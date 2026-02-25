"""Deterministic per-symbol backtest simulation engine."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from mf_etl.backtest.models import EngineResult, ExitMode


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _gross_return(side: str, entry_price: float, exit_price: float) -> float | None:
    if entry_price <= 0 or exit_price <= 0:
        return None
    if side == "LONG":
        return (exit_price / entry_price) - 1.0
    return (entry_price / exit_price) - 1.0


def _update_excursions(position: dict[str, Any], row: dict[str, Any]) -> None:
    entry = float(position["entry_price"])
    high = _finite(row.get("high"))
    low = _finite(row.get("low"))
    if entry <= 0 or high is None or low is None or high <= 0 or low <= 0:
        return

    side = str(position["side"])
    if side == "LONG":
        favorable = (high / entry) - 1.0
        adverse = (low / entry) - 1.0
    else:
        favorable = (entry / low) - 1.0
        adverse = (entry / high) - 1.0

    position["mfe"] = max(float(position["mfe"]), float(favorable))
    position["mae"] = min(float(position["mae"]), float(adverse))


def _close_position(
    *,
    position: dict[str, Any],
    exit_row: dict[str, Any],
    exit_reason: str,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
    exit_fill_fallback: bool,
) -> dict[str, Any]:
    exit_price = _finite(exit_row.get("open"))
    if exit_price is None or exit_price <= 0:
        exit_price = _finite(exit_row.get("close"))
        exit_fill_fallback = True

    entry_price = float(position["entry_price"])
    side = str(position["side"])
    gross = _gross_return(side, entry_price, float(exit_price) if exit_price is not None else -1.0)

    per_side_cost = (fee_bps_per_side + slippage_bps_per_side) / 10_000.0
    total_cost = 2.0 * per_side_cost

    is_valid = gross is not None and np.isfinite(gross)
    net = (gross - total_cost) if is_valid else None

    return {
        "position_id": position["position_id"],
        "ticker": position["ticker"],
        "side": side,
        "entry_signal_date": position["entry_signal_date"],
        "entry_date": position["entry_date"],
        "entry_price": entry_price,
        "exit_signal_date": position["exit_signal_date"],
        "exit_date": exit_row.get("trade_date"),
        "exit_price": exit_price,
        "hold_bars_realized": int(exit_row["_row_index"] - position["entry_row_index"]),
        "entry_state_id": position["entry_state_id"],
        "entry_state_label": position["entry_state_label"],
        "entry_state_class": position["entry_state_class"],
        "entry_state_direction_hint": position["entry_state_direction_hint"],
        "entry_state_score": position["entry_state_score"],
        "entry_overlay_cluster_state": position.get("entry_overlay_cluster_state"),
        "entry_overlay_policy_class": position.get("entry_overlay_policy_class"),
        "entry_overlay_direction_hint": position.get("entry_overlay_direction_hint"),
        "entry_overlay_tradability_score": position.get("entry_overlay_tradability_score"),
        "exit_reason": exit_reason,
        "gross_return": gross,
        "net_return": net,
        "max_favorable_excursion": float(position["mfe"]),
        "max_adverse_excursion": float(position["mae"]),
        "fees": 2.0 * fee_bps_per_side / 10_000.0,
        "slippage_bps": 2.0 * slippage_bps_per_side,
        "exit_fill_fallback": bool(exit_fill_fallback),
        "is_valid_trade": bool(is_valid),
    }


def simulate_trades(
    frame: pl.DataFrame,
    *,
    exit_mode: ExitMode,
    hold_bars: int,
    allow_overlap: bool,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
) -> EngineResult:
    """Simulate trades for all symbols with deterministic next-open execution."""

    if frame.height == 0:
        return EngineResult(
            trades=pl.DataFrame(schema={"position_id": pl.String}),
            signal_diagnostics={
                "input_rows": 0,
                "entry_signals": 0,
                "skipped_no_next_bar_entry": 0,
                "skipped_due_overlap": 0,
                "forced_exit_last_bar": 0,
                "invalid_trade_count": 0,
            },
        )

    sorted_df = frame.sort(["ticker", "trade_date"]).with_row_index(name="_row_index")
    trades: list[dict[str, Any]] = []
    diag = {
        "input_rows": int(sorted_df.height),
        "entry_signals": int(sorted_df.select(pl.col("entry_signal").cast(pl.Int64).sum()).item()),
        "skipped_no_next_bar_entry": 0,
        "skipped_due_overlap": 0,
        "forced_exit_last_bar": 0,
        "invalid_trade_count": 0,
    }

    for ticker, sub in sorted_df.group_by("ticker", maintain_order=True):
        rows = sub.to_dicts()
        ticker_value = str(ticker[0] if isinstance(ticker, tuple) else ticker)

        position_counter = 0
        position: dict[str, Any] | None = None
        pending_entry: dict[str, Any] | None = None
        pending_exit: dict[str, Any] | None = None

        for i, row in enumerate(rows):
            # Execute pending exit on this bar open.
            if position is not None and pending_exit is not None and pending_exit["exec_i"] == i:
                position["exit_signal_date"] = pending_exit["signal_date"]
                trade = _close_position(
                    position=position,
                    exit_row=row,
                    exit_reason=str(pending_exit["reason"]),
                    fee_bps_per_side=fee_bps_per_side,
                    slippage_bps_per_side=slippage_bps_per_side,
                    exit_fill_fallback=False,
                )
                if not trade["is_valid_trade"]:
                    diag["invalid_trade_count"] += 1
                trades.append(trade)
                position = None
                pending_exit = None

            # Execute pending entry on this bar open.
            if position is None and pending_entry is not None and pending_entry["exec_i"] == i:
                entry_price = _finite(row.get("open"))
                if entry_price is None or entry_price <= 0:
                    entry_price = _finite(row.get("close"))
                if entry_price is None or entry_price <= 0:
                    diag["skipped_no_next_bar_entry"] += 1
                else:
                    position_counter += 1
                    position = {
                        "position_id": f"{ticker_value}-{position_counter}",
                        "ticker": ticker_value,
                        "side": pending_entry["side"],
                        "entry_signal_date": pending_entry["signal_date"],
                        "entry_date": row.get("trade_date"),
                        "entry_price": entry_price,
                        "entry_row_index": int(row["_row_index"]),
                        "entry_state_id": pending_entry["state_id"],
                        "entry_state_label": pending_entry["state_label"],
                        "entry_state_class": pending_entry["state_class"],
                        "entry_state_direction_hint": pending_entry["state_direction_hint"],
                        "entry_state_score": pending_entry["state_score"],
                        "entry_overlay_cluster_state": pending_entry.get("overlay_cluster_state"),
                        "entry_overlay_policy_class": pending_entry.get("overlay_policy_class"),
                        "entry_overlay_direction_hint": pending_entry.get("overlay_direction_hint"),
                        "entry_overlay_tradability_score": pending_entry.get("overlay_tradability_score"),
                        "exit_signal_date": None,
                        "mfe": 0.0,
                        "mae": 0.0,
                    }
                    _update_excursions(position, row)
                pending_entry = None

            # Update excursions for open position on current row.
            if position is not None:
                _update_excursions(position, row)

            # Generate new entry trigger.
            if bool(row.get("entry_signal")):
                if position is not None and not allow_overlap:
                    diag["skipped_due_overlap"] += 1
                elif position is None and pending_entry is None:
                    side = row.get("entry_side")
                    if side in {"LONG", "SHORT"}:
                        if i + 1 < len(rows):
                            pending_entry = {
                                "exec_i": i + 1,
                                "side": side,
                                "signal_date": row.get("trade_date"),
                                "state_id": row.get("state_id"),
                                "state_label": row.get("state_label"),
                                "state_class": row.get("state_class"),
                                "state_direction_hint": row.get("state_direction_hint"),
                                "state_score": _finite(row.get("state_score")),
                                "overlay_cluster_state": row.get("overlay_cluster_state"),
                                "overlay_policy_class": row.get("overlay_policy_class"),
                                "overlay_direction_hint": row.get("overlay_direction_hint"),
                                "overlay_tradability_score": _finite(row.get("overlay_tradability_score")),
                            }
                        else:
                            diag["skipped_no_next_bar_entry"] += 1

            # Generate exit trigger.
            if position is not None and pending_exit is None:
                hold_realized = int(row["_row_index"] - position["entry_row_index"])
                horizon_trigger = hold_realized >= max(1, int(hold_bars))
                row_side = row.get("entry_side")
                state_exit_trigger = (not bool(row.get("signal_eligible"))) or (
                    row_side in {"LONG", "SHORT"} and row_side != position["side"]
                )

                if exit_mode == "horizon":
                    trigger = horizon_trigger
                    reason = "HORIZON"
                elif exit_mode == "state_exit":
                    trigger = state_exit_trigger
                    reason = "STATE_EXIT"
                else:
                    trigger = horizon_trigger or state_exit_trigger
                    reason = "HORIZON_OR_STATE"

                if trigger:
                    if i + 1 < len(rows):
                        pending_exit = {
                            "exec_i": i + 1,
                            "signal_date": row.get("trade_date"),
                            "reason": reason,
                        }
                    else:
                        position["exit_signal_date"] = row.get("trade_date")
                        trade = _close_position(
                            position=position,
                            exit_row=row,
                            exit_reason=f"{reason}_LAST_BAR_FALLBACK",
                            fee_bps_per_side=fee_bps_per_side,
                            slippage_bps_per_side=slippage_bps_per_side,
                            exit_fill_fallback=True,
                        )
                        if not trade["is_valid_trade"]:
                            diag["invalid_trade_count"] += 1
                        trades.append(trade)
                        diag["forced_exit_last_bar"] += 1
                        position = None
                        pending_exit = None

        if position is not None:
            last_row = rows[-1]
            position["exit_signal_date"] = last_row.get("trade_date")
            trade = _close_position(
                position=position,
                exit_row=last_row,
                exit_reason="END_OF_DATA",
                fee_bps_per_side=fee_bps_per_side,
                slippage_bps_per_side=slippage_bps_per_side,
                exit_fill_fallback=True,
            )
            if not trade["is_valid_trade"]:
                diag["invalid_trade_count"] += 1
            trades.append(trade)
            diag["forced_exit_last_bar"] += 1

    trades_df = pl.DataFrame(trades) if trades else pl.DataFrame(schema={"position_id": pl.String})
    if trades_df.height > 0:
        trades_df = trades_df.sort(["ticker", "entry_date", "position_id"])
    return EngineResult(trades=trades_df, signal_diagnostics=diag)
