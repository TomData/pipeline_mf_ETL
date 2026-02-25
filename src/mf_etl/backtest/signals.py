"""Signal generation for state-driven backtest harness modes."""

from __future__ import annotations

from typing import Any

import polars as pl

from mf_etl.backtest.models import SignalMode, SignalResult


def _entry_side(direction: str) -> str | None:
    if direction == "LONG_BIAS":
        return "LONG"
    if direction == "SHORT_BIAS":
        return "SHORT"
    return None


def generate_signals(
    frame: pl.DataFrame,
    *,
    signal_mode: SignalMode,
    confirm_bars: int = 2,
) -> SignalResult:
    """Generate entry signals from mapped state rows."""

    if frame.height == 0:
        out = frame.with_columns(
            [
                pl.lit(False).alias("entry_signal"),
                pl.lit(None).cast(pl.String).alias("entry_side"),
                pl.lit(False).alias("state_changed"),
                pl.lit(False).alias("overlay_candidate_before"),
                pl.lit(False).alias("overlay_candidate_after"),
                pl.lit(False).alias("overlay_vetoed_signal"),
                pl.lit(True).alias("overlay_gate_pass"),
                pl.lit(False).alias("overlay_direction_conflict"),
                pl.lit(True).alias("execution_gate_pass"),
                pl.lit(False).alias("execution_candidate_before"),
                pl.lit(False).alias("execution_candidate_after"),
                pl.lit(False).alias("execution_suppressed_signal"),
                pl.lit("none").alias("execution_filter_reason"),
            ]
        )
        return SignalResult(
            frame=out,
            diagnostics={
                "entry_signals": 0,
                "eligible_rows": 0,
                "signal_mode": signal_mode,
                "overlay_enabled": False,
                "overlay_mode": "none",
                "candidate_signals_before_overlay": 0,
                "candidate_signals_after_overlay": 0,
                "overlay_vetoed_signal_count": 0,
                "overlay_vetoed_signal_share": None,
                "overlay_passed_signal_count": 0,
                "overlay_vetoed_by_policy_class": {},
                "overlay_direction_conflict_count": 0,
                "overlay_direction_conflict_share": None,
                "execution_filters_enabled": False,
                "candidate_signals_before_execution": 0,
                "candidate_signals_after_execution": 0,
                "execution_suppressed_signal_count": 0,
                "execution_suppressed_signal_share": 0.0,
                "execution_suppressed_by_reason": {},
                "execution_suppressed_by_reason_count": {},
                "execution_suppressed_by_reason_share": {},
            },
        )

    sorted_df = frame.sort(["ticker", "trade_date"])
    rows_out: list[dict[str, Any]] = []
    entry_signals = 0
    eligible_rows = 0
    overlay_enabled = bool(
        "overlay_enabled" in sorted_df.columns and int(sorted_df.select(pl.col("overlay_enabled").cast(pl.Int8).sum()).item() or 0) > 0
    )
    overlay_mode = (
        str(sorted_df.select(pl.col("overlay_mode").drop_nulls().first()).item())
        if "overlay_mode" in sorted_df.columns and sorted_df.height > 0
        else "none"
    )
    candidate_before = 0
    candidate_after = 0
    overlay_vetoed = 0
    overlay_vetoed_by_policy_class: dict[str, int] = {}
    direction_conflicts = 0
    candidate_before_execution = 0
    candidate_after_execution = 0
    execution_suppressed = 0
    execution_suppressed_by_reason: dict[str, int] = {}
    execution_filters_enabled = bool(
        "execution_filters_enabled" in sorted_df.columns
        and int(
            sorted_df.select(
                pl.col("execution_filters_enabled").cast(pl.Int8, strict=False).sum()
            ).item()
            or 0
        )
        > 0
    )

    for ticker, sub in sorted_df.group_by("ticker", maintain_order=True):
        ticker_str = str(ticker[0] if isinstance(ticker, tuple) else ticker)
        data = sub.to_dicts()
        prev_state = None
        prev_side = None
        same_side_streak = 0

        for row in data:
            direction = str(row.get("state_direction_hint") or "UNCONFIRMED")
            side = _entry_side(direction)
            primary_eligible = bool(row.get("signal_eligible")) and side is not None
            overlay_pass = bool(row.get("overlay_allow_signal", True))
            execution_pass = bool(row.get("execution_eligible", True))
            eligible = bool(primary_eligible and overlay_pass and execution_pass)
            state_id = row.get("state_id")
            changed = prev_state is not None and state_id != prev_state

            if eligible:
                eligible_rows += 1

            if primary_eligible:
                candidate_before += 1
                if not overlay_pass:
                    overlay_vetoed += 1
                    policy_class = str(row.get("overlay_policy_class") or "UNKNOWN")
                    overlay_vetoed_by_policy_class[policy_class] = (
                        int(overlay_vetoed_by_policy_class.get(policy_class, 0)) + 1
                    )
                else:
                    candidate_after += 1

                execution_candidate_before_row = bool(primary_eligible and overlay_pass)
                if execution_candidate_before_row:
                    candidate_before_execution += 1
                    if not execution_pass:
                        execution_suppressed += 1
                        reason = str(row.get("execution_filter_reason") or "none")
                        execution_suppressed_by_reason[reason] = (
                            int(execution_suppressed_by_reason.get(reason, 0)) + 1
                        )
                    else:
                        candidate_after_execution += 1

                overlay_direction = str(row.get("overlay_direction_hint") or "UNCONFIRMED")
                if direction in {"LONG_BIAS", "SHORT_BIAS"} and overlay_direction in {"LONG_BIAS", "SHORT_BIAS"}:
                    if direction != overlay_direction:
                        direction_conflicts += 1

            entry_signal = False
            if signal_mode == "state_entry":
                entry_signal = eligible
            elif signal_mode == "state_transition_entry":
                entry_signal = bool(eligible and (prev_state is None or changed))
            else:
                if eligible and side == prev_side:
                    same_side_streak += 1
                elif eligible:
                    same_side_streak = 1
                else:
                    same_side_streak = 0
                entry_signal = bool(eligible and same_side_streak >= max(1, confirm_bars))

            if entry_signal:
                entry_signals += 1

            out_row = dict(row)
            out_row["ticker"] = ticker_str
            out_row["entry_side"] = side
            out_row["entry_signal"] = bool(entry_signal)
            out_row["state_changed"] = bool(changed)
            out_row["overlay_candidate_before"] = bool(primary_eligible)
            out_row["overlay_candidate_after"] = bool(primary_eligible and overlay_pass)
            out_row["overlay_vetoed_signal"] = bool(primary_eligible and (not overlay_pass))
            out_row["overlay_gate_pass"] = bool(overlay_pass)
            out_row["overlay_direction_conflict"] = bool(
                direction in {"LONG_BIAS", "SHORT_BIAS"}
                and str(row.get("overlay_direction_hint") or "UNCONFIRMED") in {"LONG_BIAS", "SHORT_BIAS"}
                and direction != str(row.get("overlay_direction_hint") or "UNCONFIRMED")
            )
            out_row["execution_gate_pass"] = bool(execution_pass)
            out_row["execution_candidate_before"] = bool(primary_eligible and overlay_pass)
            out_row["execution_candidate_after"] = bool(primary_eligible and overlay_pass and execution_pass)
            out_row["execution_suppressed_signal"] = bool(primary_eligible and overlay_pass and (not execution_pass))
            out_row["execution_filter_reason"] = str(row.get("execution_filter_reason") or "none")
            rows_out.append(out_row)

            prev_state = state_id
            prev_side = side if eligible else None

    out_df = pl.DataFrame(rows_out, infer_schema_length=None).sort(["ticker", "trade_date"])
    diagnostics = {
        "signal_mode": signal_mode,
        "rows": int(out_df.height),
        "eligible_rows": int(eligible_rows),
        "entry_signals": int(entry_signals),
        "overlay_enabled": overlay_enabled,
        "overlay_mode": overlay_mode,
        "candidate_signals_before_overlay": int(candidate_before),
        "candidate_signals_after_overlay": int(candidate_after),
        "overlay_vetoed_signal_count": int(overlay_vetoed),
        "overlay_vetoed_signal_share": (
            float(overlay_vetoed / candidate_before) if candidate_before > 0 else None
        ),
        "overlay_passed_signal_count": int(candidate_after),
        "overlay_vetoed_by_policy_class": overlay_vetoed_by_policy_class,
        "overlay_direction_conflict_count": int(direction_conflicts),
        "overlay_direction_conflict_share": (
            float(direction_conflicts / candidate_before) if candidate_before > 0 else None
        ),
        "execution_filters_enabled": execution_filters_enabled,
        "candidate_signals_before_execution": int(candidate_before_execution),
        "candidate_signals_after_execution": int(candidate_after_execution),
        "execution_suppressed_signal_count": int(execution_suppressed),
        "execution_suppressed_signal_share": (
            float(execution_suppressed / candidate_before_execution)
            if candidate_before_execution > 0
            else 0.0
        ),
        # Keep this key as share map for compatibility with earlier callers.
        "execution_suppressed_by_reason": {
            reason: (
                float(count / execution_suppressed)
                if execution_suppressed > 0
                else 0.0
            )
            for reason, count in execution_suppressed_by_reason.items()
        },
        "execution_suppressed_by_reason_count": execution_suppressed_by_reason,
        "execution_suppressed_by_reason_share": {
            reason: (
                float(count / execution_suppressed)
                if execution_suppressed > 0
                else 0.0
            )
            for reason, count in execution_suppressed_by_reason.items()
        },
    }
    return SignalResult(frame=out_df, diagnostics=diagnostics)
