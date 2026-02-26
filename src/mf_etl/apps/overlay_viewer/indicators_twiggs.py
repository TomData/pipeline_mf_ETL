"""Twiggs Money Flow + TTI proxy calculations for overlay viewer.

Implementation mirrors the TradingView/PineScript logic used in this project:
- TRH/TRL use previous close
- ADV uses true-range envelope
- Wilder smoothing uses RMA (ewm alpha=1/period, adjust=False, min_periods=period)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class TwiggsParams:
    period: int = 21
    eps: float = 1e-12
    scale_pct: bool = False
    tti_scale: float = 1.0
    level_signal: float = 0.20
    level_entry: float = 0.30


def _rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()


def compute_tmf_tti(df: pd.DataFrame, params: TwiggsParams | None = None) -> pd.DataFrame:
    """Compute TMF and TTI proxy values.

    Required columns: high, low, close, volume.
    """

    cfg = params or TwiggsParams()
    out = df.copy()
    required = {"high", "low", "close", "volume"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Missing required columns for TMF/TTI: {', '.join(missing)}")

    high = pd.to_numeric(out["high"], errors="coerce").astype(float)
    low = pd.to_numeric(out["low"], errors="coerce").astype(float)
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    volume = pd.to_numeric(out["volume"], errors="coerce").astype(float)

    prev_close = close.shift(1)
    trh = np.maximum(high, prev_close)
    trl = np.minimum(low, prev_close)
    tr = trh - trl
    tr_safe = np.maximum(tr, float(cfg.eps))

    adv = ((2.0 * close - trh - trl) / tr_safe) * volume
    tmf_num = _rma(pd.Series(adv, index=out.index), cfg.period)
    tmf_den = _rma(pd.Series(volume, index=out.index), cfg.period)
    tmf = tmf_num / np.maximum(tmf_den, float(cfg.eps))

    vol_proxy = tr
    advv = ((2.0 * close - trh - trl) / tr_safe) * vol_proxy
    tti_num = _rma(pd.Series(advv, index=out.index), cfg.period)
    tti_den = _rma(pd.Series(vol_proxy, index=out.index), cfg.period)
    tti = (tti_num / np.maximum(tti_den, float(cfg.eps))) * float(cfg.tti_scale)

    scale = 100.0 if cfg.scale_pct else 1.0
    out["prev_close"] = prev_close.astype(float)
    out["trh"] = trh.astype(float)
    out["trl"] = trl.astype(float)
    out["tr"] = tr.astype(float)
    out["adv"] = np.asarray(adv, dtype=float)
    out["advv"] = np.asarray(advv, dtype=float)
    out["tmf_raw"] = tmf.astype(float)
    out["tti_raw"] = tti.astype(float)
    out["tmf"] = out["tmf_raw"] * scale
    out["tti"] = out["tti_raw"] * scale
    out["tmf_zero_cross"] = np.sign(out["tmf"]).diff().fillna(0).ne(0)
    out["tti_zero_cross"] = np.sign(out["tti"]).diff().fillna(0).ne(0)
    out["tti_level_signal"] = float(cfg.level_signal) * scale
    out["tti_level_entry"] = float(cfg.level_entry) * scale
    return out


def _is_pivot_low(values: np.ndarray, idx: int, left: int, right: int) -> bool:
    if idx - left < 0 or idx + right >= len(values):
        return False
    window = values[idx - left : idx + right + 1]
    center = values[idx]
    if not np.isfinite(center):
        return False
    min_val = np.nanmin(window)
    if not np.isfinite(min_val):
        return False
    return center <= min_val


def _is_pivot_high(values: np.ndarray, idx: int, left: int, right: int) -> bool:
    if idx - left < 0 or idx + right >= len(values):
        return False
    window = values[idx - left : idx + right + 1]
    center = values[idx]
    if not np.isfinite(center):
        return False
    max_val = np.nanmax(window)
    if not np.isfinite(max_val):
        return False
    return center >= max_val


def compute_reading_labels(
    df: pd.DataFrame,
    *,
    value_col: str,
    date_col: str = "trade_date",
    pivot_left: int = 2,
    pivot_right: int = 2,
    zero_tolerance: float = 0.01,
) -> pd.DataFrame:
    """Build reading-mode labels similar to PineScript pivot annotations."""

    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    dates = pd.to_datetime(df[date_col], errors="coerce")

    rows: list[dict[str, object]] = []
    prev_low: float | None = None
    prev_high: float | None = None

    for idx in range(len(values)):
        val = values[idx]
        if not np.isfinite(val):
            continue

        if _is_pivot_low(values, idx, pivot_left, pivot_right):
            if abs(val) <= zero_tolerance:
                label = "AT_ZERO"
            elif val > 0 and prev_low is not None and val > prev_low:
                label = "RISING_TROUGH"
            elif val > 0:
                label = "TROUGH_ABOVE_ZERO"
            else:
                label = "PIVOT_LOW"
            rows.append(
                {
                    "trade_date": dates.iloc[idx],
                    "value": float(val),
                    "label": label,
                    "label_side": "low",
                }
            )
            prev_low = float(val)

        if _is_pivot_high(values, idx, pivot_left, pivot_right):
            if abs(val) <= zero_tolerance:
                label = "AT_ZERO"
            elif val < 0 and prev_high is not None and val < prev_high:
                label = "DECLINING_PEAK"
            elif val < 0:
                label = "PEAK_BELOW_ZERO"
            else:
                label = "PIVOT_HIGH"
            rows.append(
                {
                    "trade_date": dates.iloc[idx],
                    "value": float(val),
                    "label": label,
                    "label_side": "high",
                }
            )
            prev_high = float(val)

    if not rows:
        return pd.DataFrame(columns=["trade_date", "value", "label", "label_side"])
    return pd.DataFrame(rows)
