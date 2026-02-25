"""Parameter grid construction utilities for backtest sensitivity runs."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from typing import Any, cast

from mf_etl.backtest.models import EquityMode, ExitMode, SignalMode
from mf_etl.backtest.sensitivity_models import GridComboSpec, GridDimensionValues


def state_subset_key(state_ids: list[int]) -> str | None:
    """Return deterministic textual key for an include-state subset."""

    if not state_ids:
        return None
    return "|".join(str(x) for x in state_ids)


def build_grid_combinations(
    dimensions: GridDimensionValues,
    *,
    max_combos: int,
    shuffle_grid: bool,
    seed: int,
) -> list[GridComboSpec]:
    """Build Cartesian grid combinations with deterministic ordering/shuffle."""

    combos: list[GridComboSpec] = []
    include_sets = dimensions.include_state_sets or [[]]

    for (
        hold_bars,
        signal_mode,
        exit_mode,
        fee_bps,
        slippage_bps,
        allow_overlap,
        equity_mode,
        include_watch,
        include_state_ids,
    ) in itertools.product(
        dimensions.hold_bars,
        dimensions.signal_mode,
        dimensions.exit_mode,
        dimensions.fee_bps_per_side,
        dimensions.slippage_bps_per_side,
        dimensions.allow_overlap,
        dimensions.equity_mode,
        dimensions.include_watch,
        include_sets,
    ):
        subset = sorted(set(int(v) for v in include_state_ids))
        combos.append(
            GridComboSpec(
                hold_bars=int(hold_bars),
                signal_mode=signal_mode,
                exit_mode=exit_mode,
                fee_bps_per_side=float(fee_bps),
                slippage_bps_per_side=float(slippage_bps),
                allow_overlap=bool(allow_overlap),
                equity_mode=equity_mode,
                include_watch=bool(include_watch),
                include_state_ids=subset,
                state_subset_key=state_subset_key(subset),
            )
        )

    if shuffle_grid and combos:
        rng = random.Random(seed)
        rng.shuffle(combos)

    if len(combos) > max_combos:
        return combos[:max_combos]
    return combos


def combo_identity_payload(source_type: str, combo: GridComboSpec) -> dict[str, Any]:
    """Payload used to generate stable combo ids."""

    return {
        "source_type": source_type,
        "hold_bars": combo.hold_bars,
        "signal_mode": combo.signal_mode,
        "exit_mode": combo.exit_mode,
        "fee_bps_per_side": combo.fee_bps_per_side,
        "slippage_bps_per_side": combo.slippage_bps_per_side,
        "allow_overlap": combo.allow_overlap,
        "equity_mode": combo.equity_mode,
        "include_watch": combo.include_watch,
        "state_subset_key": combo.state_subset_key,
    }


def combo_id(source_type: str, combo: GridComboSpec) -> str:
    """Build deterministic short combo id from normalized parameter payload."""

    rendered = json.dumps(combo_identity_payload(source_type, combo), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(rendered.encode("utf-8")).hexdigest()[:16]


def default_dimensions_from_settings(settings: Any) -> GridDimensionValues:
    """Build dimensions from application backtest_sensitivity defaults."""

    cfg = settings.backtest_sensitivity.default_grid
    return GridDimensionValues(
        hold_bars=[int(x) for x in cfg.hold_bars],
        signal_mode=[cast(SignalMode, x) for x in cfg.signal_mode],
        exit_mode=[cast(ExitMode, x) for x in cfg.exit_mode],
        fee_bps_per_side=[float(x) for x in cfg.fee_bps_per_side],
        slippage_bps_per_side=[float(x) for x in cfg.slippage_bps_per_side],
        allow_overlap=[bool(x) for x in cfg.allow_overlap],
        equity_mode=[cast(EquityMode, x) for x in cfg.equity_mode],
        include_watch=[bool(x) for x in cfg.include_watch],
        include_state_sets=[
            sorted(set(int(value) for value in state_set))
            for state_set in (cfg.include_state_sets if cfg.include_state_sets else [[]])
        ],
    )
