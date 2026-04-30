from __future__ import annotations

from typing import Any


def search_mode_sequences(
    *,
    location_chain_steps: Any,
    leg_mode_costs: Any,
    mode_metadata: Any,
    k_sequences: int,
    cumulative_prob_threshold: float = 0.98,
    n_threads: int | None = None,
) -> Any: ...
