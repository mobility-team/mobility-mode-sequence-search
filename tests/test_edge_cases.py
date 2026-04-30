from __future__ import annotations

import polars as pl

from mobility_mode_sequence_search import compute_subtour_mode_probabilities


def test_infeasible_vehicle_chain_returns_empty_dataframe() -> None:
    location_chain_steps = pl.DataFrame(
        {
            "dest_seq_id": [1, 1, 1],
            "seq_step_index": [0, 1, 2],
            "location": [10, 11, 12],
        }
    )
    leg_mode_costs = pl.DataFrame(
        {
            "origin": [10, 11],
            "destination": [11, 12],
            "mode_id": [0, 0],
            "cost": [1.0, 1.0],
        }
    )
    mode_metadata = pl.DataFrame(
        {
            "mode_id": [0],
            "needs_vehicle": [True],
            "vehicle_id": [0],
            "multimodal": [False],
            "is_return_mode": [False],
            "return_mode_id": [None],
        }
    )

    result = compute_subtour_mode_probabilities(
        location_chain_steps=location_chain_steps,
        leg_mode_costs=leg_mode_costs,
        mode_metadata=mode_metadata,
        k_sequences=5,
    )

    assert result.columns == [
        "dest_seq_id",
        "mode_seq_index",
        "seq_step_index",
        "location",
        "mode_index",
    ]
    assert result.is_empty()
