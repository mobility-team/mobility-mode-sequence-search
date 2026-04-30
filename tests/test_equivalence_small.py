from __future__ import annotations

import polars as pl

from mobility_mode_sequence_search import search_mode_sequences


def test_keeps_multiple_sequences_for_simple_round_trip() -> None:
    location_chain_steps = pl.DataFrame(
        {
            "dest_seq_id": [1, 1],
            "seq_step_index": [0, 1],
            "location": [10, 11],
        }
    )
    leg_mode_costs = pl.DataFrame(
        {
            "origin": [10, 10, 11, 11],
            "destination": [11, 11, 10, 10],
            "mode_id": [0, 1, 0, 1],
            "cost": [1.0, 2.0, 1.0, 2.0],
        }
    )
    mode_metadata = pl.DataFrame(
        {
            "mode_id": [0, 1],
            "needs_vehicle": [False, False],
            "vehicle_id": [None, None],
            "multimodal": [False, False],
            "is_return_mode": [False, False],
            "return_mode_id": [None, None],
        }
    )

    result = search_mode_sequences(
        location_chain_steps=location_chain_steps,
        leg_mode_costs=leg_mode_costs,
        mode_metadata=mode_metadata,
        k_sequences=4,
        cumulative_prob_threshold=0.999,
    ).sort(["mode_seq_index", "seq_step_index"])

    assert result.height == 8
    assert result["mode_index"].to_list() == [0, 0, 0, 1, 1, 0, 1, 1]
