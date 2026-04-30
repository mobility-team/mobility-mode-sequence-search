from __future__ import annotations

import polars as pl
import pytest

from mobility_mode_sequence_search import search_mode_sequences


def test_search_mode_sequences_returns_rows() -> None:
    location_chain_steps = pl.DataFrame(
        {
            "dest_seq_id": [1],
            "locations": [[10, 11]],
        }
    )
    leg_mode_costs = pl.DataFrame(
        {
            "origin": [10, 11],
            "destination": [11, 10],
            "mode_id": [0, 0],
            "cost": [1.0, 1.0],
        }
    )
    mode_metadata = pl.DataFrame(
        {
            "mode_id": [0],
            "needs_vehicle": [False],
            "vehicle_id": [None],
            "multimodal": [False],
            "is_return_mode": [False],
            "return_mode_id": [None],
        }
    )

    result = search_mode_sequences(
        location_chain_steps=location_chain_steps,
        leg_mode_costs=leg_mode_costs,
        mode_metadata=mode_metadata,
        k_sequences=5,
    )

    assert result.to_dict(as_series=False) == {
        "dest_seq_id": [1, 1],
        "mode_seq_index": [0, 0],
        "seq_step_index": [1, 2],
        "location": [11, 10],
        "mode_index": [0, 0],
    }


def test_search_mode_sequences_accepts_integer_vehicle_ids() -> None:
    location_chain_steps = pl.DataFrame(
        {
            "dest_seq_id": [1],
            "locations": [[10, 11]],
        }
    )
    leg_mode_costs = pl.DataFrame(
        {
            "origin": [10, 11],
            "destination": [11, 10],
            "mode_id": [0, 1],
            "cost": [1.0, 1.0],
        }
    )
    mode_metadata = pl.DataFrame(
        {
            "mode_id": [0, 1],
            "needs_vehicle": [True, True],
            "vehicle_id": [0, 0],
            "multimodal": [True, True],
            "is_return_mode": [False, True],
            "return_mode_id": [1, None],
        }
    )

    result = search_mode_sequences(
        location_chain_steps=location_chain_steps,
        leg_mode_costs=leg_mode_costs,
        mode_metadata=mode_metadata,
        k_sequences=5,
    )

    assert result.to_dict(as_series=False) == {
        "dest_seq_id": [1, 1],
        "mode_seq_index": [0, 0],
        "seq_step_index": [1, 2],
        "location": [11, 10],
        "mode_index": [0, 1],
    }


def test_search_mode_sequences_accepts_string_vehicle_ids() -> None:
    location_chain_steps = pl.DataFrame(
        {
            "dest_seq_id": [1],
            "locations": [[10, 11]],
        }
    )
    leg_mode_costs = pl.DataFrame(
        {
            "origin": [10, 11],
            "destination": [11, 10],
            "mode_id": [0, 1],
            "cost": [1.0, 1.0],
        }
    )
    mode_metadata = pl.DataFrame(
        {
            "mode_id": [0, 1],
            "needs_vehicle": [True, True],
            "vehicle_id": ["car", "car"],
            "multimodal": [True, True],
            "is_return_mode": [False, True],
            "return_mode_id": [1, None],
        }
    )

    result = search_mode_sequences(
        location_chain_steps=location_chain_steps,
        leg_mode_costs=leg_mode_costs,
        mode_metadata=mode_metadata,
        k_sequences=5,
    )

    assert result.to_dict(as_series=False) == {
        "dest_seq_id": [1, 1],
        "mode_seq_index": [0, 0],
        "seq_step_index": [1, 2],
        "location": [11, 10],
        "mode_index": [0, 1],
    }


def test_search_mode_sequences_accepts_null_and_string_vehicle_ids() -> None:
    location_chain_steps = pl.DataFrame(
        {
            "dest_seq_id": [1],
            "locations": [[10, 11]],
        }
    )
    leg_mode_costs = pl.DataFrame(
        {
            "origin": [10, 10, 11],
            "destination": [11, 11, 10],
            "mode_id": [0, 2, 1],
            "cost": [1.0, 5.0, 1.0],
        }
    )
    mode_metadata = pl.DataFrame(
        {
            "mode_id": [0, 1, 2],
            "needs_vehicle": [True, True, False],
            "vehicle_id": ["car", "car", None],
            "multimodal": [True, True, False],
            "is_return_mode": [False, True, False],
            "return_mode_id": [1, None, None],
        }
    )

    result = search_mode_sequences(
        location_chain_steps=location_chain_steps,
        leg_mode_costs=leg_mode_costs,
        mode_metadata=mode_metadata,
        k_sequences=5,
    )

    assert result.to_dict(as_series=False) == {
        "dest_seq_id": [1, 1],
        "mode_seq_index": [0, 0],
        "seq_step_index": [1, 2],
        "location": [11, 10],
        "mode_index": [0, 1],
    }


def test_search_mode_sequences_rejects_mixed_vehicle_id_representations() -> None:
    location_chain_steps = pl.DataFrame(
        {
            "dest_seq_id": [1],
            "locations": [[10, 11]],
        }
    )
    leg_mode_costs = pl.DataFrame(
        {
            "origin": [10, 11],
            "destination": [11, 10],
            "mode_id": [0, 1],
            "cost": [1.0, 1.0],
        }
    )
    mode_metadata = pl.DataFrame(
        {
            "mode_id": [0, 1],
            "needs_vehicle": [True, True],
            "vehicle_id": pl.Series("vehicle_id", [0, "car"], dtype=pl.Object),
            "multimodal": [True, True],
            "is_return_mode": [False, True],
            "return_mode_id": [1, None],
        }
    )

    with pytest.raises(
        ValueError,
        match="column 'vehicle_id' must use either integer ids/null or string labels/null within one call",
    ):
        search_mode_sequences(
            location_chain_steps=location_chain_steps,
            leg_mode_costs=leg_mode_costs,
            mode_metadata=mode_metadata,
            k_sequences=5,
        )
