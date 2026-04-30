from __future__ import annotations

import polars as pl
import pytest

from mobility_mode_sequence_search import search_mode_sequences


def test_infeasible_vehicle_chain_returns_empty_dataframe() -> None:
    location_chain_steps = pl.DataFrame(
        {
            "dest_seq_id": [1, 1],
            "seq_step_index": [0, 1],
            "location": [10, 11],
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
            "vehicle_id": [0, 1],
            "multimodal": [False, False],
            "is_return_mode": [False, False],
            "return_mode_id": [None, None],
        }
    )

    result = search_mode_sequences(
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


def test_distinct_string_vehicle_labels_produce_valid_results() -> None:
    location_chain_steps = pl.DataFrame(
        {
            "dest_seq_id": [1],
            "locations": [[10, 11]],
        }
    )
    leg_mode_costs = pl.DataFrame(
        {
            "origin": [10, 10, 11, 11],
            "destination": [11, 11, 10, 10],
            "mode_id": [0, 2, 1, 3],
            "cost": [1.0, 2.0, 1.0, 2.0],
        }
    )
    mode_metadata = pl.DataFrame(
        {
            "mode_id": [0, 1, 2, 3],
            "needs_vehicle": [True, True, True, True],
            "vehicle_id": ["car", "car", "bike", "bike"],
            "multimodal": [True, True, True, True],
            "is_return_mode": [False, True, False, True],
            "return_mode_id": [1, None, 3, None],
        }
    )

    result = search_mode_sequences(
        location_chain_steps=location_chain_steps,
        leg_mode_costs=leg_mode_costs,
        mode_metadata=mode_metadata,
        k_sequences=5,
        cumulative_prob_threshold=0.999,
    ).sort(["mode_seq_index", "seq_step_index"])

    assert result.height == 4
    assert result["mode_index"].to_list() == [0, 1, 2, 3]


def test_malformed_vehicle_id_values_raise_clear_schema_error() -> None:
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
            "vehicle_id": [1.5, 1.5],
            "multimodal": [True, True],
            "is_return_mode": [False, True],
            "return_mode_id": [1, None],
        }
    )

    with pytest.raises(
        ValueError,
        match="column 'vehicle_id' must contain only integers in \\[0, 255\\], strings, or null",
    ):
        search_mode_sequences(
            location_chain_steps=location_chain_steps,
            leg_mode_costs=leg_mode_costs,
            mode_metadata=mode_metadata,
            k_sequences=5,
        )


def test_closes_all_same_chain_and_emits_final_leg() -> None:
    location_chain_steps = pl.DataFrame(
        {
            "dest_seq_id": [1],
            "locations": [[1, 1, 1, 1, 1, 1]],
        }
    )
    leg_mode_costs = pl.DataFrame(
        {
            "origin": [1],
            "destination": [1],
            "mode_id": [0],
            "cost": [1.0],
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

    assert result.height == 6
    assert result["seq_step_index"].to_list() == [1, 2, 3, 4, 5, 6]
    assert result["location"].to_list() == [1, 1, 1, 1, 1, 1]


def test_closing_chain_can_turn_open_chain_feasible() -> None:
    location_chain_steps = pl.DataFrame(
        {
            "dest_seq_id": [1],
            "locations": [[133, 1, 1, 1, 1]],
        }
    )
    leg_mode_costs = pl.DataFrame(
        {
            "origin": [133, 1, 1],
            "destination": [1, 1, 133],
            "mode_id": [0, 0, 0],
            "cost": [1.0, 1.0, 1.0],
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

    assert not result.is_empty()
    assert result.height == 5
    assert result["location"].to_list() == [1, 1, 1, 1, 133]


def test_closed_chain_includes_final_return_leg() -> None:
    location_chain_steps = pl.DataFrame(
        {
            "dest_seq_id": [1],
            "locations": [[101, 1, 101, 147]],
        }
    )
    leg_mode_costs = pl.DataFrame(
        {
            "origin": [101, 1, 101, 147],
            "destination": [1, 101, 147, 101],
            "mode_id": [0, 0, 0, 0],
            "cost": [1.0, 1.0, 1.0, 1.0],
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

    assert result.height == 4
    assert result["seq_step_index"].to_list() == [1, 2, 3, 4]
    assert result["location"].to_list() == [1, 101, 147, 101]
