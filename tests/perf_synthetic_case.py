from __future__ import annotations

import argparse
import random
import statistics
import time

import polars as pl

from mobility_mode_sequence_search import search_mode_sequences


def build_mode_metadata() -> pl.DataFrame:
    # Keep the modes simple so the fixture stresses search volume rather than
    # tricky domain semantics.
    return pl.DataFrame(
        {
            "mode_id": [0, 1, 2, 3],
            "needs_vehicle": [False, False, True, True],
            "vehicle_id": [None, None, 0, 1],
            "multimodal": [False, False, False, False],
            "is_return_mode": [False, False, False, False],
            "return_mode_id": [None, None, None, None],
        }
    )


def build_leg_mode_costs(n_locations: int) -> pl.DataFrame:
    rows: dict[str, list[int | float]] = {
        "origin": [],
        "destination": [],
        "mode_id": [],
        "cost": [],
    }

    for origin in range(n_locations):
        for destination in range(n_locations):
            if origin == destination:
                distance_cost = 1.0
            else:
                distance_cost = 1.0 + abs(origin - destination) / 10.0

            # Every ordered pair gets all four modes so the search has a
            # substantial branching factor while remaining fully feasible.
            for mode_id, mode_penalty in (
                (0, 0.0),   # walk
                (1, 0.35),  # transit
                (2, 0.15),  # car
                (3, 0.25),  # bike
            ):
                rows["origin"].append(origin)
                rows["destination"].append(destination)
                rows["mode_id"].append(mode_id)
                rows["cost"].append(distance_cost + mode_penalty)

    return pl.DataFrame(rows)


def build_location_chain_steps(
    *,
    n_chains: int,
    chain_len: int,
    n_locations: int,
    seed: int,
) -> pl.DataFrame:
    rng = random.Random(seed)
    chains: list[list[int]] = []

    for _ in range(n_chains):
        home = rng.randrange(n_locations)
        # The package closes chains internally, so this fixture provides the
        # open form callers would normally pass in.
        chain = [home]
        for _ in range(chain_len - 1):
            chain.append(rng.randrange(n_locations))
        chains.append(chain)

    return pl.DataFrame(
        {
            "dest_seq_id": list(range(n_chains)),
            "locations": chains,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a synthetic performance case against search_mode_sequences()."
    )
    parser.add_argument("--n-chains", type=int, default=2_000)
    parser.add_argument("--chain-len", type=int, default=18)
    parser.add_argument("--n-locations", type=int, default=128)
    parser.add_argument("--k-sequences", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    location_chain_steps = build_location_chain_steps(
        n_chains=args.n_chains,
        chain_len=args.chain_len,
        n_locations=args.n_locations,
        seed=args.seed,
    )
    leg_mode_costs = build_leg_mode_costs(args.n_locations)
    mode_metadata = build_mode_metadata()

    open_chain_lengths = location_chain_steps["locations"].list.len().to_list()

    start = time.perf_counter()
    result = search_mode_sequences(
        location_chain_steps=location_chain_steps,
        leg_mode_costs=leg_mode_costs,
        mode_metadata=mode_metadata,
        k_sequences=args.k_sequences,
        cumulative_prob_threshold=args.threshold,
        n_threads=args.n_threads,
    )
    elapsed_s = time.perf_counter() - start

    print("Synthetic performance case")
    print(f"chains: {args.n_chains}")
    print(f"open chain length: {args.chain_len}")
    print(f"locations in pool: {args.n_locations}")
    print(f"leg_mode_cost rows: {leg_mode_costs.height}")
    print(f"mode_metadata rows: {mode_metadata.height}")
    print(f"avg open chain length: {statistics.mean(open_chain_lengths):.2f}")
    print(f"closed legs searched per chain: {args.chain_len}")
    print(f"result rows: {result.height}")
    print(f"elapsed seconds: {elapsed_s:.3f}")
    print(f"rows per second: {result.height / elapsed_s:,.0f}")


if __name__ == "__main__":
    main()
