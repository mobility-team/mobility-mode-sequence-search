# mobility-mode-sequence-search

Rust package for mode-sequence search in `mobility`.

## Scope

This package owns:

- compact indexing of mode and cost data
- top-k mode-sequence search
- chain-level parallelism
- a Python extension API that accepts and returns columnar data

The main `mobility` repository remains responsible for orchestration,
validation, and any fallback Python implementation.

## How It Works

This package searches feasible mode sequences for an ordered chain of places.

For example, if someone goes from home to a shop, feasible answers might
include:

- walk, walk
- car outbound, car return
- bike outbound, bike return

Feasibility is constrained by a few simple rules:

- a vehicle can only be used if it is currently at the traveler's location
- if a vehicle leaves home, it must end back at home
- some outbound choices come in matched pairs, requiring a linked return mode
  later in the same subtour

Conceptually, the algorithm has four stages:

1. Split long chains into smaller home-to-home segments when possible.
2. Search each segment incrementally, always expanding the cheapest partial
   answer first.
3. Merge the best segment-level answers into full-chain answers.
4. Keep only the leading answers needed to reach the requested cumulative
   probability threshold.

Before search, each input chain is closed internally by appending the starting
location to the end. This matches the legacy Python backend, which searches a
closed tour rather than the raw caller-provided chain.

### Pseudocode

```text
for each chain:
  split the chain into home-to-home segments

  for each segment:
    start with one empty partial answer

    while there are still partial answers to explore:
      take the cheapest partial answer so far

      if it already covers every leg:
        if all vehicles are back home:
          save it as a feasible full answer
        continue

      look up the next leg's available modes

      for each allowed mode:
        update vehicle locations
        update any forced future return-mode rule

        if the partial answer is still feasible:
          put the extended partial answer back into the queue

  merge the best segment answers into full-chain answers
  prune the low-probability tail
  write the retained answers as output rows
```

Implementation details are in
[rust/search.rs](/d:/dev/mobility-mode-sequence-search/rust/search.rs) and
[rust/input.rs](/d:/dev/mobility-mode-sequence-search/rust/input.rs).

## Python API

```python
import polars as pl
from mobility_mode_sequence_search import search_mode_sequences

result = search_mode_sequences(
    location_chain_steps=pl.DataFrame(...),
    leg_mode_costs=pl.DataFrame(...),
    mode_metadata=pl.DataFrame(...),
    k_sequences=20,
    cumulative_prob_threshold=0.98,
    n_threads=None,
)
```

## Input Schemas

`location_chain_steps`

Grouped form:
- `dest_seq_id: UInt64`
- `locations: List[UInt32]`

Long-form:
- `dest_seq_id: UInt64`
- `seq_step_index: UInt32`
- `location: UInt32`

`leg_mode_costs`

- `origin: UInt32`
- `destination: UInt32`
- `mode_id: UInt16`
- `cost: Float64`

`mode_metadata`

- `mode_id: UInt16`
- `needs_vehicle: Boolean`
- `vehicle_id: UInt8 | Utf8 | null`
- `multimodal: Boolean`
- `is_return_mode: Boolean`
- `return_mode_id: UInt16 | null`

At the package boundary, `vehicle_id` may be integer/null or string/null.
String labels are normalized internally into numeric ids before search. Mixed
integer and string representations within one call are rejected.

## Output Schema

- `dest_seq_id: UInt64`
- `mode_seq_index: UInt32`
- `seq_step_index: UInt32`
- `location: UInt32`
- `mode_index: UInt16`

## Development

```bash
mamba run -n mobility python -m pip install -e .[dev]
mamba run -n mobility python -m pytest
mamba run -n mobility python -m maturin build --release
```

## Performance Fixture

A standalone synthetic performance case is available at
[tests/perf_synthetic_case.py](/d:/dev/mobility-mode-sequence-search/tests/perf_synthetic_case.py).

Example:

```bash
mamba run -n mobility python tests/perf_synthetic_case.py --n-chains 2000 --chain-len 18 --n-locations 128 --k-sequences 20
```
