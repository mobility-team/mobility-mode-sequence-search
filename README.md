# mobility-mode-sequence-search

Rust mode-sequence search kernel for `mobility`.

## Scope

This package is intended to own:

- compact indexing of mode/cost data
- top-k mode-sequence search
- chain-level parallelism
- a Python extension API that accepts and returns columnar data

The main `mobility` repository should remain responsible for orchestration, validation, and fallback Python implementations.

## How It Works

This package answers one question:

Given an ordered chain of places, what are the best feasible mode sequences?

For example, if someone goes from home to a shop and back, possible answers
could be:

- walk, walk
- car outbound, car return
- bike outbound, bike return

The key word is `feasible`. The package is not only looking for low-cost
sequences. It is also enforcing rules such as:

- a vehicle can only be used if it is currently where the traveler is
- if a vehicle leaves home, it must end up back at home by the end
- some outbound choices come in matched pairs: if the outbound leg uses one
  mode, the corresponding return leg must use its linked return mode later in
  the same subtour

Conceptually, the algorithm has four stages:

1. Split one long chain into smaller home-to-home segments when possible.
   For example:
   - `[home, shop, home, office, home]`
   becomes
   - `[home, shop, home]`
   - `[home, office, home]`

   Smaller pieces are easier to search than one long chain. The results are
   combined again later.

2. Search each segment by building answers step by step.
   Start with an empty partial answer and repeatedly extend the cheapest
   partial answer seen so far.

   At each step, keep track of:
   - which modes have already been chosen
   - where each shared vehicle currently is
   - whether a later return leg is forced to use a specific mode

   If a partial answer becomes impossible, discard it immediately.

3. Merge the best segment-level answers into full-chain answers.
   This avoids searching the whole chain as one large combinatorial problem.

4. Keep only the useful final answers.
   After feasible answers have been found:
   - sorts them from best to worst
   - converts costs into relative probabilities
   - keeps only the top part of the list until the requested cumulative
     probability threshold is reached

The output is therefore not every possible answer. It is the small set of best
answers that covers most of the probability mass.

### Implementation Notes

The implementation details are documented in the Rust source, especially
[rust/search.rs](/d:/dev/mobility-mode-sequence-search/rust/search.rs) for the
search itself and [rust/input.rs](/d:/dev/mobility-mode-sequence-search/rust/input.rs)
for input normalization.

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

## Python API

```python
import polars as pl
from mobility_mode_sequence_search import compute_subtour_mode_probabilities

result = compute_subtour_mode_probabilities(
    location_chain_steps=pl.DataFrame(...),
    leg_mode_costs=pl.DataFrame(...),
    mode_metadata=pl.DataFrame(...),
    k_sequences=20,
    cumulative_prob_threshold=0.98,
    n_threads=None,
)
```

## Input Schemas

`location_chain_steps`:

- either grouped:
- `dest_seq_id: UInt64`
- `locations: List[UInt32]`
- or long-form:
- `dest_seq_id: UInt64`
- `seq_step_index: UInt32`
- `location: UInt32`

`leg_mode_costs`:

- `origin: UInt32`
- `destination: UInt32`
- `mode_id: UInt16`
- `cost: Float64`

`mode_metadata`:

- `mode_id: UInt16`
- `needs_vehicle: Boolean`
- `vehicle_id: UInt8 | null`
- `multimodal: Boolean`
- `is_return_mode: Boolean`
- `return_mode_id: UInt16 | null`

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
