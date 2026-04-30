# mobility-mode-sequence-search

Native mode-sequence search kernel for `mobility`.

## Scope

This package is intended to own:

- compact indexing of mode/cost data
- top-k mode-sequence search
- chain-level parallelism
- a Python extension API that accepts and returns columnar data

The main `mobility` repository should remain responsible for orchestration, validation, and fallback Python implementations.

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
