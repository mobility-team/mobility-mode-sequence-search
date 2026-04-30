use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::errors::SearchError;
use crate::input::{parse_leg_mode_costs, parse_location_chains, parse_mode_metadata};
use crate::index::SearchIndex;
use crate::output::to_polars_dataframe;
use crate::search::compute_all;

/// Compute feasible subtour mode sequences for many chains at once.
///
/// The function expects long-form Polars DataFrames for the chain steps, leg
/// mode costs, and mode metadata. It normalizes them into Rust structs, builds
/// compact lookup tables, runs the search in parallel across chains, and
/// returns a Polars DataFrame containing the retained mode sequences.
///
/// # Errors
/// Returns `SearchError::InvalidSchema` when the input tables are malformed or
/// when the numeric parameters are invalid.
#[pyfunction]
#[pyo3(signature = (*, location_chain_steps, leg_mode_costs, mode_metadata, k_sequences, cumulative_prob_threshold=0.98, n_threads=None))]
pub fn compute_subtour_mode_probabilities(
    py: Python<'_>,
    location_chain_steps: &Bound<'_, PyAny>,
    leg_mode_costs: &Bound<'_, PyAny>,
    mode_metadata: &Bound<'_, PyAny>,
    k_sequences: usize,
    cumulative_prob_threshold: f64,
    n_threads: Option<usize>,
) -> PyResult<PyObject> {
    // Reject obviously invalid knobs at the boundary so the deeper search code
    // can assume these parameters are already sane.
    if k_sequences == 0 {
        return Err(SearchError::InvalidSchema("k_sequences must be > 0".to_string()).into());
    }
    if !(0.0..=1.0).contains(&cumulative_prob_threshold) || cumulative_prob_threshold == 0.0 {
        return Err(SearchError::InvalidSchema(
            "cumulative_prob_threshold must be in (0, 1]".to_string(),
        )
        .into());
    }

    // Phase 1: turn Python/Polars objects into plain Rust vectors and structs.
    // Keeping Python values out of the search loop makes the rest of the code
    // much simpler and much faster.
    let chains = parse_location_chains(location_chain_steps)?;
    let leg_mode_costs = parse_leg_mode_costs(leg_mode_costs)?;
    let mode_metadata = parse_mode_metadata(mode_metadata)?;

    // Phase 2: precompute compact lookup tables so the search loop can answer
    // "what modes are allowed here?" and "what does this mode require?" with
    // cheap lookups instead of repeated table scans.
    let index = SearchIndex::build(leg_mode_costs, mode_metadata)?;

    // Phase 3: search each chain, optionally in parallel, and then convert the
    // Rust-side column buffers back into one Polars DataFrame for Python.
    let output = compute_all(&index, &chains, k_sequences, cumulative_prob_threshold, n_threads)?;
    to_polars_dataframe(py, output)
}

/// Register the Python-callable functions exposed by this extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_subtour_mode_probabilities, m)?)?;
    Ok(())
}
