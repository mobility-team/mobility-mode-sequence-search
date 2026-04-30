//! Python extension module for native subtour mode-sequence search.

mod api;
mod errors;
mod input;
mod index;
mod output;
mod search;

use pyo3::prelude::*;

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    api::register(m)
}
