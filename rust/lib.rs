//! Python extension module for the Rust subtour mode-sequence search kernel.

mod api;
mod errors;
mod input;
mod index;
mod output;
mod search;

use pyo3::prelude::*;

#[pymodule]
fn _core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    api::register(m)
}
