use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("invalid input schema: {0}")]
    InvalidSchema(String),
    #[error("python interop error: {0}")]
    Python(String),
    #[error("missing edge for origin={origin}, destination={destination}")]
    MissingEdge { origin: u32, destination: u32 },
    #[error("missing mode metadata for mode_id={0}")]
    MissingMode(u16),
}

impl From<PyErr> for SearchError {
    fn from(value: PyErr) -> Self {
        SearchError::Python(value.to_string())
    }
}

impl From<SearchError> for PyErr {
    fn from(value: SearchError) -> Self {
        PyValueError::new_err(value.to_string())
    }
}
