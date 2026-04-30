use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Columnar output assembled in Rust before conversion back to Polars.
#[derive(Clone, Debug, Default)]
pub struct OutputTable {
    pub dest_seq_id: Vec<u64>,
    pub mode_seq_index: Vec<u32>,
    pub seq_step_index: Vec<u32>,
    pub location: Vec<u32>,
    pub mode_index: Vec<u16>,
}

impl OutputTable {
    /// Append one output row.
    pub fn push_row(
        &mut self,
        dest_seq_id: u64,
        mode_seq_index: u32,
        seq_step_index: u32,
        location: u32,
        mode_index: u16,
    ) {
        self.dest_seq_id.push(dest_seq_id);
        self.mode_seq_index.push(mode_seq_index);
        self.seq_step_index.push(seq_step_index);
        self.location.push(location);
        self.mode_index.push(mode_index);
    }

    /// Append all rows from another output table.
    pub fn extend(&mut self, other: Self) {
        self.dest_seq_id.extend(other.dest_seq_id);
        self.mode_seq_index.extend(other.mode_seq_index);
        self.seq_step_index.extend(other.seq_step_index);
        self.location.extend(other.location);
        self.mode_index.extend(other.mode_index);
    }
}

/// Convert the Rust output table back into a Polars DataFrame.
pub fn to_polars_dataframe(py: Python<'_>, output: OutputTable) -> PyResult<PyObject> {
    let polars = py.import("polars")?;
    let data = PyDict::new(py);

    // Rust keeps appending plain typed values to column buffers while it searches.
    // At the very end, we hand those columns back to Polars in one shot instead
    // of creating Python rows one by one.
    data.set_item("dest_seq_id", PyList::new(py, output.dest_seq_id)?)?;
    data.set_item("mode_seq_index", PyList::new(py, output.mode_seq_index)?)?;
    data.set_item("seq_step_index", PyList::new(py, output.seq_step_index)?)?;
    data.set_item("location", PyList::new(py, output.location)?)?;
    data.set_item("mode_index", PyList::new(py, output.mode_index)?)?;

    let df = polars.getattr("DataFrame")?.call1((data,))?;
    Ok(df.into())
}
