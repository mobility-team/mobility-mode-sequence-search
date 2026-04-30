use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::errors::SearchError;

/// One grouped chain of locations, ordered by sequence step.
#[derive(Clone, Debug)]
pub struct LocationChain {
    pub dest_seq_id: u64,
    pub locations: Vec<u32>,
}

/// One long-form `(origin, destination, mode, cost)` input row.
#[derive(Clone, Debug)]
pub struct LegModeCostRow {
    pub origin: u32,
    pub destination: u32,
    pub mode_id: u16,
    pub cost: f64,
}

/// One long-form mode metadata input row.
#[derive(Clone, Debug)]
pub struct ModeMetadataRow {
    pub mode_id: u16,
    pub needs_vehicle: bool,
    pub vehicle_id: Option<u8>,
    pub multimodal: bool,
    pub is_return_mode: bool,
    pub return_mode_id: Option<u16>,
}

fn column_as_vec<'py, T>(df: &Bound<'py, PyAny>, name: &str) -> PyResult<Vec<T>>
where
    T: FromPyObject<'py>,
{
    // The parser reads one DataFrame column at a time, converts it to a Python
    // list, then extracts a typed Rust Vec from that list.
    let series = df.call_method1("get_column", (name,))?;
    let values = series.call_method0("to_list")?;
    values.extract()
}

fn check_same_len(name: &str, expected: usize, actual: usize) -> Result<(), SearchError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SearchError::InvalidSchema(format!(
            "column '{name}' has length {actual}, expected {expected}"
        )))
    }
}

fn close_location_chains(chains: &mut [LocationChain]) {
    // The legacy Python backend always searches a closed tour by appending the
    // starting location to the end of each chain. We normalize to that shape
    // here so the search core sees the same inputs.
    for chain in chains {
        let start = chain.locations[0];
        chain.locations.push(start);
    }
}

#[derive(Clone, Debug)]
enum RawVehicleId {
    Null,
    Int(u8),
    Label(String),
}

fn parse_vehicle_id_column(df: &Bound<'_, PyAny>) -> Result<Vec<Option<u8>>, SearchError> {
    let series = df.call_method1("get_column", ("vehicle_id",))?;
    let values = series.call_method0("to_list")?;
    let values: Vec<Bound<'_, PyAny>> = values.extract()?;

    let mut raw_values = Vec::with_capacity(values.len());
    let mut saw_int = false;
    let mut saw_label = false;

    for value in values {
        if value.is_none() {
            raw_values.push(RawVehicleId::Null);
            continue;
        }

        if let Ok(label) = value.extract::<String>() {
            saw_label = true;
            raw_values.push(RawVehicleId::Label(label));
            continue;
        }

        if let Ok(number) = value.extract::<i64>() {
            let vehicle_id = u8::try_from(number).map_err(|_| {
                SearchError::InvalidSchema(
                    "column 'vehicle_id' must contain integers in [0, 255], strings, or null".to_string(),
                )
            })?;
            saw_int = true;
            raw_values.push(RawVehicleId::Int(vehicle_id));
            continue;
        }

        return Err(SearchError::InvalidSchema(
            "column 'vehicle_id' must contain only integers in [0, 255], strings, or null".to_string(),
        ));
    }

    if saw_int && saw_label {
        return Err(SearchError::InvalidSchema(
            "column 'vehicle_id' must use either integer ids/null or string labels/null within one call"
                .to_string(),
        ));
    }

    if !saw_label {
        return Ok(raw_values
            .into_iter()
            .map(|value| match value {
                RawVehicleId::Null => None,
                RawVehicleId::Int(vehicle_id) => Some(vehicle_id),
                RawVehicleId::Label(_) => unreachable!("mixed string/integer inputs are rejected above"),
            })
            .collect());
    }

    // String labels are only an ergonomic boundary feature. Normalize them
    // here so the rest of the Rust core continues to work with numeric ids.
    let mut label_to_id: BTreeMap<String, u8> = BTreeMap::new();
    for value in &raw_values {
        if let RawVehicleId::Label(label) = value {
            if label_to_id.contains_key(label) {
                continue;
            }
            let next_id = u8::try_from(label_to_id.len()).map_err(|_| {
                SearchError::InvalidSchema(
                    "column 'vehicle_id' contains more than 256 distinct string labels".to_string(),
                )
            })?;
            label_to_id.insert(label.clone(), next_id);
        }
    }

    Ok(raw_values
        .into_iter()
        .map(|value| match value {
            RawVehicleId::Null => None,
            RawVehicleId::Int(_) => unreachable!("mixed string/integer inputs are rejected above"),
            RawVehicleId::Label(label) => Some(*label_to_id
                .get(&label)
                .expect("every parsed label must have been normalized")),
        })
        .collect())
}

/// Parse chain-step rows from a Python DataFrame and group them by chain.
///
/// The function accepts two shapes:
/// - grouped rows with one `locations` list per chain
/// - long-form rows with one `(dest_seq_id, seq_step_index, location)` per step
///
/// # Errors
/// Returns `SearchError::InvalidSchema` if required columns are missing,
/// have inconsistent lengths, or define chains shorter than two locations.
pub fn parse_location_chains(df: &Bound<'_, PyAny>) -> Result<Vec<LocationChain>, SearchError> {
    // First detect the grouped representation because it is the simplest to
    // consume as-is.
    if df.hasattr("columns")? {
        let columns: Vec<String> = df.getattr("columns")?.extract()?;
        if columns.iter().any(|column| column == "locations") {
            return parse_grouped_location_chains(df);
        }
    }

    let dest_seq_id: Vec<u64> = column_as_vec(df, "dest_seq_id")?;
    let seq_step_index: Vec<u32> = column_as_vec(df, "seq_step_index")?;
    let location: Vec<u32> = column_as_vec(df, "location")?;

    check_same_len("seq_step_index", dest_seq_id.len(), seq_step_index.len())?;
    check_same_len("location", dest_seq_id.len(), location.len())?;

    // Rebuild each chain from long-form rows. Sorting by `(dest_seq_id,
    // seq_step_index)` lets callers provide rows in any order.
    let mut rows: Vec<(u64, u32, u32)> = dest_seq_id
        .into_iter()
        .zip(seq_step_index)
        .zip(location)
        .map(|((dest_seq_id, seq_step_index), location)| (dest_seq_id, seq_step_index, location))
        .collect();
    rows.sort_by_key(|row| (row.0, row.1));

    let mut chains: Vec<LocationChain> = Vec::new();
    for (dest_id, _step_index, loc) in rows {
        match chains.last_mut() {
            Some(chain) if chain.dest_seq_id == dest_id => chain.locations.push(loc),
            _ => chains.push(LocationChain {
                dest_seq_id: dest_id,
                locations: vec![loc],
            }),
        }
    }

    if chains.iter().any(|chain| chain.locations.len() < 2) {
        return Err(SearchError::InvalidSchema(
            "every location chain must contain at least two locations".to_string(),
        ));
    }

    let mut chains = chains;
    close_location_chains(&mut chains);
    Ok(chains)
}

fn parse_grouped_location_chains(df: &Bound<'_, PyAny>) -> Result<Vec<LocationChain>, SearchError> {
    let dest_seq_id: Vec<u64> = column_as_vec(df, "dest_seq_id")?;
    let locations: Vec<Vec<u32>> = column_as_vec(df, "locations")?;

    check_same_len("locations", dest_seq_id.len(), locations.len())?;

    let chains: Vec<LocationChain> = dest_seq_id
        .into_iter()
        .zip(locations)
        .map(|(dest_seq_id, locations)| LocationChain { dest_seq_id, locations })
        .collect();

    if chains.iter().any(|chain| chain.locations.len() < 2) {
        return Err(SearchError::InvalidSchema(
            "every location chain must contain at least two locations".to_string(),
        ));
    }

    let mut chains = chains;
    close_location_chains(&mut chains);
    Ok(chains)
}

/// Parse long-form leg mode costs from a Python DataFrame.
///
/// # Errors
/// Returns `SearchError::InvalidSchema` if required columns are missing or have
/// inconsistent lengths.
pub fn parse_leg_mode_costs(df: &Bound<'_, PyAny>) -> Result<Vec<LegModeCostRow>, SearchError> {
    let origin: Vec<u32> = column_as_vec(df, "origin")?;
    let destination: Vec<u32> = column_as_vec(df, "destination")?;
    let mode_id: Vec<u16> = column_as_vec(df, "mode_id")?;
    let cost: Vec<f64> = column_as_vec(df, "cost")?;

    check_same_len("destination", origin.len(), destination.len())?;
    check_same_len("mode_id", origin.len(), mode_id.len())?;
    check_same_len("cost", origin.len(), cost.len())?;

    Ok(origin
        .into_iter()
        .zip(destination)
        .zip(mode_id)
        .zip(cost)
        .map(|(((origin, destination), mode_id), cost)| LegModeCostRow {
            origin,
            destination,
            mode_id,
            cost,
        })
        .collect())
}

/// Parse long-form mode metadata from a Python DataFrame.
///
/// # Errors
/// Returns `SearchError::InvalidSchema` if required columns are missing or have
/// inconsistent lengths.
pub fn parse_mode_metadata(df: &Bound<'_, PyAny>) -> Result<Vec<ModeMetadataRow>, SearchError> {
    let mode_id: Vec<u16> = column_as_vec(df, "mode_id")?;
    let needs_vehicle: Vec<bool> = column_as_vec(df, "needs_vehicle")?;
    let vehicle_id = parse_vehicle_id_column(df)?;
    let multimodal: Vec<bool> = column_as_vec(df, "multimodal")?;
    let is_return_mode: Vec<bool> = column_as_vec(df, "is_return_mode")?;
    let return_mode_id: Vec<Option<u16>> = column_as_vec(df, "return_mode_id")?;

    check_same_len("needs_vehicle", mode_id.len(), needs_vehicle.len())?;
    check_same_len("vehicle_id", mode_id.len(), vehicle_id.len())?;
    check_same_len("multimodal", mode_id.len(), multimodal.len())?;
    check_same_len("is_return_mode", mode_id.len(), is_return_mode.len())?;
    check_same_len("return_mode_id", mode_id.len(), return_mode_id.len())?;

    Ok(mode_id
        .into_iter()
        .zip(needs_vehicle)
        .zip(vehicle_id)
        .zip(multimodal)
        .zip(is_return_mode)
        .zip(return_mode_id)
        .map(
            |(((((mode_id, needs_vehicle), vehicle_id), multimodal), is_return_mode), return_mode_id)| {
                ModeMetadataRow {
                    mode_id,
                    needs_vehicle,
                    vehicle_id,
                    multimodal,
                    is_return_mode,
                    return_mode_id,
                }
            },
        )
        .collect())
}
