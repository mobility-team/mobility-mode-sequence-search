use std::collections::{BTreeMap, HashMap, HashSet};

use crate::errors::SearchError;
use crate::input::{LegModeCostRow, ModeMetadataRow};

/// Precomputed lookup tables consumed by the search loop.
///
/// `edges` stores the available modes for each origin/destination pair.
/// `modes` stores mode properties indexed by `mode_id`.
#[derive(Clone, Debug, Default)]
pub struct SearchIndex {
    pub edges: HashMap<(u32, u32), Vec<ModeCost>>,
    pub modes: Vec<ModeInfo>,
    pub n_vehicles: usize,
}

/// Costed mode alternative available on one leg.
#[derive(Clone, Debug)]
pub struct ModeCost {
    pub mode_id: u16,
    pub cost: f64,
}

/// Static properties attached to one mode.
#[derive(Clone, Debug, Default)]
pub struct ModeInfo {
    pub needs_vehicle: bool,
    pub vehicle_index: Option<usize>,
    pub multimodal: bool,
    pub is_return_mode: bool,
    pub return_mode_id: Option<u16>,
}

impl SearchIndex {
    /// Build the compact search index from normalized tabular rows.
    ///
    /// # Errors
    /// Returns an error if later lookups would require missing mode metadata.
    pub fn build(
        leg_mode_costs: Vec<LegModeCostRow>,
        mode_metadata: Vec<ModeMetadataRow>,
    ) -> Result<Self, SearchError> {
        if mode_metadata.is_empty() {
            return Err(SearchError::InvalidSchema(
                "mode_metadata must contain at least one row".to_string(),
            ));
        }

        // Group every available mode by origin/destination so the search loop can
        // fetch candidate modes for one leg with a single hash lookup.
        let mut edges: HashMap<(u32, u32), Vec<ModeCost>> = HashMap::new();
        let referenced_mode_ids: Vec<u16> = leg_mode_costs.iter().map(|row| row.mode_id).collect();
        for row in leg_mode_costs {
            edges
                .entry((row.origin, row.destination))
                .or_default()
                .push(ModeCost {
                    mode_id: row.mode_id,
                    cost: row.cost,
                });
        }
        for options in edges.values_mut() {
            options.sort_by(|left, right| {
                left.cost
                    .total_cmp(&right.cost)
                    .then_with(|| left.mode_id.cmp(&right.mode_id))
            });
        }

        // Vehicle identifiers from Python are small external ids like 0, 1, 7.
        // We remap them into dense indexes 0..n because the search state stores
        // current vehicle locations in a Vec for speed.
        let mut vehicle_map: BTreeMap<u8, usize> = BTreeMap::new();
        let mut seen_mode_ids: HashSet<u16> = HashSet::new();
        for row in &mode_metadata {
            seen_mode_ids.insert(row.mode_id);
            if let Some(vehicle_id) = row.vehicle_id.as_ref() {
                let next_index = vehicle_map.len();
                vehicle_map.entry(vehicle_id.clone()).or_insert(next_index);
            }
        }

        for mode_id in referenced_mode_ids {
            if !seen_mode_ids.contains(&mode_id) {
                return Err(SearchError::MissingMode(mode_id));
            }
        }

        // Store mode metadata in a Vec indexed directly by mode_id. This makes
        // per-leg lookups trivial during the search.
        let max_mode_id = mode_metadata.iter().map(|row| row.mode_id as usize).max().unwrap_or(0);
        let mut modes = vec![ModeInfo::default(); max_mode_id + 1];
        for row in mode_metadata {
            if let Some(return_mode_id) = row.return_mode_id {
                if !seen_mode_ids.contains(&return_mode_id) {
                    return Err(SearchError::MissingMode(return_mode_id));
                }
            }
            modes[row.mode_id as usize] = ModeInfo {
                needs_vehicle: row.needs_vehicle,
                vehicle_index: row
                    .vehicle_id
                    .and_then(|vehicle_id| vehicle_map.get(&vehicle_id).copied()),
                multimodal: row.multimodal,
                is_return_mode: row.is_return_mode,
                return_mode_id: row.return_mode_id,
            };
        }

        Ok(Self {
            edges,
            modes,
            n_vehicles: vehicle_map.len(),
        })
    }

    /// Return the sorted mode options available on one origin/destination pair.
    pub fn edge_options(&self, origin: u32, destination: u32) -> Option<&[ModeCost]> {
        self.edges.get(&(origin, destination)).map(Vec::as_slice)
    }

    /// Return the metadata for one mode identifier.
    ///
    /// # Errors
    /// Returns `SearchError::MissingMode` if `mode_id` is not defined.
    pub fn mode(&self, mode_id: u16) -> Result<&ModeInfo, SearchError> {
        self.modes
            .get(mode_id as usize)
            .ok_or(SearchError::MissingMode(mode_id))
    }
}
