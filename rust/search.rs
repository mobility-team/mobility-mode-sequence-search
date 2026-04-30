//! Core search algorithm for feasible mode sequences.
//!
//! Given one ordered chain of locations, this module finds the best
//! travel-mode sequences that are both low-cost and feasible.
//!
//! "Actually possible" means the sequence respects rules like:
//! - you can only use a vehicle if that vehicle is currently at your location
//! - if a vehicle leaves home, it must end back at home
//! - some outbound multimodal choices force a matching return mode later
//!
//! The search does not enumerate every full mode combination up front.
//! Instead, it grows partial answers one leg at a time and always expands the
//! cheapest partial answer first.
//!
//! The flow is:
//! 1. Break one chain into smaller home-to-home segments when possible.
//! 2. Search each segment separately.
//! 3. Merge the best segment answers back into full-chain answers.
//! 4. Remove the low-probability tail of the final answer list.
//! 5. Emit one output row per retained leg.
//!
//! Conceptual pseudocode:
//!
//! ```text
//! for each chain:
//!   break the chain into smaller home-to-home segments
//!
//!   for each segment:
//!     put one empty partial answer into a priority queue
//!
//!     while the queue is not empty:
//!       take the currently cheapest partial answer
//!
//!       if it already covers every leg:
//!         if all vehicles are back home:
//!           save it as a feasible full answer
//!         continue
//!
//!       look up the next leg's available modes
//!
//!       for each allowed mode:
//!         update vehicle positions
//!         update any forced future return-mode rule
//!
//!         if the extended partial answer is still feasible:
//!           push it back into the queue
//!
//!   merge the best answers from each segment
//!   prune the low-probability tail
//! ```
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use rayon::prelude::*;

use crate::errors::SearchError;
use crate::index::SearchIndex;
use crate::input::LocationChain;
use crate::output::OutputTable;

const COST_RESCALE_FACTOR: f64 = 1e6;

// One fully chosen mode sequence with its total generalized cost.
#[derive(Clone, Debug)]
struct ResultSequence {
    cost: f64,
    sequence: Vec<u16>,
}

// One partially explored path through a chain.
//
// `leg_idx` says how many legs we have already assigned modes to.
// `vehicle_locations` tracks where each shared vehicle currently is.
// `mode_sequence` stores the chosen mode ids so far.
// `return_mode_constraints` stores "leg N must use mode X" rules introduced
// by multimodal outbound legs that require a matching return leg later.
#[derive(Clone, Debug)]
struct SearchState {
    leg_idx: usize,
    vehicle_locations: Vec<u32>,
    mode_sequence: Vec<u16>,
    return_mode_constraints: Vec<Option<u16>>,
}

// Wrapper used in the main best-first search heap.
//
// Rust's `BinaryHeap` is a max-heap by default, so the `Ord` implementation
// below reverses comparisons to make the cheapest state pop first.
#[derive(Clone, Debug)]
struct HeapState {
    cost: f64,
    counter: usize,
    state: SearchState,
}

impl PartialEq for HeapState {
    fn eq(&self, other: &Self) -> bool {
        self.cost.total_cmp(&other.cost) == Ordering::Equal && self.counter == other.counter
    }
}

impl Eq for HeapState {}

impl PartialOrd for HeapState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.counter.cmp(&self.counter))
    }
}

// Wrapper used when merging independently searched chain segments. Each heap
// entry points at one pair of partial results from the left and right sides.
#[derive(Clone, Debug)]
struct SegmentMergeState {
    cost: f64,
    i: usize,
    j: usize,
}

impl PartialEq for SegmentMergeState {
    fn eq(&self, other: &Self) -> bool {
        self.cost.total_cmp(&other.cost) == Ordering::Equal && self.i == other.i && self.j == other.j
    }
}

impl Eq for SegmentMergeState {}

impl PartialOrd for SegmentMergeState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SegmentMergeState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.i.cmp(&self.i))
            .then_with(|| other.j.cmp(&self.j))
    }
}

fn split_chain_at_home_returns(locations: &[u32]) -> Vec<Vec<u32>> {
    // Example:
    // [10, 11, 10, 12, 10] becomes [[10, 11, 10], [10, 12, 10]].
    //
    // Each home-to-home segment can be searched on its own, then later merged.
    let mut segments = Vec::new();
    let home = locations[0];
    let mut current = vec![home];

    for &loc in &locations[1..] {
        current.push(loc);
        if loc == home && current.len() > 1 {
            segments.push(current);
            current = vec![home];
        }
    }

    if current.len() > 1 {
        segments.push(current);
    }

    segments
}

fn find_repeated_location_ranges(locations: &[u32]) -> Vec<(usize, usize)> {
    // Whenever we revisit the same location, we have found a closed slice of
    // the chain that might behave like a subtour.
    let mut last_seen: HashMap<u32, usize> = HashMap::new();
    let mut subtours = Vec::new();

    for (end_idx, place) in locations.iter().copied().enumerate() {
        if let Some(start_idx) = last_seen.get(&place).copied() {
            // Treat the full home-to-home segment as a valid subtour so
            // multimodal outbound legs can force their matching return mode.
            subtours.push((start_idx, end_idx));
        }
        last_seen.insert(place, end_idx);
    }

    subtours
}

fn weight_from_cost(cost: f64, best_cost: f64) -> f64 {
    (-(cost - best_cost) / COST_RESCALE_FACTOR).exp()
}

fn can_stop_early(results: &[ResultSequence], heap: &BinaryHeap<HeapState>, k: usize, threshold: f64) -> bool {
    // Early stopping is only meaningful once we have at least one complete
    // result and there are still unexplored states in the heap.
    if results.is_empty() || results.len() >= k || heap.is_empty() {
        return false;
    }

    // We compute a conservative lower bound on how much probability mass the
    // current results already cover. If even the best possible unseen states
    // cannot change the threshold decision, we can stop exploring.
    let best_cost = results[0].cost;
    let found_weight: f64 = results.iter().map(|result| weight_from_cost(result.cost, best_cost)).sum();
    let remaining_slots = k - results.len();
    let next_cost_lower_bound = heap.peek().map(|item| item.cost).unwrap_or(best_cost);
    let max_remaining_weight = remaining_slots as f64 * weight_from_cost(next_cost_lower_bound, best_cost);
    let covered_mass_lower_bound = found_weight / (found_weight + max_remaining_weight);
    covered_mass_lower_bound >= threshold
}

fn merge_two_mode_sequences(left: &[ResultSequence], right: &[ResultSequence], k: usize) -> Vec<ResultSequence> {
    // If either side has no feasible results, the combined segment also has no
    // feasible results.
    if left.is_empty() || right.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    let mut heap = BinaryHeap::new();

    // Start from the cheapest combination: first result on the left plus first
    // result on the right. Then expand neighboring pairs lazily.
    heap.push(SegmentMergeState {
        cost: left[0].cost + right[0].cost,
        i: 0,
        j: 0,
    });
    seen.insert((0, 0));

    while let Some(entry) = heap.pop() {
        let mut sequence = left[entry.i].sequence.clone();
        sequence.extend_from_slice(&right[entry.j].sequence);
        out.push(ResultSequence {
            cost: entry.cost,
            sequence,
        });

        if out.len() >= k {
            break;
        }

        if entry.i + 1 < left.len() && seen.insert((entry.i + 1, entry.j)) {
            heap.push(SegmentMergeState {
                cost: left[entry.i + 1].cost + right[entry.j].cost,
                i: entry.i + 1,
                j: entry.j,
            });
        }

        if entry.j + 1 < right.len() && seen.insert((entry.i, entry.j + 1)) {
            heap.push(SegmentMergeState {
                cost: left[entry.i].cost + right[entry.j + 1].cost,
                i: entry.i,
                j: entry.j + 1,
            });
        }
    }

    out
}

fn merge_mode_sequences_list(lists_of_lists: &[Vec<ResultSequence>], k: usize) -> Vec<ResultSequence> {
    // Reduce many independently searched segments into one top-k list by
    // repeatedly merging two lists at a time.
    let mut iter = lists_of_lists.iter();
    let Some(first) = iter.next() else {
        return Vec::new();
    };

    let mut current = first.clone();
    current.sort_by(|a, b| a.cost.total_cmp(&b.cost).then_with(|| a.sequence.cmp(&b.sequence)));

    for next in iter {
        let mut sorted_next = next.clone();
        sorted_next.sort_by(|a, b| a.cost.total_cmp(&b.cost).then_with(|| a.sequence.cmp(&b.sequence)));
        current = merge_two_mode_sequences(&current, &sorted_next, k);
    }

    current.truncate(k);
    current
}

fn collect_symmetric_subtour_return_constraints(locations: &[u32]) -> HashMap<usize, usize> {
    // We only add return constraints for a simple symmetric pattern like
    // A -> B -> A. In that case, if the outbound leg picks a multimodal mode,
    // the matching return leg can be forced later.
    let subtours = find_repeated_location_ranges(locations);
    let mut map = HashMap::new();
    for (start, end) in subtours {
        if end > start + 1 && locations[start] == locations[end] && locations[start + 1] == locations[end - 1] {
            map.insert(start, end);
        }
    }
    map
}

fn search_chain_segment(
    index: &SearchIndex,
    locations: &[u32],
    k: usize,
    threshold: f64,
) -> Result<Vec<ResultSequence>, SearchError> {
    // Fast path: a segment with exactly one leg does not need heap search.
    if locations.len() == 2 {
        let options = index
            .edge_options(locations[0], locations[1])
            .ok_or(SearchError::MissingEdge {
                origin: locations[0],
                destination: locations[1],
            })?;
        return Ok(options
            .iter()
            .map(|option| ResultSequence {
                cost: option.cost,
                sequence: vec![option.mode_id],
            })
            .collect());
    }

    let n_legs = locations.len() - 1;
    let forced_return_legs_by_outbound_leg = collect_symmetric_subtour_return_constraints(locations);
    let mut counter = 0usize;
    let mut heap = BinaryHeap::new();
    let mut results = Vec::new();
    let empty_return_constraints = vec![None; n_legs];

    // Seed the heap with an empty path:
    // - no legs assigned yet
    // - every tracked vehicle starts at home
    // - no forced return-mode constraints yet
    heap.push(HeapState {
        cost: 0.0,
        counter,
        state: SearchState {
            leg_idx: 0,
            vehicle_locations: vec![locations[0]; index.n_vehicles],
            mode_sequence: Vec::with_capacity(n_legs),
            return_mode_constraints: empty_return_constraints,
        },
    });

    while let Some(entry) = heap.pop() {
        let SearchState {
            leg_idx,
            vehicle_locations,
            mode_sequence,
            return_mode_constraints,
        } = entry.state;

        // If we have assigned a mode to every leg, the only remaining question
        // is whether every borrowed vehicle made it back home.
        if leg_idx == n_legs {
            if vehicle_locations.iter().all(|location| *location == locations[0]) {
                results.push(ResultSequence {
                    cost: entry.cost,
                    sequence: mode_sequence,
                });
                if can_stop_early(&results, &heap, k, threshold) {
                    break;
                }
            }
            continue;
        }

        let current_location = locations[leg_idx];
        let next_location = locations[leg_idx + 1];

        // Look up every allowed mode on this leg once. Missing edges are a hard
        // schema/data error because the search cannot continue without them.
        let options = index
            .edge_options(current_location, next_location)
            .ok_or(SearchError::MissingEdge {
                origin: current_location,
                destination: next_location,
            })?;

        let enforced_mode = return_mode_constraints[leg_idx];
        let mut saw_candidate = false;
        for option in options.iter().filter(|option| enforced_mode.is_none_or(|mode_id| option.mode_id == mode_id)) {
            saw_candidate = true;
            let mode = index.mode(option.mode_id)?;
            let mut next_vehicle_locations = vehicle_locations.clone();
            let mut next_return_mode_constraints = return_mode_constraints.clone();

            // Vehicle-backed modes are only feasible if their vehicle is currently
            // at the leg origin. Multimodal outbound legs also force the matching
            // return mode later in the same symmetric subtour.
            if mode.needs_vehicle {
                let vehicle_index = mode.vehicle_index.ok_or(SearchError::MissingMode(option.mode_id))?;
                if next_vehicle_locations[vehicle_index] != current_location {
                    continue;
                }
                next_vehicle_locations[vehicle_index] = next_location;

                if mode.multimodal && !mode.is_return_mode {
                    let Some(subtour_end) = forced_return_legs_by_outbound_leg.get(&leg_idx).copied() else {
                        continue;
                    };
                    let return_mode_id = mode.return_mode_id.ok_or(SearchError::MissingMode(option.mode_id))?;
                    // `subtour_end - 1` is the last leg before we return to the
                    // repeated location, i.e. the actual return leg.
                    next_return_mode_constraints[subtour_end - 1] = Some(return_mode_id);
                }
            }

            counter += 1;
            let mut next_sequence = mode_sequence.clone();
            next_sequence.push(option.mode_id);

            // Push the extended partial solution back into the heap so it can be
            // explored later in cost order.
            heap.push(HeapState {
                cost: entry.cost + option.cost,
                counter,
                state: SearchState {
                    leg_idx: leg_idx + 1,
                    vehicle_locations: next_vehicle_locations,
                    mode_sequence: next_sequence,
                    return_mode_constraints: next_return_mode_constraints,
                },
            });
        }

        if !saw_candidate {
            continue;
        }

        if results.len() >= k {
            break;
        }
    }

    Ok(results)
}

fn prune_results(results: &mut Vec<ResultSequence>, threshold: f64) {
    // Match the Python implementation: stabilize ties first, then discard the tail
    // once the retained alternatives cover the requested probability mass.
    for result in results.iter_mut() {
        result.cost = (result.cost * 1e9).round() / 1e9;
    }
    results.sort_by(|a, b| a.cost.total_cmp(&b.cost).then_with(|| a.sequence.cmp(&b.sequence)));

    let mut best_utility = f64::NEG_INFINITY;
    let mut utilities = Vec::with_capacity(results.len());
    for result in results.iter() {
        let utility = -result.cost / COST_RESCALE_FACTOR;
        best_utility = best_utility.max(utility);
        utilities.push(utility);
    }

    // Convert utilities into normalized probabilities using a stable softmax.
    let probs: Vec<f64> = utilities.iter().map(|utility| (utility - best_utility).exp()).collect();
    let total_prob: f64 = probs.iter().sum();
    let mut cumulative = 0.0;
    let mut keep = 0usize;
    for prob in probs {
        cumulative += prob / total_prob;
        keep += 1;
        if cumulative >= threshold {
            break;
        }
    }
    keep = keep.max(1).min(results.len());
    results.truncate(keep);
}

fn search_full_chain(
    index: &SearchIndex,
    chain: &LocationChain,
    k: usize,
    threshold: f64,
) -> Result<OutputTable, SearchError> {
    // Home-to-home segments can be searched independently and then merged as
    // top-k Cartesian products, which keeps the heap search smaller.
    let segments = split_chain_at_home_returns(&chain.locations);
    let mut segment_results = Vec::with_capacity(segments.len());
    for segment in segments {
        let results = search_chain_segment(index, &segment, k, threshold)?;
        segment_results.push(results);
    }

    let mut results = merge_mode_sequences_list(&segment_results, k);
    if results.is_empty() {
        return Ok(OutputTable::default());
    }

    prune_results(&mut results, threshold);

    // Convert each retained sequence into one output row per leg so the Python
    // side receives a simple columnar long-form table.
    let mut output = OutputTable::default();
    for (mode_seq_index, result) in results.iter().enumerate() {
        for (leg_idx, mode_id) in result.sequence.iter().copied().enumerate() {
            output.push_row(
                chain.dest_seq_id,
                mode_seq_index as u32,
                (leg_idx + 1) as u32,
                chain.locations[leg_idx + 1],
                mode_id,
            );
        }
    }
    Ok(output)
}

/// Compute mode-sequence outputs for all chains, optionally using a custom
/// Rayon thread count.
///
/// # Errors
/// Returns the first search or schema-derived error encountered while
/// processing the input chains.
pub fn compute_all(
    index: &SearchIndex,
    chains: &[LocationChain],
    k: usize,
    threshold: f64,
    n_threads: Option<usize>,
) -> Result<OutputTable, SearchError> {
    // Parallelism lives here, across independent chains, so Python does not need
    // to manage a separate process pool.
    let compute = || -> Result<Vec<OutputTable>, SearchError> {
        chains
            .par_iter()
            .map(|chain| search_full_chain(index, chain, k, threshold))
            .collect()
    };

    let outputs = match n_threads {
        Some(thread_count) if thread_count > 0 => rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .map_err(|err| SearchError::InvalidSchema(format!("failed to build thread pool: {err}")))?
            .install(compute)?,
        _ => compute()?,
    };

    let mut merged = OutputTable::default();
    for output in outputs {
        merged.extend(output);
    }
    Ok(merged)
}
