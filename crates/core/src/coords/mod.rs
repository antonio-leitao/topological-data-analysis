//! Circular and toroidal coordinates from persistent cohomology.
//!
//! One selected H¹ class yields a circular coordinate; several jointly
//! decorrelated classes yield a toroidal one. Every input point remains a
//! vertex; the implementation does not use landmarks.

mod circular;

pub use circular::{circular, circular_map, circular_map_with, circular_with, CircularMap};

use crate::types::PersistenceInterval;

/// Options for the coordinate functions.
#[derive(Debug, Clone)]
pub struct Config {
    /// Interpret `data` as a row-major `n × n` distance matrix (same
    /// contract as [`crate::Config`]).
    pub distance_matrix: bool,
    /// Use the log(3)-interleaved quotient-cover filtration. The coned
    /// skeleton retains every vertex. Default `false`.
    pub quotient: bool,
    /// Evaluation scale. `None` uses each class interval's midpoint. Selected
    /// classes that are not alive at an explicit scale are dropped.
    pub scale: Option<f32>,
    /// Top-k classes by persistence. Default 1.
    pub n_classes: usize,
    /// Explicit persistence-ranked class indices; overrides `n_classes`.
    pub classes: Option<Vec<usize>>,
    /// Odd prime for the coefficient field behind the integer lift.
    /// Default 41.
    pub prime: u32,
    /// Rayon in the underlying persistence computation. Default true.
    pub parallel: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            distance_matrix: false,
            quotient: false,
            scale: None,
            n_classes: 1,
            classes: None,
            prime: 41,
            parallel: true,
        }
    }
}

/// Circular coordinates: one row per input point, row-major `(n, k)` angles
/// in `[0, 2π)`, one column per selected class.
#[derive(Debug, Clone)]
pub struct CircularCoordinates {
    pub coordinates: Vec<f32>,
    pub k: usize,
    /// Per column: the class interval, evaluation scale, and toroidal block
    /// id (columns sharing a `group` were jointly decorrelated; singleton
    /// groups are plain circular coordinates).
    pub classes: Vec<SelectedClass>,
}

/// Interval and evaluation data of one selected class (a column of
/// [`CircularCoordinates`]).
#[derive(Debug, Clone, Copy)]
pub struct SelectedClass {
    pub birth: f32,
    pub death: f32,
    pub scale: f32,
    pub group: usize,
}

/// Class indices ranked by persistence (descending), deterministic tie-break
/// by (birth, index). `death_cap` stands in for infinite deaths.
fn rank_classes(intervals: &[PersistenceInterval], death_cap: f32) -> Vec<usize> {
    let mut ranked: Vec<usize> = (0..intervals.len()).collect();
    ranked.sort_by(|&a, &b| {
        let pa = persistence(&intervals[a], death_cap);
        let pb = persistence(&intervals[b], death_cap);
        pb.total_cmp(&pa)
            .then(intervals[a].birth.total_cmp(&intervals[b].birth))
            .then(a.cmp(&b))
    });
    ranked
}

#[inline]
fn persistence(interval: &PersistenceInterval, death_cap: f32) -> f32 {
    let death = if interval.death.is_finite() {
        interval.death
    } else {
        death_cap
    };
    death - interval.birth
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranking_is_by_persistence_with_deterministic_ties() {
        let iv = |birth: f32, death: f32| PersistenceInterval { birth, death };
        let intervals = [
            iv(0.5, 1.0),
            iv(0.2, 1.7),
            iv(0.1, f32::INFINITY),
            iv(0.0, 0.5),
        ];
        let ranked = rank_classes(&intervals, 1.0);
        assert_eq!(ranked, vec![1, 2, 3, 0]);
    }
}
