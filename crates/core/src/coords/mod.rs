//! Coordinates from persistent cohomology classes.
//!
//! Maps data into the geometric target spaces classified by low-degree
//! cohomology: H¹ mod 2 → real projective space RPᵈ ([`projective`]), H¹ over
//! an odd prime → S¹ or jointly Tᵏ ([`circular`]), and H² over an odd prime
//! → complex projective space CPᵈ ([`complex_projective`]).
//!
//! Coordinates are computed on the FULL complex: every data point is a
//! vertex, so values are assigned directly — no landmark subsampling and no
//! extension step. Peel is always applied internally (it only truncates the
//! filtration tail and every class dies before the certified radius, so it
//! is transparent to coordinates and always beneficial).

mod circular;
mod complex_projective;
mod projective;

pub use circular::{circular, circular_map, circular_map_with, circular_with, CircularMap};
pub use complex_projective::{
    complex_projective, complex_projective_map, complex_projective_map_with,
    complex_projective_with, ComplexProjectiveMap,
};
pub use projective::{
    projective, projective_map, projective_map_with, projective_with, ProjectiveMap,
};

use crate::types::PersistenceInterval;
use crate::{Error, Result};

/// A projective map starts in `n` homogeneous coordinates, so a reduced
/// target can have projective dimension at most `n - 1`. Keeping this check
/// in the core protects Rust and every language binding from oversized basis
/// requests before the projection code runs.
fn validate_target_dim(target_dim: usize, n: usize) -> Result<()> {
    let max = n.saturating_sub(1);
    if target_dim == 0 || (n >= 2 && target_dim > max) {
        return Err(Error::InvalidTargetDim {
            got: target_dim,
            max,
            points: n,
        });
    }
    Ok(())
}

/// Options for the coordinates functions. One struct serves every entry
/// point; fields irrelevant to a function are ignored by it.
///
/// Deliberately absent: `collapse`, `threshold` (the certified radius is
/// always the right cutoff here), and any landmark knob (subsampling is
/// user-side preprocessing).
#[derive(Debug, Clone)]
pub struct Config {
    /// Interpret `data` as a row-major `n × n` distance matrix (same
    /// contract as [`crate::Config`]).
    pub distance_matrix: bool,
    /// Compute on the quotient-cover (coperto) filtration instead of the
    /// full VR complex: massively cheaper (near-linear skeleton), at the
    /// price of the log-3 interleaving — class selection stays robust, but
    /// intervals and scales are quotient values and the harmonic geometry
    /// is blockier (near-constant per cluster, varying along the merge
    /// structure). Every original point still receives its own value: the
    /// coned skeleton retains all vertices. Default `false`.
    pub quotient: bool,
    /// Evaluation scale. `None` picks the midpoint of the class interval.
    /// An explicit value must lie inside `[birth, death)` of every selected
    /// class.
    pub scale: Option<f32>,
    /// (`circular`) Top-k classes by persistence. Default 1.
    pub n_classes: usize,
    /// (`circular`) Explicit persistence-ranked class indices; overrides
    /// `n_classes`.
    pub classes: Option<Vec<usize>>,
    /// (`circular`) Odd prime for the coefficient field behind the integer
    /// lift. Default 41.
    pub prime: u32,
    /// (`projective`) Dimension of the target projective space RPᵈ. The
    /// output rows are unit vectors in R^(d+1), defined up to sign.
    /// Default 2.
    pub target_dim: usize,
    /// (`projective`) Persistence rank of the class to use. Default 0 (the
    /// most persistent).
    pub class_index: usize,
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
            target_dim: 2,
            class_index: 0,
            parallel: true,
        }
    }
}

/// Real projective coordinates: one row per input point, row-major
/// `(n, target_dim + 1)`, unit rows defined up to sign.
#[derive(Debug, Clone)]
pub struct ProjectiveCoordinates {
    pub coordinates: Vec<f32>,
    pub target_dim: usize,
    /// Interval of the class used (death may be infinite only under a user
    /// threshold, which this pipeline never sets).
    pub birth: f32,
    pub death: f32,
    /// The scale the map was computed at.
    pub scale: f32,
}

/// Complex projective coordinates: one row per input point, row-major
/// `(n, target_dim + 1)` COMPLEX unit rows stored as interleaved
/// `(re, im)` f32 pairs (flat length `n · (target_dim + 1) · 2`), each row
/// defined up to a global phase.
#[derive(Debug, Clone)]
pub struct ComplexProjectiveCoordinates {
    pub coordinates: Vec<f32>,
    pub target_dim: usize,
    /// Interval of the H² class used.
    pub birth: f32,
    pub death: f32,
    /// The scale the map was computed at.
    pub scale: f32,
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

// ═══════════════════════════════════════════════════════════════════════════════
// Shared selection helpers
// ═══════════════════════════════════════════════════════════════════════════════

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
            iv(0.5, 1.0),           // persistence 0.5
            iv(0.2, 1.7),           // persistence 1.5 — rank 0
            iv(0.1, f32::INFINITY), // capped at 1.0 → persistence 0.9
            iv(0.0, 0.5),           // persistence 0.5, earlier birth than [0]
        ];
        let ranked = rank_classes(&intervals, 1.0);
        assert_eq!(ranked, vec![1, 2, 3, 0]);
    }
}
