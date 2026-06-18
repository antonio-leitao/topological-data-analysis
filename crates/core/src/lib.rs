mod engine;
mod error;
mod opt;
mod preprocess;
mod types;
mod utils;

use preprocess::edgelist::EdgeList;

pub use error::{Error, Result};
pub use types::{BarcodeResult, PersistenceInterval};

pub const MAX_POINTS: usize = u16::MAX as usize;
pub const MAX_DIM: usize = 4;

// ═══════════════════════════════════════════════════════════════════════════════
// Persistent homology
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute Vietoris–Rips persistent homology from a row-major `(n, d)`
/// point cloud using the BitCSR sparse backend.
///
/// `peel` tightens the truncation radius via a sound strong collapse
/// ([`opt::peel`]) — strictly cheaper, identical barcode. `quotient` rewrites
/// the filtration into a quotient-cover (complete-linkage) coning
/// ([`opt::coperto`]) before reduction — strictly smaller, and `log 3`-interleaved
/// with VR rather than equal. The two compose: when both are set, `peel` runs
/// first and `quotient` reuses its sort.
pub fn persistent_homology(
    points: &[f32],
    n: usize,
    d: usize,
    max_dim: usize,
    threshold: Option<f32>,
    quotient: bool,
    peel: bool,
) -> Result<BarcodeResult> {
    validate_params(n, max_dim, threshold)?;
    let edges = edges_from_points(points, n, d, threshold, quotient, peel)?;
    let bitcsr = engine::BitCsrDistanceMatrix::from_edge_list(edges);
    Ok(engine::algorithm::compute(&bitcsr, max_dim, true))
}

/// Compute Vietoris–Rips persistent homology from a row-major `(n, n)`
/// distance matrix. Symmetry, zero diagonal, and non-negativity are
/// documented preconditions, not enforced invariants.
///
/// See [`persistent_homology`] for the meaning of `peel` and `quotient`.
pub fn persistent_homology_from_distances(
    distances: &[f32],
    n: usize,
    max_dim: usize,
    threshold: Option<f32>,
    quotient: bool,
    peel: bool,
) -> Result<BarcodeResult> {
    validate_params(n, max_dim, threshold)?;
    let edges = edges_from_distances(distances, n, threshold, quotient, peel)?;
    let bitcsr = engine::BitCsrDistanceMatrix::from_edge_list(edges);
    Ok(engine::algorithm::compute(&bitcsr, max_dim, true))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Filtration size
// ═══════════════════════════════════════════════════════════════════════════════

pub fn filtration_size(
    points: &[f32],
    n: usize,
    d: usize,
    max_dim: usize,
    threshold: Option<f32>,
    quotient: bool,
    peel: bool,
) -> Result<usize> {
    validate_params(n, max_dim, threshold)?;
    let edges = edges_from_points(points, n, d, threshold, quotient, peel)?;
    Ok(utils::cliques::count_cliques(edges, max_dim + 2))
}

pub fn filtration_size_from_distances(
    distances: &[f32],
    n: usize,
    max_dim: usize,
    threshold: Option<f32>,
    quotient: bool,
    peel: bool,
) -> Result<usize> {
    validate_params(n, max_dim, threshold)?;
    let edges = edges_from_distances(distances, n, threshold, quotient, peel)?;
    Ok(utils::cliques::count_cliques(edges, max_dim + 2))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shared edge-list construction
// ═══════════════════════════════════════════════════════════════════════════════

/// Build the final `EdgeList` for a point cloud: distances → optional `peel` →
/// optional `quotient` coning. The single boundary shared by the homology and
/// filtration-size paths and by both input modes.
fn edges_from_points(
    points: &[f32],
    n: usize,
    d: usize,
    threshold: Option<f32>,
    quotient: bool,
    peel: bool,
) -> Result<EdgeList> {
    validate_point_shape(points, n, d)?;
    let mut edges = EdgeList::from_points(points, n, d, threshold.unwrap_or(f32::INFINITY));
    apply_optimizations(&mut edges, peel, quotient);
    Ok(edges)
}

/// Build the final `EdgeList` for a distance matrix. See [`edges_from_points`].
fn edges_from_distances(
    distances: &[f32],
    n: usize,
    threshold: Option<f32>,
    quotient: bool,
    peel: bool,
) -> Result<EdgeList> {
    validate_distance_shape(distances, n)?;
    let mut edges =
        EdgeList::from_distance_matrix(distances, n, threshold.unwrap_or(f32::INFINITY));
    apply_optimizations(&mut edges, peel, quotient);
    Ok(edges)
}

/// Apply the optional transforms in canonical order: `peel` first (an exact,
/// barcode-preserving truncation that tightens the radius and leaves the edges
/// sorted), then `quotient` coning (`coperto`, which reuses that sort).
#[inline]
fn apply_optimizations(edges: &mut EdgeList, peel: bool, quotient: bool) {
    if peel {
        opt::peel::peel(edges);
    }
    if quotient {
        opt::coperto::cone_in_place(edges);
    }
}

#[inline]
fn validate_point_shape(points: &[f32], n: usize, d: usize) -> Result<()> {
    if d == 0 {
        return Err(Error::EmptyDimension);
    }
    if points.len() != n * d {
        return Err(Error::ShapeMismatch {
            n,
            got: points.len(),
            mode: "points (expected n*d)",
        });
    }
    Ok(())
}

#[inline]
fn validate_distance_shape(distances: &[f32], n: usize) -> Result<()> {
    if distances.len() != n * n {
        return Err(Error::ShapeMismatch {
            n,
            got: distances.len(),
            mode: "distance matrix (expected n*n)",
        });
    }
    Ok(())
}

#[inline]
fn validate_params(n: usize, max_dim: usize, threshold: Option<f32>) -> Result<()> {
    if n < 2 {
        return Err(Error::TooFewPoints { got: n });
    }
    if n > MAX_POINTS {
        return Err(Error::TooManyPoints {
            got: n,
            max: MAX_POINTS,
        });
    }
    if max_dim > MAX_DIM {
        return Err(Error::DimTooLarge {
            got: max_dim,
            max: MAX_DIM,
        });
    }
    if let Some(t) = threshold {
        if !(t >= 0.0) {
            return Err(Error::InvalidThreshold(t));
        }
    }
    Ok(())
}

#[cfg(test)]
mod pipeline_tests {
    use super::*;

    fn assert_same_barcode(a: &BarcodeResult, b: &BarcodeResult) {
        assert_eq!(a.intervals.len(), b.intervals.len());
        for (left, right) in a.intervals.iter().zip(&b.intervals) {
            let mut left: Vec<_> = left
                .iter()
                .map(|interval| (interval.birth.to_bits(), interval.death.to_bits()))
                .collect();
            let mut right: Vec<_> = right
                .iter()
                .map(|interval| (interval.birth.to_bits(), interval.death.to_bits()))
                .collect();
            left.sort_unstable();
            right.sort_unstable();
            assert_eq!(left, right);
        }
    }

    #[test]
    fn point_pipeline_peel_preserves_barcode() {
        let points = [
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05, 0.2, 0.05, 0.05, 0.2,
        ];
        let n = points.len() / 2;
        let plain = persistent_homology(&points, n, 2, 2, None, false, false).unwrap();
        let peeled = persistent_homology(&points, n, 2, 2, None, false, true).unwrap();
        assert_same_barcode(&plain, &peeled);
    }

    #[test]
    fn distance_pipeline_peel_preserves_barcode() {
        let lower = [1.5, 1.05, 1.0, 1.05, 1.5, 1.0, 1.5, 1.1, 1.5, 1.0];
        let n = 5;
        let mut distances = vec![0.0; n * n];
        let mut k = 0;
        for i in 1..n {
            for j in 0..i {
                distances[i * n + j] = lower[k];
                distances[j * n + i] = lower[k];
                k += 1;
            }
        }

        let plain =
            persistent_homology_from_distances(&distances, n, 1, Some(1.5), false, false).unwrap();
        let peeled =
            persistent_homology_from_distances(&distances, n, 1, Some(1.5), false, true).unwrap();
        assert_same_barcode(&plain, &peeled);
    }

    #[test]
    fn finite_cutoff_peel_preserves_barcode() {
        // Four-cycle at scale 1 with both diagonals outside the 1.5 cutoff. The
        // stored center has vertices outside its starting ball.
        let lower = [1.0, 10.0, 1.0, 1.0, 10.0, 1.0];
        let n = 4;
        let mut distances = vec![0.0; n * n];
        let mut k = 0;
        for i in 1..n {
            for j in 0..i {
                distances[i * n + j] = lower[k];
                distances[j * n + i] = lower[k];
                k += 1;
            }
        }

        let plain =
            persistent_homology_from_distances(&distances, n, 1, Some(1.5), false, false).unwrap();
        let peeled =
            persistent_homology_from_distances(&distances, n, 1, Some(1.5), false, true).unwrap();
        assert_same_barcode(&plain, &peeled);
    }

    #[test]
    fn filtration_size_consumes_edge_list_for_both_input_modes() {
        let points = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let diagonal = std::f32::consts::SQRT_2;
        let distances = [
            0.0, 1.0, diagonal, 1.0, 1.0, 0.0, 1.0, diagonal, diagonal, 1.0, 0.0, 1.0, 1.0,
            diagonal, 1.0, 0.0,
        ];

        let point_count = filtration_size(&points, 4, 2, 1, Some(1.0), false, false).unwrap();
        let matrix_count =
            filtration_size_from_distances(&distances, 4, 1, Some(1.0), false, false).unwrap();
        assert_eq!(point_count, 8); // 4 vertices + 4 cycle edges
        assert_eq!(matrix_count, point_count);

        let peeled = filtration_size(&points, 4, 2, 1, None, false, true).unwrap();
        let unpeeled = filtration_size(&points, 4, 2, 1, None, false, false).unwrap();
        assert!(peeled <= unpeeled);
    }

    #[test]
    fn quotient_filtration_matches_input_modes() {
        let points = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let diagonal = std::f32::consts::SQRT_2;
        let distances = [
            0.0, 1.0, diagonal, 1.0, 1.0, 0.0, 1.0, diagonal, diagonal, 1.0, 0.0, 1.0, 1.0,
            diagonal, 1.0, 0.0,
        ];

        let from_points = filtration_size(&points, 4, 2, 1, None, true, true).unwrap();
        let from_matrix =
            filtration_size_from_distances(&distances, 4, 1, None, true, true).unwrap();
        assert_eq!(from_points, from_matrix);
    }

    #[test]
    fn quotient_persistent_homology_matches_input_modes() {
        // Regression anchor for the (previously `todo!()`) sparse quotient path:
        // both input modes feed coperto the same edge set, so the barcodes agree.
        let points = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let diagonal = std::f32::consts::SQRT_2;
        let distances = [
            0.0, 1.0, diagonal, 1.0, 1.0, 0.0, 1.0, diagonal, diagonal, 1.0, 0.0, 1.0, 1.0,
            diagonal, 1.0, 0.0,
        ];

        let from_points = persistent_homology(&points, 4, 2, 1, None, true, true).unwrap();
        let from_matrix =
            persistent_homology_from_distances(&distances, 4, 1, None, true, true).unwrap();
        assert_same_barcode(&from_points, &from_matrix);
    }
}
