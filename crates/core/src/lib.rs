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

/// Vietoris–Rips persistent homology over the BitCSR sparse backend.
///
/// `data` is a row-major `&[f32]` whose length is:
///   * `n * n` when `distance_matrix` is true — the strict lower triangle is
///     read; symmetry, zero diagonal, and non-negativity are documented
///     preconditions, not enforced invariants, and `f32::INFINITY` is a valid
///     "disconnected" entry; or
///   * `n * d` otherwise — a point cloud, with `d` inferred as `data.len() / n`.
///
/// The two layouts collide at `d == n` (square data): they are byte-identical,
/// so `distance_matrix` is the *only* thing that disambiguates them — pass it
/// correctly. (The Python layer warns on this case; the core cannot, since its
/// hot caller is the benchmark loop.)
///
/// `peel` tightens the truncation radius via a sound strong collapse
/// ([`opt::peel`]) — strictly cheaper, identical barcode. `quotient` rewrites the
/// filtration into a quotient-cover (complete-linkage) coning ([`opt::coperto`]) —
/// strictly smaller, and `log 3`-interleaved with VR rather than equal. When both
/// are set, `peel` runs first and `quotient` reuses its sort. `parallel` enables
/// Rayon in sufficiently large point preprocessing, edge sorts, and engine
/// candidate assembly.
pub fn persistent_homology(
    data: &[f32],
    n: usize,
    max_dim: usize,
    threshold: Option<f32>,
    distance_matrix: bool,
    quotient: bool,
    peel: bool,
    parallel: bool,
) -> Result<BarcodeResult> {
    let edges = build_edges(
        data,
        n,
        max_dim,
        threshold,
        distance_matrix,
        quotient,
        peel,
        parallel,
    )?;
    Ok(compute_barcode(edges, max_dim, parallel))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Filtration size
// ═══════════════════════════════════════════════════════════════════════════════
//
// See [`persistent_homology`] for the `data` / `distance_matrix` layout contract
// and the meaning of `peel` / `quotient` / `parallel`.

pub fn filtration_size(
    data: &[f32],
    n: usize,
    max_dim: usize,
    threshold: Option<f32>,
    distance_matrix: bool,
    quotient: bool,
    peel: bool,
    parallel: bool,
) -> Result<usize> {
    let edges = build_edges(
        data,
        n,
        max_dim,
        threshold,
        distance_matrix,
        quotient,
        peel,
        parallel,
    )?;
    Ok(utils::cliques::count_cliques(edges, max_dim + 2, parallel))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shared construction (the hourglass waist)
// ═══════════════════════════════════════════════════════════════════════════════

/// Validate, build the truncated `EdgeList` for whichever input layout, then
/// apply the optional transforms. The single boundary shared by the homology and
/// filtration-size paths and by both input modes.
///
/// `max_dim` is validated up front, before the O(n²·d) distance build, so an
/// out-of-range value fails fast rather than after the work.
fn build_edges(
    data: &[f32],
    n: usize,
    max_dim: usize,
    threshold: Option<f32>,
    distance_matrix: bool,
    quotient: bool,
    peel: bool,
    parallel: bool,
) -> Result<EdgeList> {
    validate_params(n, max_dim, threshold)?;
    let t = threshold.unwrap_or(f32::INFINITY);

    let mut edges = if distance_matrix {
        validate_distance_shape(data, n)?;
        EdgeList::from_distance_matrix(data, n, t)
    } else {
        let d = infer_dimension(data, n)?;
        EdgeList::from_points(data, n, d, t, parallel)
    };

    apply_optimizations(&mut edges, peel, quotient, parallel);
    Ok(edges)
}

/// Apply the optional transforms in canonical order: `peel` first (exact,
/// barcode-preserving, leaves the edges sorted), then `quotient` coning
/// (`coperto`, which reuses that sort).
#[inline]
fn apply_optimizations(edges: &mut EdgeList, peel: bool, quotient: bool, parallel: bool) {
    if peel {
        opt::peel::peel(edges, parallel);
    }
    if quotient {
        opt::coperto::cone_in_place(edges, parallel);
    }
}

/// Reduce an `EdgeList` to a barcode through the BitCSR backend. The homology
/// back-half, mirroring `count_cliques` for the filtration-size path.
#[inline]
fn compute_barcode(edges: EdgeList, max_dim: usize, parallel: bool) -> BarcodeResult {
    let bitcsr = engine::BitCsrDistanceMatrix::from_edge_list(edges);
    engine::algorithm::compute(&bitcsr, max_dim, parallel)
}

/// Recover the ambient dimension of a row-major point cloud from its flat length.
/// `n` is already validated (`>= 2`), so the division is safe.
#[inline]
fn infer_dimension(points: &[f32], n: usize) -> Result<usize> {
    let len = points.len();
    if len == 0 {
        return Err(Error::EmptyDimension);
    }
    if len % n != 0 {
        return Err(Error::ShapeMismatch {
            n,
            got: len,
            mode: "points (length must be a multiple of n)",
        });
    }
    Ok(len / n)
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

    fn ring_distances() -> ([f32; 8], [f32; 16]) {
        let s = std::f32::consts::SQRT_2;
        let points = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let distances = [
            0.0, 1.0, s, 1.0, 1.0, 0.0, 1.0, s, s, 1.0, 0.0, 1.0, 1.0, s, 1.0, 0.0,
        ];
        (points, distances)
    }

    #[test]
    fn point_pipeline_peel_preserves_barcode() {
        let points = [
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05, 0.2, 0.05, 0.05, 0.2,
        ];
        let n = points.len() / 2;
        let plain = persistent_homology(&points, n, 2, None, false, false, false, true).unwrap();
        let peeled = persistent_homology(&points, n, 2, None, false, false, true, true).unwrap();
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
            persistent_homology(&distances, n, 1, Some(1.5), true, false, false, true).unwrap();
        let peeled =
            persistent_homology(&distances, n, 1, Some(1.5), true, false, true, true).unwrap();
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
            persistent_homology(&distances, n, 1, Some(1.5), true, false, false, true).unwrap();
        let peeled =
            persistent_homology(&distances, n, 1, Some(1.5), true, false, true, true).unwrap();
        assert_same_barcode(&plain, &peeled);
    }

    #[test]
    fn parallel_flag_preserves_barcode() {
        // The new flag must be behaviour-neutral: seq and par assembly agree.
        let points = [
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05, 0.2, 0.05, 0.05, 0.2,
        ];
        let n = points.len() / 2;
        let seq = persistent_homology(&points, n, 2, None, false, false, false, false).unwrap();
        let par = persistent_homology(&points, n, 2, None, false, false, false, true).unwrap();
        assert_same_barcode(&seq, &par);
    }

    #[test]
    fn filtration_size_matches_across_input_modes() {
        let (points, distances) = ring_distances();

        let point_count =
            filtration_size(&points, 4, 1, Some(1.0), false, false, false, false).unwrap();
        let matrix_count =
            filtration_size(&distances, 4, 1, Some(1.0), true, false, false, false).unwrap();
        assert_eq!(point_count, 8); // 4 vertices + 4 cycle edges
        assert_eq!(matrix_count, point_count);

        let peeled = filtration_size(&points, 4, 1, None, false, false, true, false).unwrap();
        let unpeeled = filtration_size(&points, 4, 1, None, false, false, false, false).unwrap();
        assert!(peeled <= unpeeled);
    }

    #[test]
    fn quotient_filtration_matches_input_modes() {
        let (points, distances) = ring_distances();
        let from_points = filtration_size(&points, 4, 1, None, false, true, true, false).unwrap();
        let from_matrix = filtration_size(&distances, 4, 1, None, true, true, true, false).unwrap();
        assert_eq!(from_points, from_matrix);
    }

    #[test]
    fn quotient_persistent_homology_matches_input_modes() {
        // Regression anchor for the sparse quotient path: both input modes feed
        // coperto the same edge set, so the barcodes agree.
        let (points, distances) = ring_distances();
        let from_points =
            persistent_homology(&points, 4, 1, None, false, true, true, true).unwrap();
        let from_matrix =
            persistent_homology(&distances, 4, 1, None, true, true, true, true).unwrap();
        assert_same_barcode(&from_points, &from_matrix);
    }

    #[test]
    fn points_dimension_is_inferred() {
        // 4 points in 2-D: length 8, n=4 ⇒ d=2 inferred, matching the explicit-d era.
        let (points, _) = ring_distances();
        let count = filtration_size(&points, 4, 1, Some(1.0), false, false, false, false).unwrap();
        assert_eq!(count, 8);
    }

    #[test]
    fn points_length_not_multiple_of_n_errors() {
        let points = [0.0, 0.0, 1.0, 0.0, 1.0]; // length 5, n=2 ⇒ not divisible
        let err = filtration_size(&points, 2, 1, None, false, false, false, false).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn empty_points_errors() {
        let points: [f32; 0] = [];
        let err = filtration_size(&points, 2, 1, None, false, false, false, false).unwrap_err();
        assert!(matches!(err, Error::EmptyDimension));
    }
}
