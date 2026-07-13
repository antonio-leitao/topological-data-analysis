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
// Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Optional knobs for [`persistent_homology_with`] and [`filtration_size_with`].
///
/// `Config::default()` is the recommended configuration: point-cloud input, no
/// user threshold (the certified truncation radius alone caps the filtration),
/// `peel` on (exact, strictly cheaper, identical barcode), `parallel` on, and
/// the `quotient` / `collapse` transforms off. The plain
/// [`persistent_homology`] / [`filtration_size`] entry points use it
/// implicitly; reach for `Config` only to deviate. Override individual fields
/// with functional-record-update syntax:
///
/// ```
/// use tda_core::Config;
///
/// let cfg = Config {
///     distance_matrix: true,
///     threshold: Some(1.5),
///     ..Config::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct Config {
    /// Interpret `data` as a row-major `n × n` distance matrix instead of an
    /// `n × d` point cloud. See [`persistent_homology_with`] for the layout
    /// contract and the square-input ambiguity.
    pub distance_matrix: bool,
    /// Optional filtration cutoff. The effective cutoff is always
    /// `min(threshold, certified radius)`; `None` means the certified radius
    /// alone. Must be non-negative and not NaN.
    pub threshold: Option<f32>,
    /// Tighten the truncation radius via sound strong-collapse peeling:
    /// strictly cheaper, identical barcode. Default `true`.
    pub peel: bool,
    /// Rewrite the filtration into a quotient-cover coning: strictly smaller,
    /// `log 3`-interleaved with VR rather than equal. Default `false`.
    pub quotient: bool,
    /// Remove filtration-dominated edges (exact edge collapse): strictly
    /// smaller, identical barcode, but expensive. Default `false`.
    pub collapse: bool,
    /// Enable rayon in sufficiently large point preprocessing, edge sorts, and
    /// engine candidate assembly. Default `true`.
    pub parallel: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            distance_matrix: false,
            threshold: None,
            peel: true,
            quotient: false,
            collapse: false,
            parallel: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Persistent homology
// ═══════════════════════════════════════════════════════════════════════════════

/// Vietoris–Rips persistent homology of a point cloud, with the default
/// [`Config`].
///
/// `data` is a row-major `&[f32]` point cloud of length `n * d`; `d` is
/// inferred as `data.len() / n`. For distance-matrix input, a threshold, or
/// any other option, use [`persistent_homology_with`].
pub fn persistent_homology(data: &[f32], n: usize, max_dim: usize) -> Result<BarcodeResult> {
    persistent_homology_with(data, n, max_dim, &Config::default())
}

/// Vietoris–Rips persistent homology over the BitCSR sparse backend.
///
/// `data` is a row-major `&[f32]` whose length is:
///   * `n * n` when `config.distance_matrix` is true — the strict lower
///     triangle is read; symmetry, zero diagonal, and non-negativity are
///     documented preconditions, not enforced invariants, and `f32::INFINITY`
///     is a valid "disconnected" entry; or
///   * `n * d` otherwise — a point cloud, with `d` inferred as `data.len() / n`.
///
/// The two layouts collide at `d == n` (square data): they are byte-identical,
/// so `config.distance_matrix` is the *only* thing that disambiguates them —
/// set it correctly. (The Python layer warns on this case; the core cannot,
/// since its hot caller is the benchmark loop.)
///
/// `config.peel` tightens the truncation radius via a sound strong collapse
/// ([`opt::peel`]) — strictly cheaper, identical barcode. `config.quotient`
/// rewrites the filtration into a quotient-cover (complete-linkage) coning
/// ([`opt::coperto`]) — strictly smaller, and `log 3`-interleaved with VR
/// rather than equal. `config.collapse` removes filtration-dominated edges
/// ([`opt::collapse`]) — strictly smaller, identical barcode. The canonical
/// order is peel → quotient → collapse: peel first (fewer edges, leaves them
/// sorted), collapse last (it is exact on any weighted graph's flag
/// filtration, including the coned quotient, whereas the quotient's
/// complete-linkage counting needs the un-collapsed graph). `config.parallel`
/// enables Rayon in sufficiently large point preprocessing, edge sorts, and
/// engine candidate assembly.
pub fn persistent_homology_with(
    data: &[f32],
    n: usize,
    max_dim: usize,
    config: &Config,
) -> Result<BarcodeResult> {
    let edges = build_edges(data, n, max_dim, config)?;
    Ok(compute_barcode(edges, max_dim, config.parallel))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Filtration size
// ═══════════════════════════════════════════════════════════════════════════════

/// Truncated Vietoris–Rips filtration size (simplices of dimension ≤
/// `max_dim`) of a point cloud, with the default [`Config`]. See
/// [`filtration_size_with`].
pub fn filtration_size(data: &[f32], n: usize, max_dim: usize) -> Result<usize> {
    filtration_size_with(data, n, max_dim, &Config::default())
}

/// Truncated Vietoris–Rips filtration size (simplices of dimension ≤
/// `max_dim`). See [`persistent_homology_with`] for the `data` layout contract
/// and the meaning of the [`Config`] fields.
pub fn filtration_size_with(
    data: &[f32],
    n: usize,
    max_dim: usize,
    config: &Config,
) -> Result<usize> {
    let edges = build_edges(data, n, max_dim, config)?;
    // `max_dim` is a simplex dimension; `count_cliques` is clique-size oriented.
    // A simplex of dimension d is a clique of size d + 1, so counting simplices
    // of dimension <= max_dim means cliques of size <= max_dim + 1.
    Ok(utils::cliques::count_cliques(
        edges,
        max_dim + 1,
        config.parallel,
    ))
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
fn build_edges(data: &[f32], n: usize, max_dim: usize, config: &Config) -> Result<EdgeList> {
    validate_params(n, max_dim, config.threshold)?;
    let t = config.threshold.unwrap_or(f32::INFINITY);

    let mut edges = if config.distance_matrix {
        validate_distance_shape(data, n)?;
        EdgeList::from_distance_matrix(data, n, t)
    } else {
        let d = infer_dimension(data, n)?;
        EdgeList::from_points(data, n, d, t, config.parallel)
    };

    apply_optimizations(&mut edges, config);
    Ok(edges)
}

/// Apply the optional transforms in canonical order: `peel` first (exact,
/// barcode-preserving, leaves the edges sorted), then `quotient` coning
/// (`coperto`, which reuses that sort and needs the un-collapsed graph), then
/// `collapse` last (exact on whatever flag filtration it receives).
#[inline]
fn apply_optimizations(edges: &mut EdgeList, config: &Config) {
    if config.peel {
        opt::peel::peel(edges, config.parallel);
    }
    if config.quotient {
        opt::coperto::cone_in_place(edges, config.parallel);
    }
    if config.collapse {
        opt::collapse::collapse_in_place(edges, config.parallel);
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

    /// A `Config` for the point-cloud path with no threshold; the four flags
    /// every test varies, in one place.
    fn config(peel: bool, quotient: bool, collapse: bool, parallel: bool) -> Config {
        Config {
            peel,
            quotient,
            collapse,
            parallel,
            ..Config::default()
        }
    }

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
    fn simple_api_matches_default_config() {
        let points = [
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05, 0.2, 0.05, 0.05, 0.2,
        ];
        let n = points.len() / 2;
        let simple = persistent_homology(&points, n, 2).unwrap();
        let explicit = persistent_homology_with(&points, n, 2, &Config::default()).unwrap();
        assert_same_barcode(&simple, &explicit);
        assert_eq!(
            filtration_size(&points, n, 2).unwrap(),
            filtration_size_with(&points, n, 2, &Config::default()).unwrap()
        );
    }

    #[test]
    fn point_pipeline_peel_preserves_barcode() {
        let points = [
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05, 0.2, 0.05, 0.05, 0.2,
        ];
        let n = points.len() / 2;
        let plain =
            persistent_homology_with(&points, n, 2, &config(false, false, false, true)).unwrap();
        let peeled =
            persistent_homology_with(&points, n, 2, &config(true, false, false, true)).unwrap();
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

        let base = Config {
            distance_matrix: true,
            threshold: Some(1.5),
            ..config(false, false, false, true)
        };
        let plain = persistent_homology_with(&distances, n, 1, &base).unwrap();
        let peeled = persistent_homology_with(
            &distances,
            n,
            1,
            &Config {
                peel: true,
                ..base.clone()
            },
        )
        .unwrap();
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

        let base = Config {
            distance_matrix: true,
            threshold: Some(1.5),
            ..config(false, false, false, true)
        };
        let plain = persistent_homology_with(&distances, n, 1, &base).unwrap();
        let peeled = persistent_homology_with(
            &distances,
            n,
            1,
            &Config {
                peel: true,
                ..base.clone()
            },
        )
        .unwrap();
        assert_same_barcode(&plain, &peeled);
    }

    /// REGRESSION (disconnected enclosing radius): the H1 bar of the far
    /// component must DIE at 2.0, not be truncated into an essential class by
    /// the near component's smaller enclosing radius.
    #[test]
    fn disconnected_matrix_preserves_late_h1_death() {
        // Component A: unit 4-cycle (0-1-2-3) with diagonals 2.0 → H1 [1, 2).
        // Component B: pair (4, 5) at distance 1.0 (enclosing radius 1.0).
        let n = 6;
        let mut d = vec![f32::INFINITY; n * n];
        for i in 0..n {
            d[i * n + i] = 0.0;
        }
        for &(a, b, v) in &[
            (0usize, 1usize, 1.0f32),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (0, 3, 1.0),
            (0, 2, 2.0),
            (1, 3, 2.0),
            (4, 5, 1.0),
        ] {
            d[a * n + b] = v;
            d[b * n + a] = v;
        }

        let base = Config {
            distance_matrix: true,
            ..config(false, false, false, false)
        };
        let bc = persistent_homology_with(&d, n, 1, &base).unwrap();

        // H0: exactly two essential components.
        let h0_essential = bc.intervals[0]
            .iter()
            .filter(|iv| iv.death.is_infinite())
            .count();
        assert_eq!(h0_essential, 2);

        // H1: exactly one bar, [1, 2), finite.
        assert_eq!(bc.intervals[1].len(), 1, "H1 = {:?}", bc.intervals[1]);
        let h1 = &bc.intervals[1][0];
        assert!((h1.birth - 1.0).abs() < 1e-6, "H1 birth = {}", h1.birth);
        assert!(
            h1.death.is_finite() && (h1.death - 2.0).abs() < 1e-6,
            "H1 death = {} (truncated to essential = the old bug)",
            h1.death
        );

        // Peel must remain barcode-preserving on disconnected inputs.
        let peeled = persistent_homology_with(
            &d,
            n,
            1,
            &Config {
                peel: true,
                ..base.clone()
            },
        )
        .unwrap();
        assert_same_barcode(&bc, &peeled);
    }

    /// Edge collapse is exact: identical barcode (bit-for-bit as a multiset)
    /// on deterministic random point clouds, alone and composed with peel.
    #[test]
    fn collapse_preserves_barcode_random() {
        let mut s: u64 = 0x51ce_b00c_1234_5678;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 16_777_216.0
        };
        for &(n, d) in &[(8usize, 2usize), (12, 3), (20, 2), (16, 4)] {
            let points: Vec<f32> = (0..n * d).map(|_| rng()).collect();
            for max_dim in 1..=2 {
                let plain =
                    persistent_homology_with(&points, n, max_dim, &config(false, false, false, false))
                        .unwrap();
                let collapsed =
                    persistent_homology_with(&points, n, max_dim, &config(false, false, true, false))
                        .unwrap();
                assert_same_barcode(&plain, &collapsed);

                let peel_collapse =
                    persistent_homology_with(&points, n, max_dim, &config(true, false, true, false))
                        .unwrap();
                assert_same_barcode(&plain, &peel_collapse);
            }
        }
    }

    /// Collapse composed AFTER the quotient must preserve the quotient's own
    /// barcode exactly (collapse is exact on any flag filtration).
    #[test]
    fn collapse_preserves_quotient_barcode() {
        let (points, _) = ring_distances();
        let quot =
            persistent_homology_with(&points, 4, 1, &config(true, true, false, false)).unwrap();
        let quot_collapsed =
            persistent_homology_with(&points, 4, 1, &config(true, true, true, false)).unwrap();
        assert_same_barcode(&quot, &quot_collapsed);
    }

    /// Collapse must strictly shrink the filtration on clique-rich inputs.
    #[test]
    fn collapse_reduces_filtration_size() {
        let mut points = Vec::new();
        for x in 0..4 {
            for y in 0..4 {
                points.push(x as f32);
                points.push(y as f32);
            }
        }
        let n = 16;
        let plain =
            filtration_size_with(&points, n, 2, &config(false, false, false, false)).unwrap();
        let collapsed =
            filtration_size_with(&points, n, 2, &config(false, false, true, false)).unwrap();
        assert!(
            collapsed < plain,
            "collapse should shrink the filtration ({collapsed} vs {plain})"
        );
    }

    #[test]
    fn parallel_flag_preserves_barcode() {
        // The flag must be behaviour-neutral: seq and par assembly agree.
        let points = [
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05, 0.2, 0.05, 0.05, 0.2,
        ];
        let n = points.len() / 2;
        let seq =
            persistent_homology_with(&points, n, 2, &config(false, false, false, false)).unwrap();
        let par =
            persistent_homology_with(&points, n, 2, &config(false, false, false, true)).unwrap();
        assert_same_barcode(&seq, &par);
    }

    #[test]
    fn filtration_size_matches_across_input_modes() {
        let (points, distances) = ring_distances();

        let point_cfg = Config {
            threshold: Some(1.0),
            ..config(false, false, false, false)
        };
        let matrix_cfg = Config {
            distance_matrix: true,
            ..point_cfg.clone()
        };
        let point_count = filtration_size_with(&points, 4, 1, &point_cfg).unwrap();
        let matrix_count = filtration_size_with(&distances, 4, 1, &matrix_cfg).unwrap();
        assert_eq!(point_count, 8); // 4 vertices + 4 cycle edges
        assert_eq!(matrix_count, point_count);

        let peeled =
            filtration_size_with(&points, 4, 1, &config(true, false, false, false)).unwrap();
        let unpeeled =
            filtration_size_with(&points, 4, 1, &config(false, false, false, false)).unwrap();
        assert!(peeled <= unpeeled);
    }

    #[test]
    fn quotient_filtration_matches_input_modes() {
        let (points, distances) = ring_distances();
        let point_cfg = config(true, true, false, false);
        let matrix_cfg = Config {
            distance_matrix: true,
            ..point_cfg.clone()
        };
        let from_points = filtration_size_with(&points, 4, 1, &point_cfg).unwrap();
        let from_matrix = filtration_size_with(&distances, 4, 1, &matrix_cfg).unwrap();
        assert_eq!(from_points, from_matrix);
    }

    #[test]
    fn quotient_persistent_homology_matches_input_modes() {
        // Regression anchor for the sparse quotient path: both input modes feed
        // coperto the same edge set, so the barcodes agree.
        let (points, distances) = ring_distances();
        let point_cfg = config(true, true, false, true);
        let matrix_cfg = Config {
            distance_matrix: true,
            ..point_cfg.clone()
        };
        let from_points = persistent_homology_with(&points, 4, 1, &point_cfg).unwrap();
        let from_matrix = persistent_homology_with(&distances, 4, 1, &matrix_cfg).unwrap();
        assert_same_barcode(&from_points, &from_matrix);
    }

    #[test]
    fn points_dimension_is_inferred() {
        // 4 points in 2-D: length 8, n=4 ⇒ d=2 inferred, matching the explicit-d era.
        let (points, _) = ring_distances();
        let cfg = Config {
            threshold: Some(1.0),
            ..config(false, false, false, false)
        };
        let count = filtration_size_with(&points, 4, 1, &cfg).unwrap();
        assert_eq!(count, 8);
    }

    #[test]
    fn points_length_not_multiple_of_n_errors() {
        let points = [0.0, 0.0, 1.0, 0.0, 1.0]; // length 5, n=2 ⇒ not divisible
        let err = filtration_size(&points, 2, 1).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn empty_points_errors() {
        let points: [f32; 0] = [];
        let err = filtration_size(&points, 2, 1).unwrap_err();
        assert!(matches!(err, Error::EmptyDimension));
    }
}
