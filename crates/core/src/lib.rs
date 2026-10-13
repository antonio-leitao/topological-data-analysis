#[cfg(feature = "coords")]
pub mod coords;
mod engine;
mod error;
mod input;
mod opt;
mod preprocess;
mod types;
mod utils;

use preprocess::edgelist::EdgeList;

pub use error::{Error, Result};
pub use input::{Real, SparseIndex};
pub use opt::collapse::Collapse;
pub use types::{BarcodeResult, PersistenceInterval};

/// Maximum point count allowed by the 24-bit simplex vertex slots.
pub const MAX_POINTS: usize = (1 << 24) - 1;
/// Maximum homology dimension allowed by the nine-slot simplex layout.
pub const MAX_DIM: usize = 7;

const S128_MAX_POINTS: usize = u16::MAX as usize;
const S128_MAX_DIM: usize = 4;

/// Optional knobs for [`persistent_homology_with`] and [`filtration_size_with`].
///
/// The default uses point-cloud input, the certified cutoff, exact peeling,
/// automatic edge collapse, and parallel preprocessing. Quotient coning is
/// disabled.
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
    /// Tighten the cutoff by exact strong-collapse peeling. Default `true`.
    pub peel: bool,
    /// Use a log(3)-interleaved quotient-cover filtration. Default `false`.
    pub quotient: bool,
    /// Remove filtration-dominated edges without changing the barcode.
    /// Default [`Collapse::Auto`], which decides per dimension and per dataset.
    pub collapse: Collapse,
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
            collapse: Collapse::Auto,
            parallel: true,
        }
    }
}

/// Vietoris–Rips persistent homology of a point cloud, with the default
/// [`Config`].
///
/// `data` is a row-major `f32` or `f64` point cloud of length `n * d`; `d` is
/// inferred as `data.len() / n`. For distance-matrix input, a threshold, or
/// any other option, use [`persistent_homology_with`].
pub fn persistent_homology<T: Real>(data: &[T], n: usize, max_dim: usize) -> Result<BarcodeResult> {
    persistent_homology_with(data, n, max_dim, &Config::default())
}

/// Vietoris–Rips persistent homology with explicit configuration.
///
/// `data` is row-major `f32` or `f64`; distances are computed and stored in
/// `f32`. With `distance_matrix`, it has length `n * n` and only the strict
/// lower triangle is read, in place, so `f64` input is never copied. Negative
/// entries there are an error, and infinite ones represent missing edges.
/// Otherwise it has length `n * d`, the ambient dimension is inferred, and
/// every coordinate must be finite.
///
/// Square data is ambiguous, so `config.distance_matrix` must be set correctly.
///
/// Optional transforms run in the order peel, quotient, collapse.
pub fn persistent_homology_with<T: Real>(
    data: &[T],
    n: usize,
    max_dim: usize,
    config: &Config,
) -> Result<BarcodeResult> {
    let edges = build_edges(data, n, max_dim, config, Ids::Free)?;
    compute_barcode(edges, max_dim, config.parallel)
}

/// Vietoris–Rips persistent homology of a sparse distance graph in COO form,
/// with the default [`Config`]. See [`persistent_homology_sparse_with`].
pub fn persistent_homology_sparse<I: SparseIndex, V: Real>(
    rows: &[I],
    cols: &[I],
    vals: &[V],
    n: usize,
    max_dim: usize,
) -> Result<BarcodeResult> {
    persistent_homology_sparse_with(rows, cols, vals, n, max_dim, &Config::default())
}

/// Vietoris–Rips persistent homology of a sparse distance graph in COO form.
///
/// `(rows[k], cols[k], vals[k])` is one entry `d(rows[k], cols[k]) = vals[k]`
/// of an undirected distance graph on `n` vertices. Orientation and duplicates
/// are irrelevant. The smallest finite value given for a pair wins, and
/// diagonal entries are ignored. A pair with no finite entry is never
/// connected (infinite distance), and truncation uses the same per-component
/// certificate as the dense distance-matrix path, so disconnected inputs are
/// handled soundly. Values `>= f32::MAX` (or NaN) count as absent; negative
/// values and indices outside `0..n` are errors.
///
/// Indices may be `u32`, `u64`, `i32`, or `i64` and values `f32` or `f64`; all
/// are read in place.
///
/// `config.distance_matrix` is meaningless for this entry point and ignored;
/// the other fields behave as in [`persistent_homology_with`].
pub fn persistent_homology_sparse_with<I: SparseIndex, V: Real>(
    rows: &[I],
    cols: &[I],
    vals: &[V],
    n: usize,
    max_dim: usize,
    config: &Config,
) -> Result<BarcodeResult> {
    let edges = build_sparse_edges(rows, cols, vals, n, max_dim, config)?;
    compute_barcode(edges, max_dim, config.parallel)
}

/// Vietoris–Rips persistent homology of a compressed sparse distance graph,
/// with the default [`Config`]. See [`persistent_homology_csr_with`].
pub fn persistent_homology_csr<I: SparseIndex, V: Real>(
    indptr: &[I],
    indices: &[I],
    vals: &[V],
    n: usize,
    max_dim: usize,
) -> Result<BarcodeResult> {
    persistent_homology_csr_with(indptr, indices, vals, n, max_dim, &Config::default())
}

/// Vietoris–Rips persistent homology of a compressed sparse distance graph.
///
/// Row `r` holds `d(r, indices[k]) = vals[k]` for `k` in
/// `indptr[r]..indptr[r + 1]`, so `indptr` has `n + 1` non-decreasing offsets.
/// The graph is undirected, so CSC input (offsets over columns) gives the same
/// result. Entries follow the [`persistent_homology_sparse_with`] rules, and
/// the arrays are read in place.
pub fn persistent_homology_csr_with<I: SparseIndex, V: Real>(
    indptr: &[I],
    indices: &[I],
    vals: &[V],
    n: usize,
    max_dim: usize,
    config: &Config,
) -> Result<BarcodeResult> {
    let edges = build_csr_edges(indptr, indices, vals, n, max_dim, config)?;
    compute_barcode(edges, max_dim, config.parallel)
}

/// Truncated filtration size of a sparse COO distance graph, with the default
/// [`Config`]. See [`persistent_homology_sparse_with`] for the input contract.
pub fn filtration_size_sparse<I: SparseIndex, V: Real>(
    rows: &[I],
    cols: &[I],
    vals: &[V],
    n: usize,
    max_dim: usize,
) -> Result<usize> {
    filtration_size_sparse_with(rows, cols, vals, n, max_dim, &Config::default())
}

/// Truncated filtration size of a sparse COO distance graph. See
/// [`persistent_homology_sparse_with`] for the input contract.
pub fn filtration_size_sparse_with<I: SparseIndex, V: Real>(
    rows: &[I],
    cols: &[I],
    vals: &[V],
    n: usize,
    max_dim: usize,
    config: &Config,
) -> Result<usize> {
    let edges = build_sparse_edges(rows, cols, vals, n, max_dim, config)?;
    Ok(utils::cliques::count_cliques(
        edges,
        max_dim + 1,
        config.parallel,
    ))
}

/// Truncated filtration size of a compressed sparse distance graph, with the
/// default [`Config`]. See [`persistent_homology_csr_with`].
pub fn filtration_size_csr<I: SparseIndex, V: Real>(
    indptr: &[I],
    indices: &[I],
    vals: &[V],
    n: usize,
    max_dim: usize,
) -> Result<usize> {
    filtration_size_csr_with(indptr, indices, vals, n, max_dim, &Config::default())
}

/// Truncated filtration size of a compressed sparse distance graph. See
/// [`persistent_homology_csr_with`] for the input contract.
pub fn filtration_size_csr_with<I: SparseIndex, V: Real>(
    indptr: &[I],
    indices: &[I],
    vals: &[V],
    n: usize,
    max_dim: usize,
    config: &Config,
) -> Result<usize> {
    let edges = build_csr_edges(indptr, indices, vals, n, max_dim, config)?;
    Ok(utils::cliques::count_cliques(
        edges,
        max_dim + 1,
        config.parallel,
    ))
}

/// Truncated Vietoris–Rips filtration size (simplices of dimension ≤
/// `max_dim`) of a point cloud, with the default [`Config`]. See
/// [`filtration_size_with`].
pub fn filtration_size<T: Real>(data: &[T], n: usize, max_dim: usize) -> Result<usize> {
    filtration_size_with(data, n, max_dim, &Config::default())
}

/// Truncated Vietoris–Rips filtration size (simplices of dimension ≤
/// `max_dim`). See [`persistent_homology_with`] for the `data` layout contract
/// and the meaning of the [`Config`] fields.
pub fn filtration_size_with<T: Real>(
    data: &[T],
    n: usize,
    max_dim: usize,
    config: &Config,
) -> Result<usize> {
    let edges = build_edges(data, n, max_dim, config, Ids::Free)?;
    Ok(utils::cliques::count_cliques(
        edges,
        max_dim + 1,
        config.parallel,
    ))
}

/// Whether the transforms may renumber vertices. Barcodes and filtration sizes
/// do not depend on vertex ids; coordinates are indexed by them.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ids {
    Keep,
    Free,
}

/// Build and transform the truncated edge list.
pub(crate) fn build_edges<T: Real>(
    data: &[T],
    n: usize,
    max_dim: usize,
    config: &Config,
    ids: Ids,
) -> Result<EdgeList> {
    validate_params(n, max_dim, config.threshold)?;
    let t = config.threshold.unwrap_or(f32::INFINITY);

    let mut coned = None;
    let mut edges = if config.distance_matrix {
        validate_distance_shape(data.len(), n)?;
        EdgeList::from_distance_matrix(data, n, t)?
    } else {
        let d = infer_dimension(data.len(), n)?;
        // The tiled kernel packs `f32` panels; `n * d` is small next to the
        // `n^2` pairs, so converting `f64` points here costs little.
        let points = T::as_f32_slice(data);
        validate_points(&points, d)?;
        if streams_quotient(n, config) {
            let (edges, order) = opt::coperto::cone_points(&points, n, d, t, config.parallel);
            coned = Some(order);
            edges
        } else {
            EdgeList::from_points(&points, n, d, t, config.parallel)
        }
    };

    apply_optimizations(&mut edges, max_dim, config, ids, coned)?;
    Ok(edges)
}

/// Stored edges a point cloud may need before the quotient streams them
/// instead (see [`opt::coperto::cone_points`]). The bound is every pair, so
/// about 26 700 points. Streaming costs about one extra pass over the pairs:
/// on the Stanford dragon it ties the single-pass build at 25 000-27 000
/// points, is faster from 30 000, and peaks at a fifth of its memory.
const STREAM_MIN_BYTES: usize = 1 << 32;

/// Peel needs the whole graph, so only a quotient without it can stream.
fn streams_quotient(n: usize, config: &Config) -> bool {
    let pairs = n * n.saturating_sub(1) / 2;
    config.quotient
        && !config.peel
        && pairs.saturating_mul(std::mem::size_of::<preprocess::edgelist::Edge>()) > STREAM_MIN_BYTES
}

fn build_sparse_edges<I: SparseIndex, V: Real>(
    rows: &[I],
    cols: &[I],
    vals: &[V],
    n: usize,
    max_dim: usize,
    config: &Config,
) -> Result<EdgeList> {
    validate_params(n, max_dim, config.threshold)?;
    if rows.len() != cols.len() || rows.len() != vals.len() {
        return Err(Error::SparseLengthMismatch {
            rows: rows.len(),
            cols: cols.len(),
            vals: vals.len(),
        });
    }
    let t = config.threshold.unwrap_or(f32::INFINITY);

    let mut edges = EdgeList::from_coo(n, rows, cols, vals, t)?;
    apply_optimizations(&mut edges, max_dim, config, Ids::Free, None)?;
    Ok(edges)
}

fn build_csr_edges<I: SparseIndex, V: Real>(
    indptr: &[I],
    indices: &[I],
    vals: &[V],
    n: usize,
    max_dim: usize,
    config: &Config,
) -> Result<EdgeList> {
    validate_params(n, max_dim, config.threshold)?;
    let t = config.threshold.unwrap_or(f32::INFINITY);

    let mut edges = EdgeList::from_compressed(n, indptr, indices, vals, t)?;
    apply_optimizations(&mut edges, max_dim, config, Ids::Free, None)?;
    Ok(edges)
}

/// A NaN coordinate makes every distance from its point NaN, which silently
/// zeroes the enclosing radius, so it is rejected up front.
#[inline]
fn validate_points(points: &[f32], d: usize) -> Result<()> {
    match points.iter().position(|x| !x.is_finite()) {
        None => Ok(()),
        Some(i) => Err(Error::NonFiniteCoordinate {
            point: i / d,
            value: points[i],
        }),
    }
}

/// Rows of tail-truncated bitsets above which a free-id quotient skeleton is
/// renumbered. Below it rows cost little, and keeping ids leaves results that
/// depend on them, such as which edges collapse trims, as they were.
const RELABEL_MIN_WORDS: usize = 1 << 20;

/// Quotient construction needs the uncollapsed graph, so collapse runs last.
///
/// The quotient's skeleton is sparse but its vertex ids are scattered, and
/// collapse and the engine size each row by the id range of its neighbours.
/// Where ids are free and rows would be long, the skeleton is renumbered in
/// merge order, which keeps those ranges short (see
/// [`opt::coperto::Cone::finish`]).
///
/// `coned` carries the merge order when the quotient was already built while
/// streaming the edges.
#[inline]
fn apply_optimizations(
    edges: &mut EdgeList,
    max_dim: usize,
    config: &Config,
    ids: Ids,
    coned: Option<Vec<u32>>,
) -> Result<()> {
    let order = match coned {
        Some(order) => Some(order),
        None => {
            if config.peel {
                opt::peel::peel(edges, config.parallel);
            }
            config
                .quotient
                .then(|| opt::coperto::cone_in_place(edges, config.parallel))
        }
    };
    if let Some(order) = order {
        if ids == Ids::Free && edges.tail_words() > RELABEL_MIN_WORDS {
            edges.relabel(&order);
        }
    }
    config.collapse.apply(edges, max_dim, config.parallel)?;
    Ok(())
}

/// Select the narrow packing when both its point and dimension limits fit.
#[inline]
fn compute_barcode(edges: EdgeList, max_dim: usize, parallel: bool) -> Result<BarcodeResult> {
    let use_s128 = edges.n <= S128_MAX_POINTS && max_dim <= S128_MAX_DIM;
    let bitcsr = engine::BitCsrDistanceMatrix::from_edge_list(edges)?;
    Ok(if use_s128 {
        engine::algorithm::compute::<6, engine::simplex::Simplex128>(&bitcsr, max_dim, parallel)
    } else {
        engine::algorithm::compute::<9, engine::simplex::Simplex256>(&bitcsr, max_dim, parallel)
    })
}

#[inline]
fn infer_dimension(len: usize, n: usize) -> Result<usize> {
    if len == 0 {
        return Err(Error::EmptyDimension);
    }
    if !len.is_multiple_of(n) {
        return Err(Error::ShapeMismatch {
            n,
            got: len,
            mode: "points (length must be a multiple of n)",
        });
    }
    Ok(len / n)
}

#[inline]
fn validate_distance_shape(len: usize, n: usize) -> Result<()> {
    if len != n * n {
        return Err(Error::ShapeMismatch {
            n,
            got: len,
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
        if t.partial_cmp(&0.0) == Some(std::cmp::Ordering::Less) || t.is_nan() {
            return Err(Error::InvalidThreshold(t));
        }
    }
    Ok(())
}

#[cfg(test)]
mod pipeline_tests {
    use super::*;

    fn config(peel: bool, quotient: bool, collapse: bool, parallel: bool) -> Config {
        Config {
            peel,
            quotient,
            collapse: collapse.into(),
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

    #[test]
    fn disconnected_matrix_preserves_late_h1_death() {
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

        let h0_essential = bc.intervals[0]
            .iter()
            .filter(|iv| iv.death.is_infinite())
            .count();
        assert_eq!(h0_essential, 2);

        assert_eq!(bc.intervals[1].len(), 1, "H1 = {:?}", bc.intervals[1]);
        let h1 = &bc.intervals[1][0];
        assert!((h1.birth - 1.0).abs() < 1e-6, "H1 birth = {}", h1.birth);
        assert!(
            h1.death.is_finite() && (h1.death - 2.0).abs() < 1e-6,
            "H1 death = {} (truncated to essential = the old bug)",
            h1.death
        );

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
                let plain = persistent_homology_with(
                    &points,
                    n,
                    max_dim,
                    &config(false, false, false, false),
                )
                .unwrap();
                let collapsed = persistent_homology_with(
                    &points,
                    n,
                    max_dim,
                    &config(false, false, true, false),
                )
                .unwrap();
                assert_same_barcode(&plain, &collapsed);

                let peel_collapse = persistent_homology_with(
                    &points,
                    n,
                    max_dim,
                    &config(true, false, true, false),
                )
                .unwrap();
                assert_same_barcode(&plain, &peel_collapse);
            }
        }
    }

    /// `Auto` may skip, abort after its probe, stop partway, or finish; every
    /// outcome must leave the barcode untouched.
    #[test]
    fn auto_collapse_preserves_barcode() {
        let mut s: u64 = 0x0a07_0c01_1a95_e000;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 16_777_216.0
        };
        let square: Vec<f32> = (0..60 * 2).map(|_| rng()).collect();
        let circle: Vec<f32> = (0..60)
            .flat_map(|_| {
                let a = rng() * std::f32::consts::TAU;
                [a.cos(), a.sin()]
            })
            .collect();
        for points in [&square, &circle] {
            let n = points.len() / 2;
            for max_dim in 0..=2 {
                let plain = persistent_homology_with(
                    points,
                    n,
                    max_dim,
                    &config(true, false, false, false),
                )
                .unwrap();
                let auto = persistent_homology_with(
                    points,
                    n,
                    max_dim,
                    &Config {
                        collapse: Collapse::Auto,
                        ..config(true, false, false, false)
                    },
                )
                .unwrap();
                assert_same_barcode(&plain, &auto);
            }
        }
    }

    /// `Auto` relies on any prefix of the backward pass being exact.
    #[test]
    fn collapse_prefix_preserves_barcode() {
        let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 16_777_216.0
        };
        let (n, d) = (40usize, 3usize);
        let points: Vec<f32> = (0..n * d).map(|_| rng()).collect();
        let base = EdgeList::from_points(&points, n, d, f32::INFINITY, false);
        let m = base.edges.len();
        for max_dim in 1..=2 {
            let plain = compute_barcode(
                EdgeList::from_points(&points, n, d, f32::INFINITY, false),
                max_dim,
                false,
            )
            .unwrap();
            for stop in [0, 1, m / 4, m / 2, m] {
                let mut edges = EdgeList::from_points(&points, n, d, f32::INFINITY, false);
                let progress =
                    opt::collapse::collapse_with(&mut edges, false, |p| p.processed < stop);
                assert!(progress.processed <= stop.max(1));
                let bc = compute_barcode(edges, max_dim, false).unwrap();
                assert_same_barcode(&plain, &bc);
            }
        }
    }

    #[test]
    fn collapse_preserves_quotient_barcode() {
        let (points, _) = ring_distances();
        let quot =
            persistent_homology_with(&points, 4, 1, &config(true, true, false, false)).unwrap();
        let quot_collapsed =
            persistent_homology_with(&points, 4, 1, &config(true, true, true, false)).unwrap();
        assert_same_barcode(&quot, &quot_collapsed);
    }

    /// Renumbering a quotient skeleton in merge order changes vertex ids only,
    /// so the barcode is unchanged with and without collapse.
    #[test]
    fn merge_order_relabel_preserves_barcode() {
        let n = 150;
        let points: Vec<f32> = (0..n * 3)
            .map(|i| ((i as f32) * 0.7548).sin() + ((i * i) as f32 * 0.0137).cos())
            .collect();
        let barcode = |relabel: bool, collapse: Collapse, max_dim: usize| {
            let mut edges = EdgeList::from_points(&points, n, 3, f32::INFINITY, false);
            let order = opt::coperto::cone_in_place(&mut edges, false);
            if relabel {
                edges.relabel(&order);
            }
            collapse.apply(&mut edges, max_dim, false).unwrap();
            compute_barcode(edges, max_dim, false).unwrap()
        };
        for collapse in [Collapse::Off, Collapse::On, Collapse::Auto] {
            for max_dim in 1..=3 {
                assert_same_barcode(
                    &barcode(false, collapse, max_dim),
                    &barcode(true, collapse, max_dim),
                );
            }
        }
    }

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
        let (points, _) = ring_distances();
        let cfg = Config {
            threshold: Some(1.0),
            ..config(false, false, false, false)
        };
        let count = filtration_size_with(&points, 4, 1, &cfg).unwrap();
        assert_eq!(count, 8);
    }

    #[test]
    fn sparse_pipeline_matches_dense_matrix() {
        let (_, distances) = ring_distances();
        let n = 4;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            for j in 0..n {
                rows.push(i as u32);
                cols.push(j as u32);
                vals.push(distances[i * n + j]); // diagonal zeros are ignored
            }
        }

        for peel in [false, true] {
            let base = config(peel, false, false, false);
            let dense_cfg = Config {
                distance_matrix: true,
                ..base.clone()
            };
            let dense = persistent_homology_with(&distances, n, 1, &dense_cfg).unwrap();
            let sparse = persistent_homology_sparse_with(&rows, &cols, &vals, n, 1, &base).unwrap();
            assert_same_barcode(&dense, &sparse);
            assert_eq!(
                filtration_size_sparse_with(&rows, &cols, &vals, n, 1, &base).unwrap(),
                filtration_size_with(&distances, n, 1, &dense_cfg).unwrap()
            );
        }
    }

    #[test]
    fn sparse_disconnected_preserves_late_h1_death() {
        let entries: &[(u32, u32, f32)] = &[
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 0, 1.0),
            (0, 2, 2.0),
            (1, 3, 2.0),
            (4, 5, 1.0),
            (1, 0, 7.0), // duplicate of (0, 1): the min (1.0) must win
        ];
        let rows: Vec<u32> = entries.iter().map(|e| e.0).collect();
        let cols: Vec<u32> = entries.iter().map(|e| e.1).collect();
        let vals: Vec<f32> = entries.iter().map(|e| e.2).collect();

        let bc = persistent_homology_sparse_with(
            &rows,
            &cols,
            &vals,
            6,
            1,
            &config(false, false, false, false),
        )
        .unwrap();

        let h0_essential = bc.intervals[0]
            .iter()
            .filter(|iv| iv.death.is_infinite())
            .count();
        assert_eq!(h0_essential, 2);
        assert_eq!(bc.intervals[1].len(), 1, "H1 = {:?}", bc.intervals[1]);
        let h1 = &bc.intervals[1][0];
        assert!((h1.birth - 1.0).abs() < 1e-6);
        assert!(h1.death.is_finite() && (h1.death - 2.0).abs() < 1e-6);

        let peeled = persistent_homology_sparse_with(
            &rows,
            &cols,
            &vals,
            6,
            1,
            &config(true, false, false, false),
        )
        .unwrap();
        assert_same_barcode(&bc, &peeled);
    }

    /// The default-`Config` sparse wrapper must agree with its `_with` sibling
    /// and with the dense path. The 4x4 grid is chosen because peeling changes
    /// the count here (98 vs 106 at `max_dim=1`), so a wrapper that forwarded
    /// the wrong `Config` would fail rather than coincide.
    #[test]
    fn filtration_size_sparse_matches_with_and_dense() {
        let n = 16;
        let mut points: Vec<f32> = Vec::with_capacity(n * 2);
        for i in 0..4 {
            for j in 0..4 {
                points.push(i as f32);
                points.push(j as f32);
            }
        }

        let mut distances = vec![0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let (dx, dy) = (
                    points[i * 2] - points[j * 2],
                    points[i * 2 + 1] - points[j * 2 + 1],
                );
                distances[i * n + j] = (dx * dx + dy * dy).sqrt();
            }
        }

        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            for j in 0..n {
                rows.push(i as u32);
                cols.push(j as u32);
                vals.push(distances[i * n + j]); // diagonal zeros are ignored
            }
        }

        let dense_cfg = Config {
            distance_matrix: true,
            ..Config::default()
        };
        for max_dim in 1..=2 {
            let plain = filtration_size_sparse(&rows, &cols, &vals, n, max_dim).unwrap();
            assert_eq!(
                plain,
                filtration_size_sparse_with(&rows, &cols, &vals, n, max_dim, &Config::default())
                    .unwrap(),
                "wrapper must delegate with the default Config (max_dim={max_dim})"
            );
            assert_eq!(
                plain,
                filtration_size_with(&distances, n, max_dim, &dense_cfg).unwrap(),
                "sparse complete graph must match dense (max_dim={max_dim})"
            );
        }
    }

    #[test]
    fn sparse_input_validation_errors() {
        let err = persistent_homology_sparse(&[0, 1], &[1], &[1.0, 2.0], 4, 1).unwrap_err();
        assert!(matches!(err, Error::SparseLengthMismatch { .. }));

        let err = persistent_homology_sparse(&[0, 7], &[1, 0], &[1.0, 2.0], 4, 1).unwrap_err();
        assert!(matches!(err, Error::SparseIndexOutOfRange { .. }));
    }

    #[test]
    fn deep_dimension_dispatches_to_simplex256() {
        let (points, _) = ring_distances();
        let wide = persistent_homology(&points, 4, 5).unwrap();
        let narrow = persistent_homology(&points, 4, 1).unwrap();
        assert_eq!(wide.intervals.len(), 6);
        for d in 0..=1 {
            let key = |ivs: &[PersistenceInterval]| {
                let mut v: Vec<_> = ivs
                    .iter()
                    .map(|iv| (iv.birth.to_bits(), iv.death.to_bits()))
                    .collect();
                v.sort_unstable();
                v
            };
            assert_eq!(
                key(&wide.intervals[d]),
                key(&narrow.intervals[d]),
                "dim {d}"
            );
        }
        for d in 2..=5 {
            assert!(wide.intervals[d].is_empty(), "dim {d} must be empty");
        }
    }

    #[test]
    fn quotient_composes_with_deep_dimensions() {
        let mut s: u64 = 0x0dd5_eed5_4321_8765;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 16_777_216.0
        };
        let n = 24;
        let points: Vec<f32> = (0..n * 3).map(|_| rng()).collect();
        let cfg = config(true, true, false, false); // peel + quotient

        let narrow = persistent_homology_with(&points, n, 2, &cfg).unwrap(); // S128 arm
        let wide = persistent_homology_with(&points, n, 5, &cfg).unwrap(); // S256 arm

        let key = |ivs: &[PersistenceInterval]| {
            let mut v: Vec<_> = ivs
                .iter()
                .map(|iv| (iv.birth.to_bits(), iv.death.to_bits()))
                .collect();
            v.sort_unstable();
            v
        };
        for d in 0..=2 {
            assert_eq!(
                key(&wide.intervals[d]),
                key(&narrow.intervals[d]),
                "dim {d}"
            );
        }
    }

    #[test]
    fn max_dim_above_cap_errors() {
        let (points, _) = ring_distances();
        let err = persistent_homology(&points, 4, MAX_DIM + 1).unwrap_err();
        assert!(matches!(err, Error::DimTooLarge { .. }));
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

    /// A NaN point used to zero the enclosing radius and return every vertex
    /// as an essential H0 class.
    #[test]
    fn non_finite_coordinates_error() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let (mut points, _) = ring_distances();
            points[5] = bad; // point 2, coordinate 1
            let err = persistent_homology(&points, 4, 1).unwrap_err();
            assert!(
                matches!(err, Error::NonFiniteCoordinate { point: 2, .. }),
                "{bad}: {err:?}"
            );
        }
        let mut wide: Vec<f64> = ring_distances().0.iter().map(|&x| x as f64).collect();
        wide[0] = 1e300; // finite in f64, infinite once read as f32
        let err = persistent_homology(&wide, 4, 1).unwrap_err();
        assert!(matches!(err, Error::NonFiniteCoordinate { point: 0, .. }));
    }

    /// Negative distances sort above every other value by bit pattern, which
    /// used to add a spurious H1 bar.
    #[test]
    fn negative_distances_error() {
        let (_, mut distances) = ring_distances();
        distances[3 * 4 + 1] = -1e-7; // strict lower triangle: d(3, 1)
        let cfg = Config {
            distance_matrix: true,
            ..Config::default()
        };
        let err = persistent_homology_with(&distances, 4, 1, &cfg).unwrap_err();
        assert!(
            matches!(err, Error::NegativeDistance { u: 3, v: 1, .. }),
            "{err:?}"
        );

        // -0.0 is zero, not negative.
        let (_, mut zeroed) = ring_distances();
        let (_, mut signed) = ring_distances();
        zeroed[2 * 4 + 1] = 0.0;
        signed[2 * 4 + 1] = -0.0;
        assert_same_barcode(
            &persistent_homology_with(&zeroed, 4, 1, &cfg).unwrap(),
            &persistent_homology_with(&signed, 4, 1, &cfg).unwrap(),
        );

        let err =
            persistent_homology_sparse(&[0u32, 1], &[1u32, 2], &[1.0f32, -0.5], 3, 1).unwrap_err();
        assert!(matches!(err, Error::NegativeDistance { u: 1, v: 2, .. }));
        let err = persistent_homology_csr(&[0u32, 1, 2, 2], &[1u32, 2], &[1.0f32, -0.5], 3, 1)
            .unwrap_err();
        assert!(matches!(err, Error::NegativeDistance { u: 1, v: 2, .. }));
    }

    #[test]
    fn f64_input_matches_f32() {
        let (points, distances) = ring_distances();
        let points64: Vec<f64> = points.iter().map(|&x| x as f64).collect();
        let distances64: Vec<f64> = distances.iter().map(|&x| x as f64).collect();
        let matrix = Config {
            distance_matrix: true,
            ..Config::default()
        };
        assert_same_barcode(
            &persistent_homology(&points, 4, 1).unwrap(),
            &persistent_homology(&points64, 4, 1).unwrap(),
        );
        assert_same_barcode(
            &persistent_homology_with(&distances, 4, 1, &matrix).unwrap(),
            &persistent_homology_with(&distances64, 4, 1, &matrix).unwrap(),
        );
        assert_eq!(
            filtration_size_with(&distances, 4, 1, &matrix).unwrap(),
            filtration_size_with(&distances64, 4, 1, &matrix).unwrap()
        );
    }

    /// CSR and CSC read the same undirected graph as COO and the dense path,
    /// for every index and value width, with duplicates and unsorted rows.
    #[test]
    fn compressed_matches_coo_and_dense() {
        let n = 16;
        let coord = |i: usize| ((i / 4) as f32, (i % 4) as f32);
        let mut distances = vec![0f32; n * n];
        let mut rows: Vec<Vec<(i64, f32)>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in 0..n {
                let ((xi, yi), (xj, yj)) = (coord(i), coord(j));
                let d = ((xi - xj).powi(2) + (yi - yj).powi(2)).sqrt();
                distances[i * n + j] = d;
                rows[i].push((j as i64, d));
            }
            rows[i].reverse(); // unsorted columns
            rows[i].push(((i + 1) as i64 % n as i64, 99.0)); // duplicate; the minimum wins
        }
        let mut indptr = vec![0i64];
        let (mut indices, mut vals) = (Vec::new(), Vec::new());
        let (mut coo_r, mut coo_c) = (Vec::new(), Vec::new());
        for (r, row) in rows.iter().enumerate() {
            for &(c, d) in row {
                indices.push(c);
                vals.push(d as f64);
                coo_r.push(r as i64);
                coo_c.push(c);
            }
            indptr.push(indices.len() as i64);
        }
        let indptr32: Vec<i32> = indptr.iter().map(|&x| x as i32).collect();
        let indices32: Vec<i32> = indices.iter().map(|&x| x as i32).collect();
        let vals32: Vec<f32> = vals.iter().map(|&x| x as f32).collect();

        let dense_cfg = Config {
            distance_matrix: true,
            ..Config::default()
        };
        for max_dim in 1..=2 {
            let dense = persistent_homology_with(&distances, n, max_dim, &dense_cfg).unwrap();
            let size = filtration_size_with(&distances, n, max_dim, &dense_cfg).unwrap();
            let cfg = Config::default();
            for bc in [
                persistent_homology_csr_with(&indptr, &indices, &vals, n, max_dim, &cfg).unwrap(),
                persistent_homology_csr_with(&indptr32, &indices32, &vals32, n, max_dim, &cfg)
                    .unwrap(),
                persistent_homology_csr(&indptr32, &indices32, &vals, n, max_dim).unwrap(),
                persistent_homology_sparse_with(&coo_r, &coo_c, &vals, n, max_dim, &cfg).unwrap(),
            ] {
                assert_same_barcode(&dense, &bc);
            }
            assert_eq!(
                size,
                filtration_size_csr_with(&indptr, &indices, &vals32, n, max_dim, &cfg).unwrap()
            );
            assert_eq!(
                size,
                filtration_size_csr(&indptr32, &indices32, &vals32, n, max_dim).unwrap()
            );
        }
    }

    #[test]
    fn malformed_compressed_input_errors() {
        let vals = [1.0f32, 1.0];
        let cases: [(&[i32], &[i32]); 4] = [
            (&[0, 1, 2], &[1, 0]),     // indptr too short for n = 3
            (&[0, 2, 1, 2], &[1, 0]),  // decreasing
            (&[0, 1, 2, 3], &[1, 0]),  // past the end of indices
            (&[0, 1, -1, 2], &[1, 0]), // negative offset
        ];
        for (indptr, indices) in cases {
            let err = persistent_homology_csr(indptr, indices, &vals, 3, 1).unwrap_err();
            assert!(
                matches!(err, Error::InvalidCompressed(_)),
                "{indptr:?}: {err:?}"
            );
        }
        let err =
            persistent_homology_csr(&[0i32, 1, 2, 2], &[1i32, 0], &[1.0f32], 3, 1).unwrap_err();
        assert!(matches!(err, Error::InvalidCompressed(_)));
        let err = persistent_homology_csr(&[0i32, 1, 2, 2], &[1i32, 3], &vals, 3, 1).unwrap_err();
        assert!(matches!(
            err,
            Error::SparseIndexOutOfRange { index: 3, n: 3 }
        ));
    }

    #[test]
    fn negative_sparse_index_errors() {
        let err =
            persistent_homology_sparse(&[0i32, -1], &[1i32, 0], &[1.0f32, 2.0], 4, 1).unwrap_err();
        assert!(
            matches!(err, Error::SparseIndexOutOfRange { index: -1, n: 4 }),
            "{err:?}"
        );
        let err = persistent_homology_sparse(&[0i64, 1], &[1i64, 1 << 40], &[1.0f64, 2.0], 4, 1)
            .unwrap_err();
        assert!(matches!(err, Error::SparseIndexOutOfRange { .. }));
    }
}
