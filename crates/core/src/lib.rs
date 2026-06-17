mod engine;
mod error;
mod opt;
mod preprocess;
mod types;
mod utils;

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
/// ([`opt::coperto`]) before reduction. These optimization flags still use the
/// condensed-distance compatibility path because they rewrite dense distances
/// before the filtration is built.
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

    if quotient || peel {
        // The quotient/peel optimizations rewrote dense distances and fed the
        // (now-removed) dense reduction path. Pending a re-route through BitCSR.
        todo!("quotient/peel reduction path removed in the BitCSR engine refactor");
    }

    run_bitcsr_points(points, n, d, max_dim, threshold, true)
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

    if quotient || peel {
        // The quotient/peel optimizations rewrote dense distances and fed the
        // (now-removed) dense reduction path. Pending a re-route through BitCSR.
        todo!("quotient/peel reduction path removed in the BitCSR engine refactor");
    }

    validate_distance_shape(distances, n)?;
    let user_t = threshold.unwrap_or(f32::INFINITY);
    let adj = preprocess::pdist::dmat_csr(distances, n, user_t);
    let bitcsr = engine::BitCsrDistanceMatrix::from_csr_parts(n, adj.row_ptr, adj.col, adj.val);
    Ok(engine::algorithm::compute(&bitcsr, max_dim, true))
}

fn run_bitcsr_points(
    points: &[f32],
    n: usize,
    d: usize,
    max_dim: usize,
    threshold: Option<f32>,
    parallel: bool,
) -> Result<BarcodeResult> {
    validate_point_shape(points, n, d)?;
    let user_t = threshold.unwrap_or(f32::INFINITY);
    let adj = preprocess::pdist::pdist_csr(points, n, d, user_t);
    let bitcsr = engine::BitCsrDistanceMatrix::from_csr_parts(n, adj.row_ptr, adj.col, adj.val);
    Ok(engine::algorithm::compute(&bitcsr, max_dim, parallel))
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
    let (mut lt, c_star, minimax) = condense_points(points, n, d)?;
    let t = apply_optimizations(&mut lt, c_star, minimax, threshold, quotient, peel);

    Ok(utils::cliques::count_cliques(&lt, t, max_dim + 2))
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
    let (mut lt, c_star, minimax) = condense_distances(distances, n)?;
    let t = apply_optimizations(&mut lt, c_star, minimax, threshold, quotient, peel);

    Ok(utils::cliques::count_cliques(&lt, t, max_dim + 2))
}

/// Point-cloud → condensed lower-triangular distances, Chebyshev center, and
/// minimax radius. Validates the `(n, d)` shape.
#[inline]
fn condense_points(points: &[f32], n: usize, d: usize) -> Result<(Vec<f32>, usize, f32)> {
    validate_point_shape(points, n, d)?;
    Ok(preprocess::pdist::pdist_tiled_v3(points, n, d))
}

/// Distance matrix → condensed lower-triangular distances, Chebyshev center, and
/// minimax radius. Validates the `(n, n)` shape.
#[inline]
fn condense_distances(distances: &[f32], n: usize) -> Result<(Vec<f32>, usize, f32)> {
    validate_distance_shape(distances, n)?;
    Ok(preprocess::pdist::condense_square_matrix(distances, n))
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

#[inline]
fn resolve_threshold(user: Option<f32>, minimax: f32) -> f32 {
    match user {
        None => minimax,
        Some(t) => t.min(minimax),
    }
}

#[inline]
fn apply_optimizations(
    lt: &mut Vec<f32>,
    c_star: usize,
    minimax: f32,
    threshold: Option<f32>,
    quotient: bool,
    peel: bool,
) -> f32 {
    let mut t = resolve_threshold(threshold, minimax);

    // Peel first: it reads the original geometry to certify a tighter radius.
    if peel {
        t = opt::peel::peel(lt, c_star, t);
    }
    // Quotient coning rewrites `lt` in place at the tightened radius.
    if quotient {
        opt::coperto::cone_in_place(lt, t);
    }

    t
}
