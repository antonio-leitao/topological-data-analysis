mod engine;
mod error;
mod pdist;
mod types;

use engine::DistanceMatrix;
pub use error::{Error, Result};
pub use types::{BarcodeResult, PersistenceInterval};

pub const MAX_POINTS: usize = u16::MAX as usize;
pub const MAX_DIM: usize = 4;

/// Compute Vietoris–Rips persistent homology from a row-major `(n, d)`
/// point cloud.
pub fn persistent_homology(
    points: &[f32],
    n: usize,
    d: usize,
    max_dim: usize,
    threshold: Option<f32>,
) -> Result<BarcodeResult> {
    validate_params(n, max_dim, threshold)?;
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
    let (lt, minimax) = pdist::pdist_tiled_v3(points, n, d);
    let dist = DistanceMatrix::from_lower_triangular(n, lt);
    let t = resolve_threshold(threshold, minimax);
    Ok(engine::algorithm::compute(&dist, t, max_dim))
}

/// Compute Vietoris–Rips persistent homology from a row-major `(n, n)`
/// distance matrix. Symmetry, zero diagonal, and non-negativity are
/// documented preconditions, not enforced invariants.
pub fn persistent_homology_from_distances(
    distances: &[f32],
    n: usize,
    max_dim: usize,
    threshold: Option<f32>,
) -> Result<BarcodeResult> {
    validate_params(n, max_dim, threshold)?;
    if distances.len() != n * n {
        return Err(Error::ShapeMismatch {
            n,
            got: distances.len(),
            mode: "distance matrix (expected n*n)",
        });
    }
    let (dist, minimax) = DistanceMatrix::from_square_matrix(distances, n);
    let t = resolve_threshold(threshold, minimax);
    Ok(engine::algorithm::compute(&dist, t, max_dim))
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
