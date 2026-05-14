use thiserror::Error;

/// Anything that can go wrong inside the `tda` crate.
///
/// All variants are O(1) detectable at the entry-function boundary. The crate
/// deliberately does **not** validate distance matrix contents (negativity,
/// finiteness, symmetry, zero diagonal) — those are documented preconditions,
/// not enforced invariants. Sweeping an n×n matrix just to produce a friendly
/// error costs as much as the minimax computation itself, and this library
/// optimises for the path where inputs are well-formed.
///
/// In particular, `f32::INFINITY` is a legitimate distance value (used to
/// represent disconnected pairs in graph inputs) and is allowed to flow
/// through unchecked.
#[derive(Debug, Clone, Error)]
pub enum Error {
    #[error("need at least 2 points, got {got}")]
    TooFewPoints { got: usize },

    #[error(
        "library supports at most {max} points (Simplex128 packs vertex IDs \
         into 16 bits); got {got}"
    )]
    TooManyPoints { got: usize, max: usize },

    #[error("points must have at least 1 dimension")]
    EmptyDimension,

    #[error(
        "max_dim must be at most {max} (reducing d-simplices needs (d+1)-cofacet \
         enumeration, capped at 6 vertices by Simplex128); got {got}"
    )]
    DimTooLarge { got: usize, max: usize },

    #[error("threshold must be non-negative (and not NaN), got {0}")]
    InvalidThreshold(f32),
    #[error("data length {got} is incompatible with n={n} for {mode} input")]
    ShapeMismatch {
        n: usize,
        got: usize,
        mode: &'static str,
    },
}

/// Crate-wide `Result` alias.
pub type Result<T> = std::result::Result<T, Error>;
