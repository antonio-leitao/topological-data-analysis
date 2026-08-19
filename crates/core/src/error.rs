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
        "library supports at most {max} points (vertex ids are packed into \
         24-bit simplex slots); got {got}"
    )]
    TooManyPoints { got: usize, max: usize },

    #[error("points must have at least 1 dimension")]
    EmptyDimension,

    #[error(
        "max_dim must be at most {max} (reducing d-simplices needs (d+1)-cofacet \
         enumeration, capped at 9 vertices by the widest simplex layout); got {got}"
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

    #[error(
        "sparse input arrays must have equal lengths, got rows={rows}, \
         cols={cols}, values={vals}"
    )]
    SparseLengthMismatch {
        rows: usize,
        cols: usize,
        vals: usize,
    },

    #[error("sparse entry ({u}, {v}) is out of range for n={n}")]
    SparseIndexOutOfRange { u: u32, v: u32, n: usize },

    #[error(
        "coordinates: requested class {requested} but only {available} \
         classes exist in dimension {dim}"
    )]
    NotEnoughClasses {
        requested: usize,
        available: usize,
        dim: usize,
    },

    #[error("coordinates: scale {scale} outside the class interval [{birth}, {death})")]
    InvalidScale { scale: f32, birth: f32, death: f32 },

    #[error("coordinates: target_dim must be in 1..={max} for {points} input points, got {got}")]
    InvalidTargetDim {
        got: usize,
        max: usize,
        points: usize,
    },

    #[error(
        "coordinates: the balanced integer lift of the selected H^{dim} cocycle over Z/{prime} \
         is not closed at scale {scale}; choose another class, scale, or prime"
    )]
    NonIntegralCocycleLift { dim: usize, prime: u32, scale: f32 },

    #[error("coordinates: prime must be an odd prime below 65536, got {0}")]
    InvalidPrime(u32),

    #[error("{0} is not implemented yet")]
    NotImplemented(&'static str),
}

/// Crate-wide `Result` alias.
pub type Result<T> = std::result::Result<T, Error>;
