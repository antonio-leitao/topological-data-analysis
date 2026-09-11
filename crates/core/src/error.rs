use thiserror::Error;

/// Errors returned by the core and coordinate APIs.
///
/// Dense distance matrices must be symmetric and non-negative with a zero
/// diagonal. The crate does not scan them to enforce these preconditions.
/// Infinite entries represent disconnected pairs.
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

    #[error("coordinates: requested class {requested} but only {available} H1 classes exist")]
    NotEnoughClasses { requested: usize, available: usize },

    #[error("coordinates: scale {scale} outside the class interval [{birth}, {death})")]
    InvalidScale { scale: f32, birth: f32, death: f32 },

    #[error(
        "coordinates: the balanced integer lift of the selected H^1 cocycle over Z/{prime} \
         is not closed at scale {scale}; choose another class, scale, or prime"
    )]
    NonIntegralCocycleLift { prime: u32, scale: f32 },

    #[error("coordinates: prime must be an odd prime below 65536, got {0}")]
    InvalidPrime(u32),

    #[error("{0} is not implemented yet")]
    NotImplemented(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;
