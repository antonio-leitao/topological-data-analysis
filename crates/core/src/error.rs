use thiserror::Error;

/// Errors returned by the core and coordinate APIs.
///
/// Only the strict lower triangle of a dense distance matrix is read, so
/// symmetry and the diagonal are not checked. Negative entries in that triangle
/// are rejected. Infinite entries represent disconnected pairs.
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

    #[error("sparse index {index} is out of range for n={n}")]
    SparseIndexOutOfRange { index: i128, n: usize },

    #[error("compressed sparse input is malformed: {0}")]
    InvalidCompressed(&'static str),

    #[error(
        "point {point} has a non-finite coordinate ({value}); remove or impute it \
         before computing persistence"
    )]
    NonFiniteCoordinate { point: usize, value: f32 },

    #[error(
        "distances must be non-negative, got d({u}, {v}) = {value}; clip rounding \
         noise to zero (e.g. np.clip(d, 0, None)) before computing persistence"
    )]
    NegativeDistance { u: usize, v: usize, value: f32 },

    #[error("graph too large: {what} is {got}, above the 32-bit limit of {max}")]
    GraphTooLarge {
        what: &'static str,
        got: usize,
        max: usize,
    },

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
