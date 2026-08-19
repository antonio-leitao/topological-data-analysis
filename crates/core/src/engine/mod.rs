pub mod algorithm;
mod bitcsr;
mod heap;
mod reduction;
pub(crate) mod simplex;
#[cfg(feature = "coords")]
pub(crate) mod zp;

pub(crate) use bitcsr::BitCsrDistanceMatrix;
