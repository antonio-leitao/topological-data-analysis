pub mod algorithm;
mod bitcsr;
pub(crate) mod csr;
pub(crate) mod distance;
mod filtration;
mod heap;
mod reduction;
pub(crate) mod simplex;

pub(crate) use bitcsr::{bitcsr_from_distance_matrix, BitCsrDistanceMatrix};
pub(crate) use csr::CsrDistanceMatrix;
pub(crate) use distance::DistanceMatrix;
