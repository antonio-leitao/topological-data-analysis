use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyErr;

/// Map a `tda::Error` to the appropriate Python exception.
///
/// User-correctable input problems → `ValueError`.
/// Architectural caps (vertex count, dim) → `RuntimeError`, since these are
/// limits of the library implementation rather than mistakes in user input.
pub fn into_py(e: tda_core::Error) -> PyErr {
    use tda_core::Error::*;
    match &e {
        TooManyPoints { .. } | DimTooLarge { .. } => PyRuntimeError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}
