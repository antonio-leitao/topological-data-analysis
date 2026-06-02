//! Python bindings for the `tda` core crate.
//!
//! Two endpoints, mirroring the core API but collapsing the point-cloud /
//! distance-matrix split into a single `distance_matrix` flag (the 2-D NumPy
//! array already carries the shape):
//!
//!   * `persistent_homology(data, max_dim=1, threshold=None,
//!                          distance_matrix=False, quotient=False, peel=False)`
//!       → list of `(k, 2)` float32 arrays `[[birth, death], …]`, one per
//!         homology dimension `0..=max_dim` (Ripser-compatible `dgms` layout;
//!         essential classes carry `death = inf`).
//!
//!   * `filtration_size(data, max_dim=1, threshold=None, distance_matrix=False)`
//!       → int, the simplex count of the filtered complex (dimension ≤ max_dim).
//!
//! Cargo.toml (this crate is separate from the core; e.g. `crates/python`):
//!
//! ```toml
//! [lib]
//! name = "tda"
//! crate-type = ["cdylib"]
//!
//! [dependencies]
//! # Point `package` at the core crate's actual name if it isn't `tda`.
//! tda_core = { path = "../core", package = "tda" }
//! pyo3 = { version = "0.22", features = ["extension-module"] }
//! numpy = "0.22"
//! ```

use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Flatten a 2-D array into a row-major `Vec<f32>` plus its `(rows, cols)`.
/// `ndarray`'s logical iteration is row-major regardless of the underlying
/// memory layout, so this is correct for non-contiguous / transposed views too.
fn flatten(data: &PyReadonlyArray2<'_, f32>) -> (Vec<f32>, usize, usize) {
    let arr = data.as_array();
    let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
    let flat: Vec<f32> = arr.iter().copied().collect();
    (flat, rows, cols)
}

#[inline]
fn require_square(rows: usize, cols: usize) -> PyResult<()> {
    if rows != cols {
        return Err(PyValueError::new_err(format!(
            "distance_matrix=True expects a square (n, n) array, got ({rows}, {cols})"
        )));
    }
    Ok(())
}

#[inline]
fn to_py_err(e: tda_core::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Vietoris–Rips persistent homology.
#[pyfunction]
#[pyo3(signature = (data, max_dim=1, threshold=None, distance_matrix=false, quotient=false, peel=false))]
fn persistent_homology<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    max_dim: usize,
    threshold: Option<f32>,
    distance_matrix: bool,
    quotient: bool,
    peel: bool,
) -> PyResult<Vec<Bound<'py, PyArray2<f32>>>> {
    let (flat, rows, cols) = flatten(&data);

    let barcode = if distance_matrix {
        require_square(rows, cols)?;
        tda_core::persistent_homology_from_distances(
            &flat, rows, max_dim, threshold, quotient, peel,
        )
    } else {
        tda_core::persistent_homology(&flat, rows, cols, max_dim, threshold, quotient, peel)
    }
    .map_err(to_py_err)?;

    // One (k, 2) array per dimension. `into_flat_intervals` hands back the
    // intervals already laid out as [b0, d0, b1, d1, …], so each dimension is a
    // single zero-copy reshape.
    let mut dgms = Vec::with_capacity(barcode.intervals.len());
    for (k, flat) in barcode.into_flat_intervals() {
        let arr = flat
            .into_pyarray(py)
            .reshape((k, 2))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        dgms.push(arr);
    }
    Ok(dgms)
}

/// Truncated Vietoris–Rips filtration size (simplices of dimension ≤ max_dim).
#[pyfunction]
#[pyo3(signature = (data, max_dim=1, threshold=None, distance_matrix=false, quotient=false,peel=false))]
fn filtration_size(
    data: PyReadonlyArray2<'_, f32>,
    max_dim: usize,
    threshold: Option<f32>,
    distance_matrix: bool,
    quotient: bool,
    peel: bool,
) -> PyResult<usize> {
    let (flat, rows, cols) = flatten(&data);

    if distance_matrix {
        require_square(rows, cols)?;
        tda_core::filtration_size_from_distances(&flat, rows, max_dim, threshold, quotient, peel)
    } else {
        tda_core::filtration_size(&flat, rows, cols, max_dim, threshold, quotient, peel)
    }
    .map_err(to_py_err)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(persistent_homology, m)?)?;
    m.add_function(wrap_pyfunction!(filtration_size, m)?)?;
    Ok(())
}
