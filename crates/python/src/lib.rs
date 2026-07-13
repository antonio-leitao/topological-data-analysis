//! Python bindings for the `tda` core crate.
//!
//! Two endpoints. The point-cloud / distance-matrix split collapses into a
//! single `distance_matrix` flag: the 2-D NumPy array already carries the shape,
//! so for a point cloud the ambient dimension is the column count (the core
//! re-infers it as `len / n`), and `distance_matrix` is the only disambiguator
//! when the array is square (`d == n`).
//!
//!   * `persistent_homology(data, max_dim=1, threshold=None,
//!                          distance_matrix=False, quotient=False, peel=True,
//!                          collapse=False, parallel=True)`
//!       → list of `(k, 2)` float32 arrays `[[birth, death], …]`, one per
//!         homology dimension `0..=max_dim` (Ripser-compatible `dgms` layout;
//!         essential classes carry `death = inf`).
//!
//!   * `filtration_size(data, max_dim=1, threshold=None, distance_matrix=False,
//!                      quotient=False, peel=True, collapse=False,
//!                      parallel=True)`
//!       → int, the simplex count of the filtered complex (dimension ≤ max_dim).
//!
//! `data` may be float32 or float64; float64 input is cast to float32 on the
//! way in (the core computes in f32).
//!
//! `peel` (exact truncation, ON by default — identical barcode, strictly less
//! work) and `collapse` (exact edge collapse) preserve the barcode; `quotient`
//! is a log(3)-interleaved approximation. Applied in the canonical order
//! peel → quotient → collapse.
//!
//! When `data` is square and `distance_matrix=False`, the input is ambiguous
//! (square point cloud vs. distance matrix); a `UserWarning` is emitted naming
//! the interpretation actually used.

use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Flatten a 2-D array (float32 or float64) into a row-major `Vec<f32>` plus
/// its `(rows, cols)`. Float64 input is cast element-wise. `ndarray`'s logical
/// iteration is row-major regardless of the underlying memory layout, so this
/// is correct for non-contiguous / transposed views too.
fn flatten_any(data: &Bound<'_, PyAny>) -> PyResult<(Vec<f32>, usize, usize)> {
    if let Ok(arr) = data.extract::<PyReadonlyArray2<f32>>() {
        let arr = arr.as_array();
        let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
        return Ok((arr.iter().copied().collect(), rows, cols));
    }
    if let Ok(arr) = data.extract::<PyReadonlyArray2<f64>>() {
        let arr = arr.as_array();
        let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
        return Ok((arr.iter().map(|&x| x as f32).collect(), rows, cols));
    }
    Err(PyTypeError::new_err(
        "expected a 2-D numpy array of float32 or float64 — an (n, d) point \
         cloud or an (n, n) distance matrix; for other dtypes convert with \
         np.asarray(data, dtype=np.float32)",
    ))
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

/// Best-effort `UserWarning` for the `d == n` footgun: square data on the points
/// path is byte-identical to a distance matrix. Fired only when ambiguous, so it
/// costs nothing on the common non-square call. A failed warn must never abort
/// the computation.
fn warn_square_points(py: Python<'_>, rows: usize, cols: usize) {
    if rows != cols {
        return;
    }
    let msg = format!(
        "tda: input is square ({rows}×{rows}); interpreting it as {rows} points in \
         {rows}-D. Pass distance_matrix=True if it is a distance matrix."
    );
    if let Ok(warnings) = py.import("warnings") {
        let kwargs = PyDict::new(py);
        let _ = kwargs.set_item("stacklevel", 2);
        let _ = warnings.call_method("warn", (msg,), Some(&kwargs));
    }
}

fn barcode_to_py<'py>(
    py: Python<'py>,
    barcode: tda_core::BarcodeResult,
) -> PyResult<Vec<Bound<'py, PyArray2<f32>>>> {
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

/// Branch-specific input guard, shared by both endpoints. For a distance matrix,
/// enforce squareness; for points, warn on the square (`d == n`) ambiguity.
#[inline]
fn check_input(py: Python<'_>, rows: usize, cols: usize, distance_matrix: bool) -> PyResult<()> {
    if distance_matrix {
        require_square(rows, cols)?;
    } else {
        warn_square_points(py, rows, cols);
    }
    Ok(())
}

/// Vietoris–Rips persistent homology.
#[pyfunction]
#[pyo3(signature = (data, max_dim=1, threshold=None, distance_matrix=false, quotient=false, peel=true, collapse=false, parallel=true))]
#[allow(clippy::too_many_arguments)]
fn persistent_homology<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    max_dim: usize,
    threshold: Option<f32>,
    distance_matrix: bool,
    quotient: bool,
    peel: bool,
    collapse: bool,
    parallel: bool,
) -> PyResult<Vec<Bound<'py, PyArray2<f32>>>> {
    let (flat, rows, cols) = flatten_any(data)?;

    check_input(py, rows, cols, distance_matrix)?;

    let config = tda_core::Config {
        distance_matrix,
        threshold,
        peel,
        quotient,
        collapse,
        parallel,
    };
    let barcode =
        tda_core::persistent_homology_with(&flat, rows, max_dim, &config).map_err(to_py_err)?;

    barcode_to_py(py, barcode)
}

/// Truncated Vietoris–Rips filtration size (simplices of dimension ≤ max_dim).
#[pyfunction]
#[pyo3(signature = (data, max_dim=1, threshold=None, distance_matrix=false, quotient=false, peel=true, collapse=false, parallel=true))]
#[allow(clippy::too_many_arguments)]
fn filtration_size<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    max_dim: usize,
    threshold: Option<f32>,
    distance_matrix: bool,
    quotient: bool,
    peel: bool,
    collapse: bool,
    parallel: bool,
) -> PyResult<usize> {
    let (flat, rows, cols) = flatten_any(data)?;

    check_input(py, rows, cols, distance_matrix)?;

    let config = tda_core::Config {
        distance_matrix,
        threshold,
        peel,
        quotient,
        collapse,
        parallel,
    };
    tda_core::filtration_size_with(&flat, rows, max_dim, &config).map_err(to_py_err)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(persistent_homology, m)?)?;
    m.add_function(wrap_pyfunction!(filtration_size, m)?)?;
    Ok(())
}
