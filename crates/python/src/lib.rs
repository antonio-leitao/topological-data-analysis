//! Python bindings for persistence, filtration size, and coordinates.
//!
//! Dense input is float32 or float64. Sparse COO-like input is an undirected
//! distance graph. Square dense input is treated as points unless
//! `distance_matrix=True`, with a warning for the ambiguous case.

use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

type CooInput = (Vec<u32>, Vec<u32>, Vec<f32>, usize);
type CircularCoordinateResult<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Vec<f32>,
    Vec<usize>,
);

/// Flatten a 2-D float array in logical row-major order.
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
         cloud or an (n, n) distance matrix — or a scipy.sparse distance \
         matrix; for other dtypes convert with np.asarray(data, dtype=np.float32)",
    ))
}

/// Extract an object exposing `.tocoo()`, or return `None` for dense input.
fn extract_coo(data: &Bound<'_, PyAny>) -> PyResult<Option<CooInput>> {
    if !data.hasattr("tocoo")? {
        return Ok(None);
    }
    let coo = data.call_method0("tocoo")?;
    let (rows_n, cols_n): (usize, usize) = coo.getattr("shape")?.extract()?;
    if rows_n != cols_n {
        return Err(PyValueError::new_err(format!(
            "sparse input must be a square (n, n) distance matrix, got ({rows_n}, {cols_n})"
        )));
    }
    let rows = extract_indices(&coo.getattr("row")?)?;
    let cols = extract_indices(&coo.getattr("col")?)?;
    let vals = extract_values(&coo.getattr("data")?)?;
    Ok(Some((rows, cols, vals, rows_n)))
}

fn extract_indices(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    fn convert<T: TryInto<u32> + Copy>(values: impl Iterator<Item = T>) -> PyResult<Vec<u32>> {
        values
            .map(|x| {
                x.try_into().map_err(|_| {
                    PyValueError::new_err("sparse indices must be non-negative 32-bit integers")
                })
            })
            .collect()
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<i32>>() {
        return convert(arr.as_array().iter().copied());
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<i64>>() {
        return convert(arr.as_array().iter().copied());
    }
    Err(PyTypeError::new_err(
        "sparse index arrays must be int32 or int64",
    ))
}

fn extract_values(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f32>>() {
        return Ok(arr.as_array().iter().copied().collect());
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(arr.as_array().iter().map(|&x| x as f32).collect());
    }
    Err(PyTypeError::new_err(
        "sparse value arrays must be float32 or float64",
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

/// Warn without failing when square input is interpreted as points.
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
    let config = tda_core::Config {
        distance_matrix,
        threshold,
        peel,
        quotient,
        collapse,
        parallel,
    };

    // Sparse input always represents a distance graph.
    if let Some((rows, cols, vals, n)) = extract_coo(data)? {
        let barcode =
            tda_core::persistent_homology_sparse_with(&rows, &cols, &vals, n, max_dim, &config)
                .map_err(to_py_err)?;
        return barcode_to_py(py, barcode);
    }

    let (flat, rows, cols) = flatten_any(data)?;
    check_input(py, rows, cols, distance_matrix)?;

    let barcode =
        tda_core::persistent_homology_with(&flat, rows, max_dim, &config).map_err(to_py_err)?;

    barcode_to_py(py, barcode)
}

/// Circular coordinates with aligned intervals, scales, and group ids.
#[pyfunction]
#[pyo3(signature = (data, n_classes=1, classes=None, scale=None, distance_matrix=false, quotient=false, prime=41, parallel=true))]
#[allow(clippy::too_many_arguments)]
fn circular_coordinates<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    n_classes: usize,
    classes: Option<Vec<usize>>,
    scale: Option<f32>,
    distance_matrix: bool,
    quotient: bool,
    prime: u32,
    parallel: bool,
) -> PyResult<CircularCoordinateResult<'py>> {
    if data.hasattr("tocoo")? {
        return Err(PyTypeError::new_err(
            "tda.coords does not support sparse input",
        ));
    }
    let (flat, rows, cols) = flatten_any(data)?;
    check_input(py, rows, cols, distance_matrix)?;

    let config = tda_core::coords::Config {
        distance_matrix,
        quotient,
        scale,
        n_classes,
        classes,
        prime,
        parallel,
    };
    let cc = tda_core::coords::circular_with(&flat, rows, &config).map_err(to_py_err)?;

    let k = cc.k;
    let coordinates = cc
        .coordinates
        .into_pyarray(py)
        .reshape((rows, k))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let mut intervals_flat = Vec::with_capacity(k * 2);
    let mut scales = Vec::with_capacity(k);
    let mut groups = Vec::with_capacity(k);
    for class in &cc.classes {
        intervals_flat.push(class.birth);
        intervals_flat.push(class.death);
        scales.push(class.scale);
        groups.push(class.group);
    }
    let intervals = intervals_flat
        .into_pyarray(py)
        .reshape((k, 2))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((coordinates, intervals, scales, groups))
}

/// Fitted circular map exposed as `tda.coords.CircularMap`.
#[pyclass(name = "CircularMap", module = "tda.coords")]
struct PyCircularMap {
    inner: tda_core::coords::CircularMap,
}

#[pymethods]
impl PyCircularMap {
    /// Training angles with shape `(n, k)`.
    #[getter]
    fn coordinates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.inner
            .coordinates
            .clone()
            .into_pyarray(py)
            .reshape((self.inner.n_points(), self.inner.k))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Selected class intervals with shape `(k, 2)`.
    #[getter]
    fn intervals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let mut flat = Vec::with_capacity(self.inner.k * 2);
        for class in &self.inner.classes {
            flat.push(class.birth);
            flat.push(class.death);
        }
        flat.into_pyarray(py)
            .reshape((self.inner.k, 2))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn scales(&self) -> Vec<f32> {
        self.inner.classes.iter().map(|c| c.scale).collect()
    }

    /// Jointly decorrelated column blocks.
    #[getter]
    fn groups(&self) -> Vec<Vec<usize>> {
        let mut blocks: Vec<Vec<usize>> = Vec::new();
        for (column, class) in self.inner.classes.iter().enumerate() {
            while blocks.len() <= class.group {
                blocks.push(Vec::new());
            }
            blocks[class.group].push(column);
        }
        blocks
    }

    /// Evaluate `(m, k)` angles; uncovered entries are NaN.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if data.hasattr("tocoo")? {
            return Err(PyTypeError::new_err(
                "tda.coords does not support sparse input",
            ));
        }
        let (flat, rows, cols) = flatten_any(data)?;
        if cols != self.inner.ambient_dim() {
            return Err(PyValueError::new_err(format!(
                "transform expects {} columns (the fitted ambient dimension), got {}",
                self.inner.ambient_dim(),
                cols
            )));
        }
        let values = self.inner.transform(&flat, rows).map_err(to_py_err)?;
        values
            .into_pyarray(py)
            .reshape((rows, self.inner.k))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Fit a circular map.
#[pyfunction]
#[pyo3(signature = (data, n_classes=1, classes=None, scale=None, prime=41, parallel=true))]
fn circular_map<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    n_classes: usize,
    classes: Option<Vec<usize>>,
    scale: Option<f32>,
    prime: u32,
    parallel: bool,
) -> PyResult<PyCircularMap> {
    if data.hasattr("tocoo")? {
        return Err(PyTypeError::new_err(
            "tda.coords does not support sparse input",
        ));
    }
    let (flat, rows, cols) = flatten_any(data)?;
    check_input(py, rows, cols, false)?;

    let config = tda_core::coords::Config {
        n_classes,
        classes,
        scale,
        prime,
        parallel,
        ..tda_core::coords::Config::default()
    };
    let inner = tda_core::coords::circular_map_with(&flat, rows, &config).map_err(to_py_err)?;
    Ok(PyCircularMap { inner })
}

/// Number of simplices through `max_dim` in the truncated filtration.
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
    let config = tda_core::Config {
        distance_matrix,
        threshold,
        peel,
        quotient,
        collapse,
        parallel,
    };

    if let Some((rows, cols, vals, n)) = extract_coo(data)? {
        return tda_core::filtration_size_sparse_with(&rows, &cols, &vals, n, max_dim, &config)
            .map_err(to_py_err);
    }

    let (flat, rows, cols) = flatten_any(data)?;
    check_input(py, rows, cols, distance_matrix)?;

    tda_core::filtration_size_with(&flat, rows, max_dim, &config).map_err(to_py_err)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(persistent_homology, m)?)?;
    m.add_function(wrap_pyfunction!(filtration_size, m)?)?;
    m.add_function(wrap_pyfunction!(circular_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(circular_map, m)?)?;
    m.add_class::<PyCircularMap>()?;
    Ok(())
}
