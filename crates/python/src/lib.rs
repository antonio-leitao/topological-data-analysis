//! Python bindings for the `tda` core crate.
//!
//! Two endpoints. The point-cloud / distance-matrix split collapses into a
//! single `distance_matrix` flag: the 2-D NumPy array already carries the shape,
//! so for a point cloud the ambient dimension is the column count (the core
//! re-infers it as `len / n`), and `distance_matrix` is the only disambiguator
//! when the array is square (`d == n`).
//!
//! - `persistent_homology(data, max_dim=1, threshold=None,
//!   distance_matrix=False, quotient=False, peel=True, collapse=False,
//!   parallel=True)`
//!   → list of `(k, 2)` float32 arrays `[[birth, death], …]`, one per
//!   homology dimension `0..=max_dim` (Ripser-compatible `dgms` layout;
//!   essential classes carry `death = inf`).
//!
//! - `filtration_size(data, max_dim=1, threshold=None, distance_matrix=False,
//!   quotient=False, peel=True, collapse=False, parallel=True)`
//!   → int, the simplex count of the filtered complex (dimension ≤ max_dim).
//!
//! `data` may be float32 or float64; float64 input is cast to float32 on the
//! way in (the core computes in f32). `data` may also be a `scipy.sparse`
//! matrix (anything exposing `.tocoo()`): sparse input is always treated as a
//! distance graph — entries are edges, missing pairs are never connected
//! (infinite distance), duplicates resolve to the minimum, and disconnected
//! inputs truncate soundly per component. scipy itself is never imported.
//!
//! `peel` (exact truncation, ON by default — identical barcode, strictly less
//! work) and `collapse` (exact edge collapse) preserve the barcode; `quotient`
//! is a log(3)-interleaved approximation. Applied in the canonical order
//! peel → quotient → collapse.
//!
//! When `data` is square and `distance_matrix=False`, the input is ambiguous
//! (square point cloud vs. distance matrix); a `UserWarning` is emitted naming
//! the interpretation actually used.

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
         cloud or an (n, n) distance matrix — or a scipy.sparse distance \
         matrix; for other dtypes convert with np.asarray(data, dtype=np.float32)",
    ))
}

/// Sparse (scipy-style) input: anything exposing `.tocoo()`. Returns `None`
/// for dense arrays so the caller falls through to `flatten_any`.
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
    let config = tda_core::Config {
        distance_matrix,
        threshold,
        peel,
        quotient,
        collapse,
        parallel,
    };

    // scipy.sparse input is inherently a distance graph; the
    // `distance_matrix` flag is irrelevant on this path.
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

/// Real projective coordinates (H¹ mod 2 → RP^target_dim). Returns
/// `(coordinates (n, target_dim + 1), birth, death, scale)`. Wrapped by
/// `tda.coords.projective`.
#[pyfunction]
#[pyo3(signature = (data, class_index=0, target_dim=2, scale=None, distance_matrix=false, quotient=false, parallel=true))]
#[allow(clippy::too_many_arguments)]
fn projective_coordinates<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    class_index: usize,
    target_dim: usize,
    scale: Option<f32>,
    distance_matrix: bool,
    quotient: bool,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray2<f32>>, f32, f32, f32)> {
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
        target_dim,
        class_index,
        parallel,
        ..tda_core::coords::Config::default()
    };
    let pc = tda_core::coords::projective_with(&flat, rows, &config).map_err(to_py_err)?;

    let width = pc.target_dim + 1;
    let arr = pc
        .coordinates
        .into_pyarray(py)
        .reshape((rows, width))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((arr, pc.birth, pc.death, pc.scale))
}

/// Circular coordinates (H¹ mod p → S¹). Returns `(coordinates (n, k),
/// intervals (k, 2), scales, groups)`; wrapped by `tda.coords.circular`.
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
        ..tda_core::coords::Config::default()
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

/// Complex projective coordinates (H² mod p → CP^target_dim). Returns
/// `(coordinates (n, (target_dim + 1) * 2) interleaved re/im, birth, death,
/// scale)`; wrapped by `tda.coords.complex_projective`, which views the
/// array as complex64.
#[pyfunction]
#[pyo3(signature = (data, class_index=0, target_dim=1, scale=None, distance_matrix=false, quotient=false, prime=41, parallel=true))]
#[allow(clippy::too_many_arguments)]
fn complex_projective_coordinates<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    class_index: usize,
    target_dim: usize,
    scale: Option<f32>,
    distance_matrix: bool,
    quotient: bool,
    prime: u32,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray2<f32>>, f32, f32, f32)> {
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
        prime,
        target_dim,
        class_index,
        parallel,
        ..tda_core::coords::Config::default()
    };
    let cp = tda_core::coords::complex_projective_with(&flat, rows, &config).map_err(to_py_err)?;

    let width = (cp.target_dim + 1) * 2;
    let arr = cp
        .coordinates
        .into_pyarray(py)
        .reshape((rows, width))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((arr, cp.birth, cp.death, cp.scale))
}

/// A fitted circular-coordinates map (`tda.coords.circular_map`). Holds the
/// model in Rust; `transform` evaluates the same circle-valued map on new
/// points, NaN outside the cover.
#[pyclass(name = "CircularMap", module = "tda.coords")]
struct PyCircularMap {
    inner: tda_core::coords::CircularMap,
}

#[pymethods]
impl PyCircularMap {
    /// Training coordinates, `(n, k)` angles in `[0, 2π)`.
    #[getter]
    fn coordinates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.inner
            .coordinates
            .clone()
            .into_pyarray(py)
            .reshape((self.inner.n_points(), self.inner.k))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// `(k, 2)` array of `[birth, death]` for the classes used.
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

    /// Filtration value each column was computed at.
    #[getter]
    fn scales(&self) -> Vec<f32> {
        self.inner.classes.iter().map(|c| c.scale).collect()
    }

    /// Toroidal blocks: columns listed together were jointly decorrelated.
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

    /// Evaluate the fitted map on new points: `(m, k)` angles in `[0, 2π)`,
    /// NaN where a point is outside a column's cover.
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

/// Fit a circular-coordinates map (see `tda.coords.circular_map`).
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

/// A fitted real-projective map (`tda.coords.projective_map`).
#[pyclass(name = "ProjectiveMap", module = "tda.coords")]
struct PyProjectiveMap {
    inner: tda_core::coords::ProjectiveMap,
}

#[pymethods]
impl PyProjectiveMap {
    /// Training coordinates, `(n, target_dim + 1)` unit rows up to sign.
    #[getter]
    fn coordinates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.inner
            .coordinates
            .clone()
            .into_pyarray(py)
            .reshape((self.inner.n_points(), self.inner.target_dim + 1))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn interval(&self) -> (f32, f32) {
        (self.inner.birth, self.inner.death)
    }

    #[getter]
    fn scale(&self) -> f32 {
        self.inner.scale
    }

    #[getter]
    fn target_dim(&self) -> usize {
        self.inner.target_dim
    }

    /// Evaluate the fitted map on new points: `(m, target_dim + 1)` unit
    /// rows up to sign; NaN rows outside the cover.
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
            .reshape((rows, self.inner.target_dim + 1))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Fit a real-projective map (see `tda.coords.projective_map`).
#[pyfunction]
#[pyo3(signature = (data, class_index=0, target_dim=2, scale=None, parallel=true))]
fn projective_map<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    class_index: usize,
    target_dim: usize,
    scale: Option<f32>,
    parallel: bool,
) -> PyResult<PyProjectiveMap> {
    if data.hasattr("tocoo")? {
        return Err(PyTypeError::new_err(
            "tda.coords does not support sparse input",
        ));
    }
    let (flat, rows, cols) = flatten_any(data)?;
    check_input(py, rows, cols, false)?;

    let config = tda_core::coords::Config {
        class_index,
        target_dim,
        scale,
        parallel,
        ..tda_core::coords::Config::default()
    };
    let inner = tda_core::coords::projective_map_with(&flat, rows, &config).map_err(to_py_err)?;
    Ok(PyProjectiveMap { inner })
}

/// A fitted complex-projective map, raw interleaved-re/im form; wrapped by
/// `tda.coords.ComplexProjectiveMap`, which presents complex64 arrays.
#[pyclass(name = "_ComplexProjectiveMapCore", module = "tda.coords")]
struct PyComplexProjectiveMap {
    inner: tda_core::coords::ComplexProjectiveMap,
}

#[pymethods]
impl PyComplexProjectiveMap {
    /// Training coordinates, `(n, (target_dim + 1) * 2)` interleaved re/im.
    #[getter]
    fn coordinates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.inner
            .coordinates
            .clone()
            .into_pyarray(py)
            .reshape((self.inner.n_points(), (self.inner.target_dim + 1) * 2))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn interval(&self) -> (f32, f32) {
        (self.inner.birth, self.inner.death)
    }

    #[getter]
    fn scale(&self) -> f32 {
        self.inner.scale
    }

    #[getter]
    fn target_dim(&self) -> usize {
        self.inner.target_dim
    }

    /// Evaluate on new points: `(m, (target_dim + 1) * 2)` interleaved re/im.
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
            .reshape((rows, (self.inner.target_dim + 1) * 2))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Fit a complex-projective map (see `tda.coords.complex_projective_map`).
#[pyfunction]
#[pyo3(signature = (data, class_index=0, target_dim=1, scale=None, prime=41, parallel=true))]
fn complex_projective_map<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    class_index: usize,
    target_dim: usize,
    scale: Option<f32>,
    prime: u32,
    parallel: bool,
) -> PyResult<PyComplexProjectiveMap> {
    if data.hasattr("tocoo")? {
        return Err(PyTypeError::new_err(
            "tda.coords does not support sparse input",
        ));
    }
    let (flat, rows, cols) = flatten_any(data)?;
    check_input(py, rows, cols, false)?;

    let config = tda_core::coords::Config {
        class_index,
        target_dim,
        scale,
        prime,
        parallel,
        ..tda_core::coords::Config::default()
    };
    let inner =
        tda_core::coords::complex_projective_map_with(&flat, rows, &config).map_err(to_py_err)?;
    Ok(PyComplexProjectiveMap { inner })
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
    m.add_function(wrap_pyfunction!(projective_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(circular_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(complex_projective_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(circular_map, m)?)?;
    m.add_function(wrap_pyfunction!(projective_map, m)?)?;
    m.add_function(wrap_pyfunction!(complex_projective_map, m)?)?;
    m.add_class::<PyCircularMap>()?;
    m.add_class::<PyProjectiveMap>()?;
    m.add_class::<PyComplexProjectiveMap>()?;
    Ok(())
}
