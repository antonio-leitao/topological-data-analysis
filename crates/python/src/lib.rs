//! Python bindings for persistence, filtration size, and coordinates.
//!
//! Arrays are read in place: a C-contiguous float32 or float64 array, and the
//! int32/int64 index and float32/float64 value arrays of a scipy sparse matrix,
//! reach the core without a copy. Only strided views are copied, in logical
//! order. The GIL is released while the core runs.
//!
//! Sparse input is an undirected distance graph. Square dense input is treated
//! as points unless `distance_matrix=True`, with a warning for the ambiguous
//! case.

use std::borrow::Cow;

use numpy::ndarray::Dimension;
use numpy::{
    Element, IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray, PyReadonlyArray1,
    PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tda_core::{Real, SparseIndex};

/// `collapse`: `True`, `False`, or `"auto"`.
#[derive(FromPyObject)]
enum CollapseArg {
    Flag(bool),
    Name(String),
}

impl CollapseArg {
    fn resolve(self) -> PyResult<tda_core::Collapse> {
        match self {
            CollapseArg::Flag(on) => Ok(on.into()),
            CollapseArg::Name(name) if name == "auto" => Ok(tda_core::Collapse::Auto),
            CollapseArg::Name(name) => Err(PyValueError::new_err(format!(
                "collapse must be a bool or \"auto\", got {name:?}"
            ))),
        }
    }
}
type CircularCoordinateResult<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Vec<f32>,
    Vec<usize>,
);

/// Borrow an array's buffer when it is C-contiguous, or copy a strided view in
/// logical order. `as_slice` alone also accepts Fortran order, whose buffer is
/// column-major, so C order is checked first.
fn row_major<'a, T: Element + Copy, D: Dimension>(
    arr: &'a PyReadonlyArray<'_, T, D>,
) -> Cow<'a, [T]> {
    if arr.is_c_contiguous() {
        if let Ok(slice) = arr.as_slice() {
            return Cow::Borrowed(slice);
        }
    }
    Cow::Owned(arr.as_array().iter().copied().collect())
}

/// A 2-D point cloud or distance matrix, borrowed from numpy.
enum DenseArray<'py> {
    F32(PyReadonlyArray2<'py, f32>),
    F64(PyReadonlyArray2<'py, f64>),
}

impl<'py> DenseArray<'py> {
    fn extract(data: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(arr) = data.extract::<PyReadonlyArray2<f32>>() {
            return Ok(Self::F32(arr));
        }
        if let Ok(arr) = data.extract::<PyReadonlyArray2<f64>>() {
            return Ok(Self::F64(arr));
        }
        Err(PyTypeError::new_err(
            "expected a 2-D numpy array of float32 or float64 — an (n, d) point \
             cloud or an (n, n) distance matrix — or a scipy.sparse distance \
             matrix; for other dtypes convert with np.asarray(data, dtype=np.float32)",
        ))
    }

    fn shape(&self) -> (usize, usize) {
        let shape = match self {
            Self::F32(arr) => arr.shape(),
            Self::F64(arr) => arr.shape(),
        };
        (shape[0], shape[1])
    }

    /// Row-major `f32` values, for the coordinate APIs, which take `f32` only.
    /// float64 input is converted in one pass.
    fn to_f32(&self) -> Cow<'_, [f32]> {
        match self {
            Self::F32(arr) => row_major(arr),
            Self::F64(arr) => Cow::Owned(arr.as_array().iter().map(|&x| x as f32).collect()),
        }
    }
}

/// One sparse index array. scipy gives a matrix's index arrays one dtype.
enum IndexArray<'py> {
    I32(PyReadonlyArray1<'py, i32>),
    I64(PyReadonlyArray1<'py, i64>),
}

impl<'py> IndexArray<'py> {
    fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<i32>>() {
            return Ok(Self::I32(arr));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<i64>>() {
            return Ok(Self::I64(arr));
        }
        Err(PyTypeError::new_err(
            "sparse index arrays must be int32 or int64",
        ))
    }

    /// As `i64`, for the rare pair of index arrays with different dtypes.
    fn widened(&self) -> Cow<'_, [i64]> {
        match self {
            Self::I32(arr) => Cow::Owned(arr.as_array().iter().map(|&x| x as i64).collect()),
            Self::I64(arr) => row_major(arr),
        }
    }
}

enum ValueArray<'py> {
    F32(PyReadonlyArray1<'py, f32>),
    F64(PyReadonlyArray1<'py, f64>),
}

impl<'py> ValueArray<'py> {
    fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<f32>>() {
            return Ok(Self::F32(arr));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>() {
            return Ok(Self::F64(arr));
        }
        Err(PyTypeError::new_err(
            "sparse value arrays must be float32 or float64",
        ))
    }
}

/// A scipy sparse distance graph: CSR/CSC `(indptr, indices, data)` read as
/// stored, or `(row, col, data)` from `tocoo()` for every other format.
struct SparseInput<'py> {
    compressed: bool,
    outer: IndexArray<'py>,
    inner: IndexArray<'py>,
    values: ValueArray<'py>,
    n: usize,
}

impl<'py> SparseInput<'py> {
    /// `None` for input without `tocoo()`, which is dense.
    fn extract(data: &Bound<'py, PyAny>) -> PyResult<Option<Self>> {
        if !data.hasattr("tocoo")? {
            return Ok(None);
        }
        let (rows, cols): (usize, usize) = data.getattr("shape")?.extract()?;
        if rows != cols {
            return Err(PyValueError::new_err(format!(
                "sparse input must be a square (n, n) distance matrix, got ({rows}, {cols})"
            )));
        }
        let format = data
            .getattr("format")
            .and_then(|f| f.extract::<String>())
            .ok();
        // CSC is CSR of the transpose, and the graph is undirected.
        let compressed = matches!(format.as_deref(), Some("csr" | "csc"));
        let (outer, inner, values) = if compressed {
            (
                data.getattr("indptr")?,
                data.getattr("indices")?,
                data.getattr("data")?,
            )
        } else {
            let coo = data.call_method0("tocoo")?;
            (
                coo.getattr("row")?,
                coo.getattr("col")?,
                coo.getattr("data")?,
            )
        };
        Ok(Some(Self {
            compressed,
            outer: IndexArray::extract(&outer)?,
            inner: IndexArray::extract(&inner)?,
            values: ValueArray::extract(&values)?,
            n: rows,
        }))
    }

    fn run<J: Job>(&self, py: Python<'_>, job: &J) -> PyResult<J::Out> {
        match &self.values {
            ValueArray::F32(vals) => self.with_indices(py, &row_major(vals), job),
            ValueArray::F64(vals) => self.with_indices(py, &row_major(vals), job),
        }
    }

    fn with_indices<V: Real, J: Job>(
        &self,
        py: Python<'_>,
        vals: &[V],
        job: &J,
    ) -> PyResult<J::Out> {
        match (&self.outer, &self.inner) {
            (IndexArray::I32(a), IndexArray::I32(b)) => {
                self.detached(py, &row_major(a), &row_major(b), vals, job)
            }
            (IndexArray::I64(a), IndexArray::I64(b)) => {
                self.detached(py, &row_major(a), &row_major(b), vals, job)
            }
            (a, b) => self.detached(py, &a.widened(), &b.widened(), vals, job),
        }
    }

    fn detached<I: SparseIndex, V: Real, J: Job>(
        &self,
        py: Python<'_>,
        outer: &[I],
        inner: &[I],
        vals: &[V],
        job: &J,
    ) -> PyResult<J::Out> {
        let (compressed, n) = (self.compressed, self.n);
        py.detach(|| {
            if compressed {
                job.compressed(outer, inner, vals, n)
            } else {
                job.coo(outer, inner, vals, n)
            }
        })
        .map_err(to_py_err)
    }
}

/// A core computation, generic over the element types it reads in place.
trait Job: Sync {
    type Out: Send;

    fn dense<T: Real>(&self, data: &[T], n: usize) -> tda_core::Result<Self::Out>;

    fn coo<I: SparseIndex, V: Real>(
        &self,
        rows: &[I],
        cols: &[I],
        vals: &[V],
        n: usize,
    ) -> tda_core::Result<Self::Out>;

    fn compressed<I: SparseIndex, V: Real>(
        &self,
        indptr: &[I],
        indices: &[I],
        vals: &[V],
        n: usize,
    ) -> tda_core::Result<Self::Out>;
}

struct Barcode<'a> {
    max_dim: usize,
    config: &'a tda_core::Config,
}

impl Job for Barcode<'_> {
    type Out = tda_core::BarcodeResult;

    fn dense<T: Real>(&self, data: &[T], n: usize) -> tda_core::Result<Self::Out> {
        tda_core::persistent_homology_with(data, n, self.max_dim, self.config)
    }

    fn coo<I: SparseIndex, V: Real>(
        &self,
        rows: &[I],
        cols: &[I],
        vals: &[V],
        n: usize,
    ) -> tda_core::Result<Self::Out> {
        tda_core::persistent_homology_sparse_with(rows, cols, vals, n, self.max_dim, self.config)
    }

    fn compressed<I: SparseIndex, V: Real>(
        &self,
        indptr: &[I],
        indices: &[I],
        vals: &[V],
        n: usize,
    ) -> tda_core::Result<Self::Out> {
        tda_core::persistent_homology_csr_with(indptr, indices, vals, n, self.max_dim, self.config)
    }
}

struct FiltrationSize<'a> {
    max_dim: usize,
    config: &'a tda_core::Config,
}

impl Job for FiltrationSize<'_> {
    type Out = usize;

    fn dense<T: Real>(&self, data: &[T], n: usize) -> tda_core::Result<usize> {
        tda_core::filtration_size_with(data, n, self.max_dim, self.config)
    }

    fn coo<I: SparseIndex, V: Real>(
        &self,
        rows: &[I],
        cols: &[I],
        vals: &[V],
        n: usize,
    ) -> tda_core::Result<usize> {
        tda_core::filtration_size_sparse_with(rows, cols, vals, n, self.max_dim, self.config)
    }

    fn compressed<I: SparseIndex, V: Real>(
        &self,
        indptr: &[I],
        indices: &[I],
        vals: &[V],
        n: usize,
    ) -> tda_core::Result<usize> {
        tda_core::filtration_size_csr_with(indptr, indices, vals, n, self.max_dim, self.config)
    }
}

/// Read `data` in place and run `job` without the GIL.
///
/// Another Python thread writing to the same array during the call would race
/// with the read, as with any extension that releases the GIL over a buffer.
fn run<J: Job>(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    distance_matrix: bool,
    job: &J,
) -> PyResult<J::Out> {
    // Sparse input always represents a distance graph.
    if let Some(sparse) = SparseInput::extract(data)? {
        return sparse.run(py, job);
    }

    let dense = DenseArray::extract(data)?;
    let (rows, cols) = dense.shape();
    check_input(py, rows, cols, distance_matrix)?;
    match &dense {
        DenseArray::F32(arr) => {
            let data = row_major(arr);
            py.detach(|| job.dense(&data, rows))
        }
        DenseArray::F64(arr) => {
            let data = row_major(arr);
            py.detach(|| job.dense(&data, rows))
        }
    }
    .map_err(to_py_err)
}

/// Coordinates take dense input only.
fn reject_sparse(data: &Bound<'_, PyAny>) -> PyResult<()> {
    if data.hasattr("tocoo")? {
        return Err(PyTypeError::new_err(
            "tda.coords does not support sparse input",
        ));
    }
    Ok(())
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
#[pyo3(signature = (data, max_dim=1, threshold=None, distance_matrix=false, quotient=false, peel=true, collapse=CollapseArg::Name("auto".into()), parallel=true))]
#[allow(clippy::too_many_arguments)]
fn persistent_homology<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    max_dim: usize,
    threshold: Option<f32>,
    distance_matrix: bool,
    quotient: bool,
    peel: bool,
    collapse: CollapseArg,
    parallel: bool,
) -> PyResult<Vec<Bound<'py, PyArray2<f32>>>> {
    let config = tda_core::Config {
        distance_matrix,
        threshold,
        peel,
        quotient,
        collapse: collapse.resolve()?,
        parallel,
    };
    let job = Barcode {
        max_dim,
        config: &config,
    };
    let barcode = run(py, data, distance_matrix, &job)?;
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
    reject_sparse(data)?;
    let dense = DenseArray::extract(data)?;
    let (rows, cols) = dense.shape();
    check_input(py, rows, cols, distance_matrix)?;
    let flat = dense.to_f32();

    let config = tda_core::coords::Config {
        distance_matrix,
        quotient,
        scale,
        n_classes,
        classes,
        prime,
        parallel,
    };
    let cc = py
        .detach(|| tda_core::coords::circular_with(&flat, rows, &config))
        .map_err(to_py_err)?;

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
        reject_sparse(data)?;
        let dense = DenseArray::extract(data)?;
        let (rows, cols) = dense.shape();
        if cols != self.inner.ambient_dim() {
            return Err(PyValueError::new_err(format!(
                "transform expects {} columns (the fitted ambient dimension), got {}",
                self.inner.ambient_dim(),
                cols
            )));
        }
        let flat = dense.to_f32();
        let inner = &self.inner;
        let values = py
            .detach(|| inner.transform(&flat, rows))
            .map_err(to_py_err)?;
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
    reject_sparse(data)?;
    let dense = DenseArray::extract(data)?;
    let (rows, cols) = dense.shape();
    check_input(py, rows, cols, false)?;
    let flat = dense.to_f32();

    let config = tda_core::coords::Config {
        n_classes,
        classes,
        scale,
        prime,
        parallel,
        ..tda_core::coords::Config::default()
    };
    let inner = py
        .detach(|| tda_core::coords::circular_map_with(&flat, rows, &config))
        .map_err(to_py_err)?;
    Ok(PyCircularMap { inner })
}

/// Number of simplices through `max_dim` in the truncated filtration.
#[pyfunction]
#[pyo3(signature = (data, max_dim=1, threshold=None, distance_matrix=false, quotient=false, peel=true, collapse=CollapseArg::Name("auto".into()), parallel=true))]
#[allow(clippy::too_many_arguments)]
fn filtration_size<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    max_dim: usize,
    threshold: Option<f32>,
    distance_matrix: bool,
    quotient: bool,
    peel: bool,
    collapse: CollapseArg,
    parallel: bool,
) -> PyResult<usize> {
    let config = tda_core::Config {
        distance_matrix,
        threshold,
        peel,
        quotient,
        collapse: collapse.resolve()?,
        parallel,
    };
    let job = FiltrationSize {
        max_dim,
        config: &config,
    };
    run(py, data, distance_matrix, &job)
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
