mod error;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Compute Vietoris–Rips persistent homology of a point cloud or distance matrix.
///
/// Parameters
/// ----------
/// data : np.ndarray
///     Float32 array. Shape ``(n, d)`` for a point cloud, or ``(n, n)`` for a
///     distance matrix (set ``distance_matrix=True``). C-contiguous input is
///     zero-copy; non-contiguous input is copied internally.
/// max_dim : int, default=1
///     Highest homology dimension to compute. Intervals are returned for every
///     dimension ``0..=max_dim``. Capped at 4.
/// threshold : float or None, default=None
///     Maximum filtration value. If ``None``, the enclosing radius of the data
///     is used. If given, the effective threshold is ``min(threshold, enclosing_radius)``.
/// distance_matrix : bool, default=False
///     If ``True``, ``data`` is interpreted as a precomputed square distance
///     matrix. Symmetry, zero diagonal, and non-negativity are assumed and
///     not validated. Entries ``>= f32::MAX`` are treated as "no edge".
///
/// Returns
/// -------
/// list of np.ndarray
///     One array per dimension, where ``result[d]`` has shape ``(k_d, 2)`` and
///     each row is a ``[birth, death]`` pair. ``death == inf`` marks essential
///     features.
///
/// Examples
/// --------
/// >>> import numpy as np, tda
/// >>> X = np.random.rand(100, 3).astype(np.float32)
/// >>> bars = tda.persistent_homology(X, max_dim=2)
/// >>> bars[1].shape  # H1 intervals
/// (k, 2)
#[pyfunction(name = "persistent_homology")]
#[pyo3(signature = (data, max_dim=1, threshold=None, distance_matrix=false))]
pub fn persistent_homology<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    max_dim: usize,
    threshold: Option<f32>,
    distance_matrix: bool,
) -> PyResult<Bound<'py, PyList>> {
    let view = data.as_array();
    let n = view.nrows();
    let cols = view.ncols();

    // The only Python-layer-specific check: square shape for matrix mode.
    if distance_matrix && n != cols {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "distance matrix must be square, got shape ({}, {})",
            n, cols
        )));
    }

    let owned_storage: Vec<f32>;
    let slice: &[f32] = match view.as_slice() {
        Some(s) => s,
        None => {
            owned_storage = view.iter().copied().collect();
            &owned_storage
        }
    };

    let result = py
        .detach(|| {
            if distance_matrix {
                tda::persistent_homology_from_distances(slice, n, max_dim, threshold)
            } else {
                tda::persistent_homology(slice, n, cols, max_dim, threshold)
            }
        })
        .map_err(error::into_py)?;

    let arrays: Vec<Bound<'py, PyArray2<f32>>> = result
        .into_flat_intervals()
        .into_iter()
        .map(|(rows, flat)| {
            PyArray1::from_vec(py, flat)
                .reshape([rows, 2])
                .expect("(rows, 2) reshape on a vec of len rows*2 is infallible")
        })
        .collect();

    PyList::new(py, arrays)
}

#[pymodule(gil_used = false)]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(persistent_homology, m)?)?;
    Ok(())
}
