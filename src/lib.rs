use pyo3::prelude::*;

// ROOT
#[pymodule]
fn tda(py: Python, m: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "tda.homology")?;
    homology(py, child_module)?;
    m.add("submodule", child_module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("tda.homology", child_module)?;
    Ok(())
}

// HOMOLOGY MODULE
#[pymodule]
fn homology(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(betti_numbers, m)?)?;
    Ok(())
}

#[pyfunction]
fn betti_numbers(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}
