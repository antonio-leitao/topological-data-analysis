use pyo3::prelude::*;
mod clique;
mod homology;
mod linalg;
mod simplex;
mod vecops;

// // ROOT
// #[pymodule]
// fn tda(py: Python, m: &PyModule) -> PyResult<()> {
//     let child_module = PyModule::new(py, "tda.homology")?;
//     homology(py, child_module)?;
//     m.add("submodule", child_module)?;
//     py.import("sys")?
//         .getattr("modules")?
//         .set_item("tda.homology", child_module)?;
//     Ok(())
// }
//
// // HOMOLOGY MODULE
// #[pymodule]
// fn homology(_py: Python, m: &PyModule) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(betti_numbers, m)?)?;
//     m.add_function(wrap_pyfunction!(enumerate_all_cliques, m)?)?;
//     Ok(())
// }

#[pymodule]
fn tda(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(betti, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_all_cliques, m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (data, adjacency_matrix=true, maxdim=-1))]
fn betti(data: Vec<Vec<usize>>, adjacency_matrix: bool, maxdim: i32) -> PyResult<Vec<usize>> {
    let betti = match adjacency_matrix {
        true => clique::betti(data),
        false => clique::betti(data),
    };
    // let elapsed = clique::enumerate_cliques(data);
    Ok(betti)
}

#[pyfunction]
fn enumerate_all_cliques(adjacency_matrix: Vec<Vec<usize>>) -> PyResult<Vec<Vec<usize>>> {
    Ok(clique::enumerate_cliques_list(adjacency_matrix))
}
