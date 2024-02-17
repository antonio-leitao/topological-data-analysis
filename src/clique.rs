use crate::homology;
use crate::linalg;
use crate::vecops;
use hashbrown::HashMap;
use human_repr::HumanDuration;
use rayon::prelude::*;
use std::time::{Duration, Instant};

macro_rules! timeit {
    ($name:expr, $func:expr) => {{
        let start = Instant::now();
        let result = $func;
        let duration = start.elapsed();
        println!("{}: {:?}", $name, fmt(duration));
        result
    }};
}

fn fmt(duration: Duration) -> String {
    duration.human_duration().to_string()
}

fn get_indices_smaller_than(vec: &[usize], clique_size: usize) -> Vec<usize> {
    vec.iter()
        .enumerate()
        .filter(|(_, &value)| value < clique_size)
        .map(|(index, _)| index)
        .collect()
}

fn next_cliques(
    matrix: &mut [u64],
    degrees: &[usize],
    cliques: &[u64],
    chunk_size: usize,
) -> Vec<u64> {
    let clique_size = match cliques.chunks_exact(chunk_size).nth(0) {
        Some(clique) => vecops::n_elements(clique),
        None => panic!("Error no seed cliques"), //TODO: Return Error
    };

    // clear indices where matrix has small degree
    let indices = get_indices_smaller_than(degrees, clique_size);
    matrix
        .par_chunks_exact_mut(chunk_size)
        .for_each(|row| vecops::clear_bits(row, &indices));

    let new_cliques: Vec<Vec<u64>> = cliques
        .par_chunks_exact(chunk_size)
        .flat_map(|clique| {
            let vertii = vecops::indexes(clique);
            //get common neighbours of cliques
            let common_neighbours =
                vecops::indexes(&vecops::batch_intersect(&matrix, &vertii, chunk_size));
            let mut new_cliques_chunk = Vec::with_capacity(common_neighbours.len());
            match vertii.last() {
                Some(vertex) => {
                    // Perform binary search to find the index of the first element greater than the value
                    let index = match common_neighbours.binary_search(vertex) {
                        Ok(index) => index + 1, // If the value is found, move to the next index
                        Err(index) => index, // If the value is not found, use the insertion point index
                    };

                    for neighbour in common_neighbours.into_iter().skip(index) {
                        if vecops::contains_all(
                            &matrix[neighbour * chunk_size..neighbour * chunk_size + chunk_size],
                            clique,
                        ) {
                            let mut new_clique = clique.to_vec();
                            vecops::insert(&mut new_clique, neighbour);
                            new_cliques_chunk.push(new_clique);
                        }
                    }
                }
                None => (),
            }
            new_cliques_chunk
        })
        .collect();

    new_cliques.into_iter().flatten().collect()
}

//// HOMOLOGY
fn make_boundary_map(
    dimension_n1: &[u64],
    dimension_n: &[u64],
    chunk_size: usize,
) -> (Vec<Vec<usize>>, usize) {
    let mut index_map: Vec<Vec<usize>> = vec![vec![]; dimension_n1.len()];
    let mut dimension_n_map = HashMap::with_capacity(dimension_n.len());
    for (index, chunk) in dimension_n.chunks_exact(chunk_size).enumerate() {
        dimension_n_map.insert(chunk.to_vec(), index);
    }
    for (index, simplex) in dimension_n1.chunks_exact(chunk_size).enumerate() {
        let mut indices: Vec<usize> = Vec::with_capacity(simplex.len() + 1);
        for face in vecops::faces_of(simplex) {
            let new_idx = dimension_n_map.len();
            let idx = dimension_n_map.entry(face).or_insert(new_idx);
            indices.push(*idx);
        }
        index_map[index] = indices;
    }
    (index_map, dimension_n.len())
}

pub fn betti(adjacency_matrix: Vec<Vec<usize>>) -> Vec<usize> {
    //read data
    let (mut matrix, chunk_size, mut cliques, degrees) = read_adjacency_matrix(adjacency_matrix);
    let mut betti_numbers: Vec<usize> = Vec::new();
    let mut prev_rank: usize = 0;
    //add maxdim?
    loop {
        //get them cliques
        let new_cliques = timeit!(
            "next cliques:",
            next_cliques(&mut matrix, &degrees, &cliques, chunk_size)
        );
        println!("{}", new_cliques.len() / chunk_size);
        if new_cliques.is_empty() {
            break;
        }
        //homology loop
        let (boundary_map, n_cols) = timeit!(
            "boundary_map:",
            make_boundary_map(&new_cliques, &cliques, chunk_size)
        );
        let mut boundary_matrix = timeit!(
            "boundary_matrix:",
            homology::boundary_matrix(boundary_map, n_cols)
        );
        timeit!(
            "gaussian elemination:",
            linalg::gaussian_elimination(&mut boundary_matrix, n_cols)
        );
        let rank = linalg::rank(&boundary_matrix, n_cols);
        betti_numbers.push(new_cliques.len() - prev_rank - rank);
        //reset
        prev_rank = rank;
        cliques = new_cliques; // Reassign the updated value to the outer variable
    }
    betti_numbers
}

/// USER API

// REVISE THIS!!!!!!
fn read_adjacency_matrix(
    adjacency_matrix: Vec<Vec<usize>>,
) -> (Vec<u64>, usize, Vec<u64>, Vec<usize>) {
    let n_vertices = adjacency_matrix.len();
    let chunk_size = (n_vertices + 64) / 64;
    let mut cliques: Vec<u64> = Vec::with_capacity(n_vertices * chunk_size);
    let mut matrix: Vec<u64> = Vec::with_capacity(n_vertices * chunk_size);
    let mut degrees: Vec<usize> = Vec::with_capacity(n_vertices * chunk_size);
    for (i, row) in adjacency_matrix.into_iter().enumerate() {
        cliques.extend(vecops::bitvec_from_slice(&vec![i], chunk_size));
        matrix.extend(vecops::bitvec_from_slice(&row, chunk_size));
        degrees.push(row.len());
    }
    (matrix, chunk_size, cliques, degrees)
}

/// TESTING
pub fn enumerate_cliques(adjacency_matrix: Vec<Vec<usize>>) -> String {
    let (mut matrix, chunk_size, mut cliques, degrees) = read_adjacency_matrix(adjacency_matrix);

    println!("{}", degrees.len());
    let instant = Instant::now();
    loop {
        let new_cliques = next_cliques(&mut matrix, &degrees, &cliques, chunk_size);
        if new_cliques.is_empty() {
            break;
        }
        cliques = new_cliques; // Reassign the updated value to the outer variable
    }
    fmt(instant.elapsed())
}

//DO NOT USE, ONLY FOR TESTING
pub fn enumerate_cliques_list(adjacency_matrix: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let (mut matrix, chunk_size, mut cliques, mut degrees) =
        read_adjacency_matrix(adjacency_matrix);
    let mut all_cliques: Vec<Vec<usize>> = Vec::new();
    all_cliques.extend(
        cliques
            .chunks_exact(chunk_size)
            .map(|clique| vecops::indexes(clique)),
    );
    loop {
        let new_cliques = next_cliques(&mut matrix, &degrees, &cliques, chunk_size);
        if new_cliques.is_empty() {
            break;
        }
        cliques = new_cliques; // Reassign the updated value to the outer variable
        all_cliques.extend(
            cliques
                .chunks_exact(chunk_size)
                .map(|clique| vecops::indexes(clique)),
        );
    }
    all_cliques
}
