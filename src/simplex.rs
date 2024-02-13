use crate::homology::Simplex;
use crate::vecops;
use hashbrown::hash_map::Entry;
use hashbrown::HashMap;

#[derive(Debug)]
struct Complex {
    simplices: Vec<HashMap<Vec<u64>, Simplex>>,
}

fn find_dimension_vertices(vectors: &[Vec<usize>]) -> (usize, usize) {
    let mut max_size = 0;
    let mut max_entry = 0;

    for inner_vec in vectors {
        max_size = max_size.max(inner_vec.len());
        if let Some(&max) = inner_vec.iter().max() {
            max_entry = max_entry.max(max);
        }
    }
    (max_size, max_entry)
}

fn read_complex(maximal_simplices: Vec<Vec<usize>>) -> Complex {
    let (max_dimension, n_vertices) = find_dimension_vertices(&maximal_simplices);
    let n_chunks = (n_vertices + 64) / 64;
    let mut simplices: Vec<HashMap<Vec<u64>, Simplex>> = vec![HashMap::new(); max_dimension];
    for simplex in maximal_simplices.into_iter() {
        if simplex.len() == 0 {
            continue;
        }
        let dimension = simplex.len() - 1;
        let bitvec = vecops::bitvec_from_slice(&simplex, n_chunks);
        let new_index = simplices[dimension].len().clone();
        match simplices[dimension].entry(bitvec) {
            // do nothing if it exists
            Entry::Occupied(_) => continue,
            // else insert with new index
            Entry::Vacant(v) => {
                v.insert(Simplex::new(new_index));
            }
        }
    }
    Complex { simplices }
}
