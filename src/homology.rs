use crate::vecops;
use hashbrown::hash_map::Entry;
use hashbrown::HashMap;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum SharedStatus {
    Free,                         //if a face is free
    IsShared { ids: Vec<usize> }, //if a face is shared
    Partial { id: usize },        //if it comes from at least one n+1
    Robust,                       //if has no free face
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Simplex {
    status: SharedStatus,
    index: usize,
}

impl Simplex {
    pub fn new(index: usize) -> Self {
        Simplex {
            status: SharedStatus::Free,
            index,
        }
    }
}

fn discombobulate_faces(
    dimension_n1: &mut HashMap<Vec<u64>, Simplex>,
    dimension_n: &mut HashMap<Vec<u64>, Simplex>,
) -> (Vec<Vec<usize>>, usize) {
    let mut index_map: Vec<Vec<usize>> = vec![vec![]; dimension_n1.len()];
    for (simplex, meta) in dimension_n1.iter() {
        let mut indices: Vec<usize> = Vec::with_capacity(simplex.len() + 1);
        for face in vecops::faces_of(simplex) {
            let new_idx = dimension_n.len();
            match dimension_n.entry(face) {
                Entry::Occupied(entry) => {
                    indices.push(entry.get().index);
                }
                Entry::Vacant(entry) => {
                    // Key doesn't exist, insert the value
                    entry.insert(Simplex::new(new_idx));
                    indices.push(new_idx);
                }
            }
        }
        index_map[meta.index] = indices;
    }
    (index_map, dimension_n.len())
}

pub fn boundary_matrix(index_map: Vec<Vec<usize>>, n_cols: usize) -> Vec<u64> {
    let n_chunks = (n_cols + 63) / 64;
    let mut matrix = Vec::with_capacity(index_map.len() * n_chunks);

    index_map.into_iter().for_each(|indices| {
        let bit_vec = vecops::bitvec_from_slice(&indices, n_chunks);
        matrix.extend(bit_vec);
    });

    matrix
}
