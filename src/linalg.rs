use rayon::prelude::*;

fn xor_slices(slice1: &mut [u64], slice2: &[u64]) {
    for (a, b) in slice1.iter_mut().zip(slice2.iter()) {
        *a ^= *b;
    }
}

fn swap_chunks<T>(slice: &mut [T], n_chunks: usize, i: usize, j: usize) {
    if i == j {
        return;
    }
    let start_i = i * n_chunks;
    let start_j = j * n_chunks;
    let (left, right) = slice.split_at_mut(start_j);
    left[start_i..start_i + n_chunks].swap_with_slice(&mut right[0..n_chunks]);
}

fn chunk_contains(slice: &[u64], index: usize) -> bool {
    let chunk_position = index / 64;
    let bit_position = index % 64;
    if chunk_position < slice.len() {
        // Check if the bit is set
        (slice[chunk_position] & (1 << (63 - bit_position))) != 0
    } else {
        // Out-of-bounds index is considered not contained
        false
    }
}

//pivot goes on bits
pub fn gaussian_elimination(matrix: &mut [u64], n_cols: usize) {
    let max_rank = (n_cols).min(matrix.len());
    let row_size = (n_cols + 64) / 64;
    for pivot_idx in 0..max_rank {
        let mut last_row_idx = pivot_idx;

        for (row_idx, row) in matrix.chunks_exact(row_size).enumerate().skip(pivot_idx) {
            if chunk_contains(row, pivot_idx) {
                last_row_idx = row_idx;
                break;
            }
        }
        swap_chunks(matrix, row_size, pivot_idx, last_row_idx);
        let (upper, lower) = matrix.split_at_mut(pivot_idx * row_size + row_size);
        let pivot = &upper[pivot_idx * row_size..];
        lower.par_chunks_exact_mut(row_size).for_each(|row| {
            if chunk_contains(row, pivot_idx) {
                xor_slices(row, pivot)
            }
        });
    }
}

fn transposed_index(num_rows: usize, num_cols: usize, index: usize) -> usize {
    let row_index = index / num_cols;
    let col_index = index % num_cols;
    col_index * num_rows + row_index
}

fn transpose(matrix: &mut [usize], num_rows: usize, num_cols: usize) {
    for i in 0..matrix.len() {
        let j = transposed_index(num_rows, num_cols, i);
        matrix.swap(i, j);
    }
}
