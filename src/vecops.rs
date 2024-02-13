//bitvec operations
pub fn batch_intersect(matrix: &[u64], indexes: &[usize], chunk_size: usize) -> Vec<u64> {
    // Calculate adjusted indexes
    let mut acc = matrix[indexes[0] * chunk_size..(indexes[0] + 1) * chunk_size].to_vec();
    for i in indexes.iter().skip(1) {
        //change with is_null
        if is_null(&acc) {
            // If all zeros, no need to continue, return early
            return acc;
        }
        and_slices(&mut acc, &matrix[i * chunk_size..(i + 1) * chunk_size])
    }
    acc
}

pub fn clear_bits(slice: &mut [u64], indices: &[usize]) {
    for &index in indices {
        let block_idx = index / 64;
        let bit_position = index % 64;
        slice[block_idx] &= !(1 << (63 - bit_position));
    }
}

pub fn is_null(slice: &[u64]) -> bool {
    for elem in slice.iter() {
        if *elem != 0 {
            return false;
        }
    }
    true
}

pub fn and_slices(slice1: &mut [u64], slice2: &[u64]) {
    for (a, b) in slice1.iter_mut().zip(slice2.iter()) {
        *a &= *b;
    }
}

pub fn indexes(slice: &[u64]) -> Vec<usize> {
    let mut positions = Vec::new();
    let mut offset = 0;
    for byte in slice.iter() {
        let byte_positions = find_set_bits_positions(*byte);
        positions.extend(byte_positions.iter().map(|pos| pos + offset));
        offset += 64; // Move the offset to the next byte position
    }
    positions
}

fn find_set_bits_positions(number: u64) -> Vec<usize> {
    let mut positions = Vec::new();
    let mut value = number;
    while value != 0 {
        let index = value.leading_zeros();
        positions.push(index as usize);
        value ^= 1 << (63 - index);
    }
    positions
}

pub fn n_elements(slice: &[u64]) -> usize {
    slice.iter().map(|&byte| byte.count_ones() as usize).sum()
}

pub fn contains_all(slice1: &[u64], slice2: &[u64]) -> bool {
    for (a, b) in slice1.iter().zip(slice2.iter()) {
        if *a & *b != *b {
            return false;
        };
    }
    true
}

pub fn insert(slice: &mut [u64], value: usize) -> Vec<u64> {
    let block_idx = value / 64;
    let bit_position = value % 64;
    slice[block_idx] |= 1 << (63 - bit_position);
    slice.to_vec()
}

pub fn bitvec_from_slice(slice: &[usize], n_chunks: usize) -> Vec<u64> {
    let mut vertices = vec![0; n_chunks]; // Calculate number of u64 elements needed
    for value in slice.iter() {
        let block_idx = value / 64;
        let bit_position = value % 64;
        vertices[block_idx] |= 1 << (63 - bit_position);
    }
    vertices
}

pub fn faces_of(input: &[u64]) -> Vec<Vec<u64>> {
    let mut result = Vec::new();

    for i in 0..input.len() {
        let variations = remove_single_set_bit(input[i]);

        for &variation in &variations {
            let mut clone = input.to_vec();
            clone[i] = variation;
            result.push(clone);
        }
    }
    result
}

fn remove_single_set_bit(input: u64) -> Vec<u64> {
    let mut result = Vec::new();
    let mut x = input;

    while x != 0 {
        let bit = x.trailing_zeros();
        let mask = 1 << bit;
        result.push(input & !mask);
        x &= x - 1; // Clear the lowest set bit
    }

    result
}
