// ═══════════════════════════════════════════════════════════════════════════════
// bitcsr.rs — tail-truncated bitset sparse distance matrix (the data structure)
// ═══════════════════════════════════════════════════════════════════════════════
//
// `BitCsrDistanceMatrix` stores one tail-truncated adjacency bitset per vertex,
// flattened into `words`. Distances are stored per word in descending bit order,
// so a neighbour id is implicit in the bit position and its distance is found by
// one popcount rank inside the word. The hot cofacet operation is therefore a
// k-way AND over aligned `u64` blocks followed by set-bit iteration.
//
// This file owns ONLY the representation and its primitive bit/neighbour
// operations. It knows nothing about simplices, cofacets, apparent pairs, or
// reduction columns — those live in the Ripser algorithm layer (`algorithm.rs`),
// which drives this structure through the `pub(crate)` primitives below.

use crate::engine::distance::DistanceMatrix;

/// Entries `>= NO_EDGE` are "no edge" and are excluded from the enclosing-radius
/// reduction, mirroring `distance.rs`.
const NO_EDGE: f32 = f32::MAX;

/// Sparse distance matrix as flattened tail-truncated bitsets plus per-bit
/// distances.
///
/// Row `v` occupies `words[word_ptr[v]..word_ptr[v + 1]]`; word index `b` in
/// that row represents neighbour ids `64*b..64*b+63`. Rows are truncated after
/// their highest nonzero word. `val_ptr` has one entry per flattened word plus a
/// sentinel; distances for set bits in word `wi` live in
/// `val[val_ptr[wi]..val_ptr[wi + 1]]`, ordered from high bit to low bit.
pub struct BitCsrDistanceMatrix {
    n: usize,
    threshold: f32,
    word_ptr: Vec<u32>,
    words: Vec<u64>,
    val_ptr: Vec<u32>,
    val: Vec<f32>,
}

impl BitCsrDistanceMatrix {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Pack descending-sorted CSR rows into the bitset representation.
    ///
    /// `col[row_ptr[v]..row_ptr[v + 1]]` are vertex `v`'s neighbours in
    /// DESCENDING id order, with parallel distances in `val`. `threshold` is the
    /// effective cofacet threshold to store on the matrix.
    fn from_sorted_rows(
        n: usize,
        row_ptr: &[usize],
        col: &[u16],
        val: &[f32],
        threshold: f32,
    ) -> Self {
        let nnz = col.len();
        let mut word_ptr = Vec::with_capacity(n + 1);
        let mut words = Vec::new();
        let mut val_ptr = Vec::new();
        let mut val_out = Vec::with_capacity(nnz);

        let mut counts: Vec<usize> = Vec::new();
        let mut offsets: Vec<usize> = Vec::new();
        let mut cursor: Vec<usize> = Vec::new();
        let mut row_vals: Vec<f32> = Vec::new();

        word_ptr.push(0);
        val_ptr.push(0);

        for v in 0..n {
            let s = row_ptr[v];
            let e = row_ptr[v + 1];
            let col_row = &col[s..e];
            let val_row = &val[s..e];
            if col_row.is_empty() {
                word_ptr.push(to_u32(words.len(), "bitcsr word count"));
                continue;
            }

            let word_len = ((col_row[0] as usize) >> 6) + 1;
            let row_start = words.len();
            words.resize(row_start + word_len, 0);

            counts.clear();
            counts.resize(word_len, 0);

            for &u in col_row {
                let block = (u as usize) >> 6;
                words[row_start + block] |= 1u64 << (u & 63);
                counts[block] += 1;
            }

            offsets.clear();
            offsets.resize(word_len + 1, 0);
            for block in 0..word_len {
                offsets[block + 1] = offsets[block] + counts[block];
            }

            cursor.clear();
            cursor.resize(word_len, 0);
            row_vals.clear();
            row_vals.resize(col_row.len(), 0.0);

            for (&u, &d) in col_row.iter().zip(val_row.iter()) {
                let block = (u as usize) >> 6;
                let dst = offsets[block] + cursor[block];
                row_vals[dst] = d;
                cursor[block] += 1;
            }

            for block in 0..word_len {
                val_out.extend_from_slice(&row_vals[offsets[block]..offsets[block + 1]]);
                val_ptr.push(to_u32(val_out.len(), "bitcsr value count"));
            }
            word_ptr.push(to_u32(words.len(), "bitcsr word count"));
        }

        debug_assert_eq!(word_ptr.len(), n + 1);
        debug_assert_eq!(val_ptr.len(), words.len() + 1);
        debug_assert_eq!(val_out.len(), nnz);

        Self {
            n,
            threshold,
            word_ptr,
            words,
            val_ptr,
            val: val_out,
        }
    }

    /// Build directly from raw CSR parts produced by the point-cloud preprocessor
    /// (`pdist_csr`). Rows must be descending by neighbour id; `threshold` is the
    /// effective threshold (`user_threshold.min(r_cheb)`).
    pub(crate) fn from_csr_parts(
        n: usize,
        row_ptr: Vec<usize>,
        col: Vec<u16>,
        val: Vec<f32>,
        threshold: f32,
    ) -> Self {
        debug_assert_eq!(row_ptr.len(), n + 1);
        debug_assert_eq!(col.len(), val.len());
        debug_assert_eq!(*row_ptr.last().unwrap_or(&0), col.len());
        debug_assert!((0..n).all(|v| col[row_ptr[v]..row_ptr[v + 1]]
            .windows(2)
            .all(|w| w[0] > w[1])));

        Self::from_sorted_rows(n, &row_ptr, &col, &val, threshold)
    }

    /// Build from a dense lower-triangular distance matrix, keeping only edges
    /// within the effective threshold. Returns `(matrix, r_cheb)` where `r_cheb`
    /// is the Chebyshev (minimax) radius; the matrix stores
    /// `threshold.min(r_cheb)` as its effective threshold. Supports
    /// distance-matrix inputs and tests.
    pub(crate) fn from_distance_matrix(dist: &DistanceMatrix, threshold: f32) -> (Self, f32) {
        let n = dist.n();
        if n < 2 {
            let row_ptr = vec![0usize; n + 1];
            let bitcsr = Self::from_sorted_rows(n, &row_ptr, &[], &[], threshold);
            return (bitcsr, f32::INFINITY);
        }

        let raw = dist.raw();

        let mut ecc = vec![f32::NEG_INFINITY; n];
        {
            let mut p = 0;
            for i in 1..n {
                let mut row_max = ecc[i];
                for j in 0..i {
                    let d = raw[p];
                    p += 1;
                    if d < NO_EDGE {
                        if d > row_max {
                            row_max = d;
                        }
                        if d > ecc[j] {
                            ecc[j] = d;
                        }
                    }
                }
                ecc[i] = row_max;
            }
        }
        let r_cheb = ecc
            .iter()
            .copied()
            .filter(|&m| m > f32::NEG_INFINITY)
            .fold(f32::INFINITY, f32::min);
        let eff = threshold.min(r_cheb);

        let mut row_ptr = vec![0usize; n + 1];
        {
            let mut p = 0;
            for i in 1..n {
                for j in 0..i {
                    let d = raw[p];
                    p += 1;
                    if d <= eff && d < NO_EDGE {
                        row_ptr[i + 1] += 1;
                        row_ptr[j + 1] += 1;
                    }
                }
            }
            for v in 0..n {
                row_ptr[v + 1] += row_ptr[v];
            }
        }

        let nnz = row_ptr[n];
        let mut col = vec![0u16; nnz];
        let mut val = vec![0.0f32; nnz];

        {
            let mut cur = row_ptr[..n].to_vec();
            let mut p = 0;
            for i in 1..n {
                for j in 0..i {
                    let d = raw[p];
                    p += 1;
                    if d <= eff && d < NO_EDGE {
                        let ci = cur[i];
                        col[ci] = j as u16;
                        val[ci] = d;
                        cur[i] = ci + 1;

                        let cj = cur[j];
                        col[cj] = i as u16;
                        val[cj] = d;
                        cur[j] = cj + 1;
                    }
                }
            }
        }

        // BitCSR needs neighbours descending by id.
        for v in 0..n {
            let s = row_ptr[v];
            let e = row_ptr[v + 1];
            col[s..e].reverse();
            val[s..e].reverse();
        }

        (Self::from_sorted_rows(n, &row_ptr, &col, &val, eff), r_cheb)
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    #[inline(always)]
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    /// Effective cofacet threshold baked into the matrix at construction.
    #[inline(always)]
    pub(crate) fn threshold(&self) -> f32 {
        self.threshold
    }

    #[inline(always)]
    fn row_start(&self, v: usize) -> usize {
        unsafe { *self.word_ptr.get_unchecked(v) as usize }
    }

    #[inline(always)]
    fn row_end(&self, v: usize) -> usize {
        unsafe { *self.word_ptr.get_unchecked(v + 1) as usize }
    }

    #[inline(always)]
    fn row_len(&self, v: usize) -> usize {
        self.row_end(v) - self.row_start(v)
    }

    #[inline(always)]
    fn word_distance_unchecked(&self, flat_word: usize, word: u64, bit: usize) -> f32 {
        debug_assert!((word & (1u64 << bit)) != 0);
        let higher = bits_above(bit);
        let rank = (word & higher).count_ones() as usize;
        let start = unsafe { *self.val_ptr.get_unchecked(flat_word) as usize };
        unsafe { *self.val.get_unchecked(start + rank) }
    }

    /// Distance between an existing edge `(a, b)`. Debug-asserts the edge exists.
    #[inline(always)]
    pub(crate) fn edge_dist(&self, a: u16, b: u16) -> f32 {
        let block = (b as usize) >> 6;
        let bit = (b as usize) & 63;
        let row_start = self.row_start(a as usize);
        debug_assert!(
            block < self.row_len(a as usize),
            "edge_dist on a missing edge ({a}, {b})"
        );
        let flat = row_start + block;
        let word = unsafe { *self.words.get_unchecked(flat) };
        debug_assert!(
            (word & (1u64 << bit)) != 0,
            "edge_dist on a missing edge ({a}, {b})"
        );
        self.word_distance_unchecked(flat, word, bit)
    }

    /// Encoded (raw u32 bits) distance of edge `(a, b)`.
    #[inline(always)]
    pub(crate) fn edge_dist_bits(&self, a: u16, b: u16) -> u32 {
        self.edge_dist(a, b).to_bits()
    }

    // ── Neighbour enumeration primitives ──────────────────────────────────────

    /// Enumerate the common neighbours of `verts[0..vc]` whose id exceeds `floor`
    /// (pass `-1` for no floor), youngest-first (descending id). For each, `f`
    /// receives the neighbour id and `max_extra` — the largest distance from that
    /// neighbour to any vertex in the set. Return `false` from `f` to stop.
    #[inline]
    pub(crate) fn for_each_common_neighbor(
        &self,
        verts: &[u16; 6],
        vc: usize,
        floor: i32,
        mut f: impl FnMut(u16, f32) -> bool,
    ) {
        if vc == 0 {
            return;
        }

        let mut row_start = [0usize; 6];
        let mut limit = usize::MAX;
        for i in 0..vc {
            let v = verts[i] as usize;
            row_start[i] = self.row_start(v);
            let len = self.row_len(v);
            if len == 0 {
                return;
            }
            if len < limit {
                limit = len;
            }
        }

        let floor_block = if floor >= 0 {
            let b = (floor as usize) >> 6;
            if b >= limit {
                return;
            }
            b
        } else {
            0
        };
        let floor_bit = (floor as usize) & 63;

        for block in (floor_block..limit).rev() {
            let mut mask = self.intersection_word(&row_start, vc, block);
            if floor >= 0 && block == floor_block {
                mask &= bits_above(floor_bit);
            }

            while mask != 0 {
                let bit = highest_set_bit(mask);
                mask &= !(1u64 << bit);
                let w = ((block << 6) | bit) as u16;
                let extra = self.max_extra(&row_start, vc, block, bit);
                if !f(w, extra) {
                    return;
                }
            }
        }
    }

    /// Specialised common-neighbour enumeration for an edge `(v0, v1)` that
    /// yields BOTH edge distances `(w, d(w,v0), d(w,v1))`, youngest-first above
    /// `floor`. Return `false` from `f` to stop.
    #[inline]
    pub(crate) fn for_each_common_neighbor_edges(
        &self,
        v0: u16,
        v1: u16,
        floor: i32,
        mut f: impl FnMut(u16, f32, f32) -> bool,
    ) {
        let s0 = self.row_start(v0 as usize);
        let s1 = self.row_start(v1 as usize);
        let limit = self.row_len(v0 as usize).min(self.row_len(v1 as usize));
        if limit == 0 {
            return;
        }

        let floor_block = if floor >= 0 {
            let b = (floor as usize) >> 6;
            if b >= limit {
                return;
            }
            b
        } else {
            0
        };
        let floor_bit = (floor as usize) & 63;

        for block in (floor_block..limit).rev() {
            let f0 = s0 + block;
            let f1 = s1 + block;
            let word0 = unsafe { *self.words.get_unchecked(f0) };
            let word1 = unsafe { *self.words.get_unchecked(f1) };
            let mut mask = word0 & word1;
            if floor >= 0 && block == floor_block {
                mask &= bits_above(floor_bit);
            }

            while mask != 0 {
                let bit = highest_set_bit(mask);
                mask &= !(1u64 << bit);
                let w = ((block << 6) | bit) as u16;
                let d0 = self.word_distance_unchecked(f0, word0, bit);
                let d1 = self.word_distance_unchecked(f1, word1, bit);
                if !f(w, d0, d1) {
                    return;
                }
            }
        }
    }

    /// Enumerate vertex `v`'s neighbours with id `> v`, youngest-first, each with
    /// its edge distance. This is the per-row primitive H0 edge enumeration is
    /// built from. Return `false` from `f` to stop.
    #[inline]
    pub(crate) fn for_each_neighbor_above(&self, v: usize, mut f: impl FnMut(u16, f32) -> bool) {
        let row_start = self.row_start(v);
        let row_len = self.row_len(v);
        if row_len == 0 {
            return;
        }
        let floor_block = v >> 6;
        if floor_block >= row_len {
            return;
        }
        for block in (floor_block..row_len).rev() {
            let flat = row_start + block;
            let word = unsafe { *self.words.get_unchecked(flat) };
            let mut iter_word = word;
            if block == floor_block {
                iter_word &= bits_above(v & 63);
            }
            while iter_word != 0 {
                let bit = highest_set_bit(iter_word);
                iter_word &= !(1u64 << bit);
                let w = ((block << 6) | bit) as u16;
                let d = self.word_distance_unchecked(flat, word, bit);
                if !f(w, d) {
                    return;
                }
            }
        }
    }

    #[inline(always)]
    fn intersection_word(&self, row_start: &[usize; 6], vc: usize, block: usize) -> u64 {
        let words = &self.words;
        unsafe {
            let w0 = *words.get_unchecked(row_start[0] + block);
            if vc == 1 {
                return w0;
            }
            let w1 = *words.get_unchecked(row_start[1] + block);
            let mut acc = w0 & w1;
            if acc == 0 || vc == 2 {
                return acc;
            }

            let w2 = *words.get_unchecked(row_start[2] + block);
            if w2 == 0 {
                return 0;
            }
            acc &= w2;
            if acc == 0 || vc == 3 {
                return acc;
            }

            let w3 = *words.get_unchecked(row_start[3] + block);
            if w3 == 0 {
                return 0;
            }
            acc &= w3;
            if acc == 0 || vc == 4 {
                return acc;
            }

            let w4 = *words.get_unchecked(row_start[4] + block);
            if w4 == 0 {
                return 0;
            }
            acc &= w4;
            if acc == 0 || vc == 5 {
                return acc;
            }

            let w5 = *words.get_unchecked(row_start[5] + block);
            if w5 == 0 {
                return 0;
            }
            acc & w5
        }
    }

    #[inline(always)]
    fn max_extra(&self, row_start: &[usize; 6], vc: usize, block: usize, bit: usize) -> f32 {
        unsafe {
            match vc {
                1 => {
                    let flat = row_start[0] + block;
                    self.word_distance_unchecked(flat, *self.words.get_unchecked(flat), bit)
                }
                2 => {
                    let f0 = row_start[0] + block;
                    let f1 = row_start[1] + block;
                    self.word_distance_unchecked(f0, *self.words.get_unchecked(f0), bit)
                        .max(self.word_distance_unchecked(f1, *self.words.get_unchecked(f1), bit))
                }
                3 => {
                    let f0 = row_start[0] + block;
                    let f1 = row_start[1] + block;
                    let f2 = row_start[2] + block;
                    self.word_distance_unchecked(f0, *self.words.get_unchecked(f0), bit)
                        .max(self.word_distance_unchecked(f1, *self.words.get_unchecked(f1), bit))
                        .max(self.word_distance_unchecked(f2, *self.words.get_unchecked(f2), bit))
                }
                _ => {
                    let mut extra = 0.0f32;
                    for &start in row_start.iter().take(vc) {
                        let flat = start + block;
                        let d = self.word_distance_unchecked(
                            flat,
                            *self.words.get_unchecked(flat),
                            bit,
                        );
                        if d > extra {
                            extra = d;
                        }
                    }
                    extra
                }
            }
        }
    }
}

#[inline(always)]
fn to_u32(value: usize, what: &str) -> u32 {
    assert!(value <= u32::MAX as usize, "{what} exceeds u32::MAX");
    value as u32
}

#[inline(always)]
pub(crate) fn highest_set_bit(word: u64) -> usize {
    debug_assert!(word != 0);
    63 - word.leading_zeros() as usize
}

#[inline(always)]
pub(crate) fn bits_above(bit: usize) -> u64 {
    if bit >= 63 {
        0
    } else {
        u64::MAX << (bit + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::algorithm::{compute, for_each_edge};
    use crate::types::BarcodeResult;

    fn barcodes_equal(a: &BarcodeResult, b: &BarcodeResult) -> bool {
        if a.intervals.len() != b.intervals.len() {
            return false;
        }
        let key = |p: &crate::types::PersistenceInterval| {
            let f = |x: f32| if x.is_finite() { x } else { 1.0e30 };
            (f(p.birth), f(p.death))
        };
        for (da, db) in a.intervals.iter().zip(b.intervals.iter()) {
            if da.len() != db.len() {
                return false;
            }
            let mut va: Vec<_> = da.iter().map(key).collect();
            let mut vb: Vec<_> = db.iter().map(key).collect();
            va.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
            vb.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
            for (x, y) in va.iter().zip(vb.iter()) {
                if (x.0 - y.0).abs() > 1e-5 || (x.1 - y.1).abs() > 1e-5 {
                    return false;
                }
            }
        }
        true
    }

    /// The sequential and parallel assembly paths must produce bit-identical
    /// barcodes on the same input.
    fn seq_par_check(dist: &DistanceMatrix, threshold: f32, max_dim: usize) {
        let (bitcsr, r_cheb) = BitCsrDistanceMatrix::from_distance_matrix(dist, threshold);
        let eff = threshold.min(r_cheb);
        let seq = compute(&bitcsr, eff, max_dim, false);
        let par = compute(&bitcsr, eff, max_dim, true);
        assert!(
            barcodes_equal(&seq, &par),
            "seq vs par barcode mismatch (n={}, eff={eff}, max_dim={max_dim})\n seq={:?}\n par={:?}",
            dist.n(),
            seq.intervals,
            par.intervals,
        );
    }

    fn xorshift(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    #[test]
    fn bitcsr_for_each_edge_skips_lower_tail_blocks_above_64() {
        let n = 70usize;
        let mut data = vec![f32::MAX; n * (n - 1) / 2];
        data[65 * 64 / 2 + 50] = 1.0;

        let dist = DistanceMatrix::from_lower_triangular(n, data);
        let (bitcsr, r_cheb) = BitCsrDistanceMatrix::from_distance_matrix(&dist, 1.0);
        let eff = 1.0f32.min(r_cheb);

        let mut edges = Vec::new();
        for_each_edge(&bitcsr, eff, |edge| {
            edges.push(edge.vertices());
            true
        });

        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0][0], 65);
        assert_eq!(edges[0][1], 50);
    }

    #[test]
    fn bitcsr_seq_matches_par_no_ties() {
        for &(n, seed) in &[(5usize, 1u64), (8, 7), (12, 99), (20, 31337)] {
            let mut rng = xorshift(seed);
            let m = n * (n - 1) / 2;
            let data: Vec<f32> = (0..m)
                .map(|_| (rng() >> 40) as f32 / 16_777_216.0)
                .collect();
            let dist = DistanceMatrix::from_lower_triangular(n, data);
            for &thr in &[0.3f32, 0.6, 0.9, f32::INFINITY] {
                for max_dim in 1..=3 {
                    seq_par_check(&dist, thr, max_dim);
                }
            }
        }
    }

    #[test]
    fn bitcsr_seq_matches_par_with_ties() {
        for &(n, levels, seed) in &[(6usize, 2u64, 5), (8, 3, 17), (12, 3, 2024)] {
            let mut rng = xorshift(seed);
            let m = n * (n - 1) / 2;
            let data: Vec<f32> = (0..m)
                .map(|_| ((rng() % levels) as f32 + 1.0) * 0.5)
                .collect();
            let dist = DistanceMatrix::from_lower_triangular(n, data);
            for &thr in &[0.5f32, 1.0, 1.5, f32::INFINITY] {
                for max_dim in 1..=3 {
                    seq_par_check(&dist, thr, max_dim);
                }
            }
        }
    }

    /// Exercise the rayon path (n large enough to cross
    /// `PARALLEL_ASSEMBLE_THRESHOLD`) and assert it equals the sequential sparse
    /// path, with and without diameter ties.
    #[test]
    fn bitcsr_parallel_matches_sequential() {
        for &(n, seed, levels) in &[(80usize, 12345u64, 0u64), (80, 999, 6)] {
            let mut rng = xorshift(seed);
            let m = n * (n - 1) / 2;
            let data: Vec<f32> = (0..m)
                .map(|_| {
                    if levels == 0 {
                        (rng() >> 40) as f32 / 16_777_216.0
                    } else {
                        ((rng() % levels) as f32 + 1.0) * 0.5
                    }
                })
                .collect();
            let dist = DistanceMatrix::from_lower_triangular(n, data);
            for &thr in &[0.4f32, 0.8, f32::INFINITY] {
                // max_dim 3 (finite threshold only, to bound cost) exercises the
                // parallel `build_pool` path that concatenates the next-dimension
                // simplex pool across workers.
                let max_dims: &[usize] = if thr.is_finite() { &[1, 2, 3] } else { &[1, 2] };
                for &max_dim in max_dims {
                    seq_par_check(&dist, thr, max_dim);
                }
            }
        }
    }
}
