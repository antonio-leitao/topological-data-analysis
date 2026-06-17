// ═══════════════════════════════════════════════════════════════════════════════
// bitcsr.rs — tail-truncated bitset sparse backend
// ═══════════════════════════════════════════════════════════════════════════════
//
// `BitCsrDistanceMatrix` stores one tail-truncated adjacency bitset per vertex,
// flattened into `words`. Distances are stored per word in descending bit order,
// so a neighbour id is implicit in the bit position and its distance is found by
// one popcount rank inside the word. The hot cofacet operation is therefore a
// k-way AND over aligned `u64` blocks followed by set-bit iteration.

use crate::engine::csr::{csr_from_distance_matrix, CsrDistanceMatrix};
use crate::engine::distance::DistanceMatrix;
use crate::engine::filtration::Filtration;
use crate::engine::simplex::{encode_filtration, FxHashMap, Simplex128};

/// Minimum number of input simplices before `assemble_candidates` uses rayon.
/// Below this the per-task overhead and buffer merges outweigh the work, so we
/// stay on the sequential, single-allocation path (this also covers H1 and
/// small datasets even when the caller requests `parallel`).
const PARALLEL_ASSEMBLE_THRESHOLD: usize = 512;

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
    pub(crate) fn from_csr(csr: &CsrDistanceMatrix, threshold: f32) -> Self {
        let n = csr.n();
        let mut word_ptr = Vec::with_capacity(n + 1);
        let mut words = Vec::new();
        let mut val_ptr = Vec::new();
        let mut val = Vec::with_capacity(csr.nnz());

        let mut counts: Vec<usize> = Vec::new();
        let mut offsets: Vec<usize> = Vec::new();
        let mut cursor: Vec<usize> = Vec::new();
        let mut row_vals: Vec<f32> = Vec::new();

        word_ptr.push(0);
        val_ptr.push(0);

        for v in 0..n {
            let (col, row_val) = csr.neighbors(v);
            if col.is_empty() {
                word_ptr.push(to_u32(words.len(), "bitcsr word count"));
                continue;
            }

            let word_len = ((col[0] as usize) >> 6) + 1;
            let row_start = words.len();
            words.resize(row_start + word_len, 0);

            counts.clear();
            counts.resize(word_len, 0);

            for &u in col {
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
            row_vals.resize(col.len(), 0.0);

            for (&u, &d) in col.iter().zip(row_val.iter()) {
                let block = (u as usize) >> 6;
                let dst = offsets[block] + cursor[block];
                row_vals[dst] = d;
                cursor[block] += 1;
            }

            for block in 0..word_len {
                val.extend_from_slice(&row_vals[offsets[block]..offsets[block + 1]]);
                val_ptr.push(to_u32(val.len(), "bitcsr value count"));
            }
            word_ptr.push(to_u32(words.len(), "bitcsr word count"));
        }

        debug_assert_eq!(word_ptr.len(), n + 1);
        debug_assert_eq!(val_ptr.len(), words.len() + 1);
        debug_assert_eq!(val.len(), csr.nnz());

        Self {
            n,
            threshold,
            word_ptr,
            words,
            val_ptr,
            val,
        }
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn n(&self) -> usize {
        self.n
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn nnz(&self) -> usize {
        self.val.len()
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn bit_words(&self) -> usize {
        self.words.len()
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

    #[inline(always)]
    fn edge_dist(&self, a: u16, b: u16) -> f32 {
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

    #[inline(always)]
    fn edge_dist_bits(&self, a: u16, b: u16) -> u32 {
        self.edge_dist(a, b).to_bits()
    }

    #[inline]
    fn for_each_common_neighbor(
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

    #[inline]
    fn for_each_common_neighbor_edges(
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

    #[inline]
    fn cofacets(
        &self,
        sigma: Simplex128,
        all_cofacets: bool,
        threshold: f32,
        mut f: impl FnMut(Simplex128) -> bool,
    ) {
        let vc = sigma.vertex_count();
        if vc == 0 || sigma.filtration() > threshold {
            return;
        }

        let verts = sigma.vertices();
        let base = sigma.filtration();
        let floor: i32 = if all_cofacets {
            -1
        } else {
            sigma.largest_vertex() as i32
        };
        let check_threshold = threshold < self.threshold;

        let payload = sigma.vertex_key();
        let mut vi = 0usize;
        let mut payload_upper: u128 = 0;
        let mut payload_lower_shifted: u128 = payload >> 16;
        let mut insert_shift: u32 = 80;

        self.for_each_common_neighbor(&verts, vc, floor, |w, extra| {
            let old_vi = vi;
            while vi < vc && verts[vi] > w {
                vi += 1;
            }
            if vi != old_vi {
                let split = 96 - (vi as u32) * 16;
                let upper_mask = (!0u128) << split;
                payload_upper = payload & upper_mask;
                payload_lower_shifted = (payload & !upper_mask) >> 16;
                insert_shift = split - 16;
            }

            let diam = if extra > base { extra } else { base };
            if check_threshold && diam > threshold {
                return true;
            }

            let w_ins = (w as u128) + 1;
            let cof = Simplex128(
                ((encode_filtration(diam) as u128) << 96)
                    | payload_upper
                    | (w_ins << insert_shift)
                    | payload_lower_shifted,
            );
            f(cof)
        });
    }

    #[inline]
    fn first_common_neighbor_le(&self, verts: &[u16; 6], vc: usize, max_extra: f32) -> Option<u16> {
        let mut result = None;
        self.for_each_common_neighbor(verts, vc, -1, |w, extra| {
            if extra <= max_extra {
                result = Some(w);
                false
            } else {
                true
            }
        });
        result
    }

    #[inline]
    fn facet_diameter(&self, verts: &[u16; 6], vc: usize, skip: usize) -> f32 {
        let mut diam = 0.0f32;
        for i in 0..vc {
            if i == skip {
                continue;
            }
            for j in (i + 1)..vc {
                if j == skip {
                    continue;
                }
                let d = self.edge_dist(verts[i], verts[j]);
                if d > diam {
                    diam = d;
                }
            }
        }
        diam
    }

    #[inline]
    fn diameter_edge_mask(&self, verts: &[u16; 6], vc: usize, diam_bits: u32) -> u64 {
        let mut mask = 0u64;
        for i in 0..vc {
            for j in (i + 1)..vc {
                if self.edge_dist_bits(verts[i], verts[j]) == diam_bits {
                    mask |= edge_bit(i, j);
                }
            }
        }
        mask
    }

    #[inline]
    fn zero_pivot_facet(&self, tau: Simplex128) -> Option<Simplex128> {
        let target = tau.filtration_encoded();
        let verts = tau.vertices();
        let vc = tau.vertex_count();

        if vc >= 3 {
            let diam_edges = self.diameter_edge_mask(&verts, vc, !target);
            if let Some(skip) = oldest_same_diam_facet_slot(vc, diam_edges) {
                return Some(tau.remove_vertex_at(skip, target));
            }
            return None;
        }

        for k in 0..vc {
            let diam = self.facet_diameter(&verts, vc, k);
            if encode_filtration(diam) == target {
                return Some(tau.remove_vertex_at(k, target));
            }
        }
        None
    }

    #[inline]
    fn zero_pivot_cofacet(&self, sigma: Simplex128, threshold: f32) -> Option<Simplex128> {
        let same_diam = sigma.filtration();
        if same_diam > threshold {
            return None;
        }
        let vc = sigma.vertex_count();
        if vc == 0 {
            return None;
        }
        let verts = sigma.vertices();
        let w = self.first_common_neighbor_le(&verts, vc, same_diam)?;
        Some(sigma.cofacet(w, sigma.filtration_encoded()))
    }

    pub(crate) fn assemble_candidates(
        &self,
        simplices: &mut Vec<Simplex128>,
        threshold: f32,
        cleared_pivots: &FxHashMap<Simplex128, ()>,
        build_pool: bool,
        parallel: bool,
    ) -> Vec<Simplex128> {
        if simplices.is_empty() {
            return Vec::new();
        }

        // Each simplex expands independently: the only inputs are the immutable
        // matrix and the frozen `cleared_pivots`, and outputs are pure appends.
        // So the loop parallelises by giving each worker its own buffers and
        // concatenating; the final sort makes the merge order irrelevant. The
        // size guard keeps small inputs (and every H1/small-dataset case) on the
        // allocation-free sequential path where rayon's overhead would dominate.
        let use_par = parallel && simplices.len() >= PARALLEL_ASSEMBLE_THRESHOLD;

        let (next_simplices, mut columns_to_reduce) = if use_par {
            use rayon::prelude::*;
            simplices
                .par_iter()
                .fold(
                    || (Vec::new(), Vec::new()),
                    |(mut next_local, mut cols_local), &sigma| {
                        self.assemble_one(
                            sigma,
                            threshold,
                            cleared_pivots,
                            build_pool,
                            &mut next_local,
                            &mut cols_local,
                        );
                        (next_local, cols_local)
                    },
                )
                .reduce(
                    || (Vec::new(), Vec::new()),
                    |(mut na, mut ca), (nb, cb)| {
                        na.extend(nb);
                        ca.extend(cb);
                        (na, ca)
                    },
                )
        } else {
            let mut next_simplices: Vec<Simplex128> = if build_pool {
                Vec::with_capacity(simplices.len() * 2)
            } else {
                Vec::new()
            };
            let mut columns_to_reduce: Vec<Simplex128> = Vec::with_capacity(simplices.len());
            for &sigma in simplices.iter() {
                self.assemble_one(
                    sigma,
                    threshold,
                    cleared_pivots,
                    build_pool,
                    &mut next_simplices,
                    &mut columns_to_reduce,
                );
            }
            (next_simplices, columns_to_reduce)
        };

        if use_par {
            use rayon::prelude::*;
            columns_to_reduce.par_sort_unstable();
        } else {
            columns_to_reduce.sort_unstable();
        }
        if build_pool {
            *simplices = next_simplices;
        }
        columns_to_reduce
    }

    #[inline]
    fn assemble_one(
        &self,
        sigma: Simplex128,
        threshold: f32,
        cleared_pivots: &FxHashMap<Simplex128, ()>,
        build_pool: bool,
        next_simplices: &mut Vec<Simplex128>,
        columns_to_reduce: &mut Vec<Simplex128>,
    ) {
        if sigma.vertex_count() == 2 {
            self.assemble_edge_candidates(
                sigma,
                threshold,
                cleared_pivots,
                build_pool,
                next_simplices,
                columns_to_reduce,
            );
        } else {
            self.assemble_generic_candidates(
                sigma,
                threshold,
                cleared_pivots,
                build_pool,
                next_simplices,
                columns_to_reduce,
            );
        }
    }

    #[inline]
    fn assemble_edge_candidates(
        &self,
        sigma: Simplex128,
        threshold: f32,
        cleared_pivots: &FxHashMap<Simplex128, ()>,
        build_pool: bool,
        next_simplices: &mut Vec<Simplex128>,
        columns_to_reduce: &mut Vec<Simplex128>,
    ) {
        let verts = sigma.vertices();
        let v0 = verts[0];
        let v1 = verts[1];
        let base = sigma.filtration();
        let base_bits = !sigma.filtration_encoded();
        let floor = sigma.largest_vertex() as i32;

        self.for_each_common_neighbor_edges(v0, v1, floor, |w, d0, d1| {
            let extra = d0.max(d1);
            let diam = base.max(extra);
            if diam > threshold {
                return true;
            }

            let tau = Simplex128::from_sorted_desc(diam, &[w, v0, v1]);
            if build_pool {
                next_simplices.push(tau);
            }

            if !cleared_pivots.contains_key(&tau) {
                let diam_bits = diam.to_bits();
                let mut diam_edges = 0u64;
                if d0.to_bits() == diam_bits {
                    diam_edges |= edge_bit(0, 1);
                }
                if d1.to_bits() == diam_bits {
                    diam_edges |= edge_bit(0, 2);
                }
                if base_bits == diam_bits {
                    diam_edges |= edge_bit(1, 2);
                }
                let tau_verts = [w, v0, v1, 0, 0, 0];
                if self.keep_fused_candidate(tau, &tau_verts, 3, diam_edges, threshold) {
                    columns_to_reduce.push(tau);
                }
            }
            true
        });
    }

    #[inline]
    fn assemble_generic_candidates(
        &self,
        sigma: Simplex128,
        threshold: f32,
        cleared_pivots: &FxHashMap<Simplex128, ()>,
        build_pool: bool,
        next_simplices: &mut Vec<Simplex128>,
        columns_to_reduce: &mut Vec<Simplex128>,
    ) {
        self.cofacets(sigma, false, threshold, |tau| {
            if build_pool {
                next_simplices.push(tau);
            }
            if !cleared_pivots.contains_key(&tau) {
                let vc = tau.vertex_count();
                let verts = tau.vertices();
                let diam_edges = self.diameter_edge_mask(&verts, vc, !tau.filtration_encoded());
                if self.keep_fused_candidate(tau, &verts, vc, diam_edges, threshold) {
                    columns_to_reduce.push(tau);
                }
            }
            true
        });
    }

    #[inline]
    fn keep_fused_candidate(
        &self,
        tau: Simplex128,
        tau_verts: &[u16; 6],
        vc: usize,
        diam_edges: u64,
        threshold: f32,
    ) -> bool {
        !self.is_zero_apparent_facet_side(tau, tau_verts, vc, diam_edges, threshold)
            && !self.is_zero_apparent_cofacet_side(tau, tau_verts, vc, diam_edges)
    }

    #[inline]
    fn is_zero_apparent_cofacet_side(
        &self,
        tau: Simplex128,
        tau_verts: &[u16; 6],
        vc: usize,
        diam_edges: u64,
    ) -> bool {
        let Some(skip) = oldest_same_diam_facet_slot(vc, diam_edges) else {
            return false;
        };

        let mut facet_verts = [0u16; 6];
        let mut out = 0usize;
        for i in 0..vc {
            if i != skip {
                facet_verts[out] = tau_verts[i];
                out += 1;
            }
        }

        self.first_common_neighbor_le(&facet_verts, vc - 1, tau.filtration())
            == Some(tau_verts[skip])
    }

    #[inline]
    fn is_zero_apparent_facet_side(
        &self,
        tau: Simplex128,
        tau_verts: &[u16; 6],
        vc: usize,
        diam_edges: u64,
        threshold: f32,
    ) -> bool {
        if vc >= 6 || tau.filtration() > threshold {
            return false;
        }

        let Some(w) = self.first_common_neighbor_le(tau_verts, vc, tau.filtration()) else {
            return false;
        };

        let (_rho_verts, rank) = insert_vertex_desc(tau_verts, vc, w);
        let mut rho_edges = remap_edge_mask_after_insert(diam_edges, vc, rank);
        let diam_bits = !tau.filtration_encoded();
        for i in 0..vc {
            if self.edge_dist_bits(w, tau_verts[i]) == diam_bits {
                let j = if i >= rank { i + 1 } else { i };
                let (a, b) = if rank < j { (rank, j) } else { (j, rank) };
                rho_edges |= edge_bit(a, b);
            }
        }

        oldest_same_diam_facet_slot(vc + 1, rho_edges) == Some(rank)
    }
}

impl Filtration for BitCsrDistanceMatrix {
    #[inline(always)]
    fn n(&self) -> usize {
        self.n
    }

    #[inline]
    fn for_each_edge(&self, threshold: f32, mut f: impl FnMut(Simplex128) -> bool) {
        let check_threshold = threshold < self.threshold;
        for v in 0..self.n {
            let row_start = self.row_start(v);
            let row_len = self.row_len(v);
            if row_len == 0 {
                continue;
            }

            let floor_block = v >> 6;
            if floor_block >= row_len {
                continue;
            }
            for block in (floor_block..row_len).rev() {
                let flat = row_start + block;
                let mut word = unsafe { *self.words.get_unchecked(flat) };
                if block == floor_block {
                    word &= bits_above(v & 63);
                }
                while word != 0 {
                    let bit = highest_set_bit(word);
                    word &= !(1u64 << bit);
                    let w = ((block << 6) | bit) as u16;
                    let d = self.word_distance_unchecked(
                        flat,
                        unsafe { *self.words.get_unchecked(flat) },
                        bit,
                    );
                    if (!check_threshold || d <= threshold)
                        && !f(Simplex128::from_sorted_desc(d, &[w, v as u16]))
                    {
                        return;
                    }
                }
            }
        }
    }

    #[inline]
    fn for_each_cofacet(
        &self,
        sigma: Simplex128,
        all_cofacets: bool,
        threshold: f32,
        f: impl FnMut(Simplex128) -> bool,
    ) {
        self.cofacets(sigma, all_cofacets, threshold, f);
    }

    #[inline]
    fn zero_apparent_facet(&self, tau: Simplex128, threshold: f32) -> Option<Simplex128> {
        let phi = self.zero_pivot_facet(tau)?;
        let tau_check = self.zero_pivot_cofacet(phi, threshold)?;
        if tau_check == tau {
            Some(phi)
        } else {
            None
        }
    }

    #[inline]
    fn zero_apparent_cofacet(&self, sigma: Simplex128, threshold: f32) -> Option<Simplex128> {
        let tau = self.zero_pivot_cofacet(sigma, threshold)?;
        let phi = self.zero_pivot_facet(tau)?;
        if phi == sigma {
            Some(tau)
        } else {
            None
        }
    }

    #[inline]
    fn is_apparent_cofacet(&self, tau: Simplex128, threshold: f32) -> bool {
        let sigma = match self.zero_pivot_facet(tau) {
            Some(s) => s,
            None => return false,
        };
        match self.zero_pivot_cofacet(sigma, threshold) {
            Some(t) => t == tau,
            None => false,
        }
    }
}

#[inline(always)]
fn to_u32(value: usize, what: &str) -> u32 {
    assert!(value <= u32::MAX as usize, "{what} exceeds u32::MAX");
    value as u32
}

#[inline(always)]
fn highest_set_bit(word: u64) -> usize {
    debug_assert!(word != 0);
    63 - word.leading_zeros() as usize
}

#[inline(always)]
fn bits_above(bit: usize) -> u64 {
    if bit >= 63 {
        0
    } else {
        u64::MAX << (bit + 1)
    }
}

#[inline(always)]
fn edge_bit(i: usize, j: usize) -> u64 {
    debug_assert!(i < j && j < 6);
    1u64 << (i * 6 + j)
}

#[inline]
fn oldest_same_diam_facet_slot(vc: usize, diam_edges: u64) -> Option<usize> {
    if vc < 3 || diam_edges == 0 {
        return None;
    }
    for skip in 0..vc {
        for i in 0..vc {
            if i == skip {
                continue;
            }
            for j in (i + 1)..vc {
                if j != skip && (diam_edges & edge_bit(i, j)) != 0 {
                    return Some(skip);
                }
            }
        }
    }
    None
}

#[inline]
fn insert_vertex_desc(verts: &[u16; 6], vc: usize, w: u16) -> ([u16; 6], usize) {
    let mut out = [0u16; 6];
    let mut i = 0usize;
    while i < vc && verts[i] > w {
        out[i] = verts[i];
        i += 1;
    }
    let rank = i;
    out[rank] = w;
    while i < vc {
        out[i + 1] = verts[i];
        i += 1;
    }
    (out, rank)
}

#[inline]
fn remap_edge_mask_after_insert(mask: u64, vc: usize, rank: usize) -> u64 {
    let mut out = 0u64;
    for i in 0..vc {
        for j in (i + 1)..vc {
            if (mask & edge_bit(i, j)) == 0 {
                continue;
            }
            let ni = if i >= rank { i + 1 } else { i };
            let nj = if j >= rank { j + 1 } else { j };
            out |= edge_bit(ni, nj);
        }
    }
    out
}

/// Build a bitset sparse matrix from a dense distance matrix, returning the
/// same Chebyshev radius convention as the CSR builder.
pub(crate) fn bitcsr_from_distance_matrix(
    dist: &DistanceMatrix,
    threshold: f32,
) -> (BitCsrDistanceMatrix, f32) {
    let (csr, r_cheb) = csr_from_distance_matrix(dist, threshold);
    let eff = threshold.min(r_cheb);
    (BitCsrDistanceMatrix::from_csr(&csr, eff), r_cheb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::algorithm::{compute, compute_bitcsr};
    use crate::engine::csr::csr_from_distance_matrix;
    use crate::engine::filtration::Filtration;
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

    fn diff_check(dist: &DistanceMatrix, threshold: f32, max_dim: usize) {
        let (csr, r_cheb) = csr_from_distance_matrix(dist, threshold);
        let eff = threshold.min(r_cheb);
        let bitcsr = BitCsrDistanceMatrix::from_csr(&csr, eff);
        let dense = compute(dist, eff, max_dim);
        let sparse = compute_bitcsr(&bitcsr, eff, max_dim, false);
        assert!(
            barcodes_equal(&dense, &sparse),
            "barcode mismatch (n={}, eff={eff}, max_dim={max_dim})\n dense={:?}\nsparse={:?}",
            dist.n(),
            dense.intervals,
            sparse.intervals,
        );
        // The parallel assembly must produce a bit-identical barcode.
        let sparse_par = compute_bitcsr(&bitcsr, eff, max_dim, true);
        assert!(
            barcodes_equal(&dense, &sparse_par),
            "parallel barcode mismatch (n={}, eff={eff}, max_dim={max_dim})",
            dist.n(),
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
        let (csr, r_cheb) = csr_from_distance_matrix(&dist, 1.0);
        let eff = 1.0f32.min(r_cheb);
        let bitcsr = BitCsrDistanceMatrix::from_csr(&csr, eff);

        let mut edges = Vec::new();
        bitcsr.for_each_edge(eff, |edge| {
            edges.push(edge.vertices());
            true
        });

        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0][0], 65);
        assert_eq!(edges[0][1], 50);
    }

    #[test]
    fn bitcsr_matches_dense_no_ties() {
        for &(n, seed) in &[(5usize, 1u64), (8, 7), (12, 99), (20, 31337)] {
            let mut rng = xorshift(seed);
            let m = n * (n - 1) / 2;
            let data: Vec<f32> = (0..m)
                .map(|_| (rng() >> 40) as f32 / 16_777_216.0)
                .collect();
            let dist = DistanceMatrix::from_lower_triangular(n, data);
            for &thr in &[0.3f32, 0.6, 0.9, f32::INFINITY] {
                for max_dim in 1..=3 {
                    diff_check(&dist, thr, max_dim);
                }
            }
        }
    }

    #[test]
    fn bitcsr_matches_dense_with_ties() {
        for &(n, levels, seed) in &[(6usize, 2u64, 5), (8, 3, 17), (12, 3, 2024)] {
            let mut rng = xorshift(seed);
            let m = n * (n - 1) / 2;
            let data: Vec<f32> = (0..m)
                .map(|_| ((rng() % levels) as f32 + 1.0) * 0.5)
                .collect();
            let dist = DistanceMatrix::from_lower_triangular(n, data);
            for &thr in &[0.5f32, 1.0, 1.5, f32::INFINITY] {
                for max_dim in 1..=3 {
                    diff_check(&dist, thr, max_dim);
                }
            }
        }
    }

    /// Exercise the rayon path (n large enough to cross
    /// `PARALLEL_ASSEMBLE_THRESHOLD`) and assert it equals both the sequential
    /// sparse path and the dense path, with and without diameter ties.
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
                    let (csr, r_cheb) = csr_from_distance_matrix(&dist, thr);
                    let eff = thr.min(r_cheb);
                    let bitcsr = BitCsrDistanceMatrix::from_csr(&csr, eff);
                    let seq = compute_bitcsr(&bitcsr, eff, max_dim, false);
                    let par = compute_bitcsr(&bitcsr, eff, max_dim, true);
                    let dense = compute(&dist, eff, max_dim);
                    assert!(
                        barcodes_equal(&seq, &par),
                        "seq vs par (n={n}, thr={thr}, d={max_dim})"
                    );
                    assert!(
                        barcodes_equal(&dense, &par),
                        "dense vs par (n={n}, thr={thr}, d={max_dim})"
                    );
                }
            }
        }
    }
}
