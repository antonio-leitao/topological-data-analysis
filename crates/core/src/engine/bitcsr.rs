// Each row is an adjacency bitset over the blocks its neighbours span.
// Distances follow set bits in descending bit order, so a popcount rank locates
// each value.

use crate::preprocess::edgelist::EdgeList;
use crate::{Error, Result};

/// Row `v` occupies `words[word_ptr[v]..word_ptr[v + 1]]` and covers blocks
/// `lo[v]..` (ids `64*b..64*b+63` for block `b`), or blocks `0..` when `lo` is
/// empty. `val_ptr` maps each word to values ordered from high bit to low bit.
pub struct BitCsrDistanceMatrix {
    n: usize,
    word_ptr: Vec<u32>,
    /// First stored block of each row, or empty when every row starts at block
    /// 0. Rows start at their first neighbour only when that halves the words,
    /// which needs ids that keep neighbours close; dense graphs keep the
    /// tail-truncated layout, and the hot loops compiled for it never read a
    /// lower bound. `words` opens with padding so that `word_ptr[v] - lo[v]`
    /// is never negative.
    lo: Vec<u32>,
    words: Vec<u64>,
    val_ptr: Vec<u32>,
    val: Vec<f32>,
}

impl BitCsrDistanceMatrix {
    /// Build symmetric rows from canonical half-edges.
    ///
    /// Offsets are 32-bit, which bounds both directed incidences (`2m`) and
    /// bitset words. Words grow with the id range each row's neighbours span,
    /// not with degree, so a sparse graph on several hundred thousand vertices
    /// with scattered ids can exceed the limit on few edges; ids that keep
    /// neighbours close keep rows short.
    pub(crate) fn from_edge_list(edge_list: EdgeList) -> Result<Self> {
        Self::build(edge_list, None)
    }

    /// `banded` overrides the layout choice; tests use it to run both.
    fn build(edge_list: EdgeList, banded: Option<bool>) -> Result<Self> {
        let EdgeList { n, edges, .. } = edge_list;

        let incidences = 2 * edges.len();
        if incidences > u32::MAX as usize {
            return Err(Error::GraphTooLarge {
                what: "the directed edge count (2m)",
                got: incidences,
                max: u32::MAX as usize,
            });
        }

        let mut lo = vec![u32::MAX; n];
        let mut hi = vec![0u32; n];
        for edge in &edges {
            let u = edge.u as usize;
            let v = edge.v as usize;
            debug_assert!(u < n && v < n && u > v);
            for (row, block) in [(u, (v >> 6) as u32), (v, (u >> 6) as u32)] {
                lo[row] = lo[row].min(block);
                hi[row] = hi[row].max(block + 1);
            }
        }

        let (mut tail, mut band) = (0usize, 0usize);
        for (l, &h) in lo.iter_mut().zip(&hi) {
            *l = (*l).min(h);
            tail += h as usize;
            band += (h - *l) as usize;
        }
        if !banded.unwrap_or(2 * band <= tail) {
            lo = Vec::new();
        }

        // Pad the front so that `start - lo` is never negative.
        let (mut word_count, mut pad) = (0usize, 0usize);
        for (v, &h) in hi.iter().enumerate() {
            let l = lo.get(v).map_or(0, |&l| l as usize);
            pad = pad.max(l.saturating_sub(word_count));
            word_count += h as usize - l;
        }
        word_count += pad;
        if word_count > u32::MAX as usize {
            return Err(Error::GraphTooLarge {
                what: "the engine's adjacency bitset word count",
                got: word_count,
                max: u32::MAX as usize,
            });
        }

        let mut word_ptr = Vec::with_capacity(n + 1);
        let mut start = pad;
        word_ptr.push(start as u32);
        for (v, &h) in hi.iter().enumerate() {
            start += h as usize - lo.get(v).map_or(0, |&l| l as usize);
            word_ptr.push(start as u32);
        }
        drop(hi);
        // Where block 0 of each row would sit.
        let origin: Vec<usize> = (0..n)
            .map(|v| word_ptr[v] as usize - lo.get(v).map_or(0, |&l| l as usize))
            .collect();

        let mut words = vec![0u64; word_count];
        for edge in &edges {
            set_neighbor_bit(&origin, &mut words, edge.u as usize, edge.v as usize);
            set_neighbor_bit(&origin, &mut words, edge.v as usize, edge.u as usize);
        }

        let mut val_ptr = Vec::with_capacity(word_count + 1);
        val_ptr.push(0);
        let mut value_count = 0usize;
        for &word in &words {
            value_count += word.count_ones() as usize;
            val_ptr.push(to_u32(value_count, "bitcsr value count"));
        }
        debug_assert_eq!(value_count, edges.len() * 2, "duplicate edge in EdgeList");

        let mut val = vec![0.0f32; value_count];
        for edge in edges {
            set_edge_distance(
                &origin,
                &words,
                &val_ptr,
                &mut val,
                edge.u as usize,
                edge.v as usize,
                edge.distance,
            );
            set_edge_distance(
                &origin,
                &words,
                &val_ptr,
                &mut val,
                edge.v as usize,
                edge.u as usize,
                edge.distance,
            );
        }

        Ok(Self {
            n,
            word_ptr,
            lo,
            words,
            val_ptr,
            val,
        })
    }
}

/// The BitCSR's arrays with the row layout fixed at compile time: `BANDED`
/// must match whether rows carry lower bounds. The engine queries through
/// this, so a computation tests the layout once, and every query reaches the
/// arrays in one step, as it did through the matrix itself.
#[derive(Clone, Copy)]
pub(crate) struct View<'a, const BANDED: bool> {
    n: usize,
    word_ptr: &'a [u32],
    lo: &'a [u32],
    words: &'a [u64],
    val_ptr: &'a [u32],
    val: &'a [f32],
}

impl<const BANDED: bool> View<'_, BANDED> {
    /// `(origin, lo, hi)` of row `v`: block `b` for `lo <= b < hi` is the word
    /// at `origin + b`. `BANDED` must match the layout.
    #[inline(always)]
    fn row(&self, v: usize) -> (usize, usize, usize) {
        let start = unsafe { *self.word_ptr.get_unchecked(v) as usize };
        let len = unsafe { *self.word_ptr.get_unchecked(v + 1) as usize } - start;
        if BANDED {
            let lo = unsafe { *self.lo.get_unchecked(v) as usize };
            (start - lo, lo, lo + len)
        } else {
            (start, 0, len)
        }
    }

    #[inline(always)]
    fn word_distance_unchecked(&self, flat_word: usize, word: u64, bit: usize) -> f32 {
        debug_assert!((word & (1u64 << bit)) != 0);
        let higher = bits_above(bit);
        let rank = (word & higher).count_ones() as usize;
        let start = unsafe { *self.val_ptr.get_unchecked(flat_word) as usize };
        unsafe { *self.val.get_unchecked(start + rank) }
    }

    /// `a` and `b` must be adjacent.
    #[inline(always)]
    fn edge_dist(&self, a: u32, b: u32) -> f32 {
        let block = (b as usize) >> 6;
        let bit = (b as usize) & 63;
        let (origin, lo, hi) = self.row(a as usize);
        debug_assert!(
            (lo..hi).contains(&block),
            "edge_dist on a missing edge ({a}, {b})"
        );
        let flat = origin + block;
        let word = unsafe { *self.words.get_unchecked(flat) };
        debug_assert!(
            (word & (1u64 << bit)) != 0,
            "edge_dist on a missing edge ({a}, {b})"
        );
        self.word_distance_unchecked(flat, word, bit)
    }

    /// Common neighbours above `floor` in descending id order, with the
    /// largest incident distance.
    #[inline(always)]
    fn common_neighbors<const K: usize>(
        &self,
        verts: &[u32; K],
        vc: usize,
        floor: i32,
        mut f: impl FnMut(u32, f32) -> bool,
    ) {
        if vc == 0 {
            return;
        }

        let mut row_start = [0usize; K];
        let mut start = 0usize;
        let mut limit = usize::MAX;
        for i in 0..vc {
            let (origin, lo, hi) = self.row(verts[i] as usize);
            row_start[i] = origin;
            if hi == lo {
                return;
            }
            if hi < limit {
                limit = hi;
            }
            if BANDED {
                start = start.max(lo);
            }
        }
        if BANDED && start >= limit {
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
        let first = if BANDED {
            floor_block.max(start)
        } else {
            floor_block
        };

        for block in (first..limit).rev() {
            let mut mask = self.intersection_word(&row_start, vc, block);
            if floor >= 0 && block == floor_block {
                mask &= bits_above(floor_bit);
            }

            while mask != 0 {
                let bit = highest_set_bit(mask);
                mask &= !(1u64 << bit);
                let w = ((block << 6) | bit) as u32;
                let extra = self.max_extra(&row_start, vc, block, bit);
                if !f(w, extra) {
                    return;
                }
            }
        }
    }

    /// Common neighbours as `(w, d(w,v0), d(w,v1))`, descending.
    #[inline(always)]
    fn common_neighbor_edges(
        &self,
        v0: u32,
        v1: u32,
        floor: i32,
        mut f: impl FnMut(u32, f32, f32) -> bool,
    ) {
        let (s0, lo0, hi0) = self.row(v0 as usize);
        let (s1, lo1, hi1) = self.row(v1 as usize);
        let limit = hi0.min(hi1);
        let start = lo0.max(lo1);
        if limit == 0 || (BANDED && start >= limit) {
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
        let first = if BANDED {
            floor_block.max(start)
        } else {
            floor_block
        };

        for block in (first..limit).rev() {
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
                let w = ((block << 6) | bit) as u32;
                let d0 = self.word_distance_unchecked(f0, word0, bit);
                let d1 = self.word_distance_unchecked(f1, word1, bit);
                if !f(w, d0, d1) {
                    return;
                }
            }
        }
    }

    /// Neighbours above `v`, descending.
    #[inline(always)]
    fn neighbors_above(&self, v: usize, mut f: impl FnMut(u32, f32) -> bool) {
        let (origin, lo, hi) = self.row(v);
        let floor_block = v >> 6;
        if floor_block >= hi {
            return;
        }
        let first = if BANDED {
            floor_block.max(lo)
        } else {
            floor_block
        };
        for block in (first..hi).rev() {
            let flat = origin + block;
            let word = unsafe { *self.words.get_unchecked(flat) };
            let mut iter_word = word;
            if block == floor_block {
                iter_word &= bits_above(v & 63);
            }
            while iter_word != 0 {
                let bit = highest_set_bit(iter_word);
                iter_word &= !(1u64 << bit);
                let w = ((block << 6) | bit) as u32;
                let d = self.word_distance_unchecked(flat, word, bit);
                if !f(w, d) {
                    return;
                }
            }
        }
    }

    #[inline(always)]
    fn intersection_word<const K: usize>(
        &self,
        row_start: &[usize; K],
        vc: usize,
        block: usize,
    ) -> u64 {
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

            // `K` is constant per monomorphisation, allowing full unrolling.
            let mut k = 2;
            while k + 1 < K {
                let w = *words.get_unchecked(row_start[k] + block);
                if w == 0 {
                    return 0;
                }
                acc &= w;
                if acc == 0 || vc == k + 1 {
                    return acc;
                }
                k += 1;
            }

            let w = *words.get_unchecked(row_start[K - 1] + block);
            if w == 0 {
                return 0;
            }
            acc & w
        }
    }

    #[inline(always)]
    fn max_extra<const K: usize>(
        &self,
        row_start: &[usize; K],
        vc: usize,
        block: usize,
        bit: usize,
    ) -> f32 {
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

/// The adjacency queries the engine makes.
///
/// The BitCSR answers them for either row layout, which costs a layout test on
/// every query; a [`View`] fixes the layout at compile time, so a computation
/// tests it once. On small dense inputs the per-query test cost several
/// percent of the whole run.
pub(crate) trait Rows: Sync {
    fn n(&self) -> usize;
    /// `a` and `b` must be adjacent.
    fn edge_dist(&self, a: u32, b: u32) -> f32;
    fn edge_dist_bits(&self, a: u32, b: u32) -> u32;
    /// Common neighbours above `floor`, descending, with the largest incident
    /// distance.
    fn for_each_common_neighbor<const K: usize>(
        &self,
        verts: &[u32; K],
        vc: usize,
        floor: i32,
        f: impl FnMut(u32, f32) -> bool,
    );
    /// Common neighbours as `(w, d(w,v0), d(w,v1))`, descending.
    fn for_each_common_neighbor_edges(
        &self,
        v0: u32,
        v1: u32,
        floor: i32,
        f: impl FnMut(u32, f32, f32) -> bool,
    );
    /// Neighbours above `v`, descending.
    fn for_each_neighbor_above(&self, v: usize, f: impl FnMut(u32, f32) -> bool);
}

impl BitCsrDistanceMatrix {
    /// Whether rows start at their first neighbour rather than block 0.
    #[inline]
    pub(crate) fn banded(&self) -> bool {
        !self.lo.is_empty()
    }

    /// This matrix seen through its layout; `BANDED` must equal
    /// [`Self::banded`].
    #[inline]
    pub(crate) fn view<const BANDED: bool>(&self) -> View<'_, BANDED> {
        debug_assert_eq!(BANDED, self.banded(), "view of the wrong row layout");
        View {
            n: self.n,
            word_ptr: &self.word_ptr,
            lo: &self.lo,
            words: &self.words,
            val_ptr: &self.val_ptr,
            val: &self.val,
        }
    }

    /// `a` and `b` must be adjacent.
    #[inline]
    pub(crate) fn edge_dist(&self, a: u32, b: u32) -> f32 {
        if self.banded() {
            self.view::<true>().edge_dist(a, b)
        } else {
            self.view::<false>().edge_dist(a, b)
        }
    }

    /// Common neighbours above `floor` in descending id order, with the
    /// largest incident distance.
    #[inline]
    pub(crate) fn for_each_common_neighbor<const K: usize>(
        &self,
        verts: &[u32; K],
        vc: usize,
        floor: i32,
        f: impl FnMut(u32, f32) -> bool,
    ) {
        if self.banded() {
            self.view::<true>().common_neighbors(verts, vc, floor, f)
        } else {
            self.view::<false>().common_neighbors(verts, vc, floor, f)
        }
    }

    /// Common neighbours as `(w, d(w,v0), d(w,v1))`, descending.
    #[inline]
    pub(crate) fn for_each_common_neighbor_edges(
        &self,
        v0: u32,
        v1: u32,
        floor: i32,
        f: impl FnMut(u32, f32, f32) -> bool,
    ) {
        if self.banded() {
            self.view::<true>().common_neighbor_edges(v0, v1, floor, f)
        } else {
            self.view::<false>().common_neighbor_edges(v0, v1, floor, f)
        }
    }

    /// Neighbours above `v` in descending id order.
    #[inline]
    pub(crate) fn for_each_neighbor_above(&self, v: usize, f: impl FnMut(u32, f32) -> bool) {
        if self.banded() {
            self.view::<true>().neighbors_above(v, f)
        } else {
            self.view::<false>().neighbors_above(v, f)
        }
    }
}

impl<const BANDED: bool> Rows for View<'_, BANDED> {
    #[inline(always)]
    fn n(&self) -> usize {
        self.n
    }
    #[inline(always)]
    fn edge_dist(&self, a: u32, b: u32) -> f32 {
        View::edge_dist(self, a, b)
    }
    #[inline(always)]
    fn edge_dist_bits(&self, a: u32, b: u32) -> u32 {
        View::edge_dist(self, a, b).to_bits()
    }
    #[inline(always)]
    fn for_each_common_neighbor<const K: usize>(
        &self,
        verts: &[u32; K],
        vc: usize,
        floor: i32,
        f: impl FnMut(u32, f32) -> bool,
    ) {
        self.common_neighbors(verts, vc, floor, f)
    }
    #[inline(always)]
    fn for_each_common_neighbor_edges(
        &self,
        v0: u32,
        v1: u32,
        floor: i32,
        f: impl FnMut(u32, f32, f32) -> bool,
    ) {
        self.common_neighbor_edges(v0, v1, floor, f)
    }
    #[inline(always)]
    fn for_each_neighbor_above(&self, v: usize, f: impl FnMut(u32, f32) -> bool) {
        self.neighbors_above(v, f)
    }
}

/// Per-query layout tests, for callers off the hot path.
impl Rows for BitCsrDistanceMatrix {
    #[inline]
    fn n(&self) -> usize {
        self.n
    }
    #[inline]
    fn edge_dist(&self, a: u32, b: u32) -> f32 {
        BitCsrDistanceMatrix::edge_dist(self, a, b)
    }
    #[inline]
    fn edge_dist_bits(&self, a: u32, b: u32) -> u32 {
        BitCsrDistanceMatrix::edge_dist(self, a, b).to_bits()
    }
    #[inline]
    fn for_each_common_neighbor<const K: usize>(
        &self,
        verts: &[u32; K],
        vc: usize,
        floor: i32,
        f: impl FnMut(u32, f32) -> bool,
    ) {
        BitCsrDistanceMatrix::for_each_common_neighbor(self, verts, vc, floor, f)
    }
    #[inline]
    fn for_each_common_neighbor_edges(
        &self,
        v0: u32,
        v1: u32,
        floor: i32,
        f: impl FnMut(u32, f32, f32) -> bool,
    ) {
        BitCsrDistanceMatrix::for_each_common_neighbor_edges(self, v0, v1, floor, f)
    }
    #[inline]
    fn for_each_neighbor_above(&self, v: usize, f: impl FnMut(u32, f32) -> bool) {
        BitCsrDistanceMatrix::for_each_neighbor_above(self, v, f)
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

#[inline(always)]
fn set_neighbor_bit(origin: &[usize], words: &mut [u64], row: usize, neighbor: usize) {
    let flat = origin[row] + (neighbor >> 6);
    words[flat] |= 1u64 << (neighbor & 63);
}

#[inline(always)]
fn set_edge_distance(
    origin: &[usize],
    words: &[u64],
    val_ptr: &[u32],
    val: &mut [f32],
    row: usize,
    neighbor: usize,
    distance: f32,
) {
    let flat = origin[row] + (neighbor >> 6);
    let bit = neighbor & 63;
    let rank = (words[flat] & bits_above(bit)).count_ones() as usize;
    val[val_ptr[flat] as usize + rank] = distance;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::algorithm::{compute, for_each_edge};
    use crate::engine::simplex::Simplex128;
    use crate::preprocess::edgelist::square_from_lower_tri;
    use crate::preprocess::edgelist::{Edge, EdgeList};
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

    fn bitcsr_from_lower_tri(lt: &[f32], n: usize, threshold: f32) -> (BitCsrDistanceMatrix, f32) {
        let sq = square_from_lower_tri(n, lt);
        let edges = EdgeList::from_distance_matrix(&sq, n, threshold).unwrap();
        let effective = edges.threshold;
        (
            BitCsrDistanceMatrix::from_edge_list(edges).unwrap(),
            effective,
        )
    }

    fn seq_par_check(lt: &[f32], n: usize, threshold: f32, max_dim: usize) {
        let (bitcsr, eff) = bitcsr_from_lower_tri(lt, n, threshold);
        let seq = compute::<6, Simplex128>(&bitcsr, max_dim, false);
        let par = compute::<6, Simplex128>(&bitcsr, max_dim, true);
        assert!(
            barcodes_equal(&seq, &par),
            "seq vs par barcode mismatch (n={n}, eff={eff}, max_dim={max_dim})\n seq={:?}\n par={:?}",
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

        let (bitcsr, _eff) = bitcsr_from_lower_tri(&data, n, 1.0);

        let mut edges = Vec::new();
        for_each_edge(&bitcsr, |edge: Simplex128| {
            edges.push(edge.vertices());
            true
        });

        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0][0], 65);
        assert_eq!(edges[0][1], 50);
    }

    /// A row spans its smallest to largest neighbour, so 16 385 vertices each
    /// adjacent to both the first and the last of 2^24 - 1 vertices need more
    /// than 2^32 words. That used to panic.
    #[test]
    fn wide_rows_error_instead_of_panicking() {
        let n = crate::MAX_POINTS;
        let edges = (1..16_386u32)
            .flat_map(|v| {
                [
                    Edge {
                        u: (n - 1) as u32,
                        v,
                        distance: 1.0,
                    },
                    Edge {
                        u: v,
                        v: 0,
                        distance: 1.0,
                    },
                ]
            })
            .collect();
        let edge_list = EdgeList {
            n,
            edges,
            center: 0,
            threshold: 1.0,
            sorted: false,
        };
        let err = BitCsrDistanceMatrix::from_edge_list(edge_list).err();
        assert!(
            matches!(err, Some(crate::Error::GraphTooLarge { .. })),
            "{err:?}"
        );
    }

    /// Edges into the last of 2^24 - 1 vertices cost a word per row, not a
    /// row as wide as the id; the same edges used to need more than 2^32 words.
    #[test]
    fn far_ids_with_near_neighbours_stay_compact() {
        let n = crate::MAX_POINTS;
        let last = (n - 1) as u32;
        let edges = (0..16_385u32)
            .map(|v| Edge {
                u: last,
                v,
                distance: 1.0 + v as f32,
            })
            .collect();
        let edge_list = EdgeList {
            n,
            edges,
            center: 0,
            threshold: f32::INFINITY,
            sorted: false,
        };
        let bitcsr = BitCsrDistanceMatrix::from_edge_list(edge_list).unwrap();
        // One word per short row, 257 for the hub, and a front pad of at most
        // n / 64 words.
        let stored = 16_385 + 16_385usize.div_ceil(64);
        assert!(!bitcsr.lo.is_empty());
        assert!(bitcsr.words.len() <= stored + (n >> 6), "{}", bitcsr.words.len());
        assert_eq!(bitcsr.edge_dist(last, 70), 71.0);
        assert_eq!(bitcsr.edge_dist(70, last), 71.0);
        let mut above = Vec::new();
        bitcsr.for_each_neighbor_above(70, |w, d| {
            above.push((w, d));
            true
        });
        assert_eq!(above, vec![(last, 71.0)]);
    }

    /// Both row layouts give the same barcode, on graphs sparse enough that
    /// rows start well past block 0 and on ones where they do not.
    #[test]
    fn banded_rows_match_tail_rows() {
        for &(n, keep, seed) in &[(200usize, 6u64, 3u64), (300, 3, 11), (100, 30, 5), (70, 100, 9)] {
            let mut rng = xorshift(seed);
            let mut edges = Vec::new();
            for u in 1..n as u32 {
                for v in 0..u {
                    // Mostly local edges plus a few long ones, with ties.
                    let local = u - v < 70;
                    if rng() % 100 < if local { keep } else { 1 } {
                        let d = ((rng() % 50) as f32 + 1.0) * 0.25;
                        edges.push(Edge { u, v, distance: d });
                    }
                }
            }
            // H3 on the denser graphs costs seconds and adds no coverage.
            let top = if keep < 10 { 3 } else { 2 };
            for max_dim in 1..=top {
                let make = || EdgeList {
                    n,
                    edges: edges.clone(),
                    center: 0,
                    threshold: f32::INFINITY,
                    sorted: false,
                };
                let banded = BitCsrDistanceMatrix::build(make(), Some(true)).unwrap();
                let tail = BitCsrDistanceMatrix::build(make(), Some(false)).unwrap();
                assert!(!banded.lo.is_empty() && tail.lo.is_empty());
                assert!(banded.words.len() <= tail.words.len() + (n >> 6) + 1);
                for parallel in [false, true] {
                    let a = compute::<6, Simplex128>(&banded, max_dim, parallel);
                    let b = compute::<6, Simplex128>(&tail, max_dim, parallel);
                    assert!(barcodes_equal(&a, &b), "n={n} keep={keep} max_dim={max_dim}");
                }
            }
        }
    }

    #[test]
    fn bitcsr_seq_matches_par_no_ties() {
        for &(n, seed) in &[(5usize, 1u64), (8, 7), (12, 99), (20, 31337)] {
            let mut rng = xorshift(seed);
            let m = n * (n - 1) / 2;
            let data: Vec<f32> = (0..m)
                .map(|_| (rng() >> 40) as f32 / 16_777_216.0)
                .collect();
            for &thr in &[0.3f32, 0.6, 0.9, f32::INFINITY] {
                for max_dim in 1..=3 {
                    seq_par_check(&data, n, thr, max_dim);
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
            for &thr in &[0.5f32, 1.0, 1.5, f32::INFINITY] {
                for max_dim in 1..=3 {
                    seq_par_check(&data, n, thr, max_dim);
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
            for &thr in &[0.4f32, 0.8, f32::INFINITY] {
                // max_dim 3 exercises the parallel `build_pool` path that
                // concatenates the next-dimension simplex pool across workers.
                // It runs at the tightest threshold only: there the dim-2 pool
                // is still ~5k simplices (far above `PARALLEL_ASSEMBLE_THRESHOLD`,
                // so the path is genuinely taken) while the filtration is ~37x
                // smaller than at thr=0.8, which alone cost 68% of this test.
                let max_dims: &[usize] = if thr < 0.5 { &[1, 2, 3] } else { &[1, 2] };
                for &max_dim in max_dims {
                    seq_par_check(&data, n, thr, max_dim);
                }
            }
        }
    }
}
