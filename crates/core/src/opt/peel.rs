//! Sound strong-collapse truncation over a weighted edge list.
//!
//! [`peel`] sorts and tightens an [`EdgeList`] in place. If its cutoff is `R`,
//! the resulting cutoff `R' <= R` is certified so that `VR(X, t)` is
//! contractible for every `t >= R'`.
//!
//! Neighborhoods are stored as block-sparse bitset rows ([`SparseRows`]):
//! O(E) memory instead of the dense n²/64 bytes, with the same word-parallel
//! subset/intersection walks over the occupied 64-bit blocks.

use crate::preprocess::edgelist::{Edge, EdgeList};
use rayon::prelude::*;

#[inline(always)]
fn bit_set(bits: &[u64], i: usize) -> bool {
    (bits[i >> 6] >> (i & 63)) & 1 == 1
}

/// Block-sparse closed neighborhoods: per vertex, the sorted nonzero 64-bit
/// blocks of its adjacency bitset (self bit included), flattened CSR-style.
/// After construction bits are only ever CLEARED, so the block skeleton is
/// immutable; emptied words stay as zeros and are skipped by the walks.
struct SparseRows {
    row_ptr: Vec<u32>,
    block_ids: Vec<u32>,
    words: Vec<u64>,
}

impl SparseRows {
    /// Build from a canonical edge list. `deg` are closed-neighborhood sizes
    /// (self included), as maintained by the caller for the precheck.
    fn build(n: usize, edges: &[Edge], deg: &[u32]) -> Self {
        // Transient CSR neighbor lists (2 B per directed edge).
        let mut adj_ptr = vec![0u32; n + 1];
        for v in 0..n {
            adj_ptr[v + 1] = adj_ptr[v] + (deg[v] - 1);
        }
        let mut adj = vec![0u32; adj_ptr[n] as usize];
        let mut fill: Vec<u32> = adj_ptr.clone();
        for edge in edges {
            adj[fill[edge.u as usize] as usize] = edge.v;
            fill[edge.u as usize] += 1;
            adj[fill[edge.v as usize] as usize] = edge.u;
            fill[edge.v as usize] += 1;
        }

        // Scatter each row into a dense scratch row, then harvest the touched
        // blocks in sorted order. O(deg) per row, one nw-word scratch total.
        let nw = (n + 63) >> 6;
        let mut scratch = vec![0u64; nw];
        let mut touched: Vec<u32> = Vec::new();
        let mut row_ptr = Vec::with_capacity(n + 1);
        row_ptr.push(0u32);
        let mut block_ids: Vec<u32> = Vec::new();
        let mut words: Vec<u64> = Vec::new();

        for v in 0..n {
            touched.clear();
            for id in std::iter::once(v as u32).chain(
                adj[adj_ptr[v] as usize..adj_ptr[v + 1] as usize]
                    .iter()
                    .copied(),
            ) {
                let b = (id >> 6) as usize;
                if scratch[b] == 0 {
                    touched.push(b as u32);
                }
                scratch[b] |= 1u64 << (id & 63);
            }
            touched.sort_unstable();
            for &b in &touched {
                block_ids.push(b);
                words.push(scratch[b as usize]);
                scratch[b as usize] = 0;
            }
            debug_assert!(block_ids.len() <= u32::MAX as usize);
            row_ptr.push(block_ids.len() as u32);
        }

        Self {
            row_ptr,
            block_ids,
            words,
        }
    }

    #[inline(always)]
    fn range(&self, v: usize) -> std::ops::Range<usize> {
        self.row_ptr[v] as usize..self.row_ptr[v + 1] as usize
    }

    /// N[w] ⊆ N[z], via a two-pointer walk over the sorted block ids.
    #[inline]
    fn is_subset(&self, w: usize, z: usize) -> bool {
        let rw = self.range(w);
        let rz = self.range(z);
        let (wb, ww) = (&self.block_ids[rw.clone()], &self.words[rw]);
        let (zb, zw) = (&self.block_ids[rz.clone()], &self.words[rz]);
        let mut zi = 0usize;
        for (i, &b) in wb.iter().enumerate() {
            let word = ww[i];
            if word == 0 {
                continue; // fully-removed block
            }
            while zi < zb.len() && zb[zi] < b {
                zi += 1;
            }
            if zi == zb.len() || zb[zi] != b || word & !zw[zi] != 0 {
                return false;
            }
        }
        true
    }

    /// Clear neighbor bit `u` in row `v`. The block exists by construction.
    #[inline]
    fn clear(&mut self, v: usize, u: usize) {
        let r = self.range(v);
        let block = (u >> 6) as u32;
        let idx = self.block_ids[r.clone()]
            .binary_search(&block)
            .expect("clearing an edge that was never stored");
        self.words[r.start + idx] &= !(1u64 << (u & 63));
    }
}

/// Tighten an edge list's cutoff via sound strong-collapse peeling.
///
/// The list is sorted by ascending `(distance, endpoints)`, truncated to the
/// certified radius, and left sorted for subsequent optimizations. Its center is
/// the fixed cone apex and is not changed.
pub(crate) fn peel(edge_list: &mut EdgeList, parallel: bool) {
    if !edge_list.sorted {
        if parallel {
            edge_list
                .edges
                .par_sort_unstable_by_key(|edge| (edge.distance.to_bits(), edge.u, edge.v));
        } else {
            edge_list
                .edges
                .sort_unstable_by_key(|edge| (edge.distance.to_bits(), edge.u, edge.v));
        }
        edge_list.sorted = true;
    }

    let n = edge_list.n;
    if n <= 1 {
        edge_list.edges.clear();
        edge_list.threshold = 0.0;
        return;
    }

    let center = edge_list.center as usize;
    debug_assert!(center < n, "peel: center out of range");
    debug_assert!(edge_list
        .edges
        .iter()
        .all(|edge| edge.distance <= edge_list.threshold));

    let nw = (n + 63) >> 6;

    // Center ball at the starting cutoff. One dense row of n bits is cheap;
    // only the per-vertex neighborhoods need the sparse representation.
    let mut ball_bits = vec![0u64; nw];
    ball_bits[center >> 6] |= 1u64 << (center & 63);

    // Closed-neighborhood sizes, used both to size the sparse rows and for the
    // O(1) domination precheck: N[w] ⊆ N[z] requires |N[w]| <= |N[z]|. Kept
    // exact under edge removals, so the precheck only ever skips subset tests
    // that would fail.
    let mut deg: Vec<u32> = vec![1; n];
    for edge in &edge_list.edges {
        let u = edge.u as usize;
        let v = edge.v as usize;
        deg[u] += 1;
        deg[v] += 1;

        // At the starting cutoff, the center ball is exactly the center and its
        // present neighbors. Missing center edges are outside the ball.
        if u == center {
            ball_bits[v >> 6] |= 1u64 << (v & 63);
        } else if v == center {
            ball_bits[u >> 6] |= 1u64 << (u & 63);
        }
    }

    // Neighborhoods at the current cutoff, including the diagonal.
    let mut rows = SparseRows::build(n, &edge_list.edges, &deg);

    let mut dirty: Vec<u32> = Vec::with_capacity(n);
    let mut in_dirty = vec![false; n];
    for (x, queued) in in_dirty.iter_mut().enumerate() {
        if !bit_set(&ball_bits, x) {
            *queued = true;
            dirty.push(x as u32);
        }
    }

    // Cached dominators and their reverse users. Reverse entries may become
    // stale and are filtered against `dominator` when revisited.
    let mut dominator: Vec<i32> = vec![-1; n];
    let mut users: Vec<Vec<u32>> = vec![Vec::new(); n];

    let mut certified_radius = edge_list.threshold;
    let mut top = edge_list.edges.len();

    // Descend through distinct edge radii, high to low.
    'levels: while top > 0 {
        let radius_bits = edge_list.edges[top - 1].distance.to_bits();
        let radius = f32::from_bits(radius_bits);

        // Neighborhood and ball bitsets currently represent scale `radius`.
        while let Some(w_u32) = dirty.pop() {
            let w = w_u32 as usize;
            in_dirty[w] = false;
            if bit_set(&ball_bits, w) {
                continue;
            }

            let cached = dominator[w];
            if cached >= 0 {
                let z = cached as usize;
                if bit_set(&ball_bits, z) && deg[z] >= deg[w] && rows.is_subset(w, z) {
                    continue;
                }
            }

            // Candidates z ∈ N(w) ∩ ball, ascending id over w's occupied blocks.
            let w_block = (w >> 6) as u32;
            let w_mask = !(1u64 << (w & 63));
            let mut found = false;
            'search: for i in rows.range(w) {
                let block = rows.block_ids[i];
                let mut bits = rows.words[i] & ball_bits[block as usize];
                if block == w_block {
                    bits &= w_mask;
                }
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let z = ((block as usize) << 6) | bit;
                    if deg[z] >= deg[w] && rows.is_subset(w, z) {
                        dominator[w] = z as i32;
                        users[z].push(w as u32);
                        found = true;
                        break 'search;
                    }
                }
            }

            if !found {
                break 'levels;
            }
        }

        certified_radius = radius;

        // Descend just below this radius by removing the entire tied edge level.
        while top > 0 && edge_list.edges[top - 1].distance.to_bits() == radius_bits {
            top -= 1;
            let a = edge_list.edges[top].u as usize;
            let b = edge_list.edges[top].v as usize;

            rows.clear(a, b);
            rows.clear(b, a);
            deg[a] -= 1;
            deg[b] -= 1;

            for &w_u32 in &users[a] {
                let w = w_u32 as usize;
                if dominator[w] == a as i32 && !in_dirty[w] {
                    in_dirty[w] = true;
                    dirty.push(w as u32);
                }
            }
            for &w_u32 in &users[b] {
                let w = w_u32 as usize;
                if dominator[w] == b as i32 && !in_dirty[w] {
                    in_dirty[w] = true;
                    dirty.push(w as u32);
                }
            }

            // Removing an edge incident to the center is exactly the event where
            // the other endpoint leaves the center ball.
            let departure = if a == center {
                Some(b)
            } else if b == center {
                Some(a)
            } else {
                None
            };
            if let Some(p) = departure {
                ball_bits[p >> 6] &= !(1u64 << (p & 63));
                if !in_dirty[p] {
                    in_dirty[p] = true;
                    dirty.push(p as u32);
                }
            }
        }
    }

    edge_list.threshold = certified_radius;
    let radius_bits = certified_radius.to_bits();
    let keep = edge_list
        .edges
        .partition_point(|edge| edge.distance.to_bits() <= radius_bits);
    edge_list.edges.truncate(keep);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::edgelist::EdgeList;

    fn unit_edge(u: u32, v: u32) -> Edge {
        Edge {
            u,
            v,
            distance: 1.0,
        }
    }

    #[test]
    fn sparse_rows_subset_and_clear() {
        // Triangle 0-1-2 plus pendant 3 attached to 0.
        let edges = vec![
            unit_edge(1, 0),
            unit_edge(2, 0),
            unit_edge(2, 1),
            unit_edge(3, 0),
        ];
        let mut deg = vec![1u32; 4];
        for e in &edges {
            deg[e.u as usize] += 1;
            deg[e.v as usize] += 1;
        }
        let mut rows = SparseRows::build(4, &edges, &deg);

        // N[1] = {0,1,2} ⊆ N[0] = {0,1,2,3}; the converse fails.
        assert!(rows.is_subset(1, 0));
        assert!(!rows.is_subset(0, 1));
        // Pendant: N[3] = {0,3} ⊆ N[0].
        assert!(rows.is_subset(3, 0));

        // Remove edge (2,1): N[1] = {0,1} and N[2] = {0,2}, both still ⊆ N[0].
        rows.clear(2, 1);
        rows.clear(1, 2);
        assert!(rows.is_subset(1, 0));
        assert!(rows.is_subset(2, 0));
        assert!(!rows.is_subset(0, 2));
    }

    #[test]
    fn sparse_rows_cross_block_ids() {
        // Star centered at 100 with leaves straddling three 64-bit blocks.
        let n = 200usize;
        let leaves = [3u32, 63, 64, 65, 127, 128, 199];
        let edges: Vec<Edge> = leaves
            .iter()
            .map(|&l| {
                let (u, v) = if l > 100 { (l, 100) } else { (100, l) };
                unit_edge(u, v)
            })
            .collect();
        let mut deg = vec![1u32; n];
        for e in &edges {
            deg[e.u as usize] += 1;
            deg[e.v as usize] += 1;
        }
        let rows = SparseRows::build(n, &edges, &deg);

        for &l in &leaves {
            assert!(rows.is_subset(l as usize, 100), "leaf {l} ⊆ center");
            assert!(!rows.is_subset(100, l as usize), "center ⊄ leaf {l}");
        }
        // Distinct leaves are incomparable (each contains itself).
        assert!(!rows.is_subset(3, 199));
        assert!(!rows.is_subset(199, 3));
    }

    /// End-to-end peel on a point cloud large enough that rows span multiple
    /// 64-bit blocks (n = 144 > 128).
    #[test]
    fn multi_block_grid_peels_soundly() {
        let mut points = Vec::new();
        for x in 0..12 {
            for y in 0..12 {
                points.push(x as f32);
                points.push(y as f32);
            }
        }
        let mut edge_list = EdgeList::from_points(&points, 144, 2, f32::INFINITY, false);
        let center = edge_list.center;
        let initial = edge_list.threshold;
        peel(&mut edge_list, false);
        assert!(edge_list.threshold.is_finite() && edge_list.threshold >= 0.0);
        assert_finalized(&edge_list, center, initial);
    }

    fn square_from_lower_tri(n: usize, lower: &[f32]) -> Vec<f32> {
        assert_eq!(lower.len(), n * (n - 1) / 2);
        let mut matrix = vec![0.0; n * n];
        let mut k = 0;
        for i in 1..n {
            for j in 0..i {
                matrix[i * n + j] = lower[k];
                matrix[j * n + i] = lower[k];
                k += 1;
            }
        }
        matrix
    }

    fn assert_finalized(edge_list: &EdgeList, center: u32, initial_threshold: f32) {
        assert_eq!(edge_list.center, center);
        assert!(edge_list.sorted);
        assert!(edge_list.threshold <= initial_threshold);
        assert!(edge_list
            .edges
            .iter()
            .all(|edge| edge.distance <= edge_list.threshold));
        assert!(edge_list.edges.windows(2).all(|pair| {
            (pair[0].distance.to_bits(), pair[0].u, pair[0].v)
                <= (pair[1].distance.to_bits(), pair[1].u, pair[1].v)
        }));
    }

    #[test]
    fn trivial_size_is_contractible_at_zero() {
        let mut edge_list = EdgeList::from_points(&[0.0, 0.0], 1, 2, f32::INFINITY, false);
        peel(&mut edge_list, false);
        assert_eq!(edge_list.threshold, 0.0);
        assert!(edge_list.edges.is_empty());
        assert!(edge_list.sorted);
    }

    #[test]
    fn peel_never_increases_radius_or_changes_center() {
        let mut points = Vec::new();
        for x in 0..3 {
            for y in 0..3 {
                points.push(x as f32);
                points.push(y as f32);
            }
        }
        let mut edge_list = EdgeList::from_points(&points, 9, 2, f32::INFINITY, false);
        let center = edge_list.center;
        let initial_threshold = edge_list.threshold;
        peel(&mut edge_list, false);
        assert!(edge_list.threshold.is_finite() && edge_list.threshold >= 0.0);
        assert_finalized(&edge_list, center, initial_threshold);
    }

    #[test]
    fn peel_must_not_skip_cycle_inside_center_distance_gap() {
        // 5-point kite plus center P. The H1 bar (1.1, 1.5) lies entirely
        // inside the center-distance gap (1.05, 1.5).
        let lower = [1.5, 1.05, 1.0, 1.05, 1.5, 1.0, 1.5, 1.1, 1.5, 1.0];
        let matrix = square_from_lower_tri(5, &lower);
        let mut edge_list = EdgeList::from_distance_matrix(&matrix, 5, 1.5);
        let center = edge_list.center;
        peel(&mut edge_list, false);

        assert_eq!(center, 0);
        assert!(
            edge_list.threshold >= 1.5 - 1e-6,
            "unsound truncation: {} would erase the H1 bar (1.1, 1.5)",
            edge_list.threshold
        );
        assert_finalized(&edge_list, center, 1.5);
    }

    #[test]
    fn dense_cluster_can_reduce_below_enclosing_radius() {
        let points = [
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05, 0.2, 0.05, 0.05, 0.2,
        ];
        let n = points.len() / 2;
        let mut edge_list = EdgeList::from_points(&points, n, 2, f32::INFINITY, false);
        let center = edge_list.center;
        let initial_threshold = edge_list.threshold;
        peel(&mut edge_list, false);

        assert!(edge_list.threshold.is_finite() && edge_list.threshold >= 0.0);
        assert_finalized(&edge_list, center, initial_threshold);
    }
}
