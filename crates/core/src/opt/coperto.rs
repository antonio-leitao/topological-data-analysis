//! Quotient-cover coning over a sparse weighted edge list (the `coperto` step).
//!
//! This realises the quotient-cover approximation of the Vietoris–Rips
//! filtration: complete-linkage clusters are detected from the ascending edge
//! stream, and each completed merge is emitted as a Kerber–Schreiber cone on the
//! 1-skeleton. The result is a smaller weighted graph whose flag completion is
//! `log 3`-interleaved with VR — an *approximation*, not the exact barcode
//! (contrast `opt::peel`, an exact strong-collapse truncation).
//!
//! ## Linkage choice
//!
//! We use *standard pairwise* complete linkage (merge `A,B` the instant every
//! cross pair lies within the current scale), not the conservative variant from
//! the paper. Conservative linkage exists only to make the partition a
//! well-defined functor under ties; persistence is invariant to tie-breaking, so
//! under the general-position assumption the two give identical barcodes.
//! Pairwise linkage is simpler and faster, and the deterministic
//! `(distance, u, v)` edge order plus deterministic winner selection pins down a
//! single output.
//!
//! ## Why there is no dense matrix and no dedup pass
//!
//! The previous implementation kept an `n×n` scratch matrix purely to dedup born
//! edges to their minimum birth, then scanned it to recover the sparse result.
//! Two invariants make all of that unnecessary:
//!
//!   1. Edge scales are processed non-decreasingly, so the *first* time two
//!      clusters connect is that edge's minimum birth.
//!   2. For current roots `x, y`, `nbr[x]` contains `y` **iff** the pair `{x, y}`
//!      has already been emitted (the maps are kept symmetric, and every
//!      contraction redirects the loser's references to the winner).
//!
//! Hence every born edge is pushed exactly once: the *root edge* on the first
//! cross edge (`count == 1`), the *cone edge* on the `Vacant` arm of the merge
//! drain. Those two sites are mutually exclusive for any pair (occupancy of the
//! `nbr` entry rules the other out), so the output carries no duplicate
//! endpoints and needs no `HashSet`/min scratch to enforce it.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::preprocess::edgelist::{Edge, EdgeList};
use rayon::prelude::*;

/// FxHash-style hasher specialised for the `u32` cluster-root keys. No dependency.
#[derive(Default)]
struct FxU32Hasher(u64);

const SEED: u64 = 0x517c_c1b7_2722_0a95;

impl Hasher for FxU32Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(SEED);
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Keys are u32 (write_u32); this exists only to satisfy the trait.
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(SEED);
        }
    }
}

type FxMap = HashMap<u32, u32, BuildHasherDefault<FxU32Hasher>>;

/// Union-find lookup with path halving.
#[inline]
fn find(parent: &mut [u32], mut i: u32) -> u32 {
    while parent[i as usize] != i {
        let grand = parent[parent[i as usize] as usize];
        parent[i as usize] = grand;
        i = grand;
    }
    i
}

/// Push a canonical `u > v` edge. Callers guarantee `a != b`.
#[inline(always)]
fn push_edge(out: &mut Vec<Edge>, a: u32, b: u32, distance: f32) {
    let (u, v) = if a > b { (a, b) } else { (b, a) };
    out.push(Edge {
        u: u as u16,
        v: v as u16,
        distance,
    });
}

/// Rewrite `edge_list` in place into the 1-skeleton of the coned quotient
/// filtration. `n`, `center`, and `threshold` are preserved (every emitted edge
/// is born at a scale `<= threshold`); the edge set is replaced and the list is
/// marked unsorted.
pub(crate) fn cone_in_place(edge_list: &mut EdgeList, parallel: bool) {
    let n = edge_list.n;
    if n <= 1 || edge_list.edges.is_empty() {
        return;
    }

    // Ascending (distance, endpoints). Reuses peel's sort when it already ran.
    let mut input = std::mem::take(&mut edge_list.edges);
    if !edge_list.sorted {
        if parallel {
            input.par_sort_unstable_by_key(|edge| (edge.distance.to_bits(), edge.u, edge.v));
        } else {
            input.sort_unstable_by_key(|edge| (edge.distance.to_bits(), edge.u, edge.v));
        }
    }

    // Union-find over cluster roots; roots are original vertex ids in `0..n`.
    let mut parent: Vec<u32> = (0..n as u32).collect();
    let mut size: Vec<u32> = vec![1u32; n];
    // nbr[root] : neighbour-root -> cross-edge count, kept symmetric.
    let mut nbr: Vec<FxMap> = (0..n).map(|_| FxMap::default()).collect();

    // Output skeleton. The coned filtration is near-linear in n by construction,
    // so n is a tight starting estimate; growth handles adversarial inputs.
    let mut out: Vec<Edge> = Vec::with_capacity(n);

    for edge in &input {
        let a = find(&mut parent, edge.u as u32);
        let b = find(&mut parent, edge.v as u32);
        if a == b {
            continue; // intra-cluster: nothing to record.
        }
        let ai = a as usize;
        let bi = b as usize;
        let scale = edge.distance;

        // Cross-edge tally between the two active clusters (kept symmetric).
        let count = {
            let e = nbr[ai].entry(b).or_insert(0);
            *e += 1;
            *e
        };
        *nbr[bi].entry(a).or_insert(0) += 1;

        // First cross edge => the clusters' edge is born now, at its min scale.
        if count == 1 {
            push_edge(&mut out, a, b, scale);
        }

        // Complete linkage: all size[a]*size[b] cross pairs present => contract.
        if count as u64 == size[ai] as u64 * size[bi] as u64 {
            // Union by neighbour-map size: drain the smaller map into the larger.
            let (w, l) = if nbr[ai].len() <= nbr[bi].len() {
                (b, a)
            } else {
                (a, b)
            };
            let (wi, li) = (w as usize, l as usize);

            let loser_map = std::mem::take(&mut nbr[li]);
            for (nb, cnt) in loser_map {
                if nb == w {
                    continue; // the triggering edge; already recorded.
                }
                match nbr[wi].entry(nb) {
                    Entry::Occupied(mut o) => {
                        *o.get_mut() += cnt; // already adjacent: no new edge.
                    }
                    Entry::Vacant(slot) => {
                        slot.insert(cnt);
                        // New neighbour of the merged cluster => cone edge.
                        push_edge(&mut out, w, nb, scale);
                    }
                }
                // Redirect the neighbour's back-reference from loser to winner.
                let nm = &mut nbr[nb as usize];
                nm.remove(&l);
                *nm.entry(w).or_insert(0) += cnt;
            }
            nbr[wi].remove(&l);

            size[wi] += size[li];
            parent[li] = w;
        }
    }

    edge_list.edges = out;
    edge_list.sorted = false;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn square_from_lower_tri(n: usize, lower: &[f32]) -> Vec<f32> {
        assert_eq!(lower.len(), n * (n - 1) / 2);
        let mut m = vec![0.0f32; n * n];
        let mut k = 0;
        for i in 1..n {
            for j in 0..i {
                m[i * n + j] = lower[k];
                m[j * n + i] = lower[k];
                k += 1;
            }
        }
        m
    }

    /// Output must be canonical (u > v), duplicate-free, self-loop-free, and
    /// within the (possibly tightened) threshold.
    fn assert_clean(edge_list: &EdgeList) {
        let mut seen = HashSet::new();
        for e in &edge_list.edges {
            assert!(
                e.u > e.v,
                "non-canonical / self-loop edge ({}, {})",
                e.u,
                e.v
            );
            assert!(
                e.distance <= edge_list.threshold,
                "edge {} above threshold {}",
                e.distance,
                edge_list.threshold
            );
            assert!(seen.insert((e.u, e.v)), "duplicate edge ({}, {})", e.u, e.v);
        }
    }

    fn edge_set(edge_list: &EdgeList) -> Vec<(u16, u16, u32)> {
        let mut v: Vec<_> = edge_list
            .edges
            .iter()
            .map(|e| (e.u, e.v, e.distance.to_bits()))
            .collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn trivial_input_is_a_noop() {
        let mut e = EdgeList::from_points(&[0.0, 0.0], 1, 2, f32::INFINITY, false);
        cone_in_place(&mut e, false);
        assert!(e.edges.is_empty());
    }

    #[test]
    fn complete_graph_contracts_to_a_tree() {
        // K4, all distances 1 => the quotient collapses to a point; the coned
        // 1-skeleton is an (n-1)-edge tree, all born at scale 1.
        let n = 4;
        let matrix = square_from_lower_tri(n, &[1.0; 6]);
        let mut e = EdgeList::from_distance_matrix(&matrix, n, f32::INFINITY);
        cone_in_place(&mut e, false);
        assert_clean(&e);
        assert_eq!(e.edges.len(), n - 1);
        assert!(e.edges.iter().all(|edge| edge.distance == 1.0));
    }

    #[test]
    fn reduces_below_vietoris_rips_edge_count() {
        let s = std::f32::consts::SQRT_2;
        let n = 4;
        let matrix = square_from_lower_tri(n, &[1.0, s, 1.0, 1.0, s, 1.0]);
        let vr_edges = EdgeList::from_distance_matrix(&matrix, n, f32::INFINITY)
            .edges
            .len();

        let mut quot = EdgeList::from_distance_matrix(&matrix, n, f32::INFINITY);
        cone_in_place(&mut quot, false);
        assert_clean(&quot);
        assert!(quot.edges.len() < vr_edges);
    }

    #[test]
    fn point_and_matrix_inputs_agree() {
        let s = std::f32::consts::SQRT_2;
        let points = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let matrix = square_from_lower_tri(4, &[1.0, s, 1.0, 1.0, s, 1.0]);

        let mut from_points = EdgeList::from_points(&points, 4, 2, f32::INFINITY, false);
        let mut from_matrix = EdgeList::from_distance_matrix(&matrix, 4, f32::INFINITY);
        cone_in_place(&mut from_points, false);
        cone_in_place(&mut from_matrix, false);

        assert_eq!(edge_set(&from_points), edge_set(&from_matrix));
    }

    #[test]
    fn presorted_and_unsorted_inputs_agree() {
        let s = std::f32::consts::SQRT_2;
        let matrix = square_from_lower_tri(4, &[1.0, s, 1.0, 1.0, s, 1.0]);

        let mut unsorted = EdgeList::from_distance_matrix(&matrix, 4, f32::INFINITY);
        assert!(!unsorted.sorted);

        let mut presorted = EdgeList::from_distance_matrix(&matrix, 4, f32::INFINITY);
        presorted
            .edges
            .sort_unstable_by_key(|e| (e.distance.to_bits(), e.u, e.v));
        presorted.sorted = true;

        cone_in_place(&mut unsorted, false);
        cone_in_place(&mut presorted, false);
        assert_eq!(edge_set(&unsorted), edge_set(&presorted));
    }
}
