//! Complete-linkage quotient and coning for the `log(3)` VR approximation.
//!
//! Pairwise complete linkage is deterministic under `(distance, u, v)` order.
//! A root pair is emitted on its first cross edge; symmetric neighbour maps
//! ensure each later cone edge is emitted once.

use std::collections::hash_map::Entry;

use rayon::prelude::*;

use crate::preprocess::edgelist::{band_edges, scan_points, uf_find, Edge, EdgeList, SpatialPoints};
use crate::utils::fxhash::FxHashMap;

type FxMap = FxHashMap<u32, u32>;

/// Replace the edge set with the coned quotient skeleton. Returns the vertices
/// in merge order, see [`Cone::finish`].
pub(crate) fn cone_in_place(edge_list: &mut EdgeList, parallel: bool) -> Vec<u32> {
    let n = edge_list.n;
    if n <= 1 || edge_list.edges.is_empty() {
        return (0..n as u32).collect();
    }

    edge_list.sort(parallel);
    let input = std::mem::take(&mut edge_list.edges);
    let mut cone = Cone::new(n);
    cone.extend(&input);
    drop(input);

    let (edges, order) = cone.finish();
    edge_list.edges = edges;
    edge_list.sorted = false;
    order
}

/// Edges a streamed band may hold, about 0.8 GB.
const STREAM_BAND_EDGES: u64 = 1 << 26;

/// [`cone_in_place`] on a point cloud's edge list, without building that list:
/// the edges arrive in distance bands of at most [`STREAM_BAND_EDGES`], each
/// sorted and consumed before the next is generated. The result is identical,
/// including the merge order; memory no longer grows with the number of pairs.
pub(crate) fn cone_points(
    data: &[f32],
    n: usize,
    d: usize,
    threshold: f32,
    parallel: bool,
) -> (EdgeList, Vec<u32>) {
    cone_points_in_bands(data, n, d, threshold, parallel, STREAM_BAND_EDGES)
}

fn cone_points_in_bands(
    data: &[f32],
    n: usize,
    d: usize,
    threshold: f32,
    parallel: bool,
    band_edges_max: u64,
) -> (EdgeList, Vec<u32>) {
    let scan = scan_points(data, n, d, threshold, parallel);
    let mut cone = Cone::new(n);
    if n >= 2 {
        let points = SpatialPoints::new(data, n, d);
        // Every pair within the cutoff has a bin at most the cutoff's.
        let last = scan.bin(scan.threshold);
        let mut start = 0;
        while start <= last {
            let mut end = start;
            let mut count = 0u64;
            while end <= last && (end == start || count + scan.bins[end] <= band_edges_max) {
                count += scan.bins[end];
                end += 1;
            }
            let mut band = band_edges(&points, &scan, start..end, parallel);
            if parallel {
                band.par_sort_unstable_by_key(Edge::sort_key);
            } else {
                band.sort_unstable_by_key(Edge::sort_key);
            }
            cone.extend(&band);
            start = end;
        }
    }
    let (edges, order) = cone.finish();
    let edge_list = EdgeList {
        n,
        edges,
        center: scan.center,
        threshold: scan.threshold,
        sorted: false,
    };
    (edge_list, order)
}

/// Complete-linkage contraction, fed one edge at a time in `(distance, u, v)`
/// order. Nothing here depends on the edges still to come, so the input can
/// be streamed instead of stored.
pub(crate) struct Cone {
    parent: Vec<u32>,
    size: Vec<u32>,
    /// Symmetric cross-edge counts between active roots.
    nbr: Vec<FxMap>,
    out: Vec<Edge>,
    /// Each cluster's members as a linked list, winner's members first.
    next: Vec<u32>,
    tail: Vec<u32>,
}

impl Cone {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            size: vec![1u32; n],
            nbr: (0..n).map(|_| FxMap::default()).collect(),
            out: Vec::with_capacity(n),
            next: vec![u32::MAX; n],
            tail: (0..n as u32).collect(),
        }
    }

    /// Feed the next edges, continuing the `(distance, u, v)` order. The state
    /// is unpacked once per call, so a slice runs as one tight loop.
    pub(crate) fn extend(&mut self, edges: &[Edge]) {
        let Self {
            parent,
            size,
            nbr,
            out,
            next,
            tail,
        } = self;
        for edge in edges {
            Self::push(parent, size, nbr, out, next, tail, edge);
        }
    }

    #[inline(always)]
    fn push(
        parent: &mut [u32],
        size: &mut [u32],
        nbr: &mut [FxMap],
        out: &mut Vec<Edge>,
        next: &mut [u32],
        tail: &mut [u32],
        edge: &Edge,
    ) {
        let a = uf_find(parent, edge.u);
        let b = uf_find(parent, edge.v);
        if a == b {
            return; // intra-cluster: nothing to record.
        }
        let ai = a as usize;
        let bi = b as usize;
        let scale = edge.distance;

        let count = {
            let e = nbr[ai].entry(b).or_insert(0);
            *e += 1;
            *e
        };
        *nbr[bi].entry(a).or_insert(0) += 1;

        if count == 1 {
            out.push(Edge::canonical(a, b, scale));
        }

        // Contract when every cross pair has appeared.
        if count as u64 == size[ai] as u64 * size[bi] as u64 {
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
                        out.push(Edge::canonical(w, nb, scale));
                    }
                }
                let nm = &mut nbr[nb as usize];
                nm.remove(&l);
                *nm.entry(w).or_insert(0) += cnt;
            }
            nbr[wi].remove(&l);

            size[wi] += size[li];
            parent[li] = w;
            next[tail[wi] as usize] = l;
            tail[wi] = tail[li];
        }
    }

    /// The skeleton's edges, unsorted, and every vertex in merge order: each
    /// final cluster's members contiguous, and inside it each merged cluster
    /// contiguous. Cone edges join clusters that end up merged, so numbering
    /// vertices in this order keeps every vertex near its neighbours.
    pub(crate) fn finish(self) -> (Vec<Edge>, Vec<u32>) {
        let Self {
            parent, next, out, ..
        } = self;
        let mut order = Vec::with_capacity(parent.len());
        for (r, &p) in parent.iter().enumerate() {
            if p as usize != r {
                continue;
            }
            let mut v = r as u32;
            while v != u32::MAX {
                order.push(v);
                v = next[v as usize];
            }
        }
        debug_assert_eq!(order.len(), parent.len());
        (out, order)
    }
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

    fn edge_set(edge_list: &EdgeList) -> Vec<(u32, u32, u32)> {
        let mut v: Vec<_> = edge_list
            .edges
            .iter()
            .map(|e| (e.u, e.v, e.distance.to_bits()))
            .collect();
        v.sort_unstable();
        v
    }

    /// The merge order lists every vertex once, and each final cluster's
    /// members contiguously.
    #[test]
    fn merge_order_is_a_contiguous_permutation() {
        let n = 60;
        let points: Vec<f32> = (0..n * 2)
            .map(|i| ((i as f32) * 0.618).sin() * 3.0 + (i % 7) as f32)
            .collect();
        for threshold in [0.5, 2.0, f32::INFINITY] {
            let mut e = EdgeList::from_points(&points, n, 2, threshold, false);
            e.sort(false);
            let mut cone = Cone::new(n);
            cone.extend(&e.edges);
            let roots: Vec<u32> = (0..n as u32)
                .map(|v| uf_find(&mut cone.parent, v))
                .collect();
            let (_, order) = cone.finish();
            let mut sorted = order.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..n as u32).collect::<Vec<_>>());
            // A root never reappears after its run of members ends.
            let mut seen = HashSet::new();
            for pair in order.windows(2) {
                let (a, b) = (roots[pair[0] as usize], roots[pair[1] as usize]);
                if a != b {
                    assert!(seen.insert(a), "cluster {a} split in the order");
                }
            }
        }
    }

    fn cloud(n: usize, d: usize, seed: u64, lattice: bool) -> Vec<f32> {
        let mut s = seed;
        let mut rng = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        (0..n * d)
            .map(|_| {
                if lattice {
                    (rng() % 6) as f32 // many exact ties
                } else {
                    (rng() >> 40) as f32 / 16_777_216.0 * 4.0 + (rng() % 3) as f32 * 5.0
                }
            })
            .collect()
    }

    /// Streaming the edges in bands gives exactly what building them does:
    /// same skeleton edges, merge order, cutoff and centre, across band
    /// sizes, thresholds, ties, clustered data and the parallel paths.
    #[test]
    fn streamed_cone_matches_materialised() {
        for &(n, d, seed, lattice) in &[
            (300usize, 3usize, 1u64, false),
            (257, 2, 2, true),
            (150, 7, 3, false),
            (40, 1, 4, true),
            (17, 3, 5, false),
            (1, 3, 6, false),
        ] {
            let data = cloud(n, d, seed, lattice);
            for threshold in [f32::INFINITY, 4.0, 0.7] {
                for parallel in [false, true] {
                    let mut built = EdgeList::from_points(&data, n, d, threshold, parallel);
                    let order = cone_in_place(&mut built, parallel);
                    let expect = (edge_set(&built), order, built.threshold.to_bits(), built.center);
                    // One edge per band is a band per occupied bin: thorough, but
                    // only affordable on small clouds.
                    let tiny: &[u64] = if n <= 60 { &[1] } else { &[] };
                    for &band in tiny.iter().chain(&[97, 5_000, u64::MAX]) {
                        let (streamed, order) =
                            cone_points_in_bands(&data, n, d, threshold, parallel, band);
                        let got = (edge_set(&streamed), order, streamed.threshold.to_bits(), streamed.center);
                        assert_eq!(got, expect, "n={n} d={d} lattice={lattice} t={threshold} band={band} par={parallel}");
                    }
                }
            }
        }
    }

    #[test]
    fn trivial_input_is_a_noop() {
        let mut e = EdgeList::from_points(&[0.0, 0.0], 1, 2, f32::INFINITY, false);
        cone_in_place(&mut e, false);
        assert!(e.edges.is_empty());
    }

    #[test]
    fn complete_graph_contracts_to_a_tree() {
        let n = 4;
        let matrix = square_from_lower_tri(n, &[1.0; 6]);
        let mut e = EdgeList::from_distance_matrix(&matrix, n, f32::INFINITY).unwrap();
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
            .unwrap()
            .edges
            .len();

        let mut quot = EdgeList::from_distance_matrix(&matrix, n, f32::INFINITY).unwrap();
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
        let mut from_matrix = EdgeList::from_distance_matrix(&matrix, 4, f32::INFINITY).unwrap();
        cone_in_place(&mut from_points, false);
        cone_in_place(&mut from_matrix, false);

        assert_eq!(edge_set(&from_points), edge_set(&from_matrix));
    }

    #[test]
    fn presorted_and_unsorted_inputs_agree() {
        let s = std::f32::consts::SQRT_2;
        let matrix = square_from_lower_tri(4, &[1.0, s, 1.0, 1.0, s, 1.0]);

        let mut unsorted = EdgeList::from_distance_matrix(&matrix, 4, f32::INFINITY).unwrap();
        assert!(!unsorted.sorted);

        let mut presorted = EdgeList::from_distance_matrix(&matrix, 4, f32::INFINITY).unwrap();
        presorted
            .edges
            .sort_unstable_by_key(|e| (e.distance.to_bits(), e.u, e.v));
        presorted.sorted = true;

        cone_in_place(&mut unsorted, false);
        cone_in_place(&mut presorted, false);
        assert_eq!(edge_set(&unsorted), edge_set(&presorted));
    }
}
