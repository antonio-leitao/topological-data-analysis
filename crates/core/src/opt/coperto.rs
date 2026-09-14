//! Complete-linkage quotient and coning for the `log(3)` VR approximation.
//!
//! Pairwise complete linkage is deterministic under `(distance, u, v)` order.
//! A root pair is emitted on its first cross edge; symmetric neighbour maps
//! ensure each later cone edge is emitted once.

use std::collections::hash_map::Entry;

use crate::preprocess::edgelist::{uf_find, Edge, EdgeList};
use crate::utils::fxhash::FxHashMap;

type FxMap = FxHashMap<u32, u32>;

/// Replace the edge set with the coned quotient skeleton.
pub(crate) fn cone_in_place(edge_list: &mut EdgeList, parallel: bool) {
    let n = edge_list.n;
    if n <= 1 || edge_list.edges.is_empty() {
        return;
    }

    edge_list.sort(parallel);
    let input = std::mem::take(&mut edge_list.edges);

    let mut parent: Vec<u32> = (0..n as u32).collect();
    let mut size: Vec<u32> = vec![1u32; n];
    // Symmetric cross-edge counts between active roots.
    let mut nbr: Vec<FxMap> = (0..n).map(|_| FxMap::default()).collect();

    let mut out: Vec<Edge> = Vec::with_capacity(n);

    for edge in &input {
        let a = uf_find(&mut parent, edge.u);
        let b = uf_find(&mut parent, edge.v);
        if a == b {
            continue; // intra-cluster: nothing to record.
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
