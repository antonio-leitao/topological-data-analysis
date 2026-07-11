//! Exact edge collapse of a flag filtration (the `collapse` step).
//!
//! [`collapse_in_place`] simplifies a weighted edge list while preserving the
//! persistent homology of its flag (Vietoris–Rips) filtration EXACTLY —
//! identical barcode (Boissonnat–Pritam "Edge Collapse and Persistence of
//! Flag Complexes"; Glisse–Pritam "Swap, Shift and Trim", the algorithm GUDHI
//! exposes as `reduce_graph`). Contrast `opt::peel`, which uses *vertex*
//! domination to certify a tail truncation: peel shrinks the filtration's
//! range, edge collapse thins its interior at every scale. Both are exact.
//!
//! ## Criterion
//!
//! Edge `e = (u, v)` with value `f(e)` is *dominated at time t* when some
//! common neighbor `z` is adjacent at scale <= t to every member of the
//! common neighborhood `N_t[e]` (over the retained edges with value <= t).
//! If `e` is dominated on `[f(e), T)`, its insertion can be delayed to `T`
//! (the SHIFT) without changing the persistence; if `T = ∞` the edge is
//! removed outright (the TRIM). An edge not dominated at its own scale keeps
//! its value.
//!
//! Discretization: the member set of `N_t[e]` changes only at appearance
//! times `t_w = max(f(u,w), f(v,w))`, and between events adjacencies among a
//! fixed member set only accumulate, so domination is monotone within an
//! interval. Checking at `f(e)` and at each event is sound and complete.
//!
//! ## Algorithm (after GUDHI's rewrite by Glisse)
//!
//! Edges are processed in decreasing value order over the current graph.
//! Per edge, one merge walk over the two id-sorted adjacency lists splits the
//! common neighbors into the active set (id-sorted) and a "later" event list,
//! which is heapified only if the edge survives its first domination check —
//! most edges never need it. While a dominator holds, each event costs one
//! adjacency lookup for the NEW member only (existing members stay valid by
//! monotonicity); a full dominator search re-runs only when the current
//! dominator breaks. Every adjacency list carries a `(v, -inf)` self-sentinel
//! so witness membership needs no special-casing.
//!
//! ## Order in the pipeline
//!
//! Runs LAST among the optional transforms (peel → quotient → collapse):
//! after peel because peel leaves fewer, already-sorted edges; after the
//! quotient because coperto's complete-linkage counting assumes the full
//! metric graph, while collapse is exact on ANY weighted graph's flag
//! filtration — including the coned quotient graph, so the log(3) guarantee
//! survives the composition.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::preprocess::edgelist::EdgeList;
use rayon::prelude::*;

/// Adjacency entry: (neighbor id, edge value), kept sorted by id. Each list
/// also holds a `(self, -inf)` sentinel.
type Nbr = (u16, f32);

/// Memory budget for the optional dense n×n value matrix (GUDHI's
/// `GUDHI_COLLAPSE_USE_DENSE_ARRAY`): O(1) adjacency-value lookups make a
/// FAILED domination candidate cost ~2 probes instead of an O(deg) merge
/// walk, which dominates the runtime on dense graphs. 256 MiB ⇒ n ≤ 8192;
/// larger inputs fall back to the merge walk (O(E) memory).
const DENSE_BUDGET_BYTES: usize = 1 << 28;

/// Is the id-sorted `active` set contained in `N_t[c]`? Linear merge walk
/// against the id-sorted adjacency of `c`; the self-sentinel makes `c`'s own
/// membership pass automatically. Degree precheck up front.
#[inline]
fn is_dominated_by(active: &[u16], adj_c: &[Nbr], t: f32) -> bool {
    if adj_c.len() < active.len() {
        return false;
    }
    let mut ci = 0usize;
    'members: for &ve in active {
        while ci < adj_c.len() {
            let (vc, fc) = adj_c[ci];
            if vc < ve {
                ci += 1;
                continue;
            }
            if vc > ve || fc > t {
                return false;
            }
            ci += 1;
            continue 'members;
        }
        return false;
    }
    true
}

/// First dominator of the `active` set at scale `t`, or None. With the dense
/// matrix, each candidate is O(|active|) with ~O(1) expected early exit; the
/// fallback needs `active` id-sorted for the merge walk.
#[inline]
fn find_dominator(
    adj: &[Vec<Nbr>],
    dense: &[f32],
    n: usize,
    active: &[u16],
    t: f32,
) -> Option<u16> {
    if dense.is_empty() {
        active
            .iter()
            .copied()
            .find(|&c| is_dominated_by(active, &adj[c as usize], t))
    } else {
        active.iter().copied().find(|&c| {
            // Degree precheck, then O(1) probes (self handled by the -inf
            // sentinel on the diagonal).
            adj[c as usize].len() >= active.len() && {
                let row = (c as usize) * n;
                active.iter().all(|&w| dense[row + w as usize] <= t)
            }
        })
    }
}

#[inline]
fn remove_neighbor(list: &mut Vec<Nbr>, w: u16) {
    let pos = list
        .binary_search_by_key(&w, |&(x, _)| x)
        .expect("removing an edge that is not present");
    list.remove(pos);
}

#[inline]
fn set_neighbor_value(list: &mut [Nbr], w: u16, value: f32) {
    let pos = list
        .binary_search_by_key(&w, |&(x, _)| x)
        .expect("delaying an edge that is not present");
    list[pos].1 = value;
}

/// Collapse the edge list in place: dominated edges are removed or delayed
/// (their value increased to the first time they become critical). `n`,
/// `center`, and `threshold` are preserved; the output is left in ascending
/// `(distance, u, v)` order with `sorted` true.
pub(crate) fn collapse_in_place(edge_list: &mut EdgeList, parallel: bool) {
    let use_dense = edge_list
        .n
        .checked_mul(edge_list.n)
        .and_then(|nn| nn.checked_mul(4))
        .is_some_and(|bytes| bytes <= DENSE_BUDGET_BYTES);
    collapse_impl(edge_list, parallel, use_dense);
}

/// Both paths produce identical output; `use_dense` only trades memory for
/// O(1) adjacency probes (tests exercise both).
fn collapse_impl(edge_list: &mut EdgeList, parallel: bool, use_dense: bool) {
    let n = edge_list.n;
    if n < 3 || edge_list.edges.len() < 3 {
        return; // nothing can be dominated
    }

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

    // Id-sorted closed adjacency (self-sentinel at -inf), mutated in place as
    // edges are removed or delayed.
    let mut adj: Vec<Vec<Nbr>> = vec![Vec::new(); n];
    for edge in &edge_list.edges {
        adj[edge.u as usize].push((edge.v, edge.distance));
        adj[edge.v as usize].push((edge.u, edge.distance));
    }
    for (v, list) in adj.iter_mut().enumerate() {
        list.push((v as u16, f32::NEG_INFINITY));
    }
    if parallel {
        adj.par_iter_mut()
            .for_each(|list| list.sort_unstable_by_key(|&(w, _)| w));
    } else {
        for list in &mut adj {
            list.sort_unstable_by_key(|&(w, _)| w);
        }
    }

    // Optional dense value matrix (see DENSE_BUDGET_BYTES). Missing edges are
    // +inf; the diagonal is the -inf self-sentinel.
    let mut dense: Vec<f32> = Vec::new();
    if use_dense {
        dense = vec![f32::INFINITY; n * n];
        for edge in &edge_list.edges {
            let (ui, vi) = (edge.u as usize, edge.v as usize);
            dense[ui * n + vi] = edge.distance;
            dense[vi * n + ui] = edge.distance;
        }
        for v in 0..n {
            dense[v * n + v] = f32::NEG_INFINITY;
        }
    }

    let mut keep = vec![true; edge_list.edges.len()];
    let mut any_shift = false;
    // Reused scratch: active members (id-sorted in the fallback path, where
    // the merge walk needs it) and future appearance events.
    let mut active: Vec<u16> = Vec::new();
    let mut later: Vec<Reverse<(u32, u16)>> = Vec::new();

    // Decreasing value order; ties in deterministic (u, v) order.
    for idx in (0..edge_list.edges.len()).rev() {
        let edge = edge_list.edges[idx];
        let (u, v) = (edge.u, edge.v);
        let (ui, vi) = (u as usize, v as usize);
        let start_time = edge.distance;

        // One merge walk: split common neighbors into active-now vs later.
        active.clear();
        later.clear();
        {
            let (a, b) = (&adj[ui], &adj[vi]);
            let (mut i, mut j) = (0usize, 0usize);
            while i < a.len() && j < b.len() {
                let (wa, wb) = (a[i].0, b[j].0);
                if wa < wb {
                    i += 1;
                } else if wa > wb {
                    j += 1;
                } else {
                    if wa != u && wa != v {
                        let t_w = a[i].1.max(b[j].1);
                        if t_w <= start_time {
                            active.push(wa); // id order comes from the merge
                        } else {
                            later.push(Reverse((t_w.to_bits(), wa)));
                        }
                    }
                    i += 1;
                    j += 1;
                }
            }
        }

        let mut time = start_time;
        let mut dead = false;

        if !active.is_empty() {
            // Lazily heapified: edges that are never dominated, or dominated
            // through infinity with no future events, never pay for it.
            let mut heap: Option<BinaryHeap<Reverse<(u32, u16)>>> = None;

            'outer: while let Some(dom) = find_dominator(&adj, &dense, n, &active, time) {
                let adj_dom = &adj[dom as usize];
                // Advance through events while `dom` keeps dominating: each
                // NEW member costs one lookup; existing members stay valid.
                loop {
                    let h = heap.get_or_insert_with(|| {
                        BinaryHeap::from(std::mem::take(&mut later))
                    });
                    let Some(&Reverse((t_bits, _))) = h.peek() else {
                        dead = true; // dominated on [f(e), ∞): trim
                        break 'outer;
                    };
                    time = f32::from_bits(t_bits);
                    let mut still_dominated = true;
                    while let Some(&Reverse((tb, w))) = h.peek() {
                        if tb != t_bits {
                            break;
                        }
                        h.pop();
                        let adjacent = if use_dense {
                            dense[(dom as usize) * n + w as usize] <= time
                        } else {
                            matches!(
                                adj_dom.binary_search_by_key(&w, |&(x, _)| x),
                                Ok(pos) if adj_dom[pos].1 <= time
                            )
                        };
                        if !adjacent {
                            still_dominated = false;
                        }
                        if use_dense {
                            active.push(w); // order irrelevant for probes
                        } else {
                            let pos = active.binary_search(&w).unwrap_err();
                            active.insert(pos, w);
                        }
                    }
                    if !still_dominated {
                        continue 'outer; // full re-search at the new time
                    }
                }
            }

            if let Some(h) = heap {
                later = h.into_vec(); // reclaim the allocation
                later.clear();
            }
        }

        if dead {
            keep[idx] = false;
            remove_neighbor(&mut adj[ui], v);
            remove_neighbor(&mut adj[vi], u);
            if use_dense {
                dense[ui * n + vi] = f32::INFINITY;
                dense[vi * n + ui] = f32::INFINITY;
            }
        } else if time.to_bits() != start_time.to_bits() {
            // SHIFT: dominated on [f(e), time) — delay the edge to `time`.
            any_shift = true;
            edge_list.edges[idx].distance = time;
            set_neighbor_value(&mut adj[ui], v, time);
            set_neighbor_value(&mut adj[vi], u, time);
            if use_dense {
                dense[ui * n + vi] = time;
                dense[vi * n + ui] = time;
            }
        }
    }

    let mut i = 0usize;
    edge_list.edges.retain(|_| {
        let k = keep[i];
        i += 1;
        k
    });

    // Shifts break the ascending order; restore it.
    if any_shift {
        if parallel {
            edge_list
                .edges
                .par_sort_unstable_by_key(|edge| (edge.distance.to_bits(), edge.u, edge.v));
        } else {
            edge_list
                .edges
                .sort_unstable_by_key(|edge| (edge.distance.to_bits(), edge.u, edge.v));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::edgelist::{Edge, EdgeList};

    fn edge_list(n: usize, edges: Vec<Edge>) -> EdgeList {
        EdgeList {
            n,
            edges,
            center: 0,
            threshold: f32::INFINITY,
            sorted: false,
        }
    }

    fn assert_clean(edges: &EdgeList) {
        assert!(edges.sorted);
        assert!(edges.edges.windows(2).all(|p| {
            (p[0].distance.to_bits(), p[0].u, p[0].v) <= (p[1].distance.to_bits(), p[1].u, p[1].v)
        }));
        assert!(edges.edges.iter().all(|e| e.u > e.v));
    }

    #[test]
    fn trivial_inputs_are_noops() {
        let mut e = edge_list(2, vec![Edge { u: 1, v: 0, distance: 1.0 }]);
        collapse_in_place(&mut e, false);
        assert_eq!(e.edges.len(), 1);
    }

    /// The dense-matrix and merge-walk paths must produce identical output
    /// (same removals, same shifted values) on deterministic random graphs.
    #[test]
    fn dense_and_sparse_paths_agree() {
        let mut s: u64 = 0xc0_ffee_1234_5678;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 16_777_216.0
        };
        for n in [6usize, 10, 16, 24] {
            for &thr in &[0.4f32, 0.7, 1.0] {
                let mut edges = Vec::new();
                for u in 1..n as u16 {
                    for v in 0..u {
                        let d = rng();
                        if d <= thr {
                            edges.push(Edge { u, v, distance: d });
                        }
                    }
                }
                let mut with_dense = edge_list(n, edges.clone());
                let mut without_dense = edge_list(n, edges);
                collapse_impl(&mut with_dense, false, true);
                collapse_impl(&mut without_dense, false, false);
                let key = |e: &EdgeList| -> Vec<(u16, u16, u32)> {
                    e.edges
                        .iter()
                        .map(|x| (x.u, x.v, x.distance.to_bits()))
                        .collect()
                };
                assert_eq!(key(&with_dense), key(&without_dense), "n={n} thr={thr}");
            }
        }
    }

    /// K4 at a single scale collapses to a spanning tree: every removal needs
    /// a dominating common neighbor, and the process stops exactly when the
    /// graph is triangle-free.
    #[test]
    fn complete_graph_collapses_to_tree() {
        let edges: Vec<Edge> = (1..4u16)
            .flat_map(|u| (0..u).map(move |v| Edge { u, v, distance: 1.0 }))
            .collect();
        let mut e = edge_list(4, edges);
        collapse_in_place(&mut e, false);
        assert_clean(&e);
        assert_eq!(e.edges.len(), 3, "K4 must collapse to a 3-edge tree");
        assert!(e.edges.iter().all(|edge| edge.distance == 1.0));
    }

    /// Unit square with both diagonals: the first diagonal processed is
    /// dominated through infinity (trimmed); the second has no witness at its
    /// own scale and must keep its value so the H1 bar [1, √2) dies on time.
    #[test]
    fn square_keeps_one_diagonal() {
        let s = std::f32::consts::SQRT_2;
        let edges = vec![
            Edge { u: 1, v: 0, distance: 1.0 },
            Edge { u: 2, v: 1, distance: 1.0 },
            Edge { u: 3, v: 2, distance: 1.0 },
            Edge { u: 3, v: 0, distance: 1.0 },
            Edge { u: 2, v: 0, distance: s },
            Edge { u: 3, v: 1, distance: s },
        ];
        let mut e = edge_list(4, edges);
        collapse_in_place(&mut e, false);
        assert_clean(&e);
        assert_eq!(e.edges.len(), 5, "exactly one diagonal must be removed");
        let diagonals = e.edges.iter().filter(|edge| edge.distance == s).count();
        assert_eq!(diagonals, 1);
    }

    /// An edge whose common neighbors appear only later must stay at its own
    /// value: on [f(e), t_w) its neighborhood is just its endpoints.
    #[test]
    fn late_common_neighbors_do_not_dominate() {
        // Triangle where (1,0) is the SHORTEST edge: vertex 2 becomes a
        // common neighbor only at scale 2.0 > 1.0.
        let edges = vec![
            Edge { u: 1, v: 0, distance: 1.0 },
            Edge { u: 2, v: 0, distance: 2.0 },
            Edge { u: 2, v: 1, distance: 2.0 },
        ];
        let mut e = edge_list(3, edges);
        collapse_in_place(&mut e, false);
        // (1,0) kept at 1.0 (its H0 merge is real); the longest edge (2,1) is
        // dominated by vertex 0 from its own scale onwards and is trimmed.
        assert_eq!(e.edges.len(), 2);
        assert!(e.edges.iter().any(|edge| edge.distance == 1.0));
    }

    /// A pentagon with two chords from vertex 0: neither chord has a valid
    /// witness (their two common members are never adjacent), so both must
    /// keep their exact values — no spurious trim or shift.
    #[test]
    fn pentagon_chords_are_critical() {
        let edges = vec![
            Edge { u: 1, v: 0, distance: 1.0 },
            Edge { u: 2, v: 1, distance: 1.0 },
            Edge { u: 3, v: 2, distance: 1.0 },
            Edge { u: 4, v: 3, distance: 1.0 },
            Edge { u: 4, v: 0, distance: 1.0 },
            Edge { u: 2, v: 0, distance: 2.0 },
            Edge { u: 3, v: 0, distance: 2.0 },
        ];
        let mut e = edge_list(5, edges);
        collapse_in_place(&mut e, false);
        assert_clean(&e);
        assert_eq!(e.edges.len(), 7, "nothing is dominated in this graph");
        assert!(e.edges.iter().all(|edge| edge.distance == 1.0 || edge.distance == 2.0));
    }
}
