//! Exact edge collapse for a weighted flag filtration.
//!
//! An edge dominated on `[f(e), T)` is shifted to `T`, or removed when
//! `T = infinity`. Domination changes only when a common neighbour appears, so
//! checks at `f(e)` and those event times are complete. This step runs after
//! peel and quotient; exact collapse preserves the quotient guarantee.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::preprocess::edgelist::EdgeList;
use rayon::prelude::*;

/// Id-sorted adjacency entry; each row includes `(self, -inf)`.
type Nbr = (u32, f32);

/// Memory cap for constant-time adjacency lookups; larger inputs use merge walks.
const DENSE_BUDGET_BYTES: usize = 1 << 28;

/// Test `active` containment in `N_t[c]` by merge walk.
#[inline]
fn is_dominated_by(active: &[u32], adj_c: &[Nbr], t: f32) -> bool {
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

/// First dominator of `active` at scale `t`.
#[inline]
fn find_dominator(
    adj: &[Vec<Nbr>],
    dense: &[f32],
    n: usize,
    active: &[u32],
    t: f32,
) -> Option<u32> {
    if dense.is_empty() {
        active
            .iter()
            .copied()
            .find(|&c| is_dominated_by(active, &adj[c as usize], t))
    } else {
        active.iter().copied().find(|&c| {
            adj[c as usize].len() >= active.len() && {
                let row = (c as usize) * n;
                active.iter().all(|&w| dense[row + w as usize] <= t)
            }
        })
    }
}

#[inline]
fn remove_neighbor(list: &mut Vec<Nbr>, w: u32) {
    let pos = list
        .binary_search_by_key(&w, |&(x, _)| x)
        .expect("removing an edge that is not present");
    list.remove(pos);
}

#[inline]
fn set_neighbor_value(list: &mut [Nbr], w: u32, value: f32) {
    let pos = list
        .binary_search_by_key(&w, |&(x, _)| x)
        .expect("delaying an edge that is not present");
    list[pos].1 = value;
}

/// Remove or delay dominated edges, leaving sorted output.
pub(crate) fn collapse_in_place(edge_list: &mut EdgeList, parallel: bool) {
    let use_dense = edge_list
        .n
        .checked_mul(edge_list.n)
        .and_then(|nn| nn.checked_mul(4))
        .is_some_and(|bytes| bytes <= DENSE_BUDGET_BYTES);
    collapse_impl(edge_list, parallel, use_dense);
}

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

    // Closed adjacency is mutated as edges are removed or delayed.
    let mut adj: Vec<Vec<Nbr>> = vec![Vec::new(); n];
    for edge in &edge_list.edges {
        adj[edge.u as usize].push((edge.v, edge.distance));
        adj[edge.v as usize].push((edge.u, edge.distance));
    }
    for (v, list) in adj.iter_mut().enumerate() {
        list.push((v as u32, f32::NEG_INFINITY));
    }
    if parallel {
        adj.par_iter_mut()
            .for_each(|list| list.sort_unstable_by_key(|&(w, _)| w));
    } else {
        for list in &mut adj {
            list.sort_unstable_by_key(|&(w, _)| w);
        }
    }

    // Missing dense entries are `+inf`; diagonal sentinels are `-inf`.
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
    let mut active: Vec<u32> = Vec::new();
    let mut later: Vec<Reverse<(u32, u32)>> = Vec::new();

    for idx in (0..edge_list.edges.len()).rev() {
        let edge = edge_list.edges[idx];
        let (u, v) = (edge.u, edge.v);
        let (ui, vi) = (u as usize, v as usize);
        let start_time = edge.distance;

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
            let mut heap: Option<BinaryHeap<Reverse<(u32, u32)>>> = None;

            'outer: while let Some(dom) = find_dominator(&adj, &dense, n, &active, time) {
                let adj_dom = &adj[dom as usize];
                loop {
                    let h =
                        heap.get_or_insert_with(|| BinaryHeap::from(std::mem::take(&mut later)));
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
        let mut e = edge_list(
            2,
            vec![Edge {
                u: 1,
                v: 0,
                distance: 1.0,
            }],
        );
        collapse_in_place(&mut e, false);
        assert_eq!(e.edges.len(), 1);
    }

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
                for u in 1..n as u32 {
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
                let key = |e: &EdgeList| -> Vec<(u32, u32, u32)> {
                    e.edges
                        .iter()
                        .map(|x| (x.u, x.v, x.distance.to_bits()))
                        .collect()
                };
                assert_eq!(key(&with_dense), key(&without_dense), "n={n} thr={thr}");
            }
        }
    }

    #[test]
    fn complete_graph_collapses_to_tree() {
        let edges: Vec<Edge> = (1..4u32)
            .flat_map(|u| {
                (0..u).map(move |v| Edge {
                    u,
                    v,
                    distance: 1.0,
                })
            })
            .collect();
        let mut e = edge_list(4, edges);
        collapse_in_place(&mut e, false);
        assert_clean(&e);
        assert_eq!(e.edges.len(), 3, "K4 must collapse to a 3-edge tree");
        assert!(e.edges.iter().all(|edge| edge.distance == 1.0));
    }

    #[test]
    fn square_keeps_one_diagonal() {
        let s = std::f32::consts::SQRT_2;
        let edges = vec![
            Edge {
                u: 1,
                v: 0,
                distance: 1.0,
            },
            Edge {
                u: 2,
                v: 1,
                distance: 1.0,
            },
            Edge {
                u: 3,
                v: 2,
                distance: 1.0,
            },
            Edge {
                u: 3,
                v: 0,
                distance: 1.0,
            },
            Edge {
                u: 2,
                v: 0,
                distance: s,
            },
            Edge {
                u: 3,
                v: 1,
                distance: s,
            },
        ];
        let mut e = edge_list(4, edges);
        collapse_in_place(&mut e, false);
        assert_clean(&e);
        assert_eq!(e.edges.len(), 5, "exactly one diagonal must be removed");
        let diagonals = e.edges.iter().filter(|edge| edge.distance == s).count();
        assert_eq!(diagonals, 1);
    }

    #[test]
    fn late_common_neighbors_do_not_dominate() {
        let edges = vec![
            Edge {
                u: 1,
                v: 0,
                distance: 1.0,
            },
            Edge {
                u: 2,
                v: 0,
                distance: 2.0,
            },
            Edge {
                u: 2,
                v: 1,
                distance: 2.0,
            },
        ];
        let mut e = edge_list(3, edges);
        collapse_in_place(&mut e, false);
        assert_eq!(e.edges.len(), 2);
        assert!(e.edges.iter().any(|edge| edge.distance == 1.0));
    }

    #[test]
    fn pentagon_chords_are_critical() {
        let edges = vec![
            Edge {
                u: 1,
                v: 0,
                distance: 1.0,
            },
            Edge {
                u: 2,
                v: 1,
                distance: 1.0,
            },
            Edge {
                u: 3,
                v: 2,
                distance: 1.0,
            },
            Edge {
                u: 4,
                v: 3,
                distance: 1.0,
            },
            Edge {
                u: 4,
                v: 0,
                distance: 1.0,
            },
            Edge {
                u: 2,
                v: 0,
                distance: 2.0,
            },
            Edge {
                u: 3,
                v: 0,
                distance: 2.0,
            },
        ];
        let mut e = edge_list(5, edges);
        collapse_in_place(&mut e, false);
        assert_clean(&e);
        assert_eq!(e.edges.len(), 7, "nothing is dominated in this graph");
        assert!(e
            .edges
            .iter()
            .all(|edge| edge.distance == 1.0 || edge.distance == 2.0));
    }
}
