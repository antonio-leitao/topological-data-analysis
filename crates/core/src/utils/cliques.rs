//! Count flag-complex simplices by intersecting lower-neighbour bitsets.
//! Rooting each clique at its largest vertex counts it once.

use crate::preprocess::edgelist::EdgeList;
use rayon::prelude::*;

const PARALLEL_CLIQUE_MIN_ROOTS: usize = 128;
const PARALLEL_CLIQUE_MIN_EDGES: usize = 4096;

/// Lower neighbours of one vertex as bitset words from block `lo`, so a row
/// spans its own neighbours rather than every id below them.
#[derive(Default)]
struct Row {
    lo: usize,
    bits: Vec<u64>,
}

/// Intersect `xs` (from block `xlo`) with `ys` (from `ylo`) into `dest`, which
/// starts at the returned block, and trim trailing zero words.
#[inline]
fn intersect_into(xs: &[u64], xlo: usize, ys: &[u64], ylo: usize, dest: &mut Vec<u64>) -> usize {
    let lo = xlo.max(ylo);
    let hi = (xlo + xs.len()).min(ylo + ys.len());
    dest.clear();
    if lo >= hi {
        return lo;
    }
    let (xs, ys) = (&xs[lo - xlo..hi - xlo], &ys[lo - ylo..hi - ylo]);
    dest.resize(hi - lo, 0);

    let mut last_nz = 0;
    let mut any = false;
    for (i, ((d, &x), &y)) in dest.iter_mut().zip(xs).zip(ys).enumerate() {
        let w = x & y;
        *d = w;
        if w != 0 {
            last_nz = i;
            any = true;
        }
    }
    if any {
        dest.truncate(last_nz + 1);
    } else {
        dest.clear();
    }
    lo
}

/// Count extensions from candidates common to the current stem.
fn extend(
    adj: &[Row],
    cand: &[u64],
    cand_lo: usize,
    stem_size: usize,
    max_size: usize,
    scratch: &mut [Vec<u64>],
    count: &mut usize,
) {
    if stem_size + 1 < max_size {
        let (head, tail) = scratch.split_at_mut(1);
        for (wi, &word0) in cand.iter().enumerate() {
            let mut word = word0;
            while word != 0 {
                let v = ((cand_lo + wi) << 6) | word.trailing_zeros() as usize;
                word &= word - 1;

                *count += 1; // clique {stem ∪ v}
                let row = &adj[v];
                let lo = intersect_into(cand, cand_lo, &row.bits, row.lo, &mut head[0]);
                if !head[0].is_empty() {
                    extend(adj, &head[0], lo, stem_size + 1, max_size, tail, count);
                }
            }
        }
    } else {
        for &word in cand {
            *count += word.count_ones() as usize;
        }
    }
}

/// Count cliques with at most `max_size` vertices.
pub(crate) fn count_cliques(edge_list: EdgeList, max_size: usize, parallel: bool) -> usize {
    let EdgeList { n, edges, .. } = edge_list;
    let max_size = max_size.min(n);

    if max_size <= 1 {
        return n;
    }
    if max_size == 2 {
        return n + edges.len();
    }

    let edge_count = edges.len();
    let mut span = vec![(usize::MAX, 0usize); n];
    for edge in &edges {
        let (u, b) = (edge.u as usize, edge.v as usize >> 6);
        debug_assert!(u < n && edge.u > edge.v);
        span[u] = (span[u].0.min(b), span[u].1.max(b + 1));
    }
    let mut adj: Vec<Row> = span
        .into_iter()
        .map(|(lo, hi)| match hi {
            0 => Row::default(),
            _ => Row {
                lo,
                bits: vec![0; hi - lo],
            },
        })
        .collect();
    for edge in edges {
        let row = &mut adj[edge.u as usize];
        let v = edge.v as usize;
        row.bits[(v >> 6) - row.lo] |= 1u64 << (v & 63);
    }

    let use_parallel =
        parallel && n >= PARALLEL_CLIQUE_MIN_ROOTS && edge_count >= PARALLEL_CLIQUE_MIN_EDGES;
    let rooted_count = if use_parallel {
        adj.par_iter()
            .map_init(
                || (0..max_size - 2).map(|_| Vec::new()).collect::<Vec<_>>(),
                |scratch, row| {
                    let mut count = 0;
                    if !row.bits.is_empty() {
                        extend(&adj, &row.bits, row.lo, 1, max_size, scratch, &mut count);
                    }
                    count
                },
            )
            .sum()
    } else {
        // One candidate buffer per interior recursion depth.
        let mut scratch: Vec<Vec<u64>> = (0..max_size - 2).map(|_| Vec::new()).collect();
        let mut count = 0;
        for row in &adj {
            if !row.bits.is_empty() {
                extend(&adj, &row.bits, row.lo, 1, max_size, &mut scratch, &mut count);
            }
        }
        count
    };
    n + rooted_count
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
            threshold: 1.0,
            sorted: false,
        }
    }

    fn brute(n: usize, edges: &[Edge], max_size: usize) -> usize {
        let mut adjacency = vec![vec![false; n]; n];
        for edge in edges {
            let u = edge.u as usize;
            let v = edge.v as usize;
            adjacency[u][v] = true;
            adjacency[v][u] = true;
        }

        let mut count = 0usize;
        for mask in 1u32..(1u32 << n) {
            let verts: Vec<usize> = (0..n).filter(|&k| (mask >> k) & 1 == 1).collect();
            if verts.len() > max_size {
                continue;
            }
            let mut ok = true;
            'pairs: for a in 0..verts.len() {
                for b in (a + 1)..verts.len() {
                    if !adjacency[verts[a]][verts[b]] {
                        ok = false;
                        break 'pairs;
                    }
                }
            }
            if ok {
                count += 1;
            }
        }
        count
    }

    #[test]
    fn k4_complete() {
        let edges: Vec<Edge> = (1..4)
            .flat_map(|u| {
                (0..u).map(move |v| Edge {
                    u,
                    v,
                    distance: 1.0,
                })
            })
            .collect();
        assert_eq!(count_cliques(edge_list(4, edges.clone()), 1, false), 4);
        assert_eq!(count_cliques(edge_list(4, edges.clone()), 2, false), 10);
        assert_eq!(count_cliques(edge_list(4, edges.clone()), 3, false), 14);
        assert_eq!(count_cliques(edge_list(4, edges.clone()), 4, false), 15);
        assert_eq!(count_cliques(edge_list(4, edges), 9, false), 15);
        assert_eq!(count_cliques(edge_list(4, Vec::new()), 4, false), 4);
    }

    #[test]
    fn single_point() {
        assert_eq!(count_cliques(edge_list(1, Vec::new()), 3, false), 1);
    }

    #[test]
    fn matches_brute_force() {
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 16_777_216.0
        };
        for n in 2..=8usize {
            for _ in 0..40 {
                for &thr in &[0.2f32, 0.4, 0.5, 0.7, 0.9, 1.0] {
                    let mut edges = Vec::new();
                    for u in 1..n {
                        for v in 0..u {
                            let distance = rng();
                            if distance <= thr {
                                edges.push(Edge {
                                    u: u as u32,
                                    v: v as u32,
                                    distance,
                                });
                            }
                        }
                    }
                    for max_size in 1..=n {
                        assert_eq!(
                            count_cliques(edge_list(n, edges.clone()), max_size, false),
                            brute(n, &edges, max_size),
                            "n={n} thr={thr} max_size={max_size}"
                        );
                    }
                }
            }
        }
    }

    /// Rows span their own neighbours: cliques far up the id range count the
    /// same as the same cliques near 0.
    #[test]
    fn far_ids_count_like_near_ones() {
        let shift = 1u32 << 20;
        let near: Vec<Edge> = [(1, 0), (2, 0), (2, 1), (3, 2), (70, 3), (70, 2), (130, 70)]
            .iter()
            .map(|&(u, v)| Edge { u, v, distance: 1.0 })
            .collect();
        let far: Vec<Edge> = near
            .iter()
            .map(|e| Edge { u: e.u + shift, v: e.v + shift, distance: 1.0 })
            .collect();
        for max_size in 1..=4 {
            assert_eq!(
                count_cliques(edge_list(131, near.clone()), max_size, false),
                count_cliques(edge_list((shift + 131) as usize, far.clone()), max_size, false)
                    - shift as usize
            );
        }
    }

    #[test]
    fn parallel_matches_sequential() {
        let n = 96u32;
        let edges: Vec<Edge> = (1..n)
            .flat_map(|u| {
                (0..u).map(move |v| Edge {
                    u,
                    v,
                    distance: 1.0,
                })
            })
            .collect();

        let sequential = count_cliques(edge_list(n as usize, edges.clone()), 4, false);
        let parallel = count_cliques(edge_list(n as usize, edges), 4, true);
        assert_eq!(sequential, parallel);
    }
}
