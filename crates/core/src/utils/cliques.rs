//! Vietoris–Rips clique (simplex) counting up to a fixed size.
//!
//! [`count_cliques`] consumes a truncated weighted edge list and returns the
//! number of cliques of size `1..=max_size`, where `max_size` counts vertices.
//! Equivalently, this is the Vietoris–Rips filtration size truncated at that
//! scale and dimension (a clique of size `k` is a `(k-1)`-simplex):
//!
//!   * `max_size = 1` → vertices only,
//!   * `max_size = 2` → + edges,
//!   * `max_size = 3` → + triangles, and so on.
//!
//! Method. Build, for every vertex `i`, the bitset of its lower-indexed neighbors
//! `N-(i) = {j < i : (i, j) is an edge}`. Then enumerate cliques
//! largest-vertex-first: each clique is rooted at its maximum vertex and grown
//! downward by intersecting candidate sets, which keeps every clique canonical
//! and counted once. The intersection trims trailing-zero words, scratch is
//! reused by recursion depth, and the deepest level is counted by `popcount`.

use crate::preprocess::edgelist::EdgeList;

/// Set bit `v` in a dynamically-sized bit vector, growing it if needed.
#[inline]
fn set_bit(bits: &mut Vec<u64>, v: usize) {
    let w = v >> 6;
    if w >= bits.len() {
        bits.resize(w + 1, 0);
    }
    bits[w] |= 1u64 << (v & 63);
}

/// `dest = xs ∩ ys`. Iterates only the shorter operand and trims trailing-zero
/// words; an empty intersection leaves `dest` empty (`len == 0`).
#[inline]
fn intersect_into(xs: &[u64], ys: &[u64], dest: &mut Vec<u64>) {
    let (short, long) = if xs.len() <= ys.len() {
        (xs, ys)
    } else {
        (ys, xs)
    };
    dest.clear();
    if short.is_empty() {
        return;
    }
    dest.resize(short.len(), 0);

    let mut last_nz = 0;
    let mut any = false;
    for i in 0..short.len() {
        let w = short[i] & long[i];
        dest[i] = w;
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
}

/// Count every clique that extends the current stem by one vertex drawn from
/// `cand`, recursing while the clique can still grow below `max_size`.
///
/// `cand` is the set of vertices adjacent to all stem members and smaller than
/// the most recently added one; `stem_size` is the number of vertices already
/// fixed. `scratch[d]` is the reusable candidate buffer for recursion depth `d`.
fn extend(
    adj: &[Vec<u64>],
    cand: &[u64],
    stem_size: usize,
    max_size: usize,
    scratch: &mut [Vec<u64>],
    count: &mut usize,
) {
    // Adding one vertex from `cand` yields a clique of size `stem_size + 1`.
    if stem_size + 1 < max_size {
        // Interior level: count each candidate and recurse to find larger cliques.
        let (head, tail) = scratch.split_at_mut(1);
        for (wi, &word0) in cand.iter().enumerate() {
            let mut word = word0;
            while word != 0 {
                let v = (wi << 6) | word.trailing_zeros() as usize;
                word &= word - 1;

                *count += 1; // clique {stem ∪ v}
                intersect_into(cand, &adj[v], &mut head[0]);
                if !head[0].is_empty() {
                    extend(adj, &head[0], stem_size + 1, max_size, tail, count);
                }
            }
        }
    } else {
        // Deepest level: cliques here cannot grow, so just tally the candidates.
        for &word in cand {
            *count += word.count_ones() as usize;
        }
    }
}

/// Number of cliques of size `1..=max_size` in a truncated weighted graph.
pub(crate) fn count_cliques(edge_list: EdgeList, max_size: usize) -> usize {
    let EdgeList { n, edges, .. } = edge_list;
    // No clique can exceed n vertices; clamp so absurd max_size neither
    // over-allocates scratch nor over-grows the recursion.
    let max_size = max_size.min(n);

    // Vertices are always present; sizes 0/1 ask for nothing more.
    if max_size <= 1 {
        return n;
    }
    if max_size == 2 {
        return n + edges.len();
    }

    // Lower-neighbor adjacency: every edge is canonical with u > v.
    let mut adj: Vec<Vec<u64>> = vec![Vec::new(); n];
    for edge in edges {
        let u = edge.u as usize;
        let v = edge.v as usize;
        debug_assert!(u < n && u > v);
        set_bit(&mut adj[u], v);
    }

    let mut count = n; // size-1 simplices (vertices)

    // One reusable candidate buffer per interior recursion depth. A clique of
    // size k is formed at depth k-2, and only sizes < max_size recurse, so the
    // deepest interior level is k = max_size-1 → max_size-2 buffers.
    let mut scratch: Vec<Vec<u64>> = (0..max_size - 2).map(|_| Vec::new()).collect();

    for u in 0..n {
        if !adj[u].is_empty() {
            extend(&adj, &adj[u], 1, max_size, &mut scratch, &mut count);
        }
    }
    count
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

    /// O(2ⁿ) reference: count every vertex subset of size ≤ max_size that is a clique.
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
        assert_eq!(count_cliques(edge_list(4, edges.clone()), 1), 4);
        assert_eq!(count_cliques(edge_list(4, edges.clone()), 2), 10);
        assert_eq!(count_cliques(edge_list(4, edges.clone()), 3), 14);
        assert_eq!(count_cliques(edge_list(4, edges.clone()), 4), 15);
        assert_eq!(count_cliques(edge_list(4, edges), 9), 15);
        assert_eq!(count_cliques(edge_list(4, Vec::new()), 4), 4);
    }

    #[test]
    fn single_point() {
        assert_eq!(count_cliques(edge_list(1, Vec::new()), 3), 1);
    }

    #[test]
    fn matches_brute_force() {
        // Deterministic xorshift over random graph topologies.
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
                                    u: u as u16,
                                    v: v as u16,
                                    distance,
                                });
                            }
                        }
                    }
                    for max_size in 1..=n {
                        assert_eq!(
                            count_cliques(edge_list(n, edges.clone()), max_size),
                            brute(n, &edges, max_size),
                            "n={n} thr={thr} max_size={max_size}"
                        );
                    }
                }
            }
        }
    }
}
