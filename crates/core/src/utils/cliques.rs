//! Vietoris–Rips clique (simplex) counting up to a fixed size.
//!
//! Single public entry point: [`count_cliques`]. Given a condensed distance
//! matrix, a scale `threshold`, and a maximum simplex size `max_size` (a count
//! of *vertices*), it returns the number of cliques of size `1..=max_size` in
//! the graph whose edges are the point pairs at distance `≤ threshold`.
//! Equivalently, this is the Vietoris–Rips filtration size truncated at that
//! scale and dimension (a clique of size `k` is a `(k-1)`-simplex):
//!
//!   * `max_size = 1` → vertices only,
//!   * `max_size = 2` → + edges,
//!   * `max_size = 3` → + triangles, and so on.
//!
//! Method. Build, for every vertex `i`, the bitset of its lower-indexed
//! neighbours `N⁻(i) = {j < i : d(i, j) ≤ threshold}`. Those rows are exactly
//! the contiguous blocks of the condensed matrix, so the build is one linear
//! sweep. Then enumerate cliques largest-vertex-first: each clique is rooted at
//! its maximum vertex and grown downward by intersecting candidate sets, which
//! keeps every clique canonical (counted once) with no per-candidate ordering
//! test. The intersection trims trailing-zero words so deeper intersections
//! stay short, scratch is allocated once per recursion depth and reused, and
//! the deepest level — where cliques can no longer grow — is counted by
//! `popcount` rather than bit-by-bit.

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

/// Number of cliques of size `1..=max_size` in the Vietoris–Rips graph of a
/// finite metric space at scale `threshold` — its truncated filtration size.
///
/// `d` is the lower-triangular condensed distance matrix (length `n*(n-1)/2`,
/// as produced by [`crate::pdist::pdist_tiled_v3`]); `n` is recovered from its
/// length. An edge joins two points iff their distance is `≤ threshold`. A
/// length-0 matrix is treated as a single point (`n = 1`).
pub fn count_cliques(d: &[f32], threshold: f32, max_size: usize) -> usize {
    // Recover n from the condensed length m = n(n-1)/2.
    let m = d.len();
    let n = ((1.0 + (1.0 + 8.0 * m as f64).sqrt()) / 2.0).round() as usize;
    debug_assert_eq!(
        n * (n - 1) / 2,
        m,
        "count_cliques: distance length {m} is not n(n-1)/2 for any n"
    );

    // No clique can exceed n vertices; clamp so absurd max_size neither
    // over-allocates scratch nor over-grows the recursion.
    let max_size = max_size.min(n);

    // Vertices are always present; sizes 0/1 ask for nothing more.
    if max_size <= 1 {
        return n;
    }

    // Lower-neighbour adjacency: adj[i] = {j < i : d(i, j) ≤ threshold}, read
    // straight from the contiguous condensed row d[i(i-1)/2 .. i(i-1)/2 + i].
    let mut adj: Vec<Vec<u64>> = vec![Vec::new(); n];
    let mut base = 0;
    for i in 1..n {
        let row = &d[base..base + i];
        for (j, &dist) in row.iter().enumerate() {
            if dist <= threshold {
                set_bit(&mut adj[i], j);
            }
        }
        base += i;
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

    fn edge(d: &[f32], a: usize, b: usize, thr: f32) -> bool {
        if a == b {
            return false;
        }
        let (i, j) = if a > b { (a, b) } else { (b, a) }; // i > j
        d[i * (i - 1) / 2 + j] <= thr
    }

    /// O(2ⁿ) reference: count every vertex subset of size ≤ max_size that is a clique.
    fn brute(d: &[f32], n: usize, thr: f32, max_size: usize) -> usize {
        let mut count = 0usize;
        for mask in 1u32..(1u32 << n) {
            let verts: Vec<usize> = (0..n).filter(|&k| (mask >> k) & 1 == 1).collect();
            if verts.len() > max_size {
                continue;
            }
            let mut ok = true;
            'pairs: for a in 0..verts.len() {
                for b in (a + 1)..verts.len() {
                    if !edge(d, verts[a], verts[b], thr) {
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
        let d = vec![1.0f32; 6]; // 4 points, all pairwise distance 1
        assert_eq!(count_cliques(&d, 1.0, 1), 4); // vertices
        assert_eq!(count_cliques(&d, 1.0, 2), 10); // + 6 edges
        assert_eq!(count_cliques(&d, 1.0, 3), 14); // + 4 triangles
        assert_eq!(count_cliques(&d, 1.0, 4), 15); // + 1 tetrahedron
        assert_eq!(count_cliques(&d, 1.0, 9), 15); // max_size clamps to n
        assert_eq!(count_cliques(&d, 0.5, 4), 4); // threshold below all edges
    }

    #[test]
    fn single_point() {
        assert_eq!(count_cliques(&[], 10.0, 3), 1);
    }

    #[test]
    fn matches_brute_force() {
        // Deterministic xorshift over random condensed matrices.
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 16_777_216.0
        };
        for n in 2..=8usize {
            let m = n * (n - 1) / 2;
            for _ in 0..40 {
                let d: Vec<f32> = (0..m).map(|_| rng()).collect();
                for &thr in &[0.2f32, 0.4, 0.5, 0.7, 0.9, 1.0] {
                    for max_size in 1..=n {
                        assert_eq!(
                            count_cliques(&d, thr, max_size),
                            brute(&d, n, thr, max_size),
                            "n={n} thr={thr} max_size={max_size}"
                        );
                    }
                }
            }
        }
    }
}
