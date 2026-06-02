//! Greedy complete-linkage coning, fused into one in-place pass.
//!
//! Single public entry point: [`cone_in_place`]. It takes the condensed
//! distance matrix of a finite metric space and a `threshold`, and rewrites it
//! in place into the coned (strong-collapsed) filtration: complete-linkage
//! clusters are detected from the ascending edge stream, and each completed
//! linkage is realised as a cone (a strong collapse, so the truncated
//! persistent homology is unchanged). Returns the number of contractions.
//!

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// FxHash-style hasher specialised for the `u32` cluster keys. No dependency.
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

pub fn cone_in_place(d: &mut [f32], threshold: f32) -> usize {
    let m = d.len();
    if m == 0 {
        return 0;
    }
    let n = ((1.0 + (1.0 + 8.0 * m as f64).sqrt()) / 2.0).round() as usize;
    debug_assert_eq!(
        n * (n - 1) / 2,
        m,
        "cone_in_place: distance length {m} is not n(n-1)/2 for any n"
    );
    assert!(n <= 0xFFFF, "cone_in_place: n must fit in u16");

    // ---- Gather in-threshold edges, ascending; ties broken by endpoints. ----
    let tb = threshold.to_bits();
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(m);
    {
        let mut p = 0;
        for i in 1..n {
            let i_hi = (i as u32) << 16;
            for j in 0..i {
                let db = d[p].to_bits();
                if db <= tb {
                    edges.push((db, i_hi | j as u32));
                }
                p += 1;
            }
        }
    }
    edges.sort_unstable();

    // ══════════════════════════════════════════════════════════════════════════
    // CRITICAL FIX: Reset the matrix to INFINITY to match the original Tower.
    // We only re-populate edges that happen between active roots.
    // ══════════════════════════════════════════════════════════════════════════
    d.fill(f32::INFINITY);

    // ---- Fused complete-linkage + coning ----
    let mut parent: Vec<u32> = (0..n as u32).collect();
    let mut size: Vec<u32> = vec![1; n];
    let mut nbr: Vec<FxMap> = (0..n).map(|_| FxMap::default()).collect();
    let mut contractions = 0usize;

    for &(db, ep) in &edges {
        let u_idx = ep & 0xFFFF;
        let v_idx = ep >> 16;

        let a = find(&mut parent, u_idx);
        let b = find(&mut parent, v_idx);
        if a == b {
            continue;
        }
        let ai = a as usize;
        let bi = b as usize;
        let scale = f32::from_bits(db);

        // ══════════════════════════════════════════════════════════════════════════
        // Mirror original `tower.add_edge`: Record edge between active roots
        // ══════════════════════════════════════════════════════════════════════════
        let (hi, lo) = if a > b { (ai, bi) } else { (bi, ai) };
        let idx = hi * (hi - 1) / 2 + lo;
        if d[idx] == f32::INFINITY {
            d[idx] = scale;
        }

        // Cross-edge count between the two current clusters (kept symmetric).
        let count = {
            let e = nbr[ai].entry(b).or_insert(0);
            *e += 1;
            *e
        };
        *nbr[bi].entry(a).or_insert(0) += 1;

        // Complete linkage at this scale ⇒ contract.
        if count as u64 == size[ai] as u64 * size[bi] as u64 {
            let (w, l) = if nbr[ai].len() <= nbr[bi].len() {
                (b, a)
            } else {
                (a, b)
            };
            let (wi, li) = (w as usize, l as usize);

            // Drain the loser's (smaller) map into the winner, emitting a cone
            // edge for each genuinely new neighbour.
            let loser_map = std::mem::take(&mut nbr[li]);
            for (nb, cnt) in loser_map {
                if nb == w {
                    continue;
                }
                match nbr[wi].entry(nb) {
                    Entry::Occupied(mut o) => {
                        *o.get_mut() += cnt;
                    }
                    Entry::Vacant(slot) => {
                        slot.insert(cnt);
                        // New neighbour ⇒ cone edge w—nb born at `scale`.
                        let (c_hi, c_lo) = if w > nb {
                            (w as usize, nb as usize)
                        } else {
                            (nb as usize, w as usize)
                        };
                        let idx = c_hi * (c_hi - 1) / 2 + c_lo;
                        if scale < d[idx] {
                            d[idx] = scale;
                        }
                    }
                }
                // Redirect the neighbour from the loser to the winner.
                let nm = &mut nbr[nb as usize];
                nm.remove(&l);
                *nm.entry(w).or_insert(0) += cnt;
            }
            nbr[wi].remove(&l);

            size[wi] += size[li];
            parent[li] = w;
            contractions += 1;
        }
    }

    contractions
}
