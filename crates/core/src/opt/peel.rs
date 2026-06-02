//! Strong-collapse truncation: tighten a Vietoris–Rips truncation radius,
//! **soundly**.
//!
//! Single public entry point: [`peel`]. Given the condensed distance matrix of
//! a finite metric space and any valid truncation radius `R` it returns a radius
//! `R' ≤ R` such that `VR(X, t)` is *certified contractible for every* `t ≥ R'`
//! — a drop-in, tighter truncation parameter for any PH engine that preserves
//! the entire reduced barcode up to the computed dimension.
//!

#[inline(always)]
fn is_subset(a: &[u64], b: &[u64]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    for k in 0..a.len() {
        // SAFETY: same-length slices, k in bounds.
        unsafe {
            if *a.get_unchecked(k) & !*b.get_unchecked(k) != 0 {
                return false;
            }
        }
    }
    true
}

#[inline(always)]
fn bit_set(bits: &[u64], i: usize) -> bool {
    (bits[i >> 6] >> (i & 63)) & 1 == 1
}

/// Tighten a VR truncation radius via *sound* strong-collapse peeling.
///
/// `d` is the lower-triangular condensed distance matrix (length `n*(n-1)/2`);
/// `n` is recovered from its length. `c_star` is the Chebyshev center returned
/// by [`crate::pdist::pdist_tiled_v3`] (the cone apex peeling proceeds from).
/// `threshold` is a valid truncation radius — pass the `r_cheb` from the same
/// call, or any radius `≥` it.
///
/// Returns a radius `R' ≤ threshold` such that `VR(X, t)` is contractible (hence
/// homologically trivial) for **every** `t ≥ R'`, certified by a strong-collapse
/// witness `(ball(c⋆, t), c⋆)` at every critical radius in `[R', threshold]`.
///
/// Requires `n ≤ 65535` (edge endpoints are packed into a single `u32`).
pub fn peel(d: &[f32], c_star: usize, threshold: f32) -> f32 {
    // Recover n from the condensed length m = n(n-1)/2.
    let m = d.len();
    if m == 0 {
        return 0.0; // n ≤ 1: a single point (or none) is contractible at 0.
    }
    let n = ((1.0 + (1.0 + 8.0 * m as f64).sqrt()) / 2.0).round() as usize;
    debug_assert_eq!(
        n * (n - 1) / 2,
        m,
        "peel: distance length {m} is not n(n-1)/2 for any n"
    );
    assert!(n <= 0xFFFF, "peel: n must fit in u16");
    debug_assert!(c_star < n, "peel: c_star out of range");

    let nw = (n + 63) >> 6;
    let threshold_bits = threshold.to_bits();

    // ---- Build sorted edges: only those with dist ≤ threshold ----
    // Every radius we consider is ≤ threshold, so upper-tail edges never enter
    // any neighborhood we inspect; skipping them shrinks the sort and the fill.
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(m);
    {
        let mut p = 0;
        for i in 1..n {
            let i_hi = (i as u32) << 16;
            for j in 0..i {
                let db = d[p].to_bits();
                if db <= threshold_bits {
                    edges.push((db, i_hi | j as u32));
                }
                p += 1;
            }
        }
    }
    edges.sort_unstable_by_key(|&(k, _)| k);

    // ---- Distances from c_star (used only to seed ball membership) ----
    let cs_bits: Vec<u32> = (0..n)
        .map(|u| {
            if u == c_star {
                0u32
            } else {
                let (a, b) = if c_star > u { (c_star, u) } else { (u, c_star) };
                d[a * (a - 1) / 2 + b].to_bits()
            }
        })
        .collect();

    // ---- Neighborhood bitsets: all kept edges (≤ threshold) + diagonal ----
    let mut nbr_bits = vec![0u64; n * nw];
    for &(_, endpoints) in &edges {
        let u = (endpoints >> 16) as usize;
        let v = (endpoints & 0xFFFF) as usize;
        nbr_bits[u * nw + (v >> 6)] |= 1u64 << (v & 63);
        nbr_bits[v * nw + (u >> 6)] |= 1u64 << (u & 63);
    }
    for v in 0..n {
        // v ∈ N_R(v) always — never cleared.
        nbr_bits[v * nw + (v >> 6)] |= 1u64 << (v & 63);
    }

    // ---- Chebyshev-ball membership: ball(R) = { x : d(c⋆, x) ≤ R } ----
    // At R = threshold ≥ R_cheb the ball is all of X (the contract). We still
    // honour threshold < R_cheb defensively: any point already outside the ball
    // is seeded as periphery and must be dominated at the first critical radius.
    let mut ball_bits = vec![u64::MAX; nw];
    let rem = n & 63;
    if rem != 0 {
        ball_bits[nw - 1] = (1u64 << rem) - 1;
    }
    let mut dirty: Vec<u32> = Vec::with_capacity(n);
    let mut in_dirty = vec![false; n];
    for x in 0..n {
        if cs_bits[x] > threshold_bits {
            ball_bits[x >> 6] &= !(1u64 << (x & 63));
            in_dirty[x] = true;
            dirty.push(x as u32);
        }
    }

    // Per-periphery cached dominator and its reverse index.
    let mut dominator: Vec<i32> = vec![-1; n];
    let mut users: Vec<Vec<u32>> = vec![Vec::new(); n];

    let mut r_alg = threshold;
    let mut top = edges.len();

    // ---- Descend through distinct critical radii, high to low ----
    while top > 0 {
        let v_bits = edges[top - 1].0;
        let v = f32::from_bits(v_bits);

        // ===== EVALUATE the localization predicate at scale v =====
        // `nbr_bits` and `ball_bits` already reflect "≤ v" (everything strictly
        // above v was removed in the previous descend phase). Resolve every
        // dirtied / newly-peripheral vertex; a single unfixable one means VR is
        // not contractible at v, so the last certified radius is the answer.
        while let Some(w_u32) = dirty.pop() {
            let w = w_u32 as usize;
            in_dirty[w] = false;
            if bit_set(&ball_bits, w) {
                continue; // still inside the ball ⇒ not periphery, nothing to prove
            }

            // 1) Retry the cached dominator (must still be in the ball).
            let cached = dominator[w];
            if cached >= 0 {
                let z = cached as usize;
                if bit_set(&ball_bits, z)
                    && is_subset(
                        &nbr_bits[w * nw..w * nw + nw],
                        &nbr_bits[z * nw..z * nw + nw],
                    )
                {
                    continue;
                }
            }

            // 2) Search (N_R(w) ∩ ball) \ {w} for a fresh dominator.
            let w_word = w >> 6;
            let w_mask = !(1u64 << (w & 63));
            let mut found = false;
            'search: for word_idx in 0..nw {
                let mut bits = nbr_bits[w * nw + word_idx] & ball_bits[word_idx];
                if word_idx == w_word {
                    bits &= w_mask;
                }
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let z = (word_idx << 6) | bit;
                    if is_subset(
                        &nbr_bits[w * nw..w * nw + nw],
                        &nbr_bits[z * nw..z * nw + nw],
                    ) {
                        dominator[w] = z as i32;
                        users[z].push(w as u32);
                        found = true;
                        break 'search;
                    }
                }
            }
            if !found {
                // Periphery vertex w has no ball-dominator at scale v: the
                // localization hypothesis fails here, so v is not certifiable.
                return r_alg;
            }
        }

        // Predicate held everywhere at scale v ⇒ VR(X, t) contractible on
        // [v, previous radius). Commit v as the new tightest certified radius.
        r_alg = v;

        // ===== DESCEND just below v: remove the level-v events =====
        while top > 0 && edges[top - 1].0 == v_bits {
            top -= 1;
            let endpoints = edges[top].1;
            let a = (endpoints >> 16) as usize;
            let b = (endpoints & 0xFFFF) as usize;

            // (i) edge disappears ⇒ N(a), N(b) shrink.
            nbr_bits[a * nw + (b >> 6)] &= !(1u64 << (b & 63));
            nbr_bits[b * nw + (a >> 6)] &= !(1u64 << (a & 63));

            // (ii) only dominations whose dominator lost a neighbor can break:
            //      recheck users[a] ∪ users[b] (filtering stale reverse entries).
            for k in 0..users[a].len() {
                let w = users[a][k] as usize;
                if dominator[w] == a as i32 && !in_dirty[w] {
                    in_dirty[w] = true;
                    dirty.push(w as u32);
                }
            }
            for k in 0..users[b].len() {
                let w = users[b][k] as usize;
                if dominator[w] == b as i32 && !in_dirty[w] {
                    in_dirty[w] = true;
                    dirty.push(w as u32);
                }
            }

            // (iii) ball departure: the edge to c⋆ vanishing is *exactly* the
            //       event "the other endpoint leaves the Chebyshev ball".
            let dep = if a == c_star {
                Some(b)
            } else if b == c_star {
                Some(a)
            } else {
                None
            };
            if let Some(p) = dep {
                ball_bits[p >> 6] &= !(1u64 << (p & 63));
                if !in_dirty[p] {
                    in_dirty[p] = true; // now periphery: needs a dominator next eval
                    dirty.push(p as u32);
                }
                // p's own users were already flagged in (ii) since the removed
                // edge is incident to p (so N(p) shrank); nothing more to do.
            }
        }
    }

    // Reached the bottom with no failure: every critical radius certified.
    r_alg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::pdist::pdist_tiled_v3;

    #[test]
    fn recovers_n_from_condensed_length() {
        for n in 2..500usize {
            let m = n * (n - 1) / 2;
            let nn = ((1.0 + (1.0 + 8.0 * m as f64).sqrt()) / 2.0).round() as usize;
            assert_eq!(nn, n, "n recovery failed at n={n}");
        }
    }

    #[test]
    fn trivial_sizes_are_contractible_at_zero() {
        assert_eq!(peel(&[], 0, 0.0), 0.0);
    }

    #[test]
    fn peel_never_exceeds_cheb_radius() {
        // 3×3 grid in the plane.
        let mut pts = Vec::new();
        for x in 0..3 {
            for y in 0..3 {
                pts.push(x as f32);
                pts.push(y as f32);
            }
        }
        let (d, c_star, r_cheb) = pdist_tiled_v3(&pts, 9, 2);
        let r = peel(&d, c_star, r_cheb);
        assert!(r.is_finite() && r >= 0.0);
        assert!(
            r <= r_cheb,
            "peel must not increase the radius ({r} > {r_cheb})"
        );
    }

    #[test]
    fn cone_in_cheb_gap() {
        // 5-pt metric: kite A,B,C,D (paper Example 1) + center P near B,C, far
        // from A,D. Indices: 0=P, 1=A, 2=B, 3=C, 4=D. Condensed lower-triangular:
        //   PA=1.5
        //   PB=1.05  AB=1.0
        //   PC=1.05  AC=1.5  BC=1.0
        //   PD=1.5   AD=1.1  BD=1.5  CD=1.0
        //
        // Ground truth: an H1 bar (1.1, 1.5) ⇒ Rhomol = 1.5; above 1.5, P cones X.
        // The cycle is born at AD=1.1 and dies at the diagonals=1.5, *entirely
        // inside* the c⋆-distance gap (1.05, 1.5). A validator that only checks
        // c⋆-distances commits 1.05 and erases the bar; the sound sweep must not.
        let d: [f32; 10] = [1.5, 1.05, 1.0, 1.05, 1.5, 1.0, 1.5, 1.1, 1.5, 1.0];
        let r = peel(&d, 0, 1.5);
        assert!(
            r >= 1.5 - 1e-6,
            "unsound: peel returned {r} < Rhomol = 1.5, erasing the H1 bar (1.1, 1.5)"
        );
    }

    #[test]
    fn dense_cluster_reduces_below_cheb() {
        // A tight Gaussian-ish blob plus a couple of close points: peeling should
        // certify a radius strictly below R_cheb (sanity that it does reduce).
        let pts: Vec<f32> = vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05, 0.2, 0.05, 0.05, 0.2,
        ];
        let n = pts.len() / 2;
        let (d, c_star, r_cheb) = pdist_tiled_v3(&pts, n, 2);
        let r = peel(&d, c_star, r_cheb);
        assert!(r.is_finite() && r >= 0.0 && r <= r_cheb);
    }

    #[test]
    fn peel_must_not_skip_a_cycle_in_a_cheb_distance_gap() {
        // 5-pt metric: kite A,B,C,D (paper Example 1) + center P near B,C, far from A,D.
        // Index order: 0=P, 1=A, 2=B, 3=C, 4=D.  Condensed lower-triangular layout:
        //   PA=1.5
        //   PB=1.05  AB=1.0
        //   PC=1.05  AC=1.5  BC=1.0
        //   PD=1.5   AD=1.1  BD=1.5  CD=1.0
        let d: [f32; 10] = [1.5, 1.05, 1.0, 1.05, 1.5, 1.0, 1.5, 1.1, 1.5, 1.0];
        let c_star = 0; // P: min eccentricity (all eccs = 1.5; tie -> index 0)
        let r_cheb = 1.5;

        // Ground truth: H1 bar (1.1, 1.5) -> Rhomol = 1.5. Above 1.5, P cones X.
        // A sound truncation must therefore return >= 1.5.
        let r = peel(&d, c_star, r_cheb);

        assert!(
            r >= 1.5 - 1e-6,
            "unsound truncation: peel returned {r} < Rhomol = 1.5, erasing the \
         H1 bar (1.1, 1.5). The cycle is born at AD=1.1 and dies at the \
         diagonals=1.5, entirely inside the c*-distance gap (1.05, 1.5) that \
         the batch shrink never inspects."
        );
    }
}
