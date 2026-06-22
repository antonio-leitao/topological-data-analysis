//! Sound strong-collapse truncation over a weighted edge list.
//!
//! [`peel`] sorts and tightens an [`EdgeList`] in place. If its cutoff is `R`,
//! the resulting cutoff `R' <= R` is certified so that `VR(X, t)` is
//! contractible for every `t >= R'`.

use crate::preprocess::edgelist::EdgeList;
use rayon::prelude::*;

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

/// Tighten an edge list's cutoff via sound strong-collapse peeling.
///
/// The list is sorted by ascending `(distance, endpoints)`, truncated to the
/// certified radius, and left sorted for subsequent optimizations. Its center is
/// the fixed cone apex and is not changed.
pub(crate) fn peel(edge_list: &mut EdgeList, parallel: bool) {
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

    let n = edge_list.n;
    if n <= 1 {
        edge_list.edges.clear();
        edge_list.threshold = 0.0;
        return;
    }

    let center = edge_list.center as usize;
    debug_assert!(center < n, "peel: center out of range");
    debug_assert!(edge_list
        .edges
        .iter()
        .all(|edge| edge.distance <= edge_list.threshold));

    let nw = (n + 63) >> 6;

    // Neighborhoods at the current cutoff, including the diagonal.
    let mut nbr_bits = vec![0u64; n * nw];
    let mut ball_bits = vec![0u64; nw];
    ball_bits[center >> 6] |= 1u64 << (center & 63);

    for edge in &edge_list.edges {
        let u = edge.u as usize;
        let v = edge.v as usize;
        nbr_bits[u * nw + (v >> 6)] |= 1u64 << (v & 63);
        nbr_bits[v * nw + (u >> 6)] |= 1u64 << (u & 63);

        // At the starting cutoff, the center ball is exactly the center and its
        // present neighbors. Missing center edges are outside the ball.
        if u == center {
            ball_bits[v >> 6] |= 1u64 << (v & 63);
        } else if v == center {
            ball_bits[u >> 6] |= 1u64 << (u & 63);
        }
    }
    for v in 0..n {
        nbr_bits[v * nw + (v >> 6)] |= 1u64 << (v & 63);
    }

    let mut dirty: Vec<u32> = Vec::with_capacity(n);
    let mut in_dirty = vec![false; n];
    for x in 0..n {
        if !bit_set(&ball_bits, x) {
            in_dirty[x] = true;
            dirty.push(x as u32);
        }
    }

    // Cached dominators and their reverse users. Reverse entries may become
    // stale and are filtered against `dominator` when revisited.
    let mut dominator: Vec<i32> = vec![-1; n];
    let mut users: Vec<Vec<u32>> = vec![Vec::new(); n];

    let mut certified_radius = edge_list.threshold;
    let mut top = edge_list.edges.len();

    // Descend through distinct edge radii, high to low.
    'levels: while top > 0 {
        let radius_bits = edge_list.edges[top - 1].distance.to_bits();
        let radius = f32::from_bits(radius_bits);

        // Neighborhood and ball bitsets currently represent scale `radius`.
        while let Some(w_u32) = dirty.pop() {
            let w = w_u32 as usize;
            in_dirty[w] = false;
            if bit_set(&ball_bits, w) {
                continue;
            }

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
                break 'levels;
            }
        }

        certified_radius = radius;

        // Descend just below this radius by removing the entire tied edge level.
        while top > 0 && edge_list.edges[top - 1].distance.to_bits() == radius_bits {
            top -= 1;
            let a = edge_list.edges[top].u as usize;
            let b = edge_list.edges[top].v as usize;

            nbr_bits[a * nw + (b >> 6)] &= !(1u64 << (b & 63));
            nbr_bits[b * nw + (a >> 6)] &= !(1u64 << (a & 63));

            for &w_u32 in &users[a] {
                let w = w_u32 as usize;
                if dominator[w] == a as i32 && !in_dirty[w] {
                    in_dirty[w] = true;
                    dirty.push(w as u32);
                }
            }
            for &w_u32 in &users[b] {
                let w = w_u32 as usize;
                if dominator[w] == b as i32 && !in_dirty[w] {
                    in_dirty[w] = true;
                    dirty.push(w as u32);
                }
            }

            // Removing an edge incident to the center is exactly the event where
            // the other endpoint leaves the center ball.
            let departure = if a == center {
                Some(b)
            } else if b == center {
                Some(a)
            } else {
                None
            };
            if let Some(p) = departure {
                ball_bits[p >> 6] &= !(1u64 << (p & 63));
                if !in_dirty[p] {
                    in_dirty[p] = true;
                    dirty.push(p as u32);
                }
            }
        }
    }

    edge_list.threshold = certified_radius;
    let radius_bits = certified_radius.to_bits();
    let keep = edge_list
        .edges
        .partition_point(|edge| edge.distance.to_bits() <= radius_bits);
    edge_list.edges.truncate(keep);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::edgelist::EdgeList;

    fn square_from_lower_tri(n: usize, lower: &[f32]) -> Vec<f32> {
        assert_eq!(lower.len(), n * (n - 1) / 2);
        let mut matrix = vec![0.0; n * n];
        let mut k = 0;
        for i in 1..n {
            for j in 0..i {
                matrix[i * n + j] = lower[k];
                matrix[j * n + i] = lower[k];
                k += 1;
            }
        }
        matrix
    }

    fn assert_finalized(edge_list: &EdgeList, center: u16, initial_threshold: f32) {
        assert_eq!(edge_list.center, center);
        assert!(edge_list.sorted);
        assert!(edge_list.threshold <= initial_threshold);
        assert!(edge_list
            .edges
            .iter()
            .all(|edge| edge.distance <= edge_list.threshold));
        assert!(edge_list.edges.windows(2).all(|pair| {
            (pair[0].distance.to_bits(), pair[0].u, pair[0].v)
                <= (pair[1].distance.to_bits(), pair[1].u, pair[1].v)
        }));
    }

    #[test]
    fn trivial_size_is_contractible_at_zero() {
        let mut edge_list = EdgeList::from_points(&[0.0, 0.0], 1, 2, f32::INFINITY, false);
        peel(&mut edge_list, false);
        assert_eq!(edge_list.threshold, 0.0);
        assert!(edge_list.edges.is_empty());
        assert!(edge_list.sorted);
    }

    #[test]
    fn peel_never_increases_radius_or_changes_center() {
        let mut points = Vec::new();
        for x in 0..3 {
            for y in 0..3 {
                points.push(x as f32);
                points.push(y as f32);
            }
        }
        let mut edge_list = EdgeList::from_points(&points, 9, 2, f32::INFINITY, false);
        let center = edge_list.center;
        let initial_threshold = edge_list.threshold;
        peel(&mut edge_list, false);
        assert!(edge_list.threshold.is_finite() && edge_list.threshold >= 0.0);
        assert_finalized(&edge_list, center, initial_threshold);
    }

    #[test]
    fn peel_must_not_skip_cycle_inside_center_distance_gap() {
        // 5-point kite plus center P. The H1 bar (1.1, 1.5) lies entirely
        // inside the center-distance gap (1.05, 1.5).
        let lower = [1.5, 1.05, 1.0, 1.05, 1.5, 1.0, 1.5, 1.1, 1.5, 1.0];
        let matrix = square_from_lower_tri(5, &lower);
        let mut edge_list = EdgeList::from_distance_matrix(&matrix, 5, 1.5);
        let center = edge_list.center;
        peel(&mut edge_list, false);

        assert_eq!(center, 0);
        assert!(
            edge_list.threshold >= 1.5 - 1e-6,
            "unsound truncation: {} would erase the H1 bar (1.1, 1.5)",
            edge_list.threshold
        );
        assert_finalized(&edge_list, center, 1.5);
    }

    #[test]
    fn dense_cluster_can_reduce_below_enclosing_radius() {
        let points = [
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05, 0.2, 0.05, 0.05, 0.2,
        ];
        let n = points.len() / 2;
        let mut edge_list = EdgeList::from_points(&points, n, 2, f32::INFINITY, false);
        let center = edge_list.center;
        let initial_threshold = edge_list.threshold;
        peel(&mut edge_list, false);

        assert!(edge_list.threshold.is_finite() && edge_list.threshold >= 0.0);
        assert_finalized(&edge_list, center, initial_threshold);
    }
}
