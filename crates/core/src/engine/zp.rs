//! zp.rs — mod-p persistent cohomology for H¹ (the circular-coordinates
//! backend).
//!
//! A sibling of the Z/2 reduction, NOT a replacement: the flagship engine
//! stays byte-identical, and this module reuses everything that is
//! field-independent — BitCSR, the Kruskal/apparent-pair front half
//! ([`compute_h0`]), the emergent- and apparent-pair shortcuts (pair validity
//! depends only on filtration order; the pairing coefficient is ±1, a unit in
//! any field), and the generic [`FastHeap`].
//!
//! What is field-specific:
//!   * entries carry a coefficient ([`ZpEntry`]): the derived lexicographic
//!     `Ord` compares the simplex first, so equal keys are adjacent in heap
//!     order — exactly what pivot accumulation needs — and the coefficient
//!     never influences WHICH entry is the pivot;
//!   * cofacet emission is signed: the coefficient of σ in ∂τ is (−1)^rank,
//!     and the insertion rank is already computed by the cofacet builder, so
//!     the sign is free;
//!   * pivot extraction sums coefficients of equal keys mod p (a zero sum
//!     cancels the key entirely);
//!   * column addition is scaled: add λ·column with λ = −c/cₖ mod p.
//!
//! Cocycles are always recorded here — they are the entire point of this
//! path — so there is no `RECORD` switch.

use super::algorithm::{assemble_candidates, compute_h0, zero_apparent_facet};
use super::bitcsr::BitCsrDistanceMatrix;
use super::heap::FastHeap;
use super::reduction::RawRepresentative;
use super::simplex::{encode_filtration, CofacetBuilder, FxHashMap, PackedSimplex};
use crate::types::PersistenceInterval;

// ═══════════════════════════════════════════════════════════════════════════════
// Entries and modular arithmetic
// ═══════════════════════════════════════════════════════════════════════════════

/// A packed simplex key plus its Z/p coefficient in `1..p`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ZpEntry<S> {
    pub(crate) simplex: S,
    pub(crate) coeff: u16,
}

#[inline(always)]
fn mul_mod(a: u32, b: u32, p: u32) -> u32 {
    ((a as u64 * b as u64) % p as u64) as u32
}

#[inline(always)]
fn neg_mod(a: u32, p: u32) -> u32 {
    debug_assert!(a > 0 && a < p);
    p - a
}

fn pow_mod(mut base: u32, mut exp: u32, p: u32) -> u32 {
    let mut acc = 1u32;
    base %= p;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul_mod(acc, base, p);
        }
        base = mul_mod(base, base, p);
        exp >>= 1;
    }
    acc
}

/// Modular inverse by Fermat (`p` prime, `a` nonzero mod p). One call per
/// column addition — never hot.
#[inline]
fn inv_mod(a: u32, p: u32) -> u32 {
    pow_mod(a, p - 2, p)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Signed coboundary
// ═══════════════════════════════════════════════════════════════════════════════

/// Mirror of `for_each_cofacet(all_cofacets = true)` that also yields the
/// sign: with vertices written descending and ∂ = Σᵢ (−1)^i · (omit slot i),
/// the coefficient of σ in ∂τ is (−1)^rank of the inserted vertex — delivered
/// as `1` or `p − 1`. The rank advances monotonically with the youngest-first
/// enumeration, so the sign updates only on rank changes.
#[inline]
fn for_each_signed_cofacet<const K: usize, S: PackedSimplex<K>>(
    dist: &BitCsrDistanceMatrix,
    sigma: S,
    p: u32,
    mut f: impl FnMut(S, u32) -> bool,
) {
    let vc = sigma.vertex_count();
    if vc == 0 {
        return;
    }
    let verts = sigma.vertices();
    let base = sigma.filtration();
    let mut builder = sigma.cofacet_builder();
    let mut vi = 0usize;
    let mut sign = 1u32;

    dist.for_each_common_neighbor(&verts, vc, -1, |w, extra| {
        let old_vi = vi;
        while vi < vc && verts[vi] > w {
            vi += 1;
        }
        if vi != old_vi {
            builder.set_rank(vi);
            sign = if vi.is_multiple_of(2) { 1 } else { p - 1 };
        }
        let diam = if extra > base { extra } else { base };
        f(builder.build(w, encode_filtration(diam)), sign)
    });
}

/// Coefficient of facet φ in ∂τ, where τ = φ ∪ {w}: (−1)^k with k the slot
/// of the inserted vertex in τ's descending order.
#[inline]
fn facet_sign_in_cofacet<const K: usize, S: PackedSimplex<K>>(phi: S, tau: S, p: u32) -> u32 {
    let tv = tau.vertices();
    let pv = phi.vertices();
    let vc = tau.vertex_count();
    let mut k = 0usize;
    while k < vc - 1 && tv[k] == pv[k] {
        k += 1;
    }
    if k.is_multiple_of(2) {
        1
    } else {
        p - 1
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Heap pivot extraction and Z/p cancellation
// ═══════════════════════════════════════════════════════════════════════════════

fn pop_pivot<S: Copy + Ord + std::fmt::Debug>(
    heap: &mut FastHeap<ZpEntry<S>>,
    p: u32,
) -> Option<ZpEntry<S>> {
    loop {
        let top = heap.pop()?;
        let mut acc = top.coeff as u32;
        while let Some(&next) = heap.peek() {
            if next.simplex != top.simplex {
                break;
            }
            heap.pop();
            acc = (acc + next.coeff as u32) % p;
        }
        if !acc.is_multiple_of(p) {
            return Some(ZpEntry {
                simplex: top.simplex,
                coeff: (acc % p) as u16,
            });
        }
    }
}

fn get_pivot<S: Copy + Ord + std::fmt::Debug>(
    heap: &mut FastHeap<ZpEntry<S>>,
    p: u32,
) -> Option<ZpEntry<S>> {
    let pivot = pop_pivot(heap, p)?;
    heap.push(pivot);
    Some(pivot)
}

/// Sort by simplex, sum coefficients of equal keys mod p, emit the nonzero
/// survivors. The Z/p analogue of the reduction's `for_each_cancelled`.
fn cancel_mod_p<S: Copy + Ord>(working: &mut [ZpEntry<S>], p: u32, mut f: impl FnMut(ZpEntry<S>)) {
    working.sort_unstable_by_key(|entry| entry.simplex);
    let mut i = 0usize;
    while i < working.len() {
        let simplex = working[i].simplex;
        let mut acc = 0u32;
        while i < working.len() && working[i].simplex == simplex {
            acc = (acc + working[i].coeff as u32) % p;
            i += 1;
        }
        if acc != 0 {
            f(ZpEntry {
                simplex,
                coeff: acc as u16,
            });
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// The reduction
// ═══════════════════════════════════════════════════════════════════════════════

/// Persistent cohomology of ONE dimension over Z/p with a representative
/// cocycle for every EMITTED interval, aligned index-for-index. Cocycle
/// entries are (simplex, coeff mod p) with the column simplex first at
/// coefficient 1.
pub(crate) struct DimCocycles<S> {
    pub(crate) intervals: Vec<PersistenceInterval>,
    pub(crate) cocycles: Vec<Vec<ZpEntry<S>>>,
}

pub(crate) fn h1_cocycles<const K: usize, S: PackedSimplex<K>>(
    dist: &BitCsrDistanceMatrix,
    p: u32,
    parallel: bool,
) -> DimCocycles<S> {
    // Field-independent front half: Kruskal H0 supplies the apparent-pair
    // filtered dim-1 columns in reduction order. H0 output itself is unused.
    let mut h0 = Vec::new();
    let mut h0_reps: Vec<RawRepresentative<S>> = Vec::new();
    let (_edges, columns) = compute_h0::<K, S, false>(dist, &mut h0, &mut h0_reps, parallel);
    reduce_columns(&columns, dist, p).0
}

/// H² over Z/p: reduce dimension 1 (for its pivot map), clear, assemble the
/// dimension-2 columns through the shared field-independent machinery, and
/// reduce those. Clearing with the Z/p pivots is exactly as valid as with
/// the Z/2 ones — a recorded pivot is a death simplex over THIS field.
pub(crate) fn h2_cocycles<const K: usize, S: PackedSimplex<K>>(
    dist: &BitCsrDistanceMatrix,
    p: u32,
    parallel: bool,
) -> DimCocycles<S> {
    let mut h0 = Vec::new();
    let mut h0_reps: Vec<RawRepresentative<S>> = Vec::new();
    let (mut edges, columns) = compute_h0::<K, S, false>(dist, &mut h0, &mut h0_reps, parallel);

    let (_, dim1_pivots) = reduce_columns(&columns, dist, p);
    // `assemble_candidates` only consults the key set.
    let mut cleared: FxHashMap<S, usize> =
        FxHashMap::with_capacity_and_hasher(dim1_pivots.len(), Default::default());
    for (&pivot, &(column, _)) in &dim1_pivots {
        cleared.insert(pivot, column);
    }
    let columns2 = assemble_candidates(dist, &mut edges, &cleared, false, parallel);
    reduce_columns(&columns2, dist, p).0
}

fn reduce_columns<const K: usize, S: PackedSimplex<K>>(
    columns: &[S],
    dist: &BitCsrDistanceMatrix,
    p: u32,
) -> (DimCocycles<S>, FxHashMap<S, (usize, u32)>) {
    let mut intervals = Vec::new();
    let mut cocycles = Vec::new();
    // pivot simplex → (column index, pivot coefficient in the reduced column)
    let mut pivot_column_index: FxHashMap<S, (usize, u32)> =
        FxHashMap::with_capacity_and_hasher(columns.len(), Default::default());
    if columns.is_empty() {
        return (
            DimCocycles {
                intervals,
                cocycles,
            },
            pivot_column_index,
        );
    }

    // V-matrix storage: column k's entries live in
    // `v_entries[v_starts[k]..v_starts[k + 1]]` (start pushed as each column
    // begins, so ranges for k < j are always complete while j reduces).
    let mut v_entries: Vec<ZpEntry<S>> = Vec::new();
    let mut v_starts: Vec<usize> = Vec::with_capacity(columns.len() + 1);

    let cap = (columns.len() * 4).clamp(1024, 1 << 20);
    let mut working_v: Vec<ZpEntry<S>> = Vec::with_capacity(cap);
    let mut working_coboundary: FastHeap<ZpEntry<S>> = FastHeap::with_capacity(cap);

    for (j, &sigma) in columns.iter().enumerate() {
        working_v.clear();
        working_coboundary.clear();
        v_starts.push(v_entries.len());

        let mut pivot = init_pivot(sigma, dist, p, &pivot_column_index, &mut working_coboundary);

        if working_coboundary.is_empty() {
            // Emergent pair (zero persistence, never emitted) or no cofacets.
            match pivot {
                Some(entry) => {
                    pivot_column_index.insert(entry.simplex, (j, entry.coeff as u32));
                    continue;
                }
                None => {
                    intervals.push(PersistenceInterval {
                        birth: sigma.filtration(),
                        death: f32::INFINITY,
                    });
                    cocycles.push(vec![ZpEntry {
                        simplex: sigma,
                        coeff: 1,
                    }]);
                    continue;
                }
            }
        }

        loop {
            match pivot {
                None => {
                    intervals.push(PersistenceInterval {
                        birth: sigma.filtration(),
                        death: f32::INFINITY,
                    });
                    let mut cocycle = vec![ZpEntry {
                        simplex: sigma,
                        coeff: 1,
                    }];
                    cancel_mod_p(&mut working_v, p, |e| cocycle.push(e));
                    cocycles.push(cocycle);
                    break;
                }
                Some(entry) => {
                    let tau = entry.simplex;
                    if let Some(&(k, ck)) = pivot_column_index.get(&tau) {
                        let lambda = mul_mod(neg_mod(entry.coeff as u32, p), inv_mod(ck, p), p);
                        add_simplex_coboundary(
                            columns[k],
                            lambda,
                            dist,
                            p,
                            &mut working_v,
                            &mut working_coboundary,
                        );
                        for &stored in &v_entries[v_starts[k]..v_starts[k + 1]] {
                            let scaled = mul_mod(lambda, stored.coeff as u32, p);
                            debug_assert!(scaled != 0);
                            add_simplex_coboundary(
                                stored.simplex,
                                scaled,
                                dist,
                                p,
                                &mut working_v,
                                &mut working_coboundary,
                            );
                        }
                        pivot = get_pivot(&mut working_coboundary, p);
                        continue;
                    }

                    if let Some(phi) = zero_apparent_facet(dist, tau) {
                        // δφ hits τ with coefficient (−1)^rank; scale so the
                        // pivot cancels: λ = −c / sign.
                        let sign = facet_sign_in_cofacet(phi, tau, p);
                        let lambda = mul_mod(neg_mod(entry.coeff as u32, p), inv_mod(sign, p), p);
                        add_simplex_coboundary(
                            phi,
                            lambda,
                            dist,
                            p,
                            &mut working_v,
                            &mut working_coboundary,
                        );
                        pivot = get_pivot(&mut working_coboundary, p);
                        continue;
                    }

                    let emitted = tau.filtration_encoded() != sigma.filtration_encoded();
                    if emitted {
                        intervals.push(PersistenceInterval {
                            birth: sigma.filtration(),
                            death: tau.filtration(),
                        });
                    }
                    pivot_column_index.insert(tau, (j, entry.coeff as u32));

                    let column_start = v_entries.len();
                    cancel_mod_p(&mut working_v, p, |e| v_entries.push(e));
                    if emitted {
                        let mut cocycle = Vec::with_capacity(v_entries.len() - column_start + 1);
                        cocycle.push(ZpEntry {
                            simplex: sigma,
                            coeff: 1,
                        });
                        cocycle.extend_from_slice(&v_entries[column_start..]);
                        cocycles.push(cocycle);
                    }
                    break;
                }
            }
        }
    }

    (
        DimCocycles {
            intervals,
            cocycles,
        },
        pivot_column_index,
    )
}

/// Push λ·δ(simplex) into the working coboundary and remember (simplex, λ)
/// in the working V-column.
#[inline]
fn add_simplex_coboundary<const K: usize, S: PackedSimplex<K>>(
    simplex: S,
    lambda: u32,
    dist: &BitCsrDistanceMatrix,
    p: u32,
    working_v: &mut Vec<ZpEntry<S>>,
    working_coboundary: &mut FastHeap<ZpEntry<S>>,
) {
    debug_assert!(lambda > 0 && lambda < p);
    working_v.push(ZpEntry {
        simplex,
        coeff: lambda as u16,
    });
    for_each_signed_cofacet(dist, simplex, p, |cofacet, sign| {
        working_coboundary.push(ZpEntry {
            simplex: cofacet,
            coeff: mul_mod(lambda, sign, p) as u16,
        });
        true
    });
}

/// Initial pivot with the emergent-pair shortcut (field-independent: the
/// emergent cofacet occurs once in δσ, with a unit coefficient).
fn init_pivot<const K: usize, S: PackedSimplex<K>>(
    sigma: S,
    dist: &BitCsrDistanceMatrix,
    p: u32,
    pivot_column_index: &FxHashMap<S, (usize, u32)>,
    working_coboundary: &mut FastHeap<ZpEntry<S>>,
) -> Option<ZpEntry<S>> {
    let sigma_filt = sigma.filtration_encoded();
    let mut check_for_emergent_pair = true;
    let mut emergent: Option<ZpEntry<S>> = None;

    for_each_signed_cofacet(dist, sigma, p, |cofacet, sign| {
        if check_for_emergent_pair && cofacet.filtration_encoded() == sigma_filt {
            if !pivot_column_index.contains_key(&cofacet)
                && zero_apparent_facet(dist, cofacet).is_none()
            {
                working_coboundary.clear(); // discard pollution
                emergent = Some(ZpEntry {
                    simplex: cofacet,
                    coeff: sign as u16,
                });
                return false;
            }
            check_for_emergent_pair = false;
        }
        working_coboundary.push(ZpEntry {
            simplex: cofacet,
            coeff: sign as u16,
        });
        true
    });

    if emergent.is_some() {
        return emergent;
    }

    working_coboundary.heapify();
    get_pivot(working_coboundary, p)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::algorithm::compute;
    use crate::engine::simplex::{Simplex128, Simplex256};
    use crate::preprocess::edgelist::{square_from_lower_tri, EdgeList};
    use std::collections::HashMap;

    fn fixture(n: usize, seed: u64, threshold: f32) -> BitCsrDistanceMatrix {
        let mut s = seed;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 16_777_216.0
        };
        let m = n * (n - 1) / 2;
        let lt: Vec<f32> = (0..m).map(|_| 0.2 + rng()).collect();
        let sq = square_from_lower_tri(n, &lt);
        BitCsrDistanceMatrix::from_edge_list(EdgeList::from_distance_matrix(&sq, n, threshold))
    }

    fn interval_key(iv: &PersistenceInterval) -> (u32, u32) {
        (iv.birth.to_bits(), iv.death.to_bits())
    }

    /// On torsion-free inputs (generic random fixtures) the H¹ diagram is
    /// field-independent: mod p must reproduce the Z/2 engine bit-for-bit as
    /// a multiset, across primes and both packings.
    #[test]
    fn zp_h1_matches_z2_h1() {
        for &(n, seed) in &[(8usize, 11u64), (10, 42), (12, 2026), (14, 7)] {
            let bitcsr = fixture(n, seed, f32::INFINITY);
            let z2 = compute::<6, Simplex128>(&bitcsr, 1, false);
            let mut want: Vec<_> = z2.intervals[1].iter().map(interval_key).collect();
            want.sort_unstable();

            for &p in &[3u32, 41, 65521] {
                let zp = h1_cocycles::<6, Simplex128>(&bitcsr, p, false);
                let mut got: Vec<_> = zp.intervals.iter().map(interval_key).collect();
                got.sort_unstable();
                assert_eq!(got, want, "n={n} seed={seed} p={p}");
                assert_eq!(zp.cocycles.len(), zp.intervals.len());
            }

            // The 256-bit packing must agree entry-for-entry.
            let wide = h1_cocycles::<9, Simplex256>(&bitcsr, 41, false);
            let mut got: Vec<_> = wide.intervals.iter().map(interval_key).collect();
            got.sort_unstable();
            assert_eq!(got, want, "Simplex256 packing, n={n} seed={seed}");
        }
    }

    /// The dimension-2 path (clearing + assembly + reduction) must reproduce
    /// the Z/2 engine's H² diagram on torsion-free fixtures, across primes
    /// and packings.
    #[test]
    fn zp_h2_matches_z2_h2() {
        for &(n, seed) in &[(8usize, 11u64), (10, 42), (12, 2026)] {
            let bitcsr = fixture(n, seed, f32::INFINITY);
            let z2 = compute::<6, Simplex128>(&bitcsr, 2, false);
            let mut want: Vec<_> = z2.intervals[2].iter().map(interval_key).collect();
            want.sort_unstable();

            for &p in &[3u32, 41] {
                let zp = h2_cocycles::<6, Simplex128>(&bitcsr, p, false);
                let mut got: Vec<_> = zp.intervals.iter().map(interval_key).collect();
                got.sort_unstable();
                assert_eq!(got, want, "n={n} seed={seed} p={p}");
                assert_eq!(zp.cocycles.len(), zp.intervals.len());
            }

            let wide = h2_cocycles::<9, Simplex256>(&bitcsr, 41, false);
            let mut got: Vec<_> = wide.intervals.iter().map(interval_key).collect();
            got.sort_unstable();
            assert_eq!(got, want, "Simplex256 packing, n={n} seed={seed}");
        }
    }

    /// The dimension-2 cocycles satisfy the same pivot property as the
    /// dimension-1 ones: δ(cocycle) mod p survives exactly at diameters
    /// >= death with the death simplex as the youngest survivor.
    #[test]
    fn zp_h2_cocycles_are_valid() {
        let p = 41u32;
        for &(n, seed) in &[(9usize, 5u64), (11, 99)] {
            let bitcsr = fixture(n, seed, f32::INFINITY);
            let zp = h2_cocycles::<6, Simplex128>(&bitcsr, p, false);
            for (iv, cocycle) in zp.intervals.iter().zip(&zp.cocycles) {
                assert_eq!(cocycle[0].coeff, 1);
                assert_eq!(cocycle[0].simplex.filtration(), iv.birth);
                assert!(cocycle.iter().all(|e| e.simplex.vertex_count() == 3));

                let mut sums: HashMap<Simplex128, u32> = HashMap::new();
                for entry in cocycle {
                    for_each_signed_cofacet::<6, Simplex128>(
                        &bitcsr,
                        entry.simplex,
                        p,
                        |cofacet, sign| {
                            let term = mul_mod(entry.coeff as u32, sign, p);
                            let slot = sums.entry(cofacet).or_insert(0);
                            *slot = (*slot + term) % p;
                            true
                        },
                    );
                }
                let surviving: Vec<Simplex128> = sums
                    .into_iter()
                    .filter_map(|(s, c)| (c % p != 0).then_some(s))
                    .collect();

                if iv.death.is_finite() {
                    let tau = *surviving.iter().max().expect("paired class lost its pivot");
                    assert_eq!(tau.filtration(), iv.death);
                    assert!(surviving.iter().all(|c| c.filtration() >= iv.death));
                } else {
                    assert!(surviving.is_empty());
                }
            }
        }
    }

    /// PROPERTY: every recorded cocycle is a valid representative over Z/p.
    /// Summing the signed coboundaries of its entries mod p, a paired class
    /// leaves the death simplex as its youngest survivor with everything at
    /// diameter >= death; an essential class (finite threshold) leaves
    /// nothing.
    #[test]
    fn zp_cocycles_are_valid() {
        let p = 41u32;
        // Random fixtures plus the unfilled 4-cycle at a user threshold
        // (essential H1).
        let mut cases: Vec<BitCsrDistanceMatrix> =
            vec![fixture(9, 5, f32::INFINITY), fixture(12, 99, f32::INFINITY)];
        let lower = [1.0f32, 10.0, 1.0, 1.0, 10.0, 1.0];
        let sq = square_from_lower_tri(4, &lower);
        cases.push(BitCsrDistanceMatrix::from_edge_list(
            EdgeList::from_distance_matrix(&sq, 4, 1.5),
        ));

        for bitcsr in &cases {
            let zp = h1_cocycles::<6, Simplex128>(bitcsr, p, false);
            for (iv, cocycle) in zp.intervals.iter().zip(&zp.cocycles) {
                assert_eq!(cocycle[0].coeff, 1);
                assert_eq!(cocycle[0].simplex.filtration(), iv.birth);

                let mut sums: HashMap<Simplex128, u32> = HashMap::new();
                for entry in cocycle {
                    for_each_signed_cofacet::<6, Simplex128>(
                        bitcsr,
                        entry.simplex,
                        p,
                        |cofacet, sign| {
                            let term = mul_mod(entry.coeff as u32, sign, p);
                            let slot = sums.entry(cofacet).or_insert(0);
                            *slot = (*slot + term) % p;
                            true
                        },
                    );
                }
                let surviving: Vec<Simplex128> = sums
                    .into_iter()
                    .filter_map(|(s, c)| (c % p != 0).then_some(s))
                    .collect();

                if iv.death.is_finite() {
                    let tau = *surviving.iter().max().expect("paired class lost its pivot");
                    assert_eq!(tau.filtration(), iv.death);
                    assert!(surviving.iter().all(|c| c.filtration() >= iv.death));
                } else {
                    assert!(
                        surviving.is_empty(),
                        "essential class has a nonzero coboundary mod {p}"
                    );
                }
            }
        }
    }

    #[test]
    fn modular_helpers() {
        let p = 41;
        for a in 1..p {
            assert_eq!(mul_mod(a, inv_mod(a, p), p), 1, "a = {a}");
            assert_eq!((a + neg_mod(a, p)) % p, 0);
        }
        assert_eq!(pow_mod(7, 0, p), 1);
    }
}
