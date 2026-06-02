// Packed 128-bit simplex + cofacet/facet iterators + apparent-pair primitives.
//
// LAYOUT  high 32 bits: encode_filtration(diam) = !diam.to_bits()
//         low  96 bits: 6 vertex slots, largest vertex first, each stored as id+1 (0 = empty)
//
// ORDERING  Simplex128 derives Ord, i.e. comparison IS raw u128 lexicographic
//   comparison. encode_filtration inverts the float bits ON PURPOSE: it lets us
//   keep the derived (and therefore trivially inlinable) Ord in the heap's hot
//   sift loop while still ordering a LARGER diameter as a SMALLER u128. So:
//     larger u128  ==  smaller diameter, or (equal diameter) larger vertex set.
//   The heap is a max-heap; the pivot is the max u128. CofacetIter yields the
//   max-u128 same-diameter cofacet first; FacetIter the min-u128 same-diameter
//   facet first. The exact pivot semantics are pinned by the tests
//   `pivot_primitives_match_ripser_first_yielded` and
//   `zero_apparent_facet_matches_bruteforce_on_tetrahedra` — trust those, not prose.

use std::hash::{BuildHasherDefault, Hasher};

use crate::engine::distance::{
    cofacet_diameter, cofacet_diameter_v_larger, facet_diameter, DistanceMatrix,
};

#[inline(always)]
pub fn encode_filtration(f: f32) -> u32 {
    debug_assert!(f >= 0.0, "Distances must be non-negative");
    !f.to_bits() // flip
}

#[inline(always)]
pub fn decode_filtration(u: u32) -> f32 {
    f32::from_bits(!u) // unflip
}
// ═══════════════════════════════════════════════════════════════════════════════
// Simplex128
// ═══════════════════════════════════════════════════════════════════════════════

/// Mask for the vertex payload (bits 0–95).
const VERTEX_MASK: u128 = (1u128 << 96) - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(C, align(16))]
pub struct Simplex128(pub u128);

impl Simplex128 {
    /// Sentinel for "no simplex" / invalid.
    pub const INVALID: Simplex128 = Simplex128(0);

    /// Create from a filtration value (f32) and an unsorted vertex slice.
    /// Vertices are sorted descending internally.
    #[inline]
    pub fn new(filtration: f32, vertices: &[u16]) -> Self {
        let mut sorted = [0u16; 6];
        let len = vertices.len().min(6);
        sorted[..len].copy_from_slice(&vertices[..len]);
        sorted[..len].sort_unstable_by(|a, b| b.cmp(a));
        Self::from_sorted_desc(filtration, &sorted[..len])
    }

    /// Create from an encoded filtration (u32) and a DESCENDING-sorted vertex slice.
    /// Caller must guarantee descending order — no sort is performed.
    #[inline(always)]
    pub fn from_sorted_desc(filtration: f32, vertices: &[u16]) -> Self {
        let len = vertices.len().min(6);
        let mut p: u128 = (encode_filtration(filtration) as u128) << 96;
        // Unrolled packing. Each vertex stored as (id + 1) to reserve 0 = empty.
        if len > 0 {
            p |= ((vertices[0] as u128) + 1) << 80;
        }
        if len > 1 {
            p |= ((vertices[1] as u128) + 1) << 64;
        }
        if len > 2 {
            p |= ((vertices[2] as u128) + 1) << 48;
        }
        if len > 3 {
            p |= ((vertices[3] as u128) + 1) << 32;
        }
        if len > 4 {
            p |= ((vertices[4] as u128) + 1) << 16;
        }
        if len > 5 {
            p |= (vertices[5] as u128) + 1;
        }
        Simplex128(p)
    }

    /// Encoded filtration value (u32).
    #[inline(always)]
    pub fn filtration_encoded(self) -> u32 {
        (self.0 >> 96) as u32
    }

    /// Filtration value as f32.
    #[inline(always)]
    pub fn filtration(self) -> f32 {
        decode_filtration(self.filtration_encoded())
    }

    /// Raw slot value at position k (0 = MSB slot, 5 = LSB slot).
    /// Returns 0 for empty, otherwise vertex_id + 1.
    #[inline(always)]
    fn slot(self, k: usize) -> u16 {
        ((self.0 >> (80 - k * 16)) & 0xFFFF) as u16
    }

    /// Number of vertices (= dimension + 1).
    /// Exploits the invariant that slots are packed contiguously from slot 0.
    #[inline(always)]
    pub fn vertex_count(self) -> usize {
        // Unrolled: first zero slot means all subsequent are zero.
        if self.slot(0) == 0 {
            return 0;
        }
        if self.slot(1) == 0 {
            return 1;
        }
        if self.slot(2) == 0 {
            return 2;
        }
        if self.slot(3) == 0 {
            return 3;
        }
        if self.slot(4) == 0 {
            return 4;
        }
        if self.slot(5) == 0 {
            return 5;
        }
        6
    }

    /// Dimension of the simplex (vertex_count − 1).
    #[inline(always)]
    pub fn dim(self) -> usize {
        self.vertex_count().saturating_sub(1)
    }

    /// Extract vertex IDs into a fixed-size array (descending order).
    /// Only the first `vertex_count()` entries are meaningful.
    #[inline(always)]
    pub fn vertices(self) -> [u16; 6] {
        [
            self.slot(0).wrapping_sub(1),
            self.slot(1).wrapping_sub(1),
            self.slot(2).wrapping_sub(1),
            self.slot(3).wrapping_sub(1),
            self.slot(4).wrapping_sub(1),
            self.slot(5).wrapping_sub(1),
        ]
    }

    /// Largest vertex ID (slot 0).
    #[inline(always)]
    pub fn largest_vertex(self) -> u16 {
        debug_assert!(self.slot(0) != 0, "Empty simplex has no largest vertex");
        self.slot(0) - 1
    }

    /// Vertex payload only (bits 0–95), ignoring filtration.
    /// Two simplices with the same vertex_key have the same vertex set.
    #[inline(always)]
    pub fn vertex_key(self) -> u128 {
        self.0 & VERTEX_MASK
    }

    /// Check whether vertex v is contained in this simplex.
    /// Uses descending sort order for early exit.
    #[inline(always)]
    pub fn contains_vertex(self, v: u16) -> bool {
        let target = (v as u128) + 1;
        let p = self.0;
        macro_rules! check_slot {
            ($shift:expr) => {{
                let s = (p >> $shift) & 0xFFFF;
                if s == target {
                    return true;
                }
                if s < target {
                    return false;
                }
            }};
        }
        check_slot!(80);
        check_slot!(64);
        check_slot!(48);
        check_slot!(32);
        check_slot!(16);
        (p & 0xFFFF) == target
    }

    // ── Cofacet construction ────────────────────────────────────────────────

    /// Insert `new_vertex` into this simplex with a new filtration value.
    /// The vertex is inserted into the correct sorted (descending) position.
    ///
    /// This is branchless on the insertion rank: we compute the rank (number
    /// of existing slots > new_vertex), split the payload, and recombine.
    #[inline(always)]
    pub fn cofacet(self, new_vertex: u16, new_filt: u32) -> Self {
        let payload = self.0 & VERTEX_MASK;
        let v_ins = (new_vertex as u128) + 1;

        // Count how many existing slots have value > v_ins.
        // This gives the insertion index (rank) in the descending array.
        let mut rank: u32 = 0;
        if ((payload >> 80) & 0xFFFF) > v_ins {
            rank += 1;
        }
        if ((payload >> 64) & 0xFFFF) > v_ins {
            rank += 1;
        }
        if ((payload >> 48) & 0xFFFF) > v_ins {
            rank += 1;
        }
        if ((payload >> 32) & 0xFFFF) > v_ins {
            rank += 1;
        }
        if ((payload >> 16) & 0xFFFF) > v_ins {
            rank += 1;
        }

        // Split at the insertion point.
        let split = 96 - (rank * 16); // bit offset of the split boundary
        let upper_mask = (!0u128) << split;

        let mut result = (new_filt as u128) << 96;
        result |= payload & upper_mask; // slots above insertion point (unchanged)
        result |= v_ins << (split - 16); // new vertex in the gap
        result |= (payload & !upper_mask) >> 16; // slots below, shifted right by one position
        Simplex128(result)
    }

    // ── Facet construction ──────────────────────────────────────────────────

    /// Remove the vertex at slot `k` and set a new filtration value.
    /// Slots below `k` shift up (left) to close the gap.
    #[inline(always)]
    pub fn remove_vertex_at(self, k: usize, new_filt: u32) -> Self {
        let payload = self.0 & VERTEX_MASK;
        let remove_bit = 80 - k * 16;
        let split = remove_bit + 16; // = 96 - k * 16

        // Everything above the removed slot stays in place.
        let upper = if split >= 96 {
            0u128
        } else {
            payload & ((!0u128) << split) & VERTEX_MASK
        };

        // Everything below shifts left by 16 bits (fills the gap).
        let lower = if remove_bit == 0 {
            0u128
        } else {
            (payload & ((1u128 << remove_bit) - 1)) << 16
        };

        Simplex128(((new_filt as u128) << 96) | upper | lower)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CofacetIter — enumerates cofacets in DESCENDING vertex order (j = n−1 → 0)
// ═══════════════════════════════════════════════════════════════════════════════
//
// RANK INVARIANT
// ──────────────
// At every yield point, `self.vi` equals the insertion rank of the new vertex
// in the descending-sorted vertex array. Proof: candidates `v` and `verts[]`
// both descend, and `vi` is advanced exactly when they match; so at a non-match
// yield, verts[0..vi] are all > v (consumed at larger v values) and verts[vi..]
// are all < v (not yet reached). The insertion position is therefore `vi`.
//
// We exploit this to build the cofacet without calling `Simplex128::cofacet`,
// which would otherwise do 5 compare-and-branch operations per yield to
// recompute `rank`. The cofacet-construction masks depend only on `vi`, so we
// cache them in the struct and refresh only when `vi` actually changes (at
// most (vc + 1) times per simplex, not per yield).

pub struct CofacetIter<'a> {
    dist: &'a DistanceMatrix,
    verts: [u16; 6],
    vc: usize,
    filt: f32,      // simplex's filtration as f32 (avoids repeated decode)
    v: i32,         // current candidate vertex (counts DOWN from n−1)
    vi: usize,      // index into verts[] = insertion rank (see invariant)
    stop: i32,      // iteration stops when v <= stop
    threshold: f32, // skip cofacets with diameter > threshold

    // ── Cached cofacet-construction state ───────────────────────────────────
    // Functions of (payload, vi). Updated in reset() and in the skip branch
    // of next(); NOT touched on plain yields.
    payload: u128,               // simplex.0 & VERTEX_MASK  (bits 0..96)
    payload_upper: u128,         // slots 0..vi in place     (unchanged by insert)
    payload_lower_shifted: u128, // slots vi..vc shifted right by 16 bits
    insert_shift: u32,           // bit offset at which to place the new slot
}

impl<'a> CofacetIter<'a> {
    /// Create a new cofacet iterator.
    ///
    /// - `all_cofacets = true`: enumerate ALL cofacets (j = n−1 downto 0).
    /// - `all_cofacets = false`: only cofacets where j > largest_vertex(σ),
    ///    ensuring each (d+1)-simplex is produced exactly once.
    #[inline]
    pub fn new(
        simplex: Simplex128,
        dist: &'a DistanceMatrix,
        all_cofacets: bool,
        threshold: f32,
    ) -> Self {
        let mut iter = CofacetIter {
            dist,
            verts: [0; 6],
            vc: 0,
            filt: 0.0,
            v: 0,
            vi: 0,
            stop: 0,
            threshold,
            payload: 0,
            payload_upper: 0,
            payload_lower_shifted: 0,
            insert_shift: 0,
        };
        iter.reset(simplex, all_cofacets);
        iter
    }

    /// Reset to iterate over a different simplex (avoids allocation).
    #[inline]
    pub fn reset(&mut self, simplex: Simplex128, all_cofacets: bool) {
        self.verts = simplex.vertices();
        self.vc = simplex.vertex_count();
        self.filt = simplex.filtration();
        self.payload = simplex.0 & VERTEX_MASK;
        self.v = (self.dist.n() as i32) - 1;
        self.vi = 0;
        self.stop = if all_cofacets {
            -1
        } else {
            simplex.largest_vertex() as i32
        };
        self.refresh_rank_cache();
    }

    /// Recompute the cached slot masks for the current insertion rank (= `self.vi`).
    /// Called in `reset()` and in the skip branch of `next()` — at most (vc + 1)
    /// times per simplex, NOT per yielded cofacet.
    #[inline(always)]
    fn refresh_rank_cache(&mut self) {
        // vi in [0, vc] with vc <= 5, so split in [16, 96] — always a valid
        // u128 shift amount (strictly < 128).
        let rank = self.vi as u32;
        let split = 96 - rank * 16; // bit offset just above the insertion slot
        let upper_mask = (!0u128) << split;
        self.payload_upper = self.payload & upper_mask;
        self.payload_lower_shifted = (self.payload & !upper_mask) >> 16;
        self.insert_shift = split - 16;
    }

    /// Advance to the next cofacet. Returns `None` when exhausted.
    ///
    /// Each returned `Simplex128` has its filtration set to the cofacet's
    /// diameter (encoded as u32).
    #[inline]
    pub fn next(&mut self) -> Option<Simplex128> {
        loop {
            if self.v <= self.stop {
                return None;
            }

            let v = self.v as u16;

            // Skip vertices already in the simplex. A match means the new
            // vertex would land one position deeper in the descending array —
            // i.e., the insertion rank grows by 1, so refresh the cache.
            if self.vi < self.vc && self.verts[self.vi] == v {
                self.vi += 1;
                self.v -= 1;
                self.refresh_rank_cache();
                continue;
            }

            self.v -= 1;

            // Compute cofacet diameter with early threshold exit.
            let diam = match cofacet_diameter(
                v,
                &self.verts,
                self.vc,
                self.filt,
                self.dist,
                self.threshold,
            ) {
                Some(d) => d,
                None => continue, // exceeded threshold
            };

            // Build the cofacet directly from cached state — no rank search,
            // no extra payload masking per yield. Layout:
            //
            //   [    filt (32b)    ][ payload_upper ][ v_ins << shift ][ lower_shifted ]
            //       bits 96..127     bits > split     new slot at rank   bits shifted
            //                                                            into r+1..vc
            let v_ins = (v as u128) + 1;
            let result = ((encode_filtration(diam) as u128) << 96)
                | self.payload_upper
                | (v_ins << self.insert_shift)
                | self.payload_lower_shifted;
            return Some(Simplex128(result));
        }
    }
    /// Internal iteration. Calls `f` on each cofacet; stop early by returning
    /// `false` from `f`. This is the hot-path API — `next()` is retained only
    /// for tests and external code that genuinely needs an Iterator shape.
    ///
    /// Splits iteration into two phases:
    ///   Phase 1: v > verts[0] (new vertex larger than all existing vertices)
    ///            — no skip check, no rank cache updates, fast diameter path.
    ///   Phase 2: v ≤ verts[0] — needs the full skip/rank machinery.
    ///
    /// For all_cofacets=false, stop = verts[0], so phase 2 never runs.
    #[inline(always)]
    pub fn for_each<F: FnMut(Simplex128) -> bool>(&mut self, mut f: F) {
        let pivot_v = if self.vc > 0 {
            self.verts[0] as i32
        } else {
            -1
        };

        // ── Phase 1: v > pivot_v ────────────────────────────────────────────
        // No skip check needed (v cannot equal any verts[i] because v > all of
        // them). Rank stays at 0, so payload_upper / payload_lower_shifted /
        // insert_shift remain at their reset() values for the entire phase.
        while self.v > pivot_v && self.v > self.stop {
            let v = self.v as u16;
            self.v -= 1;

            let diam = match cofacet_diameter_v_larger(
                v,
                &self.verts,
                self.vc,
                self.filt,
                self.dist,
                self.threshold,
            ) {
                Some(d) => d,
                None => continue,
            };

            let v_ins = (v as u128) + 1;
            let result = ((encode_filtration(diam) as u128) << 96)
                | self.payload_upper
                | (v_ins << self.insert_shift)
                | self.payload_lower_shifted;
            if !f(Simplex128(result)) {
                return;
            }
        }

        // ── Phase 2: v ≤ pivot_v ────────────────────────────────────────────
        loop {
            if self.v <= self.stop {
                return;
            }
            let v = self.v as u16;

            if self.vi < self.vc && self.verts[self.vi] == v {
                self.vi += 1;
                self.v -= 1;
                self.refresh_rank_cache();
                continue;
            }

            self.v -= 1;

            let diam = match cofacet_diameter(
                v,
                &self.verts,
                self.vc,
                self.filt,
                self.dist,
                self.threshold,
            ) {
                Some(d) => d,
                None => continue,
            };

            let v_ins = (v as u128) + 1;
            let result = ((encode_filtration(diam) as u128) << 96)
                | self.payload_upper
                | (v_ins << self.insert_shift)
                | self.payload_lower_shifted;
            if !f(Simplex128(result)) {
                return;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FacetIter — enumerates facets by removing vertices LARGEST-first
// ═══════════════════════════════════════════════════════════════════════════════
//
// Enumeration order: removes vertex at slot 0 first (largest vertex),
// then slot 1, etc., ending with slot (vc−1) (smallest vertex).
//
// This produces facets from OLDEST to YOUNGEST in our forward colexicographic
// ordering: removing the largest vertex yields a facet whose leading vertex
// is the second-largest, giving the smallest possible u128 (oldest). Removing
// the smallest vertex preserves the largest leading vertex (youngest).
//
// Key property: the FIRST same-diameter facet encountered is the OLDEST
// (smallest u128) — this is the "oldest same-diameter facet" in the
// apparent-pair definition (Proposition 3.9 of the Ripser paper).

pub struct FacetIter<'a> {
    dist: &'a DistanceMatrix,
    verts: [u16; 6],
    vc: usize,
    simplex: Simplex128,
    /// Next slot to remove. Starts at 0 (largest vertex), counts UP to vc.
    k: usize,
}

impl<'a> FacetIter<'a> {
    #[inline]
    pub fn new(simplex: Simplex128, dist: &'a DistanceMatrix) -> Self {
        FacetIter {
            dist,
            verts: simplex.vertices(),
            vc: simplex.vertex_count(),
            simplex,
            k: 0,
        }
    }

    /// Reset to iterate over a different simplex.
    #[inline]
    pub fn reset(&mut self, simplex: Simplex128) {
        self.verts = simplex.vertices();
        self.vc = simplex.vertex_count();
        self.simplex = simplex;
        self.k = 0;
    }

    /// Advance to the next facet. Returns `None` when exhausted.
    #[inline]
    pub fn next(&mut self) -> Option<Simplex128> {
        if self.k >= self.vc {
            return None;
        }

        let k = self.k;
        self.k += 1;

        let diam = facet_diameter(&self.verts, self.vc, k, self.dist);
        Some(self.simplex.remove_vertex_at(k, encode_filtration(diam)))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Apparent-pair primitives
// ═══════════════════════════════════════════════════════════════════════════════
//
// A pair (σ, τ) with dim τ = dim σ + 1 is a zero-persistence apparent pair
// in the lex-refined Rips filtration iff (Ripser paper, Proposition 3.9):
//
//   (C1) τ is the youngest same-diameter cofacet of σ
//   (C2) σ is the oldest  same-diameter facet   of τ
//
// Apparent pairs are persistence pairs independent of reduction (Lemma 3.3),
// and for dim ≥ 1 they have persistence zero, so they can be emitted or
// skipped without entering the reduction step.
//
// Cost: `zero_pivot_cofacet` does one pass of CofacetIter (early-exits on
// the first same-diameter hit). `zero_pivot_facet` does at most (dim+2) ≤ 7
// facet constructions, each costing ≤ O(dim²) distance lookups — all L1-hot.
// Total: cheaper than a single hashmap probe on a large pivot table.

/// Youngest same-diameter cofacet of σ, if one exists within `threshold`.
///
/// Iterates `CofacetIter` in all-cofacets mode and returns the first cofacet
/// whose encoded filtration equals σ's. Returns `None` if σ has no
/// same-diameter cofacet (or all such cofacets exceed `threshold`).
///
/// Compares encoded u32 filtrations (bit-exact), not decoded f32, to avoid
/// any f32 equality pitfalls.
#[inline]
pub fn zero_pivot_cofacet(
    simplex: Simplex128,
    dist: &DistanceMatrix,
    threshold: f32,
) -> Option<Simplex128> {
    let target = simplex.filtration_encoded();
    let mut result: Option<Simplex128> = None;
    let mut iter = CofacetIter::new(simplex, dist, true, threshold);
    iter.for_each(|cof| {
        if cof.filtration_encoded() == target {
            result = Some(cof);
            false
        } else {
            true
        }
    });
    result
}

/// Oldest same-diameter facet of τ, if one exists.
///
/// Iterates `FacetIter` (now oldest-first) and returns the first facet whose
/// encoded filtration equals τ's. Returns `None` only if τ has no
/// same-diameter facet, which cannot happen for a τ whose filtration equals
/// max pairwise distance over its vertex set (i.e., every valid simplex
/// produced by the filtration).
#[inline]
pub fn zero_pivot_facet(simplex: Simplex128, dist: &DistanceMatrix) -> Option<Simplex128> {
    let target = simplex.filtration_encoded();
    let mut iter = FacetIter::new(simplex, dist);
    while let Some(facet) = iter.next() {
        if facet.filtration_encoded() == target {
            return Some(facet);
        }
    }
    None
}

/// If σ is the column side of a zero-persistence apparent pair, return its
/// cofacet partner τ. Otherwise return `None`.
///
/// Composes `zero_pivot_cofacet` (condition C1) with `zero_pivot_facet`
/// (condition C2) and checks that σ matches the oldest same-diameter facet
/// of the youngest same-diameter cofacet.
#[inline]
pub fn zero_apparent_cofacet(
    simplex: Simplex128,
    dist: &DistanceMatrix,
    threshold: f32,
) -> Option<Simplex128> {
    let tau = zero_pivot_cofacet(simplex, dist, threshold)?;
    let phi = zero_pivot_facet(tau, dist)?;
    if phi == simplex {
        Some(tau)
    } else {
        None
    }
}

/// True iff τ is the cofacet side of some zero-persistence apparent pair.
///
/// This is the hot-path predicate used by `assemble_candidates` to filter
/// out cofacets that are guaranteed to be paired with their oldest
/// same-diameter facet — replacing the hashmap `pivots.contains(&τ)` check.
///
/// Logic: find σ = oldest same-diameter facet of τ. If that σ's youngest
/// same-diameter cofacet is τ, then (σ, τ) is the apparent pair.
#[inline]
pub fn is_apparent_cofacet(tau: Simplex128, dist: &DistanceMatrix, threshold: f32) -> bool {
    let sigma = match zero_pivot_facet(tau, dist) {
        Some(s) => s,
        None => return false,
    };
    match zero_pivot_cofacet(sigma, dist, threshold) {
        Some(t) => t == tau,
        None => false,
    }
}

/// If τ is the cofacet side of a zero-persistence apparent pair, return its
/// facet partner σ. Otherwise return `None`.
///
/// This is the dual of `zero_apparent_cofacet` and is used inside the
/// reduction loop as a shortcut: when a pivot τ has no claimed column but
/// is part of an apparent pair (φ, τ), we can immediately add δ(φ) to cancel
/// τ without going through the hashmap or growing V_j unnecessarily.
///
/// Returns `Some(φ)` iff (φ, τ) is an apparent pair, where φ is the oldest
/// same-diameter facet of τ.
#[inline]
pub fn zero_apparent_facet(
    tau: Simplex128,
    dist: &DistanceMatrix,
    threshold: f32,
) -> Option<Simplex128> {
    let phi = zero_pivot_facet(tau, dist)?;
    let tau_check = zero_pivot_cofacet(phi, dist, threshold)?;
    if tau_check == tau {
        Some(phi)
    } else {
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FxHash — fast hash for Simplex128 (u128 keys)
// ═══════════════════════════════════════════════════════════════════════════════
//
// The default SipHash is too slow for the inner reduction loop where we do
// millions of hash lookups. FxHash-style multiply-xor folding on the u128
// gives excellent performance. Simplex128's lower 96 bits (vertex content)
// provide good entropy.
//
// Currently unused in the hot path (the apparent-pair check replaced the
// hashmap lookup), but retained for future non-emergent reduction where a
// pivot-column map is genuinely needed.

const FX_SEED: u64 = 0x517cc1b727220a95;

#[derive(Default)]
pub struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        // Fallback for non-u128 writes (shouldn't be hit for Simplex128).
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ (b as u64)).wrapping_mul(FX_SEED);
        }
    }

    #[inline(always)]
    fn write_u128(&mut self, i: u128) {
        let lo = i as u64;
        let hi = (i >> 64) as u64;
        self.0 = lo.wrapping_mul(FX_SEED) ^ hi;
    }
}

pub type FxBuildHasher = BuildHasherDefault<FxHasher>;
pub type FxHashMap<K, V> = std::collections::HashMap<K, V, FxBuildHasher>;
pub type FxHashSet<T> = std::collections::HashSet<T, FxBuildHasher>;
// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triangle_dist() -> DistanceMatrix {
        // 3 points: d(0,1)=1.0, d(0,2)=2.0, d(1,2)=1.5
        // Lower triangular: [(1,0)=1.0, (2,0)=2.0, (2,1)=1.5]
        DistanceMatrix::from_lower_triangular(3, vec![1.0, 2.0, 1.5])
    }

    #[test]
    fn test_simplex_basics() {
        let s = Simplex128::new(1.5, &[3, 1, 5]);
        assert_eq!(s.vertex_count(), 3);
        assert_eq!(s.dim(), 2);
        assert_eq!(s.largest_vertex(), 5);
        let v = s.vertices();
        assert_eq!(&v[..3], &[5, 3, 1]);
        assert!(s.contains_vertex(3));
        assert!(!s.contains_vertex(4));
    }

    #[test]
    fn test_ordering_same_diameter() {
        // Forward colexicographic: simplex with larger vertex set = younger (larger u128)
        let a = Simplex128::new(1.0, &[0, 1]); // edge (1, 0)
        let b = Simplex128::new(1.0, &[0, 2]); // edge (2, 0)
        let c = Simplex128::new(1.0, &[1, 2]); // edge (2, 1)
                                               // Colex order: (1,0) < (2,0) < (2,1)
                                               // Forward colex filtration: (1,0) oldest, (2,1) youngest
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn test_cofacet_construction() {
        let edge = Simplex128::new(1.0, &[0, 2]);
        let tri = edge.cofacet(1, encode_filtration(2.0));
        assert_eq!(tri.vertex_count(), 3);
        let v = tri.vertices();
        assert_eq!(&v[..3], &[2, 1, 0]);
    }

    #[test]
    fn test_cofacet_iter_all() {
        let dist = make_triangle_dist();
        let edge = Simplex128::new(1.0, &[0, 1]); // edge (1, 0), diam=1.0
        let mut iter = CofacetIter::new(edge, &dist, true, f32::INFINITY);
        let mut cofacets = Vec::new();
        while let Some(c) = iter.next() {
            cofacets.push(c);
        }
        // Only vertex 2 can be added (n=3, vertices {0,1} already in simplex)
        assert_eq!(cofacets.len(), 1);
        assert_eq!(cofacets[0].vertex_count(), 3);
        // Triangle (2,1,0), diameter = max(1.0, d(2,0)=2.0, d(2,1)=1.5) = 2.0
        assert_eq!(cofacets[0].filtration(), 2.0);
    }

    #[test]
    fn test_cofacet_iter_unique() {
        let dist = make_triangle_dist();
        // Edge (1, 0): largest_vertex = 1. With all_cofacets=false, only j > 1.
        let edge = Simplex128::new(1.0, &[0, 1]);
        let mut iter = CofacetIter::new(edge, &dist, false, f32::INFINITY);
        let mut cofacets = Vec::new();
        while let Some(c) = iter.next() {
            cofacets.push(c);
        }
        assert_eq!(cofacets.len(), 1); // only vertex 2
    }

    #[test]
    fn test_cofacet_threshold_filter() {
        let dist = make_triangle_dist();
        let edge = Simplex128::new(1.0, &[0, 1]);
        // threshold = 1.5: the triangle has diam 2.0 > 1.5, should be filtered
        let mut iter = CofacetIter::new(edge, &dist, true, 1.5);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_facet_iter_order() {
        let dist = make_triangle_dist();
        let tri = Simplex128::new(2.0, &[0, 1, 2]);
        let mut iter = FacetIter::new(tri, &dist);
        let mut facets = Vec::new();
        while let Some(f) = iter.next() {
            facets.push(f);
        }
        assert_eq!(facets.len(), 3);
        // FacetIter removes largest vertex first (slot 0), producing facets
        // with vertex sets in ascending colex order:
        //   slot 0 = vertex 2 removed → facet {1, 0}   (smallest vertex key)
        //   slot 1 = vertex 1 removed → facet {2, 0}
        //   slot 2 = vertex 0 removed → facet {2, 1}   (largest  vertex key)
        assert_eq!(facets[0].vertices()[..2], [1, 0]);
        assert_eq!(facets[1].vertices()[..2], [2, 0]);
        assert_eq!(facets[2].vertices()[..2], [2, 1]);

        // Verify vertex-key ordering is ascending (= oldest-first within same
        // diameter). Note: full u128 ordering is dominated by filtration, and
        // the three facets here have different diameters (1.0, 2.0, 1.5), so
        // we check vertex_key() rather than raw u128.
        assert!(facets[0].vertex_key() < facets[1].vertex_key());
        assert!(facets[1].vertex_key() < facets[2].vertex_key());
    }

    #[test]
    fn test_first_same_diam_cofacet_is_youngest() {
        // 4 points in a line: d(i,j) = |i - j|
        let n = 4;
        let mut data = Vec::new();
        for i in 0..n {
            for j in 0..i {
                data.push((i - j) as f32);
            }
        }
        let dist = DistanceMatrix::from_lower_triangular(n, data);

        // Vertex 0 as a 0-simplex, diam = 0
        let v0 = Simplex128::new(0.0, &[0]);
        let mut iter = CofacetIter::new(v0, &dist, true, f32::INFINITY);

        let mut first_same_diam: Option<Simplex128> = None;
        let mut all_same_diam = Vec::new();

        while let Some(c) = iter.next() {
            if c.filtration() == v0.filtration() {
                if first_same_diam.is_none() {
                    first_same_diam = Some(c);
                }
                all_same_diam.push(c);
            }
        }

        // If there are same-diam cofacets, the first one should be the largest u128 (youngest)
        if let Some(first) = first_same_diam {
            for &c in &all_same_diam {
                assert!(
                    first >= c,
                    "First same-diam cofacet should be youngest (largest u128)"
                );
            }
        }
    }

    #[test]
    fn test_first_same_diam_facet_is_oldest() {
        let dist = make_triangle_dist();
        // Triangle (2,1,0), diam = 2.0
        let tri = Simplex128::new(2.0, &[0, 1, 2]);
        let mut iter = FacetIter::new(tri, &dist);

        let mut first_same_diam: Option<Simplex128> = None;
        let mut all_same_diam = Vec::new();

        while let Some(f) = iter.next() {
            if f.filtration() == tri.filtration() {
                if first_same_diam.is_none() {
                    first_same_diam = Some(f);
                }
                all_same_diam.push(f);
            }
        }

        if let Some(first) = first_same_diam {
            for &f in &all_same_diam {
                assert!(
                    first <= f,
                    "First same-diam facet should be oldest (smallest u128)"
                );
            }
        }
    }

    // ── Apparent-pair primitive tests ──────────────────────────────────────

    #[test]
    fn test_zero_pivot_cofacet_basic() {
        let dist = make_triangle_dist();
        // Edge (2, 0) has diam = 2.0. Its only cofacet is the triangle (2,1,0),
        // also with diam = 2.0 (max of 1.0, 2.0, 1.5). So same-diameter cofacet exists.
        let edge = Simplex128::new(2.0, &[0, 2]);
        let cof = zero_pivot_cofacet(edge, &dist, f32::INFINITY);
        assert!(cof.is_some());
        assert_eq!(cof.unwrap().vertex_count(), 3);
        assert_eq!(cof.unwrap().filtration(), 2.0);
    }

    #[test]
    fn test_zero_pivot_cofacet_none() {
        let dist = make_triangle_dist();
        // Edge (1, 0) has diam = 1.0. Its only cofacet is triangle (2,1,0) with diam 2.0.
        // No same-diameter cofacet.
        let edge = Simplex128::new(1.0, &[0, 1]);
        let cof = zero_pivot_cofacet(edge, &dist, f32::INFINITY);
        assert!(cof.is_none());
    }

    #[test]
    fn test_zero_pivot_facet_basic() {
        let dist = make_triangle_dist();
        // Triangle (2,1,0), diam = 2.0.
        // Facets and their diameters:
        //   (1,0): d(1,0) = 1.0
        //   (2,0): d(2,0) = 2.0  ← same diameter
        //   (2,1): d(2,1) = 1.5
        // Only (2,0) is same-diameter; it's the oldest (and only) same-diam facet.
        let tri = Simplex128::new(2.0, &[0, 1, 2]);
        let facet = zero_pivot_facet(tri, &dist);
        assert!(facet.is_some());
        assert_eq!(facet.unwrap().vertices()[..2], [2, 0]);
    }

    #[test]
    fn test_zero_apparent_cofacet_identifies_apparent_pair() {
        let dist = make_triangle_dist();
        // Edge (2, 0), diam = 2.0. Its youngest same-diam cofacet is tri (2,1,0), diam 2.0.
        // The oldest same-diam facet of that triangle is (2,0), which equals our edge.
        // So (edge (2,0), tri (2,1,0)) is an apparent pair.
        let edge = Simplex128::new(2.0, &[0, 2]);
        let result = zero_apparent_cofacet(edge, &dist, f32::INFINITY);
        assert!(result.is_some());
        let tau = result.unwrap();
        assert_eq!(tau.vertex_count(), 3);
        assert_eq!(tau.filtration(), 2.0);
    }

    #[test]
    fn test_zero_apparent_cofacet_rejects_non_apparent() {
        let dist = make_triangle_dist();
        // Edge (1, 0), diam = 1.0. No same-diam cofacet → not apparent.
        let edge = Simplex128::new(1.0, &[0, 1]);
        let result = zero_apparent_cofacet(edge, &dist, f32::INFINITY);
        assert!(result.is_none());
    }

    #[test]
    fn test_is_apparent_cofacet_positive() {
        let dist = make_triangle_dist();
        // Triangle (2,1,0), diam 2.0. Its oldest same-diam facet is edge (2,0).
        // The youngest same-diam cofacet of edge (2,0) is the triangle itself.
        // So the triangle IS an apparent cofacet.
        let tri = Simplex128::new(2.0, &[0, 1, 2]);
        assert!(is_apparent_cofacet(tri, &dist, f32::INFINITY));
    }

    #[test]
    fn test_apparent_pair_rectangle_example() {
        // Reproduce the rectangle example from Ripser paper §3.5.
        // Vertices 0,1,2,3 arranged so d(i,j) are distinct and ordered:
        //   d(0,1) = 1, d(2,3) = 2, d(0,2) = 3, d(1,3) = 4, d(0,3) = 5, d(1,2) = 6
        // Lower-triangular layout: index(i,j) = i*(i-1)/2 + j for i > j.
        //   idx(1,0)=0: 1.0
        //   idx(2,0)=1: 3.0
        //   idx(2,1)=2: 6.0
        //   idx(3,0)=3: 5.0
        //   idx(3,1)=4: 4.0
        //   idx(3,2)=5: 2.0
        let dist = DistanceMatrix::from_lower_triangular(4, vec![1.0, 3.0, 6.0, 5.0, 4.0, 2.0]);

        // Per paper, one apparent pair in dim 0 is ((0), (1,0)):
        // vertex 0 has diam 0. Its youngest same-diam cofacet should be... actually
        // vertex 0 has diam 0, so same-diam cofacets need diam 0 — no edges have diam 0.
        // But we can check dim-1 apparent pairs using our Rips filtration.

        // Take edge (2,0), diam=3.0. Its same-diam cofacets are triangles containing
        // {2,0} with diam=3.0. Triangle (2,1,0) has diam max(1,3,6)=6, no.
        // Triangle (3,2,0) has diam max(3,5,2)=5, no. So no same-diam cofacet.
        let edge_20 = Simplex128::new(3.0, &[0, 2]);
        assert!(zero_apparent_cofacet(edge_20, &dist, f32::INFINITY).is_none());

        // Take edge (3,2), diam=2.0. Same-diam cofacets: triangles {3,2,x} with diam=2.
        // Triangle (3,2,0) has diam max(2,5,3)=5. Triangle (3,2,1) has diam max(2,4,6)=6.
        // No same-diam cofacet.
        let edge_32 = Simplex128::new(2.0, &[2, 3]);
        assert!(zero_apparent_cofacet(edge_32, &dist, f32::INFINITY).is_none());

        // The paper's apparent pairs in this example have to do with the lexicographic
        // refinement tie-breaks, which our forward-colex convention handles differently.
        // The key property we need: the apparent-pair check correctly identifies
        // pairs where diam(τ) == diam(σ), and correctly rejects when diameters differ.
        // Both of the above correctly return None.
    }

    #[test]
    fn test_zero_apparent_facet_dual_of_cofacet() {
        // 4 points configured so that ((1,0), (2,1,0)) is an apparent pair
        // (see test_apparent_pair_with_same_diam_cofacet for the setup).
        let dist = DistanceMatrix::from_lower_triangular(
            4,
            vec![
                1.0, // (1,0)
                1.0, 1.0, // (2,0), (2,1)
                2.0, 2.0, 2.0, // (3,0), (3,1), (3,2)
            ],
        );

        // Triangle (2,1,0) is the cofacet side of an apparent pair with edge (1,0).
        // So zero_apparent_facet on the triangle should return the edge (1,0).
        let tri = Simplex128::new(1.0, &[0, 1, 2]);
        let phi = zero_apparent_facet(tri, &dist, f32::INFINITY);
        assert!(phi.is_some(), "triangle should have an apparent facet");
        let phi = phi.unwrap();
        assert_eq!(phi.vertices()[..2], [1, 0]);

        // Edge (2,1) is NOT the cofacet of an apparent pair (no same-diameter cofacet).
        let edge_21 = Simplex128::new(1.0, &[1, 2]);
        assert!(zero_apparent_facet(edge_21, &dist, f32::INFINITY).is_none());

        // Triangle (3,2,1) has diam 2.0; its facets have diams (d(2,1)=1, d(3,1)=2, d(3,2)=2).
        // Oldest same-diam facet of (3,2,1) is the smallest u128 with diam=2. Since
        // facets are removed largest-first: remove 3 → (2,1) diam 1.0; remove 2 → (3,1) diam 2.0;
        // remove 1 → (3,2) diam 2.0. So oldest same-diam facet is (3,1).
        // Edge (3,1) has diam 2.0; its cofacets include (3,2,1) diam 2 and (3,1,0) diam max(2,2,1)=2.
        // Youngest same-diam cofacet of (3,1) — youngest = largest vertex inserted = vertex 2,
        // giving (3,2,1). So zero_pivot_cofacet((3,1)) = (3,2,1). Match → apparent pair.
        let tri_321 = Simplex128::new(2.0, &[1, 2, 3]);
        let phi = zero_apparent_facet(tri_321, &dist, f32::INFINITY);
        assert!(phi.is_some());
        assert_eq!(phi.unwrap().vertices()[..2], [3, 1]);
    }

    #[test]
    fn test_apparent_pair_with_same_diam_cofacet() {
        // Construct a case where σ has a same-diameter cofacet AND the reverse
        // direction confirms the pair.
        //
        // 4 points: 0, 1, 2, 3
        // Make d(0,1) = d(0,2) = d(1,2) = 1.0 (equilateral triangle on {0,1,2})
        // and d(0,3) = d(1,3) = d(2,3) = 2.0 (vertex 3 far away)
        //
        // Edge (2,1), diam=1.0. Same-diam cofacets are triangles containing {1,2}
        // with diam=1.0. Only (2,1,0) has diam 1.0 (d(0,1)=d(0,2)=d(1,2)=1). ✓
        // Triangle (3,2,1) has diam 2.0. So youngest same-diam cofacet of (2,1) is (2,1,0).
        //
        // Now check: is (2,1) the oldest same-diam facet of (2,1,0)?
        // Facets of (2,1,0) with diam 1.0: all three have diam 1.0.
        // Oldest (smallest u128) = (1,0).
        // So (2,1) is NOT the oldest same-diam facet → not apparent.
        //
        // But edge (1,0) IS the oldest same-diam facet of (2,1,0).
        // Is (2,1,0) the youngest same-diam cofacet of (1,0)?
        // Cofacets of (1,0) with diam 1.0: only (2,1,0) qualifies (others involve
        // vertex 3 with d=2). Youngest = (2,1,0). ✓
        // So ((1,0), (2,1,0)) IS an apparent pair.
        let dist = DistanceMatrix::from_lower_triangular(
            4,
            vec![
                1.0, // (1,0)
                1.0, 1.0, // (2,0), (2,1)
                2.0, 2.0, 2.0, // (3,0), (3,1), (3,2)
            ],
        );

        let edge_10 = Simplex128::new(1.0, &[0, 1]);
        let result = zero_apparent_cofacet(edge_10, &dist, f32::INFINITY);
        assert!(
            result.is_some(),
            "edge (1,0) should be in an apparent pair with triangle (2,1,0)"
        );
        let tau = result.unwrap();
        assert_eq!(tau.vertices()[..3], [2, 1, 0]);

        // Confirm the reverse: triangle (2,1,0) is an apparent cofacet.
        let tri = Simplex128::new(1.0, &[0, 1, 2]);
        assert!(is_apparent_cofacet(tri, &dist, f32::INFINITY));

        // And edge (2,1) is NOT in an apparent pair (it's not the oldest same-diam facet).
        let edge_21 = Simplex128::new(1.0, &[1, 2]);
        assert!(zero_apparent_cofacet(edge_21, &dist, f32::INFINITY).is_none());
    }
}
