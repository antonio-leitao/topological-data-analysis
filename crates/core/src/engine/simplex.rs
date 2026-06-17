// Packed 128-bit simplex (Simplex128) + fast hashing for Simplex128 keys.
//
// LAYOUT  high 32 bits: encode_filtration(diam) = !diam.to_bits()
//         low  96 bits: 6 vertex slots, largest vertex first, each stored as id+1 (0 = empty)
//
// ORDERING  Simplex128 derives Ord, i.e. comparison IS raw u128 lexicographic
//   comparison. encode_filtration inverts the float bits ON PURPOSE: it lets us
//   keep the derived (and therefore trivially inlinable) Ord in the heap's hot
//   sift loop while still ordering a LARGER diameter as a SMALLER u128. So:
//     larger u128  ==  smaller diameter, or (equal diameter) larger vertex set.
//   The heap is a max-heap; the pivot is the max u128.
//
// This file owns ONLY the simplex representation and its intrinsic operations
// (construction, accessors, single-vertex insert/remove). All Ripser-specific
// logic — cofacet enumeration, apparent pairs, coboundary — lives in the
// algorithm layer (`algorithm.rs` / `reduction.rs`).

use std::hash::{BuildHasherDefault, Hasher};

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
    /// Create from a filtration value (f32) and an unsorted vertex slice.
    /// Vertices are sorted descending internally.
    #[cfg(test)]
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
    #[cfg(test)]
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
    #[cfg(test)]
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
// FxHash — fast hash for Simplex128 (u128 keys)
// ═══════════════════════════════════════════════════════════════════════════════
//
// The default SipHash is too slow for the inner reduction loop where we do
// millions of hash lookups. FxHash-style multiply-xor folding on the u128
// gives excellent performance. Simplex128's lower 96 bits (vertex content)
// provide good entropy.

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

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_remove_vertex_at() {
        // Triangle (2,1,0); removing slot 1 (vertex 1) yields edge (2,0).
        let tri = Simplex128::new(2.0, &[0, 1, 2]);
        let facet = tri.remove_vertex_at(1, encode_filtration(2.0));
        assert_eq!(facet.vertex_count(), 2);
        assert_eq!(facet.vertices()[..2], [2, 0]);
        assert_eq!(facet.filtration(), 2.0);
    }
}
