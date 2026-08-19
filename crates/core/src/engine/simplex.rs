// Packed simplices: the `PackedSimplex<K>` abstraction, its two layouts
// (`Simplex128`: 6 × 16-bit slots; `Simplex256`: 9 × 24-bit slots), and fast
// hashing for simplex keys.
//
// LAYOUT (both)  filtration in the top 32 bits as encode_filtration(diam) =
//   !diam.to_bits(); below it, K vertex slots, largest vertex first, each
//   stored as id+1 (0 = empty).
//
// ORDERING  both layouts derive Ord, i.e. comparison IS raw fixed-width
//   lexicographic integer comparison. encode_filtration inverts the float
//   bits ON PURPOSE: it lets us keep the derived (and therefore trivially
//   inlinable) Ord in the heap's hot sift loop while still ordering a LARGER
//   diameter as a SMALLER integer. So:
//     larger value  ==  smaller diameter, or (equal diameter) larger vertex set.
//   The heap is a max-heap; the pivot is the max value.
//
// This file owns ONLY the simplex representations and their intrinsic
// operations (construction, accessors, single-vertex insert/remove, the
// incremental cofacet builders). All Ripser-specific logic — cofacet
// enumeration, apparent pairs, coboundary — lives in the algorithm layer
// (`algorithm.rs` / `reduction.rs`), generically over `PackedSimplex<K>`.

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
// PackedSimplex — the engine's simplex abstraction
// ═══════════════════════════════════════════════════════════════════════════════

/// A packed simplex: encoded filtration + vertex set in a fixed-width integer
/// image, totally ordered so that RAW COMPARISON IS THE RIPSER FILTRATION
/// ORDER — larger value == smaller diameter, ties broken colexicographically
/// on the vertex set (see the file header for why filtration bits are
/// inverted).
///
/// The engine (`algorithm.rs`, `reduction.rs`) is generic over this trait
/// and monomorphises per layout; `lib.rs` picks the instantiation from `n`
/// at the API boundary. Every method is a hot-loop primitive:
/// implementations must be `#[inline(always)]`-cheap shift/mask work with no
/// branches beyond what the layout forces.
///
/// `K` is the layout's vertex capacity (= deepest supported dim + 2: the
/// engine builds cofacets of the deepest reduced dimension). It is a CONST
/// GENERIC rather than an associated const so that it can size every engine
/// scratch array (`[u32; K]`) and unroll depth: each monomorphisation pays
/// for exactly its own layout, never the union of all layouts.
pub trait PackedSimplex<const K: usize>:
    Copy + Clone + Eq + Ord + std::hash::Hash + std::fmt::Debug + Send + Sync + 'static
{
    /// Incremental cofacet constructor for this layout (see
    /// [`CofacetBuilder`]).
    type Builder: CofacetBuilder<Simplex = Self>;

    /// Build from a filtration value and a DESCENDING-sorted vertex slice.
    /// Caller guarantees order and that ids fit the layout's slots.
    fn from_sorted_desc(filtration: f32, vertices: &[u32]) -> Self;
    /// Encoded (inverted-bits) filtration value.
    fn filtration_encoded(self) -> u32;
    /// Filtration value as f32.
    fn filtration(self) -> f32;
    /// Number of vertices (= dimension + 1).
    fn vertex_count(self) -> usize;
    /// Vertex ids, descending; only the first `vertex_count()` entries are
    /// meaningful (the rest decode as `u32::MAX`).
    fn vertices(self) -> [u32; K];
    /// Largest vertex id (slot 0). Undefined on the empty simplex.
    fn largest_vertex(self) -> u32;
    /// Insert `new_vertex` (sorted position) with a new encoded filtration.
    fn cofacet(self, new_vertex: u32, new_filt: u32) -> Self;
    /// Remove the vertex at slot `k` and set a new encoded filtration.
    fn remove_vertex_at(self, k: usize, new_filt: u32) -> Self;
    /// Start incremental cofacet construction over this simplex.
    fn cofacet_builder(self) -> Self::Builder;
}

/// Incremental cofacet construction for a fixed base simplex σ.
///
/// Caches whatever split state depends only on the insertion rank (the
/// number of σ-vertices greater than the inserted vertex). When cofacets are
/// enumerated youngest-first the rank advances monotonically, so callers
/// invoke [`set_rank`](Self::set_rank) only on rank changes and
/// [`build`](Self::build) once per neighbour — the per-neighbour cost stays
/// a handful of ORs and one shift, independent of the packing layout.
///
/// This trait exists so packing layouts never leak out of this module: the
/// algorithm layer drives builders without touching raw words.
pub trait CofacetBuilder {
    /// The simplex layout this builder produces. (No `PackedSimplex` bound
    /// here — that would force a `K` parameter onto this trait; the tie is
    /// made by `PackedSimplex::Builder: CofacetBuilder<Simplex = Self>`.)
    type Simplex;

    /// Move to insertion rank `rank` (number of σ-vertices greater than the
    /// next inserted vertex; `0..=vertex_count`). Monotone non-decreasing.
    fn set_rank(&mut self, rank: usize);
    /// The cofacet of σ obtained by inserting `new_vertex` at the current
    /// rank, carrying encoded filtration `new_filt`.
    fn build(&self, new_vertex: u32, new_filt: u32) -> Self::Simplex;
}
// ═══════════════════════════════════════════════════════════════════════════════
// Simplex128
// ═══════════════════════════════════════════════════════════════════════════════

/// Mask for the vertex payload (bits 0–95).
const VERTEX_MASK: u128 = (1u128 << 96) - 1;

/// Vertex slots in the 128-bit layout (6 × 16 bits = the 96-bit payload).
const S128_SLOTS: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(C, align(16))]
pub struct Simplex128(pub u128);

impl Simplex128 {
    /// Create from a filtration value (f32) and an unsorted vertex slice.
    /// Vertices are sorted descending internally.
    #[cfg(test)]
    #[inline]
    pub fn new(filtration: f32, vertices: &[u32]) -> Self {
        let mut sorted = [0u32; S128_SLOTS];
        let len = vertices.len().min(S128_SLOTS);
        sorted[..len].copy_from_slice(&vertices[..len]);
        sorted[..len].sort_unstable_by(|a, b| b.cmp(a));
        Self::from_sorted_desc(filtration, &sorted[..len])
    }

    /// Create from an encoded filtration (u32) and a DESCENDING-sorted vertex slice.
    /// Caller must guarantee descending order — no sort is performed.
    ///
    /// Vertex ids are `u32` at the interface; this 128-bit layout stores them
    /// in 16-bit slots, so ids must be `< u16::MAX` (`id + 1` must fit).
    #[inline(always)]
    pub fn from_sorted_desc(filtration: f32, vertices: &[u32]) -> Self {
        debug_assert!(vertices.iter().all(|&v| v < u16::MAX as u32));
        let len = vertices.len().min(S128_SLOTS);
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
    pub fn vertices(self) -> [u32; S128_SLOTS] {
        [
            (self.slot(0) as u32).wrapping_sub(1),
            (self.slot(1) as u32).wrapping_sub(1),
            (self.slot(2) as u32).wrapping_sub(1),
            (self.slot(3) as u32).wrapping_sub(1),
            (self.slot(4) as u32).wrapping_sub(1),
            (self.slot(5) as u32).wrapping_sub(1),
        ]
    }

    /// Largest vertex ID (slot 0).
    #[inline(always)]
    pub fn largest_vertex(self) -> u32 {
        debug_assert!(self.slot(0) != 0, "Empty simplex has no largest vertex");
        (self.slot(0) as u32) - 1
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
    pub fn contains_vertex(self, v: u32) -> bool {
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
    pub fn cofacet(self, new_vertex: u32, new_filt: u32) -> Self {
        debug_assert!(new_vertex < u16::MAX as u32);
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
// Cofacet128Builder — incremental cofacet construction for Simplex128
// ═══════════════════════════════════════════════════════════════════════════════

/// [`CofacetBuilder`] for the 128-bit layout: caches the payload split
/// (`upper`, `lower >> 16`, insert shift) per insertion rank.
pub struct Cofacet128Builder {
    payload: u128,
    payload_upper: u128,
    payload_lower_shifted: u128,
    insert_shift: u32,
}

impl Cofacet128Builder {
    /// Builder for σ, starting at insertion rank 0 (inserted vertex larger
    /// than every σ-vertex).
    #[inline(always)]
    fn new(sigma: Simplex128) -> Self {
        let payload = sigma.vertex_key();
        Self {
            payload,
            payload_upper: 0,
            payload_lower_shifted: payload >> 16,
            insert_shift: 80,
        }
    }
}

impl CofacetBuilder for Cofacet128Builder {
    type Simplex = Simplex128;

    #[inline(always)]
    fn set_rank(&mut self, rank: usize) {
        let split = 96 - (rank as u32) * 16;
        let upper_mask = (!0u128) << split;
        self.payload_upper = self.payload & upper_mask;
        self.payload_lower_shifted = (self.payload & !upper_mask) >> 16;
        self.insert_shift = split - 16;
    }

    #[inline(always)]
    fn build(&self, new_vertex: u32, new_filt: u32) -> Simplex128 {
        debug_assert!(new_vertex < u16::MAX as u32);
        let w_ins = (new_vertex as u128) + 1;
        Simplex128(
            ((new_filt as u128) << 96)
                | self.payload_upper
                | (w_ins << self.insert_shift)
                | self.payload_lower_shifted,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PackedSimplex impl — Simplex128 (delegates to the inherent methods)
// ═══════════════════════════════════════════════════════════════════════════════

impl PackedSimplex<{ S128_SLOTS }> for Simplex128 {
    type Builder = Cofacet128Builder;

    #[inline(always)]
    fn from_sorted_desc(filtration: f32, vertices: &[u32]) -> Self {
        Simplex128::from_sorted_desc(filtration, vertices)
    }

    #[inline(always)]
    fn filtration_encoded(self) -> u32 {
        Simplex128::filtration_encoded(self)
    }

    #[inline(always)]
    fn filtration(self) -> f32 {
        Simplex128::filtration(self)
    }

    #[inline(always)]
    fn vertex_count(self) -> usize {
        Simplex128::vertex_count(self)
    }

    #[inline(always)]
    fn vertices(self) -> [u32; S128_SLOTS] {
        Simplex128::vertices(self)
    }

    #[inline(always)]
    fn largest_vertex(self) -> u32 {
        Simplex128::largest_vertex(self)
    }

    #[inline(always)]
    fn cofacet(self, new_vertex: u32, new_filt: u32) -> Self {
        Simplex128::cofacet(self, new_vertex, new_filt)
    }

    #[inline(always)]
    fn remove_vertex_at(self, k: usize, new_filt: u32) -> Self {
        Simplex128::remove_vertex_at(self, k, new_filt)
    }

    #[inline(always)]
    fn cofacet_builder(self) -> Cofacet128Builder {
        Cofacet128Builder::new(self)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Simplex256
// ═══════════════════════════════════════════════════════════════════════════════
//
// LAYOUT  hi: [filt:32][s0:24][s1:24][s2:24][s3:24]
//         lo: [s4:24][s5:24][s6:24][s7:24][s8:24][pad:8 = 0]
//
// 9 slots of 24 bits (id + 1, 0 = empty; ids <= 16_777_214), filtration
// encoded exactly as Simplex128. With the 32-bit filtration on top, the hi
// word holds exactly four slots (32 + 4·24 = 128): NO slot straddles the
// hi/lo boundary, so every slot extraction/insertion is single-word
// shift/mask work — only whole-payload shifts (by 24) cross the boundary.
//
// ORDERING  derived Ord compares `hi` then `lo` (FIELD ORDER MATTERS) —
// u256 lexicographic comparison with the same semantics as Simplex128's
// u128: larger == smaller diameter, ties colexicographic on the vertex set.
// The constant zero pad at the bottom of `lo` cannot affect ties.

/// Vertex slots in the 256-bit layout (9 × 24 bits across two words).
const S256_SLOTS: usize = 9;
/// 24-bit slot mask (also the exclusive upper bound for `id + 1`).
const S256_SLOT_MASK: u128 = 0xFF_FFFF;
/// hi bits 0..96 hold slots 0..3; bits 96..128 are the filtration.
const S256_HI_PAYLOAD: u128 = (1u128 << 96) - 1;

/// Bit mask over the conceptual 256-bit value `(hi, lo)`: bits `>= split`.
/// `split` must be in `1..=255`; every caller passes multiples of 8 in
/// `8..=224`, so the inner shifts never reach 128.
#[inline(always)]
fn mask_ge_256(split: u32) -> (u128, u128) {
    if split >= 128 {
        ((!0u128) << (split - 128), 0)
    } else {
        (!0u128, (!0u128) << split)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(C, align(32))]
pub struct Simplex256 {
    /// Most significant word: `[filt:32][s0:24][s1:24][s2:24][s3:24]`.
    hi: u128,
    /// Least significant word: `[s4:24][s5:24][s6:24][s7:24][s8:24][pad:8]`.
    lo: u128,
}

/// Manual `Hash`: fold both words into ONE `write_u128`. The derived impl
/// would call `write_u128` twice, and `FxHasher`'s one-shot fold OVERWRITES
/// previous state — the first word would not contribute to the hash at all.
impl std::hash::Hash for Simplex256 {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Odd 128-bit fold constant (golden-ratio style); the multiply keeps
        // `lo`'s entropy from cancelling against `hi` under the xor.
        const FOLD: u128 = 0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835;
        state.write_u128(self.lo.wrapping_mul(FOLD) ^ self.hi);
    }
}

impl Simplex256 {
    /// Create from a filtration value (f32) and an unsorted vertex slice.
    /// Vertices are sorted descending internally.
    #[cfg(test)]
    #[inline]
    pub fn new(filtration: f32, vertices: &[u32]) -> Self {
        let mut sorted = [0u32; S256_SLOTS];
        let len = vertices.len().min(S256_SLOTS);
        sorted[..len].copy_from_slice(&vertices[..len]);
        sorted[..len].sort_unstable_by(|a, b| b.cmp(a));
        Self::from_sorted_desc(filtration, &sorted[..len])
    }

    /// Create from a filtration value and a DESCENDING-sorted vertex slice.
    /// Ids must satisfy `id + 1 < 2^24`, i.e. `id <= 16_777_214`.
    #[inline(always)]
    pub fn from_sorted_desc(filtration: f32, vertices: &[u32]) -> Self {
        debug_assert!(vertices.iter().all(|&v| (v as u128) < S256_SLOT_MASK));
        let len = vertices.len().min(S256_SLOTS);
        let mut hi: u128 = (encode_filtration(filtration) as u128) << 96;
        let mut lo: u128 = 0;
        // Unrolled packing. Each vertex stored as (id + 1) to reserve 0 = empty.
        if len > 0 {
            hi |= ((vertices[0] as u128) + 1) << 72;
        }
        if len > 1 {
            hi |= ((vertices[1] as u128) + 1) << 48;
        }
        if len > 2 {
            hi |= ((vertices[2] as u128) + 1) << 24;
        }
        if len > 3 {
            hi |= (vertices[3] as u128) + 1;
        }
        if len > 4 {
            lo |= ((vertices[4] as u128) + 1) << 104;
        }
        if len > 5 {
            lo |= ((vertices[5] as u128) + 1) << 80;
        }
        if len > 6 {
            lo |= ((vertices[6] as u128) + 1) << 56;
        }
        if len > 7 {
            lo |= ((vertices[7] as u128) + 1) << 32;
        }
        if len > 8 {
            lo |= ((vertices[8] as u128) + 1) << 8;
        }
        Simplex256 { hi, lo }
    }

    /// Encoded filtration value (u32).
    #[inline(always)]
    pub fn filtration_encoded(self) -> u32 {
        (self.hi >> 96) as u32
    }

    /// Filtration value as f32.
    #[inline(always)]
    pub fn filtration(self) -> f32 {
        decode_filtration(self.filtration_encoded())
    }

    /// Raw slot value at position k (0 = most significant). Returns 0 for
    /// empty, otherwise `vertex_id + 1`. Single-word by layout.
    #[inline(always)]
    fn slot(self, k: usize) -> u32 {
        debug_assert!(k < S256_SLOTS);
        let (word, shift) = if k < 4 {
            (self.hi, 72 - 24 * k)
        } else {
            (self.lo, 104 - 24 * (k - 4))
        };
        ((word >> shift) & S256_SLOT_MASK) as u32
    }

    /// Number of vertices (= dimension + 1). Slots are packed contiguously
    /// from slot 0; the constant bound lets the loop fully unroll.
    #[inline(always)]
    pub fn vertex_count(self) -> usize {
        let mut k = 0;
        while k < S256_SLOTS {
            if self.slot(k) == 0 {
                return k;
            }
            k += 1;
        }
        S256_SLOTS
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
    pub fn vertices(self) -> [u32; S256_SLOTS] {
        [
            self.slot(0).wrapping_sub(1),
            self.slot(1).wrapping_sub(1),
            self.slot(2).wrapping_sub(1),
            self.slot(3).wrapping_sub(1),
            self.slot(4).wrapping_sub(1),
            self.slot(5).wrapping_sub(1),
            self.slot(6).wrapping_sub(1),
            self.slot(7).wrapping_sub(1),
            self.slot(8).wrapping_sub(1),
        ]
    }

    /// Largest vertex ID (slot 0).
    #[inline(always)]
    pub fn largest_vertex(self) -> u32 {
        debug_assert!(self.slot(0) != 0, "Empty simplex has no largest vertex");
        self.slot(0) - 1
    }

    // ── Cofacet construction ────────────────────────────────────────────────

    /// Insert `new_vertex` into this simplex with a new encoded filtration.
    /// One-shot spelling over the builder: compute the insertion rank, then
    /// split and recombine.
    #[inline(always)]
    pub fn cofacet(self, new_vertex: u32, new_filt: u32) -> Self {
        debug_assert!((new_vertex as u128) < S256_SLOT_MASK);
        let v_ins = new_vertex + 1;
        // Count existing slots > v_ins over slots 0..S256_SLOTS-1 (a cofacet
        // requires a free slot, so the last one is empty). Empty slots are 0
        // and never count. Constant bound: fully unrolled.
        let mut rank = 0usize;
        let mut k = 0;
        while k < S256_SLOTS - 1 {
            if self.slot(k) > v_ins {
                rank += 1;
            }
            k += 1;
        }
        let mut builder = Cofacet256Builder::new(self);
        CofacetBuilder::set_rank(&mut builder, rank);
        CofacetBuilder::build(&builder, new_vertex, new_filt)
    }

    // ── Facet construction ──────────────────────────────────────────────────

    /// Remove the vertex at slot `k` and set a new encoded filtration.
    /// Slots below `k` shift up (left by 24 bits) to close the gap.
    #[inline(always)]
    pub fn remove_vertex_at(self, k: usize, new_filt: u32) -> Self {
        debug_assert!(k < S256_SLOTS);
        let ph = self.hi & S256_HI_PAYLOAD;
        let pl = self.lo;
        let split = 224 - 24 * (k as u32); // top of the removed slot (exclusive)
        let low_edge = split - 24; // bottom of the removed slot

        // Slots above k stay in place.
        let (mhi, mlo) = mask_ge_256(split);
        let up_hi = ph & mhi;
        let up_lo = pl & mlo;

        // Everything below the removed slot shifts up by 24 bits.
        let (bhi, blo) = mask_ge_256(low_edge);
        let rem_hi = ph & !bhi;
        let rem_lo = pl & !blo;
        let sh_hi = (rem_hi << 24) | (rem_lo >> 104);
        let sh_lo = rem_lo << 24;

        Simplex256 {
            hi: ((new_filt as u128) << 96) | up_hi | sh_hi,
            lo: up_lo | sh_lo,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cofacet256Builder — incremental cofacet construction for Simplex256
// ═══════════════════════════════════════════════════════════════════════════════

/// [`CofacetBuilder`] for the 256-bit layout. The cached split state is the
/// two-word analogue of `Cofacet128Builder`'s: upper payload (in place),
/// lower payload pre-shifted down by one 24-bit slot, and the insertion
/// shift. Because slots never straddle the hi/lo boundary, the inserted
/// vertex lands in exactly one word (selected by a branch that is constant
/// per rank, i.e. perfectly predicted within a rank run).
pub struct Cofacet256Builder {
    payload_hi: u128,
    payload_lo: u128,
    upper_hi: u128,
    upper_lo: u128,
    lower_hi: u128,
    lower_lo: u128,
    insert_shift: u32,
}

impl Cofacet256Builder {
    /// Builder for σ, starting at insertion rank 0.
    #[inline(always)]
    fn new(sigma: Simplex256) -> Self {
        let payload_hi = sigma.hi & S256_HI_PAYLOAD;
        let payload_lo = sigma.lo;
        Self {
            payload_hi,
            payload_lo,
            upper_hi: 0,
            upper_lo: 0,
            lower_hi: payload_hi >> 24,
            lower_lo: (payload_lo >> 24) | (payload_hi << 104),
            insert_shift: 200,
        }
    }
}

impl CofacetBuilder for Cofacet256Builder {
    type Simplex = Simplex256;

    #[inline(always)]
    fn set_rank(&mut self, rank: usize) {
        let split = 224 - 24 * (rank as u32);
        let (mhi, mlo) = mask_ge_256(split);
        self.upper_hi = self.payload_hi & mhi;
        self.upper_lo = self.payload_lo & mlo;
        let rem_hi = self.payload_hi & !mhi;
        let rem_lo = self.payload_lo & !mlo;
        self.lower_hi = rem_hi >> 24;
        self.lower_lo = (rem_lo >> 24) | (rem_hi << 104);
        self.insert_shift = split - 24;
    }

    #[inline(always)]
    fn build(&self, new_vertex: u32, new_filt: u32) -> Simplex256 {
        debug_assert!((new_vertex as u128) < S256_SLOT_MASK);
        let v_ins = (new_vertex as u128) + 1;
        let (ins_hi, ins_lo) = if self.insert_shift >= 128 {
            (v_ins << (self.insert_shift - 128), 0)
        } else {
            (0, v_ins << self.insert_shift)
        };
        Simplex256 {
            hi: ((new_filt as u128) << 96) | self.upper_hi | ins_hi | self.lower_hi,
            lo: self.upper_lo | ins_lo | self.lower_lo,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PackedSimplex impl — Simplex256 (delegates to the inherent methods)
// ═══════════════════════════════════════════════════════════════════════════════

impl PackedSimplex<{ S256_SLOTS }> for Simplex256 {
    type Builder = Cofacet256Builder;

    #[inline(always)]
    fn from_sorted_desc(filtration: f32, vertices: &[u32]) -> Self {
        Simplex256::from_sorted_desc(filtration, vertices)
    }

    #[inline(always)]
    fn filtration_encoded(self) -> u32 {
        Simplex256::filtration_encoded(self)
    }

    #[inline(always)]
    fn filtration(self) -> f32 {
        Simplex256::filtration(self)
    }

    #[inline(always)]
    fn vertex_count(self) -> usize {
        Simplex256::vertex_count(self)
    }

    #[inline(always)]
    fn vertices(self) -> [u32; S256_SLOTS] {
        Simplex256::vertices(self)
    }

    #[inline(always)]
    fn largest_vertex(self) -> u32 {
        Simplex256::largest_vertex(self)
    }

    #[inline(always)]
    fn cofacet(self, new_vertex: u32, new_filt: u32) -> Self {
        Simplex256::cofacet(self, new_vertex, new_filt)
    }

    #[inline(always)]
    fn remove_vertex_at(self, k: usize, new_filt: u32) -> Self {
        Simplex256::remove_vertex_at(self, k, new_filt)
    }

    #[inline(always)]
    fn cofacet_builder(self) -> Cofacet256Builder {
        Cofacet256Builder::new(self)
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

    // ── Simplex256 ──────────────────────────────────────────────────────────

    #[test]
    fn s256_basics() {
        let s = Simplex256::new(1.5, &[3, 1, 5]);
        assert_eq!(s.vertex_count(), 3);
        assert_eq!(s.dim(), 2);
        assert_eq!(s.largest_vertex(), 5);
        assert_eq!(&s.vertices()[..3], &[5, 3, 1]);
        assert_eq!(s.filtration(), 1.5);
    }

    #[test]
    fn s256_nine_vertices_roundtrip() {
        // Nine vertices straddling the hi/lo boundary (slots 3 and 4), with
        // ids up to the 24-bit cap.
        let ids = [
            16_777_214u32,
            1_000_000,
            65_536,
            65_535,
            4_096,
            300,
            64,
            1,
            0,
        ];
        let s = Simplex256::new(3.25, &ids);
        assert_eq!(s.vertex_count(), 9);
        assert_eq!(s.vertices(), ids); // already descending
        assert_eq!(s.largest_vertex(), 16_777_214);
        assert_eq!(s.filtration(), 3.25);
    }

    #[test]
    fn s256_ordering_same_diameter_is_colex() {
        let a = Simplex256::new(1.0, &[0, 1]);
        let b = Simplex256::new(1.0, &[0, 2]);
        let c = Simplex256::new(1.0, &[1, 2]);
        assert!(a < b);
        assert!(b < c);
        // Larger diameter orders SMALLER (Ripser order), as for Simplex128.
        let d = Simplex256::new(2.0, &[1, 2]);
        assert!(d < a);
    }

    #[test]
    fn s256_cofacet_matches_reference_at_every_rank() {
        // 8-vertex base; inserting each w exercises every insertion rank,
        // including the hi/lo boundary (ranks 3..5).
        let base_ids = [900_000u32, 70_000, 65_536, 4_000, 512, 300, 64, 2];
        let base = Simplex256::new(1.0, &base_ids);
        for &w in &[
            16_000_000u32,
            100_000,
            66_000,
            65_000,
            1_000,
            400,
            100,
            10,
            0,
        ] {
            let cof = base.cofacet(w, encode_filtration(2.0));
            // Reference: sort the 9 ids descending and repack.
            let mut ids: Vec<u32> = base_ids.to_vec();
            ids.push(w);
            ids.sort_unstable_by(|a, b| b.cmp(a));
            let expected = Simplex256::from_sorted_desc(2.0, &ids);
            assert_eq!(cof, expected, "inserting {w}");
            assert_eq!(cof.vertex_count(), 9);
        }
    }

    #[test]
    fn s256_builder_matches_one_shot_cofacet() {
        let base = Simplex256::new(1.0, &[70_000, 65_536, 512, 64, 2]);
        let verts = base.vertices();
        let vc = base.vertex_count();
        let mut builder = Cofacet256Builder::new(base);
        // Youngest-first insertion order, monotone rank — as the engine drives it.
        let mut vi = 0usize;
        for &w in &[100_000u32, 66_000, 1_000, 100, 0] {
            let old_vi = vi;
            while vi < vc && verts[vi] > w {
                vi += 1;
            }
            if vi != old_vi {
                builder.set_rank(vi);
            }
            let filt = encode_filtration(3.0);
            assert_eq!(builder.build(w, filt), base.cofacet(w, filt), "w = {w}");
        }
    }

    #[test]
    fn s256_remove_vertex_matches_reference_at_every_slot() {
        let ids = [
            16_777_214u32,
            1_000_000,
            65_536,
            65_535,
            4_096,
            300,
            64,
            1,
            0,
        ];
        let s = Simplex256::new(2.0, &ids);
        for k in 0..9 {
            let facet = s.remove_vertex_at(k, encode_filtration(2.0));
            let mut expected_ids: Vec<u32> = ids.to_vec();
            expected_ids.remove(k);
            let expected = Simplex256::from_sorted_desc(2.0, &expected_ids);
            assert_eq!(facet, expected, "removing slot {k}");
            assert_eq!(facet.vertex_count(), 8);
        }
    }

    #[test]
    fn s256_cofacet_then_remove_roundtrips() {
        let base = Simplex256::new(1.0, &[9, 5, 3]);
        let cof = base.cofacet(7, encode_filtration(2.0));
        assert_eq!(&cof.vertices()[..4], &[9, 7, 5, 3]);
        // Vertex 7 landed in slot 1; removing it restores the base vertex set.
        let back = cof.remove_vertex_at(1, encode_filtration(1.0));
        assert_eq!(back, base);
    }

    #[test]
    fn s256_hash_uses_both_words() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        // Two simplices identical in `lo` but different in `hi` (and vice
        // versa) must hash differently under any sane fold.
        let h = |s: Simplex256| {
            let mut hasher = DefaultHasher::new();
            s.hash(&mut hasher);
            hasher.finish()
        };
        let a = Simplex256::new(1.0, &[5, 3, 1]); // hi-only payload
        let b = Simplex256::new(1.0, &[5, 3, 2]);
        let c = Simplex256::new(2.0, &[5, 3, 1]);
        assert_ne!(h(a), h(b));
        assert_ne!(h(a), h(c));
        // lo-word difference (7 vertices): slots 4+ live in `lo`.
        let d = Simplex256::new(1.0, &[9, 8, 7, 6, 5, 4, 3]);
        let e = Simplex256::new(1.0, &[9, 8, 7, 6, 5, 4, 2]);
        assert_ne!(h(d), h(e));
    }
}
