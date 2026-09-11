// The top 32 bits store `!diameter.to_bits()`. Vertex slots follow in
// descending order as `id + 1`, reserving zero for an empty slot. Raw ordering
// is therefore Ripser order: smaller diameters and then larger vertex sets.

#[inline(always)]
pub fn encode_filtration(f: f32) -> u32 {
    debug_assert!(f >= 0.0, "Distances must be non-negative");
    !f.to_bits()
}

#[inline(always)]
pub fn decode_filtration(u: u32) -> f32 {
    f32::from_bits(!u)
}

/// Packed simplex operations used by the reduction engine.
///
/// `K` is the vertex capacity and sizes the engine's fixed scratch arrays.
pub trait PackedSimplex<const K: usize>:
    Copy + Clone + Eq + Ord + std::hash::Hash + std::fmt::Debug + Send + Sync + 'static
{
    type Builder: CofacetBuilder<Simplex = Self>;

    /// `vertices` must be descending and fit the selected layout.
    fn from_sorted_desc(filtration: f32, vertices: &[u32]) -> Self;
    fn filtration_encoded(self) -> u32;
    fn filtration(self) -> f32;
    fn vertex_count(self) -> usize;
    /// Descending ids; entries after `vertex_count()` are sentinels.
    fn vertices(self) -> [u32; K];
    /// Undefined for an empty simplex.
    fn largest_vertex(self) -> u32;
    fn cofacet(self, new_vertex: u32, new_filt: u32) -> Self;
    fn remove_vertex_at(self, k: usize, new_filt: u32) -> Self;
    fn cofacet_builder(self) -> Self::Builder;
}

/// Caches the payload split for a monotone insertion rank.
pub trait CofacetBuilder {
    type Simplex;

    /// `rank` is the number of base vertices greater than the inserted vertex.
    fn set_rank(&mut self, rank: usize);
    fn build(&self, new_vertex: u32, new_filt: u32) -> Self::Simplex;
}

const VERTEX_MASK: u128 = (1u128 << 96) - 1;

const S128_SLOTS: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(C, align(16))]
pub struct Simplex128(pub u128);

impl Simplex128 {
    #[cfg(test)]
    #[inline]
    pub fn new(filtration: f32, vertices: &[u32]) -> Self {
        let mut sorted = [0u32; S128_SLOTS];
        let len = vertices.len().min(S128_SLOTS);
        sorted[..len].copy_from_slice(&vertices[..len]);
        sorted[..len].sort_unstable_by(|a, b| b.cmp(a));
        Self::from_sorted_desc(filtration, &sorted[..len])
    }

    /// `vertices` must be descending and each id must be below `u16::MAX`.
    #[inline(always)]
    pub fn from_sorted_desc(filtration: f32, vertices: &[u32]) -> Self {
        debug_assert!(vertices.iter().all(|&v| v < u16::MAX as u32));
        let len = vertices.len().min(S128_SLOTS);
        let mut p: u128 = (encode_filtration(filtration) as u128) << 96;
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

    #[inline(always)]
    pub fn filtration_encoded(self) -> u32 {
        (self.0 >> 96) as u32
    }

    #[inline(always)]
    pub fn filtration(self) -> f32 {
        decode_filtration(self.filtration_encoded())
    }

    #[inline(always)]
    fn slot(self, k: usize) -> u16 {
        ((self.0 >> (80 - k * 16)) & 0xFFFF) as u16
    }

    #[inline(always)]
    pub fn vertex_count(self) -> usize {
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

    #[cfg(test)]
    #[inline(always)]
    pub fn dim(self) -> usize {
        self.vertex_count().saturating_sub(1)
    }

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

    #[inline(always)]
    pub fn largest_vertex(self) -> u32 {
        debug_assert!(self.slot(0) != 0, "Empty simplex has no largest vertex");
        (self.slot(0) as u32) - 1
    }

    #[inline(always)]
    pub fn vertex_key(self) -> u128 {
        self.0 & VERTEX_MASK
    }

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

    #[inline(always)]
    pub fn cofacet(self, new_vertex: u32, new_filt: u32) -> Self {
        debug_assert!(new_vertex < u16::MAX as u32);
        let payload = self.0 & VERTEX_MASK;
        let v_ins = (new_vertex as u128) + 1;

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

        let split = 96 - (rank * 16);
        let upper_mask = (!0u128) << split;

        let mut result = (new_filt as u128) << 96;
        result |= payload & upper_mask;
        result |= v_ins << (split - 16);
        result |= (payload & !upper_mask) >> 16;
        Simplex128(result)
    }

    #[inline(always)]
    pub fn remove_vertex_at(self, k: usize, new_filt: u32) -> Self {
        let payload = self.0 & VERTEX_MASK;
        let remove_bit = 80 - k * 16;
        let split = remove_bit + 16;

        let upper = if split >= 96 {
            0u128
        } else {
            payload & ((!0u128) << split) & VERTEX_MASK
        };

        let lower = if remove_bit == 0 {
            0u128
        } else {
            (payload & ((1u128 << remove_bit) - 1)) << 16
        };

        Simplex128(((new_filt as u128) << 96) | upper | lower)
    }
}

pub struct Cofacet128Builder {
    payload: u128,
    payload_upper: u128,
    payload_lower_shifted: u128,
    insert_shift: u32,
}

impl Cofacet128Builder {
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

// `[filt:32][four 24-bit slots]` in `hi`, then five slots and eight padding
// bits in `lo`. No slot crosses the word boundary. Field order preserves the
// raw ordering used by `Simplex128`.

const S256_SLOTS: usize = 9;
const S256_SLOT_MASK: u128 = 0xFF_FFFF;
const S256_HI_PAYLOAD: u128 = (1u128 << 96) - 1;

/// Mask bits at or above `split` in conceptual value `(hi, lo)`.
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
    hi: u128,
    lo: u128,
}

/// `FxHasher::write_u128` replaces its state, so derived hashing would ignore
/// the first word.
impl std::hash::Hash for Simplex256 {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        const FOLD: u128 = 0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835;
        state.write_u128(self.lo.wrapping_mul(FOLD) ^ self.hi);
    }
}

impl Simplex256 {
    #[cfg(test)]
    #[inline]
    pub fn new(filtration: f32, vertices: &[u32]) -> Self {
        let mut sorted = [0u32; S256_SLOTS];
        let len = vertices.len().min(S256_SLOTS);
        sorted[..len].copy_from_slice(&vertices[..len]);
        sorted[..len].sort_unstable_by(|a, b| b.cmp(a));
        Self::from_sorted_desc(filtration, &sorted[..len])
    }

    /// `vertices` must be descending and each id at most `16_777_214`.
    #[inline(always)]
    pub fn from_sorted_desc(filtration: f32, vertices: &[u32]) -> Self {
        debug_assert!(vertices.iter().all(|&v| (v as u128) < S256_SLOT_MASK));
        let len = vertices.len().min(S256_SLOTS);
        let mut hi: u128 = (encode_filtration(filtration) as u128) << 96;
        let mut lo: u128 = 0;
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

    #[inline(always)]
    pub fn filtration_encoded(self) -> u32 {
        (self.hi >> 96) as u32
    }

    #[inline(always)]
    pub fn filtration(self) -> f32 {
        decode_filtration(self.filtration_encoded())
    }

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

    #[cfg(test)]
    #[inline(always)]
    pub fn dim(self) -> usize {
        self.vertex_count().saturating_sub(1)
    }

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

    #[inline(always)]
    pub fn largest_vertex(self) -> u32 {
        debug_assert!(self.slot(0) != 0, "Empty simplex has no largest vertex");
        self.slot(0) - 1
    }

    #[inline(always)]
    pub fn cofacet(self, new_vertex: u32, new_filt: u32) -> Self {
        debug_assert!((new_vertex as u128) < S256_SLOT_MASK);
        let v_ins = new_vertex + 1;
        // A cofacet has a free final slot, so it need not participate in rank.
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

    #[inline(always)]
    pub fn remove_vertex_at(self, k: usize, new_filt: u32) -> Self {
        debug_assert!(k < S256_SLOTS);
        let ph = self.hi & S256_HI_PAYLOAD;
        let pl = self.lo;
        let split = 224 - 24 * (k as u32);
        let low_edge = split - 24;

        let (mhi, mlo) = mask_ge_256(split);
        let up_hi = ph & mhi;
        let up_lo = pl & mlo;

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

/// Caches the two-word payload split for a cofacet insertion rank.
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
        let a = Simplex128::new(1.0, &[0, 1]);
        let b = Simplex128::new(1.0, &[0, 2]);
        let c = Simplex128::new(1.0, &[1, 2]);
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
        let tri = Simplex128::new(2.0, &[0, 1, 2]);
        let facet = tri.remove_vertex_at(1, encode_filtration(2.0));
        assert_eq!(facet.vertex_count(), 2);
        assert_eq!(facet.vertices()[..2], [2, 0]);
        assert_eq!(facet.filtration(), 2.0);
    }

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
        assert_eq!(s.vertices(), ids);
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
        let d = Simplex256::new(2.0, &[1, 2]);
        assert!(d < a);
    }

    #[test]
    fn s256_cofacet_matches_reference_at_every_rank() {
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
        let back = cof.remove_vertex_at(1, encode_filtration(1.0));
        assert_eq!(back, base);
    }

    #[test]
    fn s256_hash_uses_both_words() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let h = |s: Simplex256| {
            let mut hasher = DefaultHasher::new();
            s.hash(&mut hasher);
            hasher.finish()
        };
        let a = Simplex256::new(1.0, &[5, 3, 1]);
        let b = Simplex256::new(1.0, &[5, 3, 2]);
        let c = Simplex256::new(2.0, &[5, 3, 1]);
        assert_ne!(h(a), h(b));
        assert_ne!(h(a), h(c));
        let d = Simplex256::new(1.0, &[9, 8, 7, 6, 5, 4, 3]);
        let e = Simplex256::new(1.0, &[9, 8, 7, 6, 5, 4, 2]);
        assert_ne!(h(d), h(e));
    }
}
