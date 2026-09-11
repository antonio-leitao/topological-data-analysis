//! FxHash-style hashing for the crate's integer and packed-simplex keys.
//!
//! Shared by the engine (128/256-bit simplex keys, via `write_u128`) and the
//! quotient construction (`u32` roots, via `write_u32`). Both specialisations
//! live here so the two consumers cannot drift apart.

use std::hash::{BuildHasherDefault, Hasher};

const FX_SEED: u64 = 0x517c_c1b7_2722_0a95;

#[derive(Default)]
pub struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ (b as u64)).wrapping_mul(FX_SEED);
        }
    }

    #[inline(always)]
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(FX_SEED);
    }

    /// Replaces the state rather than mixing into it, so a derived `Hash` that
    /// writes more than one word would ignore all but the last.
    #[inline(always)]
    fn write_u128(&mut self, i: u128) {
        let lo = i as u64;
        let hi = (i >> 64) as u64;
        self.0 = lo.wrapping_mul(FX_SEED) ^ hi;
    }
}

pub type FxBuildHasher = BuildHasherDefault<FxHasher>;
pub type FxHashMap<K, V> = std::collections::HashMap<K, V, FxBuildHasher>;
