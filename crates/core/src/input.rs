//! Element types the entry points read in place, so callers holding `f64`
//! values or `i32`/`i64` indices (numpy, scipy) never pay a conversion copy.

use std::borrow::Cow;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
}

/// A distance or coordinate element: `f32`, or `f64` rounded to `f32` on read.
pub trait Real: Copy + Send + Sync + sealed::Sealed {
    fn to_f32(self) -> f32;

    /// `data` as `f32`: borrowed when it already is, converted otherwise.
    fn as_f32_slice(data: &[Self]) -> Cow<'_, [f32]>;
}

impl Real for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }

    fn as_f32_slice(data: &[f32]) -> Cow<'_, [f32]> {
        Cow::Borrowed(data)
    }
}

impl Real for f64 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }

    fn as_f32_slice(data: &[f64]) -> Cow<'_, [f32]> {
        Cow::Owned(data.iter().map(|&x| x as f32).collect())
    }
}

/// A sparse vertex index or offset: `u32`, `u64`, `i32`, or `i64`.
pub trait SparseIndex: Copy + Send + Sync + sealed::Sealed {
    /// `None` when negative or wider than `u32`.
    fn to_u32(self) -> Option<u32>;

    /// `None` when negative or wider than `usize`.
    fn to_usize(self) -> Option<usize>;

    /// Exact value, for error messages.
    fn to_i128(self) -> i128;
}

macro_rules! sparse_index {
    ($($t:ty),*) => {$(
        impl SparseIndex for $t {
            #[inline(always)]
            fn to_u32(self) -> Option<u32> {
                u32::try_from(self).ok()
            }

            #[inline(always)]
            fn to_usize(self) -> Option<usize> {
                usize::try_from(self).ok()
            }

            fn to_i128(self) -> i128 {
                self as i128
            }
        }
    )*};
}

sparse_index!(u32, u64, i32, i64);
