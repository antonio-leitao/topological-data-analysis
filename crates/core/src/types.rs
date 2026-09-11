/// A single persistence interval [birth, death).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PersistenceInterval {
    pub birth: f32,
    pub death: f32,
}

/// Barcode result organized by dimension.
#[derive(Debug, Clone)]
pub struct BarcodeResult {
    /// Persistence intervals indexed by homology dimension.
    pub intervals: Vec<Vec<PersistenceInterval>>,
}

impl BarcodeResult {
    /// Consume the barcode into one flat `[birth, death, ...]` vector per dimension.
    pub fn into_flat_intervals(self) -> Vec<(usize, Vec<f32>)> {
        self.intervals
            .into_iter()
            .map(|mut intervals| {
                let n = intervals.len();
                let ptr = intervals.as_mut_ptr() as *mut f32;
                let len = n * 2;
                let cap = intervals.capacity() * 2;
                std::mem::forget(intervals);
                // SAFETY: `PersistenceInterval` is two `f32` fields with C layout.
                // `forget` transfers ownership before rebuilding the vector.
                let flat = unsafe { Vec::from_raw_parts(ptr, len, cap) };
                (n, flat)
            })
            .collect()
    }
}
