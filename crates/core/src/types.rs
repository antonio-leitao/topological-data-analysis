// ═══════════════════════════════════════════════════════════════════════════════
// Result types
// ═══════════════════════════════════════════════════════════════════════════════

/// A single persistence interval [birth, death).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PersistenceInterval {
    pub birth: f32,
    pub death: f32, // f32::INFINITY for essential features
}

/// Barcode result organized by dimension.
#[derive(Debug, Clone)]
pub struct BarcodeResult {
    /// `intervals[d]` = persistence intervals in dimension d.
    pub intervals: Vec<Vec<PersistenceInterval>>,
}
impl BarcodeResult {
    /// Consume self, yielding one flat Vec<f32> per dimension.
    /// Zero-copy transmute: PersistenceInterval is #[repr(C)] {f32, f32}.
    pub fn into_flat_intervals(self) -> Vec<(usize, Vec<f32>)> {
        self.intervals
            .into_iter()
            .map(|mut intervals| {
                let n = intervals.len();
                let ptr = intervals.as_mut_ptr() as *mut f32;
                let len = n * 2;
                let cap = intervals.capacity() * 2;
                std::mem::forget(intervals);
                // SAFETY: PersistenceInterval is #[repr(C)] { birth: f32, death: f32 },
                // so a Vec<PersistenceInterval> with capacity C and length N has the
                // same layout as a Vec<f32> with capacity 2C and length 2N. The
                // forget() above relinquishes ownership before reconstruction.
                let flat = unsafe { Vec::from_raw_parts(ptr, len, cap) };
                (n, flat)
            })
            .collect()
    }
}
