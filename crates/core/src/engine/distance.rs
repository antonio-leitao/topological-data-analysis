// ═══════════════════════════════════════════════════════════════════════════════
// distance.rs — Dense lower-triangular distance matrix (f32)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Stores pairwise distances as f32 in a flat lower-triangular layout.
// The hot path (cofacet diameter computation) reads scattered entries from a
// single row — keeping data as f32 avoids per-lookup encoding overhead and
// lets the compiler emit native float max instructions (vmaxss/vmaxps).
//
// Construction:
//   - For point clouds, callers compute distances via `pdist::pdist_tiled_v3`
//     (which is tiled, register-blocked, and fuses the minimax computation)
//     and pass the result to `from_lower_triangular`.
//   - For precomputed distance matrices, `from_square_matrix` extracts the
//     strict lower triangle and computes minimax in a single pass.
//
// Encoding to u32 happens only once per *surviving* cofacet (when building
// the Simplex128), not per distance lookup.

/// Sentinel: entries `>= NO_EDGE` are treated as "no edge" (disconnected
/// pairs) when computing the enclosing-radius threshold. Catches both
/// `f32::INFINITY` and `f32::MAX`.
const NO_EDGE: f32 = f32::MAX;

/// Dense lower-triangular distance matrix.
///
/// Entry (i, j) with i > j is stored at index `i*(i-1)/2 + j`.
/// Diagonal entries are implicitly zero.
pub struct DistanceMatrix {
    n: usize,
    /// Flat lower-triangular data. Length = n*(n-1)/2.
    data: Vec<f32>,
}

impl DistanceMatrix {
    // ── Constructors ────────────────────────────────────────────────────────

    /// Build from a pre-computed flat lower-triangular distance vector.
    /// `data[i*(i-1)/2 + j]` = distance between points i and j, for i > j.
    ///
    /// Test-only: production builds BitCSR from `from_square_matrix` (distance
    /// matrices) or `pdist_csr` (point clouds), never from a bare lower triangle.
    ///
    /// # Panics
    /// If `data.len() != n*(n-1)/2` or `n > u16::MAX`.
    /// Negative entries trigger a debug-only assert.
    #[cfg(test)]
    pub fn from_lower_triangular(n: usize, data: Vec<f32>) -> Self {
        let expected = n * (n - 1) / 2;
        assert_eq!(
            data.len(),
            expected,
            "Expected {} entries for n={}, got {}",
            expected,
            n,
            data.len()
        );
        assert!(n <= u16::MAX as usize, "n={} exceeds u16::MAX", n);
        debug_assert!(
            data.iter().all(|&d| d >= 0.0),
            "Distances must be non-negative"
        );
        DistanceMatrix { n, data }
    }

    /// Build from a row-major `(n, n)` matrix. Reads only the strict lower
    /// triangle. Returns `(matrix, minimax_radius)`.
    ///
    /// Entries `>= f32::MAX` are stored verbatim but excluded from the
    /// minimax computation, so disconnected pairs don't poison the default
    /// threshold.
    ///
    /// Thin wrapper over [`condense_square_matrix`] for callers that only need
    /// the matrix and its radius (no Chebyshev center).
    ///
    /// # Panics
    /// If `mat.len() != n*n` or `n > u16::MAX`.
    pub fn from_square_matrix(mat: &[f32], n: usize) -> (Self, f32) {
        let (data, _c_star, minimax) = condense_square_matrix(mat, n);
        (DistanceMatrix { n, data }, minimax)
    }

    // ── Accessors ───────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn n(&self) -> usize {
        self.n
    }

    #[inline(always)]
    pub fn raw(&self) -> &[f32] {
        &self.data
    }
}

/// Extract the strict lower triangle of a row-major `(n, n)` matrix into the
/// condensed layout, together with the Chebyshev center and minimax radius.
///
/// Returns `(condensed, c_star, r_cheb)` where `condensed` has length
/// `n*(n-1)/2`, `r_cheb = min_i max_j d(i, j)`, and `c_star = argmin_i max_j
/// d(i, j)` (ties broken toward the lowest index). This mirrors the return shape
/// of [`crate::preprocess::pdist::pdist_tiled_v3`], so point-cloud and
/// distance-matrix inputs feed the same downstream pipeline — including
/// [`crate::opt::peel::peel`], which needs `c_star`.
///
/// Entries `>= f32::MAX` are stored verbatim but excluded from the minimax
/// reduction, so disconnected pairs don't poison the default threshold; a row
/// of all no-edge entries is treated as isolated and skipped when selecting the
/// center. If every row is isolated, `r_cheb` falls back to `f32::INFINITY` and
/// `c_star` to `0` (the correct degenerate behaviour).
///
/// # Panics
/// If `mat.len() != n*n` or `n > u16::MAX`.
pub fn condense_square_matrix(mat: &[f32], n: usize) -> (Vec<f32>, usize, f32) {
    assert_eq!(
        mat.len(),
        n * n,
        "expected {} entries for n={}, got {}",
        n * n,
        n,
        mat.len()
    );
    assert!(n <= u16::MAX as usize, "n={} exceeds u16::MAX", n);

    let size = n * (n - 1) / 2;
    let mut data: Vec<f32> = Vec::with_capacity(size);
    // NEG_INFINITY rather than 0 so an isolated point (row with all no-edge
    // entries) stays at NEG_INFINITY and is filtered out below.
    let mut max_dist = vec![f32::NEG_INFINITY; n];

    for i in 1..n {
        let row_base = i * n;
        let mut row_max = max_dist[i];
        for j in 0..i {
            // SAFETY: row_base + j = i*n + j < n*n = mat.len().
            let d = unsafe { *mat.get_unchecked(row_base + j) };
            data.push(d);
            // One predictable outer branch — always true for connected pairs
            // (the common case), essentially free.
            if d < NO_EDGE {
                if d > row_max {
                    row_max = d;
                }
                if d > max_dist[j] {
                    max_dist[j] = d;
                }
            }
        }
        max_dist[i] = row_max;
    }

    // Minimax radius AND Chebyshev center in one O(n) reduction over the
    // per-row maxima. Rows that never saw a finite entry (still NEG_INFINITY)
    // are isolated and ignored; if all are isolated, r_cheb stays INFINITY.
    let mut c_star = 0usize;
    let mut r_cheb = f32::INFINITY;
    for i in 0..n {
        let m = max_dist[i];
        if m > f32::NEG_INFINITY && m < r_cheb {
            r_cheb = m;
            c_star = i;
        }
    }

    (data, c_star, r_cheb)
}
