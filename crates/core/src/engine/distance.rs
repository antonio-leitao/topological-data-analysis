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
    /// # Panics
    /// If `data.len() != n*(n-1)/2` or `n > u16::MAX`.
    /// Negative entries trigger a debug-only assert.
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
    /// # Panics
    /// If `mat.len() != n*n` or `n > u16::MAX`.
    pub fn from_square_matrix(mat: &[f32], n: usize) -> (Self, f32) {
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
        // NEG_INFINITY rather than 0 so an isolated point (row with all
        // no-edge entries) stays at NEG_INFINITY and is filtered out below.
        let mut max_dist = vec![f32::NEG_INFINITY; n];

        for i in 1..n {
            let row_base = i * n;
            let mut row_max = max_dist[i];
            for j in 0..i {
                // SAFETY: row_base + j = i*n + j < n*n = mat.len().
                let d = unsafe { *mat.get_unchecked(row_base + j) };
                data.push(d);
                // One predictable outer branch — always true for connected
                // pairs (the common case), essentially free.
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

        // Filter out rows that never saw a finite entry. If every row is
        // isolated, fall back to INFINITY — the engine then processes nothing
        // above zero, which is the correct degenerate behaviour.
        let minimax = max_dist
            .iter()
            .copied()
            .filter(|&m| m > f32::NEG_INFINITY)
            .fold(f32::INFINITY, f32::min);

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

    /// Distance between points i and j. Symmetric; dist(i, i) = 0.0.
    ///
    /// # Safety contract
    /// Both i and j must be valid point indices (< n). The bounds check is
    /// elided via `get_unchecked` because this is called O(n) times per
    /// cofacet candidate in the innermost loop.
    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        if i == j {
            return 0.0;
        }
        let (r, c) = if i > j { (i, j) } else { (j, i) };
        // SAFETY: r > c ≥ 0, and both < n, so index < n*(n-1)/2 = data.len().
        unsafe { *self.data.get_unchecked(r * (r - 1) / 2 + c) }
    }

    /// Distance between points `i` and `j` where the caller guarantees `i > j`.
    /// Skips the `i == j` check and the ordering swap.
    ///
    /// # Safety
    /// Caller must ensure `i > j` and `i < self.n()`.
    #[inline(always)]
    pub unsafe fn get_unchecked_ordered(&self, i: usize, j: usize) -> f32 {
        debug_assert!(i > j && i < self.n);
        *self.data.get_unchecked(i * (i - 1) / 2 + j)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cofacet diameter — the single hottest function in the codebase
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the diameter of the cofacet formed by inserting `new_v` into a
/// simplex with vertices `verts[0..vc]` and existing diameter `base_diam`.
///
/// Returns `Some(diameter)` if the cofacet diameter ≤ threshold, `None` otherwise.
///
/// The early-exit on threshold is the most important optimization here:
/// most candidate vertices produce cofacets above threshold, and bailing at the
/// first exceeding distance avoids the remaining (vc - i - 1) lookups.
#[inline(always)]
pub fn cofacet_diameter(
    new_v: u16,
    verts: &[u16; 6],
    vc: usize,
    base_diam: f32,
    dist: &DistanceMatrix,
    threshold: f32,
) -> Option<f32> {
    let mut diam = base_diam;
    // The compiler will unroll this for small vc (≤ 6).
    // Each iteration: one distance lookup + one branch (threshold) + one cmov (max).
    for i in 0..vc {
        let d = dist.get(new_v as usize, verts[i] as usize);
        if d > threshold {
            return None;
        }
        // Branchless max — the compiler emits vmaxss or cmov.
        if d > diam {
            diam = d;
        }
    }
    Some(diam)
}

/// Same as `cofacet_diameter`, but the caller GUARANTEES `new_v > verts[i]`
/// for all i in 0..vc. This lets us:
///   - precompute the row base once (vs inside each `dist.get`)
///   - skip the `i == j` branch (caller has already filtered new_v ≠ verts[i])
///   - skip the `i > j` swap (always true here)
///
/// # Safety
/// Caller must ensure `new_v > verts[i]` for every i ∈ [0, vc), and
/// `new_v < dist.n()`.
#[inline(always)]
pub fn cofacet_diameter_v_larger(
    new_v: u16,
    verts: &[u16; 6],
    vc: usize,
    base_diam: f32,
    dist: &DistanceMatrix,
    threshold: f32,
) -> Option<f32> {
    let new_v = new_v as usize;
    let row_base = new_v * (new_v - 1) / 2;
    let raw = dist.raw();
    let mut diam = base_diam;
    for i in 0..vc {
        // SAFETY: new_v > verts[i] (caller guarantee) and both < n.
        let d = unsafe { *raw.get_unchecked(row_base + verts[i] as usize) };
        if d > threshold {
            return None;
        }
        if d > diam {
            diam = d;
        }
    }
    Some(diam)
}

/// Compute the diameter of a facet formed by removing vertex at slot `skip`
/// from `verts[0..vc]`.
///
/// This recomputes max pairwise distance over the remaining vertices.
/// For dim ≤ 5, this is at most (5 choose 2) = 10 pair checks.
#[inline]
pub fn facet_diameter(verts: &[u16; 6], vc: usize, skip: usize, dist: &DistanceMatrix) -> f32 {
    let mut diam: f32 = 0.0;
    for i in 0..vc {
        if i == skip {
            continue;
        }
        for j in (i + 1)..vc {
            if j == skip {
                continue;
            }
            let d = dist.get(verts[i] as usize, verts[j] as usize);
            if d > diam {
                diam = d;
            }
        }
    }
    diam
}
