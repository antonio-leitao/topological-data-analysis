//! Distance preprocessing: point cloud (or precomputed distance matrix) → the
//! adjacency the engine consumes, plus the Chebyshev radius and center.
//!
//! The V3 tiled kernel (`micro_kernel` / `pack_panel`) is the O(n²·d) hot path and
//! is byte-for-byte the source V3. Three things are built on top of it:
//!
//! - [`pdist_csr`] / [`dmat_csr`]: input → [`Adjacency`] (CSR parts kept at the
//!   minimax radius + `r_cheb` + `c_star`). This is what `persistent_homology`
//!   feeds to `BitCsrDistanceMatrix::from_csr_parts`.
//! - [`pdist_tiled_v3`] / [`condense_square_matrix`]: input → the full condensed
//!   lower-triangular distance vector, used by `filtration_size` (clique counting)
//!   and `opt::peel`.
//!
//! `R_cheb = min_i max_j d(i, j)` is only known after every distance is seen, so
//! the point-cloud adjacency build is *adaptive*: a single buffered pass while the
//! pair set fits a memory budget, falling back to a memory-safe two-pass build
//! (radius first, then adjacency at the known radius) for very large n.

const MR: usize = 16;
const NR: usize = 16;
const MC: usize = 64;
const NC: usize = 256;

/// Entries `>= NO_EDGE` are "no edge" (disconnected pairs): excluded from the
/// enclosing-radius reduction and from the adjacency.
const NO_EDGE: f32 = f32::MAX;

/// COO staging cost per buffered half-edge: `u16` src + `u16` dst + `f32` weight.
const COO_BYTES_PER_EDGE: usize = 8;

/// Budget for the transient single-pass COO buffer. Above this (very large n) the
/// point-cloud build switches to the two-pass path so memory stays bounded by the
/// adjacency rather than the full pair set. Sized so every shipped dataset
/// (largest `o3_8192` ≈ 268 MB worst-case COO) stays on the fast single-pass path.
const SINGLE_PASS_BUDGET_BYTES: usize = 1 << 30; // 1 GiB

// ═══════════════════════════════════════════════════════════════════════════════
// V3 tiled kernel (hot path — unchanged from source V3)
// ═══════════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn micro_kernel(a: &[f32], b: &[f32], d: usize) -> [[f32; NR]; MR] {
    let mut acc = [[0.0f32; NR]; MR];
    let a = &a[..d * MR];
    let b = &b[..d * NR];

    for (ak, bk) in a.chunks_exact(MR).zip(b.chunks_exact(NR)) {
        for m in 0..MR {
            let a_val = ak[m];
            let acc_row = &mut acc[m];
            for n in 0..NR {
                let diff = a_val - bk[n];
                acc_row[n] += diff * diff;
            }
        }
    }
    acc
}

fn pack_panel(
    data: &[f32],
    d: usize,
    start: usize,
    count: usize,
    width: usize,
    packed: &mut [f32],
) {
    let n_groups = (count + width - 1) / width;
    let used = n_groups * width * d;
    packed[..used].fill(0.0);

    for g in 0..n_groups {
        let actual = width.min(count - g * width);
        let group_base = g * width * d;
        for k in 0..d {
            let dst_off = group_base + k * width;
            for w in 0..actual {
                let src_point = start + g * width + w;
                packed[dst_off + w] = data[src_point * d + k];
            }
        }
    }
}

/// Distance of tile entry `(m, n)`. The full-tile path squares in `f64` then
/// `sqrt`s; the diagonal-straddling (partial) path uses `f32` `sqrt`. Preserving
/// this exact split keeps `R_cheb` and edge membership bit-identical to the
/// original kernel.
#[inline(always)]
fn tile_dist(tile: &[[f32; NR]; MR], m: usize, n: usize, full: bool) -> f32 {
    if full {
        ((tile[m][n] as f64).sqrt()) as f32
    } else {
        tile[m][n].sqrt()
    }
}

fn alloc_panels(d: usize) -> (Vec<f32>, Vec<f32>) {
    let a_buf_len = ((MC + MR - 1) / MR) * MR * d;
    let b_buf_len = ((NC + NR - 1) / NR) * NR * d;
    (vec![0.0f32; a_buf_len], vec![0.0f32; b_buf_len])
}

/// Drive the tiled lower-triangle traversal, invoking `on_tile` for each computed
/// `MR×NR` tile with `(i_min, j_min, mr_eff, nr_eff, full)`. `full` is true when
/// the whole tile lies strictly below the diagonal (every `i > j`); otherwise the
/// callback must apply the `i > j` filter itself. The closure inlines, so each
/// caller's loop is codegen-identical to a hand-written pass.
#[inline]
fn tiled_pass(
    data: &[f32],
    n: usize,
    d: usize,
    a_packed: &mut [f32],
    b_packed: &mut [f32],
    mut on_tile: impl FnMut(&[[f32; NR]; MR], usize, usize, usize, usize, bool),
) {
    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        pack_panel(data, d, jc, nc, NR, b_packed);
        let mut ic = jc;
        while ic < n {
            let mc = (n - ic).min(MC);
            pack_panel(data, d, ic, mc, MR, a_packed);
            let mut ir = 0;
            while ir < mc {
                let mr_eff = (mc - ir).min(MR);
                let a_off = (ir / MR) * MR * d;
                let mut jr = 0;
                while jr < nc {
                    let nr_eff = (nc - jr).min(NR);
                    let b_off = (jr / NR) * NR * d;
                    let i_min = ic + ir;
                    let i_max = i_min + mr_eff - 1;
                    let j_min = jc + jr;
                    let j_max = j_min + nr_eff - 1;
                    if i_max <= j_min {
                        jr += NR;
                        continue;
                    }
                    let tile = micro_kernel(&a_packed[a_off..], &b_packed[b_off..], d);
                    on_tile(&tile, i_min, j_min, mr_eff, nr_eff, i_min > j_max);
                    jr += NR;
                }
                ir += MR;
            }
            ic += MC;
        }
        jc += NC;
    }
}

/// `(c_star, r_cheb)` from the per-row maxima (eccentricities):
/// `r_cheb = min_i m[i]`, `c_star = argmin_i m[i]` (ties → lowest index).
#[inline]
fn reduce_minimax(max_dist: &[f32]) -> (usize, f32) {
    let mut c_star = 0usize;
    let mut r_cheb = max_dist[0];
    for i in 1..max_dist.len() {
        if max_dist[i] < r_cheb {
            r_cheb = max_dist[i];
            c_star = i;
        }
    }
    (c_star, r_cheb)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Adjacency: CSR parts + radius + center
// ═══════════════════════════════════════════════════════════════════════════════

/// Sparse adjacency in CSR form (rows DESCENDING by neighbour id, as
/// `BitCsrDistanceMatrix::from_csr_parts` requires) plus the Chebyshev radius and
/// center. Edges kept are those with `d ≤ min(threshold, r_cheb)`.
pub struct Adjacency {
    pub row_ptr: Vec<usize>,
    pub col: Vec<u16>,
    pub val: Vec<f32>,
    /// Retained as preprocessing metadata for optimization paths.
    #[allow(dead_code)]
    pub r_cheb: f32,
    /// Chebyshev center; part of pdist's contract (radius + center) for the
    /// `opt::peel` path. Currently unread by the vanilla pipeline.
    #[allow(dead_code)]
    pub c_star: usize,
}

impl Adjacency {
    fn empty(n: usize, r_cheb: f32) -> Self {
        Adjacency {
            row_ptr: vec![0usize; n + 1],
            col: Vec::new(),
            val: Vec::new(),
            r_cheb,
            c_star: 0,
        }
    }
}

/// Surviving half-edges `(i > j)` buffered during a tiled pass.
struct CooSink {
    src: Vec<u16>,
    dst: Vec<u16>,
    w: Vec<f32>,
}

impl CooSink {
    fn new() -> Self {
        CooSink {
            src: Vec::new(),
            dst: Vec::new(),
            w: Vec::new(),
        }
    }

    #[inline(always)]
    fn push(&mut self, i: usize, j: usize, d: f32) {
        self.src.push(i as u16);
        self.dst.push(j as u16);
        self.w.push(d);
    }
}

/// COO (half-edges) → symmetric CSR with rows DESCENDING by neighbour id. Keeps
/// only `d ≤ eff`; the tiled pass does not emit row-major, so each (short) row is
/// sorted descending — exactly what reference Ripser does to its neighbour lists.
fn finalize_csr(coo: &CooSink, n: usize, eff: f32) -> (Vec<usize>, Vec<u16>, Vec<f32>) {
    let mut row_ptr = vec![0usize; n + 1];
    for k in 0..coo.w.len() {
        if coo.w[k] <= eff {
            row_ptr[coo.src[k] as usize + 1] += 1;
            row_ptr[coo.dst[k] as usize + 1] += 1;
        }
    }
    for v in 0..n {
        row_ptr[v + 1] += row_ptr[v];
    }

    let nnz = row_ptr[n];
    let mut col = vec![0u16; nnz];
    let mut val = vec![0.0f32; nnz];
    let mut cur = row_ptr[..n].to_vec();
    for k in 0..coo.w.len() {
        let dd = coo.w[k];
        if dd <= eff {
            let (i, j) = (coo.src[k] as usize, coo.dst[k] as usize);
            let ci = cur[i];
            col[ci] = j as u16;
            val[ci] = dd;
            cur[i] = ci + 1;
            let cj = cur[j];
            col[cj] = i as u16;
            val[cj] = dd;
            cur[j] = cj + 1;
        }
    }

    let mut scratch: Vec<(u16, f32)> = Vec::new();
    for v in 0..n {
        let s = row_ptr[v];
        let e = row_ptr[v + 1];
        scratch.clear();
        scratch.extend(col[s..e].iter().copied().zip(val[s..e].iter().copied()));
        scratch.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        for (k, &(id, dv)) in scratch.iter().enumerate() {
            col[s + k] = id;
            val[s + k] = dv;
        }
    }

    (row_ptr, col, val)
}

/// Point cloud → adjacency + radius + center. Adaptive: single buffered pass when
/// the pair set fits [`SINGLE_PASS_BUDGET_BYTES`], else a memory-safe two-pass
/// build. Both paths produce a bit-identical [`Adjacency`].
pub fn pdist_csr(data: &[f32], n: usize, d: usize, threshold: f32) -> Adjacency {
    debug_assert_eq!(data.len(), n * d);
    if n < 2 {
        return Adjacency::empty(n, 0.0);
    }

    let pairs = n * (n - 1) / 2;
    if pairs.saturating_mul(COO_BYTES_PER_EDGE) <= SINGLE_PASS_BUDGET_BYTES {
        adjacency_single_pass(data, n, d, threshold)
    } else {
        adjacency_two_pass(data, n, d, threshold)
    }
}

/// Single buffered pass: compute every distance once, accumulate the per-row
/// maxima, and buffer the `d ≤ threshold` superset; `finalize_csr` then tightens
/// to `d ≤ eff` once `R_cheb` is known.
pub(crate) fn adjacency_single_pass(data: &[f32], n: usize, d: usize, threshold: f32) -> Adjacency {
    let (mut a_packed, mut b_packed) = alloc_panels(d);
    let mut max_dist = vec![0.0f32; n];
    let mut coo = CooSink::new();

    tiled_pass(
        data,
        n,
        d,
        &mut a_packed,
        &mut b_packed,
        |tile, i_min, j_min, mr, nr, full| {
            for m in 0..mr {
                let i = i_min + m;
                let mut row_max = max_dist[i];
                for nn in 0..nr {
                    let j = j_min + nn;
                    if full || i > j {
                        let dist = tile_dist(tile, m, nn, full);
                        if dist > row_max {
                            row_max = dist;
                        }
                        if dist > max_dist[j] {
                            max_dist[j] = dist;
                        }
                        if dist <= threshold {
                            coo.push(i, j, dist);
                        }
                    }
                }
                max_dist[i] = row_max;
            }
        },
    );

    let (c_star, r_cheb) = reduce_minimax(&max_dist);
    let eff = threshold.min(r_cheb);
    let (row_ptr, col, val) = finalize_csr(&coo, n, eff);
    Adjacency {
        row_ptr,
        col,
        val,
        r_cheb,
        c_star,
    }
}

/// Memory-safe two-pass build for large n: pass 1 computes the per-row maxima
/// (`R_cheb`, `c_star`) without storing distances; pass 2 recomputes distances
/// and buffers only the `d ≤ eff` survivors.
pub(crate) fn adjacency_two_pass(data: &[f32], n: usize, d: usize, threshold: f32) -> Adjacency {
    let (mut a_packed, mut b_packed) = alloc_panels(d);

    // Pass 1: radius + center only.
    let mut max_dist = vec![0.0f32; n];
    tiled_pass(
        data,
        n,
        d,
        &mut a_packed,
        &mut b_packed,
        |tile, i_min, j_min, mr, nr, full| {
            for m in 0..mr {
                let i = i_min + m;
                let mut row_max = max_dist[i];
                for nn in 0..nr {
                    let j = j_min + nn;
                    if full || i > j {
                        let dist = tile_dist(tile, m, nn, full);
                        if dist > row_max {
                            row_max = dist;
                        }
                        if dist > max_dist[j] {
                            max_dist[j] = dist;
                        }
                    }
                }
                max_dist[i] = row_max;
            }
        },
    );

    let (c_star, r_cheb) = reduce_minimax(&max_dist);
    let eff = threshold.min(r_cheb);

    // Pass 2: keep only the survivors at the now-known radius.
    let mut coo = CooSink::new();
    tiled_pass(
        data,
        n,
        d,
        &mut a_packed,
        &mut b_packed,
        |tile, i_min, j_min, mr, nr, full| {
            for m in 0..mr {
                let i = i_min + m;
                for nn in 0..nr {
                    let j = j_min + nn;
                    if full || i > j {
                        let dist = tile_dist(tile, m, nn, full);
                        if dist <= eff {
                            coo.push(i, j, dist);
                        }
                    }
                }
            }
        },
    );

    let (row_ptr, col, val) = finalize_csr(&coo, n, eff);
    Adjacency {
        row_ptr,
        col,
        val,
        r_cheb,
        c_star,
    }
}

/// Precomputed square `(n, n)` distance matrix → adjacency + radius + center.
/// Reads the strict lower triangle (`mat[i*n + j]`, `j < i`); entries `≥ NO_EDGE`
/// are disconnected. Memory-bounded by the output (the input is already
/// materialised), so a plain two-pass over the matrix suffices.
pub fn dmat_csr(mat: &[f32], n: usize, threshold: f32) -> Adjacency {
    debug_assert_eq!(mat.len(), n * n);
    if n < 2 {
        return Adjacency::empty(n, f32::INFINITY);
    }

    // Pass 1: per-row maxima → r_cheb, c_star (NEG_INFINITY marks isolated rows).
    let mut max_dist = vec![f32::NEG_INFINITY; n];
    for i in 1..n {
        let row_base = i * n;
        let mut row_max = max_dist[i];
        for j in 0..i {
            let dd = mat[row_base + j];
            if dd < NO_EDGE {
                if dd > row_max {
                    row_max = dd;
                }
                if dd > max_dist[j] {
                    max_dist[j] = dd;
                }
            }
        }
        max_dist[i] = row_max;
    }
    let mut c_star = 0usize;
    let mut r_cheb = f32::INFINITY;
    for i in 0..n {
        let m = max_dist[i];
        if m > f32::NEG_INFINITY && m < r_cheb {
            r_cheb = m;
            c_star = i;
        }
    }
    let eff = threshold.min(r_cheb);

    // Pass 2: count degrees → prefix-sum → scatter both directions.
    let mut row_ptr = vec![0usize; n + 1];
    for i in 1..n {
        let row_base = i * n;
        for j in 0..i {
            let dd = mat[row_base + j];
            if dd <= eff && dd < NO_EDGE {
                row_ptr[i + 1] += 1;
                row_ptr[j + 1] += 1;
            }
        }
    }
    for v in 0..n {
        row_ptr[v + 1] += row_ptr[v];
    }

    let nnz = row_ptr[n];
    let mut col = vec![0u16; nnz];
    let mut val = vec![0.0f32; nnz];
    let mut cur = row_ptr[..n].to_vec();
    for i in 1..n {
        let row_base = i * n;
        for j in 0..i {
            let dd = mat[row_base + j];
            if dd <= eff && dd < NO_EDGE {
                let ci = cur[i];
                col[ci] = j as u16;
                val[ci] = dd;
                cur[i] = ci + 1;
                let cj = cur[j];
                col[cj] = i as u16;
                val[cj] = dd;
                cur[j] = cj + 1;
            }
        }
    }

    // Scatter produces each row ascending by id; reverse to descending.
    for v in 0..n {
        let s = row_ptr[v];
        let e = row_ptr[v + 1];
        col[s..e].reverse();
        val[s..e].reverse();
    }

    Adjacency {
        row_ptr,
        col,
        val,
        r_cheb,
        c_star,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Full condensed distance vector (for filtration_size / opt::peel)
// ═══════════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn store_tile_full(
    tile: &[[f32; NR]; MR],
    out: &mut [f32],
    max_dist: &mut [f32],
    i_base: usize,
    j_base: usize,
    mr_eff: usize,
    nr_eff: usize,
) {
    for m in 0..mr_eff {
        let i = i_base + m;
        let row_base = i * (i - 1) / 2;
        let mut row_max = max_dist[i];

        for n in 0..nr_eff {
            let j = j_base + n;
            let dist = ((tile[m][n] as f64).sqrt()) as f32;
            out[row_base + j] = dist;

            if dist > row_max {
                row_max = dist;
            }
            if dist > max_dist[j] {
                max_dist[j] = dist;
            }
        }
        max_dist[i] = row_max;
    }
}

#[inline(always)]
fn store_tile_partial(
    tile: &[[f32; NR]; MR],
    out: &mut [f32],
    max_dist: &mut [f32],
    i_base: usize,
    j_base: usize,
    mr_eff: usize,
    nr_eff: usize,
) {
    for m in 0..mr_eff {
        let i = i_base + m;
        let mut row_max = max_dist[i];

        for n in 0..nr_eff {
            let j = j_base + n;
            if i > j {
                let dist = tile[m][n].sqrt();
                out[i * (i - 1) / 2 + j] = dist;

                if dist > row_max {
                    row_max = dist;
                }
                if dist > max_dist[j] {
                    max_dist[j] = dist;
                }
            }
        }
        max_dist[i] = row_max;
    }
}

/// Pairwise Euclidean distances (condensed lower-triangular), the Chebyshev
/// center, and the minimax radius `(distances, c_star, r_cheb)`. Used by
/// `filtration_size` (clique counting) and `opt::peel`.
pub fn pdist_tiled_v3(data: &[f32], n: usize, d: usize) -> (Vec<f32>, usize, f32) {
    debug_assert_eq!(data.len(), n * d);
    if n < 2 {
        return (vec![], 0, 0.0);
    }

    let out_len = n * (n - 1) / 2;
    let mut out = vec![0.0f32; out_len];
    let mut max_dist = vec![0.0f32; n];
    let (mut a_packed, mut b_packed) = alloc_panels(d);

    tiled_pass(
        data,
        n,
        d,
        &mut a_packed,
        &mut b_packed,
        |tile, i_min, j_min, mr, nr, full| {
            if full {
                store_tile_full(tile, &mut out, &mut max_dist, i_min, j_min, mr, nr);
            } else {
                store_tile_partial(tile, &mut out, &mut max_dist, i_min, j_min, mr, nr);
            }
        },
    );

    let (c_star, r_cheb) = reduce_minimax(&max_dist);
    (out, c_star, r_cheb)
}

/// Extract the strict lower triangle of a row-major `(n, n)` matrix into the
/// condensed layout, together with the Chebyshev center and minimax radius
/// `(condensed, c_star, r_cheb)` (length `n*(n-1)/2`). Entries `≥ NO_EDGE` are
/// stored verbatim but excluded from the minimax; an all-no-edge row is isolated.
/// Mirrors [`pdist_tiled_v3`]'s return shape so distance-matrix and point-cloud
/// inputs feed `filtration_size` / `opt::peel` identically.
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
    let mut max_dist = vec![f32::NEG_INFINITY; n];

    for i in 1..n {
        let row_base = i * n;
        let mut row_max = max_dist[i];
        for j in 0..i {
            let d = unsafe { *mat.get_unchecked(row_base + j) };
            data.push(d);
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

#[cfg(test)]
pub(crate) fn square_from_lower_tri(n: usize, lt: &[f32]) -> Vec<f32> {
    debug_assert_eq!(lt.len(), n * (n - 1) / 2);
    let mut sq = vec![0.0f32; n * n];
    let mut p = 0;
    for i in 1..n {
        for j in 0..i {
            let d = lt[p];
            p += 1;
            sq[i * n + j] = d;
            sq[j * n + i] = d;
        }
    }
    sq
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The adaptive single-pass and the memory-safe two-pass build must produce a
    /// bit-identical `Adjacency` on the same point cloud.
    #[test]
    fn single_pass_matches_two_pass() {
        // A few small clouds across dims, with and without a finite threshold.
        let clouds: &[(usize, usize, u64)] = &[(40, 2, 1), (33, 3, 7), (50, 5, 99)];
        for &(n, d, seed) in clouds {
            let mut s = seed;
            let mut next = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / 16_777_216.0
            };
            let data: Vec<f32> = (0..n * d).map(|_| next()).collect();

            for &thr in &[f32::INFINITY, 0.5f32, 0.3] {
                let a = adjacency_single_pass(&data, n, d, thr);
                let b = adjacency_two_pass(&data, n, d, thr);
                assert_eq!(a.r_cheb.to_bits(), b.r_cheb.to_bits(), "r_cheb (thr={thr})");
                assert_eq!(a.c_star, b.c_star, "c_star (thr={thr})");
                assert_eq!(a.row_ptr, b.row_ptr, "row_ptr (thr={thr})");
                assert_eq!(a.col, b.col, "col (thr={thr})");
                let av: Vec<u32> = a.val.iter().map(|x| x.to_bits()).collect();
                let bv: Vec<u32> = b.val.iter().map(|x| x.to_bits()).collect();
                assert_eq!(av, bv, "val (thr={thr})");
            }
        }
    }
}
