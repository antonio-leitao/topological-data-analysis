//! Tiled distance computation (V3 kernel) with a fused minimax reduction.
//!
//! Single public entry point: [`pdist_tiled_v3`]. Given a row-major point cloud
//! it returns the lower-triangular *condensed* distance vector together with the
//! minimax (Chebyshev) radius `R_cheb = min_i max_j d(i, j)` and the Chebyshev
//! center `c⋆ = argmin_i max_j d(i, j)` — the standard safe truncation radius
//! for a Vietoris–Rips filtration, plus the cone point a tighter truncation
//! peels from.
//!
//! The tile kernel, panel packing, and store functions — the entire O(n²·d) hot
//! path — are byte-for-byte identical to the source V3. The only
//! application-specific step is the trailing reduction: a single O(n) pass over
//! the per-row maxima (the eccentricities, already accumulated in the hot path)
//! that yields both `R_cheb` (their minimum) and `c⋆` (its argmin) at no extra
//! cost.

const MR: usize = 16;
const NR: usize = 16;
const MC: usize = 64;
const NC: usize = 256;

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

/// Pairwise Euclidean distances, the Chebyshev center, and the minimax radius.
///
/// Returns `(distances, c_star, r_cheb)` where `distances` is the
/// lower-triangular condensed matrix (length `n*(n-1)/2`),
/// `r_cheb = min_i max_j d(i, j)` is a valid Vietoris–Rips truncation radius,
/// and `c_star = argmin_i max_j d(i, j)` is the Chebyshev center — both read off
/// the per-row maxima for free in a single O(n) reduction. Feed all three into
/// [`crate::peeling::peel`] to tighten the radius without re-deriving the center.
pub fn pdist_tiled_v3(data: &[f32], n: usize, d: usize) -> (Vec<f32>, usize, f32) {
    debug_assert_eq!(data.len(), n * d);
    if n < 2 {
        return (vec![], 0, 0.0);
    }

    let out_len = n * (n - 1) / 2;
    let mut out = vec![0.0f32; out_len];
    let mut max_dist = vec![0.0f32; n];

    let a_buf_len = ((MC + MR - 1) / MR) * MR * d;
    let b_buf_len = ((NC + NR - 1) / NR) * NR * d;
    let mut a_packed = vec![0.0f32; a_buf_len];
    let mut b_packed = vec![0.0f32; b_buf_len];

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        pack_panel(data, d, jc, nc, NR, &mut b_packed);

        let mut ic = jc;
        while ic < n {
            let mc = (n - ic).min(MC);
            pack_panel(data, d, ic, mc, MR, &mut a_packed);

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

                    if i_min > j_max {
                        store_tile_full(
                            &tile,
                            &mut out,
                            &mut max_dist,
                            i_min,
                            j_min,
                            mr_eff,
                            nr_eff,
                        );
                    } else {
                        store_tile_partial(
                            &tile,
                            &mut out,
                            &mut max_dist,
                            i_min,
                            j_min,
                            mr_eff,
                            nr_eff,
                        );
                    }
                    jr += NR;
                }
                ir += MR;
            }
            ic += MC;
        }
        jc += NC;
    }

    // Minimax radius AND Chebyshev center in one O(n) reduction over the
    // per-row maxima (the eccentricities, accumulated for free in the hot path
    // above): r_cheb = min_i max_dist[i], c_star = argmin_i max_dist[i].
    // Tracking the argmin alongside the min is free; the hot path is untouched.
    let mut c_star = 0usize;
    let mut r_cheb = max_dist[0];
    for i in 1..n {
        if max_dist[i] < r_cheb {
            r_cheb = max_dist[i];
            c_star = i;
        }
    }

    (out, c_star, r_cheb)
}

// CSR PART
/// Surviving half-edges (i>j) plus per-row eccentricities, filled during the
/// single tiled pass. `threshold` is the *user* threshold; R_cheb is not yet
/// known here, so we buffer the superset `d ≤ threshold` and tighten in finalize.
struct CooSink {
    src: Vec<u16>,
    dst: Vec<u16>,
    w: Vec<f32>,
    max_dist: Vec<f32>,
    threshold: f32,
}

#[inline(always)]
fn store_tile_full_sparse(
    tile: &[[f32; NR]; MR],
    coo: &mut CooSink,
    i_base: usize,
    j_base: usize,
    mr_eff: usize,
    nr_eff: usize,
) {
    for m in 0..mr_eff {
        let i = i_base + m;
        let mut row_max = coo.max_dist[i];
        for n in 0..nr_eff {
            let j = j_base + n;
            let dist = ((tile[m][n] as f64).sqrt()) as f32; // matches store_tile_full
            if dist > row_max {
                row_max = dist;
            }
            if dist > coo.max_dist[j] {
                coo.max_dist[j] = dist;
            }
            if dist <= coo.threshold {
                coo.src.push(i as u16);
                coo.dst.push(j as u16);
                coo.w.push(dist);
            }
        }
        coo.max_dist[i] = row_max;
    }
}

#[inline(always)]
fn store_tile_partial_sparse(
    tile: &[[f32; NR]; MR],
    coo: &mut CooSink,
    i_base: usize,
    j_base: usize,
    mr_eff: usize,
    nr_eff: usize,
) {
    for m in 0..mr_eff {
        let i = i_base + m;
        let mut row_max = coo.max_dist[i];
        for n in 0..nr_eff {
            let j = j_base + n;
            if i > j {
                let dist = tile[m][n].sqrt(); // matches store_tile_partial (f32)
                if dist > row_max {
                    row_max = dist;
                }
                if dist > coo.max_dist[j] {
                    coo.max_dist[j] = dist;
                }
                if dist <= coo.threshold {
                    coo.src.push(i as u16);
                    coo.dst.push(j as u16);
                    coo.w.push(dist);
                }
            }
        }
        coo.max_dist[i] = row_max;
    }
}

/// Fused point-cloud → CSR + minimax radius, in a single distance-kernel pass.
///
/// Returns `(row_ptr, col, val, r_cheb)`, with `col`/`val` DESCENDING within each
/// row (the order `BitCsrDistanceMatrix::from_csr_parts` requires). `threshold`
/// is the user threshold
/// (`f32::INFINITY` if none); edges with `d ≤ min(threshold, r_cheb)` are kept.
/// `r_cheb` is the threshold-free minimax over all pairs, identical to
/// `pdist_tiled_v3`, so the caller resolves the engine threshold the same way.
pub fn pdist_csr(
    data: &[f32],
    n: usize,
    d: usize,
    threshold: f32,
) -> (Vec<usize>, Vec<u16>, Vec<f32>, f32) {
    debug_assert_eq!(data.len(), n * d);
    if n < 2 {
        return (vec![0usize; n + 1], Vec::new(), Vec::new(), 0.0);
    }

    let mut coo = CooSink {
        src: Vec::new(),
        dst: Vec::new(),
        w: Vec::new(),
        max_dist: vec![0.0f32; n],
        threshold,
    };
    // If you can estimate degree, `coo.{src,dst,w}.reserve(n * est_degree)` keeps
    // the hot-path pushes off the realloc path.

    let a_buf_len = ((MC + MR - 1) / MR) * MR * d;
    let b_buf_len = ((NC + NR - 1) / NR) * NR * d;
    let mut a_packed = vec![0.0f32; a_buf_len];
    let mut b_packed = vec![0.0f32; b_buf_len];

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        pack_panel(data, d, jc, nc, NR, &mut b_packed);
        let mut ic = jc;
        while ic < n {
            let mc = (n - ic).min(MC);
            pack_panel(data, d, ic, mc, MR, &mut a_packed);
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
                    if i_min > j_max {
                        store_tile_full_sparse(&tile, &mut coo, i_min, j_min, mr_eff, nr_eff);
                    } else {
                        store_tile_partial_sparse(&tile, &mut coo, i_min, j_min, mr_eff, nr_eff);
                    }
                    jr += NR;
                }
                ir += MR;
            }
            ic += MC;
        }
        jc += NC;
    }

    // R_cheb: min of per-row maxima — same reduction as pdist_tiled_v3.
    let mut r_cheb = coo.max_dist[0];
    for i in 1..n {
        if coo.max_dist[i] < r_cheb {
            r_cheb = coo.max_dist[i];
        }
    }
    let eff = threshold.min(r_cheb);

    // Finalize: count degrees at eff → prefix-sum row_ptr → scatter both
    // directions. (COO holds `≤ threshold`; the `≤ eff` filter drops the
    // (eff, threshold] tail that R_cheb truncation removes.)
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

    // The tiled pass does not emit row-major, so the dense builder's
    // "reverse a single slice" trick does not apply. Sort each (short) row
    // descending — exactly what reference ripser does to its neighbour lists.
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

    (row_ptr, col, val, r_cheb)
}
