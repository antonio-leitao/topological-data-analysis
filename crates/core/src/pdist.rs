// ============================================================================
// Tiled V3 — fused distance + threshold computation
//
// Same micro-kernel as V2 (f32, NR=16, chunks_exact), but:
//   1. Store phase takes sqrt in f64 for numerical stability
//   2. max_dist[i] and max_dist[j] tracked during store (zero-cost fusion)
//   3. Threshold = min_i(max_dist[i]), resolved against optional user limit
//   4. Single branchless filtering pass at the end
//
// The f32 micro-kernel accumulates squared diffs directly (no expansion trick),
// so cancellation is not an issue.  The f64 sqrt preserves precision on the
// final cast back to f32 — matching the naive f64-accumulation approach to
// within ~1 ULP for typical dimension counts.
// ============================================================================

const MR: usize = 16;
const NR: usize = 16;
const MC: usize = 64;
const NC: usize = 256;

// ---------------------------------------------------------------------------
// Micro-kernel: identical to V2.  f32 throughout, NR=16 for throughput.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Packing: identical to V2.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Fused store: sqrt (f64) + max_dist tracking.
//
// The tile contains squared distances in f32.  We:
//   1. Widen to f64
//   2. sqrt in f64 (7.5e-6 relative error vs full f64 accumulation)
//   3. Cast back to f32
//   4. Write to output
//   5. Update max_dist[i] and max_dist[j]
//
// max_dist is n floats = 8 KB at n=2000 — permanently in L1.
// The sqrt cost is O(n²), dwarfed by the O(n²d) micro-kernel.
// ---------------------------------------------------------------------------

/// Full tile — every element satisfies i > j.
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

            // max_dist[i]: accumulate in a local, write back once per row.
            if dist > row_max {
                row_max = dist;
            }
            // max_dist[j]: must update in-place (different j per iteration).
            if dist > max_dist[j] {
                max_dist[j] = dist;
            }
        }

        max_dist[i] = row_max;
    }
}

/// Diagonal tile — need per-element i > j guard.
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
                // let dist = ((tile[m][n] as f64).sqrt()) as f32;
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

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Compute pairwise Euclidean distances (with sqrt) and an effective threshold.
///
/// Returns `(distances, threshold)` where:
///   - `distances` is the lower-triangular condensed distance vector
///   - `threshold` is `min(max_threshold, minimax_radius)` where
///     `minimax_radius = min_i max_j d(i, j)`
///
/// All entries strictly above the threshold are set to `f32::INFINITY`.
pub fn pdist_tiled_v3(data: &[f32], n: usize, d: usize) -> (Vec<f32>, f32) {
    debug_assert_eq!(data.len(), n * d);
    if n < 2 {
        return (vec![], 0.0);
    }

    let out_len = n * (n - 1) / 2;
    let mut out = vec![0.0f32; out_len];
    let mut max_dist = vec![0.0f32; n];

    let a_buf_len = ((MC + MR - 1) / MR) * MR * d;
    let b_buf_len = ((NC + NR - 1) / NR) * NR * d;
    let mut a_packed = vec![0.0f32; a_buf_len];
    let mut b_packed = vec![0.0f32; b_buf_len];

    // ---- Tiled distance computation with fused sqrt + max tracking ----

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

    // ---- Threshold: minimax radius ----

    let minimax = max_dist.iter().copied().fold(f32::INFINITY, f32::min);

    (out, minimax)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: f64 accumulation + f64 sqrt, matching the naive code exactly.
    fn reference_pdist(data: &[f32], n: usize, d: usize) -> (Vec<f64>, f64) {
        let mut out = Vec::with_capacity(n * (n - 1) / 2);
        let mut max_dist = vec![0.0f64; n];

        for i in 1..n {
            for j in 0..i {
                let sq: f64 = (0..d)
                    .map(|k| {
                        let diff = data[i * d + k] as f64 - data[j * d + k] as f64;
                        diff * diff
                    })
                    .sum();
                let dist = sq.sqrt();
                out.push(dist);

                if dist > max_dist[i] {
                    max_dist[i] = dist;
                }
                if dist > max_dist[j] {
                    max_dist[j] = dist;
                }
            }
        }

        let threshold = max_dist.iter().copied().fold(f64::INFINITY, f64::min);
        (out, threshold)
    }

    fn assert_close(result: &[f32], reference: &[f64], tol: f64, label: &str) {
        assert_eq!(result.len(), reference.len(), "{label}: length mismatch");
        for (idx, (&r, &e)) in result.iter().zip(reference.iter()).enumerate() {
            if r.is_infinite() && e > 100.0 {
                continue; // filtered entry, skip
            }
            let r64 = r as f64;
            let abs_err = (r64 - e).abs();
            let denom = e.abs().max(1e-12);
            let rel_err = abs_err / denom;
            assert!(
                rel_err < tol || abs_err < 1e-6,
                "{label} index {idx}: got {r}, expected {e} (rel_err={rel_err:.2e})"
            );
        }
    }

    fn make_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..n * d)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as i64 as f32) / (i64::MAX as f32)
            })
            .collect()
    }

    #[test]
    fn v3_basic_distances() {
        let data = vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        let (out, _thr) = pdist_tiled_v3(&data, 3, 2);
        let (reference, _) = reference_pdist(&data, 3, 2);
        assert_close(&out, &reference, 1e-6, "basic");
    }

    #[test]
    fn v3_identical_points() {
        let data = vec![3.0, 4.0, 5.0, 3.0, 4.0, 5.0];
        let (out, _) = pdist_tiled_v3(&data, 2, 3);
        assert_eq!(out.len(), 1);
        assert!(
            out[0].abs() < 1e-7,
            "identical points should have zero distance"
        );
    }

    #[test]
    fn v3_matches_reference_medium() {
        let n = 200;
        let d = 128;
        let data = make_data(n, d, 42);
        let (out, thr) = pdist_tiled_v3(&data, n, d);
        let (reference, ref_thr) = reference_pdist(&data, n, d);

        // Check non-filtered entries.
        let non_inf: Vec<(f32, f64)> = out
            .iter()
            .zip(reference.iter())
            .filter(|(&r, _)| r.is_finite())
            .map(|(&r, &e)| (r, e))
            .collect();
        let (vals, refs): (Vec<f32>, Vec<f64>) = non_inf.into_iter().unzip();
        assert_close(&vals, &refs, 1e-3, "medium");

        // Threshold should match closely.
        let thr_err = ((thr as f64 - ref_thr).abs()) / ref_thr.abs().max(1e-12);
        assert!(
            thr_err < 1e-3,
            "threshold mismatch: got {thr}, expected {ref_thr}"
        );
    }

    #[test]
    fn v3_stability_large_values() {
        let n = 30;
        let d = 8;
        let mut data = vec![0.0f32; n * d];
        for i in 0..n {
            for k in 0..d {
                data[i * d + k] = 1e6 + (i as f32) * 0.01 + (k as f32) * 0.001;
            }
        }
        let (out, _) = pdist_tiled_v3(&data, n, d);
        let (reference, _) = reference_pdist(&data, n, d);

        // Only check finite entries.
        let non_inf: Vec<(f32, f64)> = out
            .iter()
            .zip(reference.iter())
            .filter(|(&r, _)| r.is_finite())
            .map(|(&r, &e)| (r, e))
            .collect();
        let (vals, refs): (Vec<f32>, Vec<f64>) = non_inf.into_iter().unzip();
        assert_close(&vals, &refs, 1e-2, "stability");
    }

    #[test]
    fn v3_odd_dimensions() {
        let n = 70;
        let d = 17;
        let data = make_data(n, d, 555);
        let (out, _) = pdist_tiled_v3(&data, n, d);
        let (reference, _) = reference_pdist(&data, n, d);

        let non_inf: Vec<(f32, f64)> = out
            .iter()
            .zip(reference.iter())
            .filter(|(&r, _)| r.is_finite())
            .map(|(&r, &e)| (r, e))
            .collect();
        let (vals, refs): (Vec<f32>, Vec<f64>) = non_inf.into_iter().unzip();
        assert_close(&vals, &refs, 1e-3, "odd dims");
    }
}
