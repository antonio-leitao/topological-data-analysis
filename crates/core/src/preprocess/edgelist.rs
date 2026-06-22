//! Sparse weighted edge-list construction from point clouds and distance matrices.
//!
//! This is the common data boundary between distance preprocessing and the
//! optional filtration transforms. Each undirected edge is stored once with
//! `u > v`. Point-cloud construction preserves the tiled distance kernel and its
//! adaptive memory policy: one buffered pass for ordinary inputs, two passes when
//! buffering every candidate edge could exceed the staging budget.

use rayon::prelude::*;

const MR: usize = 16;
const NR: usize = 16;
const MC: usize = 64;
const NC: usize = 256;

const NO_EDGE: f32 = f32::MAX;
const SINGLE_PASS_BUDGET_BYTES: usize = 1 << 30;
const PARALLEL_POINT_MIN_PAIRS: usize = 1 << 18;

/// One weighted undirected edge, stored canonically with `u > v`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub(crate) struct Edge {
    pub(crate) u: u16,
    pub(crate) v: u16,
    pub(crate) distance: f32,
}

/// The truncated weighted 1-skeleton produced by distance preprocessing.
pub(crate) struct EdgeList {
    pub(crate) n: usize,
    pub(crate) edges: Vec<Edge>,
    pub(crate) center: u16,
    pub(crate) threshold: f32,
    /// True only when `edges` is in ascending `(distance, endpoints)` order.
    pub(crate) sorted: bool,
}

impl EdgeList {
    /// Build the truncated edge list of a row-major `(n, d)` point cloud.
    pub(crate) fn from_points(
        data: &[f32],
        n: usize,
        d: usize,
        threshold: f32,
        parallel: bool,
    ) -> Self {
        debug_assert_eq!(data.len(), n * d);
        debug_assert!(n <= u16::MAX as usize);
        debug_assert!(threshold >= 0.0);

        if n < 2 {
            return Self {
                n,
                edges: Vec::new(),
                center: 0,
                threshold: 0.0,
                sorted: false,
            };
        }

        let pairs = n * (n - 1) / 2;
        let use_parallel = parallel && pairs >= PARALLEL_POINT_MIN_PAIRS;
        if pairs.saturating_mul(std::mem::size_of::<Edge>()) <= SINGLE_PASS_BUDGET_BYTES {
            if use_parallel {
                points_single_pass_parallel(data, n, d, threshold)
            } else {
                points_single_pass(data, n, d, threshold)
            }
        } else if use_parallel {
            points_two_pass_parallel(data, n, d, threshold)
        } else {
            points_two_pass(data, n, d, threshold)
        }
    }

    /// Build the truncated edge list of a row-major `(n, n)` distance matrix.
    /// Only the strict lower triangle is read; entries `>= f32::MAX` are absent.
    pub(crate) fn from_distance_matrix(mat: &[f32], n: usize, threshold: f32) -> Self {
        debug_assert_eq!(mat.len(), n * n);
        debug_assert!(n <= u16::MAX as usize);
        debug_assert!(threshold >= 0.0);

        if n < 2 {
            return Self {
                n,
                edges: Vec::new(),
                center: 0,
                threshold,
                sorted: false,
            };
        }

        let mut max_dist = vec![f32::NEG_INFINITY; n];
        for i in 1..n {
            let row_base = i * n;
            let mut row_max = max_dist[i];
            for j in 0..i {
                let distance = mat[row_base + j];
                if distance < NO_EDGE {
                    if distance > row_max {
                        row_max = distance;
                    }
                    if distance > max_dist[j] {
                        max_dist[j] = distance;
                    }
                }
            }
            max_dist[i] = row_max;
        }

        let (center, r_cheb) = reduce_minimax_disconnected(&max_dist);
        let effective = threshold.min(r_cheb);
        let mut edges = Vec::new();
        for i in 1..n {
            let row_base = i * n;
            for j in 0..i {
                let distance = mat[row_base + j];
                if distance <= effective && distance < NO_EDGE {
                    edges.push(Edge {
                        u: i as u16,
                        v: j as u16,
                        distance,
                    });
                }
            }
        }

        Self {
            n,
            edges,
            center: center as u16,
            threshold: effective,
            sorted: false,
        }
    }
}

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

struct PointWorker {
    a_packed: Vec<f32>,
    b_packed: Vec<f32>,
    max_dist: Vec<f32>,
    edges: Vec<Edge>,
}

impl PointWorker {
    fn new<const TRACK_MAX: bool>(n: usize, d: usize) -> Self {
        let (a_packed, b_packed) = alloc_panels(d);
        Self {
            a_packed,
            b_packed,
            max_dist: if TRACK_MAX {
                vec![0.0f32; n]
            } else {
                Vec::new()
            },
            edges: Vec::new(),
        }
    }

    fn process_macro_tile<const TRACK_MAX: bool, const EMIT_EDGES: bool>(
        &mut self,
        data: &[f32],
        n: usize,
        d: usize,
        jc: usize,
        ic: usize,
        threshold: f32,
    ) {
        let nc = (n - jc).min(NC);
        let mc = (n - ic).min(MC);
        pack_panel(data, d, jc, nc, NR, &mut self.b_packed);
        pack_panel(data, d, ic, mc, MR, &mut self.a_packed);

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

                let tile = micro_kernel(&self.a_packed[a_off..], &self.b_packed[b_off..], d);
                let full = i_min > j_max;
                for m in 0..mr_eff {
                    let i = i_min + m;
                    let mut row_max = if TRACK_MAX { self.max_dist[i] } else { 0.0 };
                    for nn in 0..nr_eff {
                        let j = j_min + nn;
                        if full || i > j {
                            let distance = tile_dist(&tile, m, nn, full);
                            if TRACK_MAX {
                                if distance > row_max {
                                    row_max = distance;
                                }
                                if distance > self.max_dist[j] {
                                    self.max_dist[j] = distance;
                                }
                            }
                            if EMIT_EDGES && distance <= threshold {
                                self.edges.push(Edge {
                                    u: i as u16,
                                    v: j as u16,
                                    distance,
                                });
                            }
                        }
                    }
                    if TRACK_MAX {
                        self.max_dist[i] = row_max;
                    }
                }
                jr += NR;
            }
            ir += MR;
        }
    }

    fn merge<const TRACK_MAX: bool>(mut self, other: Self) -> Self {
        if TRACK_MAX {
            for (left, right) in self.max_dist.iter_mut().zip(other.max_dist) {
                if right > *left {
                    *left = right;
                }
            }
        }
        self.edges.extend(other.edges);
        self
    }
}

fn parallel_point_pass<const TRACK_MAX: bool, const EMIT_EDGES: bool>(
    data: &[f32],
    n: usize,
    d: usize,
    threshold: f32,
) -> PointWorker {
    let mut macro_tiles = Vec::new();
    for jc in (0..n).step_by(NC) {
        for ic in (jc..n).step_by(MC) {
            macro_tiles.push((jc, ic));
        }
    }

    let workers: Vec<PointWorker> = macro_tiles
        .par_iter()
        .fold(
            || PointWorker::new::<TRACK_MAX>(n, d),
            |mut worker, &(jc, ic)| {
                worker.process_macro_tile::<TRACK_MAX, EMIT_EDGES>(data, n, d, jc, ic, threshold);
                worker
            },
        )
        .collect();

    let total_edges: usize = workers.iter().map(|worker| worker.edges.len()).sum();
    let mut workers = workers.into_iter();
    let mut output = workers
        .next()
        .unwrap_or_else(|| PointWorker::new::<TRACK_MAX>(n, d));
    output.edges.reserve(total_edges - output.edges.len());
    for worker in workers {
        output = output.merge::<TRACK_MAX>(worker);
    }
    output
}

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

#[inline]
fn reduce_minimax(max_dist: &[f32]) -> (usize, f32) {
    let mut center = 0usize;
    let mut radius = max_dist[0];
    for i in 1..max_dist.len() {
        if max_dist[i] < radius {
            radius = max_dist[i];
            center = i;
        }
    }
    (center, radius)
}

#[inline]
fn reduce_minimax_disconnected(max_dist: &[f32]) -> (usize, f32) {
    let mut center = 0usize;
    let mut radius = f32::INFINITY;
    for (i, &distance) in max_dist.iter().enumerate() {
        if distance > f32::NEG_INFINITY && distance < radius {
            radius = distance;
            center = i;
        }
    }
    (center, radius)
}

fn points_single_pass(data: &[f32], n: usize, d: usize, threshold: f32) -> EdgeList {
    let (mut a_packed, mut b_packed) = alloc_panels(d);
    let mut max_dist = vec![0.0f32; n];
    let mut edges = Vec::new();

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
                        let distance = tile_dist(tile, m, nn, full);
                        if distance > row_max {
                            row_max = distance;
                        }
                        if distance > max_dist[j] {
                            max_dist[j] = distance;
                        }
                        if distance <= threshold {
                            edges.push(Edge {
                                u: i as u16,
                                v: j as u16,
                                distance,
                            });
                        }
                    }
                }
                max_dist[i] = row_max;
            }
        },
    );

    let (center, r_cheb) = reduce_minimax(&max_dist);
    let effective = threshold.min(r_cheb);
    if effective < threshold {
        edges.retain(|edge| edge.distance <= effective);
    }

    EdgeList {
        n,
        edges,
        center: center as u16,
        threshold: effective,
        sorted: false,
    }
}

fn points_single_pass_parallel(data: &[f32], n: usize, d: usize, threshold: f32) -> EdgeList {
    let PointWorker {
        max_dist,
        mut edges,
        ..
    } = parallel_point_pass::<true, true>(data, n, d, threshold);

    let (center, r_cheb) = reduce_minimax(&max_dist);
    let effective = threshold.min(r_cheb);
    if effective < threshold {
        edges.retain(|edge| edge.distance <= effective);
    }

    EdgeList {
        n,
        edges,
        center: center as u16,
        threshold: effective,
        sorted: false,
    }
}

fn points_two_pass(data: &[f32], n: usize, d: usize, threshold: f32) -> EdgeList {
    let (mut a_packed, mut b_packed) = alloc_panels(d);
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
                        let distance = tile_dist(tile, m, nn, full);
                        if distance > row_max {
                            row_max = distance;
                        }
                        if distance > max_dist[j] {
                            max_dist[j] = distance;
                        }
                    }
                }
                max_dist[i] = row_max;
            }
        },
    );

    let (center, r_cheb) = reduce_minimax(&max_dist);
    let effective = threshold.min(r_cheb);
    let mut edges = Vec::new();

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
                        let distance = tile_dist(tile, m, nn, full);
                        if distance <= effective {
                            edges.push(Edge {
                                u: i as u16,
                                v: j as u16,
                                distance,
                            });
                        }
                    }
                }
            }
        },
    );

    EdgeList {
        n,
        edges,
        center: center as u16,
        threshold: effective,
        sorted: false,
    }
}

fn points_two_pass_parallel(data: &[f32], n: usize, d: usize, threshold: f32) -> EdgeList {
    let first = parallel_point_pass::<true, false>(data, n, d, threshold);
    let (center, r_cheb) = reduce_minimax(&first.max_dist);
    let effective = threshold.min(r_cheb);
    let second = parallel_point_pass::<false, true>(data, n, d, effective);

    EdgeList {
        n,
        edges: second.edges,
        center: center as u16,
        threshold: effective,
        sorted: false,
    }
}

/// Lower-triangular fixture → symmetric square matrix. Test-only helper shared
/// with the engine fixtures (formerly `_pdist::square_from_lower_tri`).
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

    fn sorted_edges(edge_list: &EdgeList) -> Vec<Edge> {
        let mut edges = edge_list.edges.clone();
        edges.sort_unstable_by_key(|edge| (edge.u, edge.v));
        edges
    }

    #[test]
    fn edge_is_eight_bytes() {
        assert_eq!(std::mem::size_of::<Edge>(), 8);
    }

    #[test]
    fn adaptive_point_paths_are_identical() {
        for &(n, d) in &[(5, 2), (17, 3), (33, 5), (70, 2)] {
            let points: Vec<f32> = (0..n * d)
                .map(|i| {
                    let x = i as f32;
                    (x * 0.137).sin() + (x * 0.071).cos()
                })
                .collect();
            for threshold in [0.25, 0.9, f32::INFINITY] {
                let single = points_single_pass(&points, n, d, threshold);
                let double = points_two_pass(&points, n, d, threshold);
                assert_eq!(single.n, double.n);
                assert_eq!(single.center, double.center);
                assert_eq!(single.threshold.to_bits(), double.threshold.to_bits());
                assert_eq!(sorted_edges(&single), sorted_edges(&double));
            }
        }
    }

    #[test]
    fn parallel_point_passes_match_sequential() {
        let n = 70;
        let d = 3;
        let points: Vec<f32> = (0..n * d)
            .map(|i| {
                let x = i as f32;
                (x * 0.137).sin() + (x * 0.071).cos()
            })
            .collect();

        for threshold in [0.25, 0.9, f32::INFINITY] {
            let single = points_single_pass(&points, n, d, threshold);
            let single_parallel = points_single_pass_parallel(&points, n, d, threshold);
            assert_eq!(single.center, single_parallel.center);
            assert_eq!(
                single.threshold.to_bits(),
                single_parallel.threshold.to_bits()
            );
            assert_eq!(sorted_edges(&single), sorted_edges(&single_parallel));

            let double = points_two_pass(&points, n, d, threshold);
            let double_parallel = points_two_pass_parallel(&points, n, d, threshold);
            assert_eq!(double.center, double_parallel.center);
            assert_eq!(
                double.threshold.to_bits(),
                double_parallel.threshold.to_bits()
            );
            assert_eq!(sorted_edges(&double), sorted_edges(&double_parallel));
        }
    }
}
