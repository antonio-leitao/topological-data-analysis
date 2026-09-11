//! Canonical weighted edge lists from points, dense distances, or sparse COO.
//! Each undirected edge is stored once with `u > v`.

use rayon::prelude::*;

const MR: usize = 16;
const NR: usize = 16;
const MC: usize = 64;
const NC: usize = 256;

const NO_EDGE: f32 = f32::MAX;
const PARALLEL_POINT_MIN_PAIRS: usize = 1 << 18;
const STAGING_SAMPLES: usize = 8;

/// Weighted undirected edge with `u > v`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub(crate) struct Edge {
    pub(crate) u: u32,
    pub(crate) v: u32,
    pub(crate) distance: f32,
}

pub(crate) struct EdgeList {
    pub(crate) n: usize,
    pub(crate) edges: Vec<Edge>,
    pub(crate) center: u32,
    pub(crate) threshold: f32,
    /// Whether edges are sorted by `(distance, endpoints)`.
    pub(crate) sorted: bool,
}

impl EdgeList {
    pub(crate) fn from_points(
        data: &[f32],
        n: usize,
        d: usize,
        threshold: f32,
        parallel: bool,
    ) -> Self {
        debug_assert_eq!(data.len(), n * d);
        debug_assert!(n <= crate::MAX_POINTS);
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

        // Sampled eccentricities bound the final truncation radius from above.
        let staging = staging_cutoff(data, n, d, threshold);

        let pairs = n * (n - 1) / 2;
        let use_parallel = parallel && pairs >= PARALLEL_POINT_MIN_PAIRS;
        if use_parallel {
            points_single_pass_parallel(data, n, d, threshold, staging)
        } else {
            points_single_pass(data, n, d, threshold, staging)
        }
    }

    /// Read the strict lower triangle; values at least `f32::MAX` are absent.
    pub(crate) fn from_distance_matrix(mat: &[f32], n: usize, threshold: f32) -> Self {
        debug_assert_eq!(mat.len(), n * n);
        debug_assert!(n <= crate::MAX_POINTS);
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

        // Track component membership, row maxima, and finite neighbour counts.
        let mut max_dist = vec![f32::NEG_INFINITY; n];
        let mut finite_count = vec![0u32; n];
        let mut parent: Vec<u32> = (0..n as u32).collect();
        let mut comp_size = vec![1u32; n]; // meaningful at roots only
        for i in 1..n {
            let row_base = i * n;
            let mut row_max = max_dist[i];
            let mut row_finite = 0u32;
            for j in 0..i {
                let distance = mat[row_base + j];
                if distance < NO_EDGE {
                    if distance > row_max {
                        row_max = distance;
                    }
                    if distance > max_dist[j] {
                        max_dist[j] = distance;
                    }
                    row_finite += 1;
                    finite_count[j] += 1;
                    uf_union(&mut parent, &mut comp_size, i as u32, j as u32);
                }
            }
            max_dist[i] = row_max;
            finite_count[i] += row_finite;
        }

        let (center, r_cheb) =
            per_component_minimax(&mut parent, &comp_size, &finite_count, &max_dist);
        let effective = threshold.min(r_cheb);
        let mut edges = Vec::new();
        for i in 1..n {
            let row_base = i * n;
            for j in 0..i {
                let distance = mat[row_base + j];
                if distance <= effective && distance < NO_EDGE {
                    edges.push(Edge {
                        u: i as u32,
                        v: j as u32,
                        distance,
                    });
                }
            }
        }

        Self {
            n,
            edges,
            center: center as u32,
            threshold: effective,
            sorted: false,
        }
    }

    /// Build from an undirected COO graph. The minimum finite duplicate wins;
    /// diagonal, NaN, and values at least `f32::MAX` are ignored.
    ///
    /// Indices must be `< n` (validated by the caller; debug-asserted here).
    pub(crate) fn from_coo(
        n: usize,
        rows: &[u32],
        cols: &[u32],
        vals: &[f32],
        threshold: f32,
    ) -> Self {
        debug_assert_eq!(rows.len(), cols.len());
        debug_assert_eq!(rows.len(), vals.len());
        debug_assert!(n <= crate::MAX_POINTS);
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

        let mut edges: Vec<Edge> = Vec::with_capacity(vals.len());
        for ((&r, &c), &distance) in rows.iter().zip(cols).zip(vals) {
            debug_assert!((r as usize) < n && (c as usize) < n);
            if r == c || distance.partial_cmp(&NO_EDGE) != Some(std::cmp::Ordering::Less) {
                continue; // diagonal, absent (>= f32::MAX), or NaN
            }
            let (u, v) = if r > c { (r, c) } else { (c, r) };
            edges.push(Edge { u, v, distance });
        }
        edges.sort_unstable_by_key(|edge| (edge.u, edge.v, edge.distance.to_bits()));
        edges.dedup_by_key(|edge| (edge.u, edge.v)); // keeps the first = smallest

        let mut max_dist = vec![f32::NEG_INFINITY; n];
        let mut finite_count = vec![0u32; n];
        let mut parent: Vec<u32> = (0..n as u32).collect();
        let mut comp_size = vec![1u32; n];
        for edge in &edges {
            let (u, v) = (edge.u as usize, edge.v as usize);
            if edge.distance > max_dist[u] {
                max_dist[u] = edge.distance;
            }
            if edge.distance > max_dist[v] {
                max_dist[v] = edge.distance;
            }
            finite_count[u] += 1;
            finite_count[v] += 1;
            uf_union(&mut parent, &mut comp_size, edge.u, edge.v);
        }

        let (center, r_cheb) =
            per_component_minimax(&mut parent, &comp_size, &finite_count, &max_dist);
        let effective = threshold.min(r_cheb);
        edges.retain(|edge| edge.distance <= effective);

        Self {
            n,
            edges,
            center: center as u32,
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
    let n_groups = count.div_ceil(width);
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
fn tile_dist(tile: &[[f32; NR]; MR], m: usize, n: usize) -> f32 {
    tile[m][n].sqrt()
}

fn alloc_panels(d: usize) -> (Vec<f32>, Vec<f32>) {
    let a_buf_len = MC.div_ceil(MR) * MR * d;
    let b_buf_len = NC.div_ceil(NR) * NR * d;
    (vec![0.0f32; a_buf_len], vec![0.0f32; b_buf_len])
}

struct PointWorker {
    a_packed: Vec<f32>,
    b_packed: Vec<f32>,
    max_dist: Vec<f32>,
    edges: Vec<Edge>,
}

impl PointWorker {
    fn new(n: usize, d: usize) -> Self {
        let (a_packed, b_packed) = alloc_panels(d);
        Self {
            a_packed,
            b_packed,
            max_dist: vec![0.0f32; n],
            edges: Vec::new(),
        }
    }

    fn process_macro_tile(
        &mut self,
        data: &[f32],
        n: usize,
        d: usize,
        jc: usize,
        ic: usize,
        staging: f32,
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
                    let mut row_max = self.max_dist[i];
                    for nn in 0..nr_eff {
                        let j = j_min + nn;
                        if full || i > j {
                            let distance = tile_dist(&tile, m, nn);
                            if distance > row_max {
                                row_max = distance;
                            }
                            if distance > self.max_dist[j] {
                                self.max_dist[j] = distance;
                            }
                            if distance <= staging {
                                self.edges.push(Edge {
                                    u: i as u32,
                                    v: j as u32,
                                    distance,
                                });
                            }
                        }
                    }
                    self.max_dist[i] = row_max;
                }
                jr += NR;
            }
            ir += MR;
        }
    }

    fn merge(mut self, other: Self) -> Self {
        for (left, right) in self.max_dist.iter_mut().zip(other.max_dist) {
            if right > *left {
                *left = right;
            }
        }
        self.edges.extend(other.edges);
        self
    }
}

fn parallel_point_pass(data: &[f32], n: usize, d: usize, staging: f32) -> PointWorker {
    let mut macro_tiles = Vec::new();
    for jc in (0..n).step_by(NC) {
        for ic in (jc..n).step_by(MC) {
            macro_tiles.push((jc, ic));
        }
    }

    let workers: Vec<PointWorker> = macro_tiles
        .par_iter()
        .fold(
            || PointWorker::new(n, d),
            |mut worker, &(jc, ic)| {
                worker.process_macro_tile(data, n, d, jc, ic, staging);
                worker
            },
        )
        .collect();

    let total_edges: usize = workers.iter().map(|worker| worker.edges.len()).sum();
    let mut workers = workers.into_iter();
    let mut output = workers.next().unwrap_or_else(|| PointWorker::new(n, d));
    output.edges.reserve(total_edges - output.edges.len());
    for worker in workers {
        output = output.merge(worker);
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

/// Sampled eccentricity upper bound for the enclosing radius.
///
/// A dimension-scaled epsilon covers summation-order differences with the
/// tiled distance kernel.
fn staging_cutoff(data: &[f32], n: usize, d: usize, threshold: f32) -> f32 {
    let mut bound_sq = f32::INFINITY;
    let step = (n / STAGING_SAMPLES).max(1);
    let mut p = 0usize;
    while p < n && bound_sq > 0.0 {
        let row = &data[p * d..(p + 1) * d];
        let mut ecc_sq = 0.0f32;
        for q in 0..n {
            let other = &data[q * d..(q + 1) * d];
            let mut sq = 0.0f32;
            for k in 0..d {
                let diff = row[k] - other[k];
                sq += diff * diff;
            }
            if sq > ecc_sq {
                ecc_sq = sq;
            }
        }
        if ecc_sq < bound_sq {
            bound_sq = ecc_sq;
        }
        p += step;
    }
    let safety = 1.0 + (d as f32) * f32::EPSILON * 8.0;
    threshold.min(bound_sq.sqrt() * safety)
}

fn finalize_points(
    n: usize,
    mut edges: Vec<Edge>,
    center: usize,
    threshold: f32,
    staging: f32,
    r_cheb: f32,
) -> EdgeList {
    let effective = threshold.min(r_cheb);
    // Failure would mean the staging pass omitted required edges.
    assert!(
        effective <= staging,
        "staging cutoff {staging} fell below the certified radius {r_cheb}"
    );
    if effective < staging {
        edges.retain(|edge| edge.distance <= effective);
        edges.shrink_to_fit();
    }

    EdgeList {
        n,
        edges,
        center: center as u32,
        threshold: effective,
        sorted: false,
    }
}

#[inline]
fn reduce_minimax(max_dist: &[f32]) -> (usize, f32) {
    let mut center = 0usize;
    let mut radius = max_dist[0];
    for (i, &distance) in max_dist.iter().enumerate().skip(1) {
        if distance < radius {
            radius = distance;
            center = i;
        }
    }
    (center, radius)
}

#[inline]
pub(crate) fn uf_find(parent: &mut [u32], mut i: u32) -> u32 {
    while parent[i as usize] != i {
        let grand = parent[parent[i as usize] as usize];
        parent[i as usize] = grand;
        i = grand;
    }
    i
}

#[inline]
fn uf_union(parent: &mut [u32], comp_size: &mut [u32], a: u32, b: u32) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra == rb {
        return;
    }
    let (big, small) = if comp_size[ra as usize] >= comp_size[rb as usize] {
        (ra, rb)
    } else {
        (rb, ra)
    };
    parent[small as usize] = big;
    comp_size[big as usize] += comp_size[small as usize];
}

/// Component-wise cone certificate for disconnected distance data.
///
/// Each component uses an apex adjacent to every member. The global cutoff is
/// the largest component radius; a component without an apex contributes
/// infinity. Singletons contribute zero.
fn per_component_minimax(
    parent: &mut [u32],
    comp_size: &[u32],
    finite_count: &[u32],
    max_dist: &[f32],
) -> (usize, f32) {
    let n = max_dist.len();
    let mut r_enc = vec![f32::INFINITY; n]; // indexed by component root
    let mut apex = vec![0u32; n];
    for i in 0..n {
        let root = uf_find(parent, i as u32) as usize;
        if finite_count[i] == comp_size[root] - 1 {
            // Singletons retain the negative-infinity row sentinel.
            let radius = if comp_size[root] == 1 {
                0.0
            } else {
                max_dist[i]
            };
            if radius < r_enc[root] {
                r_enc[root] = radius;
                apex[root] = i as u32;
            }
        }
    }

    let mut r_cheb = f32::NEG_INFINITY;
    let mut center = 0usize;
    for v in 0..n {
        if parent[v] as usize != v {
            continue; // roots only
        }
        let radius = r_enc[v];
        let candidate = if radius.is_finite() {
            apex[v] as usize
        } else {
            v
        };
        if radius > r_cheb {
            r_cheb = radius;
            center = candidate;
        }
    }
    debug_assert!(
        r_cheb > f32::NEG_INFINITY,
        "n >= 2 guarantees at least one component root"
    );
    (center, r_cheb)
}

fn points_single_pass(data: &[f32], n: usize, d: usize, threshold: f32, staging: f32) -> EdgeList {
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
                        let distance = tile_dist(tile, m, nn);
                        if distance > row_max {
                            row_max = distance;
                        }
                        if distance > max_dist[j] {
                            max_dist[j] = distance;
                        }
                        if distance <= staging {
                            edges.push(Edge {
                                u: i as u32,
                                v: j as u32,
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
    finalize_points(n, edges, center, threshold, staging, r_cheb)
}

fn points_single_pass_parallel(
    data: &[f32],
    n: usize,
    d: usize,
    threshold: f32,
    staging: f32,
) -> EdgeList {
    let PointWorker {
        max_dist, edges, ..
    } = parallel_point_pass(data, n, d, staging);

    let (center, r_cheb) = reduce_minimax(&max_dist);
    finalize_points(n, edges, center, threshold, staging, r_cheb)
}

/// For query `q`,
/// `cols/dists[offsets[q]..offsets[q + 1]]` are the train points within
/// `radius` (strict), ascending train index.
pub(crate) struct CrossNeighbors {
    pub(crate) offsets: Vec<u32>,
    pub(crate) cols: Vec<u32>,
    pub(crate) dists: Vec<f32>,
}

/// Find cross pairs below `radius`, preserving query and train index order.
pub(crate) fn cross_neighbors(
    queries: &[f32],
    m: usize,
    train: &[f32],
    n: usize,
    d: usize,
    radius: f32,
    parallel: bool,
) -> CrossNeighbors {
    debug_assert_eq!(queries.len(), m * d);
    debug_assert_eq!(train.len(), n * d);

    let bands: Vec<(usize, usize)> = (0..m)
        .step_by(MC)
        .map(|start| (start, (m - start).min(MC)))
        .collect();

    let band_rows = |&(band_start, band_len): &(usize, usize)| -> Vec<Vec<(u32, f32)>> {
        let (mut a_packed, mut b_packed) = alloc_panels(d);
        let mut rows: Vec<Vec<(u32, f32)>> = vec![Vec::new(); band_len];
        pack_panel(queries, d, band_start, band_len, MR, &mut a_packed);

        let mut jc = 0;
        while jc < n {
            let nc = (n - jc).min(NC);
            pack_panel(train, d, jc, nc, NR, &mut b_packed);
            let mut ir = 0;
            while ir < band_len {
                let mr_eff = (band_len - ir).min(MR);
                let a_off = (ir / MR) * MR * d;
                let mut jr = 0;
                while jr < nc {
                    let nr_eff = (nc - jr).min(NR);
                    let b_off = (jr / NR) * NR * d;
                    let tile = micro_kernel(&a_packed[a_off..], &b_packed[b_off..], d);
                    for row in 0..mr_eff {
                        let out = &mut rows[ir + row];
                        for col in 0..nr_eff {
                            let distance = tile_dist(&tile, row, col);
                            if distance < radius {
                                out.push(((jc + jr + col) as u32, distance));
                            }
                        }
                    }
                    jr += NR;
                }
                ir += MR;
            }
            jc += NC;
        }
        rows
    };

    let all_rows: Vec<Vec<Vec<(u32, f32)>>> = if parallel && m * n >= PARALLEL_POINT_MIN_PAIRS {
        bands.par_iter().map(band_rows).collect()
    } else {
        bands.iter().map(band_rows).collect()
    };

    let total: usize = all_rows
        .iter()
        .flat_map(|band| band.iter().map(Vec::len))
        .sum();
    assert!(
        total <= u32::MAX as usize,
        "cross-neighbourhood exceeds u32 entries"
    );
    let mut offsets = Vec::with_capacity(m + 1);
    offsets.push(0u32);
    let mut cols = Vec::with_capacity(total);
    let mut dists = Vec::with_capacity(total);
    for band in all_rows {
        for row in band {
            for (col, dist) in row {
                cols.push(col);
                dists.push(dist);
            }
            offsets.push(cols.len() as u32);
        }
    }

    CrossNeighbors {
        offsets,
        cols,
        dists,
    }
}

/// Expand a lower-triangular test fixture to a symmetric matrix.
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
    fn edge_is_twelve_bytes() {
        assert_eq!(std::mem::size_of::<Edge>(), 12);
    }

    fn set_sym(d: &mut [f32], n: usize, i: usize, j: usize, v: f32) {
        d[i * n + j] = v;
        d[j * n + i] = v;
    }

    #[test]
    fn disconnected_truncation_uses_max_component_radius() {
        let n = 6;
        let mut d = vec![f32::INFINITY; n * n];
        for i in 0..n {
            d[i * n + i] = 0.0;
        }
        set_sym(&mut d, n, 0, 1, 1.0);
        set_sym(&mut d, n, 1, 2, 1.0);
        set_sym(&mut d, n, 2, 3, 1.0);
        set_sym(&mut d, n, 0, 3, 1.0);
        set_sym(&mut d, n, 0, 2, 2.0);
        set_sym(&mut d, n, 1, 3, 2.0);
        set_sym(&mut d, n, 4, 5, 1.0);

        let e = EdgeList::from_distance_matrix(&d, n, f32::INFINITY);
        assert_eq!(e.threshold, 2.0, "cutoff must be max over components");
        assert_eq!(e.edges.len(), 7, "4 sides + 2 diagonals + 1 pair");
        assert!(e.center < 4, "apex must lie in the winning component");
    }

    #[test]
    fn isolated_vertex_does_not_affect_truncation() {
        let n = 4;
        let mut d = vec![f32::INFINITY; n * n];
        for i in 0..n {
            d[i * n + i] = 0.0;
        }
        set_sym(&mut d, n, 0, 1, 1.0);
        set_sym(&mut d, n, 0, 2, 1.0);
        set_sym(&mut d, n, 1, 2, 1.0);

        let e = EdgeList::from_distance_matrix(&d, n, f32::INFINITY);
        assert_eq!(e.threshold, 1.0);
        assert_eq!(e.edges.len(), 3);
    }

    #[test]
    fn component_without_apex_disables_truncation() {
        let n = 4;
        let mut d = vec![f32::INFINITY; n * n];
        for i in 0..n {
            d[i * n + i] = 0.0;
        }
        set_sym(&mut d, n, 0, 1, 1.0);
        set_sym(&mut d, n, 1, 2, 1.0);
        set_sym(&mut d, n, 2, 3, 1.0);
        set_sym(&mut d, n, 0, 3, 1.0);

        let e = EdgeList::from_distance_matrix(&d, n, f32::INFINITY);
        assert_eq!(e.threshold, f32::INFINITY);
        assert_eq!(e.edges.len(), 4);
    }

    #[test]
    fn staging_cutoff_is_transparent() {
        for &(n, d) in &[(5, 2), (17, 3), (33, 5), (70, 2)] {
            let points: Vec<f32> = (0..n * d)
                .map(|i| {
                    let x = i as f32;
                    (x * 0.137).sin() + (x * 0.071).cos()
                })
                .collect();
            for threshold in [0.25, 0.9, f32::INFINITY] {
                let staged = EdgeList::from_points(&points, n, d, threshold, false);
                let reference = points_single_pass(&points, n, d, threshold, threshold);
                assert_eq!(staged.n, reference.n);
                assert_eq!(staged.center, reference.center);
                assert_eq!(staged.threshold.to_bits(), reference.threshold.to_bits());
                assert_eq!(sorted_edges(&staged), sorted_edges(&reference));
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
            let staging = staging_cutoff(&points, n, d, threshold);
            let single = points_single_pass(&points, n, d, threshold, staging);
            let single_parallel = points_single_pass_parallel(&points, n, d, threshold, staging);
            assert_eq!(single.center, single_parallel.center);
            assert_eq!(
                single.threshold.to_bits(),
                single_parallel.threshold.to_bits()
            );
            assert_eq!(sorted_edges(&single), sorted_edges(&single_parallel));
        }
    }

    #[test]
    fn cross_neighbors_matches_brute_force() {
        let (m, n, d) = (37usize, 71usize, 3usize);
        let gen = |count: usize, phase: f32| -> Vec<f32> {
            (0..count * d)
                .map(|i| {
                    let x = i as f32 + phase;
                    (x * 0.137).sin() + (x * 0.071).cos()
                })
                .collect()
        };
        let queries = gen(m, 0.31);
        let train = gen(n, 7.7);
        let radius = 0.8f32;

        for parallel in [false, true] {
            let cross = cross_neighbors(&queries, m, &train, n, d, radius, parallel);
            assert_eq!(cross.offsets.len(), m + 1);
            for q in 0..m {
                let range = cross.offsets[q] as usize..cross.offsets[q + 1] as usize;
                let got: Vec<(u32, f32)> = range.map(|e| (cross.cols[e], cross.dists[e])).collect();
                assert!(got.windows(2).all(|w| w[0].0 < w[1].0), "ascending indices");

                let mut want = Vec::new();
                for t in 0..n {
                    let mut sq = 0f32;
                    for k in 0..d {
                        let diff = queries[q * d + k] - train[t * d + k];
                        sq += diff * diff;
                    }
                    let dist = sq.sqrt();
                    if dist < radius {
                        want.push((t as u32, dist));
                    }
                }
                assert_eq!(
                    got.iter().map(|e| e.0).collect::<Vec<_>>(),
                    want.iter().map(|e| e.0).collect::<Vec<_>>(),
                    "membership mismatch at query {q}"
                );
                for (g, w) in got.iter().zip(&want) {
                    assert!((g.1 - w.1).abs() < 1e-5, "distance mismatch at query {q}");
                }
            }
        }
    }

    #[test]
    fn coo_dedups_to_min_and_canonicalizes() {
        let rows = [0u32, 1, 1, 2, 2];
        let cols = [1u32, 0, 2, 1, 2]; // (2, 2) is a diagonal entry: ignored
        let vals = [3.0f32, 1.0, 5.0, 5.0, 9.0];
        let e = EdgeList::from_coo(3, &rows, &cols, &vals, f32::INFINITY);

        assert_eq!(sorted_edges(&e).len(), 2);
        assert!(e.edges.iter().all(|edge| edge.u > edge.v));
        assert!(e
            .edges
            .iter()
            .any(|edge| (edge.u, edge.v, edge.distance) == (1, 0, 1.0)));
        assert!(e
            .edges
            .iter()
            .any(|edge| (edge.u, edge.v, edge.distance) == (2, 1, 5.0)));
        assert_eq!(e.threshold, 5.0);
        assert_eq!(e.center, 1);
    }

    #[test]
    fn coo_matches_dense_matrix_path() {
        let n = 6;
        let mut d = vec![f32::INFINITY; n * n];
        for i in 0..n {
            d[i * n + i] = 0.0;
        }
        set_sym(&mut d, n, 0, 1, 1.0);
        set_sym(&mut d, n, 1, 2, 1.0);
        set_sym(&mut d, n, 2, 3, 1.0);
        set_sym(&mut d, n, 0, 3, 1.0);
        set_sym(&mut d, n, 0, 2, 2.0);
        set_sym(&mut d, n, 1, 3, 2.0);
        set_sym(&mut d, n, 4, 5, 1.0);

        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            for j in 0..n {
                rows.push(i as u32);
                cols.push(j as u32);
                vals.push(d[i * n + j]);
            }
        }

        let dense = EdgeList::from_distance_matrix(&d, n, f32::INFINITY);
        let sparse = EdgeList::from_coo(n, &rows, &cols, &vals, f32::INFINITY);
        assert_eq!(sparse.n, dense.n);
        assert_eq!(sparse.center, dense.center);
        assert_eq!(sparse.threshold.to_bits(), dense.threshold.to_bits());
        assert_eq!(sorted_edges(&sparse), sorted_edges(&dense));
    }
}
