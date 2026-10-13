//! Canonical weighted edge lists from points, dense distances, or sparse COO.
//! Each undirected edge is stored once with `u > v`.

use rayon::prelude::*;

use crate::input::{Real, SparseIndex};
use crate::{Error, Result};

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

impl Edge {
    /// Orient an unordered pair and fold `-0.0` into `+0.0`.
    #[inline]
    pub(crate) fn canonical(a: u32, b: u32, distance: f32) -> Self {
        let (u, v) = if a > b { (a, b) } else { (b, a) };
        let distance = if distance == 0.0 { 0.0 } else { distance };
        Edge { u, v, distance }
    }

    /// `(distance, u, v)`; bit order is numeric order for non-negative distances.
    #[inline]
    pub(crate) fn sort_key(&self) -> (u32, u32, u32) {
        (self.distance.to_bits(), self.u, self.v)
    }
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

    /// Read the strict lower triangle; values at least `f32::MAX` (or NaN) are
    /// absent, and a negative value is an error.
    pub(crate) fn from_distance_matrix<T: Real>(
        mat: &[T],
        n: usize,
        threshold: f32,
    ) -> Result<Self> {
        debug_assert_eq!(mat.len(), n * n);
        debug_assert!(n <= crate::MAX_POINTS);
        debug_assert!(threshold >= 0.0);

        if n < 2 {
            return Ok(Self {
                n,
                edges: Vec::new(),
                center: 0,
                threshold,
                sorted: false,
            });
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
                let distance = mat[row_base + j].to_f32();
                if distance < NO_EDGE {
                    // Negative bit patterns sort above every distance.
                    if distance < 0.0 {
                        return Err(Error::NegativeDistance {
                            u: i,
                            v: j,
                            value: distance,
                        });
                    }
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
                let distance = mat[row_base + j].to_f32();
                if distance <= effective && distance < NO_EDGE {
                    edges.push(Edge::canonical(i as u32, j as u32, distance));
                }
            }
        }

        Ok(Self {
            n,
            edges,
            center: center as u32,
            threshold: effective,
            sorted: false,
        })
    }

    /// Build from an undirected COO graph. The minimum finite duplicate wins;
    /// diagonal, NaN, and values at least `f32::MAX` are ignored. Indices at
    /// or above `n` and negative values are errors.
    ///
    /// Row and column lengths must match (validated by the caller).
    pub(crate) fn from_coo<I: SparseIndex, V: Real>(
        n: usize,
        rows: &[I],
        cols: &[I],
        vals: &[V],
        threshold: f32,
    ) -> Result<Self> {
        debug_assert_eq!(rows.len(), cols.len());
        debug_assert_eq!(rows.len(), vals.len());

        let mut edges: Vec<Edge> = Vec::with_capacity(vals.len());
        for ((&r, &c), &distance) in rows.iter().zip(cols).zip(vals) {
            push_entry(&mut edges, vertex(r, n)?, vertex(c, n)?, distance.to_f32())?;
        }
        Ok(Self::from_entries(n, edges, threshold))
    }

    /// Build from a compressed sparse graph: CSR rows or CSC columns, which
    /// read the same because the graph is undirected. Entries follow the
    /// [`from_coo`](Self::from_coo) rules, and `indptr` must hold `n + 1`
    /// non-decreasing offsets into `indices`.
    pub(crate) fn from_compressed<I: SparseIndex, V: Real>(
        n: usize,
        indptr: &[I],
        indices: &[I],
        vals: &[V],
        threshold: f32,
    ) -> Result<Self> {
        if indptr.len() != n + 1 {
            return Err(Error::InvalidCompressed("indptr must have n + 1 entries"));
        }
        if indices.len() != vals.len() {
            return Err(Error::InvalidCompressed(
                "indices and data have different lengths",
            ));
        }

        let offset = |p: I| {
            p.to_usize()
                .filter(|&p| p <= indices.len())
                .ok_or(Error::InvalidCompressed(
                    "indptr offset is negative or beyond indices",
                ))
        };
        let mut edges: Vec<Edge> = Vec::with_capacity(vals.len());
        let mut start = offset(indptr[0])?;
        for (r, &next) in indptr[1..].iter().enumerate() {
            let end = offset(next)?;
            if end < start {
                return Err(Error::InvalidCompressed("indptr is not non-decreasing"));
            }
            for (&c, &distance) in indices[start..end].iter().zip(&vals[start..end]) {
                push_entry(&mut edges, r as u32, vertex(c, n)?, distance.to_f32())?;
            }
            start = end;
        }
        Ok(Self::from_entries(n, edges, threshold))
    }

    /// Deduplicate raw sparse entries and apply the per-component cutoff.
    fn from_entries(n: usize, mut edges: Vec<Edge>, threshold: f32) -> Self {
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
        // Symmetric input names every pair twice. Dedup halves the length, and
        // the spare half would otherwise ride along to the engine.
        if edges.len() < edges.capacity() / 4 * 3 {
            edges.shrink_to_fit();
        }

        Self {
            n,
            edges,
            center: center as u32,
            threshold: effective,
            sorted: false,
        }
    }

    /// Sort into `Edge::sort_key` order unless already sorted.
    pub(crate) fn sort(&mut self, parallel: bool) {
        if self.sorted {
            return;
        }
        if parallel {
            self.edges.par_sort_unstable_by_key(Edge::sort_key);
        } else {
            self.edges.sort_unstable_by_key(Edge::sort_key);
        }
        self.sorted = true;
    }

    /// Words a bitset row per vertex needs when each row runs from id 0 to its
    /// largest neighbour, the layout collapse and the engine fall back to.
    pub(crate) fn tail_words(&self) -> usize {
        let mut reach = vec![0u32; self.n];
        for e in &self.edges {
            reach[e.v as usize] = reach[e.v as usize].max((e.u >> 6) + 1);
            reach[e.u as usize] = reach[e.u as usize].max((e.u >> 6) + 1);
        }
        reach.iter().map(|&r| r as usize).sum()
    }

    /// Renumber vertex `order[i]` as `i`. `order` is a permutation of `0..n`.
    pub(crate) fn relabel(&mut self, order: &[u32]) {
        debug_assert_eq!(order.len(), self.n);
        let mut position = vec![0u32; self.n];
        for (i, &v) in order.iter().enumerate() {
            position[v as usize] = i as u32;
        }
        for e in &mut self.edges {
            *e = Edge::canonical(position[e.u as usize], position[e.v as usize], e.distance);
        }
        if self.n > 0 {
            self.center = position[self.center as usize];
        }
        self.sorted = false;
    }

    /// Check the invariants the optimisation passes assume.
    pub(crate) fn assert_preconditions(&self) {
        for e in &self.edges {
            assert!(e.u > e.v && (e.u as usize) < self.n, "bad endpoints: {e:?}");
            assert!(
                e.distance.is_finite() && e.distance.is_sign_positive(),
                "bad distance: {e:?}"
            );
        }
        if self.sorted {
            for w in self.edges.windows(2) {
                assert!(
                    w[0].sort_key() < w[1].sort_key(),
                    "unsorted or duplicate: {:?}",
                    w[0]
                );
            }
        }
    }
}

/// A sparse index as a vertex id below `n`.
#[inline]
fn vertex<I: SparseIndex>(index: I, n: usize) -> Result<u32> {
    match index.to_u32() {
        Some(v) if (v as usize) < n => Ok(v),
        _ => Err(Error::SparseIndexOutOfRange {
            index: index.to_i128(),
            n,
        }),
    }
}

/// Record one sparse entry, skipping the diagonal, absent (`>= f32::MAX`) and
/// NaN values.
#[inline]
fn push_entry(edges: &mut Vec<Edge>, r: u32, c: u32, distance: f32) -> Result<()> {
    if r == c || distance.partial_cmp(&NO_EDGE) != Some(std::cmp::Ordering::Less) {
        return Ok(());
    }
    if distance < 0.0 {
        return Err(Error::NegativeDistance {
            u: r as usize,
            v: c as usize,
            value: distance,
        });
    }
    edges.push(Edge::canonical(r, c, distance));
    Ok(())
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

// ---------------------------------------------------------------------------
// Streamed point passes
// ---------------------------------------------------------------------------
//
// A point cloud's edge list under the enclosing radius is most of its pairs.
// A consumer that reads edges once in `Edge::sort_key` order, such as the
// quotient, can take them in distance bands instead: one full pass learns the
// cutoff and a histogram of distances, then each band is generated, sorted and
// handed over on its own. Distances come from the same kernel as `from_points`,
// so every edge is bit-identical to the one it would have stored.

/// Distance bins of a [`PointScan`].
pub(crate) const STREAM_BINS: usize = 1 << 16;

/// What one full pass over the pairs tells a streamed build: the cutoff and
/// centre `from_points` would choose, and how many pairs fall in each bin.
pub(crate) struct PointScan {
    pub(crate) center: u32,
    /// The truncation radius: `min(threshold, certified radius)`.
    pub(crate) threshold: f32,
    scale: f32,
    /// Pairs per bin, counting every pair up to the staging cutoff.
    pub(crate) bins: Vec<u64>,
}

impl PointScan {
    /// Bin of a distance. Monotone, so bins split pairs in distance order and
    /// never separate two equal distances.
    #[inline(always)]
    pub(crate) fn bin(&self, distance: f32) -> usize {
        ((distance * self.scale) as usize).min(STREAM_BINS - 1)
    }
}

/// One full pass: the cutoff and centre [`EdgeList::from_points`] would pick,
/// and a histogram of the distances up to its staging cutoff.
pub(crate) fn scan_points(
    data: &[f32],
    n: usize,
    d: usize,
    threshold: f32,
    parallel: bool,
) -> PointScan {
    if n < 2 {
        return PointScan {
            center: 0,
            threshold: 0.0,
            scale: 0.0,
            bins: vec![0; STREAM_BINS],
        };
    }
    let staging = staging_cutoff(data, n, d, threshold);
    let scale = if staging > 0.0 {
        STREAM_BINS as f32 / staging
    } else {
        0.0
    };
    let mut scan = PointScan {
        center: 0,
        threshold,
        scale,
        bins: vec![0; STREAM_BINS],
    };

    let tile_scan = |tile: &[[f32; NR]; MR],
                     i_min: usize,
                     j_min: usize,
                     mr: usize,
                     nr: usize,
                     full: bool,
                     max_dist: &mut [f32],
                     bins: &mut [u64]| {
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
                        bins[((distance * scale) as usize).min(STREAM_BINS - 1)] += 1;
                    }
                }
            }
            max_dist[i] = row_max;
        }
    };

    let max_dist = if parallel && n * (n - 1) / 2 >= PARALLEL_POINT_MIN_PAIRS {
        let (max_dist, bins) = macro_tiles(n)
            .par_iter()
            .fold(
                || (alloc_panels(d), vec![0.0f32; n], vec![0u64; STREAM_BINS]),
                |(mut panels, mut max_dist, mut bins), &(jc, ic)| {
                    for_each_micro_tile(data, n, d, jc, ic, &mut panels, |t, i, j, mr, nr, full| {
                        tile_scan(t, i, j, mr, nr, full, &mut max_dist, &mut bins)
                    });
                    (panels, max_dist, bins)
                },
            )
            .map(|(_, max_dist, bins)| (max_dist, bins))
            .reduce(
                || (vec![0.0f32; n], vec![0u64; STREAM_BINS]),
                |(mut a, mut ab), (b, bb)| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x = x.max(y);
                    }
                    for (x, y) in ab.iter_mut().zip(bb) {
                        *x += y;
                    }
                    (a, ab)
                },
            );
        scan.bins = bins;
        max_dist
    } else {
        let (mut a_packed, mut b_packed) = alloc_panels(d);
        let mut max_dist = vec![0.0f32; n];
        let bins = &mut scan.bins;
        tiled_pass(data, n, d, &mut a_packed, &mut b_packed, |t, i, j, mr, nr, full| {
            tile_scan(t, i, j, mr, nr, full, &mut max_dist, bins)
        });
        max_dist
    };

    let (center, r_cheb) = reduce_minimax(&max_dist);
    let effective = threshold.min(r_cheb);
    assert!(
        effective <= staging,
        "staging cutoff {staging} fell below the certified radius {r_cheb}"
    );
    scan.center = center as u32;
    scan.threshold = effective;
    scan
}

/// The points reordered so that every run of 16 is spatially compact, with
/// a bounding box per run of 16, 64 and 256: the row and column groups of a
/// micro tile and of a macro tile. A band pass skips every tile whose boxes
/// put all of its distances outside the band.
pub(crate) struct SpatialPoints {
    n: usize,
    d: usize,
    data: Vec<f32>,
    /// Original id of each reordered point.
    ids: Vec<u32>,
    /// `[min_0.., max_0..]` per group, for groups of `MR`, `MC` and `NC`.
    boxes: [Vec<f32>; 3],
}

impl SpatialPoints {
    /// k-d order: split each range at a multiple of 16 points along its widest
    /// axis, so every aligned run of 16, 64 or 256 sits in one cell.
    pub(crate) fn new(data: &[f32], n: usize, d: usize) -> Self {
        const { assert!(MC.is_multiple_of(MR) && NC.is_multiple_of(NR) && NC.is_multiple_of(MC)) };
        let mut ids: Vec<u32> = (0..n as u32).collect();
        let mut stack = vec![(0usize, n)];
        while let Some((lo, hi)) = stack.pop() {
            if hi - lo <= MR {
                continue;
            }
            let (min, max) = bounds(data, d, &ids[lo..hi]);
            let axis = (0..d)
                .max_by(|&a, &b| (max[a] - min[a]).total_cmp(&(max[b] - min[b])))
                .unwrap_or(0);
            let half = ((hi - lo) / 2).next_multiple_of(MR);
            ids[lo..hi].select_nth_unstable_by(half, |&a, &b| {
                data[a as usize * d + axis].total_cmp(&data[b as usize * d + axis])
            });
            stack.push((lo, lo + half));
            stack.push((lo + half, hi));
        }

        let mut ordered = Vec::with_capacity(n * d);
        for &id in &ids {
            ordered.extend_from_slice(&data[id as usize * d..(id as usize + 1) * d]);
        }
        let boxes = [MR, MC, NC].map(|size| {
            let mut out = Vec::with_capacity(n.div_ceil(size) * 2 * d);
            for start in (0..n).step_by(size) {
                let end = (start + size).min(n);
                let run: Vec<u32> = (start as u32..end as u32).collect();
                let (min, max) = bounds(&ordered, d, &run);
                out.extend(min);
                out.extend(max);
            }
            out
        });
        Self {
            n,
            d,
            data: ordered,
            ids,
            boxes,
        }
    }

    /// Bounds on the distances between the point groups `a` and `b` of size
    /// `sizes.0` and `sizes.1` (indices into the box tables), as squared
    /// distances in `f64`.
    #[inline]
    fn reach(&self, level_a: usize, a: usize, level_b: usize, b: usize) -> (f64, f64) {
        let d = self.d;
        let ba = &self.boxes[level_a][a * 2 * d..(a + 1) * 2 * d];
        let bb = &self.boxes[level_b][b * 2 * d..(b + 1) * 2 * d];
        let (mut near, mut far) = (0.0f64, 0.0f64);
        for k in 0..d {
            let (alo, ahi) = (ba[k] as f64, ba[d + k] as f64);
            let (blo, bhi) = (bb[k] as f64, bb[d + k] as f64);
            let gap = (alo - bhi).max(blo - ahi).max(0.0);
            let span = (ahi - blo).max(bhi - alo);
            near += gap * gap;
            far += span * span;
        }
        (near, far)
    }
}

fn bounds(data: &[f32], d: usize, ids: &[u32]) -> (Vec<f32>, Vec<f32>) {
    let mut min = vec![f32::INFINITY; d];
    let mut max = vec![f32::NEG_INFINITY; d];
    for &id in ids {
        for k in 0..d {
            let x = data[id as usize * d + k];
            min[k] = min[k].min(x);
            max[k] = max[k].max(x);
        }
    }
    (min, max)
}

/// Every pair whose bin is in `bins` and whose distance is within the scan's
/// cutoff, as canonical edges on the original ids, in no particular order.
pub(crate) fn band_edges(
    points: &SpatialPoints,
    scan: &PointScan,
    bins: std::ops::Range<usize>,
    parallel: bool,
) -> Vec<Edge> {
    let SpatialPoints { n, d, .. } = *points;
    let cap = scan.threshold;
    // Distances that can land in these bins, widened for the rounding of
    // `distance * scale`, compared squared against the box bounds. A computed
    // distance is within `(d + 2) / 2 + 1` ulps of the true one, so the box
    // test is widened by more than that.
    let slack = (d as f64 + 4.0) * f64::from(f32::EPSILON);
    let (near_limit, far_limit) = if scan.scale > 0.0 {
        let scale = scan.scale as f64;
        let lo = bins.start as f64 / scale * (1.0 - 4.0 * f64::from(f32::EPSILON));
        let hi = (bins.end as f64 / scale).min(cap as f64) * (1.0 + 4.0 * f64::from(f32::EPSILON));
        ((hi * (1.0 + slack)).powi(2), (lo * (1.0 - slack)).max(0.0).powi(2))
    } else {
        ((cap as f64 * (1.0 + slack)).powi(2), 0.0)
    };
    // A tile can hold a band pair unless all of it is too far or too near.
    let may_hit = |near: f64, far: f64| near <= near_limit && far >= far_limit;

    let tile_band = |tile: &[[f32; NR]; MR],
                     i_min: usize,
                     j_min: usize,
                     mr: usize,
                     nr: usize,
                     full: bool,
                     out: &mut Vec<Edge>| {
        for m in 0..mr {
            let i = i_min + m;
            for nn in 0..nr {
                let j = j_min + nn;
                if full || i > j {
                    let distance = tile_dist(tile, m, nn);
                    if distance <= cap && bins.contains(&scan.bin(distance)) {
                        out.push(Edge::canonical(points.ids[i], points.ids[j], distance));
                    }
                }
            }
        }
    };
    let macro_tile = |(jc, ic): (usize, usize), panels: &mut (Vec<f32>, Vec<f32>), out: &mut Vec<Edge>| {
        let (near, far) = points.reach(1, ic / MC, 2, jc / NC);
        if !may_hit(near, far) {
            return;
        }
        let (a_packed, b_packed) = panels;
        let nc = (n - jc).min(NC);
        let mc = (n - ic).min(MC);
        pack_panel(&points.data, d, jc, nc, NR, b_packed);
        pack_panel(&points.data, d, ic, mc, MR, a_packed);
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
                if i_max > j_min {
                    let (near, far) = points.reach(0, i_min / MR, 0, j_min / NR);
                    if may_hit(near, far) {
                        let tile = micro_kernel(&a_packed[a_off..], &b_packed[b_off..], d);
                        tile_band(&tile, i_min, j_min, mr_eff, nr_eff, i_min > j_max, out);
                    }
                }
                jr += NR;
            }
            ir += MR;
        }
    };

    let tiles = macro_tiles(n);
    if parallel && tiles.len() > 1 {
        let parts: Vec<Vec<Edge>> = tiles
            .par_iter()
            .fold(
                || (alloc_panels(d), Vec::new()),
                |(mut panels, mut out), &tile| {
                    macro_tile(tile, &mut panels, &mut out);
                    (panels, out)
                },
            )
            .map(|(_, out)| out)
            .collect();
        let mut out = Vec::with_capacity(parts.iter().map(Vec::len).sum());
        for part in parts {
            out.extend(part);
        }
        out
    } else {
        let mut panels = alloc_panels(d);
        let mut out = Vec::new();
        for tile in tiles {
            macro_tile(tile, &mut panels, &mut out);
        }
        out
    }
}

/// `(jc, ic)` of every macro tile that holds a pair `i > j`.
fn macro_tiles(n: usize) -> Vec<(usize, usize)> {
    let mut tiles = Vec::new();
    for jc in (0..n).step_by(NC) {
        for ic in (jc..n).step_by(MC) {
            tiles.push((jc, ic));
        }
    }
    tiles
}

/// The micro tiles of one macro tile, as `tiled_pass` visits them.
#[allow(clippy::type_complexity)]
fn for_each_micro_tile(
    data: &[f32],
    n: usize,
    d: usize,
    jc: usize,
    ic: usize,
    panels: &mut (Vec<f32>, Vec<f32>),
    mut on_tile: impl FnMut(&[[f32; NR]; MR], usize, usize, usize, usize, bool),
) {
    let (a_packed, b_packed) = panels;
    let nc = (n - jc).min(NC);
    let mc = (n - ic).min(MC);
    pack_panel(data, d, jc, nc, NR, b_packed);
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
            if i_max > j_min {
                let tile = micro_kernel(&a_packed[a_off..], &b_packed[b_off..], d);
                on_tile(&tile, i_min, j_min, mr_eff, nr_eff, i_min > j_max);
            }
            jr += NR;
        }
        ir += MR;
    }
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

        let e = EdgeList::from_distance_matrix(&d, n, f32::INFINITY).unwrap();
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

        let e = EdgeList::from_distance_matrix(&d, n, f32::INFINITY).unwrap();
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

        let e = EdgeList::from_distance_matrix(&d, n, f32::INFINITY).unwrap();
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
        let e = EdgeList::from_coo(3, &rows, &cols, &vals, f32::INFINITY).unwrap();

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

        let dense = EdgeList::from_distance_matrix(&d, n, f32::INFINITY).unwrap();
        let sparse = EdgeList::from_coo(n, &rows, &cols, &vals, f32::INFINITY).unwrap();
        assert_eq!(sparse.n, dense.n);
        assert_eq!(sparse.center, dense.center);
        assert_eq!(sparse.threshold.to_bits(), dense.threshold.to_bits());
        assert_eq!(sorted_edges(&sparse), sorted_edges(&dense));
    }
}
