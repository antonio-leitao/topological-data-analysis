// ═══════════════════════════════════════════════════════════════════════════════
// algorithm.rs — the Ripser driver over the BitCSR data structure
// ═══════════════════════════════════════════════════════════════════════════════
//
// This file owns the algorithm logic that turns a `BitCsrDistanceMatrix` into a
// barcode: the dimension driver (`compute`), H0 via union-find, cofacet/edge
// enumeration into `Simplex128`, apparent-pair detection, and next-dimension
// candidate assembly (including the rayon-parallel path). The matrix reduction
// loop itself lives in `reduction.rs`.
//
// Everything here drives the data structure through its primitive bit/neighbour
// operations (`for_each_common_neighbor`, `edge_dist`, …); the `BitCsr` type
// exposes no Ripser concepts.

use crate::engine::bitcsr::BitCsrDistanceMatrix;
use crate::engine::reduction::{compute_pairs, RawRepresentative};
use crate::engine::simplex::{encode_filtration, FxHashMap, Simplex128};
use crate::types::{BarcodeResult, PersistenceInterval, Representative, RepresentedBarcode};

/// Minimum number of input simplices before `assemble_candidates` uses rayon.
/// Below this the per-task overhead and buffer merges outweigh the work, so we
/// stay on the sequential, single-allocation path (this also covers H1 and
/// small datasets even when the caller requests `parallel`).
const PARALLEL_ASSEMBLE_THRESHOLD: usize = 512;

// ═══════════════════════════════════════════════════════════════════════════════
// Union-Find (path halving + union by rank)
// ═══════════════════════════════════════════════════════════════════════════════

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    #[inline]
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    /// Link two DISTINCT roots, returning the surviving root.
    #[inline]
    fn link(&mut self, ra: usize, rb: usize) -> usize {
        debug_assert!(ra != rb);
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
            rb
        } else {
            self.parent[rb] = ra;
            if self.rank[ra] == self.rank[rb] {
                self.rank[ra] += 1;
            }
            ra
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Edge & cofacet enumeration — build Simplex128 from the BitCSR primitives
// ═══════════════════════════════════════════════════════════════════════════════

/// Enumerate every stored 1-simplex (edge), each exactly once (the larger
/// endpoint stored first), with its diameter encoded. Used to seed H0. Return
/// `false` from `f` to stop early.
pub(crate) fn for_each_edge(dist: &BitCsrDistanceMatrix, mut f: impl FnMut(Simplex128) -> bool) {
    for v in 0..dist.n() {
        let mut stop = false;
        dist.for_each_neighbor_above(v, |w, d| {
            if !f(Simplex128::from_sorted_desc(d, &[w, v as u16])) {
                stop = true;
                return false;
            }
            true
        });
        if stop {
            return;
        }
    }
}

/// Enumerate the cofacets of `sigma`, youngest-first (descending inserted-vertex
/// order), each carrying its cofacet diameter. The matrix already contains only
/// edges admitted by preprocessing.
///
/// - `all_cofacets = true`  → every cofacet (reduction / coboundary).
/// - `all_cofacets = false` → only cofacets whose inserted vertex exceeds σ's
///   largest vertex, so each (d+1)-simplex is produced once when assembling the
///   next dimension's column pool.
///
/// Return `false` from `f` to stop early.
#[inline]
pub(crate) fn for_each_cofacet(
    dist: &BitCsrDistanceMatrix,
    sigma: Simplex128,
    all_cofacets: bool,
    mut f: impl FnMut(Simplex128) -> bool,
) {
    let vc = sigma.vertex_count();
    if vc == 0 {
        return;
    }

    let verts = sigma.vertices();
    let base = sigma.filtration();
    let floor: i32 = if all_cofacets {
        -1
    } else {
        sigma.largest_vertex() as i32
    };
    // Cache the cofacet-construction masks; they depend only on the insertion
    // rank `vi`, which advances monotonically as the youngest-first neighbour id
    // descends past σ's vertices.
    let payload = sigma.vertex_key();
    let mut vi = 0usize;
    let mut payload_upper: u128 = 0;
    let mut payload_lower_shifted: u128 = payload >> 16;
    let mut insert_shift: u32 = 80;

    dist.for_each_common_neighbor(&verts, vc, floor, |w, extra| {
        let old_vi = vi;
        while vi < vc && verts[vi] > w {
            vi += 1;
        }
        if vi != old_vi {
            let split = 96 - (vi as u32) * 16;
            let upper_mask = (!0u128) << split;
            payload_upper = payload & upper_mask;
            payload_lower_shifted = (payload & !upper_mask) >> 16;
            insert_shift = split - 16;
        }

        let diam = if extra > base { extra } else { base };
        let w_ins = (w as u128) + 1;
        let cof = Simplex128(
            ((encode_filtration(diam) as u128) << 96)
                | payload_upper
                | (w_ins << insert_shift)
                | payload_lower_shifted,
        );
        f(cof)
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Apparent-pair detection (Ripser paper, Proposition 3.9)
// ═══════════════════════════════════════════════════════════════════════════════
//
// A pair (σ, τ) with dim τ = dim σ + 1 is a zero-persistence apparent pair iff
//   (C1) τ is the youngest same-diameter cofacet of σ, and
//   (C2) σ is the oldest  same-diameter facet   of τ.

/// First common neighbour whose max edge distance to the set is ≤ `max_extra`
/// (youngest-first), or `None`.
#[inline]
fn first_common_neighbor_le(
    dist: &BitCsrDistanceMatrix,
    verts: &[u16; 6],
    vc: usize,
    max_extra: f32,
) -> Option<u16> {
    let mut result = None;
    dist.for_each_common_neighbor(verts, vc, -1, |w, extra| {
        if extra <= max_extra {
            result = Some(w);
            false
        } else {
            true
        }
    });
    result
}

/// Diameter of the facet of `verts[0..vc]` obtained by removing slot `skip`.
#[inline]
fn facet_diameter(dist: &BitCsrDistanceMatrix, verts: &[u16; 6], vc: usize, skip: usize) -> f32 {
    let mut diam = 0.0f32;
    for i in 0..vc {
        if i == skip {
            continue;
        }
        for j in (i + 1)..vc {
            if j == skip {
                continue;
            }
            let d = dist.edge_dist(verts[i], verts[j]);
            if d > diam {
                diam = d;
            }
        }
    }
    diam
}

/// Bitmask (over `edge_bit(i, j)`) of the edges whose encoded distance equals
/// `diam_bits` — the "diameter edges" of the vertex set.
#[inline]
fn diameter_edge_mask(
    dist: &BitCsrDistanceMatrix,
    verts: &[u16; 6],
    vc: usize,
    diam_bits: u32,
) -> u64 {
    let mut mask = 0u64;
    for i in 0..vc {
        for j in (i + 1)..vc {
            if dist.edge_dist_bits(verts[i], verts[j]) == diam_bits {
                mask |= edge_bit(i, j);
            }
        }
    }
    mask
}

/// Oldest same-diameter facet of `tau`, or `None`.
#[inline]
fn zero_pivot_facet(dist: &BitCsrDistanceMatrix, tau: Simplex128) -> Option<Simplex128> {
    let target = tau.filtration_encoded();
    let verts = tau.vertices();
    let vc = tau.vertex_count();

    if vc >= 3 {
        let diam_edges = diameter_edge_mask(dist, &verts, vc, !target);
        if let Some(skip) = oldest_same_diam_facet_slot(vc, diam_edges) {
            return Some(tau.remove_vertex_at(skip, target));
        }
        return None;
    }

    for k in 0..vc {
        let diam = facet_diameter(dist, &verts, vc, k);
        if encode_filtration(diam) == target {
            return Some(tau.remove_vertex_at(k, target));
        }
    }
    None
}

/// Youngest same-diameter cofacet of `sigma`, or `None`.
#[inline]
fn zero_pivot_cofacet(dist: &BitCsrDistanceMatrix, sigma: Simplex128) -> Option<Simplex128> {
    let same_diam = sigma.filtration();
    let vc = sigma.vertex_count();
    if vc == 0 {
        return None;
    }
    let verts = sigma.vertices();
    let w = first_common_neighbor_le(dist, &verts, vc, same_diam)?;
    Some(sigma.cofacet(w, sigma.filtration_encoded()))
}

/// If `tau` is the cofacet side of an apparent pair, return its facet partner φ.
/// Used as the reduction shortcut.
#[inline]
pub(crate) fn zero_apparent_facet(
    dist: &BitCsrDistanceMatrix,
    tau: Simplex128,
) -> Option<Simplex128> {
    let phi = zero_pivot_facet(dist, tau)?;
    let tau_check = zero_pivot_cofacet(dist, phi)?;
    if tau_check == tau {
        Some(phi)
    } else {
        None
    }
}

/// If `sigma` is the facet side of an apparent pair, return its cofacet partner
/// τ. Used by H0 to skip already-paired edges.
#[inline]
fn zero_apparent_cofacet(dist: &BitCsrDistanceMatrix, sigma: Simplex128) -> Option<Simplex128> {
    let tau = zero_pivot_cofacet(dist, sigma)?;
    let phi = zero_pivot_facet(dist, tau)?;
    if phi == sigma {
        Some(tau)
    } else {
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// H0: enumerate edges, Kruskal, split into simplices + columns_to_reduce
// ═══════════════════════════════════════════════════════════════════════════════

fn compute_h0<const RECORD: bool>(
    dist: &BitCsrDistanceMatrix,
    intervals: &mut Vec<PersistenceInterval>,
    reps: &mut Vec<RawRepresentative>,
    parallel: bool,
) -> (Vec<Simplex128>, Vec<Simplex128>) {
    let n = dist.n();

    let mut edges: Vec<Simplex128> = Vec::new();
    for_each_edge(dist, |edge| {
        edges.push(edge);
        true
    });

    // Kruskal needs SMALLEST DIAMETER FIRST (= oldest-first). Under the
    // Ripser-compatible Ord, oldest is "greater", so descending sort gives
    // oldest-first iteration here. Keys are distinct (one per edge), so the
    // parallel unstable sort is deterministic.
    if parallel {
        use rayon::prelude::*;
        edges.par_sort_unstable_by(|a, b| b.cmp(a));
    } else {
        edges.sort_unstable_by(|a, b| b.cmp(a));
    }

    let mut uf = UnionFind::new(n);
    // Minimum vertex id per component root, maintained only when recording.
    // Every vertex is born at 0, so the elder rule reduces to an id
    // convention: the surviving class keeps the smaller minimum, the dying
    // class is named by the larger one (matching Ripser's vertex pairing).
    let mut comp_min: Vec<u16> = if RECORD {
        (0..n).map(|v| v as u16).collect()
    } else {
        Vec::new()
    };
    // No prealloc: only non-merging, non-apparent edges land here (usually a
    // small fraction), and a full-edge-count reservation would dwarf the result.
    let mut columns_to_reduce = Vec::new();

    let mut num_merges: usize = 0;
    for &edge in &edges {
        let verts = edge.vertices();
        let ra = uf.find(verts[0] as usize);
        let rb = uf.find(verts[1] as usize);
        if ra != rb {
            num_merges += 1;
            // ZERO-PERSISTENCE FILTER: edges with diam=0 (duplicate points)
            // produce H0 pairs (0, 0) which Ripser excludes from output by
            // default. Skip them here too to match.
            let death = edge.filtration();
            if death > 0.0 {
                intervals.push(PersistenceInterval { birth: 0.0, death });
                if RECORD {
                    let dying = comp_min[ra].max(comp_min[rb]);
                    reps.push(RawRepresentative {
                        birth: Simplex128::from_sorted_desc(0.0, &[dying]),
                        death: Some(edge),
                        cocycle: Vec::new(),
                    });
                }
            }
            let winner = uf.link(ra, rb);
            if RECORD {
                comp_min[winner] = comp_min[ra].min(comp_min[rb]);
            }
        } else {
            // Edge did NOT merge components — it closes a 1-cycle candidate.
            // Filter: if the edge is already the cofacet side of an apparent pair
            // with one of its endpoints, it's guaranteed to be paired in the H1
            // reduction and we can skip it here (Ripser, compute_dim_0_pairs).
            if zero_apparent_cofacet(dist, edge).is_none() {
                columns_to_reduce.push(edge);
            }
        }
    }

    // Number of connected components at the matrix cutoff = n - num_merges. This is
    // always ≥ 1 (the whole space is at least one component). Each component
    // contributes one essential H0 class.
    let num_essential = n - num_merges;
    for _ in 0..num_essential {
        intervals.push(PersistenceInterval {
            birth: 0.0,
            death: f32::INFINITY,
        });
    }
    if RECORD {
        for v in 0..n {
            if uf.find(v) == v {
                reps.push(RawRepresentative {
                    birth: Simplex128::from_sorted_desc(0.0, &[comp_min[v]]),
                    death: None,
                    cocycle: Vec::new(),
                });
            }
        }
    }

    // Release growth slack: this vector stays alive through the whole H1
    // reduction, so dead capacity here is a long-lived cost.
    columns_to_reduce.shrink_to_fit();
    columns_to_reduce.reverse();
    (edges, columns_to_reduce)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Assemble columns_to_reduce for the next dimension
// ═══════════════════════════════════════════════════════════════════════════════

/// Build the next dimension's column list from `simplices` (the d-simplices
/// surviving from the previous dimension). For each cofacet τ enumerated:
///   1. Push into `next_simplices` (becomes the d+1-simplex pool for the
///      dimension after this), when `build_pool`.
///   2. Add to `columns_to_reduce` UNLESS τ is filtered out by clearing
///      (already a known pivot) or apparent-pair filtering.
///
/// Each simplex expands independently — inputs are the immutable matrix and the
/// frozen `cleared_pivots`, outputs are pure appends — so the loop parallelises
/// by giving each worker its own buffers and concatenating; the final sort makes
/// the merge order irrelevant. The size guard keeps small inputs (and every
/// H1/small-dataset case) on the allocation-free sequential path.
pub(crate) fn assemble_candidates(
    dist: &BitCsrDistanceMatrix,
    simplices: &mut Vec<Simplex128>,
    cleared_pivots: &FxHashMap<Simplex128, usize>,
    build_pool: bool,
    parallel: bool,
) -> Vec<Simplex128> {
    if simplices.is_empty() {
        return Vec::new();
    }

    let use_par = parallel && simplices.len() >= PARALLEL_ASSEMBLE_THRESHOLD;

    let (next_simplices, mut columns_to_reduce) = if use_par {
        use rayon::prelude::*;
        // Collect per-worker buffers, then merge once with an exact reservation.
        // (A tree reduce would recopy each element O(log workers) times.)
        let workers: Vec<(Vec<Simplex128>, Vec<Simplex128>)> = simplices
            .par_iter()
            .fold(
                || (Vec::new(), Vec::new()),
                |(mut next_local, mut cols_local), &sigma| {
                    assemble_one(
                        dist,
                        sigma,
                        cleared_pivots,
                        build_pool,
                        &mut next_local,
                        &mut cols_local,
                    );
                    (next_local, cols_local)
                },
            )
            .collect();

        let total_next: usize = workers.iter().map(|(next, _)| next.len()).sum();
        let total_cols: usize = workers.iter().map(|(_, cols)| cols.len()).sum();
        let mut workers = workers.into_iter();
        let (mut next, mut cols) = workers.next().unwrap_or_default();
        next.reserve_exact(total_next - next.len());
        cols.reserve_exact(total_cols - cols.len());
        for (next_local, cols_local) in workers {
            next.extend(next_local);
            cols.extend(cols_local);
        }
        (next, cols)
    } else {
        let mut next_simplices: Vec<Simplex128> = if build_pool {
            Vec::with_capacity(simplices.len() * 2)
        } else {
            Vec::new()
        };
        let mut columns_to_reduce: Vec<Simplex128> = Vec::with_capacity(simplices.len());
        for &sigma in simplices.iter() {
            assemble_one(
                dist,
                sigma,
                cleared_pivots,
                build_pool,
                &mut next_simplices,
                &mut columns_to_reduce,
            );
        }
        (next_simplices, columns_to_reduce)
    };

    if use_par {
        use rayon::prelude::*;
        columns_to_reduce.par_sort_unstable();
    } else {
        columns_to_reduce.sort_unstable();
    }
    if build_pool {
        *simplices = next_simplices;
    }
    columns_to_reduce
}

#[inline]
fn assemble_one(
    dist: &BitCsrDistanceMatrix,
    sigma: Simplex128,
    cleared_pivots: &FxHashMap<Simplex128, usize>,
    build_pool: bool,
    next_simplices: &mut Vec<Simplex128>,
    columns_to_reduce: &mut Vec<Simplex128>,
) {
    if sigma.vertex_count() == 2 {
        assemble_edge_candidates(
            dist,
            sigma,
            cleared_pivots,
            build_pool,
            next_simplices,
            columns_to_reduce,
        );
    } else {
        assemble_generic_candidates(
            dist,
            sigma,
            cleared_pivots,
            build_pool,
            next_simplices,
            columns_to_reduce,
        );
    }
}

#[inline]
fn assemble_edge_candidates(
    dist: &BitCsrDistanceMatrix,
    sigma: Simplex128,
    cleared_pivots: &FxHashMap<Simplex128, usize>,
    build_pool: bool,
    next_simplices: &mut Vec<Simplex128>,
    columns_to_reduce: &mut Vec<Simplex128>,
) {
    let verts = sigma.vertices();
    let v0 = verts[0];
    let v1 = verts[1];
    let base = sigma.filtration();
    let base_bits = !sigma.filtration_encoded();
    let floor = sigma.largest_vertex() as i32;

    dist.for_each_common_neighbor_edges(v0, v1, floor, |w, d0, d1| {
        let extra = d0.max(d1);
        let diam = base.max(extra);
        let tau = Simplex128::from_sorted_desc(diam, &[w, v0, v1]);
        if build_pool {
            next_simplices.push(tau);
        }

        if !cleared_pivots.contains_key(&tau) {
            let diam_bits = diam.to_bits();
            let mut diam_edges = 0u64;
            if d0.to_bits() == diam_bits {
                diam_edges |= edge_bit(0, 1);
            }
            if d1.to_bits() == diam_bits {
                diam_edges |= edge_bit(0, 2);
            }
            if base_bits == diam_bits {
                diam_edges |= edge_bit(1, 2);
            }
            let tau_verts = [w, v0, v1, 0, 0, 0];
            if keep_fused_candidate(dist, tau, &tau_verts, 3, diam_edges) {
                columns_to_reduce.push(tau);
            }
        }
        true
    });
}

#[inline]
fn assemble_generic_candidates(
    dist: &BitCsrDistanceMatrix,
    sigma: Simplex128,
    cleared_pivots: &FxHashMap<Simplex128, usize>,
    build_pool: bool,
    next_simplices: &mut Vec<Simplex128>,
    columns_to_reduce: &mut Vec<Simplex128>,
) {
    for_each_cofacet(dist, sigma, false, |tau| {
        if build_pool {
            next_simplices.push(tau);
        }
        if !cleared_pivots.contains_key(&tau) {
            let vc = tau.vertex_count();
            let verts = tau.vertices();
            let diam_edges = diameter_edge_mask(dist, &verts, vc, !tau.filtration_encoded());
            if keep_fused_candidate(dist, tau, &verts, vc, diam_edges) {
                columns_to_reduce.push(tau);
            }
        }
        true
    });
}

/// A cofacet τ survives iff it is NOT the facet side and NOT the cofacet side of
/// an apparent pair (the fused apparent-pair filter).
#[inline]
fn keep_fused_candidate(
    dist: &BitCsrDistanceMatrix,
    tau: Simplex128,
    tau_verts: &[u16; 6],
    vc: usize,
    diam_edges: u64,
) -> bool {
    !is_zero_apparent_facet_side(dist, tau, tau_verts, vc, diam_edges)
        && !is_zero_apparent_cofacet_side(dist, tau, tau_verts, vc, diam_edges)
}

#[inline]
fn is_zero_apparent_cofacet_side(
    dist: &BitCsrDistanceMatrix,
    tau: Simplex128,
    tau_verts: &[u16; 6],
    vc: usize,
    diam_edges: u64,
) -> bool {
    let Some(skip) = oldest_same_diam_facet_slot(vc, diam_edges) else {
        return false;
    };

    let mut facet_verts = [0u16; 6];
    let mut out = 0usize;
    for i in 0..vc {
        if i != skip {
            facet_verts[out] = tau_verts[i];
            out += 1;
        }
    }

    first_common_neighbor_le(dist, &facet_verts, vc - 1, tau.filtration()) == Some(tau_verts[skip])
}

#[inline]
fn is_zero_apparent_facet_side(
    dist: &BitCsrDistanceMatrix,
    tau: Simplex128,
    tau_verts: &[u16; 6],
    vc: usize,
    diam_edges: u64,
) -> bool {
    if vc >= 6 {
        return false;
    }

    let Some(w) = first_common_neighbor_le(dist, tau_verts, vc, tau.filtration()) else {
        return false;
    };

    let (_rho_verts, rank) = insert_vertex_desc(tau_verts, vc, w);
    let mut rho_edges = remap_edge_mask_after_insert(diam_edges, vc, rank);
    let diam_bits = !tau.filtration_encoded();
    for i in 0..vc {
        if dist.edge_dist_bits(w, tau_verts[i]) == diam_bits {
            let j = if i >= rank { i + 1 } else { i };
            let (a, b) = if rank < j { (rank, j) } else { (j, rank) };
            rho_edges |= edge_bit(a, b);
        }
    }

    oldest_same_diam_facet_slot(vc + 1, rho_edges) == Some(rank)
}

// ── Small combinatorial helpers for the apparent-pair / assembly logic ────────

/// Encode edge `(i, j)` (with `i < j < 6`) as a single bit in a 64-bit mask.
#[inline(always)]
fn edge_bit(i: usize, j: usize) -> u64 {
    debug_assert!(i < j && j < 6);
    1u64 << (i * 6 + j)
}

/// Given the mask of diameter edges, which slot to remove to obtain the oldest
/// same-diameter facet, or `None` if there is no same-diameter facet.
#[inline]
fn oldest_same_diam_facet_slot(vc: usize, diam_edges: u64) -> Option<usize> {
    if vc < 3 || diam_edges == 0 {
        return None;
    }
    for skip in 0..vc {
        for i in 0..vc {
            if i == skip {
                continue;
            }
            for j in (i + 1)..vc {
                if j != skip && (diam_edges & edge_bit(i, j)) != 0 {
                    return Some(skip);
                }
            }
        }
    }
    None
}

/// Insert `w` into the descending vertex array, returning the new array and the
/// insertion rank.
#[inline]
fn insert_vertex_desc(verts: &[u16; 6], vc: usize, w: u16) -> ([u16; 6], usize) {
    let mut out = [0u16; 6];
    let mut i = 0usize;
    while i < vc && verts[i] > w {
        out[i] = verts[i];
        i += 1;
    }
    let rank = i;
    out[rank] = w;
    while i < vc {
        out[i + 1] = verts[i];
        i += 1;
    }
    (out, rank)
}

/// Adjust an edge mask after inserting a new vertex at `rank`.
#[inline]
fn remap_edge_mask_after_insert(mask: u64, vc: usize, rank: usize) -> u64 {
    let mut out = 0u64;
    for i in 0..vc {
        for j in (i + 1)..vc {
            if (mask & edge_bit(i, j)) == 0 {
                continue;
            }
            let ni = if i >= rank { i + 1 } else { i };
            let nj = if j >= rank { j + 1 } else { j };
            out |= edge_bit(ni, nj);
        }
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main loop
// ═══════════════════════════════════════════════════════════════════════════════

/// Barcode only — the hot entry point. Monomorphised with recording compiled
/// out, so it pays nothing for the representatives machinery.
pub fn compute(dist: &BitCsrDistanceMatrix, max_dim: usize, parallel: bool) -> BarcodeResult {
    compute_impl::<false>(dist, max_dim, parallel).0
}

/// Barcode plus cocycle representatives (persistent cohomology), aligned
/// index-for-index with the emitted intervals of each dimension.
pub fn compute_with_representatives(
    dist: &BitCsrDistanceMatrix,
    max_dim: usize,
    parallel: bool,
) -> RepresentedBarcode {
    let (barcode, raw) = compute_impl::<true>(dist, max_dim, parallel);
    let representatives = raw
        .into_iter()
        .map(|dim| dim.into_iter().map(convert_representative).collect())
        .collect();
    RepresentedBarcode {
        barcode,
        representatives,
    }
}

/// Vertex ids (descending) of a packed simplex, for external consumption.
fn simplex_vertices(simplex: Simplex128) -> Vec<u16> {
    let vc = simplex.vertex_count();
    simplex.vertices()[..vc].to_vec()
}

fn convert_representative(raw: RawRepresentative) -> Representative {
    Representative {
        birth: simplex_vertices(raw.birth),
        death: raw.death.map(simplex_vertices),
        cocycle: raw.cocycle.into_iter().map(simplex_vertices).collect(),
    }
}

fn compute_impl<const RECORD: bool>(
    dist: &BitCsrDistanceMatrix,
    max_dim: usize,
    parallel: bool,
) -> (BarcodeResult, Vec<Vec<RawRepresentative>>) {
    let mut intervals: Vec<Vec<PersistenceInterval>> = Vec::with_capacity(max_dim + 1);
    let mut representatives: Vec<Vec<RawRepresentative>> = Vec::new();

    // H0: union-find. No clearing input needed (no previous dim).
    let mut h0 = Vec::new();
    let mut h0_reps = Vec::new();
    let (mut simplices, mut columns_to_reduce) =
        compute_h0::<RECORD>(dist, &mut h0, &mut h0_reps, parallel);
    intervals.push(h0);
    if RECORD {
        representatives.push(h0_reps);
    }

    if max_dim <= 1 {
        // The edge pool only feeds assemble_candidates (max_dim >= 2); free it
        // before the H1 reduction rather than carrying it through.
        simplices = Vec::new();
    }

    // H1 .. H_{max_dim}
    for dim in 1..=max_dim {
        let mut dim_intervals = Vec::new();
        let mut dim_reps = Vec::new();

        // Reduce this dimension's columns. The returned pivot map holds exactly
        // the pivots recorded at THIS dimension; its key set is the clearing
        // set consumed by assemble_candidates for dim+1.
        let pivots =
            compute_pairs::<RECORD>(&columns_to_reduce, dist, &mut dim_intervals, &mut dim_reps);
        intervals.push(dim_intervals);
        if RECORD {
            representatives.push(dim_reps);
        }

        if dim < max_dim {
            let build_pool = dim + 1 < max_dim;
            columns_to_reduce =
                assemble_candidates(dist, &mut simplices, &pivots, build_pool, parallel);
            if !build_pool {
                // Last assembly: the d-simplex pool is dead weight from here on;
                // free it before the final (largest) reduction.
                simplices = Vec::new();
            }
        }
    }

    (BarcodeResult { intervals }, representatives)
}

// ═══════════════════════════════════════════════════════════════════════════════
// End-to-end correctness tests (driven through the BitCSR backend)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::edgelist::square_from_lower_tri;
    use crate::preprocess::edgelist::EdgeList;

    /// Build a BitCSR from a lower-triangular fixture (through the real pdist
    /// distance-matrix path) and run the engine.
    fn run(lt: &[f32], n: usize, threshold: f32, max_dim: usize) -> BarcodeResult {
        let sq = square_from_lower_tri(n, lt);
        let edges = EdgeList::from_distance_matrix(&sq, n, threshold);
        let bitcsr = BitCsrDistanceMatrix::from_edge_list(edges);
        compute(&bitcsr, max_dim, true)
    }

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn count_finite_h_d(barcode: &BarcodeResult, d: usize) -> usize {
        barcode.intervals[d]
            .iter()
            .filter(|iv| iv.death.is_finite() && iv.death > iv.birth)
            .count()
    }

    fn count_essential_h_d(barcode: &BarcodeResult, d: usize) -> usize {
        barcode.intervals[d]
            .iter()
            .filter(|iv| iv.death.is_infinite())
            .count()
    }

    /// Two-point space: H0 has one essential class (1 component, never dies)
    /// and one finite class [0, d(0,1)). H1 is empty.
    #[test]
    fn h0_two_points() {
        let bc = run(&[1.0], 2, f32::INFINITY, 1);
        assert_eq!(bc.intervals.len(), 2);
        // H0: one finite [0,1) and one essential [0, ∞)
        assert_eq!(bc.intervals[0].len(), 2);
        let mut h0_sorted: Vec<_> = bc.intervals[0].clone();
        h0_sorted.sort_by(|a, b| a.death.partial_cmp(&b.death).unwrap());
        assert!(approx_eq(h0_sorted[0].birth, 0.0, 1e-6));
        assert!(approx_eq(h0_sorted[0].death, 1.0, 1e-6));
        assert!(h0_sorted[1].death.is_infinite());
        // H1: empty
        assert_eq!(bc.intervals[1].len(), 0);
    }

    /// 4 points forming a 4-cycle, no diagonal close enough to fill it.
    /// d(1,0)=d(2,1)=d(3,2)=d(3,0)=1, diagonals d(2,0)=d(3,1)=10.
    /// Threshold 1.5: 4-cycle alive, no triangles. Should give 1 essential H1.
    #[test]
    fn unfilled_4cycle_essential_h1() {
        let bc = run(
            &[
                1.0, // (1,0)
                10.0, 1.0, // (2,0), (2,1)
                1.0, 10.0, 1.0, // (3,0), (3,1), (3,2)
            ],
            4,
            1.5,
            1,
        );

        // H0: 4 components reduce to 1 → 3 finite [0,1) + 1 essential.
        assert_eq!(count_essential_h_d(&bc, 0), 1);
        let h0_finite_count = bc.intervals[0]
            .iter()
            .filter(|iv| iv.death.is_finite())
            .count();
        assert_eq!(h0_finite_count, 3);

        // H1: exactly 1 essential cycle (the unfilled 4-cycle).
        assert_eq!(count_essential_h_d(&bc, 1), 1);
        // No finite H1 (no triangles fill anything in).
        assert_eq!(count_finite_h_d(&bc, 1), 0);
    }

    #[test]
    fn unit_square_h0_h1() {
        let s = std::f32::consts::SQRT_2;
        // d(1,0)=1, d(2,0)=√2, d(2,1)=1, d(3,0)=1, d(3,1)=√2, d(3,2)=1
        let bc = run(&[1.0, s, 1.0, 1.0, s, 1.0], 4, f32::INFINITY, 2);

        // H0: exactly 3 finite intervals [0, 1.0) and 1 essential.
        assert_eq!(count_essential_h_d(&bc, 0), 1, "H0 essentials");
        let h0_finite: Vec<_> = bc.intervals[0]
            .iter()
            .filter(|iv| iv.death.is_finite())
            .collect();
        assert_eq!(h0_finite.len(), 3, "H0 finite intervals");
        for iv in &h0_finite {
            assert!(approx_eq(iv.birth, 0.0, 1e-6));
            assert!(approx_eq(iv.death, 1.0, 1e-6));
        }

        // H1: exactly 1 finite interval [1.0, √2), no essentials.
        assert_eq!(count_essential_h_d(&bc, 1), 0, "H1 essentials");
        assert_eq!(count_finite_h_d(&bc, 1), 1, "H1 finite intervals");
        let h1: &PersistenceInterval = bc.intervals[1]
            .iter()
            .find(|iv| iv.death.is_finite() && iv.death > iv.birth)
            .unwrap();
        assert!(
            approx_eq(h1.birth, 1.0, 1e-6),
            "H1 birth = 1.0, got {}",
            h1.birth
        );
        assert!(
            approx_eq(h1.death, s, 1e-5),
            "H1 death = √2, got {}",
            h1.death
        );
    }

    /// Two disconnected segments (4 points: 0-1 close, 2-3 close, far between).
    #[test]
    fn two_components_finite_threshold() {
        // d(1,0)=1, d(2,0)=10, d(2,1)=9, d(3,0)=11, d(3,1)=10, d(3,2)=1
        let bc = run(&[1.0, 10.0, 9.0, 11.0, 10.0, 1.0], 4, 5.0, 1);
        // With threshold 5, edges of length > 5 don't form. Components: {0,1} and {2,3}.
        // H0: 2 finite [0,1), 2 essentials.
        assert_eq!(count_essential_h_d(&bc, 0), 2);
        let finite_count = bc.intervals[0]
            .iter()
            .filter(|iv| iv.death.is_finite())
            .count();
        assert_eq!(finite_count, 2);
        for iv in &bc.intervals[0] {
            if iv.death.is_finite() {
                assert!(approx_eq(iv.death, 1.0, 1e-6));
            }
        }
        // H1: empty (no cycles possible).
        assert_eq!(bc.intervals[1].len(), 0);
    }

    /// PROPERTY: every recorded cocycle is genuinely a representative of its
    /// interval. Over Z/2, the coboundary of the cocycle (XOR of all cofacets
    /// of its simplices) must satisfy:
    ///   * paired class: the youngest surviving cofacet IS the death simplex,
    ///     and every survivor has diameter >= the death (the reduced column's
    ///     pivot property);
    ///   * essential class: no surviving cofacet at all (a true cocycle in
    ///     the truncated complex).
    /// Also: birth/death simplex diameters equal the interval endpoints, and
    /// representatives align 1:1 with emitted intervals.
    #[test]
    fn cocycle_representatives_are_valid() {
        use std::collections::HashMap;

        let mut s: u64 = 0x0c0c_5eed_1234_5678;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 16_777_216.0
        };

        // Random fixtures (enclosing-radius truncation: no essentials in
        // dim >= 1) plus the unfilled 4-cycle at a user threshold (essential
        // H1 exercises the empty-coboundary branch).
        let mut cases: Vec<(usize, Vec<f32>, f32)> = Vec::new();
        for &n in &[7usize, 9, 10] {
            let m = n * (n - 1) / 2;
            cases.push((n, (0..m).map(|_| 0.2 + rng()).collect(), f32::INFINITY));
        }
        cases.push((4, vec![1.0, 10.0, 1.0, 1.0, 10.0, 1.0], 1.5));

        for (n, lt, threshold) in cases {
            let sq = crate::preprocess::edgelist::square_from_lower_tri(n, &lt);
            let edges = crate::preprocess::edgelist::EdgeList::from_distance_matrix(
                &sq, n, threshold,
            );
            let bitcsr = BitCsrDistanceMatrix::from_edge_list(edges);
            let max_dim = 2;
            let (bc, reps) = compute_impl::<true>(&bitcsr, max_dim, false);
            let plain = compute(&bitcsr, max_dim, false);

            // Recording must not perturb the barcode, and reps align 1:1.
            assert_eq!(bc.intervals.len(), plain.intervals.len());
            for d in 0..=max_dim {
                assert_eq!(bc.intervals[d].len(), plain.intervals[d].len());
                assert_eq!(reps[d].len(), bc.intervals[d].len(), "dim {d}");
            }

            for dim in 1..=max_dim {
                for (iv, rep) in bc.intervals[dim].iter().zip(&reps[dim]) {
                    assert_eq!(rep.birth.filtration(), iv.birth);
                    assert!(rep.cocycle.contains(&rep.birth));
                    assert!(rep
                        .cocycle
                        .iter()
                        .all(|s| s.vertex_count() == dim + 1));

                    // δ(cocycle) over Z/2.
                    let mut parity: HashMap<Simplex128, bool> = HashMap::new();
                    for &simplex in &rep.cocycle {
                        for_each_cofacet(&bitcsr, simplex, true, |cofacet| {
                            *parity.entry(cofacet).or_insert(false) ^= true;
                            true
                        });
                    }
                    let surviving: Vec<Simplex128> = parity
                        .into_iter()
                        .filter_map(|(c, odd)| odd.then_some(c))
                        .collect();

                    match rep.death {
                        Some(tau) => {
                            assert_eq!(tau.filtration(), iv.death);
                            // Youngest survivor (max u128) is exactly τ.
                            assert_eq!(surviving.iter().max(), Some(&tau));
                            assert!(surviving
                                .iter()
                                .all(|c| c.filtration() >= iv.death));
                        }
                        None => {
                            assert!(iv.death.is_infinite());
                            assert!(
                                surviving.is_empty(),
                                "essential class has a nonzero coboundary"
                            );
                        }
                    }
                }
            }

            // H0: pairs only — birth vertex, death edge, empty cocycle.
            for (iv, rep) in bc.intervals[0].iter().zip(&reps[0]) {
                assert_eq!(rep.birth.vertex_count(), 1);
                assert!(rep.cocycle.is_empty());
                match rep.death {
                    Some(edge) => {
                        assert_eq!(edge.vertex_count(), 2);
                        assert_eq!(edge.filtration(), iv.death);
                    }
                    None => assert!(iv.death.is_infinite()),
                }
            }
        }
    }

    /// Octahedron: 6 points forming an octahedron. Expected H1 = 0, H2 ≥ 1
    /// (the octahedral surface is a 2-sphere).
    #[test]
    fn octahedron_h2() {
        // Indices: 0=(+1,0,0), 1=(-1,0,0), 2=(0,+1,0), 3=(0,-1,0), 4=(0,0,+1), 5=(0,0,-1)
        //   antipodal pairs (0,1), (2,3), (4,5) → 2.0; adjacent pairs → √2
        let s = std::f32::consts::SQRT_2;
        let bc = run(
            &[
                2.0, // (1,0)  antipodal
                s, s, // (2,0), (2,1)  adjacent
                s, s, 2.0, // (3,0), (3,1), (3,2)  — (3,2) antipodal
                s, s, s, s, // (4,0..3) all adjacent
                s, s, s, s, 2.0, // (5,0..3) adjacent, (5,4) antipodal
            ],
            6,
            f32::INFINITY,
            2,
        );

        // H0: 1 essential, 5 finite (all dying at √2 when octahedron connects).
        assert_eq!(count_essential_h_d(&bc, 0), 1);
        let h0_finite = bc.intervals[0]
            .iter()
            .filter(|iv| iv.death.is_finite())
            .count();
        assert_eq!(h0_finite, 5);

        // H1: no essential classes.
        assert_eq!(
            count_essential_h_d(&bc, 1),
            0,
            "H1 should have no essential classes"
        );

        // H2: at least one interval, born at √2 (the 2-sphere).
        let h2_count = bc.intervals[2].len();
        assert!(
            h2_count >= 1,
            "H2 should have at least one interval, got {}",
            h2_count
        );
        let has_h2_at_sqrt2 = bc.intervals[2]
            .iter()
            .any(|iv| approx_eq(iv.birth, s, 1e-5));
        assert!(
            has_h2_at_sqrt2,
            "expected H2 interval born at √2, got {:?}",
            bc.intervals[2]
        );
    }
}
