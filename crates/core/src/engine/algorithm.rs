use crate::engine::bitcsr::{BitCsrDistanceMatrix, Rows};
use crate::engine::reduction::compute_pairs;
use crate::engine::simplex::{encode_filtration, CofacetBuilder, PackedSimplex};
use crate::types::{BarcodeResult, PersistenceInterval};
use crate::utils::fxhash::FxHashMap;

/// Minimum input size for parallel candidate assembly.
const PARALLEL_ASSEMBLE_THRESHOLD: usize = 512;

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

    /// Link distinct roots and return the survivor.
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

/// Enumerate stored edges once, with the larger endpoint first.
pub(crate) fn for_each_edge<const K: usize, S: PackedSimplex<K>, D: Rows>(
    dist: &D,
    mut f: impl FnMut(S) -> bool,
) {
    for v in 0..dist.n() {
        let mut stop = false;
        dist.for_each_neighbor_above(v, |w, d| {
            if !f(S::from_sorted_desc(d, &[w, v as u32])) {
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

/// Enumerate cofacets youngest first with their diameters.
///
/// Unless `all_cofacets`, emit only insertions above the largest base vertex so
/// each simplex is assembled once.
#[inline]
pub(crate) fn for_each_cofacet<const K: usize, S: PackedSimplex<K>, D: Rows>(
    dist: &D,
    sigma: S,
    all_cofacets: bool,
    mut f: impl FnMut(S) -> bool,
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
    // Descending neighbour ids make the insertion rank monotone.
    let mut builder = sigma.cofacet_builder();
    let mut vi = 0usize;

    dist.for_each_common_neighbor(&verts, vc, floor, |w, extra| {
        let old_vi = vi;
        while vi < vc && verts[vi] > w {
            vi += 1;
        }
        if vi != old_vi {
            builder.set_rank(vi);
        }

        let diam = if extra > base { extra } else { base };
        f(builder.build(w, encode_filtration(diam)))
    });
}

// An apparent pair `(sigma, tau)` has `tau` as the youngest same-diameter
// cofacet of `sigma`, and `sigma` as the oldest same-diameter facet of `tau`.

/// Youngest common neighbour within `max_extra`.
#[inline]
fn first_common_neighbor_le<const K: usize, D: Rows>(
    dist: &D,
    verts: &[u32; K],
    vc: usize,
    max_extra: f32,
) -> Option<u32> {
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

#[inline]
fn facet_diameter<const K: usize, D: Rows>(
    dist: &D,
    verts: &[u32; K],
    vc: usize,
    skip: usize,
) -> f32 {
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

/// Mask of edges whose encoded distance equals `diam_bits`.
#[inline]
fn diameter_edge_mask<const K: usize, D: Rows>(
    dist: &D,
    verts: &[u32; K],
    vc: usize,
    diam_bits: u32,
) -> u64 {
    let mut mask = 0u64;
    for i in 0..vc {
        for j in (i + 1)..vc {
            if dist.edge_dist_bits(verts[i], verts[j]) == diam_bits {
                mask |= edge_bit::<K>(i, j);
            }
        }
    }
    mask
}

#[inline]
fn zero_pivot_facet<const K: usize, S: PackedSimplex<K>, D: Rows>(
    dist: &D,
    tau: S,
) -> Option<S> {
    let target = tau.filtration_encoded();
    let verts = tau.vertices();
    let vc = tau.vertex_count();

    if vc >= 3 {
        let diam_edges = diameter_edge_mask(dist, &verts, vc, !target);
        if let Some(skip) = oldest_same_diam_facet_slot::<K>(vc, diam_edges) {
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

#[inline]
fn zero_pivot_cofacet<const K: usize, S: PackedSimplex<K>, D: Rows>(
    dist: &D,
    sigma: S,
) -> Option<S> {
    let same_diam = sigma.filtration();
    let vc = sigma.vertex_count();
    if vc == 0 {
        return None;
    }
    let verts = sigma.vertices();
    let w = first_common_neighbor_le(dist, &verts, vc, same_diam)?;
    Some(sigma.cofacet(w, sigma.filtration_encoded()))
}

/// Return `tau`'s apparent facet partner.
#[inline]
pub(crate) fn zero_apparent_facet<const K: usize, S: PackedSimplex<K>, D: Rows>(
    dist: &D,
    tau: S,
) -> Option<S> {
    let phi = zero_pivot_facet(dist, tau)?;
    let tau_check = zero_pivot_cofacet(dist, phi)?;
    if tau_check == tau {
        Some(phi)
    } else {
        None
    }
}

/// Return `sigma`'s apparent cofacet partner.
#[inline]
fn zero_apparent_cofacet<const K: usize, S: PackedSimplex<K>, D: Rows>(
    dist: &D,
    sigma: S,
) -> Option<S> {
    let tau = zero_pivot_cofacet(dist, sigma)?;
    let phi = zero_pivot_facet(dist, tau)?;
    if phi == sigma {
        Some(tau)
    } else {
        None
    }
}

pub(crate) fn compute_h0<const K: usize, S: PackedSimplex<K>, D: Rows>(
    dist: &D,
    intervals: &mut Vec<PersistenceInterval>,
    parallel: bool,
) -> (Vec<S>, Vec<S>) {
    let n = dist.n();

    let mut edges: Vec<S> = Vec::new();
    for_each_edge(dist, |edge| {
        edges.push(edge);
        true
    });

    // Packed order is reversed by diameter, so descending gives Kruskal order.
    if parallel {
        use rayon::prelude::*;
        edges.par_sort_unstable_by(|a, b| b.cmp(a));
    } else {
        edges.sort_unstable_by(|a, b| b.cmp(a));
    }

    let mut uf = UnionFind::new(n);
    let mut columns_to_reduce = Vec::new();

    let mut num_merges: usize = 0;
    for &edge in &edges {
        let verts = edge.vertices();
        let ra = uf.find(verts[0] as usize);
        let rb = uf.find(verts[1] as usize);
        if ra != rb {
            num_merges += 1;
            // Match Ripser by omitting zero-persistence H0 pairs.
            let death = edge.filtration();
            if death > 0.0 {
                intervals.push(PersistenceInterval { birth: 0.0, death });
            }
            uf.link(ra, rb);
        } else {
            if zero_apparent_cofacet(dist, edge).is_none() {
                columns_to_reduce.push(edge);
            }
        }
    }

    // Each component at the cutoff contributes one essential H0 class.
    let num_essential = n - num_merges;
    for _ in 0..num_essential {
        intervals.push(PersistenceInterval {
            birth: 0.0,
            death: f32::INFINITY,
        });
    }
    columns_to_reduce.shrink_to_fit();
    columns_to_reduce.reverse();
    (edges, columns_to_reduce)
}

/// Assemble the next dimension after clearing and apparent-pair filtering.
///
/// Parallel workers use private buffers; sorting removes merge-order dependence.
pub(crate) fn assemble_candidates<const K: usize, S: PackedSimplex<K>, D: Rows>(
    dist: &D,
    simplices: &mut Vec<S>,
    cleared_pivots: &FxHashMap<S, usize>,
    build_pool: bool,
    parallel: bool,
) -> Vec<S> {
    if simplices.is_empty() {
        return Vec::new();
    }

    let use_par = parallel && simplices.len() >= PARALLEL_ASSEMBLE_THRESHOLD;

    let (next_simplices, mut columns_to_reduce) = if use_par {
        use rayon::prelude::*;
        let workers: Vec<(Vec<S>, Vec<S>)> = simplices
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
        let mut next_simplices: Vec<S> = if build_pool {
            Vec::with_capacity(simplices.len() * 2)
        } else {
            Vec::new()
        };
        let mut columns_to_reduce: Vec<S> = Vec::with_capacity(simplices.len());
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
fn assemble_one<const K: usize, S: PackedSimplex<K>, D: Rows>(
    dist: &D,
    sigma: S,
    cleared_pivots: &FxHashMap<S, usize>,
    build_pool: bool,
    next_simplices: &mut Vec<S>,
    columns_to_reduce: &mut Vec<S>,
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
fn assemble_edge_candidates<const K: usize, S: PackedSimplex<K>, D: Rows>(
    dist: &D,
    sigma: S,
    cleared_pivots: &FxHashMap<S, usize>,
    build_pool: bool,
    next_simplices: &mut Vec<S>,
    columns_to_reduce: &mut Vec<S>,
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
        let tau = S::from_sorted_desc(diam, &[w, v0, v1]);
        if build_pool {
            next_simplices.push(tau);
        }

        if !cleared_pivots.contains_key(&tau) {
            let diam_bits = diam.to_bits();
            let mut diam_edges = 0u64;
            if d0.to_bits() == diam_bits {
                diam_edges |= edge_bit::<K>(0, 1);
            }
            if d1.to_bits() == diam_bits {
                diam_edges |= edge_bit::<K>(0, 2);
            }
            if base_bits == diam_bits {
                diam_edges |= edge_bit::<K>(1, 2);
            }
            let mut tau_verts = [0u32; K];
            tau_verts[0] = w;
            tau_verts[1] = v0;
            tau_verts[2] = v1;
            if keep_fused_candidate(dist, tau, &tau_verts, 3, diam_edges) {
                columns_to_reduce.push(tau);
            }
        }
        true
    });
}

#[inline]
fn assemble_generic_candidates<const K: usize, S: PackedSimplex<K>, D: Rows>(
    dist: &D,
    sigma: S,
    cleared_pivots: &FxHashMap<S, usize>,
    build_pool: bool,
    next_simplices: &mut Vec<S>,
    columns_to_reduce: &mut Vec<S>,
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

/// Reject either side of an apparent pair.
#[inline]
fn keep_fused_candidate<const K: usize, S: PackedSimplex<K>, D: Rows>(
    dist: &D,
    tau: S,
    tau_verts: &[u32; K],
    vc: usize,
    diam_edges: u64,
) -> bool {
    !is_zero_apparent_facet_side(dist, tau, tau_verts, vc, diam_edges)
        && !is_zero_apparent_cofacet_side(dist, tau, tau_verts, vc, diam_edges)
}

#[inline]
fn is_zero_apparent_cofacet_side<const K: usize, S: PackedSimplex<K>, D: Rows>(
    dist: &D,
    tau: S,
    tau_verts: &[u32; K],
    vc: usize,
    diam_edges: u64,
) -> bool {
    let Some(skip) = oldest_same_diam_facet_slot::<K>(vc, diam_edges) else {
        return false;
    };

    let mut facet_verts = [0u32; K];
    let mut out = 0usize;
    for (i, &vertex) in tau_verts.iter().take(vc).enumerate() {
        if i != skip {
            facet_verts[out] = vertex;
            out += 1;
        }
    }

    first_common_neighbor_le(dist, &facet_verts, vc - 1, tau.filtration()) == Some(tau_verts[skip])
}

#[inline]
fn is_zero_apparent_facet_side<const K: usize, S: PackedSimplex<K>, D: Rows>(
    dist: &D,
    tau: S,
    tau_verts: &[u32; K],
    vc: usize,
    diam_edges: u64,
) -> bool {
    if vc >= K {
        return false;
    }

    let Some(w) = first_common_neighbor_le(dist, tau_verts, vc, tau.filtration()) else {
        return false;
    };

    let (_rho_verts, rank) = insert_vertex_desc(tau_verts, vc, w);
    let mut rho_edges = remap_edge_mask_after_insert::<K>(diam_edges, vc, rank);
    let diam_bits = !tau.filtration_encoded();
    for (i, &vertex) in tau_verts.iter().take(vc).enumerate() {
        if dist.edge_dist_bits(w, vertex) == diam_bits {
            let j = if i >= rank { i + 1 } else { i };
            let (a, b) = if rank < j { (rank, j) } else { (j, rank) };
            rho_edges |= edge_bit::<K>(a, b);
        }
    }

    oldest_same_diam_facet_slot::<K>(vc + 1, rho_edges) == Some(rank)
}

/// Encode edge `(i, j)` (with `i < j < K`) as a single bit in a 64-bit mask.
/// Uses row-major indexing when `K * K <= 64`, otherwise triangular indexing.
#[inline(always)]
fn edge_bit<const K: usize>(i: usize, j: usize) -> u64 {
    debug_assert!(i < j && j < K);
    if K * K <= 64 {
        1u64 << (i * K + j)
    } else {
        1u64 << (j * (j - 1) / 2 + i)
    }
}

/// Slot whose removal gives the oldest same-diameter facet.
#[inline]
fn oldest_same_diam_facet_slot<const K: usize>(vc: usize, diam_edges: u64) -> Option<usize> {
    if vc < 3 || diam_edges == 0 {
        return None;
    }
    for skip in 0..vc {
        for i in 0..vc {
            if i == skip {
                continue;
            }
            for j in (i + 1)..vc {
                if j != skip && (diam_edges & edge_bit::<K>(i, j)) != 0 {
                    return Some(skip);
                }
            }
        }
    }
    None
}

#[inline]
fn insert_vertex_desc<const K: usize>(verts: &[u32; K], vc: usize, w: u32) -> ([u32; K], usize) {
    let mut out = [0u32; K];
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

#[inline]
fn remap_edge_mask_after_insert<const K: usize>(mask: u64, vc: usize, rank: usize) -> u64 {
    let mut out = 0u64;
    for i in 0..vc {
        for j in (i + 1)..vc {
            if (mask & edge_bit::<K>(i, j)) == 0 {
                continue;
            }
            let ni = if i >= rank { i + 1 } else { i };
            let nj = if j >= rank { j + 1 } else { j };
            out |= edge_bit::<K>(ni, nj);
        }
    }
    out
}

/// Compute the barcode.
pub fn compute<const K: usize, S: PackedSimplex<K>>(
    dist: &BitCsrDistanceMatrix,
    max_dim: usize,
    parallel: bool,
) -> BarcodeResult {
    // Test the row layout once here rather than on every query.
    if dist.banded() {
        compute_rows::<K, S, _>(&dist.view::<true>(), max_dim, parallel)
    } else {
        compute_rows::<K, S, _>(&dist.view::<false>(), max_dim, parallel)
    }
}

fn compute_rows<const K: usize, S: PackedSimplex<K>, D: Rows>(
    dist: &D,
    max_dim: usize,
    parallel: bool,
) -> BarcodeResult {
    let mut intervals: Vec<Vec<PersistenceInterval>> = Vec::with_capacity(max_dim + 1);

    let mut h0 = Vec::new();
    let (mut simplices, mut columns_to_reduce) = compute_h0::<K, S, _>(dist, &mut h0, parallel);
    intervals.push(h0);

    if max_dim <= 1 {
        simplices = Vec::new();
    }

    for dim in 1..=max_dim {
        let mut dim_intervals = Vec::new();

        // This dimension's pivots clear the next dimension.
        let pivots = compute_pairs::<K, S, _>(&columns_to_reduce, dist, &mut dim_intervals);
        intervals.push(dim_intervals);

        if dim < max_dim {
            let build_pool = dim + 1 < max_dim;
            columns_to_reduce =
                assemble_candidates(dist, &mut simplices, &pivots, build_pool, parallel);
            if !build_pool {
                simplices = Vec::new();
            }
        }
    }

    BarcodeResult { intervals }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::simplex::{Simplex128, Simplex256};
    use crate::preprocess::edgelist::square_from_lower_tri;
    use crate::preprocess::edgelist::EdgeList;

    fn barcode_key(bc: &BarcodeResult) -> Vec<Vec<(u32, u32)>> {
        bc.intervals
            .iter()
            .map(|dim| {
                let mut v: Vec<_> = dim
                    .iter()
                    .map(|iv| (iv.birth.to_bits(), iv.death.to_bits()))
                    .collect();
                v.sort_unstable();
                v
            })
            .collect()
    }

    #[test]
    fn simplex256_engine_matches_simplex128() {
        let mut s: u64 = 0x5eed_cafe_0123_4567;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };

        for &(n, levels) in &[(12usize, 0u64), (20, 0), (16, 4), (24, 3)] {
            let m = n * (n - 1) / 2;
            let lt: Vec<f32> = (0..m)
                .map(|_| {
                    if levels == 0 {
                        (rng() >> 40) as f32 / 16_777_216.0
                    } else {
                        ((rng() % levels) as f32 + 1.0) * 0.5
                    }
                })
                .collect();
            let sq = square_from_lower_tri(n, &lt);

            for &thr in &[0.4f32, 0.8, f32::INFINITY] {
                for max_dim in 1..=3 {
                    let edges = EdgeList::from_distance_matrix(&sq, n, thr).unwrap();
                    let bitcsr = BitCsrDistanceMatrix::from_edge_list(edges).unwrap();

                    let narrow = compute::<6, Simplex128>(&bitcsr, max_dim, false);
                    let wide = compute::<9, Simplex256>(&bitcsr, max_dim, false);
                    assert_eq!(
                        barcode_key(&narrow),
                        barcode_key(&wide),
                        "barcode mismatch (n={n}, levels={levels}, thr={thr}, max_dim={max_dim})"
                    );
                }
            }
        }
    }

    #[test]
    fn simplex256_deep_dims_on_octahedron() {
        let s = std::f32::consts::SQRT_2;
        let lt = [
            2.0, // (1,0)
            s, s, // (2,0), (2,1)
            s, s, 2.0, // (3,0..2)
            s, s, s, s, // (4,0..3)
            s, s, s, s, 2.0, // (5,0..4)
        ];
        let sq = square_from_lower_tri(6, &lt);
        let edges = EdgeList::from_distance_matrix(&sq, 6, f32::INFINITY).unwrap();
        let bitcsr = BitCsrDistanceMatrix::from_edge_list(edges).unwrap();
        let bc = compute::<9, Simplex256>(&bitcsr, 5, false);

        assert_eq!(bc.intervals.len(), 6);
        assert!(bc.intervals[2].iter().any(|iv| (iv.birth - s).abs() < 1e-5));
        for d in 3..=5 {
            assert!(bc.intervals[d].is_empty(), "H{d} of the octahedron");
        }
    }

    fn run(lt: &[f32], n: usize, threshold: f32, max_dim: usize) -> BarcodeResult {
        let sq = square_from_lower_tri(n, lt);
        let edges = EdgeList::from_distance_matrix(&sq, n, threshold).unwrap();
        let bitcsr = BitCsrDistanceMatrix::from_edge_list(edges).unwrap();
        compute::<6, Simplex128>(&bitcsr, max_dim, true)
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

    #[test]
    fn h0_two_points() {
        let bc = run(&[1.0], 2, f32::INFINITY, 1);
        assert_eq!(bc.intervals.len(), 2);
        assert_eq!(bc.intervals[0].len(), 2);
        let mut h0_sorted: Vec<_> = bc.intervals[0].clone();
        h0_sorted.sort_by(|a, b| a.death.partial_cmp(&b.death).unwrap());
        assert!(approx_eq(h0_sorted[0].birth, 0.0, 1e-6));
        assert!(approx_eq(h0_sorted[0].death, 1.0, 1e-6));
        assert!(h0_sorted[1].death.is_infinite());
        assert_eq!(bc.intervals[1].len(), 0);
    }

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

        assert_eq!(count_essential_h_d(&bc, 0), 1);
        let h0_finite_count = bc.intervals[0]
            .iter()
            .filter(|iv| iv.death.is_finite())
            .count();
        assert_eq!(h0_finite_count, 3);

        assert_eq!(count_essential_h_d(&bc, 1), 1);
        assert_eq!(count_finite_h_d(&bc, 1), 0);
    }

    #[test]
    fn unit_square_h0_h1() {
        let s = std::f32::consts::SQRT_2;
        let bc = run(&[1.0, s, 1.0, 1.0, s, 1.0], 4, f32::INFINITY, 2);

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

    #[test]
    fn two_components_finite_threshold() {
        let bc = run(&[1.0, 10.0, 9.0, 11.0, 10.0, 1.0], 4, 5.0, 1);
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
        assert_eq!(bc.intervals[1].len(), 0);
    }

    #[test]
    fn octahedron_h2() {
        let s = std::f32::consts::SQRT_2;
        let bc = run(
            &[2.0, s, s, s, s, 2.0, s, s, s, s, s, s, s, s, 2.0],
            6,
            f32::INFINITY,
            2,
        );

        assert_eq!(count_essential_h_d(&bc, 0), 1);
        let h0_finite = bc.intervals[0]
            .iter()
            .filter(|iv| iv.death.is_finite())
            .count();
        assert_eq!(h0_finite, 5);

        assert_eq!(
            count_essential_h_d(&bc, 1),
            0,
            "H1 should have no essential classes"
        );

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
