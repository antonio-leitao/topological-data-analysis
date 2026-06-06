// ═══════════════════════════════════════════════════════════════════════════════
// csr.rs — Sparse (CSR) distance-matrix backend
// ═══════════════════════════════════════════════════════════════════════════════
//
// A full `Filtration` backend: storage type, builder, and the cofacet / facet /
// apparent-pair enumerators. Produces barcodes bit-identical to the dense
// backend (see the differential tests at the bottom). The reduction and driver
// in `reduction.rs` / `algorithm.rs` are generic and run on it unchanged.
//
// WHY CSR (not COO)
// ─────────────────
// The sparse cofacet enumerator does not scan all n vertices; a cofacet
// σ ∪ {j} exists below threshold only if j is a within-threshold neighbour of
// EVERY vertex of σ. So enumeration is a k-way intersection of the neighbour
// lists of σ's vertices — which needs O(1), cache-contiguous, sorted row
// access. CSR gives exactly that. COO (flat (row,col,val) triples) has no row
// index, so "neighbours of v" would cost a full scan or a rebuild into CSR.
// The matrix is built once and read-only during reduction, so COO's only
// edges (cheap append, easy transpose) are irrelevant here.
//
// LAYOUT (structure-of-arrays, descending rows)
// ──────────────────────────────────────────────
// `row_ptr[v]..row_ptr[v+1]` indexes `col` and `val` for vertex v. `col` holds
// neighbour vertex ids, `val` the parallel distances. Rows are stored in
// DESCENDING id order to satisfy the `Filtration::for_each_cofacet` ordering
// contract (the merge can then emit common neighbours largest-first, matching
// the dense `CofacetIter`).
//
// SoA rather than `Vec<(u16, f32)>`: the cofacet merge advances cursors by
// comparing ids on every step but reads a distance only for a surviving common
// neighbour, so keeping `col` dense (2 bytes/entry) maximises the comparison
// stream's cache density and touches `val` only on hits.

use crate::engine::distance::DistanceMatrix;
use crate::engine::filtration::Filtration;
use crate::engine::simplex::{encode_filtration, Simplex128};

/// Entries `>= NO_EDGE` are "no edge" and are excluded from the enclosing-radius
/// reduction, mirroring `distance.rs`. Catches both `f32::INFINITY` and `f32::MAX`.
const NO_EDGE: f32 = f32::MAX;

/// Sparse distance matrix in compressed-sparse-row form.
///
/// Stores only within-threshold neighbours. Symmetric: each kept edge (i, j)
/// appears in both row i and row j.
pub struct CsrDistanceMatrix {
    n: usize,
    /// Row offsets, length `n + 1`. `usize` (not `u32`) so `nnz` up to
    /// `n(n-1)` can never overflow the offsets even at `n = u16::MAX`.
    row_ptr: Vec<usize>,
    /// Neighbour vertex ids, descending within each row.
    col: Vec<u16>,
    /// Neighbour distances, parallel to `col`.
    val: Vec<f32>,
}

impl CsrDistanceMatrix {
    /// Assemble from raw CSR parts produced by `pdist_csr`. Caller guarantees
    /// `row_ptr.len() == n + 1`, `col`/`val` parallel, and each row descending
    /// by id (checked in debug).
    pub(crate) fn from_csr_parts(
        n: usize,
        row_ptr: Vec<usize>,
        col: Vec<u16>,
        val: Vec<f32>,
    ) -> Self {
        debug_assert_eq!(row_ptr.len(), n + 1);
        debug_assert_eq!(col.len(), val.len());
        debug_assert_eq!(*row_ptr.last().unwrap_or(&0), col.len());
        debug_assert!((0..n).all(|v| col[row_ptr[v]..row_ptr[v + 1]]
            .windows(2)
            .all(|w| w[0] > w[1])));
        CsrDistanceMatrix {
            n,
            row_ptr,
            col,
            val,
        }
    }
    /// Number of vertices.
    #[inline(always)]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of stored directed neighbour entries (= 2 × kept undirected edges).
    #[inline(always)]
    pub fn nnz(&self) -> usize {
        self.col.len()
    }

    /// Neighbours of `v` as parallel `(ids, distances)` slices, descending by id.
    #[inline(always)]
    pub fn neighbors(&self, v: usize) -> (&[u16], &[f32]) {
        let s = self.row_ptr[v];
        let e = self.row_ptr[v + 1];
        (&self.col[s..e], &self.val[s..e])
    }

    // ── Hot-loop primitives ───────────────────────────────────────────────────

    /// Distance of edge {a, b}, which MUST be present (true for any pair of
    /// vertices of an existing simplex — all its sub-edges are stored). Searches
    /// the shorter of the two rows.
    ///
    /// Rows are descending, so the lookup is a `partition_point` (the SIMD seam:
    /// the descending dual of `adaptive_search`). Used only by the
    /// apparent-pair facet diameters, off the 55% cofacet path.
    #[inline]
    fn edge_dist(&self, a: u16, b: u16) -> f32 {
        let (ra, rb) = (a as usize, b as usize);
        let (host, target) =
            if self.row_ptr[ra + 1] - self.row_ptr[ra] <= self.row_ptr[rb + 1] - self.row_ptr[rb] {
                (ra, b)
            } else {
                (rb, a)
            };
        let (col, val) = self.neighbors(host);
        let pos = col.partition_point(|&x| x > target); // first id ≤ target
        debug_assert!(
            pos < col.len() && col[pos] == target,
            "edge_dist on a missing edge ({a}, {b})"
        );
        val[pos]
    }

    /// Walk the common neighbours of `verts[0..vc]` in DESCENDING id order,
    /// invoking `g(w, extra)` for each, where `extra = maxᵢ d(w, vᵢ)` — the
    /// distances at the matched positions of the parallel `val` arrays. Stops
    /// when `g` returns false, when any list is exhausted, or once `w ≤ floor`
    /// (descending ⇒ nothing larger remains; bounds the `all_cofacets = false`
    /// case to `w > σ.largest_vertex`).
    ///
    /// Dispatches on arity: `vc == 2` (cofacets of an edge — the highest-volume
    /// dimension) takes a register-resident two-cursor path; everything else
    /// takes the general k-way leapfrog. Both use the galloping `advance_le`
    /// skip and `get_unchecked` on the cursor reads.
    #[inline]
    fn for_each_common_neighbor(
        &self,
        verts: &[u16; 6],
        vc: usize,
        floor: i32,
        g: impl FnMut(u16, f32) -> bool,
    ) {
        if vc == 2 {
            self.common_2way(verts[0], verts[1], floor, g);
        } else {
            self.common_kway(verts, vc, floor, g);
        }
    }

    /// Two-cursor descending intersection (cofacets of an edge). All cursor
    /// state stays in locals; the rows are read through `get_unchecked` behind
    /// `a < la` / `b < lb` guards.
    #[inline]
    fn common_2way(&self, v0: u16, v1: u16, floor: i32, mut g: impl FnMut(u16, f32) -> bool) {
        let (ca, va) = self.neighbors(v0 as usize);
        let (cb, vb) = self.neighbors(v1 as usize);
        let (la, lb) = (ca.len(), cb.len());
        let mut a = 0usize;
        let mut b = 0usize;
        unsafe {
            while a < la && b < lb {
                let xa = *ca.get_unchecked(a);
                let xb = *cb.get_unchecked(b);
                // Any common w satisfies w ≤ min(xa, xb); if that is ≤ floor, no
                // qualifying neighbour remains.
                if (xa as i32) <= floor || (xb as i32) <= floor {
                    return;
                }
                if xa == xb {
                    let extra = (*va.get_unchecked(a)).max(*vb.get_unchecked(b));
                    if !g(xa, extra) {
                        return;
                    }
                    a += 1;
                    b += 1;
                } else if xa > xb {
                    a += advance_le(ca.get_unchecked(a..), xb);
                } else {
                    b += advance_le(cb.get_unchecked(b..), xa);
                }
            }
        }
    }

    /// General k-way leapfrog intersection (`vc ≠ 2`), descending. The shortest
    /// row drives (it bounds the candidate count); each follower skips down to
    /// the candidate via `advance_le`, and a follower that overshoots re-anchors
    /// the driver and restarts.
    #[inline]
    fn common_kway(
        &self,
        verts: &[u16; 6],
        vc: usize,
        floor: i32,
        mut g: impl FnMut(u16, f32) -> bool,
    ) {
        let mut hi = [0usize; 6];
        let mut idx = [0usize; 6];
        for i in 0..vc {
            let v = verts[i] as usize;
            let s = self.row_ptr[v];
            let e = self.row_ptr[v + 1];
            if s == e {
                return; // a vertex with no neighbours ⇒ empty intersection
            }
            idx[i] = s;
            hi[i] = e;
        }
        let mut driver = 0usize;
        for i in 1..vc {
            if (hi[i] - idx[i]) < (hi[driver] - idx[driver]) {
                driver = i;
            }
        }

        let col = &self.col;
        let val = &self.val;
        unsafe {
            loop {
                if idx[driver] >= hi[driver] {
                    return;
                }
                let cand = *col.get_unchecked(idx[driver]);
                if (cand as i32) <= floor {
                    return;
                }

                let mut all_match = true;
                for i in 0..vc {
                    if i == driver {
                        continue;
                    }
                    idx[i] += advance_le(col.get_unchecked(idx[i]..hi[i]), cand);
                    if idx[i] >= hi[i] {
                        return;
                    }
                    let head = *col.get_unchecked(idx[i]);
                    if head < cand {
                        // cand absent here; head is a new, smaller upper bound.
                        idx[driver] += advance_le(col.get_unchecked(idx[driver]..hi[driver]), head);
                        all_match = false;
                        break;
                    }
                }

                if all_match {
                    let mut extra = 0.0f32;
                    for i in 0..vc {
                        let d = *val.get_unchecked(idx[i]);
                        if d > extra {
                            extra = d;
                        }
                    }
                    if !g(cand, extra) {
                        return;
                    }
                    for i in 0..vc {
                        idx[i] += 1; // advance every row past cand
                    }
                }
            }
        }
    }

    /// Emit cofacets of `sigma` youngest-first (descending inserted vertex),
    /// each carrying its diameter, skipping any whose diameter exceeds
    /// `threshold`. The shared core behind `Filtration::for_each_cofacet` and
    /// the apparent-pair cofacet probe.
    #[inline]
    fn cofacets(
        &self,
        sigma: Simplex128,
        all_cofacets: bool,
        threshold: f32,
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

        // Rank-cached cofacet construction (port of dense CofacetIter).
        // `vi` = #sigma-verts > w = insertion rank; monotone non-decreasing.
        // Refresh masks only when it advances — ≤ vc times per simplex, not per emit.
        let payload = sigma.vertex_key();
        let mut vi = 0usize;
        let mut payload_upper: u128 = 0; // rank-0: payload & (!0<<96) == 0
        let mut payload_lower_shifted: u128 = payload >> 16;
        let mut insert_shift: u32 = 80; // 96 - 0*16 - 16

        self.for_each_common_neighbor(&verts, vc, floor, |w, extra| {
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
            if diam > threshold {
                return true;
            }
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
    /// Diameter of the facet of `verts[0..vc]` obtained by removing slot `skip`
    /// (max pairwise distance over the remaining vertices). Mirrors the dense
    /// `facet_diameter`, with `edge_dist` for the lookups.
    #[inline]
    fn facet_diameter(&self, verts: &[u16; 6], vc: usize, skip: usize) -> f32 {
        let mut diam = 0.0f32;
        for i in 0..vc {
            if i == skip {
                continue;
            }
            for j in (i + 1)..vc {
                if j == skip {
                    continue;
                }
                let d = self.edge_dist(verts[i], verts[j]);
                if d > diam {
                    diam = d;
                }
            }
        }
        diam
    }

    /// Oldest same-diameter facet of `tau` (= lex-min; first in the dense
    /// `FacetIter` order, which removes slot 0 first), if one exists.
    #[inline]
    fn zero_pivot_facet(&self, tau: Simplex128) -> Option<Simplex128> {
        let target = tau.filtration_encoded();
        let verts = tau.vertices();
        let vc = tau.vertex_count();
        for k in 0..vc {
            let diam = self.facet_diameter(&verts, vc, k);
            if encode_filtration(diam) == target {
                return Some(tau.remove_vertex_at(k, target));
            }
        }
        None
    }
    /// Youngest same-diameter cofacet of `sigma` (first in descending cofacet
    /// order), if one exists within `threshold`.
    #[inline]
    fn zero_pivot_cofacet(&self, sigma: Simplex128, threshold: f32) -> Option<Simplex128> {
        let same_diam = sigma.filtration();
        if same_diam > threshold {
            return None;
        }
        let mut result = None;
        self.cofacets(sigma, true, same_diam, |cof| {
            debug_assert_eq!(cof.filtration_encoded(), sigma.filtration_encoded());
            result = Some(cof);
            false
        });
        result
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Builder — the sparse analogue of `pdist`: distance matrix → (CSR, R_cheb)
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a CSR sparse matrix from a dense distance matrix, and return the
/// minimax / Chebyshev enclosing radius `R_cheb = min_i max_j d(i, j)`.
///
/// Edges with `d ≤ min(threshold, R_cheb)` are kept; everything else is dropped
/// from storage (this is the build-time analogue of the dense path's runtime
/// threshold early-exit). `R_cheb` is computed over *all* pairs — including
/// dropped ones — exactly as `DistanceMatrix::from_square_matrix` computes its
/// `minimax`, so the caller resolves the engine threshold identically:
/// `min(user_threshold, R_cheb)`.
///
/// Cost: three sequential sweeps over the `n(n-1)/2` condensed entries
/// (eccentricities, degrees, scatter) plus an `O(nnz)` per-row reverse —
/// memory-bandwidth-bound, no `n²` auxiliary allocation, no comparison sort.
pub fn csr_from_distance_matrix(dist: &DistanceMatrix, threshold: f32) -> (CsrDistanceMatrix, f32) {
    let n = dist.n();
    if n < 2 {
        return (
            CsrDistanceMatrix {
                n,
                row_ptr: vec![0; n + 1],
                col: Vec::new(),
                val: Vec::new(),
            },
            f32::INFINITY,
        );
    }
    let raw = dist.raw(); // condensed lower-triangular; entry (i,j), i>j, at i*(i-1)/2 + j

    // ── Sweep A: eccentricities → R_cheb. Max over ALL pairs (threshold-free),
    //    excluding NO_EDGE so disconnected pairs don't poison the radius. ──────
    let mut ecc = vec![f32::NEG_INFINITY; n];
    {
        let mut p = 0;
        for i in 1..n {
            let mut row_max = ecc[i];
            for j in 0..i {
                let d = raw[p];
                p += 1;
                if d < NO_EDGE {
                    if d > row_max {
                        row_max = d;
                    }
                    if d > ecc[j] {
                        ecc[j] = d;
                    }
                }
            }
            ecc[i] = row_max;
        }
    }
    let r_cheb = ecc
        .iter()
        .copied()
        .filter(|&m| m > f32::NEG_INFINITY)
        .fold(f32::INFINITY, f32::min);

    let eff = threshold.min(r_cheb);

    // ── Sweep B: degree of each vertex at `eff` (both endpoints), prefix-sum
    //    into row_ptr. ─────────────────────────────────────────────────────────
    let mut row_ptr = vec![0usize; n + 1];
    {
        let mut p = 0;
        for i in 1..n {
            for j in 0..i {
                let d = raw[p];
                p += 1;
                if d <= eff && d < NO_EDGE {
                    row_ptr[i + 1] += 1;
                    row_ptr[j + 1] += 1;
                }
            }
        }
        for v in 0..n {
            row_ptr[v + 1] += row_ptr[v];
        }
    }
    let nnz = row_ptr[n];
    let mut col = vec![0u16; nnz];
    let mut val = vec![0.0f32; nnz];

    // ── Sweep C: scatter both directions using per-row write cursors. ─────────
    // A linear sweep appends, for each vertex v, first its j<v neighbours
    // (ascending, during row i=v) then its i>v neighbours (ascending, during
    // rows i>v). See Sweep D for why that exact order matters.
    {
        let mut cur = row_ptr[..n].to_vec();
        let mut p = 0;
        for i in 1..n {
            for j in 0..i {
                let d = raw[p];
                p += 1;
                if d <= eff && d < NO_EDGE {
                    let ci = cur[i];
                    col[ci] = j as u16;
                    val[ci] = d;
                    cur[i] = ci + 1;

                    let cj = cur[j];
                    col[cj] = i as u16;
                    val[cj] = d;
                    cur[j] = cj + 1;
                }
            }
        }
    }

    // ── Sweep D: reverse each row to DESCENDING id order. ─────────────────────
    // As scattered, row v = [j<v ascending] ++ [i>v ascending]; since every j<v
    // is below every i>v, reversing the whole slice yields a strictly
    // descending sequence — the required order, with no comparison sort.
    for v in 0..n {
        let s = row_ptr[v];
        let e = row_ptr[v + 1];
        col[s..e].reverse();
        val[s..e].reverse();
    }

    (
        CsrDistanceMatrix {
            n,
            row_ptr,
            col,
            val,
        },
        r_cheb,
    )
}

/// First index `i` of a DESCENDING `u16` slice with `col[i] <= target`, or
/// `col.len()` if every element exceeds `target`. A short linear probe (cheap
/// for the small skips typical of a high-overlap merge) backed by a galloping
/// binary search (bounded for the large skips typical of sparse rows).
///
/// This is the descending dual of the ascending `adaptive_search` from the
/// intersection crate, and the single seam where a SIMD skip drops in: widen to
/// `u16` lanes and flip the comparison to `<=`.
#[inline(always)]
fn advance_le(col: &[u16], target: u16) -> usize {
    let len = col.len();
    let probe = len.min(4);
    let mut i = 0;
    while i < probe {
        // SAFETY: i < probe ≤ len.
        if unsafe { *col.get_unchecked(i) } <= target {
            return i;
        }
        i += 1;
    }
    if probe == len {
        return len;
    }
    // col[0..4] all > target; gallop for a bound with col[bound] ≤ target.
    let mut bound = 4usize;
    // SAFETY: bound < len on each read.
    while bound < len && unsafe { *col.get_unchecked(bound) } > target {
        bound <<= 1;
    }
    let lo = bound >> 1;
    let hi = bound.min(len);
    lo + col[lo..hi].partition_point(|&x| x > target)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Filtration impl — the sparse backend
// ═══════════════════════════════════════════════════════════════════════════════
//
// Each method has the same contract as the dense impl in `filtration.rs` (see
// the trait docs and the ordering contract there). `for_each_cofacet` is the
// 55% path and delegates to the merge primitives above (2-way fast path /
// k-way leapfrog). The apparent-pair trio reuses the same cofacet probe plus a
// sparse facet enumerator over τ's own vertices.

impl Filtration for CsrDistanceMatrix {
    #[inline(always)]
    fn n(&self) -> usize {
        self.n
    }

    #[inline]
    fn for_each_edge(&self, threshold: f32, mut f: impl FnMut(Simplex128) -> bool) {
        // Each undirected edge {w, v}, w > v, lives in row v as a neighbour
        // w > v — and rows are descending, so those sit at the front. Emit it
        // once, when v is the smaller endpoint, then stop the row at w ≤ v.
        for v in 0..self.n {
            let (col, val) = self.neighbors(v);
            for k in 0..col.len() {
                let w = col[k];
                if (w as usize) <= v {
                    break;
                }
                let d = val[k];
                if d <= threshold && !f(Simplex128::from_sorted_desc(d, &[w, v as u16])) {
                    return;
                }
            }
        }
    }

    #[inline]
    fn for_each_cofacet(
        &self,
        sigma: Simplex128,
        all_cofacets: bool,
        threshold: f32,
        f: impl FnMut(Simplex128) -> bool,
    ) {
        self.cofacets(sigma, all_cofacets, threshold, f);
    }

    #[inline]
    fn zero_apparent_facet(&self, tau: Simplex128, threshold: f32) -> Option<Simplex128> {
        let phi = self.zero_pivot_facet(tau)?;
        let tau_check = self.zero_pivot_cofacet(phi, threshold)?;
        if tau_check == tau {
            Some(phi)
        } else {
            None
        }
    }

    #[inline]
    fn zero_apparent_cofacet(&self, sigma: Simplex128, threshold: f32) -> Option<Simplex128> {
        let tau = self.zero_pivot_cofacet(sigma, threshold)?;
        let phi = self.zero_pivot_facet(tau)?;
        if phi == sigma {
            Some(tau)
        } else {
            None
        }
    }

    #[inline]
    fn is_apparent_cofacet(&self, tau: Simplex128, threshold: f32) -> bool {
        let sigma = match self.zero_pivot_facet(tau) {
            Some(s) => s,
            None => return false,
        };
        match self.zero_pivot_cofacet(sigma, threshold) {
            Some(t) => t == tau,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::distance::DistanceMatrix;

    // Brute-force reference: descending neighbour list of each vertex at `eff`.
    fn ref_neighbors(dist: &DistanceMatrix, eff: f32) -> Vec<Vec<u16>> {
        let n = dist.n();
        (0..n)
            .map(|v| {
                let mut ns: Vec<u16> = (0..n)
                    .filter(|&u| u != v)
                    .filter(|&u| {
                        let d = dist.get(u, v);
                        d <= eff && d < NO_EDGE
                    })
                    .map(|u| u as u16)
                    .collect();
                ns.sort_unstable_by(|a, b| b.cmp(a)); // descending
                ns
            })
            .collect()
    }

    fn ref_r_cheb(dist: &DistanceMatrix) -> f32 {
        let n = dist.n();
        (0..n)
            .map(|i| {
                (0..n)
                    .filter(|&j| j != i)
                    .map(|j| dist.get(i, j))
                    .filter(|&d| d < NO_EDGE)
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .filter(|&m| m > f32::NEG_INFINITY)
            .fold(f32::INFINITY, f32::min)
    }

    fn check(dist: &DistanceMatrix, threshold: f32) {
        let (csr, r_cheb) = csr_from_distance_matrix(dist, threshold);
        let r_ref = ref_r_cheb(dist);
        assert!((r_cheb - r_ref).abs() < 1e-6, "R_cheb {r_cheb} vs {r_ref}");

        let eff = threshold.min(r_cheb);
        let want = ref_neighbors(dist, eff);
        for v in 0..dist.n() {
            let (col, val) = csr.neighbors(v);
            // ids match the reference exactly, in descending order
            assert_eq!(col, &want[v][..], "row {v} ids");
            // values are the actual distances, parallel to ids
            for (k, &u) in col.iter().enumerate() {
                assert!((val[k] - dist.get(u as usize, v)).abs() < 1e-6);
            }
            // strictly descending
            assert!(
                col.windows(2).all(|w| w[0] > w[1]),
                "row {v} not descending"
            );
        }
        // symmetry: u in row v  <=>  v in row u
        for v in 0..dist.n() {
            for &u in csr.neighbors(v).0 {
                assert!(csr.neighbors(u as usize).0.contains(&(v as u16)));
            }
        }
        assert_eq!(csr.nnz(), want.iter().map(|r| r.len()).sum::<usize>());
    }

    #[test]
    fn builder_matches_reference_random() {
        let mut s: u64 = 0x1234_5678_9abc_def0;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 16_777_216.0
        };
        for &n in &[2usize, 3, 5, 9, 16, 33] {
            let m = n * (n - 1) / 2;
            let data: Vec<f32> = (0..m).map(|_| rng()).collect();
            let dist = DistanceMatrix::from_lower_triangular(n, data);
            for &thr in &[0.0f32, 0.2, 0.5, 0.8, f32::INFINITY] {
                check(&dist, thr);
            }
        }
    }

    #[test]
    fn builder_octahedron_descending_and_radius() {
        let s = std::f32::consts::SQRT_2;
        let dist = DistanceMatrix::from_lower_triangular(
            6,
            vec![2.0, s, s, s, s, 2.0, s, s, s, s, s, s, s, s, 2.0],
        );
        // every vertex has eccentricity 2.0, so R_cheb = 2.0
        let (_csr, r) = csr_from_distance_matrix(&dist, f32::INFINITY);
        assert!((r - 2.0).abs() < 1e-6);
        check(&dist, f32::INFINITY);
        check(&dist, s); // drop the antipodal (=2.0) edges
    }

    // ── Differential gate: sparse engine ≡ dense engine ──────────────────────
    //
    // Build the CSR at threshold T, run BOTH backends at the resolved effective
    // threshold eff = min(T, R_cheb) so they see the same edge set, and require
    // identical barcodes. Run on tie-free AND heavily-tied inputs — ties are
    // where a wrong cofacet/facet emission order would silently corrupt
    // apparent-pair detection.

    use crate::engine::algorithm::compute;
    use crate::types::BarcodeResult;

    fn barcodes_equal(a: &BarcodeResult, b: &BarcodeResult) -> bool {
        if a.intervals.len() != b.intervals.len() {
            return false;
        }
        let key = |p: &crate::types::PersistenceInterval| {
            // sortable key; +inf maps to a large finite sentinel
            let f = |x: f32| if x.is_finite() { x } else { 1.0e30 };
            (f(p.birth), f(p.death))
        };
        for (da, db) in a.intervals.iter().zip(b.intervals.iter()) {
            if da.len() != db.len() {
                return false;
            }
            let mut va: Vec<_> = da.iter().map(key).collect();
            let mut vb: Vec<_> = db.iter().map(key).collect();
            va.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
            vb.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
            for (x, y) in va.iter().zip(vb.iter()) {
                if (x.0 - y.0).abs() > 1e-5 || (x.1 - y.1).abs() > 1e-5 {
                    return false;
                }
            }
        }
        true
    }

    fn diff_check(dist: &DistanceMatrix, threshold: f32, max_dim: usize) {
        let (csr, r_cheb) = csr_from_distance_matrix(dist, threshold);
        let eff = threshold.min(r_cheb);
        let dense = compute(dist, eff, max_dim);
        let sparse = compute(&csr, eff, max_dim);
        assert!(
            barcodes_equal(&dense, &sparse),
            "barcode mismatch (n={}, eff={eff}, max_dim={max_dim})\n dense={:?}\nsparse={:?}",
            dist.n(),
            dense.intervals,
            sparse.intervals,
        );
    }

    fn xorshift(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    #[test]
    fn sparse_matches_dense_no_ties() {
        // Continuous distances ⇒ distinct diameters (no ties).
        for &(n, seed) in &[(5usize, 1u64), (8, 7), (12, 99), (20, 31337), (30, 424242)] {
            let mut rng = xorshift(seed);
            let m = n * (n - 1) / 2;
            let data: Vec<f32> = (0..m)
                .map(|_| (rng() >> 40) as f32 / 16_777_216.0)
                .collect();
            let dist = DistanceMatrix::from_lower_triangular(n, data);
            for &thr in &[0.3f32, 0.6, 0.9, f32::INFINITY] {
                for max_dim in 1..=3 {
                    diff_check(&dist, thr, max_dim);
                }
            }
        }
    }

    #[test]
    fn sparse_matches_dense_with_ties() {
        // Quantised distances ⇒ many equal diameters: the adversarial case for
        // apparent-pair ordering.
        for &(n, levels, seed) in &[(6usize, 2u64, 5), (8, 3, 17), (12, 3, 2024), (18, 4, 88)] {
            let mut rng = xorshift(seed);
            let m = n * (n - 1) / 2;
            let data: Vec<f32> = (0..m)
                .map(|_| ((rng() % levels) as f32 + 1.0) * 0.5) // {0.5, 1.0, ...}
                .collect();
            let dist = DistanceMatrix::from_lower_triangular(n, data);
            for &thr in &[0.5f32, 1.0, 1.5, f32::INFINITY] {
                for max_dim in 1..=3 {
                    diff_check(&dist, thr, max_dim);
                }
            }
        }
    }

    #[test]
    fn sparse_matches_dense_octahedron() {
        // Geometric input with exact ties (all face edges = √2, antipodes = 2).
        let s = std::f32::consts::SQRT_2;
        let dist = DistanceMatrix::from_lower_triangular(
            6,
            vec![2.0, s, s, s, s, 2.0, s, s, s, s, s, s, s, s, 2.0],
        );
        for &thr in &[s, 1.9, 2.0, f32::INFINITY] {
            for max_dim in 1..=3 {
                diff_check(&dist, thr, max_dim);
            }
        }
    }

    // Still confirm the generic instantiates for the CSR backend end-to-end
    // (it now actually runs instead of hitting a placeholder).
    #[test]
    fn csr_backend_runs_end_to_end() {
        let dist = DistanceMatrix::from_lower_triangular(4, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let (csr, r) = csr_from_distance_matrix(&dist, f32::INFINITY);
        let res = compute(&csr, r, 2);
        assert_eq!(res.intervals.len(), 3); // dims 0,1,2 present
    }

    // Direct guard on the apparent-pair trio: compare the sparse methods against
    // the dense free functions on every edge and triangle, returned Simplex128
    // included. A false-positive here is a correctness bug that the end-to-end
    // barcode test could in principle mask, so we check it head-on, on tie-heavy
    // inputs where the same-diameter logic is exercised hardest.
    #[test]
    fn apparent_pairs_match_dense() {
        use crate::engine::filtration::Filtration;
        use crate::engine::simplex::{
            is_apparent_cofacet as dense_is_app, zero_apparent_cofacet as dense_zac,
            zero_apparent_facet as dense_zaf, Simplex128,
        };
        for &(n, levels, seed) in &[(7usize, 3u64, 11), (9, 4, 77), (11, 2, 909), (8, 5, 1234)] {
            let mut rng = xorshift(seed);
            let m = n * (n - 1) / 2;
            let data: Vec<f32> = (0..m)
                .map(|_| ((rng() % levels) as f32 + 1.0) * 0.5)
                .collect();
            let dist = DistanceMatrix::from_lower_triangular(n, data);
            let (csr, eff) = csr_from_distance_matrix(&dist, f32::INFINITY);

            // edges → zero_apparent_cofacet
            for i in 1..n {
                for j in 0..i {
                    let d = dist.get(i, j);
                    if d > eff {
                        continue;
                    }
                    let e = Simplex128::new(d, &[i as u16, j as u16]);
                    assert_eq!(
                        dense_zac(e, &dist, eff),
                        csr.zero_apparent_cofacet(e, eff),
                        "zero_apparent_cofacet edge ({i},{j})"
                    );
                }
            }
            // triangles → is_apparent_cofacet + zero_apparent_facet
            for i in 2..n {
                for j in 1..i {
                    for k in 0..j {
                        let diam = dist.get(i, j).max(dist.get(i, k)).max(dist.get(j, k));
                        if diam > eff {
                            continue;
                        }
                        let t = Simplex128::new(diam, &[i as u16, j as u16, k as u16]);
                        assert_eq!(
                            dense_is_app(t, &dist, eff),
                            csr.is_apparent_cofacet(t, eff),
                            "is_apparent_cofacet tri ({i},{j},{k})"
                        );
                        assert_eq!(
                            dense_zaf(t, &dist, eff),
                            csr.zero_apparent_facet(t, eff),
                            "zero_apparent_facet tri ({i},{j},{k})"
                        );
                    }
                }
            }
        }
    }
}
