use std::collections::HashMap;

use crate::engine::distance::DistanceMatrix;
use crate::engine::reduction::compute_pairs;
use crate::engine::simplex::{
    is_apparent_cofacet, zero_apparent_cofacet, CofacetIter, FxBuildHasher, FxHashMap, Simplex128,
};
use crate::types::{BarcodeResult, PersistenceInterval};

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

    #[inline]
    fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else {
            self.parent[rb] = ra;
            if self.rank[ra] == self.rank[rb] {
                self.rank[ra] += 1;
            }
        }
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// H0: enumerate edges, Kruskal, split into simplices + columns_to_reduce
// ═══════════════════════════════════════════════════════════════════════════════

fn compute_h0(
    dist: &DistanceMatrix,
    threshold: f32,
    intervals: &mut Vec<PersistenceInterval>,
) -> (Vec<Simplex128>, Vec<Simplex128>) {
    let n = dist.n();

    let mut edges: Vec<Simplex128> = Vec::new();
    for i in 1..n {
        for j in 0..i {
            let d = dist.get(i, j);
            if d <= threshold {
                edges.push(Simplex128::from_sorted_desc(d, &[i as u16, j as u16]));
            }
        }
    }

    // Kruskal needs SMALLEST DIAMETER FIRST (= oldest-first). Under the
    // Ripser-compatible Ord, oldest is "greater", so descending sort gives
    // oldest-first iteration here.
    edges.sort_unstable_by(|a, b| b.cmp(a));

    let mut uf = UnionFind::new(n);
    let mut columns_to_reduce = Vec::with_capacity(edges.len());

    let mut num_merges: usize = 0;
    for &edge in &edges {
        let verts = edge.vertices();
        if uf.union(verts[0] as usize, verts[1] as usize) {
            num_merges += 1;
            // ZERO-PERSISTENCE FILTER: edges with diam=0 (duplicate points)
            // produce H0 pairs (0, 0) which Ripser excludes from output by
            // default. Skip them here too to match.
            let death = edge.filtration();
            if death > 0.0 {
                intervals.push(PersistenceInterval { birth: 0.0, death });
            }
        } else {
            // Edge did NOT merge components — it closes a 1-cycle candidate.
            // Filter: if the edge is already the cofacet side of an apparent pair
            // with one of its endpoints, it's guaranteed to be paired in the H1
            // reduction and we can skip it here (Ripser, compute_dim_0_pairs).
            if zero_apparent_cofacet(edge, dist, threshold).is_none() {
                columns_to_reduce.push(edge);
            }
        }
    }

    // Number of connected components at threshold = n - num_merges. This is
    // always ≥ 1 (the whole space is at least one component). Each component
    // contributes one essential H0 class.
    let num_essential = n - num_merges;
    for _ in 0..num_essential {
        intervals.push(PersistenceInterval {
            birth: 0.0,
            death: f32::INFINITY,
        });
    }

    columns_to_reduce.reverse();
    (edges, columns_to_reduce)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Assemble columns_to_reduce for the next dimension
// ═══════════════════════════════════════════════════════════════════════════════

/// Build the next dimension's column list from `simplices` (the d-simplices
/// surviving from the previous dimension). For each cofacet τ enumerated:
///   1. Push into `next_simplices` (becomes the d+1-simplex pool for the
///      dimension after this).
///   2. Add to `columns_to_reduce` UNLESS τ is filtered out by either:
///      - clearing: τ is already a known pivot from the previous dimension's
///        reduction (its column is therefore zero by construction);
///      - apparent-pair filtering: τ is already the cofacet side of an
///        apparent pair (paired with its oldest same-diam facet).
fn assemble_candidates(
    simplices: &mut Vec<Simplex128>,
    dist: &DistanceMatrix,
    threshold: f32,
    cleared_pivots: &FxHashMap<Simplex128, ()>,
) -> Vec<Simplex128> {
    if simplices.is_empty() {
        return Vec::new();
    }

    let mut next_simplices: Vec<Simplex128> = Vec::with_capacity(simplices.len() * 2);
    let mut columns_to_reduce: Vec<Simplex128> = Vec::with_capacity(simplices.len());

    let mut iter = CofacetIter::new(simplices[0], dist, false, threshold);
    for (i, &sigma) in simplices.iter().enumerate() {
        if i > 0 {
            iter.reset(sigma, false);
        }
        iter.for_each(|tau| {
            next_simplices.push(tau);
            if !cleared_pivots.contains_key(&tau) && !is_apparent_cofacet(tau, dist, threshold) {
                columns_to_reduce.push(tau);
            }
            true
        });
    }

    columns_to_reduce.sort_unstable();
    *simplices = next_simplices;
    columns_to_reduce
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main loop
// ═══════════════════════════════════════════════════════════════════════════════

pub fn compute(dist: &DistanceMatrix, threshold: f32, max_dim: usize) -> BarcodeResult {
    let mut intervals: Vec<Vec<PersistenceInterval>> = Vec::with_capacity(max_dim + 1);

    // H0: union-find. No clearing input needed (no previous dim).
    let mut h0 = Vec::new();
    let (mut simplices, mut columns_to_reduce) = compute_h0(dist, threshold, &mut h0);
    intervals.push(h0);

    // Cleared-pivot set for the clearing optimization. Populated by
    // compute_pairs at dim d, consumed by assemble_candidates at dim d+1.
    let mut cleared_pivots: FxHashMap<Simplex128, ()> =
        HashMap::with_hasher(FxBuildHasher::default());

    // H1 .. H_{max_dim}
    for dim in 1..=max_dim {
        let mut dim_intervals = Vec::new();

        // Reduce this dimension's columns. Clear cleared_pivots first so it
        // captures only pivots from THIS dimension (used by assemble_candidates
        // for dim+1).
        cleared_pivots.clear();
        compute_pairs(
            &columns_to_reduce,
            dist,
            threshold,
            &mut dim_intervals,
            &mut cleared_pivots,
        );
        intervals.push(dim_intervals);

        if dim < max_dim {
            columns_to_reduce =
                assemble_candidates(&mut simplices, dist, threshold, &cleared_pivots);
        }
    }

    BarcodeResult { intervals }
}

// ═══════════════════════════════════════════════════════════════════════════════
// End-to-end correctness tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

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
        let dist = DistanceMatrix::from_lower_triangular(2, vec![1.0]);
        let bc = compute(&dist, f32::INFINITY, 1);
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

    /// Unit square (4 points). Vertices: (0,0), (1,0), (1,1), (0,1).
    /// Edges: 4 sides at length 1, 2 diagonals at length √2.
    /// Expected barcode (Vietoris–Rips):
    ///   H0: 4 components → 3 finite [0,1) (when sides connect them) + 1 essential
    ///   H1: 1 finite cycle [1, √2) (4 sides form a square cycle, dies when triangles fill in)
    ///   H2: empty (3-simplices fill in at √2 too, killing nothing in dim 2 above the empty cycle)
    /// Triangle perimeter at threshold below the diameter that fills it in.
    /// 3 points equilateral side 1, threshold 1.0: edges form, but the triangle
    /// (which has diameter 1.0) is also at the threshold.
    /// Use threshold 1.0 (inclusive in our pipeline). All 3 edges + the triangle
    /// form. So H1 should be 0.
    /// Use threshold strictly less than triangle diameter to keep the cycle alive:
    /// can't easily — equilateral has all distances equal.
    ///
    /// Better: 3 vertices with d(0,1)=1, d(1,2)=1, d(0,2)=2. At threshold 1.5,
    /// edges (1,0) and (2,1) form but edge (2,0) doesn't → no triangle, no cycle.
    /// At threshold 2.0, (2,0) joins and the triangle (2,1,0) forms with diam 2.
    /// At threshold 1.999, (2,0) doesn't form — the path 0-1-2 exists but no cycle.
    ///
    /// To get an actual H1 cycle we need ≥ 4 points. Use a path graph that closes:
    /// 4 points forming a 4-cycle, no diagonal close enough to fill it.
    /// d(1,0)=d(2,1)=d(3,2)=d(3,0)=1, diagonals d(2,0)=d(3,1)=10.
    /// Threshold 1.5: 4-cycle alive, no triangles. Should give 1 essential H1.
    #[test]
    fn unfilled_4cycle_essential_h1() {
        let dist = DistanceMatrix::from_lower_triangular(
            4,
            vec![
                1.0, // (1,0)
                10.0, 1.0, // (2,0), (2,1)
                1.0, 10.0, 1.0, // (3,0), (3,1), (3,2)
            ],
        );
        let bc = compute(&dist, 1.5, 1);

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
        let dist = DistanceMatrix::from_lower_triangular(4, vec![1.0, s, 1.0, 1.0, s, 1.0]);
        let bc = compute(&dist, f32::INFINITY, 2);

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

    /// Two disconnected segments (4 points: 0-1 close, 2-3 close, far between groups).
    /// Vertices in 1D: 0.0, 1.0, 10.0, 11.0.
    /// Expected: H0 has 2 essentials (one per component) when threshold < 9,
    /// 4 components reduce to 2 → 2 finite [0, 1).
    #[test]
    fn two_components_finite_threshold() {
        // d(1,0)=1, d(2,0)=10, d(2,1)=9, d(3,0)=11, d(3,1)=10, d(3,2)=1
        let dist = DistanceMatrix::from_lower_triangular(4, vec![1.0, 10.0, 9.0, 11.0, 10.0, 1.0]);
        let bc = compute(&dist, 5.0, 1);
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

    /// Octahedron-like: 6 points in 3D forming an octahedron. Expected H1 = 0,
    /// expected H2 = 1 (the octahedral surface is a 2-sphere).
    /// Use coordinates: (±1, 0, 0), (0, ±1, 0), (0, 0, ±1).
    #[test]
    fn octahedron_h2() {
        // Indices: 0=(+1,0,0), 1=(-1,0,0), 2=(0,+1,0), 3=(0,-1,0), 4=(0,0,+1), 5=(0,0,-1)
        // Distances:
        //   antipodal pairs (0,1), (2,3), (4,5) → 2.0
        //   adjacent pairs (everything else) → √2
        let s = std::f32::consts::SQRT_2;
        // Lower triangular order: (1,0), (2,0), (2,1), (3,0), (3,1), (3,2),
        //                         (4,0), (4,1), (4,2), (4,3),
        //                         (5,0), (5,1), (5,2), (5,3), (5,4)
        let dist = DistanceMatrix::from_lower_triangular(
            6,
            vec![
                2.0, // (1,0)  antipodal
                s, s, // (2,0), (2,1)  adjacent
                s, s, 2.0, // (3,0), (3,1), (3,2)  — (3,2) antipodal
                s, s, s, s, // (4,0..3) all adjacent
                s, s, s, s, 2.0, // (5,0..3) adjacent, (5,4) antipodal
            ],
        );

        let bc = compute(&dist, f32::INFINITY, 2);

        // H0: 1 essential, 5 finite (all dying at √2 when octahedron connects).
        assert_eq!(count_essential_h_d(&bc, 0), 1);
        let h0_finite = bc.intervals[0]
            .iter()
            .filter(|iv| iv.death.is_finite())
            .count();
        assert_eq!(h0_finite, 5);

        // H1: should be 0 essentials (the octahedron skeleton is connected and
        // the 1-skeleton is filled-in such that all 1-cycles die).
        // Specifically, at threshold √2 all edges except antipodals exist; the
        // 1-skeleton has many cycles but they all bound triangles (since each
        // triangle has all-√2 edges). So H1 should reduce to nothing essential.
        assert_eq!(
            count_essential_h_d(&bc, 1),
            0,
            "H1 should have no essential classes"
        );

        // H2: exactly 1 essential class — the 2-sphere.
        // The octahedral surface is closed but not a boundary (until 3-simplices
        // form, which happens only at diam 2.0 when antipodal edges enter).
        // So we expect 1 H2 interval [√2, 2.0) (the 2-sphere born at √2,
        // killed when the octahedron fills with antipodal edges and becomes a ball).
        // Actually — wait. With all antipodal edges at 2.0, at diam exactly 2.0
        // many triangles and tetrahedra appear. Let me just check there's at least
        // one H2 finite interval starting at √2.
        let h2_count = bc.intervals[2].len();
        assert!(
            h2_count >= 1,
            "H2 should have at least one interval, got {}",
            h2_count
        );
        // At least one H2 interval should be born at √2.
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
