// ═══════════════════════════════════════════════════════════════════════════════
// filtration.rs — the backend abstraction the reduction is generic over
// ═══════════════════════════════════════════════════════════════════════════════
//
// The reduction (`reduction::compute_pairs`) and the dimension driver
// (`algorithm::compute`) touch the distance representation in exactly three
// ways: enumerating the 1-simplices for H0, enumerating cofacets of a simplex
// (the 55%-of-runtime loop), and answering apparent-pair queries. Everything
// else — the heap, the V matrix, the pivot map, clearing, interval emission —
// is representation-agnostic.
//
// `Filtration` is that three-way seam. The engine is generic over `F: Filtration`
// and is monomorphized per backend, so each instantiation inlines its enumerator
// exactly as if the reduction had been hand-written for that representation. No
// vtable, no indirect call on the hot path: the cost of the abstraction is zero.
//
// IMPORTANT — this trait is deliberately NOT object-safe (`for_each_*` take
// `impl FnMut`). That is the point: it can only be used through static dispatch,
// which is the only form that preserves performance. Do not add `dyn Filtration`.
//
// ORDERING CONTRACT (read before implementing a new backend)
// ──────────────────────────────────────────────────────────
// `for_each_cofacet` MUST yield cofacets youngest-first, i.e. in DESCENDING
// inserted-vertex order (the reverse-colex order the dense `CofacetIter`
// produces by counting the candidate vertex down from n−1). The apparent-pair
// primitives depend on "first same-diameter cofacet = lex-max same-diameter
// cofacet"; a backend that yields in the wrong order stays memory-safe but
// silently computes wrong barcodes on inputs with diameter ties.

use crate::engine::simplex::Simplex128;

/// A Vietoris–Rips filtration backend: the source of edges, cofacets, and
/// apparent-pair facts that the reduction consumes.
///
/// All methods take `threshold` per call rather than baking it into the type,
/// matching the dense path where the threshold is a runtime parameter to the
/// engine and the cofacet diameter early-exits against it.
pub trait Filtration {
    /// Number of vertices (0-simplices) in the complex.
    fn n(&self) -> usize;

    /// Enumerate every 1-simplex (edge) with diameter ≤ `threshold`, each
    /// exactly once, with its diameter encoded into the returned `Simplex128`.
    /// Used to seed H0. Return `false` from `f` to stop early.
    fn for_each_edge(&self, threshold: f32, f: impl FnMut(Simplex128) -> bool);

    /// Enumerate the cofacets of `sigma` with diameter ≤ `threshold`,
    /// youngest-first (see the ordering contract above), each carrying its
    /// cofacet diameter. Return `false` from `f` to stop early.
    ///
    /// - `all_cofacets = true`  → every cofacet (reduction / coboundary).
    /// - `all_cofacets = false` → only cofacets whose inserted vertex exceeds
    ///   σ's largest vertex, so each (d+1)-simplex is produced once when
    ///   assembling the next dimension's column pool.
    fn for_each_cofacet(
        &self,
        sigma: Simplex128,
        all_cofacets: bool,
        threshold: f32,
        f: impl FnMut(Simplex128) -> bool,
    );

    /// If `tau` is the cofacet side of a zero-persistence apparent pair, return
    /// its facet partner φ (the oldest same-diameter facet). Used as a
    /// reduction shortcut.
    fn zero_apparent_facet(&self, tau: Simplex128, threshold: f32) -> Option<Simplex128>;

    /// If `sigma` is the facet side of a zero-persistence apparent pair, return
    /// its cofacet partner τ. Used by H0 to skip already-paired edges.
    fn zero_apparent_cofacet(&self, sigma: Simplex128, threshold: f32) -> Option<Simplex128>;

    /// True iff `tau` is the cofacet side of some zero-persistence apparent
    /// pair. The per-cofacet filter in next-dimension assembly.
    fn is_apparent_cofacet(&self, tau: Simplex128, threshold: f32) -> bool;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dense backend — thin delegation to the existing, untouched dense machinery
// ═══════════════════════════════════════════════════════════════════════════════
//
// Every method forwards to code that already exists and is already tuned:
// `CofacetIter` for cofacets, the free `zero_apparent_*` / `is_apparent_cofacet`
// functions for apparent pairs, and the original all-pairs loop for edges. The
// dense hot path is therefore unchanged — `compute::<DistanceMatrix>` produces
// the same inlined enumerator as before, now reached through a trait the
// compiler erases at monomorphization.

use crate::engine::distance::DistanceMatrix;
use crate::engine::simplex::{
    is_apparent_cofacet, zero_apparent_cofacet, zero_apparent_facet, CofacetIter,
};

impl Filtration for DistanceMatrix {
    #[inline(always)]
    fn n(&self) -> usize {
        // Inherent `DistanceMatrix::n` wins over the trait method here, so this
        // is a direct field read, not a recursive trait call.
        DistanceMatrix::n(self)
    }

    #[inline(always)]
    fn for_each_edge(&self, threshold: f32, mut f: impl FnMut(Simplex128) -> bool) {
        let n = DistanceMatrix::n(self);
        for i in 1..n {
            for j in 0..i {
                let d = self.get(i, j);
                if d <= threshold && !f(Simplex128::from_sorted_desc(d, &[i as u16, j as u16])) {
                    return;
                }
            }
        }
    }

    #[inline(always)]
    fn for_each_cofacet(
        &self,
        sigma: Simplex128,
        all_cofacets: bool,
        threshold: f32,
        f: impl FnMut(Simplex128) -> bool,
    ) {
        // Constructing-then-`for_each` does exactly the work the old `reset`
        // reuse did (the zero-init in `new` is dead and elided since `reset`
        // overwrites every field), so dropping the threaded `&mut CofacetIter`
        // costs nothing.
        let mut iter = CofacetIter::new(sigma, self, all_cofacets, threshold);
        iter.for_each(f);
    }

    #[inline(always)]
    fn zero_apparent_facet(&self, tau: Simplex128, threshold: f32) -> Option<Simplex128> {
        zero_apparent_facet(tau, self, threshold)
    }

    #[inline(always)]
    fn zero_apparent_cofacet(&self, sigma: Simplex128, threshold: f32) -> Option<Simplex128> {
        zero_apparent_cofacet(sigma, self, threshold)
    }

    #[inline(always)]
    fn is_apparent_cofacet(&self, tau: Simplex128, threshold: f32) -> bool {
        is_apparent_cofacet(tau, self, threshold)
    }
}
