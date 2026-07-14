// ═══════════════════════════════════════════════════════════════════════════════
// reduction.rs — Implicit cohomological matrix reduction over Z/2
// ═══════════════════════════════════════════════════════════════════════════════

use crate::engine::algorithm::{for_each_cofacet, zero_apparent_facet};
use crate::engine::bitcsr::BitCsrDistanceMatrix;
use crate::engine::heap::FastHeap;
use crate::engine::simplex::{FxHashMap, PackedSimplex};
use crate::types::PersistenceInterval;

/// Engine-internal representative for one EMITTED interval: the birth simplex,
/// the death simplex (`None` for essential classes), and the representative
/// cocycle — the birth simplex plus the recorded reduction (V) column, over
/// Z/2. Aligned index-for-index with the emitted intervals of its dimension.
pub(crate) struct RawRepresentative<S> {
    pub(crate) birth: S,
    pub(crate) death: Option<S>,
    pub(crate) cocycle: Vec<S>,
}

/// Sort and Z/2-cancel `working_v` in place, emitting each surviving entry.
/// Since elements may appear with multiplicity, equal pairs cancel.
#[inline]
fn for_each_cancelled<S: Copy + Ord>(working_v: &mut [S], mut f: impl FnMut(S)) {
    working_v.sort_unstable();
    let mut i = 0;
    while i < working_v.len() {
        let s = working_v[i];
        if i + 1 < working_v.len() && working_v[i + 1] == s {
            i += 2; // pair cancels in Z/2
        } else {
            f(s);
            i += 1;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CompressedSparseMatrix — storage for the reduction matrix V
// ═══════════════════════════════════════════════════════════════════════════════

pub struct CompressedSparseMatrix<S: Copy> {
    entries: Vec<S>,
    bounds: Vec<usize>,
}

impl<S: Copy> CompressedSparseMatrix<S> {
    #[inline]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            bounds: Vec::new(),
        }
    }

    #[inline]
    pub fn append_column(&mut self) {
        self.bounds.push(self.entries.len());
    }

    #[inline]
    pub fn push(&mut self, entry: S) {
        debug_assert!(!self.bounds.is_empty(), "must call append_column first");
        self.entries.push(entry);
        *self.bounds.last_mut().unwrap() = self.entries.len();
    }

    #[inline]
    pub fn column(&self, index: usize) -> &[S] {
        let end = self.bounds[index];
        let start = if index == 0 {
            0
        } else {
            self.bounds[index - 1]
        };
        &self.entries[start..end]
    }
}

//
// ═══════════════════════════════════════════════════════════════════════════════
// Coboundary expansion
// ═══════════════════════════════════════════════════════════════════════════════

//hottest loop accounts for 55% of runtime
#[inline]
fn add_simplex_coboundary<const K: usize, S: PackedSimplex<K>>(
    simplex: S,
    dist: &BitCsrDistanceMatrix,
    working_v: &mut Vec<S>,
    working_coboundary: &mut FastHeap<S>,
) {
    working_v.push(simplex);
    for_each_cofacet(dist, simplex, true, |cofacet| {
        working_coboundary.push(cofacet);
        true
    });
}

#[inline]
fn add_coboundary<const K: usize, S: PackedSimplex<K>>(
    v_matrix: &CompressedSparseMatrix<S>,
    columns: &[S],
    column_index: usize,
    dist: &BitCsrDistanceMatrix,
    working_v: &mut Vec<S>,
    working_coboundary: &mut FastHeap<S>,
) {
    add_simplex_coboundary(columns[column_index], dist, working_v, working_coboundary);

    for &simplex in v_matrix.column(column_index) {
        add_simplex_coboundary(simplex, dist, working_v, working_coboundary);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Initial pivot search with emergent-pair shortcut
// ═══════════════════════════════════════════════════════════════════════════════

fn init_coboundary_and_get_pivot<const K: usize, S: PackedSimplex<K>>(
    sigma: S,
    dist: &BitCsrDistanceMatrix,
    pivot_column_index: &FxHashMap<S, usize>,
    working_coboundary: &mut FastHeap<S>,
) -> Option<S> {
    let sigma_filt = sigma.filtration_encoded();
    let mut check_for_emergent_pair = true;
    let mut emergent: Option<S> = None;

    for_each_cofacet(dist, sigma, true, |cofacet| {
        if check_for_emergent_pair && cofacet.filtration_encoded() == sigma_filt {
            if !pivot_column_index.contains_key(&cofacet)
                && zero_apparent_facet(dist, cofacet).is_none()
            {
                working_coboundary.clear(); // discard pollution
                emergent = Some(cofacet);
                return false;
            }
            check_for_emergent_pair = false;
        }
        working_coboundary.append_raw(cofacet);
        true
    });

    if emergent.is_some() {
        return emergent;
    }

    working_coboundary.heapify();
    working_coboundary.get_pivot()
}

// ═══════════════════════════════════════════════════════════════════════════════
// compute_pairs — the main reduction loop
// ═══════════════════════════════════════════════════════════════════════════════

/// Reduce `columns`, appending intervals to `dim_intervals`. Returns the
/// pivot → column-index map; its key set is exactly the pivots recorded this
/// dimension, which the caller feeds to `assemble_candidates` as the clearing
/// set (avoiding a second, duplicate hash table).
///
/// `RECORD` gates cocycle-representative capture at compile time: the plain
/// path monomorphises with every recording block removed, so it stays exactly
/// the hot loop it was. When recording, one `RawRepresentative` is pushed to
/// `dim_reps` per EMITTED interval, in the same order.
pub(crate) fn compute_pairs<const K: usize, S: PackedSimplex<K>, const RECORD: bool>(
    columns: &[S],
    dist: &BitCsrDistanceMatrix,
    dim_intervals: &mut Vec<PersistenceInterval>,
    dim_reps: &mut Vec<RawRepresentative<S>>,
) -> FxHashMap<S, usize> {
    let mut pivot_column_index: FxHashMap<S, usize> =
        FxHashMap::with_capacity_and_hasher(columns.len(), Default::default());
    if columns.is_empty() {
        return pivot_column_index;
    }

    let mut v_matrix: CompressedSparseMatrix<S> = CompressedSparseMatrix::new();

    // Working buffers grow on demand; the prealloc is a warm start, not a bound.
    // Cap it so huge column counts don't reserve gigabytes speculatively
    // (2 buffers x 16 B/entry: the cap bounds the upfront cost to ~16 MiB each).
    let cap = (columns.len() * 4).clamp(1024, 1 << 20);
    let mut working_v: Vec<S> = Vec::with_capacity(cap);
    let mut working_coboundary: FastHeap<S> = FastHeap::with_capacity(cap);

    for (j, &sigma) in columns.iter().enumerate() {
        working_v.clear();
        working_coboundary.clear();
        v_matrix.append_column();

        let mut pivot = init_coboundary_and_get_pivot(
            sigma,
            dist,
            &pivot_column_index,
            &mut working_coboundary,
        );

        if working_coboundary.is_empty() {
            if let Some(tau) = pivot {
                if tau.filtration_encoded() != sigma.filtration_encoded() {
                    dim_intervals.push(PersistenceInterval {
                        birth: sigma.filtration(),
                        death: tau.filtration(),
                    });
                    if RECORD {
                        // No columns were added: the cocycle is σ alone.
                        dim_reps.push(RawRepresentative {
                            birth: sigma,
                            death: Some(tau),
                            cocycle: vec![sigma],
                        });
                    }
                }
                pivot_column_index.insert(tau, j);
                continue;
            }
            dim_intervals.push(PersistenceInterval {
                birth: sigma.filtration(),
                death: f32::INFINITY,
            });
            if RECORD {
                dim_reps.push(RawRepresentative {
                    birth: sigma,
                    death: None,
                    cocycle: vec![sigma],
                });
            }
            continue;
        }

        loop {
            match pivot {
                None => {
                    dim_intervals.push(PersistenceInterval {
                        birth: sigma.filtration(),
                        death: f32::INFINITY,
                    });
                    if RECORD {
                        // Essential class (only reachable under a user
                        // threshold): the accumulated column is its cocycle
                        // within the truncated complex.
                        let mut cocycle = vec![sigma];
                        for_each_cancelled(&mut working_v, |s| cocycle.push(s));
                        dim_reps.push(RawRepresentative {
                            birth: sigma,
                            death: None,
                            cocycle,
                        });
                    }
                    break;
                }
                Some(tau) => {
                    if let Some(&k) = pivot_column_index.get(&tau) {
                        add_coboundary(
                            &v_matrix,
                            columns,
                            k,
                            dist,
                            &mut working_v,
                            &mut working_coboundary,
                        );
                        pivot = working_coboundary.get_pivot();
                        continue;
                    }

                    if let Some(phi) = zero_apparent_facet(dist, tau) {
                        add_simplex_coboundary(phi, dist, &mut working_v, &mut working_coboundary);
                        pivot = working_coboundary.get_pivot();
                        continue;
                    }

                    let emitted = tau.filtration_encoded() != sigma.filtration_encoded();
                    if emitted {
                        dim_intervals.push(PersistenceInterval {
                            birth: sigma.filtration(),
                            death: tau.filtration(),
                        });
                    }
                    pivot_column_index.insert(tau, j);

                    // Sort and Z/2-cancel, then drain into the reduction matrix.
                    for_each_cancelled(&mut working_v, |s| v_matrix.push(s));

                    if RECORD && emitted {
                        // The cocycle of the class killed by τ: σ plus the
                        // just-recorded V column.
                        let column = v_matrix.column(j);
                        let mut cocycle = Vec::with_capacity(column.len() + 1);
                        cocycle.push(sigma);
                        cocycle.extend_from_slice(column);
                        dim_reps.push(RawRepresentative {
                            birth: sigma,
                            death: Some(tau),
                            cocycle,
                        });
                    }
                    break;
                }
            }
        }
    }

    pivot_column_index
}
