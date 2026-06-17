// ═══════════════════════════════════════════════════════════════════════════════
// reduction.rs — Implicit cohomological matrix reduction over Z/2
// ═══════════════════════════════════════════════════════════════════════════════

use crate::engine::algorithm::{for_each_cofacet, zero_apparent_facet};
use crate::engine::bitcsr::BitCsrDistanceMatrix;
use crate::engine::heap::FastHeap;
use crate::engine::simplex::{FxHashMap, Simplex128};
use crate::types::PersistenceInterval;

// ═══════════════════════════════════════════════════════════════════════════════
// CompressedSparseMatrix — storage for the reduction matrix V
// ═══════════════════════════════════════════════════════════════════════════════

pub struct CompressedSparseMatrix {
    entries: Vec<Simplex128>,
    bounds: Vec<usize>,
}

impl CompressedSparseMatrix {
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
    pub fn push(&mut self, entry: Simplex128) {
        debug_assert!(!self.bounds.is_empty(), "must call append_column first");
        self.entries.push(entry);
        *self.bounds.last_mut().unwrap() = self.entries.len();
    }

    #[inline]
    pub fn column(&self, index: usize) -> &[Simplex128] {
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
fn add_simplex_coboundary(
    simplex: Simplex128,
    dist: &BitCsrDistanceMatrix,
    working_v: &mut Vec<Simplex128>,
    working_coboundary: &mut FastHeap,
) {
    working_v.push(simplex);
    for_each_cofacet(dist, simplex, true, |cofacet| {
        working_coboundary.push(cofacet);
        true
    });
}

#[inline]
fn add_coboundary(
    v_matrix: &CompressedSparseMatrix,
    columns: &[Simplex128],
    column_index: usize,
    dist: &BitCsrDistanceMatrix,
    working_v: &mut Vec<Simplex128>,
    working_coboundary: &mut FastHeap,
) {
    add_simplex_coboundary(columns[column_index], dist, working_v, working_coboundary);

    for &simplex in v_matrix.column(column_index) {
        add_simplex_coboundary(simplex, dist, working_v, working_coboundary);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Initial pivot search with emergent-pair shortcut
// ═══════════════════════════════════════════════════════════════════════════════

fn init_coboundary_and_get_pivot(
    sigma: Simplex128,
    dist: &BitCsrDistanceMatrix,
    pivot_column_index: &FxHashMap<Simplex128, usize>,
    working_coboundary: &mut FastHeap,
) -> Option<Simplex128> {
    let sigma_filt = sigma.filtration_encoded();
    let mut check_for_emergent_pair = true;
    let mut emergent: Option<Simplex128> = None;

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

pub fn compute_pairs(
    columns: &[Simplex128],
    dist: &BitCsrDistanceMatrix,
    dim_intervals: &mut Vec<PersistenceInterval>,
    cleared_pivots: &mut FxHashMap<Simplex128, ()>,
) {
    if columns.is_empty() {
        return;
    }

    let mut v_matrix = CompressedSparseMatrix::new();
    let mut pivot_column_index: FxHashMap<Simplex128, usize> =
        FxHashMap::with_capacity_and_hasher(columns.len(), Default::default());

    let cap = (columns.len() * 4).max(1024);
    let mut working_v: Vec<Simplex128> = Vec::with_capacity(cap);
    let mut working_coboundary = FastHeap::with_capacity(cap);

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
                }
                pivot_column_index.insert(tau, j);
                cleared_pivots.insert(tau, ());
                continue;
            }
            dim_intervals.push(PersistenceInterval {
                birth: sigma.filtration(),
                death: f32::INFINITY,
            });
            continue;
        }

        loop {
            match pivot {
                None => {
                    dim_intervals.push(PersistenceInterval {
                        birth: sigma.filtration(),
                        death: f32::INFINITY,
                    });
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

                    if tau.filtration_encoded() != sigma.filtration_encoded() {
                        dim_intervals.push(PersistenceInterval {
                            birth: sigma.filtration(),
                            death: tau.filtration(),
                        });
                    }
                    pivot_column_index.insert(tau, j);
                    cleared_pivots.insert(tau, ());

                    // Sort and Z/2-cancel, then drain into the reduction matrix.
                    // Since elements may appear with multiplicity, equal pairs cancel.
                    working_v.sort_unstable();
                    let mut i = 0;
                    while i < working_v.len() {
                        let s = working_v[i];
                        if i + 1 < working_v.len() && working_v[i + 1] == s {
                            i += 2; // pair cancels in Z/2
                        } else {
                            v_matrix.push(s);
                            i += 1;
                        }
                    }
                    break;
                }
            }
        }
    }
}
