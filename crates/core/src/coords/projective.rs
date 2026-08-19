//! Real projective coordinates: H¹ mod 2 → RPᵈ.
//!
//! After Perea, *Multiscale Projective Coordinates via Persistent Cohomology
//! of Sparse Filtrations* (2018), specialised to the full-vertex cover:
//! every data point is a vertex, and the cover is the open balls of radius
//! `t/2` around all vertices (their nerve sits inside VR(t), so the
//! restricted cocycle is defined and closed on it).
//!
//! Pipeline: Z/2 persistent cohomology with representatives (the existing
//! engine) → select a class by persistence rank → restrict its cocycle to
//! edges ≤ t (sound: the reduced column's coboundary lives entirely at
//! diameters ≥ death > t) → classifying map into RP^(n−1) as one sparse
//! signed row per point → project to the top-(d+1) principal subspace
//! (deterministic subspace iteration) and renormalise rows.
//!
//! The one-shot subspace projection is a pragmatic stand-in for Perea's
//! iterative principal projective components: the subspace is the same; the
//! embeddings differ only by refinements that are irrelevant up to
//! projective isometry.

use std::collections::HashSet;

use super::{rank_classes, validate_target_dim, Config, ProjectiveCoordinates};
use crate::engine::algorithm::compute_with_representatives;
use crate::engine::simplex::{Simplex128, Simplex256};
use crate::engine::BitCsrDistanceMatrix;
use crate::{Error, Result};

/// Iteration budget for the subspace iteration; convergence usually takes
/// far fewer. The tolerance is on per-column subspace alignment.
const MAX_POWER_ITERS: usize = 128;
const POWER_TOL: f32 = 1e-7;

/// Real projective coordinates with the default [`Config`]. See
/// [`projective_with`].
pub fn projective(data: &[f32], n: usize) -> Result<ProjectiveCoordinates> {
    projective_with(data, n, &Config::default())
}

/// Real projective coordinates from the H¹ mod-2 class of rank
/// `config.class_index`, mapped into RP^(config.target_dim).
///
/// `data` follows the same layout contract as
/// [`crate::persistent_homology_with`] (point cloud, or distance matrix when
/// `config.distance_matrix` is set). Returns one unit row per input point,
/// defined up to sign.
pub fn projective_with(data: &[f32], n: usize, config: &Config) -> Result<ProjectiveCoordinates> {
    validate_target_dim(config.target_dim, n)?;

    // Full pipeline at the certified radius, peel on (transparent here).
    let core_config = crate::Config {
        distance_matrix: config.distance_matrix,
        quotient: config.quotient,
        parallel: config.parallel,
        ..crate::Config::default()
    };
    let edges = crate::build_edges(data, n, 1, &core_config)?;
    let threshold = edges.threshold;
    let bitcsr = BitCsrDistanceMatrix::from_edge_list(edges);
    // Simplex dispatch mirrors `compute_represented_barcode`; max_dim is 1
    // here, so only the point count decides the packing.
    let rb = if n <= crate::S128_MAX_POINTS {
        compute_with_representatives::<6, Simplex128>(&bitcsr, 1, config.parallel)
    } else {
        compute_with_representatives::<9, Simplex256>(&bitcsr, 1, config.parallel)
    };

    // Class selection by persistence rank.
    let intervals = &rb.barcode.intervals[1];
    let ranked = rank_classes(intervals, threshold);
    let Some(&class) = ranked.get(config.class_index) else {
        return Err(Error::NotEnoughClasses {
            requested: config.class_index + 1,
            available: ranked.len(),
            dim: 1,
        });
    };
    let interval = intervals[class];
    let death_cap = if interval.death.is_finite() {
        interval.death
    } else {
        threshold
    };
    let scale = config.scale.unwrap_or(0.5 * (interval.birth + death_cap));
    if !(scale > 0.0 && scale >= interval.birth && scale < death_cap) {
        return Err(Error::InvalidScale {
            scale,
            birth: interval.birth,
            death: death_cap,
        });
    }

    // Restricted cocycle: representative edges with diameter <= scale,
    // canonical (u, v) with u > v (the representative's vertex order).
    let mut cocycle: HashSet<(u32, u32)> = HashSet::new();
    for simplex in &rb.representatives[1][class].cocycle {
        let (u, v) = (simplex[0], simplex[1]);
        if bitcsr.edge_dist(u, v) <= scale {
            cocycle.insert((u, v));
        }
    }

    let rows = classifying_rows(&bitcsr, n, scale * 0.5, &cocycle);
    let m = config.target_dim + 1;
    let basis = top_subspace(&rows, n, m);
    let coordinates = project_rows(&rows, &basis, n, m);

    Ok(ProjectiveCoordinates {
        coordinates,
        target_dim: config.target_dim,
        birth: interval.birth,
        death: interval.death,
        scale,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Classifying map — one sparse signed row per point
// ═══════════════════════════════════════════════════════════════════════════════

/// CSR rows of the classifying map into RP^(n−1). Row `i` has an entry for
/// every vertex `k` with `d(i, k) < r` (plus `i` itself): value
/// `±sqrt(bump / total)` with the linear bump `r − d`, sign `(−1)^{η(i,k)}`
/// (each point indexes the cover by its own ball, so `j(x) = x` and the
/// diagonal is always positive). Rows are unit vectors by construction.
struct Rows {
    offsets: Vec<u32>,
    cols: Vec<u32>,
    vals: Vec<f32>,
}

fn classifying_rows(
    bitcsr: &BitCsrDistanceMatrix,
    n: usize,
    r: f32,
    cocycle: &HashSet<(u32, u32)>,
) -> Rows {
    // Degrees (self entry included), then CSR offsets.
    let mut deg = vec![1u32; n];
    for v in 0..n {
        bitcsr.for_each_neighbor_above(v, |w, d| {
            if d < r {
                deg[v] += 1;
                deg[w as usize] += 1;
            }
            true
        });
    }
    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0u32);
    let mut total = 0u64;
    for &d in &deg {
        total += d as u64;
        assert!(
            total <= u32::MAX as u64,
            "classifying map exceeds u32 entries"
        );
        offsets.push(total as u32);
    }
    let total = total as usize;

    let mut cols = vec![0u32; total];
    let mut vals = vec![0f32; total];
    let mut cursor: Vec<u32> = offsets[..n].to_vec();

    // Self entries first (bump at distance 0 is r, sign +).
    for v in 0..n {
        cols[cursor[v] as usize] = v as u32;
        vals[cursor[v] as usize] = r;
        cursor[v] += 1;
    }
    // Neighbor entries, symmetric; `for_each_neighbor_above` yields w > v,
    // matching the canonical cocycle key (w, v). Signs fold into the value.
    for v in 0..n {
        bitcsr.for_each_neighbor_above(v, |w, d| {
            if d < r {
                let bump = r - d;
                let signed = if cocycle.contains(&(w, v as u32)) {
                    -bump
                } else {
                    bump
                };
                cols[cursor[v] as usize] = w;
                vals[cursor[v] as usize] = signed;
                cursor[v] += 1;
                let wi = w as usize;
                cols[cursor[wi] as usize] = v as u32;
                vals[cursor[wi] as usize] = signed;
                cursor[wi] += 1;
            }
            true
        });
    }

    // Normalise: value = sign * sqrt(bump / total bump mass of the row).
    for i in 0..n {
        let range = offsets[i] as usize..offsets[i + 1] as usize;
        let sum: f32 = vals[range.clone()].iter().map(|v| v.abs()).sum();
        debug_assert!(sum > 0.0, "row {i} lost its own diagonal bump");
        for v in &mut vals[range] {
            *v = v.signum() * (v.abs() / sum).sqrt();
        }
    }

    Rows {
        offsets,
        cols,
        vals,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Deterministic subspace iteration + row projection
// ═══════════════════════════════════════════════════════════════════════════════

/// Top-`m` principal subspace of the second-moment matrix VᵀV of the rows,
/// via block power iteration with modified Gram–Schmidt. Column-major
/// `n × m`. Deterministic (seeded init, sequential accumulation).
fn top_subspace(rows: &Rows, n: usize, m: usize) -> Vec<f32> {
    let mut rng = xorshift(0x9e37_79b9_7f4a_7c15);
    let mut basis = vec![0f32; n * m];
    for value in basis.iter_mut() {
        *value = rng() - 0.5;
    }
    orthonormalize(&mut basis, n, m, &mut rng);

    let mut y = vec![0f32; n * m];
    let mut next = vec![0f32; n * m];
    let mut previous = basis.clone();

    for _ in 0..MAX_POWER_ITERS {
        // y[c][i] = row_i · basis[c]
        for i in 0..n {
            let range = rows.offsets[i] as usize..rows.offsets[i + 1] as usize;
            for c in 0..m {
                let col = &basis[c * n..(c + 1) * n];
                let mut acc = 0f32;
                for e in range.clone() {
                    acc += rows.vals[e] * col[rows.cols[e] as usize];
                }
                y[c * n + i] = acc;
            }
        }
        // next[c][j] = Σ_i rows[i][j] · y[c][i]
        next.fill(0.0);
        for i in 0..n {
            let range = rows.offsets[i] as usize..rows.offsets[i + 1] as usize;
            for c in 0..m {
                let yi = y[c * n + i];
                let out = &mut next[c * n..(c + 1) * n];
                for e in range.clone() {
                    out[rows.cols[e] as usize] += rows.vals[e] * yi;
                }
            }
        }
        std::mem::swap(&mut basis, &mut next);
        orthonormalize(&mut basis, n, m, &mut rng);

        // Per-column alignment with the previous iterate.
        let mut worst = 0f32;
        for c in 0..m {
            let mut dot = 0f32;
            for j in 0..n {
                dot += basis[c * n + j] * previous[c * n + j];
            }
            let misalignment = 1.0 - dot.abs();
            if misalignment > worst {
                worst = misalignment;
            }
        }
        if worst < POWER_TOL {
            break;
        }
        previous.copy_from_slice(&basis);
    }
    basis
}

/// Modified Gram–Schmidt on the `m` columns; a degenerate column is reseeded
/// (deterministically) and re-orthogonalised.
fn orthonormalize(basis: &mut [f32], n: usize, m: usize, rng: &mut impl FnMut() -> f32) {
    for c in 0..m {
        loop {
            for p in 0..c {
                let mut dot = 0f32;
                for j in 0..n {
                    dot += basis[c * n + j] * basis[p * n + j];
                }
                for j in 0..n {
                    basis[c * n + j] -= dot * basis[p * n + j];
                }
            }
            let norm: f32 = basis[c * n..(c + 1) * n]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            if norm > 1e-12 {
                for j in 0..n {
                    basis[c * n + j] /= norm;
                }
                break;
            }
            for j in 0..n {
                basis[c * n + j] = rng() - 0.5;
            }
        }
    }
}

/// Project every row onto the basis and renormalise: the RPᵈ coordinates,
/// row-major `(n, m)`.
fn project_rows(rows: &Rows, basis: &[f32], n: usize, m: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * m];
    for i in 0..n {
        let range = rows.offsets[i] as usize..rows.offsets[i + 1] as usize;
        let mut norm_sq = 0f32;
        for c in 0..m {
            let col = &basis[c * n..(c + 1) * n];
            let mut acc = 0f32;
            for e in range.clone() {
                acc += rows.vals[e] * col[rows.cols[e] as usize];
            }
            out[i * m + c] = acc;
            norm_sq += acc * acc;
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            for c in 0..m {
                out[i * m + c] /= norm;
            }
        } else {
            // Point orthogonal to the principal subspace (e.g. a far-away
            // component): any representative is valid; pick a pole.
            out[i * m] = 1.0;
        }
    }
    out
}

/// Deterministic xorshift in [0, 1).
fn xorshift(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / 16_777_216.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fitted maps: out-of-sample evaluation
// ═══════════════════════════════════════════════════════════════════════════════

/// A fitted real-projective map: [`projective_with`] with the model retained
/// — training points, cover radius `r = scale / 2`, the restricted Z/2
/// cocycle, and the principal basis. A new point inside the cover gets the
/// classifying row of its nearest covered training vertex's chart, projected
/// onto the SAVED basis and renormalised; at a training point this
/// reproduces the fit row exactly. Points outside the cover evaluate to NaN
/// rows.
#[derive(Debug)]
pub struct ProjectiveMap {
    train: Vec<f32>,
    dim: usize,
    n: usize,
    /// Training coordinates, row-major `(n, target_dim + 1)` — identical to
    /// what [`projective_with`] returns for the same input and config.
    pub coordinates: Vec<f32>,
    pub target_dim: usize,
    pub birth: f32,
    pub death: f32,
    pub scale: f32,
    r: f32,
    cocycle: HashSet<(u32, u32)>,
    basis: Vec<f32>, // column-major n × (target_dim + 1)
    parallel: bool,
}

/// Fit a real-projective map with the default [`Config`]. See
/// [`projective_map_with`] and [`ProjectiveMap`].
pub fn projective_map(data: &[f32], n: usize) -> Result<ProjectiveMap> {
    projective_map_with(data, n, &Config::default())
}

/// Fit a real-projective map: [`projective_with`]'s pipeline with the model
/// retained. `distance_matrix` and `quotient` are rejected (see
/// [`crate::coords::circular_map_with`] for the reasons — identical here).
pub fn projective_map_with(data: &[f32], n: usize, config: &Config) -> Result<ProjectiveMap> {
    if config.distance_matrix {
        return Err(Error::NotImplemented(
            "fitted projective maps on distance-matrix input",
        ));
    }
    if config.quotient {
        return Err(Error::NotImplemented(
            "fitted projective maps on the quotient filtration",
        ));
    }
    validate_target_dim(config.target_dim, n)?;

    let core_config = crate::Config {
        parallel: config.parallel,
        ..crate::Config::default()
    };
    let edges = crate::build_edges(data, n, 1, &core_config)?;
    let threshold = edges.threshold;
    let dim = data.len() / n;
    let bitcsr = BitCsrDistanceMatrix::from_edge_list(edges);
    let rb = if n <= crate::S128_MAX_POINTS {
        compute_with_representatives::<6, Simplex128>(&bitcsr, 1, config.parallel)
    } else {
        compute_with_representatives::<9, Simplex256>(&bitcsr, 1, config.parallel)
    };

    let intervals = &rb.barcode.intervals[1];
    let ranked = rank_classes(intervals, threshold);
    let Some(&class) = ranked.get(config.class_index) else {
        return Err(Error::NotEnoughClasses {
            requested: config.class_index + 1,
            available: ranked.len(),
            dim: 1,
        });
    };
    let interval = intervals[class];
    let death_cap = if interval.death.is_finite() {
        interval.death
    } else {
        threshold
    };
    let scale = config.scale.unwrap_or(0.5 * (interval.birth + death_cap));
    if !(scale > 0.0 && scale >= interval.birth && scale < death_cap) {
        return Err(Error::InvalidScale {
            scale,
            birth: interval.birth,
            death: death_cap,
        });
    }

    let mut cocycle: HashSet<(u32, u32)> = HashSet::new();
    for simplex in &rb.representatives[1][class].cocycle {
        let (u, v) = (simplex[0], simplex[1]);
        if bitcsr.edge_dist(u, v) <= scale {
            cocycle.insert((u, v));
        }
    }

    let r = scale * 0.5;
    let rows = classifying_rows(&bitcsr, n, r, &cocycle);
    let m = config.target_dim + 1;
    let basis = top_subspace(&rows, n, m);
    let coordinates = project_rows(&rows, &basis, n, m);

    Ok(ProjectiveMap {
        train: data.to_vec(),
        dim,
        n,
        coordinates,
        target_dim: config.target_dim,
        birth: interval.birth,
        death: interval.death,
        scale,
        r,
        cocycle,
        basis,
        parallel: config.parallel,
    })
}

impl ProjectiveMap {
    /// Number of training points.
    pub fn n_points(&self) -> usize {
        self.n
    }

    /// Ambient dimension of the fitted point cloud.
    pub fn ambient_dim(&self) -> usize {
        self.dim
    }

    /// Evaluate the fitted map on `m` new points (row-major `m × dim`).
    /// Returns row-major `(m, target_dim + 1)` unit rows (up to sign); NaN
    /// rows where the point is outside the cover.
    pub fn transform(&self, queries: &[f32], m: usize) -> Result<Vec<f32>> {
        if queries.len() != m * self.dim {
            return Err(Error::ShapeMismatch {
                n: m,
                got: queries.len(),
                mode: "transform queries (length must be m × the fitted ambient dimension)",
            });
        }
        let width = self.target_dim + 1;
        let mut out = vec![f32::NAN; m * width];
        if m == 0 {
            return Ok(out);
        }

        let cross = crate::preprocess::edgelist::cross_neighbors(
            queries,
            m,
            &self.train,
            self.n,
            self.dim,
            self.r,
            self.parallel,
        );

        for q in 0..m {
            let start = cross.offsets[q] as usize;
            let end = cross.offsets[q + 1] as usize;
            if start == end {
                continue; // outside the cover: NaN row
            }

            // Chart = nearest training vertex; bump mass over the neighbours.
            let mut chart = cross.cols[start];
            let mut best = cross.dists[start];
            let mut bump_total = 0f64;
            for e in start..end {
                bump_total += (self.r - cross.dists[e]) as f64;
                if cross.dists[e] < best {
                    best = cross.dists[e];
                    chart = cross.cols[e];
                }
            }

            // Row entries ± sqrt(φ̂_k), signs from the chart's cocycle values,
            // accumulated directly against the saved basis.
            let mut u = vec![0f32; width];
            for e in start..end {
                let k = cross.cols[e];
                let phi = (self.r - cross.dists[e]) as f64 / bump_total;
                let key = if chart > k { (chart, k) } else { (k, chart) };
                let sign = if chart != k && self.cocycle.contains(&key) {
                    -1.0f32
                } else {
                    1.0
                };
                let value = sign * (phi.sqrt() as f32);
                for (c, slot) in u.iter_mut().enumerate() {
                    *slot += value * self.basis[c * self.n + k as usize];
                }
            }
            let norm: f32 = u.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 1e-12 {
                for c in 0..width {
                    out[q * width + c] = u[c] / norm;
                }
            } else {
                // Orthogonal to the principal subspace: any representative
                // is valid; match the fit's pole fallback.
                out[q * width] = 1.0;
                for c in 1..width {
                    out[q * width + c] = 0.0;
                }
            }
        }
        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Points on a slightly wobbled unit circle (the wobble breaks the exact
    /// rotational symmetry, avoiding degenerate principal directions).
    fn circle(n: usize) -> Vec<f32> {
        let mut points = Vec::with_capacity(2 * n);
        for i in 0..n {
            let theta = 2.0 * std::f32::consts::PI * (i as f32) / (n as f32);
            let radius = 1.0 + 0.02 * (3.0 * theta).sin();
            points.push(radius * theta.cos());
            points.push(radius * theta.sin());
        }
        points
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// The circle's H¹ mod-2 class is the Möbius class: walking once around
    /// the loop, the lifted representative must come back with the opposite
    /// sign. Equivalently, the product over consecutive (cyclic) pairs of
    /// sign⟨uᵢ, uᵢ₊₁⟩ is −1. This is the topological correctness of the map,
    /// invariant to the choice of principal basis.
    #[test]
    fn circle_embedding_is_moebius() {
        let n = 48;
        let points = circle(n);
        let pc = projective(&points, n).unwrap();
        let m = pc.target_dim + 1;
        assert_eq!(m, 3);
        assert_eq!(pc.coordinates.len(), n * m);
        assert!(pc.birth < pc.scale && pc.scale < pc.death);

        // Unit rows.
        for i in 0..n {
            let row = &pc.coordinates[i * m..(i + 1) * m];
            let norm = dot(row, row).sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "row {i} has norm {norm}");
        }

        let mut product = 1.0f32;
        for i in 0..n {
            let a = &pc.coordinates[i * m..(i + 1) * m];
            let j = (i + 1) % n;
            let b = &pc.coordinates[j * m..(j + 1) * m];
            let d = dot(a, b);
            assert!(
                d.abs() > 1e-4,
                "consecutive points {i},{j} nearly orthogonal (d = {d})"
            );
            product *= d.signum();
        }
        assert_eq!(product, -1.0, "the loop must realise the Möbius class");
    }

    #[test]
    fn deterministic_output() {
        let n = 32;
        let points = circle(n);
        let a = projective(&points, n).unwrap();
        let b = projective(&points, n).unwrap();
        assert_eq!(a.coordinates, b.coordinates);
        assert_eq!(a.scale.to_bits(), b.scale.to_bits());
    }

    /// The Möbius invariant survives the quotient approximation: the H¹
    /// mod-2 class of the coned circle still flips sign around the loop.
    #[test]
    fn quotient_circle_is_moebius() {
        let n = 48;
        let points = circle(n);
        let pc = projective_with(
            &points,
            n,
            &Config {
                quotient: true,
                ..Config::default()
            },
        )
        .unwrap();
        let m = pc.target_dim + 1;
        let mut product = 1.0f32;
        for i in 0..n {
            let a = &pc.coordinates[i * m..(i + 1) * m];
            let j = (i + 1) % n;
            let b = &pc.coordinates[j * m..(j + 1) * m];
            let d = dot(a, b);
            assert!(
                d.abs() > 1e-4,
                "consecutive points {i},{j} nearly orthogonal (d = {d})"
            );
            product *= d.signum();
        }
        assert_eq!(product, -1.0, "quotient loop must realise the Möbius class");
    }

    #[test]
    fn selection_and_validation_errors() {
        let n = 32;
        let points = circle(n);

        let err = projective_with(
            &points,
            n,
            &Config {
                class_index: 64,
                ..Config::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::NotEnoughClasses { dim: 1, .. }));

        let reference = projective(&points, n).unwrap();
        let err = projective_with(
            &points,
            n,
            &Config {
                scale: Some(reference.birth * 0.5),
                ..Config::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidScale { .. }));

        let ok = projective_with(
            &points,
            n,
            &Config {
                scale: Some(0.5 * (reference.birth + reference.death)),
                ..Config::default()
            },
        )
        .unwrap();
        assert_eq!(ok.coordinates, reference.coordinates);

        let err = projective_with(
            &points,
            n,
            &Config {
                target_dim: 0,
                ..Config::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::InvalidTargetDim {
                got: 0,
                max: 31,
                points: 32
            }
        ));

        let err = projective_with(
            &points,
            n,
            &Config {
                target_dim: n,
                ..Config::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::InvalidTargetDim {
                got: 32,
                max: 31,
                points: 32
            }
        ));
    }

    /// The fitted map reproduces the fit on training points (same rows, same
    /// arithmetic) and realises the Möbius class on a fresh loop.
    #[test]
    fn projective_map_round_trip_and_fresh_loop() {
        let n = 48;
        let points = circle(n);
        let map = projective_map(&points, n).unwrap();
        let m = map.target_dim + 1;

        // Stored training coordinates equal the plain function's output.
        let plain = projective(&points, n).unwrap();
        assert_eq!(map.coordinates, plain.coordinates);

        // Round trip: transform(train) reproduces the fit rows (identical
        // chart, bumps, and basis) up to float noise and global sign.
        let back = map.transform(&points, n).unwrap();
        for i in 0..n {
            let a = &back[i * m..(i + 1) * m];
            let b = &map.coordinates[i * m..(i + 1) * m];
            let d: f32 = dot(a, b);
            assert!(
                d.abs() > 1.0 - 1e-4,
                "row {i} differs from the fit (|dot| = {})",
                d.abs()
            );
        }

        // A fresh loop at shifted phases realises the Möbius class.
        let f = 60;
        let mut fresh = Vec::with_capacity(2 * f);
        for i in 0..f {
            let theta = 2.0 * std::f32::consts::PI * (i as f32 + 0.37) / (f as f32);
            fresh.push(theta.cos());
            fresh.push(theta.sin());
        }
        let values = map.transform(&fresh, f).unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
        let mut product = 1.0f32;
        for i in 0..f {
            let a = &values[i * m..(i + 1) * m];
            let b = &values[((i + 1) % f) * m..((i + 1) % f + 1) * m];
            let d = dot(a, b);
            assert!(d.abs() > 1e-4);
            product *= d.signum();
        }
        assert_eq!(product, -1.0, "fresh loop must realise the Möbius class");

        // Outside the cover → NaN row; unsupported configs rejected.
        let far = map.transform(&[100.0, 100.0], 1).unwrap();
        assert!(far.iter().all(|v| v.is_nan()));
        let err = projective_map_with(
            &points,
            n,
            &Config {
                quotient: true,
                ..Config::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::NotImplemented(_)));
    }

    /// Two circles side by side: at least two H¹ classes, both selectable,
    /// giving different embeddings; an absurd rank reports the count.
    #[test]
    fn two_circles_expose_two_classes() {
        let per = 24;
        let mut points = circle(per);
        for i in 0..per {
            let theta = 2.0 * std::f32::consts::PI * (i as f32) / (per as f32);
            points.push(4.0 + theta.cos());
            points.push(theta.sin());
        }
        let n = 2 * per;

        let first = projective_with(
            &points,
            n,
            &Config {
                class_index: 0,
                ..Config::default()
            },
        )
        .unwrap();
        let second = projective_with(
            &points,
            n,
            &Config {
                class_index: 1,
                ..Config::default()
            },
        )
        .unwrap();
        assert_ne!(first.coordinates, second.coordinates);

        let err = projective_with(
            &points,
            n,
            &Config {
                class_index: 999,
                ..Config::default()
            },
        )
        .unwrap_err();
        match err {
            Error::NotEnoughClasses { available, .. } => assert!(available >= 2),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
