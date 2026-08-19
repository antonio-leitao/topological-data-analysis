//! Complex projective coordinates: H² mod p → CPᵈ.
//!
//! After Perea, *Multiscale Projective Coordinates via Persistent Cohomology
//! of Sparse Filtrations* (2018), on the full-vertex cover. An integer Čech
//! 2-cocycle η classifies a line bundle, hence a map to CP^∞; concretely,
//! with the cover `{U_j}` = balls of radius `r = t/2`, partition of unity
//! `{φ_j}`, a real 1-cochain τ, and `η' = η − δτ`, the map is, in the chart
//! of the point's own ball `j`:
//!
//! ```text
//! w_k(x) = √φ_k(x) · exp(2πi · [ τ_{kj} + Σ_l φ_l(x)·η'_{kjl} ])
//! ```
//!
//! Chart consistency is an exercise in the cocycle identities: for charts
//! `j, j'`, the phase ratio of the k-th entries is
//! `τ_{kj'} − τ_{kj} + Σ_l φ_l(η'_{kj'l} − η'_{kjl})`; expanding
//! `(δη')_{kjj'l} = 0` turns the sum into a k-independent term minus
//! `η'_{kjj'}`, and `τ_{kj'} − τ_{kj} − η'_{kjj'} = τ_{jj'} − η_{kjj'}
//! ≡ τ_{jj'} (mod 1)` because η is INTEGRAL — so the entries change by one
//! global phase and the class in CPᴸ⁻¹ is well defined. This needs Σφ = 1
//! (normalised bumps) and integer η; τ is arbitrary, and choosing it as the
//! least-squares minimiser of ‖η − δτ‖ (harmonic smoothing, one CG solve on
//! the edge up-Laplacian δ¹ᵀδ¹) makes the phases as flat as the class
//! allows. The raw map lands in CPⁿ⁻¹ and is reduced by complex subspace
//! projection + renormalisation, exactly as the real projective path.
//!
//! All triangles live in VR(t): the three cover balls of a nerve 2-simplex
//! have pairwise-intersecting radius-t/2 balls, so every edge involved is
//! shorter than t and present in the complex — under a metric. The
//! quotient's coned scales are not a metric, so there a derived pair may be
//! absent from the skeleton; such triples are simply not nerve simplices and
//! contribute nothing (the quotient map is approximate either way).
//!
//! The coefficient-wise balanced lift from Z/p to Z is verified before it is
//! used. If it is not an integer cocycle, the construction fails explicitly
//! instead of returning a phase-inconsistent projective map.

use std::collections::HashMap;

use super::{rank_classes, validate_target_dim, ComplexProjectiveCoordinates, Config};
use crate::engine::simplex::{PackedSimplex, Simplex128, Simplex256};
use crate::engine::zp::{h2_cocycles, DimCocycles};
use crate::engine::BitCsrDistanceMatrix;
use crate::types::PersistenceInterval;
use crate::{Error, Result};

const CG_TOL: f64 = 1e-9;
const CG_MAX_ITERS: usize = 2000;
const MAX_POWER_ITERS: usize = 128;
const POWER_TOL: f32 = 1e-7;

/// Complex projective coordinates with the default [`Config`]. See
/// [`complex_projective_with`].
pub fn complex_projective(data: &[f32], n: usize) -> Result<ComplexProjectiveCoordinates> {
    complex_projective_with(data, n, &Config::default())
}

/// Complex projective coordinates from the H² mod-p class of rank
/// `config.class_index`, mapped into CP^(config.target_dim).
///
/// `data` follows the same layout contract as
/// [`crate::persistent_homology_with`]. Returns one complex unit row per
/// input point (interleaved re/im), defined up to a global phase per point.
/// `config.prime` is the coefficient field behind the integer lift.
pub fn complex_projective_with(
    data: &[f32],
    n: usize,
    config: &Config,
) -> Result<ComplexProjectiveCoordinates> {
    validate_target_dim(config.target_dim, n)?;
    validate_prime(config.prime)?;

    // Full pipeline at the certified radius, peel on (transparent here).
    let core_config = crate::Config {
        distance_matrix: config.distance_matrix,
        quotient: config.quotient,
        parallel: config.parallel,
        ..crate::Config::default()
    };
    let edges = crate::build_edges(data, n, 2, &core_config)?;
    let threshold = edges.threshold;
    let bitcsr = BitCsrDistanceMatrix::from_edge_list(edges);
    let h2 = h2_classes(&bitcsr, n, config.prime, config.parallel);

    // Class selection by persistence rank.
    let ranked = rank_classes(&h2.intervals, threshold);
    let Some(&class) = ranked.get(config.class_index) else {
        return Err(Error::NotEnoughClasses {
            requested: config.class_index + 1,
            available: ranked.len(),
            dim: 2,
        });
    };
    let interval = h2.intervals[class];
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

    let eta = lift_h2_cocycle(&bitcsr, &h2.cocycles[class], scale, config.prime)?;

    // Edge index of the t-skeleton, triangle list, and the harmonic
    // 1-cochain τ = argmin ‖η − δτ‖.
    let edges_at_scale = EdgeIndex::build(&bitcsr, n, scale);
    let triangles = triangle_list(&bitcsr, n, scale, &edges_at_scale);
    let tau = harmonic_tau(&edges_at_scale, &triangles, &eta);

    // The classifying map, one sparse complex row per point.
    let rows = classifying_rows(&bitcsr, n, scale * 0.5, &eta, &tau, &edges_at_scale);
    let m = config.target_dim + 1;
    let basis = top_subspace(&rows, n, m);
    let coordinates = project_rows(&rows, &basis, n, m);

    Ok(ComplexProjectiveCoordinates {
        coordinates,
        target_dim: config.target_dim,
        birth: interval.birth,
        death: interval.death,
        scale,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Mod-p H², converted out of the generic packing
// ═══════════════════════════════════════════════════════════════════════════════

struct H2Classes {
    intervals: Vec<PersistenceInterval>,
    /// `cocycles[i]` aligns with `intervals[i]`; entries `(a, b, c, coeff)`
    /// with `a > b > c` and `coeff` in `1..p`.
    cocycles: Vec<Vec<(u32, u32, u32, u16)>>,
}

fn h2_classes(bitcsr: &BitCsrDistanceMatrix, n: usize, p: u32, parallel: bool) -> H2Classes {
    if n <= crate::S128_MAX_POINTS {
        convert(h2_cocycles::<6, Simplex128>(bitcsr, p, parallel))
    } else {
        convert(h2_cocycles::<9, Simplex256>(bitcsr, p, parallel))
    }
}

fn convert<const K: usize, S: PackedSimplex<K>>(raw: DimCocycles<S>) -> H2Classes {
    let cocycles = raw
        .cocycles
        .into_iter()
        .map(|cocycle| {
            cocycle
                .into_iter()
                .map(|entry| {
                    let verts = entry.simplex.vertices();
                    (verts[0], verts[1], verts[2], entry.coeff)
                })
                .collect()
        })
        .collect();
    H2Classes {
        intervals: raw.intervals,
        cocycles,
    }
}

/// Restrict and balance-lift a mod-p 2-cocycle, then verify its integer
/// coboundary on the tetrahedra of the scale complex. Enumerating cofacets of
/// non-zero triangles avoids materialising every tetrahedron.
fn lift_h2_cocycle(
    bitcsr: &BitCsrDistanceMatrix,
    cocycle: &[(u32, u32, u32, u16)],
    scale: f32,
    prime: u32,
) -> Result<HashMap<(u32, u32, u32), f64>> {
    let p = prime as i64;
    let mut eta = HashMap::new();
    for &(a, b, c, coeff) in cocycle {
        let diameter = bitcsr
            .edge_dist(a, b)
            .max(bitcsr.edge_dist(a, c))
            .max(bitcsr.edge_dist(b, c));
        if diameter <= scale {
            let lifted = if (coeff as i64) <= p / 2 {
                coeff as i64
            } else {
                coeff as i64 - p
            };
            if lifted != 0 {
                eta.insert((a, b, c), lifted as f64);
            }
        }
    }

    let mut coboundary: HashMap<[u32; 4], i64> = HashMap::new();
    for (&(a, b, c), &value) in &eta {
        let value = value as i64;
        let triangle = [a, b, c];
        bitcsr.for_each_common_neighbor(&triangle, 3, -1, |w, extra| {
            if extra <= scale {
                let mut tetrahedron = [a, b, c, w];
                tetrahedron.sort_unstable_by(|x, y| y.cmp(x));
                let rank = tetrahedron
                    .iter()
                    .position(|&vertex| vertex == w)
                    .expect("inserted vertex must occur in its cofacet");
                let term = if rank % 2 == 0 { value } else { -value };
                *coboundary.entry(tetrahedron).or_insert(0) += term;
            }
            true
        });
    }
    if coboundary.values().any(|&value| value != 0) {
        return Err(Error::NonIntegralCocycleLift {
            dim: 2,
            prime,
            scale,
        });
    }
    Ok(eta)
}

// ═══════════════════════════════════════════════════════════════════════════════
// t-skeleton edge index and triangle list
// ═══════════════════════════════════════════════════════════════════════════════

/// Canonical `(u > v)` edges of the t-skeleton with dense indices.
#[derive(Debug)]
struct EdgeIndex {
    index: HashMap<(u32, u32), u32>,
}

impl EdgeIndex {
    fn build(bitcsr: &BitCsrDistanceMatrix, n: usize, scale: f32) -> Self {
        let mut index = HashMap::new();
        for v in 0..n {
            bitcsr.for_each_neighbor_above(v, |w, d| {
                if d <= scale {
                    let next = index.len() as u32;
                    index.insert((w, v as u32), next);
                }
                true
            });
        }
        Self { index }
    }

    #[inline]
    fn len(&self) -> usize {
        self.index.len()
    }

    /// Index of the canonical edge for an unordered distinct pair. Panics
    /// if absent — reserved for call sites where presence is guaranteed by
    /// construction (triangle enumeration and the cocycle's own triangles).
    #[inline]
    fn get(&self, a: u32, b: u32) -> u32 {
        let key = if a > b { (a, b) } else { (b, a) };
        *self
            .index
            .get(&key)
            .expect("triangle edge missing from the t-skeleton")
    }

    /// Index of the canonical edge, or `None` when the pair is not in the
    /// t-skeleton. Under a genuine metric every classifying-map triple has
    /// all three edges present (triangle inequality); the quotient's coned
    /// scales are NOT a metric, so derived pairs can be absent there.
    #[inline]
    fn try_get(&self, a: u32, b: u32) -> Option<u32> {
        let key = if a > b { (a, b) } else { (b, a) };
        self.index.get(&key).copied()
    }

    /// Alternating 1-cochain evaluation: τ_{ab} = −τ_{ba}, stored on the
    /// canonical (larger, smaller) orientation.
    #[inline]
    fn signed(&self, tau: &[f64], a: u32, b: u32) -> f64 {
        if a > b {
            tau[*self.index.get(&(a, b)).expect("missing edge") as usize]
        } else {
            -tau[*self.index.get(&(b, a)).expect("missing edge") as usize]
        }
    }

    /// [`Self::signed`] that treats an absent pair as zero — for transform
    /// time, where derived pairs can miss the cutoff by float noise (or, in
    /// principle, under non-metric weights).
    #[inline]
    fn signed_or_zero(&self, tau: &[f64], a: u32, b: u32) -> f64 {
        match self.try_get(a, b) {
            Some(index) => {
                let value = tau[index as usize];
                if a > b {
                    value
                } else {
                    -value
                }
            }
            None => 0.0,
        }
    }
}

/// Every triangle `[a > b > c]` of the t-skeleton as the edge-index triple
/// `(e_bc, e_ac, e_ab)`, so `(δτ)_t = τ[t.0] − τ[t.1] + τ[t.2]` (standard
/// alternating coboundary in the written descending order).
fn triangle_list(
    bitcsr: &BitCsrDistanceMatrix,
    n: usize,
    scale: f32,
    edges: &EdgeIndex,
) -> Vec<[u32; 3]> {
    let mut triangles = Vec::new();
    for v in 0..n {
        bitcsr.for_each_neighbor_above(v, |u, d_uv| {
            if d_uv <= scale {
                // u > v; close with common neighbours w < v so each triangle
                // [u > v > w] is produced exactly once.
                bitcsr.for_each_common_neighbor_edges(u, v as u32, -1, |w, d_uw, d_vw| {
                    if w < v as u32 && d_uw <= scale && d_vw <= scale {
                        triangles.push([
                            edges.get(v as u32, w),
                            edges.get(u, w),
                            edges.get(u, v as u32),
                        ]);
                    }
                    true
                });
            }
            true
        });
    }
    triangles
}

// ═══════════════════════════════════════════════════════════════════════════════
// Harmonic smoothing on 1-cochains: τ = argmin ‖η − δ¹τ‖²
// ═══════════════════════════════════════════════════════════════════════════════

/// Jacobi-preconditioned CG on the edge up-Laplacian `δ¹ᵀδ¹`. The system is
/// singular (gradients and harmonic cocycles are in the kernel), but the
/// right-hand side `δ¹ᵀη` is orthogonal to it, and ANY minimiser yields a
/// valid map — the well-definedness proof holds for arbitrary τ.
fn harmonic_tau(
    edges: &EdgeIndex,
    triangles: &[[u32; 3]],
    eta: &HashMap<(u32, u32, u32), f64>,
) -> Vec<f64> {
    let m = edges.len();
    let mut rhs = vec![0f64; m];
    for (&(a, b, c), &value) in eta {
        rhs[edges.get(b, c) as usize] += value;
        rhs[edges.get(a, c) as usize] -= value;
        rhs[edges.get(a, b) as usize] += value;
    }

    let norm_b: f64 = rhs.iter().map(|b| b * b).sum::<f64>().sqrt();
    if norm_b == 0.0 {
        return vec![0.0; m];
    }
    let tol = CG_TOL * norm_b;

    // Diagonal of δ¹ᵀδ¹ = number of triangles through each edge.
    let mut diagonal = vec![0f64; m];
    for t in triangles {
        diagonal[t[0] as usize] += 1.0;
        diagonal[t[1] as usize] += 1.0;
        diagonal[t[2] as usize] += 1.0;
    }

    let matvec = |x: &[f64], out: &mut [f64]| {
        out.fill(0.0);
        for t in triangles {
            let s = x[t[0] as usize] - x[t[1] as usize] + x[t[2] as usize];
            out[t[0] as usize] += s;
            out[t[1] as usize] -= s;
            out[t[2] as usize] += s;
        }
    };

    let mut x = vec![0f64; m];
    let mut r = rhs;
    let mut z = vec![0f64; m];
    let precondition = |r: &[f64], z: &mut [f64], diagonal: &[f64]| {
        for e in 0..r.len() {
            z[e] = r[e] / diagonal[e].max(1.0);
        }
    };
    precondition(&r, &mut z, &diagonal);
    let mut d = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let mut q = vec![0f64; m];

    for _ in 0..CG_MAX_ITERS {
        matvec(&d, &mut q);
        let dq: f64 = d.iter().zip(&q).map(|(a, b)| a * b).sum();
        if dq <= 0.0 {
            break;
        }
        let alpha = rz / dq;
        for e in 0..m {
            x[e] += alpha * d[e];
            r[e] -= alpha * q[e];
        }
        let norm_r: f64 = r.iter().map(|b| b * b).sum::<f64>().sqrt();
        if norm_r <= tol {
            break;
        }
        precondition(&r, &mut z, &diagonal);
        let rz_next: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz_next / rz;
        rz = rz_next;
        for e in 0..m {
            d[e] = z[e] + beta * d[e];
        }
    }
    x
}

// ═══════════════════════════════════════════════════════════════════════════════
// The classifying map — one sparse complex row per point
// ═══════════════════════════════════════════════════════════════════════════════

/// CSR rows of complex entries (interleaved re/im). Row `i` (chart `j = i`)
/// has an entry for every vertex `k` with `d(i, k) < r` plus `i` itself:
/// `√φ̂_k · exp(2πi·[τ_{ki} + Σ_l φ̂_l·η'_{kil}])` with normalised bumps φ̂
/// (the well-definedness argument needs Σφ̂ = 1). The diagonal entry is real
/// positive (τ_{ii} = 0, η'_{iil} = 0). Rows are unit vectors.
struct ComplexRows {
    offsets: Vec<u32>,
    cols: Vec<u32>,
    vals: Vec<[f32; 2]>,
}

/// Alternating evaluation of `η' = η − δτ` on an ordered distinct triple.
fn eta_prime(
    eta: &HashMap<(u32, u32, u32), f64>,
    tau: &[f64],
    edges: &EdgeIndex,
    i0: u32,
    i1: u32,
    i2: u32,
) -> f64 {
    // Sort descending, tracking the permutation sign.
    let (mut a, mut b, mut c) = (i0, i1, i2);
    let mut sign = 1.0f64;
    if a < b {
        std::mem::swap(&mut a, &mut b);
        sign = -sign;
    }
    if b < c {
        std::mem::swap(&mut b, &mut c);
        sign = -sign;
    }
    if a < b {
        std::mem::swap(&mut a, &mut b);
        sign = -sign;
    }
    let (Some(e_bc), Some(e_ac), Some(e_ab)) = (
        edges.try_get(b, c),
        edges.try_get(a, c),
        edges.try_get(a, b),
    ) else {
        // The triple is not a simplex of the filtration at scale t. Under a
        // metric this cannot happen (d(k, l) ≤ d(k, i) + d(i, l) < t), but
        // the quotient's coned scales violate the triangle inequality; a
        // missing triple is simply not part of the nerve — no contribution.
        return 0.0;
    };
    let eta_can = eta.get(&(a, b, c)).copied().unwrap_or(0.0);
    let delta_tau = tau[e_bc as usize] - tau[e_ac as usize] + tau[e_ab as usize];
    sign * (eta_can - delta_tau)
}

fn classifying_rows(
    bitcsr: &BitCsrDistanceMatrix,
    n: usize,
    r: f32,
    eta: &HashMap<(u32, u32, u32), f64>,
    tau: &[f64],
    edges: &EdgeIndex,
) -> ComplexRows {
    // Neighbourhoods within r (open), plus the point itself.
    let mut neighbors: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
    for v in 0..n {
        bitcsr.for_each_neighbor_above(v, |w, d| {
            if d < r {
                neighbors[v].push((w, d));
                neighbors[w as usize].push((v as u32, d));
            }
            true
        });
    }

    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0u32);
    let mut total = 0u64;
    for row in &neighbors {
        total += row.len() as u64 + 1;
        assert!(
            total <= u32::MAX as u64,
            "classifying map exceeds u32 entries"
        );
        offsets.push(total as u32);
    }
    let mut cols = vec![0u32; total as usize];
    let mut vals = vec![[0f32; 2]; total as usize];

    let two_pi = 2.0 * std::f64::consts::PI;
    for i in 0..n {
        let row = &neighbors[i];
        let bump_total: f64 = (r as f64) + row.iter().map(|&(_, d)| (r - d) as f64).sum::<f64>();

        // Normalised bump weights φ̂, self first.
        let base = offsets[i] as usize;
        cols[base] = i as u32;
        let phi_self = (r as f64) / bump_total;
        vals[base] = [(phi_self.sqrt()) as f32, 0.0]; // Θ = 0 on the diagonal

        for (slot, &(k, d_ik)) in row.iter().enumerate() {
            let phi_k = ((r - d_ik) as f64) / bump_total;
            // Θ = τ_{ki} + Σ_l φ̂_l · η'_{kil} over neighbours l of i
            // (l = i and l = k contribute zero by antisymmetry).
            let mut theta = edges.signed(tau, k, i as u32);
            for &(l, d_il) in row {
                if l != k {
                    let phi_l = ((r - d_il) as f64) / bump_total;
                    theta += phi_l * eta_prime(eta, tau, edges, k, i as u32, l);
                }
            }
            let magnitude = phi_k.sqrt();
            let phase = two_pi * theta;
            cols[base + 1 + slot] = k;
            vals[base + 1 + slot] = [
                (magnitude * phase.cos()) as f32,
                (magnitude * phase.sin()) as f32,
            ];
        }
    }

    ComplexRows {
        offsets,
        cols,
        vals,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Complex subspace iteration + row projection
// ═══════════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn cmul(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

#[inline(always)]
fn conj(a: [f32; 2]) -> [f32; 2] {
    [a[0], -a[1]]
}

/// Top-`m` principal complex subspace of `VᴴV` via block power iteration
/// with complex modified Gram–Schmidt. Column-major `n × m` complex entries.
fn top_subspace(rows: &ComplexRows, n: usize, m: usize) -> Vec<[f32; 2]> {
    let mut rng = xorshift(0x51ce_c0c0_a5a5_1234);
    let mut basis = vec![[0f32; 2]; n * m];
    for value in basis.iter_mut() {
        *value = [rng() - 0.5, rng() - 0.5];
    }
    orthonormalize(&mut basis, n, m, &mut rng);

    let mut y = vec![[0f32; 2]; n * m];
    let mut next = vec![[0f32; 2]; n * m];
    let mut previous = basis.clone();

    for _ in 0..MAX_POWER_ITERS {
        // y[c][i] = Σ_j V[i][j]·B[j][c]
        for i in 0..n {
            let range = rows.offsets[i] as usize..rows.offsets[i + 1] as usize;
            for c in 0..m {
                let col = &basis[c * n..(c + 1) * n];
                let mut acc = [0f32; 2];
                for e in range.clone() {
                    let t = cmul(rows.vals[e], col[rows.cols[e] as usize]);
                    acc[0] += t[0];
                    acc[1] += t[1];
                }
                y[c * n + i] = acc;
            }
        }
        // next[j][c] = Σ_i conj(V[i][j])·y[i][c]
        next.fill([0f32; 2]);
        for i in 0..n {
            let range = rows.offsets[i] as usize..rows.offsets[i + 1] as usize;
            for c in 0..m {
                let yi = y[c * n + i];
                let out = &mut next[c * n..(c + 1) * n];
                for e in range.clone() {
                    let t = cmul(conj(rows.vals[e]), yi);
                    let slot = &mut out[rows.cols[e] as usize];
                    slot[0] += t[0];
                    slot[1] += t[1];
                }
            }
        }
        std::mem::swap(&mut basis, &mut next);
        orthonormalize(&mut basis, n, m, &mut rng);

        // |⟨b_c, previous_c⟩| → 1 at convergence (phase-invariant).
        let mut worst = 0f32;
        for c in 0..m {
            let mut dot = [0f32; 2];
            for j in 0..n {
                let t = cmul(conj(basis[c * n + j]), previous[c * n + j]);
                dot[0] += t[0];
                dot[1] += t[1];
            }
            let misalignment = 1.0 - (dot[0] * dot[0] + dot[1] * dot[1]).sqrt();
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

/// Complex modified Gram–Schmidt; degenerate columns are reseeded.
fn orthonormalize(basis: &mut [[f32; 2]], n: usize, m: usize, rng: &mut impl FnMut() -> f32) {
    for c in 0..m {
        loop {
            for p in 0..c {
                // proj = ⟨b_p, b_c⟩ = Σ conj(b_p)·b_c
                let mut proj = [0f32; 2];
                for j in 0..n {
                    let t = cmul(conj(basis[p * n + j]), basis[c * n + j]);
                    proj[0] += t[0];
                    proj[1] += t[1];
                }
                for j in 0..n {
                    let t = cmul(proj, basis[p * n + j]);
                    basis[c * n + j][0] -= t[0];
                    basis[c * n + j][1] -= t[1];
                }
            }
            let norm: f32 = basis[c * n..(c + 1) * n]
                .iter()
                .map(|v| v[0] * v[0] + v[1] * v[1])
                .sum::<f32>()
                .sqrt();
            if norm > 1e-12 {
                for j in 0..n {
                    basis[c * n + j][0] /= norm;
                    basis[c * n + j][1] /= norm;
                }
                break;
            }
            for j in 0..n {
                basis[c * n + j] = [rng() - 0.5, rng() - 0.5];
            }
        }
    }
}

/// Project every row onto the basis (`u_c = ⟨b_c, v⟩`) and renormalise:
/// CPᵈ coordinates as interleaved re/im, row-major `(n, m, 2)`.
fn project_rows(rows: &ComplexRows, basis: &[[f32; 2]], n: usize, m: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * m * 2];
    for i in 0..n {
        let range = rows.offsets[i] as usize..rows.offsets[i + 1] as usize;
        let mut norm_sq = 0f32;
        let mut u = vec![[0f32; 2]; m];
        for (c, slot) in u.iter_mut().enumerate() {
            let col = &basis[c * n..(c + 1) * n];
            let mut acc = [0f32; 2];
            for e in range.clone() {
                let t = cmul(conj(col[rows.cols[e] as usize]), rows.vals[e]);
                acc[0] += t[0];
                acc[1] += t[1];
            }
            *slot = acc;
            norm_sq += acc[0] * acc[0] + acc[1] * acc[1];
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            for c in 0..m {
                out[(i * m + c) * 2] = u[c][0] / norm;
                out[(i * m + c) * 2 + 1] = u[c][1] / norm;
            }
        } else {
            out[i * m * 2] = 1.0; // arbitrary pole for a point orthogonal to the subspace
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

fn validate_prime(p: u32) -> Result<()> {
    let is_odd_prime = p >= 3 && p % 2 == 1 && p <= u16::MAX as u32 && {
        let mut divisor = 3u32;
        let mut prime = true;
        while divisor * divisor <= p {
            if p.is_multiple_of(divisor) {
                prime = false;
                break;
            }
            divisor += 2;
        }
        prime
    };
    if is_odd_prime {
        Ok(())
    } else {
        Err(Error::InvalidPrime(p))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fitted maps: out-of-sample evaluation
// ═══════════════════════════════════════════════════════════════════════════════

/// A fitted complex-projective map: [`complex_projective_with`] with the
/// model retained — training points, cover radius `r = scale / 2`, the
/// lifted 2-cocycle η, the harmonic 1-cochain τ, the t-skeleton edge index,
/// and the complex principal basis. A new point inside the cover gets the
/// classifying row of its nearest covered training vertex's chart, projected
/// onto the SAVED basis and renormalised; at a training point this
/// reproduces the fit row exactly. Points outside the cover evaluate to NaN
/// rows.
#[derive(Debug)]
pub struct ComplexProjectiveMap {
    train: Vec<f32>,
    dim: usize,
    n: usize,
    /// Training coordinates, interleaved re/im, row-major
    /// `(n, target_dim + 1, 2)` — identical to [`complex_projective_with`].
    pub coordinates: Vec<f32>,
    pub target_dim: usize,
    pub birth: f32,
    pub death: f32,
    pub scale: f32,
    r: f32,
    eta: HashMap<(u32, u32, u32), f64>,
    tau: Vec<f64>,
    edges_at_scale: EdgeIndex,
    basis: Vec<[f32; 2]>, // column-major n × (target_dim + 1)
    parallel: bool,
}

/// Fit a complex-projective map with the default [`Config`]. See
/// [`complex_projective_map_with`] and [`ComplexProjectiveMap`].
pub fn complex_projective_map(data: &[f32], n: usize) -> Result<ComplexProjectiveMap> {
    complex_projective_map_with(data, n, &Config::default())
}

/// Fit a complex-projective map: [`complex_projective_with`]'s pipeline with
/// the model retained. `distance_matrix` and `quotient` are rejected (see
/// [`crate::coords::circular_map_with`] — identical reasons, and under the
/// quotient the nerve premise additionally fails).
pub fn complex_projective_map_with(
    data: &[f32],
    n: usize,
    config: &Config,
) -> Result<ComplexProjectiveMap> {
    if config.distance_matrix {
        return Err(Error::NotImplemented(
            "fitted complex-projective maps on distance-matrix input",
        ));
    }
    if config.quotient {
        return Err(Error::NotImplemented(
            "fitted complex-projective maps on the quotient filtration",
        ));
    }
    validate_target_dim(config.target_dim, n)?;
    validate_prime(config.prime)?;

    let core_config = crate::Config {
        parallel: config.parallel,
        ..crate::Config::default()
    };
    let edges = crate::build_edges(data, n, 2, &core_config)?;
    let threshold = edges.threshold;
    let dim = data.len() / n;
    let bitcsr = BitCsrDistanceMatrix::from_edge_list(edges);
    let h2 = h2_classes(&bitcsr, n, config.prime, config.parallel);

    let ranked = rank_classes(&h2.intervals, threshold);
    let Some(&class) = ranked.get(config.class_index) else {
        return Err(Error::NotEnoughClasses {
            requested: config.class_index + 1,
            available: ranked.len(),
            dim: 2,
        });
    };
    let interval = h2.intervals[class];
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

    let eta = lift_h2_cocycle(&bitcsr, &h2.cocycles[class], scale, config.prime)?;

    let edges_at_scale = EdgeIndex::build(&bitcsr, n, scale);
    let triangles = triangle_list(&bitcsr, n, scale, &edges_at_scale);
    let tau = harmonic_tau(&edges_at_scale, &triangles, &eta);

    let r = scale * 0.5;
    let rows = classifying_rows(&bitcsr, n, r, &eta, &tau, &edges_at_scale);
    let m = config.target_dim + 1;
    let basis = top_subspace(&rows, n, m);
    let coordinates = project_rows(&rows, &basis, n, m);

    Ok(ComplexProjectiveMap {
        train: data.to_vec(),
        dim,
        n,
        coordinates,
        target_dim: config.target_dim,
        birth: interval.birth,
        death: interval.death,
        scale,
        r,
        eta,
        tau,
        edges_at_scale,
        basis,
        parallel: config.parallel,
    })
}

impl ComplexProjectiveMap {
    /// Number of training points.
    pub fn n_points(&self) -> usize {
        self.n
    }

    /// Ambient dimension of the fitted point cloud.
    pub fn ambient_dim(&self) -> usize {
        self.dim
    }

    /// Evaluate the fitted map on `m` new points (row-major `m × dim`).
    /// Returns interleaved re/im, row-major `(m, target_dim + 1, 2)` complex
    /// unit rows (up to a global phase); NaN rows outside the cover.
    pub fn transform(&self, queries: &[f32], m: usize) -> Result<Vec<f32>> {
        if queries.len() != m * self.dim {
            return Err(Error::ShapeMismatch {
                n: m,
                got: queries.len(),
                mode: "transform queries (length must be m × the fitted ambient dimension)",
            });
        }
        let width = self.target_dim + 1;
        let mut out = vec![f32::NAN; m * width * 2];
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

        let two_pi = 2.0 * std::f64::consts::PI;
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

            // Row entries √φ̂_k · exp(2πi·[τ_{k,chart} + Σ_l φ̂_l·η'_{k,chart,l}]),
            // accumulated directly against the saved complex basis.
            let mut u = vec![[0f32; 2]; width];
            for e in start..end {
                let k = cross.cols[e];
                let phi_k = (self.r - cross.dists[e]) as f64 / bump_total;
                let phase = if k == chart {
                    0.0
                } else {
                    let mut theta = self.edges_at_scale.signed_or_zero(&self.tau, k, chart);
                    for l_entry in start..end {
                        let l = cross.cols[l_entry];
                        if l != k && l != chart {
                            let phi_l = (self.r - cross.dists[l_entry]) as f64 / bump_total;
                            theta += phi_l
                                * eta_prime(
                                    &self.eta,
                                    &self.tau,
                                    &self.edges_at_scale,
                                    k,
                                    chart,
                                    l,
                                );
                        }
                    }
                    theta
                };
                let magnitude = phi_k.sqrt();
                let angle = two_pi * phase;
                let value = [
                    (magnitude * angle.cos()) as f32,
                    (magnitude * angle.sin()) as f32,
                ];
                for (c, slot) in u.iter_mut().enumerate() {
                    let t = cmul(conj(self.basis[c * self.n + k as usize]), value);
                    slot[0] += t[0];
                    slot[1] += t[1];
                }
            }
            let norm: f32 = u
                .iter()
                .map(|v| v[0] * v[0] + v[1] * v[1])
                .sum::<f32>()
                .sqrt();
            if norm > 1e-12 {
                for c in 0..width {
                    out[(q * width + c) * 2] = u[c][0] / norm;
                    out[(q * width + c) * 2 + 1] = u[c][1] / norm;
                }
            } else {
                out[q * width * 2] = 1.0;
                for slot in out[q * width * 2 + 1..(q + 1) * width * 2].iter_mut() {
                    *slot = 0.0;
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
    use crate::preprocess::edgelist::EdgeList;

    /// Fibonacci sphere: near-uniform sample of S², the canonical H² fixture.
    fn sphere(n: usize) -> Vec<f32> {
        let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
        let mut points = Vec::with_capacity(3 * n);
        for i in 0..n {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / (n as f32);
            let radius = (1.0 - z * z).sqrt();
            let theta = golden * i as f32;
            points.push(radius * theta.cos());
            points.push(radius * theta.sin());
            points.push(z);
        }
        points
    }

    /// CP¹ ≅ S²: map a homogeneous pair to the Riemann sphere.
    fn riemann(u: &[f32]) -> [f32; 3] {
        let (a, b) = ([u[0], u[1]], [u[2], u[3]]);
        let cross = cmul(conj(a), b);
        [
            a[0] * a[0] + a[1] * a[1] - b[0] * b[0] - b[1] * b[1],
            2.0 * cross[0],
            2.0 * cross[1],
        ]
    }

    #[test]
    fn balanced_h2_lift_must_be_an_integer_cocycle() {
        let n = 4;
        let mut distances = vec![1.0; n * n];
        for i in 0..n {
            distances[i * n + i] = 0.0;
        }
        let bitcsr = BitCsrDistanceMatrix::from_edge_list(EdgeList::from_distance_matrix(
            &distances, n, 1.0,
        ));
        let err = lift_h2_cocycle(&bitcsr, &[(3, 2, 1, 1)], 1.0, 3).unwrap_err();
        assert!(matches!(
            err,
            Error::NonIntegralCocycleLift {
                dim: 2,
                prime: 3,
                ..
            }
        ));
    }

    /// The sphere's H² class through CP¹: shapes, unit rows, and the image
    /// must spread over the Riemann sphere (every axis attains both signs) —
    /// a collapsed or phase-broken map fails this immediately.
    #[test]
    fn sphere_maps_onto_cp1() {
        let n = 64;
        let points = sphere(n);
        let cp = complex_projective_with(
            &points,
            n,
            &Config {
                target_dim: 1,
                ..Config::default()
            },
        )
        .unwrap();
        assert_eq!(cp.target_dim, 1);
        assert_eq!(cp.coordinates.len(), n * 2 * 2);
        assert!(cp.birth < cp.scale && cp.scale < cp.death);

        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for i in 0..n {
            let row = &cp.coordinates[i * 4..(i + 1) * 4];
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "row {i} has norm {norm}");
            let s = riemann(row);
            for a in 0..3 {
                min[a] = min[a].min(s[a]);
                max[a] = max[a].max(s[a]);
            }
        }
        for a in 0..3 {
            assert!(
                min[a] < 0.0 && max[a] > 0.0,
                "image axis {a} does not change sign: [{}, {}]",
                min[a],
                max[a]
            );
        }
    }

    #[test]
    fn deterministic_output() {
        let n = 48;
        let points = sphere(n);
        let a = complex_projective(&points, n).unwrap();
        let b = complex_projective(&points, n).unwrap();
        assert_eq!(a.coordinates, b.coordinates);
        assert_eq!(a.scale.to_bits(), b.scale.to_bits());
    }

    /// Through the quotient: the sphere's H² class survives the log-3
    /// interleaving and the pipeline stays structurally sound (per-point
    /// unit rows). The finer Riemann-sphere coverage assertion is left to
    /// the exact path — quotient phases are legitimately blockier.
    #[test]
    fn quotient_sphere_is_structurally_sound() {
        let n = 96;
        let points = sphere(n);
        let cp = complex_projective_with(
            &points,
            n,
            &Config {
                quotient: true,
                target_dim: 1,
                ..Config::default()
            },
        )
        .unwrap();
        assert_eq!(cp.coordinates.len(), n * 2 * 2);
        assert!(cp.birth < cp.scale && cp.scale < cp.death);
        for i in 0..n {
            let row = &cp.coordinates[i * 4..(i + 1) * 4];
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "row {i} has norm {norm}");
        }
    }

    /// The fitted map reproduces the fit on training points (identical
    /// chart, bumps, phases, and basis), gives finite unit rows on fresh
    /// sphere points, NaN outside the cover, and rejects unsupported
    /// configs.
    #[test]
    fn complex_projective_map_round_trip() {
        let n = 64;
        let points = sphere(n);
        let config = Config {
            target_dim: 1,
            ..Config::default()
        };
        let map = complex_projective_map_with(&points, n, &config).unwrap();
        let width = (map.target_dim + 1) * 2;

        let plain = complex_projective_with(&points, n, &config).unwrap();
        assert_eq!(map.coordinates, plain.coordinates);

        let back = map.transform(&points, n).unwrap();
        for (i, (got, want)) in back
            .chunks(width)
            .zip(map.coordinates.chunks(width))
            .enumerate()
        {
            for (a, b) in got.iter().zip(want) {
                assert!((a - b).abs() < 1e-4, "row {i}: {a} vs {b}");
            }
        }

        // Fresh points: the same sphere rotated slightly about z.
        let alpha = 0.1f32;
        let (sin, cos) = alpha.sin_cos();
        let mut fresh = Vec::with_capacity(3 * n);
        for i in 0..n {
            let (x, y, z) = (points[3 * i], points[3 * i + 1], points[3 * i + 2]);
            fresh.push(x * cos - y * sin);
            fresh.push(x * sin + y * cos);
            fresh.push(z);
        }
        let values = map.transform(&fresh, n).unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
        for row in values.chunks(width) {
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4);
        }

        let far = map.transform(&[50.0, 50.0, 50.0], 1).unwrap();
        assert!(far.iter().all(|v| v.is_nan()));

        let err = complex_projective_map_with(
            &points,
            n,
            &Config {
                quotient: true,
                target_dim: 1,
                ..Config::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::NotImplemented(_)));
    }

    #[test]
    fn selection_and_validation() {
        let n = 48;
        let points = sphere(n);

        let err = complex_projective_with(
            &points,
            n,
            &Config {
                class_index: 99,
                ..Config::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::NotEnoughClasses { dim: 2, .. }));

        let err = complex_projective_with(
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
                max: 47,
                points: 48
            }
        ));

        let err = complex_projective_with(
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
                got: 48,
                max: 47,
                points: 48
            }
        ));

        let err = complex_projective_with(
            &points,
            n,
            &Config {
                prime: 6,
                ..Config::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidPrime(6)));

        let reference = complex_projective(&points, n).unwrap();
        let err = complex_projective_with(
            &points,
            n,
            &Config {
                scale: Some(reference.birth * 0.5),
                ..Config::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidScale { .. }));
    }
}
