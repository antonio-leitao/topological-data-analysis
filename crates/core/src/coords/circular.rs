//! Circular coordinates: H¹ mod p → S¹, jointly Tᵏ.
//!
//! After de Silva–Morozov–Vejdemo-Johansson, *Persistent Cohomology and
//! Circular Coordinates* (2011), with the multi-class decorrelation of
//! Scoccola–Gakhar–Bush–Perea, *Toroidal Coordinates* (SoCG 2023), on the
//! full-vertex complex: persistent H¹ over Z/p with representative cocycles
//! ([`crate::engine::zp`]) → select classes by persistence rank → group them
//! into co-alive blocks → per block, restrict each cocycle to edges ≤ t,
//! lift the mod-p coefficients to integers in (−p/2, p/2), verify that
//! the lift is still an integer cocycle, harmonic-smooth
//! (one Jacobi-preconditioned conjugate-gradient solve of the t-skeleton
//! graph Laplacian per class, in f64), LLL-reduce the Gram matrix of the
//! harmonic representatives and apply the unimodular transform (least
//! squares is linear, so transformed solutions need no re-solve) → the
//! coordinate of vertex v in column i is 2π·(fᵢ(v) mod 1). Every data point
//! is a vertex, so values are assigned directly.
//!
//! "Toroidal" is not an API concept: singleton blocks are plain circular
//! coordinates, and blocks of size > 1 are automatically decorrelated. Within
//! such a block the columns form the lattice-reduced basis of the block's
//! classes — the map to Tᵏ is the meaningful object, not a 1:1
//! column-to-class correspondence.

use std::collections::HashMap;

use super::{rank_classes, CircularCoordinates, Config, SelectedClass};
use crate::engine::simplex::{PackedSimplex, Simplex128, Simplex256};
use crate::engine::zp::{h1_cocycles, DimCocycles};
use crate::engine::BitCsrDistanceMatrix;
use crate::types::PersistenceInterval;
use crate::{Error, Result};

type EdgeCocycle = HashMap<(u32, u32), f64>;
type BlockSolutions = (Vec<Vec<f64>>, Vec<EdgeCocycle>);

const CG_TOL: f64 = 1e-9;
const CG_MAX_ITERS: usize = 2000;
/// Lovász parameter for the lattice reduction.
const LLL_DELTA: f64 = 0.75;
/// Step cap for ill-conditioned Gram matrices (never reached on sane input;
/// bailing keeps the current — still unimodular — transform).
const LLL_MAX_STEPS: usize = 10_000;

/// Circular coordinates with the default [`Config`] (the single most
/// persistent H¹ class). See [`circular_with`].
pub fn circular(data: &[f32], n: usize) -> Result<CircularCoordinates> {
    circular_with(data, n, &Config::default())
}

/// Circular coordinates from H¹ classes selected by persistence rank.
///
/// `data` follows the same layout contract as
/// [`crate::persistent_homology_with`]. Returns one angle column per selected
/// class, in `[0, 2π)`, in selection order. Co-alive classes are jointly
/// harmonised and lattice-decorrelated (the `group` field of
/// [`SelectedClass`] names the blocks). `config.prime` is the coefficient
/// field behind the integer lift (odd prime). An empty explicit selection
/// (`classes: Some(vec![])`) returns an `(n, 0)` result. An explicit
/// `config.scale` keeps only the selected classes alive at it (dropped
/// classes lose their column); if none survive, the scale is reported
/// invalid against the first selected class.
pub fn circular_with(data: &[f32], n: usize, config: &Config) -> Result<CircularCoordinates> {
    validate_prime(config.prime)?;

    let selection = selection_ranks(config);
    if selection.is_empty() {
        return Ok(CircularCoordinates {
            coordinates: Vec::new(),
            k: 0,
            classes: Vec::new(),
        });
    }

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
    let h1 = h1_classes(&bitcsr, n, config.prime, config.parallel);

    // Resolve ranks to classes, then group into co-alive blocks.
    let chosen = resolve_selection(&selection, &h1, threshold)?;
    let (chosen, blocks) = make_blocks(chosen, config.scale)?;

    // Per block: joint harmonisation and (for size > 1) decorrelation.
    let k = chosen.len();
    let mut coordinates = vec![0f32; n * k];
    let mut classes = vec![
        SelectedClass {
            birth: 0.0,
            death: 0.0,
            scale: 0.0,
            group: 0,
        };
        k
    ];
    for (group, block) in blocks.iter().enumerate() {
        let fs = block_solutions(&bitcsr, n, &h1, &chosen, block, config.prime)?;
        for (position, &column) in block.members.iter().enumerate() {
            for v in 0..n {
                coordinates[v * k + column] = angle(fs[position][v]);
            }
            classes[column] = SelectedClass {
                birth: chosen[column].birth,
                death: chosen[column].death,
                scale: block.scale,
                group,
            };
        }
    }

    Ok(CircularCoordinates {
        coordinates,
        k,
        classes,
    })
}

#[derive(Clone, Copy)]
struct Chosen {
    class: usize,
    birth: f32,
    death: f32,
    death_cap: f32,
}

/// Selection ranks from the config, deduplicated preserving order.
fn selection_ranks(config: &Config) -> Vec<usize> {
    match &config.classes {
        Some(list) => {
            let mut out = Vec::with_capacity(list.len());
            for &rank in list {
                if !out.contains(&rank) {
                    out.push(rank);
                }
            }
            out
        }
        None => (0..config.n_classes).collect(),
    }
}

/// Resolve persistence ranks to classes with validated intervals.
fn resolve_selection(selection: &[usize], h1: &H1Classes, threshold: f32) -> Result<Vec<Chosen>> {
    let ranked = rank_classes(&h1.intervals, threshold);
    let mut chosen: Vec<Chosen> = Vec::with_capacity(selection.len());
    for &rank in selection {
        let Some(&class) = ranked.get(rank) else {
            return Err(Error::NotEnoughClasses {
                requested: rank + 1,
                available: ranked.len(),
                dim: 1,
            });
        };
        let interval = h1.intervals[class];
        let death_cap = if interval.death.is_finite() {
            interval.death
        } else {
            threshold
        };
        if interval.birth.partial_cmp(&death_cap) != Some(std::cmp::Ordering::Less) {
            return Err(Error::InvalidScale {
                scale: interval.birth,
                birth: interval.birth,
                death: death_cap,
            });
        }
        chosen.push(Chosen {
            class,
            birth: interval.birth,
            death: interval.death,
            death_cap,
        });
    }
    Ok(chosen)
}

/// Group the chosen classes into co-alive blocks (or one block at the
/// explicit scale, dropping classes not alive at it).
fn make_blocks(chosen: Vec<Chosen>, scale: Option<f32>) -> Result<(Vec<Chosen>, Vec<Block>)> {
    if let Some(scale) = scale {
        let first = &chosen[0];
        if scale.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(Error::InvalidScale {
                scale,
                birth: first.birth,
                death: first.death_cap,
            });
        }
        let kept: Vec<Chosen> = chosen
            .iter()
            .copied()
            .filter(|c| c.birth <= scale && scale < c.death_cap)
            .collect();
        if kept.is_empty() {
            return Err(Error::InvalidScale {
                scale,
                birth: first.birth,
                death: first.death_cap,
            });
        }
        let members = (0..kept.len()).collect();
        Ok((kept, vec![Block { members, scale }]))
    } else {
        let intervals: Vec<(f32, f32)> = chosen.iter().map(|c| (c.birth, c.death_cap)).collect();
        let blocks = group_by_coalive(&intervals);
        Ok((chosen, blocks))
    }
}

/// One toroidal block: selection-order column indices, jointly evaluated at
/// one scale.
struct Block {
    members: Vec<usize>,
    scale: f32,
}

#[inline]
fn angle(value: f64) -> f32 {
    let mut t = value.fract();
    if t < 0.0 {
        t += 1.0;
    }
    (2.0 * std::f64::consts::PI * t) as f32
}

// ═══════════════════════════════════════════════════════════════════════════════
// Co-alive grouping
// ═══════════════════════════════════════════════════════════════════════════════

/// Greedy co-alive grouping over `(birth, death)` intervals: repeatedly take
/// the scale where the most remaining classes are simultaneously alive
/// (ties: widest common intersection, then earliest), form a block at the
/// midpoint of the members' common intersection, remove them, repeat.
/// Every interval is nonempty by the caller's validation, so the sweep
/// always finds a block. Pairwise-overlapping intervals share a common point
/// (Helly on the line), so blocks are exactly the co-alive sets the design
/// promises.
fn group_by_coalive(intervals: &[(f32, f32)]) -> Vec<Block> {
    let mut remaining: Vec<usize> = (0..intervals.len()).collect();
    let mut blocks = Vec::new();

    while !remaining.is_empty() {
        let mut events: Vec<f32> = remaining
            .iter()
            .flat_map(|&i| [intervals[i].0, intervals[i].1])
            .collect();
        events.sort_by(f32::total_cmp);
        events.dedup_by(|a, b| a.to_bits() == b.to_bits());

        // Best window: (count, width, −birth) lexicographic.
        let mut best: Option<(usize, f32, f32, Vec<usize>)> = None;
        for window in events.windows(2) {
            if window[1] <= window[0] {
                continue;
            }
            let probe = 0.5 * (window[0] + window[1]);
            let members: Vec<usize> = remaining
                .iter()
                .copied()
                .filter(|&i| intervals[i].0 <= probe && probe < intervals[i].1)
                .collect();
            if members.is_empty() {
                continue;
            }
            let common_birth = members
                .iter()
                .map(|&i| intervals[i].0)
                .fold(f32::NEG_INFINITY, f32::max);
            let common_death = members
                .iter()
                .map(|&i| intervals[i].1)
                .fold(f32::INFINITY, f32::min);
            let width = common_death - common_birth;
            let better = match &best {
                None => true,
                Some((count, best_width, best_birth, _)) => {
                    members.len() > *count
                        || (members.len() == *count
                            && (width > *best_width
                                || (width == *best_width && common_birth < *best_birth)))
                }
            };
            if better {
                best = Some((members.len(), width, common_birth, members));
            }
        }

        let (_, width, common_birth, members) =
            best.expect("nonempty intervals always yield a window");
        blocks.push(Block {
            scale: common_birth + 0.5 * width,
            members: members.clone(),
        });
        remaining.retain(|i| !members.contains(i));
    }

    blocks
}

// ═══════════════════════════════════════════════════════════════════════════════
// Per-block solve: lift, harmonic smoothing, decorrelation
// ═══════════════════════════════════════════════════════════════════════════════

/// The harmonic potentials of a block's classes, decorrelated when the block
/// has more than one member. `fs[position]` aligns with
/// `block.members[position]`.
fn block_solutions(
    bitcsr: &BitCsrDistanceMatrix,
    n: usize,
    h1: &H1Classes,
    chosen: &[Chosen],
    block: &Block,
    prime: u32,
) -> Result<Vec<Vec<f64>>> {
    Ok(block_solutions_full(bitcsr, n, h1, chosen, block, prime)?.0)
}

/// [`block_solutions`] plus the lifted integer cocycles, both carried through
/// the LLL transform (η and f transform identically — everything is linear).
/// The cocycles are what a fitted [`CircularMap`] evaluates on new points.
fn block_solutions_full(
    bitcsr: &BitCsrDistanceMatrix,
    n: usize,
    h1: &H1Classes,
    chosen: &[Chosen],
    block: &Block,
    prime: u32,
) -> Result<BlockSolutions> {
    let adjacency = skeleton(bitcsr, n, block.scale);
    let p = prime as i64;

    let mut etas: Vec<EdgeCocycle> = Vec::with_capacity(block.members.len());
    let mut fs: Vec<Vec<f64>> = Vec::with_capacity(block.members.len());
    for &column in &block.members {
        // Integer lift of the restricted cocycle: coefficients in
        // (−p/2, p/2), keyed by the canonical (u > v) edge.
        let mut eta: EdgeCocycle = HashMap::new();
        for &(u, v, c) in &h1.cocycles[chosen[column].class] {
            if bitcsr.edge_dist(u, v) <= block.scale {
                let lifted = if (c as i64) <= p / 2 {
                    c as i64
                } else {
                    c as i64 - p
                };
                if lifted != 0 {
                    eta.insert((u, v), lifted as f64);
                }
            }
        }
        verify_integral_lift(bitcsr, block.scale, &eta, prime)?;

        // Harmonic smoothing: minimise ‖η − δf‖² with the orientation
        // δf([u > v]) = f(v) − f(u). Normal equations: L f = δᵀη.
        let mut rhs = vec![0f64; n];
        for (&(u, v), &value) in &eta {
            rhs[v as usize] += value;
            rhs[u as usize] -= value;
        }
        fs.push(laplacian_cg(&adjacency, &rhs));
        etas.push(eta);
    }

    if fs.len() > 1 {
        // Toroidal decorrelation: LLL on the Gram matrix of the harmonic
        // representatives; least squares is linear in η, so the transformed
        // potentials are the exact solutions for the transformed cocycles.
        let gram = harmonic_gram(&adjacency, &etas, &fs);
        let transform = lll_reduce(gram);
        fs = apply_transform(&transform, &fs, n);
        etas = apply_transform_etas(&transform, &etas);
    }
    Ok((fs, etas))
}

/// The coefficient-wise balanced lift from Z/p to Z is only a heuristic: a
/// mod-p cocycle can acquire a non-zero integer coboundary. Never feed such a
/// cochain to the harmonic-coordinate construction, because the resulting
/// circle-valued map need not be continuous.
fn verify_integral_lift(
    bitcsr: &BitCsrDistanceMatrix,
    scale: f32,
    eta: &EdgeCocycle,
    prime: u32,
) -> Result<()> {
    let mut coboundary: HashMap<[u32; 3], i64> = HashMap::new();
    for (&(u, v), &value) in eta {
        let value = value as i64;
        bitcsr.for_each_common_neighbor_edges(u, v, -1, |w, d_uw, d_vw| {
            if d_uw <= scale && d_vw <= scale {
                let mut triangle = [u, v, w];
                triangle.sort_unstable_by(|a, b| b.cmp(a));
                let rank = triangle
                    .iter()
                    .position(|&vertex| vertex == w)
                    .expect("inserted vertex must occur in its cofacet");
                let term = if rank % 2 == 0 { value } else { -value };
                *coboundary.entry(triangle).or_insert(0) += term;
            }
            true
        });
    }
    if coboundary.values().any(|&value| value != 0) {
        Err(Error::NonIntegralCocycleLift {
            dim: 1,
            prime,
            scale,
        })
    } else {
        Ok(())
    }
}

/// The unimodular transform applied to the lifted integer cocycles (integer
/// linear combinations of edge maps; zero entries are dropped).
fn apply_transform_etas(
    transform: &[Vec<i64>],
    etas: &[HashMap<(u32, u32), f64>],
) -> Vec<HashMap<(u32, u32), f64>> {
    let kb = etas.len();
    let mut out = Vec::with_capacity(kb);
    for row in transform.iter().take(kb) {
        let mut merged: HashMap<(u32, u32), f64> = HashMap::new();
        for (j, &coefficient) in row.iter().enumerate() {
            if coefficient != 0 {
                let c = coefficient as f64;
                for (&edge, &value) in &etas[j] {
                    *merged.entry(edge).or_insert(0.0) += c * value;
                }
            }
        }
        merged.retain(|_, value| *value != 0.0);
        out.push(merged);
    }
    out
}

/// Gram matrix of the harmonic representatives h = η − δf over the block's
/// skeleton edges.
#[allow(clippy::needless_range_loop)] // paired matrix indices are clearer here
fn harmonic_gram(
    adjacency: &Skeleton,
    etas: &[HashMap<(u32, u32), f64>],
    fs: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let kb = fs.len();
    let n = fs[0].len();
    let mut gram = vec![vec![0f64; kb]; kb];
    let mut h = vec![0f64; kb];

    for v in 0..n {
        let range = adjacency.offsets[v] as usize..adjacency.offsets[v + 1] as usize;
        for &w in &adjacency.cols[range] {
            let wu = w as usize;
            if wu <= v {
                continue; // count each edge once, canonical (w > v)
            }
            for i in 0..kb {
                let eta_e = etas[i].get(&(w, v as u32)).copied().unwrap_or(0.0);
                h[i] = eta_e - (fs[i][v] - fs[i][wu]);
            }
            for i in 0..kb {
                for j in i..kb {
                    gram[i][j] += h[i] * h[j];
                }
            }
        }
    }
    for i in 0..kb {
        for j in 0..i {
            gram[i][j] = gram[j][i];
        }
    }
    gram
}

fn apply_transform(transform: &[Vec<i64>], fs: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let kb = fs.len();
    let mut out = Vec::with_capacity(kb);
    for row in transform.iter().take(kb) {
        let mut f = vec![0f64; n];
        for (j, &coefficient) in row.iter().enumerate() {
            if coefficient != 0 {
                let c = coefficient as f64;
                for v in 0..n {
                    f[v] += c * fs[j][v];
                }
            }
        }
        out.push(f);
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════════════
// LLL over a Gram matrix
// ═══════════════════════════════════════════════════════════════════════════════

/// LLL (δ = [`LLL_DELTA`]) in the Gram-only formulation: row operations
/// update `G ← E G Eᵀ` alongside `U ← E U`, so no explicit basis vectors are
/// needed. Returns the unimodular `U` whose rows express the reduced basis
/// in input coordinates. `k` is a handful, so the from-scratch Gram–Schmidt
/// per step (O(k³)) is irrelevant.
fn lll_reduce(mut gram: Vec<Vec<f64>>) -> Vec<Vec<i64>> {
    let k = gram.len();
    let mut u: Vec<Vec<i64>> = (0..k)
        .map(|i| (0..k).map(|j| i64::from(i == j)).collect())
        .collect();
    if k < 2 {
        return u;
    }

    let mut steps = 0usize;
    let mut i = 1usize;
    while i < k {
        steps += 1;
        if steps > LLL_MAX_STEPS {
            break; // ill-conditioned input; U is still unimodular
        }

        let (mu, _) = gram_schmidt(&gram);
        for j in (0..i).rev() {
            let r = mu[i][j].round();
            if r != 0.0 && r.is_finite() {
                row_subtract(&mut gram, &mut u, i, j, r as i64);
            }
        }
        // Recompute after size reduction (O(k³) with k a handful: clarity
        // over caching).
        let (mu, bstar) = gram_schmidt(&gram);

        if bstar[i] >= (LLL_DELTA - mu[i][i - 1] * mu[i][i - 1]) * bstar[i - 1] {
            i += 1;
        } else {
            swap_basis(&mut gram, &mut u, i, i - 1);
            i = i.max(2) - 1; // max(i − 1, 1)
        }
    }
    u
}

/// Gram–Schmidt data (μ and ‖b*‖²) straight from a Gram matrix.
#[allow(clippy::needless_range_loop)] // triangular recurrence uses prior rows
fn gram_schmidt(gram: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let k = gram.len();
    let mut mu = vec![vec![0f64; k]; k];
    let mut bstar = vec![0f64; k];
    for i in 0..k {
        bstar[i] = gram[i][i];
        for j in 0..i {
            let mut numerator = gram[i][j];
            for l in 0..j {
                numerator -= mu[i][l] * mu[j][l] * bstar[l];
            }
            mu[i][j] = if bstar[j].abs() > 1e-12 {
                numerator / bstar[j]
            } else {
                0.0
            };
            bstar[i] -= mu[i][j] * mu[i][j] * bstar[j];
        }
    }
    (mu, bstar)
}

/// `b_i ← b_i − r·b_j` on the Gram matrix (row and column) and the transform.
#[allow(clippy::needless_range_loop)] // simultaneous row/column operation
fn row_subtract(gram: &mut [Vec<f64>], u: &mut [Vec<i64>], i: usize, j: usize, r: i64) {
    let k = gram.len();
    let rf = r as f64;
    for c in 0..k {
        u[i][c] -= r * u[j][c];
    }
    for c in 0..k {
        gram[i][c] -= rf * gram[j][c];
    }
    for row in gram.iter_mut().take(k) {
        row[i] -= rf * row[j];
    }
}

fn swap_basis(gram: &mut [Vec<f64>], u: &mut [Vec<i64>], a: usize, b: usize) {
    u.swap(a, b);
    gram.swap(a, b);
    for row in gram.iter_mut() {
        row.swap(a, b);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fitted maps: out-of-sample evaluation
// ═══════════════════════════════════════════════════════════════════════════════

/// A fitted circular-coordinates map.
///
/// Fitting is [`circular_with`] with the model retained: training points,
/// per-column cover radius `r = scale / 2`, harmonic potentials `f`, and the
/// lifted integer cocycles `η` (both post-decorrelation). A new point `x`
/// inside the cover evaluates, in the chart of its nearest covered training
/// vertex `j`,
///
/// ```text
/// θ(x) = f_j + Σ_k φ_k(x) · ( f_k − f_j − η(j, k) )   (mod 1, × 2π)
/// ```
///
/// which is chart-independent mod 1 (switching charts shifts by an integer,
/// by δη = 0 and Σφ = 1), concentrates to `f_j` at the chart centre, and
/// changes by exactly −(η − δf) across an edge — the same map the training
/// coordinates sample. At training points the transform returns the
/// PoU-smoothed value: close to, but not bit-identical with, the fit output
/// (the gap is the local harmonic residual). Points farther than `r` from
/// every training vertex are outside the cover and evaluate to NaN.
#[derive(Debug)]
pub struct CircularMap {
    train: Vec<f32>,
    dim: usize,
    n: usize,
    /// Number of coordinate columns.
    pub k: usize,
    /// Training coordinates, row-major `(n, k)` — identical to what
    /// [`circular_with`] returns for the same input and config.
    pub coordinates: Vec<f32>,
    /// Per-column metadata (interval, scale, toroidal block).
    pub classes: Vec<SelectedClass>,
    potentials: Vec<Vec<f64>>,
    cocycles: Vec<SortedCocycle>,
    parallel: bool,
}

/// A lifted integer cocycle as a sorted edge array: canonical `(u > v)` keys
/// packed `(u << 32) | v`, alternating evaluation.
#[derive(Debug)]
struct SortedCocycle {
    keys: Vec<u64>,
    vals: Vec<f64>,
}

impl SortedCocycle {
    fn from_map(map: &HashMap<(u32, u32), f64>) -> Self {
        let mut entries: Vec<(u64, f64)> = map
            .iter()
            .map(|(&(u, v), &value)| (((u as u64) << 32) | v as u64, value))
            .collect();
        entries.sort_unstable_by_key(|e| e.0);
        Self {
            keys: entries.iter().map(|e| e.0).collect(),
            vals: entries.iter().map(|e| e.1).collect(),
        }
    }

    /// Alternating η(a, b): the stored value on the canonical orientation
    /// (larger index first), negated when evaluated against it.
    #[inline]
    fn eval(&self, a: u32, b: u32) -> f64 {
        if a == b {
            return 0.0;
        }
        let (hi, lo, sign) = if a > b { (a, b, 1.0) } else { (b, a, -1.0) };
        let key = ((hi as u64) << 32) | lo as u64;
        match self.keys.binary_search(&key) {
            Ok(index) => sign * self.vals[index],
            Err(_) => 0.0,
        }
    }
}

/// Fit a circular-coordinates map with the default [`Config`]. See
/// [`circular_map_with`] and [`CircularMap`].
pub fn circular_map(data: &[f32], n: usize) -> Result<CircularMap> {
    circular_map_with(data, n, &Config::default())
}

/// Fit a circular-coordinates map: [`circular_with`]'s pipeline with the
/// model retained for out-of-sample evaluation.
///
/// Not supported here: `distance_matrix` input (evaluating new points would
/// need user-supplied cross distances) and `quotient` (quotient scales are
/// not ambient distances, so the cover test is meaningless); both are
/// rejected explicitly.
pub fn circular_map_with(data: &[f32], n: usize, config: &Config) -> Result<CircularMap> {
    if config.distance_matrix {
        return Err(Error::NotImplemented(
            "fitted circular maps on distance-matrix input",
        ));
    }
    if config.quotient {
        return Err(Error::NotImplemented(
            "fitted circular maps on the quotient filtration",
        ));
    }
    validate_prime(config.prime)?;

    let core_config = crate::Config {
        parallel: config.parallel,
        ..crate::Config::default()
    };
    let edges = crate::build_edges(data, n, 1, &core_config)?;
    let threshold = edges.threshold;
    let dim = data.len() / n;
    let bitcsr = BitCsrDistanceMatrix::from_edge_list(edges);
    let h1 = h1_classes(&bitcsr, n, config.prime, config.parallel);

    let selection = selection_ranks(config);
    let (chosen, blocks) = if selection.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        let chosen = resolve_selection(&selection, &h1, threshold)?;
        make_blocks(chosen, config.scale)?
    };

    let k = chosen.len();
    let mut coordinates = vec![0f32; n * k];
    let mut classes = vec![
        SelectedClass {
            birth: 0.0,
            death: 0.0,
            scale: 0.0,
            group: 0,
        };
        k
    ];
    let mut potentials: Vec<Vec<f64>> = vec![Vec::new(); k];
    let mut cocycles: Vec<SortedCocycle> = (0..k)
        .map(|_| SortedCocycle {
            keys: Vec::new(),
            vals: Vec::new(),
        })
        .collect();

    for (group, block) in blocks.iter().enumerate() {
        let (mut fs, etas) = block_solutions_full(&bitcsr, n, &h1, &chosen, block, config.prime)?;
        for (position, &column) in block.members.iter().enumerate() {
            for v in 0..n {
                coordinates[v * k + column] = angle(fs[position][v]);
            }
            classes[column] = SelectedClass {
                birth: chosen[column].birth,
                death: chosen[column].death,
                scale: block.scale,
                group,
            };
            potentials[column] = std::mem::take(&mut fs[position]);
            cocycles[column] = SortedCocycle::from_map(&etas[position]);
        }
    }

    Ok(CircularMap {
        train: data.to_vec(),
        dim,
        n,
        k,
        coordinates,
        classes,
        potentials,
        cocycles,
        parallel: config.parallel,
    })
}

impl CircularMap {
    /// Number of training points.
    pub fn n_points(&self) -> usize {
        self.n
    }

    /// Ambient dimension of the fitted point cloud.
    pub fn ambient_dim(&self) -> usize {
        self.dim
    }

    /// Evaluate the fitted map on `m` new points (row-major `m × dim`).
    /// Returns row-major `(m, k)` angles in `[0, 2π)`; entries are NaN where
    /// the point lies outside a column's cover (farther than `r` from every
    /// training point).
    pub fn transform(&self, queries: &[f32], m: usize) -> Result<Vec<f32>> {
        if queries.len() != m * self.dim {
            return Err(Error::ShapeMismatch {
                n: m,
                got: queries.len(),
                mode: "transform queries (length must be m × the fitted ambient dimension)",
            });
        }
        let mut out = vec![f32::NAN; m * self.k];
        if self.k == 0 || m == 0 {
            return Ok(out);
        }

        let r_max = self
            .classes
            .iter()
            .map(|class| 0.5 * class.scale)
            .fold(0f32, f32::max);
        let cross = crate::preprocess::edgelist::cross_neighbors(
            queries,
            m,
            &self.train,
            self.n,
            self.dim,
            r_max,
            self.parallel,
        );

        for q in 0..m {
            let start = cross.offsets[q] as usize;
            let end = cross.offsets[q + 1] as usize;
            for column in 0..self.k {
                let r = 0.5 * self.classes[column].scale;

                // Chart = nearest covered training vertex (ties: lowest
                // index, from the ascending-index row order).
                let mut chart: Option<(u32, f32)> = None;
                let mut bump_total = 0f64;
                for e in start..end {
                    let dist = cross.dists[e];
                    if dist < r {
                        bump_total += (r - dist) as f64;
                        let better = match chart {
                            None => true,
                            Some((_, best)) => dist < best,
                        };
                        if better {
                            chart = Some((cross.cols[e], dist));
                        }
                    }
                }
                let Some((j, _)) = chart else {
                    continue; // outside this column's cover: stays NaN
                };

                let f = &self.potentials[column];
                let eta = &self.cocycles[column];
                let f_j = f[j as usize];
                let mut theta = f_j;
                for e in start..end {
                    let dist = cross.dists[e];
                    if dist < r {
                        let v = cross.cols[e];
                        let phi = (r - dist) as f64 / bump_total;
                        theta += phi * (f[v as usize] - f_j - eta.eval(j, v));
                    }
                }
                out[q * self.k + column] = angle(theta);
            }
        }
        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Mod-p persistence, converted out of the generic packing
// ═══════════════════════════════════════════════════════════════════════════════

/// H¹ intervals plus cocycles as canonical edges: the non-generic form the
/// coordinate assembly consumes (the simplex packing stays inside the
/// dispatch arms).
struct H1Classes {
    intervals: Vec<PersistenceInterval>,
    /// `cocycles[i]` aligns with `intervals[i]`; entries `(u, v, coeff)` with
    /// `u > v` and `coeff` in `1..p`.
    cocycles: Vec<Vec<(u32, u32, u16)>>,
}

fn h1_classes(bitcsr: &BitCsrDistanceMatrix, n: usize, p: u32, parallel: bool) -> H1Classes {
    if n <= crate::S128_MAX_POINTS {
        convert(h1_cocycles::<6, Simplex128>(bitcsr, p, parallel))
    } else {
        convert(h1_cocycles::<9, Simplex256>(bitcsr, p, parallel))
    }
}

fn convert<const K: usize, S: PackedSimplex<K>>(raw: DimCocycles<S>) -> H1Classes {
    let cocycles = raw
        .cocycles
        .into_iter()
        .map(|cocycle| {
            cocycle
                .into_iter()
                .map(|entry| {
                    let verts = entry.simplex.vertices();
                    (verts[0], verts[1], entry.coeff)
                })
                .collect()
        })
        .collect();
    H1Classes {
        intervals: raw.intervals,
        cocycles,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// t-skeleton and the Laplacian solve
// ═══════════════════════════════════════════════════════════════════════════════

/// Symmetric CSR adjacency (neighbour ids only) of the VR skeleton at
/// `scale`.
struct Skeleton {
    offsets: Vec<u32>,
    cols: Vec<u32>,
}

fn skeleton(bitcsr: &BitCsrDistanceMatrix, n: usize, scale: f32) -> Skeleton {
    let mut deg = vec![0u32; n];
    for v in 0..n {
        bitcsr.for_each_neighbor_above(v, |w, d| {
            if d <= scale {
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
        assert!(total <= u32::MAX as u64, "t-skeleton exceeds u32 entries");
        offsets.push(total as u32);
    }

    let mut cols = vec![0u32; total as usize];
    let mut cursor: Vec<u32> = offsets[..n].to_vec();
    for v in 0..n {
        bitcsr.for_each_neighbor_above(v, |w, d| {
            if d <= scale {
                cols[cursor[v] as usize] = w;
                cursor[v] += 1;
                let wi = w as usize;
                cols[cursor[wi] as usize] = v as u32;
                cursor[wi] += 1;
            }
            true
        });
    }

    Skeleton { offsets, cols }
}

/// Jacobi-preconditioned conjugate gradients on the graph Laplacian
/// `(Lf)(v) = deg(v)·f(v) − Σ_{w ~ v} f(w)`. The system is singular (constant
/// per component), but the right-hand side of a coboundary fit sums to zero
/// on every component, so CG converges to a valid minimiser. f64 throughout —
/// the result is taken mod 1, so accumulated error must stay far below one.
fn laplacian_cg(adjacency: &Skeleton, rhs: &[f64]) -> Vec<f64> {
    let n = rhs.len();
    let degree = |v: usize| (adjacency.offsets[v + 1] - adjacency.offsets[v]) as f64;
    let matvec = |x: &[f64], out: &mut [f64]| {
        for v in 0..n {
            let mut acc = degree(v) * x[v];
            let range = adjacency.offsets[v] as usize..adjacency.offsets[v + 1] as usize;
            for &w in &adjacency.cols[range] {
                acc -= x[w as usize];
            }
            out[v] = acc;
        }
    };

    let norm_b: f64 = rhs.iter().map(|b| b * b).sum::<f64>().sqrt();
    if norm_b == 0.0 {
        return vec![0.0; n];
    }
    let tol = CG_TOL * norm_b;

    let precondition = |r: &[f64], z: &mut [f64]| {
        for v in 0..n {
            z[v] = r[v] / degree(v).max(1.0);
        }
    };

    let mut x = vec![0f64; n];
    let mut r = rhs.to_vec();
    let mut z = vec![0f64; n];
    precondition(&r, &mut z);
    let mut d = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let mut q = vec![0f64; n];

    for _ in 0..CG_MAX_ITERS {
        matvec(&d, &mut q);
        let dq: f64 = d.iter().zip(&q).map(|(a, b)| a * b).sum();
        if dq <= 0.0 {
            break; // numerically exhausted (kernel direction)
        }
        let alpha = rz / dq;
        for v in 0..n {
            x[v] += alpha * d[v];
            r[v] -= alpha * q[v];
        }
        let norm_r: f64 = r.iter().map(|b| b * b).sum::<f64>().sqrt();
        if norm_r <= tol {
            break;
        }
        precondition(&r, &mut z);
        let rz_next: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz_next / rz;
        rz = rz_next;
        for v in 0..n {
            d[v] = z[v] + beta * d[v];
        }
    }
    x
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
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::edgelist::EdgeList;

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

    fn two_circles(per: usize) -> Vec<f32> {
        let mut points = circle(per);
        for i in 0..per {
            let theta = 2.0 * std::f32::consts::PI * (i as f32) / (per as f32);
            points.push(5.0 + theta.cos());
            points.push(theta.sin());
        }
        points
    }

    /// Winding number of a column of angles along a sampled closed loop.
    fn winding(coordinates: &[f32], k: usize, column: usize, range: std::ops::Range<usize>) -> f64 {
        let ids: Vec<usize> = range.collect();
        let m = ids.len();
        let mut total = 0f64;
        for i in 0..m {
            let a = coordinates[ids[i] * k + column] as f64;
            let b = coordinates[ids[(i + 1) % m] * k + column] as f64;
            let mut d = b - a;
            let pi = std::f64::consts::PI;
            if d > pi {
                d -= 2.0 * pi;
            } else if d < -pi {
                d += 2.0 * pi;
            }
            total += d;
        }
        total / (2.0 * std::f64::consts::PI)
    }

    fn winds(w: f64) -> bool {
        (w.abs() - 1.0).abs() < 1e-3
    }
    fn flat(w: f64) -> bool {
        w.abs() < 1e-3
    }

    #[test]
    fn balanced_lift_must_be_an_integer_cocycle() {
        let distances = [
            0.0, 1.0, 1.0, //
            1.0, 0.0, 1.0, //
            1.0, 1.0, 0.0,
        ];
        let bitcsr = BitCsrDistanceMatrix::from_edge_list(EdgeList::from_distance_matrix(
            &distances, 3, 1.0,
        ));

        let bad = HashMap::from([((2, 1), 1.0)]);
        let err = verify_integral_lift(&bitcsr, 1.0, &bad, 3).unwrap_err();
        assert!(matches!(
            err,
            Error::NonIntegralCocycleLift {
                dim: 1,
                prime: 3,
                ..
            }
        ));

        // δf with f=(0, 1, 1): η(1,0)=η(2,0)=-1 and η(2,1)=0.
        let exact = HashMap::from([((1, 0), -1.0), ((2, 0), -1.0)]);
        verify_integral_lift(&bitcsr, 1.0, &exact, 3).unwrap();
    }

    /// The circle's dominant H¹ class must produce a coordinate that winds
    /// exactly once around S¹ — the defining property of circular
    /// coordinates, invariant to harmonic wobble.
    #[test]
    fn circle_coordinate_winds_once() {
        let n = 48;
        let points = circle(n);
        let cc = circular(&points, n).unwrap();
        assert_eq!(cc.k, 1);
        assert_eq!(cc.coordinates.len(), n);
        assert_eq!(cc.classes.len(), 1);
        let class = cc.classes[0];
        assert!(class.birth < class.scale && class.scale < class.death);
        assert!(cc
            .coordinates
            .iter()
            .all(|&t| (0.0..2.0 * std::f32::consts::PI + 1e-4).contains(&t)));

        let w = winding(&cc.coordinates, 1, 0, 0..n);
        assert!(winds(w), "coordinate must wind once, got {w}");
    }

    #[test]
    fn deterministic_output() {
        let n = 40;
        let points = circle(n);
        let a = circular(&points, n).unwrap();
        let b = circular(&points, n).unwrap();
        assert_eq!(a.coordinates, b.coordinates);
    }

    #[test]
    fn selection_and_validation() {
        let n = 32;
        let points = circle(n);

        for bad_prime in [1u32, 2, 4, 9, 100_003] {
            let err = circular_with(
                &points,
                n,
                &Config {
                    prime: bad_prime,
                    ..Config::default()
                },
            )
            .unwrap_err();
            assert!(matches!(err, Error::InvalidPrime(_)), "prime {bad_prime}");
        }

        // A single circle has one dominant class: asking for two must report
        // the availability.
        let err = circular_with(
            &points,
            n,
            &Config {
                classes: Some(vec![0, 99]),
                ..Config::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::NotEnoughClasses { .. }));

        let empty = circular_with(
            &points,
            n,
            &Config {
                classes: Some(Vec::new()),
                ..Config::default()
            },
        )
        .unwrap();
        assert_eq!(empty.k, 0);
        assert!(empty.coordinates.is_empty());

        // Duplicated ranks deduplicate to a single column.
        let dedup = circular_with(
            &points,
            n,
            &Config {
                classes: Some(vec![0, 0]),
                ..Config::default()
            },
        )
        .unwrap();
        assert_eq!(dedup.k, 1);

        // Explicit rank 0 equals the default selection.
        let explicit = circular_with(
            &points,
            n,
            &Config {
                classes: Some(vec![0]),
                ..Config::default()
            },
        )
        .unwrap();
        let default = circular(&points, n).unwrap();
        assert_eq!(explicit.coordinates, default.coordinates);
    }

    /// Two circles side by side, single class: the dominant class winds on
    /// exactly one of the circles and is ~flat on the other.
    #[test]
    fn two_circles_single_class() {
        let per = 24;
        let points = two_circles(per);
        let n = 2 * per;

        let cc = circular(&points, n).unwrap();
        let first = winding(&cc.coordinates, 1, 0, 0..per);
        let second = winding(&cc.coordinates, 1, 0, per..n);
        assert!(
            (winds(first) && flat(second)) || (winds(second) && flat(first)),
            "expected one winding and one flat loop, got {first} and {second}"
        );
    }

    /// Two circles, two classes: one co-alive block of two columns; after
    /// decorrelation each column winds exactly one circle (the Gram matrix
    /// is near-diagonal, so LLL must not mix them).
    #[test]
    fn two_circles_toroidal_block() {
        let per = 24;
        let points = two_circles(per);
        let n = 2 * per;

        let cc = circular_with(
            &points,
            n,
            &Config {
                n_classes: 2,
                ..Config::default()
            },
        )
        .unwrap();
        assert_eq!(cc.k, 2);
        assert_eq!(cc.coordinates.len(), n * 2);
        // Equal circles: co-alive classes, one block.
        assert_eq!(cc.classes[0].group, cc.classes[1].group);
        assert_eq!(cc.classes[0].scale.to_bits(), cc.classes[1].scale.to_bits());

        let w = |column: usize, range: std::ops::Range<usize>| {
            winding(&cc.coordinates, 2, column, range)
        };
        let col0 = (w(0, 0..per), w(0, per..n));
        let col1 = (w(1, 0..per), w(1, per..n));
        let one_each = (winds(col0.0) && flat(col0.1) && flat(col1.0) && winds(col1.1))
            || (flat(col0.0) && winds(col0.1) && winds(col1.0) && flat(col1.1));
        assert!(
            one_each,
            "each column must wind exactly one circle, got {col0:?} and {col1:?}"
        );
    }

    /// The topological guarantee must survive the quotient approximation:
    /// the circle's coordinate, computed on the coperto-coned complex (class
    /// correspondence through the log-3 interleaving), still winds exactly
    /// once — and every original point receives its own value.
    #[test]
    fn quotient_circle_winds_once() {
        let n = 48;
        let points = circle(n);
        let cc = circular_with(
            &points,
            n,
            &Config {
                quotient: true,
                ..Config::default()
            },
        )
        .unwrap();
        assert_eq!(cc.k, 1);
        assert_eq!(cc.coordinates.len(), n);
        let w = winding(&cc.coordinates, 1, 0, 0..n);
        assert!(winds(w), "quotient coordinate must wind once, got {w}");
    }

    fn wrapped_gap(a: f32, b: f32) -> f64 {
        let mut d = (a as f64 - b as f64).abs() % (2.0 * std::f64::consts::PI);
        if d > std::f64::consts::PI {
            d = 2.0 * std::f64::consts::PI - d;
        }
        d
    }

    /// Round trip: the fitted map evaluated back on the training points is
    /// the PoU-smoothed version of the fit — same winding, values within the
    /// harmonic-residual tolerance.
    #[test]
    fn map_round_trips_on_training_points() {
        let n = 48;
        let points = circle(n);
        let map = circular_map(&points, n).unwrap();
        assert_eq!(map.k, 1);
        assert_eq!(map.coordinates.len(), n);

        // The stored training coordinates equal the plain function's output.
        let plain = circular(&points, n).unwrap();
        assert_eq!(map.coordinates, plain.coordinates);

        let transformed = map.transform(&points, n).unwrap();
        assert!(transformed.iter().all(|v| v.is_finite()));
        let w = winding(&transformed, 1, 0, 0..n);
        assert!(
            winds(w),
            "transformed training loop must wind once, got {w}"
        );

        let mut worst = 0f64;
        for (&actual, &expected) in transformed.iter().zip(&map.coordinates) {
            worst = worst.max(wrapped_gap(actual, expected));
        }
        assert!(
            worst < 1.2,
            "transform(train) deviates {worst} rad from the fit"
        );
    }

    /// Fresh points on the same circle (different phases) are covered, get
    /// finite values, and wind exactly once.
    #[test]
    fn map_transforms_fresh_points() {
        let n = 48;
        let points = circle(n);
        let map = circular_map(&points, n).unwrap();

        let m = 60;
        let mut fresh = Vec::with_capacity(2 * m);
        for i in 0..m {
            let theta = 2.0 * std::f32::consts::PI * (i as f32 + 0.37) / (m as f32);
            fresh.push(theta.cos());
            fresh.push(theta.sin());
        }
        let values = map.transform(&fresh, m).unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
        let w = winding(&values, 1, 0, 0..m);
        assert!(winds(w), "fresh loop must wind once, got {w}");
    }

    /// Outside the cover → NaN; near the data → finite. Shape errors and
    /// unsupported configs are rejected.
    #[test]
    fn map_coverage_and_validation() {
        let n = 32;
        let points = circle(n);
        let map = circular_map(&points, n).unwrap();

        let far = map.transform(&[100.0, 100.0], 1).unwrap();
        assert!(far[0].is_nan());
        let near = map.transform(&[1.0, 0.0], 1).unwrap();
        assert!(near[0].is_finite());

        let err = map.transform(&[1.0, 0.0, 0.0], 1).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));

        let err = circular_map_with(
            &points,
            n,
            &Config {
                distance_matrix: true,
                ..Config::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::NotImplemented(_)));

        let err = circular_map_with(
            &points,
            n,
            &Config {
                quotient: true,
                ..Config::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::NotImplemented(_)));

        // Empty selection: a k = 0 map that transforms to (m, 0).
        let empty = circular_map_with(
            &points,
            n,
            &Config {
                classes: Some(Vec::new()),
                ..Config::default()
            },
        )
        .unwrap();
        assert_eq!(empty.k, 0);
        assert!(empty.transform(&[1.0, 0.0], 1).unwrap().is_empty());
    }

    /// Toroidal maps: two circles, two columns; fresh loops on each circle
    /// wind in exactly one column each (same pattern as the fit).
    #[test]
    fn map_transforms_toroidal_blocks() {
        let per = 24;
        let points = two_circles(per);
        let n = 2 * per;
        let map = circular_map_with(
            &points,
            n,
            &Config {
                n_classes: 2,
                ..Config::default()
            },
        )
        .unwrap();
        assert_eq!(map.k, 2);

        let m = 30;
        let mut fresh = Vec::with_capacity(4 * m);
        for i in 0..m {
            let theta = 2.0 * std::f32::consts::PI * (i as f32 + 0.5) / (m as f32);
            fresh.push(theta.cos());
            fresh.push(theta.sin());
        }
        for i in 0..m {
            let theta = 2.0 * std::f32::consts::PI * (i as f32 + 0.5) / (m as f32);
            fresh.push(5.0 + theta.cos());
            fresh.push(theta.sin());
        }
        let values = map.transform(&fresh, 2 * m).unwrap();
        assert!(values.iter().all(|v| v.is_finite()));

        let w = |column: usize, range: std::ops::Range<usize>| winding(&values, 2, column, range);
        let col0 = (w(0, 0..m), w(0, m..2 * m));
        let col1 = (w(1, 0..m), w(1, m..2 * m));
        let one_each = (winds(col0.0) && flat(col0.1) && flat(col1.0) && winds(col1.1))
            || (flat(col0.0) && winds(col0.1) && winds(col1.0) && flat(col1.1));
        assert!(
            one_each,
            "each column must wind exactly one fresh circle, got {col0:?} and {col1:?}"
        );
    }

    /// The greedy sweep: overlapping intervals form one block at the midpoint
    /// of their common intersection; a disjoint interval gets its own block.
    #[test]
    fn grouping_sweep() {
        let intervals = [(0.0f32, 10.0f32), (1.0, 9.0), (20.0, 30.0)];
        let blocks = group_by_coalive(&intervals);
        assert_eq!(blocks.len(), 2);
        let mut first = blocks[0].members.clone();
        first.sort_unstable();
        assert_eq!(first, vec![0, 1]);
        assert!((blocks[0].scale - 5.0).abs() < 1e-6);
        assert_eq!(blocks[1].members, vec![2]);
        assert!((blocks[1].scale - 25.0).abs() < 1e-6);
    }

    /// LLL on the Gram of the basis (1, 0), (3, 1): the reduced basis is the
    /// standard one — unit diagonal, unimodular transform.
    #[test]
    fn lll_reduces_skewed_basis() {
        let gram = vec![vec![1.0, 3.0], vec![3.0, 10.0]];
        let u = lll_reduce(gram.clone());
        let det = u[0][0] * u[1][1] - u[0][1] * u[1][0];
        assert_eq!(det.abs(), 1, "transform must be unimodular");

        // Reduced Gram = U G Uᵀ.
        let mut reduced = [[0f64; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for a in 0..2 {
                    for b in 0..2 {
                        reduced[i][j] += u[i][a] as f64 * gram[a][b] * u[j][b] as f64;
                    }
                }
            }
        }
        assert!((reduced[0][0] - 1.0).abs() < 1e-9);
        assert!((reduced[1][1] - 1.0).abs() < 1e-9);
        assert!(reduced[0][1].abs() < 1e-9);
    }
}
