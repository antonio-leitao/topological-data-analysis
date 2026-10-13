//! Circular and toroidal coordinates from persistent H1 over Z/p.
//!
//! Selected cocycles are grouped by a common live scale, balanced-lifted to
//! integers, harmonic-smoothed, and LLL-decorrelated. A multi-class block
//! defines one map to a torus; its columns need not match input classes.

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
const LLL_DELTA: f64 = 0.75;
/// Preserve a valid unimodular transform if an ill-conditioned reduction stalls.
const LLL_MAX_STEPS: usize = 10_000;

/// Circular coordinates for the most persistent H1 class.
pub fn circular(data: &[f32], n: usize) -> Result<CircularCoordinates> {
    circular_with(data, n, &Config::default())
}

/// Circular coordinates from H1 classes selected by persistence rank.
///
/// Returns angle columns in `[0, 2*pi)`. An empty selection returns shape
/// `(n, 0)`. At an explicit scale, selected classes not alive there are omitted.
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

    // Coordinates live on the Rips graph itself, which collapse would rewrite.
    let core_config = crate::Config {
        distance_matrix: config.distance_matrix,
        quotient: config.quotient,
        collapse: crate::Collapse::Off,
        parallel: config.parallel,
        ..crate::Config::default()
    };
    let edges = crate::build_edges(data, n, 1, &core_config, crate::Ids::Keep)?;
    let threshold = edges.threshold;
    let bitcsr = BitCsrDistanceMatrix::from_edge_list(edges)?;
    let h1 = h1_classes(&bitcsr, n, config.prime, config.parallel);

    let chosen = resolve_selection(&selection, &h1, threshold)?;
    let (chosen, blocks) = make_blocks(chosen, config.scale)?;

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

fn resolve_selection(selection: &[usize], h1: &H1Classes, threshold: f32) -> Result<Vec<Chosen>> {
    let ranked = rank_classes(&h1.intervals, threshold);
    let mut chosen: Vec<Chosen> = Vec::with_capacity(selection.len());
    for &rank in selection {
        let Some(&class) = ranked.get(rank) else {
            return Err(Error::NotEnoughClasses {
                requested: rank + 1,
                available: ranked.len(),
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

/// Greedily group the most classes sharing a live scale.
/// Ties prefer the widest intersection, then the earliest.
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

/// Harmonic potentials aligned with `block.members`.
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

/// Return potentials and lifted cocycles after the same LLL transform.
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
        // Balanced integer lift on canonical edges `(u > v)`.
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

        // Minimise `||eta - delta(f)||^2` with `delta(f)[u > v] = f(v) - f(u)`.
        let mut rhs = vec![0f64; n];
        for (&(u, v), &value) in &eta {
            rhs[v as usize] += value;
            rhs[u as usize] -= value;
        }
        fs.push(laplacian_cg(&adjacency, &rhs));
        etas.push(eta);
    }

    if fs.len() > 1 {
        // Linearity lets the same LLL transform act on cocycles and potentials.
        let gram = harmonic_gram(&adjacency, &etas, &fs);
        let transform = lll_reduce(gram);
        fs = apply_transform(&transform, &fs, n);
        etas = apply_transform_etas(&transform, &etas);
    }
    Ok((fs, etas))
}

/// Reject balanced lifts with nonzero integer coboundary.
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
        Err(Error::NonIntegralCocycleLift { prime, scale })
    } else {
        Ok(())
    }
}

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

/// Gram matrix of `h = eta - delta(f)` over skeleton edges.
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

/// Gram-only LLL, returning the unimodular reduced-basis transform.
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

/// Gram-Schmidt data derived from a Gram matrix.
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

/// Apply `b_i = b_i - r*b_j` to the Gram matrix and transform.
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

/// A fitted circular-coordinates map.
///
/// In the chart of the nearest covered training vertex `j`, a query evaluates as
///
/// ```text
/// theta(x) = f_j + sum_k phi_k(x) * (f_k - f_j - eta(j, k))  (mod 1)
/// ```
///
/// Since `delta(eta) = 0` and weights sum to one, chart changes add an integer.
/// Points outside every cover ball evaluate to NaN.
#[derive(Debug)]
pub struct CircularMap {
    train: Vec<f32>,
    dim: usize,
    n: usize,
    pub k: usize,
    /// Training coordinates in row-major `(n, k)` layout.
    pub coordinates: Vec<f32>,
    pub classes: Vec<SelectedClass>,
    potentials: Vec<Vec<f64>>,
    cocycles: Vec<SortedCocycle>,
    parallel: bool,
}

/// Lifted cocycle keyed by packed canonical edges.
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

    /// Alternating cocycle value on `(a, b)`.
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

/// Fit a circular map with the default configuration.
pub fn circular_map(data: &[f32], n: usize) -> Result<CircularMap> {
    circular_map_with(data, n, &Config::default())
}

/// Fit a circular map for out-of-sample evaluation.
///
/// Distance matrices lack query cross-distances, and quotient scales are not
/// ambient distances, so both modes are rejected.
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
        collapse: crate::Collapse::Off,
        parallel: config.parallel,
        ..crate::Config::default()
    };
    let edges = crate::build_edges(data, n, 1, &core_config, crate::Ids::Keep)?;
    let threshold = edges.threshold;
    let dim = data.len() / n;
    let bitcsr = BitCsrDistanceMatrix::from_edge_list(edges)?;
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
    pub fn n_points(&self) -> usize {
        self.n
    }

    pub fn ambient_dim(&self) -> usize {
        self.dim
    }

    /// Evaluate row-major `(m, dim)` queries.
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

                // Ascending rows break nearest-chart ties by index.
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

/// H1 intervals and cocycles converted from packed simplices.
struct H1Classes {
    intervals: Vec<PersistenceInterval>,
    /// Aligned cocycles as `(u, v, coeff)`, with `u > v`.
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

/// Symmetric CSR adjacency at the selected scale.
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

/// Solve the singular graph Laplacian with Jacobi-preconditioned CG.
/// The coboundary-fit right side sums to zero on each component.
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
        let edges = EdgeList::from_distance_matrix(&distances, 3, 1.0).unwrap();
        let bitcsr = BitCsrDistanceMatrix::from_edge_list(edges).unwrap();

        let bad = HashMap::from([((2, 1), 1.0)]);
        let err = verify_integral_lift(&bitcsr, 1.0, &bad, 3).unwrap_err();
        assert!(matches!(
            err,
            Error::NonIntegralCocycleLift { prime: 3, .. }
        ));

        let exact = HashMap::from([((1, 0), -1.0), ((2, 0), -1.0)]);
        verify_integral_lift(&bitcsr, 1.0, &exact, 3).unwrap();
    }

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

    #[test]
    fn map_round_trips_on_training_points() {
        let n = 48;
        let points = circle(n);
        let map = circular_map(&points, n).unwrap();
        assert_eq!(map.k, 1);
        assert_eq!(map.coordinates.len(), n);

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

    #[test]
    fn lll_reduces_skewed_basis() {
        let gram = vec![vec![1.0, 3.0], vec![3.0, 10.0]];
        let u = lll_reduce(gram.clone());
        let det = u[0][0] * u[1][1] - u[0][1] * u[1][0];
        assert_eq!(det.abs(), 1, "transform must be unimodular");

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
