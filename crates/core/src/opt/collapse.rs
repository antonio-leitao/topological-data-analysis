//! Exact backward edge collapse for weighted one-parameter flag filtrations.
//!
//! This is the Glisse–Pritam backward swap/shift/trim algorithm (SoCG 2022).
//!
//! Storage: a dense `n x n` matrix of current grades, plus two adjacency
//! bitmaps.
//!
//! The matrix is the whole point. Every hot path ends in the same question —
//! "what grade is edge `ab` at now?" — and a CSR answers it with a binary
//! search: 9 to 13 *dependent, cache-missing* probes for one number, 318M
//! times on `polygon`. The matrix answers it with one load.
//!
//! It is also smaller. A CSR stores 8 bytes per directed incidence (neighbour
//! plus grade), so at density `d` it costs `8dn^2` against the matrix's `4n^2`
//! — above 50% density the matrix wins, and it wins on every dataset here, by
//! 1.27x to 2.00x. That is not a coincidence: this is a sparsification tool, so
//! a dense input is the assumption, not the exception. The cost is that a truly
//! sparse graph pays `4n^2` regardless, which is the trade the objective asks
//! for.
//!
//! Going dense also deletes the machinery a CSR needed: the counting-sort
//! build, the per-row scatter, the two-pointer merge for candidate discovery,
//! and the row-compaction problem that trimmed edges create.
//!
//! The bitmaps stay, because they answer a *cheaper* question. Testing many
//! candidates against one vertex's row touches `n/8` bytes of bitmap against
//! `4n` of matrix — 32x the locality — so `covers` remains the fast path and
//! the matrix is only read for what it cannot settle.
//!
//! Per target edge `uv` at grade `t`:
//!
//! 1. Intersect the adjacency bitmaps of `u` and `v` to get the common
//!    neighbourhood `C`, each candidate carrying its birth `b(w) = max(g(uw),
//!    g(vw))`.
//! 2. Search `C` for a dominator at `t`: some `c` adjacent by `t` to every
//!    candidate already born by `t`.
//! 3. No dominator: retain the edge at `t` and stop.
//! 4. A dominator: shift to the next birth grade, batch every arrival there,
//!    and re-check only the arrivals against the incumbent. Fall back to a full
//!    search when the incumbent fails.
//! 5. Dominated with no future arrivals left: trim.

/// A compressed filtration grade: a dense rank into `grade_values`.
pub(crate) type Grade = u32;
use crate::preprocess::edgelist::{Edge, EdgeList};
use crate::{Error, Result};

/// Sentinel for a trimmed edge. Chosen as the maximum so that every
/// `grade <= t` test rejects dead edges without a separate liveness branch.
pub(crate) const DEAD: Grade = Grade::MAX;

const NO_DOMINATOR: u32 = u32::MAX;

/// How much the dense grade matrix may cost before the CSR layout takes over.
///
/// This is the one policy constant in the file, and it exists because the two
/// layouts trade size against speed in opposite directions. Picking the
/// *smaller* one is not enough: at n = 5000 and 11% density the CSR is 1.7x
/// smaller but 1.66x slower, because a binary search costs `log(degree)` and
/// the degree there is 573. So the matrix is kept whenever it is affordable at
/// all, and the CSR takes over only when it would not be — which is exactly
/// when a sparse graph has enough vertices for `4n^2` to dominate the process.
/// At 1 GiB the crossover is around n = 16 000.
const GRADE_MATRIX_BUDGET: usize = 1 << 30;

/// Edges the `Auto` probe watches before judging the trim rate, as a share of
/// the pass (`1/PROBE_SHARE`) clamped to `PROBE_MIN..=PROBE_MAX`.
const PROBE_SHARE: usize = 50;
const PROBE_MIN: usize = 64;
const PROBE_MAX: usize = 1024;

/// Edges per marginal-cost window of the `Auto` H1 budget, as a share of the
/// pass clamped to `WINDOW_MIN..=WINDOW_MAX`.
const WINDOW_SHARE: usize = 64;
const WINDOW_MIN: usize = 256;
const WINDOW_MAX: usize = 4096;

/// Work per processed edge above which the rest of the pass costs more than
/// H1 can save on those edges.
///
/// H1 costs about 0.1–0.25 µs per edge on most data, and a unit of work costs
/// 0.4–1.4 ns. Work overstates the head of the pass, where the rows are
/// scanned fastest, so the threshold that reproduces the stops measured in
/// wall time sits above a straight conversion: 200 stops `circle_noise` after
/// 3% of a pass whose full run is 55x faster at H1, 800 lets `pbmc` run to 87%
/// where 25% was best.
const H1_WORK_PER_EDGE: u64 = 400;

/// Edge collapse policy.
///
/// Collapse never changes the barcode, only how long it takes. Whether it pays
/// depends on the target dimension and on the data:
///
/// * H0 never benefits: its reduction is cheaper than any simplification.
/// * From H2 up, the complex the engine enumerates shrinks far faster than the
///   pass costs, so a full pass wins whenever the graph collapses at all.
/// * H1 is already near its per-edge floor on most data, so only the cheap
///   head of the pass is worth running.
/// * Point clouds with no cone apex near the cutoff (spheres, flat tori, `o3`
///   without a threshold) keep ~100% of their edges, and there the pass is
///   pure overhead in every dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Collapse {
    /// Skip the collapse.
    Off,
    /// Run one full backward pass.
    On,
    /// Decide per dimension, and from how the pass behaves while it runs.
    #[default]
    Auto,
}

impl From<bool> for Collapse {
    fn from(on: bool) -> Self {
        if on {
            Collapse::On
        } else {
            Collapse::Off
        }
    }
}

impl Collapse {
    /// Run the collapse this policy asks for. Returns how far the pass got, or
    /// `None` when it did not start.
    ///
    /// Stopping the backward pass early is exact: every step is a
    /// barcode-preserving swap, shift or trim of the filtration as it stands
    /// (Glisse–Pritam, Theorem 5), so any prefix of the pass leaves a valid
    /// filtration. `Auto` uses that to decide as it goes:
    ///
    /// 1. `max_dim == 0`: skip.
    /// 2. Probe the first edges, which are the longest. When the graph
    ///    collapses they sit under a cone apex and nearly all are trimmed (at
    ///    least 90% on every dataset measured); without an apex almost none
    ///    are (at most 4%). Stop unless at least half were trimmed.
    /// 3. `max_dim >= 2`: finish the pass.
    /// 4. `max_dim == 1`: the pass costs more per edge as it descends into
    ///    short edges with long shift trajectories, while H1 saves roughly a
    ///    constant per removed edge. Stop at the first window whose work per
    ///    edge exceeds that saving.
    ///
    /// A graph too large for the pass's 32-bit indices is an error under `On`
    /// and skipped under `Auto`.
    pub(crate) fn apply(
        self,
        edge_list: &mut EdgeList,
        max_dim: usize,
        parallel: bool,
    ) -> Result<Option<Progress>> {
        let incidences = index_space(edge_list);
        let fits = incidences <= u32::MAX as usize;
        Ok(match self {
            Collapse::Off => None,
            Collapse::On if !fits => {
                return Err(Error::GraphTooLarge {
                    what: "the collapse index space (2m + n)",
                    got: incidences,
                    max: u32::MAX as usize,
                })
            }
            Collapse::On => Some(collapse_with(edge_list, parallel, |_| true)),
            Collapse::Auto if max_dim == 0 || !fits => None,
            Collapse::Auto => {
                let m = edge_list.edges.len();
                let probe = (m / PROBE_SHARE).clamp(PROBE_MIN, PROBE_MAX);
                let window = (m / WINDOW_SHARE).clamp(WINDOW_MIN, WINDOW_MAX);
                // (edges, work) at the start of the current window.
                let mut mark = (0usize, 0u64);
                Some(collapse_with(edge_list, parallel, |p| {
                    if p.processed < probe {
                        return true;
                    }
                    if p.processed == probe {
                        mark = (p.processed, p.work);
                        return 2 * p.trimmed >= probe;
                    }
                    if max_dim >= 2 || p.processed - mark.0 < window {
                        return true;
                    }
                    let per_edge = (p.work - mark.1) / (p.processed - mark.0) as u64;
                    mark = (p.processed, p.work);
                    per_edge <= H1_WORK_PER_EDGE
                }))
            }
        })
    }
}

/// `2m + n`, the widest id the pass stores in 32 bits (see `Collapser::build`).
#[inline]
fn index_space(edge_list: &EdgeList) -> usize {
    2 * edge_list.edges.len() + edge_list.n
}

/// How far a pass has got. `work` counts bitmap words and candidate entries
/// touched: a machine-independent stand-in for time.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Progress {
    pub(crate) processed: usize,
    pub(crate) trimmed: usize,
    pub(crate) work: u64,
}

/// Run one exact backward collapse pass in place, asking `keep_going` after
/// every edge whether to continue. Any prefix of the pass is exact.
///
/// One call is one pass: a pass need not reach a minimal graph, and calling
/// again may remove more. A caller that wants a fixed point calls again.
pub(crate) fn collapse_with(
    edge_list: &mut EdgeList,
    parallel: bool,
    keep_going: impl FnMut(&Progress) -> bool,
) -> Progress {
    collapse_in_layout(edge_list, parallel, None, keep_going)
}

/// [`collapse_with`], with `banded` overriding the row layout; tests use it to
/// run both.
fn collapse_in_layout(
    edge_list: &mut EdgeList,
    parallel: bool,
    banded: Option<bool>,
    keep_going: impl FnMut(&Progress) -> bool,
) -> Progress {
    edge_list.sort(parallel);
    if edge_list.edges.len() < 2 {
        return Progress::default();
    }
    debug_assert!({
        edge_list.assert_preconditions();
        true
    });
    let mut c = Collapser::build(edge_list, banded);
    let progress = c.run_pass(keep_going);
    c.write_back(edge_list);
    progress
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

/// Graph shape. Fixed for the whole pass; only the grade matrix changes.
struct Topo {
    edge_u: Vec<u32>,
    edge_v: Vec<u32>,
}

/// Row stride of the grade matrix, padded so every row starts on a cache line.
#[inline(always)]
fn grade_stride(n: usize) -> usize {
    n.next_multiple_of(16)
}

/// One CSR incidence: `(neighbour << 32) | grade`. Because the neighbour is in
/// the high half, comparing whole `u64`s compares neighbours.
type Key = u64;

/// Where an edge's current grade lives.
///
/// Two layouts, because the two regimes want opposite things and each one's
/// weakness is the other's home ground:
///
/// * `Dense` is `4n^2` bytes and answers in one load.
/// * `Sparse` is `16m` bytes and answers in a binary search over one row.
///
/// Which is *smaller* flips at 50% density (`4n^2` against `16m`). Measured,
/// so does which is *faster*: grade lookups are 6.4% of runtime on `polygon`,
/// where the matrix is 1 MB and sits in L2 — and under 1% on a 0.4%-dense
/// graph at n = 10 000, where the matrix would be 381 MB, 91% of the process,
/// and 165x the size of the edge data it describes. A binary search over a row
/// of average degree 38 is nothing next to that.
///
/// So the gate is not a tuned constant; it picks whichever representation is
/// smaller, and that turns out to be the faster one too.
enum Grades {
    Dense { data: Vec<Grade>, stride: usize },
    Sparse { offsets: Vec<u32>, keys: Vec<Key> },
}

impl Grades {
    /// Current grade of edge `ab`, or `DEAD` when it is absent or trimmed.
    #[inline(always)]
    fn get(&self, a: u32, b: u32) -> Grade {
        match self {
            Grades::Dense { data, stride } => data[a as usize * stride + b as usize],
            Grades::Sparse { offsets, keys } => {
                let row = &keys[offsets[a as usize] as usize..offsets[a as usize + 1] as usize];
                let floor = (b as u64) << 32;
                let i = row.partition_point(|&k| k < floor);
                match row.get(i) {
                    Some(&k) if (k >> 32) as u32 == b => k as u32,
                    _ => DEAD,
                }
            }
        }
    }

    /// Write an edge's new grade into both of its halves.
    #[inline]
    fn set(&mut self, u: u32, v: u32, g: Grade) {
        match self {
            Grades::Dense { data, stride } => {
                data[u as usize * *stride + v as usize] = g;
                data[v as usize * *stride + u as usize] = g;
            }
            Grades::Sparse { offsets, keys } => {
                for (a, b) in [(u, v), (v, u)] {
                    let (s, e) = (
                        offsets[a as usize] as usize,
                        offsets[a as usize + 1] as usize,
                    );
                    let floor = (b as u64) << 32;
                    let i = s + keys[s..e].partition_point(|&k| k < floor);
                    debug_assert!(i < e && (keys[i] >> 32) as u32 == b, "no such edge");
                    keys[i] = floor | g as u64;
                }
            }
        }
    }
}

/// Words in one bitmap row, padded so every row starts on a 32-byte boundary
/// and the reduction loops get a trip count the vectoriser is happy with.
#[inline(always)]
fn row_words(n: usize) -> usize {
    n.div_ceil(64).next_multiple_of(4)
}

/// `mask ⊆ row`? Both slices are the same length, iterated with `zip`, and the
/// body is a pure OR-reduction over `mask & !row` — no branches, no bounds
/// checks, nothing loop-carried but the accumulator. That is the whole reason
/// for moving to bitmaps, so keep it exactly this shape.
#[inline]
fn covers(row: &[u64], mask: &[u64]) -> bool {
    let mut acc = 0u64;
    for (&r, &m) in row.iter().zip(mask) {
        acc |= m & !r;
    }
    acc == 0
}

/// `dst = a & b`, three equal-length slices zipped so the vectoriser sees a
/// clean fixed-trip elementwise loop. Returns the OR of the result, which is
/// zero exactly when the intersection is empty — free, since every word is
/// already in a register.
#[inline]
fn and_rows(dst: &mut [u64], a: &[u64], b: &[u64]) -> u64 {
    let mut any = 0u64;
    for ((d, &x), &y) in dst.iter_mut().zip(a).zip(b) {
        let r = x & y;
        *d = r;
        any |= r;
    }
    any
}

#[inline]
fn set_bit(mask: &mut [u64], v: u32) {
    mask[v as usize >> 6] |= 1u64 << (v & 63);
}

#[inline]
fn clear_bit(mask: &mut [u64], v: u32) {
    mask[v as usize >> 6] &= !(1u64 << (v & 63));
}

/// One entry of a target edge's common neighbourhood.
#[derive(Clone, Copy)]
struct Cand {
    vertex: u32,
    /// `max(g(uw), g(vw))`: the grade at which `w` joins the neighbourhood.
    birth: Grade,
}

/// Reusable per-call buffers. Nothing in the hot loop allocates.
#[derive(Default)]
struct Scratch {
    /// The full common neighbourhood, in ascending vertex order. Membership is
    /// static for the whole trajectory; activity at `t` is `birth <= t`, so
    /// arrivals never need to be inserted into a sorted set.
    cand: Vec<Cand>,
    /// Not-yet-arrived candidates as `(birth << 32) | index`, ascending. The
    /// key is packed rather than compared through `cand`, so the sort is a
    /// plain integer sort with no gather in its comparator.
    events: Vec<u64>,
    /// Per-*vertex* stamp of the search that ruled it out, compared against
    /// `search_id` so a search never has to clear it. Keyed by vertex rather
    /// than by slot so a bitmap witness — which arrives as a vertex id — needs
    /// no lookup to record.
    rejected: Vec<u64>,
    search_id: u64,
    /// The active candidate set as a bitmap, one row wide. Rebuilt per search.
    cand_bits: Vec<u64>,
    /// Words of `cand_bits` the last target may have left set. Full-width rows
    /// overwrite the whole mask anyway; banded rows only clear this range.
    dirty: (usize, usize),
    /// Running work count for [`Progress::work`].
    work: u64,
}

struct Collapser {
    topo: Topo,
    /// The graph state: `grades[a * gstride + b]` is edge `ab`'s current grade,
    /// or `DEAD` when it is absent or trimmed. Single source of truth.
    grades: Grades,
    /// Closed adjacency of `G_t` at the current outer grade, one row per vertex
    /// laid out by `layout`. The backward loop only ever descends, so every
    /// edge leaves this exactly once: `O(m)` bit clears for the whole pass, and
    /// no level sets.
    bits: Vec<u64>,
    /// Adjacency of every *surviving* edge, at any grade. Only trimming clears
    /// a bit here, so `L[u] & L[v]` minus the active set answers "does this
    /// edge have anything left to shift across" in one pass over a row.
    live: Vec<u64>,
    /// Where each row of `bits` and `live` lives.
    layout: Layout,
    /// `grade_start[g]..grade_start[g + 1]` are the edge ids at grade `g`.
    grade_start: Vec<u32>,
    grade_values: Vec<f32>,
    scratch: Scratch,
}

/// Where a bitmap row lives: word `b` of row `r` is `bits[origin + b]` for
/// `lo <= b < hi`.
///
/// Dense graphs keep every row full width (`w` words from block 0, at
/// `r * w`), the shape the reduction loops below were tuned on. A sparse graph
/// whose ids keep neighbours close, such as the renumbered quotient skeleton,
/// would spend almost all of `n * w` on zeros, so there each row stores only
/// the blocks its closed neighbourhood spans. The pass is compiled once per
/// layout (`BANDED`), so full-width rows run with no bounds to compute.
struct Layout {
    /// Words in a full-width row, and in the candidate mask.
    w: usize,
    /// Per-row `(origin, lo, hi)`; empty when rows are full width.
    spans: Vec<(usize, u32, u32)>,
}

impl Layout {
    /// `(origin, lo, hi)` of row `r`.
    #[inline(always)]
    fn span<const BANDED: bool>(&self, r: u32) -> (usize, usize, usize) {
        if BANDED {
            let (origin, lo, hi) = self.spans[r as usize];
            (origin, lo as usize, hi as usize)
        } else {
            (r as usize * self.w, 0, self.w)
        }
    }

    /// Words `lo..hi` of row `r`; the range must lie in the row's span.
    #[inline(always)]
    fn words<'a, const BANDED: bool>(
        &self,
        bits: &'a [u64],
        r: u32,
        lo: usize,
        hi: usize,
    ) -> &'a [u64] {
        let (origin, ..) = self.span::<BANDED>(r);
        &bits[origin + lo..origin + hi]
    }

    /// Whether bit `b` is set in row `a`.
    #[inline(always)]
    fn test<const BANDED: bool>(&self, bits: &[u64], a: u32, b: u32) -> bool {
        let (origin, lo, hi) = self.span::<BANDED>(a);
        let block = b as usize >> 6;
        (!BANDED || (lo <= block && block < hi)) && bits[origin + block] & (1u64 << (b & 63)) != 0
    }

    /// Clear edge `uv` from both rows; it must be present in both spans.
    #[inline(always)]
    fn clear_edge<const BANDED: bool>(&self, bits: &mut [u64], u: u32, v: u32) {
        let (ou, ..) = self.span::<BANDED>(u);
        let (ov, ..) = self.span::<BANDED>(v);
        bits[ou + (v as usize >> 6)] &= !(1u64 << (v & 63));
        bits[ov + (u as usize >> 6)] &= !(1u64 << (u & 63));
    }
}

/// First and last-plus-one nonzero words of `mask[lo..hi]`, or an empty range.
#[inline]
fn trimmed(mask: &[u64], lo: usize, hi: usize) -> (usize, usize) {
    let mut a = lo;
    while a < hi && mask[a] == 0 {
        a += 1;
    }
    let mut b = hi;
    while b > a && mask[b - 1] == 0 {
        b -= 1;
    }
    (a, b)
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl Collapser {
    fn build(el: &EdgeList, banded: Option<bool>) -> Self {
        let n = el.n;
        let m = el.edges.len();
        // `Collapse::apply` rejects or skips larger graphs before building.
        assert!(
            index_space(el) <= u32::MAX as usize,
            "graph too large for 32-bit indices: n={n} m={m}"
        );

        // Grade compression. The edge list is canonically sorted, so equal
        // distances are already contiguous.
        let mut grade_values: Vec<f32> = Vec::new();
        let mut grade_start: Vec<u32> = Vec::new();
        let mut grade = vec![0u32; m];
        for (i, e) in el.edges.iter().enumerate() {
            if grade_values.last() != Some(&e.distance) {
                grade_values.push(e.distance);
                grade_start.push(i as u32);
            }
            grade[i] = (grade_values.len() - 1) as Grade;
        }
        grade_start.push(m as u32);
        assert!(
            grade_values.len() < DEAD as usize,
            "too many distinct filtration values"
        );

        let mut edge_u = Vec::with_capacity(m);
        let mut edge_v = Vec::with_capacity(m);
        for e in &el.edges {
            edge_u.push(e.u);
            edge_v.push(e.v);
        }

        // Keep the matrix while it is either the smaller layout or simply
        // affordable; hand over to the CSR only when it is neither.
        let gstride = grade_stride(n);
        let matrix_bytes = 4 * n * gstride;
        let grades = if matrix_bytes <= 16 * m || matrix_bytes <= GRADE_MATRIX_BUDGET {
            let mut data = vec![DEAD; n * gstride];
            for (eid, e) in el.edges.iter().enumerate() {
                data[e.u as usize * gstride + e.v as usize] = grade[eid];
                data[e.v as usize * gstride + e.u as usize] = grade[eid];
            }
            Grades::Dense {
                data,
                stride: gstride,
            }
        } else {
            build_sparse_grades(n, m, el, &grade)
        };

        // Closed adjacency bitmap of the full graph. The diagonal is set, so
        // `mask & !row` never trips over the candidate testing itself.
        let w = row_words(n);
        let layout = build_layout(n, w, el, banded);
        let words = match layout.spans.last() {
            Some(&(origin, _, hi)) => origin + hi as usize,
            None => n * w,
        };
        let mut bits = vec![0u64; words];
        let set = |bits: &mut Vec<u64>, r: u32, c: u32| {
            let origin = if layout.spans.is_empty() {
                r as usize * w
            } else {
                layout.spans[r as usize].0
            };
            bits[origin + (c as usize >> 6)] |= 1u64 << (c & 63);
        };
        for r in 0..n as u32 {
            set(&mut bits, r, r);
        }
        for e in &el.edges {
            set(&mut bits, e.u, e.v);
            set(&mut bits, e.v, e.u);
        }

        let live = bits.clone();

        let scratch = Scratch {
            cand_bits: vec![0u64; w],
            ..Scratch::default()
        };

        Collapser {
            topo: Topo { edge_u, edge_v },
            grades,
            bits,
            live,
            layout,
            grade_start,
            grade_values,
            scratch,
        }
    }

    fn write_back(self, el: &mut EdgeList) {
        let Collapser {
            topo,
            grades,
            grade_values,
            ..
        } = self;
        el.edges.clear();
        for eid in 0..topo.edge_u.len() {
            let g = grades.get(topo.edge_u[eid], topo.edge_v[eid]);
            if g == DEAD {
                continue;
            }
            el.edges.push(Edge {
                u: topo.edge_u[eid],
                v: topo.edge_v[eid],
                distance: grade_values[g as usize],
            });
        }
        // Shifted edges break the input order, so the output needs a fresh sort.
        el.edges.sort_unstable_by_key(Edge::sort_key);
        el.sorted = true;
    }
}

/// Full-width rows unless storing each row's span at most halves the words.
/// Origins are padded so they never go negative.
fn build_layout(n: usize, w: usize, el: &EdgeList, banded: Option<bool>) -> Layout {
    let mut lo: Vec<u32> = (0..n as u32).map(|r| r >> 6).collect();
    let mut hi: Vec<u32> = lo.iter().map(|&b| b + 1).collect();
    for e in &el.edges {
        let (bu, bv) = (e.u >> 6, e.v >> 6);
        lo[e.u as usize] = lo[e.u as usize].min(bv);
        hi[e.u as usize] = hi[e.u as usize].max(bv + 1);
        lo[e.v as usize] = lo[e.v as usize].min(bu);
        hi[e.v as usize] = hi[e.v as usize].max(bu + 1);
    }
    let band: usize = lo.iter().zip(&hi).map(|(&l, &h)| (h - l) as usize).sum();
    if !banded.unwrap_or(2 * band <= n * w) {
        return Layout {
            w,
            spans: Vec::new(),
        };
    }
    let (mut start, mut pad) = (0usize, 0usize);
    for (&l, &h) in lo.iter().zip(&hi) {
        pad = pad.max((l as usize).saturating_sub(start));
        start += (h - l) as usize;
    }
    let mut start = pad;
    let spans = lo
        .iter()
        .zip(&hi)
        .map(|(&l, &h)| {
            let span = (start - l as usize, l, h);
            start += (h - l) as usize;
            span
        })
        .collect();
    Layout { w, spans }
}

/// CSR grades: `offsets[r]..offsets[r + 1]` holds row `r`'s incidences as
/// `(neighbour << 32) | grade`, ascending.
///
/// Built without a comparison. Every edge is canonical (`u > v`), so a row
/// splits cleanly — all neighbours below `r` sort before all neighbours above
/// it. One counting sort by the smaller endpoint fills the lower halves in
/// ascending order, and walking the rows afterwards fills the upper halves in
/// ascending order for free.
fn build_sparse_grades(n: usize, m: usize, el: &EdgeList, grade: &[Grade]) -> Grades {
    let mut lo_deg = vec![0u32; n];
    let mut offsets = vec![0u32; n + 1];
    for e in &el.edges {
        lo_deg[e.u as usize] += 1;
        offsets[e.u as usize + 1] += 1;
        offsets[e.v as usize + 1] += 1;
    }
    for r in 0..n {
        offsets[r + 1] += offsets[r];
    }

    let by_v = {
        let mut cur = vec![0u32; n + 1];
        for e in &el.edges {
            cur[e.v as usize + 1] += 1;
        }
        for r in 0..n {
            cur[r + 1] += cur[r];
        }
        let mut order = vec![0u32; m];
        for (eid, e) in el.edges.iter().enumerate() {
            let slot = &mut cur[e.v as usize];
            order[*slot as usize] = eid as u32;
            *slot += 1;
        }
        order
    };

    let mut keys = vec![0u64; 2 * m];
    let mut cursor: Vec<u32> = offsets[..n].to_vec();
    for &eid in &by_v {
        let e = eid as usize;
        let (u, v) = (el.edges[e].u, el.edges[e].v);
        let slot = &mut cursor[u as usize];
        keys[*slot as usize] = ((v as u64) << 32) | grade[e] as u64;
        *slot += 1;
    }
    // Each cursor now sits where its row's upper half begins.
    for r in 0..n as u32 {
        let lo = offsets[r as usize] as usize;
        for p in lo..lo + lo_deg[r as usize] as usize {
            let v = (keys[p] >> 32) as u32;
            let slot = &mut cursor[v as usize];
            keys[*slot as usize] = ((r as u64) << 32) | (keys[p] & 0xFFFF_FFFF);
            *slot += 1;
        }
    }
    debug_assert!((0..n).all(|r| cursor[r] == offsets[r + 1]));
    debug_assert!((0..n).all(|r| {
        keys[offsets[r] as usize..offsets[r + 1] as usize]
            .windows(2)
            .all(|w| w[0] < w[1])
    }));
    Grades::Sparse { offsets, keys }
}

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

impl Collapser {
    /// The pass: grade groups in non-increasing order, reverse canonical order
    /// inside each group, until `keep_going` says stop.
    fn run_pass(&mut self, keep_going: impl FnMut(&Progress) -> bool) -> Progress {
        if self.layout.spans.is_empty() {
            self.pass::<false>(keep_going)
        } else {
            self.pass::<true>(keep_going)
        }
    }

    fn pass<const BANDED: bool>(
        &mut self,
        mut keep_going: impl FnMut(&Progress) -> bool,
    ) -> Progress {
        let Collapser {
            topo,
            grades,
            bits,
            live,
            layout,
            grade_start,
            scratch,
            ..
        } = self;
        let mut progress = Progress::default();

        // The bitmap invariant is maintained by the descent alone. On entry to
        // group `g` it holds exactly `{e : current_grade(e) <= g}`: everything
        // below is unprocessed and still at its original grade, and everything
        // above has either been retired here or was cleared when it shifted or
        // was trimmed. Grades are dense ranks, so retiring group `g + 1` is the
        // whole transition — no edge can land between two consecutive ranks.
        //
        // Stopping early leaves the bitmaps stale, which is fine: only the
        // grade matrix is read afterwards, and unprocessed edges still hold
        // their original grades there.
        'pass: for g in (0..grade_start.len() - 1).rev() {
            if g + 2 < grade_start.len() {
                for eid in grade_start[g + 1]..grade_start[g + 2] {
                    let e = eid as usize;
                    layout.clear_edge::<BANDED>(bits, topo.edge_u[e], topo.edge_v[e]);
                }
            }
            for eid in (grade_start[g]..grade_start[g + 1]).rev() {
                let trimmed =
                    process::<BANDED>(topo, grades, bits, live, layout, scratch, eid as usize);
                progress.processed += 1;
                progress.trimmed += trimmed as usize;
                progress.work = scratch.work;
                if !keep_going(&progress) {
                    break 'pass;
                }
            }
        }
        progress
    }
}

/// Under `BANDED`, every row-wide operation below runs over `lo..hi`, the
/// blocks both endpoints' rows span: the common neighbourhood cannot reach
/// outside it. Full-width rows make that `0..w`, the loops they were tuned as.
#[allow(clippy::too_many_arguments)]
fn process<const BANDED: bool>(
    topo: &Topo,
    grades: &mut Grades,
    bits: &mut [u64],
    live: &mut [u64],
    layout: &Layout,
    s: &mut Scratch,
    eid: usize,
) -> bool {
    let Scratch {
        cand,
        events,
        rejected,
        search_id: next_search_id,
        cand_bits,
        dirty,
        work,
    } = s;
    let w = layout.w;
    let u = topo.edge_u[eid];
    let v = topo.edge_v[eid];
    let start = grades.get(u, v);
    debug_assert_ne!(start, DEAD, "an edge is scheduled at most once per pass");

    // Keyed by vertex, and the widest vertex id is bounded by a bitmap row.
    if rejected.len() < w * 64 {
        rejected.resize(w * 64, 0);
    }

    let (_, ulo, uhi) = layout.span::<BANDED>(u);
    let (_, vlo, vhi) = layout.span::<BANDED>(v);
    let (lo, hi) = (ulo.max(vlo), uhi.min(vhi));
    if BANDED {
        cand_bits[dirty.0..dirty.1].fill(0);
        *dirty = (lo, hi);
    }

    // ---- discovery: the active common neighbourhood is one AND -------------
    //
    // `bits` is exact at the edge's own grade, so `A[u] & A[v]` is precisely
    // `C_start`. The diagonal makes both endpoints show up in the AND, so clear
    // them; everything else that survives is a live common neighbour.
    let any = {
        let ru = layout.words::<BANDED>(bits, u, lo, hi);
        let rv = layout.words::<BANDED>(bits, v, lo, hi);
        and_rows(&mut cand_bits[lo..hi], ru, rv)
    };
    *work += (hi - lo) as u64;
    clear_bit(&mut cand_bits[..w], u);
    clear_bit(&mut cand_bits[..w], v);

    if any == 0 || cand_bits[lo..hi].iter().all(|&x| x == 0) {
        // Nothing can dominate it, so it stays where it is.
        return false;
    }

    // ---- first search, entirely in the bitmap ------------------------------
    //
    // The sweep runs downward in vertex order straight off `cand_bits`. There
    // is no candidate list to materialise: building one cost more than the
    // tests it fed, and every push round-tripped the vector's length through
    // memory.
    //
    // Failed-pair rejection is deliberately absent here, and present in the
    // sorted fallback below. The asymmetry is in what a witness costs: the
    // merge in `dominates` fails *at* a candidate, so naming it is free, while
    // `covers` is a branchless reduction that deliberately does not know where
    // it failed — recovering that means a second pass over the row. Measured,
    // the second pass costs more than the tests it saves whenever the bitmap
    // path dominates (uniform2d 99.6% bitmap: rejection here cost 9.5%), and
    // the fallback path is where it pays (polygon 29.1% bitmap: 14.6%).
    //
    // Banded rows test only the mask's nonzero words, after checking that
    // they all fall inside the candidate's row.
    let (tlo, thi) = if BANDED {
        trimmed(cand_bits, lo, hi)
    } else {
        (lo, hi)
    };
    let mut dom = NO_DOMINATOR;
    'sweep: for wi in (tlo..thi).rev() {
        let mut x = cand_bits[wi];
        let base = (wi * 64) as u32;
        while x != 0 {
            let b = 63 - x.leading_zeros();
            x &= !(1u64 << b);
            let c = base + b;
            *work += (thi - tlo) as u64;
            if spans_range::<BANDED>(layout, c, tlo, thi)
                && covers(
                    layout.words::<BANDED>(bits, c, tlo, thi),
                    &cand_bits[tlo..thi],
                )
            {
                dom = c;
                break 'sweep;
            }
        }
    }

    if dom == NO_DOMINATOR {
        // Critical at its own grade: retained, and nothing else to compute.
        return false;
    }

    // ---- is there anything left to shift across? --------------------------
    //
    // `live` holds every surviving edge regardless of grade, so the common
    // neighbours that arrive later are `(L[u] & L[v])` minus the ones already
    // active. Answering this in `w` words is what lets a graph that collapses
    // almost completely skip candidate discovery altogether.
    set_bit(&mut cand_bits[..w], u);
    set_bit(&mut cand_bits[..w], v);
    let has_future = {
        let lu = layout.words::<BANDED>(live, u, lo, hi);
        let lv = layout.words::<BANDED>(live, v, lo, hi);
        let mut acc = 0u64;
        for ((&x, &y), &c) in lu.iter().zip(lv).zip(cand_bits[lo..hi].iter()) {
            acc |= (x & y) & !c;
        }
        acc != 0
    };
    *work += (hi - lo) as u64;
    if !has_future {
        grades.set(u, v, DEAD);
        layout.clear_edge::<BANDED>(bits, u, v);
        layout.clear_edge::<BANDED>(live, u, v);
        return true;
    }

    // ---- the trajectory ---------------------------------------------------
    //
    // Only edges that actually shift pay for exact births and the event list.
    collect_candidates::<BANDED>(grades, live, layout, cand_bits, cand, u, v, start, lo, hi);
    build_events(cand, events, start);
    *work += (hi - lo + 2 * cand.len()) as u64;
    let mut ev = 0usize;
    let mut t;

    'outer: loop {
        // Shift while the incumbent survives. Between two arrival grades the
        // neighbourhood is unchanged, so domination cannot break
        // (Glisse-Pritam Lemma 4) and only arrivals need re-checking.
        loop {
            if ev == events.len() {
                grades.set(u, v, DEAD); // trimmed
                layout.clear_edge::<BANDED>(bits, u, v);
                layout.clear_edge::<BANDED>(live, u, v);
                return true;
            }
            t = event_grade(events[ev]);

            // Every arrival at this grade becomes visible before the domination
            // status is re-read.
            let mut still_dominated = true;
            while ev < events.len() && event_grade(events[ev]) == t {
                let cw = cand[event_index(events[ev])].vertex;
                ev += 1;
                *work += 1;
                if cw != dom && !adjacent_by::<BANDED>(grades, bits, layout, dom, cw, t) {
                    still_dominated = false;
                }
            }
            if still_dominated {
            } else {
                break;
            }
        }

        // The active set as a bitmap, so the search is the same shape as the
        // first one: sweep the candidates, test each with a reduction over a
        // row. Rebuilding costs `w` words plus one bit per live candidate.
        cand_bits[lo..hi].fill(0);
        for c in cand.iter().filter(|c| c.birth <= t) {
            set_bit(&mut cand_bits[..w], c.vertex);
        }
        *work += (hi - lo + cand.len()) as u64;
        let (tlo, thi) = if BANDED {
            trimmed(cand_bits, lo, hi)
        } else {
            (lo, hi)
        };

        *next_search_id += 1;
        let search_id = *next_search_id;
        let mut found = NO_DOMINATOR;
        'search: for wi in (tlo..thi).rev() {
            let mut x = cand_bits[wi];
            let base = (wi * 64) as u32;
            while x != 0 {
                let b = 63 - x.leading_zeros();
                x &= !(1u64 << b);
                let c = base + b;
                if rejected[c as usize] == search_id {
                    continue;
                }
                *work += (thi - tlo) as u64;
                match dominates::<BANDED>(grades, bits, live, layout, cand_bits, tlo, thi, c, t) {
                    None => {
                        found = c;
                        break 'search;
                    }
                    Some(witness) => {
                        rejected[witness as usize] = search_id;
                    }
                }
            }
        }
        if found == NO_DOMINATOR {
            break 'outer; // critical at t
        }
        dom = found;
    }

    // Retained at `t`. We only get here after at least one shift, so the grade
    // has always moved and both the state word and the bitmap need updating.
    grades.set(u, v, t);
    layout.clear_edge::<BANDED>(bits, u, v); // its grade is now above the outer grade
    false
}

/// Whether row `c` stores every block in `lo..hi`. A mask word outside the
/// row is a candidate `c` is not adjacent to, so `c` cannot cover the mask.
#[inline(always)]
fn spans_range<const BANDED: bool>(layout: &Layout, c: u32, lo: usize, hi: usize) -> bool {
    let (_, clo, chi) = layout.span::<BANDED>(c);
    !BANDED || (clo <= lo && hi <= chi)
}

/// Open common neighbourhood of `uv`, ascending by vertex.
///
/// `live` names the surviving edges in `w` words, and the matrix turns each
/// birth into two loads. With a CSR this had to choose between merging the two
/// rows — which walks every trimmed edge still sitting in them — and probing
/// the rows by binary search; neither is needed now.
///
/// Only the candidates arriving *later* than `start` need their births read at
/// all. One already adjacent to both endpoints by `start` has `birth <= start`,
/// and every use of `birth` compares it against a grade `>= start`, so
/// recording `start` for it is indistinguishable from the true value.
#[allow(clippy::too_many_arguments)]
fn collect_candidates<const BANDED: bool>(
    grades: &Grades,
    live: &[u64],
    layout: &Layout,
    active: &[u64],
    cand: &mut Vec<Cand>,
    u: u32,
    v: u32,
    start: Grade,
    lo: usize,
    hi: usize,
) {
    cand.clear();
    let lu = layout.words::<BANDED>(live, u, lo, hi);
    let lv = layout.words::<BANDED>(live, v, lo, hi);
    for (k, ((&xu, &xv), &already)) in lu.iter().zip(lv).zip(&active[lo..hi]).enumerate() {
        let wi = lo + k;
        // The diagonal and the target's own edge both survive in `live`, so
        // both endpoints show up here and have to go.
        let mut x = xu & xv;
        let base = (wi * 64) as u32;
        while x != 0 {
            let b = x.trailing_zeros();
            x &= x - 1;
            let c = base + b;
            if c == u || c == v {
                continue;
            }
            let birth = if already & (1u64 << b) != 0 {
                start
            } else {
                let gu = grades.get(u, c);
                let gv = grades.get(v, c);
                debug_assert!(gu != DEAD && gv != DEAD, "live bit on a trimmed edge");
                gu.max(gv)
            };
            cand.push(Cand { vertex: c, birth });
        }
    }
}

/// Is `ab` present at grade `<= t`?
///
/// `bits` is exact at the outer grade `g <= t`, so a set bit settles it from a
/// row that is `n/8` bytes wide; the matrix row is `4n`, so it is worth asking
/// the cheap question first even though both are now a single load.
#[inline]
fn adjacent_by<const BANDED: bool>(
    grades: &Grades,
    bits: &[u64],
    layout: &Layout,
    a: u32,
    b: u32,
    t: Grade,
) -> bool {
    if layout.test::<BANDED>(bits, a, b) {
        return true;
    }
    // DEAD is the maximum grade, so a trimmed or absent edge fails this too.
    grades.get(a, b) <= t
}

/// Does `c` dominate the target at `t`? That is: is every candidate already
/// born by `t` adjacent to `c` by `t`?
///
/// Returns `None` when `c` dominates, or `Some(w)` with a candidate `c` fails
/// to cover. That witness is itself disqualified as a dominator, which the
/// caller exploits.
///
/// The two bitmaps bracket the answer without touching the CSR at all:
///
/// * `bits` is exact at the outer grade `g`, and `g <= t`, so a set bit proves
///   adjacency by `t`.
/// * `live` holds every untrimmed edge, so a clear bit disproves it.
///
/// What is left — an edge that survives but has shifted above `g` — is the only
/// case needing its grade read, and that set is small by construction: it is
/// exactly the edges already retained higher up in the descent. So the common
/// path is `active & !bits[c]`, the same branchless reduction the first search
/// runs, and the CSR is touched only for the residue.
///
/// `active` is nonzero only in `lo..hi`. Under `BANDED`, mask words outside
/// `c`'s row read as absent edges, in the same ascending scan.
#[allow(clippy::too_many_arguments)]
fn dominates<const BANDED: bool>(
    grades: &Grades,
    bits: &[u64],
    live: &[u64],
    layout: &Layout,
    active: &[u64],
    lo: usize,
    hi: usize,
    c: u32,
    t: Grade,
) -> Option<u32> {
    let (origin, clo, chi) = layout.span::<BANDED>(c);
    // Ask the cheap question first, and ask it branchlessly: a per-word `miss
    // != 0` test in this loop is data-dependent, which stops the vectoriser
    // dead and costs more than the residue pass it saves. `covers` is the same
    // reduction the first search runs, and it stays that shape for a reason.
    let inside = spans_range::<BANDED>(layout, c, lo, hi);
    if inside && covers(&bits[origin + lo..origin + hi], &active[lo..hi]) {
        return None;
    }
    // Under `BANDED`, mask words before or after `c`'s row read as zero rows.
    // Clamping keeps the row slices in bounds; the words outside them come
    // first or last in the scan, as they do in a full-width row.
    let (rlo, rhi) = if inside {
        (lo, hi)
    } else {
        let rlo = clo.clamp(lo, hi);
        (rlo, chi.clamp(rlo, hi))
    };
    let row = &bits[origin + rlo..origin + rhi];
    let lrow = &live[origin + rlo..origin + rhi];
    if let Some(x) = absent_outside(&active[lo..rlo], lo) {
        return Some(x);
    }
    for (k, ((&a, &r), &l)) in active[rlo..rhi].iter().zip(row).zip(lrow).enumerate() {
        let i = rlo + k;
        let miss = a & !r;
        if miss == 0 {
            continue;
        }
        let base = (i * 64) as u32;
        // A candidate `c` cannot reach at all beats one it might: the first
        // costs nothing, the second costs a binary search.
        let absent = miss & !l;
        if absent != 0 {
            return Some(base + absent.trailing_zeros());
        }
        let mut ambiguous = miss;
        while ambiguous != 0 {
            let b = ambiguous.trailing_zeros();
            ambiguous &= ambiguous - 1;
            let x = base + b;
            // DEAD is the maximum grade, so this rejects trimmed edges too.
            if grades.get(c, x) > t {
                return Some(x);
            }
        }
    }
    absent_outside(&active[rhi..hi], rhi)
}

/// The lowest set bit of `words`, whose first word is block `first`: a
/// candidate outside the dominator's row, so certainly not adjacent to it.
#[inline]
fn absent_outside(words: &[u64], first: usize) -> Option<u32> {
    let k = words.iter().position(|&x| x != 0)?;
    Some(((first + k) * 64) as u32 + words[k].trailing_zeros())
}

#[inline(always)]
fn event_grade(e: u64) -> Grade {
    (e >> 32) as Grade
}

#[inline(always)]
fn event_index(e: u64) -> usize {
    e as u32 as usize
}

/// Candidates not yet born at `t`, as `(birth << 32) | index` in ascending
/// order, so the trajectory walks them with a cursor.
///
/// The birth rides in the key instead of being fetched from `cand` inside the
/// comparator: sorting by a closure over `cand[i].birth` makes every comparison
/// a random gather, which was more expensive than the sort itself.
fn build_events(cand: &[Cand], events: &mut Vec<u64>, t: Grade) {
    events.clear();
    events.extend(
        cand.iter()
            .enumerate()
            .filter(|(_, c)| c.birth > t)
            .map(|(i, c)| ((c.birth as u64) << 32) | i as u64),
    );
    events.sort_unstable();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    /// The banded layout makes exactly the decisions the full-width one does:
    /// same shifts, same trims, same output, for full passes and prefixes.
    #[test]
    fn banded_rows_collapse_identically() {
        for &(n, keep, levels, seed) in &[
            (300usize, 8u64, 0u64, 1u64),
            (250, 5, 4, 2),
            (200, 15, 0, 3),
            (180, 30, 6, 4),
            (70, 100, 3, 5),
        ] {
            let mut rng = xorshift(seed);
            let mut edges = Vec::new();
            for u in 1..n as u32 {
                for v in 0..u {
                    // Mostly local edges plus a few long ones.
                    let local = u - v < 90;
                    if rng() % 100 < if local { keep } else { 1 } {
                        let d = if levels == 0 {
                            (rng() >> 40) as f32 / 16_777_216.0 + 0.001
                        } else {
                            ((rng() % levels) as f32 + 1.0) * 0.5
                        };
                        edges.push(Edge { u, v, distance: d });
                    }
                }
            }
            let m = edges.len();
            for stop in [m, m / 3, 7] {
                let run = |banded: bool| {
                    let mut el = EdgeList {
                        n,
                        edges: edges.clone(),
                        center: 0,
                        threshold: f32::INFINITY,
                        sorted: false,
                    };
                    let progress =
                        collapse_in_layout(&mut el, false, Some(banded), |p| p.processed < stop);
                    let out: Vec<_> = el.edges.iter().map(|e| (e.u, e.v, e.distance.to_bits())).collect();
                    (out, progress.processed, progress.trimmed)
                };
                assert_eq!(run(true), run(false), "n={n} keep={keep} levels={levels} stop={stop}");
            }
        }
    }
}
