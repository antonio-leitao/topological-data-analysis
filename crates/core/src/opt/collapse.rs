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

/// Run one exact backward collapse pass in place.
///
/// One call is one pass: a pass need not reach a minimal graph, and calling
/// again may remove more. A caller that wants a fixed point calls again.
pub fn collapse_in_place(edge_list: &mut EdgeList, parallel: bool) {
    edge_list.sort(parallel);
    if edge_list.edges.len() < 2 {
        return;
    }
    debug_assert!({
        edge_list.assert_preconditions();
        true
    });
    let mut c = Collapser::build(edge_list);
    c.run_pass();
    c.write_back(edge_list);
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
}

struct Collapser {
    topo: Topo,
    /// The graph state: `grades[a * gstride + b]` is edge `ab`'s current grade,
    /// or `DEAD` when it is absent or trimmed. Single source of truth.
    grades: Grades,
    /// Closed adjacency of `G_t` at the current outer grade, `n` rows of `w`
    /// words. The backward loop only ever descends, so every edge leaves this
    /// exactly once: `O(m)` bit clears for the whole pass, and no level sets.
    bits: Vec<u64>,
    /// Adjacency of every *surviving* edge, at any grade. Only trimming clears
    /// a bit here, so `L[u] & L[v]` minus the active set answers "does this
    /// edge have anything left to shift across" in one pass over a row.
    live: Vec<u64>,
    /// Words per bitmap row.
    w: usize,
    /// `grade_start[g]..grade_start[g + 1]` are the edge ids at grade `g`.
    grade_start: Vec<u32>,
    grade_values: Vec<f32>,
    scratch: Scratch,
}

#[inline(always)]
fn bit_row(bits: &[u64], w: usize, r: u32) -> &[u64] {
    let o = r as usize * w;
    &bits[o..o + w]
}

#[inline(always)]
fn clear_edge(bits: &mut [u64], w: usize, u: u32, v: u32) {
    bits[u as usize * w + (v as usize >> 6)] &= !(1u64 << (v & 63));
    bits[v as usize * w + (u as usize >> 6)] &= !(1u64 << (u & 63));
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl Collapser {
    fn build(el: &EdgeList) -> Self {
        let n = el.n;
        let m = el.edges.len();
        assert!(
            u32::try_from(2 * m + n).is_ok() && n < u32::MAX as usize,
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
        let mut bits = vec![0u64; n * w];
        for r in 0..n {
            bits[r * w + (r >> 6)] |= 1u64 << (r & 63);
        }
        for e in &el.edges {
            bits[e.u as usize * w + (e.v as usize >> 6)] |= 1u64 << (e.v & 63);
            bits[e.v as usize * w + (e.u as usize >> 6)] |= 1u64 << (e.u & 63);
        }

        let live = bits.clone();

        let mut scratch = Scratch::default();
        scratch.cand_bits = vec![0u64; w];

        Collapser {
            topo: Topo { edge_u, edge_v },
            grades,
            bits,
            live,
            w,
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
    /// inside each group.
    fn run_pass(&mut self) {
        let Collapser {
            topo,
            grades,
            bits,
            live,
            w,
            grade_start,
            scratch,
            ..
        } = self;
        let w = *w;

        // The bitmap invariant is maintained by the descent alone. On entry to
        // group `g` it holds exactly `{e : current_grade(e) <= g}`: everything
        // below is unprocessed and still at its original grade, and everything
        // above has either been retired here or was cleared when it shifted or
        // was trimmed. Grades are dense ranks, so retiring group `g + 1` is the
        // whole transition — no edge can land between two consecutive ranks.
        for g in (0..grade_start.len() - 1).rev() {
            if g + 2 < grade_start.len() {
                for eid in grade_start[g + 1]..grade_start[g + 2] {
                    let e = eid as usize;
                    clear_edge(bits, w, topo.edge_u[e], topo.edge_v[e]);
                }
            }
            for eid in (grade_start[g]..grade_start[g + 1]).rev() {
                process(topo, grades, bits, live, w, scratch, eid as usize);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn process(
    topo: &Topo,
    grades: &mut Grades,
    bits: &mut [u64],
    live: &mut [u64],
    w: usize,
    s: &mut Scratch,
    eid: usize,
) {
    let Scratch {
        cand,
        events,
        rejected,
        search_id: next_search_id,
        cand_bits,
    } = s;
    let u = topo.edge_u[eid];
    let v = topo.edge_v[eid];
    let start = grades.get(u, v);
    debug_assert_ne!(start, DEAD, "an edge is scheduled at most once per pass");

    // Keyed by vertex, and the widest vertex id is bounded by a bitmap row.
    if rejected.len() < w * 64 {
        rejected.resize(w * 64, 0);
    }

    // ---- discovery: the active common neighbourhood is one AND -------------
    //
    // `bits` is exact at the edge's own grade, so `A[u] & A[v]` is precisely
    // `C_start`. The diagonal makes both endpoints show up in the AND, so clear
    // them; everything else that survives is a live common neighbour.
    let any = {
        let (ru, rv) = (bit_row(bits, w, u), bit_row(bits, w, v));
        and_rows(&mut cand_bits[..w], ru, rv)
    };
    clear_bit(&mut cand_bits[..w], u);
    clear_bit(&mut cand_bits[..w], v);

    if any == 0 || cand_bits[..w].iter().all(|&x| x == 0) {
        // Nothing can dominate it, so it stays where it is.
        return;
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
    let mut dom = NO_DOMINATOR;
    'sweep: for wi in (0..w).rev() {
        let mut x = cand_bits[wi];
        let base = (wi * 64) as u32;
        while x != 0 {
            let b = 63 - x.leading_zeros();
            x &= !(1u64 << b);
            let c = base + b;
            let row = bit_row(bits, w, c);
            if covers(row, &cand_bits[..w]) {
                dom = c;
                break 'sweep;
            }
        }
    }

    if dom == NO_DOMINATOR {
        // Critical at its own grade: retained, and nothing else to compute.
        return;
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
        let (lu, lv) = (bit_row(live, w, u), bit_row(live, w, v));
        let mut acc = 0u64;
        for ((&x, &y), &c) in lu.iter().zip(lv).zip(cand_bits.iter()) {
            acc |= (x & y) & !c;
        }
        acc != 0
    };
    if !has_future {
        grades.set(u, v, DEAD);
        clear_edge(bits, w, u, v);
        clear_edge(live, w, u, v);
        return;
    }

    // ---- the trajectory ---------------------------------------------------
    //
    // Only edges that actually shift pay for exact births and the event list.
    collect_candidates(grades, live, w, &cand_bits[..w], cand, u, v, start);
    build_events(cand, events, start);
    let mut ev = 0usize;
    let mut t;

    'outer: loop {
        // Shift while the incumbent survives. Between two arrival grades the
        // neighbourhood is unchanged, so domination cannot break
        // (Glisse-Pritam Lemma 4) and only arrivals need re-checking.
        loop {
            if ev == events.len() {
                grades.set(u, v, DEAD); // trimmed
                clear_edge(bits, w, u, v);
                clear_edge(live, w, u, v);
                return;
            }
            t = event_grade(events[ev]);

            // Every arrival at this grade becomes visible before the domination
            // status is re-read.
            let mut still_dominated = true;
            while ev < events.len() && event_grade(events[ev]) == t {
                let cw = cand[event_index(events[ev])].vertex;
                ev += 1;
                if cw != dom && !adjacent_by(grades, bits, w, dom, cw, t) {
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
        cand_bits[..w].fill(0);
        for c in cand.iter().filter(|c| c.birth <= t) {
            set_bit(&mut cand_bits[..w], c.vertex);
        }

        *next_search_id += 1;
        let search_id = *next_search_id;
        let mut found = NO_DOMINATOR;
        'search: for wi in (0..w).rev() {
            let mut x = cand_bits[wi];
            let base = (wi * 64) as u32;
            while x != 0 {
                let b = 63 - x.leading_zeros();
                x &= !(1u64 << b);
                let c = base + b;
                if rejected[c as usize] == search_id {
                    continue;
                }
                match dominates(grades, bits, live, w, &cand_bits[..w], c, t) {
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
    clear_edge(bits, w, u, v); // its grade is now above the outer grade
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
fn collect_candidates(
    grades: &Grades,
    live: &[u64],
    w: usize,
    active: &[u64],
    cand: &mut Vec<Cand>,
    u: u32,
    v: u32,
    start: Grade,
) {
    cand.clear();
    let (lu, lv) = (bit_row(live, w, u), bit_row(live, w, v));
    for wi in 0..w {
        // The diagonal and the target's own edge both survive in `live`, so
        // both endpoints show up here and have to go.
        let mut x = lu[wi] & lv[wi];
        let already = active[wi];
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
fn adjacent_by(grades: &Grades, bits: &[u64], w: usize, a: u32, b: u32, t: Grade) -> bool {
    if bits[a as usize * w + (b as usize >> 6)] & (1u64 << (b & 63)) != 0 {
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
#[allow(clippy::too_many_arguments)]
fn dominates(
    grades: &Grades,
    bits: &[u64],
    live: &[u64],
    w: usize,
    active: &[u64],
    c: u32,
    t: Grade,
) -> Option<u32> {
    let row = bit_row(bits, w, c);
    let lrow = bit_row(live, w, c);
    // Ask the cheap question first, and ask it branchlessly: a per-word `miss
    // != 0` test in this loop is data-dependent, which stops the vectoriser
    // dead and costs more than the residue pass it saves. `covers` is the same
    // reduction the first search runs, and it stays that shape for a reason.
    if covers(row, active) {
        return None;
    }
    for i in 0..w {
        let miss = active[i] & !row[i];
        if miss == 0 {
            continue;
        }
        let base = (i * 64) as u32;
        // A candidate `c` cannot reach at all beats one it might: the first
        // costs nothing, the second costs a binary search.
        let absent = miss & !lrow[i];
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
    None
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
