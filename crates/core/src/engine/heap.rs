// FastHeap: a collection of local N-ary heaps (N = ARITY), with a small binary
// heap containing one head per local heap. Consecutive pushes form a batch;
// the first read Floyd-heapifies that batch in O(k), instead of paying k
// dependent sift-ups through one potentially huge global heap.

use crate::engine::simplex::Simplex128;
use std::collections::{binary_heap::PeekMut, BinaryHeap};

const ARITY: usize = 8;
const COMPACT_MIN_ENTRIES: usize = 16 * 1024;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct RunHead {
    key: Simplex128,
    run: usize,
}

#[derive(Clone, Copy)]
struct Run {
    start: usize,
    len: usize,
}

pub struct FastHeap {
    arena: Vec<Simplex128>,
    runs: Vec<Run>,
    heads: BinaryHeap<RunHead>,
    pending_start: usize,
    live_len: usize,
}

impl FastHeap {
    #[inline]
    pub fn new() -> Self {
        Self {
            arena: Vec::new(),
            runs: Vec::new(),
            heads: BinaryHeap::new(),
            pending_start: 0,
            live_len: 0,
        }
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            arena: Vec::with_capacity(cap),
            ..Self::new()
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.live_len == 0
    }

    #[inline]
    pub fn clear(&mut self) {
        self.arena.clear();
        self.runs.clear();
        self.heads.clear();
        self.pending_start = 0;
        self.live_len = 0;
    }

    #[inline]
    pub fn peek(&mut self) -> Option<&Simplex128> {
        self.finalize_pending();
        self.heads.peek().map(|head| &head.key)
    }

    #[inline(always)]
    pub fn push(&mut self, item: Simplex128) {
        self.arena.push(item);
        self.live_len += 1;
    }

    /// Append to the current batch. Kept as the explicit bulk-build spelling
    /// used by the initial coboundary path.
    #[inline(always)]
    pub fn append_raw(&mut self, item: Simplex128) {
        self.push(item);
    }

    #[inline]
    pub fn heapify(&mut self) {
        self.finalize_pending();
    }

    fn finalize_pending(&mut self) {
        let end = self.arena.len();
        let len = end - self.pending_start;
        if len == 0 {
            return;
        }

        unsafe {
            heapify_slice(self.arena.as_mut_ptr().add(self.pending_start), len);
        }

        let run = self.runs.len();
        self.runs.push(Run {
            start: self.pending_start,
            len,
        });
        self.heads.push(RunHead {
            key: self.arena[self.pending_start],
            run,
        });
        self.pending_start = end;
    }

    pub fn pop(&mut self) -> Option<Simplex128> {
        self.finalize_pending();

        let result;
        {
            let mut head = self.heads.peek_mut()?;
            let run = &mut self.runs[head.run];
            result = unsafe { pop_heap(self.arena.as_mut_ptr().add(run.start), &mut run.len) };
            debug_assert_eq!(result, head.key);

            if run.len == 0 {
                PeekMut::pop(head);
            } else {
                head.key = self.arena[run.start];
            }
        }

        self.live_len -= 1;
        self.maybe_compact();
        Some(result)
    }

    fn maybe_compact(&mut self) {
        let dead = self.arena.len() - self.live_len;
        if self.arena.len() < COMPACT_MIN_ENTRIES || dead <= self.live_len {
            return;
        }

        let mut write = 0;
        for run in &mut self.runs {
            if run.len == 0 {
                continue;
            }
            if run.start != write {
                self.arena
                    .copy_within(run.start..run.start + run.len, write);
            }
            run.start = write;
            write += run.len;
        }
        debug_assert_eq!(write, self.live_len);
        self.arena.truncate(write);
        self.pending_start = write;
    }

    // ── Z/2 cancellation (unchanged from original) ─────────────────────────

    pub fn pop_pivot(&mut self) -> Option<Simplex128> {
        loop {
            let top = self.pop()?;
            match self.peek() {
                Some(&next) if next == top => {
                    self.pop(); // cancels in Z/2
                }
                _ => return Some(top),
            }
        }
    }

    pub fn get_pivot(&mut self) -> Option<Simplex128> {
        let pivot = self.pop_pivot()?;
        self.push(pivot);
        Some(pivot)
    }
}

impl Default for FastHeap {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
unsafe fn sift_down(ptr: *mut Simplex128, mut pos: usize, end: usize, item: Simplex128) {
    loop {
        let first = ARITY * pos + 1;
        if first >= end {
            break;
        }

        let (best, best_val) = if first + 7 < end {
            // Fast path: all 8 children present. Tournament in 3 rounds.
            // The 8 loads are independent; the compiler issues them in parallel.
            let v0 = *ptr.add(first);
            let v1 = *ptr.add(first + 1);
            let v2 = *ptr.add(first + 2);
            let v3 = *ptr.add(first + 3);
            let v4 = *ptr.add(first + 4);
            let v5 = *ptr.add(first + 5);
            let v6 = *ptr.add(first + 6);
            let v7 = *ptr.add(first + 7);

            // Round 1: 4 parallel pair-compares.
            let (i01, v01) = if v1 > v0 {
                (first + 1, v1)
            } else {
                (first, v0)
            };
            let (i23, v23) = if v3 > v2 {
                (first + 3, v3)
            } else {
                (first + 2, v2)
            };
            let (i45, v45) = if v5 > v4 {
                (first + 5, v5)
            } else {
                (first + 4, v4)
            };
            let (i67, v67) = if v7 > v6 {
                (first + 7, v7)
            } else {
                (first + 6, v6)
            };

            // Round 2: 2 parallel.
            let (i0123, v0123) = if v23 > v01 { (i23, v23) } else { (i01, v01) };
            let (i4567, v4567) = if v67 > v45 { (i67, v67) } else { (i45, v45) };

            // Round 3: final.
            if v4567 > v0123 {
                (i4567, v4567)
            } else {
                (i0123, v0123)
            }
        } else {
            // Tail: < 8 children present. Runs at most once per pop (the very
            // last internal node), so sequential is fine.
            let last_excl = end;
            let mut best = first;
            let mut best_val = *ptr.add(first);
            let mut k = 1usize;
            while k < ARITY && first + k < last_excl {
                let v = *ptr.add(first + k);
                if v > best_val {
                    best = first + k;
                    best_val = v;
                }
                k += 1;
            }
            (best, best_val)
        };

        if item >= best_val {
            break;
        }
        *ptr.add(pos) = best_val;
        pos = best;
    }
    *ptr.add(pos) = item;
}

#[inline]
unsafe fn heapify_slice(ptr: *mut Simplex128, len: usize) {
    if len <= 1 {
        return;
    }
    let mut pos = ((len - 2) / ARITY) as isize;
    while pos >= 0 {
        let item = *ptr.add(pos as usize);
        sift_down(ptr, pos as usize, len, item);
        pos -= 1;
    }
}

#[inline]
unsafe fn pop_heap(ptr: *mut Simplex128, len: &mut usize) -> Simplex128 {
    let n = *len;
    debug_assert!(n > 0);
    let result = *ptr;
    *len = n - 1;
    if n > 1 {
        let last = *ptr.add(n - 1);
        sift_down(ptr, 0, n - 1, last);
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn s(x: u128) -> Simplex128 {
        Simplex128(x)
    }

    #[test]
    fn drain_returns_descending() {
        let mut h = FastHeap::with_capacity(16);
        for v in [3u128, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3] {
            h.push(s(v));
        }
        let mut out = Vec::new();
        while let Some(x) = h.pop() {
            out.push(x.0);
        }
        let mut expected = vec![3u128, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3];
        expected.sort_by(|a, b| b.cmp(a));
        assert_eq!(out, expected);
    }

    #[test]
    fn peek_matches_pop() {
        let mut h = FastHeap::new();
        for v in [10u128, 50, 20, 90, 40, 30] {
            h.push(s(v));
        }
        let peeked = *h.peek().unwrap();
        let popped = h.pop().unwrap();
        assert_eq!(peeked, popped);
        assert_eq!(popped.0, 90);
    }

    #[test]
    fn empty_returns_none() {
        let mut h = FastHeap::new();
        assert!(h.peek().is_none());
        assert!(h.pop().is_none());
        assert!(h.pop_pivot().is_none());
        assert!(h.get_pivot().is_none());
        assert!(h.is_empty());
    }

    #[test]
    fn z2_pairs_cancel() {
        let mut h = FastHeap::new();
        for _ in 0..6 {
            h.push(s(42));
        }
        assert!(h.pop_pivot().is_none());
        assert!(h.is_empty());
    }

    #[test]
    fn z2_odd_count_returns_one() {
        let mut h = FastHeap::new();
        for _ in 0..7 {
            h.push(s(42));
        }
        assert_eq!(h.pop_pivot(), Some(s(42)));
        assert!(h.is_empty());
    }

    #[test]
    fn z2_with_smaller_pivot_below() {
        let mut h = FastHeap::new();
        // Two 9s cancel, 5 should survive.
        h.push(s(9));
        h.push(s(5));
        h.push(s(9));
        assert_eq!(h.pop_pivot(), Some(s(5)));
    }

    #[test]
    fn get_pivot_keeps_element() {
        let mut h = FastHeap::new();
        h.push(s(3));
        h.push(s(7));
        h.push(s(5));
        assert_eq!(h.get_pivot(), Some(s(7)));
        assert_eq!(h.pop(), Some(s(7)));
    }

    #[test]
    fn random_inserts_preserve_invariant() {
        let mut h = FastHeap::with_capacity(2048);
        let mut x: u64 = 0xdead_beef_cafe_babe;
        let mut pushed: Vec<u128> = Vec::new();
        for _ in 0..2048 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let v = (x as u128) & 0xffff_ffff;
            h.push(s(v));
            pushed.push(v);
        }
        pushed.sort_by(|a, b| b.cmp(a));
        let mut out = Vec::with_capacity(pushed.len());
        while let Some(v) = h.pop() {
            out.push(v.0);
        }
        assert_eq!(out, pushed);
    }

    #[test]
    fn interleaved_push_pop() {
        let mut h = FastHeap::new();
        h.push(s(5));
        h.push(s(3));
        assert_eq!(h.pop(), Some(s(5)));
        h.push(s(8));
        h.push(s(1));
        h.push(s(6));
        assert_eq!(h.pop(), Some(s(8)));
        assert_eq!(h.pop(), Some(s(6)));
        assert_eq!(h.pop(), Some(s(3)));
        assert_eq!(h.pop(), Some(s(1)));
        assert!(h.is_empty());
    }

    #[test]
    fn clear_resets() {
        let mut h = FastHeap::new();
        for v in [1u128, 2, 3, 4, 5] {
            h.push(s(v));
        }
        h.clear();
        assert!(h.is_empty());
        h.push(s(7));
        assert_eq!(h.pop(), Some(s(7)));
    }
    #[test]
    fn heapify_matches_sequential_push() {
        let mut x: u64 = 0xfeed_face_dead_beef;
        let mut values = Vec::with_capacity(2048);
        for _ in 0..2048 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            values.push(s((x as u128) & 0xffff_ffff));
        }

        // Build via sequential push (the reference).
        let mut h_push = FastHeap::with_capacity(values.len());
        for &v in &values {
            h_push.push(v);
        }

        // Build via append_raw + heapify (the optimized path).
        let mut h_bulk = FastHeap::with_capacity(values.len());
        for &v in &values {
            h_bulk.append_raw(v);
        }
        h_bulk.heapify();

        // Drain both and compare. Heap order is not unique, but pop order IS.
        let mut out_push = Vec::new();
        while let Some(v) = h_push.pop() {
            out_push.push(v);
        }
        let mut out_bulk = Vec::new();
        while let Some(v) = h_bulk.pop() {
            out_bulk.push(v);
        }
        assert_eq!(out_push, out_bulk);
    }

    #[test]
    fn multiple_batches_match_binary_heap() {
        let mut h = FastHeap::new();
        let mut reference = BinaryHeap::new();
        let mut x: u64 = 0xa5a5_1234_dead_beef;

        for phase in 0..32 {
            for _ in 0..512 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let value = s(x as u128);
                h.push(value);
                reference.push(value);
            }
            h.heapify();

            if phase % 3 == 2 {
                for _ in 0..300 {
                    assert_eq!(h.pop(), reference.pop());
                }
            }
        }

        while let Some(expected) = reference.pop() {
            assert_eq!(h.pop(), Some(expected));
        }
        assert!(h.is_empty());
    }

    #[test]
    fn compaction_preserves_run_offsets() {
        let mut h = FastHeap::new();
        for value in 0..20_000u128 {
            h.push(s(value));
        }
        h.heapify();

        for expected in (5_000..20_000u128).rev() {
            assert_eq!(h.pop(), Some(s(expected)));
        }
        assert!(h.arena.len() < 20_000);

        for value in 30_000..31_000u128 {
            h.push(s(value));
        }
        for expected in (30_000..31_000u128).rev() {
            assert_eq!(h.pop(), Some(s(expected)));
        }
        for expected in (0..5_000u128).rev() {
            assert_eq!(h.pop(), Some(s(expected)));
        }
        assert!(h.is_empty());
    }
}
