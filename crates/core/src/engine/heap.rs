// ═══════════════════════════════════════════════════════════════════════════════
// FastHeap — 4-ary max-heap (eager sift-up, drop-in for the 2-ary version)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Replace the entire `pub struct FastHeap { ... }` and `impl FastHeap { ... }`
// in reduction.rs with this section. Existing call sites compile unchanged:
//
//     new, with_capacity, clear, len, is_empty,
//     peek (&self), push, pop, pop_pivot, get_pivot
//
// SEMANTICS ARE EAGER. Every `push` restores the heap invariant before
// returning. There is no dirty flag, no lazy heapify. This is the part the
// previous lazy-heapify version got wrong: in compute_pairs, each V-column
// add is followed by exactly one `get_pivot`, and a typical V-column add is
// a small handful of pushes. Lazy heapify paid O(N) per read regardless of
// batch size — catastrophic when batches are small and N grows into the
// thousands. Eager sift-up costs O(log N) per push, log N per pop.
//
// WHY 4-ARY OVER 2-ARY
// ────────────────────
// For node i:
//   - parent      = (i - 1) / 4
//   - children    = 4i + 1, 4i + 2, 4i + 3, 4i + 4
//
// Tree depth is ⌈log₄(N)⌉ ≈ ½ × ⌈log₂(N)⌉. The push loop runs ½ as many
// iterations on average — the same loop currently showing 277k samples on
// `mov x9, x23` (the loop-back) and 148k on `add x13, x8, x23, lsl #4`.
//
// Pop costs more per level (max-of-4 vs max-of-2) but pop is rare relative
// to push in this workload. The four child slots are contiguous (4·16 = 64
// bytes = one cache line on AArch64), so the 4-way max is a single cache
// fill, not four scattered loads.
//
// TUNING NOTE
// ───────────
// If after profiling 4-ary push is still hot, the next move is *not*
// lazy heapify. It's:
//   1) optimization #5 from the earlier analysis: track the running max
//      during the init-phase pushes in `init_coboundary_and_get_pivot`
//      and return it directly — no heap read needed at all for init.
//      That's a separate change in reduction.rs, not in FastHeap.
//   2) inline-attribute audit: confirm `Simplex128`'s `Ord` impl is being
//      inlined into the sift-up. Check with `cargo asm`. If you see a
//      `bl` to a comparison helper, slap `#[inline(always)]` on the impl.

use crate::engine::simplex::Simplex128;

/// Branching factor. 4 is a good default; increase to 8 only after measuring.
const ARITY: usize = 8;

pub struct FastHeap {
    data: Vec<Simplex128>,
}

impl FastHeap {
    #[inline]
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            data: Vec::with_capacity(cap),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.data.clear();
    }

    #[inline]
    pub fn peek(&self) -> Option<&Simplex128> {
        self.data.first()
    }

    // ── Push: append + sift up (4-ary) ─────────────────────────────────────
    //
    // Hot path. Same loop shape as the previous 2-ary version — the only
    // change is `parent = (pos - 1) / ARITY`. With ARITY = 4 the tree is
    // half as deep so the loop runs half as many iterations on average.

    #[inline(always)]
    pub fn push(&mut self, item: Simplex128) {
        let pos = self.data.len();
        self.data.push(item);
        if pos == 0 {
            return;
        }
        // SAFETY: we just pushed at index `pos`, so data has `pos + 1`
        // elements; ptr is valid for that range. We only ever index in
        // [0, pos], inclusive.
        unsafe {
            let ptr = self.data.as_mut_ptr();
            let mut pos = pos;
            while pos > 0 {
                let parent = (pos - 1) / ARITY;
                let p_val = *ptr.add(parent);
                if p_val >= item {
                    break;
                }
                *ptr.add(pos) = p_val;
                pos = parent;
            }
            *ptr.add(pos) = item;
        }
    }
    /// Append without restoring the heap invariant. Caller MUST call
    /// `heapify()` before the next read (`peek` / `pop` / `get_pivot` /
    /// `pop_pivot`). Useful for bulk-loading an empty heap, where Floyd's
    /// O(N) build beats N sequential O(log N) sift-ups.
    #[inline(always)]
    pub fn append_raw(&mut self, item: Simplex128) {
        self.data.push(item);
    }

    /// Restore the heap invariant over the entire buffer in O(N) using
    /// bottom-up sift-down (Floyd's algorithm, 4-ary variant).
    pub fn heapify(&mut self) {
        let n = self.data.len();
        if n <= 1 {
            return;
        }
        // Last node with at least one child: parent of the last leaf.
        // For 4-ary: parent(i) = (i - 1) / 4, so the largest internal
        // node index is (n - 2) / 4.
        let last_internal = (n - 2) / ARITY;
        // SAFETY: indices in [0, last_internal] are all < n; sift_down_4ary
        // only accesses [0, n). Simplex128 is Copy, so the temporary `item`
        // copy is sound even with the raw pointer aliasing the buffer.
        unsafe {
            let ptr = self.data.as_mut_ptr();
            let mut pos = last_internal as isize;
            while pos >= 0 {
                let item = *ptr.add(pos as usize);
                sift_down_4ary(ptr, pos as usize, n, item);
                pos -= 1;
            }
        }
    }
    // ── Pop: take root, sift last down (4-ary) ─────────────────────────────

    pub fn pop(&mut self) -> Option<Simplex128> {
        let n = self.data.len();
        if n == 0 {
            return None;
        }
        // SAFETY: n ≥ 1 verified above. set_len with n−1 leaves data valid
        // because Simplex128 is Copy. ptr remains valid for the original n
        // elements until function return.
        unsafe {
            let ptr = self.data.as_mut_ptr();
            let result = *ptr;
            if n == 1 {
                self.data.set_len(0);
                return Some(result);
            }
            let last = *ptr.add(n - 1);
            self.data.set_len(n - 1);
            sift_down_4ary(ptr, 0, n - 1, last);
            Some(result)
        }
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

// ── 4-ary sift-down ────────────────────────────────────────────────────────
//
// Walks `pos` downward toward the leaves while `item` would violate the
// heap invariant against the largest of pos's up-to-4 children. The four
// child slots are contiguous in memory (4·16 = 64 bytes = one cache line),
// so the four loads happen on a single cache fill.
//
// SAFETY: caller guarantees `pos < end ≤ data length` and that ptr is
// valid for at least `end` elements. The function only accesses indices
// in [0, end).
// #[inline(always)]
// unsafe fn sift_down_4ary(ptr: *mut Simplex128, mut pos: usize, end: usize, item: Simplex128) {
//     loop {
//         let first = ARITY * pos + 1;
//         if first >= end {
//             break;
//         }
//         // Find the index of the largest among up to 4 children.
//         // Unrolled because the iteration count is fixed and small.
//         let last_excl = (first + ARITY).min(end);
//         let mut best = first;
//         let mut best_val = *ptr.add(first);
//
//         if first + 1 < last_excl {
//             let v = *ptr.add(first + 1);
//             if v > best_val {
//                 best = first + 1;
//                 best_val = v;
//             }
//         }
//         if first + 2 < last_excl {
//             let v = *ptr.add(first + 2);
//             if v > best_val {
//                 best = first + 2;
//                 best_val = v;
//             }
//         }
//         if first + 3 < last_excl {
//             let v = *ptr.add(first + 3);
//             if v > best_val {
//                 best = first + 3;
//                 best_val = v;
//             }
//         }
//         //ONLY IF ARITY ABOVE 4!!
//         if first + 4 < last_excl {
//             let v = *ptr.add(first + 4);
//             if v > best_val {
//                 best = first + 4;
//                 best_val = v;
//             }
//         }
//         if first + 5 < last_excl {
//             let v = *ptr.add(first + 5);
//             if v > best_val {
//                 best = first + 5;
//                 best_val = v;
//             }
//         }
//         if first + 6 < last_excl {
//             let v = *ptr.add(first + 6);
//             if v > best_val {
//                 best = first + 6;
//                 best_val = v;
//             }
//         }
//         if first + 7 < last_excl {
//             let v = *ptr.add(first + 7);
//             if v > best_val {
//                 best = first + 7;
//                 best_val = v;
//             }
//         }
//
//         if item >= best_val {
//             break;
//         }
//         *ptr.add(pos) = best_val;
//         pos = best;
//     }
//     *ptr.add(pos) = item;
// }
#[inline(always)]
unsafe fn sift_down_4ary(ptr: *mut Simplex128, mut pos: usize, end: usize, item: Simplex128) {
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
}
