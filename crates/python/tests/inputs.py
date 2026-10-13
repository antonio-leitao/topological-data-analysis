"""Input handling: array layouts, dtypes, sparse formats, validation, GIL."""

import threading
import time

import numpy as np
import scipy.sparse as sp

import tda


def points(n=60, d=3, seed=0):
    return np.random.default_rng(seed).random((n, d)).astype(np.float32)


def distances(x):
    diff = x[:, None, :].astype(np.float64) - x[None, :, :]
    return np.sqrt((diff**2).sum(-1)).astype(np.float32)


def same(a, b):
    assert len(a) == len(b)
    for left, right in zip(a, b):
        left = np.asarray(left)[np.lexsort(np.asarray(left).T[::-1])]
        right = np.asarray(right)[np.lexsort(np.asarray(right).T[::-1])]
        assert np.array_equal(left, right), (left, right)


class RawCoo:
    """Minimal object exposing `tocoo()`, for arrays scipy would not accept."""

    def __init__(self, row, col, data, n):
        self.row, self.col, self.data, self.shape = row, col, data, (n, n)

    def tocoo(self):
        return self


def check_dense_layouts():
    x = points()
    want = tda.persistent_homology(x, max_dim=2)
    wide = np.zeros((x.shape[0], 2 * x.shape[1]), dtype=np.float32)
    wide[:, ::2] = x
    for label, layout in (
        ("fortran", np.asfortranarray(x)),
        ("strided", wide[:, ::2]),
        ("float64", x.astype(np.float64)),
        ("float64 fortran", np.asfortranarray(x.astype(np.float64))),
    ):
        same(want, tda.persistent_homology(layout, max_dim=2))
        assert tda.filtration_size(layout, max_dim=2) == tda.filtration_size(x, max_dim=2), label


def check_matrix_layouts():
    d = distances(points())
    # Only the strict lower triangle is read. Garbage above it makes a
    # column-major read of a Fortran array visibly wrong.
    d[np.triu_indices_from(d, 1)] *= 10.0
    want = tda.persistent_homology(d, max_dim=1, distance_matrix=True)
    for layout in (np.asfortranarray(d), d.astype(np.float64), np.asfortranarray(d.astype(np.float64))):
        same(want, tda.persistent_homology(layout, max_dim=1, distance_matrix=True))


def check_sparse_formats():
    d = distances(points())
    want = tda.persistent_homology(d, max_dim=2, distance_matrix=True)
    want_size = tda.filtration_size(d, max_dim=2, distance_matrix=True)

    csr64 = sp.csr_matrix(d)
    csr64.indptr = csr64.indptr.astype(np.int64)
    csr64.indices = csr64.indices.astype(np.int64)
    for label, graph in (
        ("csr", sp.csr_matrix(d)),
        ("csc", sp.csc_matrix(d)),
        ("coo", sp.coo_matrix(d)),
        ("csr_array", sp.csr_array(d.astype(np.float64))),
        ("lil", sp.lil_matrix(d)),
        ("csr int64", csr64),
    ):
        same(want, tda.persistent_homology(graph, max_dim=2))
        assert tda.filtration_size(graph, max_dim=2) == want_size, label

    coo = sp.coo_matrix(d)
    mixed = RawCoo(coo.row.astype(np.int32), coo.col.astype(np.int64), coo.data, d.shape[0])
    same(want, tda.persistent_homology(mixed, max_dim=2))


def expect_value_error(fn, fragment):
    try:
        fn()
    except ValueError as err:
        assert fragment in str(err), str(err)
    else:
        raise AssertionError(f"expected ValueError mentioning {fragment!r}")


def check_rejections():
    x = points()
    x[7, 1] = np.nan
    expect_value_error(lambda: tda.persistent_homology(x), "non-finite")

    d = distances(points())
    d[5, 2] = -1e-7
    expect_value_error(
        lambda: tda.persistent_homology(d, distance_matrix=True), "non-negative"
    )
    expect_value_error(lambda: tda.persistent_homology(sp.csr_matrix(d)), "non-negative")

    bad = RawCoo(
        np.array([0, -1], dtype=np.int32),
        np.array([1, 0], dtype=np.int32),
        np.array([1.0, 1.0], dtype=np.float32),
        4,
    )
    expect_value_error(lambda: tda.persistent_homology(bad), "out of range")


def check_gil_released():
    """The main thread keeps ticking while another thread computes."""
    x = np.random.default_rng(1).random((3000, 3)).astype(np.float32)
    done = threading.Event()
    elapsed = []

    def work():
        start = time.perf_counter()
        tda.persistent_homology(x, max_dim=1, peel=False, collapse=False, parallel=False)
        elapsed.append(time.perf_counter() - start)
        done.set()

    worker = threading.Thread(target=work)
    worker.start()
    last, gap = time.perf_counter(), 0.0
    while not done.is_set():
        now = time.perf_counter()
        gap = max(gap, now - last)
        last = now
    worker.join()
    assert elapsed[0] > 0.2, f"workload too small to test ({elapsed[0]:.3f}s)"
    assert gap < 0.5 * elapsed[0], f"main thread stalled {gap:.3f}s of {elapsed[0]:.3f}s"


def check_coordinate_layouts():
    theta = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    want = np.asarray(tda.coords.circular(circle).coordinates)
    for layout in (np.asfortranarray(circle), circle.astype(np.float64)):
        assert np.array_equal(want, np.asarray(tda.coords.circular(layout).coordinates))
    fitted = tda.coords.circular_map(circle)
    assert np.array_equal(
        np.asarray(fitted.transform(circle)),
        np.asarray(fitted.transform(np.asfortranarray(circle.astype(np.float64)))),
    )


def main():
    check_dense_layouts()
    check_matrix_layouts()
    check_sparse_formats()
    check_rejections()
    check_gil_released()
    check_coordinate_layouts()
    print("inputs: ok")


if __name__ == "__main__":
    main()
