import argparse
import math
import os
import time
from pathlib import Path

import gudhi
from gudhi.sklearn import RipsPersistence
import numpy as np

import tda


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
REFERENCE_TOL = 3e-3
# The quotient (coperto) filtration is multiplicatively 3-interleaved with VR,
# i.e. additively log(3)-interleaved in log scale, so by stability the
# bottleneck distance between log-transformed diagrams is <= log(3).
LOG3_TOL = math.log(3.0) + 1e-4
LOG_EPS = 1e-30  # clamp for log(0) births; both sides clamp identically


def default_max_dim():
    value = int(os.environ.get("TDA_MAX_DIM", "1"))
    if not 1 <= value <= 7:
        raise ValueError("max_dim must be in 1..=7")
    return value


def dataset_files():
    """The datasets to run are controlled by data/datasets.txt (order included)."""
    manifest = DATA_DIR / "datasets.txt"
    return [
        line.strip()
        for line in manifest.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def load_points(file):
    return np.loadtxt(DATA_DIR / file).astype(np.float32)


# peel defaults to True; each variant pins it so the four columns stay
# "vanilla" and "each transform in isolation".
def run_tda(data, max_dim):
    return tda.persistent_homology(data, max_dim=max_dim, peel=False)


def run_tda_peel(data, max_dim):
    return tda.persistent_homology(data, max_dim=max_dim, peel=True)


def run_tda_quotient(data, max_dim):
    return tda.persistent_homology(data, max_dim=max_dim, quotient=True, peel=False)


def run_tda_collapse(data, max_dim):
    return tda.persistent_homology(data, max_dim=max_dim, collapse=True, peel=False)


def gudhi_reference(data, max_dim):
    dimensions = tuple(range(max_dim + 1))
    return RipsPersistence(
        homology_dimensions=dimensions,
        n_jobs=1,
        input_type="point cloud",
        num_collapses=0,  # Do not use strong collapses for the baseline.
        homology_coeff_field=2,
    ).fit_transform([data])[0]


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


def log_diagram(dgm):
    """Map (birth, death) into log scale; inf deaths stay inf."""
    a = np.asarray(dgm, dtype=np.float64).reshape(-1, 2)
    out = a.copy()
    out[:, 0] = np.log(np.maximum(a[:, 0], LOG_EPS))
    finite = np.isfinite(a[:, 1])
    out[finite, 1] = np.log(np.maximum(a[finite, 1], LOG_EPS))
    return out


def interleaving_ok(got, want, max_dim, tol):
    """Bottleneck distance between log-scale diagrams <= tol, per dimension."""
    if len(got) < max_dim + 1 or len(want) < max_dim + 1:
        return False, None, float("inf")
    for dim in range(max_dim + 1):
        a = log_diagram(got[dim])
        b = log_diagram(want[dim])
        bd = 0.0 if len(a) == 0 and len(b) == 0 else gudhi.bottleneck_distance(a, b)
        if bd > tol:
            return False, dim, bd
    return True, None, 0.0


def bottleneck_ok(got, want, max_dim, tol):
    if len(got) < max_dim + 1 or len(want) < max_dim + 1:
        return False, None, float("inf"), len(got), len(want)
    for dim in range(max_dim + 1):
        a = np.asarray(got[dim], dtype=np.float64).reshape(-1, 2)
        b = np.asarray(want[dim], dtype=np.float64).reshape(-1, 2)
        bd = 0.0 if len(a) == 0 and len(b) == 0 else gudhi.bottleneck_distance(a, b)
        if bd > tol:
            return False, dim, bd, len(a), len(b)
    return True, None, 0.0, 0, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "max_dim", nargs="?", type=int, choices=[1, 2], default=default_max_dim()
    )
    args = parser.parse_args()

    print(f"Python correctness: BitCSR H0..H{args.max_dim} vs GUDHI")
    print(
        f"  {'dataset':<24} {'status':<6} "
        f"{'tda':>10} {'peel':>10} {'quot':>10} {'clps':>10} {'gudhi':>10}"
    )
    for file in dataset_files():
        data = load_points(file)
        got, tda_time = timed(run_tda, data, args.max_dim)
        peeled, peel_time = timed(run_tda_peel, data, args.max_dim)
        quot, quot_time = timed(run_tda_quotient, data, args.max_dim)
        clps, clps_time = timed(run_tda_collapse, data, args.max_dim)
        ref, gudhi_time = timed(gudhi_reference, data, args.max_dim)

        # Plain, peel, and collapse must all match GUDHI exactly: peel and
        # edge collapse are barcode-preserving.
        for label, dgms in (("plain", got), ("peel", peeled), ("collapse", clps)):
            ok, dim, bd, n_got, n_ref = bottleneck_ok(
                dgms, ref, args.max_dim, REFERENCE_TOL
            )
            if not ok:
                raise AssertionError(
                    f"{file} {label} H{dim} mismatch: bottleneck={bd:.3e}, "
                    f"tda={n_got}, gudhi={n_ref}"
                )

        # Quotient is an approximation: enforce the log(3) interleaving bound.
        ok, dim, bd = interleaving_ok(quot, ref, args.max_dim, LOG3_TOL)
        if not ok:
            raise AssertionError(
                f"{file} quotient H{dim} log-bottleneck {bd:.4f} "
                f"exceeds log(3)={math.log(3.0):.4f}"
            )

        print(
            f"  {file:<24} {'ok':<6} "
            f"{tda_time:9.3f}s {peel_time:9.3f}s {quot_time:9.3f}s "
            f"{clps_time:9.3f}s {gudhi_time:9.3f}s"
        )


if __name__ == "__main__":
    main()
