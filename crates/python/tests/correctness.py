import argparse
import os
import time
from pathlib import Path

import gudhi
from gudhi.sklearn import RipsPersistence
import numpy as np

import tda
import tda._core as core


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
H2_LAST_DATASET = "hiv1.txt"
REFERENCE_TOL = 3e-3


def default_max_dim():
    value = int(os.environ.get("TDA_MAX_DIM", "1"))
    if value not in (1, 2):
        raise ValueError("max_dim must be 1 or 2")
    return value


def dataset_files(max_dim):
    manifest = DATA_DIR / "datasets.txt"
    files = [
        line.strip()
        for line in manifest.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if max_dim == 2:
        return files[: files.index(H2_LAST_DATASET) + 1]
    return files


def load_points(file):
    return np.loadtxt(DATA_DIR / file).astype(np.float32)


def run_tda(data, mode, max_dim):
    if mode == "dense":
        return tda.persistent_homology(data, max_dim=max_dim)
    return core._persistent_homology_sparse(data, max_dim=max_dim)


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
    parser.add_argument("mode", nargs="?", choices=["dense", "sparse"])
    parser.add_argument(
        "max_dim", nargs="?", type=int, choices=[1, 2], default=default_max_dim()
    )
    args = parser.parse_args()

    modes = [args.mode] if args.mode else ["dense", "sparse"]
    for mode in modes:
        print(f"Python correctness: {mode} H0..H{args.max_dim} vs GUDHI")
        print(f"  {'dataset':<24} {'status':<6} {f'tda_{mode}':>10} {'gudhi':>10}")
        for file in dataset_files(args.max_dim):
            data = load_points(file)
            got, tda_time = timed(run_tda, data, mode, args.max_dim)
            ref, gudhi_time = timed(gudhi_reference, data, args.max_dim)
            ok, dim, bd, n_got, n_ref = bottleneck_ok(
                got, ref, args.max_dim, REFERENCE_TOL
            )
            if not ok:
                raise AssertionError(
                    f"{file} H{dim} mismatch: bottleneck={bd:.3e}, "
                    f"tda={n_got}, gudhi={n_ref}"
                )
            print(f"  {file:<24} {'ok':<6} {tda_time:9.3f}s {gudhi_time:9.3f}s")


if __name__ == "__main__":
    main()
