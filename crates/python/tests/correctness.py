import argparse
import os
from pathlib import Path

import gudhi
import numpy as np
import ripser
import tda
import tda._core as core


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
H2_LAST_DATASET = "dragon_2000.txt"
RIPSER_TOL = 1e-3
SPARSE_TOL = 1e-5


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


def bottleneck_ok(got, want, max_dim, tol):
    for dim in range(max_dim + 1):
        a = np.asarray(got[dim], dtype=np.float64).reshape(-1, 2)
        b = np.asarray(want[dim], dtype=np.float64).reshape(-1, 2)
        bd = 0.0 if len(a) == 0 and len(b) == 0 else gudhi.bottleneck_distance(a, b)
        if bd > tol:
            return False, dim, bd, len(a), len(b)
    return True, None, 0.0, 0, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["dense", "sparse"])
    parser.add_argument("max_dim", nargs="?", type=int, choices=[1, 2], default=default_max_dim())
    args = parser.parse_args()

    print(f"Python correctness: {args.mode} H{args.max_dim}")
    for file in dataset_files(args.max_dim):
        data = load_points(file)
        if args.mode == "dense":
            ref = ripser.ripser(data, maxdim=args.max_dim)["dgms"]
            tol = RIPSER_TOL
        else:
            ref = tda.persistent_homology(data, max_dim=args.max_dim)
            tol = SPARSE_TOL
        got = run_tda(data, args.mode, args.max_dim)
        ok, dim, bd, n_got, n_ref = bottleneck_ok(got, ref, args.max_dim, tol)
        if not ok:
            raise AssertionError(
                f"{file} H{dim} mismatch: bottleneck={bd:.3e}, "
                f"tda={n_got}, ripser={n_ref}"
            )
        print(f"  {file:<24} ok")


if __name__ == "__main__":
    main()
