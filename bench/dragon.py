"""Fetch the Stanford dragon and cut deterministic nested samples.

The full-resolution scan, dragon_vrip.ply from the Stanford 3D Scanning
Repository, has 437,645 vertices. 2,100 of them repeat an earlier vertex
exactly, so the cloud keeps the first copy of each: 435,545 points in file
order. data/dragon_1000.txt and data/dragon_2000.txt at the repository root
are subsamples of the same scan.

A sample is a prefix of a seeded permutation, so for one seed every smaller
sample is contained in every larger one.

    uv run dragon.py                         # download, verify, cache
    uv run dragon.py 1000 2000 --seed 0      # also write text slices
"""

import argparse
import hashlib
import io
import subprocess
import tarfile
from pathlib import Path

import numpy as np

URL = "https://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz"
SHA256 = "74ac1d90989c9b1732edee82d57e9ce71452144cf4355f108d8c9c616d28d02f"
MEMBER = "dragon_recon/dragon_vrip.ply"
POINTS = 435_545

DATA = Path(__file__).parent / "data"
ARCHIVE = DATA / "dragon_recon.tar.gz"
CACHE = DATA / "dragon.npy"


def download():
    """Download the archive once and refuse it if the checksum differs.

    curl uses the system trust store; python.org builds of Python on macOS
    ship without one, so urllib fails there.
    """
    if not ARCHIVE.exists():
        DATA.mkdir(exist_ok=True)
        partial = ARCHIVE.with_suffix(".part")
        print(f"downloading {URL}", flush=True)
        subprocess.run(["curl", "-fsSL", "-o", str(partial), URL], check=True)
        partial.rename(ARCHIVE)
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"{ARCHIVE} has sha256 {digest}, expected {SHA256}")


def parse(handle):
    """Vertex block of an ASCII PLY whose vertices are x, y, z."""
    text = io.TextIOWrapper(handle, encoding="ascii")
    count = None
    while (line := text.readline().strip()) != "end_header":
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
    return np.loadtxt(text, max_rows=count, dtype=np.float32)


def load():
    """The deduplicated cloud, float32, in file order. Builds the cache once."""
    if not CACHE.exists():
        download()
        with tarfile.open(ARCHIVE) as archive:
            vertices = parse(archive.extractfile(MEMBER))
        _, first = np.unique(vertices, axis=0, return_index=True)
        points = vertices[np.sort(first)]
        if len(points) != POINTS:
            raise RuntimeError(f"expected {POINTS} distinct vertices, got {len(points)}")
        np.save(CACHE, points)
    return np.load(CACHE)


def sample(n, seed, points=None):
    """The first n points of the seed's permutation of the cloud."""
    points = load() if points is None else points
    if not 0 < n <= len(points):
        raise ValueError(f"n must be in 1..{len(points)}, got {n}")
    order = np.random.default_rng(seed).permutation(len(points))
    return np.ascontiguousarray(points[order[:n]])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sizes", nargs="*", type=int, help="write these slices as text")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    points = load()
    print(f"{CACHE}: {len(points)} points")
    for n in args.sizes:
        path = DATA / f"dragon_seed{args.seed}_{n}.txt"
        np.savetxt(path, sample(n, args.seed, points), fmt="%.7g")
        print(f"{path}: {n} points")


if __name__ == "__main__":
    main()
