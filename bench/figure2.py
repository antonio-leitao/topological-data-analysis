"""Run Figure 2 timing and quotient-error panels."""

import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import psutil

RESULTS = Path(__file__).parent / "results"

REPS = 3
TIMEOUT = 3600.0
MEM_CAP_GB = 12.0
COEFF = 2

# Panel a: time vs sample size. Manifold and homology dimension held fixed.
A_DIM = 2
A_MAX_DIM = 2
A_SIZES = (250, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000)

# Panel b: time vs dimension. At x = d we sample S^d and compute up to H_d, so
# the manifold grows with the axis. Only the sample size is held fixed.
B_N = 60
B_DIMS = (1, 2, 3, 4, 5, 6, 7)

# Panel c: quotient error on T^d at max_dim = d, one row per homology dimension.
C_DIMS = (1, 2, 3, 4, 5, 6, 7, 8)
C_N = 300
C_SEEDS = 10

ENGINES = ("giotto", "gudhi", "tda", "tda_quotient")


def torus(n, dim, seed):
    """Uniform on the flat T^dim, as a product of unit circles in R^(2 dim).
    Betti numbers are comb(dim, k)."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, (n, dim))
    return np.concatenate([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)


def sphere(n, dim, seed):
    """Uniform on S^dim in R^(dim+1): normalised Gaussians are rotation
    invariant, so the direction is uniform. Betti numbers are 1, 0, ..., 0, 1."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim + 1))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


MANIFOLDS = {"torus": torus, "sphere": sphere}


def build(engine):
    """Import the engine's library and return its (points, max_dim) callable.

    Called in the child before the clock starts, so import cost stays out of
    the measurement. No truncation radius is passed and edge collapse is off
    everywhere; peel stays on for tda. See README.
    """
    if engine == "giotto":
        from gph import ripser_parallel

        return lambda pts, md: ripser_parallel(
            pts, maxdim=md, coeff=COEFF, collapse_edges=False, n_threads=1
        )["dgms"]

    if engine == "gudhi":
        from gudhi.sklearn.rips_persistence import RipsPersistence

        return lambda pts, md: RipsPersistence(
            homology_dimensions=list(range(md + 1)),
            homology_coeff_field=COEFF,
            num_collapses=0,
            n_jobs=1,
        ).fit_transform([pts])[0]

    import tda

    quotient = engine == "tda_quotient"
    return lambda pts, md: tda.persistent_homology(
        pts, max_dim=md, peel=True, collapse=False, quotient=quotient, parallel=False
    )


def guarded(target, args):
    """Run target(*args, sender) in a spawned child under TIMEOUT and
    MEM_CAP_GB. Returns (status, payload), status in ok/oom/timeout/error/
    unavailable."""
    receiver, sender = mp.Pipe(False)
    proc = mp.Process(target=target, args=(*args, sender))
    proc.start()
    watched = psutil.Process(proc.pid)
    deadline = time.monotonic() + TIMEOUT
    while proc.is_alive():
        try:
            if watched.memory_info().rss > MEM_CAP_GB * 1e9:
                proc.kill()
                return "oom", None
        except psutil.Error:
            break
        if time.monotonic() > deadline:
            proc.kill()
            return "timeout", None
        time.sleep(0.05)
    return receiver.recv() if receiver.poll() else ("error", None)


def _time_child(engine, manifold, n, dim, max_dim, seed, out):
    try:
        run = build(engine)
    except ImportError:
        out.send(("unavailable", None))
        return
    points = MANIFOLDS[manifold](n, dim, seed)
    start = time.perf_counter()
    run(points, max_dim)
    out.send(("ok", time.perf_counter() - start))


def _error_child(dim, n, seed, out):
    from gudhi import bottleneck_distance

    exact, approx = build("tda"), build("tda_quotient")
    points = torus(n, dim, seed)
    a, b = exact(points, dim), approx(points, dim)
    out.send(("ok", [(k, bottleneck_distance(logs(a[k]), logs(b[k])))
                     for k in range(1, dim + 1)]))


def logs(bars):
    """Finite bars with positive birth, log-scaled: the quotient's guarantee is
    a multiplicative 3-interleaving, which is additive log 3 here."""
    d = np.asarray(bars, dtype=np.float64).reshape(-1, 2)
    return np.log(d[np.isfinite(d).all(axis=1) & (d[:, 0] > 0.0)])


def timings(path, panel, manifold, cases):
    """Time every engine across `cases`, cheapest first. An engine's first
    failure ends its sweep, since every later case is larger."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            ["panel", "engine", "manifold", "n", "dim", "max_dim", "rep",
             "status", "seconds"],
        )
        writer.writeheader()
        for engine in ENGINES:
            for n, dim, max_dim in cases:
                ok = True
                for rep in range(REPS):
                    status, seconds = guarded(
                        _time_child, (engine, manifold, n, dim, max_dim, rep)
                    )
                    writer.writerow(dict(
                        panel=panel, engine=engine, manifold=manifold, n=n, dim=dim,
                        max_dim=max_dim, rep=rep, status=status, seconds=seconds,
                    ))
                    handle.flush()
                    print(f"{panel} {engine:<13} {manifold[0].upper()}^{dim} n={n}"
                          f" max_dim={max_dim} rep{rep} -> {status} {seconds}", flush=True)
                    if status != "ok":
                        ok = False
                        break
                if not ok:
                    break


def errors(path):
    """Quotient error on T^d at max_dim = d. Guarded like the timing panels, so
    a dimension that is out of reach is recorded and skipped instead of taking
    the run down."""
    bound = float(np.log(3.0))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            ["manifold_dim", "n", "seed", "homology_dim", "status", "bottleneck", "bound"],
        )
        writer.writeheader()
        for dim in C_DIMS:
            for seed in range(C_SEEDS):
                status, pairs = guarded(_error_child, (dim, C_N, seed))
                for k, distance in pairs if status == "ok" else [("", "")]:
                    writer.writerow(dict(
                        manifold_dim=dim, n=C_N, seed=seed, homology_dim=k,
                        status=status, bottleneck=distance, bound=bound,
                    ))
                handle.flush()
                print(f"c T^{dim} seed{seed} -> {status}", flush=True)
                if status != "ok":
                    break  # this torus dimension is out of reach; try the next


def main():
    panels = sys.argv[1] if len(sys.argv) > 1 else "abc"
    RESULTS.mkdir(exist_ok=True)
    if "a" in panels:
        timings(RESULTS / "a.csv", "a", "torus",
                [(n, A_DIM, A_MAX_DIM) for n in A_SIZES])
    if "b" in panels:
        timings(RESULTS / "b.csv", "b", "sphere",
                [(B_N, d, d) for d in B_DIMS])
    if "c" in panels:
        errors(RESULTS / "c.csv")


if __name__ == "__main__":
    main()
