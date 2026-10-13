"""Run the Figure 2 panels: timing (a, b), the quotient's reach (b1), its
filtration size (b2) and its error (c)."""

import csv
import ctypes
import ctypes.util
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# One thread everywhere, including any OpenMP, BLAS or rayon pool an engine
# might start. Set before numpy is imported; spawned children inherit it.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS")
for _var in THREAD_VARS:
    os.environ[_var] = "1"

import numpy as np
import psutil

import dragon

RESULTS = Path(__file__).parent / "results"

REPS = 3
# A case either finishes inside this or is recorded as a timeout.
TIMEOUT = 12 * 3600.0
MEM_CAP_GB = 12.0
COEFF = 2

# Panel a: time vs sample size on nested samples of the Stanford dragon, a
# surface. The largest size is the whole deduplicated scan.
A_DIM = 2
A_MAX_DIM = 2
A_SIZES = (1_000, 2_000, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 100_000,
           200_000, dragon.POINTS)

# Panel b: time vs dimension. At x = d we sample S^d and compute up to H_d, so
# the manifold grows with the axis. Only the sample size is held fixed.
B_N = 500
B_DIMS = (1, 2, 3, 4, 5, 6, 7)

# Panel b1: the largest S^d sample the quotient finishes within a budget, on
# every core, computing up to H_d. Sizes grow by sqrt(2) from B1_START until the
# first failure; the failure says whether time or memory stopped it.
B1_DIMS = (1, 2, 3, 4, 5, 6, 7)
B1_TIMEOUT = 3600.0
B1_START = 64
B1_MAX = 4_000_000

# Panel b2: filtration size, simplices up to H2, without and with the quotient
# on the datasets listed in ../data/datasets.txt. Neither pipeline peels or
# collapses, so the difference is the quotient's alone. Counts only.
B2_DATA = Path(__file__).parent.parent / "data"
B2_MAX_DIM = 2
B2_PIPELINES = {
    "vanilla": dict(peel=False, collapse=False, quotient=False),
    "quotient": dict(peel=False, collapse=False, quotient=True),
}

# Panel c: quotient error on T^d at max_dim = d, one row per homology dimension.
C_DIMS = (1, 2, 3)
C_N = 300
C_SEEDS = 10

# In the order a full run takes them: short panels first, so their results
# exist long before the multi-day ones finish.
PANELS = ("b2", "b", "c", "a", "b1")

# Run order only; the plotter keeps its own. Ours go first so a run stopped
# partway still has their curves.
ENGINES = ("tda_quotient", "tda", "gudhi")


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


MANIFOLDS = {
    "torus": torus,
    "sphere": sphere,
    "dragon": lambda n, dim, seed: dragon.sample(n, seed),
}


def build(engine, parallel=False):
    """Import the engine's library and return its (points, max_dim) callable.

    Called in the child before the clock starts, so import cost stays out of
    the measurement. No truncation radius is passed. Engines run on one thread;
    `parallel` gives tda every core. Edge collapse is automatic for both
    engines. tda peels except with the quotient. See README.
    """
    if engine == "gudhi":
        from gudhi.sklearn.rips_persistence import RipsPersistence

        return lambda pts, md: RipsPersistence(
            homology_dimensions=list(range(md + 1)),
            homology_coeff_field=COEFF,
            num_collapses="auto",
            n_jobs=1,
        ).fit_transform([pts])[0]

    import tda

    # Peel needs the whole graph, so with it the quotient cannot stream its
    # pairs; on the dragon it also leaves the quotient's barcode unchanged.
    quotient = engine == "tda_quotient"
    return lambda pts, md: tda.persistent_homology(
        pts, max_dim=md, peel=not quotient, collapse="auto", quotient=quotient,
        parallel=parallel,
    )


def _footprint():
    """Return a pid -> bytes reader for the memory cap.

    On macOS RSS leaves out compressed and swapped pages, so a process far over
    the cap can keep running while the machine swaps. The physical footprint
    counts them; it is what Activity Monitor reports as Memory.
    """
    if sys.platform != "darwin":
        return lambda pid: psutil.Process(pid).memory_info().rss
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    info = (ctypes.c_uint64 * 32)()  # rusage_info_v0 is 12 words

    def footprint(pid):
        if libc.proc_pid_rusage(pid, 0, info) != 0:  # RUSAGE_INFO_V0
            raise psutil.NoSuchProcess(pid)
        return info[9]  # ri_phys_footprint, after the uuid and 7 counters

    return footprint


def guarded(target, args, timeout=TIMEOUT):
    """Run target(*args, sender) in a spawned child under `timeout` seconds and
    MEM_CAP_GB. Returns (status, payload, peak bytes), status in ok/oom/
    timeout/error/unavailable."""
    receiver, sender = mp.Pipe(False)
    proc = mp.Process(target=target, args=(*args, sender))
    proc.start()
    footprint = _footprint()
    deadline = time.monotonic() + timeout
    peak = 0
    while proc.is_alive():
        try:
            peak = max(peak, footprint(proc.pid))
        except psutil.Error:
            break
        if peak > MEM_CAP_GB * 1e9:
            proc.kill()
            return "oom", None, peak
        if time.monotonic() > deadline:
            proc.kill()
            return "timeout", None, peak
        time.sleep(0.05)
    status, payload = receiver.recv() if receiver.poll() else ("error", None)
    return status, payload, peak


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


def _all_cores():
    """Lift the one-thread limit in this child. Libraries read it when their
    pools start, which is after `build` imports them."""
    for var in THREAD_VARS:
        os.environ.pop(var, None)


def _frontier_child(dim, n, out):
    _all_cores()
    run = build("tda_quotient", parallel=True)
    points = sphere(n, dim, 0)
    start = time.perf_counter()
    run(points, dim)
    out.send(("ok", time.perf_counter() - start))


def _size_child(name, pipeline, out):
    import tda

    points = np.loadtxt(B2_DATA / name, dtype=np.float32)
    start = time.perf_counter()
    count = tda.filtration_size(points, max_dim=B2_MAX_DIM, parallel=False,
                                **B2_PIPELINES[pipeline])
    out.send(("ok", (len(points), count, time.perf_counter() - start)))


def _error_child(dim, n, seed, out):
    # Nothing here is timed, so both barcodes may use every core.
    _all_cores()
    from gudhi import bottleneck_distance

    exact, approx = build("tda", parallel=True), build("tda_quotient", parallel=True)
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
             "status", "seconds", "peak_gb"],
        )
        writer.writeheader()
        for engine in ENGINES:
            for n, dim, max_dim in cases:
                ok = True
                for rep in range(REPS):
                    status, seconds, peak = guarded(
                        _time_child, (engine, manifold, n, dim, max_dim, rep)
                    )
                    writer.writerow(dict(
                        panel=panel, engine=engine, manifold=manifold, n=n, dim=dim,
                        max_dim=max_dim, rep=rep, status=status, seconds=seconds,
                        peak_gb=round(peak / 1e9, 3),
                    ))
                    handle.flush()
                    print(f"{panel} {engine:<13} {manifold} dim={dim} n={n}"
                          f" max_dim={max_dim} rep{rep} -> {status} {seconds}"
                          f" peak={peak / 1e9:.2f}GB", flush=True)
                    if status != "ok":
                        ok = False
                        break
                if not ok:
                    break


def frontier(path):
    """For each dimension, grow the quotient's sample until the first failure
    under B1_TIMEOUT and MEM_CAP_GB. The last success is the frontier; the
    failure's status says what stopped it."""
    sizes = []
    k = 0
    while (n := round(B1_START * 2 ** (k / 2))) <= B1_MAX:
        sizes.append(n)
        k += 1
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            ["manifold", "dim", "n", "status", "seconds", "peak_gb", "timeout_s",
             "mem_cap_gb"],
        )
        writer.writeheader()
        for dim in B1_DIMS:
            for n in sizes:
                if n < dim + 2:
                    continue
                status, seconds, peak = guarded(_frontier_child, (dim, n), timeout=B1_TIMEOUT)
                writer.writerow(dict(
                    manifold="sphere", dim=dim, n=n, status=status, seconds=seconds,
                    peak_gb=round(peak / 1e9, 3), timeout_s=B1_TIMEOUT,
                    mem_cap_gb=MEM_CAP_GB,
                ))
                handle.flush()
                print(f"b1 tda_quotient S^{dim} n={n} -> {status} {seconds}"
                      f" peak={peak / 1e9:.2f}GB", flush=True)
                if status != "ok":
                    break


def filtration_sizes(path):
    """Simplices up to H2 under each pipeline, per dataset."""
    names = [line.strip() for line in (B2_DATA / "datasets.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            ["dataset", "n", "max_dim", "pipeline", "status", "simplices", "seconds"],
        )
        writer.writeheader()
        for name in names:
            for pipeline in B2_PIPELINES:
                status, payload, _ = guarded(_size_child, (name, pipeline))
                n, count, seconds = payload if status == "ok" else ("", "", "")
                writer.writerow(dict(
                    dataset=Path(name).stem, n=n, max_dim=B2_MAX_DIM, pipeline=pipeline,
                    status=status, simplices=count, seconds=seconds,
                ))
                handle.flush()
                print(f"b2 {name} {pipeline} -> {status} {count}", flush=True)


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
                status, pairs, _ = guarded(_error_child, (dim, C_N, seed))
                for k, distance in pairs if status == "ok" else [("", "")]:
                    writer.writerow(dict(
                        manifold_dim=dim, n=C_N, seed=seed, homology_dim=k,
                        status=status, bottleneck=distance, bound=bound,
                    ))
                handle.flush()
                print(f"c T^{dim} seed{seed} -> {status}", flush=True)
                if status != "ok":
                    break  # this torus dimension is out of reach; try the next


def selected(args):
    """Panels named on the command line, or all of them. Runs of the single
    letters a, b and c may be joined, as in `bc`."""
    if not args:
        return set(PANELS)
    chosen = set()
    for arg in args:
        if arg in PANELS:
            chosen.add(arg)
        elif set(arg) <= {"a", "b", "c"}:
            chosen.update(arg)
        else:
            raise SystemExit(f"unknown panel {arg!r}; choose from {', '.join(PANELS)}")
    return chosen


def main():
    chosen = selected(sys.argv[1:])
    RESULTS.mkdir(exist_ok=True)
    for panel in PANELS:
        if panel not in chosen:
            continue
        if panel == "a":
            dragon.load()  # download and cache before the first child needs it
            timings(RESULTS / "a.csv", "a", "dragon",
                    [(n, A_DIM, A_MAX_DIM) for n in A_SIZES])
        elif panel == "b":
            timings(RESULTS / "b.csv", "b", "sphere",
                    [(B_N, d, d) for d in B_DIMS])
        elif panel == "b1":
            frontier(RESULTS / "b1.csv")
        elif panel == "b2":
            filtration_sizes(RESULTS / "b2.csv")
        elif panel == "c":
            errors(RESULTS / "c.csv")


if __name__ == "__main__":
    main()
