# Figure 2 benchmark

Run the benchmark on a dedicated machine, not in CI.

```sh
cd bench
uv sync
uv run figure2.py
uv run plot_figure2.py
```

`uv sync` builds `tda` from `../crates/python` in this environment. Rebuild
after Rust changes with:

```sh
uv sync --reinstall-package tda
```

Name panels to split a run; `bc` still means `b c`:

```sh
uv run figure2.py a
uv run figure2.py b2 b c
uv run figure2.py b1
```

A full run takes the short panels first: b2, b and c, then a and b1, which
take longest. The runner writes `results/<panel>.csv` and rewrites only the
selected panels. The plotter reads the available CSVs and writes light and dark
SVG and PNG figures: `figure2` for panels a, b and c, and `figure2-b1` and
`figure2-b2`.

Record the machine and software environment with published results.

## Data

Panel a uses the Stanford dragon. `dragon.py` downloads `dragon_recon.tar.gz`
from the Stanford 3D Scanning Repository into `data/`, checks its SHA-256, and
caches the full-resolution scan as `data/dragon.npy`. The scan has 437,645
vertices; 2,100 repeat an earlier vertex exactly, so the cloud keeps 435,545
points in file order. `data/` is not tracked. `figure2.py a` fetches the data
on first use, or run it ahead of time:

```sh
uv run dragon.py
uv run dragon.py 1000 2000 --seed 0
```

The second form also writes `data/dragon_seed0_1000.txt` and
`data/dragon_seed0_2000.txt` for tools that read text. A sample of size n is
the first n points of a seeded permutation, so for one seed every sample is a
prefix of every larger one.

## Configuration

Constants at the top of `figure2.py` define the budget and panel inputs.

| Panel | Sweeps | Fixed inputs |
|---|---|---|
| a | `A_SIZES` | `A_DIM`, `A_MAX_DIM` |
| b | `B_DIMS` | `B_N` |
| b1 | `B1_DIMS`, sizes from `B1_START` to `B1_MAX` | `B1_TIMEOUT`, `MEM_CAP_GB` |
| b2 | `B2_PIPELINES` on `../data/datasets.txt` | `B2_MAX_DIM` |
| c | `C_DIMS` | `C_N`, `C_SEEDS` |

`REPS`, `TIMEOUT`, and `MEM_CAP_GB` bound timing panels. A case that runs
past `TIMEOUT` (12 hours) is recorded as a timeout. Panels b and c set
`max_dim` to the manifold dimension. `A_DIM` is 2 because the dragon is a
surface; it only labels the rows.

`torus(n, dim, seed)`, `sphere(n, dim, seed)`, and `dragon.sample(n, seed)`
are deterministic. Each engine receives the same cloud for a case and
repetition. Panel a uses nested samples. Repetitions use different seeds, so
their median includes sampling and timing variation.

## Measurements

Panel a measures runtime against sample size on nested dragon samples, up to
the whole scan. Panel b measures runtime against dimension: at x = d it samples
S^d and computes up to H_d, so the manifold grows with the axis. Each
measurement runs in a spawned process; import time is excluded. The parent
records `oom`, `timeout`, `error`, or `unavailable` without aborting the sweep.
After an engine fails, larger cases for that engine are skipped. Engines run
in the order `tda_quotient`, `tda`, gudhi, and every row is written as it
finishes, so a run stopped partway keeps complete curves for the engines that
ran.

The memory cap and the `peak_gb` column read the child's physical footprint,
which is what Activity Monitor reports as Memory. On macOS, RSS leaves out
compressed and swapped pages, so a child far over the cap can keep running
while the machine swaps. Other platforms use RSS. The parent samples every
50 ms, so short runs may under-report their peak.

Panel b1 measures how far the quotient reaches, on every core: for every
homology dimension d it samples S^d, computes up to H_d, and grows the sample
by a factor of sqrt(2) from `B1_START` until the first failure under
`B1_TIMEOUT` and `MEM_CAP_GB`. The last size that finished is the point for
that dimension; the failure records whether time or memory stopped it. Each
size is one run with seed 0. A point costs at most a little over two budgets,
and dimensions stopped by memory fail fast, so at the default one hour the
panel takes about half a day.

Panel b2 counts simplices up to H2 with `filtration_size` for each dataset in
`../data/datasets.txt`, without and with the quotient. Neither pipeline peels
or collapses, so the difference is the quotient's alone. It takes seconds.

Panel c measures bottleneck distance between exact and quotient barcodes after
log-transforming finite bars with positive birth. The multiplicative
3-interleaving becomes an additive `log(3)` reference bound. H0 is omitted
because its births are zero. Panel c is not timed, so both barcodes are
computed with `parallel=True` on every core.

## Engines

| Key | Call |
|---|---|
| `gudhi` | `RipsPersistence(num_collapses="auto", n_jobs=1)` |
| `tda` | `persistent_homology(peel=True, collapse="auto", parallel=False)` |
| `tda_quotient` | `persistent_homology(peel=False, quotient=True, collapse="auto", parallel=False)` |

Timing uses Z/2, one thread, and no explicit threshold. gudhi's
`RipsPersistence` reduces with a fork of Ripser, so it also stands in for
Ripser-based engines. giotto-ph is not included: in an earlier run on dragon
samples of 250 to 1,000 points it finished within 15% of gudhi at every size.

gudhi and `tda` run their automatic edge collapse. gudhi's `"auto"` collapses
when `max_dim >= 2`; `tda`'s also skips data where the longest edges do not
collapse, such as spheres and flat tori. `tda` keeps its exact peel enabled.
`tda_quotient` runs without peel: peel needs the whole graph below the
cutoff, so with it the quotient could not stream its pairs past about 26,700
points. On dragon samples of 5,000 and 20,000 points the quotient's barcode is
the same with and without peel, and peel made it 2.5x slower.

The runner sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, and `RAYON_NUM_THREADS` to 1 before importing any
engine, so no library starts a thread pool behind its single-thread flag.
Panels b1 and c are not timed against other engines, so their children remove
those variables and run `tda` with `parallel=True` on every core.

## Truncation

No threshold is passed. Each engine derives an enclosing-radius cutoff from the
data. Passing an explicit threshold would replace that shared behavior with a
benchmark-specific choice.

Torus and sphere clouds both leave roughly 99% of pairs below the enclosing
radius, and the dragon about 69%, so the graphs are near-complete. The exact
engines materialize those pairs before reducing anything, so memory ends their
curves in panel a; gudhi first builds a dense float64 distance matrix. Above
about 26,700 points the quotient generates its pairs in distance bands instead
of storing them, and the whole 435,545-point scan peaks at 1.3 GB.

Panel b costs about `C(B_N, d+1)`. At `B_N = 500` the exact engines reach H2
in under a minute. The quotient reaches H6 in 37 s at 1.9 GB, but H7 passes
the 12 GB cap; at 400 points H7 takes 133 s and 8.6 GB. Sampling S^d well needs points
exponential in d, so the high-`d` points time an honest computation on a cloud
that does not resolve the sphere. Adjust sample sizes, dimensions, timeout, and
memory cap together when changing a panel.
