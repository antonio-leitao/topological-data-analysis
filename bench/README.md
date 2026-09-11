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

Pass panel letters to split a run:

```sh
uv run figure2.py a
uv run figure2.py bc
```

The runner writes `results/a.csv`, `results/b.csv`, and `results/c.csv`. It
rewrites only the selected panels. The plotter reads available CSVs and writes
light and dark SVG and PNG figures.

Record the machine and software environment with published results.

## Configuration

Constants at the top of `figure2.py` define the budget and panel inputs.

| Panel | Sweeps | Fixed inputs |
|---|---|---|
| a | `A_SIZES` | `A_DIM`, `A_MAX_DIM` |
| b | `B_DIMS` | `B_N` |
| c | `C_DIMS` | `C_N`, `C_SEEDS` |

`REPS`, `TIMEOUT`, and `MEM_CAP_GB` bound timing panels. Panels b and c set
`max_dim` to the manifold dimension.

`torus(n, dim, seed)` and `sphere(n, dim, seed)` are deterministic. Each engine
receives the same cloud for
a case and repetition. Panel a uses nested samples because a smaller cloud is a
prefix of a larger cloud generated with the same seed. Repetitions use different
seeds, so their median includes sampling and timing variation.

## Measurements

Panel a measures runtime against sample size on a fixed torus. Panel b measures
runtime against dimension: at x = d it samples S^d and computes up to H_d, so
the manifold grows with the axis. Each measurement runs in a spawned process; import
time is excluded. The parent records `oom`, `timeout`, `error`, or `unavailable`
without aborting the sweep. After an engine fails, larger cases for that engine
are skipped.

Panel c measures bottleneck distance between exact and quotient barcodes after
log-transforming finite bars with positive birth. The multiplicative
3-interleaving becomes an additive `log(3)` reference bound. H0 is omitted
because its births are zero.

## Engines

| Key | Call |
|---|---|
| `giotto` | `gph.ripser_parallel(collapse_edges=False, n_threads=1)` |
| `gudhi` | `RipsPersistence(num_collapses=0, n_jobs=1)` |
| `tda` | `persistent_homology(peel=True, collapse=False, parallel=False)` |
| `tda_quotient` | the same call with `quotient=True` |

Timing uses Z/2, one thread, no explicit threshold, and no edge collapse.
Collapse is disabled across engines so the comparison does not depend on a
shared preprocessing choice. `tda` keeps its exact peel enabled.

`giotto-ph` is used directly. The benchmark environment pins Python below 3.13
because giotto-ph 0.2.4 provides no compatible distribution there.

## Truncation

No threshold is passed. Each engine derives an enclosing-radius cutoff from the
data. Passing an explicit threshold would replace that shared behavior with a
benchmark-specific choice.

Torus and sphere clouds both leave roughly 99% of pairs below the enclosing
radius, so the graphs are near-complete. Panel b therefore costs about
`C(B_N, d+1)` and needs a small `B_N` to reach high `d`: at `B_N = 60` every
engine finishes d=7, while 400 stops at d=2. Sampling S^d well needs points
exponential in d, so the high-`d` points time an honest computation on a cloud
that does not resolve the sphere. Adjust sample sizes, dimensions, timeout, and
memory cap together when changing a panel.
