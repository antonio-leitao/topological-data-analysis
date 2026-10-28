<p align="center">
  <img
    src="https://raw.githubusercontent.com/antonio-leitao/topological-data-analysis/master/assets/logo.svg"
    width="200"
    alt="TDA logo"
  >
</p>

<div align="center">
  <h3>Topological Data Analysis</h3>
  <p>
    <i>
      Persistent homology beyond H0 and H1<br>
      A Python and Rust implementation built for performance
    </i>
  </p>
  <p>
    <img
      alt="Pepy total downloads"
      src="https://img.shields.io/pepy/dt/tda?style=for-the-badge&logo=python&labelColor=white&color=blue"
    >
  </p>
</div>

TDA does one computational job: Vietoris–Rips persistent homology, quickly.
It accepts point clouds, distance matrices, and sparse distance graphs and runs
a packed, Ripser-style[^ripser] reduction in Rust. Use the result directly as a
persistence diagram, or bring a selected class back to the data through
topological coordinates.

Limits: up to 16,777,215 points and homology dimension 7. Internally the engine
selects a packed 128-bit or 256-bit simplex layout. Deep dimensions and large
point counts are most practical with `quotient`, which trades exactness for a
`log(3)`-interleaved approximation at a fraction of the cost.[^quotient]

> [!CAUTION]
> TDA is in an early stage of development. APIs may change between releases.

## Installation

TDA requires Python 3.11 or newer. Pre-built wheels for common macOS, Windows,
and Linux platforms are published on [PyPI](https://pypi.org/project/tda/):

```sh
pip install tda
```

<details>
<summary>Compiling from source</summary>

Building requires
[Rust/Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html)
and [maturin](https://www.maturin.rs/):

```sh
git clone https://github.com/antonio-leitao/topological-data-analysis.git
cd topological-data-analysis

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install maturin numpy

cd crates/python
maturin develop --release
```

</details>

## Quick start

```python
import numpy as np
import tda

points = np.random.rand(200, 3)
barcode = tda.persistent_homology(points, max_dim=2)

# barcode[d] contains the [birth, death] intervals in dimension d.
print(barcode[1])
```

For a guided introduction, see the runnable notebooks:

- [`persistent_homology.ipynb`](examples/persistent_homology.ipynb) covers
  circles, spheres, distance matrices, and quotient preprocessing.
- [`coordinates.ipynb`](examples/coordinates.ipynb) shows circular, toroidal,
  and fitted coordinates.

### Inputs

Set `distance_matrix=True` for a dense precomputed distance matrix:

```python
barcode = tda.persistent_homology(
    distances,
    max_dim=1,
    distance_matrix=True,
)
```

Sparse matrices are treated as distance graphs. Entries are edges and missing
pairs have infinite distance:

```python
import scipy.sparse as sp

graph = sp.coo_matrix(
    ([1.0, 1.0, 1.0, 1.0], ([0, 1, 2, 3], [1, 2, 3, 0])),
    shape=(4, 4),
)
barcode = tda.persistent_homology(graph, max_dim=1)
```

Arrays are read in place. C-contiguous `float32` or `float64` input and the
`int32`/`int64` index and `float32`/`float64` value arrays of CSR, CSC, and COO
matrices reach Rust without a copy; other layouts and sparse formats are copied
once. The GIL is released while persistence runs. Coordinates must be finite
and distances non-negative, otherwise a `ValueError` names the offending entry.

### Parameters

`persistent_homology` and `filtration_size` share these options:

| Parameter         | Default | Description                                                 |
| ----------------- | ------- | ----------------------------------------------------------- |
| `max_dim`         | `1`     | Highest homology dimension to compute; at most 7.           |
| `threshold`       | `None`  | Maximum filtration value. `None` uses the enclosing radius. |
| `distance_matrix` | `False` | Interpret a dense input as a square distance matrix.        |
| `peel`            | `True`  | Apply the exact dominated-vertex peeling step.              |
| `collapse`        | `"auto"` | Exact filtration edge collapse: `True`, `False`, or `"auto"`.[^collapse] |
| `quotient`        | `False` | Use the approximate quotient-cover filtration.[^quotient]   |
| `parallel`        | `True`  | Parallelize preprocessing and assembly when useful.         |

With `collapse="auto"`, collapse is skipped for `max_dim=0`, runs in full from
`max_dim=2` up unless the longest edges show there is nothing to remove, and at
`max_dim=1` stops once it costs more than it saves. The barcode is the same
either way.

The result is a list of length `max_dim + 1`. `barcode[d]` is an array of
`[birth, death]` intervals; `inf` marks an essential feature.

## Scaling persistent homology

The filtration is normally the expensive object. TDA provides three composable
ways to reduce it before persistence:

| Step         | Guarantee             | Use it when...                                                                                         |
| ------------ | --------------------- | ------------------------------------------------------------------------------------------------------ |
| **Peel**     | Exact                 | You want a cheap default reduction. It uses dominated vertices to certify a smaller truncation radius. |
| **Collapse** | Exact                 | Extra preprocessing is worthwhile to remove filtration-dominated edges.[^collapse]                     |
| **Quotient** | `log(3)` interleaving | The exact filtration is too large and a controlled approximation is acceptable.[^quotient]             |

For the strongest exact reduction, combine peel and collapse. Peel is based on
the established dominated-vertex/strong-collapse idea; to our knowledge, this
filtration-peeling implementation is currently specific to TDA. For much larger
problems, quotient produces a near-linear skeleton and can still be combined
with the exact steps. The canonical order is **peel → quotient → collapse**.

Every exact step reads the whole graph below the truncation radius, which for
a point cloud is most of its pairs, so exact persistence needs memory
quadratic in the number of points. The quotient does not. With
`quotient=True` and `peel=False`, a point cloud above about 26,700 points
never stores its pairs: they are generated in distance bands and consumed in
order, and the result is identical to building them. Memory then follows the
skeleton, which is close to linear in the number of points. Peel needs the
whole graph, so with `peel=True` the pairs are stored as before.

### Benchmark

<p align="center">
  <img
    src="https://raw.githubusercontent.com/antonio-leitao/topological-data-analysis/master/bench/results/scaling_sample.png"
    width="49%"
    alt="Time against sample size on the Stanford dragon for GUDHI, TDA exact, and TDA quotient"
  >
  <img
    src="https://raw.githubusercontent.com/antonio-leitao/topological-data-analysis/master/bench/results/scaling_dim.png"
    width="49%"
    alt="Time against homology dimension for GUDHI, TDA exact, and TDA quotient"
  >
</p>

Left: time to compute persistence up to H_2 on samples of the Stanford dragon,
up to the full 435,545-point scan. Right: time to compute persistence up to H_d
on 500 points sampled from the d-sphere. Both run on one thread. The setup and
raw results are in [`bench/`](bench/README.md).

One run of the [`summary` example](crates/core/examples/summary.rs) with
`max_dim=2` and `parallel=false`:

```sh
cargo run --release -p tda_core --example summary
```

| Dataset     |  Vanilla |     Peel | Collapse | Peel+Collapse | Quotient | Quotient+Peel |    Q+P+C |
| ----------- | -------: | -------: | -------: | ------------: | -------: | ------------: | -------: |
| celegans    | 135.22ms | 106.47ms |  37.54ms |       35.61ms |   3.52ms |        3.90ms |   3.79ms |
| vicsek      | 134.28ms | 132.84ms |  53.73ms |       55.17ms |   2.00ms |        2.10ms |   2.23ms |
| klein_400   | 230.00ms |  92.60ms | 111.66ms |       59.80ms |   3.64ms |        4.55ms |   4.19ms |
| klein_900   |    3.30s |    1.24s |    1.23s |      618.48ms |  18.44ms |       21.13ms |  21.12ms |
| dragon_1000 |    2.98s |    1.65s |    1.21s |      654.81ms |  19.06ms |       21.56ms |  22.03ms |
| dragon_2000 |   22.43s |   11.09s |   14.10s |         6.54s | 149.59ms |      201.23ms | 213.05ms |
| hiv1        |   12.83s |    6.17s |    3.03s |         1.80s | 126.70ms |      109.23ms | 105.56ms |

Timings depend on hardware; filtration size more directly shows the work that
reaches reduction:

| Dataset     |   Vanilla |      Peel | Collapse | Peel+Collapse | Quotient | Quotient+Peel | Q+P+C |
| ----------- | --------: | --------: | -------: | ------------: | -------: | ------------: | ----: |
| celegans    |   4102122 |   3562466 |     2031 |          2031 |     8944 |          8910 |   757 |
| vicsek      |   4089966 |   4052107 |     6109 |          6054 |     2196 |          2196 |   924 |
| klein_400   |   5968412 |   2042830 |    16243 |         16243 |     5219 |          5191 |  1828 |
| klein_900   |  63867012 |  21890852 |   156044 |        156044 |    14607 |         14554 |  5077 |
| dragon_1000 |  56423455 |  25537491 |   108174 |        108174 |    14248 |         14176 |  5287 |
| dragon_2000 | 546815948 | 182326181 |   918585 |        918585 |    30418 |         30366 | 11570 |
| hiv1        | 157117515 |  70122454 |   330419 |        330376 |   117473 |        117427 |  3529 |

On `dragon_2000`, quotient reduces 546.8 million simplices to about 30.4
thousand and cuts this run from 22.43 seconds to 149.59 milliseconds. That is
the intended quotient use case: a controlled approximation when the full
filtration is the bottleneck.

## Using persistence

### Persistence diagrams

For many applications, the barcode returned by `persistent_homology` is the
final summary: long-lived intervals identify features that persist across
scales.

### Coordinates

Coordinates bring selected persistent cohomology classes back to the original
observations. One H¹ class assigns every point an angle on S¹:[^circcoords]

```python
result = tda.coords.circular(points)
angles = result.coordinates  # (n, 1), values in [0, 2π)
```

Asking for several classes gives one column each. Classes that are alive at a
common scale are jointly decorrelated by lattice reduction into a single map to
a torus; `result.groups` reports which columns were decorrelated together:[^toroidal]

```python
two_circles = np.concatenate([points, points + [4.0, 0.0]])

result = tda.coords.circular(two_circles, n_classes=2)
result.coordinates  # (n, 2)
result.groups       # [[0, 1]] — both columns in one toroidal block
```

Classes that are not co-alive stay independent and come back as singleton
groups (`[[0], [1]]`), which are plain circular coordinates.

These constructions are adapted from
[DREiMAC](https://dreimac.scikit-tda.org/en/latest/), the reference
implementation.[^dreimac] See the
[`coordinates` notebook](examples/coordinates.ipynb) for the geometric
intuition, class selection, quotient coordinates, and fitted maps for new
data.

The integer lift of the selected cocycle is checked for closedness. DREiMAC can
repair a failed lift with mixed-integer optimization; TDA instead returns an
error and leaves the choice of another class, scale, or prime to the
user.[^dreimac-lift]

## Additional APIs

Use `filtration_size` to inspect the truncated filtration under the same
preprocessing options without computing persistence:

```python
size = tda.filtration_size(points, max_dim=2, quotient=True)
```

A fitted `circular_map` transforms new point-cloud observations without
recomputing persistence. See the
[coordinates notebook](examples/coordinates.ipynb) for examples.

### Rust

The Rust crate is published as
[`tda_core`](https://crates.io/crates/tda_core):

```toml
[dependencies]
tda_core = "0.4"
```

```rust
use tda_core::{persistent_homology, Error};

fn main() -> Result<(), Error> {
    let points = vec![
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
    ];

    let barcode = persistent_homology(&points, 4, 1)?;
    println!("{:?}", barcode.intervals);
    Ok(())
}
```

Distance matrices, sparse COO and CSR/CSC distance graphs, thresholds, and
preprocessing options are available through `Config` and the corresponding
`_with`, `_sparse`, and `_csr` functions. Every entry point reads `f32` or
`f64` values and `u32`, `u64`, `i32`, or `i64` indices in place. Invalid input
returns `tda_core::Error`.

## References

[^ripser]:
    U. Bauer. _Ripser: efficient computation of Vietoris–Rips
    persistence barcodes_. Journal of Applied and Computational Topology, 2021.
    [arXiv:1908.02518](https://arxiv.org/abs/1908.02518).

[^quotient]:
    A. Leitão. _It's All About Covers: Persistent Homology of Cover
    Refinements_. [arXiv:2602.22784](https://arxiv.org/abs/2602.22784).

[^collapse]:
    J.-D. Boissonnat and S. Pritam. _Edge Collapse and Persistence of
    Flag Complexes_. SoCG 2020.
    [DOI:10.4230/LIPIcs.SoCG.2020.19](https://doi.org/10.4230/LIPIcs.SoCG.2020.19).
    M. Glisse and S. Pritam. _Swap, Shift and Trim to Edge Collapse a
    Filtration_. SoCG 2022.
    [DOI:10.4230/LIPIcs.SoCG.2022.44](https://doi.org/10.4230/LIPIcs.SoCG.2022.44).

[^circcoords]:
    V. de Silva, D. Morozov, and M. Vejdemo-Johansson. _Persistent
    Cohomology and Circular Coordinates_. Discrete & Computational
    Geometry, 2011.
    [DOI:10.1007/s00454-011-9344-x](https://doi.org/10.1007/s00454-011-9344-x).

[^toroidal]:
    L. Scoccola, H. Gakhar, J. Bush, N. Schonsheck, T. Rask, L. Zhou,
    and J. A. Perea. _Toroidal Coordinates: Decorrelating Circular
    Coordinates with Lattice Reduction_. SoCG 2023.
    [DOI:10.4230/LIPIcs.SoCG.2023.57](https://doi.org/10.4230/LIPIcs.SoCG.2023.57).

[^dreimac]:
    J. A. Perea, L. Scoccola, and C. J. Tralie. _DREiMac: Dimensionality
    Reduction with Eilenberg–MacLane Coordinates_. Journal of Open Source
    Software 8(91), 5791, 2023.
    [DOI:10.21105/joss.05791](https://doi.org/10.21105/joss.05791).
    [Source](https://github.com/scikit-tda/DREiMac) (Apache-2.0).

[^dreimac-lift]:
    DREiMAC documentation. _Choosing the parameters prime and
    check_cocycle_condition_.
    [Documentation](https://dreimac.scikit-tda.org/en/latest/notebooks/parameters_prime_and_check_cocycle_condition.html).

## License

TDA is distributed under the [MIT License](LICENSE.md).
