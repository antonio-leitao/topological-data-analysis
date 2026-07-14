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

TDA computes Vietoris–Rips persistent homology from point clouds, precomputed
distance matrices, or sparse distance graphs. The engine is an enhanced
Ripser-style[^ripser] reduction written in Rust, available through both Python
and Rust APIs.

Limits: up to 16,777,215 points and homology dimension 7. Internally the
engine picks between two packed-simplex layouts — a 128-bit one (up to 65,535
points and H4, the fastest path) and a 256-bit one (everything else) — with
identical output on shared inputs. Deep dimensions and large point counts are
most practical together with `quotient`, which trades exactness for a
`log(3)`-interleaved approximation at a fraction of the cost.

> [!CAUTION]
> TDA is in an early stage of development. APIs may change between releases.

## Installation

TDA requires Python 3.11 or newer. Pre-built wheels for common macOS, Windows,
and Linux platforms are published on [PyPI](https://pypi.org/project/tda/):

```sh
python -m pip install tda
```

### Compiling from source

Building from source requires
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

## Python usage

```python
import numpy as np
import tda

# Point cloud: an (n, d) array (float32 or float64).
points = np.random.rand(200, 3)
barcode = tda.persistent_homology(points, max_dim=2)

# barcode[d] is a (k_d, 2) array of [birth, death] intervals.
print(barcode[0])  # H0
print(barcode[1])  # H1
print(barcode[2])  # H2
```

Precomputed distance matrices are also supported:

```python
distances = np.asarray(my_distance_matrix)
barcode = tda.persistent_homology(
    distances,
    max_dim=1,
    distance_matrix=True,
)
```

Sparse distance matrices (any `scipy.sparse` format — in fact, anything with a
`.tocoo()` method; scipy itself never becomes a dependency) are treated as
distance graphs: entries are edges, missing pairs are never connected
(infinite distance), duplicate entries resolve to the smallest value, and
disconnected inputs are handled soundly:

```python
import scipy.sparse as sp

graph = sp.coo_matrix(
    ([1.0, 1.0, 1.0, 1.0], ([0, 1, 2, 3], [1, 2, 3, 0])),
    shape=(4, 4),
)
barcode = tda.persistent_homology(graph, max_dim=1)
```

Pass `representatives=True` to also get, per interval, its birth simplex, its
death simplex, and a representative cocycle of the class (persistent
cohomology, as in Ripser's `do_cocycles`):

```python
result = tda.persistent_homology(points, max_dim=1, representatives=True)
interval = result.diagrams[1][0]         # [birth, death]
rep = result.representatives[1][0]       # aligned with diagrams[1][0]
rep.birth                                # e.g. [17, 4]  — the birth edge
rep.death                                # e.g. [23, 17, 4], None if essential
rep.cocycle                              # list of simplices (vertex-id lists)
```

In dimension 0 the cocycle is empty; the birth vertex and the death (merge)
edge carry the information. Under `quotient`, representatives live on the
quotient complex (vertex ids are cluster representatives); under `collapse`,
they are cocycles of the exactly-equivalent collapsed filtration.

Use `filtration_size` to inspect the size of the truncated filtration built
with the same options:

```python
size = tda.filtration_size(points, max_dim=2)
print(size)
```

### Parameters

Both `persistent_homology` and `filtration_size` accept the following
parameters:

| Parameter         | Type                   | Default | Description                                                                                                                                                                                                                                       |
| ----------------- | ---------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data`            | `np.ndarray` or sparse | —       | Two-dimensional `float32` or `float64` array (`float64` is cast to `float32`) — shape `(n, d)` for a point cloud, `(n, n)` for a distance matrix — or a square `scipy.sparse` distance matrix. Inputs are copied into row-major internal storage. |
| `max_dim`         | `int`                  | `1`     | Highest homology dimension to compute. Capped at 7.                                                                                                                                                                                               |
| `threshold`       | `float \| None`        | `None`  | Maximum filtration value. `None` uses the enclosing radius; an explicit value is capped by that radius.                                                                                                                                           |
| `distance_matrix` | `bool`                 | `False` | Interpret `data` as a square distance matrix. Symmetry, zero diagonal, and non-negativity are assumed rather than validated. Ignored for sparse input, which is always a distance graph.                                                          |
| `quotient`        | `bool`                 | `False` | Use the smaller quotient-cover filtration.[^quotient] This is an approximation with a `log(3)` interleaving guarantee, not the exact Vietoris–Rips barcode.                                                                                       |
| `peel`            | `bool`                 | `True`  | Tighten the truncation radius with an exact strong-collapse peeling: identical barcode, strictly less work. Disable to compute over the plain enclosing radius.                                                                                   |
| `collapse`        | `bool`                 | `False` | Remove filtration-dominated edges (exact edge collapse):[^collapse] identical barcode, smaller filtration, but expensive preprocessing.                                                                                                           |
| `parallel`        | `bool`                 | `True`  | Enable parallel preprocessing, sorting, and candidate assembly for sufficiently large inputs.                                                                                                                                                     |
| `representatives` | `bool`                 | `False` | (`persistent_homology` only.) Return a `Barcode` object with `diagrams` plus aligned cocycle `representatives`. Recording is compiled separately, so the default path pays nothing for it.                                                        |

`persistent_homology` returns a list of length `max_dim + 1`. Entry
`barcode[d]` is a NumPy array of shape `(k_d, 2)` whose rows are
`[birth, death]` intervals. A death value of `inf` marks an essential feature.

## Rust usage

The Rust crate is published as
[`tda_core`](https://crates.io/crates/tda_core):

```toml
[dependencies]
tda_core = "0.4"
```

For a point cloud, pass a flat row-major `(n, d)` slice. The ambient dimension
is inferred from the slice length and `n`, and the defaults (peel on, parallel
on, no explicit threshold) are the recommended configuration:

```rust
use tda_core::{persistent_homology, Error};

fn main() -> Result<(), Error> {
    let points: Vec<f32> = vec![
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
    ];

    let barcode = persistent_homology(&points, 4 /* points */, 1 /* max_dim */)?;

    for (dim, intervals) in barcode.intervals.iter().enumerate() {
        for interval in intervals {
            println!("H{dim}: [{}, {})", interval.birth, interval.death);
        }
    }

    Ok(())
}
```

Everything else — distance-matrix input, thresholds, the optional transforms —
goes through `Config` and the `_with` variants:

```rust
use tda_core::{persistent_homology_with, Config, Error};

fn main() -> Result<(), Error> {
    let distances: Vec<f32> = vec![
        0.0, 1.0, 2.0,
        1.0, 0.0, 1.5,
        2.0, 1.5, 0.0,
    ];

    let config = Config {
        distance_matrix: true,
        ..Config::default()
    };
    let barcode = persistent_homology_with(&distances, 3, 1, &config)?;

    println!("{:?}", barcode.intervals);
    Ok(())
}
```

Sparse COO distance graphs use `persistent_homology_sparse` /
`persistent_homology_sparse_with` (and the matching `filtration_size_sparse`
functions), taking `rows`, `cols`, and `values` slices plus `n`. Cocycle
representatives come from the `_reps` siblings (`persistent_homology_reps`,
`persistent_homology_reps_with`, and the sparse variants), which return a
`RepresentedBarcode` whose `representatives[d][i]` is aligned with
`barcode.intervals[d][i]`.

Invalid inputs return `tda_core::Error`, including shape mismatches, invalid
thresholds, too few or too many points, and unsupported dimensions.

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

## License

TDA is distributed under the [MIT License](LICENSE.md).
