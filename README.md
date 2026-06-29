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

TDA computes Vietoris–Rips persistent homology from point clouds or
precomputed distance matrices. The performance-critical implementation is
written in Rust and is available through both Python and Rust APIs.

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

# Point cloud: an (n, d) float32 array.
points = np.random.rand(200, 3).astype(np.float32)
barcode = tda.persistent_homology(points, max_dim=2)

# barcode[d] is a (k_d, 2) array of [birth, death] intervals.
print(barcode[0])  # H0
print(barcode[1])  # H1
print(barcode[2])  # H2
```

Precomputed distance matrices are also supported:

```python
distances = np.asarray(my_distance_matrix, dtype=np.float32)
barcode = tda.persistent_homology(
    distances,
    max_dim=1,
    distance_matrix=True,
)
```

Use `filtration_size` to inspect the size of the truncated filtration built
with the same options:

```python
size = tda.filtration_size(points, max_dim=2, peel=True)
print(size)
```

### Parameters

Both `persistent_homology` and `filtration_size` accept the following
parameters:

| Parameter         | Type            | Default | Description                                                                                                                                             |
| ----------------- | --------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data`            | `np.ndarray`    | —       | Two-dimensional `float32` array. Shape `(n, d)` for a point cloud or `(n, n)` for a distance matrix. Inputs are copied into row-major internal storage. |
| `max_dim`         | `int`           | `1`     | Highest homology dimension to compute. Capped at 4.                                                                                                     |
| `threshold`       | `float \| None` | `None`  | Maximum filtration value. `None` uses the enclosing radius; an explicit value is capped by that radius.                                                 |
| `distance_matrix` | `bool`          | `False` | Interpret `data` as a square distance matrix. Symmetry, zero diagonal, and non-negativity are assumed rather than validated.                            |
| `quotient`        | `bool`          | `False` | Use the smaller quotient-cover filtration. This is an approximation with a `log(3)` interleaving guarantee, not the exact Vietoris–Rips barcode.        |
| `peel`            | `bool`          | `False` | Apply an exact strong-collapse reduction before computing the result.                                                                                   |
| `parallel`        | `bool`          | `True`  | Enable parallel preprocessing, sorting, and candidate assembly for sufficiently large inputs.                                                           |

`persistent_homology` returns a list of length `max_dim + 1`. Entry
`barcode[d]` is a NumPy array of shape `(k_d, 2)` whose rows are
`[birth, death]` intervals. A death value of `inf` marks an essential feature.

## Rust usage

The Rust crate is published as
[`tda_core`](https://crates.io/crates/tda_core):

```toml
[dependencies]
tda_core = "0.3"
```

For a point cloud, pass a flat row-major `(n, d)` slice. The ambient dimension
is inferred from the slice length and `n`:

```rust
use tda_core::{persistent_homology, Error};

fn main() -> Result<(), Error> {
    let points: Vec<f32> = vec![
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
    ];

    let barcode = persistent_homology(
        &points,
        4,     // number of points
        1,     // max_dim
        None,  // threshold
        false, // distance_matrix
        false, // quotient
        false, // peel
        true,  // parallel
    )?;

    for (dim, intervals) in barcode.intervals.iter().enumerate() {
        for interval in intervals {
            println!("H{dim}: [{}, {})", interval.birth, interval.death);
        }
    }

    Ok(())
}
```

Distance matrices use the same function with `distance_matrix = true`:

```rust
use tda_core::{persistent_homology, Error};

fn main() -> Result<(), Error> {
    let distances: Vec<f32> = vec![
        0.0, 1.0, 2.0,
        1.0, 0.0, 1.5,
        2.0, 1.5, 0.0,
    ];

    let barcode = persistent_homology(
        &distances,
        3,
        1,
        None,
        true,  // distance_matrix
        false, // quotient
        false, // peel
        true,  // parallel
    )?;

    println!("{:?}", barcode.intervals);
    Ok(())
}
```

Invalid inputs return `tda_core::Error`, including shape mismatches, invalid
thresholds, too few or too many points, and unsupported dimensions.

## License

TDA is distributed under the [MIT License](LICENSE.md).
