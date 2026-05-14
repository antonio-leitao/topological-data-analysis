<p align="center">
  <img src='assets/logo.svg' width='200px' align="center"></img>
</p>

<div align="center">
<h3 max-width='200px' align="center">Topological Data Analysis</h3>
  <p><i>Not limited to just H0 and H1<br/>
  Bringing topological data analysis to python ecosystem<br/>
  Built with Rust</i><br/></p>
  <p>
<img alt="Pepy Total Downlods" src="https://img.shields.io/pepy/dt/tda?style=for-the-badge&logo=python&labelColor=white&color=blue">
  </p>
</div>

#

### Contents

- [Installation](#installation)
  - [Compiling from source](#compilation-from-source)
- [Usage](#usage)
  - [Parameters](#parameters)
  - [Return value](#return-value)

TDA is a python package for topological data analysis written in Rust.

## Installation

> [!CAUTION]
> TDA is in very early stages of development.

Pre-built packages for MacOS, Windows and most Linux distributions are on
[PyPI](https://pypi.org/project/tda/) and can be installed with:

```sh
pip install tda
```

On uncommon architectures you may need to first
[install Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html)
before running `pip install tda`.

### Compilation from source

To compile from source you'll need
[Rust/Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html)
and [maturin](https://github.com/PyO3/maturin#maturin) for the python bindings.
Maturin is best used from within a Python virtual environment:

```sh
# activate your desired virtual environment first, then:
pip install maturin
git clone https://github.com/antonio-leitao/topological-data-analysis.git
cd topological-data-analysis
# build and install the package:
maturin develop --release
```

## Usage

```python
import numpy as np
import tda

# Point cloud: (n, d) float32 array
X = np.random.rand(200, 3).astype(np.float32)
barcode = tda.persistent_homology(X, max_dim=2)

# barcode[d] is an (k_d, 2) array of [birth, death] pairs for dimension d
print(barcode[0])  # H0
print(barcode[1])  # H1
print(barcode[2])  # H2
```

You can also pass a precomputed distance matrix:

```python
D = np.asarray(my_distance_matrix, dtype=np.float32)  # shape (n, n)
barcode = tda.persistent_homology(D, max_dim=1, distance_matrix=True)
```

### Parameters

| Parameter         | Type            | Default | Description                                                                                                                                   |
| ----------------- | --------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `data`            | `np.ndarray`    | —       | Float32 array. Shape `(n, d)` for a point cloud, or `(n, n)` for a distance matrix. C-contiguous input is zero-copy.                          |
| `max_dim`         | `int`           | `1`     | Highest homology dimension to compute. Intervals are returned for every dimension `0..=max_dim`. Capped at 4.                                 |
| `threshold`       | `float \| None` | `None`  | Maximum filtration value. If `None`, the enclosing radius of the data is used. Otherwise the effective threshold is `min(threshold, radius)`. |
| `distance_matrix` | `bool`          | `False` | If `True`, `data` is interpreted as a precomputed square distance matrix. Symmetry, zero diagonal, and non-negativity are assumed.            |

### Return value

A `list` of length `max_dim + 1`. Entry `barcode[d]` is a `numpy.ndarray` of
shape `(k_d, 2)` where each row is a `[birth, death]` pair for a feature in
dimension `d`. A `death` value of `inf` marks an essential (never-dying)
feature.

### From Rust

The core crate is also published as `tda`. Add it to your `Cargo.toml`:

```toml
[dependencies]
tda = "0.2"
```

Point cloud — pass a flat row-major `(n, d)` slice:

```rust
use tda::persistent_homology;

let points: Vec<f32> = vec![
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    1.0, 1.0,
];
let n = 4;
let d = 2;

let barcode = persistent_homology(&points, n, d, /* max_dim */ 1, /* threshold */ None)?;

for (dim, intervals) in barcode.intervals.iter().enumerate() {
    for iv in intervals {
        println!("H{dim}: [{}, {})", iv.birth, iv.death);
    }
}
# Ok::<(), tda::Error>(())
```

Precomputed distance matrix — pass a flat row-major `(n, n)` slice:

```rust
use tda::persistent_homology_from_distances;

let distances: Vec<f32> = vec![
    0.0, 1.0, 2.0,
    1.0, 0.0, 1.5,
    2.0, 1.5, 0.0,
];

let barcode = persistent_homology_from_distances(&distances, 3, 1, None)?;
# Ok::<(), tda::Error>(())
```

Errors come back as `tda::Error` (a `thiserror` enum) with variants for bad shapes, too few / too many points, dimension caps, and invalid thresholds.
