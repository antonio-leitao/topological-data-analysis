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
- [Roadmap](#roadmap)

TDA is a python package for topological data analysis written in Rust.

## Installation

> [!CAUTION]
> TDA is in very early stages of development.

Pre-built packages for MacOS, Windos and most Linux distributions in [PyPI](https://pypi.org/project/tda/) and can be installed with:

```sh
pip install tda
```
On uncommon architectures, you may need to first
[install Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) before running `pip install tda`.

### Compilation from source

In order to compile from source you will need to [install Rust/Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) and [maturin](https://github.com/PyO3/maturin#maturin) for the python bindings.
Maturin is best used within a Python virtual environment:

```sh
# activate your desired virtual environment first, then:
pip install maturin
git clone https://github.com/antonio-leitao/topological-data-analysis.git
cd topological-data-analysis
# build and install the package:
maturin develop --release
```

## Roadmap

- [x] Custom Simplicial Complexes
- [x] Clique Complexes
- [x] Betti Numbers
- [ ] Chunky Homology
- [ ] Homology representatives
- [ ] Optimal Homology representatives
- [ ] Persistent Homology
