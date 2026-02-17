# PtyLab with JAX

![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)

This is the experimental clone of [PtyLab.py](https://github.com/PtyLab/PtyLab.py) that is refactored to have a JAX backend. This is mainly done for adding new models flexibly and use the automatic differentiation feature.

PtyLab is an inverse modeling toolbox for Conventional (CP) and Fourier (FP) ptychography in a unified framework. For more information please check the [paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-9-13763&id=529026).

## Getting started

To explore use cases of PtyLab, check the [example_scripts](example_scripts) and [jupyter_tutorials](jupyter_tutorials) directories. However, please install the package first as described in the below sections.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for package management.

### From source (pip)

```bash
pip install git+https://github.com/ShantanuKodgirwar/PtyLabX.git
```

### With GPU support (CUDA 12)

```bash
pip install "ptylabx[cuda12]@git+https://github.com/ShantanuKodgirwar/PtyLabX.git"
```

### Development

Clone the repository and install with uv in a virtual environment:

```bash
git clone https://github.com/ShantanuKodgirwar/PtyLabX.git
cd PtyLabX
uv venv ptylabx-venv
source ptylabx-venv/bin/activate
uv sync
```

To install with GPU support and dev dependencies:

```bash
uv sync --extra cuda12,dev
```
For testing whether GPU is being used correctly

```bash
uv run python -m pytest tests/test_jax_device.py -v -s
```

If you would like to contribute to this package, especially if it involves modifying dependencies, please checkout the [`CONTRIBUTING.md`](CONTRIBUTING.md) file.

## Citation

If you use this package in your work, cite us as below.

```tex
@article{Loetgering:23,
        author = {Lars Loetgering and Mengqi Du and Dirk Boonzajer Flaes and Tomas Aidukas and Felix Wechsler and Daniel S. Penagos Molina and Max Rose and Antonios Pelekanidis and Wilhelm Eschen and J\"{u}rgen Hess and Thomas Wilhein and Rainer Heintzmann and Jan Rothhardt and Stefan Witte},
        journal = {Opt. Express},
        number = {9},
        pages = {13763--13797},
        publisher = {Optica Publishing Group},
        title = {PtyLab.m/py/jl: a cross-platform, open-source inverse modeling toolbox for conventional and Fourier ptychography},
        volume = {31},
        month = {Apr},
        year = {2023},
        doi = {10.1364/OE.485370},
}
```
