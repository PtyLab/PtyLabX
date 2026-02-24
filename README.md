# PtyLabX

![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)
![Tests](https://github.com/ShantanuKodgirwar/PtyLabX/actions/workflows/test.yml/badge.svg)
![Ruff](https://github.com/ShantanuKodgirwar/PtyLabX/actions/workflows/formatter.yml/badge.svg)

> [!WARNING]
> This project is under active development. The documentation is incomplete and does not cover all use cases. Some existing features might either be slow/unstable until everything is thoroughly tested. API could change mainly around making some interface immutable to allow gradient flow.

PtyLabX is an experimental JAX-based ptychographic reconstruction toolbox, forked from [PtyLab.py](https://github.com/PtyLab/PtyLab.py). It performs iterative phase retrieval for **Conventional Ptychographic Microscopy (CPM)** and **Fourier Ptychographic Microscopy (FPM)** in a unified framework. See the [original publication](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-9-13763&id=529026) for more details. 


[**Installation**](#installation) | [**Development and Contribution**](#development-and-contribution) | [**Getting Started**](#getting-started) | [**PtyLabX Documentation**](#documentation)

The computational backend is [JAX](https://github.com/jax-ml/jax), enabling not just GPU acceleration, but also features such as JIT, XLA (amongst others) to speed-up existing PtyLab engines and flexibility of adding newer models to use AD and enable gradient-based optimization.


## Getting started

To explore use cases of PtyLab, check the [example_scripts](example_scripts) and [jupyter_tutorials](jupyter_tutorials) directories. However, please install the package first as described in the below sections.

## Installation

### From source (pip)

```bash
pip install git+https://github.com/ShantanuKodgirwar/PtyLabX.git
```

### With GPU support

#### CUDA 12

```bash
pip install "ptylabx[cuda12]@git+https://github.com/ShantanuKodgirwar/PtyLabX.git"
```
#### CUDA 13 (Recommended for newer GPUs)

This is recommended for newer GPUs with SM version 7.5 or newer.

```bash
pip install "ptylabx[cuda13]@git+https://github.com/ShantanuKodgirwar/PtyLabX.git"
```

## Development and Contribution

This project uses the (super-fast) package manager [uv](https://docs.astral.sh/uv/). Follow the below steps:

```bash
git clone https://github.com/ShantanuKodgirwar/PtyLabX.git
cd PtyLabX
uv sync 
```

The dependencies are installed in a virtual environment which can be activated within your IDE interpreter or in the terminal `source .venv/bin/activate`.

To install with GPU support:

```bash
uv sync --extra cuda13 # alternatively `cuda12` for older GPUs
```
To test if GPU is detected:

```bash
uv run python -c "import jax; print(jax.default_backend())"
```

> [!NOTE]
> Contributions are welcome (new engines, bug fixes, example scripts, documentation). For any new implementations,
> 1. Add an appropriate test under [tests/](tests/) using [pytest](https://github.com/pytest-dev/pytest).
> 2. Run tests locally before opening a PR:

```bash
uv run pytest tests/ -v -s
```

Code formatting is handled automatically by CI when a PR is opened. Tests also run automatically on every PR. If they fail, please review your changes.

## Documentation

The documentation is a work-in-progress, but can be rendered as a webpage. Install the dependency:

```bash
uv sync --group docs
```

and build the documentation, generating a hyperlink for preview:

```bash
uv run mkdocs serve
```
## Acknowledgements

Some of the new implementation ideas mainly follow the JAX-based differentiable wave optics library [chromatix](https://github.com/chromatix-team/chromatix/) and the recent electron ptychography package [phaser](https://github.com/hexane360/phaser), also based on JAX. Additionally some ideas have also been adapted from the pytorch-based [ptyrad](https://github.com/chiahao3/ptyrad) library.

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
