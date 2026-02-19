# PtyLab with JAX

![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)

This is an experimental clone of [PtyLab.py](https://github.com/PtyLab/PtyLab.py) that is refactored to have a JAX backend. This is mainly done for adding new models flexibly for using AD and other cool features from JAX for speed-ups. 

PtyLab is an inverse modeling toolbox for Conventional (CP) and Fourier (FP) ptychography in a unified framework. For more information please check the [paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-9-13763&id=529026).

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
> [!NOTE] 
> We recommend using `cuda12` flag for older GPUs (SM version 7.5 or older). However, note that JAX will eventually discontinue support for `cuda12`. See [here](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu) for more details.

#### CUDA 13 (Recommended for newer GPUs)

```bash
pip install "ptylabx[cuda13]@git+https://github.com/ShantanuKodgirwar/PtyLabX.git"
```

### Development

This project uses the (super-fast) package manager [uv](https://docs.astral.sh/uv/). Follow the below steps:

```bash
git clone https://github.com/ShantanuKodgirwar/PtyLabX.git
cd PtyLabX
uv sync 
```

The dependencies are installed in a virtual environment which can be activated within your IDE interpreter or in the terminal `source .venv/bin/activate`.

To install with GPU support and dev dependencies:

```bash
uv sync --extra cuda13,dev # alternatively `cuda12` for older GPUs
```

For testing and formatting (after a contribution), run:

```bash
uv run python -m pytest tests/ -v -s   # run tests (add test_jax_device.py to check GPU)
uv run ruff check PtyLabX/             # check style
uv run ruff format PtyLabX/            # auto-format
```

> [!NOTE] 
> Contributions are welcome such as new engines, bug fixes, example scripts, or documentation. If its a new implementation under the package, please add a test for it and run the above commands. CI also ensures this on every push and pull request via GitHub Actions.

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
