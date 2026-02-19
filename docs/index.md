# PtyLabX

PtyLabX is a JAX-based ptychographic reconstruction toolbox, forked from [PtyLab.py](https://github.com/PtyLab/PtyLab.py). It performs iterative phase retrieval for **Conventional Ptychographic Microscopy (CPM)** and **Fourier Ptychographic Microscopy (FPM)** in a unified framework. The computational backend is [JAX](https://github.com/jax-ml/jax), enabling automatic differentiation, GPU acceleration, and JIT compilation.

For the original publication, see: [PtyLab.m/py/jl: a cross-platform, open-source inverse modeling toolbox for conventional and Fourier ptychography](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-9-13763&id=529026).

## Where to start

- **[Installation](getting-started/installation.md)** — Set up PtyLabX with CPU or GPU support
- **[Quick Start](getting-started/quickstart.md)** — Run your first reconstruction in a few lines of code
- **[CPM Workflow](cpm/overview.md)** — Understand the conventional ptychography pipeline
- **[Tutorial Notebook](tutorials/demo.ipynb)** — Step-by-step simulation and reconstruction of a CPM experiment. One can find more tutorials on different use cases.

## Citation

If you use PtyLabX in your work, please cite:

```bibtex
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
