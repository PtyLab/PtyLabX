# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PtyLabX is a JAX-based ptychographic reconstruction toolbox, forked from PtyLab.py. It performs iterative phase retrieval for Conventional Ptychographic Microscopy (CPM) and Fourier Ptychographic Microscopy (FPM). The computational backend is **JAX** (`jax.numpy`); numpy is only used for I/O (h5py) and matplotlib visualization.

## Commands

```bash
# Install dependencies
uv sync

# Install with GPU (CUDA 12) support
uv sync --extra cuda12

# Run all tests
uv run python -m pytest tests/ -v -s

# Run a single test
uv run python -m pytest tests/test_jax_device.py -v -s

# Lint and format
uv run ruff check PtyLabX/
uv run ruff format PtyLabX/
```

## Project-Specific Guidelines

- Using `uv` to manage the project and `ruff` for linting and formatting.
- When implementing any change, refer to the latest API.
- Package name is `ptylabx`, import directory is `PtyLabX/`.
- API changes must be minimal and only if required for gradient flow, speed enhancement, etc. To get a basic idea of API, see `example_scripts/`.
- All computational code uses `jax.numpy` (imported as `jnp`). Do **not** use numpy for computation.
- After major changes, update `.claude/logging.md` file for logging. Start with the date YYYY-MM-DD and title
- When adding a new implementation, if possible add a test for it under the `tests/` directory.
- Keep imports at the top of a python file and remove unused imports.
- Add docstrings for implementations and keep them consistent throughout the library.
- JAX arrays are immutable: use `array.at[idx].set(val)` instead of `array[idx] = val`. Use `jnp.where()` instead of boolean indexing assignment.
- Use `np.asarray(jax_array)` to convert JAX arrays to numpy (for h5py, matplotlib, scipy functions not in jax.scipy).
- Use `optax` for optimizers (not `jax.experimental.optimizers`, which is removed).
- For random numbers in computational code, use `jax.random` with explicit PRNG keys. Numpy random is fine for initialization/shuffling.
- `ruff` line length is 120 characters.
- When adding or changing modules/APIs, update the `docs/` directory accordingly (API docs, examples, tutorials). The docs use mkdocs (see `mkdocs.yml`). Keep docs in sync with code.
- For `PtyLabX/AutoDiff/` changes specifically, see `.claude/DESIGN.md` for the architecture plan and development roadmap.

## Architecture

### Data Flow

`ExperimentalData` (loads .hdf5) -> `Reconstruction` (holds object/probe arrays) -> `Engine.reconstruct()` (iterative updates) -> `Monitor` (visualization)

Quick start via `PtyLabX.easyInitialize(filename, engine=Engines.ePIE)`.

### Key Modules

- **`PtyLabX/__init__.py`**: `easyInitialize()` convenience function wires up all components.
- **`PtyLabX/ExperimentalData/`**: Loads experimental data from HDF5 files. Data stays as numpy arrays from h5py.
- **`PtyLabX/Reconstruction/`**: Holds mutable reconstruction state (object, probe, positions). `Reconstruction` also contains `TV_autofocus` for propagation distance optimization.
- **`PtyLabX/Params/`**: Configuration container. `gpuFlag` is auto-detected from `jax.default_backend()`.
- **`PtyLabX/Engines/`**: Reconstruction algorithms. All inherit from `BaseEngine` which provides `intensityProjection()`, `applyConstraints()`, `positionCorrection()`, `showReconstruction()`. Each engine implements `reconstruct()` with its own object/probe update rules.
  - **Engine hierarchy**: `BaseEngine` -> `ePIE`, `mPIE` (momentum), `pcPIE` (position correction), `zPIE` (z-update), `e3PIE` (multislice), `ePIE_TV`, `qNewton`, etc.
- **`PtyLabX/Operators/`**: Wave propagation operators (ASP, Fresnel, scaled ASP). `Operators.py` has `object2detector()` / `detector2object()`. `_propagation_kernels.py` caches quadratic phase terms with `@lru_cache`.
- **`PtyLabX/Regularizers/`**: TV, gradient-based regularization metrics for autofocus.
- **`PtyLabX/utils/gpuUtils.py`**: JAX backend utilities. `asNumpyArray()`, `asJaxArray()`, `check_jax_backend()`. Legacy shims (`getArrayModule`, `isGpuArray`) exist for compatibility.
- **`PtyLabX/utils/utils.py`**: `fft2c`/`ifft2c` (centered FFT), `orthogonalizeModes`, `circ`, `gaussian2D`, initialization helpers.
- **`PtyLabX/Monitor/`**: Real-time visualization during reconstruction (matplotlib/pyqtgraph/bokeh).
- **`PtyLabX/ProbeEngines/`**: Probe generation strategies (StandardProbe, OPRP).
