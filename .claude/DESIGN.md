# AutoDiff Module Architecture — PtyLabX

## Overview

A new `PtyLabX/AutoDiff/` subpackage for automatic differentiation (AD) based ptychographic reconstruction, designed to coexist with the existing iterative engines (ePIE, mPIE, etc.) without refactoring them.

**Inspired by:** Phaser (JAX), PtyRAD (PyTorch), chromatix (JAX wave optics), and the original PtyLab paper.

## Development Roadmap

Incremental feature plan, each building on the previous:

| Phase | Feature | What Changes |
|-------|---------|--------------|
| **1** | Object-only reconstruction (known probe) | `single_slice.py` forward model + `probe_lr=0.0` |
| **2** | Blind reconstruction (object + probe) | Same code, both LRs > 0 |
| **3** | Multislice | New `forward_models/multi_slice.py` — loop over `nslice` with inter-slice propagation |
| **4** | Mixed states (object & probe modes) | Forward model sums over `nosm`/`npsm` dims (already in 6D array convention) |
| **5** | OPR (Orthogonal Probe Relaxation) | New forward model with per-position probe decomposition |
| **6** | Future extensions | New forward models, losses, regularizers, state fields as research evolves |

Each phase adds a file or adjusts optimizer config — no refactoring of previous work.

## Design Principles

1. **Pure functions over classes** — Forward models, losses, regularizers are plain callables. No inheritance needed. Compose with `functools.partial` for configuration.
2. **Explicit state pytree** — `PtychographyState` (NamedTuple) holds only differentiable parameters. Static metadata in `StaticConfig`. Clean separation for `jax.grad`.
3. **Composable** — `build_loss(forward_model, data_loss, regularizers)` assembles a single differentiable objective.
4. **Per-parameter optimization** — `optax.multi_transform` for independent learning rates on object, probe, positions, background.
5. **Reuse existing infra** — Propagation via existing `fft2c`/`ifft2c` + `_make_quad_phase`. State syncs back to `Reconstruction` for monitoring/saving.
6. **JIT-compiled inner loops** — Batch processing uses `jax.vmap` over positions and `jax.jit` for the full step. Generator yields per-epoch (not per-batch) so the inner loop is never interrupted.
7. **CPM-first, extensible** — Initial scope is CPM single-slice. FPM/multislice added later as new forward model files with zero architectural changes.

## Module Map

```
PtyLabX/AutoDiff/
├── __init__.py              build_loss(), GradientReconstructor
├── _state.py                PtychographyState, StaticConfig, conversion fns
├── _propagators.py          Pure propagation (reuses Operators internals)
├── reconstructor.py         GradientReconstructor (JIT-compiled optimization loop)
├── optimizers.py            build_optimizer() — per-parameter LR via optax
├── forward_models/
│   ├── __init__.py
│   └── single_slice.py      Standard CPM: patch × probe → propagate → |·|²
├── losses/
│   ├── __init__.py
│   ├── gaussian.py          ||√I_meas - √I_pred||²
│   └── poisson.py           I_pred - I_meas·log(I_pred)
└── regularizers/
    ├── __init__.py
    └── tv.py                Differentiable TV on object

PtyLabX/Engines/
└── GradientEngine.py        BaseEngine adapter for easyInitialize() compatibility
```

## Data Flow

```
ExperimentalData ──┐
Reconstruction ────┼── state_from_reconstruction() ──→ PtychographyState
Params ────────────┘                                         │
                    static_from_reconstruction() ──→ StaticConfig
                                                             │
                              ┌───────────────────────────────┘
                              ▼
            build_loss(forward_model, data_loss, regularizers)
                              │
                              ▼
                    jax.value_and_grad(loss_fn)
                              │
                              ▼
                    optax.update() + apply_updates()
                              │
                              ▼
                    state_to_reconstruction() ──→ Reconstruction (for Monitor/save)
```

## Key Interfaces

### Forward Model
```python
def my_forward(state: PtychographyState, position_indices, positions_all, static: StaticConfig) -> jnp.ndarray:
    """Returns predicted intensities: (batch_size, Nd, Nd)"""
```

### Loss Function
```python
def my_loss(I_measured: jnp.ndarray, I_predicted: jnp.ndarray) -> jnp.ndarray:
    """Returns scalar loss."""
```

### Regularizer
```python
def my_reg(state: PtychographyState, static: StaticConfig) -> jnp.ndarray:
    """Returns scalar penalty."""
```

## Usage

### Simple (via easyInitialize)
```python
import PtyLabX
from PtyLabX import Engines

data, recon, params, monitor, engine = PtyLabX.easyInitialize("data.hdf5", engine=Engines.GradientEngine)
engine.numIterations = 100
engine.object_lr = 5e-4
engine.probe_lr = 1e-3
for loop, batch in engine.reconstruct():
    pass
```

### Advanced (composable)
```python
from PtyLabX.AutoDiff import build_loss, GradientReconstructor
from PtyLabX.AutoDiff.forward_models import single_slice_forward
from PtyLabX.AutoDiff.losses import poisson_loss
from PtyLabX.AutoDiff.regularizers import object_tv
from PtyLabX.AutoDiff.optimizers import build_optimizer

loss_fn = build_loss(
    forward_model=single_slice_forward,
    data_loss=poisson_loss,
    regularizers=[lambda s, c: object_tv(s, c, weight=1e-4)],
)
optimizer = build_optimizer(object_lr=1e-3, probe_lr=5e-4, position_lr=1e-5)
reconstructor = GradientReconstructor(loss_fn, optimizer, recon, data, params, batch_size=64)
reconstructor.reconstruct(num_iterations=200)
```

## Adding New Components

**New forward model** → Create `PtyLabX/AutoDiff/forward_models/my_model.py`, write a function matching the signature. Use with `build_loss(forward_model=my_func)`.

**New loss** → Create `PtyLabX/AutoDiff/losses/my_loss.py`, write `(I_meas, I_pred) -> scalar`. Use with `build_loss(data_loss=my_func)`.

**New regularizer** → Create `PtyLabX/AutoDiff/regularizers/my_reg.py`, write `(state, static) -> scalar`. Add to regularizers list.

**New optimizable parameter** → Add field to `PtychographyState`, update `state_from_reconstruction`/`state_to_reconstruction`, add LR to `build_optimizer`.

## Documentation Checklist

When adding or changing AutoDiff modules, update `docs/` accordingly:

- [ ] **API reference** in `docs/autodiff/` — document new forward models, losses, regularizers, optimizer options
- [ ] **Tutorials** in `docs/tutorials/` — Jupyter notebooks showing AD reconstruction (object-only, blind, multislice, etc.)
- [ ] **`docs/cpm/engines.md`** — add GradientEngine alongside existing engine docs
- [ ] **`docs/getting-started/quickstart.md`** — add AD example if it becomes a recommended workflow

Docs use **mkdocs** (`mkdocs.yml` + `mkdocs-material`). Keep docs in sync with code changes.
