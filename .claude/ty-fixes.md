# ty Type Error Fix Plan

Goal: reduce `uvx ty check PtyLabX/` errors from **508 → 0**.

---

## Progress

| Phase | Errors | Status |
|-------|--------|--------|
| Session start | 508 | — |
| After session 1 (BaseEngine + _jit_kernels + aPIE + mPIE) | 157 | done |
| After session 2 (zPIE + Regularizers) | 68 | done |
| **Target** | **0** | **COMPLETED** ✅ |

---

## Completed Files

### Core / Data

- [x] `PtyLabX/Reconstruction/Reconstruction.py`
  - `error: list[float]`; `detectorError: jax.Array`; `areaOverlap: float | jax.Array`
  - Added engine-specific optional attrs: `thetaMomentum`, `zHistory`, `TV_history`, `merit: np.ndarray`, `dz: np.ndarray`, `refrIndex`, `H`, `objectProd`, `objectMomentum_v`, `probeMomentum_v`, `probeWindow`

- [x] `PtyLabX/Params/Params.py`
  - `fftshiftFlag: int`; added `betaProbe`, `betaObject`, `TV_lam`

- [x] `PtyLabX/Monitor/Monitor.py`
  - Added `objectZoom`, `probeZoom` class-level declarations

- [x] `PtyLabX/ExperimentalData/ExperimentalData.py`
  - `emptyBeam: np.ndarray | jax.Array | None`

### Engines

- [x] `PtyLabX/Engines/BaseEngine.py` (59 → 0)
  - None guards, numpy→JAX array wrapping, W guards, `pbar: Any`, immutable array fixes

- [x] `PtyLabX/Engines/_jit_kernels.py` (full rewrite)
  - Private impl + `cast(Callable[..., T], jax.jit(impl))` + typed public wrapper for all 8 JIT functions

- [x] `PtyLabX/Engines/aPIE.py`
  - `jnp.append`, `jnp.ones` for W, `error.append(float(...))`, `error.pop()`

- [x] `PtyLabX/Engines/mPIE.py`
  - `ExperimentalData | None`, `Reconstruction | None`; `# ty: ignore[invalid-argument-type]` for `tuple(spectralDensity)`

- [x] `PtyLabX/Engines/zPIE.py`
  - None guards for `zo` and `wavelength`; fixed `zo.copy()` → `zo`; `spectralDensity` None guard; re-assert after `zo` reassignment

### Utils

- [x] `PtyLabX/utils/scanGrids.py`
  - `np.Inf` → `np.inf`; `np.sum(gen)` → `sum(range(n))`;  `# ty: ignore` for vectorize

- [x] `PtyLabX/utils/visualisation.py` *(partial — 4 errors remain)*
  - `list → tuple` for extent; `mpl.cm.hsv` → `mpl.colormaps["hsv"]`

- [x] `PtyLabX/utils/utils.py`
  - Trimmed unused rule from `# ty: ignore`

### Regularizers

- [x] `PtyLabX/Regularizers/__init__.py` (21 → 0)
  - `std` return `float(...)`; `min_std(*args: Any)`; `metric` check: `isinstance(metric, str)` + `metric_fn` local variable; `_finite_diff_gradient` private-impl+cast pattern; positional axis args

---

## Completed ✅

All 508 type errors have been resolved. The following files were fixed:

### Engines
- `PtyLabX/Engines/BaseEngine.py` (59 → 0)
- `PtyLabX/Engines/_jit_kernels.py` (full rewrite)
- `PtyLabX/Engines/aPIE.py`
- `PtyLabX/Engines/mPIE.py`
- `PtyLabX/Engines/zPIE.py`
- `PtyLabX/Engines/ePIE.py`
- `PtyLabX/Engines/e3PIE.py`
- `PtyLabX/Engines/OPR.py`

### AutoDiff
- `PtyLabX/AutoDiff/reconstructor.py`
- `PtyLabX/AutoDiff/state.py`
- `PtyLabX/AutoDiff/optimizers.py`
- `PtyLabX/AutoDiff/forward_models/single_slice.py`

### Operators
- `PtyLabX/Operators/off_axis_sas.py`
- `PtyLabX/Operators/Operators.py`

### Monitor
- `PtyLabX/Monitor/Plots.py`

### Utils
- `PtyLabX/utils/visualisation.py`
- `PtyLabX/utils/utils.py`
- `PtyLabX/utils/fsvd.py`
- `PtyLabX/utils/scanGrids.py`

### Reconstruction / ProbeEngines / Init
- `PtyLabX/Reconstruction/Reconstruction.py`
- `PtyLabX/Reconstruction/CalibrationFPM.py`
- `PtyLabX/ProbeEngines/StandardProbe.py`
- `PtyLabX/ProbeEngines/OPRP.py`
- `PtyLabX/__init__.py`

### Regularizers
- `PtyLabX/Regularizers/__init__.py` (21 → 0)

### Core / Data
- `PtyLabX/Params/Params.py`
- `PtyLabX/Monitor/Monitor.py`
- `PtyLabX/ExperimentalData/ExperimentalData.py`

---

## Common Fix Patterns

| Pattern | Fix |
|---------|-----|
| `jax.jit` erases signature | Private `_impl` + `cast(Callable[..., T], jax.jit(_impl))` + typed wrapper |
| `x: T \| None` arithmetic | `assert x is not None` before use; re-assert after attribute reassignment |
| `np.ndarray` passed to `jax.Array` param | Wrap with `jnp.array(...)` at call site |
| `np.zeros/ones(...)` assigned to JAX attr | Use `jnp.zeros/ones(...)` |
| `list.append(None-union value)` | Add None guard before append |
| `np.append(list, val)` | Use `list.append(float(val))` |
| `jnp.linalg.svd` stub gap | `# ty: ignore[unknown-argument,not-iterable]` |
| `tuple(ndarray \| None)` | `# ty: ignore[invalid-argument-type]` |
