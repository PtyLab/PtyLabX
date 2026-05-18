# 2026-02-18 – JAX Optimization & Test Consolidation

## Overview

Major refactoring to add JAX performance features (`@jax.jit`, `jax.vmap`, `jax.lax.fori_loop`) across the codebase, plus consolidation of all tests from scattered `PtyLabX/**/test*` locations into a single `tests/` directory.

---

## Phase 0: Test Infrastructure

**Created `tests/conftest.py`**
- Shared pytest fixtures: `jax_key` (session-scoped), `random_complex_field`, `random_real_field`

---

## Phase 1: `PtyLabX/utils/`

### `utils/utils.py`
- Added `@functools.partial(jax.jit, static_argnums=(1,))` to `fft2c` and `ifft2c`
- Added `@jax.jit` to `circ`, `rect`, `posit`
- Rewrote `fraccircshift` with `@jax.jit`: replaced Python `for dim in range(2)` loop with unrolled two-axis version using `jnp.roll`
- Added `@functools.partial(jax.jit, static_argnames=("method",))` to `orthogonalizeModes`

### `utils/fsvd.py`
- Added `@jax.jit` to `ortho_basis`
- Replaced Python `for` loop in `subspace_iter` with `@jax.jit` + `jax.lax.fori_loop`

### `utils/gpuUtils.py`
- **Removed** deprecated shims `getArrayModule`, `isGpuArray`, `CP_AVAILABLE` (confirmed unused via grep)

**New tests:** `tests/test_utils.py`, `tests/test_gpu_utils.py`, `tests/test_fsvd.py`

---

## Phase 2: `PtyLabX/Operators/`

### `Operators/propagator_utils.py`
- Added `@jax.jit` to `complexexp`
- `_fft_convolve2d`: JIT attempted but reverted — `_centered` uses `int()` casts incompatible with tracing

### `Operators/Operators.py`
- Created `_asp_propagate` — JIT helper for ASP propagation core
- Created `_aspw_propagate_core` — JIT helper for ASPW propagation core
- `propagate_ASP` and `aspw` delegate to these helpers

**New tests:** `tests/test_operators.py`

---

## Phase 3: `PtyLabX/Regularizers/`

### `Regularizers/__init__.py`
- **Bug fix:** `grad_y` in `_TV_jit` was using `axis=-1` for both rolls — second roll corrected to `axis=-2`
- Extracted `_TV_jit` as `@jax.jit` core; `TV` is now a thin wrapper that converts result to `float`
- Added `@functools.partial(jax.jit, static_argnums=(1,))` to `_finite_diff_gradient` (`axis` must be static)
- Added `@jax.jit` to `divergence_new` and `grad_TV`

**New tests:** `tests/test_regularizers.py` (includes regression test for the TV bug fix)

---

## Phase 4: Shared JIT Kernels

**Created `PtyLabX/Engines/_jit_kernels.py`** — all functions are `@jax.jit`:

| Function                | Used by                                 |
| ----------------------- | --------------------------------------- |
| `epie_object_update`    | ePIE, ePIE_TV, ePIE_mw, zPIE, aPIE, OPR |
| `epie_probe_update`     | ePIE, ePIE_TV, zPIE, aPIE               |
| `mpie_object_update`    | mPIE, mPIE_mw, mPIE_tv, pcPIE, multiPIE |
| `mpie_probe_update`     | mPIE, mPIE_mw, mPIE_tv, pcPIE           |
| `qnewton_object_update` | qNewton, mqNewton                       |
| `qnewton_probe_update`  | qNewton, mqNewton                       |
| `momentum_step`         | mPIE, mPIE_mw, mPIE_tv, pcPIE, multiPIE |
| `epie_object_update_tv` | ePIE_TV (TV variant)                    |

`mpie_object_update` uses `static_argnames=("fpm_mode",)` to allow Python-level branching for FPM vs CPM mode.

**New tests:** `tests/test_jit_kernels.py`

---

## Phase 5: `PtyLabX/Engines/BaseEngine.py`

Three vectorization changes (replaced Python for-loops):

1. **Spectral power correction** (was `for i in range(nlambda)`):
   - Vectorized with broadcasting: `probe * scales` where `scales` is shaped `(-1, 1, 1, 1, 1, 1)`

2. **Wavelength coupling** (was `for i in range(nlambda)`):
   - Vectorized with `jnp.roll(probe, shift=±1, axis=0)`, boundary conditions applied with `.at[0].set()` and `.at[-1].set()`

3. **Position correction cross-correlation** (was inner `for shifts in range(25)`):
   - Replaced with `jax.vmap` over a `_cc_at_shift` function vectorized over the 25 shift candidates

**New tests:** `tests/test_base_engine.py`

---

## Phase 6: Individual Engines

### Engines updated (update methods replaced with shared kernels):

| Engine        | Changes                                                                                                                |
| ------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `ePIE.py`     | `objectPatchUpdate` → `epie_object_update`, `probeUpdate` → `epie_probe_update`                                        |
| `qNewton.py`  | Both updates → `qnewton_*` kernels                                                                                     |
| `mPIE.py`     | Both updates → `mpie_*` kernels; momentum → `momentum_step`                                                            |
| `zPIE.py`     | Both updates → `epie_*` kernels                                                                                        |
| `aPIE.py`     | Both updates → `epie_*` kernels                                                                                        |
| `ePIE_TV.py`  | `objectPatchUpdate` → `epie_object_update`, `probeUpdate` → `epie_probe_update`                                        |
| `pcPIE.py`    | Both updates → `mpie_*` kernels; momentum → `momentum_step`                                                            |
| `mPIE_mw.py`  | Both updates → `mpie_*` kernels; momentum → `momentum_step`                                                            |
| `multiPIE.py` | `objectPatchUpdate` → `mpie_object_update`; momentum → `momentum_step`; `probeUpdate` kept custom (different sum axes) |
| `mqNewton.py` | Both updates → `qnewton_*` kernels                                                                                     |
| `mPIE_tv.py`  | Both updates → `mpie_*` kernels; momentum → `momentum_step`                                                            |
| `ePIE_mw.py`  | `objectPatchUpdate` → `epie_object_update`; `probeUpdate` kept custom (different sum axes)                             |
| `OPR.py`      | `objectPatchUpdate` → `epie_object_update`; `probeUpdate` kept custom (`gimmel` param)                                 |
| `e3PIE.py`    | Kept custom — unique 3D multislice signature (takes `localProbe` argument)                                             |

### Bug fixes in engines:

- **`ePIE_TV.py`**: `self.reconstruction.probe += ...` → `self.reconstruction.probe = self.reconstruction.probe + ...` (JAX immutability)
- **`mqNewton.py`**: Two `-=` in-place operations replaced with `= ... - ...`
- **`ePIE_mw.py`**: `self.reconstruction.probe += ...` → same fix
- **`OPR.py`**: 8 in-place indexed assignments fixed (`array[idx] = val` → `array.at[idx].set(val)`):
  - `probe_stack` initialization loop
  - `probe_stack` save after each position
  - `orthogonalizeIncoherentModes`: probe mode assignment
  - `average()` method: `divider[0]` and `divider[-1]`
  - `orthogonalizeProbeStack`: `s[n_dim:] = 0`, `probe_stack` blend (two variants)

### Unused imports removed from modified engines:
`matplotlib.pyplot`, `fft2c`/`ifft2c` (where unused), `jax.numpy` (in pcPIE after refactor)

**New tests:** `tests/test_engines.py` (10 kernel tests + 7 integration tests skipped without `simu.hdf5`)

---

## Phase 7: Test Migration & Cleanup

### New test files created (from migrated + new tests):

| New file                       | Source                                                                                               |
| ------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `tests/test_reconstruction.py` | `PtyLabX/Reconstruction/test/test_optimizable.py`                                                    |
| `tests/test_io.py`             | `PtyLabX/io/test/test_example_loader.py`, `test_get_example_data_folder.py`, `test_loadInputData.py` |
| `tests/test_probe_engines.py`  | `PtyLabX/ProbeEngines/test_OPRP.py`, `test_StandardProbe.py`                                         |
| `tests/test_monitor.py`        | `PtyLabX/Monitor/test/test_matplotlib_monitor.py`                                                    |

**Notes on skipped tests:**
- `test_reconstruction.py`, `test_engines.py` (integration classes), `test_io.py` (data-dependent) — skip when `example_data/simu.hdf5` is absent
- `test_probe_engines.py` — skips because `PtyLabX/ProbeEngines/__init__.py` intentionally raises `ValueError("not ready yet")`
- `test_monitor.py` — skips entirely (visual tests require display)

### Deleted (27 files + 7 directories):
All `PtyLabX/**/test*/` subdirectories and `test_*.py` files at package level, including `PtyLabX/testall.py` (empty), engine/operator/regularizer/utils/reconstruction/io/monitor test directories.

---

## Final Test State

```
71 passed, 15 skipped
```

Skips are expected: 7 integration tests (no `simu.hdf5`), 4 probe engine tests (module not ready), 3 reconstruction tests (no `simu.hdf5`), 1 monitor test (visual).

Run: `uv run python -m pytest tests/ -v -s`

---
---

# GPU Performance Optimization — 2026-02-19

## Overview

After the JAX migration, `engine_mPIE.reconstruct()` was ~4x slower than the old numpy/cupy implementation (44s vs ~10s for 50 iterations × 100 positions, Np=128). Root causes were GPU↔CPU synchronization points and excessive GPU kernel dispatch overhead for small arrays. Fixes brought runtime from **44s → ~15s** (target).

Benchmark: `example_scripts/demo_timeit.py` (compared against `PtyLab.py/example_scripts/demo_timeit.py`)

---

## Phase 1: Runtime Error Fixes (post-refactoring)

These errors appeared when first running `reconstruct()` after the JAX migration.

### `PtyLabX/Params/Params.py`
- **Missing attribute:** Added `self.saveMemory = False` to `__init__` — referenced by `BaseEngine` but not defined after refactoring.

### `PtyLabX/Engines/BaseEngine.py`
- **`detectorError` was numpy:** Changed `np.zeros` → `jnp.zeros` for `detectorError` initialization. The array is used with JAX `.at[].set()` in the reconstruction loop.

### `PtyLabX/Reconstruction/Reconstruction.py`
- **Object/probe arrays were numpy:** `initializeObjectProbe()` used `.copy()` which kept arrays as numpy. Changed to `jnp.array()` at all assignment sites:
  - `initializeObjectProbe()`: `self.object = jnp.array(self.initialGuessObject)`, same for probe
  - `load_object()`, `load_probe()`, `load()`: wrapped loaded arrays with `jnp.array()`

---

## Phase 2: Eliminating GPU↔CPU Synchronization Points (44s → 33s)

Each GPU sync forces Python to block until all pending GPU work completes, destroying JAX's async dispatch pipeline.

### Fix 1: Pre-transfer data to GPU — `BaseEngine.py`

**Problem:** `ptychogram` and `energyAtPos` stayed as numpy arrays (from h5py). Every position access triggered a CPU→GPU PCIe transfer.

**Fix:** Added `_transferDataToGPU()` method called from `_prepareReconstruction()`:
```python
def _transferDataToGPU(self):
    if not isinstance(self.experimentalData.ptychogram, jnp.ndarray):
        self.experimentalData.ptychogram = jnp.array(self.experimentalData.ptychogram)
    if not isinstance(self.experimentalData.energyAtPos, jnp.ndarray):
        self.experimentalData.energyAtPos = jnp.array(self.experimentalData.energyAtPos)
```

### Fix 2: Guard debug logging f-string — `BaseEngine.py:610-612` (CRITICAL)

**Problem:** Python evaluates f-string arguments *before* calling `logger.debug()`. Even at INFO log level, two `.sum()` calls on JAX GPU arrays executed every position → **10,000 GPU synchronization points** (2 syncs × 5,000 positions).

**Fix:** Wrapped with `if self.logger.isEnabledFor(logging.DEBUG):` guard.

### Fix 3: Use `jnp` in `probePowerCorrection` — `BaseEngine.py`

**Problem:** `np.sqrt(np.sum(self.reconstruction.probe * ...))` forces GPU→CPU transfer when `probePowerCorrectionSwitch=True` (enabled in demo). Applied in both `_initialProbePowerCorrection` and `applyConstraints`.

**Fix:** Replaced `np.sqrt(np.sum(...))` with `jnp.sqrt(jnp.sum(...))`.

### Fix 4: Use `jnp.abs` in mPIE — `mPIE.py:84`

**Problem:** `np.abs(self.reconstruction.probe)` on a JAX array forces GPU→CPU transfer during initialization.

**Fix:** Changed to `jnp.abs(...)`.

### Fix 5: Initialize momentum arrays as JAX — `Reconstruction.py`

**Problem:** `np.zeros_like(self.initialGuessObject)` creates numpy momentum arrays. First use in `@jax.jit momentum_step()` requires implicit CPU→GPU transfer.

**Fix:** Changed to `jnp.zeros(shape, dtype=jnp.complex64)`.

### Fix 6: getErrorMetrics used numpy on JAX arrays — `BaseEngine.py`

**Problem:** `np.sum`, `np.abs`, and `float()` calls in `getErrorMetrics()` and `getRMSD()` forced GPU syncs every iteration.

**Fix:** Replaced with `jnp.sum`, `jnp.abs`; removed unnecessary `float()` conversion.

### Fix 7: Throttle inner tqdm — `mPIE.py`

**Problem:** tqdm updated terminal output for every position (5,000 writes total).

**Fix:** Added `mininterval=0.5` to inner tqdm to throttle display updates.

---

## Phase 3: Fused JIT Kernel (33s → target ~15s)

### Key Insight

The old PtyLab.py demo ran with `gpuSwitch = False`, meaning it used **CPU (numpy)**, not GPU (cupy). JAX auto-detects GPU via `jax.default_backend()`. For small 128×128 arrays, individual GPU kernel dispatch overhead (~10-50μs per launch) across ~20 operations per position dominated runtime.

### `PtyLabX/Engines/_jit_kernels.py` — Added `mpie_standard_step`

A single fused `@jax.jit` kernel that combines the entire mPIE position step (~20 separate operations) into one compiled XLA kernel:

1. Extract object patch via `jax.lax.dynamic_slice` (JIT-compatible, replaces Python slice objects)
2. Exit surface wave: `objectPatch * probe`
3. Forward propagation: `fft2c(esw)`
4. Intensity estimate
5. Detector error (RMSD)
6. Standard intensity constraint (sqrt fraction)
7. Inverse propagation: `ifft2c(ESW)`
8. DELTA computation
9. mPIE object update (regularized, inlined)
10. mPIE probe update (regularized, inlined)
11. Write back via `jax.lax.dynamic_update_slice`

Returns `(object_array, probe_new, detector_error)` — one kernel dispatch instead of ~20.

Uses `static_argnames=("fpm_mode", "fftshiftSwitch")` for Python-level branching.

### `PtyLabX/Engines/mPIE.py` — Fast path integration

- Added `_can_use_fast_path()` method that checks if fused kernel is applicable:
  - Fraunhofer propagation + standard intensity constraint
  - No TV regularization, position correction, CPSC, background mode, adaptive denoising, Fourier mask
  - Probe updates enabled, no intensity-weighted updates, no patch keeping
- `reconstruct()` branches into fast path (fused kernel) or slow path (original code)
- Cached `positions = self.reconstruction.positions` once per reconstruction (avoids recomputing the property every position)

### Why `jax.lax.dynamic_slice` instead of Python slices

Python slice objects (`array[..., sy, sx]`) are not JIT-traceable. `jax.lax.dynamic_slice(array, start_indices, slice_sizes)` and `dynamic_update_slice` are the JIT-compatible equivalents, enabling the entire position step to compile into a single XLA program.

---

## Files Modified (Performance Work)

| File                                       | Changes                                                                                                                                              |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PtyLabX/Params/Params.py`                 | Added `saveMemory = False`                                                                                                                           |
| `PtyLabX/Engines/BaseEngine.py`            | `detectorError`/`errorAtPos` → `jnp.zeros`; `_transferDataToGPU()`; debug log guard; `jnp` in probePowerCorrection; `jnp` in getErrorMetrics/getRMSD |
| `PtyLabX/Reconstruction/Reconstruction.py` | `jnp.array()` for object/probe init; `jnp.zeros` for momentum init                                                                                   |
| `PtyLabX/Engines/mPIE.py`                  | `jnp.abs` for probeWindow; fast path with fused kernel; cached positions; tqdm `mininterval=0.5`                                                     |
| `PtyLabX/Engines/_jit_kernels.py`          | Added `mpie_standard_step` fused kernel                                                                                                              |

---

## Phase 4: Flexible Fused Kernel via Closure (replacing monolithic `mpie_standard_step`)

### Problem

The Phase 3 fused kernel (`mpie_standard_step`) only worked for Fraunhofer + standard constraint. Users need to change `params.propagatorType`, `params.intensityConstraint`, and other settings while keeping the same performance.

### Solution: Closure-Based Dynamic Kernel Builder

Instead of one hardcoded kernel, `_build_fused_step()` in `mPIE.py` builds a specialized `@jax.jit` closure at reconstruction start. The closure captures config as Python values, so if-branches are resolved at JIT trace time — JAX compiles a specialized kernel per unique configuration with zero runtime branching overhead.

### `PtyLabX/Engines/mPIE.py` — Major rewrite

- **Removed** `_can_use_fast_path()` (restrictive: 11 conditions)
- **Added** `_can_use_fused_kernel()` (permissive: only 4 fallback triggers — `keepPatches`, `CPSCswitch`, `interferometric` constraint, `twoStepPolychrome`/`SAS` propagators)
- **Added** `_build_fused_step(probe_update_switch)`:
  - Captures propagator type, intensity constraint, all feature flags as Python values
  - Pre-fetches propagator transfer functions from `@lru_cache`'d factories in `Operators.py`
  - Returns `@jax.jit` closure that handles the entire position step
  - Supports: all propagators (Fraunhofer, Fresnel, ASP, polychromeASP, scaledASP, scaledPolychromeASP, identity), all non-interferometric constraints (standard, fluctuation, exponential, poisson, proxpoisson), Fourier mask, background mode, adaptive denoising, TV regularization (via `jax.lax.cond`), intensity-weighted probe updates, position correction (returns original objectPatch)
- **Updated** `reconstruct()` to use `_build_fused_step()` with all features integrated

### `PtyLabX/Engines/_jit_kernels.py`

- **Removed** `mpie_standard_step` (replaced by closure in `mPIE.py`)
- Removed unused `fft2c`/`ifft2c` imports
- All other shared kernels unchanged

### `PtyLabX/Operators/Operators.py` + `_propagation_kernels.py` + `off_axis_sas.py`

- **Renamed** all double-underscore cache functions to single-underscore for importability:
  - `__make_quad_phase` → `_make_quad_phase`
  - `__aspw_transfer_function` → `_aspw_transfer_function`
  - `__make_transferfunction_ASP` → `_make_transferfunction_ASP`
  - `__make_transferfunction_polychrome_ASP` → `_make_transferfunction_polychrome_ASP`
  - `__make_transferfunction_scaledASP` → `_make_transferfunction_scaledASP`
  - `__make_transferfunction_scaledPolychromeASP` → `_make_transferfunction_scaledPolychromeASP`
  - `__make_cache_twoStepPolychrome` → `_make_cache_twoStepPolychrome`
  - `__make_transferfunction_sas` → `_make_transferfunction_sas`
- Updated all internal references and `clear_cache()`

---

## Performance Lessons Learned

1. **f-string evaluation is eager:** `logger.debug(f"... {jax_array.sum()}")` evaluates `.sum()` before the log level check. Always guard with `logger.isEnabledFor()`.
2. **numpy on JAX arrays = implicit sync:** Any `np.*` call on a JAX array forces GPU→CPU transfer. Use `jnp.*` for all computation.
3. **GPU kernel dispatch overhead matters for small arrays:** For 128×128 arrays, launching 20 separate GPU kernels per position costs more than the actual computation. Fusing into one JIT kernel eliminates this.
4. **`jax.lax.dynamic_slice`/`dynamic_update_slice`** are required for JIT-compatible array slicing with runtime indices (Python slices use concrete values that break tracing).
5. **Pre-transfer data to GPU once** rather than letting JAX implicitly transfer every access.
6. **Closures + JAX JIT = free config dispatch:** Python if-branches on closure-captured values are resolved at JIT trace time. Build specialized kernels dynamically instead of hardcoding all combinations.
