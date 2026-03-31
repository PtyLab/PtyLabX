# PtyLabX Ideas

Ideas inspired by [Chromatix](https://github.com/chromatix-team/chromatix) (differentiable wave optics), [Phaser](https://github.com/hexane360/phaser) (JAX electron ptychography), and [ptyRAD](https://github.com/chiahao3/ptyrad) (PyTorch AD-based ptychography, [paper](https://arxiv.org/abs/2505.07814)).

**Guiding principle**: Minimal API changes. The existing `BaseEngine` / `Reconstruction` / `Params` architecture stays as-is. A `GradientEngine` is a new `BaseEngine` subclass — it reuses `_prepareReconstruction()`, `applyConstraints()`, `getErrorMetrics()`, propagation operators, and monitor integration out of the box. No changes needed to `Reconstruction`, `Params`, `ExperimentalData`, `Monitor`, or any existing engine.

---

## Framework Ideas

These are directly needed to make automatic-differentiation-based reconstruction work within the current architecture.

---

### 1. Gradient-Descent Reconstruction Engine

**Source**: Phaser, ptyRAD

The core idea: define ptychographic reconstruction as minimizing a differentiable loss function, then let `jax.value_and_grad()` compute all updates automatically instead of hand-crafting update rules per engine.

**What this looks like concretely:**
- A `GradientEngine(BaseEngine)` that defines a **pure forward model** as a function: `(object_patch, probe, propagator_params) -> I_estimated`
- `jax.value_and_grad` computes object/probe gradients through the entire forward model
- Optax optimizers (Adam, SGD with momentum) replace hand-tuned step sizes
- New algorithms become: new loss function + existing optimizer, not 200+ lines of manual update code

**Key detail from Phaser**: They use **Wirtinger derivatives** for complex-valued gradients. JAX handles this natively — `jax.grad` on a real-valued loss of complex parameters gives the correct conjugate gradient. The update is simply `grad = -conj(grad)` for steepest descent.

**Alignment with PtyLabX**: The existing `intensityProjection()` already computes the forward model; the gap is extracting the forward pass into a pure function (no side effects on `self.reconstruction`) so `jax.grad` can trace through it. The `reconstruct()` loop structure stays identical to ePIE — iterate over positions, compute forward, update — just with AD-computed gradients instead of hand-derived updates.

**New files**: `PtyLabX/Engines/GradientEngine.py` (~150 lines), loss functions in `PtyLabX/Engines/_loss_functions.py`, registration in `Engines/__init__.py`.
**Modified files**: None.

---

### 2. Composable Loss Functions & Noise Models

**Source**: Phaser, ptyRAD

Currently PtyLabX hardcodes intensity projection modes (`standard`, `poisson`, `exponential`, etc.) inside `intensityProjection()`. These are effectively different noise model assumptions but aren't exposed as swappable components.

**Idea**: Factor these into explicit loss/data-fidelity functions:
- **Gaussian/Amplitude**: `L = ||sqrt(I_meas) - sqrt(I_est)||^2` (current `standard` mode)
- **Poisson/ML**: `L = sum(I_est - I_meas * log(I_est))` (negative log-likelihood)
- **Amplitude**: `L = ||sqrt(I_meas) - |psi|||^2` (Phaser's approach)
- **Anscombe**: Variance-stabilizing transform for mixed noise (from Phaser)
- **PACBED** (from ptyRAD): Compare position-averaged diffraction patterns for global self-consistency — `L = MSE(mean_j(I_est_j), mean_j(I_meas_j))`. Particularly valuable for mixed-state reconstructions to prevent mode proliferation.

Each is a simple function `(I_measured, I_estimated) -> scalar_loss` that plugs into the gradient engine. The conventional engines keep using `intensityProjection()` unchanged.

**New file**: `PtyLabX/Engines/_loss_functions.py`.
**Modified files**: None (conventional engines untouched).

---

### 3. Optax-Based Optimizer Integration

**Source**: Phaser, ptyRAD, Chromatix

Replace fixed step-size parameters (`betaObject`, `betaProbe`, `feedbackM`) with proper optimizers inside the gradient engine:

- **Adam**: Adaptive learning rates, handles ill-conditioning naturally
- **SGD + Momentum**: Direct replacement for current momentum engines (mPIE), but with well-studied convergence theory
- **Learning rate schedules**: Warm-up, cosine decay, step decay — via Optax's `optax.schedule` API
- **Per-parameter learning rates**: Different rates for object vs. probe vs. positions via `optax.multi_transform`
- **PolyakSGD** (from Phaser): Automatic step-size based on function value improvement — no learning rate tuning needed

This would subsume much of the complexity in `mPIE` (momentum), `mqNewton` (quasi-Newton), and the various `beta` parameters across engines. `optax` is already a dependency (used in `BaseEngine.position_update_to_change_in_z`).

**New code**: Inside `GradientEngine.__init__()` and `reconstruct()`.
**Modified files**: None.

---

### 4. Fourier Phase-Shift for Differentiable Position Correction

**Source**: ptyRAD's `imshift_batch()`, Phaser's position refinement

The current `pcPIE` cross-correlation approach searches on a discrete pixel grid. For the gradient engine, positions need to be differentiable. ptyRAD's approach: implement position shifts as **Fourier phase ramps**.

**The mechanism** (Fourier shift theorem):
1. FFT the object patch (or probe) to frequency domain
2. Multiply by `exp(-2πi(Δx·kx + Δy·ky))` — a phase ramp parameterized by the shift
3. IFFT back to real space

**Why this is better**:
- Arbitrary sub-pixel accuracy, no interpolation artifacts
- Differentiable: `jax.grad` flows backward through the shift — positions become first-class optimizable parameters jointly optimized with object/probe
- Vectorized: batch all N scan position shifts simultaneously with `vmap`
- Systematic drift removal: `pos_update -= mean(pos_update)` separates true corrections from global drift

**Bonus**: This naturally extends to **z-position** (defocus) correction — the propagation distance `zo` becomes a differentiable parameter, replacing the current TV-autofocus heuristic with principled gradient-based optimization.

The existing `fraccircshift()` in `utils.py` uses linear interpolation for sub-pixel shifts — the Fourier approach is a more accurate alternative that could live alongside it.

**New code**: JIT-compiled `fourier_shift()` function in `_jit_kernels.py` or `utils.py`, used by `GradientEngine`.
**Modified files**: None (pcPIE keeps its cross-correlation approach).

---

### 5. Staged Optimization with Per-Parameter Schedules

**Source**: ptyRAD

ptyRAD assigns each optimizable parameter its own activation window:
- `object_start_iter`, `object_end_iter`
- `probe_start_iter`, `probe_end_iter`
- `positions_start_iter`, `positions_end_iter`

**Why this matters in practice**:
- Start with only the object -> stable initialization
- Add probe refinement at iteration 50 (once the object is reasonable)
- Enable position correction only after iteration 100 (avoid chasing noise early on)
- Freeze the probe at the end to let the object converge cleanly

This replaces the current binary `probeSwitch` / `positionCorrectionSwitch` on/off pattern with a time-ordered curriculum. Aligns naturally with Optax's `optax.masked` for per-parameter gradient masking — at each iteration, mask out gradients for parameters outside their active window.

**New code**: Schedule parameters in `GradientEngine`, mask logic in `reconstruct()` loop.
**Modified files**: None (existing engines keep their switches).

---

### 6. Smooth (Sigmoid) Aperture Masks

**Source**: ptyRAD's `make_sigmoid_mask()`

Currently PtyLabX uses binary `circ()` for probe support and k-space masking — a hard step function. For AD mode, this creates zero gradients at the boundary where the optimizer needs to act.

ptyRAD uses a **sigmoid-based smooth mask**:

```python
mask(r) = sigmoid((R - r) / width)
```

where `width` controls the transition sharpness.

**Why this is needed in AD mode**:
- Hard thresholds create zero gradients at the boundary — gradient descent stalls
- Soft masks have smooth gradients everywhere — optimizer can act near the aperture edge
- Physically more realistic: real apertures have diffraction fringes, not perfectly sharp edges
- Tunable sharpness: `width -> 0` recovers binary `circ()` as a limit

A `smooth_circ()` utility alongside the existing `circ()` — the gradient engine uses it, conventional engines keep using `circ()`.

**New code**: `smooth_circ()` in `utils.py` (~5 lines).
**Modified files**: None.

---

### 7. Convergence Monitoring & Early Stopping

**Source**: Phaser's `PatienceObserver`

Current monitoring tracks error history but doesn't act on it:

- **Exponential moving average** of error for smooth convergence curves
- **Patience-based early stopping**: Stop if no improvement for N iterations (configurable patience)
- **Adaptive iteration count**: Run until convergence rather than fixed `numIterations`

This saves compute and prevents over-fitting to noise in later iterations. Particularly important for gradient descent where over-optimization is a real risk.

**New code**: Small addition to `BaseEngine` or `GradientEngine` — check error EMA in the loop, break early if patience exceeded.
**Modified files**: Minimal — optionally add a `patience` parameter to `Params`, or keep it engine-specific.

---

### 8. Mini-Batch Reconstruction with vmap & Gradient Accumulation

**Source**: Phaser, ptyRAD, Chromatix

Currently engines loop over scan positions sequentially. For the gradient engine:

- **vmap over positions**: Vectorize the forward model over a batch of positions — major GPU speedup
- **Stochastic gradient descent**: Random mini-batches of positions, like training a neural network
- **Gradient accumulation** (ptyRAD): When GPU memory limits batch size, accumulate gradients over sub-batches via `jax.lax.scan` before applying one optimizer step

**Trade-off**: Larger batches = faster iterations (GPU utilization) but more averaging (slower convergence per iteration). Phaser reports **6x speedup** from JIT-fusing batched operations. ptyRAD reports up to **24x** vs. traditional implementations.

**New code**: Batched forward model in `GradientEngine`, configurable `batch_size` parameter.
**Modified files**: None.

---

### 9. Hybrid Engine Pipelines

**Source**: Phaser's multi-engine `execute_plan()`

Run multiple engines sequentially in a single reconstruction:

- **Warm-up with ePIE** (fast, robust convergence to rough solution) -> **Fine-tune with gradient descent** (precise, handles complex regularization)
- **Transfer state**: Object/probe from engine 1 initializes engine 2 (already works — both read/write to the same `Reconstruction` object)
- **Per-engine configuration**: Different learning rates, regularizers, constraints per stage

This combines the robustness of conventional algorithms with the flexibility of AD-based optimization. Already partially possible by calling `engine1.reconstruct()` then `engine2.reconstruct()` in user scripts — could be formalized with a helper function.

**New code**: Optional `pipeline()` helper or just documented pattern in examples.
**Modified files**: None.

---

### 10. Modular Regularization as Loss Terms

**Source**: Phaser's `CostRegularizer` hooks, Chromatix, ptyRAD

In gradient-descent mode, regularizers are just additive loss terms with automatic gradients — no need to manually derive proximal operators:

```
total_loss = data_fidelity(I_meas, I_est) + lambda_tv * TV(object) + lambda_l1 * L1(object_phase)
```

**Regularizers to implement** (from all three libraries):
- **Total Variation** (already exists as `grad_TV` in `Regularizers/`, just needs loss-function form)
- **L1 sparsity** on object phase/amplitude (ptyRAD)
- **Tikhonov** (L2 smoothness) on object and probe (Phaser)
- **Similarity regularization** for mixed states (ptyRAD) — penalizes variance across modes: `L_sim = weight * std(modes)^2`

The existing `applyConstraints()` still runs post-step for hard constraints (orthogonalization, power normalization). Soft penalties move into the loss.

**New code**: Regularizer functions in `_loss_functions.py`.
**Modified files**: None.

---

### 11. Iteration-Conditioned Constraint Scheduling

**Source**: ptyRAD's `IterConstraint`

Currently `applyConstraints()` applies all active constraints every iteration. Parameterize each constraint with:
- `start_iter`: don't apply before this iteration
- `step`: apply every N iterations (not necessarily every one)
- `end_iter`: stop applying after this iteration

**Examples**:
- SVD probe orthogonalization: every 5 iterations (expensive, not needed every step)
- Gaussian smoothing of object: only in early iterations 0-50
- Amplitude clamping: every iteration but only after 20

**Modified files**: Small addition to `applyConstraints()` in `BaseEngine.py` — wrap each constraint block in an iteration check. Benefits all engines, not just gradient descent.

---

## Future Research Ideas

Interesting ideas that require larger architectural changes, are research explorations, or are nice-to-haves beyond the core AD framework.

---

### Mixed-State Occupancy Factors

**Source**: ptyRAD

Make mode weights explicit learnable parameters: `I_total = sum_m occupancy[m] * |FFT(probe_m * obj)|^2` where `sum(occupancy) = 1`. Occupancy values at convergence reveal source coherence. Optimized jointly with probe/object via gradient descent.

---

### Differentiable Probe Parameterization

**Source**: Chromatix, Phaser, ptyRAD

Instead of optimizing probe pixels directly, parameterize the probe:
- **Zernike coefficients**: `probe = sum(c_i * Z_i(r, theta))` — optimize ~20 coefficients instead of N^2 pixels
- **Aperture + aberrations**: `probe = circ(r) * exp(i * sum(c_i * Z_i))` — physical model
- **STEM aberration model** (ptyRAD): C1 defocus, C3 spherical, C5, astigmatism, coma
- **Hermite-Gaussian modes** (Phaser): Natural basis for laser/electron probes
- **Hybrid**: Start with parameterized probe, switch to pixel-wise refinement

`zernikeAberrations()` already exists in `utils.py` — it just needs to be used as a differentiable probe generator rather than just a one-time initialization tool.

---

### Multislice Improvements

**Source**: Phaser, ptyRAD

Current `e3PIE` implements basic multislice. Enhancements:
- **Bandwidth limiting between slices**: Prevent aliasing in inter-slice Fresnel propagation (Phaser)
- **Gaussian depth blur in real space** (ptyRAD): Avoids periodic boundary/wrap-around artifacts from k-space filtering
- **Differentiable slice thickness Δz**: Treat Δz as a learnable scalar (ptyRAD)
- **Position-dependent local tilts in propagator**: Each position j gets tilt θⱼ = (θⱼ,ₓ, θⱼ,ᵧ) in the inter-slice Fresnel propagator — crucial for twisted 2D materials, bent crystals (not too relevant for us right now)
- **Axial Fourier filter (kz)**: Arctangent-based depth frequency cutoff

---

### Equinox Module-Based Reconstruction State

**Source**: Chromatix

Make reconstruction state a proper JAX pytree via `eqx.Module`:

```python
class ReconState(eqx.Module):
    object: jnp.ndarray      # differentiable
    probe: jnp.ndarray       # differentiable
    positions: jnp.ndarray   # differentiable (if position correction on)
    wavelength: float = eqx.field(static=True)  # not differentiable
```

Benefits: `jax.grad` partitions cleanly, `jax.vmap` over state enables multi-start reconstruction, `eqx.tree_serialise_leaves` for checkpointing. But this is a **large refactor** of the `Reconstruction` class — defer until the simpler gradient engine proves its value.

---

### Field Abstraction with Metadata

**Source**: Chromatix's `Field` class

Wrap wave fields in a lightweight dataclass carrying pixel size, wavelength, propagation distance. Propagation functions auto-compute kernels from metadata. Cleaner API but significant refactor of how arrays flow through operators.

---

### Sensor/Detector Modeling

**Source**: Chromatix's `BasicSensor`

Include detector effects in the differentiable forward model: shot noise (Poisson), readout noise, quantization, detector PSF/MTF, pixel binning. When the forward model includes these, gradient-based reconstruction naturally accounts for them.

---

### Plug-and-Play Priors via Neural Networks

**Source**: Chromatix

Replace hand-crafted regularizers with learned denoisers:
- **PnP-ADMM/PnP-HQS**: Alternate between gradient step and denoiser step
- **Deep Image Prior**: Object as neural network output
- **Implicit Neural Representations**: Object as coordinate MLP `f(x,y) -> complex`

Requires the gradient-descent framework (ideas 1-3) as a prerequisite.

---

### Hyperparameter Tuning with Optuna

**Source**: ptyRAD

Wrap `engine.reconstruct()` as an Optuna objective, use TPE sampler + Hyperband pruner. Tunable: learning rates, probe modes, aberration coefficients, slice count/thickness. Even without full Optuna integration, separating hyperparameters cleanly from the loop helps manual sweeps.

---

## Summary

| Priority | Framework Idea | Effort | API Changes |
|----------|---------------|--------|-------------|
| 1 | Gradient-Descent Engine | Medium | None (new engine file) |
| 2 | Composable Loss Functions | Low | None (new file) |
| 3 | Optax Optimizers | Low | None (inside new engine) |
| 4 | Fourier Phase-Shift Positions | Low-Med | None (new JIT kernel) |
| 5 | Staged Optimization Schedules | Low | None (inside new engine) |
| 6 | Smooth Aperture Masks | Low | None (new utility) |
| 7 | Mini-Batch + Gradient Accumulation | Medium | None (inside new engine) |
| 8 | Modular Regularization as Loss | Low-Med | None (new file) |
| 9 | Convergence Monitoring / Early Stop | Low | Minimal (optional Params field) |
| 10 | Hybrid Engine Pipelines | Low | None (user-level pattern) |
| 11 | Constraint Scheduling | Low | Small (`applyConstraints` in BaseEngine) |
