# Reconstruction Engines

PtyLabX provides 14 reconstruction engines. All share the same interface: create an instance, set parameters, and call `reconstruct()`.

## Engine overview

| Engine | Description | Momentum | Special feature |
|--------|-------------|:--------:|-----------------|
| `ePIE` | Extended PIE | | Baseline algorithm |
| `mPIE` | Momentum PIE | Yes | Accelerated convergence |
| `pcPIE` | Position-correcting PIE | Yes | Corrects scan position errors |
| `zPIE` | z-updating PIE | | Optimizes propagation distance |
| `e3PIE` | Multislice PIE | | Thick sample reconstruction |
| `ePIE_TV` | ePIE + Total Variation | | TV regularization on object |
| `mPIE_tv` | mPIE + Total Variation | Yes | Momentum + TV regularization |
| `qNewton` | Quasi-Newton | | Second-order update rule |
| `mqNewton` | Momentum quasi-Newton | Yes | Second-order + momentum |
| `OPR` | Orthogonal Probe Relaxation | | Position-varying illumination |
| `aPIE` | Angle-correcting PIE | | Illumination angle correction |
| `multiPIE` | Multi-mode PIE | Yes | Multi-mode momentum updates |
| `ePIE_mw` | ePIE multi-wavelength | | Polychromatic reconstruction |
| `mPIE_mw` | mPIE multi-wavelength | Yes | Polychromatic + momentum |

## Choosing an engine

- **Start with `ePIE`** for debugging or as a baseline — simplest algorithm, easy to understand
- **Use `mPIE` for routine reconstructions** (recommended default) — momentum acceleration gives faster convergence
- **Use `pcPIE`** when scan positions may be inaccurate — enables position correction during reconstruction
- **Use `zPIE`** when the sample-to-detector distance (`zo`) is uncertain — optimizes the propagation distance
- **Use `qNewton` / `mqNewton`** for potentially better convergence at higher compute cost — second-order update
- **Use `ePIE_TV` / `mPIE_tv`** when the object benefits from total variation regularization (e.g. piecewise-constant samples)
- **Use `e3PIE`** for thick samples — multislice decomposition (set `reconstruction.nslice > 1`)
- **Use `ePIE_mw` / `mPIE_mw`** for polychromatic data — set `reconstruction.nlambda > 1` and provide `spectralDensity` in data
- **Use `OPR`** when the illumination varies across scan positions — orthogonal probe relaxation
- **Use `aPIE`** when the illumination angle is uncertain — angle correction via Luus-Jaakola optimization

## Common parameters (all engines)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `numIterations` | `50` | Number of reconstruction iterations |
| `betaObject` | `0.25` | Object update step size |
| `betaProbe` | `0.25` | Probe update step size |

## Momentum engine parameters

Engines with momentum (`mPIE`, `mPIE_tv`, `mqNewton`, `mPIE_mw`) use:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `feedbackM` | `0.3` | Momentum feedback strength |
| `frictionM` | `0.7` | Momentum friction (damping) |
| `alphaProbe` | `0.1` | Probe regularization constant |
| `alphaObject` | `0.1` | Object regularization constant |

`pcPIE` and `multiPIE` use slightly different naming:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `betaM` | `0.3` | Momentum feedback |
| `stepM` | `0.7` | Momentum friction |

## Quasi-Newton parameters

`qNewton` and `mqNewton` use different defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `betaObject` | `1.0` | Object update step size |
| `betaProbe` | `1.0` | Probe update step size |
| `regObject` | `1.0` | Object regularization constant |
| `regProbe` | `1.0` | Probe regularization constant |

`mqNewton` additionally has:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `betaObject_m` | `0.25` | Momentum step size for object |
| `betaProbe_m` | `0.25` | Momentum step size for probe |

## Usage

```python
from PtyLabX import Engines

engine = Engines.mPIE(reconstruction, experimentalData, params, monitor)
engine.numIterations = 100
engine.betaObject = 0.5
engine.betaProbe = 0.25
engine.reconstruct()
```

## Chaining engines

You can run one engine and then switch to another. The reconstruction state (object, probe) carries over:

```python
# Start with ePIE for initial convergence
epie = Engines.ePIE(reconstruction, experimentalData, params, monitor)
epie.numIterations = 20
epie.reconstruct()

# Switch to mPIE for faster convergence
mpie = Engines.mPIE(reconstruction, experimentalData, params, monitor)
mpie.numIterations = 50
mpie.reconstruct()
```

You can also toggle parameters between runs:

```python
engine = Engines.mPIE(reconstruction, experimentalData, params, monitor)
engine.numIterations = 50

for i in range(4):
    params.TV_autofocus = (i % 2 == 0)  # alternate autofocus on/off
    engine.reconstruct()
```
