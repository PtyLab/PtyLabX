# Configuration Reference

This page documents all configurable parameters in PtyLabX, organized by component.

## Params

The `Params` object holds settings shared across engines. Create and configure it before passing to an engine:

```python
from PtyLabX import Params

params = Params()
params.propagatorType = "ASP"
params.positionOrder = "random"
```

### Propagation

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `propagatorType` | `"Fraunhofer"` | `"Fraunhofer"`, `"Fresnel"`, `"ASP"`, `"scaledASP"`, `"polychromeASP"`, `"scaledPolychromeASP"`, `"twoStepPolychrome"`, `"identity"`, `"sas"` | Wave propagation method between sample and detector |
| `intensityConstraint` | `"standard"` | `"standard"`, `"sigmoid"`, `"fluctuation"`, `"exponential"`, `"poisson"` | How measured intensity is enforced |
| `fftshiftSwitch` | `False` | `True` / `False` | Pre-apply FFT shifts (internal optimization) |
| `FourierMaskSwitch` | `False` | `True` / `False` | Apply a Fourier-space mask |

!!! tip "Choosing a propagator"
    Use `"Fraunhofer"` (far-field, simple FFT) for standard experiments where the Fraunhofer approximation holds. Use `"ASP"` (angular spectrum propagation) for near-field or when you need more accurate propagation. Use `"Fresnel"` for intermediate distances. The polychromatic variants are for multi-wavelength reconstruction.

### Position ordering

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `positionOrder` | `"random"` | `"random"`, `"sequential"`, `"NA"` | Order in which scan positions are visited each iteration |

### Update timing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `probeUpdateStart` | `1` | Iteration number at which probe updates begin |
| `objectUpdateStart` | `1` | Iteration number at which object updates begin |

??? note "Delaying probe updates"
    Setting `probeUpdateStart = 5` means the probe stays fixed for the first 4 iterations while the object converges. This can help when the initial probe estimate is reasonable but the object is unknown.

### Object constraints

| Parameter | Default | Description |
|-----------|---------|-------------|
| `objectSmoothenessSwitch` | `False` | Enforce object smoothness |
| `objectSmoothenessWidth` | `2` | Number of pixels over which the object is assumed smooth |
| `objectSmoothnessAleph` | `1e-2` | Relaxation constant for smoothness regularization |
| `absObjectSwitch` | `False` | Force the object to be amplitude-only (no phase) |
| `absObjectBeta` | `1e-2` | Relaxation parameter for amplitude-only constraint |
| `objectContrastSwitch` | `False` | Push object to zero outside the region of interest |

### Probe constraints

| Parameter | Default | Description |
|-----------|---------|-------------|
| `probeSmoothenessSwitch` | `False` | Enforce probe smoothness |
| `probeSmoothnessAleph` | `5e-2` | Relaxation parameter for probe smoothness |
| `probeSmoothenessWidth` | `3` | Smoothness width in pixels |
| `probeBoundary` | `False` | Apply a window function to the probe boundary |
| `absorbingProbeBoundary` | `False` | Zero the probe at the boundary |
| `absorbingProbeBoundaryAleph` | `5e-2` | Relaxation parameter for absorbing boundary |
| `probePowerCorrectionSwitch` | `False` | Normalize probe to measured power spectral density |
| `probeSpectralPowerCorrectionSwitch` | `False` | Normalize from `experimentalData.probeSpectralPower` |
| `modulusEnforcedProbeSwitch` | `False` | Enforce probe amplitude from empty beam measurement |
| `absProbeSwitch` | `False` | Force the probe to be amplitude-only |
| `absProbeBeta` | `1e-2` | Relaxation parameter for amplitude-only probe |
| `binaryProbeSwitch` | `False` | Enforce binary probe |
| `binaryProbeThreshold` | `0.1` | Threshold for binarization |
| `binaryProbeAleph` | `0.1` | Relaxation parameter for binary constraint |
| `comStabilizationSwitch` | `False` | Stabilize probe center of mass |

### TV regularization on object

| Parameter | Default | Description |
|-----------|---------|-------------|
| `objectTVregSwitch` | `False` | Enable total variation regularization on the object |
| `objectTVfreq` | `5` | Apply TV regularization every N iterations |
| `objectTVregStepSize` | `1e-3` | Step size for TV regularization |

### L2 regularization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `l2reg` | `False` | Enable L2 regularization |
| `l2reg_probe_aleph` | `0.01` | L2 regularization strength for probe |
| `l2reg_object_aleph` | `0.001` | L2 regularization strength for object |

### TV autofocus

TV autofocus optimizes the propagation distance by maximizing a sharpness metric (e.g. total variation) on the reconstructed object or probe.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TV_autofocus` | `False` | Enable TV autofocusing |
| `TV_autofocus_metric` | `"TV"` | Sharpness metric: `"TV"`, `"std"`, `"min_std"`, or a callable |
| `TV_autofocus_intensityonly` | `False` | Only use intensity (not complex field) for metric |
| `TV_autofocus_stepsize` | `5` | Propagation distance step size |
| `TV_autofocus_aleph` | `0.01` | Regularization constant |
| `TV_autofocus_roi` | `[0.4, 0.6]` | Region of interest (fraction of array size, or pixel coordinates) |
| `TV_autofocus_range_dof` | `11` | Search range in depths of focus |
| `TV_autofocus_nplanes` | `11` | Number of planes to examine |
| `TV_autofocus_friction` | `0.7` | Momentum friction for step algorithm |
| `TV_autofocus_what` | `"object"` | What to focus: `"object"` or `"probe"` |
| `TV_autofocus_run_every` | `3` | Run autofocus every N iterations |
| `TV_autofocus_min_z` | `None` | Minimum propagation distance (meters), `None` for no limit |
| `TV_autofocus_max_z` | `None` | Maximum propagation distance (meters), `None` for no limit |

??? example "Autofocus usage"
    ```python
    params.TV_autofocus = True
    params.TV_autofocus_stepsize = 50
    params.TV_autofocus_metric = "TV"
    params.TV_autofocus_what = "object"
    params.TV_autofocus_roi = [[0.3, 0.7], [0.3, 0.7]]
    params.TV_autofocus_min_z = experimentalData.zo - 2e-2
    params.TV_autofocus_max_z = experimentalData.zo + 5e-2
    ```

### Position correction

| Parameter | Default | Description |
|-----------|---------|-------------|
| `positionCorrectionSwitch` | `False` | Enable position correction during reconstruction |
| `positionCorrectionSwitch_radius` | `1` | Search radius for position correction (pixels) |

!!! note
    Position correction is built into `pcPIE`. For other engines, set `positionCorrectionSwitch = True` in params.

### Probe orthogonalization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `orthogonalizationSwitch` | `False` | Enable probe mode orthogonalization |
| `orthogonalizationFrequency` | `10` | Orthogonalize every N iterations |

### OPR (Orthogonal Probe Relaxation) parameters

These parameters are used with the `OPR` engine:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPR_modes` | `[0]` | Indices of incoherent probe modes linked in a subspace |
| `OPR_subspace` | `4` | Size of the SVD subspace |
| `OPR_alpha` | `0.05` | Feedback parameter (higher = more probe variation allowed) |
| `OPR_tsvd_type` | `"numpy"` | SVD method: `"numpy"` or `"randomized"` |
| `OPR_orthogonalize_modes` | `True` | Orthogonalize all incoherent probe modes |
| `OPR_neighbor_constraint` | `False` | Only allow slowly changing probes |

### Other switches

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backgroundModeSwitch` | `False` | Estimate background intensity |
| `adaptiveDenoisingSwitch` | `False` | Clip estimated noise floor from raw data |
| `couplingSwitch` | `False` | Couple adjacent wavelengths |
| `couplingAleph` | `0.5` | Coupling relaxation parameter |
| `saveMemory` | `False` | Skip precomputing arrays to reduce memory |
| `CPSCswitch` | `False` | Coherent power spectral correction |
| `PSDestimationSwitch` | `False` | Power spectral density estimation |

---

## Reconstruction

The `Reconstruction` object holds the mutable state of the reconstruction. Key settings:

### Mode and slice configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `npsm` | `1` | Number of incoherent probe modes (partial coherence) |
| `nosm` | `1` | Number of incoherent object modes |
| `nlambda` | `1` | Number of wavelengths |
| `nslice` | `1` | Number of object slices (for multislice) |

### Initialization

| Parameter | Options | Description |
|-----------|---------|-------------|
| `initialObject` | `"ones"`, `"circ"`, `"upsampled"`, `"recon"` | How the object array is initialized |
| `initialProbe` | `"circ"`, `"gaussian"`, `"recon"` | How the probe array is initialized |

- `"ones"` — uniform unit amplitude
- `"circ"` — circular aperture
- `"gaussian"` — Gaussian profile
- `"upsampled"` — upsampled from a previous reconstruction
- `"recon"` — loaded from a previous reconstruction file

### Object size

```python
reconstruction.No = int(reconstruction.No * 0.9)  # reduce object array size
```

### Useful methods

| Method | Description |
|--------|-------------|
| `initializeObjectProbe()` | Initialize object and probe arrays |
| `describe_reconstruction()` | Print reconstruction parameters |
| `saveResults(filename)` | Save object, probe, and positions to HDF5 |
| `load_probe(filename)` | Load probe from a previous result |
| `load_object(filename)` | Load object from a previous result |

---

## Monitor

The `Monitor` controls real-time visualization during reconstruction:

```python
from PtyLabX import Monitor

monitor = Monitor()
monitor.figureUpdateFrequency = 5
monitor.objectPlot = "complex"
```

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `figureUpdateFrequency` | `1` | any integer | Update plot every N iterations |
| `objectPlot` | `"complex"` | `"complex"`, `"abs"`, `"angle"` | How the object is displayed |
| `verboseLevel` | `"low"` | `"low"`, `"high"` | `"low"`: one figure; `"high"`: two figures (adds diffraction data plot) |
| `objectZoom` | `1` | any float | Zoom factor for object plot field of view |
| `probeZoom` | `1` | any float | Zoom factor for probe plot field of view |
| `objectPlotContrast` | `1` | any float | Contrast adjustment for object plot |
| `probePlotContrast` | `1` | any float | Contrast adjustment for probe plot |
| `screenshot_directory` | `None` | path or `None` | Save screenshots to this directory |
| `downsample_everything` | — | any integer | Downsampling factor for all plots |
| `probe_downsampling` | — | any integer | Separate downsampling for probe plot |

!!! tip "Jupyter notebooks"
    In Jupyter, set `figureUpdateFrequency >= 5` for better performance. The monitor automatically detects inline backends and warns if the frequency is too low.

### DummyMonitor

For headless / batch runs, use `dummyMonitor=True` in `easyInitialize()`, or instantiate `AbstractMonitor()` directly:

```python
from PtyLabX.Monitor.Monitor import AbstractMonitor
monitor = AbstractMonitor()
```
