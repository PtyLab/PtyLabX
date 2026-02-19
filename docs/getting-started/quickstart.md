# Quick Start

## Using `easyInitialize`

The fastest way to run a reconstruction is with `easyInitialize()`, which wires up all components from an HDF5 file:

```python
import PtyLabX
from PtyLabX import Engines

experimentalData, reconstruction, params, monitor, engine = PtyLabX.easyInitialize(
    "path/to/data.hdf5",
    engine=Engines.mPIE,
    operationMode="CPM",
)

engine.numIterations = 50
engine.reconstruct()

reconstruction.saveResults("result.hdf5")
```

`easyInitialize` returns a 5-tuple:

| Object | Description |
|--------|-------------|
| `ExperimentalData` | Diffraction data loaded from the HDF5 file |
| `Reconstruction` | Mutable state: object array, probe array, scan positions |
| `Params` | Shared configuration (propagator type, constraints, switches) |
| `Monitor` | Real-time visualization during reconstruction |
| `Engine` | The reconstruction algorithm instance (e.g. `mPIE`) |

## Headless mode

For batch processing or server environments without a display:

```python
experimentalData, reconstruction, params, monitor, engine = PtyLabX.easyInitialize(
    "data.hdf5",
    engine=Engines.mPIE,
    dummyMonitor=True,
)
```

## Manual initialization

When you need more control (e.g. multiple probe modes, custom initial probe), skip `easyInitialize` and set up each component:

```python
from PtyLabX import ExperimentalData, Reconstruction, Params, Monitor, Engines
import numpy as np

# Load data
data = ExperimentalData("data.hdf5", operationMode="CPM")

# Create configuration and monitor
params = Params()
params.propagatorType = "ASP"
params.positionOrder = "random"

monitor = Monitor()
monitor.figureUpdateFrequency = 5
monitor.objectPlot = "complex"

# Set up reconstruction
recon = Reconstruction(data, params)
recon.npsm = 2   # two probe modes (for partial coherence)
recon.nosm = 1   # one object mode
recon.nlambda = 1
recon.nslice = 1
recon.initialProbe = "circ"
recon.initialObject = "ones"
recon.initializeObjectProbe()

# Customize initial probe with quadratic phase
recon.probe = recon.probe * np.exp(
    1.0j * 2 * np.pi / recon.wavelength
    * (recon.Xp**2 + recon.Yp**2) / (2 * 6e-3)
)

# Run reconstruction
engine = Engines.mPIE(recon, data, params, monitor)
engine.numIterations = 50
engine.betaObject = 0.25
engine.betaProbe = 0.25
engine.reconstruct()

# Save results
recon.saveResults("result.hdf5")
```

## Saving and loading

```python
# Save full reconstruction state
reconstruction.saveResults("result.hdf5")

# Reload probe or object from a previous run
reconstruction.load_probe("previous_result.hdf5")
reconstruction.load_object("previous_result.hdf5")
```

## Next steps

- [CPM Workflow Overview](../cpm/overview.md) — understand the reconstruction pipeline
- [Engines](../cpm/engines.md) — choose the right reconstruction algorithm
- [Configuration Reference](../cpm/configuration.md) — all available parameters
- [Tutorial Notebook](../tutorials/demo.ipynb) — simulate and reconstruct a CPM experiment step by step
