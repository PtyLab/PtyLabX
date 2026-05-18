from PtyLabX.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLabX.Reconstruction.Reconstruction import Reconstruction
from PtyLabX.Reconstruction.CalibrationFPM import IlluminationCalibration
from PtyLabX.Monitor.Monitor import Monitor, DummyMonitor, AbstractMonitor
from PtyLabX.Params.Params import Params
from PtyLabX import Engines
from pathlib import Path


def easyInitialize(
    filename: Path,
    engine: type[Engines.BaseEngine] = Engines.ePIE,
    operationMode: str = "CPM",
    dummyMonitor: bool = False,
) -> tuple[ExperimentalData, Reconstruction, Params, AbstractMonitor, Engines.BaseEngine]:
    """Do a 'standard' initialization, and return the items you need with some sensible defaults."""
    if operationMode == "CPM":
        return _easyInitializeCPM(filename, engine, operationMode, dummyMonitor)
    if operationMode == "FPM":
        return _easyInitializeFPM(filename, engine, operationMode, dummyMonitor)
    else:
        raise NotImplementedError()


def _easyInitializeCPM(
    filename: Path, engine_function: type[Engines.BaseEngine], operationMode: str, dummy_monitor: bool = False
) -> tuple[ExperimentalData, Reconstruction, Params, AbstractMonitor, Engines.BaseEngine]:
    experimentalData = ExperimentalData(filename, operationMode)
    params = Params()
    if dummy_monitor:
        monitor: AbstractMonitor = DummyMonitor()
    else:
        monitor: AbstractMonitor = Monitor()
    reconstruction = Reconstruction(experimentalData, params)

    reconstruction.initializeObjectProbe()

    engine = engine_function(reconstruction, experimentalData, params, monitor)
    return experimentalData, reconstruction, params, monitor, engine


def _easyInitializeFPM(
    filename: Path, engine_function: type[Engines.BaseEngine], operationMode: str, dummy_monitor: bool = False
):
    experimentalData = ExperimentalData(filename, operationMode)
    if dummy_monitor:
        monitor: AbstractMonitor = DummyMonitor()
    else:
        monitor: AbstractMonitor = Monitor()

    params = Params()
    reconstruction = Reconstruction(experimentalData, params)
    reconstruction.initializeObjectProbe()
    calib = IlluminationCalibration(reconstruction, experimentalData)

    engine = engine_function(reconstruction, experimentalData, params, monitor)
    params.positionOrder = "NA"
    params.probeBoundary = True
    return experimentalData, reconstruction, params, monitor, engine, calib
