# This file contains utilities required for Monitor
import math
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable


def hsv2rgb(hsv: np.ndarray) -> np.ndarray:
    """
    Convert a 3D hsv np.ndarray to rgb (5 times faster than colorsys).
    https://stackoverflow.com/questions/27041559/rgb-to-hsv-python-change-hue-continuously
    h,s should be a numpy arrays with values between 0.0 and 1.0
    v should be a numpy array with values between 0.0 and 255.0
    :param hsv: np.ndarray of shape (x,y,3)
    :return: hsv2rgb returns an array of uints between 0 and 255.
    """
    hsv = np.asarray(hsv)
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype("uint8")
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5, i == i]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v, v])  # , default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p, t])  # , default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q, p])  # , default=p)
    return rgb.astype("uint8")


def complex2rgb(u: np.ndarray, amplitudeScalingFactor: float | str = 1, center_phase: bool = False) -> np.ndarray:
    """
    Preparation function for a complex plot, converting a 2D complex array into an rgb array
    :param u: a 2D complex array
    :return: an rgb array for complex plot
    """
    # hue (normalize angle)
    u = np.asarray(u)
    if center_phase:
        N = u.shape[-1]
        phexp = np.sum(u[..., N // 3 : 2 * N // 3, N // 3 : 2 * N // 3], axis=(-2, -1))
        u = u * phexp.conj() / (abs(phexp) + 1e-9)
    h = np.angle(u)
    h = (h + np.pi) / (2 * np.pi)
    # saturation  (ones)
    s = np.ones_like(h)
    # value (normalize brightness to 8-bit)
    v = np.abs(u)
    if amplitudeScalingFactor == "2sigma":
        ASF = v.mean() + 2 * np.std(v)
        ASF = ASF / v.max()
    elif amplitudeScalingFactor is None:
        ASF = 1.0 / v.max()
        amplitudeScalingFactor = ASF
    else:
        ASF = amplitudeScalingFactor

    if ASF != 1 and amplitudeScalingFactor != "2sigma":
        v[v > amplitudeScalingFactor * np.max(v)] = amplitudeScalingFactor * np.max(v)
    v = v / (np.max(v) + np.finfo(float).eps) * (2**8 - 1)

    hsv = np.dstack([h, s, v])
    rgb = hsv2rgb(hsv)
    return rgb


def complex2rgb_vectorized(probe: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Turn complex image into rgb for every line.

    The individual images are all autoscaled, so you cannot compare them.
    """
    probe = np.asarray(probe)
    original_shape = probe.shape
    probe = probe.reshape(-1, *probe.shape[-2:])
    probe_rgb = np.array([complex2rgb(p, **kwargs) for p in probe])
    probe_rgb = probe_rgb.reshape(original_shape + (3,))
    return probe_rgb


def complexPlot(rgb: np.ndarray, ax: Axes | None = None, pixelSize: float = 1, axisUnit: str = "pixel") -> AxesImage:
    """
    Plot a 2D complex plot (hue for phase, brightness for amplitude). Input array need to be prepared by using
    the complex2rgb function.
    :param rgb: a rgb array that is converted from a 2D complex np.ndarray by using complex2rgb
    :param ax: Optional axis to plot in
    :param pixelSize: pixelSize in x and y, to display the physical dimension of the plot
    :param str axisUnit: Options: default 'pixel', 'm', 'cm', 'mm', 'um'
    :return: An hsv plot
    """

    if not ax:
        fig, ax = plt.subplots()
    unitRatio = {"pixel": 1, "m": 1, "cm": 1e2, "mm": 1e3, "um": 1e6}
    pixelSize = pixelSize * unitRatio[axisUnit]
    extent = [0, pixelSize * rgb.shape[1], pixelSize * rgb.shape[0], 0]

    im = ax.imshow(rgb, extent=extent, interpolation=None)
    ax.set_ylabel(axisUnit)
    ax.set_xlabel(axisUnit)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    scalar_mappable = ScalarMappable(norm=norm, cmap=mpl.cm.hsv)
    scalar_mappable.set_array([])
    cbar = plt.colorbar(scalar_mappable, ax=ax, cax=cax, ticks=[-np.pi, 0, np.pi])
    cbar.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    return im


def modeTile(P: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Tile 3D data into a single 2D array
    :param P: A complex np.ndarray
    :param normalize: normalize each mode individually
    :param pixelSize: pixelSize in x and y, to display the physical dimension of the plot
    :return: A big array with flattened modes
    """
    if P.ndim == 3 and P.shape[0] > 1:
        if normalize:
            maxs = np.max(abs(P), axis=(-1, -2)) + 1e-6
            P = (P.T / maxs).T
        S = P.shape[0]
        s = math.ceil(np.sqrt(S))
        if s > np.sqrt(S):
            P = np.pad(P, ((0, s**2 - S), (0, 0), (0, 0)), "constant")
        P = P[: s**2, ...]
        P = P.reshape((s, s) + P.shape[1:]).transpose((1, 2, 0, 3) + tuple(range(4, P.ndim + 1)))
        P = P.reshape((s * P.shape[1], s * P.shape[3]) + P.shape[4:])
    elif P.ndim == 4 and P.shape[0] > 1:
        if normalize:
            maxs = np.max(abs(P), axis=(-1, -2)) + 1e-6
            P = (P.T / maxs.T).T
        P = np.swapaxes(P, 1, 2).reshape(P.shape[0] * P.shape[2], P.shape[1] * P.shape[3])
    else:
        P = np.squeeze(P)
    return P


def hsvplot(
    u: np.ndarray,
    ax: Axes | None = None,
    pixelSize: float = 1.0,
    axisUnit: str = "pixel",
    amplitudeScalingFactor: float = 1,
) -> None:
    """
    perform complex plot
    :param ax
    :param pixelSize, default 1
    :param axisUnit, default 'pixel', options: 'm', 'cm', 'mm', 'um'
    return: a complex plot
    """
    u = np.squeeze(np.asarray(u))
    rgb = complex2rgb(u, amplitudeScalingFactor=amplitudeScalingFactor)
    complexPlot(rgb, ax, pixelSize, axisUnit)


def hsvmodeplot(
    P: np.ndarray,
    ax: Axes | None = None,
    normalize: bool = True,
    pixelSize: float = 1,
    axisUnit: str = "pixel",
    amplitudeScalingFactor: float = 1,
) -> None:
    """
    Place multi complex images in a square grid and use hsvplot to display
    :param P: A complex np.ndarray
    :param normalize: normalize each mode individually
    :param pixelSize: pixelSize in x and y, to display the physical dimension of the plot
    :return: a tiled complex plot
    """

    Q = modeTile(np.squeeze(np.asarray(P)), normalize=normalize)
    hsvplot(
        Q,
        ax=ax,
        pixelSize=pixelSize,
        axisUnit=axisUnit,
        amplitudeScalingFactor=amplitudeScalingFactor,
    )


def absplot(
    u: np.ndarray,
    ax: Axes | None = None,
    pixelSize: float = 1.0,
    axisUnit: str = "pixel",
    amplitudeScalingFactor: float = 1,
    cmap: str = "gray",
) -> None:
    U = np.abs(np.asarray(u))
    if not ax:
        fig, ax = plt.subplots()
    unitRatio = {"pixel": 1, "m": 1, "cm": 1e2, "mm": 1e3, "um": 1e6}
    pixelSize = pixelSize * unitRatio[axisUnit]
    extent = [0, pixelSize * U.shape[1], pixelSize * U.shape[0], 0]

    if amplitudeScalingFactor != 1:
        U[U > amplitudeScalingFactor * np.max(U)] = amplitudeScalingFactor * np.max(U)
    ax.imshow(U, extent=extent, interpolation=None, cmap=cmap)
    ax.set_ylabel(axisUnit)
    ax.set_xlabel(axisUnit)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    norm = Normalize(vmin=0, vmax=amplitudeScalingFactor)
    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    cbar = plt.colorbar(
        scalar_mappable,
        ax=ax,
        cax=cax,
        ticks=[0, amplitudeScalingFactor / 2, amplitudeScalingFactor],
    )
    cbar.ax.set_yticklabels(["0", str(amplitudeScalingFactor / 2), str(amplitudeScalingFactor)])


def absmodeplot(
    P: np.ndarray,
    ax: Axes | None = None,
    normalize: bool = True,
    pixelSize: float = 1,
    axisUnit: str = "pixel",
    amplitudeScalingFactor: float = 1,
) -> None:
    Q = modeTile(abs(P), normalize=normalize)
    absplot(Q, ax=ax, pixelSize=pixelSize, axisUnit=axisUnit)


def setColorMap() -> LinearSegmentedColormap:
    """
    create the colormap for diffraction data (the same as matlab)
    return: customized matplotlib colormap
    """
    colors = [
        (1, 1, 1),
        (0, 0.0875, 1),
        (0, 0.4928, 1),
        (0, 1, 0),
        (1, 0.6614, 0),
        (1, 0.4384, 0),
        (0.8361, 0, 0),
        (0.6505, 0, 0),
        (0.4882, 0, 0),
    ]

    n = 255  # Discretizes the interpolation into n bins
    cm = LinearSegmentedColormap.from_list("cmap", colors, n)
    return cm


def show3Dslider(A: np.ndarray, colormap: str = "diffraction") -> None:
    """
    show a 3D plot with a slider using pyqtgraph.
    :param A: a 3D array
    :param colormap: matplotlib colormap, default, customized colormap for plotting diffraction data
    return: a pyqtgraph plot
    """
    print(A.min(), A.max())
    app = pg.mkQApp()
    imv = pg.ImageView(view=pg.PlotItem())
    imv.setWindowTitle("Close to proceed")

    imv.setImage(A)

    # choose colormap from matplotlib colormaps
    if colormap == "diffraction":
        cmap = setColorMap()
    else:
        cmap = get_cmap(colormap)

    # set the colormap
    positions = np.linspace(0, 1, cmap.N)
    colors = [(np.array(cmap(i)[:-1]) * 255).astype("int") for i in positions]
    imv.setColorMap(pg.ColorMap(pos=positions, color=colors))
    imv.show()
    app.exec_()


def plot_alignment(reconstruction: Any, saveit: bool = False) -> Figure:
    """
    Plot position alignment (before vs after correction) and optional history metrics.

    :param reconstruction: Reconstruction object with positions, positions0, and optionally
                           zHistory, TV_history, merit, dz attributes.
    :param saveit: If True, save the figure to plots/alignment.png.
    :return: matplotlib Figure
    """
    import time
    from pathlib import Path

    p_new = np.asarray(reconstruction.positions).T
    p_old = np.asarray(reconstruction.positions0).T

    n_plots = 1
    if hasattr(reconstruction, "zHistory"):
        n_plots += 1
    if hasattr(reconstruction, "TV_history") and len(reconstruction.TV_history) >= 1:
        n_plots += 1
    if hasattr(reconstruction, "merit"):
        n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0

    # Alignment scatter
    ax = axes[ax_idx]
    ax_idx += 1
    ax.scatter(p_old[0], p_old[1], c="yellow", marker="s", s=25, label="original", edgecolors="gray")
    ax.scatter(p_new[0], p_new[1], c="red", marker="o", s=25, label="new")
    ax.set_aspect("equal")
    ax.set_xlabel("Position x [um]")
    ax.set_ylabel("Position y [um]")
    ax.set_title(f"alignment (updated {time.strftime('%Y-%m-%d %H:%M:%S')})")
    ax.legend()

    if hasattr(reconstruction, "zHistory"):
        ax = axes[ax_idx]
        ax_idx += 1
        z_hist = np.asarray(reconstruction.zHistory)
        ax.plot(np.arange(len(z_hist)), z_hist * 1e3)
        ax.set_xlabel("Iteration #")
        ax.set_ylabel("Position [mm]")
        ax.set_title("focus history")

    if hasattr(reconstruction, "TV_history") and len(reconstruction.TV_history) >= 1:
        ax = axes[ax_idx]
        ax_idx += 1
        tv = np.asarray(reconstruction.TV_history)
        ax.scatter(np.arange(len(tv)), tv, s=25)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("TV score")
        ax.set_title("TV history")

    if hasattr(reconstruction, "merit"):
        ax = axes[ax_idx]
        dz = np.asarray(reconstruction.dz)
        merit = np.asarray(reconstruction.merit)
        ax.scatter(dz * 1e3, merit, s=25, label="original")
        ax.scatter(-dz * 1e3, merit, s=25, c="red", marker="s", label="mirrored")
        ax.set_xlabel("Defocus [mm]")
        ax.set_ylabel("Score [a.u.]")
        ax.set_title("merit TV")
        ax.legend()

    fig.tight_layout()

    if saveit:
        output = Path("plots/alignment.png")
        output.parent.mkdir(exist_ok=True)
        fig.savefig(output)

    return fig


def plot_defocus_stack(defocii: np.ndarray, z_values: np.ndarray) -> Figure:
    """
    Browse a stack of defocus images interactively using a matplotlib slider.

    :param defocii: numpy array of shape (N, H, W)
    :param z_values: 1D array of z positions (in metres) for each frame
    :return: matplotlib Figure
    """

    defocii = np.asarray(defocii)
    z_values = np.asarray(z_values)
    N = defocii.shape[0]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)

    im = ax.imshow(defocii[0], cmap="gray")
    ax.set_title(f"z = {z_values[0] * 1e3:.3f} mm")
    ax.axis("off")

    ax_slider = plt.axes([0.15, 0.04, 0.7, 0.04])
    slider = Slider(ax_slider, "Frame", 0, N - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        im.set_data(defocii[idx])
        im.set_clim(defocii[idx].min(), defocii[idx].max())
        ax.set_title(f"z = {z_values[idx] * 1e3:.3f} mm")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    return fig
