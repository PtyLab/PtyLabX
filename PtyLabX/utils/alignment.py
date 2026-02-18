import math

import matplotlib.pyplot as plt
import numpy as np

from PtyLabX import Engines, ExperimentalData, Params, Reconstruction


def show_alignment(
    reconstruction: Reconstruction,
    data: ExperimentalData,
    params: Params,
    engine: Engines.BaseEngine,
):
    """Show the alignment of ptychogram frames and scan positions.

    Displays the 15 ptychogram frames nearest to the scan centre (radially ordered)
    as a grid, alongside a scatter plot of all scan positions with those 15 highlighted.
    """
    positions = np.asarray(reconstruction.positions)
    mean_pos = positions.mean(0, keepdims=True)
    order = np.argsort(np.linalg.norm(positions - mean_pos, axis=-1))[:15]
    ptycho_ordered = np.asarray(data.ptychogram[order])

    n = len(order)
    ncols = 5
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(4 * (ncols + 1), 4 * nrows))
    gs = fig.add_gridspec(nrows, ncols + 1)

    for i in range(n):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        ax.imshow(np.log1p(ptycho_ordered[i].astype(float)), cmap="gray")
        ax.set_title(f"frame {order[i]}", fontsize=8)
        ax.axis("off")

    ax_pos = fig.add_subplot(gs[:, ncols])
    ax_pos.scatter(positions[:, 1], positions[:, 0], s=10, c="steelblue", label="all positions")
    ax_pos.scatter(
        positions[order, 1], positions[order, 0], s=40, c="red", zorder=5, label="shown (nearest 15)"
    )
    ax_pos.set_aspect("equal")
    ax_pos.set_xlabel("col position")
    ax_pos.set_ylabel("row position")
    ax_pos.set_title("scan positions")
    ax_pos.legend(fontsize=7)
    ax_pos.invert_yaxis()

    fig.suptitle("Ptychogram alignment (radially ordered)", y=1.01)
    fig.tight_layout()
    plt.show()
    return fig
