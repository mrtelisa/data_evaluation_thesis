"""Stile e grafici comuni (stessa palette di questionari/utils.py)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#2a78d6"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
               "#7b5ea7", "#8c564b", "#17becf"]
SEQUENTIAL_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"])
# primo / secondo momento (arancione vs blu scuro: leggibili anche in B/N e con daltonismo)
FIRST_COLOR, SECOND_COLOR = "#ed7014", "#104281"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SUBJECT_LINE = "#c3c2b7"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
        "font.size": 9.5, "legend.frameon": False,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save_figure(fig, out_dir, name):
    plots = Path(out_dir) / "plots"
    path = plots / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def spaghetti(ax, wide: pd.DataFrame, xlabels, ylabel="", title="", color=BLUE):
    """Una linea sottile per soggetto (righe di `wide`, colonne = momenti) +
    mediana del gruppo (linea spessa) con la fascia Q1-Q3."""
    x = np.arange(wide.shape[1])
    for _, row in wide.iterrows():
        v = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        ax.plot(x, v, color=SUBJECT_LINE, lw=0.9, marker="o", ms=2.5, zorder=2)
    med = wide.median(numeric_only=True).to_numpy(dtype=float)
    q1 = wide.quantile(0.25, numeric_only=True).to_numpy(dtype=float)
    q3 = wide.quantile(0.75, numeric_only=True).to_numpy(dtype=float)
    ax.fill_between(x, q1, q3, color=color, alpha=0.15, lw=0, zorder=1)
    ax.plot(x, med, color=color, lw=2.4, marker="o", ms=5, zorder=3, label="median [IQR]")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")


def heatmap3x3(ax, values: dict, title="", fmt="{:.1f}", vmin=None, vmax=None, cmap=SEQUENTIAL_CMAP):
    """Griglia 3 x 3 delle regioni (1 in alto a sinistra ... 9 in basso a destra);
    `values`: {regione: valore} (le regioni mancanti restano grigie)."""
    grid = np.full((3, 3), np.nan)
    for region, v in values.items():
        r, c = divmod(int(region) - 1, 3)
        grid[r, c] = v
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=vmin, vmax=vmax)
    for r in range(3):
        for c in range(3):
            v = grid[r, c]
            region = r * 3 + c + 1
            txt = f"{region}\n" + (fmt.format(v) if not np.isnan(v) else "-")
            dark = not np.isnan(v) and vmax is not None and vmin is not None and \
                (v - vmin) / max(vmax - vmin, 1e-9) > 0.6
            ax.text(c, r, txt, ha="center", va="center", fontsize=8.5,
                    color="white" if dark else INK)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(AXIS)
    ax.set_title(title, loc="left", fontsize=10)


def grid_axes(n, ncols=3, width=4.0, height=3.2):
    """Figura con n pannelli su ncols colonne; i pannelli vuoti vengono nascosti."""
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]
