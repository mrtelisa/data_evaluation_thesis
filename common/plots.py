"""Stile e grafici comuni a blind test, training e robot.

- box off (solo assi sinistro e inferiore), unita' di misura fra parentesi tonde;
- box plot come la figura 4 di Najie et al. (box pieno chiaro con bordo colorato, linea
  della mediana, baffi tratteggiati fino a 1.5 IQR, un punto per soggetto con il colore
  del soggetto, lo stesso in tutti i grafici);
- versione media +- SD con lo stesso stile: box da media - SD a media + SD con la linea
  della media (senza baffi), per l'analisi parametrica;
- parentesi grigie con gli asterischi sui confronti significativi:
  * p < 0.05, ** p < 0.01, *** p < 0.001.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK2, MUTED, GRID = "#000000", "#404040", "#8c8c8c", "#e6e6e6"
SIG_COLOR = "#8c8c8c"
# box: (riempimento, bordo e mediana) come Najie: grigio/nero e rosso chiaro/rosso
GREY_BOX = ("#d9d9d9", "#000000")
RED_BOX = ("#ffd6d6", "#e8000b")
# curve della figura 4: training 1 arancione (come Bellitto), training 2 blu
TRAINING_COLORS = {"T1": "#d95319", "T2": "#0072bd"}
# un colore per soggetto (i primi sette sono quelli di MATLAB, come nei grafici di Najie)
SUBJECT_PALETTE = ["#0072bd", "#d95319", "#edb120", "#7e2f8e", "#77ac30", "#4dbeee", "#a2142f",
                   "#e377c2", "#17becf", "#8c564b", "#bcbd22", "#7f7f7f", "#1f3a93", "#2ca02c"]
BLUE, CATEGORICAL = SUBJECT_PALETTE[0], SUBJECT_PALETTE   # per i grafici delle calibrazioni


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.linewidth": 0.8, "axes.labelcolor": INK,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK, "ytick.color": INK, "xtick.direction": "out", "ytick.direction": "out",
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9.5, "axes.labelweight": "bold", "font.size": 9,
        "legend.frameon": False, "legend.fontsize": 8,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def save_figure(fig, out_dir, name):
    """plots/<name>.png (300 dpi) e plots/<name>.pdf (vettoriale, per la tesi)."""
    path = Path(out_dir) / "plots" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def subject_colors(subjects) -> dict:
    return {s: SUBJECT_PALETTE[i % len(SUBJECT_PALETTE)] for i, s in enumerate(sorted(subjects))}


def stars(p) -> str:
    if p is None or p != p:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def axis_label(label, unit) -> str:
    return f"{label} ({unit})" if unit else label


def _jitter(subjects, colors, width):
    """Spostamento orizzontale fisso per soggetto (stesso in tutte le condizioni)."""
    order = sorted(colors)
    n = max(len(order) - 1, 1)
    return {s: (order.index(s) / n - 0.5) * width for s in subjects}


def condition_plot(ax, data: dict, groups: list, conditions: list, styles: dict, colors: dict,
                   kind="box", width=None):
    """Box plot (kind="box": mediana, IQR, baffi a 1.5 IQR) o box media +- SD (kind="mean")
    con un punto per soggetto.
    data: {(gruppo, condizione): Series indicizzata per soggetto}; i gruppi vanno
    sull'asse x (etichette), le condizioni affiancate dentro ogni gruppo con lo stile
    styles[condizione] = (riempimento, bordo). Restituisce {(gruppo, condizione): x}."""
    nc = len(conditions)
    width = width or (0.32 if nc > 1 else 0.42)
    offsets = (np.arange(nc) - (nc - 1) / 2) * (width * 1.35)
    xpos, tops = {}, {}
    for gi, g in enumerate(groups):
        for ci, c in enumerate(conditions):
            v = pd.to_numeric(data.get((g, c), pd.Series(dtype=float)), errors="coerce").dropna()
            x = gi + offsets[ci]
            xpos[(g, c)] = x
            if v.empty:
                continue
            face, edge = styles[c]
            tops[x] = float(v.max())
            if kind == "box":
                ax.boxplot([v.to_numpy()], positions=[x], widths=width, patch_artist=True, showfliers=False,
                           whis=1.5, manage_ticks=False, zorder=2,
                           boxprops=dict(facecolor=face, edgecolor=edge, linewidth=1.2),
                           medianprops=dict(color=edge, linewidth=1.6),
                           whiskerprops=dict(color=edge, linewidth=1.0, linestyle=(0, (4, 3))),
                           capprops=dict(color=edge, linewidth=1.0))
            else:
                m, sd = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
                tops[x] = max(tops[x], float(m + sd))
                ax.add_patch(plt.Rectangle((x - width / 2, m - sd), width, 2 * sd, facecolor=face,
                                           edgecolor=edge, linewidth=1.2, zorder=2))
                ax.plot([x - width / 2, x + width / 2], [m, m], color=edge, lw=1.6, zorder=2,
                        solid_capstyle="butt")
                ax.update_datalim([(x, m - sd), (x, m + sd)])
            jit = _jitter(v.index, colors, width * 0.55)
            ax.scatter([x + jit[s] for s in v.index], v.to_numpy(), s=14, c=[colors[s] for s in v.index],
                       linewidths=0, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xlim(-0.6, len(groups) - 0.4)
    for gi in range(len(groups)):
        ax.axvline(gi, color=GRID, lw=0.8, zorder=0)
    ax._data_tops = tops   # per le parentesi: altezza dei dati in ogni posizione
    return xpos


def brackets(ax, items: list):
    """items = [(x1, x2, p)]: parentesi grigie con gli asterischi, solo per i confronti
    significativi. Ogni parentesi sta appena sopra i dati fra x1 e x2 (se condition_plot
    li ha registrati, altrimenti sopra tutto il grafico); quelle che si sovrappongono
    vengono impilate (come Najie)."""
    sig = [(min(x1, x2), max(x1, x2), stars(p)) for x1, x2, p in items if stars(p)]
    if not sig:
        return
    lo, hi = ax.get_ylim()
    step = 0.075 * (hi - lo)
    tops = getattr(ax, "_data_tops", None)
    sig.sort(key=lambda s_: s_[1] - s_[0])        # prima le corte, poi le lunghe sopra
    placed = []                                    # (a, b, y)
    for a, b, txt in sig:
        base = max((v for x, v in tops.items() if a - 1e-9 <= x <= b + 1e-9), default=hi) if tops else hi
        y = base + step * 0.6
        for pa, pb, py in placed:
            if not (b < pa - 0.05 or a > pb + 0.05) and y < py + step * 1.1:
                y = py + step * 1.1
        placed.append((a, b, y))
        ax.plot([a, a, b, b], [y - step * 0.3, y, y, y - step * 0.3], color=SIG_COLOR, lw=0.9, clip_on=False)
        ax.text((a + b) / 2, y - step * 0.05, txt, ha="center", va="bottom", color=SIG_COLOR, fontsize=10,
                clip_on=False)
    ax.set_ylim(lo, max(hi, max(y for _, _, y in placed) + step * 0.9))


def subject_legend(fig, colors: dict, ncol=None, y=0.0, extra=()):
    """Legenda dei colori dei soggetti (piu' eventuali `extra` handle), in una riga sotto la figura."""
    handles = list(extra) + [plt.Line2D([], [], ls="", marker="o", ms=4.5, color=c, label=s)
                             for s, c in sorted(colors.items())]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(handles),
               handletextpad=0.3, columnspacing=1.0, fontsize=7.5)


def grid_axes(n, ncols=3, width=4.0, height=3.2):
    """Figura con n pannelli su ncols colonne; i pannelli vuoti vengono nascosti."""
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]
