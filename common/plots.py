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


# ------------------------------------------------ grafici della versione ANOVA ---
# Le figure "_anova" mostrano quello che l'ANOVA e i t appaiati confrontano: le MEDIE,
# con l'errore standard (SE = SD / sqrt(n)), piu' una linea sottile per soggetto.
# Parentesi con asterischi sui confronti post hoc significativi dopo Bonferroni
# (come la figura 4 di Bellitto et al. 2025): * p < 0.05, ** p < 0.001;
# "†" = differenze non normali (Shapiro p < 0.05): vale il controllo non parametrico.

NONPARAM_TAG = "[NON-PARAMETRIC version] median [IQR]; tests in report: Wilcoxon (Holm) / Friedman"
ANOVA_TAG = "[ANOVA version] mean ± SE; repeated-measures ANOVA + post hoc paired t (Bonferroni)"
STARS_NOTE = "* p < 0.05, ** p < 0.001 (post hoc paired t, Bonferroni); † differences not normal: check Wilcoxon"


def p_stars(p) -> str:
    if p is None or p != p:
        return ""
    return "**" if p < 0.001 else "*" if p < 0.05 else ""


def mean_se(ax, wide: pd.DataFrame, x=None, color=BLUE, label=None, subject_lines=True, band=False):
    """Linee sottili per soggetto + media +- SE tra soggetti (barre, o fascia se band)."""
    w = wide.apply(pd.to_numeric, errors="coerce")
    x = np.arange(w.shape[1]) if x is None else np.asarray(x, float)
    if subject_lines:
        for _, row in w.iterrows():
            ax.plot(x, row.to_numpy(float), color=SUBJECT_LINE, lw=0.8, zorder=1)
    mean = w.mean().to_numpy(float)
    se = (w.std(ddof=1) / np.sqrt(w.notna().sum())).to_numpy(float)
    if band:
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2, lw=0, zorder=2)
        ax.plot(x, mean, color=color, lw=2, zorder=3, label=label)
    else:
        ax.errorbar(x, mean, yerr=se, color=color, lw=2, marker="o", ms=4, capsize=3, zorder=3, label=label)
    return mean, se


def brackets(ax, pairs):
    """pairs: [(x1, x2, p, non_normale)]; disegna solo quelle significative, una sopra
    l'altra sopra i dati (asterischi come nel paper)."""
    sig = [(x1, x2, p, nn) for x1, x2, p, nn in pairs if p_stars(p)]
    if not sig:
        return
    lo, hi = ax.get_ylim()
    step = 0.07 * (hi - lo)
    for i, (x1, x2, p, nn) in enumerate(sig):
        y = hi + step * (i + 0.3)
        ax.plot([x1, x1, x2, x2], [y - step * 0.25, y, y, y - step * 0.25], color=INK, lw=0.9, clip_on=False)
        ax.text((x1 + x2) / 2, y, p_stars(p) + ("†" if nn else ""), ha="center", va="bottom", fontsize=10,
                color=INK, clip_on=False)
    ax.set_ylim(lo, hi + step * (len(sig) + 0.6))


def anova_caption(rows, names=None) -> str:
    """Testo breve con gli effetti dell'ANOVA (p GG quando ci sono piu' di 2 livelli)."""
    parts = []
    for r in rows:
        if r.get("F") is None or r.get("F") != r.get("F"):
            continue
        p = r["p_gg"] if r.get("df1", 1) > 1 else r["p"]
        name = (names or {}).get(r["effect"], r["effect"])
        ptxt = "<0.001" if p < 0.001 else f"={p:.3f}"
        parts.append(f"{name}: F({int(r['df1'])},{int(r['df2'])})={r['F']:.1f}, "
                     f"p{' GG' if r.get('df1', 1) > 1 else ''}{ptxt}{p_stars(p)}")
    if not parts:
        return "ANOVA not computable"
    return "ANOVA – " + parts[0] if len(parts) == 1 else "ANOVA:\n" + "\n".join(parts)
