"""Stile e grafici comuni (new_stats, ritorni del training, questionari, calibrazioni).

- box off (solo assi sinistro e inferiore), unita' di misura fra parentesi tonde;
- box plot come la figura 4 di Najie et al. (box pieno chiaro con bordo colorato, linea
  della mediana, baffi tratteggiati fino a 1.5 IQR, un punto per soggetto con il colore
  del soggetto, lo stesso in tutti i grafici);
- parentesi grigie con gli asterischi sui confronti significativi, a livelli sopra l'asse:
  * p < 0.05, ** p < 0.01, *** p < 0.001; se una figura ha degli asterischi, save_figure
  scrive la loro legenda;
- grafici singoli: niente titolo, sotto il grafico una legenda scarna (save_figure(..., note=...)):
  cosa si vede (es. BOX_NOTE) piu' gli asterischi.
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
BOX_NOTE = "Box: median [IQR], whiskers 1.5 IQR; dots: participants."
STARS_NOTE = "* p < 0.05, ** p < 0.01, *** p < 0.001"
# figure della tesi: scritte direttamente fra le figure di LaTeX, cosi' si aggiornano a ogni analisi
THESIS_DIR = Path(__file__).resolve().parents[3] / "Tesi_latex" / "Chapters" / "Figures" / "results"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.linewidth": 0.8, "axes.labelcolor": INK,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK, "ytick.color": INK, "xtick.labelcolor": "inherit", "ytick.labelcolor": "inherit",
        "xtick.direction": "out", "ytick.direction": "out",
        # tutte le scritte a 11 pt e in grassetto (si devono leggere bene nella tesi), tranne i numeri
        # sugli assi, non in grassetto (_style_ticks)
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 11, "axes.labelweight": "bold", "font.size": 11, "font.weight": "bold",
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.frameon": False, "legend.fontsize": 11,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def _is_number(text: str) -> bool:
    try:
        float(text.replace("\u2212", "-"))
        return True
    except ValueError:
        return False


def _style_ticks(fig):
    """Etichette degli assi: i numeri non in grassetto, le etichette di testo (Outbound, Test 1, ...)
    in grassetto."""
    fig.canvas.draw()
    for ax in fig.axes:
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            if not lab.get_text():
                continue
            lab.set_fontweight("normal" if _is_number(lab.get_text()) else "bold")


def save_figure(fig, out_dir, name, note=None):
    """plots/<name>.png (300 dpi) e plots/<name>.pdf (vettoriale, per la tesi). Se brackets ha
    disegnato degli asterischi aggiunge la loro legenda: in alto a destra nelle figure con titolo,
    sotto il grafico insieme a `note` (cosa si vede) nei grafici singoli."""
    _style_ticks(fig)
    star_text = STARS_NOTE if getattr(fig, "_has_stars", False) else None
    if note is not None:
        fig.canvas.draw()   # posizione delle etichette dell'asse x, per scrivere la nota sotto
        bottom = min(ax.get_tightbbox().transformed(fig.transFigure.inverted()).y0 for ax in fig.axes
                     if ax.get_visible())
        text = "\n".join(t for t in (note, star_text) if t)
        fig.text(0.01, bottom - 0.02, text, ha="left", va="top", fontsize=11, color=INK2, wrap=True)
    elif star_text:
        title = fig._suptitle
        y, va = (title.get_position()[1], "top") if title is not None else (1.0, "bottom")
        fig.text(0.995, y, star_text, ha="right", va=va, fontsize=11, color=SIG_COLOR)
    path = Path(out_dir) / "plots" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def save_thesis_figure(fig, name, note=None):
    """<THESIS_DIR>/<name>.pdf, senza titolo. Sotto tutta la figura (anche sotto la legenda dei
    soggetti) una sola nota per la figura intera: `note` (cosa si vede, es. BOX_NOTE) e, se ci sono
    parentesi, la legenda degli asterischi. Se la cartella della tesi non c'e' la figura non viene
    scritta."""
    _style_ticks(fig)
    star_text = STARS_NOTE if getattr(fig, "_has_stars", False) else None
    text = "\n".join(t for t in (note, star_text) if t)
    if text:
        fig.canvas.draw()
        to_fig = fig.transFigure.inverted()
        artists = [ax for ax in fig.axes if ax.get_visible()] + list(fig.legends)
        bottom = min(a.get_tightbbox().transformed(to_fig).y0 for a in artists)
        fig.text(0.01, bottom - 0.01, text, ha="left", va="top", fontsize=11, color=INK2)
    if THESIS_DIR.parent.is_dir():
        THESIS_DIR.mkdir(exist_ok=True)
        fig.savefig(THESIS_DIR / f"{name}.pdf")
        print(f"[{name}] figura tesi -> {THESIS_DIR / f'{name}.pdf'}")
    plt.close(fig)


def subject_colors(subjects) -> dict:
    return {s: SUBJECT_PALETTE[i % len(SUBJECT_PALETTE)] for i, s in enumerate(sorted(subjects))}


def stars(p) -> str:
    if p is None or p != p:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def wrap_unit(ax, max_len=24):
    """Etichetta dell'asse y su una riga; se e' piu' lunga di max_len caratteri va a capo solo l'unita'
    di misura (tra parentesi tonde). Senza unita' resta su una riga."""
    label = ax.get_ylabel()
    if len(label) > max_len and " (" in label:
        name, unit = label.rsplit(" (", 1)
        ax.set_ylabel(f"{name}\n({unit}")


def axis_label(label, unit) -> str:
    return f"{label} ({unit})" if unit else label


def _jitter(subjects, colors, width):
    """Spostamento orizzontale fisso per soggetto (stesso in tutte le condizioni)."""
    order = sorted(colors)
    n = max(len(order) - 1, 1)
    return {s: (order.index(s) / n - 0.5) * width for s in subjects}


def condition_plot(ax, data: dict, groups: list, conditions: list, styles: dict, colors: dict, width=None,
                   whisker_color=None):
    """Box plot (mediana, IQR, baffi a 1.5 IQR) con un punto per soggetto.
    data: {(gruppo, condizione): Series indicizzata per soggetto}; i gruppi vanno
    sull'asse x (etichette), le condizioni affiancate dentro ogni gruppo con lo stile
    styles[condizione] = (riempimento, bordo); baffi del colore del bordo o di whisker_color.
    Restituisce {(gruppo, condizione): x}."""
    nc = len(conditions)
    width = width or (0.32 if nc > 1 else 0.42)
    offsets = (np.arange(nc) - (nc - 1) / 2) * (width * 1.35)
    xpos = {}
    for gi, g in enumerate(groups):
        for ci, c in enumerate(conditions):
            v = pd.to_numeric(data.get((g, c), pd.Series(dtype=float)), errors="coerce").dropna()
            x = gi + offsets[ci]
            xpos[(g, c)] = x
            if v.empty:
                continue
            face, edge = styles[c]
            ax.boxplot([v.to_numpy()], positions=[x], widths=width, patch_artist=True, showfliers=False,
                       whis=1.5, manage_ticks=False, zorder=2,
                       boxprops=dict(facecolor=face, edgecolor=edge, linewidth=1.2),
                       medianprops=dict(color=edge, linewidth=1.6),
                       whiskerprops=dict(color=whisker_color or edge, linewidth=1.0, linestyle=(0, (4, 3))),
                       capprops=dict(color=whisker_color or edge, linewidth=1.0))
            jit = _jitter(v.index, colors, width * 0.55)
            ax.scatter([x + jit[s] for s in v.index], v.to_numpy(), s=14, c=[colors[s] for s in v.index],
                       linewidths=0, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xlim(-0.6, len(groups) - 0.4)
    for gi in range(len(groups)):
        ax.axvline(gi, color=GRID, lw=0.8, zorder=0)
    return xpos


BRACKET_STEP_PT = 14   # distanza fra due livelli di parentesi (punti): asterischi e linee non si toccano
BRACKET_GAP_PT = 5     # distanza del primo livello dai dati


def _assign_levels(sig):
    """Livello di ogni parentesi (prima le corte): il piu' basso in cui non tocca in orizzontale quelle
    gia' messe. sig = [(a, b, testo)], ordinata qui. Restituisce [(a, b, testo, livello)] e n. livelli."""
    levels, out = [], []
    for a, b, txt in sorted(sig, key=lambda s_: s_[1] - s_[0]):
        k = next((i for i, lv in enumerate(levels) if all(b < pa - 0.08 or a > pb + 0.08 for pa, pb in lv)), None)
        if k is None:
            levels.append([])
            k = len(levels) - 1
        levels[k].append((a, b))
        out.append((a, b, txt, k))
    return out, len(levels)


def bracket_levels(items: list) -> int:
    """Quanti livelli di parentesi servono per items = [(x1, x2, p)] (solo i significativi)."""
    sig = [(min(x1, x2), max(x1, x2), stars(p)) for x1, x2, p in items if stars(p)]
    return _assign_levels(sig)[1]


def bracket_space_pt(n_levels: int) -> float:
    """Spazio sopra l'asse occupato da n_levels livelli di parentesi (punti)."""
    return BRACKET_GAP_PT + n_levels * BRACKET_STEP_PT if n_levels else 0.0


def brackets(ax, items: list):
    """items = [(x1, x2, p)]: parentesi grigie con gli asterischi, solo per i confronti significativi,
    a livelli sopra il bordo superiore dell'asse, a distanza fissa in punti, senza cambiare la scala y:
    tante parentesi non schiacciano i dati e non si sovrappongono. Salva in ax._title_pad lo spazio
    da lasciare per il titolo del pannello."""
    from matplotlib.transforms import ScaledTranslation
    sig = [(min(x1, x2), max(x1, x2), stars(p)) for x1, x2, p in items if stars(p)]
    if not sig:
        return
    ax.figure._has_stars = True   # save_figure aggiunge la legenda degli asterischi
    placed, n = _assign_levels(sig)
    tick = 3.0 / 72   # dentini di 3 pt verso il basso
    for a, b, txt, k in placed:
        y = (BRACKET_GAP_PT + k * BRACKET_STEP_PT) / 72
        top = ax.get_xaxis_transform() + ScaledTranslation(0, y, ax.figure.dpi_scale_trans)
        low = ax.get_xaxis_transform() + ScaledTranslation(0, y - tick, ax.figure.dpi_scale_trans)
        ax.plot([a, b], [1, 1], color=SIG_COLOR, lw=0.9, clip_on=False, transform=top)
        for x in (a, b):
            ax.annotate("", xy=(x, 1), xycoords=low, xytext=(x, 1), textcoords=top,
                        arrowprops=dict(arrowstyle="-", color=SIG_COLOR, lw=0.9, shrinkA=0, shrinkB=0))
        ax.text((a + b) / 2, 1, txt, ha="center", va="bottom", color=SIG_COLOR, fontsize=11, clip_on=False,
                transform=top)
    ax._title_pad = max(6.0, bracket_space_pt(n) + 4)


def subject_handles(colors: dict) -> list:
    return [plt.Line2D([], [], ls="", marker="o", ms=4.5, color=c, label=participant_label(s))
            for s, c in sorted(colors.items())]


def participant_label(s) -> str:
    """Etichetta nelle figure della tesi: P01..P10 al posto dei codici soggetto S001..S010."""
    s = str(s)
    return f"P{int(s[1:]):02d}" if s[:1] == "S" and s[1:].isdigit() else s


def side_legend(ax, handles: list):
    """Legenda a destra del grafico, fuori dagli assi (grafici singoli)."""
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1, fontsize=11,
              handletextpad=0.3, borderaxespad=0)


def subject_legend(fig, colors: dict, ncol=None, y=0.0, extra=()):
    """Legenda dei colori dei soggetti (piu' eventuali `extra` handle), in una riga sotto la figura."""
    handles = list(extra) + subject_handles(colors)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(handles),
               handletextpad=0.3, columnspacing=1.0, fontsize=11)


def share_ylim(axes) -> tuple:
    """Stessa scala y (stessi limiti) per pannelli con la stessa unita' e grandezze simili, es. le
    percentuali del blind; da chiamare dopo aver disegnato dati e parentesi. Restituisce i limiti."""
    lims = [ax.get_ylim() for ax in axes]
    lim = (min(lo for lo, _ in lims), max(hi for _, hi in lims))
    for ax in axes:
        ax.set_ylim(lim)
    return lim
