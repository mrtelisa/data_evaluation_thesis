"""Analisi statistica dei tre task, quella della dottoranda (Aurora: cartelle newStats_blind,
newStats_reachy, newStats_training con i suoi file jamovi e i riassunti), usata nella tesi. Legge le
metriche calcolate dagli script dei task (reaching_regions, reaching_blind, real_robot), quindi va
lanciata dopo (run_all.py lo fa).

Training (andate, reaching_regions/statistics/training_trials.csv):
    3 blocchi da 32 andate per training (B1-B3, ognuno con 4 ripetizioni di ogni target), valore di
    un blocco = media delle sue 32 andate; ANOVA a misure ripetute Training (T1, T2) x Blocco (B1-B3);
    movement duration su log. Post hoc su TUTTE le 15 coppie dei 6 blocchi (ANOVA a un fattore con 2
    livelli, F = t^2), p corretti con Holm x15 (scelta di Elisa: tutti gli asterischi, come il post hoc di
    jamovi; la dottoranda faceva solo 4 confronti pianificati).
Blind (reaching_blind/statistics/blind_trials.csv): un valore per test (media dei 3 blocchi da 4
    prove); ANOVA a un
    fattore Test (5 livelli); Euclidean error su log; post hoc su tutte le 10 coppie di test, Holm x10.
    Linearity index escluso (va rifatto dopo la correzione del trial anomalo).
Robot (real_robot/statistics/robot_subjects.csv): test 1 vs test 2, andata e ritorno, ANOVA a misure
    ripetute (2 livelli);
    completion time su log; Holm x2 (andata e ritorno) per ogni metrica. Quello che la dottoranda non
    analizza e la tesi riporta e' trattato come lei ha trattato il resto ("le stesse cose"): Shapiro-Wilk
    sulle differenze fra i test, log solo se una delle due fasi non e' normale (il completion time resta
    su log in tutte le fasi), Holm x2 sulle due fasi in coppia:
        jerk e metriche esplorative (soste, cambi al minuto, distanza del cursore dal centro): andata e ritorno
        completion time: grasp e posizionamento; totale e andata + ritorno
    Il grasp senza S001 e' un confronto singolo di controllo, non corretto.
Blind, metrica esplorativa della tesi: % di prove che finiscono nella regione del target, stessa
    analisi delle altre.
In tutte: Greenhouse-Geisser quando il test di Mauchly e' significativo (p < 0.05); medie, SD e
grafici sui dati originali, la trasformazione log serve solo per i test. Controllo non parametrico:
Friedman e Wilcoxon (Holm), utile per le metriche con residui non normali (XE, HR).
Le parentesi dei grafici sono tutti i confronti significativi (p di Holm); le tabelle della tesi
riportano solo le ANOVA, i p dei confronti sono nel report.

Output:
  una sottocartella per task, <task> = training, blind, robot:
    <task>/statistics/report.txt                 tutti i numeri (F, gdl, p, eta^2 parziale, confronti)
    <task>/statistics/<task>_descriptives.csv    media, SD, mediana, IQR per cella
    <task>/statistics/<task>_tests.csv           ANOVA e confronti a coppie
    <task>/statistics/plots/single/...           i pannelli uno per uno
    plot_ppt/                                    copie per la presentazione (PPT_PLOTS): EE, XE, DE del
                                                 blind e le curve di movement duration e direction error
    thesis/figures/*_newstats.pdf        figure per la tesi
    thesis/tables/*_newstats.tex         tabelle per la tesi (da includere con \\input)
    Con --thesis le stesse vanno direttamente in Tesi_latex/Chapters/Figures/results e Chapters/Tables
    (sovrascrivono quelle della tesi: di default NO, Elisa non vuole che la tesi cambi da sola).

Uso:
    python3 new_stats_analysis.py [--thesis]
"""

import argparse
import itertools
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reaching_regions"))
from common.plots import (BOX_NOTE, GREY_BOX, INK2, RED_BOX, THESIS_DIR, TRAINING_COLORS, _jitter,   # noqa: E402
                          axis_label, brackets, condition_plot, save_figure, save_thesis_figure, setup_style,
                          share_ylim, side_legend, subject_colors, subject_handles, subject_legend)
from common.report import Report   # noqa: E402
from common.stats import describe, friedman, holm, paired_t, rm_anova, wilcoxon_paired   # noqa: E402
from scipy.stats import shapiro   # noqa: E402
import training_analysis as ta   # noqa: E402  (curve della figura 4: media mobile, asse delle prove)

OUT = {task: BASE_DIR / task / "statistics" for task in ("training", "blind", "robot")}   # una cartella per task
import common.plots as _plots   # noqa: E402  (per scegliere dove scrive save_thesis_figure)
LOCAL_THESIS = BASE_DIR / "thesis"
TABLES_DIR = LOCAL_THESIS / "tables"   # con --thesis: THESIS_DIR.parent.parent / "Tables"
SUFFIX = "_newstats"
CURVE_NOTE = "Line: median; band: IQR."
TRAINING_BOX = {"T1": ("#f7d9c9", TRAINING_COLORS["T1"]), "T2": ("#cfe3f3", TRAINING_COLORS["T2"])}

# metrica -> (sigla, etichetta, unita', log per i test, cifre decimali)
TRAINING = {
    "euclidean_error":   ("EE", "Euclidean error at 1 s", "cm", False, 2),
    "direction_error":   ("DE", "Direction error at 1 s", "deg", False, 1),
    "extent_error":      ("XE", "Extent error at 1 s", "cm", False, 2),
    "movement_duration": ("MD", "Movement duration", "s", True, 2),
    "linearity_index":   ("LI", "Linearity index", "-", False, 2),
}
BLOCK = 32
BLOCKS = (1, 2, 3)
CELLS = [(tr, b) for tr in ("T1", "T2") for b in BLOCKS]
TRAINING_PAIRS = [(a, b, f"{a[0]} B{a[1]} vs {b[0]} B{b[1]}") for a, b in itertools.combinations(CELLS, 2)]
# blind: metrica -> (sigla, etichetta, unita', log, cifre); nei grafici come la figura della tesi
BLIND = {
    "hit":             ("HR", "Hit rate", "%", False, 1),
    "euclidean_error": ("EE", "Euclidean error at 4 s", "cm", True, 2),
    "extent_error":    ("XE", "Extent error at 4 s", "cm", False, 2),
    "direction_error": ("DE", "Direction error at 4 s", "deg", False, 1),
    "movement_duration": ("MD", "Movement duration", "s", False, 2),
}
BLIND_PLOTTED = ["hit", "euclidean_error", "extent_error", "direction_error"]
TESTS = (1, 2, 3, 4, 5)
BLIND_PAIRS = [(a, b, f"Test {a} vs {b}") for a, b in itertools.combinations(TESTS, 2)]
ROBOT = {
    "completion_time":    ("CT", "Completion time", "s", True, 0),
    "path_length":        ("PL", "Path length", "m", False, 2),
    "mean_speed":         ("MS", "Mean speed", "m/s", False, 2),
    "jerk":               ("JK", "Jerk", "m/s³", None, 2),   # None: log deciso da Shapiro (metrica nuova)
    "region_transitions": ("RT", "Region transitions", "", False, 0),
}
LEGS = {"andata": "Outbound", "ritorno": "Return"}
# fasi del completion time che la tesi riporta oltre alle due di guida: confronti singoli, non corretti
# (la dottoranda analizza solo andata e ritorno), anche loro su log
# metriche esplorative della tesi (real_robot/statistics/robot_subjects.csv): solo report e testo
ROBOT_EXPLORATORY = {"stop_percent": ("Stop time while driving", "%"),
                     "transitions_per_min": ("Region transitions per minute of driving", "/min"),
                     "cursor_eccentricity": ("Cursor distance from the centre while moving", "")}
# coppie di fasi del completion time corrette insieme con Holm, come andata e ritorno
CT_PAIRS = [("grasp", "posizionamento"), ("totale", "andata_ritorno")]
CT_OTHER = {"grasp": "Grasp", "posizionamento": "Placement", "totale": "Total", "andata_ritorno": "Outbound + return"}
ROBOT_PARTS = [["completion_time", "path_length"], ["mean_speed", "jerk", "region_transitions"]]
GRASP_WITHOUT = "S001"   # grasp del test 2 con 5 riposizionamenti (519 s): la tesi lo riporta anche senza


# ------------------------------------------------------------------ statistica ---

def anova(wide: pd.DataFrame, factors: list, log: bool) -> list:
    """rm_anova (su log se `log`), con la riga da riportare: Greenhouse-Geisser se Mauchly p < 0.05."""
    rows = rm_anova(np.log(wide) if log else wide, factors)
    for r in rows:
        r["gg"] = bool(r.get("mauchly_p") == r.get("mauchly_p") and r.get("mauchly_p", 1) < 0.05)
        r["df1_rep"], r["df2_rep"], r["p_rep"] = ((r["df1_gg"], r["df2_gg"], r["p_gg"]) if r["gg"]
                                                  else (r["df1"], r["df2"], r["p"]))
    return rows


def pairwise(wide: pd.DataFrame, pairs: list, log: bool) -> pd.DataFrame:
    """Confronti a coppie come ANOVA a un fattore con 2 livelli (F = t^2, gdl 1 e n-1), Holm sulla
    famiglia; controllo Wilcoxon (Holm) sui dati originali."""
    data = np.log(wide) if log else wide
    rows = []
    for a, b, name in pairs:
        t = paired_t(data[a], data[b])
        wx = wilcoxon_paired(wide[a], wide[b])
        rows.append({"comparison": name, "a": a, "b": b, "n": t["n"], "F": t["statistic"] ** 2, "df2": t["df"],
                     "p": t["p"], "p_wilcoxon": wx["p"]})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p"])
    df["p_wilcoxon_holm"] = holm(df["p_wilcoxon"])
    return df


def write_anova(report: Report, rows: list, log: bool):
    for r in rows:
        df1 = f"{r['df1_rep']:.2f}" if r["gg"] else f"{r['df1_rep']:.0f}"
        df2 = f"{r['df2_rep']:.2f}" if r["gg"] else f"{r['df2_rep']:.0f}"
        mauchly = f", Mauchly p = {r['mauchly_p']:.3f}" if r["mauchly_p"] == r["mauchly_p"] else ""
        report.text(f"  {r['effect']:<20} F({df1}, {df2}) = {r['F']:.2f}, p {p_text(r['p_rep'])}"
                    f"{' (GG)' if r['gg'] else ''}, eta2p = {r['partial_eta2']:.2f}{mauchly}")
    if log:
        report.text("  (log-transformed data)")


def write_pairwise(report: Report, ph: pd.DataFrame):
    for _, r in ph.iterrows():
        report.text(f"  {r['comparison']:<18} F(1, {r['df2']:.0f}) = {r['F']:.2f}, p {p_text(r['p'])}, "
                    f"p Holm {p_text(r['p_holm'])}{stars_text(r['p_holm'])} | Wilcoxon p Holm "
                    f"{p_text(r['p_wilcoxon_holm'])}")


def p_text(p) -> str:
    """'= 0.012' o '< .001', da scrivere dopo 'p '."""
    return "< .001" if p < 0.001 else f"= {p:.3f}"


def stars_text(p) -> str:
    return " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""


def describe_rows(task: str, metric: str, wide: pd.DataFrame, names: dict) -> list:
    return [{"task": task, "metric": metric, "cell": names[c], **describe(wide[c])} for c in wide.columns]


def test_rows(task: str, metric: str, rows: list, ph: pd.DataFrame, extra: dict = None) -> list:
    out = [{"task": task, "metric": metric, **(extra or {}), "test": f"ANOVA {r['effect']}", "F": r["F"],
            "df1": r["df1_rep"], "df2": r["df2_rep"], "p": r["p_rep"], "gg": r["gg"],
            "partial_eta2": r["partial_eta2"], "mauchly_p": r["mauchly_p"]} for r in rows]
    out += [{"task": task, "metric": metric, **(extra or {}), "test": r["comparison"], "F": r["F"], "df1": 1,
             "df2": r["df2"], "p": r["p"], "p_holm": r["p_holm"], "p_wilcoxon_holm": r["p_wilcoxon_holm"]}
            for _, r in ph.iterrows()]
    return out


# ------------------------------------------------------------------ dati ---

def training_data() -> tuple:
    trials = pd.read_csv(ROOT / "reaching_regions" / "statistics" / "training_trials.csv")
    trials["block32"] = (trials["target_number"] - 1) // BLOCK + 1
    blocks = trials.groupby(["subject", "training", "block32"])[list(TRAINING)].mean()
    return trials, blocks


def training_wide(blocks: pd.DataFrame, m: str) -> pd.DataFrame:
    w = blocks[m].unstack(["training", "block32"])
    return w[[(tr, b) for tr in ("T1", "T2") for b in BLOCKS]]


def blind_trials() -> pd.DataFrame:
    return pd.read_csv(ROOT / "reaching_blind" / "statistics" / "blind_trials.csv")


def blind_wide(trials: pd.DataFrame, m: str) -> pd.DataFrame:
    """soggetti x test: media dei 3 blocchi (valore di un blocco = media delle sue 4 prove)."""
    blocks = trials.groupby(["subject", "test", "block"])[m].mean().unstack(["test", "block"])
    return pd.DataFrame({t: blocks[[(t, b) for b in (1, 2, 3)]].mean(axis=1) for t in TESTS})


def needs_log(wides: list) -> tuple:
    """Shapiro-Wilk sulle differenze fra test 2 e test 1 di ogni fase (con 2 livelli = i residui del modello):
    log se in una fase p < 0.05 e i valori sono tutti positivi. Restituisce (log, [p di Shapiro])."""
    ps = [float(shapiro((w[2] - w[1]).dropna())[1]) for w in wides]
    return (min(ps) < 0.05 and all((w > 0).all().all() for w in wides)), ps


def robot_subjects() -> pd.DataFrame:
    return pd.read_csv(ROOT / "real_robot" / "statistics" / "robot_subjects.csv")


def exploratory_wide(subjects: pd.DataFrame, m: str, leg: str) -> pd.DataFrame:
    d = subjects[(subjects.metric == m) & (subjects.phase == leg)]
    return d.pivot(index="subject", columns="run", values="value")[[1, 2]]


def robot_pair(report: Report, label: str, wides: dict, log, desc: list, tests: list, metric: str) -> dict:
    """Test 1 vs test 2 in due fasi corrette insieme con Holm. log: True/False, o None = deciso da Shapiro.
    Restituisce {fase: (riga dell'ANOVA, p Holm)}."""
    sw = None
    if log is None:
        log, sw = needs_log(list(wides.values()))
    rows = {ph: anova(w, ["Test"], log)[0] for ph, w in wides.items()}
    report.text(label + (" (tests on log-transformed data)" if log else "")
                + (f" – Shapiro on the differences: {', '.join(f'{x:.3f}' for x in sw)}" if sw else ""))
    out = {}
    for (ph, w), ph_holm in zip(wides.items(), holm([r["p"] for r in rows.values()])):
        r, wx = rows[ph], wilcoxon_paired(w[1], w[2])
        out[ph] = (r, ph_holm)
        report.text(f"  {ph:<15} test 1 {describe(w[1])['mean']:.3f} ± {describe(w[1])['sd']:.3f}, test 2 "
                    f"{describe(w[2])['mean']:.3f} ± {describe(w[2])['sd']:.3f} | F(1, {r['df2']:.0f}) = {r['F']:.2f}, "
                    f"p {p_text(r['p'])}, p Holm {p_text(ph_holm)}{stars_text(ph_holm)}, eta2p = "
                    f"{r['partial_eta2']:.2f} | Wilcoxon p {p_text(wx['p'])}")
        desc += describe_rows("robot", metric, w, {1: f"{ph} Test 1", 2: f"{ph} Test 2"})
        tests.append({"task": "robot", "metric": metric, "leg": ph, "test": "ANOVA Test", "F": r["F"], "df1": 1,
                      "df2": r["df2"], "p": r["p"], "p_holm": ph_holm, "partial_eta2": r["partial_eta2"], "log": log})
    report.text()
    return out


def robot_wide(subjects: pd.DataFrame, m: str, leg: str) -> pd.DataFrame:
    """soggetti x test (1, 2) di una fase; tutti i soggetti, anche senza valore (NaN)."""
    d = subjects[(subjects.metric == m) & (subjects.phase == leg)]
    return (d.pivot(index="subject", columns="run", values="value")
            .reindex(index=sorted(subjects["subject"].unique()), columns=[1, 2]))


# ------------------------------------------------------------------ grafici ---

def block_x(tr, b):
    """Posizione dei box del training: i 3 blocchi del training 1, uno spazio, quelli del 2."""
    return (b - 1) + (3.6 if tr == "T2" else 0)


def training_box_panel(ax, wide: pd.DataFrame, m: str, ph: pd.DataFrame, colors: dict):
    _, label, unit, _, _ = TRAINING[m]
    width = 0.55
    for tr in ("T1", "T2"):
        face, edge = TRAINING_BOX[tr]
        for b in BLOCKS:
            v = wide[(tr, b)].dropna()
            x = block_x(tr, b)
            ax.boxplot([v.to_numpy()], positions=[x], widths=width, patch_artist=True, showfliers=False,
                       whis=1.5, manage_ticks=False, zorder=2,
                       boxprops=dict(facecolor=face, edgecolor=edge, linewidth=1.2),
                       medianprops=dict(color=edge, linewidth=1.6),
                       whiskerprops=dict(color=edge, linewidth=1.0, linestyle=(0, (4, 3))),
                       capprops=dict(color=edge, linewidth=1.0))
            jit = _jitter(v.index, colors, width * 0.55)
            ax.scatter([x + jit[s] for s in v.index], v.to_numpy(), s=14, c=[colors[s] for s in v.index],
                       linewidths=0, zorder=3)
    if m == "extent_error":
        ax.axhline(0, color="#8c8c8c", lw=0.8, ls=":", zorder=0)
    xs = [block_x(tr, b) for tr in ("T1", "T2") for b in BLOCKS]
    ax.set_xticks(xs)
    ax.set_xticklabels([f"B{b}" for _ in ("T1", "T2") for b in BLOCKS])
    ax.set_xlim(-0.6, block_x("T2", 3) + 0.6)
    for tr, name in (("T1", "Training 1"), ("T2", "Training 2")):
        ax.annotate(name, xy=(block_x(tr, 2), 0), xycoords=("data", "axes fraction"), xytext=(0, -24),
                    textcoords="offset points", ha="center", va="top", color=TRAINING_COLORS[tr],
                    fontweight="bold")
    ax.set_ylabel(axis_label(label, "" if unit == "-" else unit))
    brackets_out(ax, [(block_x(*r["a"]), block_x(*r["b"]), r["p_holm"]) for _, r in ph.iterrows()])


def training_curve_panel(ax, trials: pd.DataFrame, m: str):
    """Curva di apprendimento senza parentesi: gli asterischi dei confronti fra blocchi sono nei box plot."""
    _, label, unit, _, _ = TRAINING[m]
    ta.figure4_panel(ax, trials, m, label, unit)


def brackets_out(ax, items):
    """Parentesi senza sovrapposizioni, a livelli sopra i dati (anche fuori dall'asse)."""
    brackets(ax, items)


def panel_title(ax, text):
    """Titolo (A), (B)... sopra le eventuali parentesi uscite dall'asse."""
    ax.set_title(text, pad=getattr(ax, "_title_pad", 6.0))


def two_line_ylabel(ax):
    """Etichette lunghe nelle figure della tesi (pannelli stretti): a capo solo l'unita' di misura."""
    _plots.wrap_unit(ax)


def stacked_figure(row_levels: list, ncols=2, width=7.6, axes_h=1.9, below=0.62, bottom=0.5, left=0.9,
                   right=0.12, wgap=1.0):
    """Figura della tesi impaginata a mano, riga per riga: sopra ogni riga lo spazio per le parentesi
    (row_levels = livelli di parentesi della riga, il massimo fra i suoi pannelli) e per il titolo,
    sotto ogni riga `below` pollici per etichette e nome dell'asse x, in fondo `bottom` per la legenda.
    Cosi' tante parentesi allungano la figura invece di schiacciare i dati. Restituisce (fig, righe di
    assi, frazione dell'altezza occupata dal fondo)."""
    tops = [(_plots.bracket_space_pt(lv) + 18) / 72 for lv in row_levels]   # + titolo (A), (B)...
    height = sum(tops) + len(row_levels) * (axes_h + below) + bottom
    fig = plt.figure(figsize=(width, height))
    axw = (width - left - right - (ncols - 1) * wgap) / ncols
    y, rows = height, []
    for t in tops:
        y -= t + axes_h
        rows.append([fig.add_axes([(left + c * (axw + wgap)) / width, y / height, axw / width, axes_h / height])
                     for c in range(ncols)])
        y -= below
    return fig, rows, bottom / height


def plot_training_effect_curves(trials, anovas):
    """Curve singole per le slide (movement duration e direction error): senza asterischi e senza note sotto.
    (Prima avevano la parentesi dell'effetto Training dell'ANOVA, tolta su richiesta di Elisa.)"""
    for m in ("movement_duration", "direction_error"):   # le due che servono alla presentazione
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        training_curve_panel(ax, trials, m)
        save_figure(fig, OUT["training"], f"single/{m}_curve_training_effect")


def plot_training(trials, wides, posthocs, colors):
    handles = subject_handles(colors)
    metrics = list(TRAINING)
    for kind in ("boxplots", "curves"):
        lv = [_plots.bracket_levels([(block_x(*r["a"]), block_x(*r["b"]), r["p_holm"])
                                     for _, r in posthocs[m].iterrows()]) if kind == "boxplots" else 0
              for m in metrics]
        row_levels = [max(lv[i:i + 2]) for i in range(0, len(lv), 2)]
        fig, rows, bottom = stacked_figure(row_levels, below=0.75 if kind == "boxplots" else 0.6,
                                           bottom=0.55 if kind == "boxplots" else 0.05)
        axes = [ax for row in rows for ax in row]
        for ax in axes[len(metrics):]:
            ax.set_visible(False)
        for ax, m, letter in zip(axes, metrics, "ABCDE"):
            if kind == "boxplots":
                training_box_panel(ax, wides[m], m, posthocs[m], colors)
            else:
                training_curve_panel(ax, trials, m)
            panel_title(ax, f"({letter})")
            two_line_ylabel(ax)
        if kind == "boxplots":
            subject_legend(fig, colors, ncol=5, y=bottom * 0.9)
        save_thesis_figure(fig, f"training_{kind}{SUFFIX}", note=BOX_NOTE if kind == "boxplots" else CURVE_NOTE)
    for m in metrics:
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        training_box_panel(ax, wides[m], m, posthocs[m], colors)
        side_legend(ax, handles)
        save_figure(fig, OUT["training"], f"single/{m}_boxplot", note=BOX_NOTE)
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        training_curve_panel(ax, trials, m)
        save_figure(fig, OUT["training"], f"single/{m}_curve", note=CURVE_NOTE)


def blind_panel(ax, wide, m, ph, colors, numbers=False):
    _, label, unit, _, _ = BLIND[m]
    groups = [f"Test {t}" for t in TESTS]
    xpos = condition_plot(ax, {(f"Test {t}", "all"): wide[t] for t in TESTS}, groups, ["all"],
                          {"all": GREY_BOX}, colors)
    if numbers:   # figura della tesi: pannelli stretti, solo il numero del test
        ax.set_xticklabels([str(t) for t in TESTS])
        ax.set_xlabel("Test")
    else:
        ax.set_xticklabels(groups)
    ax.set_ylabel(axis_label(label, unit))
    if m == "extent_error":
        ax.axhline(0, color="#8c8c8c", lw=0.8, ls=":", zorder=0)
    brackets_out(ax, [(xpos[(f"Test {r['a']}", "all")], xpos[(f"Test {r['b']}", "all")], r["p_holm"])
                  for _, r in ph.iterrows()])


def plot_blind(wides, posthocs, colors):
    lv = [_plots.bracket_levels([(a - 1, b - 1, p) for (_, r), (a, b, _) in zip(posthocs[m].iterrows(), BLIND_PAIRS)
                                 for p in [r["p_holm"]]]) for m in BLIND_PLOTTED]
    fig, rows, bottom = stacked_figure([max(lv[0:2]), max(lv[2:4])], below=0.6, bottom=0.55)
    for ax, m, letter in zip([ax for row in rows for ax in row], BLIND_PLOTTED, "ABCD"):
        blind_panel(ax, wides[m], m, posthocs[m], colors, numbers=True)
        panel_title(ax, f"({letter})")
        two_line_ylabel(ax)
    subject_legend(fig, colors, ncol=5, y=bottom * 0.9)
    save_thesis_figure(fig, f"blind_boxplots{SUFFIX}", note=BOX_NOTE)
    for m in BLIND:
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        blind_panel(ax, wides[m], m, posthocs[m], colors)
        side_legend(ax, subject_handles(colors))
        save_figure(fig, OUT["blind"], f"single/{m}_boxplot", note=BOX_NOTE)


def robot_panel(ax, wide, m, leg, p, colors, ylim=None, draw=True):
    """Test 1 vs test 2 in una fase. ylim: scala y da fissare prima delle parentesi (che si posizionano sui
    dati). draw=False: le parentesi non vengono disegnate, restituisce gli elementi per brackets_out."""
    _, label, unit, _, _ = ROBOT[m]
    xpos = condition_plot(ax, {(LEGS[leg], "Test 1"): wide[1], (LEGS[leg], "Test 2"): wide[2]}, [LEGS[leg]],
                          ["Test 1", "Test 2"], {"Test 1": GREY_BOX, "Test 2": RED_BOX}, colors)
    ax.set_xticklabels([LEGS[leg]])
    ax.set_ylabel(axis_label(label, unit))
    if ylim is not None:
        ax.set_ylim(ylim)
    items = [(xpos[(LEGS[leg], "Test 1")], xpos[(LEGS[leg], "Test 2")], p)]
    if draw:
        brackets_out(ax, items)
    return items


def phases_panel(ax, wides, phases, p, colors):
    """Completion time di piu' fasi nello stesso pannello (andata + ritorno; grasp e posizionamento)."""
    data = {}
    for ph in phases:
        data[(CT_OTHER[ph], "Test 1")], data[(CT_OTHER[ph], "Test 2")] = wides[ph][1], wides[ph][2]
    groups = [CT_OTHER[ph] for ph in phases]
    xpos = condition_plot(ax, data, groups, ["Test 1", "Test 2"], {"Test 1": GREY_BOX, "Test 2": RED_BOX}, colors)
    ax.set_xticklabels(groups)
    ax.set_ylabel(axis_label("Completion time", "s"))
    brackets_out(ax, [(xpos[(g, "Test 1")], xpos[(g, "Test 2")], p[ph]) for g, ph in zip(groups, phases)])


def plot_robot(wides, pholm, ct_other, p_other, colors):
    """Figura della tesi con la stessa disposizione di quella dell'analisi del robot: parte 1 = completion
    time di andata + ritorno (A) e di grasp e posizionamento (B), completion time (C, D) e path length
    (E, F) di andata e ritorno; parte 2 = mean speed (G, H), jerk (I, J), region transitions (K, L)."""
    run_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=f, edgecolor=e, lw=1.2, label=f"Test {r}")
                   for r, (f, e) in zip((1, 2), (GREY_BOX, RED_BOX))]
    ylim = {}
    letters = iter("ABCDEFGHIJKL")
    for part, metrics in enumerate(ROBOT_PARTS, 1):
        n = len(metrics) + (part == 1)
        row_levels = ([1] if part == 1 else []) + [1] * len(metrics)   # al massimo una parentesi per pannello
        fig, axes, bottom = stacked_figure(row_levels, axes_h=1.75, below=0.4, bottom=0.55)
        rows = axes[1:] if part == 1 else axes
        if part == 1:
            phases_panel(axes[0][0], ct_other, ["andata_ritorno"], p_other, colors)
            panel_title(axes[0][0], f"({next(letters)})")
            phases_panel(axes[0][1], ct_other, ["grasp", "posizionamento"], p_other, colors)
            panel_title(axes[0][1], f"({next(letters)})")
        for row, m in zip(rows, metrics):
            # prima la scala y comune ad andata e ritorno, poi parentesi e titoli
            items = [robot_panel(row[col], wides[(m, leg)], m, leg, pholm[(m, leg)], colors, draw=False)
                     for col, leg in enumerate(LEGS)]
            ylim[m] = share_ylim(row)
            for ax, it in zip(row, items):
                brackets_out(ax, it)
                panel_title(ax, f"({next(letters)})")
        subject_legend(fig, colors, ncol=6, y=bottom * 0.9, extra=run_handles)
        save_thesis_figure(fig, f"robot_boxplots{SUFFIX}_{part}", note=BOX_NOTE)
    for m in ROBOT:
        for leg in LEGS:
            fig, ax = plt.subplots(figsize=(3.1, 3.6))
            robot_panel(ax, wides[(m, leg)], m, leg, pholm[(m, leg)], colors, ylim=ylim[m])
            side_legend(ax, run_handles + subject_handles(colors))
            save_figure(fig, OUT["robot"], f"single/{m}_{'outbound' if leg == 'andata' else 'return'}_boxplot",
                        note=BOX_NOTE)
    # i due pannelli del completion time con piu' fasi
    for phases, name in ((["andata_ritorno"], "completion_time_outbound_return"),
                         (["grasp", "posizionamento"], "completion_time_grasp_placement")):
        fig, ax = plt.subplots(figsize=(1.9 + 1.2 * len(phases), 3.6))
        phases_panel(ax, ct_other, phases, p_other, colors)
        side_legend(ax, run_handles + subject_handles(colors))
        save_figure(fig, OUT["robot"], f"single/{name}_boxplot", note=BOX_NOTE)


# ------------------------------------------------------------------ tabelle LaTeX ---

def tex_ms(v: pd.Series, digits: int) -> str:
    d = describe(v)
    return f"${d['mean']:.{digits}f} \\pm {d['sd']:.{digits}f}$"


def tex_p(p, eta=None, gg=False) -> str:
    s = "${<}0.001$" if p < 0.001 else f"${p:.3f}$"
    if gg:
        s += "\\textsuperscript{a}"
    return s + (f" (${eta:.2f}$)" if eta is not None else "")


def tex_header(label: str, unit: str, log: bool) -> str:
    words = label.split()
    first, second = (" ".join(words[:-1]), words[-1]) if len(words) > 1 else (label, "")
    if unit and unit != "-":
        unit_tex = unit.replace("%", "\\%").replace("m/s³", "m/s$^3$")
        second = f"{second} ({unit_tex})".strip()
    mark = "\\textsuperscript{b}" if log else ""
    return (f"\\begin{{tabular}}[b]{{@{{}}c@{{}}}}\\textbf{{{first}}}\\\\"
            f"\\textbf{{\\boldmath {second}}}{mark}\\end{{tabular}}")


def write_table(name: str, cols: list, body: list, caption: str, short: str, label: str, colsep=3, prefix=()):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["% generata da new_stats/new_stats_analysis.py: non modificare a mano", "\\begin{table}[tbp]",
             "    \\centering", "    \\small", f"    \\setlength{{\\tabcolsep}}{{{colsep}pt}}", *prefix,
             FIT, f"    \\begin{{tabular}}{{l{'c' * len(cols)}}}", "        \\toprule",
             "        & " + " & ".join(cols) + " \\\\", "        \\midrule"]
    lines += [f"        {row}" for row in body]
    lines += ["        \\bottomrule", "    \\end{tabular}}", f"    \\caption[{short}]{{{caption}}}",
              f"    \\label{{{label}}}", "\\end{table}", ""]
    path = TABLES_DIR / f"{name}{SUFFIX}.tex"   # TABLES_DIR scelto in main
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[{name}] tabella tesi -> {path}")


# rimpicciolisce la tabella solo se e' piu' larga del testo (come le tabelle dell'appendice)
FIT = "    \\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%"


def row(name: str, cells: list) -> str:
    return f"\\textbf{{\\boldmath {name}}} & " + " & ".join(cells) + " \\\\"


def section(title: str, ncols: int) -> str:
    return f"\\multicolumn{{{ncols + 1}}}{{l}}{{\\textbf{{\\boldmath {title}}}}} \\\\"


NOTES = ("\\textsuperscript{a}~Greenhouse--Geisser-corrected (Mauchly's test $p < 0.05$). "
         "\\textsuperscript{b}~Tests on log-transformed data (values in the original units).")


def training_table(wides, anovas, posthocs):
    ms = list(TRAINING)
    cols = [tex_header(TRAINING[m][1].replace(" at 1 s", ""), TRAINING[m][2], TRAINING[m][3]) for m in ms]
    body = [row(f"{tr}, B{b}", [tex_ms(wides[m][(tr, b)], TRAINING[m][4]) for m in ms])
            for tr in ("T1", "T2") for b in BLOCKS]
    body += ["\\midrule", section("ANOVA, $p$ ($\\eta^2_p$)", len(ms))]
    for i, eff in enumerate(("Training", "Block", "Interaction")):
        body.append(row(eff, [tex_p(anovas[m][i]["p_rep"], anovas[m][i]["partial_eta2"], anovas[m][i]["gg"])
                              for m in ms]))
    write_table("training", cols, body,
                "Metrics of the reaching training in the three blocks of $32$ outward movements (B1--B3) of "
                "training 1 (T1) and training 2 (T2), errors at $1$~s (mean $\\pm$ SD across participants); "
                "$p$-values of the repeated-measures ANOVA Training $\\times$ Block, with the partial eta squared "
                "in brackets. " + NOTES,
                "Reaching training: results", "tab:results_training")


def blind_table(wides, anovas, posthocs):
    ms = BLIND_PLOTTED   # le metriche della tesi (la movement duration del blind non c'e')
    cols = [tex_header(BLIND[m][1].replace(" at 4 s", ""), BLIND[m][2], BLIND[m][3]) for m in ms]
    body = [row(f"Test {t}", [tex_ms(wides[m][t], BLIND[m][4]) for m in ms]) for t in TESTS]
    body += ["\\midrule", section("ANOVA, $p$ ($\\eta^2_p$)", len(ms)),
             row("Test", [tex_p(anovas[m][0]["p_rep"], anovas[m][0]["partial_eta2"], anovas[m][0]["gg"])
                          for m in ms])]
    write_table("blind", cols, body,
                "Metrics of the blind reaching test in its five repetitions, errors at the end of the trial "
                "(mean $\\pm$ SD across participants); $p$-values of the repeated-measures ANOVA on the effect "
                "of Test, with the partial eta squared in brackets. " + NOTES,
                "Blind reaching test: results", "tab:results_blind", colsep=4)


def robot_table(wides, anovas, pholm, ct_other, p_other, a_other):
    phases = ["andata", "grasp", "ritorno", "posizionamento", "totale", "andata_ritorno"]
    names = {"andata": "Outbound", "ritorno": "Return", **CT_OTHER}
    w = {ph: wides[("completion_time", ph)] if ph in LEGS else ct_other[ph] for ph in phases}
    p = {ph: pholm[("completion_time", ph)] if ph in LEGS else p_other[ph] for ph in phases}
    eta = {ph: (anovas[("completion_time", ph)] if ph in LEGS else a_other[ph])["partial_eta2"] for ph in phases}
    head = ["\\textbf{%s}" % names[ph] if " + " not in names[ph] else
            "\\begin{tabular}[b]{@{}c@{}}\\textbf{Outbound}\\\\\\textbf{+ return}\\end{tabular}" for ph in phases]
    top = [FIT, "    \\begin{tabular}{l%s}" % ("c" * len(phases)), "        \\toprule",
           "        " + section("Completion time (s)\\textsuperscript{b}", len(phases)),
           "        & " + " & ".join(head) + " \\\\", "        \\midrule"]
    top += ["        " + row(f"Test {t}", [tex_ms(w[ph][t], 0) for ph in phases]) for t in (1, 2)]
    top += ["        " + row("$p$ ($\\eta^2_p$)", [tex_p(p[ph], eta[ph]) for ph in phases]),
            "        \\bottomrule", "    \\end{tabular}}", "", "    \\vspace{1em}"]
    ms = [m for m in ROBOT if m != "completion_time"]
    cols = [tex_header(ROBOT[m][1], ROBOT[m][2], ROBOT[m][3]) for m in ms]
    body = []
    for i, leg in enumerate(LEGS):
        if i:
            body.append("\\midrule")
        body.append(section(f"{LEGS[leg]} leg", len(ms)))
        for t in (1, 2):
            body.append(row(f"Test {t}", [tex_ms(wides[(m, leg)][t], ROBOT[m][4]) for m in ms]))
        body.append(row("$p$ ($\\eta^2_p$)", [tex_p(pholm[(m, leg)], anovas[(m, leg)]["partial_eta2"]) for m in ms]))
    write_table("robot", cols, body,
                "Metrics of the test on the real robot in the two tests (mean $\\pm$ SD across participants): "
                "completion time of each phase, of the whole test and of the two driving legs together (top), and "
                "metrics of the outbound and return legs (bottom). For every metric, $p$-value of the "
                "repeated-measures ANOVA between the tests, with the partial eta squared in brackets, Holm-corrected for the two "
                "legs of each metric and, for the completion time, within the pairs grasp--placement and "
                "total--outbound + return. \\textsuperscript{b}~Tests on log-transformed data (values in the original units).",
                "Test on the real robot: results", "tab:results_robot", prefix=top)


# ------------------------------------------------------------------ main ---

def main():
    global TABLES_DIR
    parser = argparse.ArgumentParser(description="Second analysis (newStats of the PhD student)")
    parser.add_argument("--thesis", action="store_true",
                        help="scrive figure e tabelle direttamente nella cartella della tesi (di default no)")
    args = parser.parse_args()
    if args.thesis:
        TABLES_DIR = THESIS_DIR.parent.parent / "Tables"
    else:
        LOCAL_THESIS.mkdir(exist_ok=True)
        _plots.THESIS_DIR = LOCAL_THESIS / "figures"
        TABLES_DIR = LOCAL_THESIS / "tables"
    for out in OUT.values():
        out.mkdir(parents=True, exist_ok=True)
    setup_style()

    def new_report(task: str) -> Report:
        r = Report(f"SECOND ANALYSIS (newStats of the PhD student): {task}")
        r.text("Same data as the task analyses. GG = Greenhouse-Geisser (when Mauchly p < 0.05); p of the")
        r.text("post hoc comparisons between all pairs, Holm-corrected within each metric; log only for the tests.")
        return r

    report = new_report("reaching training")

    # --- training
    trials, blocks = training_data()
    colors = subject_colors(trials["subject"].unique())
    report.section("REACHING TRAINING – Training (T1, T2) x Block (3 blocks of 32 outward movements)")
    wides, anovas, posthocs, desc, tests = {}, {}, {}, [], []
    for m, (abbr, label, unit, log, _) in TRAINING.items():
        w = training_wide(blocks, m)
        wides[m], anovas[m], posthocs[m] = w, anova(w, ["Training", "Block"], log), pairwise(w, TRAINING_PAIRS, log)
        report.text(f"{abbr} – {label}{f' ({unit})' if unit != '-' else ''}")
        for (tr, b) in w.columns:
            d = describe(w[(tr, b)])
            report.text(f"  {tr} B{b}: {d['mean']:.3f} ± {d['sd']:.3f}")
        write_anova(report, anovas[m], log)
        fr = friedman(w)
        report.text(f"  Friedman (6 blocks): chi2({fr['df']}) = {fr['statistic']:.2f}, p {p_text(fr['p'])}")
        write_pairwise(report, posthocs[m])
        report.text()
        desc += describe_rows("training", m, w, {c: f"{c[0]} B{c[1]}" for c in w.columns})
        tests += test_rows("training", m, anovas[m], posthocs[m])
    pd.DataFrame(desc).to_csv(OUT["training"] / "training_descriptives.csv", index=False)
    pd.DataFrame(tests).to_csv(OUT["training"] / "training_tests.csv", index=False)
    plot_training(trials, wides, posthocs, colors)
    plot_training_effect_curves(trials, anovas)
    training_table(wides, anovas, posthocs)

    # --- blind
    report.save(OUT["training"])
    report = new_report("blind reaching test")
    report.section("BLIND TEST – one value per test (mean of the 12 trials), Test 1-5")
    report.text("Linearity index left out until the anomalous trial is corrected.")
    bw, ba, bp, desc, tests = {}, {}, {}, [], []
    trials = blind_trials()
    for m, (abbr, label, unit, log, _) in BLIND.items():
        w = blind_wide(trials, m)
        bw[m], ba[m], bp[m] = w, anova(w, ["Test"], log), pairwise(w, BLIND_PAIRS, log)
        report.text(f"{abbr} – {label} ({unit})")
        report.text("  " + "  ".join(f"T{t}: {describe(w[t])['mean']:.2f} ± {describe(w[t])['sd']:.2f}" for t in TESTS))
        write_anova(report, ba[m], log)
        fr = friedman(w)
        report.text(f"  Friedman: chi2({fr['df']}) = {fr['statistic']:.2f}, p {p_text(fr['p'])}")
        write_pairwise(report, bp[m])
        report.text()
        desc += describe_rows("blind", m, w, {t: f"Test {t}" for t in TESTS})
        tests += test_rows("blind", m, ba[m], bp[m])
    report.section("EXPLORATORY (in the thesis): trials ending in the region of the target (%)")
    w = trials.groupby(["subject", "test"])["region_correct"].mean().unstack("test")[list(TESTS)]
    report.text("  " + "  ".join(f"T{t}: {describe(w[t])['mean']:.1f} ± {describe(w[t])['sd']:.1f}" for t in TESTS))
    write_anova(report, anova(w, ["Test"], False), False)
    fr = friedman(w)
    report.text(f"  Friedman: chi2({fr['df']}) = {fr['statistic']:.2f}, p {p_text(fr['p'])}")
    write_pairwise(report, pairwise(w, BLIND_PAIRS, False))
    pd.DataFrame(desc).to_csv(OUT["blind"] / "blind_descriptives.csv", index=False)
    pd.DataFrame(tests).to_csv(OUT["blind"] / "blind_tests.csv", index=False)
    plot_blind(bw, bp, colors)
    blind_table(bw, ba, bp)

    # --- robot
    report.save(OUT["blind"])
    report = new_report("real robot")
    report.section("REAL ROBOT – test 1 vs test 2, outbound and return (Holm x2 per metric)")
    rw, ra, pholm, desc, tests = {}, {}, {}, [], []
    subjects = robot_subjects()
    for m, (abbr, label, unit, log, _) in ROBOT.items():
        wides = {leg: robot_wide(subjects, m, leg) for leg in LEGS}
        res = robot_pair(report, f"{abbr} – {label}{f' ({unit})' if unit else ''}", wides, log, desc, tests, m)
        for leg in LEGS:
            rw[(m, leg)], (ra[(m, leg)], pholm[(m, leg)]) = wides[leg], res[leg]
    report.section("COMPLETION TIME OF THE OTHER PHASES (log, Holm x2 within each pair)")
    ct_other, p_other, a_other = {}, {}, {}
    for pair in CT_PAIRS:
        wides = {ph: robot_wide(subjects, "completion_time", ph) for ph in pair}
        res = robot_pair(report, " and ".join(CT_OTHER[ph] for ph in pair), wides, True, desc, tests,
                         "completion_time")
        for ph in pair:
            ct_other[ph], (a_other[ph], p_other[ph]) = wides[ph], res[ph]
    w = ct_other["grasp"].drop(GRASP_WITHOUT)
    r, wx = anova(w, ["Test"], True)[0], wilcoxon_paired(w[1], w[2])
    report.text(f"Grasp without {GRASP_WITHOUT} (single check, log, not corrected): test 1 {describe(w[1])['mean']:.1f} "
                f"± {describe(w[1])['sd']:.1f}, test 2 {describe(w[2])['mean']:.1f} ± {describe(w[2])['sd']:.1f} | "
                f"F(1, {r['df2']:.0f}) = {r['F']:.2f}, p {p_text(r['p'])}, eta2p = {r['partial_eta2']:.2f} | "
                f"Wilcoxon p {p_text(wx['p'])}")
    report.section("EXPLORATORY METRICS OF THE THESIS (Holm x2, outbound and return)")
    for m, (label, unit) in ROBOT_EXPLORATORY.items():
        robot_pair(report, f"{label}{f' ({unit})' if unit else ''}", {leg: exploratory_wide(subjects, m, leg) for leg in LEGS},
                   None, desc, tests, m)
    pd.DataFrame(desc).to_csv(OUT["robot"] / "robot_descriptives.csv", index=False)
    pd.DataFrame(tests).to_csv(OUT["robot"] / "robot_tests.csv", index=False)
    plot_robot(rw, pholm, ct_other, p_other, colors)
    robot_table(rw, ra, pholm, ct_other, p_other, a_other)

    report.save(OUT["robot"])
    plot_ppt(bw, bp, colors)


# grafici per la presentazione in plot_ppt/: solo nome dell'asse con l'unita', grafico, asterischi e
# Test 1, 2, ...; niente legenda dei partecipanti ne' note sotto (richiesta di Elisa)
PPT_DIR = BASE_DIR / "plot_ppt"
PPT_BLIND = ["euclidean_error", "extent_error", "direction_error"]
PPT_CURVES = {"training_movement_duration_curve": "movement_duration_curve_training_effect",
              "training_direction_error_curve": "direction_error_curve_training_effect"}


def plot_ppt(blind_wides, blind_posthocs, colors):
    PPT_DIR.mkdir(exist_ok=True)
    for m in PPT_BLIND:
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        blind_panel(ax, blind_wides[m], m, blind_posthocs[m], colors)
        fig._has_stars = False   # niente legenda degli asterischi sotto
        png = save_figure(fig, PPT_DIR, f"blind_{m}")   # save_figure scrive in PPT_DIR/plots: sposta su
        for f in (png, png.with_suffix(".pdf")):
            f.replace(PPT_DIR / f.name)
    shutil.rmtree(PPT_DIR / "plots", ignore_errors=True)
    for name, src in PPT_CURVES.items():   # le curve senza asterischi e senza note: copiate
        for ext in (".png", ".pdf"):
            shutil.copy(OUT["training"] / "plots" / "single" / f"{src}{ext}", PPT_DIR / f"{name}{ext}")


if __name__ == "__main__":
    main()
