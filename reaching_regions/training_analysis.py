"""Training a 9 regioni (reaching_regions.py), fatto due volte: training 1 (pre) e
training 2 (post). Statistica solo sulle ANDATE (centro -> target), 96 per training; dei
RITORNI al centro (home) ci sono solo i grafici di movement duration e linearity index.

Input: <soggetto>_regions_<pre|post>[_k]_raw.csv (+ _summary.json per hidden_s), ovunque
dentro questa cartella (es. reaching_regions/results/). Se una sessione e' stata
ripetuta si usa l'ultima.

Metriche di ogni andata, definite come Bellitto et al. 2025 (par. 2.4.2) e
computereachmetrics.m (vedi common/reach_metrics.py), misurate dalla posizione del
cursore quando compare il target:
    EE  Euclidean error a 1 s (cm)   all'istante in cui il cursore ricompare (hidden_s)
    DE  Direction error a 1 s (deg)
    XE  Extent error a 1 s (cm)
    MD  Movement duration (s)        comparsa del target -> entrata nel target
    LI  Linearity index (-)          sullo stesso tratto
Blocchi di 8 target: 12 per training. Valore di un blocco = media degli 8 target.
Ritorno: il goal home che segue ogni target (il primo home di ogni blocco non e' un ritorno);
MD dalla comparsa della home all'entrata, LI sullo stesso tratto. Nessuna statistica.

Analisi (per metrica), come il modello jamovi della dottoranda:
    ANOVA a misure ripetute Training (T1, T2) x Blocco (B1-B12), Greenhouse-Geisser;
    post hoc pianificati, t appaiati con Bonferroni (controllo: Wilcoxon con Holm):
        T1 B1 vs T1 B12   apprendimento nel training 1
        T1 B12 vs T2 B1   mantenimento fra i due training
        T1 B12 vs T2 B12  fine training 1 vs fine training 2
    controllo non parametrico: Friedman sui 12 blocchi di ogni training.

Output:
    statistics/report.txt, training_trials.csv (ogni andata), training_blocks.csv (soggetto x
    training x blocco), training_descriptives/anova/friedman/posthoc.csv,
    training_return_trials.csv (ogni ritorno)
    statistics/plots/figure4_median_iqr.(png|pdf)
        come la figura 4 di Bellitto: media mobile su 12 prove, mediana [IQR]
    statistics/plots/return_median_iqr.(png|pdf)  ritorni (MD, LI), stesso stile, senza statistica
    statistics/plots/single/<metrica>.(png|pdf), single/return_<metrica>.(png|pdf)  i pannelli uno per uno
    Tesi_latex/Chapters/Figures/results/training_curves.pdf, training_return_curves.pdf  figure della tesi:
        gli stessi pannelli, due per riga, senza titolo e note
    ../jamovi/training/<sigla>_<metrica>.(omv|csv): un file per metrica come training_reaching_out.omv,
        una riga per soggetto, colonne <sigla>_T1_B1 ... <sigla>_T2_B12 (es. EE_T1_B1)

Uso:
    python3 training_analysis.py [--results DIR] [--out DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import find_sessions   # noqa: E402
from common.jamovi import write_tables   # noqa: E402
from common.plots import (TRAINING_COLORS, MUTED, axis_label, brackets, save_figure, save_thesis_figure,   # noqa: E402
                          setup_style)
from common.reach_metrics import canvas_scale, canvas_to_cm, goals_from_raw, training_metrics   # noqa: E402
from common.report import Report, analyse_metric, save_tables   # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
PATTERN = r"(?P<subject>.+)_regions_(?P<phase>pre|post)(?:_(?P<rep>\d+))?_raw\.csv"
TRAININGS = {"pre": "T1", "post": "T2"}
BLOCK_SIZE = 8
HIDDEN_S_DEFAULT = 1.0
WINDOW = 12          # figura 4: media mobile sulla prova e le 11 precedenti (come il paper)
GAP = 16             # prove di spazio fra i due training nella figura 4

# metrica -> (sigla jamovi, etichetta, unita')
METRICS = {
    "euclidean_error":   ("EE", "Euclidean error at 1 s", "cm"),
    "direction_error":   ("DE", "Direction error at 1 s", "deg"),
    "extent_error":      ("XE", "Extent error at 1 s", "cm"),
    "movement_duration": ("MD", "Movement duration", "s"),
    "linearity_index":   ("LI", "Linearity index", "-"),
}
# ritorno al centro: solo grafici
RETURN_METRICS = {
    "movement_duration": ("MD", "Return movement duration", "s"),
    "linearity_index":   ("LI", "Return linearity index", "-"),
}


def load(root: Path) -> tuple:
    """Una riga per andata (soggetto, training, numero del target 1-96, blocco, metriche) e una
    per ritorno (stesso numero del target da cui parte; solo movement duration e linearity index)."""
    chosen, dropped = find_sessions(root, PATTERN, ("phase",))
    rows, returns, problems = [], [], []
    for (subject, phase), path in sorted(chosen.items()):
        raw = pd.read_csv(path)
        summary = Path(str(path)[:-len("_raw.csv")] + "_summary.json")
        info = json.loads(summary.read_text()) if summary.exists() else {}
        cfg = info.get("config", {})
        hidden_s = float(cfg.get("hidden_s", HIDDEN_S_DEFAULT))
        scale = canvas_scale(info)
        all_goals = goals_from_raw(raw)
        goals = [g for g in all_goals if g["kind"] == "region"]
        if len(goals) % BLOCK_SIZE:
            problems.append(f"{subject} {TRAININGS[phase]}: {len(goals)} targets (not a multiple of {BLOCK_SIZE})")
        for n, g in enumerate(goals, start=1):
            row = {"subject": subject, "training": TRAININGS[phase], "target_number": n,
                   "block": (n - 1) // BLOCK_SIZE + 1}
            row.update(training_metrics(g["t"], canvas_to_cm(g["xy"], scale), canvas_to_cm(g["goal"], scale),
                                        g["t_enter"], hidden_s))
            rows.append(row)
        n = 0
        for prev, g in zip([None] + all_goals[:-1], all_goals):
            if g["kind"] == "region":
                n += 1
            elif prev is not None and prev["kind"] == "region":   # ritorno dopo il target n
                m = training_metrics(g["t"], canvas_to_cm(g["xy"], scale), canvas_to_cm(g["goal"], scale),
                                     g["t_enter"], hidden_s)
                returns.append({"subject": subject, "training": TRAININGS[phase], "target_number": n,
                                "block": (n - 1) // BLOCK_SIZE + 1,
                                **{k: m[k] for k in RETURN_METRICS}})
    return pd.DataFrame(rows), pd.DataFrame(returns), dropped, problems


def block_wide(blocks: pd.DataFrame, m: str) -> pd.DataFrame:
    """soggetti x (training, blocco), in ordine T1 B1..B12, T2 B1..B12."""
    w = blocks.pivot_table(index="subject", columns=["training", "block"], values=m)
    n_blocks = int(blocks["block"].max())
    return w.reindex(columns=pd.MultiIndex.from_product([list(TRAININGS.values()), range(1, n_blocks + 1)]))


def jamovi_tables(blocks: pd.DataFrame, out_dir: Path):
    """Un file per metrica, come training_reaching_out.omv: colonne <sigla>_T1_B1 ... <sigla>_T2_B12."""
    subjects = sorted(blocks["subject"].unique())
    for m, (abbr, label, unit) in METRICS.items():
        w = block_wide(blocks, m)
        cols, desc = {"subject": subjects}, {}
        for (tr, b) in w.columns:
            name = f"{abbr}_{tr}_B{b}"
            cols[name] = w[(tr, b)].reindex(subjects).to_numpy()
            desc[name] = (f"{label}{f' ({unit})' if unit != '-' else ''}, training {tr[1]}, block {b} = mean of "
                          f"targets {(b - 1) * BLOCK_SIZE + 1}-{b * BLOCK_SIZE}")
        n_blocks = int(w.columns.get_level_values(1).max())
        spec = {"title": f"{abbr} – {label}{f' ({unit})' if unit != '-' else ''}",
                "dep_label": f"{label}{f' ({unit})' if unit != '-' else ''}",
                "factors": [("Training", ["T1", "T2"]), ("Blocco", [f"B{b}" for b in range(1, n_blocks + 1)])],
                "cells": [(f"{abbr}_{tr}_B{b}", [tr, f"B{b}"]) for tr, b in w.columns],
                "post_hoc": []}   # niente post hoc: con 12 blocchi le tabelle diventano troppo lunghe
        write_tables(pd.DataFrame(cols), out_dir / f"{abbr}_{m}", desc, [spec])


def analyse(blocks: pd.DataFrame, report: Report) -> tuple:
    results, posthocs = [], {}
    for m, (abbr, label, unit) in METRICS.items():
        w = block_wide(blocks, m)
        n_blocks = w.columns.get_level_values(1).max()
        cells = [((tr, b), f"{tr} B{b}") for tr, b in w.columns]
        pairs = [(("T1", 1), ("T1", n_blocks), f"T1 B1 vs T1 B{n_blocks}"),
                 (("T1", n_blocks), ("T2", 1), f"T1 B{n_blocks} vs T2 B1"),
                 (("T1", n_blocks), ("T2", n_blocks), f"T1 B{n_blocks} vs T2 B{n_blocks}")]
        sets = [(f"training {tr[1]}, {n_blocks} blocks", [(tr, b) for b in range(1, n_blocks + 1)])
                for tr in TRAININGS.values()]
        res = analyse_metric(report, m, label, "" if unit == "-" else unit, w, ["Training", "Block"], cells,
                             pairs, digits=1 if unit == "deg" else 2, friedman_sets=sets)
        results.append(res)
        posthocs[m] = (res["posthoc"], pairs)
    return results, posthocs


# ---------------------------------------------------------------- figura 4 ---

def moving_average(trials: pd.DataFrame, m: str) -> pd.DataFrame:
    """prova x soggetto per training: media mobile della prova e delle WINDOW-1 precedenti,
    ricominciando all'inizio di ogni training. Colonne = soggetti."""
    out = {}
    for tr, g in trials.groupby("training"):
        w = g.pivot(index="target_number", columns="subject", values=m).sort_index()
        out[tr] = w.rolling(WINDOW, min_periods=1).mean()
    return out


def trial_x(tr, trial, n_trials):
    return np.asarray(trial, float) + (n_trials + GAP if tr == "T2" else 0)


def figure4_panel(ax, trials: pd.DataFrame, m: str, label: str, unit: str, posthoc=None):
    """Un pannello: media mobile per soggetto, poi mediana [IQR] tra soggetti, T1 e T2 affiancati.
    posthoc = (tabella, confronti): parentesi dai t appaiati (Bonferroni); None = senza statistica."""
    n_trials = int(trials["target_number"].max())
    for tr, w in moving_average(trials, m).items():
        x = trial_x(tr, w.index, n_trials)
        mid, lo, hi = w.median(axis=1), w.quantile(0.25, axis=1), w.quantile(0.75, axis=1)
        ax.fill_between(x, lo, hi, color=TRAINING_COLORS[tr], alpha=0.22, lw=0)
        ax.plot(x, mid, color=TRAINING_COLORS[tr], lw=1.6)
    if m == "extent_error":
        ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ticks = [1, n_trials // 2, n_trials]
    ax.set_xticks(list(trial_x("T1", ticks, n_trials)) + list(trial_x("T2", ticks, n_trials)))
    labels = ax.set_xticklabels([str(t) for t in ticks] + [str(t + n_trials) for t in ticks])
    # fine del training 1 e inizio del 2 sono vicini: etichette allineate verso l'esterno per non toccarsi
    labels[len(ticks) - 1].set_ha("right")
    labels[len(ticks)].set_ha("left")
    ax.set_xlim(0, 2 * n_trials + GAP + 1)
    ax.set_xlabel("Trial")
    ax.set_ylabel(axis_label(label, unit))
    # etichette dei training sopra i dati, le parentesi dei post hoc sopra le etichette
    lo, top = ax.get_ylim()
    top += 0.1 * (top - lo)
    ax.set_ylim(lo, top)
    for tr, name in (("T1", "Training 1"), ("T2", "Training 2")):
        ax.text(trial_x(tr, n_trials / 2, n_trials), top, name, ha="center", va="top", fontsize=11,
                color=TRAINING_COLORS[tr], fontweight="bold")
    if posthoc is not None:
        ph, pairs = posthoc
        col = "p_t_bonferroni"   # parentesi sempre dai t appaiati (Bonferroni), come i post hoc dell'ANOVA
        centre = lambda tr, b: float(trial_x(tr, (b - 1) * BLOCK_SIZE + (BLOCK_SIZE + 1) / 2, n_trials))
        brackets(ax, [(centre(*a), centre(*b), p) for (a, b, _), p in zip(pairs, ph[col])])


def plot_training(trials: pd.DataFrame, metrics: dict, posthocs: dict, out: Path, name: str, title: str,
                  note: str, single_prefix: str = "", thesis_name: str = None):
    """Figura con un pannello per metrica (3 per riga) e i pannelli uno per uno in single/,
    senza titolo e con `note` (cosa si vede) sotto il grafico. Con `thesis_name` anche la figura
    della tesi: gli stessi pannelli, due per riga, senza titolo."""
    ncols = min(3, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.15 * ncols + 0.05, 3.3 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    for ax, (m, (_, label, unit)), letter in zip(axes, metrics.items(), "ABCDEFGH"):
        figure4_panel(ax, trials, m, label, unit, posthocs.get(m))
        ax.set_title(f"({letter})")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save_figure(fig, out, name)
    for m, (_, label, unit) in metrics.items():
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        figure4_panel(ax, trials, m, label, unit, posthocs.get(m))
        save_figure(fig, out, f"single/{single_prefix}{m}", note=note)
    if thesis_name:
        nrows = int(np.ceil(len(metrics) / 2))
        fig, axes = plt.subplots(nrows, 2, figsize=(7.6, 2.9 * nrows), squeeze=False)
        axes = axes.ravel()
        for ax in axes[len(metrics):]:
            ax.set_visible(False)
        for ax, (m, (_, label, unit)), letter in zip(axes, metrics.items(), "ABCDEFGH"):
            figure4_panel(ax, trials, m, label, unit, posthocs.get(m))
            ax.set_title(f"({letter})")
            # etichette lunghe (es. "Return movement duration (s)") su due righe, se no escono dal pannello
            if len(ax.get_ylabel()) > 24:
                words = ax.get_ylabel().split()
                half = len(words) // 2
                ax.set_ylabel(" ".join(words[:half]) + "\n" + " ".join(words[half:]))
        fig.tight_layout(h_pad=1.4)
        save_thesis_figure(fig, thesis_name, note=note)


def main():
    p = argparse.ArgumentParser(description="Reaching training analysis (outward movements)")
    p.add_argument("--results", type=Path, default=BASE_DIR)
    p.add_argument("--out", type=Path, default=BASE_DIR / "statistics")
    args = p.parse_args()

    trials, returns, dropped, problems = load(args.results)
    if trials.empty:
        print(f"No *_regions_<pre|post>_raw.csv in {args.results}")
        return
    report = Report("REACHING TRAINING – outward movements (centre -> target)")
    subjects = sorted(trials["subject"].unique())
    report.text(f"Participants: {len(subjects)} ({', '.join(subjects)})")
    for tr in TRAININGS.values():
        who = trials[trials.training == tr].groupby("subject")["target_number"].max()
        report.text(f"  {tr}: {len(who)} participants, {int(who.min())}-{int(who.max())} targets each")
    if dropped:
        report.text("Repeated sessions discarded (the last one is used): " + ", ".join(p.name for p in dropped))
    for msg in problems:
        report.text("WARNING: " + msg)
    report.text(f"Blocks of {BLOCK_SIZE} targets; block value = mean of its {BLOCK_SIZE} targets.")
    report.text("Errors at 1 s after the target onset (when the cursor reappears), measured from the cursor")
    report.text("position at target onset; movement duration and linearity index from target onset to target entry.")

    blocks = trials.groupby(["subject", "training", "block"], as_index=False)[list(METRICS)].mean()
    results, posthocs = analyse(blocks, report)

    args.out.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.out / "training_trials.csv", index=False)
    returns.to_csv(args.out / "training_return_trials.csv", index=False)
    blocks.to_csv(args.out / "training_blocks.csv", index=False)
    save_tables(results, args.out, "training")
    jamovi_tables(blocks, ROOT / "jamovi" / "training")

    setup_style()
    curve = "Line: median; band: IQR."
    plot_training(trials, METRICS, posthocs, args.out, "figure4_median_iqr",
                  "Reaching training: cursor control across trials (median [IQR] across participants)", curve,
                  thesis_name="training_curves")
    plot_training(returns, RETURN_METRICS, {}, args.out, "return_median_iqr",
                  "Reaching training: return to the centre (median [IQR] across participants, no statistics)",
                  curve, single_prefix="return_", thesis_name="training_return_curves")
    report.save(args.out)


if __name__ == "__main__":
    main()
