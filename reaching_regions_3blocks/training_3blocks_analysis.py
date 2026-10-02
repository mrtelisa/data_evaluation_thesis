"""Training a 9 regioni, versione a 3 blocchi (richiesta della dottoranda, EE_EuclidianError.omv):
per ogni training si confrontano i PRIMI 3 blocchi (start) con gli ULTIMI 3 blocchi (end).

Dati ricalcolati dai _raw.csv con lo stesso codice del training completo
(reaching_regions/training_analysis.py: load, stesse metriche, blocchi da 8 target):
    valore start = media dei blocchi 1-3 (target 1-24)
    valore end   = media dei blocchi 10-12 (target 73-96)
Nota: nel file della dottoranda l'EE e' il logaritmo naturale, media dei ln dei 3 blocchi
(riprodotto con differenza massima 0.0005); qui, come per gli altri file jamovi, i dati
sono nelle unita' originali.

Analisi (per metrica), come il modello jamovi della dottoranda:
    ANOVA a misure ripetute Training (T1, T2) x Blocco (start, end), post hoc sui due fattori;
    post hoc pianificati, t appaiati con Bonferroni (controllo: Wilcoxon con Holm):
        T1 start vs T1 end   apprendimento nel training 1
        T2 start vs T2 end   apprendimento nel training 2
        T1 end vs T2 start   mantenimento fra i due training
        T1 end vs T2 end     fine training 1 vs fine training 2
    controllo non parametrico: Friedman sulle 4 celle.

Output:
    statistics/report.txt, training_3blocks_values.csv (soggetto x training x fase),
    training_3blocks_descriptives/anova/friedman/posthoc.csv
    statistics/plots/training_3blocks_boxplots.(png|pdf)  un pannello per metrica
    statistics/plots/single/<metrica>_boxplot.(png|pdf)   i pannelli uno per uno
    ../jamovi/training_3blocks/<sigla>_<metrica>.(omv|csv): colonne <sigla>_T1_start, <sigla>_T1_end,
        <sigla>_T2_start, <sigla>_T2_end (come EE_EuclidianError.omv), una riga per soggetto

Uso:
    python3 training_3blocks_analysis.py [--results DIR] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reaching_regions"))
import training_analysis as full   # noqa: E402  (training completo: caricamento e metriche)
from common.jamovi import write_tables   # noqa: E402
from common.plots import (BOX_NOTE, GREY_BOX, RED_BOX, axis_label, brackets, condition_plot, save_figure,   # noqa: E402
                          setup_style, side_legend, subject_colors, subject_handles, subject_legend)
from common.report import Report, analyse_metric, save_tables   # noqa: E402

METRICS = full.METRICS
TRAININGS = list(full.TRAININGS.values())          # T1, T2
N_BLOCKS = 3                                        # blocchi all'inizio e alla fine di ogni training
PHASES = ["start", "end"]
PAIRS = [(("T1", "start"), ("T1", "end"), "T1 start vs T1 end"),
         (("T2", "start"), ("T2", "end"), "T2 start vs T2 end"),
         (("T1", "end"), ("T2", "start"), "T1 end vs T2 start"),
         (("T1", "end"), ("T2", "end"), "T1 end vs T2 end")]
GROUPS = {"T1": "Training 1", "T2": "Training 2"}
STYLES = {"start": GREY_BOX, "end": RED_BOX}


def phase_values(blocks: pd.DataFrame) -> tuple:
    """Una riga per soggetto x training x fase: start = media dei primi N_BLOCKS blocchi,
    end = media degli ultimi N_BLOCKS. Restituisce anche le etichette dei blocchi usati."""
    last = int(blocks["block"].max())
    sets = {"start": range(1, N_BLOCKS + 1), "end": range(last - N_BLOCKS + 1, last + 1)}
    rows = []
    for phase, bl in sets.items():
        sel = blocks[blocks["block"].isin(bl)]
        v = sel.groupby(["subject", "training"], as_index=False)[list(METRICS)].mean()
        v.insert(2, "phase", phase)
        v.insert(3, "n_blocks", sel.groupby(["subject", "training"])["block"].count().to_numpy())
        rows.append(v)
    labels = {p: f"blocks {bl[0]}-{bl[-1]}" for p, bl in sets.items()}
    return pd.concat(rows, ignore_index=True), labels


def wide(values: pd.DataFrame, m: str) -> pd.DataFrame:
    """soggetti x (training, fase), in ordine T1 start, T1 end, T2 start, T2 end."""
    w = values.pivot_table(index="subject", columns=["training", "phase"], values=m)
    return w.reindex(columns=pd.MultiIndex.from_product([TRAININGS, PHASES]))


def jamovi_tables(values: pd.DataFrame, labels: dict, out_dir: Path):
    """Un file per metrica, come EE_EuclidianError.omv: colonne <sigla>_T1_start ... <sigla>_T2_end."""
    subjects = sorted(values["subject"].unique())
    for m, (abbr, label, unit) in METRICS.items():
        w = wide(values, m)
        full_label = axis_label(label, "" if unit == "-" else unit)
        cols, desc = {"subject": subjects}, {}
        for tr, ph in w.columns:
            name = f"{abbr}_{tr}_{ph}"
            cols[name] = w[(tr, ph)].reindex(subjects).to_numpy()
            desc[name] = f"{full_label}, training {tr[1]}, {ph} = mean of {labels[ph]}"
        spec = {"title": f"{abbr} – {full_label}, first vs last {N_BLOCKS} blocks",
                "dep_label": full_label,
                "factors": [("Training", TRAININGS), ("Blocco", PHASES)],
                "cells": [(f"{abbr}_{tr}_{ph}", [tr, ph]) for tr, ph in w.columns],
                "post_hoc": [["Training"], ["Blocco"]]}   # come il modello della dottoranda
        write_tables(pd.DataFrame(cols), out_dir / f"{abbr}_{m}", desc, [spec])


def analyse(values: pd.DataFrame, labels: dict, report: Report) -> tuple:
    results, posthocs = [], {}
    for m, (abbr, label, unit) in METRICS.items():
        w = wide(values, m)
        cells = [((tr, ph), f"{tr} {ph} ({labels[ph]})") for tr, ph in w.columns]
        res = analyse_metric(report, m, label, "" if unit == "-" else unit, w, ["Training", "Block"], cells,
                             PAIRS, digits=1 if unit == "deg" else 2)
        results.append(res)
        posthocs[m] = res["posthoc"]
    return results, posthocs


# ------------------------------------------------------------------ grafici ---

def plot_panel(ax, values, m, ph, colors):
    _, label, unit = METRICS[m]
    w = wide(values, m)
    groups = [GROUPS[tr] for tr in TRAININGS]
    data = {(GROUPS[tr], p): w[(tr, p)] for tr in TRAININGS for p in PHASES}
    xpos = condition_plot(ax, data, groups, PHASES, STYLES, colors)
    ax.set_xticklabels(groups)
    ax.set_ylabel(axis_label(label, "" if unit == "-" else unit))
    if m == "extent_error":
        ax.axhline(0, color="#8c8c8c", lw=0.8, ls=":", zorder=0)
    col = "p_t_bonferroni"   # parentesi sempre dai t appaiati (Bonferroni), come i post hoc dell'ANOVA
    brackets(ax, [(xpos[(GROUPS[a[0]], a[1])], xpos[(GROUPS[b[0]], b[1])], p)
                  for (a, b, _), p in zip(PAIRS, ph[col])])


def phase_handles(labels: dict) -> list:
    return [Patch(facecolor=STYLES[p][0], edgecolor=STYLES[p][1], label=f"{p.capitalize()} ({labels[p]})")
            for p in PHASES]


def plots(values: pd.DataFrame, labels: dict, posthocs: dict, out: Path):
    setup_style()
    colors = subject_colors(values["subject"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    axes = axes.ravel()
    for ax in axes[len(METRICS):]:
        ax.set_visible(False)
    for ax, m, letter in zip(axes, METRICS, "ABCDEF"):
        plot_panel(ax, values, m, posthocs[m], colors)
        ax.set_title(f"({letter})")
    fig.suptitle(f"Reaching training: first vs last {N_BLOCKS} blocks (median [IQR] across participants)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(h_pad=1.6, rect=(0, 0.03, 1, 0.985))
    subject_legend(fig, colors, y=0.035, extra=phase_handles(labels))
    save_figure(fig, out, "training_3blocks_boxplots")
    for m in METRICS:
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        plot_panel(ax, values, m, posthocs[m], colors)
        side_legend(ax, phase_handles(labels) + subject_handles(colors))
        save_figure(fig, out, f"single/{m}_boxplot", note=BOX_NOTE)


def main():
    p = argparse.ArgumentParser(description="Reaching training, first vs last 3 blocks")
    p.add_argument("--results", type=Path, default=ROOT / "reaching_regions")
    p.add_argument("--out", type=Path, default=BASE_DIR / "statistics")
    args = p.parse_args()

    trials, _, dropped, problems = full.load(args.results)
    if trials.empty:
        print(f"No *_regions_<pre|post>_raw.csv in {args.results}")
        return
    blocks = trials.groupby(["subject", "training", "block"], as_index=False)[list(METRICS)].mean()
    values, labels = phase_values(blocks)

    report = Report(f"REACHING TRAINING – first vs last {N_BLOCKS} blocks (outward movements)")
    subjects = sorted(trials["subject"].unique())
    report.text(f"Participants: {len(subjects)} ({', '.join(subjects)})")
    if dropped:
        report.text("Repeated sessions discarded (the last one is used): " + ", ".join(p.name for p in dropped))
    for msg in problems:
        report.text("WARNING: " + msg)
    incomplete = values[values["n_blocks"] < N_BLOCKS]
    for _, r in incomplete.iterrows():
        report.text(f"WARNING: {r['subject']} {r['training']} {r['phase']}: only {r['n_blocks']} blocks")
    report.text(f"Blocks of {full.BLOCK_SIZE} targets (block value = mean of its targets); per training,")
    report.text(f"start = mean of {labels['start']}, end = mean of {labels['end']}.")
    report.text("Errors at 1 s after the target onset; movement duration and linearity index from target onset")
    report.text("to target entry (same metrics as reaching_regions).")
    results, posthocs = analyse(values, labels, report)

    args.out.mkdir(parents=True, exist_ok=True)
    values.to_csv(args.out / "training_3blocks_values.csv", index=False)
    save_tables(results, args.out, "training_3blocks")
    jamovi_tables(values, labels, ROOT / "jamovi" / "training_3blocks")
    plots(values, labels, posthocs, args.out)
    report.save(args.out)


if __name__ == "__main__":
    main()
