"""NASA Task Load Index (6 dimensioni, scala 1-20).

Score complessivo per soggetto = Raw TLX, cioè la media delle 6 dimensioni
senza pesatura. Riportato sia sulla scala 1-20 sia riscalato 0-100 con
(x - 1) / 19 * 100. Nel form la scala di Performance va al contrario delle altre
(20 = prestazione percepita migliore, quindi meno carico): nel Raw TLX entra invertita,
21 - valore; le statistiche della singola dimensione restano sulla scala del form.

Output:
- statistics/nasa_statistics.csv : per ogni dimensione media/SD/mediana/
  min/max e n/% in fascia low (1-7) / medium (8-13) / high (14-20); stessa
  cosa per il Raw TLX.
- statistics/nasa_scores.csv : ID, 6 dimensioni, Raw TLX (1-20), Raw TLX
  (0-100).
- statistics/plots/nasa_dimensions.png (media ± SD)
- statistics/plots/nasa_boxplot.(png|pdf): box plot come quelli dei test e del robot (common/plots.py:
  mediana [IQR], baffi 1.5 IQR, un pallino per soggetto, legenda dei soggetti a lato, senza titolo);
  il pdf e' copiato anche nelle figure della tesi
"""

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (BLUE, band_series, count_rows, describe_row,
                   load_responses, parse_args, save_figure, save_scores,
                   save_statistics, setup_style)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import plots as cp   # noqa: E402

DIMENSIONS = ["Mental demand", "Physical demand", "Temporal demand",
              "Performance", "Effort", "Frustration"]
BAND_EDGES = [1, 7, 13, 20]
BAND_LABELS = ["Low (1-7)", "Medium (8-13)", "High (14-20)"]
REVERSED = ["Performance"]   # 20 = buona prestazione: invertita nel Raw TLX
THESIS_FIG = (Path(__file__).resolve().parents[3]
              / "Tesi_latex/Chapters/Figures/results/nasa_boxplot.pdf")


def main():
    args = parse_args("Statistiche del NASA-TLX")
    df = load_responses("Nasa", args.risposte)

    rows = [{"section": "Sample", "item": "Subjects", "n": len(df)}]
    if df.empty:
        save_statistics("nasa", rows, args.out)
        print("Nessuna risposta: grafici non generati.")
        return

    # le 6 dimensioni sono le colonne dopo ID, nell'ordine del form
    item_cols = [c for c in df.columns if c != "ID"]
    assert len(item_cols) == 6, f"attese 6 dimensioni, trovate {len(item_cols)}"
    scores = df[["ID"]].copy()
    for dim, col in zip(DIMENSIONS, item_cols):
        scores[dim] = pd.to_numeric(df[col], errors="coerce")
    workload = scores[DIMENSIONS].copy()
    workload[REVERSED] = 21 - workload[REVERSED]   # tutte le dimensioni: alto = piu' carico
    scores["Raw TLX (1-20)"] = workload.mean(axis=1).round(2)
    scores["Raw TLX (0-100)"] = ((scores["Raw TLX (1-20)"] - 1) / 19 * 100).round(1)
    save_scores("nasa", scores, args.out)

    for dim in DIMENSIONS:
        rows.append(describe_row("Dimension", dim, scores[dim]))
    rows.append(describe_row("Overall", "Raw TLX (1-20)", scores["Raw TLX (1-20)"]))
    rows.append(describe_row("Overall", "Raw TLX (0-100)", scores["Raw TLX (0-100)"]))
    for dim in DIMENSIONS + ["Raw TLX (1-20)"]:
        bands = band_series(scores[dim], BAND_EDGES, BAND_LABELS)
        rows += count_rows(f"Band – {dim}", bands, order=BAND_LABELS)
    save_statistics("nasa", rows, args.out)

    # --- grafici -----------------------------------------------------------
    setup_style()
    means = scores[DIMENSIONS].mean()
    sds = scores[DIMENSIONS].std(ddof=1).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(DIMENSIONS))
    ax.bar(x, means, color=BLUE, width=0.6)
    # la SD tra soggetti e' la banda in cui sfuma la cima della barra
    for xi, m, s in zip(x, means, sds):
        if s > 0:
            ax.fill_between([xi - 0.3, xi + 0.3], [m, m], [m + s, m + s], color=BLUE, alpha=0.3, lw=0, zorder=3)
            ax.fill_between([xi - 0.3, xi + 0.3], [m - s, m - s], [m, m], color="white", alpha=0.45, lw=0, zorder=3)
    for xi, m in zip(x, means):
        ax.text(xi, 0.5, f"{m:.1f}", ha="center", va="bottom",
                color="white", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(DIMENSIONS, rotation=20, ha="right")
    ax.set_ylim(0, 20); ax.set_ylabel("Score (1-20)")
    ax.set_title("NASA-TLX – mean ± SD per dimension", loc="left")
    save_figure(fig, "nasa_dimensions", args.out)

    # box plot come i test e il robot (box blu, baffi grigio scuro), senza titolo
    cp.setup_style()
    by_subject = scores.set_index("ID")
    colors = cp.subject_colors(by_subject.index)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    cp.condition_plot(ax, {(d, "NASA"): by_subject[d] for d in DIMENSIONS}, DIMENSIONS, ["NASA"],
                      {"NASA": ("#cde2fb", BLUE)}, colors, width=0.5, whisker_color=cp.INK2)
    ax.set_xticklabels(DIMENSIONS, rotation=20, ha="right")
    ax.set_ylim(0.5, 20.5)   # scala del questionario: 1-20
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.set_ylabel("Score (1-20)")
    cp.side_legend(ax, cp.subject_handles(colors))
    png = cp.save_figure(fig, args.out, "nasa_boxplot", note=cp.BOX_NOTE)
    print(f"[nasa_boxplot] grafico -> {png} (+ .pdf)")
    if THESIS_FIG.parent.exists():
        shutil.copy(png.with_suffix(".pdf"), THESIS_FIG)
        print(f"[nasa_boxplot] figura tesi -> {THESIS_FIG}")


if __name__ == "__main__":
    main()
