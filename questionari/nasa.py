"""NASA Task Load Index (6 dimensioni, scala 1-20).

Score complessivo per soggetto = Raw TLX, cioè la media delle 6 dimensioni
senza pesatura. Riportato sia sulla scala 1-20 sia riscalato 0-100 con
(x - 1) / 19 * 100. Nessuna dimensione viene invertita (Performance è usata
così com'è).

Output:
- statistics/nasa_statistics.csv : per ogni dimensione media/SD/mediana/
  min/max e n/% in fascia low (1-7) / medium (8-13) / high (14-20); stessa
  cosa per il Raw TLX.
- statistics/nasa_scores.csv : ID, 6 dimensioni, Raw TLX (1-20), Raw TLX
  (0-100).
- statistics/plots/nasa_dimensions.png (media ± SD), nasa_boxplot.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (BLUE, INK, MUTED, band_series, count_rows, describe_row,
                   load_responses, parse_args, save_figure, save_scores,
                   save_statistics, setup_style)

DIMENSIONS = ["Mental demand", "Physical demand", "Temporal demand",
              "Performance", "Effort", "Frustration"]
BAND_EDGES = [1, 7, 13, 20]
BAND_LABELS = ["Low (1-7)", "Medium (8-13)", "High (14-20)"]


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
    scores["Raw TLX (1-20)"] = scores[DIMENSIONS].mean(axis=1).round(2)
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
    ax.bar(x, means, yerr=sds, color=BLUE, width=0.6, capsize=3,
           error_kw={"ecolor": MUTED, "elinewidth": 1})
    for xi, m in zip(x, means):
        ax.text(xi, 0.5, f"{m:.1f}", ha="center", va="bottom",
                color="white", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(DIMENSIONS, rotation=20, ha="right")
    ax.set_ylim(0, 20); ax.set_ylabel("Score (1-20)")
    ax.set_title("NASA-TLX – mean ± SD per dimension", loc="left")
    save_figure(fig, "nasa_dimensions", args.out)

    fig, ax = plt.subplots(figsize=(7, 4))
    data = [scores[d].dropna().values for d in DIMENSIONS]
    ax.boxplot(data, widths=0.5, patch_artist=True, showmeans=True,
               boxprops={"facecolor": "#cde2fb", "edgecolor": BLUE},
               medianprops={"color": BLUE, "linewidth": 2},
               whiskerprops={"color": MUTED}, capprops={"color": MUTED},
               flierprops={"marker": "o", "markersize": 4,
                           "markerfacecolor": MUTED, "markeredgecolor": "none"},
               meanprops={"marker": "D", "markersize": 4,
                          "markerfacecolor": INK, "markeredgecolor": "none"})
    ax.set_xticks(range(1, 7)); ax.set_xticklabels(DIMENSIONS, rotation=20, ha="right")
    ax.set_ylim(0, 21); ax.set_ylabel("Score (1-20)")
    ax.set_title("NASA-TLX – distribution per dimension", loc="left")
    save_figure(fig, "nasa_boxplot", args.out)


if __name__ == "__main__":
    main()
