"""Edinburgh Handedness Inventory (10 item, Oldfield 1971).

Risposte per item (opzioni del Google Form):
    Left hand (strong preference) / Left hand / Indifferent/Both /
    Right hand / Right hand (strong preference)
convertite in punti R e L:
    Right hand (strong preference) -> R=2, L=0
    Right hand                     -> R=1, L=0
    Indifferent/Both               -> R=1, L=1
    Left hand                      -> R=0, L=1
    Left hand (strong preference)  -> R=0, L=2
Laterality Quotient: LQ = (R - L) / (R + L) * 100, in [-100, +100].
Classificazione: LQ < -40 left-handed, -40..40 ambidextrous, > 40 right-handed.

Output:
- statistics/edinburgh_statistics.csv : media/SD/mediana/min/max dell'LQ,
  n e % per classe, per ogni item la % di ogni risposta.
- statistics/edinburgh_scores.csv : ID, R, L, LQ, classe.
- statistics/plots/edinburgh_lq.png, edinburgh_items.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (AXIS, BLUE, DIVERGING5, INK, INK2, count_rows,
                   describe_row, diverging_likert, load_responses, parse_args,
                   save_figure, save_scores, save_statistics, setup_style)

LEVELS = ["Left hand (strong preference)", "Left hand", "Indifferent/Both",
          "Right hand", "Right hand (strong preference)"]
POINTS = {"left hand (strong preference)": (0, 2), "left hand": (0, 1),
          "indifferent/both": (1, 1), "right hand": (1, 0),
          "right hand (strong preference)": (2, 0)}
# etichette brevi per la legenda del grafico
SHORT_LEVELS = ["Left (strong)", "Left", "Indifferent/Both",
                "Right", "Right (strong)"]
CLASSES = ["Left-handed", "Ambidextrous", "Right-handed"]


def classify(lq):
    if pd.isna(lq):
        return np.nan
    if lq < -40:
        return CLASSES[0]
    if lq > 40:
        return CLASSES[2]
    return CLASSES[1]


def normalize_answer(a):
    a = str(a).strip().lower()
    if a not in POINTS:
        raise ValueError(f"Risposta Edinburgh non riconosciuta: {a!r}")
    return a


def main():
    args = parse_args("Statistiche dell'Edinburgh Handedness Inventory")
    df = load_responses("Edinburgh", args.risposte)

    rows = [{"section": "Sample", "item": "Subjects", "n": len(df)}]
    if df.empty:
        save_statistics("edinburgh", rows, args.out)
        print("Nessuna risposta: grafici non generati.")
        return

    item_cols = [c for c in df.columns if c != "ID"]
    assert len(item_cols) == 10, f"attesi 10 item, trovati {len(item_cols)}"
    answers = df[item_cols].map(normalize_answer)

    scores = df[["ID"]].copy()
    scores["R"] = answers.map(lambda a: POINTS[a][0]).sum(axis=1)
    scores["L"] = answers.map(lambda a: POINTS[a][1]).sum(axis=1)
    scores["LQ"] = ((scores["R"] - scores["L"]) / (scores["R"] + scores["L"]) * 100).round(1)
    scores["Class"] = scores["LQ"].map(classify)
    save_scores("edinburgh", scores, args.out)

    rows.append(describe_row("Laterality Quotient", "LQ", scores["LQ"]))
    rows += count_rows("Handedness class", scores["Class"], order=CLASSES)
    # per ogni item, % di ogni risposta (nome item accorciato alla parte
    # prima della parentesi)
    short = {c: c.split("(")[0].strip() for c in item_cols}
    pct = pd.DataFrame(index=[short[c] for c in item_cols], columns=LEVELS,
                       dtype=float)
    for c in item_cols:
        counts = answers[c].value_counts()
        for lvl in LEVELS:
            pct.loc[short[c], lvl] = 100 * counts.get(lvl.lower(), 0) / len(answers)
        for lvl in LEVELS:
            rows.append({"section": f"Item – {short[c]}", "item": lvl,
                         "n": int(counts.get(lvl.lower(), 0)),
                         "percent": round(pct.loc[short[c], lvl], 1)})
    save_statistics("edinburgh", rows, args.out)

    # --- grafici -----------------------------------------------------------
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.axvspan(-100, -40, color=DIVERGING5[1], alpha=0.25, lw=0)
    ax.axvspan(-40, 40, color=DIVERGING5[2], alpha=0.5, lw=0)
    ax.axvspan(40, 100, color=DIVERGING5[3], alpha=0.25, lw=0)
    counts, _, _ = ax.hist(scores["LQ"].dropna(), bins=np.arange(-100, 101, 20),
                           color=BLUE, edgecolor="white", linewidth=1)
    top = max(counts.max(), 1) * 1.3   # spazio sopra le barre per le etichette
    ax.set_ylim(0, top)
    for x, lab in [(-70, "Left-handed"), (0, "Ambidextrous"), (70, "Right-handed")]:
        ax.text(x, top * 0.97, lab, ha="center", va="top", color=INK2, fontsize=9)
    ax.set_xlim(-100, 100); ax.set_xlabel("Laterality Quotient (LQ)")
    ax.set_ylabel("Subjects")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_title("Edinburgh Handedness Inventory – LQ distribution", loc="left")
    save_figure(fig, "edinburgh_lq", args.out)

    pct_plot = pct.copy(); pct_plot.columns = SHORT_LEVELS
    diverging_likert(pct_plot, "Edinburgh Handedness Inventory – answers per item",
                     "edinburgh_items", args.out)


if __name__ == "__main__":
    main()
