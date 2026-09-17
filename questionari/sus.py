"""System Usability Scale (10 item, scala 1-5).

Score standard (Brooke 1996):
    item dispari (1,3,5,7,9): contributo = risposta - 1
    item pari   (2,4,6,8,10): contributo = 5 - risposta
    SUS = somma dei contributi * 2.5   -> 0..100
La colonna "Score" eventualmente presente nel file viene ignorata: lo score
viene sempre ricalcolato.

Interpretazione (Bangor, Kortum & Miller 2009):
    voto:      A >= 90, B 80-89, C 70-79, D 60-69, F < 60
    aggettivo: < 25 Worst imaginable, 25-38 Poor, 39-51 OK, 52-72 Good,
               73-84 Excellent, >= 85 Best imaginable
Soglia di riferimento: 68 (media dei sistemi in letteratura).

Output:
- statistics/sus_statistics.csv : media/SD/mediana/min/max dello score,
  % sopra 68, n e % per voto e aggettivo, media/SD per item.
- statistics/sus_scores.csv : ID, 10 risposte, SUS score, voto, aggettivo.
- statistics/plots/sus_scores.png, sus_items.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (BLUE, INK2, MUTED, count_rows, describe_row,
                   diverging_likert, load_responses, parse_args, save_figure,
                   save_scores, save_statistics, setup_style)

LIKERT = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
TEXT_TO_NUM = {"strongly disagree": 1, "disagree": 2, "neutral": 3,
               "neither agree nor disagree": 3, "agree": 4, "strongly agree": 5}
BENCHMARK = 68
GRADES = [(90, "A"), (80, "B"), (70, "C"), (60, "D"), (0, "F")]
ADJECTIVES = [(85, "Best imaginable"), (73, "Excellent"), (52, "Good"),
              (39, "OK"), (25, "Poor"), (0, "Worst imaginable")]


def lookup(score, table):
    if pd.isna(score):
        return np.nan
    for threshold, label in table:
        if score >= threshold:
            return label


def to_numeric_answer(a):
    """Accetta sia 1-5 sia il testo della scala Likert."""
    n = pd.to_numeric(a, errors="coerce")
    if pd.notna(n):
        return int(n)
    key = str(a).strip().lower()
    if key in TEXT_TO_NUM:
        return TEXT_TO_NUM[key]
    raise ValueError(f"Risposta SUS non riconosciuta: {a!r}")


def main():
    args = parse_args("Statistiche del SUS")
    df = load_responses("SUS", args.risposte)

    rows = [{"section": "Sample", "item": "Subjects", "n": len(df)}]
    if df.empty:
        save_statistics("sus", rows, args.out)
        print("Nessuna risposta: grafici non generati.")
        return

    item_cols = [c for c in df.columns if c not in ("ID", "Score")]
    assert len(item_cols) == 10, f"attesi 10 item, trovati {len(item_cols)}"
    answers = df[item_cols].map(to_numeric_answer)
    if not answers.isin([1, 2, 3, 4, 5]).all().all():
        raise ValueError("Risposte SUS fuori dal range 1-5")

    scores = df[["ID"]].copy()
    contrib = pd.DataFrame(index=answers.index)
    for i, c in enumerate(item_cols, start=1):
        scores[f"Q{i}"] = answers[c]
        contrib[f"Q{i}"] = answers[c] - 1 if i % 2 == 1 else 5 - answers[c]
    scores["SUS"] = (contrib.sum(axis=1) * 2.5).round(1)
    scores["Grade"] = scores["SUS"].map(lambda s: lookup(s, GRADES))
    scores["Adjective"] = scores["SUS"].map(lambda s: lookup(s, ADJECTIVES))
    save_scores("sus", scores, args.out)

    rows.append(describe_row("SUS score", "SUS (0-100)", scores["SUS"]))
    above = int((scores["SUS"] >= BENCHMARK).sum())
    rows.append({"section": "SUS score", "item": f"Subjects >= {BENCHMARK}",
                 "n": above, "percent": round(100 * above / len(scores), 1)})
    rows += count_rows("Grade", scores["Grade"], order=[g for _, g in GRADES])
    rows += count_rows("Adjective rating", scores["Adjective"],
                       order=[a for _, a in ADJECTIVES][::-1])
    for i in range(1, 11):
        rows.append(describe_row("Item (raw answer 1-5)", f"Q{i}", scores[f"Q{i}"]))
    save_statistics("sus", rows, args.out)

    # --- grafici -----------------------------------------------------------
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    counts, _, _ = ax.hist(scores["SUS"], bins=np.arange(0, 101, 10), color=BLUE,
                           edgecolor="white", linewidth=1)
    top = max(counts.max(), 1) * 1.3   # spazio sopra le barre per le etichette
    ax.set_ylim(0, top)
    ax.axvline(BENCHMARK, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(BENCHMARK + 1, top * 0.97, f"benchmark {BENCHMARK}",
            color=INK2, fontsize=9, va="top")
    mean = scores["SUS"].mean()
    ax.axvline(mean, color=INK2, linewidth=1.5)
    ax.text(mean + 1, top * 0.87, f"mean {mean:.1f}",
            color=INK2, fontsize=9, va="top")
    ax.set_xlim(0, 100); ax.set_xlabel("SUS score"); ax.set_ylabel("Subjects")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_title("System Usability Scale – score distribution", loc="left")
    save_figure(fig, "sus_scores", args.out)

    pct = pd.DataFrame(index=[f"Q{i}" for i in range(1, 11)], columns=LIKERT,
                       dtype=float)
    for i in range(1, 11):
        counts = scores[f"Q{i}"].value_counts()
        for v, lvl in enumerate(LIKERT, start=1):
            pct.loc[f"Q{i}", lvl] = 100 * counts.get(v, 0) / len(scores)
    diverging_likert(pct, "System Usability Scale – answers per item",
                     "sus_items", args.out)


if __name__ == "__main__":
    main()
