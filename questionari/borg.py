"""Borg scale (sforzo percepito), una sola domanda.

Scala usata: Borg RPE 6-20 (impostata in SCALE_NAME; è disponibile anche
la CR10 0-10). Le opzioni del Google Form sono del tipo "6: No exertion at
all": dalla risposta viene estratto il numero iniziale. I valori fuori dal
range della scala fanno fermare lo script.

Output:
- statistics/borg_statistics.csv : media/SD/mediana/min/max, n e % per ogni
  valore della scala, n e % per fascia (light / moderate / hard).
- statistics/plots/borg_distribution.png
"""

import re

import numpy as np
import pandas as pd

from utils import (band_series, bar_counts, count_rows, describe_row,
                   load_responses, parse_args, save_statistics)

SCALES = {
    "CR10 (0-10)": {"values": list(range(0, 11)),
                    "edges": [0, 2, 5, 10],
                    "labels": ["Light (0-2)", "Moderate (3-5)", "Hard (6-10)"],
                    "anchors": {0: "Nothing at all", 1: "Very weak", 2: "Weak",
                                3: "Moderate", 4: "Somewhat strong", 5: "Strong",
                                7: "Very strong", 10: "Very, very strong"}},
    "RPE (6-20)":  {"values": list(range(6, 21)),
                    "edges": [6, 11, 14, 20],
                    "labels": ["Light (6-11)", "Moderate (12-14)", "Hard (15-20)"],
                    "anchors": {6: "No exertion at all", 7: "Extremely light",
                                9: "Very light", 11: "Light", 12: "Moderate",
                                13: "Somewhat hard", 15: "Hard", 17: "Very hard",
                                19: "Extremely hard", 20: "Maximal exertion"}},
}


SCALE_NAME = "RPE (6-20)"


def parse_score(a):
    """'6: No exertion at all' -> 6.0 ; '13' -> 13.0 ; vuoto -> NaN."""
    m = re.match(r"\s*(\d+(?:[.,]\d+)?)", str(a))
    return float(m.group(1).replace(",", ".")) if m else np.nan


def main():
    args = parse_args("Statistiche della Borg scale")
    df = load_responses("BORG", args.risposte)

    rows = [{"section": "Sample", "item": "Subjects", "n": len(df)}]
    if df.empty:
        save_statistics("borg", rows, args.out)
        print("Nessuna risposta: grafici non generati.")
        return

    # l'unica colonna oltre a ID è la risposta
    effort_col = [c for c in df.columns if c != "ID"][0]
    effort = df[effort_col].map(parse_score).dropna()

    scale = SCALES[SCALE_NAME]
    lo, hi = scale["values"][0], scale["values"][-1]
    if not effort.between(lo, hi).all():
        bad = effort[~effort.between(lo, hi)].tolist()
        raise ValueError(f"Valori Borg fuori dal range {lo}-{hi}: {bad}")
    rows.append({"section": "Sample", "item": f"Scale: Borg {SCALE_NAME}",
                 "n": len(effort)})
    rows.append(describe_row("Perceived effort", "Score", effort))
    rows += count_rows("Score distribution", effort.round(1),
                       order=scale["values"])
    bands = band_series(effort, scale["edges"], scale["labels"])
    rows += count_rows("Effort band", bands, order=scale["labels"])
    save_statistics("borg", rows, args.out)

    # --- grafico -----------------------------------------------------------
    # conteggio per ogni valore della scala (anche quelli a zero)
    rounded = effort.round(0).astype(int)
    counts = rounded.value_counts().reindex(scale["values"], fill_value=0)
    # etichetta di tick: numero + ancora testuale dove la scala la prevede;
    # se due ancore sono adiacenti (6/7, 19/20) la seconda scende di una riga
    anchors = scale["anchors"]
    labels = []
    for v in counts.index:
        if v not in anchors:
            labels.append(str(v))
        elif (v - 1) in anchors and not labels[-1].endswith("\n" + anchors[v - 1] + "\n"):
            labels.append(f"{v}\n\n{anchors[v]}")
        else:
            labels.append(f"{v}\n{anchors[v]}")
    counts.index = labels
    bar_counts(counts, f"Perceived effort – Borg {SCALE_NAME}",
               "Borg score", "borg_distribution", args.out, wide=True)


if __name__ == "__main__":
    main()
