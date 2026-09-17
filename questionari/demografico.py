"""Questionario demografico (Subject Information).

Colonne: ID, Gender, Age, esperienza con interfacce/videogiochi.

Output:
- statistics/demografico_statistics.csv : n e % per Gender, fasce d'età,
  esperienza; media/SD/mediana/min/max dell'età.
- statistics/plots/demografico_gender.png, demografico_age.png,
  demografico_experience.png
"""

import pandas as pd

from utils import (band_series, bar_counts, count_rows, describe_row,
                   load_responses, parse_args, save_statistics, CATEGORICAL)

AGE_EDGES = [0, 17, 25, 35, 50, 200]
AGE_LABELS = ["<18", "18-25", "26-35", "36-50", ">50"]
GENDER_ORDER = ["Male", "Female"]       # opzioni del Google Form
EXPERIENCE_ORDER = ["Yes", "No"]


def ordered(series, known):
    """Ordine fisso per le categorie note; eventuali altre in coda."""
    extra = [c for c in series.dropna().unique() if c not in known]
    return known + sorted(extra)


def find_column(df, keyword):
    """Prima colonna il cui nome contiene `keyword` (case-insensitive)."""
    for c in df.columns:
        if keyword.lower() in c.lower():
            return c
    raise KeyError(f"Colonna con '{keyword}' non trovata: {list(df.columns)}")


def main():
    args = parse_args("Statistiche del questionario demografico")
    df = load_responses("Subject Information", args.risposte)

    rows = [{"section": "Sample", "item": "Subjects", "n": len(df)}]
    if df.empty:
        save_statistics("demografico", rows, args.out)
        print("Nessuna risposta: grafici non generati.")
        return

    gender_col = find_column(df, "gender")
    age_col = find_column(df, "age")
    exp_col = find_column(df, "experience")

    gender = df[gender_col].astype(str).str.strip()
    age = pd.to_numeric(df[age_col], errors="coerce")
    age_band = band_series(age, AGE_EDGES, AGE_LABELS)
    exp = df[exp_col].astype(str).str.strip()

    rows += count_rows("Gender", gender, order=ordered(gender, GENDER_ORDER))
    rows.append(describe_row("Age", "Age (years)", age))
    rows += count_rows("Age band", age_band, order=AGE_LABELS)
    rows += count_rows("Experience with interfaces/videogames", exp,
                       order=ordered(exp, EXPERIENCE_ORDER))
    save_statistics("demografico", rows, args.out)

    # --- grafici -----------------------------------------------------------
    g_counts = gender.value_counts().reindex(ordered(gender, GENDER_ORDER), fill_value=0)
    bar_counts(g_counts, "Gender", "Gender", "demografico_gender", args.out,
               colors=CATEGORICAL[:len(g_counts)])
    a_counts = age_band.value_counts().reindex(AGE_LABELS, fill_value=0)
    bar_counts(a_counts, "Age distribution", "Age band (years)",
               "demografico_age", args.out)
    e_counts = exp.value_counts().reindex(ordered(exp, EXPERIENCE_ORDER), fill_value=0)
    bar_counts(e_counts, "Experience with interfaces / videogames",
               "Answer", "demografico_experience", args.out, horizontal=True,
               colors=CATEGORICAL[:len(e_counts)])


if __name__ == "__main__":
    main()
