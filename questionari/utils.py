"""Funzioni comuni a tutti gli script dei questionari.

- load_responses(prefix): legge il file in risposte/ che inizia con `prefix`
  (.xlsx o .csv), pulisce i nomi colonna, normalizza la colonna ID e toglie
  Timestamp.
- save_statistics(name, rows): scrive statistics/<name>_statistics.csv
- save_scores(name, df):      scrive statistics/<name>_scores.csv
- save_figure(fig, name):     scrive statistics/plots/<name>.png

Ogni script accetta `--risposte DIR` e `--out DIR` per usare cartelle diverse
da quelle di default (utile per i test).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RISPOSTE = BASE_DIR / "risposte"
DEFAULT_OUT = BASE_DIR / "statistics"

# ---------------------------------------------------------------------------
# Palette (una sola tinta per grandezze, categoriali in ordine fisso,
# rosso/blu per scale divergenti tipo Likert)
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
# scala divergente a 5 livelli: sinistra/negativo -> neutro -> destra/positivo
DIVERGING5 = ["#e34948", "#f2a3a2", "#e1e0d9", "#86b6ef", "#2a78d6"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def parse_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--risposte", type=Path, default=DEFAULT_RISPOSTE,
                   help="cartella con i file delle risposte")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="cartella di output per statistiche e grafici")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Caricamento
# ---------------------------------------------------------------------------
def load_responses(prefix, risposte_dir=DEFAULT_RISPOSTE):
    """Carica il file di risposte il cui nome inizia con `prefix`.

    Preferisce .xlsx; se non lo trova cerca un .csv. Restituisce un DataFrame
    con colonne ripulite (spazi tolti), colonna "ID" normalizzata e senza
    Timestamp. Le righe completamente vuote vengono scartate.
    """
    risposte_dir = Path(risposte_dir)
    candidates = sorted(risposte_dir.glob(f"{prefix}*.xlsx")) + \
        sorted(risposte_dir.glob(f"{prefix}*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Nessun file che inizia con '{prefix}' in {risposte_dir}")
    path = candidates[0]
    if path.suffix == ".xlsx":
        df = pd.read_excel(path, sheet_name=0)
    else:
        df = pd.read_csv(path)

    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={c: "ID" for c in df.columns if c.upper() == "ID"})
    df = df.drop(columns=[c for c in df.columns if c.lower() == "timestamp"])
    df = df.dropna(how="all").reset_index(drop=True)
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(str).str.strip()
    print(f"[{prefix}] letto {path.name}: {len(df)} risposte")
    return df


# ---------------------------------------------------------------------------
# Statistiche
# ---------------------------------------------------------------------------
STAT_COLUMNS = ["section", "item", "n", "percent",
                "mean", "sd", "var", "sem", "median", "min", "max"]


def count_rows(section, series, order=None):
    """Righe (n, %) per ogni categoria di `series`.

    `order` fissa l'ordine delle categorie e include anche quelle a zero.
    """
    series = series.dropna()
    total = len(series)
    counts = series.value_counts()
    if order is not None:
        counts = counts.reindex(order, fill_value=0)
    rows = []
    for cat, n in counts.items():
        rows.append({"section": section, "item": str(cat), "n": int(n),
                     "percent": round(100 * n / total, 1) if total else 0.0})
    rows.append({"section": section, "item": "Total", "n": total,
                 "percent": 100.0 if total else 0.0})
    return rows


def describe_row(section, item, series):
    """Riga con media, SD (campionaria), mediana, min, max di `series`."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"section": section, "item": item, "n": 0}
    return {"section": section, "item": item, "n": int(len(s)),
            "mean": round(s.mean(), 2),
            "sd": round(s.std(ddof=1), 2) if len(s) > 1 else 0.0,
            "var": round(s.var(ddof=1), 2) if len(s) > 1 else 0.0,
            "sem": round(s.std(ddof=1) / len(s) ** 0.5, 2) if len(s) > 1 else 0.0,
            "median": round(s.median(), 2),
            "min": round(s.min(), 2), "max": round(s.max(), 2)}


def band_series(series, edges, labels):
    """Assegna ogni valore alla fascia definita da `edges` (inclusivi)."""
    return pd.cut(pd.to_numeric(series, errors="coerce"),
                  bins=edges, labels=labels, include_lowest=True)


def save_statistics(name, rows, out_dir=DEFAULT_OUT):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).reindex(columns=STAT_COLUMNS)
    path = out_dir / f"{name}_statistics.csv"
    df.to_csv(path, index=False)
    print(f"[{name}] statistiche -> {path}")
    return df


def save_scores(name, df, out_dir=DEFAULT_OUT):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_scores.csv"
    df.to_csv(path, index=False)
    print(f"[{name}] punteggi per soggetto -> {path}")


# ---------------------------------------------------------------------------
# Grafici
# ---------------------------------------------------------------------------
def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "font.size": 10,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def save_figure(fig, name, out_dir=DEFAULT_OUT):
    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[{name}] grafico -> {path}")


def bar_counts(counts, title, xlabel, name, out_dir=DEFAULT_OUT,
               horizontal=False, colors=None, ylabel="Subjects", wide=False):
    """Barre di conteggio con etichetta 'n (x%)' su ogni barra.

    `counts`: Series con indice = categorie (nell'ordine voluto).
    """
    setup_style()
    total = counts.sum()
    labels = [str(i) for i in counts.index]
    values = counts.values
    colors = colors or BLUE
    if horizontal:
        fig, ax = plt.subplots(figsize=(6, 0.5 * len(labels) + 1.2))
        ax.barh(labels, values, color=colors, height=0.6)
        ax.invert_yaxis()
        ax.grid(axis="x"); ax.grid(False, axis="y")
        ax.set_xlabel(ylabel); ax.set_ylabel(xlabel)
        for i, v in enumerate(values):
            pct = 100 * v / total if total else 0
            ax.text(v + max(values.max(), 1) * 0.01, i, f"{v} ({pct:.0f}%)",
                    va="center", ha="left", color=INK, fontsize=9)
        ax.set_xlim(0, max(values.max(), 1) * 1.25)
    else:
        per_bar = 0.85 if wide else 0.6
        fig, ax = plt.subplots(figsize=(max(5, per_bar * len(labels) + 2), 4.2))
        ax.bar(labels, values, color=colors, width=0.6)
        if wide:
            ax.tick_params(axis="x", labelsize=7.5)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        for i, v in enumerate(values):
            pct = 100 * v / total if total else 0
            ax.text(i, v + max(values.max(), 1) * 0.02, f"{v} ({pct:.0f}%)",
                    ha="center", va="bottom", color=INK, fontsize=9)
        ax.set_ylim(0, max(values.max(), 1) * 1.2)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title(title, loc="left")
    save_figure(fig, name, out_dir)


def diverging_likert(pct, title, name, out_dir=DEFAULT_OUT,
                     legend_title=None):
    """Barre impilate divergenti (una per item) centrate sul livello neutro.

    `pct`: DataFrame item x livello (5 livelli in ordine da negativo a
    positivo) con le percentuali di risposta.
    """
    setup_style()
    levels = list(pct.columns)
    assert len(levels) == 5, "servono esattamente 5 livelli"
    items = list(pct.index)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(items) + 1.6))
    y = np.arange(len(items))
    # offset: le due categorie negative + metà della neutra stanno a sinistra
    left = -(pct[levels[0]] + pct[levels[1]] + pct[levels[2]] / 2)
    for lvl, col in zip(levels, DIVERGING5):
        w = pct[lvl].values
        ax.barh(y, w, left=left, color=col, height=0.6, label=lvl,
                edgecolor="white", linewidth=1)
        for yi, (l, wi) in enumerate(zip(left, w)):
            if wi >= 8:
                ax.text(l + wi / 2, yi, f"{wi:.0f}%", ha="center",
                        va="center", fontsize=8,
                        color="white" if col in (DIVERGING5[0], DIVERGING5[4]) else INK)
        left = left + w
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(items)
    ax.invert_yaxis()
    ax.grid(axis="x"); ax.grid(False, axis="y")
    ax.set_xlim(-100, 100)
    ax.set_xticks(range(-100, 101, 25))
    ax.set_xticklabels([f"{abs(v)}%" for v in range(-100, 101, 25)])
    ax.set_xlabel("Percentage of subjects")
    ax.legend(title=legend_title, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=5, fontsize=8)
    ax.set_title(title, loc="left")
    save_figure(fig, name, out_dir)
