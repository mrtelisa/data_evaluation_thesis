"""Training a 9 regioni (reaching_regions.py), fatto due volte: training 1 (pre) e
training 2 (post). Metriche di ogni ANDATA (centro -> target, 96 per training) per
new_stats/new_stats_analysis.py (statistica, grafici e tabelle della tesi; usa anche
figure4_panel per le curve) e figura dei RITORNI al centro (home), senza statistica.

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
Blocchi di 8 target (colonna block); new_stats usa blocchi da 32 (target_number).
Ritorno: il goal home che segue ogni target (il primo home di ogni blocco non e' un ritorno);
MD dalla comparsa della home all'entrata, LI sullo stesso tratto.

Output:
    statistics/training_trials.csv          ogni andata
    statistics/training_return_trials.csv   ogni ritorno
    statistics/training_return_curves.pdf   figura della tesi dei ritorni (MD, LI): curve come la
        figura 4 di Bellitto, media mobile su 12 prove, mediana [IQR]
    Con --thesis la figura va direttamente in Tesi_latex/Chapters/Figures/results (sovrascrive
    quella della tesi: di default NO, come new_stats).

Uso:
    python3 training_analysis.py [--results DIR] [--out DIR] [--thesis]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common.plots as _plots   # noqa: E402  (per scegliere dove scrive save_thesis_figure)
from common.data import find_sessions   # noqa: E402
from common.plots import (TRAINING_COLORS, MUTED, axis_label, wrap_unit, save_thesis_figure,   # noqa: E402
                          setup_style)
from common.reach_metrics import canvas_scale, canvas_to_cm, goals_from_raw, training_metrics   # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
PATTERN = r"(?P<subject>.+)_regions_(?P<phase>pre|post)(?:_(?P<rep>\d+))?_raw\.csv"
TRAININGS = {"pre": "T1", "post": "T2"}
BLOCK_SIZE = 8
HIDDEN_S_DEFAULT = 1.0
WINDOW = 12          # figura 4: media mobile sulla prova e le 11 precedenti (come il paper)
GAP = 16             # prove di spazio fra i due training nella figura 4

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


def figure4_panel(ax, trials: pd.DataFrame, m: str, label: str, unit: str):
    """Un pannello: media mobile per soggetto, poi mediana [IQR] tra soggetti, T1 e T2 affiancati."""
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
    # etichette dei training sopra i dati
    lo, top = ax.get_ylim()
    top += 0.1 * (top - lo)
    ax.set_ylim(lo, top)
    for tr, name in (("T1", "Training 1"), ("T2", "Training 2")):
        ax.text(trial_x(tr, n_trials / 2, n_trials), top, name, ha="center", va="top", fontsize=11,
                color=TRAINING_COLORS[tr], fontweight="bold")


def return_figure(returns: pd.DataFrame):
    """Figura della tesi dei ritorni: un pannello per metrica, due per riga, senza titolo."""
    nrows = int(np.ceil(len(RETURN_METRICS) / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(7.6, 2.9 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(RETURN_METRICS):]:
        ax.set_visible(False)
    for ax, (m, (_, label, unit)), letter in zip(axes, RETURN_METRICS.items(), "ABCDEFGH"):
        figure4_panel(ax, returns, m, label, unit)
        ax.set_title(f"({letter})")
        # etichette lunghe: a capo solo l'unita' di misura, se no escono dal pannello
        wrap_unit(ax)
    fig.tight_layout(h_pad=1.4)
    save_thesis_figure(fig, "training_return_curves", note="Line: median; band: IQR.")


def main():
    p = argparse.ArgumentParser(description="Reaching training: metrics of every movement, return curves")
    p.add_argument("--results", type=Path, default=BASE_DIR)
    p.add_argument("--out", type=Path, default=BASE_DIR / "statistics")
    p.add_argument("--thesis", action="store_true",
                   help="scrive la figura dei ritorni direttamente nella cartella della tesi (di default no)")
    args = p.parse_args()

    trials, returns, dropped, problems = load(args.results)
    if trials.empty:
        print(f"No *_regions_<pre|post>_raw.csv in {args.results}")
        return
    subjects = sorted(trials["subject"].unique())
    print(f"Participants: {len(subjects)} ({', '.join(subjects)})")
    if dropped:
        print("Repeated sessions discarded (the last one is used): " + ", ".join(p.name for p in dropped))
    for msg in problems:
        print("WARNING: " + msg)
    args.out.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.out / "training_trials.csv", index=False)
    returns.to_csv(args.out / "training_return_trials.csv", index=False)
    print(f"-> {args.out / 'training_trials.csv'}, {args.out / 'training_return_trials.csv'}")

    if not args.thesis:
        _plots.THESIS_DIR = args.out
    setup_style()
    return_figure(returns)


if __name__ == "__main__":
    main()
