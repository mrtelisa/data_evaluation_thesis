"""Test sul robot reale (reachy_control.py), fatto due volte: test 1 e test 2
(nei file run1 e run2).

Input: <soggetto>_run<N>[_k]_session.json, _regions.csv, _odometry.csv, ovunque dentro
questa cartella (es. real_robot/results/). Se una run e' stata ripetuta si usa l'ultima.

Fasi del compito, delimitate dagli eventi del task (session_metrics.py): andata = dalla
partenza alla selezione dell'oggetto (outbound + approach), grasp = selezione -> oggetto in
mano, ritorno con la borraccia, posizionamento (placement); la loro somma e' il tempo totale.
Andata + ritorno = tempo di guida senza la manipolazione (grasp e posizionamento).
Metriche (come Najie et al.):
    CT  Completion time (s)       per fase e totale, dal _session.json (durate degli eventi
                                  del task; nel raw mancano gli istanti); solo fasi completate
    PL  Path length (m)           andata e ritorno, dall'odometria      (robot_raw.py)
    MS  Mean speed (m/s)          andata e ritorno, dall'odometria
    JK  Jerk (m/s^3)              andata e ritorno, jerk medio dall'odometria
    RT  Region transitions        andata e ritorno, cambi di regione del cursore

Analisi, come il modello jamovi della dottoranda (una volta con le andate, una con i
ritorni, e per ogni fase del completion time): ANOVA a misure ripetute test 1 vs test 2
(con 2 livelli coincide con il t appaiato: F = t^2); controllo Wilcoxon.

Output:
    statistics/report.txt, robot_subjects.csv, robot_descriptives/anova/posthoc.csv
    statistics/plots/robot_boxplots.(png|pdf)  box plot come Najie (mediana, IQR) + t appaiato:
        una riga per metrica (CT, PL, MS, JK, RT), andata a sinistra e ritorno a destra, stessa scala y per riga
    statistics/plots/robot_manipulation_boxplot.(png|pdf)  completion time di grasp e posizionamento
    statistics/plots/robot_total_time_boxplot.(png|pdf)    completion time andata + ritorno
    statistics/plots/single/<metrica>_<outbound|return>_boxplot.(png|pdf)  i pannelli uno per uno
    ../jamovi/reachy/<sigla>_<metrica>.(omv|csv): un file per metrica come reachy_stats.omv, una riga
        per soggetto, colonne Test1_andata, Test2_andata, Test1_ritorno, Test2_ritorno
        (completion time: anche Test1_grasp ... Test2_totale, Test1_andata_ritorno, Test2_andata_ritorno)

Uso:
    python3 robot_analysis.py [--results DIR] [--out DIR]
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
from common.plots import (BOX_NOTE, GREY_BOX, RED_BOX, axis_label, brackets, condition_plot,   # noqa: E402
                          save_figure, setup_style, share_ylim, side_legend, subject_colors, subject_handles,
                          subject_legend)
from common.report import Report, analyse_metric, save_tables   # noqa: E402
from robot_raw import navigation_metrics   # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
RUNS = (1, 2)
PATTERN = r"(?P<subject>.+)_run(?P<run>\d)(?:_(?P<rep>\d+))?_session\.json"
# fase (nome delle colonne jamovi, come il file della dottoranda) -> (etichetta, fasi del task)
PHASES = {"andata": ("Outbound", ("outbound", "approach")), "grasp": ("Grasp", ("grasp",)),
          "ritorno": ("Return", ("return",)), "posizionamento": ("Placement", ("placement",)),
          "totale": ("Total", None),
          "andata_ritorno": ("Outbound + return", ("outbound", "approach", "return"))}
# metriche esplorative (robot_raw.py): solo report e .csv, per la discussione
EXPLORATORY = {
    "stop_percent":        ("Stop time while driving", "%", ["andata", "ritorno"]),
    "transitions_per_min": ("Region transitions per minute of driving", "/min", ["andata", "ritorno"]),
    "cursor_eccentricity": ("Cursor distance from the centre while moving (0 = centre, 1 = border)", "", ["andata", "ritorno"]),
}
# metrica -> (sigla jamovi, etichetta, unita', fasi)
METRICS = {
    "completion_time":    ("CT", "Completion time", "s", list(PHASES)),
    "path_length":        ("PL", "Path length", "m", ["andata", "ritorno"]),
    "mean_speed":         ("MS", "Mean speed", "m/s", ["andata", "ritorno"]),
    "jerk":               ("JK", "Jerk", "m/s³", ["andata", "ritorno"]),
    "region_transitions": ("RT", "Region transitions", "", ["andata", "ritorno"]),
}


def get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return np.nan
        d = d[k]
    return np.nan if d is None else d


def load(root: Path) -> tuple:
    """Una riga per soggetto, run, metrica e fase."""
    chosen, dropped = find_sessions(root, PATTERN, ("run",))
    rows, problems = [], []
    for (subject, run), path in sorted(chosen.items()):
        s = json.loads(path.read_text(encoding="utf-8"))
        run = int(run)
        if not s.get("success", False):
            problems.append(f"{subject} run {run}: not completed ({s.get('end_reason')})")
        values = {}
        for phase, (_, parts) in PHASES.items():
            if parts is None:
                values[("completion_time", phase)] = get(s, "test_duration") if s.get("success") else np.nan
            elif all(get(s, f"phase_completed.{p}") is True for p in parts):
                values[("completion_time", phase)] = float(sum(get(s, f"phase_durations.{p}") for p in parts))
        nav, missing = navigation_metrics(path)
        values.update(nav)
        if missing:
            problems.append(f"{subject} run {run}: missing {', '.join(missing)}")
        rows += [{"subject": subject, "run": run, "metric": m, "phase": ph, "value": v} for (m, ph), v in values.items()]
    return pd.DataFrame(rows), dropped, problems


def wide(df: pd.DataFrame, m: str, phase: str) -> pd.DataFrame:
    d = df[(df.metric == m) & (df.phase == phase)]
    return d.pivot(index="subject", columns="run", values="value").reindex(columns=list(RUNS))


LEGS = ("andata", "ritorno")   # colonne della figura generale


def plot_panel(ax, df, m, phases, posthocs, colors):
    """Test 1 vs test 2 per ognuna delle `phases` (gruppi sull'asse x)."""
    _, label, unit, _ = METRICS[m]
    groups = [PHASES[p][0] for p in phases]
    data = {}
    for p in phases:
        w = wide(df, m, p)
        for r in RUNS:
            data[(PHASES[p][0], f"Test {r}")] = w[r]
    xpos = condition_plot(ax, data, groups, ["Test 1", "Test 2"], {"Test 1": GREY_BOX, "Test 2": RED_BOX}, colors)
    ax.set_xticklabels(groups)
    ax.set_ylabel(axis_label(label, unit))
    col = "p_t_bonferroni"   # parentesi dal t appaiato, come i post hoc dell'ANOVA (un solo confronto: p non corretto)
    brackets(ax, [(xpos[(PHASES[p][0], "Test 1")], xpos[(PHASES[p][0], "Test 2")], posthocs[(m, p)][col].iloc[0])
                  for p in phases])


def single_plot(df, m, phases, posthocs, colors, handles, out, name, ylim=None):
    """Grafico singolo: senza titolo, legenda di run e soggetti a lato, nota sotto."""
    fig, ax = plt.subplots(figsize=(1.9 + 1.2 * len(phases), 3.6))
    plot_panel(ax, df, m, phases, posthocs, colors)
    if ylim is not None:
        ax.set_ylim(ylim)
    side_legend(ax, handles)
    save_figure(fig, out, name, note=BOX_NOTE)


def plots(df: pd.DataFrame, posthocs: dict, out: Path):
    setup_style()
    colors = subject_colors(df["subject"].unique())
    run_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=f, edgecolor=e, lw=1.2, label=f"Test {r}")
                   for r, (f, e) in zip(RUNS, (GREY_BOX, RED_BOX))]
    handles = run_handles + subject_handles(colors)
    # figura generale: andata a sinistra, ritorno a destra (lunghezze e velocita' del robot diverse);
    # una riga per metrica, stessa scala y nella riga
    fig, axes = plt.subplots(len(METRICS), 2, figsize=(7.6, 3.0 * len(METRICS)))
    letters = iter("ABCDEFGHIJ")
    for row, m in enumerate(METRICS):
        for col, leg in enumerate(LEGS):
            plot_panel(axes[row, col], df, m, [leg], posthocs, colors)
            axes[row, col].set_title(f"({next(letters)})")
    ylim = {m: share_ylim(axes[row]) for row, m in enumerate(METRICS)}
    fig.suptitle("Real robot: test 1 vs test 2\noutbound (left) and return (right), median [IQR] across participants",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(h_pad=1.6, rect=(0, 0.035, 1, 0.975))
    subject_legend(fig, colors, ncol=6, y=0.03, extra=run_handles)
    save_figure(fig, out, "robot_boxplots")
    for m in METRICS:
        for leg in LEGS:
            single_plot(df, m, [leg], posthocs, colors, handles, out,
                        f"single/{m}_{'outbound' if leg == 'andata' else 'return'}_boxplot", ylim[m])
    # a parte: manipolazione (selezione dell'oggetto e posizionamento) e tempo di guida andata + ritorno
    single_plot(df, "completion_time", ["grasp", "posizionamento"], posthocs, colors, handles, out,
                "robot_manipulation_boxplot")
    single_plot(df, "completion_time", ["andata_ritorno"], posthocs, colors, handles, out,
                "robot_total_time_boxplot")


def main():
    p = argparse.ArgumentParser(description="Real robot analysis (test 1 vs test 2)")
    p.add_argument("--results", type=Path, default=BASE_DIR)
    p.add_argument("--out", type=Path, default=BASE_DIR / "statistics")
    args = p.parse_args()

    df, dropped, problems = load(args.results)
    if df.empty:
        print(f"No *_run<N>_session.json in {args.results}")
        return
    report = Report("REAL ROBOT – test 1 vs test 2")
    ids = sorted(df["subject"].unique())
    report.text(f"Participants: {len(ids)} ({', '.join(ids)})")
    for r in RUNS:
        report.text(f"  test {r}: {df[df.run == r]['subject'].nunique()} participants")
    if dropped:
        report.text("Repeated tests discarded (the last one is used): " + ", ".join(p.name for p in dropped))
    for msg in problems:
        report.text("WARNING: " + msg)
    report.text("Outbound = from the start to the object selection (outbound + approach phases); the four")
    report.text("phases (outbound, grasp, return, placement) add up to the total time.")
    report.text("Path length, mean speed and jerk from the base odometry; region transitions = changes of")
    report.text("the cursor region in the 3 x 3 control interface.")

    results, posthocs = [], {}
    for m, (abbr, label, unit, phases) in METRICS.items():
        for ph in phases:
            res = analyse_metric(report, m, f"{label} – {PHASES[ph][0].lower()}", unit, wide(df, m, ph), ["Test"],
                                 [(r, f"test {r}") for r in RUNS], [(1, 2, "test 1 vs test 2")],
                                 digits=0 if m in ("completion_time", "region_transitions") else 2)
            for key in ("descriptives", "anova", "posthoc"):
                res[key].insert(1, "phase", ph)
            results.append(res)
            posthocs[(m, ph)] = res["posthoc"]

    report.section("EXPLORATORY METRICS (not in the figures nor in the jamovi tables)")
    report.text("Why is the base faster in test 2? Share of stops, rate of command changes and how far from")
    report.text("the centre the cursor is pushed (the commanded speed grows with that distance).")
    exploratory = []
    for m, (label, unit, phases) in EXPLORATORY.items():
        for ph in phases:
            res = analyse_metric(report, m, f"{label} – {PHASES[ph][0].lower()}", unit, wide(df, m, ph), ["Test"],
                                 [(r, f"test {r}") for r in RUNS], [(1, 2, "test 1 vs test 2")], digits=2)
            for key in ("descriptives", "anova", "posthoc"):
                res[key].insert(1, "phase", ph)
            exploratory.append(res)

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "robot_subjects.csv", index=False)
    save_tables(results, args.out, "robot")
    save_tables(exploratory, args.out, "robot_exploratory")
    # un file per metrica, come reachy_stats.omv: Test1_andata, Test2_andata, Test1_ritorno, Test2_ritorno
    # (per il completion time seguono grasp, posizionamento e totale)
    for m, (abbr, label, unit, phases) in METRICS.items():
        cols, desc, specs = {"subject": ids}, {}, []
        for ph in ["andata", "ritorno"] + [p_ for p_ in phases if p_ not in ("andata", "ritorno")]:
            specs.append({"title": f"{abbr} – {label} – {ph}", "dep_label": f"{label}{f' ({unit})' if unit else ''}",
                          "factors": [("test", ["test1", "test2"])],
                          "cells": [(f"Test1_{ph}", ["test1"]), (f"Test2_{ph}", ["test2"])],
                          "post_hoc": [["test"]]})
            w = wide(df, m, ph)
            for r in RUNS:
                name = f"Test{r}_{ph}"
                cols[name] = w[r].reindex(ids).to_numpy()
                desc[name] = f"{label}{f' ({unit})' if unit else ''}, {PHASES[ph][0].lower()}, test {r}"
        write_tables(pd.DataFrame(cols), ROOT / "jamovi" / "reachy" / f"{abbr}_{m}", desc, specs)

    plots(df, posthocs, args.out)
    report.save(args.out)


if __name__ == "__main__":
    main()
