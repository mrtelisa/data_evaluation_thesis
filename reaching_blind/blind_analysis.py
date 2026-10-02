"""Blind reaching test (reaching_blind.py), ripetuto 5 volte nel protocollo.

Input: <soggetto>_blind_test<N>[_k]_raw.csv (+ _summary.json con la configurazione del
test), ovunque dentro questa cartella (es. reaching_blind/results/). Se un test e'
stato ripetuto si usa l'ultima ripetizione. Ogni test: 12 prove da 4 s, cursore sempre
nascosto; ogni target parte da dove e' finita la prova precedente. Le prove interrotte
(piu' corte di 4 s) sono escluse.

Metriche di ogni prova (errori e linearity index con le definizioni di Bellitto et al.
2025 e computereachmetrics.m, vedi common/reach_metrics.py; "partenza" = posizione del
cursore quando compare il target):
    HR   Hit rate (%)                 cursore nel target (raggio 70 px) per 0.5 s di fila
    TC   Target crossing rate (%)     il cursore entra nel target almeno una volta
    TT   Time on target (s)           tempo totale dentro il target nella prova
    EE   Euclidean error at 4 s (cm)  alla fine della prova, prima del target successivo
    XE   Extent error at 4 s (cm)
    DE   Direction error at 4 s (deg)
    MinD Minimum distance (cm)        distanza minima dal centro del target nella prova
    TTR  Time to reach (s)            comparsa del target -> inizio della permanenza di
                                      0.5 s (solo prove con hit)
    LI   Linearity index (-)          dalla partenza all'entrata nel target (hit) o alla
                                      fine della prova
    MD   Movement duration (s)        inizio del movimento (velocita' > soglia del task,
                                      40 px/s) -> entrata nel target (hit) o ultimo istante
                                      sopra la soglia (prove senza hit)
Metrica esplorativa (solo report e .csv, per la discussione):
    region_correct  % di prove che finiscono nella regione del target (la regione e' il comando
                    sul robot: senza feedback il cursore arriva almeno nella regione giusta?)
Blocchi di 4 prove: 3 per test. Valore di un blocco = media delle sue 4 prove; valore di
un test = media delle 12 prove (TTR: media delle prove con hit, mancante se non ci sono hit).

Analisi (per metrica), come il modello jamovi del training:
    ANOVA a misure ripetute Test (1-5) x Blocco (B1-B3), Greenhouse-Geisser; post hoc t
    appaiati fra test consecutivi (medie dei test) con Bonferroni; controllo Friedman sui
    5 test e Wilcoxon (Holm).

Output:
    statistics/report.txt, blind_trials.csv, blind_subjects.csv (soggetto x test),
    blind_blocks.csv (soggetto x test x blocco), blind_descriptives/anova/friedman/posthoc.csv
    statistics/plots/blind_boxplots.(png|pdf)  box plot (mediana, IQR) + t appaiati (Bonferroni), senza TT e LI
    statistics/plots/single/<metrica>_boxplot.(png|pdf)  i pannelli uno per uno (senza MD e TTR, che
        restano nel report e nelle tabelle jamovi) e linearity_index_no_outlier_boxplot senza il valore
        piu' alto del linearity index, per vedere gli altri soggetti
    ../jamovi/blind/<sigla>_<metrica>.(omv|csv): un file per metrica, una riga per soggetto,
        colonne <sigla>_T1_B1 ... <sigla>_T5_B3 (fattore Test, livelli T1-T5; fattore Blocco, B1-B3)

Uso:
    python3 blind_analysis.py [--results DIR] [--out DIR]
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
from common.plots import (BOX_NOTE, GREY_BOX, axis_label, brackets, condition_plot, save_figure, setup_style,   # noqa: E402
                          share_ylim, side_legend, subject_colors, subject_handles, subject_legend)
from common.reach_metrics import canvas_scale, canvas_to_cm, endpoint_errors, linearity_index, speed   # noqa: E402
from common.report import Report, analyse_metric, save_tables   # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
TESTS = (1, 2, 3, 4, 5)
BLOCK_SIZE_DEFAULT = 4   # summary.json: block_size
PATTERN = r"(?P<subject>.+)_blind_test(?P<test>\d)(?:_(?P<rep>\d+))?_raw\.csv"
# se il summary.json non ha il config: i valori del task
DEFAULTS = {"trial_s": 4.0, "dwell_s": 0.5, "target_radius": 70.0, "motion_onset_speed": 40.0}

# metrica -> (sigla jamovi, etichetta, unita')
METRICS = {
    "hit":               ("HR", "Hit rate", "%"),
    "crossed":           ("TC", "Target crossing rate", "%"),
    "time_on_target":    ("TT", "Time on target", "s"),
    "euclidean_error":   ("EE", "Euclidean error at 4 s", "cm"),
    "extent_error":      ("XE", "Extent error at 4 s", "cm"),
    "direction_error":   ("DE", "Direction error at 4 s", "deg"),
    "min_distance":      ("MinD", "Minimum distance to target", "cm"),
    "time_to_reach":     ("TTR", "Time to reach (hits)", "s"),
    "linearity_index":   ("LI", "Linearity index", "-"),
    "movement_duration": ("MD", "Movement duration", "s"),
}
# metriche nei grafici (MD e TTR solo nel report e nelle tabelle jamovi)
PLOTTED = [m for m in METRICS if m not in ("movement_duration", "time_to_reach")]
# figura generale: senza time on target e linearity index (solo nei grafici singoli)
OVERVIEW = [m for m in PLOTTED if m not in ("time_on_target", "linearity_index")]
# stessa unita' e stessa grandezza (quota di prove): stessa scala y, anche nei grafici singoli
SHARED_Y = [("hit", "crossed")]


def trials_from_raw(raw: pd.DataFrame, cfg: dict, scale: float) -> pd.DataFrame:
    """Una riga per prova, tutto dal _raw.csv. Hit e ingresso nel target in px del canvas
    (come il task), errori e distanze in cm."""
    c = {k: float(cfg.get(k, v)) for k, v in DEFAULTS.items()}
    x_edges, y_edges = cfg.get("region_x_edges", [0, 400, 800, 1200]), cfg.get("region_y_edges", [0, 650 / 3, 1300 / 3, 650])
    region = lambda x, y: (min(max(int(np.searchsorted(y_edges, y, "right")) - 1, 0), 2) * 3
                           + min(max(int(np.searchsorted(x_edges, x, "right")) - 1, 0), 2) + 1)
    fs = 1.0 / float(raw["time"].diff().median())
    rows = []
    for k, g in raw.groupby("trial"):
        t_shown = (int(k) - 1) * c["trial_s"]          # "time" = secondi dal primo target; un target ogni trial_s
        g = g[g["time"] < t_shown + c["trial_s"]]
        t = g["time"].to_numpy(float)
        xy = g[["cursor_x", "cursor_y"]].to_numpy(float)
        goal = np.array([g["target_x"].iloc[0], g["target_y"].iloc[0]], float)
        inside = np.hypot(xy[:, 0] - goal[0], xy[:, 1] - goal[1]) < c["target_radius"]
        xy_cm, goal_cm = canvas_to_cm(xy, scale), canvas_to_cm(goal, scale)
        d = np.hypot(xy_cm[:, 0] - goal_cm[0], xy_cm[:, 1] - goal_cm[1])
        # hit: prima permanenza di dwell_s nel cerchio senza uscirne; t_enter = inizio della permanenza
        t_enter, t_in = None, None
        for ti, ok in zip(t, inside):
            t_in = (t_in if t_in is not None else ti) if ok else None
            if t_in is not None and ti - t_in >= c["dwell_s"]:
                t_enter = t_in
                break
        dt = np.diff(t, append=t_shown + c["trial_s"])
        row = {"trial": int(k), "target": int(g["target_id"].iloc[0]),
               "complete": bool(t[-1] - t_shown >= c["trial_s"] - 2.0 / fs),
               "hit": 100.0 if t_enter is not None else 0.0,
               "crossed": 100.0 if inside.any() else 0.0,
               "time_on_target": float(dt[inside].sum()),
               "min_distance": float(d.min()),
               "time_to_reach": t_enter - t_shown if t_enter is not None else np.nan}
        row.update(endpoint_errors(xy_cm[0], xy_cm[-1], goal_cm))
        reach = t <= (t_enter if t_enter is not None else t[-1]) + 1e-9
        row["linearity_index"] = linearity_index(xy_cm[reach])
        row["region_correct"] = 100.0 if region(*xy[-1]) == region(*goal) else 0.0
        v = speed(t, xy) if len(t) >= 3 else np.zeros(len(t))
        moving = np.flatnonzero(v > c["motion_onset_speed"])
        row["movement_duration"] = np.nan
        if moving.size:
            t_end = t_enter if t_enter is not None else t[moving[-1]]
            if t_end > t[moving[0]]:
                row["movement_duration"] = float(t_end - t[moving[0]])
        row["_x"], row["_y"] = xy[:, 0], xy[:, 1]
        rows.append(row)
    return pd.DataFrame(rows)


def load(root: Path) -> tuple:
    chosen, dropped = find_sessions(root, PATTERN, ("test",))
    frames, problems = [], []
    for (subject, test), path in sorted(chosen.items()):
        summary = Path(str(path)[:-len("_raw.csv")] + "_summary.json")
        info = json.loads(summary.read_text()) if summary.exists() else {}
        df = trials_from_raw(pd.read_csv(path), info.get("config", {}), canvas_scale(info))
        n_bad = int((~df["complete"]).sum())
        if n_bad:
            problems.append(f"{subject} test {test}: {n_bad} interrupted trial(s) excluded")
        df = df[df["complete"]].drop(columns="complete")
        df.insert(1, "block", (df["trial"] - 1) // int(info.get("block_size", BLOCK_SIZE_DEFAULT)) + 1)
        df.insert(0, "test", int(test))
        df.insert(0, "subject", subject)
        frames.append(df)
    trials = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return trials, dropped, problems


def wide(subjects: pd.DataFrame, m: str) -> pd.DataFrame:
    return subjects.pivot(index="subject", columns="test", values=m).reindex(columns=list(TESTS))


def block_wide(blocks: pd.DataFrame, m: str, n_blocks: int) -> pd.DataFrame:
    """soggetti x (test, blocco), in ordine Test1 B1..B3, ..., Test5 B1..B3."""
    w = blocks.pivot_table(index="subject", columns=["test", "block"], values=m)
    return w.reindex(columns=pd.MultiIndex.from_product([list(TESTS), range(1, n_blocks + 1)]))


def plot_panel(ax, subjects, m, ph, colors, hide=()):
    """`hide`: (soggetto, test) da non disegnare; le parentesi restano quelle dei post hoc su tutti i dati."""
    _, label, unit = METRICS[m]
    w = wide(subjects, m)
    for subj, t in hide:
        w.loc[subj, t] = np.nan
    groups = [f"Test {t}" for t in TESTS]
    data = {(f"Test {t}", "all"): w[t] for t in TESTS}
    xpos = condition_plot(ax, data, groups, ["all"], {"all": GREY_BOX}, colors)
    ax.set_xticklabels(groups)
    ax.set_ylabel(axis_label(label, unit))
    if m == "extent_error":
        ax.axhline(0, color="#8c8c8c", lw=0.8, ls=":", zorder=0)
    col = "p_t_bonferroni"   # parentesi sempre dai t appaiati (Bonferroni), come i post hoc dell'ANOVA
    brackets(ax, [(xpos[(f"Test {a}", "all")], xpos[(f"Test {b}", "all")], p)
                  for (a, b), p in zip(zip(TESTS, TESTS[1:]), ph[col])])


def plots(subjects: pd.DataFrame, posthocs: dict, out: Path):
    setup_style()
    colors = subject_colors(subjects["subject"].unique())
    # valore piu' alto del linearity index: grafico in piu' senza di lui, per vedere gli altri soggetti
    top = subjects.loc[subjects["linearity_index"].idxmax()]
    hide = [(top["subject"], int(top["test"]))]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    by_metric = dict(zip(OVERVIEW, axes.ravel()))
    for (m, ax), letter in zip(by_metric.items(), "ABCDEF"):
        plot_panel(ax, subjects, m, posthocs[m], colors)
        ax.set_title(f"({letter})")
    ylim = {}
    for group in SHARED_Y:
        lim = share_ylim([by_metric[m] for m in group])
        ylim.update({m: lim for m in group})
    fig.suptitle("Blind reaching test (median [IQR] across participants)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(h_pad=1.6, rect=(0, 0.03, 1, 0.985))
    subject_legend(fig, colors, y=0.035)
    save_figure(fig, out, "blind_boxplots")
    for m in PLOTTED:
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        plot_panel(ax, subjects, m, posthocs[m], colors)
        if m in ylim:
            ax.set_ylim(ylim[m])
        side_legend(ax, subject_handles(colors))
        save_figure(fig, out, f"single/{m}_boxplot", note=BOX_NOTE)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    plot_panel(ax, subjects, "linearity_index", posthocs["linearity_index"], colors, hide)
    side_legend(ax, subject_handles(colors))
    save_figure(fig, out, "single/linearity_index_no_outlier_boxplot",
                note=f"{BOX_NOTE} {top['subject']} test {int(top['test'])} not shown "
                     f"(LI = {top['linearity_index']:.1f}).")


def main():
    p = argparse.ArgumentParser(description="Blind reaching test analysis (5 tests)")
    p.add_argument("--results", type=Path, default=BASE_DIR)
    p.add_argument("--out", type=Path, default=BASE_DIR / "statistics")
    args = p.parse_args()

    trials, dropped, problems = load(args.results)
    if trials.empty:
        print(f"No *_blind_test<N>_raw.csv in {args.results}")
        return
    report = Report("BLIND REACHING TEST – 5 repetitions")
    ids = sorted(trials["subject"].unique())
    report.text(f"Participants: {len(ids)} ({', '.join(ids)})")
    for t in TESTS:
        who = trials[trials.test == t].groupby("subject").size()
        report.text(f"  test {t}: {len(who)} participants, {who.min() if len(who) else 0}-"
                    f"{who.max() if len(who) else 0} trials each")
    if dropped:
        report.text("Repeated tests discarded (the last one is used): " + ", ".join(p.name for p in dropped))
    for msg in problems:
        report.text("WARNING: " + msg)
    report.text("Blocks of 4 trials (3 per test); block value = mean of its trials, test value = mean of the 12")
    report.text("trials (time to reach: mean of the hits). ANOVA on Test x Block, post hoc on the test values.")
    report.text("Errors at the end of the 4 s trial, from the cursor position at target onset.")

    subjects = trials.groupby(["subject", "test"], as_index=False)[list(METRICS)].mean()
    blocks = trials.groupby(["subject", "test", "block"], as_index=False)[list(METRICS)].mean()
    n_blocks = int(blocks["block"].max())
    # colonne (test, blocco) per l'ANOVA e (test, 0) = media del test per post hoc e Friedman
    pairs = [((a, 0), (b, 0), f"test {a} vs test {b}") for a, b in zip(TESTS, TESTS[1:])]
    results, posthocs = [], {}
    for m, (abbr, label, unit) in METRICS.items():
        wb = block_wide(blocks, m, n_blocks)
        wt = wide(subjects, m)
        wt.columns = pd.MultiIndex.from_tuples([(t, 0) for t in wt.columns])
        res = analyse_metric(report, m, label, "" if unit == "-" else unit, pd.concat([wb, wt], axis=1),
                             ["Test", "Block"], [(c, f"test {c[0]} B{c[1]}") for c in wb.columns], pairs,
                             digits=1 if unit in ("deg", "%") else 2,
                             friedman_sets=[("5 tests", [(t, 0) for t in TESTS])])
        n_missing = int(wb.isna().any(axis=1).sum())
        if n_missing:
            report.text(f"  NOTE: {n_missing} participant(s) without a value in some block (excluded from the ANOVA)")
        results.append(res)
        posthocs[m] = res["posthoc"]

    report.section("EXPLORATORY METRIC (not in the figures nor in the jamovi tables)")
    report.text("Does the cursor end at least in the region of the target, which is the command on the robot?")
    rc = trials.groupby(["subject", "test"], as_index=False)["region_correct"].mean()
    exploratory = [analyse_metric(report, "region_correct", "Trials ending in the region of the target", "%",
                                  wide(rc, "region_correct"), ["Test"], [(t, f"test {t}") for t in TESTS],
                                  [(a, b, f"test {a} vs test {b}") for a, b in zip(TESTS, TESTS[1:])], digits=1)]

    args.out.mkdir(parents=True, exist_ok=True)
    save_tables(exploratory, args.out, "blind_exploratory")
    trials.drop(columns=["_x", "_y"]).to_csv(args.out / "blind_trials.csv", index=False)
    subjects.to_csv(args.out / "blind_subjects.csv", index=False)
    blocks.to_csv(args.out / "blind_blocks.csv", index=False)
    save_tables(results, args.out, "blind")
    for m, (abbr, label, unit) in METRICS.items():
        w = block_wide(blocks, m, n_blocks)
        cols, desc = {"subject": ids}, {}
        for t, b in w.columns:
            name = f"{abbr}_T{t}_B{b}"
            cols[name] = w[(t, b)].reindex(ids).to_numpy()
            desc[name] = (f"{label}{f' ({unit})' if unit != '-' else ''}, blind test {t}, block {b} = mean of trials "
                          f"{(b - 1) * 4 + 1}-{b * 4}")
        spec = {"title": f"{abbr} – {label}{f' ({unit})' if unit != '-' else ''}",
                "dep_label": f"{label}{f' ({unit})' if unit != '-' else ''}",
                "factors": [("Test", [f"T{t}" for t in TESTS]), ("Blocco", [f"B{b}" for b in range(1, n_blocks + 1)])],
                "cells": [(f"{abbr}_T{t}_B{b}", [f"T{t}", f"B{b}"]) for t, b in w.columns],
                "post_hoc": []}   # niente post hoc: con 2 fattori le tabelle diventano troppo lunghe
        write_tables(pd.DataFrame(cols), ROOT / "jamovi" / "blind" / f"{abbr}_{m}", desc, [spec])

    plots(subjects, posthocs, args.out)
    report.save(args.out)


if __name__ == "__main__":
    main()
