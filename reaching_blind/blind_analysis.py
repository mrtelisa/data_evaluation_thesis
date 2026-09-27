"""Blind reaching test (reaching_blind.py), ripetuto 5 volte nel protocollo.

Input: i file di results_blind/ del repo reachy2_teleop_grasping_and_simulation,
incollati ovunque dentro questa cartella (es. reaching_blind/results/):
    <soggetto>_blind_test<N>[_k]_trials.csv       una riga per prova (12 per test)
    <soggetto>_blind_test<N>[_k]_trajectory.csv   ogni campione del cursore (per i grafici)
Se un test e' stato ripetuto (_1, _2, ...) si usa l'ultima ripetizione.
Le prove interrotte da un abort (success = False) sono escluse.

Analisi (vedi README.md):
1. per ogni soggetto e test, un valore per metrica: la percentuale per le
   metriche si'/no (hit, region accuracy), la mediana per le altre;
2. descrittive tra soggetti per ogni test: mediana [Q1-Q3], media, SD;
3. Friedman sui 5 test (c'e' qualche differenza?) con W di Kendall;
4. Wilcoxon sui 4 confronti fra test consecutivi, ognuno legato a una fase del
   protocollo, con correzione di Holm sui 4 e rank-biserial r come effect size:
       1 vs 2  effetto del training 1
       2 vs 3  effetto del training 2 (plateau?)
       3 vs 4  effetto di free exploration + run 1 sul robot (fatica?)
       4 vs 5  effetto della run 2
Le 5 metriche principali (MAIN) sono quelle dell'analisi; le altre sono
calcolate allo stesso modo ma sono esplorative.

Output in reaching_blind/statistics/:
    report.txt              il riassunto da leggere (italiano)
    blind_subjects.csv      soggetto x test: un valore per metrica
    blind_statistics.csv    metrica x test: descrittive tra soggetti
    blind_tests.csv         Friedman + 4 Wilcoxon per metrica (p, p Holm, effetto)
    blind_per_target.csv    metrica x target x test: mediana tra soggetti
    blind_trials_all.csv    tutte le prove usate, con soggetto e test
    plots/blind_main.png                le 5 metriche principali (una linea per soggetto)
    plots/metrics/<metrica>.png         lo stesso grafico per ogni metrica
    plots/blind_per_target.png          metriche principali per target
    plots/trajectories/<soggetto>.png   traiettorie del cursore nei 5 test

Uso:
    python3 blind_analysis.py [--results DIR] [--out DIR] [--skip-first-trial]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import aggregate, find_sessions, to_bool   # noqa: E402
from common.plots import (CATEGORICAL, INK2, MUTED, grid_axes, save_figure,   # noqa: E402
                          setup_style, spaghetti)
from common.report import Report   # noqa: E402
from common.stats import describe, friedman, holm, wilcoxon_paired   # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
TESTS = (1, 2, 3, 4, 5)
TEST_LABELS = [f"test {t}" for t in TESTS]
# confronti pianificati: (primo, secondo, cosa misura)
PAIRS = ((1, 2, "effect of training 1"), (2, 3, "effect of training 2"),
         (3, 4, "effect of free exploration + robot run 1"), (4, 5, "effect of robot run 2"))
PATTERN = r"(?P<subject>.+)_blind_test(?P<test>\d)(?:_(?P<rep>\d+))?_trials\.csv"
CANVAS_W, CANVAS_H, TARGET_RADIUS = 1200, 650, 70

# metrica -> (etichetta, unita', meglio se 'up'/'down'/None, aggregazione, principale)
METRICS = {
    "hit":                     ("Hit rate", "%", "up", "rate", True),
    "end_error":               ("End-point error", "px", "down", "median", True),
    "reach_time":              ("Reach time (hits)", "s", "down", "median", True),
    "initial_direction_error": ("Initial direction error", "deg", "down", "median", True),
    "region_at_end_correct":   ("Region accuracy", "%", "up", "rate", True),
    "final_error":             ("Final error", "px", "down", "median", False),
    "min_error":               ("Minimum error", "px", "down", "median", False),
    "relative_final_error":    ("Relative final error", "", "down", "median", False),
    "time_in_target":          ("Time in target", "%", "up", "mean", False),
    "on_target_at_end":        ("On target at end", "%", "up", "rate", False),
    "chosen_correct":          ("Closest target correct", "%", "up", "rate", False),
    "region_hit":              ("Region hit (0.5 s)", "%", "up", "rate", False),
    "region_reach_time":       ("Region reach time", "s", "down", "median", False),
    "end_time_in_region":      ("Last second in region", "%", "up", "mean", False),
    "time_in_region":          ("Time in region", "%", "up", "mean", False),
    "initial_direction_error_peak": ("Initial direction error (peak)", "deg", "down", "median", False),
    "reaction_time":           ("Reaction time", "s", "down", "median", False),
    "movement_time":           ("Movement time", "s", "down", "median", False),
    "normalized_path_length":  ("Normalized path length", "", "down", "median", False),
    "max_deviation":           ("Max deviation", "px", "down", "median", False),
    "log_dimensionless_jerk":  ("Smoothness (-ln DJ)", "", "up", "median", False),
    "n_speed_peaks":           ("Speed peaks", "", "down", "median", False),
    "mean_speed":              ("Mean speed", "px/s", None, "median", False),
    "peak_speed":              ("Peak speed", "px/s", None, "median", False),
    "hand_lost":               ("Hand lost", "%", "down", "mean", False),
}
MAIN = [k for k, v in METRICS.items() if v[4]]


def load_trials(root: Path, skip_first: bool) -> tuple:
    chosen, dropped = find_sessions(root, PATTERN, ("test",))
    frames = []
    for (subject, test), path in sorted(chosen.items()):
        df = pd.read_csv(path)
        df = df[to_bool(df["success"])] if "success" in df.columns else df
        if skip_first:
            df = df[df["trial"] != df["trial"].min()]
        df.insert(0, "test", int(test))
        df.insert(0, "subject", subject)
        df["_trajectory"] = str(path).replace("_trials.csv", "_trajectory.csv")
        frames.append(df)
    trials = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return trials, dropped


def subject_table(trials: pd.DataFrame, by=("subject", "test")) -> pd.DataFrame:
    """Un valore per soggetto (e test, target...) per ogni metrica presente."""
    rows = []
    for key, g in trials.groupby(list(by)):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row["n_trials"] = len(g)
        for m, (_, _, _, how, _) in METRICS.items():
            if m in g.columns:
                row[m] = aggregate(g[m], how)
        rows.append(row)
    return pd.DataFrame(rows)


def wide(subjects: pd.DataFrame, metric: str) -> pd.DataFrame:
    """soggetti x test (colonne 1..5) per una metrica."""
    return subjects.pivot(index="subject", columns="test", values=metric).reindex(columns=list(TESTS))


def analyse(subjects: pd.DataFrame, report: Report) -> tuple:
    stat_rows, test_rows = [], []
    for m, (label, unit, better, _, main) in METRICS.items():
        if m not in subjects.columns:
            continue
        w = wide(subjects, m)
        descr = {t: describe(w[t]) for t in TESTS}
        for t in TESTS:
            stat_rows.append({"metric": m, "label": label, "unit": unit, "main": main, "test": t, **descr[t]})
        fr = friedman(w)
        test_rows.append({"metric": m, "label": label, "main": main, "comparison": "Friedman tests 1-5",
                          "test": "Friedman", "n": fr["n"], "statistic": fr["statistic"], "p": fr["p"],
                          "p_holm": np.nan, "effect": fr["effect"], "effect_name": "Kendall W"})
        pair_tests = [wilcoxon_paired(w[a], w[b]) for a, b, _ in PAIRS]
        p_holm = holm([t["p"] for t in pair_tests])
        for (a, b, what), t, ph in zip(PAIRS, pair_tests, p_holm):
            delta = descr[b].get("median", np.nan) - descr[a].get("median", np.nan) \
                if descr[a].get("n") and descr[b].get("n") else np.nan
            test_rows.append({"metric": m, "label": label, "main": main, "comparison": f"test {a} vs test {b}",
                              "meaning": what, "test": "Wilcoxon", "n": t["n"], "statistic": t["statistic"],
                              "p": t["p"], "p_holm": ph, "effect": t["effect"],
                              "effect_name": "rank-biserial r", "median_first": descr[a].get("median"),
                              "median_second": descr[b].get("median"), "delta_median": delta,
                              "better": better})
        # report: solo le metriche principali per esteso
        if main:
            report.section(f"{label}{f' [{unit}]' if unit else ''}")
            report.text("  per test, mediana [Q1-Q3]: " + " | ".join(
                f"test {t}: " + (f"{descr[t]['median']:.2f} [{descr[t]['q1']:.2f}-{descr[t]['q3']:.2f}]"
                                 if descr[t].get("n") else "-") for t in TESTS))
            report.text(f"  Friedman (c'e' qualche differenza fra i 5 test?): n={fr['n']}, "
                        f"p={fr['p']:.3f}, W={fr['effect']:.2f}" if fr["n"] >= 2 and not np.isnan(fr["p"])
                        else "  Friedman: servono almeno 2 soggetti con tutti e 5 i test")
            for (a, b, what), t, ph in zip(PAIRS, pair_tests, p_holm):
                report.comparison(f"{what} (test {a} vs {b})", "", better, descr[a], descr[b], t, ph,
                                  f"test {a}", f"test {b}")
    return pd.DataFrame(stat_rows), pd.DataFrame(test_rows)


def plot_metric(ax, subjects, m):
    label, unit, better, _, _ = METRICS[m]
    spaghetti(ax, wide(subjects, m), TEST_LABELS, ylabel=f"{label}{f' [{unit}]' if unit else ''}",
              title=label + ("  (higher = better)" if better == "up" else "  (lower = better)" if better == "down" else ""))
    # what happens between two tests, under the axis between the ticks
    trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, (_, _, what) in enumerate(PAIRS):
        ax.text(i + 0.5, -0.1, what.replace("effect of ", "").replace(" + ", "\n+ "), transform=trans,
                ha="center", va="top", fontsize=6.5, color=MUTED)


def plots(subjects, per_target, trials, out):
    setup_style()
    fig, axes = grid_axes(len(MAIN), ncols=3, width=4.6, height=3.9)
    for ax, m in zip(axes, MAIN):
        plot_metric(ax, subjects, m)
    fig.suptitle("Blind reaching test across the protocol (one line per subject)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out, "blind_main")
    for m in METRICS:
        if m in subjects.columns and subjects[m].notna().any():
            fig, ax = plt.subplots(figsize=(5.5, 4.1))
            plot_metric(ax, subjects, m)
            save_figure(fig, out, f"metrics/{m}")

    # metriche principali per target: mediana tra soggetti, una linea per target
    targets = sorted(per_target["target"].unique())
    fig, axes = grid_axes(len(MAIN), ncols=3, width=4.6, height=3.4)
    for ax, m in zip(axes, MAIN):
        for i, tg in enumerate(targets):
            s = per_target[(per_target.metric == m) & (per_target.target == tg)].set_index("test")["median"]
            s = s.reindex(list(TESTS))
            ax.plot(range(len(TESTS)), s.values, marker="o", color=CATEGORICAL[i % len(CATEGORICAL)],
                    lw=1.8, label=f"target {tg}")
        label, unit = METRICS[m][0], METRICS[m][1]
        ax.set_xticks(range(len(TESTS)))
        ax.set_xticklabels(TEST_LABELS)
        ax.set_ylabel(f"{label}{f' [{unit}]' if unit else ''}")
        ax.set_title(label, loc="left")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Blind reaching test per target (median across subjects)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out, "blind_per_target")

    # traiettorie: una figura per soggetto, un pannello per test
    for subject, g in trials.groupby("subject"):
        fig, axes = plt.subplots(1, len(TESTS), figsize=(3.3 * len(TESTS), 2.3), squeeze=False)
        for ax, t in zip(axes[0], TESTS):
            ax.set_xlim(0, CANVAS_W)
            ax.set_ylim(CANVAS_H, 0)
            ax.set_aspect("equal")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for x in (400, 800):
                ax.axvline(x, color="#eeeeee", lw=0.8)
            for y in (CANVAS_H / 3, 2 * CANVAS_H / 3):
                ax.axhline(y, color="#eeeeee", lw=0.8)
            ax.set_title(f"test {t}", loc="left", fontsize=9)
            gt = g[g.test == t]
            if gt.empty:
                continue
            tpos = gt.groupby("target")[["goal_x", "goal_y"]].first()
            colors = {tg: CATEGORICAL[i % len(CATEGORICAL)] for i, tg in enumerate(sorted(tpos.index))}
            for tg, (gx, gy) in tpos.iterrows():
                ax.add_patch(plt.Circle((gx, gy), TARGET_RADIUS, fill=False, color=colors[tg], lw=1.2))
            traj_path = Path(gt["_trajectory"].iloc[0])
            if traj_path.exists():
                tr = pd.read_csv(traj_path)
                used = set(gt["trial"])
                for trial, seg in tr.groupby("trial"):
                    if trial in used:
                        ax.plot(seg["x"], seg["y"], color=colors.get(seg["target"].iloc[0], INK2), lw=0.7, alpha=0.8)
        fig.suptitle(f"{subject} – cursor paths (colour = target)", x=0.01, ha="left", fontweight="bold", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        save_figure(fig, out, f"trajectories/{subject}")


def main():
    p = argparse.ArgumentParser(description="Analisi dei 5 blind reaching test")
    p.add_argument("--results", type=Path, default=BASE_DIR, help="cartella con i file di results_blind/")
    p.add_argument("--out", type=Path, default=BASE_DIR / "statistics")
    p.add_argument("--skip-first-trial", action="store_true",
                   help="esclude la prima prova di ogni test (parte dalla posizione della mano, non da un target)")
    args = p.parse_args()

    trials, dropped = load_trials(args.results, args.skip_first_trial)
    if trials.empty:
        print(f"Nessun file *_blind_test<N>_trials.csv in {args.results}")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    report = Report("BLIND REACHING TEST – 5 ripetizioni")
    subjects_ids = sorted(trials["subject"].unique())
    report.text(f"Soggetti: {len(subjects_ids)} ({', '.join(subjects_ids)})")
    for t in TESTS:
        who = sorted(trials.loc[trials.test == t, "subject"].unique())
        missing = sorted(set(subjects_ids) - set(who))
        report.text(f"  test {t}: {len(who)} soggetti" + (f" (mancano: {', '.join(missing)})" if missing else ""))
    if dropped:
        report.text("Ripetizioni scartate (si usa l'ultima): " + ", ".join(p.name for p in dropped))
    if args.skip_first_trial:
        report.text("La prima prova di ogni test e' esclusa (--skip-first-trial).")
    report.text("Confronti pianificati: 4 Wilcoxon fra test consecutivi, Holm sui 4, per ogni metrica.")

    subjects = subject_table(trials)
    stats_df, tests_df = analyse(subjects, report)
    per_target_subj = subject_table(trials, by=("subject", "test", "target"))
    per_target = []
    for m in METRICS:
        if m in per_target_subj.columns:
            for (tg, t), g in per_target_subj.groupby(["target", "test"]):
                per_target.append({"metric": m, "target": tg, "test": t, **describe(g[m])})
    per_target = pd.DataFrame(per_target)

    subjects.to_csv(args.out / "blind_subjects.csv", index=False)
    stats_df.to_csv(args.out / "blind_statistics.csv", index=False)
    tests_df.to_csv(args.out / "blind_tests.csv", index=False)
    per_target.to_csv(args.out / "blind_per_target.csv", index=False)
    trials.drop(columns=["_trajectory"]).to_csv(args.out / "blind_trials_all.csv", index=False)
    plots(subjects, per_target, trials, args.out)
    report.section("Metriche esplorative")
    report.text("  Tutte le altre metriche sono in blind_tests.csv e plots/metrics/, calcolate allo stesso modo.")
    report.write(args.out)


if __name__ == "__main__":
    main()
