"""Blind reaching test (reaching_blind.py), ripetuto 5 volte nel protocollo.

Input: i file di results_blind/ del repo reachy2_teleop_grasping_and_simulation,
incollati ovunque dentro questa cartella (es. reaching_blind/results/):
    <soggetto>_blind_test<N>[_k]_raw.csv          ogni frame: da qui si calcola TUTTO
    <soggetto>_blind_test<N>[_k]_summary.json     config del test (durata prova, raggio, dwell...)
    <soggetto>_blind_test<N>[_k]_trials.csv       solo per controllo (numero di prove e hit)
Se un test e' stato ripetuto (_1, _2, ...) si usa l'ultima ripetizione.
Le prove interrotte da un abort (piu' corte di trial_s) sono escluse.
Metriche di ogni prova: hit, errori, regioni, target scelto come nel task
(reaching_blind.py); cinematica (linearity index, aspect ratio, jerk, picchi,
velocita', percorso) con le funzioni del laboratorio (common/reach_metrics.py);
piu' le metriche del test intero (tt.m, cursor_distance.m, speed1.m, jerk1.m,
smooth.m, movement_train.m).

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
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import aggregate, find_sessions   # noqa: E402
from common.plots import (CATEGORICAL, INK2, MUTED, grid_axes, save_figure,   # noqa: E402
                          setup_style, spaghetti)
from common.reach_metrics import onset_metrics, reach_metrics, region_of, session_metrics   # noqa: E402
from common.report import Report   # noqa: E402
from common.stats import describe, friedman, holm, wilcoxon_paired   # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
TESTS = (1, 2, 3, 4, 5)
TEST_LABELS = [f"test {t}" for t in TESTS]
# confronti pianificati: (primo, secondo, cosa misura)
PAIRS = ((1, 2, "effect of training 1"), (2, 3, "effect of training 2"),
         (3, 4, "effect of free exploration + robot run 1"), (4, 5, "effect of robot run 2"))
PATTERN = r"(?P<subject>.+)_blind_test(?P<test>\d)(?:_(?P<rep>\d+))?_raw\.csv"
CANVAS_W, CANVAS_H, TARGET_RADIUS = 1200, 650, 70

# metrica -> (etichetta, unita', meglio se 'up'/'down'/None, aggregazione, principale)
METRICS = {
    "hit":                     ("Hit rate", "%", "up", "rate", True),
    "end_error":               ("End-point error", "px", "down", "median", True),
    "reach_time":              ("Reach time = movement time (hits)", "s", "down", "median", True),
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
    "linearity_index":         ("Linearity index", "", "down", "median", False),
    "aspect_ratio":            ("Aspect ratio", "", "down", "median", False),
    "dimensionless_jerk":      ("Dimensionless jerk", "", "down", "median", False),
    "n_speed_peaks":           ("Speed peaks", "", "down", "median", False),
    "path_length":             ("Path length", "px", None, "median", False),
    "mean_speed":              ("Mean speed", "px/s", None, "median", False),
    "sd_speed":                ("SD of speed", "px/s", None, "median", False),
    "peak_speed":              ("Peak speed", "px/s", None, "median", False),
    "hand_lost":               ("Hand lost", "%", "down", "mean", False),
    # test intero (tutti i frame): gli script del laboratorio sul log completo
    "session_log_duration":    ("Test duration (log, tt.m)", "s", None, "mean", False),
    "session_path_length":     ("Cursor path length (cursor_distance.m)", "px", None, "mean", False),
    "session_mean_speed":      ("Mean cursor speed (speed1.m)", "px/s", None, "mean", False),
    "session_peak_speed":      ("Peak cursor speed (speed1.m)", "px/s", None, "mean", False),
    "session_mean_jerk":       ("Mean jerk (jerk1.m)", "px/s^3", "down", "mean", False),
    "session_speed_peaks":     ("Speed peaks (smooth.m)", "", "down", "mean", False),
    "session_region_changes":  ("Region changes (movement_train.m)", "", None, "mean", False),
}
# cinematica di ogni prova dalle funzioni del laboratorio (i "_1s" del training non servono:
# nel blind test il cursore non ricompare mai)
KINEMATIC_KEYS = ("linearity_index", "aspect_ratio", "dimensionless_jerk", "n_speed_peaks", "path_length",
                  "mean_speed", "sd_speed", "peak_speed")
MAIN = [k for k, v in METRICS.items() if v[4]]


def trials_from_raw(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Una riga per prova, tutto dal _raw.csv con le definizioni del task
    (reaching_blind.py: hit, errori, regioni, target scelto) e, per la cinematica,
    le funzioni del laboratorio (common/reach_metrics.py)."""
    trial_s = float(cfg.get("trial_s", 4.0))
    dwell_s = float(cfg.get("dwell_s", 0.5))
    end_s = float(cfg.get("end_window_s", 1.0))
    radius = float(cfg.get("target_radius", TARGET_RADIUS))
    onset_speed = float(cfg.get("motion_onset_speed", 40.0))
    x_edges = cfg.get("region_x_edges", [0, CANVAS_W / 3, 2 * CANVAS_W / 3, CANVAS_W])
    y_edges = cfg.get("region_y_edges", [0, CANVAS_H / 3, 2 * CANVAS_H / 3, CANVAS_H])
    positions = {int(k): tuple(v) for k, v in cfg.get("targets", {}).items()}
    fs = 1.0 / float(raw["time"].diff().median())
    t0 = 0.0   # "time" del raw = secondi dall'ENTER, quando compare il primo target
    rows = []
    for k, g in raw.groupby("trial"):
        t_shown = t0 + (int(k) - 1) * trial_s        # il task cambia target ogni trial_s esatti
        g = g[g["time"] < t_shown + trial_s]         # l'ultimo frame del test cade gia' dopo la prova
        t = g["time"].to_numpy(float)
        xy = g[["cursor_x", "cursor_y"]].to_numpy(float)
        target = int(g["target_id"].iloc[0])
        goal = np.array([g["target_x"].iloc[0], g["target_y"].iloc[0]], float)
        positions.setdefault(target, tuple(goal))
        d = np.hypot(xy[:, 0] - goal[0], xy[:, 1] - goal[1])
        inside = d < radius
        # hit: prima permanenza di dwell_s nel cerchio senza uscirne; t_enter = inizio della permanenza
        t_enter, t_in = None, None
        for ti, ok in zip(t, inside):
            t_in = (t_in if t_in is not None else ti) if ok else None
            if t_in is not None and ti - t_in >= dwell_s:
                t_enter = t_in
                break
        dt = np.diff(t, append=t[-1])
        total = float(dt.sum())
        reg = np.array([region_of(px, py, x_edges, y_edges) for px, py in xy])
        end = t >= t[-1] - end_s
        row = {"trial": int(k), "target": target, "target_number": int(k), "goal_x": goal[0], "goal_y": goal[1],
               "success": bool(t[-1] - t_shown >= trial_s - 2.0 / fs),
               "hit": 100.0 if t_enter is not None else 0.0,
               "reach_time": (t_enter - t_shown) if t_enter is not None else np.nan,
               "time_in_target": 100.0 * float(dt[inside].sum()) / total if total > 0 else np.nan,
               "hand_lost": 100.0 * float(dt[g["hand_detected"].to_numpy() == 0].sum()) / total if total > 0 else np.nan,
               "on_target_at_end": 100.0 if inside[-1] else 0.0,
               "initial_error": float(d[0]), "final_error": float(d[-1]),
               "end_error": float(d[end].mean()), "min_error": float(d.min()),
               "relative_final_error": float(d[-1] / d[0]) if d[0] > 1e-6 else np.nan}
        chosen = min(positions, key=lambda c: np.hypot(xy[-1, 0] - positions[c][0], xy[-1, 1] - positions[c][1]))
        row["chosen_target"], row["chosen_correct"] = chosen, 100.0 if chosen == target else 0.0
        row["region_at_end"] = int(reg[-1])
        row["region_at_end_correct"] = 100.0 if reg[-1] == target else 0.0
        row["time_in_region"] = 100.0 * float(dt[reg == target].sum()) / total if total > 0 else np.nan
        row["end_time_in_region"] = 100.0 * float(dt[end & (reg == target)].sum()) / max(float(dt[end].sum()), 1e-9)
        t_stay, row["region_reach_time"] = None, np.nan
        for ti, ok in zip(t, reg == target):
            t_stay = (t_stay if t_stay is not None else ti) if ok else None
            if t_stay is not None and ti - t_stay >= dwell_s:
                row["region_reach_time"] = t_stay - t_shown
                break
        row["region_hit"] = 0.0 if np.isnan(row["region_reach_time"]) else 100.0
        # cinematica: dall'inizio della prova all'entrata del hit (hit) o alla fine (mancati)
        kin = reach_metrics(t, xy, goal, t_enter if t_enter is not None else t[-1], 0.0, fs)
        row.update({k2: kin[k2] for k2 in KINEMATIC_KEYS})
        om = onset_metrics(t, xy, goal, onset_speed, fs)
        om["reaction_time"] += t[0] - t_shown
        row.update(om)
        row["_x"], row["_y"] = xy[:, 0], xy[:, 1]
        rows.append(row)
    df = pd.DataFrame(rows)
    for key, v in session_metrics(raw["time"], raw[["cursor_x", "cursor_y"]], raw["region"], fs).items():
        df[key] = v
    return df


def load_trials(root: Path, skip_first: bool) -> tuple:
    chosen, dropped = find_sessions(root, PATTERN, ("test",))
    frames, problems = [], []
    for (subject, test), path in sorted(chosen.items()):
        stem = str(path)[:-len("_raw.csv")]
        summary = Path(stem + "_summary.json")
        cfg = json.loads(summary.read_text()).get("config", {}) if summary.exists() else {}
        df = trials_from_raw(pd.read_csv(path), cfg)
        check = Path(stem + "_trials.csv")
        if check.exists():
            tr = pd.read_csv(check)
            if len(tr) != len(df) or (tr["hit"].to_numpy() != df["hit"].to_numpy()).any():
                problems.append(f"{subject} test {test}: prove o hit diversi fra raw e trials.csv")
        df = df[df["success"]]
        if skip_first:
            df = df[df["trial"] != df["trial"].min()]
        df.insert(0, "test", int(test))
        df.insert(0, "subject", subject)
        frames.append(df)
    trials = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return trials, dropped, problems


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
            for _, r in gt.iterrows():
                ax.plot(r["_x"], r["_y"], color=colors.get(r["target"], INK2), lw=0.7, alpha=0.8)
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

    trials, dropped, problems = load_trials(args.results, args.skip_first_trial)
    if trials.empty:
        print(f"Nessun file *_blind_test<N>_raw.csv in {args.results}")
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
    for msg in problems:
        report.text("ATTENZIONE: " + msg)
    report.text("Tutte le metriche sono calcolate dai _raw.csv (cinematica con le funzioni del laboratorio).")
    if args.skip_first_trial:
        report.text("La prima prova di ogni test e' esclusa (--skip-first-trial).")
    report.text("Confronti pianificati: 4 Wilcoxon fra test consecutivi, Holm sui 4, per ogni metrica.")

    subjects = subject_table(trials)
    stats_df, tests_df = analyse(subjects, report)
    per_target_subj = subject_table(trials, by=("subject", "test", "target"))
    per_target = []
    for m in METRICS:
        if m in per_target_subj.columns and not m.startswith("session_"):
            for (tg, t), g in per_target_subj.groupby(["target", "test"]):
                per_target.append({"metric": m, "target": tg, "test": t, **describe(g[m])})
    per_target = pd.DataFrame(per_target)

    subjects.to_csv(args.out / "blind_subjects.csv", index=False)
    stats_df.to_csv(args.out / "blind_statistics.csv", index=False)
    tests_df.to_csv(args.out / "blind_tests.csv", index=False)
    per_target.to_csv(args.out / "blind_per_target.csv", index=False)
    trials.drop(columns=["_x", "_y"]).to_csv(args.out / "blind_trials_all.csv", index=False)
    plots(subjects, per_target, trials, args.out)
    report.section("Metriche esplorative")
    report.text("  Tutte le altre metriche sono in blind_tests.csv e plots/metrics/, calcolate allo stesso modo.")
    report.write(args.out)


if __name__ == "__main__":
    main()
