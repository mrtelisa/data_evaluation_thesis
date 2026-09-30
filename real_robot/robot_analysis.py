"""Test sul robot reale (reachy_control.py), fatto due volte: run 1 e run 2.

Input: i file di results_robot/ del repo reachy2_teleop_grasping_and_simulation,
incollati ovunque dentro questa cartella (es. real_robot/results/):
    <soggetto>_run<N>[_k]_session.json   metriche della sessione (session_metrics.py)
    <soggetto>_run<N>[_k]_odometry.csv   odometria a 20 Hz (per i grafici delle traiettorie)
Se una run e' stata ripetuta (_1, _2, ...) si usa l'ultima ripetizione.

Il compito e' diviso in fasi: andata (outbound), avvicinamento al tavolo
(approach), presa (grasp), ritorno con la borraccia (return), appoggio
(placement). Run 1 e run 2 si confrontano fase per fase (andata con andata,
ritorno con ritorno): stessa velocita', stesso percorso, stesso carico.
La durata di una fase conta solo se la fase e' stata completata.

Analisi (vedi README.md):
1. metriche PRIMARIE (4): Wilcoxon run 1 vs run 2, Holm sulle 4, rank-biserial r;
2. tutte le altre metriche: stessi numeri, ma esplorative (p non corretto);
3. conteggi (successo, collisioni, borracce cadute...): solo descrittivi;
4. uso delle regioni, matrice dei cambi di comando e traiettorie: grafici.

Output in real_robot/statistics/:
    report.txt                 il riassunto da leggere (italiano)
    robot_subjects.csv         soggetto x run: tutte le metriche
    robot_statistics.csv       metrica x run: descrittive tra soggetti
    robot_tests.csv            run 1 vs run 2 per metrica (primarie con Holm)
    robot_counts.csv           successi, collisioni, cadute, ... per run
    robot_regions.csv          uso delle regioni per fase e run
    plots/robot_primary.png, plots/robot_phases.png, plots/metrics/<metrica>.png,
    plots/robot_regions.png, plots/robot_transitions.png,
    plots/trajectories/<soggetto>.png, plots/robot_trajectories_all.png

Uso:
    python3 robot_analysis.py [--results DIR] [--out DIR]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import find_sessions   # noqa: E402
from common.plots import (CATEGORICAL, FIRST_COLOR, INK, INK2, MUTED, SECOND_COLOR,   # noqa: E402
                          SEQUENTIAL_CMAP, grid_axes, heatmap3x3, save_figure, setup_style, spaghetti)
from common.report import Report   # noqa: E402
from robot_raw import raw_metrics   # noqa: E402
from common.stats import describe, fmt_num, holm, wilcoxon_paired   # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
RUNS = (1, 2)
PATTERN = r"(?P<subject>.+)_run(?P<run>\d)(?:_(?P<rep>\d+))?_session\.json"
PHASES = ("outbound", "approach", "grasp", "return", "placement")
DRIVING = ("outbound", "approach", "return")
MODE_COLORS = {"max_speed": CATEGORICAL[0], "reduced_speed": CATEGORICAL[2],
               "repositioning": CATEGORICAL[3], "transport": CATEGORICAL[1]}

# Le 4 metriche primarie (la famiglia della correzione di Holm): cambiale qui se servono altre
PRIMARY = ("phase_durations.outbound", "phase_durations.return",
           "raw.outbound.non_adjacent_percent", "raw.outbound.command_changes_per_min")

# chiave nel json (con i punti) -> (etichetta, unita', meglio se)
METRICS = {
    "test_duration": ("Test duration", "s", "down"),
    **{f"phase_durations.{p}": (f"{p.capitalize()} time", "s", "down") for p in PHASES},
    "n_dwell_declined": ("Dwells declined", "", "down"),
    "n_repositioning": ("Repositionings", "", "down"),
    "n_repositioning_per_phase.grasp": ("Repositionings – grasp", "", "down"),
    "n_repositioning_per_phase.placement": ("Repositionings – placement", "", "down"),
    "manual.n_collisions": ("Collisions", "", "down"),
}
# metriche di guida: tutte ricalcolate dai file raw (robot_raw.py)
for _ph in DRIVING:
    METRICS.update({
        f"raw.{_ph}.stop_percent": (f"Stop time – {_ph}", "%", "down"),
        f"raw.{_ph}.command_changes_per_min": (f"Command changes – {_ph}", "/min", "down"),
        f"raw.{_ph}.non_adjacent_percent": (f"Non-adjacent changes – {_ph}", "%", "down"),
        f"raw.{_ph}.reversal_percent": (f"Reversals – {_ph}", "%", "down"),
        f"raw.{_ph}.mean_command_duration": (f"Mean command duration – {_ph}", "s", "up"),
        f"raw.{_ph}.stop_passage_percent": (f"Passages through stop – {_ph}", "%", "down"),
        f"raw.{_ph}.sequence_entropy": (f"Sequence entropy – {_ph}", "bits", "down"),
        f"raw.{_ph}.driving_time": (f"Driving time – {_ph}", "s", "down"),
        f"raw.{_ph}.robot_path_length": (f"Path length – {_ph} (cursor_distance.m)", "m", "down"),
        f"raw.{_ph}.cumulative_rotation": (f"Cumulative rotation – {_ph}", "rad", "down"),
        # script del laboratorio
        f"raw.{_ph}.region_changes_per_min": (f"Region changes – {_ph} (movement_train.m)", "/min", "down"),
        f"raw.{_ph}.robot_lin_speed_mean": (f"Robot linear speed – {_ph} (speed1.m)", "m/s", None),
        f"raw.{_ph}.robot_lin_speed_peak": (f"Robot peak linear speed – {_ph} (speed1.m)", "m/s", None),
        f"raw.{_ph}.robot_ang_speed_mean": (f"Robot angular speed – {_ph} (speed1.m)", "rad/s", None),
        f"raw.{_ph}.robot_mean_jerk": (f"Robot mean jerk – {_ph} (jerk1.m)", "m/s^3", "down"),
        f"raw.{_ph}.robot_log10_dimensionless_jerk": (f"Robot dimensionless jerk, log10 – {_ph} (jerk1.m)", "", "down"),
        f"raw.{_ph}.cursor_path_length": (f"Cursor path length – {_ph} (cursor_distance.m)", "px", None),
        f"raw.{_ph}.cursor_mean_speed": (f"Cursor speed – {_ph} (speed1.m)", "px/s", None),
        f"raw.{_ph}.cursor_mean_jerk": (f"Cursor mean jerk – {_ph} (jerk1.m)", "px/s^3", "down"),
        f"raw.{_ph}.cursor_speed_peaks": (f"Cursor speed peaks – {_ph} (smooth.m)", "", "down"),
    })
# conteggi rari: solo descrittivi (i dropped dipendono dal robot, non dal soggetto)
COUNTS = {"success": "Task completed", "manual.n_collisions": "Collisions",
          "manual.n_drops_grasp": "Bottles dropped – grasp (robot)",
          "manual.n_drops_transport": "Bottles dropped – transport/placement (robot)",
          "n_dwell_declined": "Dwells declined", "n_repositioning": "Repositionings"}


def get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return np.nan
        d = d[k]
    return np.nan if d is None else d


def load(root: Path) -> tuple:
    chosen, dropped = find_sessions(root, PATTERN, ("run",))
    rows, sessions, problems = [], {}, []
    for (subject, run), path in sorted(chosen.items()):
        s = json.loads(path.read_text(encoding="utf-8"))
        run = int(run)
        sessions[(subject, run)] = (s, path)
        row = {"subject": subject, "run": run, "end_reason": s.get("end_reason"),
               "success": bool(s.get("success", s.get("n_objects_moved", 0) > 0))}
        for key in set(METRICS) | set(COUNTS):
            if key != "success":
                row[key] = get(s, key)
        raw, missing = raw_metrics(path)
        row.update(raw)
        if missing:
            problems.append(f"{subject} run {run}: mancano {', '.join(missing)}")
        # durata di una fase solo se completata
        for ph in PHASES:
            if not get(s, f"phase_completed.{ph}") is True:
                row[f"phase_durations.{ph}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows), sessions, dropped, problems


def analyse(df, report):
    stat_rows, test_rows = [], []
    results = {}
    for key, (label, unit, better) in METRICS.items():
        w = df.pivot(index="subject", columns="run", values=key).reindex(columns=list(RUNS)).apply(
            pd.to_numeric, errors="coerce")
        d = {r: describe(w[r]) for r in RUNS}
        for r in RUNS:
            stat_rows.append({"metric": key, "label": label, "unit": unit, "primary": key in PRIMARY, "run": r, **d[r]})
        results[key] = (w, d, wilcoxon_paired(w[1], w[2]))
    p_holm = dict(zip(PRIMARY, holm([results[k][2]["p"] for k in PRIMARY])))
    for key, (w, d, t) in results.items():
        label, unit, better = METRICS[key]
        test_rows.append({"metric": key, "label": label, "unit": unit, "primary": key in PRIMARY,
                          "comparison": "run 1 vs run 2", "n": t["n"], "statistic": t["statistic"], "p": t["p"],
                          "p_holm": p_holm.get(key, np.nan), "effect": t["effect"], "effect_name": "rank-biserial r",
                          "median_run1": d[1].get("median"), "median_run2": d[2].get("median"), "better": better})

    report.section("Metriche primarie (Holm sulle 4)")
    for key in PRIMARY:
        label, unit, better = METRICS[key]
        w, d, t = results[key]
        report.comparison(label, unit, better, d[1], d[2], t, p_holm[key], "run 1", "run 2")
    report.section("Le 13 metriche, per fase (esplorative: p non corretto)")
    groups = [
        ("1. Successo e tempo per fase", ["test_duration"] + [f"phase_durations.{p}" for p in PHASES]),
        ("2. Cambi di comando", [f"raw.{p}.command_changes_per_min" for p in DRIVING]),
        ("3. Insicurezza: dwell rifiutati e repositioning",
         ["n_dwell_declined", "n_repositioning_per_phase.grasp", "n_repositioning_per_phase.placement"]),
        ("4. Tempo fermo (regione 5, dwell esclusi)", [f"raw.{p}.stop_percent" for p in DRIVING]),
        ("5. Collisioni (contate a mano)", ["manual.n_collisions"]),
        ("6. Salti non adiacenti", [f"raw.{p}.non_adjacent_percent" for p in DRIVING]),
        ("7. Inversioni (sovracorrezioni)", [f"raw.{p}.reversal_percent" for p in DRIVING]),
        ("8. Durata media di un comando", [f"raw.{p}.mean_command_duration" for p in DRIVING]),
        ("9. Passaggi per lo stop (stop-and-go vs fluido)", [f"raw.{p}.stop_passage_percent" for p in DRIVING]),
        ("10. Entropia di sequenza", [f"raw.{p}.sequence_entropy" for p in DRIVING]),
        ("11. Path length", [f"raw.{p}.robot_path_length" for p in ("outbound", "return")]),
        ("12. Rotazione cumulata", [f"raw.{p}.cumulative_rotation" for p in ("outbound", "return")]),
        ("Script del laboratorio (esplorative)",
         [f"raw.{p}.{m}" for p in DRIVING for m in (
             "region_changes_per_min", "robot_lin_speed_mean", "robot_lin_speed_peak", "robot_ang_speed_mean",
             "robot_mean_jerk", "robot_log10_dimensionless_jerk", "cursor_path_length", "cursor_mean_speed",
             "cursor_mean_jerk", "cursor_speed_peaks")]),
    ]
    for title, keys in groups:
        report.text(f"  {title}")
        for key in keys:
            label, unit, better = METRICS[key]
            w, d, t = results[key]
            report.comparison("  " + label, unit, better, d[1], d[2], t, None, "run 1", "run 2")
    return pd.DataFrame(stat_rows), pd.DataFrame(test_rows)


def counts(df, report):
    rows = []
    report.section("13. Conteggi (solo descrittivi)")
    for r in RUNS:
        g = df[df.run == r]
        row = {"run": r, "n_sessions": len(g)}
        for key, label in COUNTS.items():
            v = pd.to_numeric(g[key], errors="coerce") if key != "success" else g[key].astype(float)
            row[f"{key}_total"] = float(v.sum())
            row[f"{key}_sessions_with"] = int((v > 0).sum())
        rows.append(row)
        report.text(f"  run {r}: {int(g['success'].sum())}/{len(g)} compiti completati; "
                    f"collisioni {row['manual.n_collisions_total']:.0f} "
                    f"(in {row['manual.n_collisions_sessions_with']} sessioni); borracce cadute al grasp "
                    f"{row['manual.n_drops_grasp_total']:.0f}, nel trasporto {row['manual.n_drops_transport_total']:.0f} "
                    f"(dipendono dal robot); dwell rifiutati {row['n_dwell_declined_total']:.0f}; "
                    f"repositioning {row['n_repositioning_total']:.0f}")
        failed = g[~g["success"]]
        for _, f in failed.iterrows():
            report.text(f"    {f['subject']} run {r} non completata: {f['end_reason']}")
    return pd.DataFrame(rows)


def region_table(sessions):
    rows = []
    for (subject, run), (s, _) in sessions.items():
        for ph in DRIVING + ("all",):
            pct = get(s, f"driving.{ph}.region_time_percent")
            if isinstance(pct, dict):
                for k, v in pct.items():
                    rows.append({"subject": subject, "run": run, "phase": ph, "region": int(k.replace("region", "")),
                                 "percent": v})
    return pd.DataFrame(rows)


def load_odometry(json_path: Path):
    """Odometria allineata: parte da (0, 0) con direzione iniziale lungo +x
    (ogni run ha il suo zero di odometria)."""
    csv = json_path.with_name(json_path.name.replace("_session.json", "_odometry.csv"))
    if not csv.exists():
        return None
    od = pd.read_csv(csv)
    if od.empty:
        return None
    x0, y0 = od["x"].iloc[0], od["y"].iloc[0]
    th0 = od["theta"].dropna().iloc[0] if od["theta"].notna().any() else 0.0
    c, s = math.cos(-th0), math.sin(-th0)
    dx, dy = od["x"] - x0, od["y"] - y0
    od["xa"], od["ya"] = c * dx - s * dy, s * dx + c * dy
    return od


def plots(df, sessions, regions, out):
    setup_style()
    fig, axes = grid_axes(len(PRIMARY), ncols=4, width=3.3, height=3.4)
    for ax, key in zip(axes, PRIMARY):
        label, unit, _ = METRICS[key]
        w = df.pivot(index="subject", columns="run", values=key).reindex(columns=list(RUNS))
        spaghetti(ax, w, ["run 1", "run 2"], ylabel=f"{label}{f' [{unit}]' if unit else ''}", title=label)
    fig.suptitle("Real robot – primary metrics, run 1 vs run 2 (one line per subject)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, out, "robot_primary")

    fig, axes = grid_axes(len(PHASES), ncols=5, width=2.9, height=3.3)
    for ax, ph in zip(axes, PHASES):
        w = df.pivot(index="subject", columns="run", values=f"phase_durations.{ph}").reindex(columns=list(RUNS))
        spaghetti(ax, w, ["run 1", "run 2"], ylabel="time [s]", title=ph)
    fig.suptitle("Duration of every phase (completed phases only)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, out, "robot_phases")

    for key, (label, unit, _) in METRICS.items():
        w = df.pivot(index="subject", columns="run", values=key).reindex(columns=list(RUNS))
        if w.apply(pd.to_numeric, errors="coerce").notna().any().any():
            fig, ax = plt.subplots(figsize=(3.8, 3.6))
            spaghetti(ax, w.apply(pd.to_numeric, errors="coerce"), ["run 1", "run 2"],
                      ylabel=f"{label}{f' [{unit}]' if unit else ''}", title=label)
            save_figure(fig, out, f"metrics/{key.replace('.', '_')}")

    if not regions.empty:
        fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.6), squeeze=False)
        for i, r in enumerate(RUNS):
            for j, ph in enumerate(("outbound", "return", "all")):
                sub = regions[(regions.run == r) & (regions.phase == ph)]
                vals = sub.groupby("region")["percent"].median().to_dict()
                heatmap3x3(axes[i, j], vals, title=f"run {r} – {ph}", vmin=0, vmax=max(regions["percent"].max(), 1),
                           fmt="{:.0f}%")
        fig.suptitle("Driving time per region (median across subjects; 5 = stop, dwells excluded)", x=0.01,
                     ha="left", fontweight="bold", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure(fig, out, "robot_regions")

    # matrice dei cambi di comando (tutti i soggetti insieme): riga = da, colonna = a
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, r in zip(axes, RUNS):
        mat = np.zeros((9, 9))
        for (subject, run), (s, _) in sessions.items():
            if run != r:
                continue
            for seq in get(s, "driving.all.command_sequences") if isinstance(get(s, "driving.all.command_sequences"), list) else []:
                for a, b in zip(seq, seq[1:]):
                    mat[a - 1, b - 1] += 1
        rows = mat.sum(axis=1, keepdims=True)
        share = np.divide(mat, rows, out=np.zeros_like(mat), where=rows > 0) * 100
        ax.imshow(share, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=100)
        for a in range(9):
            for b in range(9):
                if mat[a, b]:
                    adjacent = max(abs(a // 3 - b // 3), abs(a % 3 - b % 3)) <= 1
                    ax.text(b, a, f"{share[a, b]:.0f}", ha="center", va="center", fontsize=6.5,
                            color="white" if share[a, b] > 60 else (INK if adjacent else "#c0392b"),
                            fontweight="normal" if adjacent else "bold")
        ax.set_xticks(range(9))
        ax.set_xticklabels(range(1, 10))
        ax.set_yticks(range(9))
        ax.set_yticklabels(range(1, 10))
        ax.set_xlabel("to region")
        ax.set_ylabel("from region")
        ax.grid(False)
        ax.set_title(f"run {r} – command changes (% of row)", loc="left", fontsize=10)
    fig.text(0.01, 0.01, "Red bold = non-adjacent change (the cursor swept across the grid).", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, out, "robot_transitions")

    # traiettorie
    subjects = sorted({s for s, _ in sessions})
    fig_all, axes_all = plt.subplots(1, 2, figsize=(10, 4.5), squeeze=False)
    for subject in subjects:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), squeeze=False)
        for ax, r in zip(axes[0], RUNS):
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(f"run {r}", loc="left", fontsize=9)
            ax.grid(True, axis="both")
            if (subject, r) not in sessions:
                continue
            od = load_odometry(sessions[(subject, r)][1])
            if od is None:
                continue
            for mode, seg in od.groupby((od["mode"] != od["mode"].shift()).cumsum()):
                m = seg["mode"].iloc[0]
                ax.plot(seg["xa"], seg["ya"], color=MODE_COLORS.get(m, INK2), lw=1.4)
            ax.plot(0, 0, "o", color=INK, ms=4)
            ax_all = axes_all[0][r - 1]
            ax_all.plot(od["xa"], od["ya"], lw=0.9, alpha=0.7,
                        color=CATEGORICAL[subjects.index(subject) % len(CATEGORICAL)], label=subject)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
        handles = [plt.Line2D([], [], color=c, lw=2, label=m.replace("_", " ")) for m, c in MODE_COLORS.items()]
        axes[0][1].legend(handles=handles, fontsize=7.5, loc="best")
        fig.suptitle(f"{subject} – base trajectory (odometry, start at the origin)", x=0.01, ha="left",
                     fontweight="bold", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure(fig, out, f"trajectories/{subject}")
    for r, ax in zip(RUNS, axes_all[0]):
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, axis="both")
        ax.set_title(f"run {r} – all subjects", loc="left", fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    axes_all[0][1].legend(fontsize=7, loc="best")
    fig_all.tight_layout()
    save_figure(fig_all, out, "robot_trajectories_all")


def main():
    p = argparse.ArgumentParser(description="Analisi del test sul robot reale (run 1 vs run 2)")
    p.add_argument("--results", type=Path, default=BASE_DIR, help="cartella con i file di results_robot/")
    p.add_argument("--out", type=Path, default=BASE_DIR / "statistics")
    args = p.parse_args()

    df, sessions, dropped, problems = load(args.results)
    if df.empty:
        print(f"Nessun file *_run<N>_session.json in {args.results}")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    report = Report("ROBOT REALE – run 1 vs run 2")
    ids = sorted(df["subject"].unique())
    report.text(f"Soggetti: {len(ids)} ({', '.join(ids)})")
    for r in RUNS:
        who = sorted(df.loc[df.run == r, "subject"])
        missing = sorted(set(ids) - set(who))
        report.text(f"  run {r}: {len(who)} soggetti" + (f" (mancano: {', '.join(missing)})" if missing else ""))
    if dropped:
        report.text("Ripetizioni scartate (si usa l'ultima): " + ", ".join(p.name for p in dropped))
    for msg in problems:
        report.text("ATTENZIONE: " + msg)
    report.text("Metriche di guida ricalcolate dai file raw (_regions, _odometry, _cursor.csv); durate delle fasi,")
    report.text("dwell, repositioning e conteggi manuali dal _session.json (nel raw mancano gli istanti degli eventi).")
    report.text("Confronti: Wilcoxon run 1 vs run 2; Holm sulle 4 metriche primarie, le altre sono esplorative.")

    stats_df, tests_df = analyse(df, report)
    counts_df = counts(df, report)
    regions = region_table(sessions)
    df.to_csv(args.out / "robot_subjects.csv", index=False)
    stats_df.to_csv(args.out / "robot_statistics.csv", index=False)
    tests_df.to_csv(args.out / "robot_tests.csv", index=False)
    counts_df.to_csv(args.out / "robot_counts.csv", index=False)
    regions.to_csv(args.out / "robot_regions.csv", index=False)
    plots(df, sessions, regions, args.out)
    report.write(args.out)


if __name__ == "__main__":
    main()
