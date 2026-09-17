"""Test sul robot reale: confronto pre-training (test1) vs post-training (test2).

File attesi: `<soggetto>_session.json` (+ `<soggetto>_odometry.csv`) come li
salva session_metrics.py. Convenzione: `S000` = test1 (pre), `S000_1` = test2
(post). I file possono stare tutti in una cartella o divisi in test1/ e
test2/ (in quel caso la cartella decide la fase).

Sessioni fallite (abort/quit o meno di --n-objects oggetti spostati) sono
escluse: il soggetto resta nel confronto solo se ha pre e post validi.

Per ogni metrica: media/SD/mediana pre e post, differenza post-pre,
Shapiro-Wilk sulle differenze -> t-test appaiato (p > 0.05) o Wilcoxon,
d di Cohen e r di Wilcoxon.

Output (statistics/), ogni cartella con la sua sottocartella plots/:
- test1/  robot_test1_subjects.csv (una riga per soggetto), robot_test1_statistics.csv
          (per metrica n/mean/sd/sem/median/min/max), robot_test1_regions.csv,
          plots/test1_<metrica>.png (istogramma per i conteggi, boxplot + soggetti
          per le continue), test1_grid.png, test1_regions.png, test1_trajectories.png
- test2/  idem
- test1-vs-test2/  robot_test_prepost_statistics.csv (una riga per metrica),
          robot_test_sessions.csv (tutte le sessioni, con flag valid),
          plots/prepost_<metrica>.png (segmenti per soggetto + media +- SEM, o
          istogramma affiancato per i conteggi), prepost_grid.png,
          regions_heatmap.png, trajectories_prepost.png, trajectories_subjects.png

Uso:
    python3 robot_test_analysis.py [--dirs test1 test2] [--out DIR]
        [--n-objects 3] [--optimal-path M]
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robot_utils import (BASE_DIR, BLUE, DEFAULT_OUT, INK, INK2, ORANGE, box_dots,
                         count_hist, describe, grouped_count_hist, load_session,
                         odometry_metrics, paired_plot, paired_stats, parse_session_name,
                         save_figure, set_title, setup_style)

FAIL_REASONS = ("quit", "abort", "aborted", "error")

# metrica -> (etichetta, meglio se)
METRICS = {
    "test_duration":            ("Test duration [s]", "down"),
    "navigation_duration":      ("Navigation duration [s]", "down"),
    "path_length_navigation":   ("Navigation path length [m]", "down"),
    "path_length_total":        ("Total path length [m]", "down"),
    "path_length_repositioning": ("Repositioning path length [m]", "down"),
    "normalized_path_length":   ("Normalized path length (1 = optimal)", "down"),
    "log_dimensionless_jerk":   ("Log dimensionless jerk (higher = smoother)", "up"),
    "n_repositioning":          ("Repositionings", "down"),
    "n_dwell":                  ("Dwells while driving", "down"),
    "n_dwell_declined":         ("Dwells declined (unintended)", "down"),
    "n_objects_moved":          ("Objects moved", "up"),
    "edges_percent":            ("Time in edge regions (pure motion) [%]", None),
    "corners_percent":          ("Time in corner regions (combined motion) [%]", None),
    "center_percent":           ("Time in central region [%]", None),
    # dall'odometria (navigazione)
    "mean_speed":               ("Mean speed [m/s]", "up"),
    "stopped_percent":          ("Time stopped [%]", "down"),
    "n_stops":                  ("Stops", "down"),
    "backward_percent":         ("Time driving backward [%]", "down"),
    "total_rotation_deg":       ("Total rotation [deg]", "down"),
    "time_max_speed_percent":   ("Navigation time at max speed [%]", "up"),
}
# metriche intere: istogramma (quanti soggetti per valore) invece di linee/boxplot
COUNT_METRICS = {"n_repositioning", "n_dwell", "n_dwell_declined", "n_objects_moved", "n_stops"}
GRID_METRICS = ["test_duration", "navigation_duration", "path_length_navigation",
                "normalized_path_length", "log_dimensionless_jerk", "n_dwell_declined",
                "mean_speed", "stopped_percent", "total_rotation_deg"]
REGIONS = [f"region{i}" for i in range(1, 10)]
EDGES, CORNERS = ["region2", "region4", "region6", "region8"], ["region1", "region3", "region7", "region9"]


def session_row(json_path, phase_hint, n_objects, optimal_path):
    summary, odom = load_session(json_path)
    name = os.path.basename(json_path).replace("_session.json", "")
    subject, rep = parse_session_name(name)
    phase = phase_hint or ("pre" if rep == 0 else "post")
    row = {"subject": subject, "session": name, "phase": phase, "file": str(json_path),
           "end_reason": summary.get("end_reason"), "timestamp": summary.get("timestamp")}
    for k in ("test_duration", "navigation_duration", "path_length_max_speed",
              "path_length_reduced_speed", "path_length_repositioning", "path_length_navigation",
              "path_length_total", "optimal_path_length", "normalized_path_length",
              "log_dimensionless_jerk", "n_repositioning", "n_objects_moved", "n_dwell",
              "n_dwell_declined", "reached_object_selection"):
        row[k] = summary.get(k)
    if optimal_path and row.get("path_length_navigation") is not None:
        row["optimal_path_length"] = optimal_path
        row["normalized_path_length"] = row["path_length_navigation"] / optimal_path
    pct = summary.get("region_time_percent") or {}
    for r in REGIONS:
        row[f"{r}_percent"] = pct.get(r)
    if pct:
        row["edges_percent"] = sum(pct.get(r, 0) for r in EDGES)
        row["corners_percent"] = sum(pct.get(r, 0) for r in CORNERS)
        row["center_percent"] = pct.get("region5", 0)
    reason = str(row["end_reason"] or "").lower()
    row["valid"] = (row.get("n_objects_moved") or 0) >= n_objects and not any(f in reason for f in FAIL_REASONS)
    # odometria: solo la navigazione (fino alla prima selezione)
    if odom is not None and len(odom) > 1:
        nav_end = row.get("navigation_duration")
        nav = odom[odom["t_test"] <= nav_end] if nav_end is not None else odom
        row.update({k: v for k, v in odometry_metrics(nav).items()
                    if k in METRICS or k == "odom_path_length"})
        if "mode" in nav.columns and len(nav):
            row["time_max_speed_percent"] = 100 * float((nav["mode"] == "max_speed").mean())
        row["_odom"] = odom
    return row


def load_all(dirs, n_objects, optimal_path):
    rows = []
    for d in dirs:
        d = Path(d)
        hint = "pre" if d.name.lower() in ("test1", "pre") else "post" if d.name.lower() in ("test2", "post") else None
        for f in sorted(glob.glob(str(d / "*_session.json"))):
            rows.append(session_row(f, hint, n_objects, optimal_path))
            r = rows[-1]
            print(f"[load] {r['session']:12s} {r['phase']:4s} objects={r['n_objects_moved']} "
                  f"end={r['end_reason']} valid={r['valid']}")
    if not rows:
        raise FileNotFoundError(f"Nessun *_session.json in {dirs}")
    df = pd.DataFrame(rows)
    dup = df[df.duplicated(["subject", "phase"], keep=False)]
    if len(dup):
        print(f"[WARN] piu' sessioni per la stessa fase, tengo l'ultima: {sorted(dup['session'])}")
        df = df.sort_values("session").drop_duplicates(["subject", "phase"], keep="last")
    return df


def regions_grid(sub_phase):
    g = sub_phase[[f"{r}_percent" for r in REGIONS]].mean()
    return g.values.reshape(3, 3)


def draw_region_grid(ax, g, title, cmap="Blues", vmin=0, vmax=None, diff=False):
    if diff:
        lim = max(1e-6, np.nanmax(np.abs(g)))
        im = ax.imshow(g, cmap="RdBu_r", vmin=-lim, vmax=lim)
    else:
        im = ax.imshow(g, cmap=cmap, vmin=vmin, vmax=vmax if vmax else max(1e-6, np.nanmax(g)))
    for i in range(3):
        for j in range(3):
            v = g[i, j]
            ax.text(j, i, f"{v:+.1f}" if diff else f"{v:.1f}%", ha="center", va="center", fontsize=10,
                    color="white" if abs(v) > 0.6 * im.get_clim()[1] else INK)
            ax.text(j - 0.42, i - 0.38, str(3 * i + j + 1), fontsize=7, color=INK2)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(title, fontsize=10)


def draw_trajectories(ax, rows, color, label=None):
    for _, r in rows.iterrows():
        o = r["_odom"]
        ax.plot(o["x"] - o["x"].iloc[0], o["y"] - o["y"].iloc[0], color=color, linewidth=1,
                alpha=0.6, label=label)
    ax.plot(0, 0, marker="s", color=INK, markersize=6)
    ax.set_aspect("equal"); ax.grid(True, axis="both"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")


# ---------------------------------------------------------------------------
# Un test (test1 o test2): descrittive su tutti i soggetti
# ---------------------------------------------------------------------------
def analyse_phase(sub, phase, metrics, out_dir, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = sub[sub["phase"] == phase]
    sp.drop(columns=["_odom"]).to_csv(out_dir / f"robot_{tag}_subjects.csv", index=False)
    rows = []
    for m in metrics:
        rows.append({"metric": m, "label": METRICS[m][0], **describe(sp[m])})
    pd.DataFrame(rows).to_csv(out_dir / f"robot_{tag}_statistics.csv", index=False)
    if sp["region1_percent"].notna().any():
        reg = sp[[f"{r}_percent" for r in REGIONS]].agg(["mean", "std", "median"]).T
        reg.to_csv(out_dir / f"robot_{tag}_regions.csv")

    setup_style()
    n = len(sp)
    for m in metrics:
        if sp[m].notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(4, 4))
        if m in COUNT_METRICS:
            count_hist(ax, sp[m]); ax.set_xlabel(METRICS[m][0])
        else:
            box_dots(ax, sp[m], METRICS[m][0])
        set_title(ax, f"{METRICS[m][0].split(' [')[0]} – {tag} (N = {n})", width=32)
        save_figure(fig, f"{tag}_{m}", out_dir)
    gm = [m for m in GRID_METRICS if m in metrics and sp[m].notna().any()]
    if gm:
        ncol = 3; nrow = int(np.ceil(len(gm) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.6 * nrow), squeeze=False)
        for ax, m in zip(axes.flat, gm):
            if m in COUNT_METRICS:
                count_hist(ax, sp[m]); ax.set_xlabel(METRICS[m][0])
            else:
                box_dots(ax, sp[m], "")
            set_title(ax, METRICS[m][0], width=34, fontsize=9.5)
        for ax in list(axes.flat)[len(gm):]:
            ax.axis("off")
        fig.suptitle(f"Real-robot {tag}: all subjects (N = {n})", fontweight="bold")
        fig.tight_layout()
        save_figure(fig, f"{tag}_grid", out_dir)
    if sp["region1_percent"].notna().any():
        fig, ax = plt.subplots(figsize=(4.2, 4))
        draw_region_grid(ax, regions_grid(sp), f"{tag} – % of driving time per region (mean, N = {n})")
        save_figure(fig, f"{tag}_regions", out_dir)
    have = sp[sp["_odom"].notna()]
    if len(have):
        fig, ax = plt.subplots(figsize=(7, 6))
        draw_trajectories(ax, have, BLUE)
        set_title(ax, f"Base trajectories (odometry) – {tag}, all subjects (N = {len(have)})")
        save_figure(fig, f"{tag}_trajectories", out_dir)
    return sp


# ---------------------------------------------------------------------------
# test1 vs test2: confronto appaiato
# ---------------------------------------------------------------------------
def analyse_prepost(sub, df_all, metrics, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    df_all.drop(columns=["_odom"]).to_csv(out_dir / "robot_test_sessions.csv", index=False)
    wide = {m: sub.pivot(index="subject", columns="phase", values=m) for m in metrics}
    rows = []
    for m in metrics:
        w = wide[m]
        if "pre" not in w.columns or "post" not in w.columns:
            continue
        rows.append({"metric": m, "label": METRICS[m][0], "better_if": METRICS[m][1],
                     **paired_stats(w["pre"], w["post"])})
    tests = pd.DataFrame(rows)
    tests.to_csv(out_dir / "robot_test_prepost_statistics.csv", index=False)
    pvals = dict(zip(tests["metric"], tests["p"])) if "p" in tests.columns else {}

    setup_style()
    labels = ("Test 1 (pre)", "Test 2 (post)")

    def draw(ax, m, ylabel):
        w = wide[m]
        if m in COUNT_METRICS:
            n = grouped_count_hist(ax, w["pre"], w["post"], labels, p=pvals.get(m))
            ax.set_xlabel(ylabel or METRICS[m][0])
        else:
            n = paired_plot(ax, w["pre"], w["post"], ylabel, labels, p=pvals.get(m))
        return n

    for m in tests["metric"]:
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        n = draw(ax, m, METRICS[m][0])
        set_title(ax, f"{METRICS[m][0].split(' [')[0]} – test 1 vs test 2 (N = {n})", width=32)
        save_figure(fig, f"prepost_{m}", out_dir)
    gm = [m for m in GRID_METRICS if m in set(tests["metric"])]
    if gm:
        ncol = 3; nrow = int(np.ceil(len(gm) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.8 * nrow), squeeze=False)
        for ax, m in zip(axes.flat, gm):
            n = draw(ax, m, "")
            set_title(ax, METRICS[m][0], width=34, fontsize=9.5)
        for ax in list(axes.flat)[len(gm):]:
            ax.axis("off")
        fig.suptitle(f"Real-robot test: test 1 (pre) vs test 2 (post) – N = {n}, grey = single "
                     f"subjects, blue = mean ± SEM", fontweight="bold")
        fig.tight_layout()
        save_figure(fig, "prepost_grid", out_dir)

    if sub["region1_percent"].notna().any():
        pre, post = regions_grid(sub[sub["phase"] == "pre"]), regions_grid(sub[sub["phase"] == "post"])
        vmax = max(1e-6, np.nanmax([pre, post]))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        draw_region_grid(axes[0], pre, "Test 1 (pre) – % of driving time", vmax=vmax)
        draw_region_grid(axes[1], post, "Test 2 (post) – % of driving time", vmax=vmax)
        draw_region_grid(axes[2], post - pre, "post − pre", diff=True)
        fig.suptitle("Region usage (3×3 cursor grid, mean over subjects)", fontweight="bold")
        fig.tight_layout()
        save_figure(fig, "regions_heatmap", out_dir)

    have = sub[sub["_odom"].notna()]
    if len(have):
        fig, ax = plt.subplots(figsize=(7, 6))
        draw_trajectories(ax, have[have["phase"] == "pre"], ORANGE, labels[0])
        draw_trajectories(ax, have[have["phase"] == "post"], BLUE, labels[1])
        h, l = ax.get_legend_handles_labels(); uniq = dict(zip(l, h))
        ax.legend(uniq.values(), uniq.keys(), loc="best")
        set_title(ax, "Base trajectories (odometry), all subjects – test 1 vs test 2")
        save_figure(fig, "trajectories_prepost", out_dir)
        subjects = sorted(have["subject"].unique())
        ncol = min(4, len(subjects)); nrow = int(np.ceil(len(subjects) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow), squeeze=False)
        for ax, s in zip(axes.flat, subjects):
            for phase, color, lab in (("pre", ORANGE, labels[0]), ("post", BLUE, labels[1])):
                r = have[(have["subject"] == s) & (have["phase"] == phase)]
                if len(r):
                    o = r.iloc[0]["_odom"]
                    ax.plot(o["x"] - o["x"].iloc[0], o["y"] - o["y"].iloc[0], color=color, linewidth=1.2, label=lab)
            ax.set_aspect("equal"); ax.grid(True, axis="both"); ax.set_title(s, fontsize=10)
            ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[len(subjects):]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle("Base trajectories per subject – test 1 (orange) vs test 2 (blue)", fontweight="bold")
        fig.tight_layout()
        save_figure(fig, "trajectories_subjects", out_dir)
    return tests


def main():
    p = argparse.ArgumentParser(description="Test sul robot: test1, test2 e confronto pre/post")
    p.add_argument("--dirs", nargs="+", type=Path, default=[BASE_DIR / "test1", BASE_DIR / "test2"],
                   help="cartelle con i *_session.json (default: test1 test2)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="cartella statistics/ (dentro: test1/, test2/, test1-vs-test2/)")
    p.add_argument("--n-objects", type=int, default=3, help="oggetti da spostare per un test riuscito")
    p.add_argument("--optimal-path", type=float, default=None,
                   help="lunghezza del percorso ottimo [m]; se data ricalcola normalized_path_length")
    args = p.parse_args()

    df = load_all(args.dirs, args.n_objects, args.optimal_path)
    invalid = df[~df["valid"]]
    if len(invalid):
        print(f"[filter] sessioni fallite escluse: {sorted(invalid['session'])}")
    sub = df[df["valid"]].copy()
    metrics = [m for m in METRICS if m in sub.columns and sub[m].notna().any()]

    analyse_phase(sub, "pre", metrics, args.out / "test1", "test1")
    analyse_phase(sub, "post", metrics, args.out / "test2", "test2")
    tests = analyse_prepost(sub, df, metrics, args.out / "test1-vs-test2")

    cols = [c for c in ["metric", "n", "pre_mean", "post_mean", "diff_percent", "test", "p", "cohen_d"]
            if c in tests.columns]
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print("\n" + tests[cols].to_string(index=False))
    print(f"\nOutput in {args.out}/test1, test2, test1-vs-test2")


if __name__ == "__main__":
    main()
