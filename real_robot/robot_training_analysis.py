"""Training di navigazione sul robot reale: reaching dei coni dall'odometria.

File attesi in training/: `<soggetto>_session.json` + `<soggetto>_odometry.csv`
(l'odometria parte azzerata all'inizio del training). Le posizioni dei
bersagli sono in training_targets.csv (name, x, y, radius [m], nel frame
dell'odometria: origine = partenza del robot); `home` e' il punto di partenza.

Un bersaglio e' "raggiunto" quando il robot entra nel suo raggio. La
traiettoria viene divisa in tratte (leg) fra un'uscita da una zona e
l'ingresso nella successiva: le tratte home -> cono sono i reaching, quelle
cono -> home i rientri. Per ogni tratta: durata, lunghezza del percorso,
percorso normalizzato (percorso / distanza rettilinea uscita-ingresso),
velocita' media, arresti, smoothness (-ln DJ). Per ogni cono la ripetizione
(1a, 2a, 3a visita) da' la curva di apprendimento. Dei coni mai raggiunti si
riporta la distanza minima.

Output (statistics/training/), statistiche solo aggregate su tutti i soggetti;
l'unico risultato per singolo utente sono le traiettorie:
- robot_training_statistics.csv   metriche di sessione: n/mean/sd/sem/median/min/max
- robot_training_repetitions.csv  per ripetizione: media/SD/SEM tra soggetti (reaching)
- robot_training_tests.csv        Friedman sulle ripetizioni, Wilcoxon 1a vs ultima
- robot_training_per_cone.csv     per cono e ripetizione: media/SD tra soggetti
- robot_training_targets.csv      per cono: quanti soggetti l'hanno raggiunto, distanza minima media
- robot_training_legs.csv         tutte le tratte di tutti i soggetti (dato grezzo)
- plots/training_curve_<metrica>.png, training_curves_grid.png,
  training_trajectories_all.png, trajectories/<soggetto>.png

Uso:
    python3 robot_training_analysis.py [--dir training] [--targets training_targets.csv]
        [--out DIR]
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from robot_utils import (BASE_DIR, BLUE, DEFAULT_OUT, INK, INK2, ORANGE, SUBJECT_LINE,
                         count_stops, describe, load_session, load_targets,
                         log_dimensionless_jerk, odometry_metrics, parse_session_name,
                         path_length, save_figure, set_title, setup_style)

LEG_METRICS = {
    "duration":               ("Time to reach [s]", "down"),
    "path_length":            ("Path length [m]", "down"),
    "normalized_path_length": ("Normalized path length (1 = straight)", "down"),
    "mean_speed":             ("Mean speed [m/s]", "up"),
    "n_stops":                ("Stops", "down"),
    "stopped_percent":        ("Time stopped [%]", "down"),
    "total_rotation_deg":     ("Total rotation [deg]", "down"),
    "log_dimensionless_jerk": ("Log dimensionless jerk (higher = smoother)", "up"),
}
GRID_METRICS = ["duration", "path_length", "normalized_path_length", "mean_speed",
                "n_stops", "log_dimensionless_jerk"]


def segment_legs(odom, targets):
    """Tratte fra le zone (home e coni) attraversate dal robot."""
    x, y, t = odom["x"].values, odom["y"].values, odom["t_test"].values
    names = targets["name"].tolist()
    inside = np.zeros((len(names), len(x)), dtype=bool)
    for k, (_, tg) in enumerate(targets.iterrows()):
        inside[k] = np.hypot(x - tg["x"], y - tg["y"]) <= tg["radius"]
    # sequenza di visite: (zona, i_ingresso, i_uscita)
    visits, current, i_enter = [], None, None
    for i in range(len(x)):
        zone = next((k for k in range(len(names)) if inside[k, i]), None)
        if zone != current:
            if current is not None:
                visits.append((current, i_enter, i - 1))
            current, i_enter = zone, i
    if current is not None:
        visits.append((current, i_enter, len(x) - 1))
    legs = []
    for (z0, _, i_out), (z1, i_in, _) in zip(visits[:-1], visits[1:]):
        if i_in - i_out < 2:
            continue
        seg = odom.iloc[i_out:i_in + 1]
        sx, sy, st = seg["x"].values, seg["y"].values, seg["t"].values
        plen = path_length(sx, sy)
        straight = float(np.hypot(sx[-1] - sx[0], sy[-1] - sy[0]))
        n_stops, stopped = count_stops(st, seg["speed"].values)
        dur = float(seg["t_test"].iloc[-1] - seg["t_test"].iloc[0])
        leg = {"from": names[z0], "to": names[z1],
               "t_start": float(seg["t_test"].iloc[0]), "t_end": float(seg["t_test"].iloc[-1]),
               "duration": dur, "path_length": plen, "straight_distance": straight,
               "normalized_path_length": plen / straight if straight > 0.05 else np.nan,
               "mean_speed": float(seg["speed"].mean()), "n_stops": n_stops,
               "stopped_percent": 100 * stopped / dur if dur > 0 else np.nan,
               "log_dimensionless_jerk": log_dimensionless_jerk(st, sx, sy)}
        if "theta" in seg.columns and seg["theta"].notna().any():
            leg["total_rotation_deg"] = float(np.degrees(np.sum(np.abs(np.diff(np.unwrap(seg["theta"].values))))))
        legs.append(leg)
    legs = pd.DataFrame(legs)
    if len(legs):
        legs["kind"] = np.where(legs["from"] == "home", "reach",
                                np.where(legs["to"] == "home", "return", "cone_to_cone"))
        legs["repetition"] = legs.groupby(["kind", "to"]).cumcount() + 1
        legs["leg"] = np.arange(1, len(legs) + 1)
    return legs


def target_summary(odom, targets, legs):
    x, y = odom["x"].values, odom["y"].values
    rows = []
    for _, tg in targets.iterrows():
        if tg["name"] == "home":
            continue
        d = np.hypot(x - tg["x"], y - tg["y"])
        n_reached = int(((legs["to"] == tg["name"]) & (legs["kind"] == "reach")).sum()) if len(legs) else 0
        rows.append({"target": tg["name"], "reached": bool(n_reached), "n_reached": n_reached,
                     "min_distance": float(d.min()), "t_min_distance": float(odom["t_test"].values[d.argmin()])})
    return rows


def plot_trajectory(subject, odom, targets, legs, out_dir):
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(odom["x"], odom["y"], c=odom["t_test"], cmap="Blues", s=3, zorder=2)
    ax.plot(odom["x"], odom["y"], color=SUBJECT_LINE, linewidth=0.6, zorder=1)
    for _, tg in targets.iterrows():
        col = INK if tg["name"] == "home" else ORANGE
        ax.add_patch(plt.Circle((tg["x"], tg["y"]), tg["radius"], fill=False, color=col, linewidth=1.5))
        ax.plot(tg["x"], tg["y"], marker="x", color=col, markersize=6)
        ax.text(tg["x"], tg["y"] + tg["radius"] + 0.1, tg["name"], ha="center", color=col, fontsize=9)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8); cb.set_label("time [s]", color=INK2)
    ax.set_aspect("equal"); ax.grid(True, axis="both")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    n_reach = int((legs["kind"] == "reach").sum()) if len(legs) else 0
    set_title(ax, f"{subject} – training trajectory ({n_reach} reaches)")
    fig.savefig(Path(out_dir) / "plots" / "trajectories" / f"{subject}.png"); plt.close(fig)


def draw_curve(ax, wide, metric, style_color=BLUE):
    reps = sorted(wide.columns); x = np.arange(len(reps))
    for _, s in wide.iterrows():
        ax.plot(x, s[reps].values, color=SUBJECT_LINE, linewidth=0.8, marker="o", markersize=2.5, zorder=1)
    mean = wide[reps].mean(); n = wide[reps].count()
    sem = (wide[reps].std(ddof=1) / np.sqrt(n)).fillna(0)
    ax.fill_between(x, mean - sem, mean + sem, color=style_color, alpha=0.15, lw=0, zorder=2)
    ax.errorbar(x, mean, yerr=sem, color=style_color, linewidth=2, marker="o", markersize=5, capsize=3, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f"{r}" for r in reps]); ax.set_xlabel("Repetition")
    ax.set_ylabel(LEG_METRICS[metric][0]); ax.set_xlim(-0.5, len(reps) - 0.5)
    return int(n.max())


def main():
    p = argparse.ArgumentParser(description="Training sul robot: reaching dei coni")
    p.add_argument("--dir", type=Path, default=BASE_DIR / "training")
    p.add_argument("--targets", type=Path, default=BASE_DIR / "training_targets.csv")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT / "training")
    p.add_argument("--max-repetitions", type=int, default=None,
                   help="ripetizioni per cono da usare nella curva (default: tutte)")
    args = p.parse_args()
    (args.out / "plots" / "trajectories").mkdir(parents=True, exist_ok=True)
    targets = load_targets(args.targets)
    if "home" not in set(targets["name"]):
        raise ValueError("training_targets.csv deve avere una riga 'home'")
    print(f"[targets] {', '.join(f'{r.name}({r.x},{r.y},r={r.radius})' for r in targets.itertuples())}")

    files = sorted(glob.glob(str(args.dir / "*_session.json")))
    if not files:
        raise FileNotFoundError(f"Nessun *_session.json in {args.dir}")
    all_legs, all_targets, sessions = [], [], []
    for f in files:
        summary, odom = load_session(f)
        subject, _ = parse_session_name(os.path.basename(f).replace("_session.json", ""))
        srow = {"subject": subject, "end_reason": summary.get("end_reason"),
                "test_duration": summary.get("test_duration"),
                "path_length_total": summary.get("path_length_total"),
                "log_dimensionless_jerk_session": summary.get("log_dimensionless_jerk"),
                "n_dwell": summary.get("n_dwell"), "n_dwell_declined": summary.get("n_dwell_declined")}
        for r, v in (summary.get("region_time_percent") or {}).items():
            srow[f"{r}_percent"] = v
        if odom is None or len(odom) < 5:
            print(f"[load] {subject}: odometria assente, solo metriche di sessione")
            sessions.append(srow); continue
        odom = odom.copy()
        odom["x"] -= odom["x"].iloc[0]; odom["y"] -= odom["y"].iloc[0]   # partenza = origine
        srow.update(odometry_metrics(odom))
        legs = segment_legs(odom, targets)
        legs.insert(0, "subject", subject)
        tsum = target_summary(odom, targets, legs)
        for t in tsum:
            t["subject"] = subject
        all_legs.append(legs); all_targets += tsum; sessions.append(srow)
        n_reach = int((legs["kind"] == "reach").sum()) if len(legs) else 0
        reached = [t["target"] for t in tsum if t["reached"]]
        print(f"[load] {subject}: {len(odom)} campioni, {n_reach} reaching, coni raggiunti: {reached}")
        plot_trajectory(subject, odom, targets, legs, args.out)

    sess = pd.DataFrame(sessions)
    num = [c for c in sess.columns if c not in ("subject", "end_reason") and pd.api.types.is_numeric_dtype(sess[c])]
    pd.DataFrame([{"metric": c, **describe(sess[c])} for c in num]).to_csv(
        args.out / "robot_training_statistics.csv", index=False)
    legs = pd.concat(all_legs, ignore_index=True) if all_legs else pd.DataFrame()
    legs.to_csv(args.out / "robot_training_legs.csv", index=False)
    tg = pd.DataFrame(all_targets)
    if len(tg):
        tg.groupby("target").agg(n_subjects=("subject", "nunique"), n_subjects_reached=("reached", "sum"),
                                 mean_n_reached=("n_reached", "mean"), min_distance_mean=("min_distance", "mean"),
                                 min_distance_sd=("min_distance", "std")).reset_index().to_csv(
            args.out / "robot_training_targets.csv", index=False)
    if legs.empty or not (legs["kind"] == "reach").any():
        print("Nessun reaching rilevato: controlla training_targets.csv (unita' in metri, raggio)")
        return

    # --- curva per ripetizione (media sui coni per soggetto, poi tra soggetti) ---
    reach = legs[legs["kind"] == "reach"]
    if args.max_repetitions:
        reach = reach[reach["repetition"] <= args.max_repetitions]
    metrics = [m for m in LEG_METRICS if m in reach.columns and reach[m].notna().any()]
    per_subj = reach.groupby(["subject", "repetition"])[metrics].mean().reset_index()
    rows, tests = [], []
    for m in metrics:
        wide = per_subj.pivot(index="subject", columns="repetition", values=m)
        for rep in wide.columns:
            v = wide[rep].dropna(); n = len(v)
            row = {"metric": m, "repetition": rep, "n": n}
            if n:
                row.update(mean=v.mean(), median=v.median(), min=v.min(), max=v.max())
            if n > 1:
                row["sd"] = v.std(ddof=1); row["sem"] = row["sd"] / np.sqrt(n)
            rows.append(row)
        reps = sorted(wide.columns)
        trow = {"metric": m, "better_if": LEG_METRICS[m][1], "n_repetitions": len(reps),
                "first_mean": wide[reps[0]].mean(), "last_mean": wide[reps[-1]].mean()}
        trow["delta_percent"] = 100 * (trow["last_mean"] - trow["first_mean"]) / trow["first_mean"] if trow["first_mean"] else np.nan
        pair = wide[[reps[0], reps[-1]]].dropna()
        if len(pair) >= 5:
            try:
                trow["wilcoxon_first_last_p"] = stats.wilcoxon(pair.iloc[:, 0], pair.iloc[:, 1], zero_method="zsplit").pvalue
            except ValueError:
                pass
        full = wide.dropna()
        if len(full) >= 3 and len(reps) >= 3:
            trow["friedman_p"] = stats.friedmanchisquare(*[full[r] for r in reps]).pvalue
        tests.append(trow)
    pd.DataFrame(rows).to_csv(args.out / "robot_training_repetitions.csv", index=False)
    tests = pd.DataFrame(tests)
    tests.to_csv(args.out / "robot_training_tests.csv", index=False)
    # per cono
    per_cone = reach.groupby(["to", "repetition"])[metrics].agg(["mean", "std", "count"])
    per_cone.to_csv(args.out / "robot_training_per_cone.csv")

    # --- grafici ---
    setup_style()
    for m in metrics:
        wide = per_subj.pivot(index="subject", columns="repetition", values=m)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        n = draw_curve(ax, wide, m)
        set_title(ax, f"{LEG_METRICS[m][0].split(' [')[0]} per repetition, home → cone "
                      f"(mean ± SEM, N = {n})")
        save_figure(fig, f"training_curve_{m}", args.out)
    gm = [m for m in GRID_METRICS if m in metrics]
    ncol = 3; nrow = int(np.ceil(len(gm) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow), squeeze=False)
    for ax, m in zip(axes.flat, gm):
        n = draw_curve(ax, per_subj.pivot(index="subject", columns="repetition", values=m), m)
        set_title(ax, LEG_METRICS[m][0], width=36, fontsize=10); ax.set_ylabel("")
    for ax in list(axes.flat)[len(gm):]:
        ax.axis("off")
    fig.suptitle(f"Navigation training – reaching per repetition (mean ± SEM, N = {n}, grey = subjects)",
                 fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "training_curves_grid", args.out)

    # tutte le traiettorie sovrapposte
    fig, ax = plt.subplots(figsize=(7, 6))
    for f in files:
        _, odom = load_session(f)
        if odom is None or len(odom) < 5:
            continue
        ax.plot(odom["x"] - odom["x"].iloc[0], odom["y"] - odom["y"].iloc[0], color=BLUE, linewidth=0.7, alpha=0.5)
    for _, tg in targets.iterrows():
        col = INK if tg["name"] == "home" else ORANGE
        ax.add_patch(plt.Circle((tg["x"], tg["y"]), tg["radius"], fill=False, color=col, linewidth=1.5))
        ax.text(tg["x"], tg["y"] + tg["radius"] + 0.1, tg["name"], ha="center", color=col, fontsize=9)
    ax.set_aspect("equal"); ax.grid(True, axis="both"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    set_title(ax, "Training trajectories, all subjects")
    save_figure(fig, "training_trajectories_all", args.out)

    cols = [c for c in ["metric", "first_mean", "last_mean", "delta_percent", "wilcoxon_first_last_p", "friedman_p"]
            if c in tests.columns]
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print("\n" + tests[cols].to_string(index=False))
    print(f"\nOutput in {args.out}")


if __name__ == "__main__":
    main()
