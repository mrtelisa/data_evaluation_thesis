"""Training a 9 regioni (reaching_regions.py), fatto due volte: pre e post
(con il secondo blind test in mezzo).

Input: i file di results_regions/ del repo reachy2_teleop_grasping_and_simulation,
incollati ovunque dentro questa cartella (es. reaching_regions/results/):
    <soggetto>_regions_<pre|post>[_k]_trials.csv       una riga per goal (target o ritorno al centro)
    <soggetto>_regions_<pre|post>[_k]_trajectory.csv   ogni campione del cursore (per i grafici)
    <soggetto>_regions_<pre|post>[_k]_summary.json     durata della sessione (time_total)
Se una sessione e' stata ripetuta (_1, _2, ...) si usa l'ultima ripetizione.
Si analizzano i goal verso i target (centro -> regione); i ritorni al centro
sono riportati a parte, solo come descrittive (il primo centro, che avvia la
sessione, non e' un ritorno ed e' escluso).

Analisi (vedi README.md):
1. blocchi di 8 target (ogni regione una volta): 12 per sessione, 24 in tutto.
   Per ogni soggetto e blocco un valore per metrica (mediana, o percentuale
   per le metriche si'/no) -> curva di apprendimento;
2. 3 confronti pianificati (Wilcoxon, Holm sui 3, rank-biserial r):
       pre: primo blocco vs ultimo blocco    quanto insegna il training 1
       ultimo blocco pre vs primo blocco post  si mantiene dopo il blind test?
       ultimo blocco pre vs ultimo blocco post cosa aggiunge il training 2
   piu' il confronto esplorativo fra le due sessioni intere (pre vs post);
3. fit esponenziale y = a + b * exp(-k * (blocco - 1)) della curva di gruppo
   (e di ogni soggetto): a = plateau, k = velocita' di apprendimento, e il
   blocco in cui si arriva al 95% del miglioramento (<= 12: gia' nel training 1);
4. per regione (heatmap 3 x 3): quali direzioni sono piu' difficili.

Output in reaching_regions/statistics/:
    report.txt                    il riassunto da leggere (italiano)
    training_subjects.csv         soggetto x sessione: un valore per metrica (+ durata)
    training_blocks.csv           soggetto x blocco (1-24): un valore per metrica
    training_statistics.csv       metrica x sessione: descrittive tra soggetti
    training_block_statistics.csv metrica x blocco: descrittive tra soggetti
    training_tests.csv            i 3 confronti + pre vs post per metrica
    training_fits.csv             fit esponenziale (gruppo e soggetti)
    training_per_region.csv       metrica x regione x sessione: mediana tra soggetti
    training_homes.csv            ritorni al centro: descrittive per sessione
    plots/training_learning_curves.png, plots/metrics/<metrica>.png,
    plots/training_sessions.png, plots/training_regions.png,
    plots/trajectories/<soggetto>.png

Uso:
    python3 training_analysis.py [--results DIR] [--out DIR]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import aggregate, find_sessions, to_bool   # noqa: E402
from common.plots import (BLUE, CATEGORICAL, INK2, MUTED, grid_axes, heatmap3x3,   # noqa: E402
                          save_figure, setup_style, spaghetti, SUBJECT_LINE)
from common.report import Report   # noqa: E402
from common.stats import describe, fmt_num, holm, wilcoxon_paired   # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
PHASES = ("pre", "post")
BLOCK_SIZE = 8
PATTERN = r"(?P<subject>.+)_regions_(?P<phase>pre|post)(?:_(?P<rep>\d+))?_trials\.csv"
CANVAS_W, CANVAS_H, TARGET_RADIUS = 1200, 650, 70

# metrica -> (etichetta, unita', meglio se, aggregazione, ruolo)
# ruolo: "main" = le 4 principali, "extra" = richieste in piu', "" = esplorative
METRICS = {
    "reach_time":               ("Reach time", "s", "down", "median", "main"),
    "normalized_path_length":   ("Normalized path length", "", "down", "median", "main"),
    "initial_direction_error":  ("Initial direction error", "deg", "down", "median", "main"),
    "region_at_reveal_correct": ("Correct region at reveal", "%", "up", "rate", "main"),
    "n_speed_peaks":            ("Speed peaks", "", "down", "median", "extra"),
    "movement_time":            ("Movement time", "s", "down", "median", "extra"),
    "reached_hidden":           ("Reached while hidden", "%", "up", "rate", "extra"),
    "reaction_time":            ("Reaction time", "s", "down", "median", ""),
    "error_at_reveal":          ("Error at reveal", "px", "down", "median", ""),
    "first_region_correct":     ("First region correct", "%", "up", "rate", ""),
    "n_wrong_regions":          ("Wrong regions", "", "down", "mean", ""),
    "n_entries":                ("Entries before the dwell", "", "down", "mean", ""),
    "dwell_time":               ("Dwell time", "s", "down", "median", ""),
    "max_deviation":            ("Max deviation", "px", "down", "median", ""),
    "log_dimensionless_jerk":   ("Smoothness (-ln DJ)", "", "up", "median", ""),
    "initial_direction_error_peak": ("Initial direction error (peak)", "deg", "down", "median", ""),
    "path_length":              ("Path length", "px", "down", "median", ""),
    "mean_speed":               ("Mean speed", "px/s", None, "median", ""),
    "peak_speed":               ("Peak speed", "px/s", None, "median", ""),
}
FOCUS = [k for k, v in METRICS.items() if v[4] in ("main", "extra")]
HOME_METRICS = ["reach_time", "movement_time", "normalized_path_length", "n_speed_peaks", "n_entries", "dwell_time"]


def load(root: Path) -> tuple:
    chosen, dropped = find_sessions(root, PATTERN, ("phase",))
    frames, durations = [], {}
    for (subject, phase), path in sorted(chosen.items()):
        df = pd.read_csv(path)
        df = df[to_bool(df["success"])] if "success" in df.columns else df
        df.insert(0, "phase", phase)
        df.insert(0, "subject", subject)
        df["_trajectory"] = str(path).replace("_trials.csv", "_trajectory.csv")
        frames.append(df)
        summary = Path(str(path).replace("_trials.csv", "_summary.json"))
        if summary.exists():
            durations[(subject, phase)] = json.loads(summary.read_text()).get("time_total")
    goals = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not goals.empty:
        goals["block"] = (goals["target_number"] - 1) // BLOCK_SIZE + 1
    return goals, durations, dropped


def per_unit(df: pd.DataFrame, by: list, metrics=METRICS) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row["n_goals"] = len(g)
        for m in metrics:
            if m in g.columns:
                row[m] = aggregate(g[m], METRICS[m][3] if m in METRICS else "median")
        rows.append(row)
    return pd.DataFrame(rows)


def expo(x, a, b, k):
    return a + b * np.exp(-k * (x - 1))


def fit_curve(x, y) -> dict:
    """Fit esponenziale; block95 = blocco in cui si e' fatto il 95% del miglioramento."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = ~np.isnan(y)
    x, y = x[keep], y[keep]
    out = {"a": np.nan, "b": np.nan, "k": np.nan, "r2": np.nan, "block95": np.nan, "n_points": int(len(x))}
    if len(x) < 5 or np.allclose(y, y[0]):
        return out
    try:
        p, _ = curve_fit(expo, x, y, p0=(y[-3:].mean(), y[:3].mean() - y[-3:].mean(), 0.2),
                         bounds=([-np.inf, -np.inf, 1e-4], [np.inf, np.inf, 5.0]), maxfev=20000)
    except (RuntimeError, ValueError):
        return out
    res = y - expo(x, *p)
    ss_tot = np.sum((y - y.mean()) ** 2)
    out.update(a=float(p[0]), b=float(p[1]), k=float(p[2]),
               r2=float(1 - np.sum(res ** 2) / ss_tot) if ss_tot > 0 else np.nan,
               block95=float(1 + math.log(20) / p[2]))
    return out


def analyse(sessions, blocks, n_blocks, report):
    stat_rows, test_rows, block_rows, fit_rows = [], [], [], []
    last = n_blocks
    contrasts = (("pre", 1, "pre", last, "training 1: first vs last block"),
                 ("pre", last, "post", 1, "retention: last pre vs first post block"),
                 ("pre", last, "post", last, "training 2: last pre vs last post block"))
    for m, (label, unit, better, _, role) in METRICS.items():
        if m not in sessions.columns:
            continue
        ws = sessions.pivot(index="subject", columns="phase", values=m).reindex(columns=list(PHASES))
        d_sess = {ph: describe(ws[ph]) for ph in PHASES}
        for ph in PHASES:
            stat_rows.append({"metric": m, "label": label, "unit": unit, "role": role, "phase": ph, **d_sess[ph]})
        wb = blocks.pivot(index="subject", columns="gblock", values=m)
        for gb in wb.columns:
            block_rows.append({"metric": m, "block": gb, **describe(wb[gb])})

        def col(ph, b):
            gb = b + (n_blocks if ph == "post" else 0)
            return wb[gb] if gb in wb.columns else pd.Series(np.nan, index=wb.index)

        tests = [wilcoxon_paired(col(a, ba), col(b, bb)) for a, ba, b, bb, _ in contrasts]
        p_holm = holm([t["p"] for t in tests])
        descr = []
        for (a, ba, b, bb, what), t, ph in zip(contrasts, tests, p_holm):
            da, db = describe(col(a, ba)), describe(col(b, bb))
            descr.append((da, db))
            test_rows.append({"metric": m, "label": label, "role": role, "comparison": f"{a} block {ba} vs {b} block {bb}",
                              "meaning": what, "planned": True, "n": t["n"], "statistic": t["statistic"], "p": t["p"],
                              "p_holm": ph, "effect": t["effect"], "effect_name": "rank-biserial r",
                              "median_first": da.get("median"), "median_second": db.get("median"), "better": better})
        t_sess = wilcoxon_paired(ws["pre"], ws["post"])
        test_rows.append({"metric": m, "label": label, "role": role, "comparison": "session pre vs post",
                          "meaning": "whole sessions (exploratory)", "planned": False, "n": t_sess["n"],
                          "statistic": t_sess["statistic"], "p": t_sess["p"], "p_holm": np.nan,
                          "effect": t_sess["effect"], "effect_name": "rank-biserial r",
                          "median_first": d_sess["pre"].get("median"), "median_second": d_sess["post"].get("median"),
                          "better": better})
        # fit: curva di gruppo (mediana per blocco) e di ogni soggetto
        med = wb.median()
        fg = fit_curve(med.index.to_numpy(), med.to_numpy())
        fit_rows.append({"metric": m, "subject": "GROUP (median curve)", **fg})
        for subj, row in wb.iterrows():
            fit_rows.append({"metric": m, "subject": subj, **fit_curve(row.index.to_numpy(), row.to_numpy())})

        if role in ("main", "extra"):
            report.section(f"{label}{f' [{unit}]' if unit else ''}" + ("  (metrica principale)" if role == "main" else ""))
            report.text(f"  sessione intera: pre {fmt_num(d_sess['pre'].get('median'))} "
                        f"[{fmt_num(d_sess['pre'].get('q1'))}-{fmt_num(d_sess['pre'].get('q3'))}] -> post "
                        f"{fmt_num(d_sess['post'].get('median'))} [{fmt_num(d_sess['post'].get('q1'))}-"
                        f"{fmt_num(d_sess['post'].get('q3'))}] (esplorativo: p={fmt_num(t_sess['p'], 3)}, "
                        f"r={fmt_num(t_sess['effect'])})")
            for (a, ba, b, bb, what), t, ph, (da, db) in zip(contrasts, tests, p_holm, descr):
                report.comparison(what, "", better, da, db, t, ph, f"{a} b{ba}", f"{b} b{bb}")
            if not np.isnan(fg["k"]):
                where = "gia' nel training 1" if fg["block95"] <= n_blocks else \
                    ("nel training 2" if fg["block95"] <= 2 * n_blocks else "non ancora raggiunto a fine training 2")
                report.text(f"  curva di gruppo: plateau stimato {fg['a']:.2f}, 95% del miglioramento al blocco "
                            f"{fg['block95']:.1f} ({where}); R2 del fit {fg['r2']:.2f}")
            else:
                report.text("  curva di gruppo: fit esponenziale non riuscito (curva piatta o troppo rumorosa)")
    return (pd.DataFrame(stat_rows), pd.DataFrame(test_rows), pd.DataFrame(block_rows), pd.DataFrame(fit_rows))


def plot_curve(ax, blocks, fits, m, n_blocks):
    label, unit, better, _, _ = METRICS[m]
    wb = blocks.pivot(index="subject", columns="gblock", values=m)
    x = wb.columns.to_numpy(dtype=float)
    for _, row in wb.iterrows():
        ax.plot(x, row.to_numpy(dtype=float), color=SUBJECT_LINE, lw=0.8, zorder=2)
    med, q1, q3 = wb.median(), wb.quantile(0.25), wb.quantile(0.75)
    ax.fill_between(x, q1, q3, color=BLUE, alpha=0.15, lw=0, zorder=1)
    ax.plot(x, med, color=BLUE, lw=2.2, marker="o", ms=3, zorder=3, label="median [IQR]")
    f = fits[(fits.metric == m) & (fits.subject == "GROUP (median curve)")]
    if not f.empty and not np.isnan(f["k"].iloc[0]):
        xx = np.linspace(x.min(), x.max(), 200)
        ax.plot(xx, expo(xx, f["a"].iloc[0], f["b"].iloc[0], f["k"].iloc[0]), color=INK2, ls="--", lw=1.2,
                label="exponential fit", zorder=4)
    ax.axvline(n_blocks + 0.5, color=MUTED, ls=":", lw=1)
    ylim = ax.get_ylim()
    ax.text(n_blocks / 2 + 0.5, ylim[1], "training 1 (pre)", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.text(1.5 * n_blocks + 0.5, ylim[1], "training 2 (post)", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xlabel("block of 8 targets")
    ax.set_ylabel(f"{label}{f' [{unit}]' if unit else ''}")
    ax.set_title(label + ("  (higher = better)" if better == "up" else "  (lower = better)" if better == "down" else ""),
                 loc="left", pad=14)


def plots(goals, sessions, blocks, fits, per_region, n_blocks, out):
    setup_style()
    fig, axes = grid_axes(len(FOCUS), ncols=2, width=6.2, height=3.6)
    for ax, m in zip(axes, FOCUS):
        plot_curve(ax, blocks, fits, m, n_blocks)
    axes[0].legend(fontsize=7.5, loc="best")
    fig.suptitle("Reaching training – learning curves (one line per subject)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out, "training_learning_curves")
    for m in METRICS:
        if m in blocks.columns and blocks[m].notna().any():
            fig, ax = plt.subplots(figsize=(7, 3.8))
            plot_curve(ax, blocks, fits, m, n_blocks)
            save_figure(fig, out, f"metrics/{m}")

    fig, axes = grid_axes(len(FOCUS), ncols=4, width=3.2, height=3.2)
    for ax, m in zip(axes, FOCUS):
        label, unit = METRICS[m][0], METRICS[m][1]
        ws = sessions.pivot(index="subject", columns="phase", values=m).reindex(columns=list(PHASES))
        spaghetti(ax, ws, ["pre", "post"], ylabel=f"{label}{f' [{unit}]' if unit else ''}", title=label)
    fig.suptitle("Reaching training – whole session, pre vs post", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out, "training_sessions")

    main = [m for m in FOCUS if METRICS[m][4] == "main"]
    fig, axes = plt.subplots(len(main), 2, figsize=(5.4, 2.6 * len(main)), squeeze=False)
    for i, m in enumerate(main):
        sub = per_region[per_region.metric == m]
        vmin, vmax = sub["median"].min(), sub["median"].max()
        for j, ph in enumerate(PHASES):
            vals = sub[sub.phase == ph].set_index("region")["median"].to_dict()
            unit = METRICS[m][1]
            heatmap3x3(axes[i, j], vals, title=f"{METRICS[m][0]} – {ph}" + (f" [{unit}]" if unit else ""),
                       vmin=vmin, vmax=vmax, fmt="{:.2f}" if abs(vmax) < 10 else "{:.0f}")
    fig.suptitle("Per region (median across subjects); centre = home", x=0.01, ha="left", fontweight="bold",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out, "training_regions")

    targets = goals[goals.kind == "region"]
    for subject, g in targets.groupby("subject"):
        fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), squeeze=False)
        for ax, ph in zip(axes[0], PHASES):
            ax.set_xlim(0, CANVAS_W)
            ax.set_ylim(CANVAS_H, 0)
            ax.set_aspect("equal")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for xx in (400, 800):
                ax.axvline(xx, color="#dddddd", lw=0.8)
            for yy in (CANVAS_H / 3, 2 * CANVAS_H / 3):
                ax.axhline(yy, color="#dddddd", lw=0.8)
            ax.set_title(f"training {ph}", loc="left", fontsize=9)
            gp = g[g.phase == ph]
            if gp.empty:
                continue
            tr_path = Path(gp["_trajectory"].iloc[0])
            if not tr_path.exists():
                continue
            tr = pd.read_csv(tr_path)
            tr = tr[tr["kind"] == "region"]
            used = set(gp["trial"])
            for trial, seg in tr.groupby("trial"):
                if trial in used:
                    c = CATEGORICAL[(int(seg["region"].iloc[0]) - 1) % len(CATEGORICAL)]
                    ax.plot(seg["x"], seg["y"], color=c, lw=0.5, alpha=0.5)
        fig.suptitle(f"{subject} – cursor paths to the targets (colour = region)", x=0.01, ha="left",
                     fontweight="bold", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        save_figure(fig, out, f"trajectories/{subject}")


def main():
    p = argparse.ArgumentParser(description="Analisi dei due training a 9 regioni")
    p.add_argument("--results", type=Path, default=BASE_DIR, help="cartella con i file di results_regions/")
    p.add_argument("--out", type=Path, default=BASE_DIR / "statistics")
    args = p.parse_args()

    goals, durations, dropped = load(args.results)
    if goals.empty:
        print(f"Nessun file *_regions_<pre|post>_trials.csv in {args.results}")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    targets = goals[goals["kind"] == "region"].copy()
    n_blocks = int(targets["block"].max())
    targets["gblock"] = targets["block"] + np.where(targets["phase"] == "post", n_blocks, 0)

    report = Report("TRAINING A 9 REGIONI – pre e post")
    ids = sorted(goals["subject"].unique())
    report.text(f"Soggetti: {len(ids)} ({', '.join(ids)})")
    for ph in PHASES:
        who = sorted(goals.loc[goals.phase == ph, "subject"].unique())
        missing = sorted(set(ids) - set(who))
        report.text(f"  training {ph}: {len(who)} soggetti" + (f" (mancano: {', '.join(missing)})" if missing else ""))
    if dropped:
        report.text("Ripetizioni scartate (si usa l'ultima): " + ", ".join(p.name for p in dropped))
    report.text(f"Blocchi di {BLOCK_SIZE} target: {n_blocks} per sessione, {2 * n_blocks} in tutto.")
    report.text("Confronti pianificati: 3 Wilcoxon fra blocchi, Holm sui 3, per ogni metrica.")

    sessions = per_unit(targets, ["subject", "phase"])
    sessions["session_time"] = [durations.get((s, ph)) for s, ph in zip(sessions.subject, sessions.phase)]
    blocks = per_unit(targets, ["subject", "gblock"])
    stats_df, tests_df, block_stats, fits = analyse(sessions, blocks, n_blocks, report)

    per_region = []
    reg_units = per_unit(targets, ["subject", "phase", "region"])
    for m in METRICS:
        if m in reg_units.columns:
            for (ph, reg), g in reg_units.groupby(["phase", "region"]):
                per_region.append({"metric": m, "phase": ph, "region": reg, **describe(g[m])})
    per_region = pd.DataFrame(per_region)
    # the first home is the start of the session (the timer starts there), not a return: excluded
    first_home = goals.groupby(["subject", "phase"])["trial"].transform("min") == goals["trial"]
    homes = per_unit(goals[(goals["kind"] == "home") & ~first_home], ["subject", "phase"], HOME_METRICS)
    home_rows = [{"metric": m, "phase": ph, **describe(g[m])} for m in HOME_METRICS if m in homes.columns
                 for ph, g in homes.groupby("phase")]

    report.section("Durata della sessione [s] (timer a schermo)")
    ws = sessions.pivot(index="subject", columns="phase", values="session_time").reindex(columns=list(PHASES))
    report.comparison("durata del training", "s", "down", describe(ws["pre"]), describe(ws["post"]),
                      wilcoxon_paired(ws["pre"], ws["post"]), None, "pre", "post", digits=0)
    report.section("Metriche esplorative")
    report.text("  Tutte le altre metriche sono in training_tests.csv e plots/metrics/, calcolate allo stesso modo.")
    report.text("  I ritorni al centro sono in training_homes.csv.")

    sessions.to_csv(args.out / "training_subjects.csv", index=False)
    blocks.to_csv(args.out / "training_blocks.csv", index=False)
    stats_df.to_csv(args.out / "training_statistics.csv", index=False)
    block_stats.to_csv(args.out / "training_block_statistics.csv", index=False)
    tests_df.to_csv(args.out / "training_tests.csv", index=False)
    fits.to_csv(args.out / "training_fits.csv", index=False)
    per_region.to_csv(args.out / "training_per_region.csv", index=False)
    pd.DataFrame(home_rows).to_csv(args.out / "training_homes.csv", index=False)
    plots(goals, sessions, blocks, fits, per_region, n_blocks, args.out)
    report.write(args.out)


if __name__ == "__main__":
    main()
