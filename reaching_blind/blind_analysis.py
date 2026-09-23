"""Test di reaching "blind" a 3 target (reaching_blind.py): pre vs post training.

File attesi (come li salva reaching_blind.py), in results/pre e results/post
oppure tutti insieme in results/ (la fase si legge dal nome del file):
    <soggetto>_blind_<fase>_trials.csv        una riga per trial
    <soggetto>_blind_<fase>_trajectory.csv    ogni campione del cursore
    <soggetto>_blind_<fase>_summary.json      medie e config (posizioni target)
Se un soggetto ha piu' sessioni della stessa fase (_1, _2, ...) si tiene
l'ultima. I trial non completati (success = False, solo su abort) sono esclusi.

Ogni soggetto pesa 1: prima la media dei suoi trial, poi la statistica tra
soggetti (media, SD, varianza, SEM, mediana). Il confronto pre/post e'
appaiato: Shapiro-Wilk sulle differenze -> t-test appaiato o Wilcoxon, con
d di Cohen e r di Wilcoxon (come nel protocollo).

Oltre alle metriche gia' nel trials.csv vengono calcolate dalla traiettoria:
    reaction_time_blind   tempo dal cambio di target al primo movimento VERSO
                          il nuovo target (velocita' radiale > soglia per
                          almeno REACTION_MIN_S): il reaction_time del task e'
                          ~0 perche' il cursore e' gia' in movimento
    time_to_min_error     tempo in cui il cursore arriva alla distanza minima
    border_time           % di tempo con il cursore fermo sul bordo del canvas
    settled_error         distanza media nell'ultimo secondo (= end_error) e
                          deviazione standard della posizione nello stesso
                          tratto (quanto e' stabile l'arrivo)

Output (statistics/), ogni cartella con la sua plots/:
- pre/ e post/    blind_<fase>_subjects.csv, blind_<fase>_trials.csv,
                  blind_<fase>_statistics.csv (tra soggetti), blind_<fase>_per_target.csv,
                  blind_<fase>_blocks.csv, plots/<fase>_<metrica>.png,
                  <fase>_grid.png, <fase>_targets.png,
                  plots/trajectories/<soggetto>.png
- pre-vs-post/    blind_prepost_statistics.csv (una riga per metrica: pre, post,
                  delta, test, p, effect size), blind_prepost_subjects.csv,
                  blind_prepost_blocks.csv, plots/prepost_<metrica>.png,
                  prepost_grid.png, prepost_main.png, blocks_curve.png,
                  prepost_targets.png, plots/trajectories/<soggetto>.png
  e la tabella riassuntiva delle MAIN_METRICS stampata a terminale.

Uso:
    python3 blind_analysis.py [--results DIR] [--out DIR]
"""

import argparse
import glob
import json
import math
import os
import re
import textwrap
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", message="Sample size too small")
warnings.filterwarnings("ignore", message="Exact p-value calculation")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = BASE_DIR / "results"
DEFAULT_OUT = BASE_DIR / "statistics"

CANVAS_W, CANVAS_H = 1200, 650
BORDER_TOL = 1.0          # px: piu' vicino di cosi' al bordo = cursore saturato
REACTION_SPEED = 40.0     # px/s di velocita' radiale verso il target
REACTION_MIN_S = 0.15     # ...mantenuta almeno per questo tempo
SMOOTH_S = 0.1            # finestra di smoothing della velocita'

# metrica -> (etichetta, meglio se, e' una percentuale/conteggio da istogramma)
METRICS = {
    # precisione
    "hit":                    ("Trials with a hit [%]", "up", False),
    "chosen_correct":         ("Closest target is the correct one [%]", "up", False),
    "on_target_at_end":       ("On target when it changes [%]", "up", False),
    "final_error":            ("Final error [px]", "down", False),
    "end_error":              ("End error, last 1 s [px]", "down", False),
    "min_error":              ("Closest approach [px]", "down", False),
    "initial_error":          ("Initial error [px]", "down", False),
    "relative_final_error":   ("Relative final error (0 = perfect, 1 = no gain)", "down", False),
    "settled_sd":             ("Endpoint scatter, last 1 s [px]", "down", False),
    "time_in_target":         ("Time inside the target [%]", "up", False),
    # tempi
    "reaction_time_blind":    ("Reaction time toward the target [s]", "down", False),
    "reaction_time":          ("Reaction time, task (movement onset) [s]", "down", False),
    "time_to_min_error":      ("Time of closest approach [s]", "down", False),
    "reach_time":             ("Reach time (hits only) [s]", "down", False),
    "movement_time":          ("Movement time [s]", "down", False),
    # cinematica / controllo
    "normalized_path_length": ("Normalized path length (1 = straight)", "down", False),
    "path_length":            ("Path length [px]", "down", False),
    "max_deviation":          ("Max deviation from ideal line [px]", "down", False),
    "log_dimensionless_jerk": ("Log dimensionless jerk (higher = smoother)", "up", False),
    "n_speed_peaks":          ("Speed peaks per trial", "down", True),
    "mean_speed":             ("Mean speed [px/s]", "up", False),
    "peak_speed":             ("Peak speed [px/s]", "up", False),
    "border_time":            ("Time with the cursor on the canvas border [%]", "down", False),
    "hand_lost":              ("Time without hand tracking [%]", "down", False),
}
# tabella riassuntiva stampata a terminale (come quella del task)
MAIN_METRICS = ["hit", "chosen_correct", "final_error", "end_error", "relative_final_error",
                "time_in_target", "reach_time", "normalized_path_length"]
PRECISION_METRICS = ["hit", "chosen_correct", "on_target_at_end", "final_error", "end_error",
                     "min_error", "relative_final_error", "settled_sd", "time_in_target"]
TIME_METRICS = ["reaction_time_blind", "time_to_min_error", "reach_time", "movement_time"]
GRID_METRICS = ["hit", "chosen_correct", "final_error", "end_error", "relative_final_error",
                "time_in_target", "reaction_time_blind", "time_to_min_error", "normalized_path_length"]

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
TARGET_COLORS = [BLUE, ORANGE, GREEN, "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SUBJECT_LINE = "#c3c2b7"
PHASES = ("pre", "post")
# confronto prima/dopo: arancione carota vs blu scuro (tinte opposte e molto
# diverse in luminosita': leggibili anche in bianco/nero e con daltonismo)
PRE_COLOR, POST_COLOR = "#ed7014", "#104281"
PHASE_COLOR = {"pre": PRE_COLOR, "post": POST_COLOR}
PHASE_LABEL = {"pre": "Pre-training", "post": "Post-training"}


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
        "font.size": 10, "legend.frameon": False,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def set_title(ax, text, width=55, **kw):
    ax.set_title("\n".join(textwrap.wrap(text, width)), loc="center", **kw)


def save_figure(fig, name, out_dir, sub=None):
    d = Path(out_dir) / "plots" / (sub or "")
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png")
    plt.close(fig)


def describe(values):
    v = pd.Series(values).dropna().astype(float)
    n = len(v)
    row = {"n": n}
    if n:
        row.update(mean=v.mean(), median=v.median(), min=v.min(), max=v.max())
    if n > 1:
        row["sd"] = v.std(ddof=1); row["var"] = v.var(ddof=1); row["sem"] = row["sd"] / math.sqrt(n)
    return row


# ---------------------------------------------------------------------------
# Metriche dalla traiettoria
# ---------------------------------------------------------------------------
def trajectory_metrics(g, goal):
    """g: campioni di un trial (t, x, y); goal: (x, y) del target."""
    if len(g) < 3:
        return {}
    t = g["t"].values.astype(float)
    x, y = g["x"].values.astype(float), g["y"].values.astype(float)
    d = np.hypot(x - goal[0], y - goal[1])
    out = {}
    # tempo in cui arriva piu' vicino
    out["time_to_min_error"] = float(t[int(np.argmin(d))] - t[0])
    # dispersione dell'arrivo nell'ultimo secondo
    last = t >= t[-1] - 1.0
    if last.sum() > 1:
        out["settled_sd"] = float(np.hypot(x[last].std(ddof=1), y[last].std(ddof=1)))
    # tempo sul bordo del canvas (cursore saturato dal clipping)
    on_border = (x <= BORDER_TOL) | (x >= CANVAS_W - BORDER_TOL) | \
                (y <= BORDER_TOL) | (y >= CANVAS_H - BORDER_TOL)
    dt = np.diff(t, append=t[-1])
    total = float(dt.sum())
    if total > 0:
        out["border_time"] = 100.0 * float(dt[on_border].sum()) / total
    # reaction time "blind": primo movimento verso il target, sostenuto
    dts = np.diff(t)
    if np.all(dts > 0) and len(t) > 5:
        radial = -np.gradient(d, t)          # > 0 = si avvicina
        k = max(1, int(round(SMOOTH_S / np.median(dts))))
        radial = np.convolve(radial, np.ones(k) / k, mode="same")
        toward = radial > REACTION_SPEED
        i = 0
        while i < len(toward):
            if toward[i]:
                j = i
                while j + 1 < len(toward) and toward[j + 1]:
                    j += 1
                if t[j] - t[i] >= REACTION_MIN_S:
                    out["reaction_time_blind"] = float(t[i] - t[0])
                    break
                i = j + 1
            else:
                i += 1
    return out


# ---------------------------------------------------------------------------
# Caricamento
# ---------------------------------------------------------------------------
def find_sessions(results_dir):
    """{(soggetto, fase): path del trials.csv} tenendo l'ultima sessione."""
    files = sorted(glob.glob(str(Path(results_dir) / "**" / "*_blind_*_trials.csv"), recursive=True))
    sessions = {}
    for f in files:
        m = re.match(r"^(.*)_blind_(pre|post)(?:_(\d+))?_trials\.csv$", os.path.basename(f))
        if not m:
            continue
        subject, phase, n = m.group(1), m.group(2), int(m.group(3) or 0)
        key = (subject, phase)
        if key in sessions and n <= sessions[key][0]:
            continue
        if key in sessions:
            print(f"[WARN] {subject} {phase}: piu' sessioni, tengo la piu' recente")
        sessions[key] = (n, f)
    if not sessions:
        raise FileNotFoundError(f"Nessun *_blind_<fase>_trials.csv in {results_dir}")
    return {k: v[1] for k, v in sorted(sessions.items())}


def load_sessions(results_dir):
    """(trials di tutti, traiettorie {(soggetto,fase): df}, posizioni dei target)."""
    frames, trajs, targets = [], {}, None
    for (subject, phase), f in find_sessions(results_dir).items():
        df = pd.read_csv(f)
        df.insert(0, "phase", phase)
        df.insert(0, "subject", subject)
        df["success"] = df["success"].astype(str).str.lower().eq("true")
        spath = f.replace("_trials.csv", "_summary.json")
        if os.path.exists(spath):
            with open(spath) as fh:
                cfg = json.load(fh).get("config", {})
            if targets is None and cfg.get("targets"):
                targets = {int(k): tuple(v) for k, v in cfg["targets"].items()}
        tpath = f.replace("_trials.csv", "_trajectory.csv")
        if os.path.exists(tpath):
            traj = pd.read_csv(tpath)
            trajs[(subject, phase)] = traj
            extra = []
            for _, r in df.iterrows():
                g = traj[traj["trial"] == r["trial"]]
                extra.append(trajectory_metrics(g, (r["goal_x"], r["goal_y"])) if len(g) else {})
            df = pd.concat([df.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
        # blocco = gruppo di trial in cui ogni target compare una volta
        n_targets = df["target"].nunique() or 1
        df["block"] = (df["trial"] - 1) // n_targets + 1
        frames.append(df)
        print(f"[load] {subject} {phase:4s}: {len(df)} trial"
              + ("" if (subject, phase) in trajs else " (senza traiettoria)"))
    return pd.concat(frames, ignore_index=True), trajs, targets


# ---------------------------------------------------------------------------
# Statistica appaiata
# ---------------------------------------------------------------------------
def paired_stats(pre, post):
    d = pd.concat([pre, post], axis=1).dropna()
    n = len(d)
    row = {"n": n}
    if n == 0:
        return row
    a, b = d.iloc[:, 0].values.astype(float), d.iloc[:, 1].values.astype(float)
    diff = b - a
    row.update(pre_mean=a.mean(), pre_sd=a.std(ddof=1) if n > 1 else np.nan,
               pre_var=a.var(ddof=1) if n > 1 else np.nan, pre_median=np.median(a),
               post_mean=b.mean(), post_sd=b.std(ddof=1) if n > 1 else np.nan,
               post_var=b.var(ddof=1) if n > 1 else np.nan, post_median=np.median(b),
               delta=diff.mean(),
               delta_percent=100 * diff.mean() / a.mean() if a.mean() else np.nan,
               n_better=np.nan, n_worse=np.nan)
    if n < 3:
        return row
    sd_diff = diff.std(ddof=1)
    # d solo se le differenze hanno una dispersione reale: con differenze tutte
    # uguali (o quasi) il rapporto esplode e non significa nulla
    row["cohen_d"] = diff.mean() / sd_diff if sd_diff > 1e-9 * max(1.0, abs(diff.mean())) else np.nan
    if np.allclose(diff, 0):
        row["test"] = "none (no change)"
        return row
    row["shapiro_p"] = stats.shapiro(diff).pvalue
    if row["shapiro_p"] > 0.05:
        row["test"] = "paired t-test"; row["p"] = stats.ttest_rel(b, a).pvalue
    else:
        row["test"] = "Wilcoxon"
        try:
            row["p"] = stats.wilcoxon(b, a, zero_method="zsplit").pvalue
        except ValueError:
            row["p"] = np.nan
    nz = diff[diff != 0]; k = len(nz)
    if k >= 2:
        ranks = stats.rankdata(np.abs(nz)); w_plus = ranks[nz > 0].sum()
        mu, sigma = k * (k + 1) / 4, math.sqrt(k * (k + 1) * (2 * k + 1) / 24)
        row["wilcoxon_r"] = abs(w_plus - mu) / sigma / math.sqrt(k) if sigma > 0 else np.nan
    return row


# ---------------------------------------------------------------------------
# Grafici
# ---------------------------------------------------------------------------
def draw_canvas(ax, targets, radius=60):
    ax.set_xlim(0, CANVAS_W); ax.set_ylim(CANVAS_H, 0); ax.set_aspect("equal"); ax.grid(False)
    ax.add_patch(plt.Rectangle((0, 0), CANVAS_W, CANVAS_H, fill=False, edgecolor=GRID))
    for k, (x, y) in sorted((targets or {}).items()):
        c = TARGET_COLORS[(k - 1) % len(TARGET_COLORS)]
        ax.add_patch(plt.Circle((x, y), radius, fill=False, edgecolor=c, linewidth=1.5))
        ax.text(x, y - radius - 8, f"T{k}", ha="center", color=c, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def plot_trajectories(subject, trajs, trials, targets, out_dirs):
    """Traiettorie del soggetto: pre, post (nelle rispettive cartelle) e le due
    sovrapposte in pre-vs-post."""
    setup_style()
    for phase in PHASES:
        traj = trajs.get((subject, phase))
        if traj is None:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        draw_canvas(ax, targets)
        for trial, g in traj.groupby("trial"):
            tgt = int(g["target"].iloc[0])
            ax.plot(g["x"], g["y"], color=TARGET_COLORS[(tgt - 1) % len(TARGET_COLORS)],
                    linewidth=1, alpha=0.8)
            ax.plot(g["x"].iloc[-1], g["y"].iloc[-1], marker="o", markersize=4,
                    color=TARGET_COLORS[(tgt - 1) % len(TARGET_COLORS)])
        set_title(ax, f"{subject} – {PHASE_LABEL[phase]} (colour = target, dot = endpoint)")
        save_figure(fig, subject, out_dirs[phase], "trajectories")

    if all((subject, p) in trajs for p in PHASES):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
        for ax, phase in zip(axes, PHASES):
            draw_canvas(ax, targets)
            traj = trajs[(subject, phase)]
            for trial, g in traj.groupby("trial"):
                ax.plot(g["x"], g["y"], color=PHASE_COLOR[phase], linewidth=1, alpha=0.75)
                ax.plot(g["x"].iloc[-1], g["y"].iloc[-1], marker="o", markersize=4,
                        color=PHASE_COLOR[phase])
            t = trials[(trials["subject"] == subject) & (trials["phase"] == phase)]
            err = t["final_error"].mean(); hit = t["hit"].mean()
            ax.set_title(f"{PHASE_LABEL[phase]} – hit {hit:.0f} %, final error {err:.0f} px", fontsize=10)
        fig.suptitle(f"{subject} – cursor trajectories, pre (orange) vs post (blue)", fontweight="bold")
        fig.tight_layout()
        save_figure(fig, subject, out_dirs["prepost"], "trajectories")


def box_dots(ax, values, ylabel, color=BLUE):
    v = pd.Series(values).dropna().values
    if len(v) == 0:
        return 0
    ax.boxplot([v], positions=[0], widths=0.45, patch_artist=True, showmeans=True, showfliers=False,
               boxprops={"facecolor": "#cde2fb", "edgecolor": color},
               medianprops={"color": color, "linewidth": 2},
               whiskerprops={"color": MUTED}, capprops={"color": MUTED},
               meanprops={"marker": "D", "markersize": 5, "markerfacecolor": INK, "markeredgecolor": "none"})
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(v))
    ax.scatter(jitter, v, color=INK2, s=16, zorder=3, alpha=0.85)
    ax.set_xticks([0]); ax.set_xticklabels([f"N = {len(v)}"]); ax.set_xlim(-0.6, 0.6)
    ax.set_ylabel(ylabel)
    return len(v)


def bar_band(ax, xs, means, errs, width, color=BLUE):
    """Variabilita' di una barra: la cima sfuma nella banda +- err (sopra il
    colore della barra in trasparenza, sotto una schiaritura della barra)."""
    for xi, m, e in zip(np.atleast_1d(xs), np.atleast_1d(means), np.atleast_1d(errs)):
        if not np.isfinite(m) or not np.isfinite(e) or e == 0:
            continue
        xs_ = [xi - width / 2, xi + width / 2]
        ax.fill_between(xs_, [m, m], [m + e, m + e], color=color, alpha=0.3, lw=0, zorder=3)
        ax.fill_between(xs_, [m - e, m - e], [m, m], color="white", alpha=0.45, lw=0, zorder=3)


def empty_panel(ax, ylabel):
    ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color=MUTED)
    ax.set_ylabel(ylabel); ax.set_xticks([])


def paired_plot(ax, pre, post, ylabel, p=None):
    d = pd.concat([pre, post], axis=1).dropna()
    a, b = d.iloc[:, 0].values, d.iloc[:, 1].values
    for ai, bi in zip(a, b):
        ax.plot([0, 1], [ai, bi], color=SUBJECT_LINE, linewidth=1, marker="o", markersize=3.5, zorder=1)
    n = len(d)
    if n:
        means = [a.mean(), b.mean()]
        sems = [a.std(ddof=1) / math.sqrt(n) if n > 1 else 0, b.std(ddof=1) / math.sqrt(n) if n > 1 else 0]
        lo = [m - s for m, s in zip(means, sems)]
        hi = [m + s for m, s in zip(means, sems)]
        ax.fill_between([0, 1], lo, hi, color=POST_COLOR, alpha=0.22, lw=0, zorder=2)
        ax.plot([0, 1], means, color=POST_COLOR, linewidth=2.2, marker="o", markersize=7, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pre", "Post"]); ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel(ylabel)
    if n == 0:   # es. reach_time quando nessuno ha centrato un target
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color=MUTED)
        return 0
    if p is not None and not np.isnan(p):
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(0.5, 0.97, f"p = {p:.3f} {stars}", transform=ax.transAxes, ha="center",
                va="top", color=INK2, fontsize=9)
    return n


def phase_out(out, phase):
    return out / phase


# ---------------------------------------------------------------------------
def analyse_phase(trials, subj, phase, metrics, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    tp = trials[trials["phase"] == phase]
    sp = subj[subj["phase"] == phase]
    tp.to_csv(out_dir / f"blind_{phase}_trials.csv", index=False)
    sp.to_csv(out_dir / f"blind_{phase}_subjects.csv", index=False)
    pd.DataFrame([{"metric": m, "label": METRICS[m][0], **describe(sp[m])} for m in metrics]).to_csv(
        out_dir / f"blind_{phase}_statistics.csv", index=False)
    # per target e per blocco: prima media per soggetto, poi tra soggetti
    rows = []
    for key, col in (("target", "per_target"), ("block", "per_block")):
        per_subj = tp.groupby(["subject", key])[metrics].mean().reset_index()
        for m in metrics:
            for k, grp in per_subj.groupby(key):
                rows.append({"grouping": key, key: k, "metric": m, **describe(grp[m])})
    by = pd.DataFrame(rows)
    by[by["grouping"] == "target"].to_csv(out_dir / f"blind_{phase}_per_target.csv", index=False)
    by[by["grouping"] == "block"].to_csv(out_dir / f"blind_{phase}_blocks.csv", index=False)

    setup_style()
    n = len(sp)
    for m in metrics:
        if sp[m].notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(4, 4))
        box_dots(ax, sp[m], METRICS[m][0], PHASE_COLOR[phase])
        set_title(ax, f"{METRICS[m][0].split(' [')[0]} – {PHASE_LABEL[phase]} (N = {n})", width=32)
        save_figure(fig, f"{phase}_{m}", out_dir)
    gm = [m for m in GRID_METRICS if m in metrics and sp[m].notna().any()]
    ncol = 3; nrow = int(np.ceil(len(gm) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.6 * nrow), squeeze=False)
    for ax, m in zip(axes.flat, gm):
        if box_dots(ax, sp[m], "", PHASE_COLOR[phase]) == 0:
            empty_panel(ax, "")
        set_title(ax, METRICS[m][0], width=34, fontsize=9.5)
    for ax in list(axes.flat)[len(gm):]:
        ax.axis("off")
    fig.suptitle(f"Blind reaching – {PHASE_LABEL[phase]}, all subjects (N = {n}, dots = subjects)",
                 fontweight="bold")
    fig.tight_layout()
    save_figure(fig, f"{phase}_grid", out_dir)
    # per target
    tm = ["final_error", "hit", "chosen_correct", "reaction_time_blind"]
    tm = [m for m in tm if m in metrics]
    fig, axes = plt.subplots(1, len(tm), figsize=(3.6 * len(tm), 3.6), squeeze=False)
    per_subj = tp.groupby(["subject", "target"])[metrics].mean().reset_index()
    for ax, m in zip(axes.flat, tm):
        w = per_subj.pivot(index="subject", columns="target", values=m)
        ts = sorted(w.columns); x = np.arange(len(ts))
        mean = w[ts].mean(); sem = (w[ts].std(ddof=1) / np.sqrt(w[ts].count())).fillna(0)
        cols = [TARGET_COLORS[(t - 1) % len(TARGET_COLORS)] for t in ts]
        ax.bar(x, mean, width=0.6, color=cols)
        for xi, mi, ei, ci in zip(x, mean.values, sem.values, cols):
            bar_band(ax, [xi], [mi], [ei], 0.6, ci)
        ax.set_xticks(x); ax.set_xticklabels([f"T{t}" for t in ts]); ax.set_xlabel("Target")
        set_title(ax, METRICS[m][0], width=30, fontsize=9.5)
    fig.suptitle(f"{PHASE_LABEL[phase]} – per target (mean ± SEM, N = {n})", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, f"{phase}_targets", out_dir)


def analyse_prepost(trials, subj, metrics, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    subj.to_csv(out_dir / "blind_prepost_subjects.csv", index=False)
    wide = {m: subj.pivot(index="subject", columns="phase", values=m) for m in metrics}
    rows = []
    for m in metrics:
        w = wide[m]
        if not {"pre", "post"} <= set(w.columns):
            continue
        rows.append({"metric": m, "label": METRICS[m][0], "better_if": METRICS[m][1],
                     **paired_stats(w["pre"], w["post"])})
    tests = pd.DataFrame(rows)
    tests.to_csv(out_dir / "blind_prepost_statistics.csv", index=False)
    pvals = dict(zip(tests["metric"], tests["p"])) if "p" in tests.columns else {}

    # per blocco, pre e post
    per_subj_block = trials.groupby(["subject", "phase", "block"])[metrics].mean().reset_index()
    rows = []
    for m in metrics:
        for (phase, b), grp in per_subj_block.groupby(["phase", "block"]):
            rows.append({"phase": phase, "block": b, "metric": m, **describe(grp[m])})
    pd.DataFrame(rows).to_csv(out_dir / "blind_prepost_blocks.csv", index=False)

    setup_style()
    for m in tests["metric"]:
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        n = paired_plot(ax, wide[m]["pre"], wide[m]["post"], METRICS[m][0], pvals.get(m))
        set_title(ax, f"{METRICS[m][0].split(' [')[0]} – pre vs post (N = {n})", width=32)
        save_figure(fig, f"prepost_{m}", out_dir)

    for name, mlist, title in (("prepost_grid", GRID_METRICS, "Blind reaching: pre vs post training"),
                               ("prepost_main", MAIN_METRICS, "Blind reaching: main metrics, pre vs post"),
                               ("prepost_precision", PRECISION_METRICS, "Blind reaching: precision, pre vs post"),
                               ("prepost_times", TIME_METRICS, "Blind reaching: times, pre vs post")):
        gm = [m for m in mlist if m in set(tests["metric"])]
        if not gm:
            continue
        ncol = min(3, len(gm)); nrow = int(np.ceil(len(gm) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.8 * nrow), squeeze=False)
        for ax, m in zip(axes.flat, gm):
            n = paired_plot(ax, wide[m]["pre"], wide[m]["post"], "", pvals.get(m))
            set_title(ax, METRICS[m][0], width=34, fontsize=9.5)
        for ax in list(axes.flat)[len(gm):]:
            ax.axis("off")
        fig.suptitle(f"{title} (N = {n}, grey = single subjects, blue = mean ± SEM)", fontweight="bold")
        fig.tight_layout()
        save_figure(fig, name, out_dir)

    # curva per blocco, pre vs post
    cm = [m for m in ["final_error", "hit", "chosen_correct", "reaction_time_blind"] if m in metrics]
    fig, axes = plt.subplots(1, len(cm), figsize=(4.2 * len(cm), 3.8), squeeze=False)
    for ax, m in zip(axes.flat, cm):
        for phase in PHASES:
            w = per_subj_block[per_subj_block["phase"] == phase].pivot(
                index="subject", columns="block", values=m)
            blocks = sorted(w.columns); x = np.arange(len(blocks))
            mean = w[blocks].mean(); sem = (w[blocks].std(ddof=1) / np.sqrt(w[blocks].count())).fillna(0)
            ax.fill_between(x, mean - sem, mean + sem, color=PHASE_COLOR[phase], alpha=0.22, lw=0, zorder=2)
            ax.plot(x, mean, color=PHASE_COLOR[phase], linewidth=2, marker="o", markersize=5,
                    zorder=3, label=PHASE_LABEL[phase])
        ax.set_xticks(x); ax.set_xticklabels([str(b) for b in blocks]); ax.set_xlabel("Block")
        set_title(ax, METRICS[m][0], width=30, fontsize=9.5)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Blind reaching – per block of trials (mean ± SEM across subjects)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "blocks_curve", out_dir)

    # per target, pre vs post
    per_subj_t = trials.groupby(["subject", "phase", "target"])[metrics].mean().reset_index()
    tm = [m for m in ["final_error", "hit", "chosen_correct", "reaction_time_blind"] if m in metrics]
    fig, axes = plt.subplots(1, len(tm), figsize=(3.8 * len(tm), 3.8), squeeze=False)
    for ax, m in zip(axes.flat, tm):
        ts = sorted(per_subj_t["target"].unique()); x = np.arange(len(ts))
        for off, phase in zip((-0.19, 0.19), PHASES):
            w = per_subj_t[per_subj_t["phase"] == phase].pivot(index="subject", columns="target", values=m)
            mean = w.reindex(columns=ts).mean()
            sem = (w.reindex(columns=ts).std(ddof=1) / np.sqrt(w.reindex(columns=ts).count())).fillna(0)
            ax.bar(x + off, mean, width=0.36, color=PHASE_COLOR[phase], label=PHASE_LABEL[phase])
            bar_band(ax, x + off, mean.values, sem.values, 0.36, PHASE_COLOR[phase])
        ax.set_xticks(x); ax.set_xticklabels([f"T{t}" for t in ts]); ax.set_xlabel("Target")
        set_title(ax, METRICS[m][0], width=30, fontsize=9.5)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Blind reaching – per target, pre vs post (mean ± SEM across subjects)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "prepost_targets", out_dir)
    return tests


def print_table(tests):
    """Tabella riassuntiva come quella stampata dal task, ma su tutti i soggetti."""
    t = tests.set_index("metric")
    n = int(t["n"].max()) if "n" in t.columns else 0
    print(f"\n  post vs pre, all subjects (N = {n}):")
    num = lambda v, w=8: f"{v:{w}.2f}" if pd.notna(v) else "-".rjust(w)
    for m in MAIN_METRICS:
        if m not in t.index:
            continue
        r = t.loc[m]
        sd = lambda v: f"{v:.2f}" if pd.notna(v) else "-"
        line = (f"    {m:24s} pre {num(r['pre_mean'])} ± {sd(r.get('pre_sd')):>6s}   "
                f"post {num(r['post_mean'])} ± {sd(r.get('post_sd')):>6s}   delta {num(r['delta'])}")
        if pd.notna(r.get("n")) and int(r["n"]) != n:
            line += f"   [n = {int(r['n'])}]"
        if pd.notna(r.get("p")):
            stars = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
            line += f"   p {r['p']:.3f}{stars}"
        if pd.notna(r.get("cohen_d")):
            line += f"   d {r['cohen_d']:.2f}"
        print(line)


def main():
    p = argparse.ArgumentParser(description="Analisi del test blind: pre vs post training")
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    trials, trajs, targets = load_sessions(args.results)
    valid = trials[trials["success"]].copy()
    dropped = len(trials) - len(valid)
    metrics = [m for m in METRICS if m in valid.columns and valid[m].notna().any()]
    subj = valid.groupby(["subject", "phase"])[metrics].mean().reset_index()
    n_pre = (subj["phase"] == "pre").sum(); n_post = (subj["phase"] == "post").sum()
    both = set(subj.loc[subj.phase == "pre", "subject"]) & set(subj.loc[subj.phase == "post", "subject"])
    print(f"[data] {subj['subject'].nunique()} soggetti ({n_pre} pre, {n_post} post, {len(both)} con entrambe), "
          f"{len(valid)} trial validi, {dropped} esclusi (non completati)")

    for phase in PHASES:
        analyse_phase(valid, subj, phase, metrics, args.out / phase)
    tests = analyse_prepost(valid[valid["subject"].isin(both)],
                            subj[subj["subject"].isin(both)], metrics, args.out / "pre-vs-post")

    out_dirs = {"pre": args.out / "pre", "post": args.out / "post", "prepost": args.out / "pre-vs-post"}
    for subject in sorted(subj["subject"].unique()):
        plot_trajectories(subject, trajs, valid, targets, out_dirs)

    print_table(tests)
    print(f"\nOutput in {args.out}/pre, post, pre-vs-post")


if __name__ == "__main__":
    main()
