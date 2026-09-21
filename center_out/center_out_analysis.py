"""Analisi del training center-out: curva di apprendimento per blocchi.

Legge tutti i `<soggetto>_center_out_trials.csv` in results/ (un file per
soggetto, una sessione ciascuno) e valuta se e quanto i soggetti migliorano
blocco dopo blocco (blocco = `block_size` target consecutivi, default 8:
target 1-8, 9-16, ... -> il numero di blocchi segue il numero di target).

Trial usati: solo quelli riusciti (`success`), non `over_time`, di tipo
`target` per l'analisi principale; i ritorni al centro (`home`) sono
analizzati a parte. Ogni soggetto pesa 1 in ogni blocco (prima la media per
soggetto, poi la statistica tra soggetti).

Output (in statistics/):
- center_out_subject_blocks.csv       soggetto x blocco x metrica (tabella lunga)
- center_out_blocks_statistics.csv    per blocco e metrica: n, mean, sd, sem,
                                      median, ci95 (target e home)
- center_out_learning_tests.csv       per metrica: Friedman sui blocchi,
                                      Wilcoxon primo vs ultimo blocco e prima
                                      vs seconda meta', delta %, pendenza media
- center_out_subject_slopes.csv       learning rate (pendenza) per soggetto
- center_out_directions_statistics.csv metriche per direzione del target
- center_out_excluded.csv             trial esclusi (over_time / non riusciti)
- plots/learning_curve_<metrica>.png, learning_curves_grid.png,
  home_vs_target_reach_time.png, directions_polar.png, score_per_block.png

Uso:
    python3 center_out_analysis.py [--results DIR] [--out DIR]
        [--block-size 8] [--include-over-time] [--style line|bar|box]
"""

import argparse
import glob
import json
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

# con N piccolo scipy avvisa che usa l'approssimazione normale: e' atteso
warnings.filterwarnings("ignore", message="Sample size too small")
warnings.filterwarnings("ignore", message="Exact p-value calculation")


def wilcoxon_p(a, b=None):
    """p del test di Wilcoxon (coppie a-b, o a contro 0); NaN se non calcolabile."""
    try:
        return stats.wilcoxon(a, b, zero_method="zsplit").pvalue
    except ValueError:
        return np.nan


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = BASE_DIR / "results"
DEFAULT_OUT = BASE_DIR / "statistics"

# metrica -> (etichetta asse, "meglio se" scende/sale)
METRICS = {
    "reach_time":             ("Reach time [s]", "down"),
    "reaction_time":          ("Reaction time [s]", "down"),
    "movement_time":          ("Movement time [s]", "down"),
    "normalized_path_length": ("Normalized path length (1 = straight)", "down"),
    "max_deviation":          ("Max deviation from ideal line [px]", "down"),
    "log_dimensionless_jerk": ("Log dimensionless jerk (higher = smoother)", "up"),
    "n_speed_peaks":          ("Speed peaks per movement", "down"),
    "mean_speed":             ("Mean speed [px/s]", "up"),
    "peak_speed":             ("Peak speed [px/s]", "up"),
    # derivate per blocco
    "reach_time_sd":          ("Reach time within-block SD [s]", "down"),
    "points":                 ("Score (points per block)", "up"),
    "block_duration":         ("Block duration [s]", "down"),
}
# metriche nella figura riassuntiva
GRID_METRICS = ["reach_time", "reaction_time", "movement_time",
                "normalized_path_length", "max_deviation", "log_dimensionless_jerk",
                "n_speed_peaks", "reach_time_sd", "points"]
RAW_METRICS = [m for m in METRICS if m not in ("reach_time_sd", "points", "block_duration")]

# --- stile grafici (stessa palette dei questionari) ---
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SUBJECT_LINE = "#c3c2b7"


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


# ---------------------------------------------------------------------------
# Caricamento
# ---------------------------------------------------------------------------
def load_sessions(results_dir):
    """Un DataFrame con tutti i trial di tutti i soggetti (+ colonna subject)
    e il block_size letto dai summary.json (None se assente)."""
    files = sorted(glob.glob(str(Path(results_dir) / "*_center_out_trials.csv")))
    if not files:
        raise FileNotFoundError(f"Nessun *_center_out_trials.csv in {results_dir}")
    frames, block_sizes = [], set()
    for f in files:
        subject = re.sub(r"_center_out_trials\.csv$", "", os.path.basename(f))
        df = pd.read_csv(f)
        df.insert(0, "subject", subject)
        frames.append(df)
        summary = f.replace("_trials.csv", "_summary.json")
        if os.path.exists(summary):
            with open(summary) as fh:
                cfg = json.load(fh).get("config", {})
            if cfg.get("block_size"):
                block_sizes.add(int(cfg["block_size"]))
        print(f"[load] {subject}: {len(df)} trial")
    if len(block_sizes) > 1:
        print(f"[WARN] block_size diversi nei summary: {sorted(block_sizes)}")
    df = pd.concat(frames, ignore_index=True)
    df["success"] = df["success"].astype(str).str.lower().eq("true")
    if "over_time" in df.columns:
        df["over_time"] = df["over_time"].astype(str).str.lower().eq("true")
    else:
        df["over_time"] = False
    return df, (block_sizes.pop() if len(block_sizes) == 1 else None)


# ---------------------------------------------------------------------------
# Statistiche
# ---------------------------------------------------------------------------
def block_label(b, block_size):
    return f"{(b - 1) * block_size + 1}-{b * block_size}"


def subject_block_means(trials, all_trials, block_size):
    """Una riga per soggetto x blocco: media delle metriche sui trial del
    blocco, SD del reach time, punti totali e durata del blocco (calcolata su
    tutti i trial del blocco, home compresi)."""
    trials = trials.copy()
    trials["block"] = (trials["target_number"] - 1) // block_size + 1
    g = trials.groupby(["subject", "block"])
    out = g[RAW_METRICS].mean()
    out["reach_time_sd"] = g["reach_time"].std(ddof=1)
    out["points"] = g["points"].sum()
    out["n_trials"] = g.size()
    at = all_trials.copy()
    at["block"] = (at["target_number"] - 1) // block_size + 1
    ga = at.groupby(["subject", "block"])
    out["block_duration"] = ga["t_end"].max() - ga["t_shown"].min()
    out["n_targets_in_block"] = ga["target_number"].nunique()
    out["complete"] = out["n_targets_in_block"] == block_size
    return out.reset_index()


def across_subjects(sb, metrics, block_size, kind):
    """Statistica tra soggetti per ogni blocco e metrica."""
    rows = []
    for metric in metrics:
        for b, grp in sb.groupby("block"):
            v = grp[metric].dropna()
            n = len(v)
            row = {"kind": kind, "metric": metric, "block": b,
                   "targets": block_label(b, block_size), "n": n}
            if n:
                row.update(mean=v.mean(), median=v.median(),
                           min=v.min(), max=v.max())
            if n > 1:
                sd = v.std(ddof=1); sem = sd / np.sqrt(n)
                h = stats.t.ppf(0.975, n - 1) * sem
                row.update(sd=sd, var=v.var(ddof=1), sem=sem,
                           ci95_low=v.mean() - h, ci95_high=v.mean() + h)
            rows.append(row)
    return pd.DataFrame(rows)


def learning_tests(sb, metrics, block_size, kind):
    """Per ogni metrica: Friedman sui blocchi (soggetti con tutti i blocchi),
    Wilcoxon primo vs ultimo blocco, prima vs seconda meta', delta %,
    pendenza per soggetto e test sulla pendenza media."""
    n_blocks = int(sb["block"].max())
    wide_all = {m: sb.pivot(index="subject", columns="block", values=m) for m in metrics}
    rows, slopes = [], []
    for metric in metrics:
        wide = wide_all[metric]
        better = METRICS[metric][1]
        row = {"kind": kind, "metric": metric, "better_if": better,
               "n_blocks": n_blocks, "n_subjects": len(wide)}
        first, last = wide[1], wide[n_blocks]
        paired = pd.concat([first, last], axis=1).dropna()
        row["first_block_mean"] = first.mean()
        row["last_block_mean"] = last.mean()
        row["delta_percent"] = 100 * (last.mean() - first.mean()) / first.mean() if first.mean() else np.nan
        row["n_paired_first_last"] = len(paired)
        if len(paired) >= 5:
            row["wilcoxon_first_last_p"] = wilcoxon_p(paired.iloc[:, 0], paired.iloc[:, 1])
        # prima meta' vs seconda meta' dei blocchi
        half = n_blocks // 2
        if half >= 1:
            h1 = wide[[b for b in wide.columns if b <= half]].mean(axis=1)
            h2 = wide[[b for b in wide.columns if b > half]].mean(axis=1)
            hp = pd.concat([h1, h2], axis=1).dropna()
            row["first_half_mean"] = h1.mean(); row["second_half_mean"] = h2.mean()
            if len(hp) >= 5:
                row["wilcoxon_halves_p"] = wilcoxon_p(hp.iloc[:, 0], hp.iloc[:, 1])
        # Friedman: solo soggetti con tutti i blocchi
        full = wide.dropna()
        row["n_subjects_all_blocks"] = len(full)
        if len(full) >= 3 and n_blocks >= 3:
            row["friedman_p"] = stats.friedmanchisquare(*[full[b] for b in full.columns]).pvalue
        # pendenza (learning rate) per soggetto: metrica vs numero di blocco
        subj_slopes = []
        for subject, series in wide.iterrows():
            s = series.dropna()
            if len(s) >= 3:
                slope = stats.linregress(s.index.astype(float), s.values).slope
                subj_slopes.append(slope)
                slopes.append({"kind": kind, "subject": subject, "metric": metric,
                               "slope_per_block": slope, "n_blocks": len(s)})
        if subj_slopes:
            row["mean_slope_per_block"] = float(np.mean(subj_slopes))
            row["n_subjects_slope"] = len(subj_slopes)
            if len(subj_slopes) >= 5:
                row["wilcoxon_slope_vs_0_p"] = wilcoxon_p(subj_slopes)
            improving = [s < 0 if better == "down" else s > 0 for s in subj_slopes]
            row["subjects_improving_percent"] = 100 * np.mean(improving)
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(slopes)


def direction_stats(trials):
    """Metriche per direzione del target (index 0..7 -> angolo), tra soggetti."""
    per_subject = trials.groupby(["subject", "index"])[RAW_METRICS].mean().reset_index()
    rows = []
    for metric in ["reach_time", "movement_time", "normalized_path_length",
                   "max_deviation", "log_dimensionless_jerk", "n_speed_peaks"]:
        for idx, grp in per_subject.groupby("index"):
            v = grp[metric].dropna(); n = len(v)
            row = {"metric": metric, "index": int(idx), "n": n}
            if n:
                row.update(mean=v.mean(), median=v.median())
            if n > 1:
                row["sd"] = v.std(ddof=1); row["var"] = v.var(ddof=1); row["sem"] = row["sd"] / np.sqrt(n)
            rows.append(row)
    return pd.DataFrame(rows), per_subject


# ---------------------------------------------------------------------------
# Grafici
# ---------------------------------------------------------------------------
def set_title(ax, text, width=55, **kw):
    """Titolo centrato sul grafico, a capo se piu' lungo di `width` caratteri."""
    ax.set_title("\n".join(textwrap.wrap(text, width)), loc="center", **kw)


def save_figure(fig, name, out_dir):
    plots = Path(out_dir) / "plots"; plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots / f"{name}.png"); plt.close(fig)


def draw_learning_curve(ax, sb, metric, block_size, style="line", color=BLUE,
                        label=None, show_subjects=True):
    """Curva media +- SEM tra soggetti; ogni soggetto in grigio sottile."""
    wide = sb.pivot(index="subject", columns="block", values=metric)
    blocks = np.array(sorted(wide.columns))
    x = np.arange(len(blocks))
    if show_subjects:
        for _, s in wide.iterrows():
            ax.plot(x, s[blocks].values, color=SUBJECT_LINE, linewidth=0.8,
                    marker="o", markersize=2.5, alpha=0.9, zorder=1)
    mean = wide[blocks].mean(); n = wide[blocks].count()
    sem = (wide[blocks].std(ddof=1) / np.sqrt(n)).fillna(0)   # n = 1 -> nessuna barra
    if style == "bar":
        ax.bar(x, mean, yerr=sem, color=color, width=0.6, capsize=3, zorder=2,
               error_kw={"ecolor": INK2, "elinewidth": 1}, label=label)
    elif style == "box":
        ax.boxplot([wide[b].dropna() for b in blocks], positions=x, widths=0.5,
                   patch_artist=True, showmeans=True,
                   boxprops={"facecolor": "#cde2fb", "edgecolor": color},
                   medianprops={"color": color, "linewidth": 2},
                   whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                   flierprops={"marker": "o", "markersize": 3, "markerfacecolor": MUTED,
                               "markeredgecolor": "none"},
                   meanprops={"marker": "D", "markersize": 4, "markerfacecolor": INK,
                              "markeredgecolor": "none"})
    else:
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.15, lw=0, zorder=2)
        ax.errorbar(x, mean, yerr=sem, color=color, linewidth=2, marker="o",
                    markersize=5, capsize=3, zorder=3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([block_label(b, block_size) for b in blocks],
                       rotation=45 if len(blocks) > 8 else 0,
                       ha="right" if len(blocks) > 8 else "center")
    ax.set_xlabel("Targets (block)")
    ax.set_ylabel(METRICS[metric][0])
    ax.set_xlim(-0.5, len(blocks) - 0.5)
    return int(n.max())


def plot_learning_curves(sb, block_size, out_dir, style):
    setup_style()
    for metric in METRICS:
        if sb[metric].notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        n = draw_learning_curve(ax, sb, metric, block_size, style)
        set_title(ax, f"{METRICS[metric][0].split(' [')[0]} per block of {block_size} targets "
                      f"(mean ± SEM, N = {n})")
        save_figure(fig, f"learning_curve_{metric}", out_dir)

    metrics = [m for m in GRID_METRICS if sb[m].notna().sum() > 0]
    ncol = 3; nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow))
    for ax, metric in zip(axes.flat, metrics):
        n = draw_learning_curve(ax, sb, metric, block_size, style)
        set_title(ax, METRICS[metric][0].split(" [")[0], width=40, fontsize=10)
        ax.set_ylabel(""); ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=8)
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    fig.suptitle(f"Learning curves – targets, blocks of {block_size}, mean ± SEM (N = {n}), "
                 f"grey = single subjects", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "learning_curves_grid", out_dir)


def plot_home_vs_target(sb_target, sb_home, block_size, out_dir):
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    draw_learning_curve(ax, sb_target, "reach_time", block_size, "line", BLUE,
                        label="Target (centre → target)", show_subjects=False)
    draw_learning_curve(ax, sb_home, "reach_time", block_size, "line", ORANGE,
                        label="Home (target → centre)", show_subjects=False)
    ax.legend(loc="upper right")
    set_title(ax, "Reach time per block – target vs home return (mean ± SEM)")
    save_figure(fig, "home_vs_target_reach_time", out_dir)


def plot_directions(dir_stats, n_circle, out_dir):
    if dir_stats.empty or dir_stats["index"].max() >= n_circle:
        print("[plot] directions: index fuori dal cerchio a 8 target, polare saltato")
        return
    setup_style()
    metrics = [("reach_time", "Reach time [s]"),
               ("normalized_path_length", "Normalized path length"),
               ("n_speed_peaks", "Speed peaks")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.2),
                             subplot_kw={"projection": "polar"})
    for ax, (metric, label) in zip(axes, metrics):
        d = dir_stats[dir_stats["metric"] == metric].sort_values("index")
        theta = np.deg2rad(d["index"] * 360.0 / n_circle)
        theta_c = np.append(theta, theta.iloc[0])
        mean = np.append(d["mean"].values, d["mean"].values[0])
        sem = np.append(d["sem"].fillna(0).values, d["sem"].fillna(0).values[0])
        ax.fill_between(theta_c, mean - sem, mean + sem, color=BLUE, alpha=0.15, lw=0)
        ax.plot(theta_c, mean, color=BLUE, linewidth=2, marker="o", markersize=4)
        ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
        lo, hi = np.nanmin(mean - sem), np.nanmax(mean + sem)
        ax.set_ylim(max(0, lo - 0.3 * (hi - lo)), hi + 0.1 * (hi - lo))
        ax.set_xticks(theta); ax.set_xticklabels([f"{int(a)}°" for a in d["index"] * 360.0 / n_circle])
        ax.set_title(label, fontsize=10, pad=12)
        ax.grid(True, color=GRID); ax.spines["polar"].set_color(AXIS)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Per-direction performance (mean ± SEM across subjects)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "directions_polar", out_dir)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Analisi del training center-out")
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--block-size", type=int, default=None,
                   help="target per blocco (default: dal summary.json, altrimenti 8)")
    p.add_argument("--include-over-time", action="store_true",
                   help="tieni anche i trial over_time (default: esclusi)")
    p.add_argument("--style", choices=["line", "bar", "box"], default="line")
    p.add_argument("--n-circle", type=int, default=8, help="target sul cerchio (direzioni)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df, bs_from_summary = load_sessions(args.results)
    block_size = args.block_size or bs_from_summary or 8
    print(f"[config] block_size = {block_size}, soggetti = {df['subject'].nunique()}")

    # --- esclusioni ---
    keep = df["success"]
    if not args.include_over_time:
        keep &= ~df["over_time"]
    excluded = df.loc[~keep, ["subject", "trial", "kind", "target_number", "index",
                              "success", "end_reason", "over_time", "reach_time"]]
    excluded.to_csv(args.out / "center_out_excluded.csv", index=False)
    print(f"[filter] esclusi {len(excluded)} trial "
          f"({int((~df['success']).sum())} non riusciti, {int(df['over_time'].sum())} over_time)")
    valid = df[keep]
    targets = valid[valid["kind"] == "target"]
    homes = valid[valid["kind"] == "home"]

    # --- soggetto x blocco ---
    sb_t = subject_block_means(targets, df, block_size)
    sb_h = subject_block_means(homes, df, block_size)
    sb_t.insert(0, "kind", "target"); sb_h.insert(0, "kind", "home")
    sb_all = pd.concat([sb_t, sb_h], ignore_index=True)
    sb_all.insert(3, "targets", sb_all["block"].map(lambda b: block_label(b, block_size)))
    sb_all.to_csv(args.out / "center_out_subject_blocks.csv", index=False)

    # --- statistiche tra soggetti e test ---
    metrics = [m for m in METRICS if sb_t[m].notna().sum() > 0]
    stats_df = pd.concat([across_subjects(sb_t, metrics, block_size, "target"),
                          across_subjects(sb_h, metrics, block_size, "home")])
    stats_df.to_csv(args.out / "center_out_blocks_statistics.csv", index=False)
    tests_t, slopes_t = learning_tests(sb_t, metrics, block_size, "target")
    tests_h, slopes_h = learning_tests(sb_h, metrics, block_size, "home")
    pd.concat([tests_t, tests_h]).to_csv(args.out / "center_out_learning_tests.csv", index=False)
    pd.concat([slopes_t, slopes_h]).to_csv(args.out / "center_out_subject_slopes.csv", index=False)

    # --- direzioni ---
    dir_stats, _ = direction_stats(targets)
    dir_stats.to_csv(args.out / "center_out_directions_statistics.csv", index=False)

    # --- grafici ---
    plot_learning_curves(sb_t, block_size, args.out, args.style)
    plot_home_vs_target(sb_t, sb_h, block_size, args.out)
    plot_directions(dir_stats, args.n_circle, args.out)

    # --- riepilogo a schermo ---
    print(f"\nBlocchi di {block_size} target, N = {sb_t['subject'].nunique()} soggetti")
    cols = ["metric", "first_block_mean", "last_block_mean", "delta_percent",
            "wilcoxon_first_last_p", "friedman_p", "mean_slope_per_block"]
    cols = [c for c in cols if c in tests_t.columns]
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(tests_t[cols].to_string(index=False))
    print(f"\nOutput in {args.out}")


if __name__ == "__main__":
    main()
