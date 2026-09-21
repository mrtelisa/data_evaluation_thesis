"""Analisi del test a 9 regioni con target invisibili (reaching_regions.py).

Legge da results/ i `<soggetto>_regions_trials.csv` (+ `_trajectory.csv` e
`_summary.json`) e valuta, per i goal di tipo `area` (centro -> area
invisibile, quello che il test misura) e a parte per i ritorni `home`:
- tempo (reach_time), percorso normalizzato, deviazione massima, picchi di
  velocita', smoothness (-ln DJ)
- metriche "blind" dalla traiettoria: prima regione in cui entra il cursore
  (accuratezza = era quella giusta), regioni sbagliate visitate prima di
  quella giusta, tempo per entrare nella regione giusta, tempo di ricerca
  dentro la regione (ingresso nella regione -> ingresso nell'area)
- divisione per regione (1..9, heatmap 3x3) e per ripetizione (1a, 2a, 3a
  visita di ogni regione = blocchi di 8 aree, curva di apprendimento).
Ogni soggetto pesa 1 (prima la media per soggetto, poi tra soggetti). Il
primo home (prima dell'avvio del timer) e i goal non riusciti sono esclusi.

Output (statistics/):
- regions_trials.csv          tutti i goal di tutti i soggetti + metriche blind
- regions_subjects.csv        soggetto x (area/home) x metriche medie
- regions_statistics.csv      tra soggetti: n/mean/sd/sem/median/min/max (area e home)
- regions_per_region.csv      per regione: tra soggetti
- regions_repetitions.csv     per ripetizione: tra soggetti
- regions_learning_tests.csv  Friedman sulle ripetizioni, Wilcoxon 1a vs ultima, delta %
- plots/heatmap_<metrica>.png, heatmaps_grid.png, learning_curve_<metrica>.png,
  learning_curves_grid.png, hist_<metrica>.png (conteggi), home_vs_area_reach_time.png,
  trajectories/<soggetto>.png, trajectories_all.png

Uso:
    python3 regions_analysis.py [--results DIR] [--out DIR]
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

warnings.filterwarnings("ignore", message="Sample size too small")
warnings.filterwarnings("ignore", message="Exact p-value calculation")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = BASE_DIR / "results"
DEFAULT_OUT = BASE_DIR / "statistics"

CANVAS_W, CANVAS_H = 1200, 650
HOME_REGION = 5
REGION_NAMES = {1: "top-left", 2: "top", 3: "top-right", 4: "left", 5: "centre",
                6: "right", 7: "bottom-left", 8: "bottom", 9: "bottom-right"}

# metrica -> (etichetta, meglio se, e' un conteggio)
METRICS = {
    "reach_time":             ("Reach time [s]", "down", False),
    "reaction_time":          ("Reaction time [s]", "down", False),
    "movement_time":          ("Movement time [s]", "down", False),
    "normalized_path_length": ("Normalized path length (1 = straight)", "down", False),
    "max_deviation":          ("Max deviation from ideal line [px]", "down", False),
    "log_dimensionless_jerk": ("Log dimensionless jerk (higher = smoother)", "up", False),
    "n_speed_peaks":          ("Speed peaks per movement", "down", True),
    "mean_speed":             ("Mean speed [px/s]", "up", False),
    "peak_speed":             ("Peak speed [px/s]", "up", False),
    # blind (dalla traiettoria, solo aree)
    "first_region_correct":   ("First region entered is the target [%]", "up", False),
    "n_wrong_regions":        ("Wrong regions visited before the target", "down", True),
    "time_to_region":         ("Time to enter the target region [s]", "down", False),
    "search_time":            ("Search time inside the region [s]", "down", False),
}
HEATMAP_METRICS = ["reach_time", "normalized_path_length", "first_region_correct",
                   "n_wrong_regions", "search_time", "log_dimensionless_jerk"]
GRID_METRICS = ["reach_time", "normalized_path_length", "max_deviation", "log_dimensionless_jerk",
                "n_speed_peaks", "first_region_correct", "n_wrong_regions", "time_to_region", "search_time"]

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SUBJECT_LINE = "#c3c2b7"
REGION_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#898781", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


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
    d = Path(out_dir) / "plots" / (sub or ""); d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png"); plt.close(fig)


def region_of(x, y):
    col = np.where(x < CANVAS_W / 3, 0, np.where(x < 2 * CANVAS_W / 3, 1, 2))
    row = np.where(y < CANVAS_H / 3, 0, np.where(y < 2 * CANVAS_H / 3, 1, 2))
    return row * 3 + col + 1


# ---------------------------------------------------------------------------
# Caricamento + metriche blind
# ---------------------------------------------------------------------------
MIN_VISIT_S = 0.25   # una regione conta come "visitata" solo se il cursore ci resta almeno tanto


def blind_metrics(traj_trial, target_region, t_start):
    """Dalla traiettoria di un goal `area` (fase go): prima regione visitata,
    regioni sbagliate, tempo per entrare nella regione giusta, ricerca.
    Una regione attraversata per meno di MIN_VISIT_S (es. l'angolo di una
    regione laterale lungo la diagonale verso un angolo) non conta."""
    g = traj_trial[traj_trial["phase"] == "go"]
    if g.empty:
        return {}
    reg = region_of(g["x"].values, g["y"].values)
    t = g["t"].values
    # soggiorni: (regione, t_in, t_out) fuori dal centro
    stays, i = [], 0
    while i < len(reg):
        j = i
        while j + 1 < len(reg) and reg[j + 1] == reg[i]:
            j += 1
        if reg[i] != HOME_REGION:
            stays.append((int(reg[i]), t[i], t[j]))
        i = j + 1
    visited = []
    for r, t_in, t_out in stays:
        if r == target_region or t_out - t_in >= MIN_VISIT_S:
            if not visited or visited[-1] != r:
                visited.append(r)
    out = {"first_region": visited[0] if visited else np.nan,
           "first_region_correct": 100.0 * (visited[0] == target_region) if visited else np.nan}
    hit = np.flatnonzero(reg == target_region)
    if hit.size:
        out["time_to_region"] = float(t[hit[0]] - t_start)
        out["n_wrong_regions"] = len(set(visited[:visited.index(target_region)])) if target_region in visited else np.nan
    else:
        out["time_to_region"] = np.nan; out["n_wrong_regions"] = np.nan
    return out


def load_sessions(results_dir):
    # <soggetto>_regions_trials.csv (prima sessione) o <soggetto>_regions_<n>_trials.csv
    files = sorted(set(glob.glob(str(Path(results_dir) / "*_regions_trials.csv")) +
                       glob.glob(str(Path(results_dir) / "*_regions_*_trials.csv"))))
    if not files:
        raise FileNotFoundError(f"Nessun *_regions_trials.csv / *_regions_*_trials.csv in {results_dir}")
    # se un soggetto ha piu' sessioni si tiene l'ultima (suffisso _n piu' alto)
    by_subject = {}
    for f in files:
        m = re.match(r"^(.*)_regions(?:_(\d+))?_trials\.csv$", os.path.basename(f))
        if not m:
            continue
        subject, n = m.group(1), int(m.group(2) or 0)
        if subject in by_subject:
            print(f"[WARN] {subject}: piu' sessioni, tengo la piu' recente")
        if subject not in by_subject or n > by_subject[subject][0]:
            by_subject[subject] = (n, f)
    frames, trajs, areas_cfg = [], {}, None
    for subject, (_, f) in sorted(by_subject.items()):
        df = pd.read_csv(f)
        df.insert(0, "subject", subject)
        df["success"] = df["success"].astype(str).str.lower().eq("true")
        tpath = f.replace("_trials.csv", "_trajectory.csv")
        traj = pd.read_csv(tpath) if os.path.exists(tpath) else None
        spath = f.replace("_trials.csv", "_summary.json")
        if areas_cfg is None and os.path.exists(spath):
            with open(spath) as fh:
                areas_cfg = json.load(fh).get("config", {}).get("areas")
        # metriche blind per i goal area: dal trials.csv se il task le ha gia' calcolate
        # (reaching_regions.py aggiornato), altrimenti dalla traiettoria
        blind_cols = ["first_region", "first_region_correct", "n_wrong_regions", "time_to_region", "search_time"]
        if all(c in df.columns for c in blind_cols):
            if traj is not None:
                trajs[subject] = traj
        elif traj is not None:
            rows = []
            for _, r in df.iterrows():
                if r["kind"] != "area" or not r["success"] or pd.isna(r["t_start"]):
                    rows.append({}); continue
                rows.append(blind_metrics(traj[traj["trial"] == r["trial"]], int(r["region"]), r["t_start"]))
            df = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
            trajs[subject] = traj
        if "time_to_region" in df.columns and "search_time" not in df.columns:
            df["search_time"] = (df["reach_time"] - df["time_to_region"]).clip(lower=0)
        # ripetizione = blocco di n_outer aree (target_number 1-8, 9-16, ...): per le
        # aree coincide con l'ennesima visita della regione, e vale anche per gli home
        n_outer = df.loc[df["kind"] == "area", "region"].nunique() or 8
        df["repetition"] = (df["target_number"] - 1) // n_outer + 1
        # il primo home (prima dell'avvio del timer) non e' un ritorno
        df["first_home"] = (df["kind"] == "home") & (df["t_start"] < 0)
        frames.append(df)
        print(f"[load] {subject}: {len(df)} goal, {int((df['kind'] == 'area').sum())} aree"
              + ("" if traj is not None else " (senza traiettoria: niente metriche blind)"))
    return pd.concat(frames, ignore_index=True), trajs, areas_cfg


# ---------------------------------------------------------------------------
# Statistiche
# ---------------------------------------------------------------------------
def describe(v):
    v = pd.Series(v).dropna().astype(float); n = len(v)
    row = {"n": n}
    if n:
        row.update(mean=v.mean(), median=v.median(), min=v.min(), max=v.max())
    if n > 1:
        row["sd"] = v.std(ddof=1); row["var"] = v.var(ddof=1); row["sem"] = row["sd"] / np.sqrt(n)
    return row


def across(per_subject, metrics, by, kind):
    rows = []
    for m in metrics:
        for key, grp in per_subject.groupby(by):
            rows.append({"kind": kind, "metric": m, by: key, **describe(grp[m])})
    return pd.DataFrame(rows)


def learning_tests(per_subject_rep, metrics, kind):
    rows = []
    reps = sorted(per_subject_rep["repetition"].unique())
    for m in metrics:
        wide = per_subject_rep.pivot(index="subject", columns="repetition", values=m)
        row = {"kind": kind, "metric": m, "better_if": METRICS[m][1], "n_repetitions": len(reps),
               "n_subjects": len(wide), "first_mean": wide[reps[0]].mean(), "last_mean": wide[reps[-1]].mean()}
        row["delta_percent"] = 100 * (row["last_mean"] - row["first_mean"]) / row["first_mean"] if row["first_mean"] else np.nan
        pair = wide[[reps[0], reps[-1]]].dropna()
        if len(pair) >= 5:
            try:
                row["wilcoxon_first_last_p"] = stats.wilcoxon(pair.iloc[:, 0], pair.iloc[:, 1], zero_method="zsplit").pvalue
            except ValueError:
                pass
        full = wide.dropna()
        if len(full) >= 3 and len(reps) >= 3:
            row["friedman_p"] = stats.friedmanchisquare(*[full[r] for r in reps]).pvalue
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Grafici
# ---------------------------------------------------------------------------
def heatmap(ax, values_by_region, title, fmt="{:.2f}", cmap="Blues", reverse=False, sd_by_region=None):
    """values_by_region: dict regione -> media tra soggetti (NaN dove manca);
    sd_by_region: dict regione -> SD tra soggetti, scritta sotto la media."""
    g = np.full((3, 3), np.nan); gs = np.full((3, 3), np.nan)
    for r, v in values_by_region.items():
        g[(r - 1) // 3, (r - 1) % 3] = v
    for r, v in (sd_by_region or {}).items():
        gs[(r - 1) // 3, (r - 1) % 3] = v
    finite = g[np.isfinite(g)]
    vmin, vmax = (finite.min(), finite.max()) if finite.size else (0, 1)
    im = ax.imshow(np.where(np.isfinite(g), g, np.nan), cmap=cmap + ("_r" if reverse else ""),
                   vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-6)
    for i in range(3):
        for j in range(3):
            v = g[i, j]
            if np.isfinite(v):
                frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                if reverse: frac = 1 - frac
                txt = fmt.format(v)
                if np.isfinite(gs[i, j]):
                    txt += "\n± " + fmt.format(gs[i, j])
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if frac > 0.6 else INK)
            else:
                ax.text(j, i, "—", ha="center", va="center", color=MUTED)
            ax.text(j - 0.42, i - 0.38, str(3 * i + j + 1), fontsize=7, color=INK2)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(title, fontsize=10)


def draw_curve(ax, wide, metric):
    reps = sorted(wide.columns); x = np.arange(len(reps))
    for _, s in wide.iterrows():
        ax.plot(x, s[reps].values, color=SUBJECT_LINE, linewidth=0.8, marker="o", markersize=2.5, zorder=1)
    mean = wide[reps].mean(); n = wide[reps].count()
    sem = (wide[reps].std(ddof=1) / np.sqrt(n)).fillna(0)
    ax.fill_between(x, mean - sem, mean + sem, color=BLUE, alpha=0.15, lw=0, zorder=2)
    ax.errorbar(x, mean, yerr=sem, color=BLUE, linewidth=2, marker="o", markersize=5, capsize=3, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([str(r) for r in reps]); ax.set_xlabel("Repetition (block of 8 areas)")
    ax.set_ylabel(METRICS[metric][0]); ax.set_xlim(-0.5, len(reps) - 0.5)
    return int(n.max())


def count_hist(ax, values, xlabel):
    v = pd.Series(values).dropna().round().astype(int)
    if v.empty:
        return
    xs = np.arange(int(v.min()), int(v.max()) + 1)
    counts = v.value_counts().reindex(xs, fill_value=0)
    ax.bar(xs, counts.values, width=0.7, color=BLUE, zorder=2)
    total = counts.sum()
    for x, c in zip(xs, counts.values):
        if c:
            ax.text(x, c + 0.05, f"{c} ({100 * c / total:.0f}%)", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(xs); ax.set_xlabel(xlabel); ax.set_ylabel("Trials")
    ax.set_ylim(0, max(counts.max(), 1) * 1.25)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))


def draw_canvas(ax, areas_cfg):
    ax.set_xlim(0, CANVAS_W); ax.set_ylim(CANVAS_H, 0); ax.set_aspect("equal")
    ax.grid(False)
    for x in (CANVAS_W / 3, 2 * CANVAS_W / 3):
        ax.axvline(x, color=GRID, linewidth=1)
    for y in (CANVAS_H / 3, 2 * CANVAS_H / 3):
        ax.axhline(y, color=GRID, linewidth=1)
    if areas_cfg:
        for r, a in areas_cfg.items():
            if int(r) == HOME_REGION:
                continue
            ax.add_patch(plt.Rectangle((a["x0"], a["y0"]), a["x1"] - a["x0"], a["y1"] - a["y0"],
                                       fill=False, edgecolor=REGION_COLORS[int(r) - 1], linewidth=1.2))
            ax.text(a["x0"] + 6, a["y0"] + 16, r, fontsize=8, color=REGION_COLORS[int(r) - 1])
    ax.add_patch(plt.Circle((CANVAS_W / 2, CANVAS_H / 2), 40, fill=False, edgecolor=INK, linewidth=1.2))
    ax.set_xticks([]); ax.set_yticks([])


def plot_trajectory(subject, traj, df_subj, areas_cfg, out_dir):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, kind, title in ((axes[0], "area", "centre → area"), (axes[1], "home", "area → centre")):
        draw_canvas(ax, areas_cfg)
        for _, r in df_subj[(df_subj["kind"] == kind) & ~df_subj["first_home"]].iterrows():
            g = traj[(traj["trial"] == r["trial"]) & (traj["phase"] == "go")]
            # colore = regione dell'area (per i ritorni: la regione da cui torna)
            reg = int(r["region"]) if kind == "area" else int(df_subj.loc[df_subj["trial"] == r["trial"] - 1, "region"].iloc[0]) if r["trial"] > 1 else 5
            ax.plot(g["x"], g["y"], color=REGION_COLORS[reg - 1], linewidth=0.9, alpha=0.8)
        ax.set_title(f"{subject} – {title}", fontsize=10)
    fig.suptitle("Cursor trajectories (colour = region of the area)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, subject, out_dir, "trajectories")


def main():
    p = argparse.ArgumentParser(description="Analisi del test a 9 regioni (target invisibili)")
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df, trajs, areas_cfg = load_sessions(args.results)
    df.to_csv(args.out / "regions_trials.csv", index=False)
    valid = df[df["success"] & ~df["first_home"]]
    areas = valid[valid["kind"] == "area"]
    homes = valid[valid["kind"] == "home"]
    n_subj = df["subject"].nunique()
    print(f"[data] {n_subj} soggetti, {len(areas)} aree valide, {len(homes)} ritorni; "
          f"esclusi {int((~df['success']).sum())} non riusciti e {int(df['first_home'].sum())} primi home")

    metrics_a = [m for m in METRICS if m in areas.columns and areas[m].notna().any()]
    metrics_h = [m for m in METRICS if m in homes.columns and homes[m].notna().any()
                 and m not in ("first_region_correct", "n_wrong_regions", "time_to_region", "search_time")]

    # --- soggetto x kind ---
    subj_a = areas.groupby("subject")[metrics_a].mean(); subj_a.insert(0, "kind", "area")
    subj_h = homes.groupby("subject")[metrics_h].mean(); subj_h.insert(0, "kind", "home")
    pd.concat([subj_a, subj_h]).reset_index().to_csv(args.out / "regions_subjects.csv", index=False)
    stats_rows = [{"kind": "area", "metric": m, "label": METRICS[m][0], **describe(subj_a[m])} for m in metrics_a]
    stats_rows += [{"kind": "home", "metric": m, "label": METRICS[m][0], **describe(subj_h[m])} for m in metrics_h]
    pd.DataFrame(stats_rows).to_csv(args.out / "regions_statistics.csv", index=False)

    # --- per regione ---
    ps_reg_a = areas.groupby(["subject", "region"])[metrics_a].mean().reset_index()
    per_region = across(ps_reg_a, metrics_a, "region", "area")
    per_region["region_name"] = per_region["region"].map(REGION_NAMES)
    per_region.to_csv(args.out / "regions_per_region.csv", index=False)

    # --- per ripetizione ---
    ps_rep_a = areas.groupby(["subject", "repetition"])[metrics_a].mean().reset_index()
    ps_rep_h = homes.groupby(["subject", "repetition"])[metrics_h].mean().reset_index()
    pd.concat([across(ps_rep_a, metrics_a, "repetition", "area"),
               across(ps_rep_h, metrics_h, "repetition", "home")]).to_csv(
        args.out / "regions_repetitions.csv", index=False)
    tests = pd.concat([learning_tests(ps_rep_a, metrics_a, "area"), learning_tests(ps_rep_h, metrics_h, "home")])
    tests.to_csv(args.out / "regions_learning_tests.csv", index=False)

    # --- grafici ---
    setup_style()
    hm = [m for m in HEATMAP_METRICS if m in metrics_a]
    for m in hm:
        pr = per_region[per_region["metric"] == m].set_index("region")
        vals, sds = pr["mean"].to_dict(), pr["sd"].to_dict() if "sd" in pr.columns else None
        fig, ax = plt.subplots(figsize=(4.4, 3.6))
        heatmap(ax, vals, "", fmt="{:.0f}%" if m == "first_region_correct" else "{:.2f}",
                reverse=(METRICS[m][1] == "up"), sd_by_region=sds)
        set_title(ax, f"{METRICS[m][0].split(' [')[0]} per region (mean ± SD, N = {n_subj})", width=40)
        save_figure(fig, f"heatmap_{m}", args.out)
    ncol = 3; nrow = int(np.ceil(len(hm) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    for ax, m in zip(axes.flat, hm):
        pr = per_region[per_region["metric"] == m].set_index("region")
        heatmap(ax, pr["mean"].to_dict(), "\n".join(textwrap.wrap(METRICS[m][0], 34)),
                fmt="{:.0f}%" if m == "first_region_correct" else "{:.2f}", reverse=(METRICS[m][1] == "up"),
                sd_by_region=pr["sd"].to_dict() if "sd" in pr.columns else None)
    for ax in list(axes.flat)[len(hm):]:
        ax.axis("off")
    fig.suptitle(f"Centre → invisible area, per region (mean ± SD over N = {n_subj} subjects; darker = worse)",
                 fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "heatmaps_grid", args.out)

    for m in metrics_a:
        wide = ps_rep_a.pivot(index="subject", columns="repetition", values=m)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        n = draw_curve(ax, wide, m)
        set_title(ax, f"{METRICS[m][0].split(' [')[0]} per repetition, centre → area (mean ± SEM, N = {n})")
        save_figure(fig, f"learning_curve_{m}", args.out)
    gm = [m for m in GRID_METRICS if m in metrics_a]
    ncol = 3; nrow = int(np.ceil(len(gm) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.5 * nrow), squeeze=False)
    for ax, m in zip(axes.flat, gm):
        n = draw_curve(ax, ps_rep_a.pivot(index="subject", columns="repetition", values=m), m)
        set_title(ax, METRICS[m][0], width=36, fontsize=10); ax.set_ylabel(""); ax.set_xlabel("")
    for ax in list(axes.flat)[len(gm):]:
        ax.axis("off")
    fig.suptitle(f"9-region test – learning over repetitions (mean ± SEM, N = {n}, grey = subjects)",
                 fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "learning_curves_grid", args.out)

    for m in [m for m in metrics_a if METRICS[m][2]]:
        fig, ax = plt.subplots(figsize=(5, 3.8))
        count_hist(ax, areas[m], METRICS[m][0])
        set_title(ax, f"{METRICS[m][0]} – all area trials ({len(areas)} trials, N = {n_subj})", width=45)
        save_figure(fig, f"hist_{m}", args.out)

    fig, ax = plt.subplots(figsize=(6, 4))
    for kind, ps, color, lab in (("area", ps_rep_a, BLUE, "Centre → area"), ("home", ps_rep_h, ORANGE, "Area → centre")):
        wide = ps.pivot(index="subject", columns="repetition", values="reach_time")
        reps = sorted(wide.columns); x = np.arange(len(reps)); n = wide.count()
        sem = (wide.std(ddof=1) / np.sqrt(n)).fillna(0)
        ax.errorbar(x, wide.mean(), yerr=sem, color=color, linewidth=2, marker="o", capsize=3, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([str(r) for r in reps]); ax.set_xlabel("Repetition")
    ax.set_ylabel("Reach time [s]"); ax.legend()
    set_title(ax, "Reach time per repetition – area vs return to centre (mean ± SEM)")
    save_figure(fig, "home_vs_area_reach_time", args.out)

    for subject, traj in trajs.items():
        plot_trajectory(subject, traj, df[df["subject"] == subject], areas_cfg, args.out)
    if trajs:
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        draw_canvas(ax, areas_cfg)
        for subject, traj in trajs.items():
            d = df[(df["subject"] == subject) & (df["kind"] == "area") & df["success"]]
            for _, r in d.iterrows():
                g = traj[(traj["trial"] == r["trial"]) & (traj["phase"] == "go")]
                ax.plot(g["x"], g["y"], color=REGION_COLORS[int(r["region"]) - 1], linewidth=0.6, alpha=0.5)
        set_title(ax, f"Centre → area trajectories, all subjects (N = {n_subj}, colour = region)")
        save_figure(fig, "trajectories_all", args.out)

    cols = [c for c in ["kind", "metric", "first_mean", "last_mean", "delta_percent", "wilcoxon_first_last_p", "friedman_p"]
            if c in tests.columns]
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print("\n" + tests[cols].to_string(index=False))
    print(f"\nOutput in {args.out}")


if __name__ == "__main__":
    main()
