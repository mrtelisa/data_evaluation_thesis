"""Funzioni comuni alle analisi sul robot reale (test pre/post e training).

- load_session(json_path): il json della sessione + l'odometria (csv) accanto
- parse_session_name("S000_1") -> ("S000", 1)
- odometry_metrics(odom, ...): velocita' media, % fermo, arresti, retromarcia,
  rotazione totale, dall'odometria a 20 Hz
- log_dimensionless_jerk(t, x, y): -ln(DJ) (Hogan & Sternad 2009), come in
  session_metrics.py
- paired_stats(pre, post): Shapiro-Wilk sulle differenze -> t-test appaiato
  o Wilcoxon, con d di Cohen e r di Wilcoxon
- grafici: stile comune, paired_plot (pre/post per soggetto), set_title
"""

import json
import math
import os
import re
import warnings
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", message="Sample size too small")
warnings.filterwarnings("ignore", message="Exact p-value calculation")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = BASE_DIR / "statistics"

ODOM_HZ = 20.0
STOP_SPEED = 0.05        # [m/s] sotto questa velocita' il robot e' fermo
STOP_MIN_S = 0.5         # [s] un arresto conta se dura almeno tanto
BACKWARD_SPEED = 0.02    # [m/s] vx sotto -questo = retromarcia

# --- stile (stessa palette degli altri script) ---
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
# confronto prima/dopo: arancione carota vs blu scuro (tinte opposte e molto
# diverse in luminosita': leggibili anche in bianco/nero e con daltonismo)
PRE_COLOR, POST_COLOR = "#ed7014", "#104281"
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


def set_title(ax, text, width=55, **kw):
    ax.set_title("\n".join(textwrap.wrap(text, width)), loc="center", **kw)


def save_figure(fig, name, out_dir=DEFAULT_OUT):
    plots = Path(out_dir) / "plots"; plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots / f"{name}.png"); plt.close(fig)


# ---------------------------------------------------------------------------
# Caricamento
# ---------------------------------------------------------------------------
def parse_session_name(name):
    """'S000' -> ('S000', 0); 'S000_1' -> ('S000', 1)."""
    m = re.match(r"^(.*?)_(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, 0)


def load_session(json_path):
    """(summary dict, odometria DataFrame o None). La colonna `speed` [m/s] e'
    calcolata da vx, vy se presenti, altrimenti dalle posizioni."""
    json_path = Path(json_path)
    with open(json_path) as f:
        summary = json.load(f)
    odom_name = summary.get("odometry_file") or json_path.name.replace("_session.json", "_odometry.csv")
    odom_path = json_path.parent / odom_name
    odom = None
    if odom_path.exists():
        odom = pd.read_csv(odom_path)
        for c in ("t", "t_test", "x", "y", "theta", "vx", "vy", "vtheta"):
            if c in odom.columns:
                odom[c] = pd.to_numeric(odom[c], errors="coerce")
        if "vx" in odom.columns and odom["vx"].notna().any():
            odom["speed"] = np.hypot(odom["vx"].fillna(0), odom["vy"].fillna(0))
        else:
            dt = odom["t"].diff().replace(0, np.nan)
            odom["speed"] = (np.hypot(odom["x"].diff(), odom["y"].diff()) / dt).fillna(0)
    return summary, odom


def load_targets(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df["name"] = df["name"].astype(str).str.strip()
    df = df.dropna(subset=["x", "y"])
    if "radius" not in df.columns:
        df["radius"] = 0.7
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metriche dall'odometria
# ---------------------------------------------------------------------------
def log_dimensionless_jerk(t, x, y, hz=ODOM_HZ):
    t, x, y = (np.asarray(a, dtype=float) for a in (t, x, y))
    if t.size < 5:
        return np.nan
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, x, y = t[keep], x[keep], y[keep]
    dt = 1.0 / hz
    tu = np.arange(t[0], t[-1], dt)
    if tu.size < 5:
        return np.nan
    xu, yu = np.interp(tu, t, x), np.interp(tu, t, y)
    vx, vy = np.gradient(xu, dt), np.gradient(yu, dt)
    ax, ay = np.gradient(vx, dt), np.gradient(vy, dt)
    jx, jy = np.gradient(ax, dt), np.gradient(ay, dt)
    duration = tu[-1] - tu[0]
    amplitude = math.hypot(xu[-1] - xu[0], yu[-1] - yu[0])
    if duration <= 0 or amplitude < 1e-3:
        return np.nan
    dj = math.sqrt(0.5 * float(np.sum(jx ** 2 + jy ** 2) * dt) * duration ** 5 / amplitude ** 2)
    return -math.log(dj) if dj > 0 else np.nan


def count_stops(t, speed, thr=STOP_SPEED, min_s=STOP_MIN_S):
    """Numero di arresti (velocita' < thr per almeno min_s) e secondi fermi."""
    t, speed = np.asarray(t, dtype=float), np.asarray(speed, dtype=float)
    stopped = speed < thr
    n, total, i = 0, 0.0, 0
    while i < len(stopped):
        if stopped[i]:
            j = i
            while j + 1 < len(stopped) and stopped[j + 1]:
                j += 1
            dur = t[j] - t[i] + (1.0 / ODOM_HZ)
            if dur >= min_s:
                n += 1; total += dur
            i = j + 1
        else:
            i += 1
    return n, total


def path_length(x, y):
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))


def odometry_metrics(odom):
    """Metriche di un tratto di odometria (DataFrame con t, x, y, theta, speed)."""
    if odom is None or len(odom) < 2:
        return {}
    t = odom["t"].values
    duration = float(t[-1] - t[0])
    m = {"odom_duration": duration,
         "odom_path_length": path_length(odom["x"].values, odom["y"].values),
         "mean_speed": float(odom["speed"].mean()),
         "peak_speed": float(odom["speed"].max())}
    n_stops, stopped_s = count_stops(t, odom["speed"].values)
    m["n_stops"] = n_stops
    m["stopped_percent"] = 100 * stopped_s / duration if duration > 0 else np.nan
    if "vx" in odom.columns and odom["vx"].notna().any():
        m["backward_percent"] = 100 * float((odom["vx"] < -BACKWARD_SPEED).mean())
    if "theta" in odom.columns and odom["theta"].notna().any():
        m["total_rotation_deg"] = float(np.degrees(np.sum(np.abs(np.diff(np.unwrap(odom["theta"].values))))))
    m["log_dimensionless_jerk_odom"] = log_dimensionless_jerk(t, odom["x"].values, odom["y"].values)
    return m


# ---------------------------------------------------------------------------
# Statistica appaiata (protocollo: Shapiro -> t-test appaiato o Wilcoxon)
# ---------------------------------------------------------------------------
def paired_stats(pre, post):
    """pre, post: Series allineate per soggetto. Restituisce un dict con
    descrittive, test scelto, p ed effect size."""
    d = pd.concat([pre, post], axis=1).dropna()
    n = len(d)
    row = {"n": n}
    if n == 0:
        return row
    a, b = d.iloc[:, 0].values, d.iloc[:, 1].values
    diff = b - a
    row.update(pre_mean=a.mean(), pre_sd=a.std(ddof=1) if n > 1 else np.nan,
               pre_var=a.var(ddof=1) if n > 1 else np.nan, pre_median=np.median(a),
               post_mean=b.mean(), post_sd=b.std(ddof=1) if n > 1 else np.nan,
               post_var=b.var(ddof=1) if n > 1 else np.nan, post_median=np.median(b),
               diff_mean=diff.mean(),
               diff_percent=100 * diff.mean() / a.mean() if a.mean() else np.nan,
               n_improved_lower=int((diff < 0).sum()), n_improved_higher=int((diff > 0).sum()))
    if n < 3:
        return row
    sd_diff = diff.std(ddof=1)
    row["cohen_d"] = diff.mean() / sd_diff if sd_diff > 0 else np.nan
    if np.allclose(diff, 0):
        row["test"] = "none (no change)"
        return row
    row["shapiro_p"] = stats.shapiro(diff).pvalue
    if row["shapiro_p"] > 0.05:
        row["test"] = "paired t-test"
        row["p"] = stats.ttest_rel(b, a).pvalue
    else:
        row["test"] = "Wilcoxon"
        try:
            row["p"] = stats.wilcoxon(b, a, zero_method="zsplit").pvalue
        except ValueError:
            row["p"] = np.nan
    # r di Wilcoxon = |Z| / sqrt(n), sempre riportato
    nz = diff[diff != 0]; k = len(nz)
    if k >= 2:
        ranks = stats.rankdata(np.abs(nz))
        w_plus = ranks[nz > 0].sum()
        mu, sigma = k * (k + 1) / 4, math.sqrt(k * (k + 1) * (2 * k + 1) / 24)
        row["wilcoxon_r"] = abs(w_plus - mu) / sigma / math.sqrt(k) if sigma > 0 else np.nan
    return row


# ---------------------------------------------------------------------------
# Grafico pre/post appaiato
# ---------------------------------------------------------------------------
def paired_plot(ax, pre, post, ylabel, labels=("Pre", "Post"), p=None):
    """Due colonne; un segmento grigio per soggetto; media +- SEM in blu."""
    d = pd.concat([pre, post], axis=1).dropna()
    a, b = d.iloc[:, 0].values, d.iloc[:, 1].values
    for ai, bi in zip(a, b):
        ax.plot([0, 1], [ai, bi], color=SUBJECT_LINE, linewidth=1, marker="o",
                markersize=3.5, zorder=1)
    n = len(d)
    if n:
        means = [a.mean(), b.mean()]
        sems = [a.std(ddof=1) / math.sqrt(n) if n > 1 else 0,
                b.std(ddof=1) / math.sqrt(n) if n > 1 else 0]
        lo = [m - s for m, s in zip(means, sems)]
        hi = [m + s for m, s in zip(means, sems)]
        ax.fill_between([0, 1], lo, hi, color=POST_COLOR, alpha=0.22, lw=0, zorder=2)
        ax.plot([0, 1], means, color=POST_COLOR, linewidth=2.2, marker="o", markersize=7, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels); ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel(ylabel)
    if p is not None and not np.isnan(p):
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(0.5, 0.97, f"p = {p:.3f} {stars}", transform=ax.transAxes,
                ha="center", va="top", color=INK2, fontsize=9)
    return n


# ---------------------------------------------------------------------------
# Grafici per metriche di conteggio (istogrammi) e descrittive di un gruppo
# ---------------------------------------------------------------------------
def count_hist(ax, values, ylabel="Subjects", color=BLUE, label=None, offset=0.0, width=0.7):
    """Istogramma di una metrica intera: quanti soggetti hanno ogni valore."""
    v = pd.Series(values).dropna().round().astype(int)
    if v.empty:
        return
    lo, hi = int(v.min()), int(v.max())
    xs = np.arange(lo, hi + 1)
    counts = v.value_counts().reindex(xs, fill_value=0)
    ax.bar(xs + offset, counts.values, width=width, color=color, label=label, zorder=2)
    for x, c in zip(xs + offset, counts.values):
        if c:
            ax.text(x, c + 0.05, str(c), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(xs); ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylim(0, max(counts.max(), 1) * 1.25)


def grouped_count_hist(ax, pre, post, labels=("Pre", "Post"), p=None):
    """Istogrammi affiancati pre/post di una metrica intera."""
    d = pd.concat([pre, post], axis=1).dropna()
    a, b = d.iloc[:, 0].round().astype(int), d.iloc[:, 1].round().astype(int)
    if d.empty:
        return 0
    lo, hi = int(min(a.min(), b.min())), int(max(a.max(), b.max()))
    xs = np.arange(lo, hi + 1)
    ca, cb = a.value_counts().reindex(xs, fill_value=0), b.value_counts().reindex(xs, fill_value=0)
    ax.bar(xs - 0.2, ca.values, width=0.38, color=PRE_COLOR, label=labels[0], zorder=2)
    ax.bar(xs + 0.2, cb.values, width=0.38, color=POST_COLOR, label=labels[1], zorder=2)
    for x, c in zip(xs - 0.2, ca.values):
        if c: ax.text(x, c + 0.05, str(c), ha="center", va="bottom", fontsize=8, color=INK)
    for x, c in zip(xs + 0.2, cb.values):
        if c: ax.text(x, c + 0.05, str(c), ha="center", va="bottom", fontsize=8, color=INK)
    top = max(ca.max(), cb.max(), 1)
    ax.set_xticks(xs); ax.set_ylabel("Subjects"); ax.set_ylim(0, top * 1.35)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(loc="upper right", fontsize=8)
    if p is not None and not np.isnan(p):
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(0.02, 0.97, f"p = {p:.3f} {stars}", transform=ax.transAxes, ha="left", va="top",
                color=INK2, fontsize=9)
    return len(d)


def box_dots(ax, values, ylabel, color=BLUE):
    """Boxplot di un gruppo con i singoli soggetti sovrapposti e la media (rombo)."""
    v = pd.Series(values).dropna().values
    if len(v) == 0:
        return 0
    ax.boxplot([v], positions=[0], widths=0.45, patch_artist=True, showmeans=True,
               boxprops={"facecolor": "#cde2fb", "edgecolor": color},
               medianprops={"color": color, "linewidth": 2},
               whiskerprops={"color": MUTED}, capprops={"color": MUTED}, showfliers=False,
               meanprops={"marker": "D", "markersize": 5, "markerfacecolor": INK, "markeredgecolor": "none"})
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(v))
    ax.scatter(jitter, v, color=INK2, s=14, zorder=3, alpha=0.8)
    ax.set_xticks([0]); ax.set_xticklabels([f"N = {len(v)}"]); ax.set_xlim(-0.6, 0.6)
    ax.set_ylabel(ylabel)
    return len(v)


def describe(values):
    """n, mean, sd, sem, median, min, max di una Series."""
    v = pd.Series(values).dropna().astype(float)
    n = len(v)
    row = {"n": n}
    if n:
        row.update(mean=v.mean(), median=v.median(), min=v.min(), max=v.max())
    if n > 1:
        row["sd"] = v.std(ddof=1); row["var"] = v.var(ddof=1); row["sem"] = row["sd"] / math.sqrt(n)
    return row
