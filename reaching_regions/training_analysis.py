"""Training a 9 regioni (reaching_regions.py), fatto due volte: pre e post
(con il secondo blind test in mezzo).

Input: i file di results_regions/ del repo reachy2_teleop_grasping_and_simulation,
incollati ovunque dentro questa cartella (es. reaching_regions/results/):
    <soggetto>_regions_<pre|post>[_k]_raw.csv       ogni frame: tempo, cursore, eventi del task
                                                    -> TUTTE le metriche cinematiche
    <soggetto>_regions_<pre|post>[_k]_summary.json  config del task (hidden_s, regioni, ...)
    <soggetto>_regions_<pre|post>[_k]_trials.csv    solo per controllo (target e tempi uguali al raw)
Se una sessione e' stata ripetuta (_1, _2, ...) si usa l'ultima ripetizione.
Si analizzano i goal verso i target (centro -> regione); i ritorni al centro
sono riportati a parte, solo come descrittive (il primo centro, che avvia la
sessione, non e' un ritorno ed e' escluso).

Metriche (common/reach_metrics.py, le definizioni degli script MATLAB del
laboratorio e di Bellitto et al. 2025): a 1 s dalla comparsa del target, cioe'
quando il cursore ricompare, euclidean / direction / extent error; su tutto il
movimento movement time e linearity index (le 5 della figura 4 del paper, le
principali), piu' aspect ratio, dimensionless jerk, picchi di velocita'.
Per la sessione intera: percorso, velocita', picchi, cambi di regione.

Analisi (vedi README.md):
1. figura 4 del paper: ogni metrica prova per prova (96 target del training 1,
   poi i 96 del training 2), media mobile sulla prova e le 11 precedenti; per
   ogni soggetto e per il gruppo (mediana +- SD tra soggetti);
2. blocchi di 8 target (ogni regione una volta): 12 per sessione, 24 in tutto.
   Per ogni soggetto e blocco un valore per metrica (mediana, o percentuale
   per le metriche si'/no) -> curva di apprendimento;
3. ANOVA a misure ripetute sessione (pre/post) x blocco (1-12), Greenhouse-Geisser,
   eta^2 parziale; post hoc: 3 t appaiati pianificati (Bonferroni sui 3, d di Cohen):
       pre: primo blocco vs ultimo blocco    quanto insegna il training 1
       ultimo blocco pre vs primo blocco post  si mantiene dopo il blind test?
       ultimo blocco pre vs ultimo blocco post cosa aggiunge il training 2
   con il controllo non parametrico (Wilcoxon, Holm) e il confronto esplorativo
   fra le due sessioni intere;
3b. modello lineare misto su tutte le prove (96 per sessione, senza blocchi):
   y ~ sessione * log(prova) + regione, soggetto casuale;
4. fit esponenziale y = a + b * exp(-k * (blocco - 1)) della curva di gruppo
   (e di ogni soggetto): a = plateau, k = velocita' di apprendimento, e il
   blocco in cui si arriva al 95% del miglioramento (<= 12: gia' nel training 1);
5. per regione (heatmap 3 x 3): quali direzioni sono piu' difficili.

Output in reaching_regions/statistics/:
    report.txt                    il riassunto da leggere (italiano)
    training_trials.csv           una riga per target: tutte le metriche di ogni prova
    training_subjects.csv         soggetto x sessione: un valore per metrica + sessione intera
    training_blocks.csv           soggetto x blocco (1-24): un valore per metrica
    training_statistics.csv       metrica x sessione: descrittive tra soggetti
    training_block_statistics.csv metrica x blocco: descrittive tra soggetti
    training_anova.csv            ANOVA sessione x blocco per metrica
    training_tests.csv            i 3 post hoc (t e Wilcoxon) + pre vs post per metrica
    training_mixed_models.csv     modello misto prova per prova (coefficienti, IC 95%, p)
    training_fits.csv             fit esponenziale (gruppo e soggetti)
    training_per_region.csv       metrica x regione x sessione: mediana tra soggetti
    training_homes.csv            ritorni al centro: descrittive per sessione
    plots/figure4_group.png, plots/figure4/<soggetto>.png   come la figura 4 del paper
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
from common.data import aggregate, find_sessions   # noqa: E402
from common.plots import (ANOVA_TAG, NONPARAM_TAG, STARS_NOTE, anova_caption, brackets, mean_se,   # noqa: E402
                          BLUE, CATEGORICAL, INK2, MUTED, grid_axes, heatmap3x3,
                          save_figure, setup_style, spaghetti, SUBJECT_LINE)
from common.reach_metrics import (blind_region_metrics, goals_from_raw, onset_metrics,   # noqa: E402
                                  reach_metrics, session_metrics)
from common.report import Report   # noqa: E402
from common.stats import (bonferroni, describe, fmt_num, holm, mixed_model, paired_t, rm_anova,   # noqa: E402
                          wilcoxon_paired)

BASE_DIR = Path(__file__).resolve().parent
PHASES = ("pre", "post")
BLOCK_SIZE = 8
PATTERN = r"(?P<subject>.+)_regions_(?P<phase>pre|post)(?:_(?P<rep>\d+))?_raw\.csv"
CANVAS_W, CANVAS_H, TARGET_RADIUS = 1200, 650, 70
# se il summary.json non ha il config: i valori del task (reaching_regions.py)
HIDDEN_S_DEFAULT, MIN_VISIT_S_DEFAULT, ONSET_SPEED_DEFAULT = 1.0, 0.25, 40.0
# figura 4: media mobile sulla prova e le 11 precedenti (come il paper), poi
# centro e dispersione tra soggetti (il paper usa media +- errore standard)
FIG4_WINDOW = 12
FIG4_CENTER, FIG4_SPREAD = "median", "sd"   # "mean" / "sem" per rifare esattamente il paper
FIG4_GAP = 12                               # prove di spazio fra training 1 e training 2 nel grafico
# modello misto: trasformazione delle metriche asimmetriche (sempre > 0 / >= 0)
MIXED_TRANSFORM = {"movement_time": "log", "linearity_index": "log1p"}
ANOVA_NAMES = {"phase": "session", "block": "block", "phase x block": "session × block"}

# metrica -> (etichetta, unita', meglio se, aggregazione, ruolo)
# ruolo: "main" = le 5 della figura 4 del paper, "extra" = richieste in piu', "" = esplorative
METRICS = {
    "euclidean_error_1s":       ("Euclidean error at 1 s", "px", "down", "median", "main"),
    "direction_error_1s":       ("Direction error at 1 s", "deg", "down", "median", "main"),
    "extent_error_1s":          ("Extent error at 1 s", "px", None, "median", "main"),
    "movement_time":            ("Movement time", "s", "down", "median", "main"),
    "linearity_index":          ("Linearity index", "", "down", "median", "main"),
    "region_at_reveal_correct": ("Correct region at reveal", "%", "up", "rate", "extra"),
    "reached_hidden":           ("Reached while hidden", "%", "up", "rate", "extra"),
    "n_speed_peaks":            ("Speed peaks", "", "down", "median", "extra"),
    "aspect_ratio":             ("Aspect ratio", "", "down", "median", ""),
    "dimensionless_jerk":       ("Dimensionless jerk", "", "down", "median", ""),
    "reaction_time":            ("Reaction time", "s", "down", "median", ""),
    "initial_direction_error":  ("Initial direction error (0.2 s after onset)", "deg", "down", "median", ""),
    "initial_direction_error_peak": ("Initial direction error (first speed peak)", "deg", "down", "median", ""),
    "first_region_correct":     ("First region correct", "%", "up", "rate", ""),
    "n_wrong_regions":          ("Wrong regions", "", "down", "mean", ""),
    "n_entries":                ("Entries before the dwell", "", "down", "mean", ""),
    "dwell_time":               ("Dwell time", "s", "down", "median", ""),
    "path_length":              ("Path length", "px", "down", "median", ""),
    "mean_speed":               ("Mean speed", "px/s", None, "median", ""),
    "sd_speed":                 ("SD of speed", "px/s", None, "median", ""),
    "peak_speed":               ("Peak speed", "px/s", None, "median", ""),
    "smoothness":               ("Smoothness (1 / speed peaks)", "", "up", "median", ""),
}
FOCUS = [k for k, v in METRICS.items() if v[4] in ("main", "extra")]
MAIN = [k for k, v in METRICS.items() if v[4] == "main"]
# sessione intera (tutti i frame del raw) -> (etichetta, unita', meglio se, script di origine)
SESSION_METRICS = {
    "session_log_duration":   ("Session duration (log)", "s", "down", "tt.m"),
    "session_path_length":    ("Cursor path length", "px", "down", "cursor_distance.m"),
    "session_mean_speed":     ("Mean cursor speed", "px/s", None, "speed1.m"),
    "session_sd_speed":       ("SD of cursor speed", "px/s", None, "speed1.m"),
    "session_peak_speed":     ("Peak cursor speed", "px/s", None, "speed1.m"),
    "session_mean_jerk":      ("Mean jerk", "px/s^3", "down", "jerk1.m"),
    "session_log10_dimensionless_jerk": ("Dimensionless jerk (segments, log10)", "", "down", "jerk1.m"),
    "session_speed_peaks":    ("Speed peaks", "", "down", "smooth.m"),
    "session_smoothness":     ("Smoothness (1 / speed peaks)", "", "up", "smooth.m"),
    "session_region_changes": ("Region changes", "", "down", "movement_train.m"),
}
HOME_METRICS = ["movement_time", "linearity_index", "n_speed_peaks", "n_entries", "dwell_time"]



def check_against_trials(goals, trials_path: Path, where: str, problems: list):
    """Controllo: i target trovati nel raw devono coincidere (numero, tempi) con il
    _trials.csv scritto dal task. Il _trials.csv non entra nei calcoli."""
    if not trials_path.exists():
        return
    tr = pd.read_csv(trials_path)
    tr = tr[tr["kind"] == "region"].sort_values("target_number")
    reg = [g for g in goals if g["kind"] == "region"]
    if len(reg) != len(tr):
        problems.append(f"{where}: {len(reg)} target nel raw, {len(tr)} nel trials.csv")
        return
    if reg:
        off = reg[0]["t_shown"] - tr["t_shown"].iloc[0]
        bad = sum(abs(g["t_shown"] - off - a) > 0.01 or abs(g["t_enter"] - off - b) > 0.01
                  for g, a, b in zip(reg, tr["t_shown"], tr["t_enter"]))
        if bad:
            problems.append(f"{where}: {bad} target con tempi diversi fra raw e trials.csv")


def load_session(subject, phase, path: Path, problems: list):
    """Tutti i goal di una sessione e le metriche della sessione intera, dal _raw.csv."""
    stem = str(path)[:-len("_raw.csv")]
    raw = pd.read_csv(path)
    summary_path = Path(stem + "_summary.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    cfg = summary.get("config", {})
    hidden_s = float(cfg.get("hidden_s", HIDDEN_S_DEFAULT))
    x_edges = cfg.get("region_x_edges", [0, CANVAS_W / 3, 2 * CANVAS_W / 3, CANVAS_W])
    y_edges = cfg.get("region_y_edges", [0, CANVAS_H / 3, 2 * CANVAS_H / 3, CANVAS_H])
    min_visit_s = float(cfg.get("min_visit_s", MIN_VISIT_S_DEFAULT))
    onset_speed = float(cfg.get("motion_onset_speed", ONSET_SPEED_DEFAULT))
    fs = 1.0 / float(raw["time"].diff().median())

    goals = goals_from_raw(raw, x_edges, y_edges)
    check_against_trials(goals, Path(stem + "_trials.csv"), f"{subject} {phase}", problems)
    rows, n_target = [], 0
    for i, g in enumerate(goals):
        row = {"subject": subject, "phase": phase, "kind": g["kind"], "goal_index": i + 1,
               "region": g["target_region"], "n_entries": g["n_entries"], "dwell_time": g["dwell_time"]}
        if g["kind"] == "region":
            n_target += 1
            row["target_number"] = n_target
        row.update(reach_metrics(g["t"], g["xy"], g["goal"], g["t_enter"], hidden_s, fs))
        row.update(onset_metrics(g["t"], g["xy"], g["goal"], onset_speed, fs))
        if g["kind"] == "region":
            row.update(blind_region_metrics(g, hidden_s, min_visit_s, x_edges, y_edges))
        row["_x"], row["_y"] = g["xy"][:, 0], g["xy"][:, 1]
        rows.append(row)
    session = session_metrics(raw["time"], raw[["cursor_x", "cursor_y"]], raw["region"], fs)
    session["session_log10_dimensionless_jerk"] = np.log10(session["session_dimensionless_jerk"])
    session["fs"] = fs
    session["hidden_s"] = hidden_s
    return pd.DataFrame(rows), session


def load(root: Path) -> tuple:
    chosen, dropped = find_sessions(root, PATTERN, ("phase",))
    frames, sessions, problems = [], {}, []
    for (subject, phase), path in sorted(chosen.items()):
        df, session = load_session(subject, phase, path, problems)
        frames.append(df)
        sessions[(subject, phase)] = session
    goals = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not goals.empty:
        goals["block"] = (goals["target_number"] - 1) // BLOCK_SIZE + 1
    return goals, sessions, dropped, problems


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
    # k al limite inferiore = curva quasi lineare: plateau e blocco al 95% non hanno senso
    span = y.max() - y.min()
    if p[2] < 1e-3 or not (y.min() - span <= p[0] <= y.max() + span):
        return out
    res = y - expo(x, *p)
    ss_tot = np.sum((y - y.mean()) ** 2)
    out.update(a=float(p[0]), b=float(p[1]), k=float(p[2]),
               r2=float(1 - np.sum(res ** 2) / ss_tot) if ss_tot > 0 else np.nan,
               block95=float(1 + math.log(20) / p[2]))
    return out


def test_row(m, label, role, comparison, meaning, planned, t, p_corr, test, effect_name, da, db, better):
    return {"metric": m, "label": label, "role": role, "comparison": comparison, "meaning": meaning,
            "planned": planned, "test": test, "n": t["n"], "statistic": t["statistic"], "p": t["p"],
            "p_corrected": p_corr, "correction": ("Bonferroni" if test == "t" else "Holm") if planned else "",
            "effect": t["effect"], "effect_name": effect_name, "shapiro_p": t.get("shapiro_p", np.nan),
            "median_first": da.get("median"), "median_second": db.get("median"), "better": better}


def analyse(sessions, blocks, n_blocks, report):
    stat_rows, test_rows, block_rows, fit_rows, anova_rows = [], [], [], [], []
    last = n_blocks
    contrasts = (("pre", 1, "pre", last, "training 1: first vs last block"),
                 ("pre", last, "post", 1, "retention: last pre vs first post block"),
                 ("pre", last, "post", last, "training 2: last pre vs last post block"))
    long = blocks.assign(phase=np.where(blocks["gblock"] > n_blocks, "post", "pre"),
                         block=(blocks["gblock"] - 1) % n_blocks + 1)
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

        # ANOVA a misure ripetute: sessione (pre/post) x blocco (1-12)
        anova = rm_anova(long, m, ["phase", "block"])
        for r in anova:
            anova_rows.append({"metric": m, "label": label, "role": role, **r})
        # post hoc pianificati: t appaiati (Bonferroni sui 3) + controllo Wilcoxon (Holm sui 3)
        t_tests = [paired_t(col(a, ba), col(b, bb)) for a, ba, b, bb, _ in contrasts]
        p_bonf = bonferroni([t["p"] for t in t_tests])
        w_tests = [wilcoxon_paired(col(a, ba), col(b, bb)) for a, ba, b, bb, _ in contrasts]
        p_holm = holm([t["p"] for t in w_tests])
        descr = []
        for (a, ba, b, bb, what), tt, pb, wt, ph in zip(contrasts, t_tests, p_bonf, w_tests, p_holm):
            da, db = describe(col(a, ba)), describe(col(b, bb))
            descr.append((da, db))
            comp = f"{a} block {ba} vs {b} block {bb}"
            test_rows.append(test_row(m, label, role, comp, what, True, tt, pb, "t", "Cohen dz", da, db, better))
            test_rows.append(test_row(m, label, role, comp, what, True, wt, ph, "Wilcoxon", "rank-biserial r",
                                      da, db, better))
        t_sess, w_sess = paired_t(ws["pre"], ws["post"]), wilcoxon_paired(ws["pre"], ws["post"])
        for tt, name, eff in ((t_sess, "t", "Cohen dz"), (w_sess, "Wilcoxon", "rank-biserial r")):
            test_rows.append(test_row(m, label, role, "session pre vs post", "whole sessions (exploratory)", False,
                                      tt, np.nan, name, eff, d_sess["pre"], d_sess["post"], better))
        # fit: curva di gruppo (mediana per blocco) e di ogni soggetto
        med = wb.median()
        fg = fit_curve(med.index.to_numpy(), med.to_numpy())
        fit_rows.append({"metric": m, "subject": "GROUP (median curve)", **fg})
        for subj, row in wb.iterrows():
            fit_rows.append({"metric": m, "subject": subj, **fit_curve(row.index.to_numpy(), row.to_numpy())})

        if role in ("main", "extra"):
            report.section(f"{label}{f' [{unit}]' if unit else ''}" + ("  (metrica principale)" if role == "main" else ""))
            report.anova(anova, "ANOVA sessione x blocco")
            for (a, ba, b, bb, what), tt, pb, ph, (da, db) in zip(contrasts, t_tests, p_bonf, p_holm, descr):
                report.ttest(what, "", better, da, db, tt, pb, f"{a} b{ba}", f"{b} b{bb}", np_check=ph)
            report.text(f"  sessione intera (esplorativo): pre {fmt_num(d_sess['pre'].get('median'))} -> post "
                        f"{fmt_num(d_sess['post'].get('median'))}, t p={fmt_num(t_sess['p'], 3)}, "
                        f"d={fmt_num(t_sess['effect'])}; Wilcoxon p={fmt_num(w_sess['p'], 3)}")
            if not np.isnan(fg["k"]):
                where = "gia' nel training 1" if fg["block95"] <= n_blocks else \
                    ("nel training 2" if fg["block95"] <= 2 * n_blocks else "non ancora raggiunto a fine training 2")
                report.text(f"  curva di gruppo: plateau stimato {fg['a']:.2f}, 95% del miglioramento al blocco "
                            f"{fg['block95']:.1f} ({where}); R2 del fit {fg['r2']:.2f}")
            else:
                report.text("  curva di gruppo: fit esponenziale non riuscito (curva piatta o troppo rumorosa)")
            if better is None and m == "extent_error_1s":
                report.text("  (< 0 = il cursore resta corto rispetto al target; meglio = piu' vicino a 0)")

    report.section("Sessione intera (tutti i frame, script del laboratorio; esplorativo)")
    for m, (label, unit, better, script) in SESSION_METRICS.items():
        if m not in sessions.columns:
            continue
        ws = sessions.pivot(index="subject", columns="phase", values=m).reindex(columns=list(PHASES))
        d = {ph: describe(ws[ph]) for ph in PHASES}
        for ph in PHASES:
            stat_rows.append({"metric": m, "label": label, "unit": unit, "role": "session", "phase": ph, **d[ph]})
        tt, wt = paired_t(ws["pre"], ws["post"]), wilcoxon_paired(ws["pre"], ws["post"])
        for t, name, eff in ((tt, "t", "Cohen dz"), (wt, "Wilcoxon", "rank-biserial r")):
            test_rows.append(test_row(m, label, "session", "session pre vs post",
                                      f"whole session ({script}, exploratory)", False, t, np.nan, name, eff,
                                      d["pre"], d["post"], better))
        report.ttest(f"{label} ({script})", unit, better, d["pre"], d["post"], tt, None, "pre", "post",
                     digits=0 if unit in ("s", "px", "px/s", "px/s^3") or "changes" in m or "peaks" in m else 3,
                     np_check=wt["p"])
    return (pd.DataFrame(stat_rows), pd.DataFrame(test_rows), pd.DataFrame(block_rows), pd.DataFrame(fit_rows),
            pd.DataFrame(anova_rows))


# ---------------------------------------------------- modello misto, prova per prova ---

def mixed_models(targets: pd.DataFrame, report) -> pd.DataFrame:
    """Per ogni metrica principale, tutte le prove (96 per sessione, niente blocchi):
        y ~ sessione * log(prova) + regione,  soggetto casuale (intercetta + pendenza di log(prova))
    log(prova): la curva di apprendimento (veloce all'inizio, poi piatta); la regione toglie
    la differenza di difficolta' fra i target. Movement time e linearity index sono
    asimmetrici: si modella log(y) e log(1 + y) e l'effetto si legge in %."""
    report.section("Modello misto, prova per prova (96 prove per sessione, senza blocchi)")
    report.text("  y ~ sessione * log(prova) + regione, soggetto casuale. log(prova): 'per ogni raddoppio")
    report.text("  del numero di prova' = coefficiente x ln 2. Movement time e linearity index in scala log (%).")
    rows = []
    d = targets.assign(log_trial=np.log(targets["target_number"]),
                       phase=pd.Categorical(targets["phase"], categories=list(PHASES)))
    for m in MAIN:
        label, unit = METRICS[m][0], METRICS[m][1]
        tr = MIXED_TRANSFORM.get(m)
        y = pd.to_numeric(d[m], errors="coerce")
        y = np.log(y) if tr == "log" else np.log1p(y) if tr == "log1p" else y
        dm = d.assign(y=y).dropna(subset=["y"])
        dm = dm[np.isfinite(dm["y"])]
        res = mixed_model(dm, "y ~ C(phase) * log_trial + C(region)", slope="log_trial")
        terms = res["terms"]
        for term, r in terms.iterrows():
            rows.append({"metric": m, "label": label, "transform": tr or "", "term": term, "random_effects": res["re"],
                         "converged": res["converged"], "n_obs": res["n_obs"], "n_subjects": res["n_groups"], **r})
        if terms.empty:
            report.text(f"  {label}: modello non stimabile")
            continue

        def effect(term, per_doubling=False):
            if term not in terms.index:
                return "-"
            r = terms.loc[term]
            k = math.log(2) if per_doubling else 1.0
            c, lo, hi = r["coef"] * k, r["ci_low"] * k, r["ci_high"] * k
            if tr:
                c, lo, hi = [(math.exp(v) - 1) * 100 for v in (c, lo, hi)]
                txt = f"{c:+.1f}% [{lo:+.1f}, {hi:+.1f}]"
            else:
                txt = f"{c:+.2f} [{lo:+.2f}, {hi:+.2f}]{f' {unit}' if unit else ''}"
            return f"{txt}, p={fmt_num(r['p'], 3)}"

        report.text(f"  {label}{' (' + ('log' if tr == 'log' else 'log(1 + y)') + ')' if tr else ''}: "
                    f"apprendimento nel training 1 (per raddoppio della prova) {effect('log_trial', True)} | "
                    f"training 2 vs 1 all'inizio {effect('C(phase)[T.post]')} | "
                    f"differenza di apprendimento nel training 2 {effect('C(phase)[T.post]:log_trial', True)}"
                    + f" | effetti casuali: {res['re']}"
                    + ("" if res["converged"] else " | ATTENZIONE: nessun modello stimabile senza problemi"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figura 4 ---

def moving_average(targets: pd.DataFrame, m: str) -> pd.DataFrame:
    """soggetto x prova (1..96 pre, poi 97..192 post): media mobile della prova e
    delle FIG4_WINDOW-1 precedenti, ricominciando all'inizio di ogni training."""
    parts = []
    for (subject, phase), g in targets.groupby(["subject", "phase"]):
        s = g.sort_values("target_number").set_index("target_number")[m].astype(float)
        s = s.rolling(FIG4_WINDOW, min_periods=1).mean()
        parts.append(pd.DataFrame({"subject": subject, "phase": phase, "trial": s.index, m: s.to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def fig4_x(phase, trial, n_trials):
    """Posizione sull'asse x: training 2 spostato di FIG4_GAP prove dopo il training 1."""
    return trial + (n_trials + FIG4_GAP if phase == "post" else 0)


def fig4_axes(n_trials, title, hspace=0.45):
    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(2, 3, hspace=hspace, wspace=0.28)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return fig, axes, fig.add_subplot(gs[1, 2])


def fig4_clip(ax, values):
    """Asse y sul 1-99 percentile delle prove singole, cosi' le poche prove fuori
    scala (es. un movement time di 19 s) non schiacciano le curve."""
    lo, hi = np.nanpercentile(np.asarray(values, float), [1, 99])
    pad = 0.08 * (hi - lo if hi > lo else 1)
    ax.set_ylim(lo - pad, hi + pad)


def fig4_decorate(ax, m, n_trials, letter):
    label, unit = METRICS[m][0], METRICS[m][1]
    x_post = n_trials + FIG4_GAP
    ax.axvspan(x_post + 0.5, x_post + n_trials + 0.5, color="#f1f0ec", zorder=0, lw=0)
    ticks = [1, n_trials // 2, n_trials]
    ax.set_xticks(ticks + [t + x_post for t in ticks])
    ax.set_xticklabels([str(t) for t in ticks] + [str(t + n_trials) for t in ticks], fontsize=8)
    ax.set_xlim(0, 2 * n_trials + FIG4_GAP + 1)
    if m == "extent_error_1s":
        ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("trial (target)")
    ax.set_ylabel(f"{label}{f' [{unit}]' if unit else ''}")
    ax.set_title(f"({letter}) {label}", loc="left", fontsize=10)


def fig4_phase_labels(ax, n_trials):
    """'training 1 / 2' in alto, alla quota attuale dei dati (le parentesi dei post hoc,
    aggiunte dopo, vanno sopra e non ci finiscono addosso)."""
    x_post = n_trials + FIG4_GAP
    top = ax.get_ylim()[1]
    for xc, txt in ((n_trials / 2, "training 1 (pre)"), (x_post + n_trials / 2, "training 2 (post)")):
        ax.text(xc, top, txt, ha="center", va="top", fontsize=7.5, color=MUTED, zorder=5)


def fig4_legend(ax, handles, text):
    ax.axis("off")
    ax.legend(handles=handles, loc="upper left", fontsize=8.5)
    ax.text(0.0, 0.35, text, transform=ax.transAxes, fontsize=8, color=INK2, va="top", wrap=True)


def plot_figure4(targets, out, tests, anova):
    n_trials = int(targets["target_number"].max())
    smoothed = {m: moving_average(targets, m) for m in MAIN}
    note = (f"Moving average over each trial and the {FIG4_WINDOW - 1} preceding ones\n"
            "(restarted at the start of each training).\n"
            f"Errors at 1 s = when the cursor reappears.\nUnits: canvas pixels (1200 x 650).")

    # un soggetto per figura: ogni prova (punti) + media mobile (linea)
    for subject, g in targets.groupby("subject"):
        fig, axes, lax = fig4_axes(n_trials, f"{subject} – cursor control during the two trainings")
        for ax, m, letter in zip(axes, MAIN, "ABCDE"):
            all_x, all_v, all_c = [], [], []
            for ph, color in zip(PHASES, (CATEGORICAL[1], CATEGORICAL[0])):
                gp = g[g.phase == ph].sort_values("target_number")
                if gp.empty:
                    continue
                x = fig4_x(ph, gp["target_number"].to_numpy(), n_trials)
                ax.plot(x, gp[m], "o", ms=2.2, color=color, alpha=0.35, zorder=2)
                all_x.append(x)
                all_v.append(gp[m].to_numpy(float))
                all_c.append(np.full(len(x), color))
                sm = smoothed[m]
                sm = sm[(sm.subject == subject) & (sm.phase == ph)]
                ax.plot(fig4_x(ph, sm["trial"].to_numpy(), n_trials), sm[m], color=color, lw=1.8, zorder=3)
            fig4_decorate(ax, m, n_trials, letter)
            if all_v:
                v, x, c = np.concatenate(all_v), np.concatenate(all_x), np.concatenate(all_c)
                fig4_clip(ax, v)
            fig4_phase_labels(ax, n_trials)
            if all_v:
                for col in set(c):
                    lo, hi = ax.get_ylim()
                    sel = c == col
                    for out_sel, y, mk in ((v[sel] > hi, hi, "^"), (v[sel] < lo, lo, "v")):
                        if out_sel.any():
                            ax.plot(x[sel][out_sel], np.full(out_sel.sum(), y), mk, ms=4, color=col,
                                    clip_on=False, zorder=4)
        handles = [plt.Line2D([], [], color=INK2, marker="o", ls="", ms=3, alpha=0.5, label="single trial"),
                   plt.Line2D([], [], color=INK2, marker="^", ls="", ms=4, label="trial off the axis (1-99th pct.)"),
                   plt.Line2D([], [], color=INK2, lw=1.8, label=f"moving average ({FIG4_WINDOW} trials)")]
        fig4_legend(lax, handles, note)
        save_figure(fig, out, f"figure4/{subject}")

    # gruppo, due versioni:
    #  figure4_group.png        [NON PARAMETRICO] mediana +- SD (FIG4_CENTER, FIG4_SPREAD)
    #  figure4_group_anova.png  [ANOVA] media +- SE, parentesi sui post hoc significativi
    plot_figure4_group(targets, smoothed, out, FIG4_CENTER, FIG4_SPREAD, "figure4_group",
                       "[NON-PARAMETRIC version] no test on the figure; tests in report: Wilcoxon (Holm)", note)
    plot_figure4_group(targets, smoothed, out, "mean", "sem", "figure4_group_anova", ANOVA_TAG, note, tests, anova)


def contrast_pairs(tests, m, xpos):
    """[(x1, x2, p Bonferroni, differenze non normali)] dei 3 post hoc (t) di una metrica;
    xpos(fase, blocco) -> posizione sull'asse x."""
    t = tests[(tests.metric == m) & (tests.test == "t") & tests.planned.astype(bool)]
    pairs = []
    for _, r in t.iterrows():
        pa, _, na, pb, _, nb = r["comparison"].replace(" vs ", " ").split()
        pairs.append((xpos(pa, int(na)), xpos(pb, int(nb)), r["p_corrected"], r["shapiro_p"] < 0.05))
    return pairs


def plot_figure4_group(targets, smoothed, out, center, spread, name, tag, note, tests=None, anova=None):
    n_trials = int(targets["target_number"].max())
    fig, axes, lax = fig4_axes(n_trials, "Cursor control during the two trainings – "
                                         f"{center} ± {'SE' if spread == 'sem' else spread.upper()} across subjects\n{tag}",
                               hspace=0.75 if anova is not None else 0.45)
    for ax, m, letter in zip(axes, MAIN, "ABCDE"):
        sm = smoothed[m]
        for ph, color in zip(PHASES, (CATEGORICAL[1], CATEGORICAL[0])):
            wide = sm[sm.phase == ph].pivot(index="trial", columns="subject", values=m)
            if wide.empty:
                continue
            x = fig4_x(ph, wide.index.to_numpy(), n_trials)
            for s_ in wide.columns:
                ax.plot(x, wide[s_], color=SUBJECT_LINE, lw=0.6, zorder=1)
            mid = wide.median(axis=1) if center == "median" else wide.mean(axis=1)
            sp = wide.std(axis=1, ddof=1)
            if spread == "sem":
                sp = sp / np.sqrt(wide.notna().sum(axis=1))
            ax.fill_between(x, mid - sp, mid + sp, color=color, alpha=0.2, lw=0, zorder=2)
            ax.plot(x, mid, color=color, lw=2, zorder=3)
        fig4_decorate(ax, m, n_trials, letter)
        fig4_phase_labels(ax, n_trials)
        if tests is not None:
            # blocco b = prove 8(b-1)+1 ... 8b: la parentesi va al centro del blocco
            brackets(ax, contrast_pairs(tests, m, lambda ph, b: fig4_x(ph, (b - 1) * BLOCK_SIZE + 4.5, n_trials)))
        if anova is not None:
            ax.text(0.0, -0.24, anova_caption(anova[anova.metric == m].to_dict("records"), ANOVA_NAMES),
                    transform=ax.transAxes, fontsize=7, color=INK2, va="top")
    n = targets["subject"].nunique()
    lab = f"{center} ± {'SE' if spread == 'sem' else spread.upper()}"
    handles = [plt.Line2D([], [], color=CATEGORICAL[1], lw=2, label=f"training 1 (pre): {lab}"),
               plt.Line2D([], [], color=CATEGORICAL[0], lw=2, label=f"training 2 (post): {lab}"),
               plt.Line2D([], [], color=SUBJECT_LINE, lw=1, label=f"single subjects (n = {n})")]
    extra = ("\nBrackets: planned post hoc between blocks of 8\n(first vs last block of training 1, last pre vs\n"
             "first post, last pre vs last post).\n" + STARS_NOTE.replace("; ", ";\n")) if tests is not None else ""
    fig4_legend(lax, handles, note + extra)
    save_figure(fig, out, name)


# ----------------------------------------------------------- altri grafici ---

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


def plot_curve_anova(ax, blocks, m, n_blocks, tests, anova):
    """Versione ANOVA della curva per blocchi: media +- SE di ogni blocco (training 1 e 2),
    parentesi sui 3 post hoc significativi, risultato dell'ANOVA sotto il grafico."""
    label, unit, better, _, _ = METRICS[m]
    wb = blocks.pivot(index="subject", columns="gblock", values=m)
    for ph, color in zip(PHASES, (CATEGORICAL[1], CATEGORICAL[0])):
        cols = [c for c in wb.columns if (c > n_blocks) == (ph == "post")]
        if cols:
            mean_se(ax, wb[cols], x=cols, color=color, label=f"training {'1 (pre)' if ph == 'pre' else '2 (post)'}")
    ax.axvline(n_blocks + 0.5, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("block of 8 targets (1-12 training 1, 13-24 training 2)")
    ax.set_ylabel(f"{label}{f' [{unit}]' if unit else ''}")
    ax.set_title(label + ("  (higher = better)" if better == "up" else "  (lower = better)" if better == "down" else ""),
                 loc="left", pad=16)
    brackets(ax, contrast_pairs(tests, m, lambda ph, b: b + (n_blocks if ph == "post" else 0)))
    ax.text(1.0, -0.2, anova_caption(anova[anova.metric == m].to_dict("records"), ANOVA_NAMES),
            transform=ax.transAxes, fontsize=7, color=INK2, va="top", ha="right")


def plot_mixed_model(targets, mixed, out):
    """Modello misto: punti = media tra soggetti di ogni prova (nella scala del modello,
    riportata indietro: per movement time e linearity index e' la media geometrica),
    linea = curva stimata dal modello (effetti fissi, media sulle regioni)."""
    if mixed.empty:
        return
    n_trials = int(targets["target_number"].max())
    fig, axes, lax = fig4_axes(n_trials, "Trial-by-trial learning – linear mixed model "
                                         "(y ~ session × log(trial) + region, subject random)\n[ANOVA version – trial by trial, no blocks] "
                                         "points = mean across subjects per trial, line = model fit",
                               hspace=0.75)
    trials = np.arange(1, n_trials + 1)
    for ax, m, letter in zip(axes, MAIN, "ABCDE"):
        tr = MIXED_TRANSFORM.get(m)
        fwd = np.log if tr == "log" else np.log1p if tr == "log1p" else (lambda v: v)
        back = np.exp if tr == "log" else np.expm1 if tr == "log1p" else (lambda v: v)
        terms = mixed[mixed.metric == m].set_index("term")
        if terms.empty:
            continue
        c = terms["coef"]
        region_mean = c[[t for t in c.index if t.startswith("C(region)")]].sum() / max(targets["region"].nunique(), 1)
        for ph, color in zip(PHASES, (CATEGORICAL[1], CATEGORICAL[0])):
            g = targets[targets.phase == ph]
            y = pd.to_numeric(g[m], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                g = g.assign(_y=fwd(y))
            g = g[np.isfinite(g["_y"])]
            pts = back(g.groupby("target_number")["_y"].mean())
            ax.plot(fig4_x(ph, pts.index.to_numpy(), n_trials), pts.to_numpy(), "o", ms=2.2, color=color,
                    alpha=0.45, zorder=2)
            post = ph == "post"
            eta = (c.get("Intercept", 0) + region_mean + post * c.get("C(phase)[T.post]", 0)
                   + np.log(trials) * (c.get("log_trial", 0) + post * c.get("C(phase)[T.post]:log_trial", 0)))
            ax.plot(fig4_x(ph, trials, n_trials), back(eta), color=color, lw=2.2, zorder=3)
        fig4_decorate(ax, m, n_trials, letter)
        fig4_phase_labels(ax, n_trials)
        r = terms.iloc[0]
        re_en = (str(r["random_effects"]).replace("solo intercetta", "intercept only")
                 .replace("intercetta", "intercept").replace("pendenza di", "slope of")
                 .replace("(correlate)", "(correlated)").replace("(indipendenti)", "(independent)"))
        ax.text(0.0, -0.24, f"random effects per subject: {re_en}" + ("" if r["converged"] else " (check!)"),
                transform=ax.transAxes, fontsize=6.8, color=INK2, va="top")
    handles = [plt.Line2D([], [], color=INK2, marker="o", ls="", ms=3, alpha=0.6, label="mean across subjects, per trial"),
               plt.Line2D([], [], color=INK2, lw=2.2, label="mixed-model fit (average region)")]
    fig4_legend(lax, handles, "Every trial, no blocks. Movement time and\nlinearity index modelled on a log scale.\n"
                              "Coefficients, 95% CI and p in\ntraining_mixed_models.csv and in report.txt.")
    save_figure(fig, out, "training_mixed_model")


def plots(targets, sessions, blocks, fits, per_region, n_blocks, out, tests, anova, mixed):
    setup_style()
    plot_figure4(targets, out, tests, anova)
    plot_mixed_model(targets, mixed, out)

    fig, axes = grid_axes(len(FOCUS), ncols=2, width=6.2, height=4.8)
    for ax, m in zip(axes, FOCUS):
        plot_curve_anova(ax, blocks, m, n_blocks, tests, anova)
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Reaching training – learning curves by block (grey = single subjects)\n" + ANOVA_TAG,
                 x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.003, STARS_NOTE, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    save_figure(fig, out, "training_learning_curves_anova")

    fig, axes = grid_axes(len(FOCUS), ncols=2, width=6.2, height=3.6)
    for ax, m in zip(axes, FOCUS):
        plot_curve(ax, blocks, fits, m, n_blocks)
    axes[0].legend(fontsize=7.5, loc="best")
    fig.suptitle("Reaching training – learning curves (one line per subject)\n" + NONPARAM_TAG
                 + "; dashed = exponential fit", x=0.01, ha="left", fontweight="bold")
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

    fig, axes = plt.subplots(len(MAIN), 2, figsize=(5.4, 2.6 * len(MAIN)), squeeze=False)
    for i, m in enumerate(MAIN):
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
            for _, r in g[g.phase == ph].iterrows():
                c = CATEGORICAL[(int(r["region"]) - 1) % len(CATEGORICAL)]
                ax.plot(r["_x"], r["_y"], color=c, lw=0.5, alpha=0.5)
        fig.suptitle(f"{subject} – cursor paths to the targets (colour = region)", x=0.01, ha="left",
                     fontweight="bold", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        save_figure(fig, out, f"trajectories/{subject}")


def main():
    p = argparse.ArgumentParser(description="Analisi dei due training a 9 regioni")
    p.add_argument("--results", type=Path, default=BASE_DIR, help="cartella con i file di results_regions/")
    p.add_argument("--out", type=Path, default=BASE_DIR / "statistics")
    args = p.parse_args()

    goals, session_info, dropped, problems = load(args.results)
    if goals.empty:
        print(f"Nessun file *_regions_<pre|post>_raw.csv in {args.results}")
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
    for msg in problems:
        report.text("ATTENZIONE: " + msg)
    hidden = sorted({round(v["hidden_s"], 3) for v in session_info.values()})
    fs = [v["fs"] for v in session_info.values()]
    report.text(f"Metriche calcolate dai _raw.csv (webcam {min(fs):.1f}-{max(fs):.1f} Hz); cursore nascosto "
                f"per {', '.join(map(str, hidden))} s dopo la comparsa del target: gli errori sono calcolati li'.")
    report.text(f"Blocchi di {BLOCK_SIZE} target: {n_blocks} per sessione, {2 * n_blocks} in tutto.")
    report.text("Analisi principale: ANOVA a misure ripetute sessione (2) x blocco (12), correzione di")
    report.text("Greenhouse-Geisser; post hoc: 3 t appaiati pianificati fra blocchi, Bonferroni sui 3.")
    report.text("Controllo non parametrico: Wilcoxon, Holm sui 3. In fondo il modello misto prova per prova.")

    sessions = per_unit(targets, ["subject", "phase"])
    for m in list(SESSION_METRICS) + ["session_dimensionless_jerk", "fs"]:
        sessions[m] = [session_info.get((s, ph), {}).get(m) for s, ph in zip(sessions.subject, sessions.phase)]
    blocks = per_unit(targets, ["subject", "gblock"])
    stats_df, tests_df, block_stats, fits, anova_df = analyse(sessions, blocks, n_blocks, report)
    mixed_df = mixed_models(targets, report)

    per_region = []
    reg_units = per_unit(targets, ["subject", "phase", "region"])
    for m in METRICS:
        if m in reg_units.columns:
            for (ph, reg), g in reg_units.groupby(["phase", "region"]):
                per_region.append({"metric": m, "phase": ph, "region": reg, **describe(g[m])})
    per_region = pd.DataFrame(per_region)
    # the first home is the start of the session (the timer starts there), not a return: excluded
    first_home = goals["goal_index"] == 1
    homes = per_unit(goals[(goals["kind"] == "home") & ~first_home], ["subject", "phase"], HOME_METRICS)
    home_rows = [{"metric": m, "phase": ph, **describe(g[m])} for m in HOME_METRICS if m in homes.columns
                 for ph, g in homes.groupby("phase")]

    report.section("Metriche esplorative")
    report.text("  Tutte le altre metriche sono in training_tests.csv e plots/metrics/, calcolate allo stesso modo.")
    report.text("  I ritorni al centro sono in training_homes.csv.")
    report.text("  plots/figure4/<soggetto>.png: ogni soggetto prova per prova; plots/figure4_group.png: il gruppo.")

    targets.drop(columns=["_x", "_y"]).to_csv(args.out / "training_trials.csv", index=False)
    sessions.to_csv(args.out / "training_subjects.csv", index=False)
    blocks.to_csv(args.out / "training_blocks.csv", index=False)
    stats_df.to_csv(args.out / "training_statistics.csv", index=False)
    block_stats.to_csv(args.out / "training_block_statistics.csv", index=False)
    tests_df.to_csv(args.out / "training_tests.csv", index=False)
    anova_df.to_csv(args.out / "training_anova.csv", index=False)
    mixed_df.to_csv(args.out / "training_mixed_models.csv", index=False)
    fits.to_csv(args.out / "training_fits.csv", index=False)
    per_region.to_csv(args.out / "training_per_region.csv", index=False)
    pd.DataFrame(home_rows).to_csv(args.out / "training_homes.csv", index=False)
    plots(targets, sessions, blocks, fits, per_region, n_blocks, args.out, tests_df, anova_df, mixed_df)
    report.write(args.out)


if __name__ == "__main__":
    main()
