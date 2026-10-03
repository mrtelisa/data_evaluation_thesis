"""Tabelle statistiche complete per l'appendice della tesi, dai .csv scritti dalle analisi
(reaching_regions, reaching_blind, real_robot). Solo le metriche usate nella tesi.

Uscita: Tesi_latex/Appendix1/stats_tables.tex (incluso dall'appendice con \\input), riscritto a ogni
esecuzione, cosi' le tabelle si aggiornano insieme alle analisi. Va lanciato dopo le analisi
(run_all.py lo fa per ultimo).

    python3 thesis_tables.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[1] / "Tesi_latex" / "Appendix1" / "stats_tables.tex"

TRAINING = {"euclidean_error": "Euclidean error", "direction_error": "Direction error",
            "extent_error": "Extent error", "movement_duration": "Movement duration",
            "linearity_index": "Linearity index"}
BLIND = {"hit": "Hit rate", "euclidean_error": "Euclidean error", "extent_error": "Extent error",
         "direction_error": "Direction error"}
ROBOT = [("completion_time", "andata", "Completion time", "Outbound"),
         ("completion_time", "ritorno", "", "Return"),
         ("completion_time", "andata_ritorno", "", "Outbound + return"),
         ("completion_time", "totale", "", "Total"),
         ("completion_time", "grasp", "Grasping time", ""),
         ("completion_time", "posizionamento", "Placing time", ""),
         ("path_length", "andata", "Path length", "Outbound"), ("path_length", "ritorno", "", "Return"),
         ("mean_speed", "andata", "Mean speed", "Outbound"), ("mean_speed", "ritorno", "", "Return"),
         ("jerk", "andata", "Jerk", "Outbound"), ("jerk", "ritorno", "", "Return"),
         ("region_transitions", "andata", "Region transitions", "Outbound"),
         ("region_transitions", "ritorno", "", "Return")]
EXPLORATORY_ROBOT = [("stop_percent", "Stop time while driving"),
                     ("transitions_per_min", "Region transitions per minute"),
                     ("cursor_eccentricity", "Cursor distance from the center")]
LEG = {"andata": "Outbound", "ritorno": "Return"}


def p_fmt(p):
    return "--" if pd.isna(p) else ("$<0.001$" if p < 0.001 else f"${p:.3f}$")


def num(x, d=2):
    return "--" if pd.isna(x) else f"${x:.{d}f}$"


def table(caption, short, label, head, rows, cols):
    body = "\n".join("        " + " & ".join(r) + r" \\" for r in rows)
    return rf"""
\begin{{table}}[htbp]
    \centering
    \small
    \setlength{{\tabcolsep}}{{4pt}}
    \resizebox{{\ifdim\width>\linewidth\linewidth\else\width\fi}}{{!}}{{%   % rimpicciolisce solo se troppo larga
    \begin{{tabular}}{{{cols}}}
        \toprule
        {head} \\
        \midrule
{body}
        \bottomrule
    \end{{tabular}}%
    }}
    \caption[{short}]{{{caption}}}
    \label{{{label}}}
\end{{table}}
"""


def anova_rows(anova, names, effects):
    rows = []
    for m, name in names.items():
        for k, eff in enumerate(effects):
            r = anova[(anova.metric == m) & (anova.effect == eff)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([name if k == 0 else "", eff.replace("×", r"$\times$"),
                         f"$F({r.df1_gg:.2f}, {r.df2_gg:.2f}) = {r.F:.2f}$", p_fmt(r.p_gg),
                         num(r.partial_eta2), num(r.eps_gg)])
    return rows


def posthoc_rows(ph, names, label_of=lambda c: c):
    rows = []
    for m, name in names.items():
        for k, (_, r) in enumerate(ph[ph.metric == m].iterrows()):
            rows.append([name if k == 0 else "", label_of(r.comparison), f"$t({int(r.df)}) = {r.t:.2f}$",
                         p_fmt(r.p_t_bonferroni), num(r.cohen_dz), num(r.shapiro_p, 3),
                         p_fmt(r.p_wilcoxon_holm), num(r.rank_biserial_r)])
    return rows


def friedman_rows(fr, names, set_label=lambda s: s):
    rows = []
    for m, name in names.items():
        for k, (_, r) in enumerate(fr[fr.metric == m].iterrows()):
            rows.append([name if k == 0 else "", set_label(r.set), rf"$\chi^2({int(r.df)}) = {r.statistic:.2f}$",
                         p_fmt(r.p), num(r.effect)])
    return rows


ANOVA_HEAD = r"Metric & Effect & $F$ (Greenhouse--Geisser $df$) & $p$ & $\eta^2_p$ & $\varepsilon$"
POSTHOC_HEAD = (r"Metric & Comparison & $t$ & $p$ (Bonf.) & $d_z$ & Shapiro $p$ & Wilcoxon $p$ (Holm) & $r$")
FRIEDMAN_HEAD = r"Metric & Set & $\chi^2$ & $p$ & Kendall's $W$"
POSTHOC_NOTE = (r"paired $t$-tests with the Bonferroni correction and Cohen's $d_z$; Shapiro--Wilk test on the "
                r"paired differences; Wilcoxon signed-rank tests with the Holm correction and rank-biserial "
                r"correlation $r$.")


def training(parts):
    d = ROOT / "reaching_regions" / "statistics"
    anova, ph, fr = (pd.read_csv(d / f"training_{k}.csv") for k in ("anova", "posthoc", "friedman"))
    parts.append(r"\section{Reaching training}")
    parts.append(table(
        r"Reaching training: repeated-measures ANOVA Training $\times$ Block on the outward movements, with the "
        r"Greenhouse--Geisser correction ($\varepsilon$ and corrected degrees of freedom; with two levels, "
        r"$\varepsilon = 1$) and the partial eta squared.", "Reaching training: ANOVA", "tab:app_training_anova",
        ANOVA_HEAD, anova_rows(anova, TRAINING, ["Training", "Block", "Training × Block"]), "llcccc"))
    parts.append(table(
        r"Reaching training: planned comparisons between blocks (B1 and B12 of training 1, B12 of training 1 "
        r"and B1 of training 2, B12 of the two sessions): " + POSTHOC_NOTE,
        "Reaching training: planned comparisons", "tab:app_training_posthoc", POSTHOC_HEAD,
        posthoc_rows(ph, TRAINING), "llcccccc"))
    parts.append(table(
        r"Reaching training: Friedman tests on the twelve blocks of each session, with Kendall's $W$ as effect "
        r"size.", "Reaching training: Friedman tests", "tab:app_training_friedman", FRIEDMAN_HEAD,
        friedman_rows(fr, TRAINING, lambda s: s.replace(", 12 blocks", "").capitalize()), "llccc"))


def blind(parts):
    d = ROOT / "reaching_blind" / "statistics"
    anova, ph, fr = (pd.read_csv(d / f"blind_{k}.csv") for k in ("anova", "posthoc", "friedman"))
    parts.append(r"\clearpage" "\n" r"\section{Blind reaching test}")   # ogni sezione su una pagina nuova, con le sue tabelle
    parts.append(table(
        r"Blind reaching test: repeated-measures ANOVA Test $\times$ Block, with the Greenhouse--Geisser "
        r"correction and the partial eta squared.", "Blind reaching test: ANOVA", "tab:app_blind_anova",
        ANOVA_HEAD, anova_rows(anova, BLIND, ["Test", "Block", "Test × Block"]), "llcccc"))
    parts.append(table(
        r"Blind reaching test: planned comparisons between consecutive tests: " + POSTHOC_NOTE,
        "Blind reaching test: planned comparisons", "tab:app_blind_posthoc", POSTHOC_HEAD,
        posthoc_rows(ph, BLIND, lambda c: c.replace("test ", "T")), "llcccccc"))
    parts.append(table(
        r"Blind reaching test: Friedman tests on the five tests, with Kendall's $W$ as effect size.",
        "Blind reaching test: Friedman tests", "tab:app_blind_friedman", FRIEDMAN_HEAD,
        friedman_rows(fr, BLIND, lambda s: "Tests 1--5"), "llccc"))


def robot(parts):
    d = ROOT / "real_robot" / "statistics"
    anova, ph = pd.read_csv(d / "robot_anova.csv"), pd.read_csv(d / "robot_posthoc.csv")
    rows = []
    for m, phase, name, leg in ROBOT:
        a = anova[(anova.metric == m) & (anova.phase == phase)]
        t = ph[(ph.metric == m) & (ph.phase == phase)]
        if a.empty or t.empty:
            continue
        a, t = a.iloc[0], t.iloc[0]
        rows.append([name, leg, f"$F(1, {int(a.df2)}) = {a.F:.2f}$", p_fmt(a.p), num(a.partial_eta2),
                     num(t.cohen_dz), num(t.shapiro_p, 3), p_fmt(t.p_wilcoxon_holm), num(t.rank_biserial_r)])
    parts.append(r"\clearpage" "\n" r"\section{Test on the real robot}")   # ogni sezione su una pagina nuova, con le sue tabelle
    parts.append(table(
        r"Test on the real robot: repeated-measures ANOVA between run 1 and run 2 for every phase and leg "
        r"(with two levels, equivalent to a paired $t$-test, $t^2 = F$), with the partial eta squared and "
        r"Cohen's $d_z$; Shapiro--Wilk test on the paired differences; Wilcoxon signed-rank test and "
        r"rank-biserial correlation $r$.", "Test on the real robot: ANOVA and Wilcoxon tests",
        "tab:app_robot", r"Metric & Leg & $F$ & $p$ & $\eta^2_p$ & $d_z$ & Shapiro $p$ & Wilcoxon $p$ & $r$",
        rows, "llccccccc"))


def exploratory(parts):
    db, dr = ROOT / "reaching_blind" / "statistics", ROOT / "real_robot" / "statistics"
    ba, bp = pd.read_csv(db / "blind_exploratory_anova.csv"), pd.read_csv(db / "blind_exploratory_posthoc.csv")
    bf = pd.read_csv(db / "blind_exploratory_friedman.csv")
    ra, rp = pd.read_csv(dr / "robot_exploratory_anova.csv"), pd.read_csv(dr / "robot_exploratory_posthoc.csv")
    parts.append(r"\clearpage" "\n" r"\section{Exploratory metrics}")   # ogni sezione su una pagina nuova, con le sue tabelle
    rows = anova_rows(ba, {"region_correct": "Final region"}, ["Test"])
    rows += [["", "Friedman", rf"$\chi^2({int(r.df)}) = {r.statistic:.2f}$", p_fmt(r.p), num(r.effect), "--"]
             for _, r in bf.iterrows()]
    parts.append(table(
        r"Blind reaching test, exploratory metric (share of trials ending in the region of the target): "
        r"repeated-measures ANOVA on Test (Greenhouse--Geisser-corrected) and Friedman test (Kendall's $W$ "
        r"in the $\eta^2_p$ column). The $p$-values are not corrected for the exploratory analysis.",
        "Blind reaching test: exploratory metric", "tab:app_blind_exploratory", ANOVA_HEAD, rows, "llcccc"))
    parts.append(table(
        r"Blind reaching test, exploratory metric: comparisons between consecutive tests: " + POSTHOC_NOTE,
        "Blind reaching test: exploratory comparisons", "tab:app_blind_exploratory_posthoc", POSTHOC_HEAD,
        posthoc_rows(bp, {"region_correct": "Final region"}, lambda c: c.replace("test ", "T")), "llcccccc"))
    rows = []
    for m, name in EXPLORATORY_ROBOT:
        for k, phase in enumerate(("andata", "ritorno")):
            a = ra[(ra.metric == m) & (ra.phase == phase)]
            t = rp[(rp.metric == m) & (rp.phase == phase)]
            if a.empty or t.empty:
                continue
            a, t = a.iloc[0], t.iloc[0]
            rows.append([name if k == 0 else "", LEG[phase], f"$F(1, {int(a.df2)}) = {a.F:.2f}$", p_fmt(a.p),
                         num(a.partial_eta2), num(t.cohen_dz), num(t.shapiro_p, 3), p_fmt(t.p_wilcoxon_holm),
                         num(t.rank_biserial_r)])
    parts.append(table(
        r"Test on the real robot, exploratory metrics: repeated-measures ANOVA between the two runs and "
        r"Wilcoxon signed-rank test, as in Table \ref{tab:app_robot}. The $p$-values are not corrected for the "
        r"exploratory analysis.", "Test on the real robot: exploratory metrics", "tab:app_robot_exploratory",
        r"Metric & Leg & $F$ & $p$ & $\eta^2_p$ & $d_z$ & Shapiro $p$ & Wilcoxon $p$ & $r$", rows,
        "llccccccc"))


def main():
    parts = ["% Generato da data_evaluation_thesis/thesis_tables.py: non modificare a mano."]
    training(parts)
    blind(parts)
    robot(parts)
    exploratory(parts)
    if OUT.parent.is_dir():
        OUT.write_text("\n".join(parts), encoding="utf-8")
        print(f"[thesis_tables] tabelle -> {OUT}")


if __name__ == "__main__":
    main()
