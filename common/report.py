"""Report testuale (in inglese, come la tesi), scritto in statistics/report.txt e
stampato a terminale, e le tabelle .csv delle statistiche.

Per ogni metrica: descrittive di ogni condizione (media ± SD e mediana [Q1, Q3]),
ANOVA a misure ripetute (come il modello jamovi), post hoc t appaiati (Bonferroni)
e, come controllo non parametrico, Friedman e Wilcoxon (Holm)."""

from pathlib import Path

import pandas as pd

from .stats import (describe, fmt_mean_sd, fmt_median_iqr, fmt_num, fmt_p, friedman, posthoc, posthoc_jamovi,
                    rm_anova)

LEGEND = """\
HOW TO READ THIS REPORT
- One value per participant and condition (the mean of their trials), then statistics
  across participants: mean ± SD (parametric) and median [Q1, Q3] (non-parametric).
- Parametric analysis (as Bellitto et al. 2025 and the jamovi models): repeated-measures
  ANOVA with the same tables and numbers as jamovi (jmv::anovaRM, type 3 sums of squares,
  checked against jamovi output):
    Within-subjects effects: SS, df, MS, F, p for every effect and its residual, with
      sphericity assumed (None) and with the Greenhouse-Geisser and Huynh-Feldt corrections
      (read the Greenhouse-Geisser row when a factor has more than 2 levels);
      eta2G = generalised eta2, eta2 = SS effect / SS total, eta2p = partial eta2.
    Between-subjects effects: residual (differences between participants).
    Sphericity tests: Mauchly's W and p (p < 0.05: sphericity violated, use the corrected
      rows), Greenhouse-Geisser and Huynh-Feldt epsilon. Not computable with 2 levels or
      with more levels than participants (e.g. 12 blocks, 11 participants).
    No Levene test: there are no between-subjects groups.
  Robot (2 levels): jamovi post hoc test1 - test2 (mean difference, SE, df, t, p; with a
  single comparison the Tukey, Scheffe, Bonferroni and Holm p are all equal to p).
  Planned comparisons (not in jamovi): paired t-tests, Bonferroni-corrected within each
  metric; dz = Cohen's d for paired data; Shapiro = normality of the differences
  (p < 0.05: check the Wilcoxon).
- Non-parametric control: Friedman (Kendall's W) and Wilcoxon signed-rank (Holm-corrected,
  r = rank-biserial correlation).
- Significance in the figures: * p < 0.05, ** p < 0.01, *** p < 0.001 (corrected p).
"""


class Report:
    def __init__(self, title: str, legend: bool = True):
        self.lines = [title, "=" * len(title), ""] + ([LEGEND] if legend else [])

    def section(self, title: str):
        self.lines += ["", title, "-" * len(title)]

    def text(self, line: str = ""):
        self.lines.append(line)

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        text = "\n".join(self.lines) + "\n"
        (out_dir / "report.txt").write_text(text, encoding="utf-8")
        print(text)


def analyse_metric(report: Report, metric: str, label: str, unit: str, wide: pd.DataFrame, factors: list,
                   cells: list, pairs: list, digits=2, friedman_sets=None) -> dict:
    """Statistica completa di una metrica.
    wide: soggetti x celle (colonne = chiavi di `cells`); cells = [(colonna, nome leggibile)];
    pairs = [(colonna a, colonna b, nome)] confronti pianificati; friedman_sets =
    [(nome, [colonne])] per il controllo non parametrico (default: tutte le celle, se >= 3).
    Scrive nel report e restituisce le righe per i .csv."""
    report.section(f"{label}{f' ({unit})' if unit else ''}")
    desc_rows = []
    for col, name in cells:
        d = describe(wide[col])
        desc_rows.append({"metric": metric, "condition": name, **d})
        report.text(f"  {name:<22} mean ± SD {fmt_mean_sd(d, digits):<22} median [IQR] {fmt_median_iqr(d, digits)}"
                    f"   (n={d.get('n', 0)})")
    anova = rm_anova(wide[[c for c, _ in cells]], factors)
    anova_tables(report, anova, len(wide))
    if len(cells) == 2 and len(factors) == 1:
        pj = posthoc_jamovi(wide, cells[0][0], cells[1][0])
        report.text(f"  Post hoc (jamovi) {cells[0][1]} - {cells[1][1]}: mean difference {_num(pj['md'])}, "
                    f"SE {_num(pj['se'])}, df {fmt_num(pj['df'], 0)}, t {_num(pj['t'])}, p {fmt_p(pj['p'])} "
                    "(= Tukey = Scheffe = Bonferroni = Holm)")
    fr_rows = []
    sets = friedman_sets if friedman_sets is not None else ([("all", [c for c, _ in cells])] if len(cells) >= 3 else [])
    for name, cols in sets:
        fr = friedman(wide[cols])
        fr_rows.append({"metric": metric, "set": name, **fr})
        report.text(f"  Friedman ({name}): chi2({fr['df']})={fmt_num(fr['statistic'])}, p={fmt_p(fr['p'])}, "
                    f"W={fmt_num(fr['effect'])} (n={fr['n']})")
    ph = posthoc(wide, pairs)
    if not ph.empty:
        report.text("  Post hoc (planned comparisons)")
        for _, r in ph.iterrows():
            report.text(f"    {r['comparison']}: t({int(r['df'])})={fmt_num(r['t'])}, p Bonf={fmt_p(r['p_t_bonferroni'])}, "
                        f"dz={fmt_num(r['cohen_dz'])}, Shapiro p={fmt_p(r['shapiro_p'])} | Wilcoxon p Holm="
                        f"{fmt_p(r['p_wilcoxon_holm'])}, r={fmt_num(r['rank_biserial_r'])}")
        ph.insert(0, "metric", metric)
    return {"descriptives": pd.DataFrame(desc_rows),
            "anova": pd.DataFrame([{"metric": metric, **r} for r in anova]),
            "friedman": pd.DataFrame(fr_rows), "posthoc": ph}


def _num(x) -> str:
    """Numero come nelle tabelle di jamovi: 3 decimali, 1 se il valore e' grande."""
    if x is None or x != x:
        return "-"
    if x != 0 and abs(x) < 0.001:
        return f"{x:.2e}"
    return f"{x:.1f}" if abs(x) >= 1000 else f"{x:.3f}"


def anova_tables(report: Report, anova: list, n_total: int):
    """Le tabelle dell'ANOVA a misure ripetute di jamovi, in testo."""
    n = anova[0].get("n", 0) if anova else 0
    report.text(f"  Repeated-measures ANOVA (n={n} participants with all cells, {n_total - n} excluded)")
    if not anova or anova[0].get("F") is None or anova[0].get("F") != anova[0].get("F"):
        report.text("    not computable")
        return
    report.text("    Within-subjects effects")
    report.text(f"      {'':<20}{'Correction':<20}{'SS':>11}{'df':>9}{'MS':>11}{'F':>9}{'p':>8}"
                f"{'eta2G':>8}{'eta2':>8}{'eta2p':>8}")
    for r in anova:
        for corr, sfx in (("None", ""), ("Greenhouse-Geisser", "_gg"), ("Huynh-Feldt", "_hf")):
            df1 = r["df1"] if not sfx else r[f"df1{sfx}"]
            p = r["p"] if not sfx else r[f"p{sfx}"]
            eff = (f"{_num(r['ges']):>8}{_num(r['eta2']):>8}{_num(r['partial_eta2']):>8}")
            report.text(f"      {r['effect'] if not sfx else '':<20}{corr:<20}{_num(r['ss']):>11}"
                        f"{fmt_num(df1, 0 if not sfx else 3):>9}{_num(r['ss'] / df1):>11}{_num(r['F']):>9}"
                        f"{fmt_p(p):>8}{eff}")
        for corr, sfx in (("None", ""), ("Greenhouse-Geisser", "_gg"), ("Huynh-Feldt", "_hf")):
            df2 = r["df2"] if not sfx else r[f"df2{sfx}"]
            report.text(f"      {'Residual' if not sfx else '':<20}{corr:<20}{_num(r['ss_error']):>11}"
                        f"{fmt_num(df2, 0 if not sfx else 3):>9}{_num(r['ss_error'] / df2):>11}")
    r0 = anova[0]
    report.text("    Between-subjects effects")
    report.text(f"      {'Residual':<40}{_num(r0['ss_subjects']):>11}{r0['df_subjects']:>9}{_num(r0['ms_subjects']):>11}")
    report.text("    Sphericity tests")
    report.text(f"      {'':<20}{'Mauchly W':>11}{'p':>8}{'GG eps':>9}{'HF eps':>9}")
    for r in anova:
        w = 1.0 if r["df1"] == 1 else r["mauchly_w"]
        report.text(f"      {r['effect']:<20}{_num(w):>11}{fmt_p(r['mauchly_p']):>8}{_num(r['eps_gg']):>9}"
                    f"{_num(r['eps_hf']):>9}")


def save_tables(results: list, out_dir: Path, prefix: str):
    """Unisce le righe di tutte le metriche e scrive <prefix>_descriptives/anova/friedman/posthoc.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in ("descriptives", "anova", "friedman", "posthoc"):
        frames = [r[key] for r in results if not r[key].empty]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out_dir / f"{prefix}_{key}.csv", index=False)
