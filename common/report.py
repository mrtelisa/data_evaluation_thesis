"""Report testuale (in inglese, come la tesi), scritto in statistics/report.txt e
stampato a terminale, e le tabelle .csv delle statistiche.

Per ogni metrica: descrittive di ogni condizione (media ± SD e mediana [Q1, Q3]),
ANOVA a misure ripetute (come il modello jamovi), post hoc t appaiati (Bonferroni)
e, come controllo non parametrico, Friedman e Wilcoxon (Holm)."""

from pathlib import Path

import pandas as pd

from .stats import (describe, fmt_mean_sd, fmt_median_iqr, fmt_num, fmt_p, friedman, posthoc, rm_anova)

LEGEND = """\
HOW TO READ THIS REPORT
- One value per participant and condition (the mean of their trials), then statistics
  across participants: mean ± SD (parametric) and median [Q1, Q3] (non-parametric).
- Parametric analysis (as Bellitto et al. 2025 and the jamovi models): repeated-measures
  ANOVA, F(df1, df2), p with sphericity assumed and with the Greenhouse-Geisser correction
  (p GG, to be read when a factor has more than 2 levels), eta2 = SS effect / SS total
  (the jamovi "eta2"), eta2p = partial eta2. Post hoc: paired t-tests on the planned
  comparisons, Bonferroni-corrected within each metric; dz = Cohen's d for paired data;
  Shapiro = normality of the differences (p < 0.05: check the Wilcoxon).
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
    report.text("  Repeated-measures ANOVA" + (f" (n={anova[0]['n']} complete participants)" if anova else ""))
    for r in anova:
        if r.get("F") != r.get("F") or r.get("F") is None:
            report.text(f"    {r['effect']}: not computable")
            continue
        gg = f", p GG={fmt_p(r['p_gg'])} (eps={r['eps_gg']:.2f})" if r["df1"] > 1 else ""
        report.text(f"    {r['effect']}: F({r['df1']}, {r['df2']})={r['F']:.2f}, p={fmt_p(r['p'])}{gg}, "
                    f"eta2={r['eta2']:.3f}, eta2p={r['partial_eta2']:.3f}")
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


def save_tables(results: list, out_dir: Path, prefix: str):
    """Unisce le righe di tutte le metriche e scrive <prefix>_descriptives/anova/friedman/posthoc.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in ("descriptives", "anova", "friedman", "posthoc"):
        frames = [r[key] for r in results if not r[key].empty]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out_dir / f"{prefix}_{key}.csv", index=False)
