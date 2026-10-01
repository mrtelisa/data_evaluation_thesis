"""Report testuale in italiano, scritto in statistics/report.txt e stampato a terminale.

Pensato per essere letto senza sapere di statistica: per ogni confronto dice i
valori tipici dei due momenti, se la differenza e' significativa (dopo la
correzione di Holm) e quanto e' grande l'effetto.
"""

from pathlib import Path

from .stats import d_magnitude, direction, eta_magnitude, fmt_median_iqr, fmt_num, magnitude, significance

LEGEND = """\
COME LEGGERE QUESTO REPORT
- "mediana [Q1-Q3]": il valore tipico del gruppo e la fascia in cui sta la meta' centrale
  dei soggetti. Ogni soggetto pesa 1: prima un valore per soggetto, poi il gruppo.
- ANALISI PRINCIPALE (come il laboratorio): ANOVA a misure ripetute e post hoc con t di
  Student appaiati.
  ANOVA: F(gdl1, gdl2), p; "p GG" = p con la correzione di Greenhouse-Geisser (epsilon < 1
  quando i livelli non hanno la stessa variabilita'), da leggere quando ci sono piu' di 2
  livelli; eta2p = eta^2 parziale (0.01 piccolo, 0.06 medio, 0.14 grande).
  Post hoc: t(gdl), p, "p Bonf" = p x numero di confronti; d = d di Cohen appaiato
  (0.2 piccolo, 0.5 medio, 0.8 grande; segno + = il secondo momento e' piu' alto);
  Shapiro = p del test di normalita' sulle differenze (< 0.05: non normali, guardare
  anche il controllo non parametrico).
  Significativo = p Bonf (o p GG) < 0.05; i confronti esplorativi non sono corretti.
- CONTROLLO NON PARAMETRICO: Wilcoxon con p corretto con Holm (e r rank-biserial nei .csv).
  Con 6 soggetti il p piu' piccolo possibile del Wilcoxon e' 0.031: dopo la correzione
  non puo' uscire significativo, conta la direzione.
- migliorato / peggiorato: in base a cosa e' meglio per la metrica (es. tempo: meno
  e' meglio; hit rate: piu' e' meglio).
- "cambiano k": quanti soggetti hanno un valore diverso fra i due momenti. Se sono pochi
  (es. 1 su 9) l'effetto puo' risultare "grande" ma riguarda solo quei soggetti.
"""


class Report:
    def __init__(self, title: str):
        self.lines = [title, "=" * len(title), "", LEGEND]

    def section(self, title: str):
        self.lines += ["", title, "-" * len(title)]

    def text(self, line: str = ""):
        self.lines.append(line)

    def comparison(self, label, unit, better, d_first, d_second, test, p_holm=None,
                   name_first="1", name_second="2", digits=2):
        """Una riga per un confronto fra due momenti."""
        delta = (d_second.get("median") - d_first.get("median")) \
            if d_first.get("n") and d_second.get("n") else None
        p = test.get("p")
        eff = test.get("effect")
        # identical medians: the direction comes from the effect (most subjects moved that way)
        if delta == 0 and eff is not None and eff == eff and abs(eff) >= 0.1:
            delta = eff
        sig = significance(p_holm if p_holm is not None else p)
        ptxt = f"p={fmt_num(p, 3)}" + (f", p Holm={fmt_num(p_holm, 3)}" if p_holm is not None else "")
        self.lines.append(
            f"  {label}{f' [{unit}]' if unit else ''}: {name_first} {fmt_median_iqr(d_first, digits)} -> "
            f"{name_second} {fmt_median_iqr(d_second, digits)} | {direction(delta, better)} | "
            f"n={test.get('n', 0)} (cambiano {test.get('n_changed', '-')}), {ptxt}, significativo: {sig} | "
            f"effetto r={fmt_num(eff, 2)} ({magnitude(eff)})")

    def anova(self, rows, prefix="ANOVA"):
        """Una riga con tutti gli effetti di rm_anova()."""
        parts = []
        for r in rows:
            if r.get("F") is None or r.get("F") != r.get("F"):
                parts.append(f"{r['effect']}: -")
                continue
            gg = f", eps GG={r['eps_gg']:.2f}, p GG={fmt_num(r['p_gg'], 3)}" if r["df1"] > 1 else ""
            parts.append(f"{r['effect']} F({r['df1']},{r['df2']})={r['F']:.2f}, p={fmt_num(r['p'], 3)}{gg}, "
                         f"eta2p={r['partial_eta2']:.2f} ({eta_magnitude(r['partial_eta2'])}), "
                         f"significativo: {significance(r['p_gg'] if r['df1'] > 1 else r['p'])}")
        n = rows[0].get("n", 0) if rows else 0
        self.lines.append(f"  {prefix} (n={n}): " + " | ".join(parts))

    def ttest(self, label, unit, better, d_first, d_second, t, p_corr=None, name_first="1", name_second="2",
              digits=2, np_check=None):
        """Una riga per un confronto con t appaiato (+ controllo non parametrico)."""
        delta = (d_second.get("median") - d_first.get("median")) \
            if d_first.get("n") and d_second.get("n") else None
        if delta == 0 and t.get("mean_diff"):
            delta = t["mean_diff"]
        p = t.get("p")
        ptxt = f"p={fmt_num(p, 3)}" + (f", p Bonf={fmt_num(p_corr, 3)}" if p_corr is not None else "")
        tt = f"t({t['n'] - 1})={fmt_num(t.get('statistic'), 2)}" if t.get("n", 0) > 1 else "t=-"
        np_txt = f" | Wilcoxon p Holm={fmt_num(np_check, 3)}" if np_check is not None else ""
        sp = t.get("shapiro_p")
        if sp is not None and sp == sp and sp < 0.05:
            np_txt += "  <- differenze NON normali: vale il controllo non parametrico"
        self.lines.append(
            f"  {label}{f' [{unit}]' if unit else ''}: {name_first} {fmt_median_iqr(d_first, digits)} -> "
            f"{name_second} {fmt_median_iqr(d_second, digits)} | {direction(delta, better)} | "
            f"n={t.get('n', 0)} (cambiano {t.get('n_changed', '-')}), {tt}, {ptxt}, "
            f"significativo: {significance(p_corr if p_corr is not None else p)} | "
            f"d={fmt_num(t.get('effect'), 2)} ({d_magnitude(t.get('effect'))}), Shapiro p={fmt_num(t.get('shapiro_p'), 2)}"
            f"{np_txt}")

    def write(self, out_dir):
        path = Path(out_dir) / "report.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print("\n".join(self.lines))
        print(f"\n[report] -> {path}")
        return path
