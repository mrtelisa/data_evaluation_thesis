"""Report testuale in italiano, scritto in statistics/report.txt e stampato a terminale.

Pensato per essere letto senza sapere di statistica: per ogni confronto dice i
valori tipici dei due momenti, se la differenza e' significativa (dopo la
correzione di Holm) e quanto e' grande l'effetto.
"""

from pathlib import Path

from .stats import direction, fmt_median_iqr, fmt_num, magnitude, significance

LEGEND = """\
COME LEGGERE QUESTO REPORT
- "mediana [Q1-Q3]": il valore tipico del gruppo e la fascia in cui sta la meta' centrale
  dei soggetti. Ogni soggetto pesa 1: prima un valore per soggetto, poi il gruppo.
- p: probabilita' di vedere una differenza cosi' grande se in realta' non ci fosse nessun
  effetto. p Holm: lo stesso p corretto perche' si fanno piu' confronti sulla stessa
  domanda. Significativo = p Holm < 0.05 (per i confronti esplorativi si guarda p).
- effetto: quanto e' grande la differenza (rank-biserial r per 2 momenti, W di Kendall per
  Friedman, rho per le correlazioni). |0.1| piccolo, |0.3| medio, |0.5| grande.
  Con pochi soggetti conta piu' l'effetto del p-value.
- migliorato / peggiorato: in base a cosa e' meglio per quella metrica (es. tempo: meno
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

    def write(self, out_dir):
        path = Path(out_dir) / "report.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print("\n".join(self.lines))
        print(f"\n[report] -> {path}")
        return path
