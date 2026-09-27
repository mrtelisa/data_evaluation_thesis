"""Statistica usata in tutte le analisi (solo scipy).

Scelte (vedi README.md):
- ogni soggetto pesa 1: prima un valore per soggetto (mediana o percentuale
  delle sue prove), poi la statistica tra soggetti;
- descrittive: mediana [Q1-Q3] (IQR), oltre a media e SD per completezza;
- 2 momenti appaiati: Wilcoxon signed-rank, effect size = correlazione
  rank-biserial appaiata (da -1 a +1; il segno dice se il secondo momento e'
  piu' alto (+) o piu' basso (-) del primo);
- piu' di 2 momenti: Friedman, effect size = W di Kendall (da 0 a 1);
- confronti multipli: correzione di Holm dentro ogni famiglia di confronti;
- correlazioni: Spearman (rho e' gia' l'effect size).
"""

import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", message="Sample size too small")
warnings.filterwarnings("ignore", message="Exact p-value calculation")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")


def describe(values) -> dict:
    """n, mediana, Q1, Q3, IQR, media, SD, min, max dei valori (NaN esclusi)."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if v.empty:
        return {"n": 0}
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    return {"n": int(len(v)), "median": float(v.median()), "q1": float(q1), "q3": float(q3),
            "iqr": float(q3 - q1), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}


def fmt_num(x, digits=2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{digits}f}"


def fmt_median_iqr(d: dict, digits=2) -> str:
    """'85.00 [60.00-110.00]' (mediana [Q1-Q3])."""
    if not d or d.get("n", 0) == 0:
        return "-"
    return f"{d['median']:.{digits}f} [{d['q1']:.{digits}f}-{d['q3']:.{digits}f}]"


def magnitude(effect) -> str:
    """Soglie indicative di Cohen per r, W e rho: 0.1 piccolo, 0.3 medio, 0.5 grande."""
    if effect is None or (isinstance(effect, float) and math.isnan(effect)):
        return "-"
    a = abs(effect)
    if a < 0.1:
        return "trascurabile"
    if a < 0.3:
        return "piccolo"
    if a < 0.5:
        return "medio"
    return "grande"


def wilcoxon_paired(first, second) -> dict:
    """Wilcoxon appaiato fra due momenti (stessi soggetti, stesso ordine).
    Coppie con un valore mancante escluse. Restituisce n, statistic, p,
    rank-biserial r (positivo = il secondo momento e' piu' alto)."""
    a = pd.to_numeric(pd.Series(first), errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(second), errors="coerce").to_numpy(dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    out = {"n": int(len(a)), "n_changed": 0, "statistic": np.nan, "p": np.nan, "effect": np.nan}
    if len(a) < 2:
        return out
    d = b - a
    nz = d[d != 0]
    out["n_changed"] = int(len(nz))
    if len(nz) == 0:   # nessuna differenza: nessun effetto
        out.update(statistic=0.0, p=1.0, effect=0.0)
        return out
    ranks = stats.rankdata(np.abs(nz))
    t_plus, t_minus = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    out["effect"] = float((t_plus - t_minus) / (t_plus + t_minus))
    try:
        res = stats.wilcoxon(b, a)
        out.update(statistic=float(res.statistic), p=float(res.pvalue))
    except ValueError:
        pass
    return out


def friedman(wide: pd.DataFrame) -> dict:
    """Friedman su una tabella soggetti x momenti (solo i soggetti completi).
    Restituisce n, k, chi2, p, W di Kendall."""
    w = wide.apply(pd.to_numeric, errors="coerce").dropna()
    n, k = w.shape
    out = {"n": int(n), "k": int(k), "statistic": np.nan, "p": np.nan, "effect": np.nan}
    if n < 2 or k < 3:
        return out
    try:
        res = stats.friedmanchisquare(*[w[c].to_numpy() for c in w.columns])
    except ValueError:
        return out
    chi2 = float(res.statistic)
    if math.isnan(chi2):   # tutti i valori identici
        out.update(statistic=0.0, p=1.0, effect=0.0)
        return out
    out.update(statistic=chi2, p=float(res.pvalue), effect=chi2 / (n * (k - 1)))
    return out


def holm(pvalues) -> list:
    """p-value corretti con Holm (NaN restano NaN): ordinati dal piu' piccolo,
    il primo x m, il secondo x (m-1), ... senza mai scendere sotto il precedente."""
    p = np.asarray([np.nan if v is None else v for v in pvalues], dtype=float)
    idx = [i for i in np.argsort(p) if not np.isnan(p[i])]
    m = len(idx)
    adj = np.full(len(p), np.nan)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def spearman(x, y) -> dict:
    """Correlazione di Spearman sulle coppie complete."""
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    out = {"n": int(len(a)), "rho": np.nan, "p": np.nan}
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return out
    res = stats.spearmanr(a, b)
    out.update(rho=float(res.statistic), p=float(res.pvalue))
    return out


def direction(delta, better) -> str:
    """'migliorato' / 'peggiorato' / 'invariato' dal segno della differenza
    (secondo - primo) e da cosa e' meglio per la metrica ('up' o 'down')."""
    if delta is None or (isinstance(delta, float) and math.isnan(delta)) or better not in ("up", "down"):
        return "-"
    if delta == 0:
        return "invariato"
    return "migliorato" if (delta > 0) == (better == "up") else "peggiorato"


def significance(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "-"
    return "si'" if p < 0.05 else "no"
