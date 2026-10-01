"""Statistica usata in tutte le analisi (scipy).

Due versioni di ogni analisi (vedi README.md):
- PARAMETRICA (come il laboratorio, Bellitto et al. 2025, e i modelli jamovi della
  dottoranda): media +- SD tra soggetti; ANOVA a misure ripetute (1 o 2 fattori entro
  soggetto, somme dei quadrati di tipo 3, correzione di Greenhouse-Geisser, eta^2 ed
  eta^2 parziale); post hoc t di Student appaiati con correzione di Bonferroni,
  d di Cohen (dz) e Shapiro-Wilk sulle differenze;
- NON PARAMETRICA: mediana [Q1-Q3] tra soggetti; Friedman (W di Kendall) al posto
  dell'ANOVA; post hoc Wilcoxon signed-rank (rank-biserial r) con correzione di Holm.
Ogni soggetto pesa 1: prima un valore per soggetto, poi la statistica tra soggetti.
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
    """n, media, SD, mediana, Q1, Q3, IQR, min, max (NaN esclusi)."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if v.empty:
        return {"n": 0}
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "median": float(v.median()), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
            "min": float(v.min()), "max": float(v.max())}


def fmt_num(x, digits=2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{digits}f}"


def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_mean_sd(d: dict, digits=2) -> str:
    if not d or d.get("n", 0) == 0:
        return "-"
    return f"{d['mean']:.{digits}f} ± {d['sd']:.{digits}f}"


def fmt_median_iqr(d: dict, digits=2) -> str:
    if not d or d.get("n", 0) == 0:
        return "-"
    return f"{d['median']:.{digits}f} [{d['q1']:.{digits}f}, {d['q3']:.{digits}f}]"


def bonferroni(pvalues) -> list:
    """p-value x numero di confronti (massimo 1); NaN restano NaN."""
    p = np.asarray([np.nan if v is None else v for v in pvalues], dtype=float)
    m = int(np.sum(~np.isnan(p)))
    return [float(min(1.0, v * m)) if not np.isnan(v) else np.nan for v in p]


def holm(pvalues) -> list:
    """p-value corretti con Holm (NaN restano NaN)."""
    p = np.asarray([np.nan if v is None else v for v in pvalues], dtype=float)
    idx = [i for i in np.argsort(p) if not np.isnan(p[i])]
    m = len(idx)
    adj = np.full(len(p), np.nan)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def _pair(first, second):
    a = pd.to_numeric(pd.Series(first), errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(second), errors="coerce").to_numpy(dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep], b[keep]


def paired_t(first, second) -> dict:
    """t di Student appaiato (secondo - primo): n, t, gdl, p, d di Cohen (dz),
    differenza media e Shapiro-Wilk sulle differenze. Coppie incomplete escluse."""
    a, b = _pair(first, second)
    d = b - a
    out = {"n": int(len(d)), "statistic": np.nan, "df": max(len(d) - 1, 0), "p": np.nan, "effect": np.nan,
           "mean_diff": float(d.mean()) if len(d) else np.nan, "shapiro_p": np.nan}
    if len(d) < 2:
        return out
    if np.allclose(d, 0):
        out.update(statistic=0.0, p=1.0, effect=0.0)
        return out
    res = stats.ttest_rel(b, a)
    sd = d.std(ddof=1)
    out.update(statistic=float(res.statistic), p=float(res.pvalue), effect=float(d.mean() / sd) if sd else np.nan)
    if len(d) >= 3:
        out["shapiro_p"] = float(stats.shapiro(d).pvalue)
    return out


def wilcoxon_paired(first, second) -> dict:
    """Wilcoxon signed-rank (secondo vs primo): n, W, p, rank-biserial r
    (positivo = il secondo e' piu' alto). Coppie incomplete escluse."""
    a, b = _pair(first, second)
    out = {"n": int(len(a)), "statistic": np.nan, "p": np.nan, "effect": np.nan}
    if len(a) < 2:
        return out
    d = b - a
    nz = d[d != 0]
    if len(nz) == 0:
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
    """Friedman su soggetti x condizioni (solo i soggetti completi): n, k, chi2, gdl, p, W di Kendall."""
    w = wide.apply(pd.to_numeric, errors="coerce").dropna()
    n, k = w.shape
    out = {"n": int(n), "k": int(k), "statistic": np.nan, "df": k - 1, "p": np.nan, "effect": np.nan}
    if n < 2 or k < 3:
        return out
    try:
        res = stats.friedmanchisquare(*[w[c].to_numpy() for c in w.columns])
    except ValueError:
        return out
    chi2 = float(res.statistic)
    if math.isnan(chi2):
        out.update(statistic=0.0, p=1.0, effect=0.0)
        return out
    out.update(statistic=chi2, p=float(res.pvalue), effect=chi2 / (n * (k - 1)))
    return out


def _contrasts(k: int) -> np.ndarray:
    """k x (k-1) contrasti ortonormali (ortogonali alla media)."""
    h = np.linalg.qr(np.column_stack([np.ones(k), np.eye(k)[:, :k - 1]]))[0]
    return h[:, 1:]


def rm_anova(wide: pd.DataFrame, factors: list) -> list:
    """ANOVA a misure ripetute come jmv::anovaRM di jamovi (afex, tipo 3; con il disegno
    bilanciato coincide con questa). `wide`: soggetti x condizioni; con 2 fattori le
    colonne sono un MultiIndex (livello del primo, livello del secondo), in ordine
    prodotto (il primo fattore esterno). Si tengono solo
    i soggetti con tutte le celle (come jamovi). Una riga per effetto: SS, gdl, F, p
    con sfericita' assunta e con Greenhouse-Geisser (epsilon), eta^2 (SS effetto / SS
    totale, l'"eta" di jamovi) ed eta^2 parziale."""
    w = wide.apply(pd.to_numeric, errors="coerce").dropna()
    n = len(w)
    sizes = [w.columns.get_level_values(i).unique().size for i in range(w.columns.nlevels)]
    names = list(factors)
    if n < 2:
        return [{"effect": " × ".join(names), "n": n}]
    y = w.to_numpy(float)
    if len(sizes) == 1:
        effects = {names[0]: _contrasts(sizes[0])}
    else:
        ca, cb = _contrasts(sizes[0]), _contrasts(sizes[1])
        ua, ub = np.ones((sizes[0], 1)) / math.sqrt(sizes[0]), np.ones((sizes[1], 1)) / math.sqrt(sizes[1])
        effects = {names[0]: np.kron(ca, ub), names[1]: np.kron(ua, cb),
                   f"{names[0]} × {names[1]}": np.kron(ca, cb)}
    ss_total = float(np.sum((y - y.mean()) ** 2))
    out = []
    for name, c in effects.items():
        z = y @ c
        q = c.shape[1]
        zm = z.mean(axis=0)
        ss_eff = n * float(zm @ zm)
        ss_err = float(np.sum((z - zm) ** 2))
        df1, df2 = q, q * (n - 1)
        row = {"effect": name, "n": n, "ss": ss_eff, "df1": df1, "ss_error": ss_err, "df2": df2,
               "F": np.nan, "p": np.nan, "eps_gg": np.nan, "df1_gg": np.nan, "df2_gg": np.nan, "p_gg": np.nan,
               "eta2": ss_eff / ss_total if ss_total > 0 else np.nan, "partial_eta2": np.nan}
        if ss_err > 0:
            f = (ss_eff / df1) / (ss_err / df2)
            s_cov = np.cov(z, rowvar=False).reshape(q, q)
            eps = float(np.trace(s_cov) ** 2 / (q * np.trace(s_cov @ s_cov))) if q > 1 else 1.0
            eps = min(1.0, max(eps, 1.0 / q))
            row.update(F=f, p=float(stats.f.sf(f, df1, df2)), eps_gg=eps, df1_gg=df1 * eps, df2_gg=df2 * eps,
                       p_gg=float(stats.f.sf(f, df1 * eps, df2 * eps)), partial_eta2=ss_eff / (ss_eff + ss_err))
        out.append(row)
    return out


def posthoc(wide: pd.DataFrame, pairs: list) -> pd.DataFrame:
    """Confronti pianificati fra colonne di `wide` (pairs = [(colonna a, colonna b, nome)]):
    t appaiato (Bonferroni sulla famiglia) e Wilcoxon (Holm sulla famiglia)."""
    rows = []
    for a, b, name in pairs:
        t = paired_t(wide[a], wide[b])
        wx = wilcoxon_paired(wide[a], wide[b])
        rows.append({"comparison": name, "n": t["n"], "mean_diff": t["mean_diff"], "t": t["statistic"],
                     "df": t["df"], "p_t": t["p"], "cohen_dz": t["effect"], "shapiro_p": t["shapiro_p"],
                     "W": wx["statistic"], "p_wilcoxon": wx["p"], "rank_biserial_r": wx["effect"]})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(df.columns.get_loc("cohen_dz"), "p_t_bonferroni", bonferroni(df["p_t"]))
        df.insert(df.columns.get_loc("rank_biserial_r"), "p_wilcoxon_holm", holm(df["p_wilcoxon"]))
    return df
