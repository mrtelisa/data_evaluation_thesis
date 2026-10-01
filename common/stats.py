"""Statistica usata in tutte le analisi (scipy; il modello misto in statsmodels).

Scelte (vedi README.md):
- ogni soggetto pesa 1: prima un valore per soggetto (mediana o percentuale
  delle sue prove), poi la statistica tra soggetti;
- descrittive: mediana [Q1-Q3] (IQR), oltre a media e SD per completezza;
- ANALISI PRINCIPALE (come il laboratorio, Bellitto et al. 2025):
  ANOVA a misure ripetute (1 o 2 fattori entro soggetto) con correzione di
  Greenhouse-Geisser ed eta^2 parziale; post hoc t di Student appaiati con
  d di Cohen (dz) e correzione di Bonferroni; Shapiro-Wilk sulle differenze;
- CONTROLLO NON PARAMETRICO (nei .csv): Wilcoxon signed-rank con rank-biserial r
  e Holm, Friedman con W di Kendall;
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


# ------------------------------------------------------ analisi parametrica ---

def bonferroni(pvalues) -> list:
    """p-value x numero di confronti (massimo 1); NaN restano NaN."""
    p = np.asarray([np.nan if v is None else v for v in pvalues], dtype=float)
    m = int(np.sum(~np.isnan(p)))
    return [float(min(1.0, v * m)) if not np.isnan(v) else np.nan for v in p]


def paired_t(first, second) -> dict:
    """t di Student appaiato (secondo - primo): n, t, p, d di Cohen (dz = media delle
    differenze / SD delle differenze; il segno dice se il secondo e' piu' alto),
    differenza media e Shapiro-Wilk sulle differenze (normalita', n >= 3)."""
    a = pd.to_numeric(pd.Series(first), errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(second), errors="coerce").to_numpy(dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    d = b[keep] - a[keep]
    out = {"n": int(len(d)), "n_changed": int(np.sum(d != 0)), "statistic": np.nan, "p": np.nan,
           "effect": np.nan, "mean_diff": float(d.mean()) if len(d) else np.nan, "shapiro_p": np.nan}
    if len(d) < 2:
        return out
    if np.allclose(d, 0):
        out.update(statistic=0.0, p=1.0, effect=0.0)
        return out
    sd = d.std(ddof=1)
    if sd == 0:   # tutti cambiano della stessa quantita': t infinito
        out.update(statistic=np.inf * np.sign(d.mean()), p=0.0, effect=np.inf * np.sign(d.mean()))
        return out
    res = stats.ttest_rel(b[keep], a[keep])
    out.update(statistic=float(res.statistic), p=float(res.pvalue), effect=float(d.mean() / sd))
    if len(d) >= 3:
        out["shapiro_p"] = float(stats.shapiro(d).pvalue)
    return out


def _contrasts(k: int) -> np.ndarray:
    """k x (k-1) contrasti ortonormali (ortogonali alla media)."""
    h = np.linalg.qr(np.column_stack([np.ones(k), np.eye(k)[:, :k - 1]]))[0]
    return h[:, 1:]


def rm_anova(long: pd.DataFrame, dv: str, within: list, subject: str = "subject") -> list:
    """ANOVA a misure ripetute con 1 o 2 fattori entro soggetto (disegno bilanciato:
    si tengono solo i soggetti con tutte le celle). Una riga per effetto:
    F, gradi di liberta', p con sfericita' assunta e con la correzione di
    Greenhouse-Geisser (epsilon), eta^2 parziale. Con piu' livelli che soggetti
    il test di Mauchly non si puo' calcolare: per questo la GG si applica sempre."""
    d = long[[subject] + list(within) + [dv]].copy()
    d[dv] = pd.to_numeric(d[dv], errors="coerce")
    wide = d.pivot_table(index=subject, columns=list(within), values=dv, aggfunc="mean")
    wide = wide.dropna()
    n = len(wide)
    levels = [sorted(d[f].dropna().unique()) for f in within]
    wide = wide.reindex(columns=pd.MultiIndex.from_product(levels) if len(within) > 1 else levels[0])
    out = []
    if n < 2 or wide.isna().any().any():
        return [{"effect": " x ".join(within), "n": n, "F": np.nan, "p": np.nan, "p_gg": np.nan}]
    y = wide.to_numpy(float)
    sizes = [len(lv) for lv in levels]
    if len(within) == 1:
        effects = {within[0]: _contrasts(sizes[0])}
    else:
        ca, cb = _contrasts(sizes[0]), _contrasts(sizes[1])
        ua, ub = np.ones((sizes[0], 1)) / math.sqrt(sizes[0]), np.ones((sizes[1], 1)) / math.sqrt(sizes[1])
        effects = {within[0]: np.kron(ca, ub), within[1]: np.kron(ua, cb),
                   f"{within[0]} x {within[1]}": np.kron(ca, cb)}
    for name, c in effects.items():
        z = y @ c
        q = c.shape[1]
        zm = z.mean(axis=0)
        ss_eff = n * float(zm @ zm)
        ss_err = float(np.sum((z - zm) ** 2))
        df1, df2 = q, q * (n - 1)
        row = {"effect": name, "n": n, "df1": df1, "df2": df2, "F": np.nan, "p": np.nan,
               "eps_gg": np.nan, "p_gg": np.nan, "partial_eta2": np.nan}
        if ss_err > 0:
            f = (ss_eff / df1) / (ss_err / df2)
            s_cov = np.cov(z, rowvar=False).reshape(q, q)
            eps = float(np.trace(s_cov) ** 2 / (q * np.trace(s_cov @ s_cov))) if q > 1 else 1.0
            eps = min(1.0, max(eps, 1.0 / q))
            row.update(F=f, p=float(stats.f.sf(f, df1, df2)), eps_gg=eps,
                       p_gg=float(stats.f.sf(f, df1 * eps, df2 * eps)),
                       partial_eta2=ss_eff / (ss_eff + ss_err))
        out.append(row)
    return out


def eta_magnitude(eta2) -> str:
    """Soglie indicative di Cohen per eta^2 parziale: 0.01 piccolo, 0.06 medio, 0.14 grande."""
    if eta2 is None or (isinstance(eta2, float) and math.isnan(eta2)):
        return "-"
    return "trascurabile" if eta2 < 0.01 else "piccolo" if eta2 < 0.06 else "medio" if eta2 < 0.14 else "grande"


def d_magnitude(d) -> str:
    """Soglie indicative di Cohen per d: 0.2 piccolo, 0.5 medio, 0.8 grande."""
    if d is None or (isinstance(d, float) and math.isnan(d)):
        return "-"
    a = abs(d)
    return "trascurabile" if a < 0.2 else "piccolo" if a < 0.5 else "medio" if a < 0.8 else "grande"


def mixed_model(df: pd.DataFrame, formula: str, group: str = "subject", slope: str = None) -> dict:
    """Modello lineare misto (statsmodels MixedLM, REML): effetti fissi della formula e
    effetti casuali per soggetto. Con `slope` (nome di una variabile) si prova, in ordine:
      1. intercetta + pendenza casuali correlate,
      2. intercetta + pendenza casuali indipendenti,
      3. sola intercetta casuale,
    e si tiene il primo modello stimabile: convergenza, verosimiglianza finita, varianze > 0
    e correlazione non a +-1 (al bordo: il modello e' troppo complesso per i dati).
    Ogni modello si prova con piu' ottimizzatori e si tiene la verosimiglianza piu' alta.
    Restituisce {"terms": DataFrame (coefficiente, SE, z, p, IC 95%), "re": descrizione degli
    effetti casuali, "converged", "n_obs", "n_groups"}."""
    import warnings as _w
    import statsmodels.formula.api as smf
    specs = []
    if slope:
        specs += [(f"~{slope}", None, f"intercetta + pendenza di {slope} (correlate)"),
                  ("1", {"slope": f"0 + {slope}"}, f"intercetta + pendenza di {slope} (indipendenti)")]
    specs.append(("1", None, "solo intercetta"))

    def usable(res, vc):
        if not (res.converged and np.isfinite(res.llf)):
            return False
        cov = np.atleast_2d(np.asarray(res.cov_re, dtype=float))
        var = np.diag(cov)
        if np.any(var <= 1e-10):
            return False
        if cov.shape[0] == 2 and abs(cov[0, 1] / math.sqrt(var[0] * var[1])) > 0.99:
            return False
        if vc is not None and np.any(np.asarray(res.vcomp) <= 1e-10):
            return False
        return True

    fallback = None
    for re_f, vc, desc in specs:
        best = None
        for method in ("bfgs", "lbfgs", "nm", "powell"):
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                try:
                    res = smf.mixedlm(formula, df, groups=df[group], re_formula=re_f,
                                      vc_formula=vc).fit(reml=True, method=method)
                except (ValueError, np.linalg.LinAlgError):
                    continue
            if np.isfinite(res.llf) and (best is None or res.llf > best.llf):
                best = res
        if best is None:
            continue
        if fallback is None:
            fallback = (best, desc)
        if usable(best, vc):
            fallback = None
            chosen, chosen_desc, ok = best, desc, True
            break
    else:
        if fallback is None:
            return {"terms": pd.DataFrame(), "re": None, "converged": False, "n_obs": len(df), "n_groups": 0}
        chosen, chosen_desc, ok = fallback[0], fallback[1], False
    res = chosen
    idx = res.fe_params.index
    ci = res.conf_int().loc[idx]
    terms = pd.DataFrame({"coef": res.fe_params, "se": res.bse_fe, "z": res.tvalues[idx], "p": res.pvalues[idx],
                          "ci_low": ci[0], "ci_high": ci[1]})
    return {"terms": terms, "re": chosen_desc, "converged": ok, "n_obs": int(res.nobs),
            "n_groups": int(res.model.n_groups)}
