"""Metriche di reaching calcolate dai file _raw.csv (un frame per riga), con le
definizioni degli script MATLAB del laboratorio, portate in Python:

  computereachmetrics.m (un valore per movimento, Bellitto et al. 2025, J. Neural Eng. 22 046036)
    euclidean_error_1s   distanza cursore - centro del target a HIDDEN_S dall'inizio del
                         movimento (quando il cursore ricompare)
    direction_error_1s   angolo [deg, 0-180] fra partenza -> target e partenza -> cursore a HIDDEN_S
    extent_error_1s      |partenza -> cursore a HIDDEN_S| - |partenza -> target|
                         (< 0 = il cursore e' rimasto corto)
    movement_time        durata del movimento: target mostrato -> entrata nel target
                         (quella che completa il dwell)
    linearity_index      lunghezza del percorso / distanza partenza-arrivo - 1 (0 = rettilineo;
                         il "- 1" e' quello del paper, computereachmetrics.m da' il rapporto)
    aspect_ratio         massima distanza perpendicolare dalla retta partenza-arrivo / distanza
    dimensionless_jerk   sqrt(mean(|jerk|^2) * MT^6 / L^2 / 2), L = lunghezza del percorso,
                         jerk da Savitzky-Golay (ordine 4, ~0.22 s = 11 campioni a 50 Hz)
  smooth.m
    n_speed_peaks        picchi della velocita' (SG ordine 3, 0.2 s) > 25% del massimo,
                         distanti almeno 0.5 s; smoothness = 1 / n_speed_peaks
  speed1.m
    mean_speed, sd_speed, peak_speed   velocita' istantanea fra campioni consecutivi
                         (passo / dt, esclusi i buchi > 0.2 s)
  cursor_distance.m
    path_length          somma dei passi (esclusi i buchi > 0.2 s)
  movement_train.m
    count_movements      1 + numero di cambi di regione (NaN tolti)
  tt.m
    session_log_duration ultimo - primo tempo del log
  jerk1.m
    session_jerk         jerk medio e dimensionless jerk su tratti continui del log
Metriche in piu' del task (non negli script del laboratorio), ricalcolate dal raw
con le definizioni del task: onset_metrics (reaction time, initial direction error),
blind_region_metrics (regione alla ricomparsa, prima regione visitata, ...).

Differenze volute rispetto agli script MATLAB (vedi README):
- tempi in secondi, non in campioni: gli script assumono 50-100 Hz (tt = 50 campioni,
  fc = 50), la webcam registra a ~30 Hz. Le posizioni a HIDDEN_S sono interpolate;
- prima delle derivate (SG) il segnale viene ricampionato su una griglia uniforme alla
  frequenza della sessione (i timestamp della webcam oscillano);
- unita': pixel del canvas (1200 x 650), non cm/m (manca la misura fisica dello schermo).
"""

import math

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

GAP_S = 0.2              # cursor_distance.m / speed1.m: passi con dt > 0.2 s esclusi
JERK_WINDOW_S = 0.22     # computereachmetrics.m: 11 campioni a 50 Hz
JERK_ORDER = 4
SPEED_WINDOW_S = 0.20    # smooth.m
SPEED_ORDER = 3
PEAK_FRACTION = 0.25     # smooth.m: picchi > 25% del massimo
PEAK_DISTANCE_S = 0.50   # smooth.m: picchi distanti almeno 0.5 s

TRIAL_KEYS = ("euclidean_error_1s", "direction_error_1s", "extent_error_1s", "movement_time",
              "linearity_index", "aspect_ratio", "dimensionless_jerk", "n_speed_peaks", "smoothness",
              "path_length", "mean_speed", "sd_speed", "peak_speed")


def _odd_window(seconds: float, fs: float, order: int, n: int) -> int:
    """Finestra SG in campioni: dispari, > ordine, non piu' lunga del segnale (0 = impossibile)."""
    w = int(round(seconds * fs))
    w += 1 - w % 2
    w = max(w, order + 2 + (order % 2))
    if w > n:
        w = n - (1 - n % 2)
    return w if w > order else 0


def _uniform(t, xy, fs):
    """Ricampiona (t, xy) su una griglia uniforme a fs (interpolazione lineare)."""
    tu = np.arange(t[0], t[-1] + 1e-9, 1.0 / fs)
    return tu, np.column_stack([np.interp(tu, t, xy[:, 0]), np.interp(tu, t, xy[:, 1])])


def _steps(t, xy, gap_s=GAP_S):
    """Passi validi fra campioni consecutivi (0 < dt <= gap_s): lunghezze e dt."""
    d = np.hypot(*np.diff(xy, axis=0).T)
    dt = np.diff(t)
    ok = np.isfinite(d) & (dt > 0) & (dt <= gap_s)
    return d[ok], dt[ok]


def speed_stats(t, xy, gap_s=GAP_S) -> dict:
    """speed1.m e cursor_distance.m: percorso e velocita' istantanea."""
    d, dt = _steps(t, xy, gap_s)
    if d.size == 0:
        return {"path_length": np.nan, "mean_speed": np.nan, "sd_speed": np.nan, "peak_speed": np.nan}
    v = d / dt
    return {"path_length": float(d.sum()), "mean_speed": float(v.mean()),
            "sd_speed": float(v.std(ddof=1)) if v.size > 1 else 0.0, "peak_speed": float(v.max())}


def speed_peaks(t, xy, fs) -> int:
    """smooth.m: velocita' da SG (derivata 1), lisciata con SG, primo valore ignorato;
    picchi > 25% del massimo e distanti almeno 0.5 s. NaN se il segnale e' troppo corto."""
    if len(t) < 3:
        return np.nan
    tu, _ = _uniform(t, xy, fs)
    if not _odd_window(SPEED_WINDOW_S, fs, SPEED_ORDER, len(tu)):
        return np.nan
    return int(speed_peak_times(t, xy, fs).size)


def session_jerk(t, xy, fs) -> dict:
    """jerk1.m: il log diviso in tratti continui (buchi > 0.2 s, tratti < 1 s scartati),
    posizione lisciata con SG (ordine 3, 11 campioni), derivate con il gradiente;
    jerk medio e SD (px/s^3) su tutti i campioni, dimensionless jerk
    T^5 / D^2 * int |j|^2 dt per tratto (D = percorso), media pesata sulla durata."""
    t, xy = np.asarray(t, float), np.asarray(xy, float)
    breaks = np.flatnonzero(~((np.diff(t) > 0) & (np.diff(t) <= GAP_S))) + 1
    jerk_all, dj, weights = [], [], []
    for idx in np.split(np.arange(len(t)), breaks):
        if len(idx) < 10 or t[idx[-1]] - t[idx[0]] < 1.0:
            continue
        tk, xk = t[idx], xy[idx]
        w = min(11, len(tk) - (1 - len(tk) % 2))
        xs = savgol_filter(xk, w, 3, axis=0) if w >= 5 else xk
        v = np.gradient(xs, tk, axis=0)
        a = np.gradient(v, tk, axis=0)
        j = np.gradient(a, tk, axis=0)
        j2 = j[:, 0] ** 2 + j[:, 1] ** 2
        jerk_all.append(np.sqrt(j2))
        T = tk[-1] - tk[0]
        D = float(np.sum(np.hypot(*np.diff(xs, axis=0).T)))
        if D > 0:
            dj.append(T ** 5 / D ** 2 * float(np.trapz(j2, tk)))
            weights.append(T)
    if not jerk_all:
        return {"mean_jerk": np.nan, "sd_jerk": np.nan, "dimensionless_jerk": np.nan}
    allj = np.concatenate(jerk_all)
    return {"mean_jerk": float(allj.mean()), "sd_jerk": float(allj.std(ddof=1)),
            "dimensionless_jerk": float(np.average(dj, weights=weights)) if dj else np.nan}


def dimensionless_jerk(t, xy, fs, movement_time, path_length) -> float:
    """computereachmetrics.m: posizione lisciata (SG ordine 4) e poi derivata terza (SG),
    sqrt(mean(|j|^2) * MT^6 / L^2 / 2) con L = lunghezza del percorso."""
    if len(t) < 3 or not path_length or not movement_time:
        return np.nan
    tu, u = _uniform(t, xy, fs)
    w = _odd_window(JERK_WINDOW_S, fs, JERK_ORDER, len(tu))
    if not w:
        return np.nan
    traj = savgol_filter(u, w, JERK_ORDER, axis=0, mode="nearest")
    j = savgol_filter(traj, w, JERK_ORDER, deriv=3, delta=1.0 / fs, axis=0, mode="nearest")
    mjerk = (j[:, 0] ** 2 + j[:, 1] ** 2) * movement_time ** 6 / path_length ** 2
    return float(math.sqrt(np.mean(mjerk) / 2))


def position_at(t, xy, when):
    """Posizione interpolata al tempo `when`; l'ultima se il segnale finisce prima
    (come computereachmetrics.m quando MT < 1 s)."""
    if when >= t[-1]:
        return xy[-1]
    return np.array([np.interp(when, t, xy[:, 0]), np.interp(when, t, xy[:, 1])])


def reach_metrics(t, xy, target, t_enter, hidden_s, fs) -> dict:
    """Un movimento: t, xy = tutti i frame dal target mostrato (t[0]) alla fine del goal
    (dwell completato); t_enter = entrata nel target che completa il dwell."""
    m = {k: np.nan for k in TRIAL_KEYS}
    t = np.asarray(t, float)
    xy = np.asarray(xy, float)
    target = np.asarray(target, float)
    if len(t) < 2:
        return m
    start = xy[0]
    # a HIDDEN_S dall'inizio (il cursore ricompare): errori di feedforward
    p1 = position_at(t, xy, t[0] + hidden_s)
    to_target, to_cursor = target - start, p1 - start
    m["euclidean_error_1s"] = float(np.hypot(*(p1 - target)))
    m["extent_error_1s"] = float(np.hypot(*to_cursor) - np.hypot(*to_target))
    if np.hypot(*to_cursor) > 1e-9 and np.hypot(*to_target) > 1e-9:
        cross = to_target[0] * to_cursor[1] - to_target[1] * to_cursor[0]
        m["direction_error_1s"] = math.degrees(math.atan2(abs(cross), float(np.dot(to_target, to_cursor))))
    # tutto il movimento: dall'inizio all'entrata nel target
    mv = t <= t_enter + 1e-9
    tm, xm = t[mv], xy[mv]
    if len(tm) < 2:
        return m
    mt = float(tm[-1] - tm[0])
    m["movement_time"] = mt
    m.update(speed_stats(tm, xm))
    path = float(np.sum(np.hypot(*np.diff(xm, axis=0).T)))   # computereachmetrics.m: tutti i passi
    straight = float(np.hypot(*(xm[-1] - xm[0])))
    if straight > 1e-9:
        m["linearity_index"] = path / straight - 1
        d = (xm[-1] - xm[0]) / straight
        rel = xm - xm[0]
        m["aspect_ratio"] = float(np.max(np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]))) / straight
    m["dimensionless_jerk"] = dimensionless_jerk(tm, xm, fs, mt, path)
    peaks = speed_peaks(tm, xm, fs)
    m["n_speed_peaks"] = peaks
    m["smoothness"] = 1.0 / peaks if peaks and peaks == peaks else np.nan
    return m


def count_movements(regions) -> int:
    """movement_train.m: NaN tolti, la prima regione conta 1, +1 a ogni cambio."""
    reg = pd.to_numeric(pd.Series(regions), errors="coerce").dropna().to_numpy()
    return int(1 + np.sum(np.diff(reg) != 0)) if reg.size else 0


def session_metrics(t, xy, regions, fs) -> dict:
    """Un log intero (tutti i frame): gli script del laboratorio che lavorano sul log
    completo. tt.m (durata), cursor_distance.m (percorso), speed1.m (velocita'),
    jerk1.m (jerk), smooth.m (picchi), movement_train.m (cambi di regione)."""
    t, xy = np.asarray(t, float), np.asarray(xy, float)
    out = {"session_log_duration": float(t[-1] - t[0]) if len(t) else np.nan}
    out.update({f"session_{k}": v for k, v in speed_stats(t, xy).items()})
    out.update({f"session_{k}": v for k, v in session_jerk(t, xy, fs).items()})
    peaks = speed_peaks(t, xy, fs)
    out["session_speed_peaks"] = peaks
    out["session_smoothness"] = 1.0 / peaks if peaks else np.nan
    out["session_region_changes"] = count_movements(regions)
    return out


# ------------------------------------------------ metriche in piu' del task ---
# Non sono negli script del laboratorio: stesse definizioni di reaching_metrics.py /
# reaching_regions.py del task, ma ricalcolate qui dal raw.

INITIAL_DIRECTION_S = 0.2   # reaching_metrics.INITIAL_DIRECTION_S


def angle_between(a, b) -> float:
    """Angolo [deg, 0-180] fra due vettori (NaN se uno e' nullo), come direction_error
    di computereachmetrics.m: atan2(|a x b|, a . b)."""
    if np.hypot(*a) < 1e-9 or np.hypot(*b) < 1e-9:
        return np.nan
    return math.degrees(math.atan2(abs(a[0] * b[1] - a[1] * b[0]), float(np.dot(a, b))))


def onset_metrics(t, xy, goal, onset_speed, fs) -> dict:
    """reaction_time: comparsa del target -> prima velocita' > onset_speed (velocita' da
    np.gradient, come il task); initial_direction_error: direzione inizio movimento ->
    posizione 0.2 s dopo, rispetto a inizio movimento -> target; _peak: lo stesso con
    la posizione al primo picco di velocita' (picchi di smooth.m)."""
    out = {"reaction_time": np.nan, "initial_direction_error": np.nan, "initial_direction_error_peak": np.nan}
    t, xy, goal = np.asarray(t, float), np.asarray(xy, float), np.asarray(goal, float)
    if len(t) < 3 or np.any(np.diff(t) <= 0):
        return out
    speed = np.hypot(np.gradient(xy[:, 0], t), np.gradient(xy[:, 1], t))
    moving = np.flatnonzero(speed > onset_speed)
    if moving.size == 0:
        return out
    i0 = int(moving[0])
    out["reaction_time"] = float(t[i0] - t[0])
    p0 = xy[i0]
    if t[i0] + INITIAL_DIRECTION_S <= t[-1]:
        out["initial_direction_error"] = angle_between(position_at(t, xy, t[i0] + INITIAL_DIRECTION_S) - p0, goal - p0)
    peaks = speed_peak_times(t[i0:], xy[i0:], fs)
    if peaks.size:
        out["initial_direction_error_peak"] = angle_between(position_at(t, xy, peaks[0]) - p0, goal - p0)
    return out


def speed_peak_times(t, xy, fs) -> np.ndarray:
    """Tempi dei picchi di velocita' con le regole di smooth.m (vedi speed_peaks)."""
    if len(t) < 3:
        return np.array([])
    tu, u = _uniform(t, xy, fs)
    w = _odd_window(SPEED_WINDOW_S, fs, SPEED_ORDER, len(tu))
    if not w:
        return np.array([])
    v = savgol_filter(u, w, SPEED_ORDER, deriv=1, delta=1.0 / fs, axis=0, mode="nearest")
    speed = savgol_filter(np.hypot(v[:, 0], v[:, 1]), w, SPEED_ORDER, mode="nearest")
    speed[speed < 0] = 0
    speed[0] = 0
    if speed.max() <= 0:
        return np.array([])
    peaks, _ = find_peaks(speed, height=PEAK_FRACTION * speed.max(), distance=max(1, round(PEAK_DISTANCE_S * fs)))
    return tu[peaks]


def region_of(x, y, x_edges, y_edges) -> int:
    """Regione 1-9 (1 in alto a sinistra) di un punto del canvas, come region_of del task."""
    col = min(max(int(np.searchsorted(x_edges, x, side="right")) - 1, 0), 2)
    row = min(max(int(np.searchsorted(y_edges, y, side="right")) - 1, 0), 2)
    return row * 3 + col + 1


def visited_regions(t, regions, target_region, min_visit_s, home_region=5) -> list:
    """Regioni (fuori dal centro) visitate in ordine: contano se il cursore ci resta
    almeno min_visit_s, il target subito (visited_regions del task)."""
    visited, i, n = [], 0, len(regions)
    while i < n:
        j = i
        while j + 1 < n and regions[j + 1] == regions[i]:
            j += 1
        r = regions[i]
        if r != home_region and (r == target_region or t[j] - t[i] >= min_visit_s):
            if not visited or visited[-1] != r:
                visited.append(r)
        i = j + 1
    return visited


def blind_region_metrics(g: dict, hidden_s, min_visit_s, x_edges, y_edges) -> dict:
    """Metriche a regioni di un target del training (blind_metrics del task), dal raw:
    reached_hidden, region_at_reveal(_correct), first_region(_correct), n_wrong_regions."""
    t, xy, target_region = g["t"], g["xy"], g["target_region"]
    t_reveal = t[0] + hidden_s
    out = {"reached_hidden": 100.0 if g["t_enter"] < t_reveal else 0.0}
    if g["t_enter"] < t_reveal:
        at_reveal = target_region
    else:
        p = position_at(t, xy, t_reveal)
        at_reveal = region_of(p[0], p[1], x_edges, y_edges)
    out["region_at_reveal"] = at_reveal
    out["region_at_reveal_correct"] = 100.0 if at_reveal == target_region else 0.0
    visited = visited_regions(t, g["regions"], target_region, min_visit_s)
    out["first_region"] = visited[0] if visited else np.nan
    out["first_region_correct"] = (100.0 if visited[0] == target_region else 0.0) if visited else np.nan
    out["n_wrong_regions"] = len(set(visited[:visited.index(target_region)])) if target_region in visited else np.nan
    return out


def goals_from_raw(raw: pd.DataFrame, x_edges=None, y_edges=None) -> list:
    """I goal di una sessione del training dal _raw.csv, in ordine: dict con kind
    ('region'/'home'), t_shown, t_enter (entrata che completa il dwell), t_first_enter,
    n_entries, t_end (dwell completato), i frame (t, xy, regions) da t_shown a t_end,
    il centro del goal e la sua regione. Il primo home (avvio della sessione) e' incluso."""
    ev = raw[["time", "event"]].dropna()
    ev = ev.assign(event=ev["event"].str.split(";")).explode("event")
    goals, current, enters = [], None, []
    for time, e in zip(ev["time"], ev["event"]):
        if e in ("target_on", "home_on"):
            current = {"kind": "region" if e == "target_on" else "home", "t_shown": float(time)}
            enters = []
        elif e == "enter_target" and current is not None:
            enters.append(float(time))
        elif e in ("target_acquired", "home_acquired") and current is not None:
            current["t_end"] = float(time)
            current["t_enter"] = enters[-1] if enters else float(time)
            current["t_first_enter"] = enters[0] if enters else float(time)
            current["n_entries"] = max(len(enters), 1)
            goals.append(current)
            current = None
    t = raw["time"].to_numpy(float)
    xy = raw[["cursor_x", "cursor_y"]].to_numpy(float)
    reg = pd.to_numeric(raw["region"], errors="coerce").to_numpy()
    for g in goals:
        sel = (t >= g["t_shown"] - 1e-9) & (t <= g["t_end"] + 1e-9)
        g["t"], g["xy"], g["regions"] = t[sel], xy[sel], reg[sel]
        first = np.flatnonzero(sel)[0]
        g["goal"] = (float(raw["target_x"].iloc[first]), float(raw["target_y"].iloc[first]))
        g["dwell_time"] = g["t_end"] - g["t_first_enter"]
        if x_edges is not None:
            g["target_region"] = region_of(*g["goal"], x_edges, y_edges)
    return goals
