"""Metriche di reaching calcolate dai file _raw.csv (un frame per riga), con le
definizioni di Bellitto et al. 2025 (J. Neural Eng. 22 046036, par. 2.4.2) e dello
script del laboratorio computereachmetrics.m:

  euclidean_error   distanza fra il centro del target e il cursore all'istante di
                    misura (cm nel paper, qui px del canvas)
  extent_error      |partenza -> cursore| - |partenza -> target| (< 0 = cursore corto)
  direction_error   angolo [deg, 0-180] fra partenza -> target e partenza -> cursore
                    (atan2(|det|, dot) come computereachmetrics.m)
  movement_duration tempo per raggiungere il target
  linearity_index   lunghezza del percorso / distanza fra il primo e l'ultimo punto - 1
                    (0 = rettilineo; il "- 1" e' quello del paper, lo script da' il rapporto)

"Partenza" = posizione del cursore quando compare il target (primo campione del
movimento, come P1 = CursorXY(1,:) in computereachmetrics.m). L'istante di misura
cambia fra i task: 1 s nel training (quando il cursore ricompare; nel paper 0.5 s),
la fine dei 4 s nel blind test. Le posizioni a un istante sono interpolate.

Errori e distanze sono in cm come nel paper (canvas_to_cm): il canvas (1200 x 650 px) e'
disegnato a scala canvas_scale (dal summary.json, 1.53) sullo schermo del portatile,
1920 x 1080 px = 340 x 190 mm (xrandr), quindi 1 px del canvas = 1.53 x 0.177 mm.

Per il robot: speed_stats (speed1.m / cursor_distance.m), session_jerk (jerk1.m),
count_movements (movement_train.m).
"""

import math

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

GAP_S = 0.2   # cursor_distance.m / speed1.m: passi con dt > 0.2 s esclusi
SCREEN_PX = (1920, 1080)     # risoluzione dello schermo dei task
SCREEN_MM = (340.0, 190.0)   # dimensione fisica dello stesso schermo
CANVAS_SCALE_DEFAULT = 1.53  # px dello schermo per px del canvas (summary.json: setup.screen.canvas_scale)


def canvas_scale(summary: dict) -> float:
    return float(summary.get("setup", {}).get("screen", {}).get("canvas_scale", CANVAS_SCALE_DEFAULT))


def canvas_to_cm(xy, scale: float) -> np.ndarray:
    """Coordinate del canvas (px) -> cm sullo schermo (x e y con il proprio passo dei pixel)."""
    k = np.array([SCREEN_MM[0] / SCREEN_PX[0], SCREEN_MM[1] / SCREEN_PX[1]]) * scale / 10.0
    return np.asarray(xy, float) * k


def position_at(t, xy, when):
    """Posizione interpolata al tempo `when`; l'ultima se il segnale finisce prima
    (come computereachmetrics.m quando il movimento dura meno del tempo di misura)."""
    if when >= t[-1]:
        return xy[-1]
    return np.array([np.interp(when, t, xy[:, 0]), np.interp(when, t, xy[:, 1])])


def endpoint_errors(start, cursor, target) -> dict:
    """Euclidean, extent e direction error del cursore rispetto al target, misurati
    dalla posizione di partenza."""
    start, cursor, target = (np.asarray(v, float) for v in (start, cursor, target))
    to_target, to_cursor = target - start, cursor - start
    out = {"euclidean_error": float(np.hypot(*(cursor - target))),
           "extent_error": float(np.hypot(*to_cursor) - np.hypot(*to_target)),
           "direction_error": np.nan}
    if np.hypot(*to_cursor) > 1e-9 and np.hypot(*to_target) > 1e-9:
        cross = to_target[0] * to_cursor[1] - to_target[1] * to_cursor[0]
        out["direction_error"] = math.degrees(math.atan2(abs(cross), float(np.dot(to_target, to_cursor))))
    return out


def linearity_index(xy) -> float:
    """Percorso (somma di tutti i passi) / distanza primo-ultimo punto - 1."""
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return np.nan
    path = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
    straight = float(np.hypot(*(xy[-1] - xy[0])))
    return path / straight - 1 if straight > 1e-9 else np.nan


def speed(t, xy) -> np.ndarray:
    """Velocita' istantanea (np.gradient, come il task per l'inizio del movimento)."""
    return np.hypot(np.gradient(xy[:, 0], t), np.gradient(xy[:, 1], t))


# ------------------------------------------------------------------ training ---

def goals_from_raw(raw: pd.DataFrame) -> list:
    """I goal di una sessione del training dal _raw.csv, in ordine: dict con kind
    ('region' = andata verso il target / 'home' = ritorno al centro), t_shown,
    t_enter (entrata che completa il dwell), t_end (dwell completato), i frame
    (t, xy) da t_shown a t_end e il centro del goal."""
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
            goals.append(current)
            current = None
    t = raw["time"].to_numpy(float)
    xy = raw[["cursor_x", "cursor_y"]].to_numpy(float)
    for g in goals:
        sel = (t >= g["t_shown"] - 1e-9) & (t <= g["t_end"] + 1e-9)
        g["t"], g["xy"] = t[sel], xy[sel]
        first = np.flatnonzero(sel)[0]
        g["goal"] = (float(raw["target_x"].iloc[first]), float(raw["target_y"].iloc[first]))
    return goals


def training_metrics(t, xy, target, t_enter, at_s) -> dict:
    """Un movimento di andata: t, xy = frame dal target mostrato (t[0]) alla fine del
    dwell; t_enter = entrata nel target che completa il dwell; errori a at_s secondi
    dalla comparsa del target; movement duration e linearity index dalla comparsa
    all'entrata nel target."""
    t, xy = np.asarray(t, float), np.asarray(xy, float)
    out = {"euclidean_error": np.nan, "extent_error": np.nan, "direction_error": np.nan,
           "movement_duration": np.nan, "linearity_index": np.nan}
    if len(t) < 2:
        return out
    out.update(endpoint_errors(xy[0], position_at(t, xy, t[0] + at_s), target))
    mv = t <= t_enter + 1e-9
    if mv.sum() >= 2:
        out["movement_duration"] = float(t[mv][-1] - t[0])
        out["linearity_index"] = linearity_index(xy[mv])
    return out


# --------------------------------------------------------------------- robot ---

def speed_stats(t, xy, gap_s=GAP_S) -> dict:
    """speed1.m e cursor_distance.m: percorso e velocita' istantanea (passi con
    0 < dt <= gap_s)."""
    d = np.hypot(*np.diff(xy, axis=0).T)
    dt = np.diff(t)
    ok = np.isfinite(d) & (dt > 0) & (dt <= gap_s)
    d, dt = d[ok], dt[ok]
    if d.size == 0:
        return {"path_length": np.nan, "mean_speed": np.nan}
    return {"path_length": float(d.sum()), "mean_speed": float((d / dt).mean())}


def session_jerk(t, xy, gap_s=GAP_S) -> dict:
    """jerk1.m: il log diviso in tratti continui (buchi > gap_s, tratti < 1 s scartati),
    posizione lisciata con SG (ordine 3, 11 campioni), derivate con il gradiente;
    jerk medio su tutti i campioni e dimensionless jerk T^5 / D^2 * int |j|^2 dt per
    tratto (D = percorso), media pesata sulla durata."""
    t, xy = np.asarray(t, float), np.asarray(xy, float)
    breaks = np.flatnonzero(~((np.diff(t) > 0) & (np.diff(t) <= gap_s))) + 1
    jerk_all, dj, weights = [], [], []
    for idx in np.split(np.arange(len(t)), breaks):
        if len(idx) < 10 or t[idx[-1]] - t[idx[0]] < 1.0:
            continue
        tk, xk = t[idx], xy[idx]
        w = min(11, len(tk) - (1 - len(tk) % 2))
        xs = savgol_filter(xk, w, 3, axis=0) if w >= 5 else xk
        j = np.gradient(np.gradient(np.gradient(xs, tk, axis=0), tk, axis=0), tk, axis=0)
        j2 = j[:, 0] ** 2 + j[:, 1] ** 2
        jerk_all.append(np.sqrt(j2))
        T = tk[-1] - tk[0]
        D = float(np.sum(np.hypot(*np.diff(xs, axis=0).T)))
        if D > 0:
            dj.append(T ** 5 / D ** 2 * float(np.trapz(j2, tk)))
            weights.append(T)
    if not jerk_all:
        return {"mean_jerk": np.nan, "dimensionless_jerk": np.nan}
    return {"mean_jerk": float(np.concatenate(jerk_all).mean()),
            "dimensionless_jerk": float(np.average(dj, weights=weights)) if dj else np.nan}


def count_movements(regions) -> int:
    """movement_train.m: NaN tolti, la prima regione conta 1, +1 a ogni cambio."""
    reg = pd.to_numeric(pd.Series(regions), errors="coerce").dropna().to_numpy()
    return int(1 + np.sum(np.diff(reg) != 0)) if reg.size else 0
