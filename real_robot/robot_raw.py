"""Metriche di guida del robot reale ricalcolate dai file raw di una run:
    <soggetto>_run<N>_regions.csv    regione del cursore a ogni tick del controllo (20 Hz),
                                     per tratto di guida (stretch), con la modalita' del robot
    <soggetto>_run<N>_odometry.csv   odometria della base (x, y, theta, vx, vy, vtheta)

Solo le due navigazioni (la modalita' del robot dice la fase):
    andata  = max_speed + reduced_speed (dalla partenza alla selezione dell'oggetto)
    ritorno = transport
Per ognuna:
    path_length         percorso della base dall'odometria (m): somma di tutti i passi fra
                        campioni consecutivi della fase, anche dopo un buco nei campioni (le
                        posizioni sono assolute: lo spostamento c'e' stato davvero). Il passo
                        fra l'ultimo campione di un'altra fase e il primo di questa NON conta
                        (il task invece lo attribuisce alla fase nuova)
    mean_speed          velocita' media della base (m/s), speed1.m: velocita' istantanea dei
                        passi con buchi fino a 1 s
    jerk                jerk medio della base (m/s^3), jerk1.m sull'odometria
    region_transitions  numero di cambi di regione del cursore nell'interfaccia
                        (movement_train.m sui tick, come Najie et al.: ogni cambio)
Metriche esplorative (per la discussione: perche' la base va piu' veloce nel test 2?):
    stop_percent        % del tempo di guida con il cursore nella regione di stop (5)
    transitions_per_min region_transitions / tempo di guida: i cambi calano solo perche' si
                        guida per meno tempo, o il comando cambia davvero meno spesso?
    cursor_eccentricity distanza media del cursore dal centro dell'interfaccia, fuori dalla
                        regione di stop, normalizzata (0 = centro, 1 = bordo): la velocita'
                        comandata cresce con questa distanza
Il tempo di guida e' la somma degli intervalli fra tick consecutivi (buchi > TICK_GAP_S esclusi).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from common.reach_metrics import count_movements, session_jerk, speed_stats

NAVIGATION = {"andata": ("max_speed", "reduced_speed"), "ritorno": ("transport",)}
ODOM_GAP_S = 1.0
TICK_GAP_S = 0.5      # config.region_tick_max_gap_s del task
CURSOR_PHASES = {"andata": ("outbound", "approach"), "ritorno": ("return",)}


def navigation_metrics(json_path: Path) -> tuple:
    """{(metrica, fase): valore} di una run e i file raw mancanti."""
    stem = str(json_path)[:-len("_session.json")]
    files = {k: Path(f"{stem}_{k}.csv") for k in ("regions", "odometry", "cursor")}
    missing = [p.name for p in files.values() if not p.exists()]
    data = {k: pd.read_csv(p) for k, p in files.items() if p.exists()}
    out = {}
    for phase, modes in NAVIGATION.items():
        if "odometry" in data:
            od = data["odometry"]
            od = od[od["mode"].isin(modes)]
            if len(od) >= 2:
                t, xy = od["t_test"].to_numpy(float), od[["x", "y"]].to_numpy(float)
                out[("path_length", phase)] = speed_stats(t, xy, np.inf)["path_length"]
                out[("mean_speed", phase)] = speed_stats(t, xy, ODOM_GAP_S)["mean_speed"]
                out[("jerk", phase)] = session_jerk(t, xy)["mean_jerk"]
        if "regions" in data:
            reg = data["regions"]
            reg = reg[reg["mode"].isin(modes)]
            # i cambi non attraversano due tratti di guida diversi
            out[("region_transitions", phase)] = float(sum(count_movements(g["region"]) - 1
                                                           for _, g in reg.groupby("stretch")))
            t = reg["t_test"].to_numpy(float)
            dt = np.diff(t, append=t[-1]) if len(t) else np.array([])
            dt[(dt < 0) | (dt > TICK_GAP_S)] = 0.0
            driving = float(dt.sum())
            if driving > 0:
                out[("stop_percent", phase)] = 100.0 * float(dt[reg["region"].to_numpy() == 5].sum()) / driving
                out[("transitions_per_min", phase)] = 60.0 * out[("region_transitions", phase)] / driving
        if "cursor" in data:
            import json
            screen = json.loads(Path(json_path).read_text(encoding="utf-8")).get("screen", {})
            w, h = float(screen.get("canvas_width", 2550)), float(screen.get("canvas_height", 1500))
            cur = data["cursor"]
            cur = cur[cur["phase"].isin(CURSOR_PHASES[phase]) & (cur["region"] != 5)]
            if len(cur):
                out[("cursor_eccentricity", phase)] = float(np.mean(np.hypot((cur["cursor_x"] - w / 2) / (w / 2),
                                                                             (cur["cursor_y"] - h / 2) / (h / 2))))
    return out, missing
