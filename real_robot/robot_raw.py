"""Metriche di guida del robot reale ricalcolate dai file raw di una run:
    <soggetto>_run<N>_regions.csv    regione del cursore a ogni tick del controllo (20 Hz),
                                     per tratto di guida (stretch), con la modalita' del robot
    <soggetto>_run<N>_odometry.csv   odometria della base (x, y, theta, vx, vy, vtheta)

Solo le due navigazioni (la modalita' del robot dice la fase):
    andata  = max_speed + reduced_speed (dalla partenza alla selezione dell'oggetto)
    ritorno = transport
Per ognuna:
    path_length         percorso della base dall'odometria (m), cursor_distance.m con
                        buchi fino a 1 s (le posizioni sono assolute)
    mean_speed          velocita' media della base (m/s), speed1.m sugli stessi passi
    jerk                jerk medio della base (m/s^3), jerk1.m sull'odometria
    region_transitions  numero di cambi di regione del cursore nell'interfaccia
                        (movement_train.m sui tick, come Najie et al.: ogni cambio)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from common.reach_metrics import count_movements, session_jerk, speed_stats

NAVIGATION = {"andata": ("max_speed", "reduced_speed"), "ritorno": ("transport",)}
ODOM_GAP_S = 1.0


def navigation_metrics(json_path: Path) -> tuple:
    """{(metrica, fase): valore} di una run e i file raw mancanti."""
    stem = str(json_path)[:-len("_session.json")]
    files = {k: Path(f"{stem}_{k}.csv") for k in ("regions", "odometry")}
    missing = [p.name for p in files.values() if not p.exists()]
    data = {k: pd.read_csv(p) for k, p in files.items() if p.exists()}
    out = {}
    for phase, modes in NAVIGATION.items():
        if "odometry" in data:
            od = data["odometry"]
            od = od[od["mode"].isin(modes)]
            if len(od) >= 2:
                t, xy = od["t_test"].to_numpy(float), od[["x", "y"]].to_numpy(float)
                s = speed_stats(t, xy, ODOM_GAP_S)
                out[("path_length", phase)] = s["path_length"]
                out[("mean_speed", phase)] = s["mean_speed"]
                out[("jerk", phase)] = session_jerk(t, xy)["mean_jerk"]
        if "regions" in data:
            reg = data["regions"]
            reg = reg[reg["mode"].isin(modes)]
            # i cambi non attraversano due tratti di guida diversi
            out[("region_transitions", phase)] = float(sum(count_movements(g["region"]) - 1
                                                           for _, g in reg.groupby("stretch")))
    return out, missing
