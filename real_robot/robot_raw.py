"""Metriche del robot reale ricalcolate dai file raw di una run:
    <soggetto>_run<N>_regions.csv    regione del cursore a ogni tick del controllo, per tratto
                                     di guida (stretch), con la modalita' e il dwell da togliere
    <soggetto>_run<N>_odometry.csv   odometria della base (x, y, theta, vx, vy, vtheta)
    <soggetto>_run<N>_cursor.csv     ogni frame della webcam: cursore e regione

Per ogni fase di guida (outbound, approach, return, repositioning; la modalita'
del robot dice la fase) e per tutta la guida ("all"):
- metriche di guida del task (session_metrics.py del repo, portate qui):
  driving_time, stop_time/percent, n_commands, n_command_changes, command_changes_per_min,
  mean/median_command_duration, non_adjacent_percent, reversal_percent,
  stop_passage_percent, sequence_entropy (comandi = visite di almeno MIN_COMMAND_S);
- script del laboratorio (common/reach_metrics.py):
  region_changes             movement_train.m sui tick (ogni cambio di regione, nessun filtro)
  robot_path_length          cursor_distance.m sull'odometria [m] (buchi fino a 1 s inclusi)
  robot_lin_speed_*, robot_ang_speed_*   speed1.m: |v lineare| [m/s] e |v angolare| [rad/s]
                             (dall'odometria: velocita' misurate, non i comandi inviati)
  robot_mean_jerk, robot_log10_dimensionless_jerk   jerk1.m sull'odometria
  cursor_*                   tt.m, cursor_distance.m, speed1.m, jerk1.m, smooth.m sul cursore [px]
- cumulative_rotation        somma di |d theta| dall'odometria [rad] (metrica del task)
Le durate delle fasi NON si ricavano dal raw (mancano gli istanti degli eventi):
restano quelle del _session.json.
"""

import collections
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common.reach_metrics import count_movements, session_jerk, session_metrics, speed_stats

MODE_PHASE = {"max_speed": "outbound", "reduced_speed": "approach", "transport": "return",
              "repositioning": "repositioning"}
PHASES = ("outbound", "approach", "return", "repositioning")
MIN_COMMAND_S = 0.25
# l'odometria da' posizioni assolute: un buco nei campioni non e' un dropout come per il
# cursore (0.2 s di cursor_distance.m), il robot si e' davvero spostato. Si spezza solo
# fra tratti di guida diversi (fermi di almeno 1 s)
ODOM_GAP_S = 1.0
STOP_REGION = 5
REGION_SIGNS = {1: (1, 1), 2: (1, 0), 3: (1, -1), 4: (0, 1), 5: (0, 0), 6: (0, -1),
                7: (-1, 1), 8: (-1, 0), 9: (-1, -1)}


def adjacent(a, b) -> bool:
    ra, ca = divmod(a - 1, 3)
    rb, cb = divmod(b - 1, 3)
    return max(abs(ra - rb), abs(ca - cb)) <= 1


def region_visits(t, regions, trim_end=0.0) -> list:
    """[[regione, secondi], ...] di un tratto: ogni tick vale fino al successivo; l'ultima
    visita (lo stop del dwell completato) si accorcia di trim_end."""
    visits = []
    for i, r in enumerate(regions):
        held = t[i + 1] - t[i] if i + 1 < len(t) else 0.0
        if visits and visits[-1][0] == r:
            visits[-1][1] += held
        else:
            visits.append([r, held])
    if trim_end > 0 and visits and visits[-1][0] == STOP_REGION:
        visits[-1][1] = max(0.0, visits[-1][1] - trim_end)
    return visits


def commands(visits, min_s=MIN_COMMAND_S) -> list:
    out = []
    for r, s in visits:
        if s < min_s:
            continue
        if out and out[-1][0] == r:
            out[-1][1] += s
        else:
            out.append([r, s])
    return out


def command_metrics(stretches: list, driving_time: float) -> dict:
    """command_metrics di session_metrics.py (stesse definizioni)."""
    n_commands = n_changes = n_non_adjacent = n_motion = n_via_stop = n_reversals = 0
    durations, pairs = [], collections.Counter()
    for cmds in stretches:
        n_commands += len(cmds)
        durations += [s for r, s in cmds if r != STOP_REGION]
        for (a, _), (b, _) in zip(cmds, cmds[1:]):
            n_changes += 1
            pairs[(a, b)] += 1
            n_non_adjacent += not adjacent(a, b)
        last_lin = last_ang = 0
        first, stop_before = True, False
        for r, _ in cmds:
            if r == STOP_REGION:
                stop_before = True
                continue
            lin, ang = REGION_SIGNS[r]
            if not first:
                n_motion += 1
                n_via_stop += stop_before
                if (lin and last_lin and lin != last_lin) or (ang and last_ang and ang != last_ang):
                    n_reversals += 1
            last_lin, last_ang = lin or last_lin, ang or last_ang
            first, stop_before = False, False
    entropy = np.nan
    if n_changes:
        frm = collections.Counter()
        for (a, _), c in pairs.items():
            frm[a] += c
        entropy = -sum((c / n_changes) * math.log2(c / frm[a]) for (a, _), c in pairs.items()) + 0.0
    pct = lambda n, d: 100.0 * n / d if d else np.nan
    return {"n_commands": n_commands, "n_command_changes": n_changes,
            "command_changes_per_min": 60.0 * n_changes / driving_time if driving_time > 0 else np.nan,
            "mean_command_duration": float(np.mean(durations)) if durations else np.nan,
            "median_command_duration": float(np.median(durations)) if durations else np.nan,
            "non_adjacent_percent": pct(n_non_adjacent, n_changes),
            "reversal_percent": pct(n_reversals, n_motion),
            "stop_passage_percent": pct(n_via_stop, n_motion),
            "sequence_entropy": entropy}


def driving_metrics(reg: pd.DataFrame) -> dict:
    """Tick di una fase (uno o piu' tratti) -> metriche di guida del task + movement_train.m."""
    stretches, driving_time, stop_time, changes = [], 0.0, 0.0, 0
    for _, g in reg.groupby("stretch", sort=True):
        t, r = g["t_test"].to_numpy(float), g["region"].astype(int).to_numpy()
        visits = region_visits(t, r, float(g["stretch_dwell_s"].iloc[0]))
        driving_time += sum(s for _, s in visits)
        stop_time += sum(s for rr, s in visits if rr == STOP_REGION)
        stretches.append(commands(visits))
        changes += count_movements(r) - 1          # i cambi non attraversano due tratti
    out = {"driving_time": driving_time, "stop_time": stop_time,
           "stop_percent": 100.0 * stop_time / driving_time if driving_time > 0 else np.nan}
    out.update(command_metrics(stretches, driving_time))
    out["region_changes"] = changes                 # movement_train.m: ogni cambio, nessun filtro
    out["region_changes_per_min"] = 60.0 * changes / driving_time if driving_time > 0 else np.nan
    return out


def odometry_metrics(od: pd.DataFrame) -> dict:
    """speed1.m, cursor_distance.m, jerk1.m sull'odometria; rotazione cumulata."""
    if len(od) < 2:
        return {}
    t = od["t_test"].to_numpy(float)
    xy = od[["x", "y"]].to_numpy(float)
    ok = (np.diff(t) > 0) & (np.diff(t) <= ODOM_GAP_S)
    dtheta = np.abs(np.diff(np.unwrap(od["theta"].to_numpy(float))))
    out = {"robot_path_length": speed_stats(t, xy, ODOM_GAP_S)["path_length"],
           "cumulative_rotation": float(dtheta[ok].sum())}
    for name, v in (("lin", np.hypot(od["vx"], od["vy"]).to_numpy(float)), ("ang", od["vtheta"].abs().to_numpy(float))):
        out[f"robot_{name}_speed_mean"] = float(np.mean(v))
        out[f"robot_{name}_speed_sd"] = float(np.std(v, ddof=1))
        out[f"robot_{name}_speed_peak"] = float(np.max(v))
    fs = 1.0 / float(np.median(np.diff(t)))
    j = session_jerk(t, xy, fs)
    out["robot_mean_jerk"] = j["mean_jerk"]
    out["robot_log10_dimensionless_jerk"] = math.log10(j["dimensionless_jerk"]) \
        if j["dimensionless_jerk"] and j["dimensionless_jerk"] > 0 else np.nan
    return out


def cursor_metrics(cur: pd.DataFrame) -> dict:
    if len(cur) < 3:
        return {}
    t = cur["t_test"].to_numpy(float)
    fs = 1.0 / float(np.median(np.diff(t)))
    m = session_metrics(t, cur[["cursor_x", "cursor_y"]], cur["region"], fs)
    keep = ("path_length", "mean_speed", "sd_speed", "peak_speed", "mean_jerk", "speed_peaks")
    return {f"cursor_{k}": m[f"session_{k}"] for k in keep}


def raw_metrics(json_path: Path) -> tuple:
    """{"raw.<fase>.<metrica>": valore} di una run e i file raw mancanti."""
    stem = str(json_path)[:-len("_session.json")]
    files = {k: Path(f"{stem}_{k}.csv") for k in ("regions", "odometry", "cursor")}
    missing = [p.name for p in files.values() if not p.exists()]
    data = {k: pd.read_csv(p) for k, p in files.items() if p.exists()}
    out = {}
    for ph in PHASES + ("all",):
        vals = {}
        if "regions" in data:
            reg = data["regions"]
            reg = reg[reg["mode"].map(MODE_PHASE).isin(PHASES if ph == "all" else (ph,))]
            if len(reg):
                vals.update(driving_metrics(reg))
        if "odometry" in data:
            od = data["odometry"]
            od = od[od["mode"].map(MODE_PHASE).isin(PHASES if ph == "all" else (ph,))]
            vals.update(odometry_metrics(od))
        if "cursor" in data:
            cur = data["cursor"]
            cur = cur[cur["phase"].isin(PHASES if ph == "all" else (ph,))]
            vals.update(cursor_metrics(cur))
        out.update({f"raw.{ph}.{k}": v for k, v in vals.items()})
    return out, missing
