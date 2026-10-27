"""Test sul robot reale (reachy_control.py), fatto due volte: test 1 e test 2
(nei file run1 e run2): metriche di ogni soggetto e test, per new_stats/new_stats_analysis.py
(statistica, grafici e tabelle della tesi).

Input: <soggetto>_run<N>[_k]_session.json, _regions.csv, _odometry.csv, _cursor.csv, ovunque dentro
questa cartella (es. real_robot/results/). Se una run e' stata ripetuta si usa l'ultima.

Fasi del compito, delimitate dagli eventi del task (session_metrics.py): andata = dalla
partenza alla selezione dell'oggetto (outbound + approach), grasp = selezione -> oggetto in
mano, ritorno con la borraccia, posizionamento (placement); la loro somma e' il tempo totale.
Andata + ritorno = tempo di guida senza la manipolazione (grasp e posizionamento).
Metriche (come Najie et al.):
    completion_time      per fase e totale (s), dal _session.json (durate degli eventi
                         del task; nel raw mancano gli istanti); solo fasi completate
    path_length          andata e ritorno (m), dall'odometria      (robot_raw.py)
    mean_speed           andata e ritorno (m/s), dall'odometria
    jerk                 andata e ritorno (m/s^3), jerk medio dall'odometria
    region_transitions   andata e ritorno, cambi di regione del cursore
    e le metriche esplorative di robot_raw.py (stop_percent, transitions_per_min, cursor_eccentricity)

Output:
    statistics/robot_subjects.csv   una riga per soggetto, run (= test), metrica e fase
                                    (fasi: andata, grasp, ritorno, posizionamento, totale, andata_ritorno)

Uso:
    python3 robot_analysis.py [--results DIR] [--out DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import find_sessions   # noqa: E402
from robot_raw import navigation_metrics   # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
PATTERN = r"(?P<subject>.+)_run(?P<run>\d)(?:_(?P<rep>\d+))?_session\.json"
# fase (nome usato anche da new_stats, come le colonne dei file jamovi della dottoranda) -> fasi del task
PHASES = {"andata": ("outbound", "approach"), "grasp": ("grasp",), "ritorno": ("return",),
          "posizionamento": ("placement",), "totale": None, "andata_ritorno": ("outbound", "approach", "return")}


def get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return np.nan
        d = d[k]
    return np.nan if d is None else d


def load(root: Path) -> tuple:
    """Una riga per soggetto, run, metrica e fase."""
    chosen, dropped = find_sessions(root, PATTERN, ("run",))
    rows, problems = [], []
    for (subject, run), path in sorted(chosen.items()):
        s = json.loads(path.read_text(encoding="utf-8"))
        run = int(run)
        if not s.get("success", False):
            problems.append(f"{subject} run {run}: not completed ({s.get('end_reason')})")
        values = {}
        for phase, parts in PHASES.items():
            if parts is None:
                values[("completion_time", phase)] = get(s, "test_duration") if s.get("success") else np.nan
            elif all(get(s, f"phase_completed.{p}") is True for p in parts):
                values[("completion_time", phase)] = float(sum(get(s, f"phase_durations.{p}") for p in parts))
        nav, missing = navigation_metrics(path)
        values.update(nav)
        if missing:
            problems.append(f"{subject} run {run}: missing {', '.join(missing)}")
        rows += [{"subject": subject, "run": run, "metric": m, "phase": ph, "value": v} for (m, ph), v in values.items()]
    return pd.DataFrame(rows), dropped, problems


def main():
    p = argparse.ArgumentParser(description="Real robot: metrics of test 1 and test 2")
    p.add_argument("--results", type=Path, default=BASE_DIR)
    p.add_argument("--out", type=Path, default=BASE_DIR / "statistics")
    args = p.parse_args()

    df, dropped, problems = load(args.results)
    if df.empty:
        print(f"No *_run<N>_session.json in {args.results}")
        return
    ids = sorted(df["subject"].unique())
    print(f"Participants: {len(ids)} ({', '.join(ids)})")
    if dropped:
        print("Repeated tests discarded (the last one is used): " + ", ".join(p.name for p in dropped))
    for msg in problems:
        print("WARNING: " + msg)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "robot_subjects.csv", index=False)
    print(f"-> {args.out / 'robot_subjects.csv'}")


if __name__ == "__main__":
    main()
