"""Blind reaching test (reaching_blind.py), ripetuto 5 volte nel protocollo: metriche di ogni
prova, per new_stats/new_stats_analysis.py (statistica, grafici e tabelle della tesi).

Input: <soggetto>_blind_test<N>[_k]_raw.csv (+ _summary.json con la configurazione del
test), ovunque dentro questa cartella (es. reaching_blind/results/). Se un test e'
stato ripetuto si usa l'ultima ripetizione. Ogni test: 12 prove da 4 s, cursore sempre
nascosto; ogni target parte da dove e' finita la prova precedente. Le prove interrotte
(piu' corte di 4 s) sono escluse.

Metriche di ogni prova (errori con le definizioni di Bellitto et al. 2025 e
computereachmetrics.m, vedi common/reach_metrics.py; "partenza" = posizione del cursore
quando compare il target):
    HR   Hit rate (%)                 cursore nel target (raggio 70 px) per 0.5 s di fila
    EE   Euclidean error at 4 s (cm)  alla fine della prova, prima del target successivo
    XE   Extent error at 4 s (cm)
    DE   Direction error at 4 s (deg)
    MD   Movement duration (s)        inizio del movimento (velocita' > soglia del task,
                                      40 px/s) -> entrata nel target (hit) o ultimo istante
                                      sopra la soglia (prove senza hit)
Metrica esplorativa della tesi:
    region_correct  % di prove che finiscono nella regione del target (la regione e' il comando
                    sul robot: senza feedback il cursore arriva almeno nella regione giusta?)
Blocchi di 4 prove: 3 per test (colonna block).

Output:
    statistics/blind_trials.csv   una riga per prova (soggetto, test, blocco, prova, target, metriche)

Uso:
    python3 blind_analysis.py [--results DIR] [--out DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import find_sessions   # noqa: E402
from common.reach_metrics import canvas_scale, canvas_to_cm, endpoint_errors, speed   # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
BLOCK_SIZE_DEFAULT = 4   # summary.json: block_size
PATTERN = r"(?P<subject>.+)_blind_test(?P<test>\d)(?:_(?P<rep>\d+))?_raw\.csv"
# se il summary.json non ha il config: i valori del task
DEFAULTS = {"trial_s": 4.0, "dwell_s": 0.5, "target_radius": 70.0, "motion_onset_speed": 40.0}


def trials_from_raw(raw: pd.DataFrame, cfg: dict, scale: float) -> pd.DataFrame:
    """Una riga per prova, tutto dal _raw.csv. Hit e ingresso nel target in px del canvas
    (come il task), errori in cm."""
    c = {k: float(cfg.get(k, v)) for k, v in DEFAULTS.items()}
    x_edges, y_edges = cfg.get("region_x_edges", [0, 400, 800, 1200]), cfg.get("region_y_edges", [0, 650 / 3, 1300 / 3, 650])
    region = lambda x, y: (min(max(int(np.searchsorted(y_edges, y, "right")) - 1, 0), 2) * 3
                           + min(max(int(np.searchsorted(x_edges, x, "right")) - 1, 0), 2) + 1)
    fs = 1.0 / float(raw["time"].diff().median())
    rows = []
    for k, g in raw.groupby("trial"):
        t_shown = (int(k) - 1) * c["trial_s"]          # "time" = secondi dal primo target; un target ogni trial_s
        g = g[g["time"] < t_shown + c["trial_s"]]
        t = g["time"].to_numpy(float)
        xy = g[["cursor_x", "cursor_y"]].to_numpy(float)
        goal = np.array([g["target_x"].iloc[0], g["target_y"].iloc[0]], float)
        inside = np.hypot(xy[:, 0] - goal[0], xy[:, 1] - goal[1]) < c["target_radius"]
        xy_cm, goal_cm = canvas_to_cm(xy, scale), canvas_to_cm(goal, scale)
        # hit: prima permanenza di dwell_s nel cerchio senza uscirne; t_enter = inizio della permanenza
        t_enter, t_in = None, None
        for ti, ok in zip(t, inside):
            t_in = (t_in if t_in is not None else ti) if ok else None
            if t_in is not None and ti - t_in >= c["dwell_s"]:
                t_enter = t_in
                break
        row = {"trial": int(k), "target": int(g["target_id"].iloc[0]),
               "complete": bool(t[-1] - t_shown >= c["trial_s"] - 2.0 / fs),
               "hit": 100.0 if t_enter is not None else 0.0}
        row.update(endpoint_errors(xy_cm[0], xy_cm[-1], goal_cm))
        row["region_correct"] = 100.0 if region(*xy[-1]) == region(*goal) else 0.0
        v = speed(t, xy) if len(t) >= 3 else np.zeros(len(t))
        moving = np.flatnonzero(v > c["motion_onset_speed"])
        row["movement_duration"] = np.nan
        if moving.size:
            t_end = t_enter if t_enter is not None else t[moving[-1]]
            if t_end > t[moving[0]]:
                row["movement_duration"] = float(t_end - t[moving[0]])
        rows.append(row)
    return pd.DataFrame(rows)


def load(root: Path) -> tuple:
    chosen, dropped = find_sessions(root, PATTERN, ("test",))
    frames, problems = [], []
    for (subject, test), path in sorted(chosen.items()):
        summary = Path(str(path)[:-len("_raw.csv")] + "_summary.json")
        info = json.loads(summary.read_text()) if summary.exists() else {}
        df = trials_from_raw(pd.read_csv(path), info.get("config", {}), canvas_scale(info))
        n_bad = int((~df["complete"]).sum())
        if n_bad:
            problems.append(f"{subject} test {test}: {n_bad} interrupted trial(s) excluded")
        df = df[df["complete"]].drop(columns="complete")
        df.insert(1, "block", (df["trial"] - 1) // int(info.get("block_size", BLOCK_SIZE_DEFAULT)) + 1)
        df.insert(0, "test", int(test))
        df.insert(0, "subject", subject)
        frames.append(df)
    trials = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return trials, dropped, problems


def main():
    p = argparse.ArgumentParser(description="Blind reaching test: metrics of every trial")
    p.add_argument("--results", type=Path, default=BASE_DIR)
    p.add_argument("--out", type=Path, default=BASE_DIR / "statistics")
    args = p.parse_args()

    trials, dropped, problems = load(args.results)
    if trials.empty:
        print(f"No *_blind_test<N>_raw.csv in {args.results}")
        return
    ids = sorted(trials["subject"].unique())
    print(f"Participants: {len(ids)} ({', '.join(ids)})")
    if dropped:
        print("Repeated tests discarded (the last one is used): " + ", ".join(p.name for p in dropped))
    for msg in problems:
        print("WARNING: " + msg)
    args.out.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.out / "blind_trials.csv", index=False)
    print(f"-> {args.out / 'blind_trials.csv'}")


if __name__ == "__main__":
    main()
