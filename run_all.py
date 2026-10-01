"""Lancia tutte le analisi:
    1. questionari (punteggi per soggetto)
    2. blind test, training, robot reale (statistiche, grafici, tabelle jamovi in jamovi/)
    3. calibrazioni BoMI

    python3 run_all.py

Ogni analisi scrive in <cartella>/statistics/ un report.txt da leggere, le
tabelle .csv e i grafici in plots/. Un'analisi senza dati viene saltata.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("questionari", "run_all.py"),
    ("reaching_blind", "blind_analysis.py"),
    ("reaching_regions", "training_analysis.py"),
    ("real_robot", "robot_analysis.py"),
    ("calibrations", "calibration_analysis.py"),
]

if __name__ == "__main__":
    failed = []
    for folder, script in STEPS:
        print(f"\n{'#' * 70}\n# {folder}/{script}\n{'#' * 70}", flush=True)
        # ogni script dalla sua cartella (i questionari importano utils.py da li')
        res = subprocess.run([sys.executable, script], cwd=ROOT / folder)
        if res.returncode != 0:
            failed.append(f"{folder}/{script}")
    print("\n" + ("Tutto fatto." if not failed else "ERRORI in: " + ", ".join(failed)))
    sys.exit(1 if failed else 0)
