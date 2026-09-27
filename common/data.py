"""Caricamento dei file prodotti dagli script di reachy2_teleop_grasping_and_simulation.

I file si possono incollare ovunque dentro la cartella di un'analisi (per
esempio reaching_blind/results/): vengono cercati in tutte le sottocartelle,
tranne statistics/ (l'output).

Se un soggetto ha ripetuto una sessione, gli script del robot aggiungono _1,
_2, ... al nome del file: qui si tiene sempre l'ULTIMA ripetizione (quella con
il numero piu' alto), e le altre vengono elencate come scartate.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_id(raw) -> str:
    """ID del soggetto in forma unica, per unire file e questionari:
    ' s1 ', 'S01', 'S001' -> 'S001'; ogni altro ID viene solo messo in maiuscolo."""
    s = str(raw).strip().upper()
    m = re.fullmatch(r"S?\s*0*(\d+)", s)
    return f"S{int(m.group(1)):03d}" if m else s


def find_sessions(root, pattern: str, key_groups: tuple) -> tuple:
    """File sotto `root` il cui nome rispetta `pattern` (regex con i gruppi
    `subject`, i gruppi di `key_groups` e un gruppo opzionale `rep`).
    Restituisce (scelti, scartati): scelti = {(soggetto, *chiave): Path} con
    l'ultima ripetizione di ogni sessione; scartati = lista di Path."""
    root = Path(root)
    regex = re.compile(pattern)
    found = {}
    for path in sorted(root.rglob("*")):
        if "statistics" in path.relative_to(root).parts or not path.is_file():
            continue
        m = regex.fullmatch(path.name)
        if not m:
            continue
        key = (normalize_id(m.group("subject")),) + tuple(m.group(g) for g in key_groups)
        rep = int(m.group("rep")) if m.group("rep") else 0
        found.setdefault(key, []).append((rep, path))
    chosen, dropped = {}, []
    for key, items in found.items():
        items.sort()
        chosen[key] = items[-1][1]
        dropped += [p for _, p in items[:-1]]
    return chosen, dropped


def to_bool(series: pd.Series) -> pd.Series:
    """True/False anche quando nel csv sono stringhe ('True', 'False')."""
    return series.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))


def aggregate(values: pd.Series, how: str):
    """Un valore per soggetto da tutte le sue prove:
    'rate'   -> media di valori 0/100 = percentuale di prove riuscite
    'median' -> mediana (robusta alle prove anomale)
    'mean'   -> media (per i conteggi piccoli, spesso a 0)"""
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return np.nan
    if how == "median":
        return float(v.median())
    return float(v.mean())
