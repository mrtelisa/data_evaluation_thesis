"""Tabelle per jamovi, nel formato dei file della dottoranda (training_reaching_out.omv,
reachy_stats.omv): una riga per soggetto, una colonna per cella del disegno a misure
ripetute (es. EE_T1_B1 ... EE_T2_B12, Test1_andata ... Test2_ritorno).

Scrive sia il .csv sia il .omv. Il .omv contiene solo i dati (con la descrizione di
ogni colonna): jamovi riapre le analisi salvate senza ricalcolarle, quindi un'analisi
scritta da qui mostrerebbe risultati vuoti. Il modello (ANOVA a misure ripetute) si
imposta in jamovi come nei file della dottoranda; gli stessi numeri sono nel report.

Formato .omv (jamovi/server/formatio/omv.py): zip con
    meta           manifest (jamovi-Archive-Version)
    metadata.json  colonne (nome, tipo, descrizione) e numero di righe
    xdata.json     livelli delle colonne nominali
    data.bin       i valori colonna per colonna: double little-endian per le colonne
                   Decimal (NaN = mancante), int32 per le colonne di testo (indice del livello)
    index.html     anteprima dei risultati (vuota)
"""

import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST = ("Manifest-Version: 1.0\nData-Archive-Version: 1.0.2\njamovi-Archive-Version: 11.0\n"
            "Created-By: data_evaluation_thesis\n")
EMPTY_HTML = "<!DOCTYPE html><html><head><meta charset=\"utf-8\"></head><body></body></html>"


def _field(name, col_id, text, description):
    return {"name": name, "id": col_id, "columnType": "Data",
            "dataType": "Text" if text else "Decimal",
            "measureType": "Nominal" if text else "Continuous",
            "formula": "", "formulaMessage": "", "parentId": 0, "width": 100,
            "type": "text" if text else "number", "importName": name, "description": description,
            "transform": 0, "edits": [], "missingValues": [], "trimLevels": True}


def write_tables(table: pd.DataFrame, path_stem: Path, descriptions: dict = None):
    """table: colonna 'subject' + colonne numeriche. Scrive <stem>.csv e <stem>.omv."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path_stem.with_suffix(".csv"), index=False)
    descriptions = descriptions or {}

    fields, xdata, chunks = [], {}, []
    for i, name in enumerate(table.columns, start=1):
        col = table[name]
        if name == "subject":
            levels = list(dict.fromkeys(col.astype(str)))
            fields.append(_field(name, i, True, descriptions.get(name, "Participant")))
            xdata[name] = {"labels": [[k, lv, lv, False] for k, lv in enumerate(levels)]}
            chunks.append(struct.pack(f"<{len(col)}i", *[levels.index(v) for v in col.astype(str)]))
        else:
            v = pd.to_numeric(col, errors="coerce").to_numpy(float)
            fields.append(_field(name, i, False, descriptions.get(name, "")))
            chunks.append(struct.pack(f"<{len(v)}d", *np.where(np.isfinite(v), v, np.nan)))
    metadata = {"document": {"resultsLanguage": "en"},
                "dataSet": {"rowCount": len(table), "columnCount": len(table.columns), "removedRows": [],
                            "addedRows": [], "fields": fields, "transforms": [], "weights": None}}
    with zipfile.ZipFile(path_stem.with_suffix(".omv"), "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("meta", MANIFEST)
        z.writestr("index.html", EMPTY_HTML)
        z.writestr("metadata.json", json.dumps(metadata))
        z.writestr("xdata.json", json.dumps(xdata))
        z.writestr("data.bin", b"".join(chunks))
