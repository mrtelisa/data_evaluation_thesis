"""Tabelle per jamovi, nel formato dei file della dottoranda (training_reaching_out.omv,
reachy_stats.omv): una riga per soggetto, una colonna per cella del disegno a misure
ripetute (es. EE_T1_B1 ... EE_T2_B12, Test1_andata ... Test2_ritorno).

Scrive il .csv e il .omv. Il .omv contiene i dati (con la descrizione di ogni colonna) e,
se richieste, le ANOVA a misure ripetute gia' impostate come il modello della dottoranda
(jamovi_template/anovaRM.pb, preso da reachy_stats.omv): fattori e celle, eta^2 ed eta^2
parziale, test di sfericita' con Greenhouse-Geisser, post hoc sui fattori principali con
tutte le correzioni (solo dove spec['post_hoc'] li chiede).

jamovi apre le analisi salvate senza ricalcolarle (server/analyses/analyses.py:
create_from_serial). I risultati qui sono quindi vuoti e segnati "stale", come jamovi fa
nei template (analysis.py: serialize(strip_content=True)): jamovi li ricalcola tutti
quando si cambia un'impostazione generale (tema dei grafici, simbolo decimale:
instance.py _main_settings_changed) o, per una sola analisi, quando se ne cambia
un'opzione.

Formato .omv (jamovi/server/formatio/omv.py): zip con
    meta           manifest (jamovi-Archive-Version)
    metadata.json  colonne (nome, tipo, descrizione) e numero di righe
    xdata.json     livelli delle colonne nominali
    data.bin       i valori colonna per colonna: double little-endian per le colonne
                   Decimal (NaN = mancante), int32 per le colonne di testo (indice del livello)
    index.html     anteprima dei risultati (vuota)
    NN <nome>/analysis   un'analisi (protobuf AnalysisResponse, jamovi_template/jamovi.proto)
"""

import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATE_DIR = Path(__file__).resolve().parent / "jamovi_template"
sys.path.insert(0, str(TEMPLATE_DIR))
import jamovi_pb2 as jpb   # noqa: E402  (generato da jamovi.proto con protoc)

MANIFEST = ("Manifest-Version: 1.0\nData-Archive-Version: 1.0.2\njamovi-Archive-Version: 11.0\n"
            "Created-By: data_evaluation_thesis\n")
EMPTY_HTML = "<!DOCTYPE html><html><head><meta charset=\"utf-8\"></head><body></body></html>"
LANG = "it"   # lingua dei risultati, come i file della dottoranda


def _field(name, col_id, text, description):
    return {"name": name, "id": col_id, "columnType": "Data",
            "dataType": "Text" if text else "Decimal",
            "measureType": "Nominal" if text else "Continuous",
            "formula": "", "formulaMessage": "", "parentId": 0, "width": 100,
            "type": "text" if text else "number", "importName": name, "description": description,
            "transform": 0, "edits": [], "missingValues": [], "trimLevels": True}


# ------------------------------------------------------------ opzioni jamovi ---

def _set(opt, value):
    """Scrive un valore Python in un AnalysisOption: str, bool/None, int, lista, dict."""
    opt.Clear()
    if isinstance(value, str):
        opt.s = value
    elif value is None:
        opt.o = jpb.AnalysisOption.NONE
    elif isinstance(value, bool):
        opt.o = jpb.AnalysisOption.TRUE if value else jpb.AnalysisOption.FALSE
    elif isinstance(value, int):
        opt.i = value
    elif isinstance(value, dict):
        opt.c.hasNames = True
        for k, v in value.items():
            opt.c.names.append(k)
            _set(opt.c.options.add(), v)
    else:   # lista
        opt.c.SetInParent()
        for v in value:
            _set(opt.c.options.add(), v)


def _strip(elem):
    """Come jamovi per i template: risultati vuoti e 'stale' (da ricalcolare)."""
    elem.status = jpb.AnalysisStatus.Value("ANALYSIS_COMPLETE")
    kind = elem.WhichOneof("type")
    if kind in ("group", "array"):
        for e in getattr(elem, kind).elements:
            _strip(e)
        return
    elem.stale = True
    if kind == "table":
        for col in elem.table.columns:
            for cell in col.cells:
                cell.o = 0
                del cell.footnotes[:]
                del cell.symbols[:]
        del elem.table.notes[:]
    elif kind == "preformatted":
        elem.preformatted = ""
    elif kind == "image":
        elem.image.path = ""


def _anova(spec: dict, analysis_id: int, index: int) -> bytes:
    """Un'ANOVA a misure ripetute (jmv::anovaRM) dal modello della dottoranda.
    spec: title, dep_label, factors [(nome, [livelli])], cells [(colonna, [livello per fattore])],
    post_hoc [[fattore], ...]."""
    r = jpb.AnalysisResponse()
    r.ParseFromString((TEMPLATE_DIR / "anovaRM.pb").read_bytes())
    names = [f for f, _ in spec["factors"]]
    terms = [[f] for f in names] + ([names] if len(names) == 2 else [])
    values = {
        "rm": [{"label": f, "levels": list(lv)} for f, lv in spec["factors"]],
        "rmCells": [{"measure": col, "cell": list(cell)} for col, cell in spec["cells"]],
        "effectSize": ["eta", "partEta"],
        "depLabel": spec["dep_label"],
        "rmTerms": terms,
        "contrasts": [{"var": f, "type": "none"} for f in names],   # uno per fattore, come jamovi
        "spherTests": True,
        "spherCorr": ["none", "GG"],
        "postHoc": [list(t) for t in spec["post_hoc"]] or None,   # None = nessun termine, come il modello
        "postHocCorr": ["none", "tukey", "scheffe", "holm", "bonf"],
        ".lang": LANG,
    }
    for name, opt in zip(r.options.names, r.options.options):
        if name in values:
            _set(opt, values[name])
    missing = set(values) - set(r.options.names)
    assert not missing, f"options not in the template: {missing}"
    r.analysisId, r.index, r.revision = analysis_id, index, 0
    r.instanceId = ""
    r.title, r.hasTitle = spec["title"], True
    _strip(r.results)
    return r.SerializeToString()


def _header(analysis_id: int) -> bytes:
    r = jpb.AnalysisResponse()
    r.ParseFromString((TEMPLATE_DIR / "empty.pb").read_bytes())
    r.analysisId, r.index = analysis_id, 0
    return r.SerializeToString()


# ------------------------------------------------------------------ scrittura ---

def write_tables(table: pd.DataFrame, path_stem: Path, descriptions: dict = None, analyses: list = None):
    """table: colonna 'subject' + colonne numeriche. Scrive <stem>.csv e <stem>.omv, con le
    analisi `analyses` (lista di spec per _anova) gia' impostate."""
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
    metadata = {"document": {"resultsLanguage": LANG},
                "dataSet": {"rowCount": len(table), "columnCount": len(table.columns), "removedRows": [],
                            "addedRows": [], "fields": fields, "transforms": [], "weights": None}}
    for spec in analyses or []:
        for col, _ in spec["cells"]:
            assert col in table.columns, f"{col} not in the table"
    with zipfile.ZipFile(path_stem.with_suffix(".omv"), "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("meta", MANIFEST)
        z.writestr("index.html", EMPTY_HTML)
        z.writestr("metadata.json", json.dumps(metadata))
        z.writestr("xdata.json", json.dumps(xdata))
        z.writestr("data.bin", b"".join(chunks))
        if analyses:
            z.writestr("01 empty/analysis", _header(1))
            for k, spec in enumerate(analyses):
                aid = 2 * (k + 1)   # id pari, come le analisi create dall'utente
                z.writestr(f"{aid:02d} anovaRM/analysis", _anova(spec, aid, k + 1))
