"""Collegamenti fra le parti del protocollo e questionari.

Va lanciato DOPO le altre analisi (run_all.py lo fa nell'ordine giusto): legge
le tabelle per soggetto che producono.
    reaching_blind/statistics/blind_subjects.csv
    reaching_regions/statistics/training_blocks.csv
    real_robot/statistics/robot_subjects.csv
    questionari/statistics/nasa_scores.csv, sus_scores.csv, edinburgh_scores.csv
Gli ID dei questionari vengono normalizzati (' s1 ', 'S01' -> 'S001') per
unirli a quelli dei file del robot.

1. Trasferimento (il filo biomedico): chi va meglio con il cursore va meglio
   anche con il robot? Correlazioni di Spearman fra la stessa abilita' misurata
   in contesti diversi (TRANSFER qui sotto).
2. Questionari: descrittive di SUS (con il riferimento 68), NASA-TLX raw ed
   Edinburgh, e correlazioni con le prestazioni (QUESTIONNAIRE).
Con pochi soggetti le correlazioni sono ESPLORATIVE: nessuna correzione, da
leggere come tendenze (rho e' l'effect size).

Output in cross_analysis/statistics/:
    report.txt, cross_subjects.csv (una riga per soggetto con tutti i valori usati),
    cross_correlations.csv, questionnaires_summary.csv,
    plots/cross_transfer.png, plots/cross_questionnaires.png

Uso:
    python3 cross_analysis.py [--root DIR] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import normalize_id   # noqa: E402
from common.plots import BLUE, INK2, MUTED, grid_axes, save_figure, setup_style   # noqa: E402
from common.report import Report   # noqa: E402
from common.stats import describe, fmt_median_iqr, fmt_num, magnitude, spearman   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUS_BENCHMARK = 68

# variabile per soggetto -> (etichetta, come ricavarla); solo le metriche dei task (README):
# ("blind", test, metrica) | ("training_block", training, blocco, metrica) |
# ("robot", run, metrica, fase) | ("q", file, colonna) |
# ("delta", var_a, var_b) = var_a - var_b (miglioramento)
VARIABLES = {
    "blind3_hit":          ("Blind test 3 – hit rate (%)", ("blind", 3, "hit")),
    "blind3_ee":           ("Blind test 3 – Euclidean error (cm)", ("blind", 3, "euclidean_error")),
    "blind3_de":           ("Blind test 3 – direction error (deg)", ("blind", 3, "direction_error")),
    "blind4_ee":           ("Blind test 4 – Euclidean error (cm)", ("blind", 4, "euclidean_error")),
    "blind_fatigue":       ("Blind Euclidean error: test 4 – test 3 (cm) (> 0 = worse after run 1)",
                            ("delta", "blind4_ee", "blind3_ee")),
    "train_first_md":      ("Training T1 B1 – movement duration (s)", ("training_block", "T1", 1, "movement_duration")),
    "train_last_md":       ("Training T2 B12 – movement duration (s)", ("training_block", "T2", 12, "movement_duration")),
    "train_last_li":       ("Training T2 B12 – linearity index", ("training_block", "T2", 12, "linearity_index")),
    "train_gain":          ("Training gain: movement duration T1 B1 – T2 B12 (s)",
                            ("delta", "train_first_md", "train_last_md")),
    "run1_rt":             ("Run 1 – region transitions, outbound", ("robot", 1, "region_transitions", "andata")),
    "run1_pl":             ("Run 1 – path length, outbound (m)", ("robot", 1, "path_length", "andata")),
    "run1_outbound":       ("Run 1 – completion time, outbound (s)", ("robot", 1, "completion_time", "andata")),
    "run2_outbound":       ("Run 2 – completion time, outbound (s)", ("robot", 2, "completion_time", "andata")),
    "robot_gain":          ("Robot gain: outbound completion time run 1 – run 2 (s)",
                            ("delta", "run1_outbound", "run2_outbound")),
    "run2_total":          ("Run 2 – total completion time (s)", ("robot", 2, "completion_time", "totale")),
    "sus":                 ("SUS score (0-100)", ("q", "sus_scores.csv", "SUS")),
    "tlx":                 ("NASA-TLX raw (0-100)", ("q", "nasa_scores.csv", "Raw TLX (0-100)")),
    "tlx_physical":        ("NASA-TLX physical demand (1-20)", ("q", "nasa_scores.csv", "Physical demand")),
    "lq":                  ("Edinburgh laterality quotient", ("q", "edinburgh_scores.csv", "LQ")),
}
# (x, y, segno atteso, perche')
TRANSFER = [
    ("blind3_hit", "run1_rt", "-", "reaching the target blind vs sweeping across the grid on the robot"),
    ("train_last_li", "run1_pl", "+", "curved cursor paths (training) vs longer robot paths"),
    ("blind3_de", "run1_rt", "+", "planning errors (blind) vs corrections (robot)"),
    ("blind3_ee", "run1_outbound", "+", "map knowledge (blind) vs driving time (robot)"),
    ("train_gain", "robot_gain", "+", "who improves in the training improves on the robot"),
]
QUESTIONNAIRE = [
    ("sus", "run2_total", "-", "perceived usability vs time on the task"),
    ("tlx", "run2_total", "+", "perceived workload vs time on the task"),
    ("tlx_physical", "blind_fatigue", "+", "perceived physical demand vs fatigue in the blind test"),
]


def read(path):
    return pd.read_csv(path) if Path(path).exists() else None


def build(root: Path) -> tuple:
    blind = read(root / "reaching_blind/statistics/blind_subjects.csv")
    tr_blocks = read(root / "reaching_regions/statistics/training_blocks.csv")
    robot = read(root / "real_robot/statistics/robot_subjects.csv")
    q = {}
    for name in ("sus_scores.csv", "nasa_scores.csv", "edinburgh_scores.csv"):
        df = read(root / "questionari/statistics" / name)
        if df is not None and "ID" in df.columns:
            df["subject"] = df["ID"].map(normalize_id)
            q[name] = df.set_index("subject")
    missing = [n for n, d in (("blind", blind), ("training", tr_blocks), ("robot", robot)) if d is None] + \
        [n for n in ("sus_scores.csv", "nasa_scores.csv", "edinburgh_scores.csv") if n not in q]

    cols = {}

    def series(spec):
        kind = spec[0]
        if kind == "blind" and blind is not None and spec[2] in blind.columns:
            return blind[blind.test == spec[1]].set_index("subject")[spec[2]]
        if kind == "training_block" and tr_blocks is not None and spec[3] in tr_blocks.columns:
            sel = tr_blocks[(tr_blocks.training == spec[1]) & (tr_blocks.block == spec[2])]
            return sel.set_index("subject")[spec[3]]
        if kind == "robot" and robot is not None:
            sel = robot[(robot.run == spec[1]) & (robot.metric == spec[2]) & (robot.phase == spec[3])]
            return pd.to_numeric(sel.set_index("subject")["value"], errors="coerce")
        if kind == "q" and spec[1] in q and spec[2] in q[spec[1]].columns:
            return pd.to_numeric(q[spec[1]][spec[2]], errors="coerce")
        if kind == "delta":
            return cols.get(spec[1], pd.Series(dtype=float)) - cols.get(spec[2], pd.Series(dtype=float))
        return pd.Series(dtype=float)

    for name, (_, spec) in VARIABLES.items():   # le delta vengono dopo le variabili che usano
        if spec[0] != "delta":
            cols[name] = series(spec)
    for name, (_, spec) in VARIABLES.items():
        if spec[0] == "delta":
            cols[name] = series(spec)
    table = pd.DataFrame(cols)
    table.index.name = "subject"
    return table.sort_index(), q, missing


def correlate(table, pairs, family, report):
    rows = []
    for x, y, expected, why in pairs:
        r = spearman(table[x], table[y])
        agrees = "-" if np.isnan(r["rho"]) else ("si'" if (r["rho"] > 0) == (expected == "+") else "no")
        rows.append({"family": family, "x": x, "x_label": VARIABLES[x][0], "y": y, "y_label": VARIABLES[y][0],
                     "expected_sign": expected, "meaning": why, **r, "magnitude": magnitude(r["rho"]),
                     "as_expected": agrees})
        report.text(f"  {VARIABLES[x][0]}  vs  {VARIABLES[y][0]}")
        if np.isnan(r["rho"]):
            report.text(f"      ({why}) non calcolabile: n={r['n']} coppie complete, servono almeno 3 "
                        "e valori non tutti uguali")
        else:
            report.text(f"      ({why}; atteso {expected}) rho={fmt_num(r['rho'])} ({magnitude(r['rho'])}), "
                        f"p={fmt_num(r['p'], 3)}, n={r['n']}, nella direzione attesa: {agrees}")
    return rows


def scatter(table, pairs, out, name, title):
    setup_style()
    fig, axes = grid_axes(len(pairs), ncols=3, width=4.2, height=3.6)
    for ax, (x, y, _, _) in zip(axes, pairs):
        d = table[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        ax.grid(True, axis="both")
        ax.scatter(d[x], d[y], color=BLUE, s=28, zorder=3)
        for subj, row in d.iterrows():
            ax.annotate(subj, (row[x], row[y]), fontsize=6.5, color=MUTED, xytext=(3, 3), textcoords="offset points")
        r = spearman(table[x], table[y])
        ax.set_xlabel(VARIABLES[x][0], fontsize=8)
        ax.set_ylabel(VARIABLES[y][0], fontsize=8)
        ax.set_title(f"rho = {fmt_num(r['rho'])}, n = {r['n']}", loc="left", fontsize=9.5, color=INK2)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, out, name)


def main():
    p = argparse.ArgumentParser(description="Collegamenti fra le parti del protocollo e questionari")
    p.add_argument("--root", type=Path, default=ROOT, help="cartella data_evaluation_thesis")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "statistics")
    args = p.parse_args()

    table, q, missing = build(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    report = Report("COLLEGAMENTI FRA LE PARTI DEL PROTOCOLLO E QUESTIONARI", legend=False)
    report.text(f"Soggetti con almeno un dato: {len(table)}")
    if missing:
        report.text("Mancano (lancia prima le altre analisi / i questionari): " + ", ".join(missing))
    report.text("Correlazioni di Spearman ESPLORATIVE: nessuna correzione, da leggere come tendenze.")

    rows = []
    report.section("Trasferimento: la stessa abilita' con il cursore e con il robot")
    rows += correlate(table, TRANSFER, "transfer", report)
    report.section("Questionari e prestazioni")
    rows += correlate(table, QUESTIONNAIRE, "questionnaire", report)

    report.section("Questionari: descrittive")
    summary = []
    if "sus_scores.csv" in q:
        sus = pd.to_numeric(q["sus_scores.csv"]["SUS"], errors="coerce")
        d = describe(sus)
        above = int((sus >= SUS_BENCHMARK).sum())
        summary.append({"questionnaire": "SUS", "item": "SUS (0-100)", **d, "n_above_68": above})
        report.text(f"  SUS: {fmt_median_iqr(d, 1)}; {above}/{d.get('n', 0)} soggetti sopra il riferimento "
                    f"{SUS_BENCHMARK} (media degli studi in letteratura)")
    if "nasa_scores.csv" in q:
        nasa = q["nasa_scores.csv"]
        for col in [c for c in nasa.columns if c not in ("ID",)]:
            v = pd.to_numeric(nasa[col], errors="coerce")
            if v.notna().any():
                d = describe(v)
                summary.append({"questionnaire": "NASA-TLX", "item": col, **d})
                report.text(f"  NASA-TLX {col}: {fmt_median_iqr(d, 1)}")
    if "edinburgh_scores.csv" in q:
        d = describe(pd.to_numeric(q["edinburgh_scores.csv"]["LQ"], errors="coerce"))
        summary.append({"questionnaire": "Edinburgh", "item": "LQ", **d})
        report.text(f"  Edinburgh LQ: {fmt_median_iqr(d, 0)}")

    table.to_csv(args.out / "cross_subjects.csv")
    pd.DataFrame(rows).to_csv(args.out / "cross_correlations.csv", index=False)
    pd.DataFrame(summary).to_csv(args.out / "questionnaires_summary.csv", index=False)
    scatter(table, TRANSFER, args.out, "cross_transfer",
            "Transfer: the same ability with the cursor and with the robot (one dot per subject)")
    scatter(table, QUESTIONNAIRE, args.out, "cross_questionnaires", "Questionnaires vs performance")
    report.save(args.out)


if __name__ == "__main__":
    main()
