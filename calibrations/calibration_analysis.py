"""Calibrazione / customizzazione della BoMI (customize_bomi.py), una per soggetto.

Input: i file di calibrations/ del repo reachy2_teleop_grasping_and_simulation,
incollati ovunque dentro questa cartella:
    <soggetto>_<AAAAMMGG_hhmmss>.npz                    la mappa usata dal soggetto
    <soggetto>_<AAAAMMGG_hhmmss>_customization.json     cosa e' stato cambiato rispetto
                                                        alla mappa condivisa (shared)
Se un soggetto ha piu' mappe si usa l'ultima (e' quella che caricano gli script
del robot con --subject); le altre sono elencate come scartate.

Tutti i soggetti partono dalla stessa mappa condivisa (autoencoder 42 -> 2,
calibrate_bomi.py): VAF e campioni sono quindi uguali per tutti e vengono
riportati una volta sola. Quello che cambia fra i soggetti e' solo la
customizzazione (rotazione, flip, scala, offset) fatta con i tasti di
customize_bomi.py. Una mappa salvata senza nessun tasto premuto ha i passi
"not recorded" (null nel file): vale come 0 passi, scala 1 e offset 0.

Analisi: solo descrittive tra soggetti (mediana [Q1-Q3], media, SD) e conteggi
(quanti soggetti hanno cambiato scala, offset, rotazione, flip). Non ci sono
due momenti da confrontare, quindi nessun test.

Se in questa cartella ci sono anche i campioni della calibrazione condivisa
(shared_calib.npy, 42 coordinate della mano per frame, dal repo del task) si
rifanno i calcoli dai dati raw con le funzioni del laboratorio:
  - compute_bomi_map.compute_latent_variance: quota di varianza di ogni unita'
    latente (cu), dall'encoder salvato nella mappa, stessa divisione 80/20 del task;
  - pca_analysis.py (PrincipalComponentAnalysis + compute_vaf): VAF della
    ricostruzione con 2 PC su train e test, varianza spiegata da ogni PC e
    distribuzione PC1/PC2: quanto i movimenti della mano stanno su un piano.
  La VAF dell'autoencoder NON si puo' ricalcolare: il task salva solo l'encoder
  (manca il decoder), resta quella scritta nella mappa.

Output in calibrations/statistics/:
    report.txt                     il riassunto da leggere (italiano)
    calibration_subjects.csv       un soggetto per riga: customizzazione finale e passi
    calibration_statistics.csv     metrica: descrittive tra soggetti
    plots/calibration_customization.png   scala, offset e passi per soggetto
    calibration_pca.csv, plots/calibration_pca.png   (solo con shared_calib.npy)

Uso:
    python3 calibration_analysis.py [--results DIR] [--out DIR]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import normalize_id   # noqa: E402
from common.plots import BLUE, CATEGORICAL, INK2, MUTED, save_figure, setup_style   # noqa: E402
from common.report import Report   # noqa: E402
from common.stats import describe, fmt_median_iqr, fmt_num   # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
PATTERN = re.compile(r"(?P<subject>.+)_(?P<stamp>\d{8}_\d{6})\.npz")
STEP_TYPES = ("scale", "offset", "flip", "rotation")

# metrica -> (etichetta, unita')
METRICS = {
    "n_steps":         ("Customization steps (key presses)", ""),
    "n_scale":         ("Scale steps", ""),
    "n_offset":        ("Offset steps", ""),
    "n_flip":          ("Flip steps", ""),
    "n_rotation":      ("Rotation steps", ""),
    "scale":           ("Final scale", "x"),
    "rot_deg":         ("Final rotation", "deg"),
    "offset_x":        ("Final offset X", "px"),
    "offset_y":        ("Final offset Y", "px"),
    "offset_norm":     ("Final offset (distance)", "px"),
}


def step_types(step: dict) -> list:
    """Tipi di un passo di customize(), con la stessa logica di _steps_text()
    in customize_bomi.py (un flip ha guadagno negativo, non conta come scala)."""
    out = []
    if step["rot_deg"]:
        out.append("rotation")
    if step["gain_x"] < 0 or step["gain_y"] < 0:
        out.append("flip")
    elif step["gain_x"] != 1.0 or step["gain_y"] != 1.0:
        out.append("scale")
    if step["off_x"] or step["off_y"]:
        out.append("offset")
    return out


def find_maps(root: Path) -> tuple:
    """(scelte, scartate): scelte = {soggetto: Path dell'ultima mappa .npz}."""
    found = {}
    for path in sorted(root.rglob("*.npz")):
        if "statistics" in path.relative_to(root).parts:
            continue
        m = PATTERN.fullmatch(path.name)
        if m:
            found.setdefault(normalize_id(m.group("subject")), []).append((m.group("stamp"), path))
    chosen, dropped = {}, []
    for subject, items in found.items():
        items.sort()
        chosen[subject] = items[-1][1]
        dropped += [p for _, p in items[:-1]]
    return chosen, dropped


def load_map(subject: str, path: Path) -> dict:
    data = np.load(path)
    metrics = json.loads(str(data["metrics"])) if "metrics" in data else {}
    row = {"subject": subject, "map": path.stem,
           "vaf_train": metrics.get("vaf_train"), "vaf_test": metrics.get("vaf_test"),
           "n_train": metrics.get("n_train"), "n_test": metrics.get("n_test"),
           "_weights": np.concatenate([data[k].ravel() for k in ("w1", "b1", "w2", "b2", "w3", "b3")])}
    js = path.with_name(path.stem + "_customization.json")
    if not js.exists():
        row["missing_json"] = True
        return row
    rec = json.loads(js.read_text(encoding="utf-8"))
    steps = rec.get("steps") or []   # null = nessun tasto premuto
    counts = {t: 0 for t in STEP_TYPES}
    for st in steps:
        for t in step_types(st):
            counts[t] += 1
    row.update({
        "base": rec.get("base"), "steps_recorded": rec.get("steps") is not None,
        "n_steps": len(steps), **{f"n_{t}": c for t, c in counts.items()},
        "scale": rec.get("scale"), "rot_deg": rec.get("rot_deg"), "flip_x": bool(rec.get("flip_x")),
        "offset_x": rec.get("offset_x"), "offset_y": rec.get("offset_y"),
        "residual": rec.get("residual"), "steps_text": rec.get("steps_text"),
    })
    # -0 e 1e-13 px dagli arrotondamenti: offset nullo
    for k in ("offset_x", "offset_y", "rot_deg"):
        if row[k] is not None and abs(row[k]) < 1e-6:
            row[k] = 0.0
    row["offset_norm"] = float(np.hypot(row["offset_x"], row["offset_y"]))
    return row


AE_SEED, AE_TEST_SPLIT, N_MAP_COMPONENTS = 20, 0.2, 2   # come bomi.py (fit) e pca_analysis.py


def compute_vaf(x, x_rec) -> float:
    """compute_bomi_map.compute_vaf: VAF (%) della ricostruzione, entrambe a media nulla."""
    x_zm, x_rec_zm = x - x.mean(0), x_rec - x_rec.mean(0)
    return float(100 * (1 - np.sum((x_zm - x_rec_zm) ** 2) / np.sum(x_zm ** 2)))


def compute_latent_variance(codes) -> np.ndarray:
    """compute_bomi_map.compute_latent_variance: quota (%) della varianza di ogni unita'."""
    v = np.sum((codes - codes.mean(0)) ** 2, axis=0)
    return 100 * v / max(float(v.sum()), 1e-8)


def encoder(npz, x):
    h = np.tanh(x @ npz["w1"] + npz["b1"])
    h = np.tanh(h @ npz["w2"] + npz["b2"])
    return h @ npz["w3"] + npz["b3"]


def shared_calibration(samples_path: Path, map_path: Path, report: Report, out: Path):
    """Calcoli dai campioni raw della calibrazione condivisa (vedi il docstring)."""
    x = np.load(samples_path).astype(float)
    x = x[np.any(x != 0, axis=1)]                       # frame prima della prima detection
    x = x[np.all(np.isfinite(x), axis=1)]
    np.random.seed(AE_SEED)                             # come fit() in bomi.py: seed 20, shuffle, 80/20
    np.random.shuffle(x)
    split = int(round(len(x) * (1 - AE_TEST_SPLIT)))
    train, test = x[:split], x[split:]
    report.section("Calibrazione condivisa, dai campioni raw (shared_calib.npy)")
    report.text(f"  {len(x)} frame validi x {x.shape[1]} coordinate; train {len(train)}, test {len(test)} "
                "(stessa divisione del task)")
    npz = np.load(map_path)
    stored = json.loads(str(npz["metrics"])) if "metrics" in npz else {}
    cu_tr, cu_te = compute_latent_variance(encoder(npz, train)), compute_latent_variance(encoder(npz, test))
    report.text(f"  Varianza latente (cu x / cu y): train {cu_tr[0]:.1f}% / {cu_tr[1]:.1f}%, "
                f"test {cu_te[0]:.1f}% / {cu_te[1]:.1f}% (nella mappa: train "
                f"{' / '.join(f'{v:.1f}%' for v in stored.get('cu_train', []))})")
    report.text(f"  VAF autoencoder (dalla mappa, non ricalcolabile: manca il decoder): train "
                f"{fmt_num(stored.get('vaf_train'), 1)}%, test {fmt_num(stored.get('vaf_test'), 1)}%")
    # pca_analysis.py / PrincipalComponentAnalysis.train_pca
    mean = train.mean(0)
    _, sv, vt = np.linalg.svd(train - mean, full_matrices=False)
    coeff = vt.T
    ratios = sv ** 2 / np.sum(sv ** 2)

    def rec(data):
        score = (data - mean) @ coeff
        score[:, N_MAP_COMPONENTS:] = 0
        return score @ coeff.T + mean

    vaf_tr, vaf_te = compute_vaf(train, rec(train)), compute_vaf(test, rec(test))
    within = 100 * ratios[:2] / ratios[:2].sum()
    report.text(f"  PCA con {N_MAP_COMPONENTS} PC (pca_analysis.py): VAF train {vaf_tr:.1f}%, test {vaf_te:.1f}%; "
                f"PC1 {100 * ratios[0]:.1f}%, PC2 {100 * ratios[1]:.1f}% della varianza totale "
                f"(fra le prime due: {within[0]:.0f}% / {within[1]:.0f}%)")
    report.text("  Confronto: l'autoencoder (non lineare) spiega "
                f"{fmt_num(stored.get('vaf_test'), 1)}% sul test, la PCA lineare con 2 componenti {vaf_te:.1f}%.")
    pca = pd.DataFrame({"PC": [f"PC{i + 1}" for i in range(len(ratios))],
                        "ExplainedVariancePercent": 100 * ratios,
                        "CumulativeExplainedVariancePercent": 100 * np.cumsum(ratios)})
    pca.to_csv(out / "calibration_pca.csv", index=False)
    pd.DataFrame([{"n_train": len(train), "n_test": len(test), "pca_vaf_train": vaf_tr, "pca_vaf_test": vaf_te,
                   "ae_vaf_train": stored.get("vaf_train"), "ae_vaf_test": stored.get("vaf_test"),
                   "cu_train_x": cu_tr[0], "cu_train_y": cu_tr[1], "cu_test_x": cu_te[0], "cu_test_y": cu_te[1]}]
                 ).to_csv(out / "calibration_vaf.csv", index=False)
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    n = min(15, len(ratios))
    ax.bar(np.arange(1, n + 1), 100 * ratios[:n], color=BLUE, width=0.7, label="explained variance")
    ax.plot(np.arange(1, n + 1), 100 * np.cumsum(ratios)[:n], color=CATEGORICAL[1], marker="o", ms=3,
            label="cumulative")
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance (%)")
    ax.set_title("Shared calibration – PCA of the 42 hand coordinates", loc="left")
    ax.legend(fontsize=8)
    save_figure(fig, out, "calibration_pca")


def plots(subjects: pd.DataFrame, out: Path):
    setup_style()
    s = subjects.dropna(subset=["scale"])
    x = np.arange(len(s))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.bar(x, s["scale"], color=BLUE, width=0.6)
    ax.axhline(1.0, color=MUTED, ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(s["subject"])
    ax.set_ylabel("scale (x shared map)")
    ax.set_title("Final scale", loc="left")

    ax = axes[1]
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    # soggetti con lo stesso offset (es. tutti a 0) in un'unica etichetta
    for (ox, oy), g in s.groupby([s["offset_x"].round(), s["offset_y"].round()]):
        ax.plot(ox, oy, "o", color=BLUE, ms=6)
        ax.annotate(", ".join(g["subject"]), (ox, oy), xytext=(5, 5 if oy <= 0 else -11),
                    textcoords="offset points", fontsize=8, color=INK2)
    lim = max(50.0, float(np.nanmax(np.abs(s[["offset_x", "offset_y"]].to_numpy()))) * 1.2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lim, -lim)   # asse y dello schermo verso il basso
    ax.set_aspect("equal")
    ax.grid(True, axis="both")
    ax.set_xlabel("offset X [px]")
    ax.set_ylabel("offset Y [px], down = +")
    ax.set_title("Final offset", loc="left")

    ax = axes[2]
    bottom = np.zeros(len(s))
    for t, color in zip(STEP_TYPES, CATEGORICAL):
        v = s[f"n_{t}"].to_numpy(dtype=float)
        ax.bar(x, v, bottom=bottom, color=color, width=0.6, label=t)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(s["subject"])
    ax.set_ylabel("key presses")
    ax.set_title("Customization steps", loc="left")
    ax.legend(fontsize=8)
    return save_figure(fig, out, "calibration_customization")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=BASE_DIR, help="cartella con i file di calibrations/")
    p.add_argument("--out", type=Path, default=BASE_DIR / "statistics")
    p.add_argument("--shared-dir", type=Path, default=BASE_DIR,
                   help="cartella con shared_calib.npy e shared.npz (default: questa)")
    args = p.parse_args()

    chosen, dropped = find_maps(args.results)
    if not chosen:
        print(f"Nessun file <soggetto>_<data>_<ora>.npz in {args.results}")
        return
    rows = [load_map(s, chosen[s]) for s in sorted(chosen)]
    weights = [r.pop("_weights") for r in rows]
    subjects = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)

    report = Report("CALIBRAZIONE BoMI – customizzazione per soggetto")
    report.text(f"Soggetti: {len(subjects)} ({', '.join(subjects.subject)})")
    if dropped:
        report.text("Mappe scartate (si usa l'ultima): " + ", ".join(p.name for p in dropped))
    if "missing_json" in subjects and subjects["missing_json"].fillna(False).any():
        miss = subjects.loc[subjects["missing_json"].fillna(False).astype(bool), "subject"]
        report.text("Senza _customization.json (esclusi dalle descrittive): " + ", ".join(miss))

    report.section("Mappa condivisa")
    same = all(np.array_equal(weights[0], w) for w in weights[1:])
    if same:
        r0 = subjects.iloc[0]
        report.text("  Stessa rete (autoencoder) per tutti i soggetti: cambia solo la customizzazione.")
        report.text(f"  VAF train {fmt_num(r0.vaf_train, 1)}% (n={r0.n_train}), "
                    f"VAF test {fmt_num(r0.vaf_test, 1)}% (n={r0.n_test}).")
    else:
        report.text("  ATTENZIONE: le reti NON sono tutte uguali; VAF per soggetto in calibration_subjects.csv.")
        report.text(f"  VAF test: {fmt_median_iqr(describe(subjects.vaf_test), 1)} %")

    report.section("Customizzazione (descrittive tra soggetti)")
    c = subjects.dropna(subset=["scale"])
    n = len(c)
    customized = int((c.n_steps > 0).sum())
    report.text(f"  Soggetti con almeno un passo: {customized}/{n}"
                + (f" (senza customizzazione: {', '.join(c.loc[c.n_steps == 0, 'subject'])})"
                   if customized < n else ""))
    report.text(f"  Scala cambiata: {int((~np.isclose(c.scale, 1.0)).sum())}/{n}, "
                f"offset cambiato: {int((c.offset_norm > 0).sum())}/{n}, "
                f"rotazione cambiata: {int((c.rot_deg != 0).sum())}/{n}, "
                f"flip X finale: {int(c.flip_x.sum())}/{n}")
    stats_rows = []
    for m, (label, unit) in METRICS.items():
        d = describe(c[m])
        stats_rows.append({"metric": m, "label": label, "unit": unit, **d})
        report.text(f"  {label}{f' [{unit}]' if unit else ''}: {fmt_median_iqr(d)} "
                    f"(min {fmt_num(d.get('min'))}, max {fmt_num(d.get('max'))})")

    report.section("Per soggetto")
    for _, r in c.iterrows():
        report.text(f"  {r.subject}: scala x{r.scale:.3f}, rotazione {r.rot_deg:+.0f} deg, "
                    f"flip X {'si' if r.flip_x else 'no'}, offset ({r.offset_x:+.0f}, {r.offset_y:+.0f}) px | "
                    f"{r.n_steps} passi (scala {r.n_scale}, offset {r.n_offset}, flip {r.n_flip}, "
                    f"rotazione {r.n_rotation})")
    report.text("  Il numero di passi dipende anche da come e' stata fatta la sessione (es. l'offset")
    report.text("  si sposta di 10 px a tasto): e' una misura di quanto la mappa e' stata adattata,")
    report.text("  non della difficolta' del soggetto.")

    samples = args.shared_dir / "shared_calib.npy"
    shared_map = args.shared_dir / "shared.npz"
    if samples.exists():
        shared_calibration(samples, shared_map if shared_map.exists() else chosen[subjects.subject.iloc[0]],
                           report, args.out)
    else:
        report.section("Calibrazione condivisa, dai campioni raw")
        report.text("  Manca shared_calib.npy (e shared.npz): copiali qui da calibrations/ del repo del task")
        report.text("  per ricalcolare varianza latente e PCA (pca_analysis.py) dai dati raw.")

    subjects.to_csv(args.out / "calibration_subjects.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(args.out / "calibration_statistics.csv", index=False)
    plots(subjects, args.out)
    report.save(args.out)


if __name__ == "__main__":
    main()
