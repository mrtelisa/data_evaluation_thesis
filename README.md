# data_evaluation_thesis

Analisi dei dati del protocollo sperimentale: questionari, 5 blind reaching test, 2 training a 9 regioni, 2 run sul robot reale, e i collegamenti fra le parti.

## 1. Dove incollare i file

Gli script del repo `reachy2_teleop_grasping_and_simulation` scrivono i risultati in tre cartelle. Copia il **contenuto** di ognuna nella cartella `results/` dell'analisi corrispondente (vanno bene anche sottocartelle: i file vengono cercati ovunque dentro la cartella dell'analisi, tranne in `statistics/`).

| Script del repo | Cartella del repo | Incolla in | File usati |
|---|---|---|---|
| `reaching_blind.py` (branch `new_protocol`) | `results_blind/` | `reaching_blind/results/` | `<soggetto>_blind_test<N>_raw.csv`, `_summary.json` (+ `_trials.csv` solo per controllo) |
| `reaching_regions.py` (branch `new_protocol`) | `results_regions/` | `reaching_regions/results/` | `<soggetto>_regions_<pre\|post>_raw.csv`, `_summary.json` (+ `_trials.csv` solo per controllo) |
| `reachy_control.py` (branch `main`) | `results_robot/` | `real_robot/results/` | `<soggetto>_run<N>_regions.csv`, `_odometry.csv`, `_cursor.csv`, `_session.json` |
| `customize_bomi.py` (branch `new_protocol`) | `calibrations/` | `calibrations/` | `<soggetto>_<data>_<ora>.npz`, `_customization.json`; in più `shared_calib.npy` e `shared.npz` per PCA e varianza latente |
| Google Form | (export .xlsx) | `questionari/risposte/` | i 4 file .xlsx (il Borg non è più analizzato) |

- **Tutto si calcola dai file raw** (un frame per riga), con le funzioni del laboratorio (vedi §6). I `_trials.csv` scritti dai task servono solo come controllo: il report avvisa se prove, tempi o hit non coincidono con il raw. `_blocks.csv` e `_trajectory.csv` non servono.
- Dal `_session.json` del robot restano solo quello che nel raw non c'è: durate delle fasi (mancano gli istanti degli eventi), dwell, repositioning, conteggi manuali.
- **Non analizzati**: la bag della free exploration in simulazione (non ha metriche nel protocollo).
- **Ripetizioni**: se una sessione è stata rifatta, il repo aggiunge `_1`, `_2`, ... al nome. Si usa sempre l'**ultima**; le altre sono elencate nel report come scartate.
- **ID dei soggetti**: devono essere gli stessi negli script e nei questionari. `S1`, `s01`, `S001` vengono riconosciuti come lo stesso soggetto (`S001`).

## 2. Come lanciare

```bash
cd data_evaluation_thesis
python3 run_all.py
```

Lancia tutto nell'ordine giusto: questionari → blind test → training → robot → calibrazioni → collegamenti. Ogni parte si può lanciare anche da sola (`python3 reaching_blind/blind_analysis.py`, ...). Un'analisi senza dati viene saltata.

Serve Python 3 con `pandas`, `scipy`, `matplotlib`, `openpyxl`, `statsmodels` (`pip install --user pandas scipy matplotlib openpyxl statsmodels`; sul Python di sistema di Ubuntu serve anche `--break-system-packages`).

## 3. Cosa leggere

**`summary/riepilogo_analisi.pdf`**: il riassunto di tutto in un unico PDF (dati usati, metriche con le formule, test, risultati, tutti i grafici). Lo rigenera `run_all.py` alla fine; da solo: `python3 summary/make_summary.py` (serve LaTeX: `pdflatex` + `latexmk`).

Ogni analisi scrive in `<cartella>/statistics/`:

- **`report.txt`**: il riassunto in italiano, da leggere per primo. Per ogni confronto: valori tipici prima e dopo, se è migliorato o peggiorato, se è significativo, quanto è grande l'effetto.
- **`plots/`**: i grafici (in inglese, pronti per la tesi).
- **`.csv`**: tutti i numeri, per le tabelle della tesi.

| Cartella | Report | Grafico principale |
|---|---|---|
| `reaching_blind/` | 5 metriche principali, ANOVA sui 5 test + 4 confronti fra test consecutivi | `blind_main.png` |
| `reaching_regions/` | le 5 metriche della figura 4 di Bellitto et al. 2025 + 3 richieste, ANOVA sessione × blocco, 3 confronti, modello misto prova per prova, plateau, sessione intera | `figure4_group.png`, `figure4/<soggetto>.png`, `training_learning_curves.png` |
| `real_robot/` | 4 metriche primarie + le 13 metriche per fase, conteggi | `robot_primary.png`, `robot_transitions.png` |
| `calibrations/` | customizzazione della mappa per soggetto (scala, offset, passi), solo descrittive; PCA della calibrazione condivisa | `calibration_customization.png`, `calibration_pca.png` |
| `cross_analysis/` | trasferimento cursore → robot, questionari | `cross_transfer.png` |

### Due versioni dei grafici principali

Ogni grafico riassuntivo esiste in due versioni. Il titolo lo dice sempre nella seconda riga: **[NON-PARAMETRIC version]** o **[ANOVA version]**.

| Versione | File | Cosa mostra | A quale analisi corrisponde |
|---|---|---|---|
| **Non parametrica** (quella di prima) | `blind_main.png`, `training_learning_curves.png`, `figure4_group.png`, `robot_primary.png` | mediana e fascia [Q1–Q3] (la figura 4: mediana ± SD), una linea per soggetto, nessun test sul grafico | Wilcoxon (Holm) / Friedman, i "Wilcoxon p Holm" del report |
| **ANOVA** (nuova) | `blind_main_anova.png`, `training_learning_curves_anova.png`, `figure4_group_anova.png`, `robot_primary_anova.png` | media ± SE (errore standard), una linea grigia per soggetto, parentesi `*` / `**` sui confronti post hoc significativi, risultato dell'ANOVA sotto ogni pannello | ANOVA a misure ripetute + t appaiati con Bonferroni: l'analisi principale |
| Modello misto | `training_mixed_model.png` | media fra soggetti di ogni prova (punti) e curva stimata dal modello (linea) | modello misto prova per prova |

Nella tesi va la versione **ANOVA**, perché è quella dell'analisi principale. La non parametrica serve come confronto. `*` = p < 0.05, `**` = p < 0.001 (post hoc, Bonferroni); `†` = le differenze non sono normali, in quel caso vale il Wilcoxon. Le figure per singolo soggetto (`figure4/<soggetto>.png`, traiettorie) non hanno statistica e sono una sola.

## 4. Le scelte statistiche, in parole semplici

- **Un valore per soggetto.** Ogni soggetto fa tante prove: le riassumiamo con la **mediana** (robusta alle prove anomale) o con la **percentuale** per le metriche sì/no (hit, regione corretta). Così ogni soggetto pesa 1.
- **Descrittive: mediana [Q1–Q3].** La mediana è il valore tipico del gruppo, [Q1–Q3] la fascia della metà centrale dei soggetti. Media e SD sono comunque nei .csv.
- **Grafici con una linea per soggetto**: mostrano se tutti migliorano o solo alcuni. Con pochi soggetti sono la parte più importante.
- **Analisi principale: ANOVA a misure ripetute** (come il laboratorio, Bellitto et al. 2025), perché tutti i fattori sono entro soggetto:
  - training: sessione (pre/post) × blocco (12 blocchi da 8 target);
  - blind test: un fattore, i 5 test;
  - robot: run 1 vs run 2 = t appaiato (è l'ANOVA con 2 livelli).
  Con più livelli che soggetti il test di sfericità di Mauchly non si può calcolare: la **correzione di Greenhouse-Geisser** (p GG) si applica sempre. Effect size: **η² parziale** (0.01 piccolo, 0.06 medio, 0.14 grande).
- **Post hoc: t di Student appaiati**, solo i confronti decisi prima, con **Bonferroni** dentro ogni famiglia, **d di Cohen** (0.2 piccolo, 0.5 medio, 0.8 grande) e **Shapiro-Wilk** sulle differenze:
  - training: inizio vs fine training 1, fine training 1 vs inizio training 2, fine training 1 vs fine training 2;
  - blind test: i 4 test consecutivi (1–2, 2–3, 3–4, 4–5);
  - robot: le 4 metriche primarie (`PRIMARY` in `robot_analysis.py`).
  Le altre metriche sono **esplorative** (p non corretto).
- **Controllo non parametrico** accanto a ogni confronto: Wilcoxon (Holm) e Friedman. Se lo Shapiro dice che le differenze non sono normali il report lo segnala: in quel caso vale il controllo. Con 6 soggetti il p più piccolo del Wilcoxon è 0.031, quindi dopo la correzione non può uscire significativo: conta la direzione.
- **Modello lineare misto, prova per prova** (training): tutte le 96 prove di ogni sessione, senza blocchi. `y ~ sessione × log(prova) + regione`, soggetto casuale (intercetta e, se stimabile, pendenza). log(prova) descrive la curva di apprendimento; la regione toglie la differenza di difficoltà fra i target. Movement time e linearity index in scala log (effetti in %). Serve `statsmodels` (`pip install --user statsmodels`).
- **Correlazioni (Spearman)** in `cross_analysis`: solo esplorative, da leggere come tendenze.

## 5. Le metriche

**Blind test** (principali): hit rate (cursore nel cerchio per 0.5 s), end-point error, reach time (= movement time, solo hit), initial direction error, region accuracy.

**Training** (principali, come la figura 4 di Bellitto et al. 2025): a **1 s** dalla comparsa del target, cioè quando il cursore ricompare (nel paper 0.5 s, perché lì era nascosto 0.5 s), Euclidean error, direction error ed extent error; su **tutto il movimento** movement time e linearity index. In più correct region at reveal, reached while hidden, speed peaks. La figura 4 mostra ogni prova (96 del training 1, poi le 96 del training 2) con la media mobile su 12 prove; per il gruppo mediana ± SD tra soggetti (`FIG4_CENTER`, `FIG4_SPREAD` in `training_analysis.py`: `"mean"`, `"sem"` per rifare esattamente il paper). La curva sui 24 blocchi da 8 target viene fittata con `y = a + b·exp(−k·(blocco−1))`: `a` è il plateau, e il report dice in quale blocco si raggiunge il 95% del miglioramento (≤ 12: già nel training 1).

**Robot** (primarie): tempo dell'andata, tempo del ritorno, salti non adiacenti e cambi di comando/min nell'andata. Poi le 13 metriche per fase (tempi, cambi di comando, dwell rifiutati e repositioning, tempo fermo, collisioni, salti non adiacenti, inversioni, durata media dei comandi, passaggi per lo stop, entropia, path length e rotazione cumulata per andata e ritorno, borracce cadute). Run 1 e run 2 si confrontano fase per fase.

**Unità**: pixel del canvas (training e blind 1200 × 650, robot 2550 × 1500) e metri/radianti per l'odometria. Per avere cm come nel paper serve la misura fisica dello schermo usato.

**Collegamenti**: la stessa abilità con il cursore e con il robot (es. region accuracy nel blind test 3 vs salti non adiacenti nella run 1; dwell time a fine training vs durata dei comandi sul robot), e questionari vs prestazioni. SUS confrontato con il riferimento 68.

## 6. Le funzioni del laboratorio e dove si usano

Gli script MATLAB/Python ricevuti (nella radice del repo) sono portati in `common/reach_metrics.py` (e `real_robot/robot_raw.py`, `calibrations/calibration_analysis.py`). Dove prima una grandezza era calcolata in un altro modo, ora si usa la loro definizione; le metriche in più dei task restano, ricalcolate dal raw.

| Script | Cosa calcola | Dove si usa |
|---|---|---|
| `computereachmetrics.m` | per movimento: Euclidean / direction / extent error a un tempo fisso, movement time, linearity index, aspect ratio, dimensionless jerk | training (errori a 1 s = alla ricomparsa); blind test (linearity, aspect ratio, jerk: gli errori "a 1 s" no, il cursore non ricompare mai) |
| `smooth.m` | picchi della velocità (> 25% del massimo, distanti ≥ 0.5 s) e smoothness = 1/picchi | per prova (training, blind) e sul log intero (training, blind, cursore del robot) |
| `speed1.m` | velocità istantanea media / SD / picco; per il robot \|v lineare\| e \|v angolare\| | per prova e log intero; robot dall'odometria |
| `cursor_distance.m` | lunghezza del percorso, spezzata ai buchi > 0.2 s | per prova e log intero; odometria del robot con buchi fino a 1 s (le posizioni sono assolute) |
| `jerk1.m` | jerk medio e dimensionless jerk su tratti continui del log | log intero (training, blind), fasi di guida del robot (cursore e odometria) |
| `tt.m` | durata del log | log intero del training e del blind; per il robot le durate restano quelle degli eventi del task |
| `movement_train.m` | numero di "movimenti" = 1 + cambi di regione | log intero del training e del blind; fasi di guida del robot (`region_changes`, accanto ai comandi del task che ignorano le visite < 0.25 s) |
| `compute_bomi_map.py`, `pca_analysis.py` | VAF, varianza latente (cu), PCA | calibrazione condivisa (`shared_calib.npy`); la VAF dell'autoencoder non si ricalcola, il task non salva il decoder |

Differenze volute: i tempi sono in secondi e non in campioni (gli script assumono 50–100 Hz, la webcam registra a ~30 Hz, il cursore del robot a ~15 Hz); le posizioni a 1 s sono interpolate; prima delle derivate il segnale è ricampionato su una griglia uniforme; il linearity index ha il "− 1" del paper (lo script dà il rapporto).
