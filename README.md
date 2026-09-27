# data_evaluation_thesis

Analisi dei dati del protocollo sperimentale: questionari, 5 blind reaching test, 2 training a 9 regioni, 2 run sul robot reale, e i collegamenti fra le parti.

## 1. Dove incollare i file

Gli script del repo `reachy2_teleop_grasping_and_simulation` scrivono i risultati in tre cartelle. Copia il **contenuto** di ognuna nella cartella `results/` dell'analisi corrispondente (vanno bene anche sottocartelle: i file vengono cercati ovunque dentro la cartella dell'analisi, tranne in `statistics/`).

| Script del repo | Cartella del repo | Incolla in | File usati |
|---|---|---|---|
| `reaching_blind.py` (branch `new_protocol`) | `results_blind/` | `reaching_blind/results/` | `<soggetto>_blind_test<N>_trials.csv`, `_trajectory.csv` |
| `reaching_regions.py` (branch `new_protocol`) | `results_regions/` | `reaching_regions/results/` | `<soggetto>_regions_<pre\|post>_trials.csv`, `_trajectory.csv`, `_summary.json` |
| `reachy_control.py` (branch `main`) | `results_robot/` | `real_robot/results/` | `<soggetto>_run<N>_session.json`, `_odometry.csv` |
| Google Form | (export .xlsx) | `questionari/risposte/` | i 5 file .xlsx |

- I file `_blocks.csv` e `_summary.json` del blind test, e `_regions.csv` del robot, si possono incollare lo stesso ma non servono: le statistiche vengono ricalcolate dalle singole prove.
- **Non analizzati**: le mappe in `calibrations/` (customizzazione) e la bag della free exploration in simulazione (non ha metriche nel protocollo).
- **Ripetizioni**: se una sessione è stata rifatta, il repo aggiunge `_1`, `_2`, ... al nome. Si usa sempre l'**ultima**; le altre sono elencate nel report come scartate.
- **ID dei soggetti**: devono essere gli stessi negli script e nei questionari. `S1`, `s01`, `S001` vengono riconosciuti come lo stesso soggetto (`S001`).

## 2. Come lanciare

```bash
cd data_evaluation_thesis
python3 run_all.py
```

Lancia tutto nell'ordine giusto: questionari → blind test → training → robot → collegamenti. Ogni parte si può lanciare anche da sola (`python3 reaching_blind/blind_analysis.py`, ...). Un'analisi senza dati viene saltata.

Serve Python 3 con `pandas`, `scipy`, `matplotlib`, `openpyxl` (il `python3` di sistema li ha già; altrimenti `pip install pandas scipy matplotlib openpyxl`).

## 3. Cosa leggere

Ogni analisi scrive in `<cartella>/statistics/`:

- **`report.txt`**: il riassunto in italiano, da leggere per primo. Per ogni confronto: valori tipici prima e dopo, se è migliorato o peggiorato, se è significativo, quanto è grande l'effetto.
- **`plots/`**: i grafici (in inglese, pronti per la tesi).
- **`.csv`**: tutti i numeri, per le tabelle della tesi.

| Cartella | Report | Grafico principale |
|---|---|---|
| `reaching_blind/` | 5 metriche principali, Friedman + 4 confronti fra test consecutivi | `blind_main.png` |
| `reaching_regions/` | 4 metriche principali + 3 richieste, curve di apprendimento, 3 confronti, plateau | `training_learning_curves.png` |
| `real_robot/` | 4 metriche primarie + le 13 metriche per fase, conteggi | `robot_primary.png`, `robot_transitions.png` |
| `cross_analysis/` | trasferimento cursore → robot, questionari | `cross_transfer.png` |

## 4. Le scelte statistiche, in parole semplici

- **Un valore per soggetto.** Ogni soggetto fa tante prove: le riassumiamo con la **mediana** (robusta alle prove anomale) o con la **percentuale** per le metriche sì/no (hit, regione corretta). Così ogni soggetto pesa 1.
- **Descrittive: mediana [Q1–Q3].** La mediana è il valore tipico del gruppo, [Q1–Q3] la fascia della metà centrale dei soggetti. Più robuste di media ± SD con pochi soggetti e dati asimmetrici (tempi, percentuali vicine al 100%). Media e SD sono comunque nei .csv.
- **Grafici con una linea per soggetto** (spaghetti plot): mostrano se tutti migliorano o solo alcuni. Sono la parte più importante con pochi soggetti.
- **Wilcoxon** per confrontare due momenti (es. run 1 vs run 2): guarda, soggetto per soggetto, se migliora o peggiora e di quanto.
- **Friedman** per i 5 blind test insieme: c'è qualche differenza fra i 5?
- **Pochi confronti, decisi prima** di guardare i dati, e **correzione di Holm** dentro ogni famiglia (più confronti fai, più è facile trovare qualcosa per caso):
  - blind test: 4 confronti fra test consecutivi (1–2, 2–3, 3–4, 4–5), per ogni metrica;
  - training: 3 confronti (inizio vs fine training 1, fine training 1 vs inizio training 2, fine training 1 vs fine training 2), per ogni metrica;
  - robot: run 1 vs run 2 sulle 4 metriche primarie (`PRIMARY` in `robot_analysis.py`).
  Le altre metriche sono calcolate allo stesso modo ma sono **esplorative** (p non corretto).
- **Effect size** per ogni confronto, significativo o no: rank-biserial r (Wilcoxon), W di Kendall (Friedman), rho (Spearman). 0.1 piccolo, 0.3 medio, 0.5 grande. Con pochi soggetti conta più del p-value.
- **"cambiano k"** nel report: quanti soggetti cambiano davvero fra i due momenti. Con 1 soggetto su 9 l'effetto risulta "grande" ma riguarda solo lui.
- **Con 9 soggetti** un confronto può risultare significativo dopo Holm solo se quasi tutti (8–9 su 9) vanno nella stessa direzione. Con 7 soggetti e 4 confronti non è possibile: in quel caso si discutono effect size e grafici.
- **Correlazioni (Spearman)** in `cross_analysis`: solo esplorative, da leggere come tendenze.

## 5. Le metriche

**Blind test** (principali): hit rate (cursore nel cerchio per 0.5 s), end-point error, reach time, initial direction error, region accuracy.

**Training** (principali): reach time, normalized path length, initial direction error, correct region at reveal; in più speed peaks, movement time, reached while hidden. La curva sui 24 blocchi da 8 target viene fittata con `y = a + b·exp(−k·(blocco−1))`: `a` è il plateau, e il report dice in quale blocco si raggiunge il 95% del miglioramento (≤ 12: già nel training 1).

**Robot** (primarie): tempo dell'andata, tempo del ritorno, salti non adiacenti e cambi di comando/min nell'andata. Poi le 13 metriche per fase (tempi, cambi di comando, dwell rifiutati e repositioning, tempo fermo, collisioni, salti non adiacenti, inversioni, durata media dei comandi, passaggi per lo stop, entropia, path length e rotazione cumulata per andata e ritorno, borracce cadute). Run 1 e run 2 si confrontano fase per fase.

**Collegamenti**: la stessa abilità con il cursore e con il robot (es. region accuracy nel blind test 3 vs salti non adiacenti nella run 1; dwell time a fine training vs durata dei comandi sul robot), e questionari vs prestazioni. SUS confrontato con il riferimento 68.
