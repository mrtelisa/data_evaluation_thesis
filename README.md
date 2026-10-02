# data_evaluation_thesis

Analysis of the experimental protocol: blind reaching test (5 repetitions), reaching training (2 sessions), real robot (2 tests), plus questionnaires and BoMI calibrations.

Reference papers: Bellitto et al. 2025, J. Neural Eng. 22 046036 (metric definitions, Fig. 4) and Najie et al. (box plots).

## Run

```bash
python3 run_all.py
```

Each task can also be run on its own (`python3 reaching_blind/blind_analysis.py`, ...). Requires Python 3 with `pandas`, `scipy`, `matplotlib`, `openpyxl`.

## Input

Copy the result files of `reachy2_teleop_grasping_and_simulation` into the `results/` folder of each analysis (files are searched in every subfolder except `statistics/`; if a session was repeated, `_1`, `_2`, ... the last one is used):

| Task | Folder | Files used |
|---|---|---|
| Blind test | `reaching_blind/results/` | `<subject>_blind_test<N>_raw.csv`, `_summary.json` |
| Training | `reaching_regions/results/` | `<subject>_regions_<pre\|post>_raw.csv`, `_summary.json` |
| Real robot | `real_robot/results/` | `<subject>_run<N>_session.json`, `_regions.csv`, `_odometry.csv` |

## Output

- `jamovi/`: one file per metric (`.omv` with column descriptions, and `.csv`), in the layout of the jamovi models. One row per participant.
  - `jamovi/training/<abbr>_<metric>`: `EE_T1_B1 ... EE_T2_B12` (as `training_reaching_out.omv`; block = mean of 8 targets) for EE, DE, XE, MD, LI. Model: repeated-measures ANOVA Training (T1, T2) × Block (B1–B12).
  - `jamovi/blind/<abbr>_<metric>`: `HR_T1_B1 ... HR_T5_B3` (block = mean of 4 trials) for HR, TC, TT, EE, XE, DE, MinD, TTR, LI, MD. Model: repeated-measures ANOVA Test (T1–T5) × Block (B1–B3).
  - `jamovi/reachy/<abbr>_<metric>`: `Test1_andata`, `Test2_andata`, `Test1_ritorno`, `Test2_ritorno` (as `reachy_stats.omv`) for CT, PL, MS, JK, RT; the CT file also has grasp, posizionamento and totale. the CT file also has `andata_ritorno` (outbound + return, without manipulation). Model: repeated-measures ANOVA test 1 vs test 2, once on the outbound and once on the return columns.
  - Every `.omv` already contains the repeated-measures ANOVA(s), set up as the supervisor's models (`common/jamovi_template/`): factors and cells, η² and partial η², sphericity tests with Greenhouse-Geisser, post hoc tests only in the robot files (training and blind: no post hoc terms, the tables would be too long for jamovi). The results are saved empty and marked stale (as jamovi does for templates), because jamovi does not recompute saved analyses when opening a file: after opening, change the plot theme or the decimal symbol once in the jamovi settings (⋮ menu, top right) and every analysis is computed.
- `cross_analysis/statistics/`: exploratory Spearman correlations (no correction) between the tasks (blind/training vs robot) and between questionnaires and performance, plus questionnaire descriptives; uses only the metrics below. Runs last in `run_all.py`.
- `<task>/statistics/`: `report.txt` (descriptives, ANOVA, post hoc for every metric), `.csv` tables, `plots/` (`.png` 300 dpi and `.pdf`; `plots/single/` has every panel on its own).

## Metrics

Definitions as Bellitto et al. 2025 (par. 2.4.2) and the lab script `computereachmetrics.m`. The start position is the cursor position at target onset. Errors and distances in cm, as in the paper: the 1200 × 650 px canvas is drawn at scale 1.53 (`canvas_scale` in the summary.json) on a 1920 × 1080 px, 340 × 190 mm screen, so 1 canvas px = 0.27 mm (`canvas_to_cm` in `common/reach_metrics.py`). Hits and target entries stay in canvas pixels, as in the task.

**Training** (outward movements only, 96 per session, 12 blocks of 8 targets):
- Euclidean error, extent error, direction error at 1 s after target onset (when the cursor reappears);
- movement duration and linearity index from target onset to target entry.

**Blind test** (12 trials of 4 s per test, cursor always hidden; 3 blocks of 4 trials; block value = mean of its trials, test value = mean of the 12 trials):
- hit rate (0.5 s inside the target), target crossing rate (cursor enters the target at least once), time on target;
- Euclidean, extent and direction error at the end of the 4 s, minimum distance from the target;
- time to reach (hits only), linearity index (to target entry or end of trial), movement duration (movement onset at 40 px/s to target entry, or to the last sample above 40 px/s).

**Real robot** (test 1 vs test 2; input files `run1`/`run2`):
- completion time per phase and total: outbound = from the start to the object selection (task phases outbound + approach), grasp = object selection to object in hand, return, placement;
- outbound and return: path length, mean speed, mean jerk (base odometry), region transitions of the cursor in the 3 × 3 interface.

**Exploratory metrics** (report and `*_exploratory_*.csv` only, not in figures or jamovi tables; used in the thesis discussion, p not corrected):
- blind test: share of trials ending in the region of the target;
- robot, per driving leg: share of driving time in the stop region, region transitions per minute of driving, mean distance of the cursor from the centre while moving (0 = centre, 1 = border).

## Statistics

Every participant counts once: one value per participant and condition, then statistics across participants. Every analysis has two versions of the tests (mean ± SD and median [IQR] are both in `report.txt`):

| Version | Tests |
|---|---|
| Parametric | repeated-measures ANOVA (Greenhouse-Geisser, η², partial η²), post hoc paired t-tests (Bonferroni) |
| Non-parametric | Friedman, Wilcoxon signed-rank (Holm) |

Figures show only medians: box plots with whiskers at 1.5 IQR (`*_boxplots`, blind and robot) and median [IQR] curves for the training (`figure4_median_iqr`). The training also has `return_median_iqr` (return to the centre: movement duration and linearity index, plots only, no statistics; data in `training_return_trials.csv`). Every panel is also saved on its own in `plots/single/`: the overview figures have a title, the single plots have no title and a note under the plot saying what is shown (median [IQR], whiskers, dots) and what the asterisks mean. In the blind test, movement duration and time to reach are not plotted, and time on target and linearity index are only in their single plots (`plots/single/`); `plots/single/linearity_index_no_outlier_boxplot` repeats the linearity index without its highest value, to show the other participants.

Planned post hoc comparisons: training T1 B1 vs T1 B12, T1 B12 vs T2 B1, T1 B12 vs T2 B12; blind consecutive tests (1–2, 2–3, 3–4, 4–5); robot test 1 vs test 2. In the figures: `*` p < 0.05, `**` p < 0.01, `***` p < 0.001 (corrected p). The brackets of every figure, box plots included, come from the paired t-tests (Bonferroni), as the post hoc of the ANOVA; the Wilcoxon tests are only reported in `report.txt`.
