# data_evaluation_thesis

Analysis of the experimental protocol used in the thesis: blind reaching test (5 repetitions), reaching training (2 sessions), real robot (2 tests), plus questionnaires and BoMI calibrations.

The statistics, figures and tables of the thesis come from `new_stats/new_stats_analysis.py`, the analysis of the PhD student (her jamovi files and summaries are in `newStats_blind/`, `newStats_reachy/`, `newStats_training/`). The older analyses (full ANOVA reports, jamovi tables, cross-task correlations, 3-block training, summary PDF) were removed: they are in the git history (commit `dd8bac3`, "Checkpoint before cleanup").

## Run

```bash
python3 run_all.py
```

Each step can also be run on its own, in this order (new_stats reads the files of the three task scripts):

| Step | Script | Output |
|---|---|---|
| Questionnaires | `questionari/run_all.py` | `questionari/statistics/` (scores, descriptives, plots) |
| Blind test | `reaching_blind/blind_analysis.py` | `reaching_blind/statistics/blind_trials.csv` (one row per trial) |
| Training | `reaching_regions/training_analysis.py` | `reaching_regions/statistics/training_trials.csv`, `training_return_trials.csv`, `training_return_curves.pdf` (thesis figure of the returns to the home) |
| Real robot | `real_robot/robot_analysis.py` | `real_robot/statistics/robot_subjects.csv` (one row per participant, test, metric and phase) |
| Statistics of the thesis | `new_stats/new_stats_analysis.py` | `new_stats/` (see below) |
| Calibrations | `calibrations/calibration_analysis.py` | `calibrations/statistics/` |

Requires Python 3 with `pandas`, `scipy`, `matplotlib`, `openpyxl`.

The thesis figures and tables are written inside this repository by default (`new_stats/thesis/`, `reaching_regions/statistics/training_return_curves.pdf`). With `--thesis` (`new_stats_analysis.py --thesis`, `training_analysis.py --thesis`) they go directly into `Tesi_latex/Chapters/Figures/results` and `Chapters/Tables`, overwriting the ones of the thesis. `questionari/nasa.py` always copies `nasa_boxplot.pdf` into the thesis folder.

## Input

The result files of `reachy2_teleop_grasping_and_simulation` are in the `results/` folder of each task (files are searched in every subfolder except `statistics/`; if a session was repeated, `_1`, `_2`, ... the last one is used):

| Task | Folder | Files used |
|---|---|---|
| Blind test | `reaching_blind/results/` | `<subject>_blind_test<N>_raw.csv`, `_summary.json` |
| Training | `reaching_regions/results/` | `<subject>_regions_<pre\|post>_raw.csv`, `_summary.json` |
| Real robot | `real_robot/results/` | `<subject>_run<N>_session.json`, `_regions.csv`, `_odometry.csv`, `_cursor.csv` |

Questionnaire answers: `questionari/risposte/` (Google Forms exports). Calibrations: `calibrations/` (`.npz` maps, `_customization.json`, `shared_calib.npy`).

## Output of new_stats

One subfolder per task (`training/`, `blind/`, `robot/`), each with `statistics/report.txt` (every number: F, df, p, partial η², comparisons), `<task>_descriptives.csv`, `<task>_tests.csv` and `statistics/plots/single/` (every panel on its own, `.png` 300 dpi and `.pdf`). Plus:
- `thesis/figures/*_newstats.pdf`: figures of the thesis (chapter 6);
- `thesis/tables/*_newstats.tex`: tables of the thesis (included with `\input`);
- `plot_ppt/`: plots for the presentation (no legend of the participants, no notes).

The statistical choices (blocks of 32 for the training, one value per test for the blind test, log transforms, Greenhouse-Geisser when Mauchly p < 0.05, Holm corrections, Friedman/Wilcoxon checks) are described at the top of `new_stats/new_stats_analysis.py`.

## Metrics

Definitions as Bellitto et al. 2025 (par. 2.4.2) and the lab script `computereachmetrics.m`. The start position is the cursor position at target onset. Errors and distances in cm, as in the paper: the 1200 × 650 px canvas is drawn at scale 1.53 (`canvas_scale` in the summary.json) on a 1920 × 1080 px, 340 × 190 mm screen, so 1 canvas px = 0.27 mm (`canvas_to_cm` in `common/reach_metrics.py`). Hits and target entries stay in canvas pixels, as in the task.

**Training** (outward movements, 96 per session; new_stats uses 3 blocks of 32 targets):
- Euclidean error, extent error, direction error at 1 s after target onset (when the cursor reappears);
- movement duration and linearity index from target onset to target entry.

**Blind test** (12 trials of 4 s per test, cursor always hidden; 3 blocks of 4 trials; block value = mean of its trials, test value = mean of the 3 blocks):
- hit rate (0.5 s inside the target);
- Euclidean, extent and direction error at the end of the 4 s;
- movement duration (movement onset at 40 px/s to target entry, or to the last sample above 40 px/s; single plot only, not in the thesis table).

**Real robot** (test 1 vs test 2; input files `run1`/`run2`):
- completion time per phase and total: outbound = from the start to the object selection (task phases outbound + approach), grasp = object selection to object in hand, return, placement;
- outbound and return: path length, mean speed, mean jerk (base odometry), region transitions of the cursor in the 3 × 3 interface.

Exploratory metrics (report only): blind test, share of trials ending in the region of the target; robot, per driving leg, share of driving time in the stop region, region transitions per minute of driving, mean distance of the cursor from the centre while moving (0 = centre, 1 = border).
