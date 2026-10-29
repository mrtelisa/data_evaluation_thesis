# BoMI Teleoperation: Data Analysis

Data analysis of the experimental protocol of my Master's thesis, in which 10 participants learned a markerless Body-Machine Interface (BoMI) and used it to teleoperate the **Reachy 2** humanoid robot. The experiment code is in [reachy2_teleop_grasping_and_simulation](https://github.com/mrtelisa/reachy2_teleop_grasping_and_simulation).

The repository contains the raw results of each task, the scripts that compute the metrics, and the statistical analysis that produces the figures and tables of the thesis.

## Experimental protocol

Each participant went through four parts in a single session. The scripts come from two branches of the experiment repository: the full task on the real robot (`reachy_control.py`) from [`main`](https://github.com/mrtelisa/reachy2_teleop_grasping_and_simulation/tree/main), all the other steps from [`new_protocol`](https://github.com/mrtelisa/reachy2_teleop_grasping_and_simulation/tree/new_protocol).

| Part | Step | Script |
|---|---|---|
| **A. Questionnaires** | Subject information, Edinburgh Handedness Inventory | Google Forms |
| **B. Simulation** | 1. BoMI customization | `customize_bomi.py` |
| | 2. Blind test 1 | `reaching_blind.py --test 1` |
| | 3. Reaching training 1 | `reaching_regions.py --phase pre` |
| | 4. Blind test 2 | `reaching_blind.py --test 2` |
| | 5. Reaching training 2 | `reaching_regions.py --phase post` |
| | 6. Blind test 3 | `reaching_blind.py --test 3` |
| | 7. Free exploration in Gazebo | `socket_client.py` |
| **C. Real robot** | 1. Full task, test 1 | `reachy_control.py --run 1` |
| | 2. Blind test 4 | `reaching_blind.py --test 4` |
| | 3. Full task, test 2 | `reachy_control.py --run 2` |
| | 4. Blind test 5 | `reaching_blind.py --test 5` |
| **D. Questionnaires** | NASA-TLX, System Usability Scale (SUS) | Google Forms |

The blind test is repeated five times to track how the map is learned across the whole session. In the full task on the real robot, the participant drives Reachy 2 to a table, selects an object that the robot grasps, and brings it to a second table where it is placed.

## Metrics

Reaching metrics follow Bellitto et al. 2025 (*J. Neural Eng.* 22, 046036). Distances are converted from screen pixels to cm.

| Task | Metrics |
|---|---|
| **Reaching training** (96 targets per session) | Euclidean, extent and direction error at 1 s from target onset (when the cursor reappears), movement duration, linearity index |
| **Blind test** (12 trials of 4 s per test) | Hit rate, Euclidean, extent and direction error at the end of the trial, movement duration |
| **Real robot** (test 1 vs test 2) | Completion time per phase (outbound, grasp, return, placement, total); for outbound and return: path length, mean speed, jerk, cursor region transitions |
| **Questionnaires** | Raw NASA-TLX (0-100), SUS score (Brooke 1996), Edinburgh laterality quotient, demographics |

The BoMI calibrations are analysed descriptively: variance accounted for (VAF) and PCA of the shared map, and the customization parameters of each participant.

## Statistical analysis

| Task | Design | Test |
|---|---|---|
| Reaching training | 2 trainings × 3 blocks of 32 targets | Two-way repeated-measures ANOVA (Training × Block) |
| Blind test | 5 tests (one value per test: mean of 3 blocks of 4 trials) | One-way repeated-measures ANOVA (Test) |
| Real robot | Test 1 vs test 2, outbound and return analysed separately | Repeated-measures ANOVA (2 levels) |

- **Normality:** log transform for skewed metrics (training movement duration, blind Euclidean error, robot completion time). For the other robot metrics, the transform is applied only if the Shapiro-Wilk test on the differences is significant.
- **Sphericity:** Mauchly's test, with Greenhouse-Geisser correction when p < 0.05.
- **Post hoc:** pairwise comparisons on all pairs (15 for the training blocks, 10 for the blind tests), Holm-corrected within each metric. For the robot, Holm correction on the outbound/return pair.
- **Effect size:** partial η².
- **Non-parametric check:** Friedman and Wilcoxon signed-rank tests (Holm-corrected), for metrics with non-normal residuals.

Descriptive statistics and plots use the original data, while the log transform is used only for the tests. The jamovi files in `jamovi/` were used as a reference to check the results.

## Repository structure

```
run_all.py                  # runs every analysis in order
common/                     # shared code: data loading, reaching metrics, statistical tests, plots
questionari/                # questionnaires: answers (risposte/), scoring scripts, statistics
reaching_regions/           # reaching training: results/ and per-trial metrics
reaching_blind/             # blind test: results/ and per-trial metrics
real_robot/                 # real robot: results/ and per-participant metrics
new_stats/                  # statistical analysis: reports, tables, plots per task (training/, blind/, robot/)
calibrations/               # BoMI maps of each participant and their analysis
jamovi/                     # jamovi reference analyses
```

Each analysis writes its output in a `statistics/` subfolder. The statistical analysis writes, for each task, a `report.txt` with every test (F, df, p, Holm-corrected p, partial η²), CSV tables of descriptives and tests, and the plots.

## Usage

Requires Python 3 with `pandas`, `numpy`, `scipy`, `matplotlib` and `openpyxl`.

```bash
python3 run_all.py
```

The scripts can also be run one at a time, in this order:

```bash
python3 questionari/run_all.py
python3 reaching_blind/blind_analysis.py
python3 reaching_regions/training_analysis.py
python3 real_robot/robot_analysis.py
python3 new_stats/new_stats_analysis.py      # needs the outputs of the three scripts above
python3 calibrations/calibration_analysis.py
```
