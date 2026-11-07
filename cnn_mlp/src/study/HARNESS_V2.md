# Robustness / physics harness v2

Owner files: `pipeline_robustness.py`, `test_corruptions.py`, `aggregate_seeds.py`, `build_ablation_table.py`,
`run_local_eval.py`, this note, `tokamark_robustness_study_template.ipynb` (+ `scripts/build_lstm_robust_kernel.py`).

## 1. What changed in `pipeline_robustness.py`

| Area | v1 (what produced every earlier number) | v2 |
|---|---|---|
| Corruption axes | treated the last axis as time on every branch. Profile branches are `(B,1,1,T,n_sensors)`, so temporal-gap / disruption zeroed sensor columns, ffill ran across sensors, and channel / category / group ablations zeroed one time step | time axis `-2` on profile branches, `-1` on solenoid / ip; whole-branch loss blanks the whole branch (already fixed before this package, kept and now under test) |
| Physics sample | 35%-level heuristic on every 100th window (~35 windows / pass) | Module 1 v2 against EFIT truth (`TruthIndex`) on ~1 in `TM_ROB_PHYS_STRIDE` (default 4) windows, selected by a hash of (shot, window) so the same windows are scored in every pass; windows with no truth are skipped and counted |
| Uncertainty | none | `PHYS_ci95`: 95 % shot-level cluster bootstrap CI (1000 resamples) for `boundary_mean_cm`, `axis_shift_cm`, `boundary_hausdorff_cm` |
| Per-window records | none | `PHYS_records`: one row per scored window (`shot, win, boundary_mean_cm, boundary_hausdorff_cm, axis_shift_cm, xpoint_err_cm, kappa_err, delta_err, boundary_failed, is_false_alarm, is_missed_alarm`) so any statistic can be re-bootstrapped offline |
| Counts | none | `PHYS_counts`: `n_selected, n_used, n_no_truth, n_no_efit_lcfs, n_boundary_fail` (the aggregate in `PHYS` also has `n_boundary_valid`, `boundary_fail_rate_pct`) |
| Paths | `rob_paths()` ignored `TM_RUN_TAG` (model was read from the tagged dir, results written to the untagged one) | `rob_paths()` uses `pipeline.paths()`: `out/<split>/<TM_RUN_TAG>/robustness`; `TM_ROB_DIR` overrides the output dir |
| Resume | none | `TM_ROB_RESUME=1`: clean pass and every finished pass are reused from `robust_metrics.json` (it is rewritten after every pass) |
| Entry point | `cmd_eval_robust` only | `eval_sweep(loader, model, dev, meta, split, rc, P_, train_means=None)`: any re-iterable of collated batches works (the local CPU run feeds a disk cache) |

`robust_metrics.json` schema is unchanged for `report/build_assets.py` (`clean`, `corrupted["family|param|impute"]`, `PHYS`,
`viz_samples`, `spatial_abs_err`, ...). New keys only: `PHYS_counts`, `PHYS_records`, `PHYS_ci95` (inside `clean` and every
corrupted cell) and `train_means` (top level). `PHYS` is now an EFIT-mode aggregate: **do not compare it with any v1 PHYS**.

Notes on what the sweep does and does not do:
* `zero` fill is applied on STANDARDISED inputs, i.e. zero = the split's dataset mean for every signal. `mean` fill uses the
  mean of the non-zero standardised train values (first 40 train batches), which is close to zero for most signals, so
  zero- and mean-fill are nearly the same intervention for the profile branches; only `ffill` is a genuinely different imputation.
* A failure-aware checkpoint receives a health mask: 0 for every branch removed whole (channel / category / group loss),
  1 for partial damage (random dropout, gaps), exactly as before.
* Checkpoints in `runs/` were trained with the legacy flux-loop PINN (`TM_PINN_MODE=legacy`, pixels vs the TRUE psi). The default of
  the current `pipeline.py` is `measured`. Evaluating the old checkpoints is unaffected, but they are labelled "mask + PINN (legacy)"
  everywhere in this package; a `measured`-PINN checkpoint has yet to be trained.

## 2. Tests (`test_corruptions.py`)

`python -m pytest cnn_mlp/src/study/test_corruptions.py -q` (needs torch + numpy; the real-batch half needs `REAL_BATCH=<batch.npz>`,
default is the local scratchpad batch built from shot 13222 by the real `TokaMarkDataset`, 16 windows, layout
`[(16,1,1,25,15),(16,1,1,25,40),(16,1,1,25,18),(16,1,1,25,18),(16,1,1,250,12),(16,1,1,20,10),(16,1,1,20),(16,1,1,20)]`; skipped when absent).
Each test runs on a synthetic tensor of that layout and on the real batch.
Covered: layout / time-axis helper; channel ablation (n = 1, 3, 6) x zero / mean / ffill zeroes whole branches, others bit-identical,
`dead_channels` correct; category and correlated-group loss; temporal gap and disruption-proximate (contiguous span of
round(frac * T) steps over all sensors, nothing outside it modified, span at the end for disruption); ffill carries the last pre-gap
value of the SAME sensor along time (the pre-fix code filled across sensors); mean fill touches only holes; random dropout zeroes
exactly round(sev * n) values and ffill runs along time; failure-aware path gets health 0 for lost branches only (1 for dropout /
gaps / disruption) and the plain path never receives a health argument; hash window selection; shot bootstrap CI; `rob_paths` honours `TM_RUN_TAG`.
Mutation check: re-introducing the old axis (`_time_axis -> -1`) makes 16 of the gap / ffill / mean tests fail.

## 3. Stale legacy copy

`cnn_mlp/src/robustness.py` (796 lines) is the pre-fix copy of this harness: it still has the last-axis-as-time bug (the lines the fix touched
are at ~153 / 220 / 242), no TruthIndex, no health mask, no `eval_sweep`. It was NOT edited. Do not use it; `scripts/sync_robustness_nb.py`
(which embeds it into `tokamark_robustness_kaggle.ipynb`) still ships the stale copy; use `scripts/build_lstm_robust_kernel.py` (template notebook) instead.

## 4. Multi-seed aggregation and the results table

```
python aggregate_seeds.py OUT.json RUN_DIR [RUN_DIR ...]          # mean / std (ddof=1) / n per cell over seeds
python build_ablation_table.py --out results_v2/ablation \
    --plain DIR [DIR..] --mask DIR [DIR..] --mask_pinn DIR [DIR..] [--mitigation MIT.json [MIT.json ..]]
```
Rows: baseline (plain, zero-fill); + best fixed imputation (plain, lower-NRMSE of mean / ffill per cell); + failure-aware
training (mask-only, no imputation); + flux-loop consistency loss (mask + PINN); supplementary rows (mask / mask+PINN with the best
fixed imputation); + adaptive mitigation; Oracle mitigation. Columns: Clean, Channel loss (6 sensors), Group loss (magnetics_all),
Temporal gap (40 %), Pre-disruption (`disruption_proximate` 25 %), Latency. Every cell is `boundary cm (NRMSE)`, with `mean ± std`
when more than one seed; `--` when missing. Writes `<out>.md/.tex/.csv`. The Latency column and the two mitigation rows are `--` until
a mitigation JSON is supplied.

Mitigation JSON schema (one file per seed; several are averaged):
```json
{"adaptive": {"clean": {"boundary_mean_cm": 1.2, "nrmse": 0.15},
              "channel_loss_6": {...}, "group_loss": {...}, "temporal_gap_40": {...}, "pre_disruption_25": {...},
              "latency_ms": 2.1},
 "oracle":   {"clean": {...}, "channel_loss_6": {...}, "group_loss": {...}, "temporal_gap_40": {...}, "pre_disruption_25": {...}}}
```
Numbers may also be `{"mean": x, "std": s, "n": n}`. The boundary error must be the Module 1 v2 EFIT-truth `boundary_mean_cm` on the same
window protocol; `nrmse` is optional (cell prints `--` for it).

## 5. Commands

Kaggle (dual T4, per model): rebuild the kernel with `python cnn_mlp/scripts/build_lstm_robust_kernel.py` (it now also ships `efit_truth.py`
and `aggregate_seeds.py`) and push it; the notebook sweep step is `python -u robustness.py eval-robust --split random` with
`TM_RUN_TAG=<tag>` (and `TM_ROB_PHYS_STRIDE=4`, the default).
Modal: `python cnn_mlp/modal/app.py robustness ...` (see `cnn_mlp/modal/RUN_MATRIX.md`); outputs land in `out/random/<tag>/robustness/`.
Local CPU: see section 6.

## 6. Local CPU re-evaluation (corrected harness) - `run_local_eval.py`

Environment that works on this laptop (no GPU): system Python 3.12 + torch 2.2 CPU, with `PYTHONPATH`/`--pylib` pointing at a
directory holding numpy 1.26.4, zarr 3.0.10, xarray 2026.9, numcodecs 0.15, torchinfo (`scratchpad\pylib5`; numpy 2 breaks torch's
numpy bridge, zarr 2 cannot read the v3 stores, xarray < 2025 raises FileNotFoundError instead of KeyError on shots with a missing group).

```
python cnn_mlp/src/study/run_local_eval.py --runs mlp_fa_v3 lstm_fa <run_dir> [<run_dir> ...] --skip-done --rob-seed 42 \
    --data-root <dir with random/<shot>.zarr> --cache-dir <outside repo> --work-dir <outside repo> --pylib <pylib5>
python cnn_mlp/src/study/run_local_eval.py --discover cnn_mlp/results_v2 ...        # every dir with best_model.pt + run_config.json
```
A run is a legacy alias (`mlp_fa_v3` = runs/mlp_fa_robust_v3, `lstm_fa` = runs/lstm_fa_robust) or any directory containing
`model/best_model.pt` next to `run_config.json` (model / failure_aware / variant / dropout are read from it; nothing is hard-coded to a
name). One subprocess per run, per-pass checkpoint in `results_v2/corrected_robustness/sweeps/<tag>/robust_metrics.json`; re-running resumes.
Protocol: all 60 official test shots of the random split (46 yield windows: 3 450 windows, identical to the Kaggle run), batch 64, `--rob-seed 42`
(the seed of the old sweeps; the per-batch RNG stream still differs because the batch size is 64 not 512, so channel choices of
`channel_ablation` / dropout masks are not bit-identical to the old run), physics on ~1 in 4 windows (867 windows, 768 with an EFIT
LCFS, 99 without), mean-fill values from 8 train shots (the Kaggle run used the first 40 train batches of 606 shots).
Clean gate: **harness clean NRMSE = 0.151380 vs 0.151380 in the Kaggle run -> PASS** (same checkpoint, same windows).
Cost: ~1.1 min per pass on 4 CPU cores (about 65 min per model for the 63-pass sweep, shared with other jobs).

Checkpoint labels: `mlp_fa_v3` and `lstm_fa` are the OLD Kaggle checkpoints = mask + augmentation + **legacy** flux-loop PINN (pixels vs the
true psi); the `measured` PINN of the current `pipeline.py` has no checkpoint yet. The old plain CNN+MLP checkpoint (the report's
baseline, `cnn_mlp/results/robustness`) is NOT in the repo (only its metrics); `results_v2/kaggle_mlp_plain_s1` is a new plain CNN+MLP.

### 6.1 Result: failure-aware CNN+MLP (`mlp_fa_v3`, legacy PINN) - `results_v2/corrected_robustness/sweeps/mlp_fa_v3/`

All numbers below are from `robust_metrics.json` / `compare_old_new.json` in that folder (46 shots, 3 450 windows, 63 passes).

| quantity | old report | corrected |
|---|---|---|
| clean NRMSE | 0.1514 | 0.1514 (identical) |
| clean boundary error (cm), EFIT truth | 3.48 (heuristic 35 % contour, 35 windows: INVALID) | **3.65 [3.40, 3.90]** (95 % shot-bootstrap, 768 windows); measurement floor on the true psi 0.17 cm |
| clean axis shift (cm) | n/a | 6.11 [5.36, 7.02]; X-point detection 53 %; boundary failures 0 % |
| RS, zero-fill (mean of clean/NRMSE over 18 settings) | 0.562 | 0.581 |
| RS, best of zero / mean / ffill per setting | 0.583 | **0.836** |

Per-family RS (clean NRMSE / corrupted NRMSE, mean over severities; old -> corrected):

| family | zero-fill | mean-fill | ffill |
|---|---|---|---|
| random dropout | 0.109 -> 0.115 | 0.111 -> 0.108 | 0.065 -> **0.668** |
| temporal gap | 0.251 -> 0.345 | 0.271 -> 0.349 | 0.118 -> **0.835** |
| disruption-proximate | 0.520 -> 0.519 | 0.520 -> 0.510 | 0.533 -> **1.004** |
| channel ablation | 0.724 -> 0.739 | 0.699 -> 0.718 | 0.697 -> 0.700 |
| group failure | 0.800 -> 0.800 | 0.800 -> 0.800 | 0.800 -> 0.800 |
| category ablation | 0.970 -> 0.970 | 0.970 -> 0.970 | 0.970 -> 0.970 |

Boundary error (cm, EFIT truth, mean [95 % CI]) / NRMSE for the table settings, zero-fill unless noted:
channel loss (6 sensors) 21.5 [18.6, 24.3] / 0.399; magnetics group loss 15.3 [14.1, 16.6] / 0.263 (identical for mean and ffill: the dead branch is gated out by the health mask);
temporal gap 40 % 23.3 [20.7, 25.8] / 0.484 (ffill: 5.3 [4.6, 6.2] / 0.164); pre-disruption 25 % 12.1 [10.4, 14.1] / 0.315 (ffill: 3.65 / 0.151 = clean).
Random dropout 10 %: zero-fill 17.4 cm / 0.857, ffill 5.4 cm / 0.190.

Which conclusions of the earlier report survive (for this checkpoint):
* **Invalid, must be dropped:** every Module 1 boundary / axis / X-point / alarm number (35 % contour, wrong grid scale, 35 windows) - e.g. the 0.9 cm clean boundary error is really about 3.7 cm vs EFIT.
* **Reversed:** "forward-fill is the worst imputation / does not help temporal gaps". The old ffill carried values across sensors and zeroed sensor columns (temporal_gap / disruption) and also mis-filled dropout; with the fix ffill is by far the best fixed imputation (gap RS 0.12 -> 0.84, dropout 0.07 -> 0.67, disruption 0.53 -> 1.00). Zero- and mean-fill stay almost equal (zero = dataset mean in standardised space).
* **Unchanged:** random dropout with zero/mean fill is catastrophic (RS ~0.11); whole-branch loss is cheap for a failure-aware model because the health mask removes the branch (group / category RS identical old vs new; the pre-fix code had zeroed one time step only, which a gated model ignores anyway); the ranking channel ablation < gap < dropout in damage. Channel ablation changes only by the random channel draw.
* **Not claimed here:** conclusions about the plain CNN+MLP (old checkpoint missing) or about PINN vs mask-only (needs the new multi-seed runs). Clean boundary error of 3.7 cm is large against the 3-6 cm grid pitch; do not present sub-pixel numbers.

### 6.2 Other runs

Status of the other checkpoints is in `results_v2/corrected_robustness/sweeps/*/DONE` (a run is complete only if that file exists);
`python cnn_mlp/src/study/aggregate_seeds.py --summary OUT.md <sweep dirs>` prints the headline row of each.

## 7. Verified / not verified

Verified: `test_corruptions.py` 64 passed (synthetic + real batch of shot 13222; mutation check with the old axis fails 16 tests);
Module 1 v2 integration ran end to end on 60 test shots (867 scored windows per pass, 0 windows without truth, 99 without an EFIT LCFS,
0 boundary failures on the clean pass); clean-gate PASS for `mlp_fa_v3` (NRMSE identical to 6 digits); `build_ablation_table.py` and
`aggregate_seeds.py` ran on real sweep files (`results_v2/corrected_robustness/ablation_partial.*`, `summary_partial.md`); the Kaggle template
notebook and `build_lstm_robust_kernel.py` were re-synced (adds `efit_truth.py`, `aggregate_seeds.py` cells) and the kernel notebook rebuilt.

Not verified: sweeps of `lstm_fa` and of the new runs (`kaggle_mlp_plain_s1`, `kaggle_mlp_mask_s1`, `modal_cnn_lstm_*`, `kaggle_transformer_plain_s1`)
were still running/queued when this was written (check for `sweeps/<tag>/DONE`); the transformer per-sensor health mask is only unit-tested
against a stub model (no mask-trained transformer checkpoint exists); the Kaggle/Modal run of the new `pipeline_robustness.py` has not been
executed (only CPU); multi-seed statistics (std over seeds) cannot exist before more seeds are trained; the table's mitigation rows and
latency column are `--` until a mitigation JSON exists; bootstrap CIs are over shots within ONE checkpoint and do not include training-seed variance.
