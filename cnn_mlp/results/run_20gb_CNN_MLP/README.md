# Run: 20 GB subset (`results.zip`)

Kaggle run of `pipeline.py` on a ~20 GB budget-limited subset.

## Data
- selected / official: train 280/8963 (~3.1%), val 32/1115 (~2.9%), test 42/1110 (~3.8%)
- git revs: tokamark + tokamark_baseline@1e7c489

## Training
- stop_reason: early stopping (patience rule)
- steps: 3500 ; epochs: 97 ; best_val: 0.0250 ; duration: 146.5 min
- cuda_oom: 0 ; data_errors: 0

## Evaluation (official evaluator, un-standardised)
- equilibrium-psi: NRMSE 0.1395 (+/-0.2005), NMAE 0.0920, RMSE 0.00766, MAE 0.00505
- n_shots: 35

## Notes
- This run **generalised cleanly**: val loss 0.556 -> 0.0250, train/val gap ~ +/-0.01.
- `figures/loss_curves.png` present. `psi_reconstruction_samples.png` was NOT generated
  in this run (only the 40 GB run produced it).

## Contents
`out/random/{run_config.json, selection.json, summary.md, train_summary.json,
figures/, model/, results/task_1-3/}`, plus `out/tar_index.json`.

## Caveat
Subset of the official splits. Absolute numbers are NOT the official full-set
results. See docs/EXPLAINED.md.
