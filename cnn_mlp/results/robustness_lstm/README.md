# TokaMark task_1-3 CNN+LSTM baseline: training + sensor-robustness results

- `training/` - baseline training run (best_model.pt, final.log, loss_curve.png, prediction.png, task_metrics.csv, windows_metrics.csv, shots_random.csv, history/config).
- `robustness/` - frozen best_model.pt under 6 sensor-failure scenarios x 3 imputations (zero, mean, ffill), 63 passes on the 66 test shots, 23908 windows.
  - `robustness.md` full report, `robust_results.csv` / `robust_metrics.json` raw numbers, `robust_manifest.json` the sweep, PNG figures, `robustness.log`.

Clean gate: robustness-harness clean NRMSE = 0.1969 vs training `task_metrics.csv` NRMSE = 0.1969 -> PASS.

Worst-case NRMSE per scenario (zero / mean / ffill):

| scenario | zero | mean | ffill |
|---|---|---|---|
| random_dropout | 0.698 | 0.614 | 0.290 |
| channel_ablation | 0.520 | 0.477 | 0.477 |
| category_abl | 0.344 | 0.384 | 0.384 |
| temporal_gap | 0.626 | 0.496 | 0.248 |
| group_failure | 0.445 | 0.493 | 0.493 |
| disruption_proximate | 0.531 | 0.510 | 0.198 |

A second sweep via the notebook (tokamark_robustness_kaggle.ipynb, kernel v3) ended in ERROR on Kaggle with an empty log and is not included.
