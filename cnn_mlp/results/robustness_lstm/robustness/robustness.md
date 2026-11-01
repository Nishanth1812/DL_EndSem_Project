# Robustness report - task_1-3, trained baseline

Frozen `best_model.pt` from `/kaggle/input/datasets/nishanthdevabathini/tokamark-lstm-baseline`, test shots of its split only (all from `configs/selection.json`). Sensor-failure scenarios after Gupta (arXiv 2607.11915); the model is never retrained on damaged data.

## Clean baseline

- NRMSE = **0.1969** (+-0.1153), NMAE = 0.1252, RMSE = 0.01081, MAE = 0.00688
- 66 test shots, 23908 windows
- clean gate vs the run's `task_metrics.csv`: **PASS**

## Worst case per scenario

RS = NRMSE_clean / NRMSE_corrupted: 1.0 = no damage, lower = worse.

| scenario | worst NRMSE (zero) | RS (zero) | worst NRMSE (mean) | RS (mean) | worst NRMSE (ffill) | RS (ffill) |
|---|---|---|---|---|---|---|
| random_dropout | 0.6980 | 0.282 | 0.6139 | 0.321 | 0.2899 | 0.679 |
| channel_ablation | 0.5201 | 0.378 | 0.4769 | 0.413 | 0.4769 | 0.413 |
| category_abl | 0.3444 | 0.572 | 0.3838 | 0.513 | 0.3838 | 0.513 |
| temporal_gap | 0.6256 | 0.315 | 0.4957 | 0.397 | 0.2475 | 0.795 |
| group_failure | 0.4449 | 0.442 | 0.4925 | 0.400 | 0.4925 | 0.400 |
| disruption_proximate | 0.5308 | 0.371 | 0.5103 | 0.386 | 0.1983 | 0.992 |

## Full sweep

| pass | dead signals | NRMSE | change | RS | shots |
|---|---|---|---|---|---|
| random_dropout|0.1|zero | - | 0.5311 | +169.8% | 0.371 | 66 |
| random_dropout|0.1|mean | - | 0.4001 | +103.2% | 0.492 | 66 |
| random_dropout|0.1|ffill | - | 0.2234 | +13.5% | 0.881 | 66 |
| random_dropout|0.25|zero | - | 0.6344 | +222.3% | 0.310 | 66 |
| random_dropout|0.25|mean | - | 0.5300 | +169.2% | 0.371 | 66 |
| random_dropout|0.25|ffill | - | 0.2530 | +28.5% | 0.778 | 66 |
| random_dropout|0.5|zero | - | 0.6980 | +254.6% | 0.282 | 66 |
| random_dropout|0.5|mean | - | 0.6139 | +211.9% | 0.321 | 66 |
| random_dropout|0.5|ffill | - | 0.2899 | +47.3% | 0.679 | 66 |
| channel_ablation|1|zero | flux_loop | 0.2241 | +13.9% | 0.878 | 66 |
| channel_ablation|1|mean | flux_loop | 0.2778 | +41.1% | 0.709 | 66 |
| channel_ablation|1|ffill | flux_loop | 0.2778 | +41.1% | 0.709 | 66 |
| channel_ablation|3|zero | obr saddle_v coil_i | 0.4057 | +106.1% | 0.485 | 66 |
| channel_ablation|3|mean | obr saddle_v coil_i | 0.4445 | +125.8% | 0.443 | 66 |
| channel_ablation|3|ffill | obr saddle_v coil_i | 0.4445 | +125.8% | 0.443 | 66 |
| channel_ablation|6|zero | flux_loop ccbv obv saddle_v solenoid_i ip | 0.5201 | +164.2% | 0.378 | 66 |
| channel_ablation|6|mean | flux_loop ccbv obv saddle_v solenoid_i ip | 0.4769 | +142.3% | 0.413 | 66 |
| channel_ablation|6|ffill | flux_loop ccbv obv saddle_v solenoid_i ip | 0.4769 | +142.3% | 0.413 | 66 |
| category_abl|flux_loop|zero | flux_loop | 0.2241 | +13.9% | 0.878 | 66 |
| category_abl|flux_loop|mean | flux_loop | 0.2778 | +41.1% | 0.709 | 66 |
| category_abl|flux_loop|ffill | flux_loop | 0.2778 | +41.1% | 0.709 | 66 |
| category_abl|pickup_fields|zero | ccbv obr obv | 0.3040 | +54.4% | 0.648 | 66 |
| category_abl|pickup_fields|mean | ccbv obr obv | 0.3838 | +95.0% | 0.513 | 66 |
| category_abl|pickup_fields|ffill | ccbv obr obv | 0.3838 | +95.0% | 0.513 | 66 |
| category_abl|saddle_voltage|zero | saddle_v | 0.2041 | +3.7% | 0.965 | 66 |
| category_abl|saddle_voltage|mean | saddle_v | 0.2058 | +4.5% | 0.957 | 66 |
| category_abl|saddle_voltage|ffill | saddle_v | 0.2058 | +4.5% | 0.957 | 66 |
| category_abl|active_coils|zero | coil_i | 0.2774 | +40.9% | 0.710 | 66 |
| category_abl|active_coils|mean | coil_i | 0.2728 | +38.6% | 0.722 | 66 |
| category_abl|active_coils|ffill | coil_i | 0.2728 | +38.6% | 0.722 | 66 |
| category_abl|solenoid|zero | solenoid_i | 0.2943 | +49.5% | 0.669 | 66 |
| category_abl|solenoid|mean | solenoid_i | 0.2865 | +45.5% | 0.687 | 66 |
| category_abl|solenoid|ffill | solenoid_i | 0.2865 | +45.5% | 0.687 | 66 |
| category_abl|plasma_current|zero | ip | 0.3444 | +75.0% | 0.572 | 66 |
| category_abl|plasma_current|mean | ip | 0.2106 | +7.0% | 0.935 | 66 |
| category_abl|plasma_current|ffill | ip | 0.2106 | +7.0% | 0.935 | 66 |
| temporal_gap|0.2|zero | - | 0.5479 | +178.3% | 0.359 | 66 |
| temporal_gap|0.2|mean | - | 0.4180 | +112.4% | 0.471 | 66 |
| temporal_gap|0.2|ffill | - | 0.2013 | +2.3% | 0.978 | 66 |
| temporal_gap|0.4|zero | - | 0.6256 | +217.8% | 0.315 | 66 |
| temporal_gap|0.4|mean | - | 0.4569 | +132.1% | 0.431 | 66 |
| temporal_gap|0.4|ffill | - | 0.2220 | +12.8% | 0.887 | 66 |
| temporal_gap|0.6|zero | - | 0.5301 | +169.3% | 0.371 | 66 |
| temporal_gap|0.6|mean | - | 0.4957 | +151.8% | 0.397 | 66 |
| temporal_gap|0.6|ffill | - | 0.2475 | +25.7% | 0.795 | 66 |
| group_failure|magnetic_probes|zero | ccbv obr obv | 0.3040 | +54.4% | 0.648 | 66 |
| group_failure|magnetic_probes|mean | ccbv obr obv | 0.3838 | +95.0% | 0.513 | 66 |
| group_failure|magnetic_probes|ffill | ccbv obr obv | 0.3838 | +95.0% | 0.513 | 66 |
| group_failure|power_supply|zero | coil_i solenoid_i | 0.4244 | +115.6% | 0.464 | 66 |
| group_failure|power_supply|mean | coil_i solenoid_i | 0.4563 | +131.8% | 0.431 | 66 |
| group_failure|power_supply|ffill | coil_i solenoid_i | 0.4563 | +131.8% | 0.431 | 66 |
| group_failure|magnetics_all|zero | flux_loop ccbv obr obv saddle_v | 0.4449 | +126.0% | 0.442 | 66 |
| group_failure|magnetics_all|mean | flux_loop ccbv obr obv saddle_v | 0.4925 | +150.2% | 0.400 | 66 |
| group_failure|magnetics_all|ffill | flux_loop ccbv obr obv saddle_v | 0.4925 | +150.2% | 0.400 | 66 |
| disruption_proximate|0.1|zero | - | 0.3820 | +94.0% | 0.515 | 66 |
| disruption_proximate|0.1|mean | - | 0.2312 | +17.4% | 0.852 | 66 |
| disruption_proximate|0.1|ffill | - | 0.1969 | +0.0% | 1.000 | 66 |
| disruption_proximate|0.25|zero | - | 0.4326 | +119.7% | 0.455 | 66 |
| disruption_proximate|0.25|mean | - | 0.3599 | +82.8% | 0.547 | 66 |
| disruption_proximate|0.25|ffill | - | 0.1972 | +0.2% | 0.998 | 66 |
| disruption_proximate|0.5|zero | - | 0.5308 | +169.6% | 0.371 | 66 |
| disruption_proximate|0.5|mean | - | 0.5103 | +159.2% | 0.386 | 66 |
| disruption_proximate|0.5|ffill | - | 0.1983 | +0.8% | 0.992 | 66 |

## Notes

- Subset of the data (`configs/selection.json`): absolute NRMSE is not the paper's full-set figure.
- Fill values are in the loader's standardised units: zero = a physical 0 reading, mean = the official train-split mean (no test information), ffill = last undamaged value in the window.
- random_dropout zeroes exactly that fraction of the present values of every signal; time gaps hit all sensors of a window at the same time; disruption_proximate puts the gap at the window end.
