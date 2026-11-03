# Robustness report - task_1-3 (random split)

Sensor-failure study mirroring Gupta (arXiv 2607.11915), on the equilibrium
reconstruction task (Gupta used task_4-4). Model: frozen `best_model.pt`.
**Test split only. Model never retrained on corrupted data.**

## Clean baseline (sensor-healthy)
- NRMSE = **0.1419** (±0.1199)
- NMAE = 0.0708, RMSE = 0.00779, MAE = 0.00389
- n_shots = 46, n_windows = 3450

## Robustness Score (RS) — worst case per scenario, zero-fill
RS = NRMSE_clean / NRMSE_corrupted; 1.0 = no damage, lower = worse.

| scenario | worst NRMSE | RS |
|---|---|---|
| random_dropout | 1.9941 | 0.071 |
| channel_ablation | 0.2855 | 0.497 |
| category_abl | 0.2222 | 0.638 |
| temporal_gap | 0.5991 | 0.237 |
| group_failure | 0.2266 | 0.626 |
| disruption_proximate | 0.6097 | 0.233 |

## Full sweep

| task | NRMSE | NMAE | RS | n_shots |
|---|---|---|---|---|
| category_abl|active_coils|ffill | 0.1774 | 0.1038 | 0.800 | 46 |
| category_abl|active_coils|mean | 0.1774 | 0.1038 | 0.800 | 46 |
| category_abl|active_coils|zero | 0.1774 | 0.1038 | 0.800 | 46 |
| category_abl|flux_loop|ffill | 0.1651 | 0.0958 | 0.859 | 46 |
| category_abl|flux_loop|mean | 0.1651 | 0.0958 | 0.859 | 46 |
| category_abl|flux_loop|zero | 0.1651 | 0.0958 | 0.859 | 46 |
| category_abl|pickup_fields|ffill | 0.2214 | 0.1317 | 0.641 | 46 |
| category_abl|pickup_fields|mean | 0.2214 | 0.1317 | 0.641 | 46 |
| category_abl|pickup_fields|zero | 0.2214 | 0.1317 | 0.641 | 46 |
| category_abl|plasma_current|ffill | 0.1833 | 0.0987 | 0.774 | 46 |
| category_abl|plasma_current|mean | 0.1833 | 0.0987 | 0.774 | 46 |
| category_abl|plasma_current|zero | 0.1833 | 0.0987 | 0.774 | 46 |
| category_abl|saddle_voltage|ffill | 0.1441 | 0.0727 | 0.985 | 46 |
| category_abl|saddle_voltage|mean | 0.1441 | 0.0727 | 0.985 | 46 |
| category_abl|saddle_voltage|zero | 0.1441 | 0.0727 | 0.985 | 46 |
| category_abl|solenoid|ffill | 0.2222 | 0.1437 | 0.638 | 46 |
| category_abl|solenoid|mean | 0.2222 | 0.1437 | 0.638 | 46 |
| category_abl|solenoid|zero | 0.2222 | 0.1437 | 0.638 | 46 |
| channel_ablation|1|ffill | 0.1740 | 0.1000 | 0.815 | 46 |
| channel_ablation|1|mean | 0.1692 | 0.0980 | 0.839 | 46 |
| channel_ablation|1|zero | 0.1744 | 0.0951 | 0.814 | 46 |
| channel_ablation|3|ffill | 0.2256 | 0.1409 | 0.629 | 46 |
| channel_ablation|3|mean | 0.2232 | 0.1313 | 0.636 | 46 |
| channel_ablation|3|zero | 0.2264 | 0.1295 | 0.627 | 46 |
| channel_ablation|6|ffill | 0.2649 | 0.1681 | 0.536 | 46 |
| channel_ablation|6|mean | 0.2971 | 0.1885 | 0.477 | 46 |
| channel_ablation|6|zero | 0.2855 | 0.1793 | 0.497 | 46 |
| disruption_proximate|0.1|ffill | 0.2033 | 0.1194 | 0.698 | 46 |
| disruption_proximate|0.1|mean | 0.2103 | 0.1233 | 0.674 | 46 |
| disruption_proximate|0.1|zero | 0.2103 | 0.1233 | 0.674 | 46 |
| disruption_proximate|0.25|ffill | 0.3355 | 0.2275 | 0.423 | 46 |
| disruption_proximate|0.25|mean | 0.2417 | 0.1508 | 0.587 | 46 |
| disruption_proximate|0.25|zero | 0.2417 | 0.1508 | 0.587 | 46 |
| disruption_proximate|0.5|ffill | 0.3724 | 0.2465 | 0.381 | 46 |
| disruption_proximate|0.5|mean | 0.6097 | 0.4279 | 0.233 | 46 |
| disruption_proximate|0.5|zero | 0.6097 | 0.4279 | 0.233 | 46 |
| group_failure|magnetic_probes|ffill | 0.2214 | 0.1317 | 0.641 | 46 |
| group_failure|magnetic_probes|mean | 0.2214 | 0.1317 | 0.641 | 46 |
| group_failure|magnetic_probes|zero | 0.2214 | 0.1317 | 0.641 | 46 |
| group_failure|magnetics_all|ffill | 0.1930 | 0.1164 | 0.735 | 46 |
| group_failure|magnetics_all|mean | 0.1930 | 0.1164 | 0.735 | 46 |
| group_failure|magnetics_all|zero | 0.1930 | 0.1164 | 0.735 | 46 |
| group_failure|power_supply|ffill | 0.2266 | 0.1473 | 0.626 | 46 |
| group_failure|power_supply|mean | 0.2266 | 0.1473 | 0.626 | 46 |
| group_failure|power_supply|zero | 0.2266 | 0.1473 | 0.626 | 46 |
| random_dropout|0.1|ffill | 0.9532 | 0.5582 | 0.149 | 46 |
| random_dropout|0.1|mean | 0.6844 | 0.3677 | 0.207 | 46 |
| random_dropout|0.1|zero | 0.6724 | 0.3610 | 0.211 | 46 |
| random_dropout|0.25|ffill | 1.3095 | 0.8615 | 0.108 | 46 |
| random_dropout|0.25|mean | 1.3710 | 0.8080 | 0.103 | 46 |
| random_dropout|0.25|zero | 1.3489 | 0.7908 | 0.105 | 46 |
| random_dropout|0.5|ffill | 1.8980 | 1.2806 | 0.075 | 46 |
| random_dropout|0.5|mean | 2.0407 | 1.2940 | 0.070 | 46 |
| random_dropout|0.5|zero | 1.9941 | 1.2837 | 0.071 | 46 |
| temporal_gap|0.2|ffill | 0.3654 | 0.2394 | 0.388 | 46 |
| temporal_gap|0.2|mean | 0.4077 | 0.2752 | 0.348 | 46 |
| temporal_gap|0.2|zero | 0.4637 | 0.3112 | 0.306 | 46 |
| temporal_gap|0.4|ffill | 0.4038 | 0.2690 | 0.351 | 46 |
| temporal_gap|0.4|mean | 0.4970 | 0.3384 | 0.285 | 46 |
| temporal_gap|0.4|zero | 0.5627 | 0.3806 | 0.252 | 46 |
| temporal_gap|0.6|ffill | 0.6596 | 0.4645 | 0.215 | 46 |
| temporal_gap|0.6|mean | 0.5600 | 0.4059 | 0.253 | 46 |
| temporal_gap|0.6|zero | 0.5991 | 0.4237 | 0.237 | 46 |

## Caveats

- Subset data: absolute NRMSE is NOT the official full-set figure.
- mean-imputation uses TRAIN-split channel means (no test leakage).
- Corruption is inference-time only; a model retrained on damaged data
  could behave differently.
