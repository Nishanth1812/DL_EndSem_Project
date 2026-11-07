# Module 3 + 4 on real checkpoints (study v2)

Code: `physics_study.py` (windows, frozen-network wrapper, EFIT scoring, Module 4 study, latency), `sensor_criticality.py` (Module 3 on
`sensor_attribution.py` + `greens_sensitivity.py`), `run_module34_local.py` (driver), `test_module34_study.py` (9 assert tests, ~20 s).
Outputs: `cnn_mlp/results_v2/module34/<tag>/` (`mlp_fa_v3` = failure-aware CNN+MLP, `runs/mlp_fa_robust_v3`; `lstm_fa` = failure-aware
CNN+LSTM, `runs/lstm_fa_robust`). Both checkpoints are "pinn legacy" (trained before TM_PINN_MODE=measured); evaluation is unaffected.
All numbers below are in those files; nothing was trained; everything ran on CPU (torch 2.2 CPU, 22-core laptop).

## What was run
* Data: random split, shots on disk from the runs' `selection.json` (train / val / test are disjoint at every step). Window cache =
  every window (stride 1 val/test, stride 2 train, <= 60 per shot) of the official TokaMark dataset; windows without an EFIT LCFS are
  dropped (train 90 / val 91 / test 155 of the cached windows, counted in `mitigation.json: dropped_no_efit_lcfs`).
  Train 1059 windows / 32 shots; val 1013 / 21 shots; test 1110 / 24 shots (of which 300, or 200 for the LSTM, are used).
* Scoring: Module 1 v2 EFIT mode only (`evaluate_window_physics(..., truth=TruthIndex.get(shot, window))`), wall alarm = PROXY wall
  (EFIT grid box, no vessel data in the zarr): in the data every EFIT clearance is 13.5-25 cm, so there are NO true wall alarms; the alarm
  numbers are a protocol demonstration only.
* Clean accuracy of the checkpoint (this protocol): boundary 3.9-4.4 cm (mean, test), Hausdorff about 20 cm, NRMSE 0.088. The 2 cm design
  tolerance is therefore infeasible even with all 115 sensors; Module 3 reports sets for tolerances relative to the clean error.

## Module 3 (mlp_fa_v3; `criticality.json`, `criticality_report.md`, `figures/*.png`, `spatial_maps.npz`)
value_fn(mask) = mean EFIT boundary error (cm) on 80 val windows (21 shots) with the lost sensors zero-filled; confirmation on 80 disjoint
test windows (24 shots). X-point error is cached in the same evaluations. 2556 + 271 unique model evaluations (1.2 h CPU).
* Failure-aware CNN: the health input exists per BRANCH only. Per-sensor loss is therefore an UNMASKED zero for the network; health = 0
  only when a whole branch is gone (group table). A model with per-sensor masks (the Transformer variant) was not evaluated here (no
  checkpoint locally); its numbers will differ (a masked sensor is dropped, not seen as zero).
* Groups, exact Shapley over 256 coalitions (cm added; efficiency check sum = f(all) - f(none) = 28.5 cm holds): flux_loop 9.8,
  solenoid 6.9, ccbv 6.2, obv 1.7, coil_i 1.5, obr 1.4, saddle_v 1.2, ip -0.1. X-point (cm): flux_loop 19.0, solenoid 17.3, ccbv 12.6.
  Val vs test group-Shapley Spearman 0.83. Drop-one with the network told (cm): ccbv 1.18, flux_loop 0.75, ip 0.68, solenoid 0.49, others ~0.
  Pair interactions: `group.interaction` (exact, 8x8). Group worst-k exhaustive: k=1 ccbv 5.4; k=2 flux_loop+solenoid 10.4 (greedy picks
  flux_loop+ccbv = 8.9, suboptimal); k=3 flux_loop+ccbv+solenoid 23.5. The network is robust to losing a whole ANNOUNCED branch
  (core fact behind the "+ failure-aware" row) and fragile to silent zeros.
* Sensors (115): drop-one for all 115; sampled Shapley (antithetic permutations, K<=20 simultaneous failures, **only 18 pairs / 800
  evaluations** because of the CPU budget, 95 % CIs in `sensor.shapley_ci95`, mostly +-3-7 cm, so only the top ~6 are distinguishable
  from zero). Classes: critical 26, backup 33, redundant 24, minor 32 (thresholds = 50 % / 5 % of the headroom). A single silently dead
  flux loop (CC09) costs +14 cm (18.1 vs 4.3); AMB_OBV06, AMC_P2IL, AMB_CCBV40, AMB_OBR01, AMB_FL/CC05, AMC_P4L follow.
  Pairwise sensor interactions were not sampled (`pairs=0`).
* Worst-k sensors (k=1..5, greedy vs beam-2; test confirmation in `worst_k_beam[k].value_test`): 18.1/25.3/29.8/34.0/40.8 (greedy),
  beam 18.1/25.3/29.8/35.5/41.5; random k sensors 4.4/4.8/5.5/7.0/8.6. k=6..8 were not run.
* Minimum sensor set (lazy greedy backward = approximate CELF greedy, Shapley-ranked, drop-one-ranked, Green's-ranked, exhaustive on groups):
  abs 2 cm: infeasible. clean x 1.25 (5.33 cm): lazy greedy 55 sensors (val 5.29, test 6.08 cm), Shapley-ranked 93, drop-one-ranked 110,
  Green's-ranked 102, groups: 4 of 8. clean x 1.5 (6.40): 50 sensors (test 7.13). clean + 1 cm (5.27): 56 (test 5.65). Kept per group for
  the 55-set: flux_loop 8, ccbv 19, obr 9, obv 8, saddle 2, coil 7, solenoid 1, ip 1. 80 windows: the selection is noisy (test error is
  higher than val for the greedy sets), treat "about half the sensors" as the finding, not "55".
* Green's-function comparison (vacuum linear-Gaussian model; saddle loops unmodelled; modelling choices are parameters): sensor level
  Spearman(data Shapley, Green's leave-one-out) 0.08 (p 0.39), Kendall 0.06; drop-one vs LOO -0.03; per group no significant agreement
  (obv 0.39, p 0.11). Group level (n=8) Spearman 0.52 (p 0.20). Conclusion: no evidence that the vacuum geometry predicts which sensors
  the trained network depends on; the data-driven ranking mostly reflects how the network uses zero inputs.
* Spatial maps (`figures/spatial_error_vs_sensor_location.png`, stats in `spatial.cases[*].stats_*`): effect of losing ccbv / flux loops
  / single sensors does not concentrate at the lost sensor (Spearman(error, distance to sensor) +0.47 ccbv, -0.34 flux loops, |rho| < 0.3 for
  single sensors; error-centroid-to-sensor 0.2-1.8 m): the error appears around the divertor regions (R 1.0-1.6 m, Z about +-1 m) and the core, wherever the lost sensor is,
  and correlate with the sensor's response map only moderately (0.3-0.5 for CC09, P2IL, OBV06).

## Module 4 (`mitigation.json`, `mitigation_table.json`, `mitigation_table.md`, `module4_boundary_error.png`)
Fit on 32 train shots (detector statistics, virtual sensors, physics fallback); val shots split by shot into A1 (detector thresholds),
A2 (operating point + best-fixed-imputation choice), B (conformal calibration); report on test shots. Operating point searched on A2
(16 settings: threshold scale, forward-fill horizon, fallback crossover, critical policy): mlp chose threshold x4, forward-fill <= 2.5 ms,
fallback beyond 5 masked branches, critical policy "virtual"; mean boundary over scenarios on A2: 7.2 cm vs 12.2 cm with the default 1 ms forward-fill horizon (the gap scenarios dominate).
Tiers come from Module 3 (`tiers_from_criticality`: group drop-one >= 1.0 cm tier 1, >= 0.25 cm tier 2): ccbv 1; flux_loop, solenoid, ip 2;
rest 3. Scenarios (definitions in `make_faults`): clean, channel loss (3 branches), channel loss 6 (6 branches, the table column),
group loss (magnetics_all), temporal gap 40 %, pre-disruption (last 25 %), 6 dead probes, 3 stuck, 3 drift, 3 spikes. 300 test windows
(24 shots); 95 % CIs are shot-level bootstraps (1000).
Boundary error cm (NRMSE), mlp_fa_v3: no mitigation / best fixed imputation / adaptive / Oracle
| scenario | none | best imputation | adaptive | Oracle |
|---|---|---|---|---|
| clean | 3.94 (0.088) | 3.94 (0.088) | 4.29 (0.099) | 3.94 (0.088) |
| channel loss 6 | 30.2 (0.539) | 19.2 (0.522) | 11.9 (0.391) | 10.9 (0.310) |
| group loss | 20.7 (0.298) | 20.7 (0.298) | 11.6 (0.183) | 11.6 (0.183) |
| temporal gap 40 % | 21.1 (0.474) | 11.6 (0.245) | 9.3 (0.213) | 10.4 (0.208) |
| pre-disruption | 21.4 (0.310) | 5.8 (0.123) | 7.7 (0.179) | 5.8 (0.123) |
| stuck / drift / spike | 27.4 / 9.5 / 35.1 | 6.5 / 4.4 / 7.5 | 13.5 / 5.3 / 5.8 | 5.8 / 4.5 / 7.5 |
Full table with CIs and the `known-health mask only` row: `mitigation_table.md`. LSTM: `lstm_fa/` (200 windows; clean 5.26 none / 5.09 adaptive;
channel loss 6 22.9 -> 11.9; group loss 18.7 -> 10.2; same ordering).
* Rows "+ adaptive mitigation" / "Oracle mitigation" for `build_ablation_table.py --mitigation` are `mitigation_table.json` (schema of
  HARNESS_V2.md section 4, verified to load). Caveat: both rows use the FAILURE-AWARE network as backbone and this study's own windows
  (all EFIT-valid windows of 24 shots, not the harness hash-selected phys_stride sample), one seed.
* Adaptive loses 0.35 cm on clean windows: the real data contain probes that are zero for whole shots (dead probes; 64 % of clean test windows
  have >= 1, 4.5 channels per window on average); the detector correctly flags them and repairs them although the network was trained on them
  as zeros. Detection on injected faults (flags new relative to the un-injected window, `detection.<scen>.adaptive`): recall 0.90-1.00,
  precision 0.96-1.00, isolation top-1 0.99-1.0 (stuck/drift recall 0.90). Stuck rails are still handled worse than imputation with known
  holes (adaptive 13.5 vs Oracle 5.8 cm): detection is fine, the routing (module2 branch masking in 197 of 300 windows) is the weak part.
* Routes used (`routes`): mostly `module2` (branch masks), `transient_ffill` for gaps/spikes, `virtual_sensor` for single stuck/drift probes,
  `physics_fallback` for channel-loss-6 only. The linear fallback beats the masked network beyond 5 masked branches (val A2 mean boundary: crossover 5 = 7.2 cm vs 8 = 8.3 cm).
* Conformal (alpha 0.1, calibrated on val_B on the adaptive pipeline's errors, 3000 windows, scenario mixture): empirical coverage on the
  disjoint test shots 0.94-0.95 (Hausdorff gauge: route-global 0.95, marginal 0.94, adaptive 0.94; mean-error gauge 0.94 / 0.93);
  conservative, per scenario 0.81-1.0 (marginal guarantee only). The intervals are WIDE: Hausdorff delta 90-123 cm, mean-error delta 20-30 cm,
  so with the 5 cm margin every window is ABSTAIN (0 HIGH, 0 LOW in 3000) and the conservative alarm fires on every window (100 % "false
  alarms" against proxy ground truth; the point-estimate alarm: 295 false alarms in 3000, 0 true alarms exist). Module 4's claim of cutting
  false alarms is NOT supported by this model: its boundary error (4 cm clean, 10+ cm under failures) is larger than the 5 cm margin.
  The adaptive gauge (difficulty features) did not narrow intervals measurably here.
* Latency (quiet machine, CPU, batch 1, p50 / p95 / p99, ms; `latency`, `latency_by_threads`): total 16.5 / 237 / 261 (mlp) and 15.7 / 339 / 416
  (lstm); detector 2.2 (p95 217: the peeling isolation on windows with ~100 faulty channels), router 0.5, mitigation 0.6, network forward
  10.9 (includes numpy->torch conversion), gauge 0.5. This exceeds a 5 ms control cycle on CPU; the GPU figure must come from Kaggle.

## Transformer, failure-aware (mask), seed 1 (`transformer_mask_s1`)

Checkpoint `results_v2/kaggle_transformer_mask_s1/out/random/transformer_mask_s1` (Transformer, per-sensor mask training, one seed). Per-sensor
health is a (115,) vector: a masked sensor is dropped from attention. Analysis sizes: 80 val windows (21 shots) for the Shapley games, 80 disjoint
test windows (24 shots) for confirmation, 300 test windows for Module 4. What was reduced against the plan in `HARNESS_V2.md`: sensor Shapley
21 antithetic pairs (914 unique evaluations, truncation K=20; only 3 sensors are statistically distinguishable at the top), pair
interactions skipped (`pairs=0`), worst-k to k=5 with beam width 2, minimum-set by an APPROXIMATE lazy greedy (CELF-style, 8 candidates
re-evaluated per round, at most 24 re-evaluations per step; the run was resumed from the cache after a restart, so the early steps used an
uncapped search), 80 windows rather than the hundreds a final table needs. Not announced losses (silent) use the same masks with the health input
left at ones. Green's model is vacuum-only (saddle loops unmodelled, no plasma response), so its agreement numbers say how far a vacuum sensitivity
explains the learned network, nothing more. One seed: differences below about 1 cm between sensors or models are not established.
Module 4 uses sensor-level repair (`Mitigators._prepare_sensor_level`): short transients forward-filled, other flagged channels masked per sensor
(not per branch), physics fallback when more than 5 branches are fully masked. The Oracle rows use the injected-fault labels plus the fitted
set of whole-shot dead probes; in `channel_loss_6` the labels miss pre-existing dead channels inside a lost branch, so the Oracle falls back
to the linear model in only 128 of 300 windows and is NOT an upper bound there (adaptive 9.27 vs Oracle 19.14 cm); in `pre_disruption` the Oracle
equals the forward-fill imputation (8.46 cm) because the labelled gap (25 % of 5 ms) is a short fault. The first Module 4 run of this checkpoint
(before the dead-probe fix, Oracle `channel_loss_6` 27.7 cm) was discarded.

### Module 3 (`transformer_mask_s1/criticality.json`, `criticality_report.md`, `figures/`)
Clean boundary error 4.76 cm (val, 80 windows / 21 shots), 4.60 cm (test, 80 windows / 24 shots). Primary tolerance `clean_x1.25` = 5.95 cm. Evaluations: 3762 val + 381 test (unique).

Group Shapley (cm of boundary error, exact over 256 coalitions) and whole-group loss (drop-one):

| group | #sensors | Shapley | Shapley (test) | drop-one | alone |
|---|---|---|---|---|---|
| flux_loop | 15 | 15.62 | 16.39 | 2.34 | 17.52 |
| ccbv | 40 | 12.00 | 11.65 | 12.84 | 27.90 |
| obr | 18 | 2.59 | 3.34 | 3.67 | 36.36 |
| obv | 18 | 3.37 | 2.64 | 1.98 | 30.79 |
| saddle_v | 12 | 1.27 | -0.01 | 1.15 | 50.63 |
| coil_i | 10 | 2.98 | 3.12 | 0.79 | 37.95 |
| solenoid_i | 1 | 6.70 | 8.33 | 0.91 | 32.55 |
| ip | 1 | -0.58 | 1.39 | 0.44 | 52.88 |

Efficiency check: sum Shapley 43.96 = f(all lost) - f(none lost) 43.96; val/test Shapley Spearman 0.88.

Worst k groups lost (greedy / exhaustive, cm): k=1: 17.6; k=2: 29.0; k=3: 40.0; k=4: 43.4; k=5: 47.8.

Sensors: sampled antithetic Shapley, 21 pairs, truncation K=20, stop = budget; classes {'backup': 35, 'critical': 6, 'minor': 40, 'redundant': 34}; top-5 by Shapley: AMB_FL/CC07 (1.01), AMB_FL/P5U/1 (0.95), AMB_CCBV22 (0.70), AMB_FL/CC09 (0.65), AMB_FL/CC03 (0.65). Distinguishable top: 3.

Worst k single sensors lost (cm, val): k=1: greedy 5.87 / beam 5.87 / random mean 4.88; k=2: greedy 7.85 / beam 7.85 / random mean 4.93; k=3: greedy 10.47 / beam 10.47 / random mean 4.93; k=4: greedy 13.87 / beam 13.87 / random mean 5.17; k=5: greedy 17.29 / beam 17.29 / random mean 5.24.

Minimum sensor set that keeps the error within tolerance (kept sensors of 115; confirmation on disjoint test shots):

| tolerance | lazy greedy (val / test cm) | Shapley-ranked | drop-one-ranked | Green's-ranked | groups (exhaustive best) |
|---|---|---|---|---|---|
| abs_2cm (2.00) | infeasible | infeasible | infeasible | infeasible | infeasible |
| clean_x1.25 (5.95) | 40 (5.94 / 7.90) | 103 (5.91 / 6.68) | 110 (5.72 / 5.70) | 103 (5.91) | 7 groups (5.20) |
| clean_x1.5 (7.14) | 21 (7.10 / 9.41) | 101 (6.96 / 7.56) | 108 (7.03 / 6.72) | 102 (6.77) | 5 groups (6.53) |
| clean_plus_1cm (5.76) | 41 (5.67 / 7.62) | 106 (5.63 / 5.85) | 110 (5.72 / 5.70) | 104 (5.74) | 7 groups (5.20) |

Green's comparison (vacuum linear-Gaussian model; saddle loops unmodelled; Ip as a Rogowski sum): rank agreement with the data-driven sensor ranking, all modelled sensors: Shapley vs leave-one-out information loss Spearman -0.07 (Kendall -0.06, n=103); drop-one vs LOO -0.03; Shapley vs Green's Shapley 0.02; group level Shapley 0.64 (p=0.10), group drop-one 0.48.

#### Silent vs announced loss (Transformer, per-sensor mask)
Silent = the lost sensors read as zeros and nothing is announced (health input all ones). Announced = per-sensor mask, the lost sensors are dropped from attention. Same 80 val windows, clean 4.76 / 4.76 cm.

| group | whole-group loss silent (cm) | announced (cm) | mean single-sensor loss silent | announced |
|---|---|---|---|---|
| flux_loop | 26.27 | 2.34 | 0.75 | 0.32 |
| ccbv | 8.51 | 12.84 | 0.07 | 0.06 |
| obr | 0.58 | 3.67 | 0.02 | -0.02 |
| obv | 4.10 | 1.98 | 0.03 | -0.00 |
| saddle_v | 0.19 | 1.15 | -0.01 | 0.03 |
| coil_i | 5.03 | 0.79 | 0.23 | 0.12 |
| solenoid_i | 7.20 | 0.91 | 7.20 | 0.91 |
| ip | 0.49 | 0.44 | 0.49 | 0.44 |

Group Shapley silent: flux_loop 20.0, ccbv 2.7, obr -0.7, obv 1.0, saddle_v -0.2, coil_i 2.5, solenoid_i 3.0, ip 0.4; announced: flux_loop 15.6, ccbv 12.0, obr 2.6, obv 3.4, saddle_v 1.3, coil_i 3.0, solenoid_i 6.7, ip -0.6.
Single sensors whose loss adds >= 1 cm: silent 7 vs announced 1; >= 2 cm: 1 vs 0. Spearman of the per-sensor drop-one, silent vs announced: 0.60. Worst silent single sensors: AMC_SOL CURRENT 7.2, AMB_FL/CC03 1.9, AMB_FL/CC09 1.8, AMB_FL/CC05 1.5, AMB_FL/P3L/4 1.5; worst announced: AMB_FL/CC03 1.11, AMB_FL/CC05 0.92, AMC_SOL CURRENT 0.91, AMB_FL/CC07 0.76, AMB_FL/CC09 0.72 (cm).

### Module 4 (`transformer_mask_s1/mitigation.json`, `mitigation_table.md/.json`, `module4_boundary_error.png`)
300 test windows / 24 shots; operating point chosen on val_A2: `{'thr_scale': 4.0, 'short_ms': 2.5, 'max_masked_branches': 5, 'critical_policy': 'mask', 'long_fault': 'router'}`. Mean boundary error cm (test; 95% shot-bootstrap CIs in `mitigation_table.md`):

| scenario | none | best imputation | adaptive | Oracle |
|---|---|---|---|---|
| clean | 4.19 | 4.19 | 5.00 | 4.19 |
| channel_loss | 16.77 | 13.71 | 15.16 | 14.98 |
| channel_loss_6 | 25.29 | 21.07 | 9.27 | 19.14 |
| group_loss | 26.20 | 20.90 | 22.12 | 22.75 |
| temporal_gap | 21.88 | 8.80 | 6.88 | 6.85 |
| pre_disruption | 17.54 | 8.46 | 5.76 | 8.46 |
| sensor_loss | 5.72 | 4.99 | 5.54 | 4.95 |
| stuck | 4.83 | 4.28 | 5.31 | 4.61 |
| drift | 4.29 | 4.18 | 5.42 | 4.65 |
| spike | 4.78 | 4.24 | 5.07 | 4.20 |

Conformal (alpha 0.1): empirical coverage on test shots 0.91 (marginal), 0.95 (adaptive); Hausdorff interval half-width 88 cm; ABSTAIN in 3000 of 3000 window-scenarios (the interval is wider than the 5 cm margin).
Latency (CPU, batch 1, measured while other jobs were running, p50 / p95 ms): total 67.7 / 126.7; network 56.4; detector 4.7.


## Verified vs not
Verified: sensor-axis masking on a real collated window cache (`test_sensor_axis_masking_real_batch`, `M34_CACHE=<cache dir>`), exact Shapley
efficiency, shot-disjoint calibration/test, schema load by `build_ablation_table.py`, batched variants equal the brain's per-window path
in structure (same prepare/decide/apply code; the brain's `process` is used for latency only).
Not verified / limits: sensor Shapley precision (18 pairs), 80 analysis windows, one seed per checkpoint, no Transformer checkpoint, no
Oracle for the physics fallback beyond the router's labels, the 35 % heuristic contour inside the brain's gauge (latency only), a "real"
wall (proxy only), tiers rule cuts (1.0 / 0.25 cm) and the 0.5 / 1.0 flag fractions are policy parameters, the LSTM got the MLP's tiers,
`kernels/*` and `scripts/build_lstm_robust_kernel.py` still ship the OLD physics_study.py CLI (`criticality`/`mitigation` subcommands were
removed; `sensor_criticality.TIER1_CM/tier_of` are kept) and must be rebuilt before Kaggle use.

## To re-run (needs GPU / Kaggle / Modal for the final multi-seed table)
Per seed/checkpoint: window cache once, then the stages (numpy<2 + zarr3 pylib as in `scratchpad\pylib5;pylib4`):
```
python run_module34_local.py --model mlp_fa_v3 --data-root DIR --work-dir SCRATCH --stage cache --splits train,val,test --stride 1 --max-per-shot 60 --pylib "..."
python run_module34_local.py ... --stage m3 --out-dir cnn_mlp/results_v2/module34/mlp_fa_v3 --extra n_eval=80,sensor_budget=700,worst_k=5,beam=2,pairs=0,procs=8,threads=4
python run_module34_local.py ... --stage m4 --extra n_val_a=150,n_val_b=300,n_test=300,n_train=2000,n_lat=100,n_sel=80,procs=7,threads=3,tiers_json=<criticality.json>
python run_module34_local.py ... --stage lat --extra n_lat=150,threads_sweep=1-2-4-8        # quiet machine
python test_module34_study.py
```
For the final table: more seeds (aggregate with `aggregate_seeds.py`), n_eval >= 200 and sensor_budget >= 3000 (GPU forward), pair interactions
(`pairs=1`), worst-k to 8, the Transformer per-sensor-mask checkpoint, GPU latency, and the plain / mask / mask+PINN rows from the harness.
