# Module 4 v2: adaptive failure mitigation with calibrated confidence

Files: `fault_models.py` (injection + labels + scoring), `adaptive_mitigation.py` (detector, router, gauge, brain),
`virtual_sensor.py` (virtual sensors, physics fallback), `test_mitigation_v2.py` (`python test_mitigation_v2.py`, about 1 minute, numpy only).
The legacy classes (`RealTimeFaultDetector`, `AdaptiveRouter`, `ConformalGauge`, `SmartAdaptiveMitigationBrain(tiers, gauge, known_missing).process(window, model_fn)`)
behave as before when no v2 component is passed, so `physics_study.py` runs unchanged. The old "2.1 ms / 0.25 ms detector" figures were never measured and are not reused.

## Data layout
A window is a list of 8 branch arrays `(T_b, P_b)` (15, 40, 18, 18, 12, 10, 1, 1 probes = 115 channels; T_b differs per branch: 25 magnetics, 250 saddle, 20 PF/Ip in 5 ms).
`WindowLayout(window)` flattens it to a canonical `(25, 115)` matrix (block mean / interpolation) for detection and regression, and writes repairs back at native resolution
(only the flagged samples change). Limitation: a 1-sample spike on a decimated channel (saddle) is diluted by the block mean.

## Fault models (`fault_models.py`)
`FaultSpec(channel, kind, onset_ms, duration_ms, magnitude, n, fill)`; kinds `stuck_rail`, `stuck_value`, `drift` (sigma/ms), `spike`, `saturation`, `dropout` (0 or NaN fill), plus
`group_loss_specs(layout, branches)`. `inject(X, specs)` (canonical) and `inject_window(window, layout, specs)` (native) return the faulty data and `FaultLabels`
(kind, onset_ms, duration_ms per channel, sample mask). `benchmark_detector(detect_fn, X_windows, ...)` gives per-kind recall / precision / F1, isolation top-1 / top-3 and the clean false-positive rate.

## Detector (`FaultDetector`)
1. Local rules: NaN / exact-zero runs (>= 2 samples) -> dropout; run of identical samples longer than any clean run, or window variance below every clean window -> flatline / rail (|z| > 4.5);
   |z| above max(6.5, 1.25 x clean maximum) -> spike. Probes that are constant/zero in training are "absent", never faults.
2. Cross-sensor residuals: Gaussian conditional of each channel given all others (ridge-regularised precision matrix, ridge picked out of sample on calibration windows); residuals standardised
   with their out-of-sample scale; three statistics per channel (peak, rms, total change across the window = drift) against pooled peaks-over-threshold quantiles of clean calibration
   residuals (`fp_rate` per channel per statistic). **Isolation by peeling**: take the largest ratio, flag it, marginalise it out of the precision matrix (rank-1 Schur update), re-score; so a fault does not smear to neighbours and the flag order is the isolation order.
3. `FluxLoopCheck(loop_pixels, loop_channels)`: measured loop flux vs the reconstruction's psi at the loop pixels (per-loop linear map and robust sigma fitted on clean windows, pooled threshold).
   `loop_pixels_from_coords(pipeline.FLUX_LOOP_COORDS)` converts (r_idx, z_idx) to (row, col) of `psi[z, r]`. Pass `psi_pred` to `detect`. Caveat: if the faulty loop fed the same reconstruction, the model partly follows it; run the check on a pass where the flagged loop is already masked.
`demean=True` cross-checks only intra-window dynamics (insensitive to shot-level shifts, blind to pure offsets).
Fit: `FaultDetector().fit(train_X (N,25,C), calib_X (N2,25,C))` with clean windows; use separate calibration windows or the false-positive rate is optimistic.

## Router (`RouterV2`, defaults are policy placeholders, set them from module 3 / validation)
| situation | action |
|---|---|
| nothing flagged | healthy |
| spike or dropout no longer than `short_ms` (1 ms), any tier | forward-fill (hold the last good sample) |
| long fault, redundant sensor, virtual-sensor R^2 >= `virtual_r2_min` (0.9) given the surviving neighbours | virtual sensor |
| long fault on a tier-1 probe, a branch with >= 50 % of probes lost, or >= 6 lost probes (branch threshold 25 %), or virtual R^2 too low | failure-aware model, branch(es) masked (the model's mask is per branch) |
| more than `max_masked_branches` (5, what structured augmentation trained: magnetics_all) masked, or no model | physics fallback |
Tiers default to `DEFAULT_TIERS` (module 3's last published tiering, per branch); pass `tiers=` from `criticality.json`. `route_and_mitigate(x_window, mask=None, diagnoses=<Detection>, oracle_faults=<FaultLabels>)` returns `(repaired window, branch mask, route, per-channel actions)`.
Oracle: `oracle_faults=` (or `SmartAdaptiveMitigationBrain(oracle=True).process(window, model_fn, oracle_faults=labels)`) skips detection and routes on the true labels, the upper bound of detector + router for the table's "Oracle mitigation" row.

## Physics fallback (`LinearFluxFallback`, `fit_physics_fallback(window_mean_features, psi)`)
Gaussian-conditional (linear least-squares) estimate of the whole psi map from whichever channels survive (flux loops, PF currents, solenoid, Ip). Statistical linear reconstruction. It is not EFIT and does not solve Grad-Shafranov.
Whether it beats the failure-aware network anywhere is unmeasured; `max_masked_branches` decides, so compare on validation before relying on the route.

## Confidence
`ConformalGauge` (per route): delta = the ceil((n+1)(1-alpha))-th smallest calibration error (correct split-conformal index). Coverage caveats: (a) `physics_study` replays the same validation windows under 8 scenarios and
neighbouring windows of a shot are correlated, so the calibration set is not exchangeable and the guarantee is approximate; (b) one delta per route is marginal coverage only: too wide on easy windows, too tight on hard ones (synthetic: 100 % coverage on easy, 71 % on hard at a 90 % target);
(c) routes with < 20 calibration windows inherit the largest delta (conservative). `AdaptiveConformalGauge.calibrate_adaptive(errors, difficulty_features)` fits a linear difficulty model on half the calibration set, conformalises the normalised errors on the other half, `delta(f) = q * sigma_hat(f)`; features from `difficulty_features(detection, decision)`.
Flag per window (`confidence_flag`): ABSTAIN if delta > 1.0 x wall margin (5 cm) or the output is invalid; HIGH if |clearance - 5 cm| > delta and delta <= 0.5 x margin; else LOW. The alarm rule stays `clearance - delta < 5 cm`. The 0.5 / 1.0 fractions are policy parameters, not tuned.

## Measured here (this laptop, numpy CPU; see the test output for the exact run)
- Synthetic, C=115, 300 windows, one fault per window, clean-calibrated thresholds: recall / precision / isolation top-1: stuck_rail 1.00/0.99/1.00, stuck_value 1.00/0.99/1.00, spike 1.00/0.99/1.00, dropout 1.00/0.99/1.00, drift (0.6 sigma/ms) 0.94/0.99/0.99, saturation (clip at 0.8 sigma, mild) 0.63/0.99/0.63; clean false-positive rate 0.7 % of windows. Three simultaneous faults: recall >= 0.97.
- Real MAST data (14 local shots, split by shot 1719/410/249 train/calib/test windows, 19 NaN probes removed; faults injected into real clean windows): clean false-positive rate 9.2 % of windows (0.68 % of channels); recall stuck_rail 0.98, stuck_value 0.90, drift 0.88, spike 0.88, dropout 1.00; isolation top-1 0.98 / 0.90 / 0.84 / 0.87 / 1.00. Precision is only 0.57-0.61 because of those false positives.
  With a single shot as training data the same detector flagged nearly every held-out window: the cross-sensor model needs many shots.
- Virtual sensor (synthetic, 80 % gap): RMSE 0.23 sigma vs forward-fill 0.46 vs zero-fill 0.96. Physics fallback (synthetic linear world): see test output. Real-data virtual-sensor numbers: not measured.
- Latency (stub model, C=115, 300 windows, p50 / p99 ms; two runs, machine-load dependent): detector 0.75-1.8 / 1.7-4.7, router 0.23-0.49 / 0.8-1.6, mitigation 0.22-0.54 / 0.7-2.1, gauge 0.03-0.08 / 0.1-0.3; total 1.3-3.1 / 2.7-8.1. The model stage is the real network's forward pass and is not included.
- Conformal (synthetic exchangeable, alpha = 0.1): marginal coverage 0.89 (global) and 0.89 (adaptive); adaptive coverage easy / hard 0.91 / 0.88 vs 1.00 / 0.71 for the global delta.

## Unverified (needs the trained network on Kaggle/Modal)
Boundary error in cm of every route, the Oracle row, the fallback vs failure-aware crossover, real calibrated delta and its coverage on held-out shots, flux-loop residual on real model output, and the router thresholds (`short_ms`, `virtual_r2_min`, ...).

## Integration edits (owned by others)
- `physics_study.py::run_mitigation`: build `layout = WindowLayout(twin[0])`; canonical train/calibration matrices `layout.flatten(w)` from the train and val loaders;
  `det = FaultDetector().fit(Xtr, Xcal)`; `bank = fit_virtual_sensors(Xtr)`; `fb = fit_physics_fallback(np.stack([fb.features(x) for x in Xtr]), psi_train_in_the_model_output_units)`;
  `brain = SmartAdaptiveMitigationBrain(tiers, gauge, known_missing, detector=det, bank=bank, fallback=fb)`; Oracle row: `SmartAdaptiveMitigationBrain(..., oracle=True, bank=bank, fallback=fb)` and `process(damaged, model_fn, oracle_faults=labels)`.
- Replace `inject()` by `fault_models.inject_window(window, layout, specs)` (`group_loss_specs`, `FaultSpec(..., "dropout", onset, duration)` for gaps / pre-disruption, `spike`, plus new `drift` / `stuck_*` scenarios) so each damaged window carries its labels.
- Calibrate `AdaptiveConformalGauge` on `(error, difficulty_features(res["detection"], res["decision"]))` pairs; report `summarise_latencies([r["latencies_ms"] ...])` (p50/p95/p99) instead of mean/p95 only; `process` now also returns `detection` and `decision` (v2 mode only).
- `pipeline_robustness.py`: no edit needed. `make_corrupt_fn` still produces zero-fill corruption for the sweep; the new fault kinds are only used by the mitigation study.
