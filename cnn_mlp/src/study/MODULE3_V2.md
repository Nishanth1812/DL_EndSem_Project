# Module 3 v2: physics-grounded sensor criticality (model-agnostic)

Files: `sensor_attribution.py` (Shapley, worst-k, minimum set), `sensor_geometry.py` + `geometry/mast_sensor_geometry.json`
(real geometry), `greens_sensitivity.py` (Green's functions, linear-Gaussian information model, rank agreement).
Tests: `python test_sensor_attribution.py`, `python test_greens_sensitivity.py` (no torch, no zarr needed).

## Sensor inventory (115, verified on zarr shots 12384/13222 and the run log shapes)
flux_loop 15, ccbv 40, obr 18, obv 18, saddle_v 12, coil_i 10 (P2IL P2IU P2OL P2OU P3L P3U P4L P4U P5L P5U), solenoid_i 1, ip 1.
Model inputs per window: `(1,1,T,n)` for branches 0-5 (sensor axis LAST), `(1,1,T)` for solenoid and ip.
`SensorInventory.apply_mask_to_inputs` zeroes sensor columns (whole tensor for the two scalar branches).
The failure-aware model's health input is per branch only (8), so a single dead sensor reaches it only as zeros.
Caution: `robustness._zero_channels` indexes axis -2, which for branches 0-5 is TIME, not sensors (confirmed on a real batch, see the section at the end).

## Attribution (all on `value_fn(mask) -> error`, mask True = available)
Failure game: f(F) = value_fn(~F). Shapley phi_i = mean damage when i joins the failed set (efficiency: sum = f(all) - f(none)).
* groups: exact, 256 coalitions in one batch (`full_table`, `exact_shapley`, `exact_interaction`).
* sensors: antithetic permutation sampling, Welford mean/SE, stops on `ci_tol`, `budget_evals` or `max_pairs`;
  `max_failed=K` truncates to realistic K simultaneous failures (cost K+1 per path) and estimates the mean marginal damage over
  coalition sizes 0..K-1. All calls cached by mask bytes; `vectorised=True` batches masks into one model pass.
* interaction I_ij > 0 backup/redundant pair (damage only when both are lost), < 0 overlap (either loss already hurts).
* classes from drop-one delta and Shapley: critical / backup (harmless alone, harmful together) / redundant / minor;
  thresholds default to 50 % and 5 % of the headroom `tol - clean`.
* worst-k: greedy, beam, exhaustive (groups: all k; sensors: only if C(n,k) is small). Greedy is provably suboptimal on
  synergistic pairs (tested), so report greedy/beam against the group optimum.
* minimum set: greedy backward and beam backward on sensors, greedy on groups, Shapley-ranked backward, exhaustive on groups,
  with the full size-vs-error curve. The earlier "47 -> 25 probes" claim is unsupported: 47 came from a hard-coded per-branch
  table, not the 115-channel inventory.

Cost (unique model passes): groups 256; drop-one 116; sensor Shapley = `sensor_budget_evals` (default 6000);
worst-k about beam_width * 115 per level; greedy minimum set about 115^2/2 = 6.6k. One pass = value_fn on the evaluation windows.

## Geometry (all copied, provenance per sensor in the JSON)
Source: FAIR-MAST level-2 zarr, shot 30421, `https://s3.echo.stfc.ac.uk/mast/level2/shots/30421.zarr`, groups magnetics, pf_active,
wall (ingested from the MAST EFIT++ `magnetics.xml`, `pfSystems.xml`, `limiter.xml`; mapping pointers in
`github.com/ukaea/IMAS_MAST_mappings`). Flux loops, CCBV/OBR/OBV pickup coils, saddle loops (polygons), PF coil turn rectangles,
solenoid, limiter outline. Cross-check: PF coil centroids vs FreeGS `MAST()` agree to 0.3-2.9 cm (P3-P5), 1.6-6.1 cm (P2, FreeGS has
one P2 for inner and outer). Missing: coil orientation vectors (component inferred from the name: CCBV, OBV = B_Z, OBR = B_R),
geometry for older shots (only shot 30421 fetched), OBR19/OBV19 (geometry has 19, TokaMark files carry 18, matched by name),
passive structure currents and the unit/turn convention of "feed current" (each element is taken as one turn).
Ip has no position.
Saddle loops: 12 identical rectangles on R = 1.999 m, Z in [-0.4, 0.4], 20 deg wide every 30 deg, measuring a voltage.

## Green's functions
`green_psi`, `green_br`, `green_bz` (complete elliptic integrals, vectorised; small-m series avoids cancellation). Validated against
the on-axis loop, G(a,b)=G(b,a), Neumann and Biot-Savart quadrature, and B = curl psi by finite differences.
`build_model(geo)` gives a linear-Gaussian model whose `value_fn` is sqrt(mean posterior variance of psi at the target points)
and plugs into `sensor_attribution`. Scores: `loo_loss` (information lost when s is removed), `alone_gain`, `noise_prop`.
Prior scale, `rel_noise`, kernel length and coil share are modelling choices: in a check, the leave-one-out ranking kept Spearman
0.91-0.99 (Kendall 0.75-0.91) against the default for rel_noise 0.01/0.05, ell 0.1/0.4, coil_share 0.1/10. Vacuum fields only.
`rank_agreement(a, b)` and `rank_agreement_by_group` give Spearman and Kendall with permutation p-values;
`error_map_vs_location` correlates a 65x65 error map with the distance to a sensor (and optionally its response map).

## Open items
* The hard-coded `FLUX_LOOP_COORDS` in `pipeline.py` do not match the real loops on the default grid (median distance to the nearest real
  loop 0.35 m, max 0.86 m; to the same-index loop 1.2 m). Grid axes are parameters here; the re-derived grid should be passed in.
* Nothing here has run against a trained model. The Green's comparison needs the data-driven Shapley from a real `value_fn`.

## Confirmed input-axis bug in robustness.py / pipeline_robustness.py (real batch, shot 13222)
Built with the real TokaMark dataset, `ModelTransform_1/2` and `model_collate_fn` (batch of 16, local zarr, official stats file).
Collated shapes: profile branches `(B,1,1,T,n)` = flux_loop (16,1,1,25,15), ccbv (..,25,40), obr (..,25,18), obv (..,25,18),
saddle_v (..,250,12), coil_i (..,20,10); scalar branches solenoid_i, ip `(B,1,1,20)`. Axis -1 = SENSORS and axis -2 = TIME on the
profile branches (a window column matches the raw zarr row of the same sensor with correlation 1.000000); on the scalar branches
axis -1 = time and axis -2 is a singleton. Real `apply_scenario` / `apply_group` then do this (fraction of values zeroed):
* `channel_ablation`, `category_abl`, `group_failure` (`_zero_channels(arr,[0])` = `out[...,0,:]`): solenoid_i and ip 100 % (correct);
  every profile branch only TIME STEP 0 of all sensors: flux_loop 3.7 %, ccbv 3.9 %, obr 4.0 %, obv 3.8 %, coil_i 5 %,
  saddle_v 0.3 %. So "lose flux loops / pickup coils / saddle / coil currents / all magnetics" was a 1-sample dropout.
* `temporal_gap`, `disruption_proximate` (`out[...,s:s+g]` on the last axis): scalar branches correct (time block); profile branches
  lose a contiguous block of SENSOR COLUMNS for the whole window (gap 0.4: flux_loop 6/15 sensors, ccbv 16/40, saddle 5/12), not a
  time gap; `where="end"` drops the highest-index sensors.
* `ffill` imputation runs along the last axis, i.e. across sensors on the profile branches (verified: mismatch with "previous time
  sample"); `mean` is elementwise and fine.
* `random_dropout` is correct (elementwise, fraction = severity). `physics_study.predict(dead=[c])` zeroes whole branches (correct), and
  `physics_study.inject` is axis-aware, so Module 3 criticality and Module 4 injection in the current code are not affected.
Affected old numbers: every robustness.py / pipeline_robustness.py result for channel_ablation, category_abl, group_failure,
temporal_gap, disruption_proximate and their mean / ffill variants, on the profile branches (e.g. `results/robustness_lstm/robustness`
rows such as "flux_loop lost +13.9 %", "saddle_v lost +3.7 %"); any tier or ranking derived from them. The failure-aware path in the
study copy passes `dead_channels` as a health mask, which removes the branch regardless of the input, so those rows were whole-branch
there but 1-sample for plain models. Fix (patched copy verified on the real batch; diff in the scratchpad `axis_check/robustness_axis_fix.diff`):
add `_time_axis(arr) = -2 if arr.ndim >= 5 else -1`; use it in `_mask_temporal_gap` and in the ffill loop; make
`_zero_channels(arr, chans=None)` blank the whole branch for `chans=None` or scalar branches and `out[..., chans]` otherwise; call
`_zero_channels(arr)` in `apply_scenario` and `apply_group`. The same lines exist in `cnn_mlp/src/robustness.py`.
