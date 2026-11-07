# Module 1 v2: physics evaluation against EFIT truth

Files: `physics_evaluation.py` (metrics), `efit_truth.py` (EFIT truth + window alignment), `xpoint_diagnostic.py` (checks),
`test_physics_evaluation.py`.

## What changed and why

1. **Geometry.** The zarr says R = 0.06..1.98 m and Z = -2..+2 m on 65 points: **3.00 cm per column, 6.25 cm per row**,
   `psi[z_row, r_col]` (zarr dims `z, major_radius, time`; the flux along the EFIT LCFS is constant only in this
   orientation). The model target is that raw 65x65 map (the TokaMark pipeline only standardises it: `(psi - mean) / std`;
   no crop, no resample). v1 used 2.8125 / 4.6875 cm per pixel: radial errors were reported 6.3% too small, vertical
   errors 25% too small.
2. **The 35% contour was not the plasma boundary.** On EFIT's own psi the LCFS flux sits at 0.73 of (psi_min..psi_max)
   (5th..95th percentile 0.61..0.80, 941 windows, 16 shots), because PF-coil lobes stretch the range. A 35% contour lies far outside the plasma
   and runs along the grid edge. v2 takes `psi_b` = median of the true psi sampled along the EFIT LCFS.
3. **Argmax is not always the axis.** In ~7% of windows the global psi maximum is a coil lobe outside the LCFS. v2 takes the
   extremum inside the EFIT LCFS (sub-pixel step on the unmasked field).
4. **X-points.** EFIT lists up to two (upper and lower). v1 searched the lower half only, at the 35% level, with a flux
   tolerance (15%) that flags an X-point in ~95% of windows that have none (8-shot check). v2 uses the true `psi_b` and a 2% tolerance.
5. **kappa / delta.** On EFIT's own LCFS polygon the formulas below reproduce `elongation`, `triangularity_upper/lower`
   and `minor_radius` to 1e-6: `a = (Rmax-Rmin)/2, R_geo = (Rmax+Rmin)/2, kappa = (Zmax-Zmin)/(2a),
   delta_up = (R_geo - R(Zmax))/a, delta_low = (R_geo - R(Zmin))/a`. (`elongation_axis` is a different quantity.)
6. **Axis bug.** `compute_axis_shift` masked the prediction to one half and refined the masked copy: identical fields gave
   2.3 cm. The peak is now searched in the region but refined on the unmasked field (identical fields give 0).

## API

```python
from physics_evaluation import evaluate_window_physics, aggregate_physics_metrics
from efit_truth import TruthIndex

ti = TruthIndex(data_dir, outlier_yaml=None)          # data_dir/<shot>.zarr ; outlier_yaml = MAST_tools/metadata/<split>_split_outlier_metadata.yaml
tr = ti.get(shot_id, window_index)                     # dict, or None when the dataset never emits that window
rec = evaluate_window_physics(yt_65x65, yp_65x65, truth=tr)          # mode "efit"
rec = evaluate_window_physics(yt, yp)                                 # mode "heuristic": 35% method, real geometry
rec = evaluate_window_physics(yt, yp, mode="legacy")                  # v1 numbers (old geometry)
agg = aggregate_physics_metrics(list_of_recs)                        # do not mix modes in one list
```

`yt`, `yp` may be in any affine unit of psi (standardised or Wb/rad), the same for both. `truth` keys: `shot_id, window_index,
t_cut, t_psi, slice_index, grid{r0,dr,z0,dz in m}, psi_raw[65,65], axis_rz_m(2), lcfs_rz_m(n,2 | None), xpoints_rz_m(k,2),
n_xpoints, kappa, kappa_axis, delta_upper, delta_lower, delta, minor_radius_m, ip_rating`.

Record keys (all modes): `boundary_mean_cm, boundary_hausdorff_cm, axis_shift_cm, xpoint_err_cm, xpoint_true_found,
xpoint_pred_found, kappa_{true,pred,err}, delta_{true,pred,err}, d_wall_{true,pred}_cm, is_false_alarm, is_missed_alarm, mode,
psi_b`. efit mode adds `psi_b_spread_pct, efit_lcfs_available, boundary_closed, boundary_failed, level_bump,
delta_{upper,lower}_{true,pred}, xpoint_n_true, xpoint_n_hit, xpoint_oracle_err_cm, xpoint_true_field_found,
*_oracle` (the TRUE psi scored against EFIT: the measurement floor), `kappa_err_field, delta_err_field`.

## Honest limits

* **No wall.** The zarr and TokaMark metadata contain no vessel / limiter signal. `WALL_BOX_CM` is the EFIT grid box
  (R 6..198, Z -200..+200 cm), looser than the real vessel (clearances are over-estimated). Alarm rates are a protocol
  demonstration, not a safety result. The outermost EFIT LCFS seen in the data is printed by `xpoint_diagnostic.py`.
* Windows without an EFIT LCFS (13% of dataset windows) get no boundary / shape score (nan); they are counted
  (`n_boundary_valid`). A prediction with no contour at `psi_b` is scored `BOUNDARY_FAIL_CM` and counted
  (`boundary_fail_rate_pct`), not dropped.
* kappa / delta from a 65x65 contour carry a floor: kappa is biased about -0.025, delta about -0.01 even on error-free psi
  (`*_oracle` keys). Compare models with `kappa_err_field` (prediction vs the contour of the true psi) when that matters.
* `TruthIndex` re-implements the dataset's window arithmetic (checked against the real `TokaMarkDataset`, see
  `xpoint_diagnostic.py --prove-alignment`). Without `outlier_yaml` no signal is treated as an outlier.
