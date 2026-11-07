# Training variants, seeds and tuning (pipeline.py)

One training loop (`cmd_train` -> `train_loss`) serves every model and variant: Adam, lr 1e-4, wd 1e-4, batch 512,
ReduceLROnPlateau, validate every 100 steps, patience 10, grad-clip 1.0. Only the knobs below differ between runs.

## Models (`TM_MODEL`)

| value | model | params (115 sensors, 65x65 psi) |
|---|---|---|
| `mlp` | CNN encoder per group + MLP | 287,377 plain / 287,889 failure-aware |
| `lstm` | CNN encoder per group + LSTM | 344,049 plain / 345,073 failure-aware |
| `transformer` | one token per sensor, default d_model 64, 3 layers, 4 heads | 288,449 (d_model 128: 776,769) |

Counts measured on CPU with the real shapes (decoder family `decoder_mlp` + `output_branches` = 92,897 params in all three).

Real input shapes (per sample, from a Kaggle log): 8 groups `(1,1,25,15) (1,1,25,40) (1,1,25,18) (1,1,25,18) (1,1,250,12)
(1,1,20,10) (1,1,20) (1,1,20)` = 15+40+18+18+12+10+1+1 = 115 sensors (batch tensors add a leading B); target `(B,1,65,65)`;
no exogenous inputs. Existing failure-aware masks are group level `(B, 8)`; the transformer accepts `(B, 8)` or per-sensor `(B, 115)`.

Transformer: per-group `Linear->GELU->Linear` embeds each sensor's series (time length differs per group: 25/250/20),
plus learned sensor-ID and group embeddings, pre-LN encoder, learned always-valid CLS token. Missing sensors are removed
with `key_padding_mask` and from the masked-mean pool (not zero-filled; verified bit-exact on CPU). Pooling = masked
mean + learned-query attention; the head feeds the same decoder as the CNN models. With no mask given, an exactly-zero
sensor is treated as missing (same rule as the failure-aware CNNs). Training augmentation for it is per sensor
(`apply_structured_failure(..., per_channel=True)`: 40% 1-6 random sensors, 30% one correlated group, 30% temporal gap on
1-6 sensors); the CNN models keep the unchanged group-level augmentation.

## Ablation variants (`TM_VARIANT`)

| `TM_VARIANT` | failure_aware | pinn_lambda | meaning |
|---|---|---|---|
| `plain` | 0 | - | baseline |
| `mask` | 1 | 0 | failure-aware mask + structured augmentation (p = `TM_FAIL_AUG_P`, default 0.5); flux-loop term is not computed at all |
| `mask_pinn` | 1 | `TM_PINN_LAMBDA` (default 0.1, must be > 0) | + flux-loop consistency loss (`TM_PINN_MODE`, below) |

If `TM_VARIANT` is unset the legacy `TM_FAILURE_AWARE` / `TM_PINN_LAMBDA` pair decides (unchanged behaviour); the variant
name is derived from them. `plain` and `mask` never compute any flux-loop term.

## Flux-loop consistency loss (`TM_PINN_MODE`, variant `mask_pinn` only)

* `measured` (default for new runs): the 15 real loop positions (`FLUX_LOOP_RZ`, from `study/geometry/mast_sensor_geometry.json`)
  are mapped to fractional pixels on the real grid (R = 0.06 + 0.03 i, Z = -2 + 0.0625 j, `psi[z_row, r_col]`), the predicted psi is
  sampled there by differentiable bilinear interpolation, and compared with the loop's **own measured flux in the same window**
  (input group 0, first time sample = the sample nearest the psi slice) converted to the psi standardisation:
  `psi_target_std = ((x_std*sd_f + mu_f)/(2*pi) - mu_p)/sd_p`, with `mu/sd` taken at run time from the task metadata
  (so the temporal split uses its own stats). Equivalent affine map: `psi_std = 1.1093*x_std - 0.2650` (random split).
  A loop contributes only if it is available: not failure-masked (per-sensor column for the transformer, flux-loop group flag for the CNN
  models), not NaN-filled, and not dead (|flux| < 2 mWb over the window; CC04 reads ~0 in the 27xxx shots).
  Per-loop weight `min(1, (0.1/resid)^2)`, resid = measured-vs-EFIT rms in standardised-psi units: 1.0 for the five inboard CC loops,
  0.10-0.19 for the outer loops (EFIT psi is unreliable next to the PF coils).
* `legacy` (`TM_PINN_MODE=legacy`): the old term - hard-coded `FLUX_LOOP_COORDS` pixels compared with the **true** psi target, weighted by the
  flux-loop group flag. Those pixels are NOT the real loops (median 0.35 m from the nearest real loop), and it is not a measurement
  consistency check. **All checkpoints trained before this change (e.g. `results/run_40gb_CNN_MLP_failure_aware`, the `mlp_fa`/`lstm_fa` runs) are "pinn legacy".**
  Resuming such a run needs `TM_PINN_MODE=legacy` (a mode mismatch with the stored `run_config.json` aborts); `auto` tags add `_legacy` for it.
* Mode is recorded in `run_config.json` / `train_summary.json` as `pinn_mode` (`none` for plain and mask).

Measured relation (16 local shots, 1083 windows, EFIT psi bilinear at the real loops vs `magnetics/flux_loop_flux`): flux = 2*pi*psi, positive sign.
Inboard loops (CC03,05,07,09 and CC04 where alive): fitted slope 6.20-6.31 (2*pi = 6.283), offset -0.007 Wb (window mean) / -0.002 Wb (first sample),
rms residual 0.0097 Wb = 0.028 in standardised-psi units, corr 0.9985. Outer loops (P3-P6): corr 0.94-0.98, rms 0.08-0.11 Wb = 0.23-0.32 standardised units,
which is the floor of the check, not model error. Standardisation: flux_loop_flux mean -0.2555 / std 0.3829 vs psi mean -0.0261 / std 0.0549 -
comparing the standardised numbers directly is wrong (slope 1.109 and shift 0.265 sigma; naive diff mean +0.27, std 0.28).
On the real dataset tensors (40 windows, shots 12811/27080/13222) the converted first sample matches standardised EFIT psi at the inboard loops to 0.021 rms.

## Run tags and outputs (`TM_RUN_TAG`)

* unset: legacy layout `out/<split>/{model,results,figures,run_config.json,train_summary.json}`.
* `auto`: `out/<split>/<model>_<variant>_s<seed>/...`; any other value is used literally as the sub-directory.
* `selection.json` and `tar_index.json` never move: always `out/<split>/selection.json`, shared by all runs.
* `train_summary.json` now also has `model, variant, run_tag, seed, params, steps, epochs, best_val, duration_min` (since
  session start, includes prep) and `train_duration_min` (training only), plus lr/weight_decay/dropout/d_model/pinn_lambda/tuned.
* The robustness harness loads `P.paths(split)["model_dir"]`, so it honours the tag (its own report dir is still `out/<split>/robustness`).

## Seeds

`TM_SEED` changes initialisation, augmentation and data order only. Shot selection uses `TM_SELECT_SEED` (default 23) and
is skipped entirely when `selection.json` exists (`prep` reuses it; `train/eval/plot/summary` only read it), so every seed
and variant trains/evaluates on identical shots. Use e.g. `TM_SEED=1..5 TM_RUN_TAG=auto`.

## Fair tuning (`python pipeline.py tune`)

Same budget and search space for every model: `TM_TUNE_TRIALS` (6) random trials of `TM_TUNE_STEPS` (300) steps through the
same `cmd_train` loop, over lr (log 3e-5..1e-3), weight decay (log 1e-6..1e-3), latent dropout {0, 0.1, 0.2} (and d_model
{64, 128} for the transformer only). The (lr, wd, dropout) list is generated from `TM_TUNE_SEED` (7) and is identical for
every model. Selection metric: lowest **validation** loss; only train/val shots are read, the test split is never touched.
Dropout is applied to the latent feeding the psi decoder in all models (transformer also uses it inside its layers).
Trial seed is `TM_TUNE_SEED`, not the final seeds. Finished trials are reused if `tune` is restarted.

* Output: `out/<split>/tune/<model>_<variant>/tuning.json` = `{model, variant, split, selection, budget, tune_seed,
  trials: [{trial, lr, weight_decay, dropout, [d_model], val_loss, steps, params}], chosen_trial, chosen: {...}}`
  (trial dirs `.../t<i>/train_summary.json`; their checkpoints are deleted).
* Consumption: set `TM_TUNED=<path to tuning.json>` for the final runs. At import `pipeline.py` reads `["chosen"]` and
  `hp(var, key, default)` resolves each value as **explicit env var > tuned file > default**:

| CFG key | env var (explicit override) | tuning.json key | default |
|---|---|---|---|
| `lr` | `TM_LR` | `lr` | 1e-4 |
| `weight_decay` | `TM_WEIGHT_DECAY` | `weight_decay` | 1e-4 |
| `dropout` | `TM_DROPOUT` | `dropout` | 0.0 |
| `d_model` | `TM_D_MODEL` | `d_model` | 64 |

Recommended protocol: tune each model once on `TM_VARIANT=plain` and reuse that file for its `mask` and `mask_pinn` runs, so
the ablation rows differ only by the method, not by tuning. Cost: 6 trials x 300 steps ~ 1800 steps per model (roughly 40% of one full run).

## Other new env vars

`TM_PINN_MODE` (measured|legacy), `TM_SELECT_SEED`, `TM_VARIANT`, `TM_RUN_TAG`, `TM_TUNED`, `TM_LR`, `TM_DROPOUT`, `TM_D_MODEL`, `TM_TX_LAYERS` (3), `TM_TX_HEADS` (4),
`TM_TUNE_TRIALS`, `TM_TUNE_STEPS`, `TM_TUNE_SEED`. After editing `pipeline.py`: `python cnn_mlp/scripts/sync_nb.py && python cnn_mlp/scripts/check_nb.py`.
Tests: `python cnn_mlp/src/test_pipeline_models.py` (CPU, ~1 min, synthetic tensors).
