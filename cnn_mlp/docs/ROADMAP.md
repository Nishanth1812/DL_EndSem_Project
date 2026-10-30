# Roadmap / Next Stages

Living plan for where this project goes next. Ordered by dependency, not by
glamour. Each stage lists goal, method, and a concrete "done" test.

---

## Stage 0 — Understand & stabilise  (current)

**Goal:** real understanding of the code, results, and external references.

- [x] Reorganise repo (src / results / scripts / docs).
- [x] Standalone architecture reference (`docs/reference/architecture_multiconv_mlp_standalone.py`).
- [x] Plain-English guide (`docs/EXPLAINED.md`).
- [x] Fetch + summarise the two key papers (`docs/PAPERS.md`).
- [x] Two baseline runs captured (20 GB, 40 GB) with honest caveats.
- [ ] (optional) `git init` later — `.gitignore` already prepared.

**Done when:** you can explain NRMSE, the two model variants, and why 40 GB
overfit, without re-reading the code.

---

## Stage 1 — Robustness study on task_1-3  (the main event)

**Goal:** quantify how the trained task_1-3 model degrades when sensors fail —
mirroring Gupta (arXiv 2607.11915) but on the equilibrium-reconstruction task
(which Gupta did NOT cover; he used task_4-4). This is the novel contribution.

### 1.1 Build the inference harness
- Load `best_model.pt` (raw `state_dict`; build empty model first via
  `make_model`, then `load_state_dict`).
- Re-build the test loader for the trained subset, run forward, un-standardise ψ.
- Reproduce the clean baseline NRMSE as a sanity check (must match ~0.14).
- **Deliverable:** `src/robustness.py` (`corrupt | eval-robust | report` commands).
  **STATUS: COMPLETED & CAPTURED** (2026-10-22). Successfully executed 63-task sweep on Kaggle dual T4 GPUs.
  Clean baseline gate passed with $\Delta = 0.0000$ (`NRMSE = 0.1419`). Generated publication artifacts:
  `robustness_criticality.png`, `robustness_mitigation.png`, `robustness_psi_reconstruction.png`,
  `signal_importance.png`, `spatial_error_clean.png`, and `robustness.md`. Captured in `results/robustness/`.

### 1.2 Implement the six corruption scenarios (Gupta taxonomy)
Applied to the **test split only**; model always trained on clean data.
1. **Random dropout** — p in {0.10, 0.25, 0.50} of nonzero values zeroed.
2. **Channel ablation** — n in {1, 3, 6} whole channels zeroed.
3. **Per-category importance** — zero each signal category in isolation.
   (task_1-3 has 8 signals; group them: flux loop / pickup fields / saddle /
   active coil / solenoid / ip.)
4. **Temporal gap** — f in {0.20, 0.40, 0.60}, at front / random / **pre-event**.
5. **Correlated group failure** — zero a physically related group at once.
6. **Disruption-proximate failure** — f in {0.10, 0.25, 0.50} at the **end** of
   the window (most safety-critical).

### 1.3 Imputation strategies
- zero-fill (default), mean-fill (train mean), forward-fill.

### 1.4 Metrics
- Per-window and per-shot **NRMSE / NMAE / RMSE / MAE**, corrupted vs. clean.
- **Robustness Score** RS = mean over scenarios of NRMSE_clean / NRMSE_s.
- **Spatial error maps**: for a 2-D output we can show *where* in the ψ grid
  errors concentrate (missing from Gupta's scalar task) — core/edge/LCFS.

### 1.5 Plots & report
- Degradation curves (NRMSE vs severity) per scenario.
- Signal-importance bar chart (which of the 8 inputs matters most).
- Robustness Score table across scenarios x imputation.
- **Deliverable:** `results/robustness/` with figures + `robustness.md`.

**Done when:** a table like Gupta's Table 4 exists for task_1-3, with honest
subset caveats.

---

## Stage 2 — Address the overfitting finding  (cheap, high-value)

**Goal:** test whether 40 GB's val collapse is fixable, and whether data size or
regularisation is the lever.

- **2a. Patience experiment:** re-run 40 GB with `TM_PATIENCE` raised (e.g. 25)
  and deadline extended → does val keep falling past 3500 steps, or truly plateau?
- **2b. Regularisation experiment:** 40 GB with weight_decay 1e-3 (10x) and/or
  added dropout → does train loss rise while val loss falls? (confirms memorisation)
- **2c. Data-size curve:** the 20 vs 40 GB pair already hints at the trend; a
  3rd point (e.g. 60 GB / more tars) would make it a genuine curve.

**Done when:** we can state empirically whether 40 GB overfits from *data
scarcity* or *insufficient regularisation*.

---

## Stage 3 — Model comparison  (optional, for depth)

- Train the **CNN+LSTM** sibling (`LSTM_v3`) on task_1-3 and compare, or
- compare CNN vs a Transformer/XGBoost head, as Gupta did.
- Note: task_1-3 is markovian, so LSTM is not expected to help much — an
  interesting *negative* result if it does not.

---

## Stage 4 — Write-up / packaging

- Consolidate: clean README, results tables, figures, and a short methods note.
- Decide whether to (a) keep as private study, or (b) frame as a task_1-3
  robustness extension of Gupta for the community.

---

## Risk register

| Risk | Mitigation |
|------|------------|
| 7 h GPU wall | keep budget at 40 GB; lower patience; deadline-aware train |
| Dataloader hang | `DL_WORKERS=0` always on Kaggle |
| Subset not official | never quote absolute numbers as official; label everywhere |
| `best_model.pt` needs torche | build model first, then load state_dict |
| Temp dirs wiped | all durable scripts/refs copied into the repo already |
| Local RAM ~0.4 GB free | never run heavy code locally; static checks only |
