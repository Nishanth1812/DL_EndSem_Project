# TokaMark `task_1-3` Kaggle baseline — changelog & runbook

Deliverable: `src/tokamark_baseline_kaggle.ipynb` (14 cells; cell 3 is byte-identical to the
tested `src/pipeline.py`). All fixes verified by static checks + a local end-to-end prep
integration test against real HF data and the official split CSVs (training/eval are
Kaggle-only and untested locally by design — no torch on this machine).

## Changelog (vs the original notebook)

### Dependencies
- Added `torchinfo==1.8.0` — imported by `src/multi_conv_mlp_model.py`; missing it crashed
  training before step 1. Added `filelock`; pinned `fsspec==2025.3.0` / `s3fs==2025.3.0`.
- `pip install` now runs with `check=True` (was `| tail -2` with `check=False`, which
  swallowed errors), followed by an explicit import verification.

### Data prep
- Tar indexing: per-thread `requests.Session` (TLS keep-alive) + 4 KB coalesced range
  reads (~3x fewer requests than per-512 B reads). Tar walks abort cleanly at the prep
  deadline and resume on the next run (index saved per tar).
- Indexing stops **early** once ~1.6x the byte budget *and* >= N archives are covered
  (`TM_INDEX_STOP_GB`, `TM_INDEX_MIN_TARS`, `TM_INDEX_THREADS`) — no need to walk the
  whole 565 GB repo.
- **HTTP 429 (rate-limit) hardening** (seen on the 40 GB run: 3 tars failed to index and
  6 shots failed to download, all `RuntimeError('range ... failed: HTTP 429')`):
  - Shared token-bucket throttle for all index + download threads with an **adaptive**
    rate: the configured `TM_HF_RPS` (default 12) is a *ceiling*; every 429 halves the
    live rate (floor `TM_HF_RPS_MIN`, default 2) and it recovers slowly on success. An
    over-aggressive setting therefore self-corrects rather than failing — the server
    decides the real pace.
  - `ranged_get` honours `Retry-After`, backs off hard on 429 (up to 120 s, and pushes
    the shared clock so sibling threads slow too), retries 5xx separately, and retries up
    to 10 times (was 6) with jitter.
  - Transiently-failed tars are **re-queued** (bounded, 3 attempts) and never walked
    twice concurrently (`inflight` guard); failed shots are retried in-loop and in a
    final pass before the deadline.
  - Knobs: `TM_HF_RPS` (ceiling req/s), `TM_HF_RPS_MIN` (floor), `TM_INDEX_THREADS`
    (default 16), `TM_FETCH_WORKERS` (default 4).
  - Safety: the re-queue state machines are provably terminating (bounded retries, no
    duplicate in-flight work) — verified by simulation; the fatal `< 5 shots` guard
    remains as the last line of defence against training on a misleading subset.
- **Two-phase prep deadline protection**: `cmd_prep` runs indexing *then* downloading,
  so a slow/re-throttled index could otherwise starve the download phase. Indexing now
  gets a **soft** sub-deadline (`TM_INDEX_TIME_FRAC`, default 0.25 of the prep window):
  once it passes, in-flight walks are cancelled *if the index is already usable*
  (>= min(3, `TM_INDEX_MIN_TARS`) tars) or complete; otherwise indexing continues up to
  the **hard** prep deadline, because an under-sized index is what would trip the fatal
  `<5-shot` guard. Worst-case prep measured/simulated: index <=~42 min at the 2 req/s
  floor + download <=~27 min at a slow 25 MB/s = ~70 min, inside the 2 h window.
- Selection guarantees **>= 5 shots per split** (the 5 smallest available) before
  filling each split's proportional byte quota — prevents val/test starvation when
  quotas are smaller than a single shot. Selection is seeded and **reused** across
  re-runs (`TM_FORCE_RESELECT=1` to redo); `selection.json` records config + git revs.
- If a split ends with < 5 downloaded shots the run fails loudly with an actionable
  message instead of training on a misleading subset.

### Training
- Official hyperparameters verified against the pinned baseline repo and preserved
  (Adam lr 1e-4 / wd 1e-4, MultiOutputMSELoss, patience 10, validate every 100 steps,
  `stride_window=0.005` for task_1-x, `shuffle_windows=False` only for test/eval).
- Atomic checkpointing (`last.pt`, `best_model.pt`) every `TM_CKPT_EVERY_S` and at the
  deadline; re-running the train cell **resumes from `last.pt`**.
- Wall-clock deadline; `train_summary.json` records stop_reason, shot counts, OOM count.
- CUDA OOM: clear guidance to lower `BATCH` and re-run (resumes); aborts after 3 OOMs
  instead of looping. Non-finite-loss batches are skipped, not crashed.
- `training_history.{pt,json}` saved for plotting without reloading the model.

### Dataloader hang fix (silent freeze after "val loader ready")
- Real incident: overnight run (`BATCH=512`, `workers=4`) produced **no log line for 12 h**
  after `val loader ready`; the 7 h deadline never fired because it is checked only inside
  the training loop. Root cause class: workers stuck on tensor IPC (`/dev/shm`) or
  fork-after-CUDA-init, plus the first batch taking very long at `BATCH=512`.
- Fixes:
  - `torch.multiprocessing.set_sharing_strategy("file_system")` (no `/dev/shm`; batch
    tensors go through `/tmp`).
  - Notebook config cell remounts `/dev/shm` to 64 GB and prints `df -h /dev/shm`.
  - `TM_DL_WORKERS` semantics fixed: `-1` = auto, `0` = main-process loading (no
    workers/IPC), `N` = N workers. Set `DL_WORKERS = 0` in the config cell as fallback.
  - New config knob `DL_WORKERS` is forwarded as `TM_DL_WORKERS`.
  - Heartbeat thread logs the current phase every 60 s (dataloader build, model build,
    first batch, training step N, validation), so a freeze is immediately locatable.
  - `first_batch_timed`: logs every 30 s while waiting and hard-fails after 30 min with
    an actionable message instead of hanging forever.
  - Per-item window-probe per dataset + estimated first-batch time. `TokaMarkDataset`
    is an **`IterableDataset` (no `__getitem__`)**, so the probe pulls one sample via the
    iterator; the old `ds[0]` probe always raised `NotImplementedError` (the misleading
    "harmless warning"). An empty dataset is now **fatal** here, not swallowed.
  - First 10 steps log every step (not every 20), and `last.pt` is written after step 1.
  - Workers use `persistent_workers=True` / `prefetch_factor=2` to avoid re-forking.

### Evaluation / plots / summary
- `get_signals_metadata` is patched in **both** `tokamark.tasks` and
  `tokamark.evaluator` (upstream binds it at import time), so metadata and NRMSE
  normalisation always match the current split protocol.
- New `plot` command: `figures/loss_curves.png` + `figures/psi_reconstruction_samples.png`
  (optional part is try/except-guarded).
- New `summary` command: `out/<split>/summary.md` — metrics table + artifact checklist.

## Recommended overnight settings (T4 x2, `SESSION_HOURS = 9`)

| Setting | Preflight run | Overnight run |
|---|---|---|
| `DRY_RUN` | `True` (~30 min) | `False` |
| `DATA_GB` | 2.0 (forced) | `40.0` (raise only after a clean preflight) |
| `BATCH` | 16 (forced) | `512` (256/128 if OOM) |
| Prep deadline | 0.5 h | 2.0 h |
| Train deadline | 1.25 h | `SESSION_HOURS - 2.0` = 7 h (2 h left for prep overrun + eval) |
| `INDEX_THREADS` / `HF_RPS` / `HF_RPS_MIN` | 16 / 12 / 2 | 16 / 12 / 2 |

Budget/time notes measured on real data:
- Indexing ~6 archives (seeded spread) takes ~5–8 min on a normal connection.
- Download speed dominates: Kaggle typically does 50–200 MB/s (20 GB in ~5–10 min);
  a slow home connection measured here at ~4–5 MB/s would take hours — hence the prep
  deadline, which always leaves a usable subset.
- Shots average ~10–140 MB; 20 GB ≈ 150–2000 shots depending on mix, always drawn only
  from the official split lists.

## Launch checklist

1. Kaggle: **Internet ON**, **GPU T4 x2**, *Save & Run All*.
2. First session: set `DRY_RUN = True`, Run All. Expect, in order:
   dependency imports OK → repo checkouts at pinned SHAs → ~6 archives indexed →
   selection with >= 5 shots in train/val/test → training steps + checkpoints →
   `results/.../task_metrics.csv` → `figures/` → `summary.md`.
3. Commit `/kaggle/working` (artifacts, and optionally the downloaded data).
   Note: Kaggle wipes `/tmp` and `/kaggle/temp` on restart; `/kaggle/working` survives
   restarts **only via commits**. Resume-from-checkpoint therefore works reliably
   *within* a session; across restarts commit first or re-run prep (the selection is
   reused; downloads repeat).
4. Set `DRY_RUN = False` (optionally raise `DATA_GB`), **restart the session**, Run All.
5. Before the session ends: commit `/kaggle/working`; copy out
   `out/random/{summary.md, train_summary.json, selection.json, results/, figures/,
   model/}`.
6. Recovery:
   - CUDA OOM → lower `BATCH` (256/128) and re-run the train cell (resumes).
   - Session died mid-train, same session → just re-run the train cell.
   - Prep too slow / cut by deadline → re-run the prep cell; it continues the index and
     keeps the same selection; the fatal guard only fires if a split would have < 5
     shots.
7. Never mix protocols: random is the default; the temporal protocol is a separate
   opt-in cell with its own shots/stats/outliers.

## Known limitations / not verified
- Training, evaluation and plotting depend on torch and were not executed locally
  (machine constraint: no torch install). They were statically checked and mirror the
  official training/eval code paths; the DRY_RUN preflight is the intended live test.
- A parallel working copy `tokamark_baseline_kaggle_fixed_v4.ipynb` (and `_fixed`,
  `_fixed_v2`) exists in this folder from earlier iterations; the canonical deliverable
  is `tokamark_baseline_kaggle.ipynb`.
