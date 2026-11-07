# RUNBOOK_MODAL - run the TokaMark study on Modal (with Kaggle as the second engine)

Files here: `app.py` (Modal app), `run_matrix.py` -> `RUN_MATRIX.md` (what to run where, hours, cost), this runbook.
Run every command from `cnn_mlp/modal/` in a shell where `modal` works (Git Bash or PowerShell; commands below are shell-neutral).

## 0. What the app does (one paragraph)

One image (Python 3.11, torch 2.6.0 CUDA wheel, zarr 3.1.5, fsspec/s3fs 2025.3.0, torchinfo 1.8.0, tokamark@1a200f4 and
tokamark_baseline@1e7c489 cloned and pip-installed, everything else tokamark's pyproject pulls in). `cnn_mlp/src/` (pipeline.py and
`study/`) is mounted at `/opt/study_src` WITHOUT rebuilding the image, so edits to pipeline.py are picked up on the next launch.
Data: volume `tokamark-data` holds one tar per split (`tars/<split>.tar`) plus the shot contract (`work/out/tar_index.json`,
`work/out/<split>/selection.json`). A job untars to local disk, runs `pipeline.py ...` with `TM_WORK=/results/runs/<split>/<tag>` on volume
`tokamark-results`, commits that volume every 2 min (so `last.pt` survives preemption), and appends a cost record to `costlog/`.
Preemption: Modal relaunches the input; the steps carry `_done_<step>` markers and the pipeline resumes from `last.pt`. A failed step is
retried 3x in-container, a failed container 2x by `modal.Retries`, and a hard wall-clock budget (`--hours`) aborts overruns.

## 1. Install and authenticate (once)

```
pip install modal            # 1.2.1 is installed on this machine
modal token new              # browser login; writes %USERPROFILE%\.modal.toml  (never commit it)
modal profile list           # this machine has TWO profiles; pick the one that holds your credit:
modal profile activate <name>
```
Windows gotcha seen on this machine: the `modal.exe` shim printed nothing (not even `--help`) in the agent shell, while `python -m modal ...` worked
and was already authenticated (workspace has other volumes). If `modal` prints nothing for you, replace `modal` by `python -m modal` in every command below.

Check the credit balance in the dashboard (Settings -> Billing). Modal bills dollars; your "20 hrs" is either about $20 (about 22 T4-hours
all-in at the list price) or 20 literal hours (about $18 on a T4). `RUN_MATRIX.md` section 1 has the price table and its source.

Optional Hugging Face token (public dataset works without it; a token raises the rate limit): put it in YOUR shell only,
`$env:HF_TOKEN = "hf_..."` (PowerShell) or `export HF_TOKEN=hf_...`. `app.py` forwards it as a Modal Secret object at launch. Do not write it in any file.

## 2. Look before you spend

```
python app.py plan train --model cnn_mlp --variant mask --seed 1 --split random --gpu T4     # offline, no account needed
modal run app.py::check_env       # builds the image once (CPU only, cents) and imports torch/zarr/tokamark/our harness
modal run app.py::status          # lists volumes; creates them (free) on first use
```
`modal run app.py::<entry> --dry-run` also exists on every entrypoint (it prints the plan and launches nothing, but `modal run` itself builds the image first).

## 3. Put the shared shot list in place BEFORE anything downloads (the contract)

Reuse the selection already used by your Kaggle runs so Modal and Kaggle see identical shots (random split: 606/74/60):
```
modal volume put tokamark-data "../results/run_40gb_CNN_MLP_failure_aware/out/tar_index.json" /work/out/tar_index.json
modal volume put tokamark-data "../results/run_40gb_CNN_MLP_failure_aware/out/random/selection.json" /work/out/random/selection.json
```
The temporal split has no selection yet: its first `prep` creates one (then immediately `modal volume get tokamark-data work/out ./contract` and
store it in the repo/Kaggle dataset so all environments use it). Without these uploads, `prep` re-selects a DIFFERENT random subset.

## 4. Prep (CPU only, ~2-3 h wall, about $1 per split at $0.317/h; resumable)

```
modal run --detach app.py::prep --split random
modal run app.py::status          # expect: tars/random.tar ~40 GB, selection.json {'train': 606, 'val': 74, 'test': 60}
```
If it stops at its deadline (`--hours`, default 3) just run it again: it untars what it has, downloads the rest, and re-tars.
Temporal (a second ~40 GB download; only needed for tier 5+ of the plan): `modal run --detach app.py::prep --split temporal`.

## 5. Smoke test (about $0.3, ~10-15 min)

```
modal run app.py::smoke --model cnn_mlp --variant mask_pinn
```
Fetches an isolated 2 GB subset into /tmp, trains 30 steps at batch 16, evaluates, writes nothing but a cost record. Verifies GPU, image, pipeline.py env
contract and the cost log. Repeat with `--model transformer` once the transformer exists, and with `--gpu L4` to compare step time
(L4 beats T4 on cost only if it is more than 1.23x faster; read the step times in the log).
Then a 5-minute contract test on real data: `modal run app.py::train --model cnn_mlp --variant plain --seed 99 --split random --hours 0.4 --env TM_MAX_STEPS=40,TM_VALIDATE_EVERY=20 --skip-eval`
(untars 40 GB first, so allow a few minutes; the run dir `runs/random/cnn_mlp_plain_s99` can be deleted afterwards).

## 6. Real jobs

Always `--detach` so closing the laptop does not kill the job:
```
modal run --detach app.py::tune       --model cnn_mlp                       # once per model; tunes on `plain`
modal run --detach app.py::train      --model cnn_mlp --variant mask --seed 1 --split random --env TM_LR=...,TM_DROPOUT=...   # tuned values
modal run --detach app.py::robustness --model cnn_mlp --variant mask_pinn --seed 1 --split random
modal run         app.py::evaluate    --model cnn_mlp --variant plain --seed 1                       # eval/plot/summary only
```
`--env` takes comma-separated `TM_X=y` pairs and wins over the variant mapping. Model names: `cnn_mlp|cnn_lstm|transformer` (-> `TM_MODEL=mlp|lstm|transformer`);
variants `plain` (FAILURE_AWARE=0), `mask` (FAILURE_AWARE=1, PINN_LAMBDA=0), `mask_pinn` (FAILURE_AWARE=1, PINN_LAMBDA=0.1); each run also sets
`TM_VARIANT`, `TM_SEED`, `TM_RUN_TAG=<model>_<variant>_s<seed>`, `TM_USE_DP=0`. `--gpu T4|L4|A10G|L40S|A100-40GB|A100-80GB|H100`, `--hours` hard budget
(default = 1.6 x estimate + 1 h, max 23 h). Follow a detached job with `modal app logs <app-id>` (id printed at launch) or the dashboard.
Re-launching the same command after a crash continues it (markers + `last.pt`); `--force` re-runs steps already marked done.
Order and who runs what: `RUN_MATRIX.md`. A tuned hyper-parameter file written by `pipeline.py tune` lands under
`runs/random/tune_<model>/` - read the chosen values there (or in the log) and pass them with `--env` (the other engineer owns the `tune`/`hp()` code: confirm the env names it reads).

## 7. Pull results and hand them back to the report build

```
python app.py pull                                  # prints the commands below
modal volume ls  tokamark-results runs/random
modal volume get tokamark-results runs/random/cnn_mlp_mask_s1 ../results/modal/random/cnn_mlp_mask_s1
modal volume get tokamark-results costlog ../results/modal/costlog
modal volume get tokamark-data work/out ../results/modal/contract       # keep: selection.json is the experiment's identity
```
Per run you get `out/<split>/{run_config.json,train_summary.json,summary.md,results/,figures/,model/}` (the `model/` dir holds `best_model.pt`; skip `last.pt`
with `--include`-style copies if size matters) and for robustness `out/<split>/robustness/{robust_metrics.json,robustness.md,physics/...}`.
Tell the report engineer: (1) the folder `cnn_mlp/results/modal/...`, (2) each run's `<model>_<variant>_s<seed>` tag, (3) `costlog/` for the compute table,
(4) the Kaggle runs' equivalents (same tag convention), so one script can aggregate mean +/- std per (model, variant) across both engines.

## 8. Kaggle side (the other engine)

Per notebook: copy the same `selection.json` + `tar_index.json` into `out/` (and `out/<split>/`), set `TM_MODEL`, `TM_VARIANT` (or the FAILURE_AWARE/PINN pair),
`TM_SEED`, `TM_RUN_TAG`, `TM_USE_DP=0`, and train the 2 T4s as two independent runs only after you measured that two concurrent processes do not slow each other
(4 vCPU shared). To skip the 2-hour HF re-download per session: `modal volume get tokamark-data tars/random.tar .` and upload it as a Kaggle dataset
(single file, so no file-count problem; check Kaggle's current dataset size limit before relying on it), then `tar -xf` it onto `/kaggle/temp` at session start.

## 9. Cost control and safety

- Modal dashboard shows live spend; `modal run app.py::status` prints the total logged by this app. Stop a runaway job: `modal app stop <app-id>`.
- Hard budget per job (`--hours`) + `timeout`; at most 3 in-container attempts x 3 containers per input. Worst-case cost of one job is printed by `--dry-run`/`plan`.
- Nothing here commits tokens: `.modal.toml` lives in your home directory; `HF_TOKEN` stays in your shell environment. Do not paste either into notebooks, issues or the report.
- Volumes are not backups. Pull `costlog/`, finished run folders and `work/out` after each batch. Check the Modal pricing page for any Volume storage charge
  (not verified here) and delete `tars/temporal.tar` when you no longer need it: `modal volume rm tokamark-data tars/temporal.tar`.
- The data volume is mounted read-only in GPU jobs, so a buggy training run cannot corrupt the shared shots.
