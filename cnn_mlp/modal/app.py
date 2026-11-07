"""Modal runner for the TokaMark task_1-3 study (prep / smoke / tune / train / eval / robustness).

Local, no network, no Modal account needed:
    python app.py plan train --model cnn_mlp --variant mask --seed 1 --split random --gpu T4
Real runs (see RUNBOOK_MODAL.md):
    modal run app.py::status
    modal run app.py::prep --split random
    modal run --detach app.py::train --model cnn_mlp --variant mask --seed 1 --split random
Every entrypoint also takes --dry-run (prints the plan, launches nothing; note `modal run` still builds the image once).

Design (all choices explained in RUNBOOK_MODAL.md):
 - Shots live on volume `tokamark-data` as ONE tar per split (a zarr store is hundreds of thousands of tiny files;
   Modal Volumes v1 cap at 500,000 inodes). A job untars it to the container's local disk (fast I/O for the DataLoader).
 - Every run gets a private TM_WORK=/results/runs/<split>/<tag> on volume `tokamark-results`, seeded with the shared
   selection.json + tar_index.json (the contract). The pipeline resumes from last.pt inside it after a preemption/retry.
 - src/ is mounted with add_local_dir(copy=False): editing pipeline.py never rebuilds the image.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

APP_NAME = "tokamark-study"
GH = "https://github.com/UKAEA-IBM-STFC-Fusion-FMs"
REVS = (("tokamark", "1a200f4"), ("tokamark_baseline", "1e7c489"))  # = git_revs in run_config.json
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"                  # cnn_mlp/src: pipeline.py + study/
APP_SRC = "/opt/study_src"                 # NOT "/app/src": the baseline repo has its own top-level `src` package
DATA_LOCAL = "/tmp/tokamark_data"
RUNS = "/results/runs"
TORCH, TORCHVISION = "2.6.0", "0.21.0"     # ASSUMPTION: Kaggle's torch version was not recorded in the repo; any 2.x works
PIN = ["zarr==3.1.5", "fsspec==2025.3.0", "s3fs==2025.3.0", "torchinfo==1.8.0", "line-profiler==5.0.2"]  # as notebooks / READMEs

# ---- prices: https://modal.com/pricing (fetched 2026-11-04), USD per second -----------------------------------------
GPU_S = {"T4": 0.000164, "L4": 0.000222, "A10G": 0.000306, "L40S": 0.000542, "A100-40GB": 0.000583,
         "A100-80GB": 0.000694, "H100": 0.001097}
CPU_CORE_S, MEM_GIB_S = 0.0000131, 0.00000222


def usd_per_hour(gpu, cpu, mem_gib):
    return 3600 * ((GPU_S[gpu] if gpu else 0.0) + cpu * CPU_CORE_S + mem_gib * MEM_GIB_S)


MODEL_ENV = {"cnn_mlp": "mlp", "cnn_lstm": "lstm", "transformer": "transformer"}
VARIANT_ENV = {"plain": {"TM_FAILURE_AWARE": "0", "TM_PINN_LAMBDA": "0.0"},
               "mask": {"TM_FAILURE_AWARE": "1", "TM_PINN_LAMBDA": "0.0"},
               "mask_pinn": {"TM_FAILURE_AWARE": "1", "TM_PINN_LAMBDA": "0.1"}}
# wall-hours of one run on a T4 for the plan printout; run_matrix.py is the documented source (measured / extrapolated)
EST_H = {"cnn_mlp": {"plain": 2.83, "mask": 3.25, "mask_pinn": 3.42}, "cnn_lstm": {"plain": 2.85, "mask": 3.25, "mask_pinn": 3.42},
         "transformer": {"plain": 3.65, "mask": 4.15, "mask_pinn": 4.35}}
KIND_STEPS = {  # (step name, script relative to APP_SRC, args)
    "train": [("train", "pipeline.py", ["train"]), ("eval", "pipeline.py", ["eval"]),
              ("plot", "pipeline.py", ["plot"]), ("summary", "pipeline.py", ["summary"])],
    "eval": [("eval", "pipeline.py", ["eval"]), ("plot", "pipeline.py", ["plot"]), ("summary", "pipeline.py", ["summary"])],
    "robustness": [("eval-robust", "study/pipeline_robustness.py", ["eval-robust"]),
                   ("report", "study/pipeline_robustness.py", ["report"]),
                   ("physics", "study/physics_study.py", ["all"])],
    "tune": [("tune", "pipeline.py", ["tune"])],
}

# ---- image -----------------------------------------------------------------------------------------------------------
_constraints = "\\n".join([f"torch=={TORCH}", f"torchvision=={TORCHVISION}"] + PIN)
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")          # libgl1/libglib: opencv (a tokamark dependency) imports them
    .run_commands(f"pip install torch=={TORCH} torchvision=={TORCHVISION}")   # PyPI wheel bundles the CUDA 12.4 runtime
    .run_commands(f"printf '{_constraints}\\n' > /opt/constraints.txt")
    .run_commands(*[f"git clone -q {GH}/{n}.git /opt/{n} && git -C /opt/{n} checkout -q {sha} && git -C /opt/{n} log -1 --format='{n} @ %h'"
                    for n, sha in REVS])
    # tokamark's pyproject pulls tensordict, opencv, sklearn, optuna, ...; the constraint file stops pip upgrading torch/zarr
    .run_commands("pip install -c /opt/constraints.txt -e /opt/tokamark huggingface_hub pyyaml scipy pandas matplotlib "
                  "contourpy requests filelock xarray " + " ".join(PIN))
    .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": f"{APP_SRC}/study:{APP_SRC}",   # study/ first: its modules are the newer ones
          "TM_TOKAMARK": "/opt/tokamark", "TM_BASELINE": "/opt/tokamark_baseline",
          "TM_GIT_REVS": ",".join(f"{n}@{s}" for n, s in REVS)})
)
image = base_image.add_local_dir(str(SRC), APP_SRC, copy=False, ignore=["**/__pycache__", "**/*.pyc", "**/*.ipynb"])
test_image = base_image.pip_install("pytest").add_local_dir(str(SRC), APP_SRC, copy=False,
                                                          ignore=["**/__pycache__", "**/*.pyc", "**/*.ipynb"])
slim = modal.Image.debian_slim(python_version="3.11")

app = modal.App(APP_NAME)
data_vol = modal.Volume.from_name("tokamark-data", create_if_missing=True)
res_vol = modal.Volume.from_name("tokamark-results", create_if_missing=True)
# Optional HF token (higher download rate limit). Read from YOUR shell when you launch; stored only as a Modal Secret object.
SECRETS = [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})] if os.environ.get("HF_TOKEN") else []


# ---- helpers ---------------------------------------------------------------------------------------------------------
def _norm_gpu(g):
    g = {"A100": "A100-40GB", "A10": "A10G"}.get(g, g)
    if g not in GPU_S:
        raise SystemExit(f"unknown --gpu {g!r}; choose from {sorted(GPU_S)}")
    return g


def _kv(s):
    out = {}
    for item in [x for x in (s or "").split(",") if x.strip()]:
        k, _, v = item.partition("=")
        if not k.strip().startswith("TM_") or not _:
            raise SystemExit(f"--env expects TM_X=y,TM_Z=w (got {item!r})")
        out[k.strip()] = v.strip()
    return out


def make_spec(kind, model="cnn_mlp", variant="plain", seed=1, split="random", gpu="T4", hours=0.0, env="",
              force=False, skip_eval=False, cpu=4.0, memory_gib=16, smoke=False, tag=""):
    if model not in MODEL_ENV or variant not in VARIANT_ENV or split not in ("random", "temporal"):
        raise SystemExit("model in %s, variant in %s, split in random|temporal" % (list(MODEL_ENV), list(VARIANT_ENV)))
    gpu = _norm_gpu(gpu)
    tag = tag or f"{model}_{variant}_s{seed}"
    est = 0.3 if smoke else EST_H[model][variant]
    if kind == "tune":
        tag, est = f"tune_{model}", 1.5
    if kind in ("eval", "robustness"):
        est = 1.5
    budget = hours or (0.5 if smoke else min(est * 1.6 + 1.0, 23.0))   # hard budget, hours
    e = {"TM_MODEL": MODEL_ENV[model], "TM_VARIANT": variant, "TM_SEED": str(seed), "TM_RUN_TAG": tag,
         "TM_USE_DP": "0", "TM_CKPT_EVERY_S": "300", "TM_DL_WORKERS": "-1", **VARIANT_ENV[variant]}
    if kind == "tune":
        e["TM_PINN_LAMBDA"], e["TM_FAILURE_AWARE"] = "0.0", "0"          # tune on `plain`, reuse for all variants
    if smoke:
        e.update(TM_MAX_STEPS="30", TM_VALIDATE_EVERY="10", TM_PATIENCE="2", TM_BATCH="16", TM_DATA_GB="2",
                 TM_PREP_DEADLINE_H="0.3", TM_CKPT_EVERY_S="60", TM_PLOT_WINDOWS="1")
    e.update(_kv(env))                                                     # explicit overrides win (TM_LR=..., TM_DROPOUT=...)
    return dict(kind=kind, model=model, variant=variant, seed=seed, split=split, tag=tag, gpu=gpu, budget_h=float(budget),
                est_h=float(est), env=e, force=force, skip_eval=skip_eval, cpu=float(cpu), mem_gib=int(memory_gib), smoke=smoke)


def plan_text(spec):
    r = usd_per_hour(spec["gpu"], spec["cpu"], spec["mem_gib"])
    steps = KIND_STEPS[spec["kind"]]
    work = f"{RUNS}/{spec['split']}/{spec['tag']}"
    if spec["smoke"]:
        steps, work = [("prep", "pipeline.py", ["prep"])] + KIND_STEPS["train"][:2], "/tmp/smoke/work (isolated, nothing saved)"
    lines = [f"PLAN {spec['kind']}{' (smoke)' if spec['smoke'] else ''}: {spec['tag']}  split={spec['split']}  gpu={spec['gpu']}  "
             f"cpu={spec['cpu']}  mem={spec['mem_gib']} GiB",
             f"  TM_WORK={work}   data: " + ("2 GB fresh prep into /tmp" if spec["smoke"] else f"/data/tars/{spec['split']}.tar -> {DATA_LOCAL}/{spec['split']}"),
             f"  steps: " + " -> ".join(f"python {s[1]} {' '.join(s[2])} --split {spec['split']}" for s in steps),
             "  env: " + " ".join(f"{k}={v}" for k, v in sorted(spec["env"].items())),
             f"  estimate: {spec['est_h']:.2f} h (T4 basis, not GPU-scaled)  hard budget: {spec['budget_h']:.1f} h  "
             f"rate: ${r:.3f}/h all-in  ->  est ${spec['est_h'] * r:.2f}, worst case ${spec['budget_h'] * r:.2f}"]
    return "\n".join(lines)


def _sh(cmd, env, cwd=None):
    """Run cmd streaming merged output to the container log; return rc. Terminates the child if we are interrupted."""
    print("$", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
        return p.wait()
    except BaseException:                      # Modal preemption/timeout arrives as an interrupt -> let the child die cleanly
        p.terminate()
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
        raise


def _committer(vol, every_s, stop):
    while not stop.wait(every_s):
        try:
            vol.commit()                        # makes last.pt survive a preemption
        except Exception as e:                  # noqa
            print("commit warning:", e, flush=True)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def _log_cost(spec, t0, status, extra=None):
    wall = time.time() - t0
    rec = dict(ts=datetime.datetime.utcnow().isoformat() + "Z", kind=spec["kind"], tag=spec["tag"], split=spec["split"], gpu=spec["gpu"],
               cpu=spec["cpu"], mem_gib=spec["mem_gib"], wall_s=round(wall, 1), wall_h=round(wall / 3600, 3), status=status,
               est_usd=round(wall / 3600 * usd_per_hour(spec["gpu"], spec["cpu"], spec["mem_gib"]), 3),
               price_basis="modal.com/pricing 2026-11-04 list price, GPU+CPU+memory; excludes image build/volume storage",
               **(extra or {}))
    d = Path("/results/costlog")
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{rec['ts'].replace(':', '')}_{spec['kind']}_{spec['tag']}.json").write_text(json.dumps(rec, indent=1))
    return rec


# ---- GPU/CPU worker --------------------------------------------------------------------------------------------------
@app.cls(image=image, gpu="T4", cpu=4, memory=16 * 1024, timeout=12 * 3600, secrets=SECRETS,
         volumes={"/data": data_vol.read_only(), "/results": res_vol}, retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=30))
class Worker:
    @modal.method()
    def job(self, spec: dict) -> dict:
        t0 = time.time()
        smoke = spec["smoke"]
        work = "/tmp/smoke/work" if smoke else f"{RUNS}/{spec['split']}/{spec['tag']}"
        data_root = "/tmp/smoke/data" if smoke else DATA_LOCAL
        split = spec["split"]
        out = Path(work) / "out"
        (out / split).mkdir(parents=True, exist_ok=True)
        Path(data_root).mkdir(parents=True, exist_ok=True)
        stop = threading.Event()
        threading.Thread(target=_committer, args=(res_vol, 120, stop), daemon=True).start()
        env = dict(os.environ, **spec["env"], TM_WORK=work, TM_DATA_ROOT=data_root)
        try:
            # --- stage the shared contract + data ---
            base = Path("/data/work/out")
            idx = base / "tar_index.json"
            if idx.exists() and not (out / "tar_index.json").exists():
                shutil.copy(idx, out / "tar_index.json")
            steps = [(n, s, a) for n, s, a in KIND_STEPS[spec["kind"]]]
            contract = {}
            if smoke:
                steps = [("prep", "pipeline.py", ["prep"])] + KIND_STEPS["train"][:2]   # isolated 2 GB selection, never saved
            else:
                sel = base / split / "selection.json"
                if not sel.exists():
                    raise SystemExit(f"{sel} missing on volume tokamark-data: run `modal run app.py::prep --split {split}` "
                                     f"(or `modal volume put` your selection.json) first")
                if not (out / split / "selection.json").exists():
                    shutil.copy(sel, out / split / "selection.json")
                elif json.loads(sel.read_text())["shots"] != json.loads((out / split / "selection.json").read_text())["shots"]:
                    raise SystemExit("selection.json in the run dir differs from the shared contract; refusing to mix shot sets")
                contract = dict(selection_sha=_sha(sel))
                tar = f"/data/tars/{split}.tar"
                if not Path(tar).exists():
                    if spec["env"].get("TM_PREP_INLINE") != "1":
                        raise SystemExit(f"{tar} missing: run prep for split {split}")
                    rc = _sh([sys.executable, f"{APP_SRC}/pipeline.py", "prep", "--split", split], env, cwd=APP_SRC)
                    if rc:
                        raise RuntimeError(f"inline data preparation failed: {rc}")
                    if json.loads(sel.read_text())["shots"] != json.loads((out / split / "selection.json").read_text())["shots"]:
                        raise RuntimeError("inline preparation changed the shared shot selection")
                elif not (Path(data_root) / split).is_dir():
                    t = time.time()
                    rc = _sh(["tar", "-C", data_root, "-xf", tar], env)
                    if rc:
                        raise RuntimeError("untar failed")
                    print(f"restored data in {time.time() - t:.0f}s", flush=True)
            print("selection:", {k: len(v) for k, v in json.load(open(out / split / "selection.json")).get("shots", {}).items()}
                  if (out / split / "selection.json").exists() else "none yet", contract, flush=True)
            _sh(["bash", "-c", "nvidia-smi -L; nproc; free -g | head -2; df -h /tmp | tail -1"], env)
            # --- steps, each with a done-marker so a retry/preemption skips finished ones ---
            for name, script, args in steps:
                if spec["skip_eval"] and name in ("eval", "plot", "summary"):
                    continue
                marker = Path(work) / f"_done_{name}"
                if marker.exists() and not spec["force"]:
                    print(f"[skip] {name}: done marker present", flush=True)
                    continue
                for attempt in range(1, 4):                      # in-container retry; the pipeline resumes from last.pt
                    left_h = spec["budget_h"] - (time.time() - t0) / 3600
                    if left_h <= 0.05:
                        raise TimeoutError(f"hard budget {spec['budget_h']} h exhausted before/while running {name}")
                    e2 = dict(env, TM_TRAIN_DEADLINE_H=f"{max(0.1, left_h - (0.0 if spec['skip_eval'] else 0.3)):.2f}")
                    rc = _sh([sys.executable, f"{APP_SRC}/{script}", *args, "--split", split], e2, cwd=f"{APP_SRC}")
                    if rc == 0:
                        marker.write_text(datetime.datetime.utcnow().isoformat())
                        break
                    print(f"[{name}] rc={rc} attempt {attempt}/3", flush=True)
                    time.sleep(20)
                else:
                    raise RuntimeError(f"step {name} failed 3x")
            rec = _log_cost(spec, t0, "ok", contract)
            print("COST", json.dumps(rec), flush=True)
            return rec
        except BaseException as ex:
            _log_cost(spec, t0, f"{type(ex).__name__}", None)
            raise
        finally:
            stop.set()
            res_vol.commit()


# ---- CPU-only functions ----------------------------------------------------------------------------------------------
@app.function(image=image, cpu=4, memory=16 * 1024, timeout=12 * 3600, secrets=SECRETS, volumes={"/data": data_vol})
def prep_remote(split: str, hours: float, data_gb: float, force_reselect: bool, extra_env: dict):
    """Idempotent: restore existing tar + contract, let pipeline.py prep fetch what is missing, tar back, commit."""
    t0 = time.time()
    work, droot = "/tmp/work", DATA_LOCAL
    Path(f"{work}/out/{split}").mkdir(parents=True, exist_ok=True)
    Path(droot).mkdir(parents=True, exist_ok=True)
    vbase = Path("/data/work/out")
    for src, dst in ((vbase / "tar_index.json", Path(work) / "out/tar_index.json"),
                     (vbase / split / "selection.json", Path(work) / f"out/{split}/selection.json")):
        if src.exists():
            shutil.copy(src, dst)
            print("restored", src, flush=True)
    tar = Path(f"/data/tars/{split}.tar")
    env = dict(os.environ, TM_WORK=work, TM_DATA_ROOT=droot, TM_DATA_GB=str(data_gb), TM_PREP_DEADLINE_H=str(hours),
               TM_FORCE_RESELECT="1" if force_reselect else "0", TM_HF_RPS=os.environ.get("TM_HF_RPS", "12"),
               TM_INDEX_MIN_TARS="12", TM_INDEX_THREADS="16", **extra_env)
    if tar.exists():
        t = time.time()
        _sh(["tar", "-C", droot, "-xf", str(tar)], env)
        print(f"restored {tar} in {time.time() - t:.0f}s", flush=True)
    rc = _sh([sys.executable, f"{APP_SRC}/pipeline.py", "prep", "--split", split], env, cwd=APP_SRC)
    # save progress even when prep stopped at its deadline: the next run resumes from it
    if (Path(droot) / split).is_dir():
        Path("/data/tars").mkdir(exist_ok=True)
        tmp = Path(f"/data/tars/{split}.tar.tmp")
        _sh(["tar", "-C", droot, "-cf", str(tmp), split], env)
        os.replace(tmp, tar)
    (vbase / split).mkdir(parents=True, exist_ok=True)
    for name, src in (("tar_index.json", Path(work) / "out/tar_index.json"), (f"{split}/selection.json", Path(work) / f"out/{split}/selection.json")):
        if src.exists():
            shutil.copy(src, vbase / name)
    data_vol.commit()
    sel = vbase / split / "selection.json"
    info = dict(rc=rc, wall_h=round((time.time() - t0) / 3600, 2), tar_gb=round(tar.stat().st_size / 1e9, 2) if tar.exists() else 0,
                shots={k: len(v) for k, v in json.load(open(sel)).get("shots", {}).items()} if sel.exists() else {},
                selection_sha=_sha(sel) if sel.exists() else None,
                est_usd=round((time.time() - t0) / 3600 * usd_per_hour(None, 4, 16), 2))
    print("PREP", json.dumps(info), flush=True)
    if rc:
        raise RuntimeError(f"pipeline prep rc={rc}; partial progress was saved, re-run to resume")
    return info


@app.function(image=image, cpu=1, memory=4096, timeout=1800)
def check_env_remote():
    env = dict(os.environ, TM_WORK="/tmp/w", TM_DATA_ROOT="/tmp/d")
    code = "\n".join([
        "import sys, os, torch, zarr, xarray, fsspec, s3fs, huggingface_hub, torchinfo, scipy, yaml, pandas, matplotlib, contourpy, tensordict",
        "print('torch', torch.__version__, 'cuda build', torch.version.cuda, '| zarr', zarr.__version__, '| fsspec', fsspec.__version__)",
        "sys.path[:0] = ['/opt/tokamark/src', '/opt/tokamark_baseline']; os.chdir('/opt/tokamark_baseline')",
        "from tokamark.tools.path import RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE as RS, TEMPORAL_SPLIT_TOKAMARK_DATA_SPLITS_FILE as TS",
        "from MAST_tools.utils.path_utils import RANDOM_SPLIT_OUTLIER_METADATA_FILE as RO",
        "from tokamark.tasks import get_task_config, get_task_metadata; from tokamark.data import initialize_TokaMark_dataset",
        "import src.trainer, src.multi_conv_mlp_model, src.multi_conv_lstm_model, utils",
        "print('tokamark + baseline imports OK; split csv exists:', os.path.exists(str(RS)), os.path.exists(str(TS)))",
        "try:",
        "    import pipeline, pipeline_robustness; print('our pipeline.py + pipeline_robustness.py import OK')",
        "except Exception as e:",
        "    print('WARNING: our pipeline import failed (is src/ mid-edit?):', repr(e))",
    ])
    rc = _sh([sys.executable, "-c", code], env, cwd=APP_SRC)
    return rc


@app.function(image=test_image, gpu="A10G", cpu=4, memory=16 * 1024, timeout=1800)
def module3_tests_remote():
    env = dict(os.environ, PYTHONPATH=f"{APP_SRC}/study:{APP_SRC}")
    for name in ("test_corruptions.py", "test_greens_sensitivity.py", "test_mitigation_v2.py", "test_module34_study.py",
                 "test_physics_evaluation.py", "test_sensor_attribution.py"):
        rc = _sh([sys.executable, f"{APP_SRC}/study/{name}"], env, cwd=f"{APP_SRC}/study")
        if rc:
            raise RuntimeError(f"{name} failed with exit code {rc}")
    return "Module 3 study, sensor attribution, and Green's sensitivity tests passed on Modal A10G"


@app.function(image=slim, timeout=600, volumes={"/data": data_vol.read_only(), "/results": res_vol.read_only()})
def status_remote():
    lines = []
    for p in sorted(Path("/data/tars").glob("*")) if Path("/data/tars").exists() else []:
        lines.append(f"data   {p.name}: {p.stat().st_size / 1e9:.2f} GB")
    for p in sorted(Path("/data/work/out").rglob("selection.json")) if Path("/data/work/out").exists() else []:
        shots = {k: len(v) for k, v in json.load(open(p)).get("shots", {}).items()}
        lines.append(f"data   {p.relative_to('/data')}: {shots} sha={_sha(p)}")
    runs = Path(RUNS)
    for sp in sorted(runs.glob("*")) if runs.exists() else []:
        for r in sorted(sp.glob("*")):
            done = sorted(m.name[6:] for m in r.glob("_done_*"))
            best = any(r.rglob("best_model.pt"))
            last = any(r.rglob("last.pt"))
            lines.append(f"result {sp.name}/{r.name}: done={done} best_model={best} last.pt={last}")
    tot = 0.0
    for p in sorted(Path("/results/costlog").glob("*.json")) if Path("/results/costlog").exists() else []:
        tot += json.load(open(p)).get("est_usd", 0)
    lines.append(f"estimated spend logged by this app: ${tot:.2f} (list price, excludes image builds/storage)")
    return "\n".join(lines) or "volumes are empty"


# ---- entrypoints -----------------------------------------------------------------------------------------------------
def _run_gpu(spec, dry_run):
    print(plan_text(spec))
    if dry_run:
        print("DRY RUN: nothing launched")
        return
    to = int(min(spec["budget_h"] + 0.5, 23.9) * 3600)
    w = Worker.with_options(gpu=spec["gpu"], cpu=spec["cpu"], memory=spec["mem_gib"] * 1024, timeout=to)()
    print(json.dumps(w.job.remote(spec), indent=1))


@app.local_entrypoint()
def status():
    print(status_remote.remote())


@app.local_entrypoint()
def check_env():
    rc = check_env_remote.remote()
    print("check_env rc =", rc)


@app.local_entrypoint()
def prep(split: str = "random", hours: float = 3.0, data_gb: float = 40.0, force_reselect: bool = False, dry_run: bool = False):
    """Download the byte-budgeted shot subset once into volume tokamark-data (CPU only, resumable)."""
    print(f"PLAN prep split={split} budget={hours} h data_gb={data_gb} force_reselect={force_reselect} "
          f"CPU-only ~${usd_per_hour(None, 4, 16):.2f}/h; temporal split = a SECOND ~40 GB download")
    if force_reselect:
        print("WARNING: --force-reselect changes the shot set; every later run must use the new selection.json")
    if dry_run:
        return
    print(json.dumps(prep_remote.remote(split, hours, data_gb, force_reselect, {}), indent=1))


@app.function(image=slim, timeout=10 * 3600)
def prepare_and_train_lstm():
    """Run both stages remotely so the training handoff survives laptop shutdown."""
    prepared = prep_remote.remote("random", 3.0, 40.0, False, {})
    if prepared["rc"] or not prepared["tar_gb"]:
        raise RuntimeError(f"Data preparation failed: {prepared}")
    spec = make_spec("train", "cnn_lstm", "plain", 1, "random", "A10G",
                     env="TM_DL_WORKERS=0,TM_BATCH=512,TM_VALIDATE_EVERY=100,TM_PATIENCE=10")
    print(plan_text(spec), flush=True)
    worker = Worker.with_options(gpu=spec["gpu"], cpu=spec["cpu"],
                                 memory=spec["mem_gib"] * 1024,
                                 timeout=int(min(spec["budget_h"] + 0.5, 23.9) * 3600))()
    return worker.job.remote(spec)


@app.local_entrypoint()
def detached_lstm():
    print(json.dumps(prepare_and_train_lstm.remote(), indent=1))


@app.function(image=slim, timeout=10 * 3600)
def prepare_and_train(model: str, variant: str, seed: int, gpu: str, env: str):
    """Like prepare_and_train_lstm for any model/variant: restore or rebuild the data tar (CPU), then train (GPU)."""
    prepared = prep_remote.remote("random", 3.0, 40.0, False, {})
    if prepared["rc"] or not prepared["tar_gb"]:
        raise RuntimeError(f"Data preparation failed: {prepared}")
    spec = make_spec("train", model, variant, seed, "random", gpu, env=env)
    print(plan_text(spec), flush=True)
    worker = Worker.with_options(gpu=spec["gpu"], cpu=spec["cpu"], memory=spec["mem_gib"] * 1024,
                                 timeout=int(min(spec["budget_h"] + 0.5, 23.9) * 3600))()
    return worker.job.remote(spec)


@app.local_entrypoint()
def detached_train(model: str = "cnn_lstm", variant: str = "mask", seed: int = 1, gpu: str = "A10G",
                   env: str = "TM_DL_WORKERS=0,TM_BATCH=512,TM_VALIDATE_EVERY=100,TM_PATIENCE=10"):
    """Prepare data if needed, then train, as one remote job that survives the laptop going offline. Use `modal run --detach`."""
    print(json.dumps(prepare_and_train.remote(model, variant, seed, gpu, env), indent=1))


@app.local_entrypoint()
def smoke(model: str = "cnn_mlp", variant: str = "mask_pinn", gpu: str = "T4", dry_run: bool = False):
    """~10-15 min: 2 GB prep into /tmp + 30 train steps + eval; saves nothing but a cost record. Needs no selection.json."""
    spec = make_spec("train", model, variant, 0, "random", gpu, smoke=True, tag=f"smoke_{model}_{variant}")
    _run_gpu(spec, dry_run)


@app.local_entrypoint()
def train(model: str = "cnn_mlp", variant: str = "plain", seed: int = 1, split: str = "random", gpu: str = "T4", hours: float = 0.0,
          env: str = "", skip_eval: bool = False, force: bool = False, cpu: float = 4.0, memory_gib: int = 16, dry_run: bool = False):
    """train + eval + plot + summary. --env TM_LR=3e-4,TM_DROPOUT=0.1 passes tuned values. Use `modal run --detach`."""
    _run_gpu(make_spec("train", model, variant, seed, split, gpu, hours, env, force, skip_eval, cpu, memory_gib), dry_run)


@app.local_entrypoint()
def train_detached(model: str = "cnn_mlp", variant: str = "plain", seed: int = 1, split: str = "random", gpu: str = "T4",
                   hours: float = 0.0, env: str = "", skip_eval: bool = False, force: bool = False, cpu: float = 4.0,
                   memory_gib: int = 16):
    """Fire-and-forget train: spawn the job and return at once, so a laptop network drop cannot stop it.
    Use `modal run --detach app.py::train_detached ...`; watch with `modal app list` and the results volume."""
    spec = make_spec("train", model, variant, seed, split, gpu, hours, env, force, skip_eval, cpu, memory_gib)
    print(plan_text(spec))
    w = Worker.with_options(gpu=spec["gpu"], cpu=spec["cpu"], memory=spec["mem_gib"] * 1024,
                            timeout=int(min(spec["budget_h"] + 0.5, 23.9) * 3600))()
    call = w.job.spawn(spec)
    print("spawned", call.object_id, "- the job runs on Modal independently of this machine")


@app.local_entrypoint()
def evaluate(model: str = "cnn_mlp", variant: str = "plain", seed: int = 1, split: str = "random", gpu: str = "T4", hours: float = 0.0,
             env: str = "", force: bool = False, dry_run: bool = False):
    """Re-run eval/plot/summary on an existing best_model.pt (same tag)."""
    _run_gpu(make_spec("eval", model, variant, seed, split, gpu, hours, env, force), dry_run)


@app.local_entrypoint()
def robustness(model: str = "cnn_mlp", variant: str = "plain", seed: int = 1, split: str = "random", gpu: str = "T4", hours: float = 0.0,
               env: str = "", force: bool = False, dry_run: bool = False):
    """63-pass sensor-failure sweep + report + physics_study on the run's best_model.pt (same tag)."""
    _run_gpu(make_spec("robustness", model, variant, seed, split, gpu, hours, env, force), dry_run)


@app.local_entrypoint()
def tune(model: str = "cnn_mlp", split: str = "random", gpu: str = "T4", hours: float = 0.0, env: str = "", force: bool = False,
         dry_run: bool = False):
    """One-time hyper-parameter search per model (pipeline.py tune), on `plain`; reuse the values for every variant."""
    _run_gpu(make_spec("tune", model, "plain", 7, split, gpu, hours, env, force), dry_run)


@app.local_entrypoint()
def pull():
    print("""Retrieve results (run on your machine, no GPU involved):
  modal volume ls tokamark-results runs/random
  modal volume get tokamark-results runs/random/cnn_mlp_mask_s1 ./modal_results/random/cnn_mlp_mask_s1
  modal volume get tokamark-results costlog ./modal_results/costlog
  modal volume get tokamark-data work/out ./modal_results/contract      # selection.json + tar_index.json (keep a copy!)""")


@app.local_entrypoint()
def module3_tests():
    """Run the existing study test scripts on Modal A10G without training a model."""
    print(module3_tests_remote.remote())


if __name__ == "__main__":      # offline planner: python app.py plan train --model cnn_mlp ...
    ap = argparse.ArgumentParser(description="offline dry-run planner (no Modal account or network needed)")
    ap.add_argument("plan")
    ap.add_argument("kind", choices=["train", "eval", "robustness", "tune", "smoke"])
    ap.add_argument("--model", default="cnn_mlp")
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split", default="random")
    ap.add_argument("--gpu", default="T4")
    ap.add_argument("--hours", type=float, default=0.0)
    ap.add_argument("--env", default="")
    a = ap.parse_args()
    k = "train" if a.kind == "smoke" else a.kind
    print(plan_text(make_spec(k, a.model, a.variant, a.seed, a.split, a.gpu, a.hours, a.env, smoke=a.kind == "smoke",
                              tag=f"smoke_{a.model}_{a.variant}" if a.kind == "smoke" else "")))
