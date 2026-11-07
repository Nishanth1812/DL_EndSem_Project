#!/usr/bin/env python
"""Local CPU re-evaluation of trained checkpoints with the corrected harness (axis fix + EFIT-truth physics).

    python run_local_eval.py --runs mlp_fa_v3 lstm_fa [DIR ...] | --discover results_v2 | --run DIR
        --data-root DIR --cache-dir DIR --work-dir DIR [--out-root DIR] [--stage cache|sweep|all] [--skip-done]
        [--phys-stride 4] [--batch 64] [--rob-seed 23] [--pylib "A;B"]

A run is a legacy alias (mlp_fa_v3, lstm_fa = the old Kaggle checkpoints in runs/) or ANY directory that contains
model/best_model.pt next to run_config.json (model, failure_aware, variant, dropout, ... are read from it), so new
checkpoints are picked up without code changes (--discover scans a folder). Several runs run as one subprocess each.

Per run (nothing is trained):
  1. cache : streams the official-split test shots found in <data-root>/random through TokaMarkDataset once (batch 64,
             num_workers 0) and stores the collated batches as float32 npz in --cache-dir (shared by every model and pass).
  2. sweep : loads best_model.pt on CPU, uses mean-fill values from the train shots on disk, then runs
             pipeline_robustness.eval_sweep (clean + 63 corrupted passes) on the cached batches. Each finished pass is
             written to <out-root>/<tag>/robust_metrics.json; re-running resumes (TM_ROB_RESUME=1).
  3. compare: old (Kaggle) vs corrected numbers -> compare_old_new.json (legacy aliases only).
Environment: torch (CPU) + numpy<2 + zarr 3 + xarray (the scratchpad `pylib5`; pass --pylib).
"""
import argparse
import glob
import json
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(SRC))                      # .../Deep Learning
FAMILY_ORDER = ["random_dropout", "temporal_gap", "disruption_proximate", "channel_ablation", "group_failure", "category_abl"]


def t():
    return time.strftime("%H:%M:%S")


class CachedLoader:
    """Re-iterable stand-in for the test DataLoader: yields (shot_ids, window_ids, x_list, y_list) from npz files."""

    def __init__(self, d):
        self.files = sorted(glob.glob(os.path.join(d, "b*.npz")))

    def __iter__(self):
        import numpy as np
        import torch
        for f in self.files:
            z = np.load(f)
            nx = len([k for k in z.files if k.startswith("x")])
            yield (torch.as_tensor(z["sid"]), torch.as_tensor(z["wid"]), [torch.from_numpy(z[f"x{i}"]) for i in range(nx)],
                   [torch.from_numpy(z["y0"])])


def stub_s3fs():
    try:
        import s3fs  # noqa: F401
    except ImportError:  # local zarr only: the s3 backend of the dataset code is never contacted
        import fsspec
        m = types.ModuleType("s3fs")
        m.__version__ = "2024.1.0"

        class S3FileSystem(fsspec.AbstractFileSystem):
            protocol = "s3"

            def __init__(self, *a, **k):
                super().__init__()
        m.S3FileSystem = S3FileSystem
        sys.modules["s3fs"] = m


def trimmed_selection(run_dir, data_dir, work_dir, n_train):
    """selection.json restricted to shots that exist on disk (test: all present; train: first n_train present; val: none)."""
    sel = json.load(open(os.path.join(run_dir, "out", "random", "selection.json")))
    have = {int(f[:-5]) for f in os.listdir(data_dir) if f.endswith(".zarr") and os.path.isdir(os.path.join(data_dir, f))}
    test_shots = [s for s in sel["shots"]["test"] if int(s) in have]
    if not test_shots:
        raise SystemExit(f"FATAL: no selected test shots from {run_dir} are present in {data_dir}")
    out = dict(sel)
    out["shots"] = dict(test=test_shots,
                        train=[s for s in sel["shots"]["train"] if int(s) in have][:n_train], val=[])
    p = os.path.join(work_dir, "out", "random")
    os.makedirs(p, exist_ok=True)
    json.dump(out, open(os.path.join(p, "selection.json"), "w"))
    return out


def build_cache(R, cache_dir, max_batches=0):
    marker = os.path.join(cache_dir, "DONE")
    if os.path.exists(marker):
        print(t(), "cache complete:", open(marker).read().strip(), flush=True)
        return
    import numpy as np
    os.makedirs(cache_dir, exist_ok=True)
    for f in glob.glob(os.path.join(cache_dir, "b*.npz")):
        os.remove(f)
    loaders, meta = R.build_data("random", ["test"])
    n_win, t0 = 0, time.time()
    for bi, batch in enumerate(loaders["test"]):
        if batch is None:
            continue
        sid, wid, x, y = batch
        tmp = os.path.join(cache_dir, f"tmp{bi:05d}.npz")
        np.savez(tmp, sid=np.array([int(s) for s in sid]), wid=np.array([int(w) for w in wid]),
                 y0=np.asarray(y[0].tolist(), dtype=np.float32),
                 **{f"x{i}": np.asarray(v.tolist(), dtype=np.float32) for i, v in enumerate(x)})
        os.replace(tmp, os.path.join(cache_dir, f"b{bi:05d}.npz"))
        n_win += len(sid)
        if bi % 5 == 0:
            print(t(), f"cache batch {bi}: {n_win} windows, {n_win / (time.time() - t0):.1f} win/s", flush=True)
        if max_batches and bi + 1 >= max_batches:
            break
    open(marker, "w").write(f"{n_win} windows in {time.time() - t0:.0f}s" + (" (PARTIAL: --max-batches)" if max_batches else ""))
    print(t(), "cache done:", n_win, "windows", flush=True)


def compare_old_new(new_path, old_path, out_path):
    """Per-family worst NRMSE (zero / best-of-imputation) old Kaggle sweep vs corrected local sweep, plus clean numbers."""
    new, old = json.load(open(new_path)), json.load(open(old_path))
    FAM = {"random_dropout": ["0.1", "0.25", "0.5"], "temporal_gap": ["0.2", "0.4", "0.6"], "disruption_proximate": ["0.1", "0.25", "0.5"],
           "channel_ablation": ["1", "3", "6"], "group_failure": ["magnetic_probes", "power_supply", "magnetics_all"],
           "category_abl": ["flux_loop", "pickup_fields", "saddle_voltage", "active_coils", "solenoid", "plasma_current"]}
    out = dict(clean=dict(old=old["clean"]["NRMSE"], new=new["clean"]["NRMSE"], old_n_windows=old["clean"]["n_windows"],
                          new_n_windows=new["clean"]["n_windows"]), families={})
    for fam, pars in FAM.items():
        row = {}
        for imp in ("zero", "mean", "ffill"):
            ks = [f"{fam}|{p}|{imp}" for p in pars]
            if all(k in new["corrupted"] and k in old["corrupted"] for k in ks):
                row[imp] = dict(old_mean_NRMSE=sum(old["corrupted"][k]["NRMSE"] for k in ks) / len(ks),
                                new_mean_NRMSE=sum(new["corrupted"][k]["NRMSE"] for k in ks) / len(ks),
                                old_RS=sum(old["clean"]["NRMSE"] / old["corrupted"][k]["NRMSE"] for k in ks) / len(ks),
                                new_RS=sum(new["clean"]["NRMSE"] / new["corrupted"][k]["NRMSE"] for k in ks) / len(ks))
        out["families"][fam] = row
    json.dump(out, open(out_path, "w"), indent=1)
    return out


LEGACY = {  # alias -> runs/ dir of an OLD Kaggle checkpoint (trained with the legacy flux-loop PINN, pinn_mode=legacy)
    "mlp_fa_v3": "mlp_fa_robust_v3",
    "lstm_fa": "lstm_fa_robust",
}
REF = os.path.join(REPO, "runs", "mlp_fa_robust_v3")  # tokamark + tokamark_baseline checkouts used for every model (identical sources)
REF_SEL = os.path.join(REPO, "runs", "lstm_fa_robust", "out", "random", "selection.json")  # the 60 official test shots (same for all runs)


def resolve_run(spec):
    """spec: legacy alias, or a directory holding (anywhere below it) model/best_model.pt next to run_config.json.
    Returns dict(tag, ckpt, cfg, old_metrics)."""
    if spec in LEGACY:
        d = os.path.join(REPO, "runs", LEGACY[spec])
        cfg = json.load(open(os.path.join(d, "out", "random", "run_config.json")))
        cfg.setdefault("model", "mlp")
        cfg["variant"] = "mask_pinn(legacy-pinn)"
        return dict(tag=spec, ckpt=os.path.join(d, "out", "random", "model", "best_model.pt"), cfg=cfg,
                    old_metrics=os.path.join(d, "out", "random", "robustness", "robust_metrics.json"))
    cks = sorted(glob.glob(os.path.join(spec, "**", "best_model.pt"), recursive=True))
    if len(cks) != 1:
        raise SystemExit(f"{spec}: expected exactly one best_model.pt below it, found {len(cks)}")
    cfgp = os.path.join(os.path.dirname(os.path.dirname(cks[0])), "run_config.json")
    cfg = json.load(open(cfgp))
    cfg.setdefault("model", "mlp")
    return dict(tag=cfg.get("run_tag") or os.path.basename(os.path.normpath(spec)), ckpt=cks[0], cfg=cfg, old_metrics=None)


def discover(root):
    """Every run directory below `root` that holds a best_model.pt (new checkpoints keep arriving: nothing is hard-coded)."""
    out = []
    for ck in sorted(glob.glob(os.path.join(root, "**", "best_model.pt"), recursive=True)):
        mdir = os.path.dirname(ck)
        run = os.path.dirname(mdir)
        if os.path.exists(os.path.join(run, "run_config.json")):
            out.append(run)
    return out


def run_one(a):
    r = resolve_run(a.run)
    cfg, tag = r["cfg"], r["tag"]
    out_dir = os.path.join(a.out_root, tag)
    os.makedirs(out_dir, exist_ok=True)
    work = os.path.abspath(a.work_dir)
    data_dir = os.path.join(os.path.abspath(a.data_root), "random")
    # one trimmed selection for every model: the official 60 test shots that are on disk (+ a few train shots for the mean-fill values)
    sel = json.load(open(REF_SEL))
    have = {int(f[:-5]) for f in os.listdir(data_dir) if f.endswith(".zarr") and os.path.isdir(os.path.join(data_dir, f))}
    sel["shots"] = dict(test=[s for s in sel["shots"]["test"] if int(s) in have],
                        train=[s for s in sel["shots"]["train"] if int(s) in have][:a.train_shots_for_means], val=[])
    os.makedirs(os.path.join(work, "out", "random"), exist_ok=True)
    json.dump(sel, open(os.path.join(work, "out", "random", "selection.json"), "w"))
    n_test = len(sel["shots"]["test"])
    env = dict(TM_TOKAMARK=os.path.join(REF, "tokamark"), TM_BASELINE=os.path.join(REF, "tokamark_baseline"), TM_WORK=work,
               TM_DATA_ROOT=os.path.abspath(a.data_root), TM_DL_WORKERS="0", TM_BATCH=str(a.batch), TM_MODEL=cfg["model"],
               TM_ROB_DIR=out_dir, TM_ROB_RESUME="1", TM_ROB_SEED=str(a.rob_seed), TM_ROB_PHYS_STRIDE=str(a.phys_stride),
               TM_ROB_BATCH=str(a.batch), TM_USE_DP="0", TM_FAILURE_AWARE="1" if cfg.get("failure_aware") else "0")
    for k, e in (("dropout", "TM_DROPOUT"), ("d_model", "TM_D_MODEL"), ("tx_layers", "TM_TX_LAYERS"), ("tx_heads", "TM_TX_HEADS")):
        if cfg.get(k) is not None:
            env[e] = str(cfg[k])
    os.environ.update(env)
    for k in ("TM_VARIANT", "TM_RUN_TAG"):
        os.environ.pop(k, None)
    if a.only:
        os.environ["TM_ROB_SCENARIOS"] = a.only
    for p in reversed([q for q in a.pylib.split(os.pathsep) if q]):  # first entry ends up first on sys.path
        sys.path.insert(0, p)
    sys.path.insert(0, SRC)
    sys.path.insert(0, HERE)
    stub_s3fs()
    import numpy as np
    import torch
    import pipeline_robustness as R
    assert R.TruthIndex is not None, "efit_truth not importable"
    print(t(), f"[{tag}] model={cfg['model']} failure_aware={cfg.get('failure_aware')} variant={cfg.get('variant')} "
          f"pinn_mode={cfg.get('pinn_mode', 'legacy')} | {n_test}/60 test shots on disk | torch {torch.__version__} numpy {np.__version__}", flush=True)

    build_cache(R, a.cache_dir, a.max_batches)
    if a.stage == "cache":
        return
    loader = CachedLoader(a.cache_dir)
    if a.max_batches:
        loader.files = loader.files[:a.max_batches]
    meta = test_meta(R)
    xs0 = next(iter(loader))[2]
    shapes = ([tuple(v.shape[1:]) for v in xs0], [], [(1, 65, 65)])
    # plain CNN/LSTM builders read shapes from loader.dataset (failure-aware / transformer ones take `shapes`): give them a one-window stand-in
    one = dict(shot_id=0, input=[np.zeros(sh, np.float32) for sh in shapes[0]], exogenous=[], y=[np.zeros(shapes[2][0], np.float32)])
    model = R.make_model(types.SimpleNamespace(dataset=[one]), meta, shapes=shapes)
    model.load_state_dict(torch.load(r["ckpt"], map_location="cpu", weights_only=False))
    model.eval()
    print(t(), "loaded", r["ckpt"], "| params", sum(p.numel() for p in model.parameters()), flush=True)
    json.dump(dict(tag=tag, ckpt=r["ckpt"], run_config=cfg, n_test_shots_on_disk=n_test, cache=a.cache_dir,
                   pinn_label="legacy flux-loop PINN (pixels vs true psi)" if cfg.get("pinn_mode", "legacy") == "legacy" and (cfg.get("pinn_lambda") or 0) > 0
                   else f"pinn_mode={cfg.get('pinn_mode', 'none')}"), open(os.path.join(out_dir, "run_info.json"), "w"), indent=1)

    tm_path = os.path.join(a.out_root, "train_means.json")  # same train shots for every model: compute once
    if os.path.exists(tm_path):
        train_means = {int(k): v for k, v in json.load(open(tm_path)).items()}
    else:
        tl, _ = R.build_data("random", ["train"])
        train_means = R._train_means_pass(tl["train"], max_batches=40)
        json.dump({str(k): float(v) for k, v in train_means.items()}, open(tm_path, "w"))

    rc = R.rob_cfg()
    P_ = R.rob_paths("random")
    R.cmd_corrupt("random")
    print(t(), "sweep:", len(R.build_sweep_tasks(rc)), "passes, phys_stride", rc["phys_stride"], flush=True)
    R.eval_sweep(loader, model, torch.device("cpu"), meta, "random", rc, P_, train_means=train_means)
    if r["old_metrics"] and os.path.exists(r["old_metrics"]) and not a.only:
        res = compare_old_new(P_["metrics"], r["old_metrics"], os.path.join(out_dir, "compare_old_new.json"))
        print(t(), "clean NRMSE old/new:", res["clean"], flush=True)
    try:
        R.cmd_report("random")
    except Exception as e:  # noqa  (figures are optional)
        print(t(), "report skipped:", repr(e)[:150], flush=True)
    print(t(), f"[{tag}] DONE", flush=True)


def test_meta(R):
    """Task metadata with the split's standardisation stats (what build_data computes before touching any zarr)."""
    R.P.setup_imports()
    from tokamark.tasks import get_task_config, get_task_metadata
    _, stats, _o = R.P.split_files("random")
    R.P.patch_stats_module(stats)
    cfgt = get_task_config(task_name=R.CFG["task"])
    cfgt["stride_window"] = 0.005
    return get_task_metadata(config_task=cfgt, verbose=False) | cfgt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="", help="one run: legacy alias (%s) or a run directory" % "|".join(LEGACY))
    ap.add_argument("--runs", nargs="*", default=[], help="several runs, evaluated one subprocess each (crash-isolated, resumable)")
    ap.add_argument("--discover", default="", help="evaluate every run dir below this folder that has best_model.pt + run_config.json")
    ap.add_argument("--skip-done", action="store_true", help="skip runs whose out dir already has a DONE marker")
    ap.add_argument("--data-root", required=True, help="dir holding random/<shot>.zarr")
    ap.add_argument("--cache-dir", required=True, help="OUTSIDE the repo (batch cache, ~120 MB)")
    ap.add_argument("--work-dir", required=True, help="scratch TM_WORK (trimmed selection.json); NOT inside the repo")
    ap.add_argument("--out-root", default=os.path.join(REPO, "cnn_mlp", "results_v2", "corrected_robustness", "sweeps"))
    ap.add_argument("--stage", default="all", choices=["cache", "sweep", "all"])
    ap.add_argument("--phys-stride", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rob-seed", type=int, default=23)
    ap.add_argument("--train-shots-for-means", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=0, help="debug: cache/evaluate only the first N batches")
    ap.add_argument("--only", default="", help="debug: comma list of scenario families")
    ap.add_argument("--pylib", default="", help="dirs (os.pathsep-separated) prepended to sys.path: numpy 1.26 + zarr 3 + xarray + torchinfo")
    a = ap.parse_args()
    specs = ([a.run] if a.run else []) + list(a.runs) + (discover(a.discover) if a.discover else [])
    if len(specs) > 1 or (a.discover and specs):
        import subprocess
        fixed = []  # re-use our own argv minus the run selection flags
        argv = sys.argv[1:]
        i = 0
        while i < len(argv):
            if argv[i] in ("--run", "--discover"):
                i += 2
                continue
            if argv[i] == "--runs":
                i += 1
                while i < len(argv) and not argv[i].startswith("--"):
                    i += 1
                continue
            fixed.append(argv[i])
            i += 1
        for s in specs:
            tag = resolve_run(s)["tag"]
            if a.skip_done and os.path.exists(os.path.join(a.out_root, tag, "DONE")):
                print(t(), "skip (done):", tag, flush=True)
                continue
            rc = subprocess.call([sys.executable, os.path.abspath(__file__), "--run", s] + fixed)
            print(t(), f"run {tag} exit code {rc}", flush=True)
        return
    if not specs:
        raise SystemExit("give --run, --runs or --discover")
    a.run = specs[0]
    run_one(a)
    tag = resolve_run(a.run)["tag"]
    if a.stage != "cache" and not a.only and not a.max_batches:
        open(os.path.join(a.out_root, tag, "DONE"), "w").write(t())


if __name__ == "__main__":
    main()
