#!/usr/bin/env python
"""Local CPU driver for Module 3 (sensor criticality) and Module 4 (adaptive mitigation) on a REAL trained checkpoint.

    python run_module34_local.py --model mlp_fa_v3 --stage cache      # stream train/val/test shots once -> npz window cache
    python run_module34_local.py --model mlp_fa_v3 --stage m3         # Module 3 (Shapley, worst-k, minimum set, Green's)
    python run_module34_local.py --model mlp_fa_v3 --stage m4         # Module 4 (detector, router, Oracle, conformal, latency)
    python run_module34_local.py --model mlp_fa_v3 --stage all

Environment: torch (CPU) + numpy<2 + zarr 2: pass --pylib "<pylib3>;<pylib>;<pylib2>" (the scratchpad setup).
Shots: every <data-root>/random/<shot>.zarr listed in the run's selection.json (train / val / test); downloading is not done here
(see STUDY_V2.md for the fetch script). Nothing here trains anything.  Outputs: cnn_mlp/results_v2/module34/<model>/.
"""
import argparse
import json
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(SRC))
RES = os.path.join(REPO, "cnn_mlp", "results_v2")
# tag -> (checkpoint dir that holds model/best_model.pt, backbone, runs/ dir with the tokamark + tokamark_baseline checkouts).
# All checkpoints use the same shot selection (selection.json identical), so one window cache serves every model.
_TK = "mlp_fa_robust_v3"
MODELS = {
    "mlp_fa_v3": (os.path.join(REPO, "runs", "mlp_fa_robust_v3", "out", "random"), "mlp", _TK),
    "lstm_fa": (os.path.join(REPO, "runs", "lstm_fa_robust", "out", "random"), "lstm", "lstm_fa_robust"),
    "transformer_mask_s1": (os.path.join(RES, "kaggle_transformer_mask_s1", "out", "random", "transformer_mask_s1"), "transformer", _TK),
    "transformer_mask_pinn_s1": (os.path.join(RES, "modal_transformer_maskpinn_s1", "transformer_mask_pinn_s1"), "transformer", _TK),
    "cnn_lstm_mask_s1": (os.path.join(RES, "modal_cnn_lstm_mask_s1", "cnn_lstm_mask_s1", "out", "random", "cnn_lstm_mask_s1"), "lstm", _TK),
    "cnn_mlp_mask_pinn_s1": (os.path.join(RES, "modal_cnn_mlp_maskpinn_s1", "cnn_mlp_mask_pinn_s1"), "mlp", _TK),
}


def t():
    return time.strftime("%H:%M:%S")


def stub_s3fs():
    try:
        import s3fs  # noqa: F401
    except ImportError:  # local zarr only: the s3 backend is never contacted
        import fsspec
        m = types.ModuleType("s3fs")
        m.__version__ = "2024.1.0"

        class S3FileSystem(fsspec.AbstractFileSystem):
            protocol = "s3"

            def __init__(self, *a, **k):
                super().__init__()
        m.S3FileSystem = S3FileSystem
        sys.modules["s3fs"] = m


def stub_torchinfo():
    try:
        import torchinfo  # noqa: F401
    except ImportError:  # only used by the upstream model files for a printed summary
        m = types.ModuleType("torchinfo")
        m.summary = lambda *a, **k: None
        sys.modules["torchinfo"] = m


def setup(a):
    """Set the TM_* contract, sys.path and import the repo pipeline. Returns (pipeline module, run_dir, data_dir)."""
    ckpt_dir, backbone, tk_name = MODELS[a.model]
    run_dir = os.path.join(REPO, "runs", tk_name)
    work = os.path.abspath(a.work_dir)
    data_dir = os.path.join(os.path.abspath(a.data_root), "random")
    sel = json.load(open(os.path.join(REPO, "runs", "mlp_fa_robust_v3", "out", "random", "selection.json")))
    have = {int(f[:-5]) for f in os.listdir(data_dir) if f.endswith(".zarr") and os.path.isdir(os.path.join(data_dir, f))}
    shots = {k: [s for s in sel["shots"][k] if int(s) in have] for k in ("train", "val", "test")}
    out = dict(sel)
    out["shots"] = shots
    p = os.path.join(work, "out", "random")
    os.makedirs(p, exist_ok=True)
    json.dump(out, open(os.path.join(p, "selection.json"), "w"))
    os.environ.update(TM_TOKAMARK=os.path.join(run_dir, "tokamark"), TM_BASELINE=os.path.join(run_dir, "tokamark_baseline"),
                      TM_WORK=work, TM_DATA_ROOT=os.path.abspath(a.data_root), TM_DL_WORKERS="0", TM_BATCH="32",
                      TM_MODEL=backbone, TM_FAILURE_AWARE="1", TM_USE_DP="0")
    os.environ.pop("TM_VARIANT", None)
    os.environ.pop("TM_RUN_TAG", None)
    for q in reversed([q for q in a.pylib.split(os.pathsep) if q]):
        sys.path.insert(0, q)
    sys.path.insert(0, SRC)
    sys.path.insert(0, HERE)
    stub_s3fs()
    stub_torchinfo()
    import pipeline as P
    return P, ckpt_dir, data_dir, shots


def cache_split(P, split, out_path, stride, max_per_shot, shots):
    import numpy as np
    import torch  # noqa: F401
    from physics_study import WindowSet
    if os.path.exists(out_path):
        print(t(), "cache exists", out_path, flush=True)
        return
    # P.build_data reads TM_WORK/out/random/selection.json (trimmed to the shots on disk)
    loaders, meta = P.build_data("random", [split], shuffle_windows=False)
    ld = loaders[split]
    sid, wid, xs, ys = [], [], [None] * 8, []
    seen, t0, per_shot = 0, time.time(), {}
    for batch in ld:
        if batch is None:
            continue
        s, w, x, y = batch
        for j in range(len(s)):
            seen += 1
            key = int(s[j])
            if (int(w[j]) % stride) != 0 or per_shot.get(key, 0) >= max_per_shot:
                continue
            per_shot[key] = per_shot.get(key, 0) + 1
            sid.append(key)
            wid.append(int(w[j]))
            for b in range(8):
                xs[b] = (xs[b] or []) + [np.asarray(x[b][j].tolist(), dtype=np.float32)]
            ys.append(np.asarray(y[0][j].tolist(), dtype=np.float32).reshape(65, 65))
        if (seen // 32) % 10 == 0:
            print(t(), f"{split}: seen {seen}, kept {len(sid)}, shots {len(per_shot)}, {seen / (time.time() - t0):.1f} win/s", flush=True)
    ws = WindowSet(np.array(sid), np.array(wid), [np.stack(v) for v in xs], np.stack(ys))
    ws.save(out_path)
    print(t(), f"{split}: saved {len(ws)} windows from {len(per_shot)} shots in {time.time() - t0:.0f}s ->", out_path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp_fa_v3", choices=list(MODELS))
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--work-dir", required=True, help="scratch TM_WORK (trimmed selection.json) + window cache; NOT in the repo")
    ap.add_argument("--stage", default="all", choices=["cache", "m3", "m4", "lat", "all"])
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--stride", type=int, default=2, help="keep every k-th window index")
    ap.add_argument("--max-per-shot", type=int, default=60)
    ap.add_argument("--pylib", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--extra", default="", help="stage arguments as key=value,key=value (see STUDY_V2.md)")
    a = ap.parse_args()
    P, ckpt_dir, data_dir, shots = setup(a)
    import numpy as np  # noqa: F401
    print(t(), "pipeline", P.__file__, "| shots on disk", {k: len(v) for k, v in shots.items()}, flush=True)
    cache = os.path.join(os.path.abspath(a.work_dir), "cache")
    os.makedirs(cache, exist_ok=True)
    if a.stage in ("cache", "all"):
        for sp in a.splits.split(","):
            cache_split(P, sp, os.path.join(cache, f"{sp}.npz"), a.stride, a.max_per_shot, shots[sp])
    if a.stage in ("m3", "m4", "lat", "all"):
        import physics_study as S
        out = os.path.abspath(a.out_dir) if a.out_dir else os.path.join(REPO, "cnn_mlp", "results_v2", "module34", a.model)
        os.makedirs(out, exist_ok=True)
        kw = dict(kv.split("=", 1) for kv in a.extra.split(",") if kv)
        ctx = S.make_context(P, ckpt_dir, data_dir, cache, a.model, **kw)
        if a.stage in ("m3", "all"):
            S.run_criticality(ctx, out, **kw)
        if a.stage in ("m4", "all"):
            S.run_mitigation(ctx, out, **kw)
        if a.stage == "lat":
            S.run_latency_only(ctx, out, **kw)
        ctx.scorer.close()


if __name__ == "__main__":
    main()
