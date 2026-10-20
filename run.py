"""Train + evaluate one upstream baseline run on task_1-3, then print its NRMSE.

python run.py --model lstm --split random   # --split temporal for the other column
--gpu pins one CUDA device; --timeout (s) stops training early and still evaluates best_model.pt.
"""
import argparse
import os
import shutil
import subprocess
import sys

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["cnn", "lstm"], default="lstm")
ap.add_argument("--split", choices=["random", "temporal"], default="random")
ap.add_argument("--seed", type=int, default=23)
ap.add_argument("--gpu", help="CUDA device index to pin")
ap.add_argument("--timeout", type=int, help="training wall-clock budget in seconds")
a = ap.parse_args()

repo = "external/tokamark_baseline"
splits = f"data/shots_{a.split}.csv"
if not os.path.exists(splits):  # download.py writes it; without it the loader would ask for shots we don't have
    sys.exit(f"missing {splits} - run download.py first")
plan = pd.read_csv(splits, usecols=["train", "val", "test"]).sum().astype(int)
print(f"task_1-3, {a.split} split, {a.model}: train {plan.train}, val {plan.val}, test {plan.test} shots", flush=True)
# upstream builds paths as REPO_ROOT + arg, so "/../../" reaches our configs/ and runs/
args = ["--task", "task_1-3", "--config", "/../../configs/baseline.yaml",
        "--model", a.model, "--split", a.split, "--seed", str(a.seed)]
env = {**os.environ, **({"CUDA_VISIBLE_DEVICES": a.gpu} if a.gpu else {})}
res = f"{repo}/results/{a.split}/{a.model}/seed_{a.seed}"  # upstream hardcodes this; clear it so we never read a stale run
ckpt = f"runs/baseline/{a.split}/{a.model}/task_1-3/seed_{a.seed}"  # training output (config paths.data_output_directory)
for d in (res, ckpt):
    shutil.rmtree(d, ignore_errors=True)


def run(script, timeout=None):
    subprocess.run([sys.executable, f"{repo}/{script}", *args], check=True, env=env, timeout=timeout)


try:
    run("run_training.py", a.timeout)
except subprocess.TimeoutExpired:  # upstream trainer cannot resume, so best_model.pt is all we keep
    print("TRAINING STOPPED AT TIME BUDGET, NOT CONVERGED: evaluating best_model.pt", flush=True)
run("run_evaluation.py")
df = pd.read_csv(f"{res}/task_metrics.csv")
print(df[df.feature_name == "task_1-3"][["n_shots", *[c for c in df if c.startswith("NRMSE")]]].to_string(index=False))
