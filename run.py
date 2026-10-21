"""Train + evaluate one upstream baseline run on task_1-3, then print its NRMSE.

python run.py --model lstm
--gpu pins CUDA devices ("0,1" = both, trained with DataParallel); --timeout (s) stops training early and still evaluates best_model.pt.
"""
import argparse
import os
import shutil
import subprocess
import sys

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["cnn", "lstm"], default="lstm")
ap.add_argument("--seed", type=int, default=23)
ap.add_argument("--gpu", help="CUDA device indices to use, e.g. 0,1")
ap.add_argument("--timeout", type=int, help="training wall-clock budget in seconds")
a = ap.parse_args()

repo = "external/tokamark_baseline"
split = "random"
splits = "data/shots_random.csv"
if not os.path.exists(splits):  # download.py writes it; without it the loader would ask for shots we don't have
    sys.exit(f"missing {splits} - run download.py first")
plan = pd.read_csv(splits, usecols=["train", "val", "test"]).sum().astype(int)
print(f"task_1-3, random split, {a.model}: train {plan.train}, val {plan.val}, test {plan.test} shots", flush=True)
if not (plan > 0).all():
    sys.exit(f"an empty train/val/test set under {splits}; download more batches")
# upstream builds paths as REPO_ROOT + arg, so "/../../" reaches our configs/ and runs/
args = ["--task", "task_1-3", "--config", "/../../configs/baseline.yaml",
        "--model", a.model, "--split", split, "--seed", str(a.seed)]
env = {**os.environ, **({"CUDA_VISIBLE_DEVICES": a.gpu} if a.gpu else {})}
res = f"{repo}/results/{split}/{a.model}/seed_{a.seed}"  # upstream hardcodes this; clear it so we never read a stale run
ckpt = f"runs/baseline/{split}/{a.model}/task_1-3/seed_{a.seed}"  # training output (config paths.data_output_directory)
for d in (res, ckpt):
    shutil.rmtree(d, ignore_errors=True)


def run(script, timeout=None):
    subprocess.run([sys.executable, f"{repo}/{script}", *args], check=True, env=env, timeout=timeout)


try:
    run("run_training.py", a.timeout)
except subprocess.TimeoutExpired:  # upstream trainer cannot resume, so best_model.pt is all we keep
    print("TRAINING STOPPED AT TIME BUDGET, NOT CONVERGED: evaluating best_model.pt", flush=True)
if not os.path.exists(f"{ckpt}/best_model.pt"):  # the time budget ended before the first epoch finished
    sys.exit("no best_model.pt: not one epoch completed in the time budget")
run("run_evaluation.py")
df = pd.read_csv(f"{res}/task_metrics.csv")
print(df[df.feature_name == "task_1-3"][["n_shots", *[c for c in df if c.startswith("NRMSE")]]].to_string(index=False))
