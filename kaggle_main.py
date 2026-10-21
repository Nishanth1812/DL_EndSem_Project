"""Kaggle batch entry: pull a slice of TokaMark, then train the upstream LSTM on the random split.

Notebook settings: accelerator GPU T4, Internet on, the project files attached as a dataset
(kaggle_main.py, setup.sh, download.py, run.py, configs/, patches/). One cell:
    !python /kaggle/input/<dataset>/kaggle_main.py
then Save Version -> Save & Run All. Logs, checkpoints and metrics land in /kaggle/working/out.

Data is pulled from the Hub for DATA_MIN minutes and then training starts with whatever arrived,
so a slow download shrinks the dataset instead of eating the training budget.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
SESSION_S = 12 * 3600 - 90 * 60  # session cap minus a reserve for evaluation + copying outputs
DATA_MIN = 75  # how long to spend on data; training gets everything after that
NEED_GB = 20  # ~1.5 GB per 1000 shots of task_1-3 arrays, plus the 2 GB tar being staged
SRC = Path(__file__).resolve().parent
W = Path("/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp") / "proj"  # not saved as notebook output
OUT = Path("/kaggle/working/out")

W.mkdir(parents=True, exist_ok=True)
free = shutil.disk_usage(W).free / 1e9
if free < NEED_GB:  # fail now, not after hours of downloading
    sys.exit(f"only {free:.0f} GB free under {W}, need ~{NEED_GB} GB")
os.chdir(W)
for f in ("setup.sh", "download.py", "run.py"):
    shutil.copy(SRC / f, f)
shutil.copytree(SRC / "configs", "configs", dirs_exist_ok=True)
shutil.copytree(SRC / "patches", "patches", dirs_exist_ok=True)

print(f"[{(time.time() - T0) / 60:.0f} min] {free:.0f} GB free; installing", flush=True)
subprocess.run(["bash", "setup.sh"], check=True, env={**os.environ, "PIP": f"{sys.executable} -m pip install -q"})
print(f"[{(time.time() - T0) / 60:.0f} min] installed; pulling TokaMark from the Hub "
      f"(budget {DATA_MIN} min, one line per batch and every 50 shots)", flush=True)
subprocess.run([sys.executable, "-u", "download.py", "--minutes", str(DATA_MIN)], check=True)

if not os.path.exists("data/shots_random.csv"):
    sys.exit("no split files landed; see the download log above")
print(f"[{(time.time() - T0) / 60:.0f} min] data ready", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
budget = int(SESSION_S - (time.time() - T0) - 20 * 60)  # 20 min left for evaluation + copying results
proc = subprocess.Popen([sys.executable, "-u", "run.py", "--model", "lstm", "--gpu", "0,1",
                         "--timeout", str(budget)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
with open(OUT / "random.log", "w") as f:
    for line in proc.stdout:
        f.write(line)
        f.flush()
        print(f"[random] {line}", end="", flush=True)
code = proc.wait()

for src, dst in (("runs", "runs"), ("external/tokamark_baseline/results", "results")):
    if os.path.isdir(src):
        shutil.copytree(src, OUT / dst, dirs_exist_ok=True)
(OUT / "splits").mkdir(exist_ok=True)  # the exact shot lists; reuse them via download.py --shots-from
shutil.copy("data/shots_random.csv", OUT / "splits")
print("--- random ---\n" + "".join((OUT / "random.log").read_text().splitlines(True)[-6:]))
sys.exit(code)
