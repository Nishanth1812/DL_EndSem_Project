import json, io, os

NB_PATH = "src/tokamark_robustness_kaggle.ipynb"
PY_PATH = "src/pipeline.py"
ROB_PATH = "src/robustness.py"

nb = json.load(io.open(NB_PATH, encoding="utf-8"))

# 1. Sync Cell 2 (pipeline.py)
marker_p = "%%writefile /kaggle/working/pipeline.py\n"
disk_p = io.open(PY_PATH, encoding="utf-8").read().replace("\r\n", "\n").replace("\r", "\n")
body_p = marker_p + disk_p
lines_p = body_p.splitlines(keepends=True)
if lines_p and not lines_p[-1].endswith("\n"):
    lines_p[-1] += "\n"
nb["cells"][2]["source"] = lines_p

# 2. Sync Cell 3 (robustness.py)
marker_r = "%%writefile /kaggle/working/robustness.py\n"
disk_r = io.open(ROB_PATH, encoding="utf-8").read().replace("\r\n", "\n").replace("\r", "\n")
body_r = marker_r + disk_r
lines_r = body_r.splitlines(keepends=True)
if lines_r and not lines_r[-1].endswith("\n"):
    lines_r[-1] += "\n"
nb["cells"][3]["source"] = lines_r

# 3. Update Cell 5 to use TM_PREP_SPLITS=test
nb["cells"][5]["source"] = [
    'run(f"cd /kaggle/working && TM_PREP_SPLITS=test python -u pipeline.py prep --split {DATA_SPLIT}")\n'
]

json.dump(nb, io.open(NB_PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("Updated tokamark_robustness_kaggle.ipynb: Cell 2, Cell 3, Cell 5.")
