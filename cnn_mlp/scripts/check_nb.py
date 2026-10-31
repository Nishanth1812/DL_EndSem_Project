"""Verify that the notebook's %%writefile cell (cell 3) is byte-identical to src/pipeline.py.

Run from anywhere:  python scripts/check_nb.py
"""
import json, io, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "src", "tokamark_baseline_kaggle.ipynb")
PY = os.path.join(ROOT, "src", "pipeline.py")

nb = json.load(io.open(NB, encoding="utf-8"))
print("notebook JSON: valid,", len(nb["cells"]), "cells")

cell3 = nb["cells"][2]
src = "".join(cell3["source"])
marker = "%%writefile /kaggle/working/pipeline.py\n"
assert src.startswith(marker), "cell3 is not the writefile cell!"
body = src[len(marker):]

disk = io.open(PY, encoding="utf-8").read()
if body == disk:
    print("cell3 body == pipeline.py : BYTE-IDENTICAL")
else:
    print("MISMATCH: cell3 body != pipeline.py")
    print("  cell3 len:", len(body), " pipeline len:", len(disk))
    for i, (a, b) in enumerate(zip(body, disk)):
        if a != b:
            print("  first diff at", i, repr(body[i - 40:i + 40]), "VS", repr(disk[i - 40:i + 40]))
            break

cfg = "".join(nb["cells"][1]["source"])
for key in ("DATA_GB", "DL_WORKERS", "INDEX_MIN_TARS", "TM_INDEX_MIN_TARS"):
    for line in cfg.splitlines():
        if line.strip().startswith(key) or f"{key}=" in line:
            print(f"  cfg: {line.strip()}")
            break
