"""Write src/pipeline.py into the notebook's %%writefile cell (cell 3), keeping them in sync.

Run from anywhere:  python scripts/sync_nb.py
"""
import json, io, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "src", "tokamark_baseline_kaggle.ipynb")
PY = os.path.join(ROOT, "src", "pipeline.py")
marker = "%%writefile /kaggle/working/pipeline.py\n"

nb = json.load(io.open(NB, encoding="utf-8"))
disk = io.open(PY, encoding="utf-8").read().replace("\r\n", "\n").replace("\r", "\n")

cell3 = nb["cells"][2]
assert "".join(cell3["source"]).startswith(marker), "cell3 is not the writefile cell"

body = marker + disk
lines = body.splitlines(keepends=True)
if lines and not lines[-1].endswith("\n"):
    lines[-1] += "\n"
cell3["source"] = lines

json.dump(nb, io.open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

nb2 = json.load(io.open(NB, encoding="utf-8"))
src = "".join(nb2["cells"][2]["source"])
b = src[len(marker):]
print("notebook JSON valid,", len(nb2["cells"]), "cells")
print("cell3 body == pipeline.py :", "BYTE-IDENTICAL" if b == disk else "STILL MISMATCH")
