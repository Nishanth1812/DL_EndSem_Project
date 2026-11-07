#!/usr/bin/env python
"""Instructor's results table: rows = pipeline stages, columns = Clean / Channel loss / Group loss / Temporal gap /
Pre-disruption / Latency; every cell "boundary error cm (NRMSE)", mean +/- std over seeds when more than one.

    python build_ablation_table.py --out results_v2/ablation \
        --plain DIR [DIR...] --mask DIR [DIR...] --mask_pinn DIR [DIR...] \
        [--mitigation MIT.json [MIT.json...]]

Each DIR holds a robust_metrics.json (one DIR = one seed). Any of --plain/--mask/--mask_pinn may be omitted (rows are
skipped). Writes <out>.md, <out>.tex, <out>.csv. Cells that are not available print "--".
Boundary error = PHYS.boundary_mean_cm (EFIT-truth Module 1 v2, mean over scored windows; a prediction with no contour at
the LCFS flux counts BOUNDARY_FAIL_CM = 50 cm). NRMSE = official hierarchical NRMSE.

Mitigation JSON (written by the adaptive-mitigation study, schema in HARNESS_V2.md):
  {"adaptive": {"<scenario>": {"boundary_mean_cm": x, "nrmse": y}, ..., "latency_ms": z}, "oracle": {...same...}}
  scenario in clean | channel_loss_6 | group_loss | temporal_gap_40 | pre_disruption_25; x, y numbers or {"mean","std","n"}.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_seeds import aggregate, _ms  # noqa: E402

# column -> (cell id in robust_metrics.json, key in the mitigation JSON)
COLUMNS = [
    ("Clean", "clean", "clean"),
    ("Channel loss (6)", "channel_ablation|6|{imp}", "channel_loss_6"),
    ("Group loss (magnetics)", "group_failure|magnetics_all|{imp}", "group_loss"),
    ("Temporal gap (40%)", "temporal_gap|0.4|{imp}", "temporal_gap_40"),
    ("Pre-disruption (25%)", "disruption_proximate|0.25|{imp}", "pre_disruption_25"),
]
MISSING = "--"


def fmt(b, n, tex=False):
    """b, n: {mean,std,n} or None."""
    if b is None and n is None:
        return MISSING

    def one(x, d):
        if x is None:
            return MISSING
        s = f"{x['mean']:.{d}f}"
        if x["n"] > 1:
            s += (r" $\pm$ " if tex else " ± ") + f"{x['std']:.{d}f}"
        return s
    return f"{one(b, 2)} ({one(n, 3)})"


def _pick(agg, col_cell, imp):
    cid = col_cell.format(imp=imp)
    c = agg["cells"].get(cid)
    return (c["boundary_mean_cm"], c["NRMSE"]) if c else (None, None)


def stage_row(label, agg, imps):
    """Cells of one training variant. `imps`: ("zero",) or ("mean","ffill") -> best (lowest mean NRMSE) per column."""
    cells = []
    for _, cid, _k in COLUMNS:
        cand = [_pick(agg, cid, i) for i in imps]
        cand = [c for c in cand if c[1] is not None]
        cells.append(min(cand, key=lambda c: c[1]["mean"]) if cand else (None, None))
    return label, cells


def _num(v):
    if v is None:
        return None
    return v if isinstance(v, dict) and "mean" in v else dict(mean=float(v), std=0.0, n=1)


def mitigation_rows(paths):
    out = {}
    docs = [json.load(open(p)) for p in paths]
    for name in ("adaptive", "oracle"):
        docs_n = [d[name] for d in docs if name in d]
        if not docs_n:
            continue
        cells = []
        for _, _c, key in COLUMNS:
            bs = [_num(d.get(key, {}).get("boundary_mean_cm")) for d in docs_n]
            ns = [_num(d.get(key, {}).get("nrmse")) for d in docs_n]
            cells.append((_ms([b["mean"] for b in bs if b]), _ms([n["mean"] for n in ns if n])))
        lat = _ms([_num(d.get("latency_ms"))["mean"] for d in docs_n if d.get("latency_ms") is not None])
        out[name] = (cells, lat)
    return out


def build(variants, mit_paths):
    """variants: {"plain": [dirs], "mask": [dirs], "mask_pinn": [dirs]} -> list of (label, cells, latency, n_seeds)."""
    rows = []
    A = {k: aggregate(v) for k, v in variants.items() if v}
    if "plain" in A:
        rows.append(("baseline: plain CNN, zero-fill", *stage_row("", A["plain"], ("zero",))[1:], None, A["plain"]["n_seeds"]))
        rows.append(("+ best fixed imputation (mean/ffill)", *stage_row("", A["plain"], ("mean", "ffill"))[1:], None, A["plain"]["n_seeds"]))
    if "mask" in A:
        rows.append(("+ failure-aware training (mask + augmentation)", *stage_row("", A["mask"], ("zero",))[1:], None, A["mask"]["n_seeds"]))
    if "mask_pinn" in A:
        rows.append(("+ flux-loop consistency loss (mask + PINN)", *stage_row("", A["mask_pinn"], ("zero",))[1:], None, A["mask_pinn"]["n_seeds"]))
    for k, lab in (("mask", "(supp.) mask-only + best fixed imputation"), ("mask_pinn", "(supp.) mask+PINN + best fixed imputation")):
        if k in A:
            rows.append((lab, *stage_row("", A[k], ("mean", "ffill"))[1:], None, A[k]["n_seeds"]))
    mit = mitigation_rows(mit_paths) if mit_paths else {}
    for name, lab in (("adaptive", "+ adaptive mitigation"), ("oracle", "Oracle mitigation")):
        if name in mit:
            cells, lat = mit[name]
            rows.append((lab, cells, lat, len(mit_paths)))
        else:
            rows.append((lab, [(None, None)] * len(COLUMNS), None, 0))
    return rows


def write(rows, out):
    head = ["Configuration"] + [c[0] for c in COLUMNS] + ["Latency (ms)", "seeds"]
    body = []
    for lab, cells, lat, ns in rows:
        body.append([lab] + [fmt(b, n) for b, n in cells] + [(f"{lat['mean']:.2f}" if lat else MISSING), str(ns) if ns else MISSING])
    md = ["Each cell: boundary error in cm (NRMSE). mean ± std over seeds when more than one.", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in body]
    open(out + ".md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    with open(out + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config"] + [f"{c[0]} boundary_cm_mean,boundary_cm_std,nrmse_mean,nrmse_std" for c in COLUMNS] + ["latency_ms", "seeds"])
        for lab, cells, lat, ns in rows:
            flat = []
            for b, n in cells:
                flat.append(" ".join(f"{x:.5g}" if x is not None else "" for x in
                                     ((b or {}).get("mean"), (b or {}).get("std"), (n or {}).get("mean"), (n or {}).get("std"))))
            w.writerow([lab] + flat + [f"{lat['mean']:.4g}" if lat else "", ns])
    tex = [r"\begin{tabular}{l" + "c" * (len(head) - 1) + "}", r"\hline", " & ".join(head).replace("%", r"\%") + r" \\", r"\hline"]
    for lab, cells, lat, ns in rows:
        tex.append(" & ".join([lab.replace("%", r"\%").replace("&", r"\&")] + [fmt(b, n, tex=True) for b, n in cells] +
                              [(f"{lat['mean']:.2f}" if lat else MISSING), str(ns) if ns else MISSING]) + r" \\")
    tex += [r"\hline", r"\end{tabular}"]
    open(out + ".tex", "w", encoding="utf-8").write("\n".join(tex) + "\n")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    for k in ("plain", "mask", "mask_pinn"):
        ap.add_argument("--" + k, nargs="*", default=[])
    ap.add_argument("--mitigation", nargs="*", default=[])
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(write(build(dict(plain=a.plain, mask=a.mask, mask_pinn=a.mask_pinn), a.mitigation), a.out))


if __name__ == "__main__":
    main()
