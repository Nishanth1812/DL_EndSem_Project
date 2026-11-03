#!/usr/bin/env python
"""Module 3: Physics-Grounded Sensor Criticality & Minimal Sensor Set.

  1. Centimeter criticality: how far the plasma boundary (LCFS) moves, in cm, when a sensor branch is lost.
  2. Three safety tiers on the boundary error with that sensor lost:
        Tier 1 safety-critical  > 2.0 cm      Tier 2 plasma shaping  1.0 - 2.0 cm      Tier 3 redundant  < 1.0 cm
  3. Minimal sensor set: greedy backward elimination, dropping the sensor whose loss hurts least while the boundary
     error stays within the 2.0 cm tolerance.
  4. Spatial error-propagation heatmaps: where on the 65 x 65 grid each lost sensor puts its error.

The numbers come from the trained model, not from this file: `mlp/physics_study.py criticality` ablates every sensor
branch of the model on test windows, scores the result with Module 1 and writes <run>/physics/criticality.json.
This module holds the tier rule and turns that file into a report and figures.

    python modules/sensor_criticality.py --run runs/lstm_fa/out/random/robustness
"""
import argparse
import json
import os

import numpy as np

TIER1_CM, TIER2_CM = 2.0, 1.0
TIER_NAMES = {1: "Tier 1: Safety-Critical", 2: "Tier 2: Plasma Shaping", 3: "Tier 3: Auxiliary / Redundant"}


def tier_of(boundary_error_cm):
    """Tier from the boundary error (cm) with that sensor lost."""
    return 1 if boundary_error_cm > TIER1_CM else 2 if boundary_error_cm >= TIER2_CM else 3


def render_report(data, out_dir):
    """Write criticality_report.md and the two figures next to criticality.json; returns the report path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    sig = sorted(data["signals"], key=lambda s: -s["shift_cm"])
    el, ms = data["elimination"], data["minimal_set"]
    lines = [
        "# Module 3: Physics-Grounded Sensor Criticality & Minimal Sensor Set", "",
        f"**Model:** CNN+{data['model'].upper()} ({'failure-aware' if data['failure_aware'] else 'baseline'}), "
        f"{data['n_windows']} test windows | **Clean boundary error:** `{data['clean']['boundary_cm']:.2f} cm` | "
        f"**Sensors:** {data['total_probes']} probes in {len(sig)} branches | **Tolerance:** `{data['threshold_cm']:.2f} cm`", "",
        "## 1. Criticality ranking (one branch lost at a time, zero-filled)", "",
        "| Rank | Branch | Probes | Boundary error (cm) | Shift vs clean (cm) | Axis shift (cm) | X-point (cm) | Tier |",
        "| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |"]
    for i, s in enumerate(sig, 1):
        lines.append(f"| {i} | **{s['short']}** (`{s['signal']}`) | {s['n_probes']} | `{s['boundary_cm']:.2f}` | "
                     f"**`{s['shift_cm']:+.2f}`** | `{s['axis_shift_cm']:.2f}` | `{s['xpoint_err_cm']:.2f}` | {TIER_NAMES[s['tier']]} |")
    lines += ["", "## 2. Minimal sensor set (greedy backward elimination)", "",
              f"- **Kept:** {', '.join(f'`{k}`' for k in ms['kept'])} ({ms['probes']} of {data['total_probes']} probes, "
              f"{100 * (1 - ms['probes'] / data['total_probes']):.1f}% fewer)",
              f"- **Dropped:** {', '.join(f'`{k}`' for k in ms['dropped']) or 'none'}",
              f"- **Boundary error at the minimal set:** `{ms['boundary_cm']:.2f} cm` (tolerance `{data['threshold_cm']:.2f} cm`)", "",
              "| Step | Removed | Remaining probes | Boundary error (cm) | Within tolerance |", "| :-- | :-- | :-- | :-- | :-- |"]
    for e in el:
        lines.append(f"| {e['step']} | {e['removed'] or 'none (full set)'} | {e['remaining_probes']} | "
                     f"`{e['boundary_cm']:.2f}` | {'yes' if e['safe'] else 'NO'} |")
    lines += ["", "## 3. Figures", "", "- `minimal_sensor_set_curve.png`, `spatial_error_propagation_heatmaps.png`"]
    report = os.path.join(out_dir, "criticality_report.md")
    open(report, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot([e["remaining_probes"] for e in el], [e["boundary_cm"] for e in el], "o-", c="steelblue")
    ax.axhline(data["threshold_cm"], c="r", ls="--", label=f"{data['threshold_cm']:.1f} cm tolerance")
    ax.invert_xaxis()
    ax.set_xlabel("probes remaining"); ax.set_ylabel("boundary error (cm)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Greedy backward elimination")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "minimal_sensor_set_curve.png"), dpi=150); plt.close(fig)

    maps = [s for s in sig if s.get("spatial_abs_err")]
    if maps:
        cols = min(4, len(maps)); rows = -(-len(maps) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
        vmax = max(np.max(m["spatial_abs_err"]) for m in maps)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, s in zip(axes.ravel(), maps):
            ax.imshow(np.array(s["spatial_abs_err"]), cmap="magma", vmin=0, vmax=vmax)
            ax.set_title(f"{s['short']} lost", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "spatial_error_propagation_heatmaps.png"), dpi=150); plt.close(fig)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/robustness", help="robustness folder holding physics/criticality.json")
    a = ap.parse_args()
    path = os.path.join(a.run, "physics", "criticality.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}: run `python mlp/physics_study.py criticality` on a trained model first")
    print("report:", render_report(json.load(open(path)), os.path.join(a.run, "physics")))
