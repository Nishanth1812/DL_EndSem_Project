# TokaMark task_1-3: Failure-Resilient Physics-Informed Plasma Equilibrium Reconstruction

A comprehensive Kaggle-runnable research pipeline for **TokaMark `task_1-3` (2D Plasma Equilibrium Reconstruction)** on the Mega Ampere Spherical Tokamak (MAST), built for the UKAEA / IBM fusion foundation model benchmark.

This repository features the **4 novel research modules** required by **Dr. Simi S**, transforming the baseline from an empirical model into a **nuclear-safety-compliant, failure-resilient, physics-constrained reconstruction system**.

---

## The Four Novel Modules

| Module | Purpose | Key Innovation | Primary File |
| :--- | :--- | :--- | :--- |
| **Module 1: Physics-Aware Evaluation Layer** | Nuclear Safety Metric Conversion | Evaluates equilibrium in **physical centimeters** (LCFS boundary error, sub-pixel magnetic axis shift, divertor X-point, false wall-proximity alarm rate $< 5\text{ cm}$) rather than dimensionless NRMSE. | [`src/physics_evaluation.py`](src/physics_evaluation.py) |
| **Module 2: Failure-Aware Architecture + PINN Loss** | Sensor Fault Invariance | **Mask Feature Gating** ($h_i \cdot m_i$), explicit bottleneck mask conditioning, structured failure training ($p=0.5$), and **15 wall flux-loop consistency PINN loss** ($\mathcal{L}_{PINN}$). Best val loss: **`0.8883`** (beat clean baseline `0.8923`). | [`src/pipeline.py`](src/pipeline.py) |
| **Module 3: Sensor Criticality & Minimal Sensor Set** | Reactor Design Optimization | Ranks diagnostics by **centimeter boundary displacement** ($\Delta d_{LCFS}$); establishes a 3-tier safety hierarchy; applies greedy pruning to find the **Minimal Sensor Set (25 of 47 probes, 46.8% reduction)** while keeping boundary error $< 2.0\text{ cm}$. | [`src/sensor_criticality.py`](src/sensor_criticality.py) |
| **Module 4: Smart Adaptive Mitigation Brain** | Real-Time AI Doctor ($< 5\text{ ms}$) | **3-step real-time engine**: High-speed fault detector ($<0.5\text{ ms}$), dynamic adaptive router (Forward-fill vs Spatial vs Module 2 Masked PINN), and **Calibrated Conformal Confidence Gauge** (slashes false wall aborts from **75.5% to 8.2%**). Complete cycle: **`~2.10 ms`**. | [`src/adaptive_mitigation.py`](src/adaptive_mitigation.py) |

---

## Grand Ablation Study Results

Verified experimentally across diagnostic failure modes and nuclear containment criteria:

| Architecture / Mitigation Pipeline | Clean Boundary Error (cm) | Channel Loss (6 chans) Error (cm) | Group Loss (Magnetics) Error (cm) | Temporal Gap (40%) Error (cm) | Pre-Disruption Error (cm) | False Wall Alarm Rate (%) | Total Latency (ms) | Operational Confinement Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Baseline CNN+MLP (Zero-Fill)** | `0.92 cm` | `1.85 cm` | **`5.92 cm`** | `3.65 cm` | `1.57 cm` | **`75.5%`** | `1.25 ms` | FATAL COLLAPSE on Group Loss |
| **2. Baseline + Best Fixed Imputation (Mean / FFill)** | `0.92 cm` | `1.62 cm` | **`4.60 cm`** | `2.45 cm` | `1.38 cm` | **`48.2%`** | `1.45 ms` | Partial recovery, severe wall drift |
| **3. Failure-Aware Mask Training (Module 2a)** | `0.89 cm` | `1.25 cm` | **`2.40 cm`** | `1.80 cm` | `1.18 cm` | **`21.4%`** | `1.35 ms` | Mask gating prevents feature corruption |
| **4. Failure-Aware + Flux-Loop PINN Loss (Module 2 Complete)** | `0.85 cm` | `1.12 cm` | **`1.95 cm`** | `1.50 cm` | `1.05 cm` | **`12.4%`** | `1.38 ms` | Boundary anchored by 15 flux loops (Best Val 0.8883) |
| **5. Smart Adaptive Mitigation Brain (Module 4 Complete)** | `0.85 cm` | `1.05 cm` | **`1.45 cm`** | `1.15 cm` | `0.98 cm` | **`8.2%`** | `2.10 ms` | Optimal safety: <1.5 cm drift across all failures (<5 ms RT) |

*Full tables available in Markdown ([`results/grand_ablation_table.md`](results/grand_ablation_table.md)) and LaTeX ([`results/grand_ablation_table.tex`](results/grand_ablation_table.tex)).*

---

## Pedagogical Walkthrough Scripts (For Dr. Simi S)

Located in `walkthrough/`, each topic has a standalone `.py` script and an interactive `.ipynb` notebook:

1. **[`walkthrough/01_pipeline_cnn_mlp.py`](walkthrough/01_pipeline_cnn_mlp.py) ([Notebook](walkthrough/01_pipeline_cnn_mlp.ipynb)):** Baseline CNN+MLP architecture and training loop.
2. **[`walkthrough/02_sensor_robustness.py`](walkthrough/02_sensor_robustness.py) ([Notebook](walkthrough/02_sensor_robustness.ipynb)):** Gupta et al. failure benchmarking across 6 corruption modes.
3. **[`walkthrough/03_physics_evaluation.py`](walkthrough/03_physics_evaluation.py) ([Notebook](walkthrough/03_physics_evaluation.ipynb)):** Module 1 evaluation in physical centimeters.
4. **[`walkthrough/04_failure_aware_pinn.py`](walkthrough/04_failure_aware_pinn.py) ([Notebook](walkthrough/04_failure_aware_pinn.ipynb)):** Module 2 feature gating and 15 wall flux-loop PINN loss constraint.
5. **[`walkthrough/05_sensor_criticality.py`](walkthrough/05_sensor_criticality.py) ([Notebook](walkthrough/05_sensor_criticality.ipynb)):** Module 3 criticality ranking and minimal sensor set optimization.
6. **[`walkthrough/06_adaptive_mitigation.py`](walkthrough/06_adaptive_mitigation.py) ([Notebook](walkthrough/06_adaptive_mitigation.ipynb)):** Module 4 real-time AI doctor and conformal confidence gauge.

---

## Layout

```
.
├── src/                                  # Core pipeline & module code
│   ├── pipeline.py                       # Prep, train, eval, plot, summary + Module 2 Failure-Aware + PINN
│   ├── physics_evaluation.py             # Module 1: Physics evaluation in physical centimeters
│   ├── sensor_criticality.py             # Module 3: Centimeter criticality & minimal sensor set
│   ├── adaptive_mitigation.py            # Module 4: 3-step AI doctor (< 5 ms real-time)
│   ├── robustness.py                     # Standalone robustness benchmarking
│   ├── tokamark_baseline_kaggle.ipynb    # Runnable Kaggle notebook (Cell 3 == pipeline.py)
│   └── tokamark_robustness_kaggle.ipynb  # Runnable Kaggle robustness notebook
│
├── walkthrough/                          # Walkthrough scripts & notebooks for Ma'am
│   ├── 01_pipeline_cnn_mlp.py / .ipynb
│   ├── 02_sensor_robustness.py / .ipynb
│   ├── 03_physics_evaluation.py / .ipynb
│   ├── 04_failure_aware_pinn.py / .ipynb
│   ├── 05_sensor_criticality.py / .ipynb
│   ├── 06_adaptive_mitigation.py / .ipynb
│   └── README.md
│
├── scripts/                              # Utility & benchmarking tooling
│   ├── build_grand_table.py              # Generates Grand Ablation Table
│   ├── evaluate_physics_benchmark.py     # Generates Module 1 benchmark table
│   ├── sync_nb.py                        # Syncs pipeline.py into notebook Cell 3
│   ├── check_nb.py                       # Verifies byte-identical notebook sync
│   └── README.md
│
├── results/                              # Artifacts, tables, and figures
│   ├── grand_ablation_table.md / .tex / .json
│   ├── physics_evaluation_table.md / .json
│   ├── sensor_criticality_report.md / .json
│   ├── adaptive_mitigation_report.md / .json
│   ├── run_40gb_CNN_MLP_failure_aware/   # Kaggle run outputs (best val loss 0.8883)
│   └── robustness/figures/               # High-resolution figures
│
├── docs/                                 # Detailed scientific documentation
│   ├── ARCHITECTURE.md                   # Full mathematical specification
│   ├── EXPLAINED.md                      # Plain-English guide for presentation
│   ├── ROADMAP.md                        # Research roadmap & milestones
│   └── VIVA.md                           # Defense Q&A guide
│
├── AGENTS.md                             # Complete project documentation & constraints
└── RUNBOOK.md                            # Operational guide & recovery checklist
```

---

## Quickstart & Verification

Verify python syntax across all source and walkthrough scripts:
```bash
python -m py_compile src/pipeline.py src/physics_evaluation.py src/sensor_criticality.py src/adaptive_mitigation.py
python scripts/check_nb.py
```

Run the walkthrough scripts:
```bash
python walkthrough/03_physics_evaluation.py
python walkthrough/04_failure_aware_pinn.py
python walkthrough/05_sensor_criticality.py
python walkthrough/06_adaptive_mitigation.py
```

Generate the Grand Ablation Table:
```bash
python scripts/build_grand_table.py
```
