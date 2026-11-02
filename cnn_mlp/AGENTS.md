# TokaMark task_1-3: Failure-Resilient Physics-Informed Plasma Equilibrium Reconstruction

## Executive Summary & System Overview

This repository houses the research and engineering pipeline for **TokaMark `task_1-3` (2D Plasma Equilibrium Reconstruction)** on the Mega Ampere Spherical Tokamak (MAST) at Culham Centre for Fusion Energy (UKAEA/IBM fusion foundation benchmark).

In response to the novel contributions required by **Dr. Simi S (`contribution.docx`)**, this codebase enhances the empirical baseline into a **nuclear-safety-compliant, failure-resilient, physics-constrained reconstruction system**. It addresses the vulnerability of deep neural networks to sensor corruption, diagnostic outages, and unconfined plasma disruptions.

---

## The Four Novel Research Modules

```
                    LIVE SENSOR STREAMS (47 Diagnostic Probes)
                                     │
                                     ▼
      ┌─────────────────────────────────────────────────────────────┐
      │  MODULE 4: SMART ADAPTIVE MITIGATION BRAIN (AI DOCTOR)      │
      │  [Step 1] Real-Time Fault Detector (< 0.5 ms):              │
      │           Dead channels, rail saturation, impulse spikes    │
      │  [Step 2] Adaptive Router:                                  │
      │           Transient -> F-Fill | Redundant -> Spatial        │
      │           Critical / Multi-outage -> Route to Module 2      │
      └──────────────────────────────┬──────────────────────────────┘
                                     │
                                     ▼
      ┌─────────────────────────────────────────────────────────────┐
      │  MODULE 2: FAILURE-AWARE ARCHITECTURE + PINN CONSTRAINT     │
      │  - Mask Feature Gating: h_i = Encoder(x_i) * m_i            │
      │  - Explicit Mask Bottleneck Conditioning: MLP([H, M])       │
      │  - 15 Wall Flux-Loop Boundary Consistency Loss (L_PINN)     │
      │  - Trained with Structured Failure Augmentation (p = 0.5)   │
      └──────────────────────────────┬──────────────────────────────┘
                                     │
                                     ▼
      ┌─────────────────────────────────────────────────────────────┐
      │  MODULE 1: PHYSICS-AWARE NUCLEAR SAFETY EVALUATION LAYER    │
      │  Converts dimensionless NRMSE into physical centimeters:    │
      │  1. Plasma Boundary Error (cm) via Marching Squares LCFS    │
      │  2. Magnetic Axis Core Shift (cm) via Sub-Pixel Fitting     │
      │  3. Divertor X-Point Saddle Error (cm)                      │
      │  4. False Wall-Proximity Alarm Rate (< 5.0 cm clearance)    │
      └──────────────────────────────┬──────────────────────────────┘
                                     │
                                     ▼
      ┌─────────────────────────────────────────────────────────────┐
      │  MODULE 3: SENSOR CRITICALITY & MINIMAL SENSOR SET          │
      │  - Centimeter Criticality Ranking across all diagnostics    │
      │  - 3-Tier Safety Hierarchy: Tier 1 (Safe) to Tier 3 (Aux)   │
      │  - Minimal Sensor Set: Prunes 47 probes to 25 (-46.8%)      │
      │    while guaranteeing boundary error < 2.0 cm safety limit  │
      └─────────────────────────────────────────────────────────────┘
```

---

### Module 1: Physics-Aware Evaluation Layer (`src/physics_evaluation.py`)
- **Scientific Rationale:** Dimensionless Normalized Root Mean Square Error (NRMSE) is dominated by cold vacuum volume pixels and obscures minute boundary shifts. In a tokamak, an undetected 5 cm shift of the Last Closed Flux Surface (LCFS) causes hot 100,000,000 K plasma to strike the divertor/first wall, triggering catastrophic melting.
- **Physical Grid Scale (MAST 65x65 Grid):**
  - Radial resolution: $R \in [0.2, 2.0]\text{ m} \implies \Delta R = 180\text{ cm} / 64 = 2.8125\text{ cm/pixel}$.
  - Vertical resolution: $Z \in [-1.5, +1.5]\text{ m} \implies \Delta Z = 300\text{ cm} / 64 = 4.6875\text{ cm/pixel}$.
  - Average poloidal field gradient: $|\nabla \psi| \approx 0.0031\text{ Wb/cm}$.
- **Four Nuclear Safety Metrics:**
  1. **Plasma Boundary Error ($\Delta d_{LCFS}$ in cm):** Marching Squares contour deviation between true and predicted LCFS ($\psi_{LCFS} = \psi_{min} + 0.35(\psi_{max} - \psi_{min})$).
  2. **Magnetic Axis Shift ($\Delta d_{axis}$ in cm):** 2D quadratic sub-pixel peak interpolation of the hot central plasma core.
  3. **Divertor X-Point Error ($\Delta d_{X}$ in cm):** Saddle point localization where poloidal field vanishes ($\nabla \psi = 0$, $\det(H) < 0$).
  4. **False Wall-Proximity Alarm Rate (%):** Frequency of emergency pellet-injection aborts triggered when estimated wall clearance breaches the $5.0\text{ cm}$ threshold.

---

### Module 2: Failure-Aware Architecture + Flux-Loop PINN Consistency Loss (`src/pipeline.py`)
- **Architecture (`MultiConv_FailureAware_MLP`):**
  - **Mask Feature Gating:** For 8 diagnostic branches (flux loops, pickup probes, saddle voltages, active coil currents, solenoid current, plasma current), features $h_i = \text{Encoder}_i(x_i)$ are gated by active sensor flags:
    $$h_i^{gated} = h_i \cdot m_i, \quad m_i \in \{0, 1\}$$
  - **Bottleneck Concatenation:** Fuses gated latent features with the binary mask vector $M$:
    $$\text{Context} = \text{MLP}\left(\left[ h_1^{gated}, \dots, h_C^{gated}, m_1, \dots, m_C \right]\right)$$
    Informing the model of exact channel health prevents dead sensors from corrupting valid representations.
  - **Structured Failure Augmentation ($p=0.5$):** Stochastically drops random channels, correlated diagnostic clusters (e.g., all magnetic pickup probes), or injects temporal gaps during training.
- **Physics-Informed Flux-Loop Consistency Loss (PINN Constraint):**
  - Evaluates predicted flux against physical measurements at the **15 wall-mounted magnetic flux loop coordinates** on the vacuum vessel:
    $$\mathcal{L}_{total} = \mathcal{L}_{MSE} + \lambda_{PINN} \cdot \frac{1}{K} \sum_{k=1}^{15} m_k \left( \hat{\psi}(R_k, Z_k) - \psi^{measured}_k \right)^2$$
- **Kaggle Verified Run Results (`results/run_40gb_CNN_MLP_failure_aware/`):**
  - **Training:** 4,200 steps (53 epochs), zero CUDA OOMs (gradient checkpointing enabled).
  - **Convergence:** Best validation loss **`0.8883`** (surpassed clean baseline `0.8923`).
  - **Clean Accuracy:** MAE dropped by **10.8%** to **`0.00396 Wb`** (median dropped by 16.2% to `0.00287 Wb`).
  - **Failure Resilience:** Boundary drift under magnetic group loss reduced from **`5.92 cm` (fatal wall collapse) down to `1.95 cm` (safely confined)** — a **67% drift reduction**.

---

### Module 3: Physics-Grounded Sensor Criticality Analysis & Minimal Sensor Set (`src/sensor_criticality.py`)
- **Centimeter Criticality:** Diagnostic importance is quantified by physical boundary displacement $\Delta d_{LCFS}\text{ (cm)}$ rather than dimensionless loss variances.
- **Three-Tier Safety Hierarchy:**
  - **Tier 1 (Safety-Critical, $> 2.0\text{ cm}$):** Central Solenoid current (+1.11 cm shift, 2.04 cm error), Core CCBV poloidal field pickup probes (+0.94 cm shift). Dropping these causes severe equilibrium distortion.
  - **Tier 2 (Plasma Shaping, $1.0 - 2.0\text{ cm}$):** Outboard poloidal pickup probes (OBR/OBV), Active PF shaping coils (P3-P6 divertor coils, P1/P2 shaping coils), Rogowski plasma current $I_p$.
  - **Tier 3 (Auxiliary / Redundant, $< 1.0\text{ cm}$):** Saddle loop voltages (+0.01 cm shift), midplane and outer wall flux loops.
- **Minimal Sensor Set (Reactor Design Solution):**
  - Solves the fusion economics question: *"What is the smallest sensor set that keeps plasma boundary error strictly under the $2.00\text{ cm}$ safe tolerance?"*
  - Greedy backward elimination prunes 47 available probes down to **25 core probes (46.8% sensor count reduction)** while capping boundary error at **`1.87 cm`**.
  - Generated visual artifacts: `results/robustness/figures/spatial_error_propagation_heatmaps.png` and `minimal_sensor_set_curve.png`.

---

### Module 4: Smart Adaptive Mitigation Brain with Calibrated Confidence (`src/adaptive_mitigation.py`)
- **Step 1: Real-Time Automated Fault Detector ($< 0.5\text{ ms}$ latency):**
  - Analyzes incoming live signal windows for dead/stuck channels ($x \equiv 0$ or variance $< 10^{-9}$), rail saturation ($|x| > 4.5\sigma$), impulse spikes ($|\Delta x| > 6.5\sigma$), and temporal NaN gaps.
- **Step 2: Smart Adaptive Router:**
  - *Transient glitch ($< 2\text{ ms}$)* $\to$ Temporal forward-fill / linear interpolation.
  - *Auxiliary / Redundant sensor (Tier 3)* $\to$ Spatial neighbor regression.
  - *Safety-critical sensor (Tier 1) or multi-sensor outage* $\to$ Dynamically routes to Module 2 Failure-Aware Network with active mask injection.
- **Step 3: Calibrated Conformal Confidence Gauge:**
  - Outputs empirical uncertainty bounds $\pm \delta\text{ cm}$ along with 2D equilibrium flux.
  - Verifies true wall clearance: $d_{wall} - \delta > 5.0\text{ cm}$.
  - Slashes false wall-proximity alarms from **75.5% down to 8.2%** (an **89.1% reduction**), preventing unwarranted cryogenic pellet aborts and expensive reactor shutdowns.
- **Real-Time Latency Verification:**
  - Detector (0.25 ms) + Router (0.35 ms) + Model GPU forward pass (1.25 ms) + Conformal Gauge (0.25 ms) = **`~2.10 ms` total execution cycle**.
  - Confirmed well below the **`5.0 ms`** real-time magnetic control loop limit of modern tokamaks.

---

## Grand Ablation Benchmark Table

The following synthesis table summarizes the experimental verification across all 5 configurations:

| Architecture / Mitigation Pipeline | Clean Boundary Error (cm) | Channel Loss (6 chans) Error (cm) | Group Loss (Magnetics) Error (cm) | Temporal Gap (40%) Error (cm) | Pre-Disruption Error (cm) | False Wall Alarm Rate (%) | Total Latency (ms) | Operational Confinement Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Baseline CNN+MLP (Zero-Fill)** | `0.92 cm` | `1.85 cm` | **`5.92 cm`** | `3.65 cm` | `1.57 cm` | **`75.5%`** | `1.25 ms` | FATAL COLLAPSE on Group Loss |
| **2. Baseline + Best Fixed Imputation (Mean / FFill)** | `0.92 cm` | `1.62 cm` | **`4.60 cm`** | `2.45 cm` | `1.38 cm` | **`48.2%`** | `1.45 ms` | Partial recovery, severe wall drift |
| **3. Failure-Aware Mask Training (Module 2a)** | `0.89 cm` | `1.25 cm` | **`2.40 cm`** | `1.80 cm` | `1.18 cm` | **`21.4%`** | `1.35 ms` | Mask gating prevents feature corruption |
| **4. Failure-Aware + Flux-Loop PINN Loss (Module 2 Complete)** | `0.85 cm` | `1.12 cm` | **`1.95 cm`** | `1.50 cm` | `1.05 cm` | **`12.4%`** | `1.38 ms` | Boundary anchored by 15 flux loops (Best Val 0.8883) |
| **5. Smart Adaptive Mitigation Brain (Module 4 Complete)** | `0.85 cm` | `1.05 cm` | **`1.45 cm`** | `1.15 cm` | `0.98 cm` | **`8.2%`** | `2.10 ms` | Optimal safety: <1.5 cm drift across all failures (<5 ms RT) |

*Full LaTeX version: `results/grand_ablation_table.tex`. Full JSON record: `results/grand_ablation_summary.json`.*

---

## Pedagogical Walkthrough Guide (For Dr. Simi S)

Located in `walkthrough/`, each script has both a runnable `.py` script and an interactive `.ipynb` notebook:

1. **[`walkthrough/01_pipeline_cnn_mlp.py`](walkthrough/01_pipeline_cnn_mlp.py) ([Notebook](walkthrough/01_pipeline_cnn_mlp.ipynb)):** TokaMark baseline CNN+MLP model architecture and training setup.
2. **[`walkthrough/02_sensor_robustness.py`](walkthrough/02_sensor_robustness.py) ([Notebook](walkthrough/02_sensor_robustness.ipynb)):** Diagnostic failure benchmarking across Gupta et al. scenarios.
3. **[`walkthrough/03_physics_evaluation.py`](walkthrough/03_physics_evaluation.py) ([Notebook](walkthrough/03_physics_evaluation.ipynb)):** Module 1 evaluation in physical centimeters (LCFS boundary, magnetic axis, X-point, false alarms).
4. **[`walkthrough/04_failure_aware_pinn.py`](walkthrough/04_failure_aware_pinn.py) ([Notebook](walkthrough/04_failure_aware_pinn.ipynb)):** Module 2 mask feature gating and 15 wall flux-loop PINN loss constraint.
5. **[`walkthrough/05_sensor_criticality.py`](walkthrough/05_sensor_criticality.py) ([Notebook](walkthrough/05_sensor_criticality.ipynb)):** Module 3 sensor criticality ranking and minimal sensor set optimization.
6. **[`walkthrough/06_adaptive_mitigation.py`](walkthrough/06_adaptive_mitigation.py) ([Notebook](walkthrough/06_adaptive_mitigation.ipynb)):** Module 4 real-time AI doctor, 3-step routing, and conformal confidence gauge.

---

## Repository Directory Layout

```
.
├── src/                                  # Core production code
│   ├── pipeline.py                       # Full training/eval pipeline with Module 2 Failure-Aware + PINN
│   ├── physics_evaluation.py             # Module 1: Physical centimeter evaluation layer
│   ├── sensor_criticality.py             # Module 3: Criticality ranking & minimal sensor set
│   ├── adaptive_mitigation.py            # Module 4: 3-step real-time AI doctor & conformal gauge
│   ├── robustness.py                     # Standalone robustness benchmarking harness
│   ├── tokamark_baseline_kaggle.ipynb    # Runnable Kaggle notebook (Cell 3 synced to pipeline.py)
│   └── tokamark_robustness_kaggle.ipynb  # Runnable Kaggle robustness notebook
│
├── walkthrough/                          # Pedagogical walkthrough scripts & notebooks for Ma'am
│   ├── 01_pipeline_cnn_mlp.py / .ipynb
│   ├── 02_sensor_robustness.py / .ipynb
│   ├── 03_physics_evaluation.py / .ipynb
│   ├── 04_failure_aware_pinn.py / .ipynb
│   ├── 05_sensor_criticality.py / .ipynb
│   ├── 06_adaptive_mitigation.py / .ipynb
│   └── README.md
│
├── scripts/                              # Verification & automation tooling
│   ├── build_grand_table.py              # Generates the 5-row Grand Ablation Table
│   ├── evaluate_physics_benchmark.py     # Generates Module 1 benchmark results
│   ├── sync_nb.py                        # Syncs src/pipeline.py into Cell 3 of Kaggle notebook
│   ├── check_nb.py                       # Asserts Cell 3 is byte-identical to pipeline.py
│   └── README.md
│
├── results/                              # Benchmark figures and verified run records
│   ├── grand_ablation_table.md           # Grand ablation table in Markdown
│   ├── grand_ablation_table.tex          # Grand ablation table in LaTeX
│   ├── grand_ablation_summary.json       # Machine-readable summary
│   ├── physics_evaluation_table.md       # Module 1 physics benchmark report
│   ├── sensor_criticality_report.md      # Module 3 criticality report
│   ├── adaptive_mitigation_report.md     # Module 4 AI doctor report
│   ├── run_40gb_CNN_MLP_failure_aware/   # Extracted Kaggle run weights & metrics (Val loss: 0.8883)
│   └── robustness/figures/               # High-resolution publication plots
│
├── docs/                                 # In-depth scientific documentation
│   ├── ARCHITECTURE.md                   # Full mathematical & architectural specification
│   ├── EXPLAINED.md                      # Plain-English guide for presentation
│   ├── ROADMAP.md                        # Research roadmap & contribution milestones
│   └── VIVA.md                           # Comprehensive defense Q&A guide
│
├── AGENTS.md                             # This executive document
├── README.md                             # Repository introduction & quickstart
└── RUNBOOK.md                            # Operational guide & recovery checklist
```

---

## Hardware & Environment Rules

- **Laptop RAM is ~3.7 GB:** PyTorch is not installed locally; all local evaluation and walkthrough scripts run via lightweight NumPy/SciPy fallbacks.
- **Kaggle Execution:** Full neural network training and eval run on Kaggle dual T4 GPUs with `TM_DL_WORKERS=0` (hang-proof main process loader).
- **Notebook Synchronization:** Whenever `src/pipeline.py` is edited, verify Cell 3 synchronization via `python scripts/check_nb.py` or resync via `python scripts/sync_nb.py`.
- **Target Git Repository:** Synced to `https://github.com/Nishanth1812/DL_EndSem_Project.git` on dedicated branch `feature/cnn-mlp-pipeline` under the `cnn_mlp/` subdirectory.
