# Robust 2D Tokamak Equilibrium Reconstruction from Sparse Wall Diagnostics Under Sensor Failure

**Course:** Deep Learning End-Semester Project  
**Team 15:** TokaMark Benchmark Reproduction & Novel Physics-Aware Robustness Research  
**Benchmark:** TokaMark `task_1-3` (UKAEA / IBM Fusion Foundation Models on FAIR-MAST Data)

---

## 1. Project Overview

In magnetic confinement fusion (tokamaks like MAST), superheated plasma ($100,000,000^\circ\text{C}$) must be confined by magnetic fields without touching the vessel walls. Operators require continuous, real-time knowledge of the 2D poloidal magnetic flux map $\psi(R, Z)$ (a $65 \times 65$ grid) reconstructed from 8 sparse diagnostic signals on the vacuum vessel wall.

This repository hosts Team 15's dual-architecture investigation and novel research modules:
1. **Recurrent Track (`CNN + LSTM`):** Evaluates temporal memory for bridging signal dropouts across successive time windows.
2. **Spatial & Physics Track (`CNN + MLP`):** Located in [`cnn_mlp/`](cnn_mlp/), implements spatial channel isolation alongside **4 novel deep learning modules** (physics evaluation in cm, failure-aware mask training, PINN boundary loss, sensor criticality pruning, and adaptive mitigation).

---

## 2. Comparative Model Architecture & Results

Both architectures ingest 8 multi-rate diagnostic signals (flux loops, magnetic pickup coils, saddle loops, poloidal field coil currents, and plasma current $I_p$) sampled across 5 ms windows, mapping them into the continuous 2D equilibrium $\psi(R, Z)$.

| Evaluation Metric | CNN + MLP Track (`cnn_mlp/`) | CNN + LSTM Track (Root) | Physical & Engineering Takeaway |
| :--- | :---: | :---: | :--- |
| **Clean NRMSE** | **0.1419** | 0.1969 | MLP achieves **28% lower error** under clean conditions. |
| **Clean NMAE** | **0.0807** | 0.1118 | Consistent error reduction across all flux surfaces. |
| **Clean Physical MAE** | **0.00444 Wb** | 0.00688 Wb | Lower magnetic flux error across the cross-section. |
| **Sensor Robustness Sweep** | **Won 43 / 63 tasks** | Won 20 / 63 tasks | MLP wins on spatial channel dropouts; LSTM wins on temporal gaps. |
| **Novel Module 2 (PINN + Mask)** | **Val Loss: 0.8883** | — | **Lowest val loss recorded**; 11% physical MAE drop under 50% sensor failure. |

---

## 3. The 4 Novel Deep Learning Modules

To transition beyond baseline reproduction into defense-worthy novel DL research (aligned with supervisor Dr. Simi S's roadmap), the [`cnn_mlp/`](cnn_mlp/) track introduces:

1. **Module 1: Physics-Aware Evaluation Layer:**
   * Replaces computer vision pixel NRMSE (which is masked by 75% cold vacuum) with true physical metrics:
     * **Plasma Boundary Error:** Last Closed Flux Surface (LCFS) contour deviation in **centimeters**.
     * **Magnetic Axis Shift:** Sub-pixel quadratic peak shift in **centimeters**.
     * **X-Point Location Error:** Lower divertor magnetic saddle point ($\nabla \psi = 0$) shift in **centimeters**.
     * **False Wall-Proximity Alarm Rate:** Measuring emergency abort triggers when plasma approaches within $5\text{ cm}$ of the wall.

2. **Module 2: Failure-Aware Architecture & PINN Loss (`cnn_mlp/src/pipeline.py`):**
   * Conditions the network on a diagnostic health mask $M \in \{0, 1\}^8$ with feature gating.
   * Trains under structured sensor dropout (random probes, correlated magnetic groups, temporal gaps).
   * Enforces a Physics-Informed Neural Network (PINN) boundary loss matching measured Webers (Wb) at the 15 wall flux loops.

3. **Module 3: Physics-Grounded Sensor Criticality & Minimal Sensor Set:**
   * Ranks all 40 diagnostics by induced boundary displacement in centimeters.
   * Employs greedy backward pruning to discover the minimal sensor set (e.g. 18 of 40 probes) guaranteeing containment within $2\text{ cm}$.

4. **Module 4: Smart Adaptive Mitigation Brain:**
   * Real-time fault detector ($<0.5\text{ ms}$).
   * Dynamic mitigation router: temporal forward-fill for transient glitches, spatial neighbor interpolation for redundant probes, and routing to Module 2 for critical losses.
   * Outputs calibrated conformal confidence intervals ($\pm \delta\text{ cm}$).

---

## 4. Repository Structure

```
DL_EndSem_Project/
├── README.md                      <-- This team overview
├── ARCHITECTURE.md                <-- Upstream S3 vs. Hugging Face data pipeline documentation
├── kaggle_main.py                 <-- Entry point for Kaggle batch training (LSTM)
├── run.py                         <-- CLI wrapper for upstream baseline training/eval
├── configs/
│   └── baseline.yaml              <-- Model hyperparameters & hang-proof dataloader configuration
├── patches/
│   └── tokamark_baseline.patch    <-- Validation frequency patch
├── download.py                    <-- Hugging Face tar archive downloader & zarr extractor
├── setup.sh                       <-- Kaggle environment initialization script
│
└── cnn_mlp/                       <-- Dedicated CNN-MLP & Novel Research Modules Track
    ├── README.md                  <-- CNN-MLP architecture reference & guide
    ├── src/
    │   ├── pipeline.py            <-- Unified baseline + Module 2 Failure-Aware pipeline
    │   ├── robustness.py          <-- 63-task sensor corruption evaluation harness
    │   ├── tokamark_baseline_kaggle.ipynb   <-- 1-click Kaggle training notebook (Dual T4 GPUs)
    │   └── tokamark_robustness_kaggle.ipynb <-- 1-click Kaggle robustness sweep notebook
    ├── walkthrough/
    │   ├── 01_pipeline_cnn_mlp.ipynb        <-- Interactive training walkthrough
    │   └── 02_sensor_robustness.ipynb       <-- Interactive robustness walkthrough
    ├── docs/
    │   ├── presentation_4_modules.html      <-- 20-slide interactive browser presentation
    │   ├── NOVEL_MODULES_DEEP_DIVE.md       <-- Comprehensive 30 KB technical deep dive
    │   ├── ARCHITECTURE.md                  <-- CNN-MLP tensor shapes and mathematics
    │   ├── EXPLAINED.md                     <-- Plain-English system guide
    │   └── VIVA.md                          <-- Comprehensive defense Q&A guide
    ├── PPT/
    │   ├── Team_15_Final_Review.tex         <-- 38-frame widescreen LaTeX Beamer slide deck
    │   └── figures/                         <-- Publication loss curves & 2D reconstruction plots
    └── scripts/
        ├── check_nb.py                      <-- Validates pipeline.py and notebook synchronization
        ├── check_slides.py                  <-- Beamer slide compilation checker
        └── sync_nb.py                       <-- Keeps Kaggle notebook writefile cell byte-identical
```

---

## 5. How to Run on Kaggle

### Running the CNN+LSTM Track
```bash
python kaggle_main.py
```
*(Uses `configs/baseline.yaml` with `num_workers: 0` to prevent Docker shared-memory freezes).*

### Running the CNN+MLP & Novel Failure-Aware Track
1. Upload `cnn_mlp/src/tokamark_baseline_kaggle.ipynb` to Kaggle.
2. Select Accelerator: **GPU T4 x2**, Internet: **On**.
3. Toggle `FAILURE_AWARE = 1` in Cell 2 to train the novel Failure-Aware + PINN model, or `FAILURE_AWARE = 0` for the baseline.
4. Click **Run All**. Training finishes in ~2.5 hours and automatically saves `results.zip`.
