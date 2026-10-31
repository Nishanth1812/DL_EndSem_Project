# CNN + MultiConv_MLP: Equilibrium Reconstruction & Novel Failure-Aware Modules

Part of **Team 15**'s TokaMark `task_1-3` equilibrium-reconstruction project on MAST tokamak data.

This subfolder houses the **CNN + MLP** track and the **4 Novel Research Modules** designed to make 2D plasma equilibrium reconstruction robust against sensor dropouts and physical boundary violations.

---

## 1. Overview & Architecture

While `task_1-3` is Markovian (instantaneous mapping from 8 wall diagnostics to a $65 \times 65$ poloidal flux field $\psi$), standard deep learning models fail catastrophically when sensors drop out or drift in the harsh fusion environment.

* **Spatial MultiConv Encoder:** 8 parallel 1D convolutional branches map diagnostic channels into a unified spatial latent vector.
* **Bottleneck Core:** Spatial MLP bottleneck ($520 \rightarrow 128 \rightarrow 64$).
* **2D Transpose Decoder:** Upsamples latent features into the continuous $65 \times 65$ magnetic flux grid $\psi(R, Z)$.

### Head-to-Head Baseline Comparison (Clean Test Split)
| Metric | CNN + MLP *(This Track)* | CNN + LSTM *(Recurrent Track)* | Advantage |
| :--- | :---: | :---: | :--- |
| **Clean NRMSE** | **0.1419** | 0.1969 | **MLP wins by 28%** |
| **Clean NMAE** | **0.0807** | 0.1118 | **MLP wins by 28%** |
| **Physical MAE** | **0.00444 Wb** | 0.00688 Wb | **MLP wins by 35%** |
| **63 Failure Scenarios** | **Won 43 / 63 tasks** | Won 20 / 63 tasks | **Spatial channel isolation** |

---

## 2. The 4 Novel Deep Learning Modules

To transition from benchmark evaluation into novel AI research, this track implements the four modules mandated in our supervision roadmap:

1. **Module 1: Physics-Aware Evaluation Layer (`docs/NOVEL_MODULES_DEEP_DIVE.md`):**
   * Translates computer vision pixel loss (which is masked by 75% cold vacuum) into true **physical centimeters**:
     * Plasma Boundary Error (LCFS contour distance in cm).
     * Magnetic Axis Shift (sub-pixel quadratic peak shift in cm).
     * Divertor X-Point Saddle Error ($\nabla \psi = 0$, $\det(H) < 0$ in cm).
     * False Wall-Proximity Alarm Rate ($<5\text{ cm}$ threshold).

2. **Module 2: Failure-Aware Architecture & Flux-Loop PINN Loss (`src/pipeline.py`):**
   * **Mask Conditioning:** Network receives an explicit diagnostic health mask $M \in \{0, 1\}^8$ with feature gating and bottleneck concatenation.
   * **Structured Failure Augmentation:** 50% of training mini-batches undergo random probe dropouts, group failures, and temporal gaps.
   * **PINN Boundary Constraint:** Bilinear sampling at the 15 wall flux loop coordinates penalizes boundary deviations directly in Webers (Wb).
   * **Result:** Achieved **best validation loss ever recorded (`0.8883`)** and reduced physical MAE by **11% (`0.00396 Wb`)**.

3. **Module 3: Physics-Grounded Sensor Criticality & Minimal Sensor Set:**
   * Ranks all 40 diagnostics by physical centimeter boundary displacement.
   * Applies greedy backward pruning to discover the minimal sensor set (e.g., 18 of 40 probes) needed to preserve the $2.0\text{ cm}$ safety envelope.

4. **Module 4: Smart Adaptive Mitigation Brain:**
   * Real-time fault detector ($<0.5\text{ ms}$).
   * Adaptive router: temporal forward-fill for transient blips, spatial neighbor interpolation for redundant probes, and routing to Module 2 for critical losses.
   * Calibrated conformal confidence intervals ($\pm \delta\text{ cm}$).

---

## 3. Directory Layout

```
cnn_mlp/
├── README.md                      <-- This reference guide
├── src/
│   ├── pipeline.py                <-- Unified pipeline (Baseline, Failure-Aware, and PINN loss)
│   ├── robustness.py              <-- 63-task sensor corruption evaluation harness
│   ├── tokamark_baseline_kaggle.ipynb   <-- Kaggle training notebook (Dual T4 GPUs)
│   └── tokamark_robustness_kaggle.ipynb <-- Kaggle 63-task robustness sweep notebook
├── walkthrough/
│   ├── 01_pipeline_cnn_mlp.ipynb  <-- Pedagogical training walkthrough notebook
│   ├── 01_pipeline_cnn_mlp.py     <-- Clean standalone training script
│   ├── 02_sensor_robustness.ipynb <-- Pedagogical robustness sweep notebook
│   └── 02_sensor_robustness.py    <-- Clean standalone robustness script
├── docs/
│   ├── presentation_4_modules.html<-- Interactive 20-slide browser walkthrough (open in browser)
│   ├── NOVEL_MODULES_DEEP_DIVE.md <-- 30 KB comprehensive technical reference
│   ├── ARCHITECTURE.md            <-- MultiConv_MLP tensor shapes and mathematics
│   ├── EXPLAINED.md               <-- Plain-English system guide
│   └── VIVA.md                    <-- Comprehensive defense Q&A guide
├── PPT/
│   ├── Team_15_Final_Review.tex   <-- 38-frame LaTeX Beamer slide deck
│   └── figures/                   <-- Publication loss curves, reconstructions, and degradation plots
└── scripts/
    ├── check_nb.py                <-- Verifies pipeline.py and notebook synchronization
    ├── check_slides.py            <-- Beamer slide compilation and overflow checker
    ├── compare_robustness.py      <-- Head-to-head MLP vs. LSTM comparative analyzer
    └── sync_nb.py                 <-- Keeps Kaggle notebook writefile cell byte-identical
```

---

## 4. How to Run on Kaggle

1. Upload `src/tokamark_baseline_kaggle.ipynb` to Kaggle.
2. Select Accelerator: **GPU T4 x2**, Internet: **On**.
3. In Cell 2:
   * To train the **Clean Baseline**: set `FAILURE_AWARE = 0`.
   * To train the **Novel Failure-Aware + PINN Model**: set `FAILURE_AWARE = 1`.
4. Click **Run All**. The notebook will download the budget-limited 40 GB data slice, train the model (~2.5 hours), evaluate clean test shots, and package `results.zip`.
