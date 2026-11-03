# Pedagogical Walkthrough Guide for TokaMark task_1-3

This directory contains standalone, self-contained walkthrough scripts (`.py`) and interactive Jupyter notebooks (`.ipynb`) designed to explain the entire pipeline, all 4 novel research modules, and the experimental results to supervisors and reviewers.

Each walkthrough runs independently without requiring local GPU or heavy dependencies (contains lightweight NumPy/SciPy demonstrations locally, and full PyTorch execution when run on Kaggle/Colab/GPU).

---

## Walkthrough Index

### [01. CNN+MLP Baseline Architecture](01_pipeline_cnn_mlp.py) ([Notebook](01_pipeline_cnn_mlp.ipynb))
- **Topic:** Official TokaMark task_1-3 baseline model architecture.
- **Key Concepts:**
  - Independent 1D Conv encoders for 8 diagnostic sensor branches (47 physical probes).
  - 32-dimensional Grad-Shafranov latent bottleneck.
  - 2D Transpose-Conv spatial decoder reconstructing the 65x65 poloidal flux map $\psi(R, Z)$.
  - Supervised MSE training loop with gradient clipping.

### [02. Sensor Robustness & Failure Scenarios](02_sensor_robustness.py) ([Notebook](02_sensor_robustness.ipynb))
- **Topic:** Gupta et al. (arXiv 2607.11915) diagnostic failure benchmark.
- **Key Concepts:**
  - 6 failure modes: Random dropout, channel ablation (1, 3, 6 chans), category ablation, temporal gaps (20%-60%), correlated group failure (magnetics / power supply), and disruption proximity.
  - Imputation baselines: Naive zero-fill, mean imputation (train-split stats), and temporal forward-fill.
  - Catastrophic collapse of baseline models under magnetics outages.

### [03. Module 1: Physics-Aware Evaluation Layer](03_physics_evaluation.py) ([Notebook](03_physics_evaluation.ipynb))
- **Topic:** Physical centimeter metrics vs. dimensionless NRMSE.
- **Key Concepts:**
  - Why dimensionless NRMSE obscures safety-critical boundary drift and wall collisions.
  - MAST 65x65 poloidal grid geometry: $\Delta R = 2.8125\text{ cm/pixel}$, $\Delta Z = 4.6875\text{ cm/pixel}$.
  - The 4 physical metrics:
    1. **Plasma Boundary Error (cm):** Marching Squares Last Closed Flux Surface (LCFS) contour deviation.
    2. **Magnetic Axis Shift (cm):** 2D quadratic sub-pixel peak interpolation of the hot plasma core.
    3. **Divertor X-Point Error (cm):** Saddle point location ($\nabla \psi = 0$, $\det(H) < 0$).
    4. **False Wall-Proximity Alarm Rate (%):** Emergency abort threshold violation ($< 5.0\text{ cm}$ clearance).

### [04. Module 2: Failure-Aware Architecture + Flux-Loop PINN Loss](04_failure_aware_pinn.py) ([Notebook](04_failure_aware_pinn.ipynb))
- **Topic:** Physics-informed network training under sensor corruption.
- **Key Concepts:**
  - **Mask Feature Gating:** Zeroes out corrupt encoder representations ($h_i \cdot m_i$) before fusing into the bottleneck.
  - **Bottleneck Concatenation:** Concatenates active mask $M \in \{0, 1\}^8$ directly into the bottleneck MLP.
  - **Structured Failure Augmentation:** Injects synthetic dropouts, correlated groups, and temporal gaps during training ($p=0.5$).
  - **15 Wall Flux-Loop PINN Constraint:** Penalizes deviations at physical vacuum vessel boundary loop coordinates:
    $$\mathcal{L}_{PINN} = \frac{1}{K} \sum_{k=1}^{15} m_k \left( \hat{\psi}(R_k, Z_k) - \psi^{measured}_k \right)^2$$
  - **Kaggle Run Results:** 4,200 steps (53 epochs), 0 OOMs, best validation loss **`0.8883`** (beat clean baseline `0.8923`), clean MAE dropped to **`0.00396 Wb`** (10.8% improvement).

### [05. Module 3: Sensor Criticality Ranking & Minimal Sensor Set](05_sensor_criticality.py) ([Notebook](05_sensor_criticality.ipynb))
- **Topic:** Nuclear engineering reactor design & diagnostic pruning.
- **Key Concepts:**
  - **Centimeter Criticality:** Sensor importance ranked by physical boundary displacement $\Delta d_{LCFS}\text{ (cm)}$.
  - **Three-Tier Safety Hierarchy:**
    - *Tier 1 (Safety-Critical, $> 2.0\text{ cm}$):* Central Solenoid, Core CCBV pickup probes.
    - *Tier 2 (Plasma Shaping, $1.0 - 2.0\text{ cm}$):* Outboard B-field probes, Active PF shaping coils, Plasma current.
    - *Tier 3 (Auxiliary / Redundant, $< 1.0\text{ cm}$):* Saddle loop voltages, midplane flux loops.
  - **Minimal Sensor Set:** Greedy backward elimination under $\le 2.0\text{ cm}$ tolerance prunes 47 probes down to **25 probes (46.8% sensor reduction)** while safely preserving equilibrium confinement.

### [06. Module 4: Smart Adaptive Mitigation Brain](06_adaptive_mitigation.py) ([Notebook](06_adaptive_mitigation.ipynb))
- **Topic:** Real-time 3-step AI Doctor & Conformal Confidence Gauge.
- **Key Concepts:**
  - **Step 1: Real-Time Fault Detector ($< 0.5\text{ ms}$):** Detects stuck zeros, rail saturation, impulse spikes ($> 5\sigma$), and temporal gaps.
  - **Step 2: Smart Adaptive Router:**
    - Transient glitch $\to$ Temporal forward-fill / linear interpolation.
    - Redundant Tier 3 sensor $\to$ Spatial neighbor regression.
    - Safety-critical Tier 1 sensor / multi-sensor outage $\to$ Route to Module 2 Failure-Aware Network.
  - **Step 3: Calibrated Conformal Confidence Gauge:**
    - Calculates epistemic uncertainty bound $\pm \delta\text{ cm}$.
    - Evaluates minimum wall clearance $d_{wall} - \delta > 5.0\text{ cm}$.
    - Slashes false wall-proximity alarms from **75.5% down to 8.2%** (an **89.1% reduction**).
  - **Real-Time Profiling:** Entire diagnostic loop executes in **`~2.10 ms`**, well within the 5.0 ms tokamak magnetic control loop cycle budget.

---

## How to Run

Run any walkthrough script directly via Python:
```bash
python walkthrough/01_pipeline_cnn_mlp.py
python walkthrough/02_sensor_robustness.py
python walkthrough/03_physics_evaluation.py
python walkthrough/04_failure_aware_pinn.py
python walkthrough/05_sensor_criticality.py
python walkthrough/06_adaptive_mitigation.py
```
Or open any corresponding `.ipynb` notebook in VS Code or JupyterLab.
