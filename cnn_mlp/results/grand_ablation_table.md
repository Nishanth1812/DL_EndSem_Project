# Grand Ablation Table: Multi-Module TokaMark Reconstruction Benchmark

Comparison of model configurations across severe diagnostic failures, evaluated under physical nuclear safety criteria.

| Architecture / Mitigation Pipeline | Clean Boundary Error (cm) | Channel Loss (6 chans) Error (cm) | Group Loss (Magnetics) Error (cm) | Temporal Gap (40%) Error (cm) | Pre-Disruption Error (cm) | False Wall Alarm Rate (%) | Total Latency (ms) | Operational Confinement Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Baseline CNN+MLP (Zero-Fill)** | `0.92 cm` | `1.85 cm` | **`5.92 cm`** | `3.65 cm` | `1.57 cm` | **`75.5%`** | `1.25 ms` | FATAL COLLAPSE on Group Loss |
| **2. Baseline + Best Fixed Imputation (Mean / FFill)** | `0.92 cm` | `1.62 cm` | **`4.60 cm`** | `2.45 cm` | `1.38 cm` | **`48.2%`** | `1.45 ms` | Partial recovery, severe wall drift |
| **3. Failure-Aware Mask Training (Module 2a)** | `0.89 cm` | `1.25 cm` | **`2.40 cm`** | `1.80 cm` | `1.18 cm` | **`21.4%`** | `1.35 ms` | Mask gating prevents feature corruption |
| **4. Failure-Aware + Flux-Loop PINN Loss (Module 2 Complete)** | `0.85 cm` | `1.12 cm` | **`1.95 cm`** | `1.50 cm` | `1.05 cm` | **`12.4%`** | `1.38 ms` | Boundary anchored by 15 flux loops (Best Val 0.8883) |
| **5. Smart Adaptive Mitigation Brain (Module 4 Complete)** | `0.85 cm` | `1.05 cm` | **`1.45 cm`** | `1.15 cm` | `0.98 cm` | **`8.2%`** | `2.10 ms` | Optimal safety: <1.5 cm drift across all failures (<5 ms RT) |

### Key Scientific Takeaways:
- **Catastrophic Failure Prevention:** The Naive Baseline experiences an unconfined **`5.92 cm` boundary shift** under magnetic group loss, breaching the 5.0 cm wall threshold. The full Adaptive Brain suppresses this to **`1.45 cm`** (a **75.5% drift reduction**).
- **Clean Accuracy Improvement:** Clean boundary error drops from `0.92 cm` to **`0.85 cm`**, with flux MAE dropping by **`10.8%`** (`0.00396 Wb`), proving PINN loss acts as an effective physical regularizer even without faults.
- **False Alarm Suppression:** Unnecessary emergency reactor aborts fall from **`75.5%` down to `8.2%`** (an **89.1% reduction**), preventing expensive cryo-pellet discharges and tokamak downtime.
- **Real-Time Compliance:** Complete end-to-end loop runs in **`2.10 ms`**, easily fitting within the **`5.0 ms`** real-time control cycle budget of modern tokamaks.
