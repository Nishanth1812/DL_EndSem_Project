# Research papers relevant to this project

Two papers anchor the work here: the benchmark we are reproducing (TokaMark) and
the *only* prior robustness study on it (Gupta). Both verified as of the latest
available version on the date this file was written.

---

## 1. TokaMark — the benchmark and baseline we reproduce

- **Title:** TokaMark: A Comprehensive Benchmark for MAST Tokamak Plasma Models
- **Authors:** Cécile Rousseau, Samuel Jackson, Rodrigo H. Ordonez-Hurtado, Nicola C. Amorisco,
  Tobia Boschi, George K. Holt, Andrea Loreti, Eszter Székely, Alexander Whittle,
  Adriano Agnello, Stanislas Pamela, Alessandra Pascale, Robert Akers, Juan Bernabe Moreno,
  Sue Thorne, Mykhaylo Zayats
- **Affiliation:** IBM Research Europe, UK Atomic Energy Authority, STFC Hartree Centre
- **arXiv:** 2602.10132  (physics.plasm-ph)
- **Latest version:** v3, 11 Jun 2026
- **Links:**
  - Abstract: https://arxiv.org/abs/2602.10132
  - HTML (v3): https://arxiv.org/html/2602.10132
  - PDF: https://arxiv.org/pdf/2602.10132
- **Code/data:** https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark_baseline

**What it gives us (the facts our project is built on):**
- 14 tasks in 4 groups over real FAIR-MAST data; Group 1 = Equilibrium Reconstruction.
- task_1-3 uses 8 input signals (flux loops, 3 pickup-coil fields, saddle voltage,
  coil current, solenoid current, summary ip) -> output `equilibrium/psi` (65x65).
- 80/10/10 shot-level split; official counts train/val/test = 8963/1115/1110 (565.3 GB total).
- The CNN baseline reaches low NRMSE on Group 1; used as our reference architecture
  (`MultiConv_MLP`, see docs/reference/).

---

## 2. Gupta — the only prior robustness study on TokaMark  (KEY REFERENCE)

- **Title:** Benchmarking Sensor Robustness in Plasma Diagnostic Models:
  A Systematic Evaluation on TokaMark
- **Author:** Neerav Gupta (Independent Researcher)
- **arXiv:** 2607.11915  (physics.plasm-ph)
- **Version seen:** v1, 05 Jul 2026
- **Links:**
  - Abstract: https://arxiv.org/abs/2607.11915
  - HTML: https://arxiv.org/html/2607.11915v1
  - PDF: https://arxiv.org/pdf/2607.11915
- **Code:** https://github.com/Neerav-Gupta/tokamark-robustness
- **Data + checkpoints:** https://huggingface.co/datasets/Neerav-Gupta/tokamark-robustness-data

**Why it matters to us:** it is the only paper that systematically stresses the
TokaMark CNN baseline under sensor failure. It defines the six corruption
scenarios and the Robustness Score (RS) that our robustness study will mirror.
Note: Gupta evaluates **Task 4-4** (plasma-current quench forecasting), NOT
task_1-3 — so our task_1-3 robustness study would be novel even relative to it.

### 2.1 The six corruption scenarios (we will reuse these)
1. **Random dropout** — zero a random fraction p in {0.10, 0.25, 0.50} of nonzero values.
2. **Channel ablation** — zero n in {1, 3, 6} whole channels.
3. **Per-category importance** — zero each diagnostic category in isolation.
4. **Temporal gap** — contiguous gap of fraction f in {0.20, 0.40, 0.60}, at
   front / random / pre-event positions.
5. **Correlated group failure** — zero a physically-related group at once
   (motivated by r=1.000 NaN correlation between interferometer and D-alpha).
6. **Disruption-proximate failure** — gap of fraction f in {0.10, 0.25, 0.50}
   at the very end of the window (most safety-critical).

### 2.2 Imputation / mitigation strategies
- **Zero-fill** (no imputation; TokaMark's implicit default baseline).
- **Mean-fill** (per-channel training mean).
- **Forward-fill** (carry last valid value; most physical for real-time control).

### 2.3 The Robustness Score (RS)
```
RS = (1/|S|) * sum over scenarios s in S of  NRMSE_clean / NRMSE_s
```
RS = 1.0 means corruption has no effect; below 1.0 scales linearly with damage.
Computed over the zero-fill corruption set S. Gupta encourages its adoption.

### 2.4 Headline findings (for comparison with our results)
- **Disruption-proximate failure is catastrophic for sequence models**: LSTM
  +212% NRMSE; Transformer +143% at 20% gap; CNN more gradual (+67.7% to +167.7%).
- **Plasma current (summary-ip) is the most critical diagnostic** for every
  architecture: removing it costs LSTM +139.8%, Transformer +89.1%, XGBoost
  +72.7%, CNN +27.7%.
- **Forward-fill nearly eliminates random-dropout degradation** for sequence
  models (LSTM +57% -> ~0%), but cannot fix end-of-window corruption.
- **CNN clean NRMSE 0.528** (200 train shots) vs TokaMark's 0.429 (full 9,270
  shots); CNN RS = 0.764. CNN is *most* sensitive to active-coil removal (+46.9%)
  because per-channel encoders cannot share info across channels.
- Gupta's absolute NRMSEs are window-level and NOT directly comparable to
  TokaMark's shot-level hierarchical aggregation.

### 2.5 Gupta's setup (so we can position ours)
- Task 4-4; 14 diagnostic + 4 actuator signals -> 600x18 tensor.
- Trains on 9,950 windows / 200 shots; tests on 2,420 windows / 50 shots.
- CNN: N=3 conv layers, K=3, s=3, p=1, D=16; MLP backbone 64->32->16->1; ~458K params.
  (Note: this differs from our task_1-3 model, which is the full MultiConv_MLP.)

---

## 3. Positioning: what our project adds

- We stress the **task_1-3 (equilibrium reconstruction / 65x65 psi)** model,
  whereas Gupta stressed **task_4-4 (scalar current forecasting)**. Same
  benchmark, complementary task.
- Our output is a **2-D field**, so "degradation" can be measured spatially
  (where in the psi map errors concentrate), which a scalar task cannot show.
- We can reuse Gupta's scenario taxonomy and the RS metric to stay comparable.
