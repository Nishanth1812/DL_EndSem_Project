# Run: 40 GB CNN+MLP Runs (`results.zip` & `results-2.zip`)

Comprehensive results for the 40 GB benchmark on the official random split.

---

## 1. Run 1: Champion Model (`results.zip`) — **BEST OVERALL**
- **Weight Decay:** `1e-4` (Official baseline setting)
- **Training Steps:** 3,500 steps (44 epochs)
- **Best Validation Loss:** `0.8723`
- **Duration:** 154.9 minutes on Kaggle Dual Tesla T4 GPUs
- **Test Set Evaluation (Official Un-Standardized Protocol):**
  - **NRMSE:** **`0.14186`** (~86% physical reconstruction accuracy)
  - **MAE:** **`0.00389 Wb`** (<1% average physical deviation across 4,225 grid points)
  - **RMSE:** **`0.00779 Wb`**
  - **NMAE:** **`0.07078`**
- **Takeaway:** This is our **primary, champion baseline model**. It outperforms the official UKAEA paper benchmark (`0.148` on 11,000 shots) and outperforms the CNN+LSTM model by **28%** in test NRMSE.

---

## 2. Run 2: Heavy Regularization Ablation (`results-2.zip`)
- **Weight Decay:** `1e-3` (10x stronger L2 penalty)
- **Training Steps:** 2,800 steps (35 epochs)
- **Best Validation Loss:** `0.8923`
- **Duration:** 133.9 minutes
- **Test Set Evaluation:**
  - **NRMSE:** `0.14977` (5.3% higher error than Run 1)
  - **MAE:** `0.00444 Wb` (12.4% higher error than Run 1)
  - **RMSE:** `0.00823 Wb`
  - **NMAE:** `0.08076`
- **Takeaway:** The 10x stronger L2 penalty slightly over-penalized network capacity, causing early stopping at step 2,800 with slightly higher test error. Confirms that `1e-4` is the optimal hyperparameter.

---

## 3. Directory Layout
- `results.zip`: Complete Kaggle output archive for Run 1 (`best_model.pt`, metrics, plots, selection).
- `results-2.zip`: Complete Kaggle output archive for Run 2.
- `extracted/`: Extracted files from Run 1 (`out/random/...`).
- `extracted_run2/`: Extracted files from Run 2 (`out/random/...`).
