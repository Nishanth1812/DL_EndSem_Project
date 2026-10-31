# Scripts Directory: Tools & Analysis Runners

This folder contains verified evaluation, analysis, and verification scripts for the project.

---

## 1. Physics Evaluation & Benchmark Runners

| Script | Purpose | Command |
| :--- | :--- | :--- |
| **`evaluate_physics_benchmark.py`** | **Module 1**: Calculates all 4 physical metrics in **centimeters** (boundary error, axis shift, X-point error, and false alarms) and produces the head-to-head comparison table against the Module 2 model. | `python scripts/evaluate_physics_benchmark.py` |
| **`compare_robustness.py`** | Compares **CNN+MLP vs. CNN+LSTM** head-to-head across all 63 sensor failure tasks of Gupta (arXiv:2607.11915). | `python scripts/compare_robustness.py` |
| **`compare_runs.py`** | Compares the 20 GB vs. 40 GB training runs to evaluate data scaling and regularisation. | `python scripts/compare_runs.py` |
| **`generate_robustness_figures.py`** | Renders degradation curves, signal importance charts, and mitigation bar plots. | `python scripts/generate_robustness_figures.py` |

---

## 2. Verification & Synchronization Tools

| Script | Purpose | Command |
| :--- | :--- | :--- |
| **`check_nb.py`** | Verifies that the Kaggle notebook's writefile cell (Cell 3) is **byte-identical** to `src/pipeline.py`. | `python scripts/check_nb.py` |
| **`sync_nb.py`** | Automatically copies any changes in `src/pipeline.py` into Cell 3 of the Kaggle notebook. | `python scripts/sync_nb.py` |
| **`sync_robustness_nb.py`** | Keeps `src/robustness.py` and `src/tokamark_robustness_kaggle.ipynb` in sync. | `python scripts/sync_robustness_nb.py` |
| **`check_slides.py`** | Lints the 38-frame LaTeX Beamer slide deck (`PPT/Team_15_Final_Review.tex`) for bracket balance, math delimiters, and visual overflow. | `python scripts/check_slides.py` |

---

## 3. Archive Subdirectory

* **`scripts/archive/`**: Contains legacy one-off helper scripts and migration tools from earlier development sessions.
