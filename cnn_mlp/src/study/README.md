# study/ - model-driven robustness + Modules 1/3/4 (works for CNN+MLP and CNN+LSTM)

Unlike the hard-coded benchmark scripts in `src/`, everything here is computed from a trained checkpoint:
`pipeline_robustness.py` (63-pass sensor-failure sweep, dual-GPU inference, per-pass physics metrics),
`physics_study.py` (Module 3 branch ablation + Module 4 conformal mitigation on validation/test windows),
`physics_evaluation.py` (Module 1; predicted LCFS cut at the TRUE flux level), `sensor_criticality.py`, `adaptive_mitigation.py`.
`scripts/build_lstm_robust_kernel.py` embeds them into the Kaggle kernel `kernels/tokamark-robust-lstm-fa`.
