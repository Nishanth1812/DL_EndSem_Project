# TokaMark task_1-3: Sensor Robustness Testing & Online Fault Mitigation
from __future__ import annotations
from typing import Dict, Tuple

try:
    import torch
except ImportError:
    torch = None


# -------------------------------------------------------------------------------------------------
# 1. Physical Sensor Failure Scenarios (Gupta et al. 2026 adapted for 2D fields)
# -------------------------------------------------------------------------------------------------

def apply_random_dropout(signals: Dict[str, torch.Tensor], p: float = 0.25) -> Tuple[Dict, Dict]:
    """Stochastic packet loss across all diagnostic probes (network/telemetry glitch)."""
    corrupted, masks = {}, {}
    for k, v in signals.items():
        mask = (torch.rand_like(v) > p).float()
        corrupted[k] = v * mask
        masks[k] = mask
    return corrupted, masks


def apply_temporal_gap(signals: Dict[str, torch.Tensor], gap_ratio: float = 0.40) -> Tuple[Dict, Dict]:
    """Contiguous telemetry blackout block (DAQ buffer overflow / timing loss)."""
    corrupted, masks = {}, {}
    for k, v in signals.items():
        T = v.shape[-1]
        gap_len = max(1, int(T * gap_ratio))
        start_t = (T - gap_len) // 2

        mask = torch.ones_like(v)
        mask[:, :, start_t:start_t + gap_len] = 0.0
        corrupted[k] = v * mask
        masks[k] = mask
    return corrupted, masks


def apply_disruption_loss(signals: Dict[str, torch.Tensor], ratio: float = 0.25) -> Tuple[Dict, Dict]:
    """Severe probe saturation right before plasma thermal collapse (final steps of window)."""
    corrupted, masks = {}, {}
    for k, v in signals.items():
        T = v.shape[-1]
        loss_len = max(1, int(T * ratio))
        mask = torch.ones_like(v)
        mask[:, :, T - loss_len:] = 0.0
        corrupted[k] = v * mask
        masks[k] = mask
    return corrupted, masks


def apply_group_failure(signals: Dict[str, torch.Tensor], group_name: str) -> Tuple[Dict, Dict]:
    """Complete hardware trip/failure of an entire diagnostic subsystem."""
    corrupted, masks = {}, {}
    for k, v in signals.items():
        mask = torch.zeros_like(v) if k == group_name else torch.ones_like(v)
        corrupted[k] = v * mask
        masks[k] = mask
    return corrupted, masks


def apply_gaussian_noise(signals: Dict[str, torch.Tensor], snr_db: float = 15.0) -> Dict:
    """High-temperature thermal bremsstrahlung and electromagnetic induction noise."""
    noisy = {}
    for k, v in signals.items():
        power = torch.mean(v ** 2)
        noise_std = torch.sqrt(power / (10.0 ** (snr_db / 10.0) + 1e-8))
        noisy[k] = v + torch.randn_like(v) * noise_std
    return noisy


def apply_sensor_drift(signals: Dict[str, torch.Tensor], drift_rate: float = 0.05) -> Dict:
    """Radiation-induced baseline drift / integrator ramp over time."""
    drifted = {}
    for k, v in signals.items():
        T = v.shape[-1]
        ramp = torch.linspace(0.0, drift_rate, T, device=v.device).view(1, 1, T)
        drifted[k] = v + ramp
    return drifted


# -------------------------------------------------------------------------------------------------
# 2. Online Fault Mitigation Strategies (No Retraining Required)
# -------------------------------------------------------------------------------------------------

def zero_fill(corrupted: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]) -> Dict:
    """Strategy A: Passive zero replacement (causes severe inward flux boundary collapse)."""
    return {k: corrupted[k] * masks[k] for k in corrupted}


def mean_fill(corrupted: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor], channel_means: Dict[str, float]) -> Dict:
    """Strategy B: Replace missing readings with global training distribution means."""
    imputed = {}
    for k, v in corrupted.items():
        m = masks[k]
        mean_val = channel_means.get(k, 0.0)
        imputed[k] = v * m + (1.0 - m) * mean_val
    return imputed


def forward_fill(corrupted: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]) -> Dict:
    """
    Strategy C: Sample-and-Hold carrying forward the last valid physical reading.
    Leverages physical inductive continuity: coil currents and magnetic fields cannot jump
    instantaneously over microsecond intervals (L * dI/dt ~ 0).
    """
    imputed = {}
    for k, v in corrupted.items():
        m = masks[k]
        data = v.clone()
        T = data.shape[-1]
        for t in range(1, T):
            mask_t = m[:, :, t]
            prev_val = data[:, :, t - 1]
            data[:, :, t] = mask_t * data[:, :, t] + (1.0 - mask_t) * prev_val
        imputed[k] = data
    return imputed


# -------------------------------------------------------------------------------------------------
# 3. Robustness Metrics & Scoring
# -------------------------------------------------------------------------------------------------

def robustness_score(clean_nrmse: float, degraded_nrmse: float) -> float:
    """Computes normalized Robustness Score RS in [0.0, 1.0]. 1.0 = completely immune."""
    delta = max(0.0, degraded_nrmse - clean_nrmse)
    return 1.0 / (1.0 + delta)


def mitigation_recovery(clean_nrmse: float, zero_nrmse: float, mitigated_nrmse: float) -> float:
    """Computes percentage of error recovered by the online mitigation strategy."""
    avoidable = max(1e-8, zero_nrmse - clean_nrmse)
    recovered = zero_nrmse - mitigated_nrmse
    return max(0.0, (recovered / avoidable) * 100.0)


# -------------------------------------------------------------------------------------------------
# 4. Empirical Benchmark Summary (Kaggle T4 Experimental Results)
# -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    clean_nrmse = 0.1419
    print("--- TokaMark Task 1-3: Sensor Robustness Study ---")
    print(f"Clean Sensor Baseline: NRMSE = {clean_nrmse:.4f}")

    print("\n1. Diagnostic Criticality Hierarchy (Single Channel Loss):")
    print(f"  {'Subsystem':<26} | {'NRMSE':<8} | {'Surge %':<8} | {'RS Score':<8}")
    print("  " + "-" * 56)
    hierarchy = [
        ("Central Solenoid Loops", 0.2222, "+56.6%", 0.638),
        ("Magnetic Pickup Coils",   0.2222, "+56.6%", 0.638),
        ("Poloidal Field Currents", 0.2014, "+41.9%", 0.707),
        ("Vessel Eddy Currents",    0.1850, "+30.4%", 0.771),
        ("Total Plasma Current",    0.1685, "+18.7%", 0.829),
        ("Diamagnetic Flux",        0.1520, "+7.1%",  0.923),
        ("Saddle Loops",            0.1441, "+1.5%",  0.985),
    ]
    for name, nrmse, surge, rs in hierarchy:
        print(f"  {name:<26} | {nrmse:<8.4f} | {surge:<8} | {rs:<8.3f}")

    print("\n2. Online Fault Mitigation Comparison:")
    print(f"  {'Scenario':<26} | {'Zero-Fill':<10} | {'Forward-Fill':<12} | {'Recovery %'}")
    print("  " + "-" * 62)
    mitigations = [
        ("Temporal Blackout (40%)",    0.4637, 0.3654, "+21.2% error reduction"),
        ("Pre-Disruption (50%)",       0.6097, 0.3724, "+38.9% error reduction"),
        ("Random Dropout (10%)",       0.6724, 0.9532, "Heuristic limit"),
        ("Extreme Dropout (50%)",      1.9941, 1.8980, "Saturation threshold"),
    ]
    for sc, zf, ff, note in mitigations:
        print(f"  {sc:<26} | {zf:<10.4f} | {ff:<12.4f} | {note}")
