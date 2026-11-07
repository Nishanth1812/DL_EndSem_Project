#!/usr/bin/env python
"""Module 3 v2: Green's-function (vacuum, axisymmetric) sensitivity of the MAST magnetics to plasma current, and comparison with
the data-driven (Shapley / drop-one) ranking.

Physics. A toroidal current filament I at (Rc, Zc) produces, at (R, Z), with k^2 = m = 4 R Rc / ((R + Rc)^2 + (Z - Zc)^2):
    psi = M I / (2 pi),   M = mu0 sqrt(R Rc) [(2 - m) K(m) - 2 E(m)] / sqrt(m)          (psi = poloidal flux function, Wb/rad;
                                                                                        total flux through a loop = 2 pi psi = M I)
    B_Z =  (1/R) d psi/dR = mu0 I / (2 pi sqrt(D)) [K + (Rc^2 - R^2 - dz^2)/((Rc - R)^2 + dz^2) E],      D = (R+Rc)^2 + dz^2
    B_R = -(1/R) d psi/dZ = mu0 I dz / (2 pi R sqrt(D)) [-K + (Rc^2 + R^2 + dz^2)/((Rc - R)^2 + dz^2) E],   dz = Z - Zc
Validated in test_greens_sensitivity.py against: on-axis circular loop, symmetry G(a,b)=G(b,a), Neumann/Biot-Savart quadrature,
finite-difference B = curl(psi).

Sensor model. Vacuum fields only: NO vessel/passive-structure eddy currents, NO iron, plasma modelled as a smooth current
distribution on pixels inside the limiter. Sensor types: flux loop -> psi, CCBV/OBV -> B_Z, OBR -> B_R (component inferred from
the channel names), PF-coil and solenoid current -> direct readout of that coil current, Ip -> sum of plasma current.
Saddle loops (real geometry: 12 identical loops on the cylinder R=1.999 m, Z in [-0.4, 0.4], 20 deg wide, every 30 deg) measure
the VOLTAGE dPhi/dt, Phi = -dphi*(psi(R,Zt) - psi(R,Zb)); the model input carries no DC level, so they are excluded by default
(include_saddle=True treats them as level sensors = optimistic upper bound; all 12 are then exact duplicates).

Information model (linear-Gaussian, closed form). Unknowns x = [plasma pixel currents J ; PF/solenoid currents I].
Target = psi at target points (boundary region) = T x. Sensors y = A x + noise, noise sd = rel_noise * prior RMS of that signal.
Prior: J smooth (squared-exponential kernel, length ell) scaled so the plasma contributes unit prior variance on the target;
coils share `coil_share` times that, split equally. value_fn(mask) = sqrt(mean posterior variance of the target given the
available sensors) -- an ERROR-like value that plugs straight into sensor_attribution (Shapley, worst-k, minimum set), which
makes the data-driven and the Green's rankings directly comparable. Scale conventions (prior shares, rel_noise, ell) are MODELLING
CHOICES, exposed as parameters; test the sensitivity of any ranking to them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import special, stats

from sensor_geometry import DEFAULT_R_AXIS, DEFAULT_Z_AXIS, SensorGeometry

MU0 = 4e-7 * math.pi
_COILS = ("P2IL", "P2IU", "P2OL", "P2OU", "P3L", "P3U", "P4L", "P4U", "P5L", "P5U", "SOL")


# ----------------------------------------------------------------------------------------------------------------------
# Green's functions (vectorised, broadcasting)
# ----------------------------------------------------------------------------------------------------------------------
def _series_2mK_minus_2E(m):
    """(2 - m) K(m) - 2 E(m) from the power series of K and E (no cancellation for small m)."""
    out = np.zeros_like(m)
    c, k_prev = 1.0, 0.0
    for n in range(0, 16):
        if n:
            c *= (2 * n - 1) / (2 * n)
        k_n = c * c
        out += (2 * k_n - k_prev - 2 * k_n / (1 - 2 * n)) * m ** n
        k_prev = k_n
    return out * (math.pi / 2)


def green_psi(R, Z, Rc, Zc):
    """Poloidal flux function psi [Wb/rad] at (R, Z) per ampere of a toroidal filament at (Rc, Zc). Total flux = 2 pi psi."""
    R, Z, Rc, Zc = np.broadcast_arrays(*(np.asarray(v, float) for v in (R, Z, Rc, Zc)))
    den = (R + Rc) ** 2 + (Z - Zc) ** 2
    m = np.clip(4 * R * Rc / den, 1e-300, 1 - 1e-14)
    small = m < 0.1
    big = np.where(small, 0.5, m)
    f_big = (2 - big) * special.ellipk(big) - 2 * special.ellipe(big)
    f = np.where(small, _series_2mK_minus_2E(np.where(small, m, 0.0)), f_big)
    M = MU0 * np.sqrt(R * Rc) * f / np.sqrt(m)
    return M / (2 * math.pi)


def _kek(R, Z, Rc, Zc):
    R, Z, Rc, Zc = np.broadcast_arrays(*(np.asarray(v, float) for v in (R, Z, Rc, Zc)))
    dz = Z - Zc
    den = (R + Rc) ** 2 + dz ** 2
    m = np.clip(4 * R * Rc / den, 0.0, 1 - 1e-14)
    return R, dz, den, special.ellipk(m), special.ellipe(m), (Rc - R) ** 2 + dz ** 2, Rc


def green_bz(R, Z, Rc, Zc):
    """B_Z [T/A] at (R, Z) from a unit toroidal current filament at (Rc, Zc)."""
    R, dz, den, K, E, d2, Rc = _kek(R, Z, Rc, Zc)
    return MU0 / (2 * math.pi * np.sqrt(den)) * (K + (Rc ** 2 - R ** 2 - dz ** 2) / d2 * E)


def green_br(R, Z, Rc, Zc):
    """B_R [T/A] at (R, Z) from a unit toroidal current filament at (Rc, Zc) (zero on the axis R = 0)."""
    R, dz, den, K, E, d2, Rc = _kek(R, Z, Rc, Zc)
    Rs = np.where(R > 1e-9, R, 1.0)
    out = MU0 * dz / (2 * math.pi * Rs * np.sqrt(den)) * (-K + (Rc ** 2 + R ** 2 + dz ** 2) / d2 * E)
    return np.where(R > 1e-9, out, 0.0)


# ----------------------------------------------------------------------------------------------------------------------
# Sensor response matrices
# ----------------------------------------------------------------------------------------------------------------------
def plasma_pixels(R_axis=DEFAULT_R_AXIS, Z_axis=DEFAULT_Z_AXIS, limiter=None, stride=3, margin=0.0):
    """(R, Z) of grid pixels (every `stride`-th) that lie inside the limiter polygon (shrunk by nothing; `margin` unused for
    non-convex polygons -- keep 0). Without a limiter all pixels are used."""
    RR, ZZ = np.meshgrid(R_axis[::stride], Z_axis[::stride])
    pts = np.c_[RR.ravel(), ZZ.ravel()]
    if limiter is not None:
        from matplotlib.path import Path
        pts = pts[Path(np.c_[limiter[0], limiter[1]]).contains_points(pts)]
    return pts[:, 0], pts[:, 1]


def _row(kind, quantity, R, Z, Rp, Zp):
    if kind == "flux_loop":
        return green_psi(R, Z, Rp, Zp)
    if kind == "pickup_coil":
        return green_br(R, Z, Rp, Zp) if quantity == "B_R" else green_bz(R, Z, Rp, Zp)
    raise ValueError(kind)


@dataclass
class LinearGaussianModel:
    geo: SensorGeometry
    has_model: np.ndarray          # (n_sens,) bool: sensor has a static axisymmetric response
    A: np.ndarray                  # (n_sens, n_x) rows for modelled sensors (zeros elsewhere)
    T: np.ndarray                  # (n_t, n_x)
    Sigma_x: np.ndarray            # (n_x, n_x)
    noise_sd: np.ndarray           # (n_sens,)
    n_pix: int
    target_pts: np.ndarray
    cm_per_unit: Optional[float] = None
    # cached
    _C: Optional[np.ndarray] = None
    _S: Optional[np.ndarray] = None
    _prior: float = 0.0

    def __post_init__(self):
        P = self.Sigma_x @ self.A.T
        self._C = self.T @ P                                  # (n_t, n_sens) cov(target, y)
        self._S = self.A @ P + np.diag(self.noise_sd ** 2)   # cov(y)
        self._prior = float(np.trace(self.T @ self.Sigma_x @ self.T.T) / len(self.T))

    @property
    def n_sens(self):
        return len(self.has_model)

    def post_var(self, mask) -> float:
        """Mean posterior variance of the target given available sensors `mask` (bool, n_sens); unmodelled sensors ignored."""
        idx = np.flatnonzero(np.asarray(mask, bool) & self.has_model)
        if idx.size == 0:
            return self._prior
        S = self._S[np.ix_(idx, idx)]
        C = self._C[:, idx]
        red = np.einsum("ti,ti->", C, np.linalg.solve(S, C.T).T) / len(self.T)
        return max(self._prior - red, 0.0)

    def value_fn(self, mask) -> float:
        """sqrt(mean posterior variance) [Wb/rad]; times cm_per_unit if given. Error-like: larger = worse."""
        v = math.sqrt(self.post_var(mask))
        return v * self.cm_per_unit if self.cm_per_unit else v

    def value_fn_vec(self, masks) -> np.ndarray:
        return np.array([self.value_fn(m) for m in masks])

    def kalman_gain(self) -> np.ndarray:
        idx = np.flatnonzero(self.has_model)
        K = np.zeros((len(self.T), self.n_sens))
        K[:, idx] = np.linalg.solve(self._S[np.ix_(idx, idx)], self._C[:, idx].T).T
        return K

    def scores(self) -> dict:
        """Per-sensor Green's-function scores (NaN where the sensor has no model):
        loo_loss    value(all but s) - value(all)             (Green's analogue of drop-one)
        alone_gain  prior variance - posterior variance with only s   (information carried by s alone)
        noise_prop  RMS_t |K_ts| * noise_sd_s                  (target error caused by a 1-sigma error on s, full-data estimator)
        """
        n = self.n_sens
        allm = np.ones(n, bool)
        base = self.value_fn(allm)
        loo, alone = np.full(n, np.nan), np.full(n, np.nan)
        for s in np.flatnonzero(self.has_model):
            m = allm.copy()
            m[s] = False
            loo[s] = self.value_fn(m) - base
            one = np.zeros(n, bool)
            one[s] = True
            alone[s] = self._prior - self.post_var(one)
        K = self.kalman_gain()
        noise_prop = np.sqrt((K ** 2).mean(0)) * self.noise_sd
        noise_prop[~self.has_model] = np.nan
        return dict(loo_loss=loo, alone_gain=alone, noise_prop=noise_prop, value_all=base, prior_value=math.sqrt(self._prior))


def build_model(geo: SensorGeometry, R_axis=DEFAULT_R_AXIS, Z_axis=DEFAULT_Z_AXIS, stride=3, ell=0.2, rel_noise=0.02,
                coil_share=1.0, target_pts: Optional[np.ndarray] = None, cm_per_unit: Optional[float] = None,
                include_saddle: bool = False) -> LinearGaussianModel:
    """Assemble the linear-Gaussian model for the 115 sensors of `geo`.

    target_pts: (n_t, 2) (R, Z) where the boundary flux is wanted. Default: pixels inside the limiter on a coarse ring-free
    sampling (every 2*stride-th pixel), i.e. the whole plasma region -- pass the reference boundary contour for a sharper test.
    """
    Rp, Zp = plasma_pixels(R_axis, Z_axis, geo.limiter, stride)
    n_pix = len(Rp)
    coils = []
    for c in _COILS:
        r, z = geo.coil_elements(c)
        coils.append((r, z))
    n_x = n_pix + len(coils)

    def src_row(f_pix, f_coil):
        row = np.zeros(n_x)
        row[:n_pix] = f_pix
        row[n_pix:] = [f_coil(r, z).sum() for r, z in coils]
        return row

    if target_pts is None:
        tR, tZ = plasma_pixels(R_axis, Z_axis, geo.limiter, 2 * stride)
        target_pts = np.c_[tR, tZ]
    T = np.array([src_row(green_psi(R, Z, Rp, Zp), lambda r, z, R=R, Z=Z: green_psi(R, Z, r, z)) for R, Z in target_pts])

    n = len(geo.names)
    A = np.zeros((n, n_x))
    has = np.zeros(n, bool)
    for i in range(n):
        k, q = geo.kind[i], geo.quantity[i]
        if k in ("flux_loop", "pickup_coil"):
            R, Z = geo.R[i], geo.Z[i]
            A[i] = src_row(_row(k, q, R, Z, Rp, Zp), lambda r, z, R=R, Z=Z, k=k, q=q: _row(k, q, R, Z, r, z))
            has[i] = True
        elif k in ("pf_coil_current", "solenoid_current"):
            name = geo.names[i]
            c = "SOL" if k == "solenoid_current" else name.split("_")[1].split(" ")[0]
            A[i, n_pix + _COILS.index(c)] = 1.0
            has[i] = True
        elif k == "rogowski":
            A[i, :n_pix] = 1.0
            has[i] = True
        elif k == "saddle_loop" and include_saddle:
            # loop on the cylinder R=Rs, Z in [Zb, Zt], toroidal span dphi: flux = -dphi * (psi(Rs, Zt) - psi(Rs, Zb)).
            # The measured signal is its TIME DERIVATIVE, so this treats it as a level sensor: an optimistic upper bound.
            p = geo.raw["sensors"][i]["polygon"]
            Rs, Zt, Zb = float(np.mean(p["R"])), max(p["Z"]), min(p["Z"])
            dphi = math.radians(max(p["phi_deg"]) - min(p["phi_deg"]))
            A[i] = src_row(dphi * (green_psi(Rs, Zt, Rp, Zp) - green_psi(Rs, Zb, Rp, Zp)),
                           lambda r, z, Rs=Rs, Zt=Zt, Zb=Zb: dphi * (green_psi(Rs, Zt, r, z) - green_psi(Rs, Zb, r, z)))
            has[i] = True
        # saddle_loop with include_saddle=False: left unmodelled

    # prior: smooth plasma current + equal-share coil currents, both calibrated on the target variance
    d2 = (Rp[:, None] - Rp[None]) ** 2 + (Zp[:, None] - Zp[None]) ** 2
    Kj = np.exp(-d2 / (2 * ell ** 2)) + 1e-9 * np.eye(n_pix)
    Tp, Tc = T[:, :n_pix], T[:, n_pix:]
    plasma_var = np.trace(Tp @ Kj @ Tp.T) / len(T)
    Sigma = np.zeros((n_x, n_x))
    Sigma[:n_pix, :n_pix] = Kj / plasma_var
    coil_var = coil_share / (len(coils) * np.maximum((Tc ** 2).sum(0) / len(T), 1e-300))
    Sigma[n_pix:, n_pix:] = np.diag(coil_var)
    sig = np.sqrt(np.einsum("ij,jk,ik->i", A, Sigma, A))
    return LinearGaussianModel(geo, has, A, T, Sigma, rel_noise * sig, n_pix, np.asarray(target_pts), cm_per_unit)


# ----------------------------------------------------------------------------------------------------------------------
# Rank agreement with permutation p-values
# ----------------------------------------------------------------------------------------------------------------------
def rank_agreement(a, b, n_perm: int = 10000, seed: int = 0) -> dict:
    """Spearman rho and Kendall tau between two score vectors (NaN pairs dropped) with two-sided permutation p-values."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return dict(n=int(n), spearman=float("nan"), p_spearman=float("nan"), kendall=float("nan"), p_kendall=float("nan"))
    rho, tau = stats.spearmanr(a, b)[0], stats.kendalltau(a, b)[0]
    rng = np.random.default_rng(seed)
    ra = stats.rankdata(a)
    cnt_r = cnt_t = 0
    for _ in range(n_perm):
        pb = rng.permutation(b)
        cnt_r += abs(np.corrcoef(ra, stats.rankdata(pb))[0, 1]) >= abs(rho) - 1e-12
        cnt_t += abs(stats.kendalltau(a, pb)[0]) >= abs(tau) - 1e-12
    return dict(n=int(n), spearman=float(rho), p_spearman=(cnt_r + 1) / (n_perm + 1), kendall=float(tau),
                p_kendall=(cnt_t + 1) / (n_perm + 1))


def rank_agreement_by_group(a, b, group_labels, n_perm: int = 5000, seed: int = 0) -> dict:
    out = {"all": rank_agreement(a, b, n_perm, seed)}
    labels = np.asarray(group_labels)
    for g in dict.fromkeys(labels.tolist()):
        m = labels == g
        out[g] = rank_agreement(np.asarray(a)[m], np.asarray(b)[m], n_perm, seed)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Spatial error map vs sensor location
# ----------------------------------------------------------------------------------------------------------------------
def sensor_response_map(geo: SensorGeometry, i: int, R_axis=DEFAULT_R_AXIS, Z_axis=DEFAULT_Z_AXIS) -> Optional[np.ndarray]:
    """(nZ, nR) map |response of sensor i to a unit current filament at each pixel| (flux loops, pickup coils), or, for a PF
    coil, |psi that the coil produces at each pixel per unit coil current|. None for sensors without a model."""
    RR, ZZ = np.meshgrid(R_axis, Z_axis)
    k = geo.kind[i]
    if k in ("flux_loop", "pickup_coil"):
        return np.abs(_row(k, geo.quantity[i], geo.R[i], geo.Z[i], RR, ZZ))
    if k in ("pf_coil_current", "solenoid_current"):
        name = geo.names[i]
        c = "SOL" if k == "solenoid_current" else name.split("_")[1].split(" ")[0]
        r, z = geo.coil_elements(c)
        out = np.zeros_like(RR)
        for a, b in zip(r, z):
            out += green_psi(RR, ZZ, a, b)
        return np.abs(out)
    return None


def error_map_vs_location(err_map, sensor_R, sensor_Z, R_axis=DEFAULT_R_AXIS, Z_axis=DEFAULT_Z_AXIS, region=None,
                          response_map=None) -> dict:
    """Relate a (65, 65) [Z, R] error map (e.g. mean |psi_pred - psi_true| with sensor s lost) to where the sensor sits.
    Returns Spearman(error, distance-to-sensor) (negative = error concentrates near the lost sensor), the distance between
    the error-weighted centroid and the sensor, and (if a response map is given) Spearman(error, |response|). `region`: bool
    (65, 65) mask of pixels to use (e.g. inside the limiter / plasma)."""
    RR, ZZ = np.meshgrid(R_axis, Z_axis)
    E = np.asarray(err_map, float)
    ok = np.isfinite(E) if region is None else (np.isfinite(E) & np.asarray(region, bool))
    out = dict(n_pixels=int(ok.sum()))
    if not (np.isfinite(sensor_R) and np.isfinite(sensor_Z)):
        out.update(spearman_distance=float("nan"), centroid_distance_m=float("nan"))
    else:
        d = np.hypot(RR - sensor_R, ZZ - sensor_Z)
        out["spearman_distance"] = float(stats.spearmanr(E[ok], d[ok])[0])
        w = E[ok]
        out["centroid_distance_m"] = float(math.hypot((w * RR[ok]).sum() / w.sum() - sensor_R, (w * ZZ[ok]).sum() / w.sum() - sensor_Z))
    if response_map is not None:
        out["spearman_response"] = float(stats.spearmanr(E[ok], np.asarray(response_map)[ok])[0])
    return out
