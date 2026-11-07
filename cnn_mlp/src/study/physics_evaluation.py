#!/usr/bin/env python
"""Module 1 (v2): Physics-Aware Evaluation Layer for 2D tokamak equilibrium reconstruction.

Scores a predicted psi map in physical units (cm) instead of pixel NRMSE: plasma boundary, magnetic axis, X-points,
elongation / triangularity, and wall clearance.

Three modes of evaluate_window_physics(yt_2d, yp_2d, truth=None, mode=None):
  "efit"      (default when `truth` is given, see efit_truth.py): scored against TokaMark's EFIT outputs. psi_b is the flux of
              the TRUE field sampled along the EFIT LCFS; the predicted boundary is the loop of the predicted psi at psi_b
              around the EFIT axis; the axis, X-points, kappa and delta are compared with EFIT's own numbers.
  "heuristic" (default without `truth`): the original flux-fraction method (LCFS = psi_min + 0.35 * range, argmax axis,
              saddle search), but with the REAL grid geometry.
  "legacy"    the heuristic method with the old WRONG geometry (R 0.2..2.0 m, Z +-1.5 m), to reproduce the old numbers.

Grid geometry, read from the TokaMark zarr (equilibrium/major_radius, equilibrium/z; the model target is the raw 65x65
psi, no crop or resample): psi[z_row, r_col], R = 0.06 + 0.03 i m (0.06..1.98), Z = -2 + 0.0625 j m (-2..+2), i.e.
3.00 cm per column and 6.25 cm per row. The old constants (2.8125 / 4.6875 cm) were wrong by +6.7% / +25%.
"""
from collections import namedtuple

import numpy as np
import contourpy
from matplotlib.path import Path
from scipy.ndimage import map_coordinates, minimum_filter
from scipy.spatial import cKDTree

# ---- real geometry (verified against the zarr by test_physics_evaluation.test_geometry_matches_zarr) ----
GRID_SIZE = 65
R_MIN_M, R_MAX_M = 0.06, 1.98
Z_MIN_M, Z_MAX_M = -2.0, +2.0
DR_CM = (R_MAX_M - R_MIN_M) * 100.0 / (GRID_SIZE - 1)  # 3.00 cm / column
DZ_CM = (Z_MAX_M - Z_MIN_M) * 100.0 / (GRID_SIZE - 1)  # 6.25 cm / row
LEGACY_DR_CM, LEGACY_DZ_CM = 2.8125, 4.6875  # what v1 assumed (R 0.2..2.0 m, Z -1.5..+1.5 m): wrong

Geom = namedtuple("Geom", "r0 dr z0 dz")  # cm: R = r0 + dr * col, Z = z0 + dz * row
GEOM = Geom(R_MIN_M * 100.0, DR_CM, Z_MIN_M * 100.0, DZ_CM)
GEOM_LEGACY = Geom(0.0, LEGACY_DR_CM, 0.0, LEGACY_DZ_CM)  # v1 measured from the grid corner

# Wall clearance. The TokaMark zarr has NO vessel / limiter signal, so the real MAST wall is not available. PROXY: the
# edge of the EFIT computational grid (R 6..198 cm, Z -200..+200 cm). This is looser than the real vessel (the centre
# column is at R ~ 20 cm), so clearances are over-estimated; treat alarm rates as a protocol demonstration, not a safety
# claim. Assign a different (rmin, rmax, zmin, zmax) here when a real wall becomes available.
WALL_BOX_CM = (R_MIN_M * 100.0, R_MAX_M * 100.0, Z_MIN_M * 100.0, Z_MAX_M * 100.0)
WALL_BOX_LEGACY_CM = (0.0, 180.0, 0.0, 300.0)  # v1 box in corner-relative coordinates
SAFETY_CLEARANCE_CM = 5.0

LCFS_FRAC = 0.35  # heuristic mode only: psi_LCFS = psi_min + frac * (psi_max - psi_min)

# X-point = saddle (det H < 0, zero of grad psi within 1 px) whose flux is within XPOINT_PSI_TOL * range of psi_LCFS.
XPOINT_PSI_TOL = 0.15  # heuristic modes: their 35% level is not the separatrix flux, so the window is wide
# efit mode cuts at the true psi_b. Chosen on error-free EFIT psi (8 shots, 356 windows: 15% would flag 95% of the windows
# that have no X-point, 2% flags 3%); re-measured on 16 shots / 941 windows with xpoint_diagnostic.py: at 2% of the range
# 93.8% of windows with an EFIT X-point have a saddle within 10 cm and 6.4% of windows without one have a spurious saddle.
XPOINT_PSI_TOL_EFIT = 0.02
XPOINT_HIT_CM = 10.0  # a predicted X-point is a hit when within this distance of the true one
# A prediction with no contour at all at psi_b (flat, offset or inverted field) has no boundary to compare. It is NOT dropped
# from the means (that would reward the worst predictions): it is scored with this fixed penalty (cm), flagged
# boundary_failed, and counted in boundary_fail_rate_pct. The value is a convention (order of the plasma size), not a
# measurement.
BOUNDARY_FAIL_CM = 50.0
LEVEL_BUMPS = (0.0, 0.002, 0.005, 0.01, 0.02)  # fraction of the psi range tried when no closed loop exists at psi_b


def _geom(mode, truth=None):
    if mode == "legacy":
        return GEOM_LEGACY
    g = truth.get("grid") if truth else None
    return GEOM if not g else Geom(g["r0"] * 100.0, g["dr"] * 100.0, g["z0"] * 100.0, g["dz"] * 100.0)


def _to_cm(px, geom):
    px = np.asarray(px, dtype=np.float64)
    return np.column_stack([geom.r0 + px[:, 0] * geom.dr, geom.z0 + px[:, 1] * geom.dz])


def grid_to_physical_cm(r_idx, z_idx, geom=GEOM):
    """Convert (col, row) pixel indices (may be fractional) to (R, Z) in cm."""
    return geom.r0 + r_idx * geom.dr, geom.z0 + z_idx * geom.dz


def lcfs_level(field_2d, frac=LCFS_FRAC):
    """Heuristic psi_LCFS = psi_min + frac * (psi_max - psi_min) of a (true) field, or None for a flat field."""
    f_min, f_max = float(field_2d.min()), float(field_2d.max())
    return None if f_max - f_min < 1e-6 else f_min + frac * (f_max - f_min)


def extract_lcfs_contour(field_2d, frac=LCFS_FRAC, level=None, min_len=8, geom=GEOM):
    """Heuristic contour(s) at `level` (default: the 35% level of this field) as (R_cm, Z_cm) points, or None.

    All segments of at least `min_len` points are kept (two flux lobes give two loops; speckle is dropped). Pass the level
    of the TRUE field when scoring a prediction: psi_LCFS is a physical value.
    """
    if level is None:
        level = lcfs_level(field_2d, frac)
        if level is None:
            return None
    segs = contourpy.contour_generator(z=np.asarray(field_2d, dtype=np.float64)).lines(level)
    segs = [sg for sg in segs if len(sg) >= min_len]
    if not segs:
        return None
    return _to_cm(np.concatenate(segs), geom)


# ---------------------------------------------------------------- magnetic axis
def _refine_peak(f, z, r):
    """Separable 3-point parabola refinement of the maximum of `f` at integer (z, r), on the UNMASKED field."""
    r_sub = z_sub = 0.0
    if 0 < z < f.shape[0] - 1 and 0 < r < f.shape[1] - 1:
        dr_ = 2.0 * f[z, r] - f[z, r + 1] - f[z, r - 1]
        if abs(dr_) > 1e-12:
            r_sub = float(np.clip(0.5 * (f[z, r + 1] - f[z, r - 1]) / dr_, -0.5, 0.5))
        dz_ = 2.0 * f[z, r] - f[z + 1, r] - f[z - 1, r]
        if abs(dz_) > 1e-12:
            z_sub = float(np.clip(0.5 * (f[z + 1, r] - f[z - 1, r]) / dz_, -0.5, 0.5))
    return r + r_sub, z + z_sub


def get_subpixel_peak(field_2d, region=None, sign=1.0):
    """(r_px, z_px) of the extremum of the field (maximum of sign * field) with sub-pixel refinement.

    `region` is an optional boolean mask: the integer peak is searched inside it, but the sub-pixel refinement uses the
    unmasked field, so a peak on the region border is not distorted (identical fields give identical peaks).
    """
    f = sign * np.asarray(field_2d, dtype=np.float64)
    g = f if region is None else np.where(region, f, -np.inf)
    if not np.isfinite(g).any():
        g = f
    z, r = np.unravel_index(np.argmax(g), g.shape)
    return _refine_peak(f, z, r)


# ---------------------------------------------------------------- X-points
def find_xpoints(field_2d, level=None, psi_tol=XPOINT_PSI_TOL, side="lower"):
    """All X-points of psi as (r_px, z_px, psi, |det H|), best candidate first. side: "lower", "upper" or "both".

    X-point: grad(psi) = 0 and det(Hessian) < 0. Candidates are saddle pixels at a local minimum of |grad psi|, refined
    by one Newton step (rejected if the zero is more than one pixel away). With `level` (the LCFS flux) candidates whose
    flux differs from it by more than psi_tol * (psi_max - psi_min) are dropped and the rest ordered by closeness to it.
    Returns [] when there is none: it never invents one.
    """
    f = np.asarray(field_2d, dtype=np.float64)
    rng = float(f.max() - f.min())
    if rng < 1e-12:
        return []
    gz, gr = np.gradient(f)
    g2 = gr ** 2 + gz ** 2
    hrr = np.gradient(gr, axis=1)
    hzz = np.gradient(gz, axis=0)
    hrz = 0.5 * (np.gradient(gr, axis=0) + np.gradient(gz, axis=1))
    det = hrr * hzz - hrz ** 2
    cand = (g2 == minimum_filter(g2, size=3)) & (det < 0)
    cand[[0, -1], :] = False
    cand[:, [0, -1]] = False
    half = f.shape[0] // 2
    rows = np.arange(f.shape[0])[:, None]
    if side == "lower":
        cand &= rows < half
    elif side == "upper":
        cand &= rows > half
    found = []
    for z, r in zip(*np.nonzero(cand)):
        h = np.array([[hrr[z, r], hrz[z, r]], [hrz[z, r], hzz[z, r]]])
        try:
            d = -np.linalg.solve(h, np.array([gr[z, r], gz[z, r]]))
        except np.linalg.LinAlgError:
            continue
        if abs(d[0]) > 1.0 or abs(d[1]) > 1.0:
            continue
        rr, zz = r + d[0], z + d[1]
        psi = float(map_coordinates(f, [[zz], [rr]], order=1, mode="nearest")[0])
        if level is not None and abs(psi - level) > psi_tol * rng:
            continue
        found.append((float(rr), float(zz), psi, float(abs(det[z, r]))))
    found.sort(key=(lambda c: abs(c[2] - level)) if level is not None else (lambda c: -c[3]))
    return found


def find_divertor_xpoint(field_2d, level=None, psi_tol=XPOINT_PSI_TOL):
    """Best lower-divertor X-point as (r_px, z_px), or None."""
    xs = find_xpoints(field_2d, level=level, psi_tol=psi_tol, side="lower")
    return None if not xs else (xs[0][0], xs[0][1])


# ---------------------------------------------------------------- heuristic / legacy metrics
def _densify(poly_cm, step=0.5):
    """Resample a polyline to ~`step` cm spacing so point-to-point distances approximate point-to-curve distances."""
    p = np.asarray(poly_cm, dtype=np.float64)
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return p
    u = np.arange(0.0, s[-1], step)
    return np.column_stack([np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])])


def curve_distances(a_cm, b_cm):
    """(symmetric mean, Hausdorff) distance in cm between two curves (arrays of (R, Z) cm), densified to 0.5 cm."""
    a, b = _densify(a_cm), _densify(b_cm)
    d_ab = cKDTree(b).query(a)[0]
    d_ba = cKDTree(a).query(b)[0]
    return 0.5 * float(d_ab.mean() + d_ba.mean()), float(max(d_ab.max(), d_ba.max()))


def compute_boundary_error(yt_2d, yp_2d, frac=LCFS_FRAC, geom=GEOM):
    """Heuristic mean and Hausdorff boundary error (cm) between the 35% contours of truth and prediction.

    Both are cut at the flux level of the TRUE field. Falls back to a pixel-error based estimate when a contour is missing.
    """
    level = lcfs_level(yt_2d, frac)
    ct = extract_lcfs_contour(yt_2d, frac=frac, geom=geom)
    cp = None if level is None else extract_lcfs_contour(yp_2d, frac=frac, level=level, geom=geom)
    if ct is None or cp is None:
        diff = np.abs(yp_2d - yt_2d)
        err_px = float(diff.mean()) / (float(yt_2d.max() - yt_2d.min()) + 1e-6) * 10.0
        return err_px * geom.dr, err_px * 2.0 * geom.dz
    d_t_to_p = cKDTree(cp).query(ct)[0]
    d_p_to_t = cKDTree(ct).query(cp)[0]
    return 0.5 * float(d_t_to_p.mean() + d_p_to_t.mean()), float(max(d_t_to_p.max(), d_p_to_t.max()))


def compute_axis_shift(yt_2d, yp_2d, geom=GEOM):
    """Heuristic axis shift (cm): argmax of the prediction vs argmax of the truth, sub-pixel.

    A two-lobe field has two near-equal peaks and the prediction may favour the other one, so the predicted peak is
    searched in the same vertical half (+3 rows over the midline) as the true peak. The sub-pixel step uses the unmasked prediction, so identical
    fields give exactly 0.
    """
    rt, zt = get_subpixel_peak(yt_2d)
    half = yt_2d.shape[0] // 2
    region = np.zeros(np.shape(yp_2d), dtype=bool)
    if zt < half:  # the half of the true peak plus 3 rows over the midline, so a peak straddling it is not cut in two
        region[:half + 3] = True
    else:
        region[half - 3:] = True
    rp, zp = get_subpixel_peak(yp_2d, region=region)
    return float(np.hypot((rp - rt) * geom.dr, (zp - zt) * geom.dz)), (rt, zt), (rp, zp)


def compute_xpoint_error(yt_2d, yp_2d, frac=LCFS_FRAC, geom=GEOM):
    """Heuristic lower X-point comparison: (error_cm or nan, true_found, pred_found). Both fields at the TRUE 35% level."""
    level = lcfs_level(yt_2d, frac)
    xt = find_divertor_xpoint(yt_2d, level=level)
    xp = None if level is None else find_divertor_xpoint(yp_2d, level=level)
    if xt is None or xp is None:
        return float("nan"), xt is not None, xp is not None
    return float(np.hypot((xp[0] - xt[0]) * geom.dr, (xp[1] - xt[1]) * geom.dz)), True, True


# ---------------------------------------------------------------- loops and shape
def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def loop_around(field_2d, level, point_px):
    """Largest closed contour (pixel coords) of `field_2d` at `level` that encloses `point_px` = (r_px, z_px), or None."""
    best, best_a = None, 0.0
    for sg in contourpy.contour_generator(z=np.asarray(field_2d, dtype=np.float64)).lines(level):
        if len(sg) < 8 or not np.allclose(sg[0], sg[-1]) or not Path(sg).contains_point(point_px):
            continue
        a = _area(sg)
        if a > best_a:
            best, best_a = sg, a
    return best


def loop_around_bumped(field_2d, level, point_px, sign=1.0):
    """loop_around at `level`; if none exists, retry slightly inside (towards the axis). Returns (loop_px, bump_fraction)."""
    rng = float(np.max(field_2d) - np.min(field_2d))
    for b in LEVEL_BUMPS:
        lp = loop_around(field_2d, level + sign * b * rng, point_px)
        if lp is not None:
            return lp, b
    return None, float("nan")


def main_lcfs_loop(field_2d, level, geom=GEOM):
    """Heuristic: outermost closed contour at `level` that encloses the argmax axis, in cm (R, Z), or None."""
    if level is None:
        return None
    f = np.asarray(field_2d, dtype=np.float64)
    lp = loop_around(f, level, get_subpixel_peak(f))
    return None if lp is None else _to_cm(lp, geom)


def _vertex_r(r, z, top, refine):
    """R of the highest (top) / lowest point of a boundary; `refine` fits a parabola z(r) to the points near the extreme."""
    zi = np.argmax(z) if top else np.argmin(z)
    if not refine:
        return r[zi]
    zs = z.max() if top else z.min()
    near = np.abs(z - zs) <= 0.06 * (z.max() - z.min())
    if near.sum() >= 4 and np.ptp(r[near]) > 1e-6:
        c = np.polyfit(r[near], z[near], 2)
        if c[0] * (-1 if top else 1) > 0:  # opens the right way: a rounded top / bottom
            rv = -c[1] / (2 * c[0])
            if r[near].min() <= rv <= r[near].max():
                return rv
    return r[zi]


def shape_parameters(loop_cm, refine=False):
    """Boundary shape of a (R, Z) polyline in cm, or None. EFIT / IMAS definitions:
    a = (Rmax - Rmin) / 2, R_geo = (Rmax + Rmin) / 2, kappa = (Zmax - Zmin) / (2a),
    delta_upper = (R_geo - R(Zmax)) / a, delta_lower = (R_geo - R(Zmin)) / a, delta = their mean.
    dict(kappa, delta_upper, delta_lower, delta, a_cm, r_geo_cm). On EFIT's own LCFS polygon this reproduces TokaMark's
    elongation, triangularity_upper/lower and minor_radius to 1e-4 (checked on 356 windows), so these ARE EFIT's
    definitions. `refine` (parabola at the extremes) is NOT EFIT's definition and makes it worse (bias -0.045).
    """
    if loop_cm is None or len(loop_cm) < 4:
        return None
    r, z = loop_cm[:, 0], loop_cm[:, 1]
    a = 0.5 * (r.max() - r.min())
    if a < 1e-6:
        return None
    r0 = 0.5 * (r.max() + r.min())
    du = (r0 - _vertex_r(r, z, True, refine)) / a
    dl = (r0 - _vertex_r(r, z, False, refine)) / a
    return dict(kappa=float((z.max() - z.min()) / (2.0 * a)), delta_upper=float(du), delta_lower=float(dl),
                delta=float(0.5 * (du + dl)), a_cm=float(a), r_geo_cm=float(r0))


def compute_shape_error(yt_2d, yp_2d, frac=LCFS_FRAC, geom=GEOM):
    """Heuristic (kappa_true, delta_true, kappa_pred, delta_pred); nan where no closed loop around the axis exists."""
    level = lcfs_level(yt_2d, frac)
    st = shape_parameters(main_lcfs_loop(yt_2d, level, geom))
    sp = shape_parameters(main_lcfs_loop(yp_2d, level, geom))
    nan = float("nan")
    return (st["kappa"] if st else nan, st["delta"] if st else nan, sp["kappa"] if sp else nan, sp["delta"] if sp else nan)


def compute_wall_proximity(contour_cm, wall=None):
    """Minimum clearance (cm) from a boundary to the wall box `wall` = (rmin, rmax, zmin, zmax). See WALL_BOX_CM: a PROXY."""
    if contour_cm is None:
        return 50.0  # safe default if no boundary
    rmin, rmax, zmin, zmax = WALL_BOX_CM if wall is None else wall
    r_pts, z_pts = contour_cm[:, 0], contour_cm[:, 1]
    return max(0.0, float(min(r_pts.min() - rmin, rmax - r_pts.max(), z_pts.min() - zmin, zmax - z_pts.max())))


# ---------------------------------------------------------------- EFIT-grounded scoring
def _px_of(rz_m, g):
    """EFIT (R, Z) in metres (n, 2) -> pixel (col, row) given grid dict g (metres)."""
    rz = np.atleast_2d(np.asarray(rz_m, dtype=np.float64))
    return np.column_stack([(rz[:, 0] - g["r0"]) / g["dr"], (rz[:, 1] - g["z0"]) / g["dz"]])


def _sample(f, px):
    return map_coordinates(f, [px[:, 1], px[:, 0]], order=1, mode="nearest")


def _region_mask(poly_px, shape):
    cc, rr = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    return Path(poly_px).contains_points(np.column_stack([cc.ravel(), rr.ravel()])).reshape(shape)


def efit_psi_b(yt_2d, lcfs_px):
    """(psi_b, spread_pct): median flux of the true field along the EFIT LCFS, and std(flux) as % of the field's range.

    The spread is a self-consistency check: on the real EFIT psi it is ~0.01% (the LCFS IS a flux contour); a large
    value means the grid / orientation / units do not match the truth.
    """
    v = _sample(np.asarray(yt_2d, dtype=np.float64), lcfs_px)
    rng = float(np.max(yt_2d) - np.min(yt_2d)) + 1e-12
    return float(np.median(v)), float(100.0 * np.std(v) / rng)


def _nan_shape():
    return dict(kappa=np.nan, delta_upper=np.nan, delta_lower=np.nan, delta=np.nan)


def _efit_field_scores(f, truth, geom, g, psi_b, sign, axis_px, lcfs_cm, region, xp_px):
    """Boundary / axis / shape / X-point scores of ONE field (prediction or true psi) against EFIT."""
    loop_px, bump = loop_around_bumped(f, psi_b, axis_px, sign) if axis_px is not None else (None, np.nan)
    closed = loop_px is not None
    if closed:
        loop_cm = _to_cm(loop_px, geom)
    else:  # no closed loop: use whatever segments exist at psi_b, so the error is large and not silently missing
        loop_cm = extract_lcfs_contour(f, level=psi_b, geom=geom)
    failed = False
    if loop_cm is not None and lcfs_cm is not None:
        b_mean, b_haus = curve_distances(lcfs_cm, loop_cm)
    elif lcfs_cm is not None:  # nothing at psi_b in this field
        b_mean = b_haus = BOUNDARY_FAIL_CM
        failed = True
    else:
        b_mean = b_haus = float("nan")
    rp, zp = get_subpixel_peak(f, region=region, sign=sign)
    axis_cm = float("nan")
    if axis_px is not None:
        axis_cm = float(np.hypot((rp - axis_px[0]) * geom.dr, (zp - axis_px[1]) * geom.dz))
    shp = shape_parameters(loop_cm) if closed else None
    cands = find_xpoints(f, level=psi_b, psi_tol=XPOINT_PSI_TOL_EFIT, side="both")
    xd = [min([np.hypot((c[0] - x[0]) * geom.dr, (c[1] - x[1]) * geom.dz) for c in cands], default=np.inf) for x in xp_px]
    return dict(loop_cm=loop_cm, closed=closed, failed=failed, bump=bump, b_mean=b_mean, b_haus=b_haus, axis_cm=axis_cm,
                shape=shp or _nan_shape(), xcands=cands, xdist=xd)


def _evaluate_efit(yt, yp, truth, frac):
    geom = _geom("efit", truth)
    g = truth.get("grid") or dict(r0=R_MIN_M, dr=DR_CM / 100.0, z0=Z_MIN_M, dz=DZ_CM / 100.0)
    nan = float("nan")
    lcfs = truth.get("lcfs_rz_m")
    axis_rz = np.asarray(truth.get("axis_rz_m", [nan, nan]), dtype=np.float64)
    xp_rz = np.asarray(truth.get("xpoints_rz_m", np.zeros((0, 2))), dtype=np.float64).reshape(-1, 2)
    have_lcfs = lcfs is not None and len(lcfs) >= 8
    have_axis = bool(np.all(np.isfinite(axis_rz)))
    lcfs_px = _px_of(lcfs, g) if have_lcfs else None
    axis_px = _px_of(axis_rz, g)[0] if have_axis else None
    xp_px = _px_of(xp_rz, g) if len(xp_rz) else np.zeros((0, 2))

    if have_lcfs:
        psi_b, spread = efit_psi_b(yt, lcfs_px)
    else:  # no EFIT LCFS for this slice: no physical level, boundary / shape scores are undefined (nan), counted in aggregate
        psi_b, spread = nan, nan
    sign = 1.0
    if have_axis and np.isfinite(psi_b):
        sign = 1.0 if float(_sample(yt, axis_px[None])[0]) >= psi_b else -1.0
    elif have_axis:
        sign = 1.0 if float(_sample(yt, axis_px[None])[0]) >= float(np.median(yt)) else -1.0
    region = _region_mask(lcfs_px, yt.shape) if have_lcfs and not (~_region_mask(lcfs_px, yt.shape)).all() else None
    if region is not None and not region.any():
        region = None
    lcfs_cm = _to_cm(lcfs_px, geom) if have_lcfs else None

    if np.isfinite(psi_b):
        sp = _efit_field_scores(yp, truth, geom, g, psi_b, sign, axis_px, lcfs_cm, region, xp_px)
        so = _efit_field_scores(yt, truth, geom, g, psi_b, sign, axis_px, lcfs_cm, region, xp_px)
    else:  # axis only
        sp = dict(loop_cm=None, closed=False, failed=False, bump=nan, b_mean=nan, b_haus=nan, shape=_nan_shape(), xcands=[], xdist=[np.inf] * len(xp_px),
                  axis_cm=nan)
        so = dict(sp)
        if have_axis:
            for s, f in ((sp, yp), (so, yt)):
                rp, zp = get_subpixel_peak(f, region=region, sign=sign)
                s["axis_cm"] = float(np.hypot((rp - axis_px[0]) * geom.dr, (zp - axis_px[1]) * geom.dz))

    # wall clearance: truth = EFIT LCFS polygon, prediction = predicted loop (PROXY wall, see WALL_BOX_CM)
    d_true = compute_wall_proximity(lcfs_cm) if have_lcfs else nan
    d_pred = compute_wall_proximity(sp["loop_cm"]) if np.isfinite(psi_b) else nan
    fa = bool(np.isfinite(d_true) and np.isfinite(d_pred) and d_pred < SAFETY_CLEARANCE_CM <= d_true)
    ma = bool(np.isfinite(d_true) and np.isfinite(d_pred) and d_true < SAFETY_CLEARANCE_CM <= d_pred)

    # X-points: the primary EFIT X-point is the one whose true-field flux is closest to psi_b (the active separatrix)
    k = len(xp_px)
    if k and np.isfinite(psi_b):
        prim = int(np.argmin(np.abs(_sample(yt, xp_px) - psi_b)))
        x_err = sp["xdist"][prim]
        x_err = nan if not np.isfinite(x_err) else float(x_err)
        x_or = so["xdist"][prim]
        x_or = nan if not np.isfinite(x_or) else float(x_or)
        n_hit = int(sum(d <= XPOINT_HIT_CM for d in sp["xdist"]))
    else:
        x_err = x_or = nan
        n_hit = 0
    kt = truth.get("kappa", nan)
    dt = truth.get("delta", nan)
    return dict(
        boundary_mean_cm=sp["b_mean"], boundary_hausdorff_cm=sp["b_haus"], axis_shift_cm=sp["axis_cm"],
        xpoint_err_cm=x_err, xpoint_true_found=bool(k), xpoint_pred_found=bool(len(sp["xcands"])),
        kappa_true=float(kt), kappa_pred=sp["shape"]["kappa"], kappa_err=abs(sp["shape"]["kappa"] - kt),
        delta_true=float(dt), delta_pred=sp["shape"]["delta"], delta_err=abs(sp["shape"]["delta"] - dt),
        d_wall_true_cm=d_true if np.isfinite(d_true) else 50.0, d_wall_pred_cm=d_pred if np.isfinite(d_pred) else 50.0,
        is_false_alarm=fa, is_missed_alarm=ma,
        mode="efit", psi_b=psi_b, psi_b_spread_pct=spread, efit_lcfs_available=bool(have_lcfs),
        boundary_closed=bool(sp["closed"]), boundary_failed=bool(sp["failed"]), level_bump=sp["bump"],
        delta_upper_pred=sp["shape"]["delta_upper"], delta_lower_pred=sp["shape"]["delta_lower"],
        delta_upper_true=float(truth.get("delta_upper", nan)), delta_lower_true=float(truth.get("delta_lower", nan)),
        xpoint_n_true=int(k), xpoint_n_hit=n_hit, xpoint_oracle_err_cm=x_or, xpoint_true_field_found=bool(len(so["xcands"])),
        boundary_mean_cm_oracle=so["b_mean"], boundary_hausdorff_cm_oracle=so["b_haus"], axis_shift_cm_oracle=so["axis_cm"],
        kappa_oracle=so["shape"]["kappa"], kappa_err_oracle=abs(so["shape"]["kappa"] - kt),
        delta_oracle=so["shape"]["delta"], delta_err_oracle=abs(so["shape"]["delta"] - dt),
        kappa_err_field=abs(sp["shape"]["kappa"] - so["shape"]["kappa"]), delta_err_field=abs(sp["shape"]["delta"] - so["shape"]["delta"]),
    )


def _evaluate_heuristic(yt, yp, frac, mode):
    geom = _geom(mode)
    wall = WALL_BOX_LEGACY_CM if mode == "legacy" else WALL_BOX_CM
    nan = float("nan")
    b_mean_cm, b_hausdorff_cm = compute_boundary_error(yt, yp, frac=frac, geom=geom)
    axis_shift_cm, _, _ = compute_axis_shift(yt, yp, geom=geom)
    xpoint_err_cm, xt_found, xp_found = compute_xpoint_error(yt, yp, frac=frac, geom=geom)
    kt, dt, kp, dp = compute_shape_error(yt, yp, frac=frac, geom=geom)
    level = lcfs_level(yt, frac)
    ct = extract_lcfs_contour(yt, frac=frac, geom=geom)
    cp = None if level is None else extract_lcfs_contour(yp, frac=frac, level=level, geom=geom)
    d_true, d_pred = compute_wall_proximity(ct, wall), compute_wall_proximity(cp, wall)
    return dict(
        boundary_mean_cm=b_mean_cm, boundary_hausdorff_cm=b_hausdorff_cm, axis_shift_cm=axis_shift_cm,
        xpoint_err_cm=xpoint_err_cm, xpoint_true_found=bool(xt_found), xpoint_pred_found=bool(xp_found),
        kappa_true=kt, kappa_pred=kp, kappa_err=abs(kp - kt), delta_true=dt, delta_pred=dp, delta_err=abs(dp - dt),
        d_wall_true_cm=d_true, d_wall_pred_cm=d_pred,
        is_false_alarm=bool(d_pred < SAFETY_CLEARANCE_CM <= d_true),  # predicted breach while the truth was safe
        is_missed_alarm=bool(d_true < SAFETY_CLEARANCE_CM <= d_pred),  # predicted safe while the truth was a breach
        mode=mode, psi_b=level if level is not None else nan, psi_b_spread_pct=nan, efit_lcfs_available=False,
        boundary_closed=bool(cp is not None),
    )


def evaluate_window_physics(yt_2d, yp_2d, frac=LCFS_FRAC, truth=None, mode=None):
    """All physical metrics for one 65x65 window (arrays [z_row, r_col], any affine unit of psi, same unit for both).

    truth: dict from efit_truth.TruthIndex.get(shot_id, window_index) / efit_truth.window_truth. mode: "efit" (needs
    truth, default when given), "heuristic" (default otherwise) or "legacy"; see the module docstring.
    Distances in cm; missing quantities are nan (never a fabricated number), counted by aggregate_physics_metrics.
    """
    yt = np.asarray(yt_2d, dtype=np.float64)
    yp = np.asarray(yp_2d, dtype=np.float64)
    mode = mode or ("efit" if truth is not None else "heuristic")
    if mode == "efit":
        if truth is None:
            raise ValueError("mode='efit' needs the `truth` dict (efit_truth.TruthIndex.get)")
        return _evaluate_efit(yt, yp, truth, frac)
    if mode not in ("heuristic", "legacy"):
        raise ValueError(f"unknown mode {mode!r}")
    return _evaluate_heuristic(yt, yp, frac, mode)


# ---------------------------------------------------------------- aggregation
def _nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    return float(x.mean()) if x.size else float("nan")


def _col(rs, k):
    return np.array([r.get(k, np.nan) for r in rs], dtype=np.float64)


def aggregate_physics_metrics(results_list):
    """Mean / std across window records. Metrics are averaged over the windows where they exist; counts are returned.

    boundary_* and axis_shift_cm are nan-means (windows without an EFIT LCFS have no boundary score: n_boundary_valid says
    how many do). Predictions with no contour at psi_b count BOUNDARY_FAIL_CM (boundary_fail_rate_pct;
    boundary_mean_cm_valid_only excludes them); boundary_closed_rate_pct is the share of valid windows with a closed
    predicted loop around the axis, and kappa / delta errors exist only for those.
    xpoint_detect_rate_pct: of the windows whose TRUTH has an X-point, the share whose prediction has one within
    XPOINT_HIT_CM of it. xpoint_false_rate_pct: of the windows without a true X-point, the share where the prediction has
    a saddle at the LCFS flux anyway (xpoint_false_rate_truefield_pct is the same detector on the error-free psi, the
    floor of that number). *_oracle keys score the TRUE psi against EFIT (the measurement floor of the method).
    """
    if not results_list:
        return {}
    rs = results_list
    n = len(rs)
    b_means, b_haus, axes, xpoints = (_col(rs, k) for k in ("boundary_mean_cm", "boundary_hausdorff_cm", "axis_shift_cm", "xpoint_err_cm"))
    d_true = _col(rs, "d_wall_true_cm")
    n_safe = int(np.sum(d_true >= SAFETY_CLEARANCE_CM))
    n_hazard = int(np.sum(d_true < SAFETY_CLEARANCE_CM))
    false_alarms = sum(1 for r in rs if r["is_false_alarm"])
    missed_alarms = sum(1 for r in rs if r["is_missed_alarm"])
    valid_b = ~np.isnan(b_means)

    def mean_std(a):
        a = a[~np.isnan(a)]
        return (float(a.mean()), float(a.std())) if a.size else (float("nan"), float("nan"))

    out = {}
    if all("xpoint_true_found" in r for r in rs):
        xt = np.array([r["xpoint_true_found"] for r in rs], bool)
        xpf = np.array([r["xpoint_pred_found"] for r in rs], bool)
        hit = xt & xpf & (xpoints <= XPOINT_HIT_CM)
        out.update(
            xpoint_true_count=int(xt.sum()),
            xpoint_detect_rate_pct=float(100.0 * hit.sum() / xt.sum()) if xt.any() else float("nan"),
            xpoint_false_rate_pct=float(100.0 * (~xt & xpf).sum() / (~xt).sum()) if (~xt).any() else float("nan"),
            xpoint_err_n=int(np.sum(~np.isnan(xpoints))),
            kappa_err=_nanmean(_col(rs, "kappa_err")),
            kappa_err_rel_pct=_nanmean(100.0 * _col(rs, "kappa_err") / _col(rs, "kappa_true")),
            delta_err=_nanmean(_col(rs, "delta_err")),
            kappa_true_mean=_nanmean(_col(rs, "kappa_true")), delta_true_mean=_nanmean(_col(rs, "delta_true")),
            shape_valid_n=int(np.sum(~np.isnan(_col(rs, "kappa_err")))),
        )
    if any(r.get("mode") == "efit" for r in rs):
        ef = [r for r in rs if r.get("mode") == "efit"]
        xt = np.array([r["xpoint_true_found"] for r in ef], bool)
        xo = np.array([r["xpoint_true_field_found"] for r in ef], bool)
        out.update(
            n_efit_windows=len(ef),
            n_boundary_valid=int(valid_b.sum()),
            boundary_closed_rate_pct=float(100.0 * np.mean([r["boundary_closed"] for r in ef if not np.isnan(r["boundary_mean_cm"])]))
            if valid_b.any() else float("nan"),
            boundary_fail_rate_pct=float(100.0 * np.mean([r["boundary_failed"] for r in ef if not np.isnan(r["boundary_mean_cm"])]))
            if valid_b.any() else float("nan"),
            boundary_mean_cm_valid_only=_nanmean([r["boundary_mean_cm"] for r in ef if not r["boundary_failed"]]),
            psi_b_spread_pct_mean=_nanmean(_col(ef, "psi_b_spread_pct")),
            psi_b_spread_pct_max=float(np.nanmax(_col(ef, "psi_b_spread_pct"))) if np.isfinite(_col(ef, "psi_b_spread_pct")).any() else float("nan"),
            xpoint_false_rate_truefield_pct=float(100.0 * (~xt & xo).sum() / (~xt).sum()) if (~xt).any() else float("nan"),
            xpoint_oracle_err_cm=_nanmean(_col(ef, "xpoint_oracle_err_cm")),
            boundary_mean_cm_oracle=_nanmean(_col(ef, "boundary_mean_cm_oracle")),
            boundary_hausdorff_cm_oracle=_nanmean(_col(ef, "boundary_hausdorff_cm_oracle")),
            axis_shift_cm_oracle=_nanmean(_col(ef, "axis_shift_cm_oracle")),
            kappa_err_oracle=_nanmean(_col(ef, "kappa_err_oracle")), delta_err_oracle=_nanmean(_col(ef, "delta_err_oracle")),
            kappa_err_field=_nanmean(_col(ef, "kappa_err_field")), delta_err_field=_nanmean(_col(ef, "delta_err_field")),
        )
    bm, bs = mean_std(b_means)
    am, asd = mean_std(axes)
    hs = b_haus[~np.isnan(b_haus)]
    return dict(
        n_windows=n,
        boundary_mean_cm=bm, boundary_mean_cm_std=bs,
        boundary_hausdorff_cm=_nanmean(b_haus), boundary_hausdorff_cm_p95=float(np.percentile(hs, 95)) if hs.size else float("nan"),
        axis_shift_cm=am, axis_shift_cm_std=asd,
        xpoint_err_cm=_nanmean(xpoints),
        false_alarm_count=false_alarms, false_alarm_rate_pct=float(false_alarms / max(1, n_safe) * 100.0),
        missed_alarm_count=missed_alarms, missed_alarm_rate_pct=float(missed_alarms / max(1, n_hazard) * 100.0),
        **out,
    )


if __name__ == "__main__":
    # Self-test on a synthetic Gaussian: a one-pixel shift in R is DR_CM = 3.0 cm
    rr, zz = np.meshgrid(np.linspace(0, 1, GRID_SIZE), np.linspace(0, 1, GRID_SIZE))
    y_true = np.exp(-((rr - 0.5) ** 2 + (zz - 0.5) ** 2) / 0.05)
    y_pred = np.exp(-((rr - 0.5 - 1.0 / 64) ** 2 + (zz - 0.5) ** 2) / 0.05)
    for k, v in evaluate_window_physics(y_true, y_pred).items():
        print(f"  {k:25}: {v}")
