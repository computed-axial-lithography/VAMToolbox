"""Physics-aware evaluation of an optimized sinogram.

The optimizers (OSMO/BCLP) judge a sinogram by the *ideal* linear dose model
``dose = backprojection(sinogram)`` sampled exactly at the optimization angles.
The printer does not deliver that dose:

* the vial rotates **continuously** while each frame is displayed, so every frame
  is smeared over its angular dwell (``360 / n_angles`` degrees);
* free-radical acrylate resins do not obey exposure reciprocity — cure rate
  scales as ``I**beta`` with ``beta ~ 0.5-0.65`` (DeRosa 2021, Wydra 2014), so
  a bright frame delivers *less* effective dose per unit energy than a dim one;
* the delivered dose is blurred by radical/oxygen diffusion and by the projector
  point-spread function (Orth et al., Nat. Commun. 2023).

This module re-projects a sinogram through those effects so a candidate pattern
set can be scored by what would actually cure, and provides the dose-quality
metrics used by the Tomo GUI (process window, voxel error rate, IoU) plus a
kernel-identification helper (Ye et al. 2026: recover the diffusion kernel from
one uncorrected print by scanning kernel candidates for best overlap).
"""
import numpy as np

import vamtoolbox


# ───────────────────────────── dose under real physics ──────────────────────────
def _clone_proj_geo(proj_geo, angles):
    pg = vamtoolbox.geometry.ProjectionGeometry(
        angles=angles, ray_type=proj_geo.ray_type, CUDA=proj_geo.CUDA,
        absorption_coeff=proj_geo.absorption_coeff,
        container_radius=proj_geo.container_radius,
        projector_pixel_size=proj_geo.projector_pixel_size,
        inclination_angle=getattr(proj_geo, "inclination_angle", None))
    pg.sparse = getattr(proj_geo, "sparse", False)
    pg.dof = getattr(proj_geo, "dof", None)
    return pg


def with_measured_optics(proj_geo, voxel_mm, focal_offset_mm=0.0, n_resin=1.51, **dof):
    """Copy of proj_geo that projects through the measured depth-of-focus PSF
    (ray_type "dof"), e.g. to score a sinogram optimized with the ideal model under the
    optics it will actually meet:  evaluate_sinogram(sino, tg, with_measured_optics(pg, 0.05))."""
    pg = _clone_proj_geo(proj_geo, proj_geo.angles)
    pg.ray_type = "dof"
    pg.dof = dict(voxel_mm=voxel_mm, focal_offset_mm=focal_offset_mm, n_resin=n_resin, **dof)
    return pg


def dose_from_sinogram(sinogram, target_geo, proj_geo, sub_steps=1, beta=1.0,
                       kernel=None, normalize=False):
    """Delivered dose for a sinogram under the printer's physics.

    Parameters
    ----------
    sinogram : ndarray (nX, nA, nZ) or geometry.Sinogram
    target_geo, proj_geo : the geometry the sinogram was optimized for
    sub_steps : int   angular sub-positions per frame.  1 = the optimizer's
                      instantaneous model; k>1 averages the backprojection over
                      the frame's dwell ``[-dtheta/2, +dtheta/2)`` (continuous
                      rotation; frame timing assumed centred on the optimization
                      angle, i.e. a calibrated trigger).
    beta : float      reciprocity exponent: effective per-frame dose ~ I**beta.
                      1.0 = linear (ideal).  Applied per sinogram pixel, which is
                      exact for the per-frame dose a voxel receives.
    kernel : ndarray  optional 3D blur PSF (from response.blur_ker) applied to the
                      accumulated dose (diffusion + optical spread).
    normalize : bool  scale result to max = 1.
    """
    g = np.asarray(getattr(sinogram, "array", sinogram), dtype=np.float32)
    g = np.clip(g, 0.0, None)
    if beta != 1.0:
        g = np.power(g, np.float32(beta), dtype=np.float32)
    angles = np.asarray(proj_geo.angles, dtype=np.float64)
    n = angles.size
    dtheta = 360.0 / n
    dose = None
    k = max(1, int(sub_steps))
    for j in range(k):
        off = 0.0 if k == 1 else ((j + 0.5) / k - 0.5) * dtheta
        pg = proj_geo if k == 1 else _clone_proj_geo(proj_geo, angles + off)
        if pg.absorption_coeff is not None and getattr(pg, "absorption_mask", None) is None:
            pg.calcAbsorptionMask(target_geo)
        A = vamtoolbox.projectorconstructor.projectorconstructor(target_geo, pg)
        d = np.asarray(A.backward(g), dtype=np.float32).reshape(target_geo.array.shape)
        dose = d if dose is None else dose + d
    dose = dose / np.float32(k)
    if kernel is not None:
        dose = vamtoolbox.response._diffusion_convolve(dose, np.asarray(kernel, np.float32))
        dose = np.asarray(dose, dtype=np.float32)
    if normalize:
        m = float(dose.max())
        if m > 0:
            dose = dose / m
    return dose


# ───────────────────────────────── metrics ───────────────────────────────────────
def _inside_vial_mask(shape):
    nX, nY = shape[0], shape[1]
    yy, xx = np.ogrid[:nX, :nY]
    circ = ((xx - nX / 2.0) ** 2 + (yy - nY / 2.0) ** 2) <= (min(nX, nY) / 2.0) ** 2
    return circ[..., None] if len(shape) == 3 else circ


def dose_metrics(dose, target, threshold=None, n_levels=256):
    """Gel/void separation metrics on a normalized dose.

    Returns dict with: in_min, in_mean, out_max, out_mean, window (in_min-out_max),
    iou_best + threshold_best (IoU of the cured set vs the target over a threshold
    scan), and — if ``threshold`` is given — in_under_pct, out_over_pct, ver_pct,
    iou at that fixed gel threshold.  Out-of-part voxels are restricted to the
    inscribed circle (inside the vial), as in the Tomo GUI.
    """
    r = np.asarray(dose, dtype=np.float32)
    m = float(r.max())
    r = r / m if m > 0 else r
    tgt = np.asarray(target) > (0.5 * float(np.max(target)) if np.max(target) > 0 else 0.5)
    circ = _inside_vial_mask(r.shape)
    in_d = r[tgt]
    out_d = r[(~tgt) & circ]
    n_in, n_out = in_d.size, out_d.size
    res = {
        "in_min": float(in_d.min()) if n_in else 0.0,
        "in_mean": float(in_d.mean()) if n_in else 0.0,
        "out_max": float(out_d.max()) if n_out else 0.0,
        "out_mean": float(out_d.mean()) if n_out else 0.0,
    }
    res["window"] = res["in_min"] - res["out_max"]
    # IoU over a threshold scan (cured = dose >= t): |in>=t| / (|in| + |out>=t|)
    levels = np.linspace(0.0, 1.0, n_levels, endpoint=False)
    in_sorted = np.sort(in_d)
    out_sorted = np.sort(out_d)
    in_ge = n_in - np.searchsorted(in_sorted, levels, side="left")
    out_ge = n_out - np.searchsorted(out_sorted, levels, side="left")
    iou = in_ge / np.maximum(n_in + out_ge, 1)
    b = int(np.argmax(iou))
    res["iou_best"] = float(iou[b])
    res["threshold_best"] = float(levels[b])
    if threshold is not None:
        T = float(threshold)
        in_under = float((in_d < T).mean()) if n_in else 0.0
        out_over = float((out_d >= T).mean()) if n_out else 0.0
        res.update({
            "threshold": T,
            "in_under_pct": 100.0 * in_under,
            "out_over_pct": 100.0 * out_over,
            "ver_pct": 100.0 * (in_under * n_in + out_over * n_out) / max(1, n_in + n_out),
            "iou": float((in_d >= T).sum() / max(1, n_in + (out_d >= T).sum())),
        })
    return res


def evaluate_sinogram(sinogram, target_geo, proj_geo, threshold=None, **physics):
    """dose_from_sinogram + dose_metrics in one call.  ``physics`` = sub_steps,
    beta, kernel (see dose_from_sinogram)."""
    dose = dose_from_sinogram(sinogram, target_geo, proj_geo, **physics)
    return dose_metrics(dose, target_geo.array, threshold=threshold), dose


# ───────────────────────── kernel identification (Ye 2026) ───────────────────────
def volume_matched_threshold(dose, n_cured):
    """Dose level at which exactly ``n_cured`` voxels are at/above threshold."""
    flat = np.sort(np.asarray(dose, dtype=np.float32).reshape(-1))
    n_cured = int(np.clip(n_cured, 1, flat.size))
    return float(flat[flat.size - n_cured])


def identify_diffusion_D(dose, printed, pitch_mm, print_time_s, rotation_deg_s,
                         D_candidates, psf_size=19, optical=False, threshold_rule="volume",
                         **blur_kwargs):
    """Recover the resin's dose-diffusion coefficient from ONE uncorrected print.

    Given the dose the optimizer *thinks* it delivered (``dose``, e.g. the
    reconstruction of the printed sinogram) and the measured printed geometry
    (``printed``, binary from micro-CT / OST / a photo stack registered to the
    grid), scan candidate D values: blur the dose with blur_ker(D), threshold it,
    and pick the D whose predicted part best overlaps the real one.

    threshold_rule : "volume" (default) — the cure threshold is set so the
        predicted cured volume equals the measured printed volume.  This uses
        only the print itself and is what makes the scan discriminative: an
        under-blurred model keeps thin features that the real print lost, an
        over-blurred one loses features the print kept.  "best_iou" instead
        frees the threshold per candidate (much weaker discrimination).
    Returns (best_D, table) with rows (D, iou, threshold).
    """
    dose = np.asarray(dose, dtype=np.float32)
    printed = np.asarray(printed) > 0
    n_print = int(printed.sum())
    rows = []
    for D in D_candidates:
        ker = vamtoolbox.response.blur_ker(pitch_mm, float(D), print_time_s, rotation_deg_s,
                                           psf_size=psf_size, optical=optical, **blur_kwargs)
        eff = np.asarray(vamtoolbox.response._diffusion_convolve(dose, ker), dtype=np.float32)
        if threshold_rule == "volume":
            thr = volume_matched_threshold(eff, n_print)
            cured = eff >= thr
            iou = float((cured & printed).sum() / max(1, (cured | printed).sum()))
            rows.append((float(D), iou, thr / float(eff.max()) if eff.max() > 0 else 0.0))
        else:
            m = dose_metrics(eff, printed)
            rows.append((float(D), m["iou_best"], m["threshold_best"]))
    best = max(rows, key=lambda r: r[1])
    return best[0], rows
