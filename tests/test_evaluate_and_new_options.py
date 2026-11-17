"""Physics-aware evaluation module + the new optimizer options (PIHE, dose-rate
beta, in-loop diffusion mode).  CPU sparse projector so the tests stay portable."""
import numpy as np
import pytest

import vamtoolbox
from vamtoolbox import evaluate
from vamtoolbox.pipeline import PrintConfig, VAMPipeline


def _osmo(tg, pg, **kw):
    o = vamtoolbox.optimize.Options(method="OSMO", n_iter=6, d_h=0.85, d_l=0.65,
                                    filter="hanning", verbose="off", **kw)
    return vamtoolbox.optimize.optimize(tg, pg, o)


def _bclp(tg, pg, **kw):
    rm = vamtoolbox.response.ResponseModel(type="analytical", form="identity")
    o = vamtoolbox.optimize.Options(method="BCLP", response_model=rm, eps=0.1, weight=1,
                                    n_iter=6, d_h=0.85, d_l=0.65, learning_rate=0.005,
                                    verbose="off", **kw)
    return vamtoolbox.optimize.optimize(tg, pg, o)


def test_dose_from_sinogram_matches_optimizer_recon(cylinder_target, proj_geo_factory):
    sino, recon, _ = _osmo(cylinder_target, proj_geo_factory())
    d = evaluate.dose_from_sinogram(sino.array, cylinder_target, proj_geo_factory())
    a, b = d / d.max(), recon.array / recon.array.max()
    assert np.abs(a - b).max() < 1e-5


def test_physics_variants_run_and_degrade_gracefully(cylinder_target, proj_geo_factory):
    sino, _, _ = _osmo(cylinder_target, proj_geo_factory())
    m_ideal, _ = evaluate.evaluate_sinogram(sino.array, cylinder_target, proj_geo_factory(), threshold=0.65)
    m_cont, _ = evaluate.evaluate_sinogram(sino.array, cylinder_target, proj_geo_factory(), threshold=0.65, sub_steps=3)
    m_beta, _ = evaluate.evaluate_sinogram(sino.array, cylinder_target, proj_geo_factory(), threshold=0.65, beta=0.6)
    ker = vamtoolbox.response.blur_ker(0.5, 1e-4, 60, 40, psf_size=7)
    m_blur, _ = evaluate.evaluate_sinogram(sino.array, cylinder_target, proj_geo_factory(), threshold=0.65, kernel=ker)
    for m in (m_ideal, m_cont, m_beta, m_blur):
        assert 0.0 <= m["iou_best"] <= 1.0 and np.isfinite(m["window"])
        assert 0.0 <= m["ver_pct"] <= 100.0
    # (no ordering assertion: at coarse angular sampling, sub-stepping acts like
    #  extra angles and can legitimately reduce streak artefacts)


def test_dose_metrics_perfect_and_worst_case():
    tgt = np.zeros((8, 8, 8), np.uint8); tgt[2:6, 2:6, 2:6] = 1
    perfect = tgt.astype(np.float32)
    m = evaluate.dose_metrics(perfect, tgt, threshold=0.5)
    assert m["iou_best"] == pytest.approx(1.0) and m["window"] > 0 and m["ver_pct"] == 0.0
    flat = np.ones_like(perfect)
    m2 = evaluate.dose_metrics(flat, tgt)
    assert m2["window"] <= 0.0 and m2["iou_best"] < 1.0


def test_identify_diffusion_D_recovers_kernel():
    # fine-feature phantom: thin plates so blur width is discriminative
    arr = np.zeros((40, 40, 60), np.uint8)
    for z0 in (10, 25, 40, 52):
        arr[8:32, 8:32, z0:z0 + 2] = 1
    dose = arr.astype(np.float32)
    D_true = 1.5e-4
    ker = vamtoolbox.response.blur_ker(0.08, D_true, 60, 40, psf_size=13)
    eff = np.asarray(vamtoolbox.response._diffusion_convolve(dose, ker), np.float32)
    # a real uncorrected print stops when the BULK cures: threshold = the dose the
    # thickest feature's interior reaches -> thin plates come out thin/missing
    thick = np.zeros_like(arr); thick[8:32, 8:32, 10:12] = 1
    printed = eff >= float(eff[thick > 0].min())
    assert printed.sum() < arr.sum()                    # blur really lost material
    best, table = evaluate.identify_diffusion_D(dose, printed, 0.08, 60, 40,
                                                [3e-5, 8e-5, 1.5e-4, 3e-4, 6e-4], psf_size=13)
    assert best == pytest.approx(D_true), table


def test_osmo_pihe_options_run(cylinder_target, proj_geo_factory):
    s0, r0, _ = _osmo(cylinder_target, proj_geo_factory())
    s1, r1, _ = _osmo(cylinder_target, proj_geo_factory(), k_i=0.5, hist_eq=True)
    assert np.isfinite(s1.array).all() and s1.array.max() <= 1.0 + 1e-6
    assert not np.allclose(s0.array, s1.array)          # the options actually change the result
    m = evaluate.dose_metrics(r1.array, cylinder_target.array, threshold=0.65)
    assert m["iou_best"] > 0.9


def test_bclp_dose_rate_beta_runs_and_differs(cylinder_target, proj_geo_factory):
    s1, r1, loss1 = _bclp(cylinder_target, proj_geo_factory())
    s6, r6, loss6 = _bclp(cylinder_target, proj_geo_factory(), dose_rate_beta=0.6)
    assert np.isfinite(loss6).all() and np.isfinite(s6.array).all()
    assert loss6[-1] < loss6[0]                          # descends under the nonlinear model
    assert not np.allclose(s1.array, s6.array)
    # the returned reconstruction must be the beta-model dose
    d6 = evaluate.dose_from_sinogram(s6.array, cylinder_target, proj_geo_factory(), beta=0.6)
    assert np.abs(d6 / d6.max() - r6.array / r6.array.max()).max() < 1e-5


def test_pipeline_inloop_diffusion_and_new_config_fields(cylinder_array):
    n = cylinder_array.shape[2]
    base = dict(part_height_mm=25.4, voxel_pitch_um=25.4 / n * 1000, use_cuda=False,
                slab="off", n_angles=60, n_iterations=4, absorption=False)
    cfg = PrintConfig(method="BCLP", diffusion=True, diffusion_mode="inloop", **base)
    res = VAMPipeline(cfg).run(target_array=cylinder_array, do_rebin=False)
    assert np.isfinite(res.sinogram).all() and res.quality["contrast"] > 0
    cfg2 = PrintConfig(method="OSMO", osmo_k_i=0.5, osmo_hist_eq=True, **base)
    res2 = VAMPipeline(cfg2).run(target_array=cylinder_array, do_rebin=False)
    assert res2.quality["contrast"] > 0
    cfg3 = PrintConfig(method="BCLP", dose_rate_beta=0.6, **base)
    assert PrintConfig.from_dict(cfg3.to_dict()).dose_rate_beta == 0.6
