"""Measured depth-of-focus projector (ray_type "dof"): adjoint, conventions, scale, pipeline."""
import numpy as np
import pytest

import vamtoolbox
from vamtoolbox.pipeline import PrintConfig, VAMPipeline
from vamtoolbox.projector.ProjectorDOF import DepthPSF, DEFAULT_PSF_TABLE, dof_z_halo


def _dof_geo(angles, voxel_mm=0.2, **dof):
    pg = vamtoolbox.geometry.ProjectionGeometry(angles=angles, ray_type="dof", CUDA=False)
    pg.dof = dict(voxel_mm=voxel_mm, device="cpu", **dof)
    return pg


def _target(shape):
    rng = np.random.default_rng(0)
    return vamtoolbox.geometry.TargetGeometry(target=(rng.random(shape) > 0.5).astype(np.float32))


@pytest.mark.parametrize("shape", [(40, 40), (32, 32, 10)])
def test_dof_adjoint_and_shapes(shape):
    ang = np.linspace(0, 360, 36, endpoint=False)
    tg = _target(shape)
    A = vamtoolbox.projectorconstructor.projectorconstructor(tg, _dof_geo(ang))
    rng = np.random.default_rng(1)
    x = vamtoolbox.util.data.clipToCircle(rng.random(shape).astype(np.float32))
    b = A.forward(x)
    expect = (shape[0], ang.size) if len(shape) == 2 else (shape[0], ang.size, shape[2])
    assert b.shape == expect
    y = rng.random(b.shape).astype(np.float32)
    lhs, rhs = float((b.astype(np.float64) * y).sum()), float((x.astype(np.float64) * A.backward(y)).sum())
    assert abs(lhs - rhs) / abs(lhs) < 1e-5


def test_dof_scale_matches_parallel():
    """With the blur conserving energy, per-angle sinogram totals match the ideal projector."""
    ang = np.linspace(0, 180, 12, endpoint=False)
    n = 48
    yy, xx = np.mgrid[:n, :n]
    disk = ((xx - n / 2 + .5) ** 2 + (yy - n / 2 + .5) ** 2 < (n / 3) ** 2).astype(np.float32)
    tg = vamtoolbox.geometry.TargetGeometry(target=disk)
    A_dof = vamtoolbox.projectorconstructor.projectorconstructor(tg, _dof_geo(ang))
    pg = vamtoolbox.geometry.ProjectionGeometry(angles=ang, ray_type="parallel", CUDA=False)
    pg.sparse = True
    A_par = vamtoolbox.projectorconstructor.projectorconstructor(tg, pg)
    s_dof, s_par = A_dof.forward(disk).sum(axis=0), A_par.forward(disk).sum(axis=0)
    assert np.allclose(s_dof, s_par, rtol=0.03)


def test_psf_extends_past_measurement():
    psf = DepthPSF(DEFAULT_PSF_TABLE)
    d = [float(psf.d4sigma_um(z)) for z in (0, 15, 30, 38)]
    assert d[0] < 100 and d[1] < d[2] < d[3]                  # grows along the cone
    r = np.linspace(0, 8000, 8001)
    for z in (0, 30):
        assert abs((psf.radial_profile(z, r) * 2 * np.pi * r).sum() - 1) < 0.02   # energy kept


def test_dof_z_halo_positive():
    ang = np.linspace(0, 360, 12, endpoint=False)
    assert dof_z_halo(_dof_geo(ang, voxel_mm=0.1), 300) >= 2


def test_pipeline_measured_dof_run(cylinder_array):
    n = cylinder_array.shape[2]
    cfg = PrintConfig(part_height_mm=25.4, voxel_pitch_um=25.4 / n * 1000, use_cuda=False, slab="off",
                      n_angles=36, n_iterations=4, method="BCLP", absorption=False, optics_model="measured_dof")
    res = VAMPipeline(cfg).run(target_array=cylinder_array, do_rebin=False)
    assert np.isfinite(res.sinogram).all()
    assert res.quality["contrast"] > 0
    with pytest.raises(ValueError):
        PrintConfig(optics_model="nonsense").validate()
