"""Diffusion correction: kernel build, convolution paths, dmapdf caching."""
import numpy as np
import pytest
from scipy.ndimage import convolve as nd_convolve
from scipy.signal import fftconvolve, oaconvolve

from vamtoolbox import response


def test_blur_ker_normalized():
    k = response.blur_ker(0.1, 1e-4, 10.0, 24.0)
    assert k.ndim == 3
    assert k.shape == (19, 19, 19)
    assert np.isclose(k.sum(), 1.0, atol=1e-5)
    assert k.min() > -1e-6                       # PSF is non-negative
    assert np.issubdtype(k.dtype, np.floating)
    # ResponseModel stores the kernel as float32 (matches the float32 dose flow)
    m = response.ResponseModel(type="analytical", form="identity", diffusion_kernel=k)
    assert m.diffusion_kernel.dtype == np.float32


def test_identity_no_diffusion_is_passthrough():
    m = response.ResponseModel(type="analytical", form="identity")
    f = np.random.default_rng(0).random((8, 8, 8)).astype(np.float32)
    assert np.allclose(m.map(f), f)


def test_diffusion_map_equals_reference_convolution():
    k = response.blur_ker(0.1, 1e-4, 10.0, 24.0)
    m = response.ResponseModel(type="analytical", form="identity", diffusion_kernel=k)
    f = np.zeros((24, 24, 16), np.float32)
    f[8:16, 8:16, 4:12] = 1.0
    out = m.map(f)
    ref = nd_convolve(f.astype(np.float64),
                      (k / k.sum()).astype(np.float64), mode="constant", cval=0.0)
    rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-9)
    assert rel < 1e-5


def test_adaptive_convolve_fft_matches_overlap_add():
    k = response.blur_ker(0.1, 1e-4, 10.0, 24.0)
    f = np.random.default_rng(1).random((32, 32, 32)).astype(np.float32)
    a = fftconvolve(f, k, mode="same")
    b = oaconvolve(f, k, mode="same")
    assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < 1e-4


def test_diffusion_convolve_helper_threshold():
    # small volume -> fftconvolve path; both must agree with the module helper
    k = response.blur_ker(0.1, 1e-4, 10.0, 24.0)
    f = np.random.default_rng(2).random((20, 20, 20)).astype(np.float32)
    out = response._diffusion_convolve(f, k)
    ref = fftconvolve(f, k, mode="same")
    assert out.shape == f.shape
    assert np.max(np.abs(out - ref)) / np.max(np.abs(ref)) < 1e-5


def test_dmapdf_identity_diffusion_is_cached():
    k = response.blur_ker(0.1, 1e-4, 10.0, 24.0)
    m = response.ResponseModel(type="analytical", form="identity", diffusion_kernel=k)
    f = np.ones((16, 16, 12), np.float32)
    d1 = m.dmapdf(f)
    d2 = m.dmapdf(f)
    assert d1 is d2                              # constant D^T(ones) cached, not recomputed
    assert d1.shape == f.shape


def test_correct_blurring_shape_range_nonneg():
    # Orth-2023 pre-deconvolution: shape preserved, grey-scale [0,1], non-negative
    # (required for physically realizable light projection), and reports progress.
    k = response.blur_ker(0.054, 1.5e-4, 60.0, 54.0)
    v = np.zeros((24, 24, 24), np.float32)
    v[6:18, 6:18, 6:18] = 1.0
    seen = []
    out = response.correct_blurring(k, v, n_iter=3, progress_cb=lambda i, n: seen.append((i, n)))
    assert out.shape == v.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert seen == [(1, 3), (2, 3), (3, 3)]      # one callback per RL iteration


def test_correct_blurring_raw_is_unnormalized():
    # normalize=False returns the raw RL output (bulk ~1, fine features overshoot >1)
    k = response.blur_ker(0.054, 1.5e-4, 60.0, 54.0)
    v = np.zeros((40, 40, 40), np.float32)
    v[10:30, 8:32, 20:21] = 1.0                  # a thin (1-voxel) sheet overshoots
    raw = response.correct_blurring(k, v, n_iter=3, normalize=False)
    norm = response.correct_blurring(k, v, n_iter=3, normalize=True)
    assert raw.max() > 1.5                        # raw is not capped at 1
    assert norm.max() <= 1.0 + 1e-6               # normalized is
    # same shape up to a global scale
    assert raw.shape == v.shape


def test_correct_blurring_streamed_matches_whole_volume():
    # z-streaming must be numerically identical to the whole-volume result (seam-free):
    # the overlap halo equals the exact kernel reach, and a single global max is used.
    k = response.blur_ker(0.054, 1.5e-4, 60.0, 54.0)
    v = np.zeros((40, 40, 120), np.float32)
    yy, xx = np.mgrid[0:40, 0:40]
    disk = ((xx - 20) ** 2 + (yy - 20) ** 2) <= 14 ** 2
    for z in range(4, 116):
        v[:, :, z] = disk
    v[16:24, 16:24, 60] = 1.0                     # feature near a forced slab joint
    whole = np.rint(response.correct_blurring(k, v, n_iter=3, normalize=True) * 255).astype(np.uint8)
    streamed = response.correct_blurring_streamed(
        k, v, n_iter=3, working_bytes=40 * 40 * 30 * 4 * 6)   # tiny budget -> many slabs
    assert streamed.dtype == np.uint8
    assert streamed.shape == v.shape
    assert int(np.abs(streamed.astype(int) - whole.astype(int)).max()) <= 1   # identical (±rounding)


def test_blur_ker_diffusion_only_vs_optical():
    # default is diffusion-only (sum of isotropic Gaussians); optical=True adds blur
    kd = response.blur_ker(0.1, 1e-4, 10.0, 24.0)                 # diffusion-only (default)
    ko = response.blur_ker(0.1, 1e-4, 10.0, 24.0, optical=True)  # + optical PSF
    assert np.isclose(kd.sum(), 1.0, atol=1e-5) and np.isclose(ko.sum(), 1.0, atol=1e-5)
    assert kd.shape == ko.shape == (19, 19, 19)
    # optical adds blur -> a lower, wider peak than diffusion-only
    assert ko.max() < kd.max()


def test_separable_decompose_reconstructs_kernel():
    k = response.blur_ker(0.054, 1.5e-4, 60.0, 54.0)
    terms = response.separable_decompose(k, max_rank=12, tol=0.01)
    approx = sum(lam * np.einsum("i,j,k->ijk", u, v, w) for lam, u, v, w in terms)
    assert np.linalg.norm(approx - k) / np.linalg.norm(k) < 0.02     # within tol


def test_sep_convolve_matches_fft():
    # separable conv (sum of 1D convs) must match the full FFT convolution
    k = response.blur_ker(0.054, 1.5e-4, 60.0, 54.0)
    terms = response.separable_decompose(k, max_rank=12, tol=0.005)
    v = np.zeros((40, 40, 40), np.float32)
    v[10:30, 10:30, 12:16] = 1.0
    v[15:25, 15:25, 28] = 1.0
    ref = response._diffusion_convolve(v, k)
    sep = response._sep_convolve(v, terms)
    assert sep.shape == v.shape
    assert np.max(np.abs(sep - ref)) / np.max(np.abs(ref)) < 0.05    # few-% separable approx


def test_correct_blurring_equalizes_feature_dose():
    # The published effect: after the resin/optics blur, a thin feature should reach
    # a peak dose much closer to a thick feature's when the target is pre-corrected.
    k = response.blur_ker(0.054, 1.5e-4, 60.0, 54.0)
    v = np.zeros((40, 40, 40), np.float32)
    v[10:30, 8:32, 6:10] = 1.0                   # thick feature (4 voxels)
    v[10:30, 8:32, 20:21] = 1.0                  # thin feature (1 voxel)
    cor = response.correct_blurring(k, v, n_iter=3)

    reblur_uncorr = fftconvolve(v, k, mode="same")
    reblur_corr = fftconvolve(cor, k, mode="same")
    thick, thin = (slice(10, 30), slice(8, 32), 8), (slice(10, 30), slice(8, 32), 20)
    ratio_uncorr = reblur_uncorr[thin].max() / reblur_uncorr[thick].max()
    ratio_corr = reblur_corr[thin].max() / reblur_corr[thick].max()
    assert ratio_uncorr < 0.5                    # uncorrected: thin feature badly under-dosed
    assert ratio_corr > 0.8                      # corrected: thin/thick dose nearly equal
    assert ratio_corr > ratio_uncorr
