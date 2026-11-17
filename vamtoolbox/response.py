# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the GNU GPLv3 license.

import time
import warnings
from cmath import isnan

import matplotlib.pyplot as plt
import numpy as np


import os as _os
from concurrent.futures import ThreadPoolExecutor as _ThreadPool

# Max threads for the tiled diffusion convolution.  scipy's FFT releases the GIL so
# threads give real parallelism, but each concurrent tile holds its own FFT workspace
# (~1 GB at the default tile size), so cap the count to keep peak RAM bounded.
_CONV_THREADS = min(6, max(1, (_os.cpu_count() or 4) - 2))

# torch's CPU FFT (Intel MKL, multithreaded) is ~4.5x faster than scipy's pocketfft
# (measured, numerically identical to 6e-7).  Use it for the diffusion convolution
# when available; fall back to the scipy thread-pool path otherwise.
try:
    import torch as _torch
    _torch.set_num_threads(_os.cpu_count() or 4)
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

# cupy GPU FFT.  The whole-slab deconvolution runs GPU-resident (transfer the slab
# once; all convolutions + pointwise math on the GPU) — ~2.4x faster than torch CPU
# on large parts, ~2.5 GB VRAM, numerically equivalent (<=2 uint8 levels).  Probed
# lazily so a broken/absent CUDA install can't crash import.
_HAVE_CUPY = None        # tri-state: None=unprobed, True/False after first check
_cp = None


def _gpu_ok():
    """True if cupy + a usable CUDA device are available (probed once)."""
    global _HAVE_CUPY, _cp
    if _HAVE_CUPY is None:
        try:
            import cupy as _c
            _c.cuda.runtime.getDeviceCount()
            _ = _c.zeros(1, dtype=_c.float32) + 1     # smoke-test an actual alloc/op
            # CAP the cupy memory pool.  Without a limit it greedily reserves VRAM for
            # transient FFT temporaries and fragments — measured 13.5 GB reserved for a
            # slab that only USES 0.9 GB, choking the 12 GB card.  A limit forces it to
            # reuse freed blocks instead, keeping VRAM bounded (and leaving room for the
            # astra projector afterwards).  Single allocs over the limit OOM -> CPU.
            try:
                free_b, _tot = _c.cuda.Device().mem_info
                _c.get_default_memory_pool().set_limit(size=int(free_b * 0.92))
            except Exception:
                pass
            # also bound the cuFFT plan cache (plans hold workspace, separate from pool)
            try:
                _c.fft.config.get_plan_cache().set_memsize(1 << 30)   # 1 GiB
            except Exception:
                pass
            _cp, _HAVE_CUPY = _c, True
        except Exception:
            _HAVE_CUPY = False
    return _HAVE_CUPY


def _gpu_free():
    """Release cupy's device memory back so the next slab / astra starts clean."""
    if _cp is not None:
        try:
            _cp.get_default_memory_pool().free_all_blocks()
            _cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


def _array_module(a):
    """numpy for host arrays, cupy for device arrays."""
    if _cp is not None and isinstance(a, _cp.ndarray):
        return _cp
    return np


def _gpu_convolve(vol, kernel, tile_z=12):
    """Linear 3D convolution, mode='same', for a GPU-resident (cupy) volume.

    Picks the backend by what FITS VRAM: FFT (fast) when the padded transform fits,
    else DIRECT spatial convolution (memory-light, ~XY*tile only) for wide parts.
    The FFT workspace is XY-dominated — a 1407^2 plane needs ~10 GB per conv even
    z-tiled — so wide / billion-voxel parts use the direct path and still run on GPU."""
    cp = _cp
    from scipy.fft import next_fast_len
    try:
        free_b, _tot = cp.cuda.Device().mem_info
    except Exception:
        free_b = 4 << 30
    fnx = next_fast_len(vol.shape[0] + kernel.shape[0] - 1)
    fny = next_fast_len(vol.shape[1] + kernel.shape[1] - 1)
    ftz = next_fast_len(min(tile_z, vol.shape[2]) + kernel.shape[2] - 1)
    # cuFFT 3D scratch + cupyx's padded copies measure ~20x the single complex array
    # (a 1407^2 x 30 tile -> ~10 GB), so use a conservative multiplier; under-estimating
    # here picks FFT and OOMs the card.
    fft_b = fnx * fny * ftz * 8 * 24
    if fft_b < 0.4 * free_b:
        return _gpu_convolve_fft(vol, kernel, tile_z)
    return _gpu_convolve_direct(vol, kernel)


def _gpu_convolve_fft(vol, kernel, tile_z=12):
    """FFT path: z-tiled cupyx fftconvolve (fast, but XY-dominated memory)."""
    from cupyx.scipy.signal import fftconvolve as _gfft
    cp = _cp
    h = kernel.shape[2] // 2
    out = cp.empty_like(vol)
    nZ = vol.shape[2]
    pool = cp.get_default_memory_pool()
    for z0 in range(0, nZ, tile_z):
        z1 = min(z0 + tile_z, nZ)
        a0, a1 = max(0, z0 - h), min(nZ, z1 + h)
        block = _gfft(vol[:, :, a0:a1], kernel, mode="same").astype(cp.float32)
        out[:, :, z0:z1] = block[:, :, z0 - a0 : z0 - a0 + (z1 - z0)]
        del block
        # Return each tile's transient FFT workspace immediately — cupyx fftconvolve
        # pads input+kernel to the full tile size, so without this the pool fragments
        # to >10 GB across the tiles and OOMs the 12 GB card.  Keeps VRAM ~ one tile.
        pool.free_all_blocks()
    return out


def _gpu_convolve_direct(vol, kernel):
    """DIRECT spatial convolution (cupyx.ndimage) — no FFT padding, so memory is just
    input+output+kernel (a few GB even for 1407^2).  Handles wide / billion-voxel parts
    the FFT path can't fit on a 12 GB card.  mode='constant' cval=0 == fftconvolve(
    mode='same') for the symmetric PSF (verified to ~7e-6).  Done in one call — direct
    conv is memory-light, so no z-tiling / per-tile free overhead."""
    from cupyx.scipy.ndimage import convolve as _dconv
    out = _dconv(vol, kernel, mode="constant", cval=0.0)
    if out.dtype != _cp.float32:
        out = out.astype(_cp.float32)
    return out


def _torch_convolve(vol, kernel):
    """Linear 3D convolution, mode='same', via torch's CPU FFT (MKL, multithreaded).
    Numerically identical to scipy.signal.fftconvolve(mode='same')."""
    from scipy.fft import next_fast_len
    vt = _torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))
    kt = _torch.from_numpy(np.ascontiguousarray(kernel, dtype=np.float32))
    full = [vol.shape[i] + kernel.shape[i] - 1 for i in range(3)]
    fsh = [next_fast_len(full[i]) for i in range(3)]                 # fast FFT sizes
    F = _torch.fft.rfftn(vt, s=fsh)
    F *= _torch.fft.rfftn(kt, s=fsh)
    out = _torch.fft.irfftn(F, s=fsh)
    sl = tuple(slice((kernel.shape[i] - 1) // 2,
                     (kernel.shape[i] - 1) // 2 + vol.shape[i]) for i in range(3))
    return np.ascontiguousarray(out[sl].numpy())


def _diffusion_convolve(vol, kernel, tile_bytes=100_000_000, threads=None):
    """3D linear convolution for the diffusion blur, with BOUNDED peak memory.

    A single full-volume FFT convolution spikes to ~20x the array size (measured:
    ~20 GB for a 0.9 GB grid) — enough to OOM a machine on a large part.  Because
    the PSF only couples +/- (kz//2) z-slices, the result is identical when
    computed in z-tiles: each tile is FFT-convolved with a kernel-half halo and
    cropped.  This keeps the FFT workspace bounded to one tile (numerically
    identical to the full convolution; verified to ~2e-7) regardless of how tall
    the volume is.  Small volumes take the direct fftconvolve path (faster).

    The z-tiles are independent (disjoint output regions), so they run on a small
    thread pool — scipy's FFT releases the GIL, giving multi-core speedup with the
    arrays shared (no extra copies).  Peak RAM ~ threads * one-tile workspace.

    A cupy (GPU) input dispatches to the GPU-resident convolution instead."""
    if _cp is not None and isinstance(vol, _cp.ndarray):
        return _gpu_convolve(vol, kernel)
    from scipy.signal import fftconvolve
    vol = np.ascontiguousarray(vol)
    slice_bytes = vol.shape[0] * vol.shape[1] * 4          # one z-slice, float32
    # backend: torch FFT (MKL, ~4.5x faster, threads internally) when available;
    # bigger tiles for torch since it's efficient on large FFTs and self-parallel.
    if _HAVE_TORCH:
        conv1 = lambda a: _torch_convolve(a, kernel)
        tile_bytes = max(tile_bytes, 400_000_000)
        n_threads = 1                                      # torch already uses all cores
    else:
        conv1 = lambda a: fftconvolve(a, kernel, mode="same")
        n_threads = _CONV_THREADS if threads is None else int(threads)
    # direct path when the whole thing is small enough that the FFT spike is safe
    if vol.nbytes <= 300_000_000 or slice_bytes <= 0:
        return conv1(vol)
    kz = kernel.shape[2]
    h = kz // 2
    chunk = max(kz, int(tile_bytes / slice_bytes))         # z-slices per tile
    out = np.empty_like(vol)
    nZ = vol.shape[2]
    starts = list(range(0, nZ, chunk))

    def _do_tile(z0):
        z1 = min(z0 + chunk, nZ)
        a0, a1 = max(0, z0 - h), min(nZ, z1 + h)           # halo-padded read window
        block = conv1(vol[:, :, a0:a1])
        out[:, :, z0:z1] = block[:, :, z0 - a0 : z0 - a0 + (z1 - z0)]

    n_threads = max(1, min(n_threads, len(starts)))
    if n_threads > 1:
        with _ThreadPool(max_workers=n_threads) as ex:
            list(ex.map(_do_tile, starts))
    else:
        for z0 in starts:
            _do_tile(z0)
    return out


def separable_decompose(kernel, max_rank=12, tol=0.01):
    """Approximate a 3D PSF as a sum of SEPARABLE rank-1 terms (greedy CP):

        kernel ≈ Σ_r  lam_r · (u_r ⊗ v_r ⊗ w_r)

    so a 3D convolution becomes a handful of 1D convolutions — ~120x less compute
    than a 19³ direct conv and no FFT padding, which makes the diffusion deconv
    fast AND memory-light on the GPU (the diffusion PSF is ~Gaussian so a few terms
    suffice).  Returns a list of (lam, u, v, w) float32 numpy 1D arrays; rank grows
    until the relative reconstruction error is < `tol` (capped at `max_rank`)."""
    K = np.asarray(kernel, dtype=np.float64)
    Knorm = np.linalg.norm(K) or 1.0
    resid = K.copy()
    terms = []
    for _ in range(int(max_rank)):
        # leading rank-1 term via ALS power iteration (init from marginals)
        u = resid.sum(axis=(1, 2)); v = resid.sum(axis=(0, 2)); w = resid.sum(axis=(0, 1))
        u /= (np.linalg.norm(u) or 1); v /= (np.linalg.norm(v) or 1); w /= (np.linalg.norm(w) or 1)
        for _ in range(200):
            u = np.einsum("ijk,j,k->i", resid, v, w); u /= (np.linalg.norm(u) or 1)
            v = np.einsum("ijk,i,k->j", resid, u, w); v /= (np.linalg.norm(v) or 1)
            w = np.einsum("ijk,i,j->k", resid, u, v); w /= (np.linalg.norm(w) or 1)
        lam = float(np.einsum("ijk,i,j,k->", resid, u, v, w))
        terms.append((np.float32(lam), u.astype(np.float32), v.astype(np.float32), w.astype(np.float32)))
        resid = resid - lam * np.einsum("i,j,k->ijk", u, v, w)
        if np.linalg.norm(resid) / Knorm < tol:
            break
    return terms


def _sep_convolve(vol, terms):
    """Separable 3D convolution: Σ_r lam_r · conv1d_x(conv1d_y(conv1d_z(vol, w), v), u).
    Array-module agnostic — cupyx.ndimage on the GPU (fast, memory-light), scipy on
    the host.  Matches a full convolution with Σ lam (u⊗v⊗w) (symmetric PSF)."""
    xp = _array_module(vol)
    if xp is _cp:
        from cupyx.scipy.ndimage import convolve1d as _c1d
    else:
        from scipy.ndimage import convolve1d as _c1d
    out = None
    for lam, u, v, w in terms:
        t = _c1d(vol, u, axis=0, mode="constant")
        t = _c1d(t, v, axis=1, mode="constant")
        t = _c1d(t, w, axis=2, mode="constant")
        if out is None:
            out = t
            out *= lam
        else:
            out += lam * t                  # accumulate in place (one fewer full temp)
    return out


class ResponseModel:

    _default_gen_log_fun = {"A": 0, "K": 1, "B": 25, "M": 0.5, "nu": 1}
    _default_linear = {"M": 1, "C": 0}
    _default_interpolation = {"interp_min": 0, "interp_max": 1, "n_pts": 512}

    def __init__(
        self, type: str = "interpolation", form: str = "gen_log_fun", **kwargs
    ):
        """
        Parameters
        ----------
        type : str ('analytical', 'interpolation')
            Select analytical function evaluation or interpolate on pre-built interpolant arrays.
            Interpolation method handles edge cases of input explicitly and hence is more robust.

        form : str ('gen_log_fun', 'linear', 'identity', 'freeform')

        A : float, optional
            parameter in generalized logistic function (Richard's curve)
            Left asymptote

        K : float, optional
            parameter in generalized logistic function (Richard's curve)
            Right asymptote

        B : float, optional
            parameter in generalized logistic function (Richard's curve)
            Steepness of the curve

        M : float, optional
            parameter in generalized logistic function (Richard's curve)
            M shifts the curve left or right. It is the location of inflextion point when nu = 1.

        nu : float, optional
            parameter in generalized logistic function (Richard's curve)
            Influence location of maximum slope relative to the two asymptotes. 'Skew' the curve towards either end.

        M : float, optional
            parameter in linear (affine) function
            M is the slope of the curve: map = M*f + C

        C : float, optional
            parameter in linear (affine) function
            M is the y-intercept of the curve: map = M*f + C

        diffusion_kernel : np.ndarray, optional
            Optional 3D PSF for diffusion convolution (e.g. from blur_ker()).  When
            provided, the analytical forward map convolves the dose with this kernel
            before the non-linearity, and the derivative uses its adjoint (flipped
            kernel) -- so a BCLP optimization pre-compensates for light diffusion in
            the resin.  None (default) -> no diffusion, original behaviour.
        """
        self.type = type
        self.form = form

        # Optional diffusion kernel (only used when provided, analytical mode).
        self.diffusion_kernel = None
        self._diffusion_kernel_adjoint = None
        kernel = kwargs.pop("diffusion_kernel", None)
        if kernel is not None:
            k = np.asarray(kernel, dtype=np.float32)
            if k.ndim != 3:
                raise ValueError(
                    f"diffusion_kernel must be a 3D array, got ndim={k.ndim}"
                )
            if np.allclose(k, 0):
                warnings.warn(
                    "diffusion_kernel is all zeros; diffusion will have no effect."
                )
            k_sum = float(k.sum())  # normalize to preserve total dose (sum=1)
            if k_sum != 0.0:
                k = k / k_sum
            self.diffusion_kernel = k
            # adjoint of convolution is convolution with the flipped kernel
            self._diffusion_kernel_adjoint = np.flip(k)

        if self.type == "analytical":
            if self.form == "gen_log_fun":
                self.map = self._map_glf
                self.dmapdf = self._dmapdf_glf
                self.map_inv = self._map_inv_glf
                # Shallow copy avoid editing dict '_default_gen_log_fun' in place
                self.params = self._default_gen_log_fun.copy()
                # up-to-date parameters. Default dict is not updated
                self.params.update(kwargs)

            elif self.form == "linear":
                self.map = self._map_lin
                self.dmapdf = self._dmapdf_lin
                self.map_inv = self._map_inv_lin
                # Shallow copy avoid editing dict '_default_linear' in place
                self.params = self._default_linear.copy()  # type: ignore
                # up-to-date parameters. Default dict is not updated
                self.params.update(kwargs)

            elif self.form == "identity":
                self.map = self._map_id
                self.dmapdf = self._dmapdf_id
                self.map_inv = self._map_inv_id

            else:
                raise Exception(
                    "Form: Other analytical functions are not supported yet."
                )

        elif self.type == "interpolation":
            # Interpolation method stores three 1-D arrays as interpolant and query them upon each mapping call.
            # Stored arrays : (1)Sampling point on f, (2)corresponding forward map values, and (3)the first derviative of forward map.
            # All arrays are of the same size. The inverse mapping use (1) and (2) for memory efficiency and avoid singularity problem at asymptotes.

            # function alias
            self.map = self._map_interp
            self.dmapdf = self._dmapdf_interp  # type: ignore
            # Inverse mapping uses the same set of data generated for forward mapping.
            self.map_inv = self._map_inv_interp
            # Shallow copy avoid editing dict '_default_interpolation' in place
            self.params = self._default_interpolation.copy()  # type: ignore

            # build or import interpolant dataset
            if self.form == "gen_log_fun":
                self.params.update(self._default_gen_log_fun)  # Add relevant parameters
                self.params.update(
                    kwargs
                )  # up-to-date parameters. Default dict is not updated

                # build interpolant arrays
                self.interp_f_0 = np.linspace(  # type: ignore
                    self.params["interp_min"],
                    self.params["interp_max"],
                    self.params["n_pts"],
                )
                self.interp_map_0 = self._map_glf(self.interp_f_0)
                self.interp_dmapdf_0 = self._dmapdf_glf(self.interp_f_0)

            elif self.form == "linear":
                self.params.update(self._default_linear)  # Add relevant parameters
                self.params.update(
                    kwargs
                )  # up-to-date parameters. Default dict is not updated

                # build interpolant arrays
                self.interp_f_0 = np.linspace(  # type: ignore
                    self.params["interp_min"],
                    self.params["interp_max"],
                    self.params["n_pts"],
                )
                self.interp_map_0 = self._map_lin(self.interp_f_0)
                self.interp_dmapdf_0 = self._dmapdf_lin(self.interp_f_0)

            elif self.form == "identity":
                self.params.update(
                    kwargs
                )  # up-to-date parameters. Default dict is not updated

                # build interpolant arrays
                self.interp_f_0 = np.linspace(  # type: ignore
                    self.params["interp_min"],
                    self.params["interp_max"],
                    self.params["n_pts"],
                )
                self.interp_map_0 = self._map_id(self.interp_f_0)
                self.interp_dmapdf_0 = self._dmapdf_id(self.interp_f_0)

            elif self.form == "freeform":  # Directly import data instead of generating.
                self.interp_f_0 = kwargs.get(
                    "interp_f_0", None
                )  # Input data points are designated with 0 subscript
                self.interp_map_0 = kwargs.get(
                    "interp_map_0", None
                )  # Input data points are designated with 0 subscript

                # Check inputs
                if (len(self.interp_f_0.shape) > 1) or (
                    len(self.interp_map_0.shape) > 1
                ):
                    raise Exception(
                        'Imported data for material response curve should be 1D. Check "interp_f_0" and "interp_map_0".'
                    )
                if (self.interp_f_0.shape) != (self.interp_map_0.shape):
                    raise Exception(
                        'Size mismatch between "interp_f_0" and "interp_map_0".'
                    )

                # Extending the diff curve by assuming continuity of 1st derivative at the end of the curve
                self.interp_dmapdf_0 = np.diff(
                    self.interp_map_0,
                    n=1,
                    append=(
                        self.interp_map_0[-1]
                        + (self.interp_map_0[-1] - self.interp_map_0[-2])
                    ),
                )
                # Alternative solution to the differed array size is simply using shorter arrays.

            else:
                raise Exception("Other interpolation functions are not supported yet.")

        else:
            raise Exception(
                'Mapping type ("type") should be either "analytical" or "interpolation".'
            )

    # =================================Analytic: Generalized logistic function================================================

    # Definition of generalized logistic function: https://en.wikipedia.org/wiki/Generalised_logistic_function
    def _map_glf(self, f: np.ndarray) -> np.ndarray:
        numerator = self.params["K"] - self.params["A"]

        # Apply diffusion to the dose before the non-linearity (analytical only;
        # interpolation builds its tables from the closed-form curve directly).
        f_eff = self._apply_diffusion(f) if self.type == "analytical" else f

        self.cached_exp = np.exp(
            -self.params["B"] * (f_eff - self.params["M"])
        )  # cache result for later computation of derivative
        denominator = (1 + self.cached_exp) ** (1 / self.params["nu"])

        self.cached_map = self.params["A"] + (
            numerator / denominator
        )  # cache result for later use
        self._cached_f_eff = f_eff  # effective (diffused) dose, for the derivative
        return self.cached_map

    def _dmapdf_glf(self, f: np.ndarray, use_cached_result: bool = False) -> np.ndarray:
        # This function allows pre-computed results to be used to avoid duplicated computations
        # If 'map' is already executed for the exact current input f, use of cached results avoid recomputing the forward map in derivative evaluation.

        coef_1 = (1 / (self.params["K"] - self.params["A"])) ** self.params["nu"]
        coef_2 = self.params["B"] / self.params["nu"]

        # With diffusion (analytical) the non-linearity acts on the diffused dose;
        # the derivative w.r.t. the pre-diffusion dose needs the diffusion adjoint.
        if self.type == "analytical":
            if use_cached_result and hasattr(self, "_cached_f_eff"):
                map_val = self.cached_map
                exponential = self.cached_exp
            else:
                map_val = self._map_glf(f)  # caches exp, map, f_eff
                exponential = self.cached_exp
        else:
            if use_cached_result:
                map_val = self.cached_map
                exponential = self.cached_exp
            else:
                map_val = self._map_glf(f)
                exponential = np.exp(-self.params["B"] * (f - self.params["M"]))

        coef_3 = (map_val - self.params["A"]) ** (self.params["nu"] + 1)
        dmapdf_eff = coef_1 * coef_2 * coef_3 * exponential

        if self.type == "analytical" and self.diffusion_kernel is not None:
            self.cached_dmapdf = self._apply_diffusion_adjoint(dmapdf_eff)
        else:
            self.cached_dmapdf = dmapdf_eff
        return self.cached_dmapdf

    def _map_inv_glf(self, mapped: np.ndarray) -> np.ndarray:

        numerator = -np.log(
            ((self.params["K"] - self.params["A"]) / (mapped - self.params["A"]))
            ** self.params["nu"]
            - 1
        )  # Given C=1 and Q=1 --> log(Q)=log(1)=0
        f = (numerator / self.params["B"]) + self.params["M"]

        return f

    # =================================Analytic: Linear (affine) function=====================================================
    # Definition of linear function: mapped = M*f + C
    def _map_lin(self, f: np.ndarray) -> np.ndarray:
        f_eff = self._apply_diffusion(f) if self.type == "analytical" else f
        self.cached_map = self.params["M"] * f_eff + self.params["C"]
        return self.cached_map

    def _dmapdf_lin(self, f: np.ndarray, use_cached_result: bool = False) -> np.ndarray:
        dmapdf_eff = np.ones_like(f) * self.params["M"]
        if self.type == "analytical" and self.diffusion_kernel is not None:
            return self._apply_diffusion_adjoint(dmapdf_eff)
        return dmapdf_eff

    def _map_inv_lin(self, mapped: np.ndarray) -> np.ndarray:
        return (mapped - self.params["C"]) / self.params["M"]

    # =================================Analytic: Identity function============================================================
    # Definition of identity: mapped = f
    def _map_id(self, f: np.ndarray) -> np.ndarray:
        # With diffusion enabled (analytical), identity becomes "diffusion only".
        f_eff = self._apply_diffusion(f) if self.type == "analytical" else f
        self.cached_map = f_eff
        return self.cached_map

    def _dmapdf_id(self, f: np.ndarray, use_cached_result: bool = False) -> np.ndarray:
        if self.type == "analytical" and self.diffusion_kernel is not None:
            # d(identity-map)/d(dose) factor with diffusion is D^T(ones), which is
            # constant (dose-independent) -> compute the adjoint convolution once
            # and cache it (saves one FFT convolution per optimizer iteration).
            if getattr(self, "_dmapdf_id_cache_key", None) != f.shape:
                self._dmapdf_id_cache = self._apply_diffusion_adjoint(
                    np.ones(f.shape, dtype=np.float32)
                )
                self._dmapdf_id_cache_key = f.shape
            return self._dmapdf_id_cache
        return np.ones_like(f)

    def _map_inv_id(self, mapped: np.ndarray) -> np.ndarray:
        return mapped

    # =================================Interpolation==========================================================================
    def _map_interp(self, f: np.ndarray) -> np.ndarray:
        """
        Map optical dose to response via interpolation.
        More robust for asymptote values and potentially faster than computing exponentials in generalized logistic function.
        """
        return np.interp(
            f,
            self.interp_f_0,
            self.interp_map_0,
            left=self.interp_map_0[0],
            right=self.interp_map_0[-1],
        )  # Extrapolation points are taken as nearest neighbor, same as default

    def _dmapdf_interp(self, f: np.ndarray) -> np.ndarray:
        """
        Map optical dose to response 1st derivative via interpolation.
        """
        return np.interp(
            f,
            self.interp_f_0,
            self.interp_dmapdf_0,
            left=self.interp_dmapdf_0[0],
            right=self.interp_dmapdf_0[-1],
        )  # Extrapolation points are taken as nearest neighbor, same as default

    def _map_inv_interp(self, mapped: np.ndarray) -> np.ndarray:
        """
        Map material response back to optical dose via interpolation.
        """
        return np.interp(
            mapped,
            self.interp_map_0,
            self.interp_f_0,
            left=self.interp_f_0[0],
            right=self.interp_f_0[-1],
        )  # Extrapolation points are taken as nearest neighbor, same as default

    # =================================Diffusion=============================================================================
    def _apply_diffusion(self, f: np.ndarray):
        """Convolve the (3D) dose with the diffusion PSF.  No-op if no kernel.

        Uses FFT-based convolution (scipy.signal.fftconvolve), which is ~50-100x
        faster than direct ndimage.convolve for the ~19^3 PSF and numerically
        identical (linear zero-padded convolution, central 'same' output).
        """
        if self.diffusion_kernel is None:
            return f
        f_arr = np.asarray(f)
        original_shape = f_arr.shape
        f_3d = np.atleast_3d(f_arr).astype(np.float32, copy=False)
        diffused = _diffusion_convolve(f_3d, self.diffusion_kernel)
        if original_shape == ():
            return float(diffused.reshape(-1)[0])
        return diffused.reshape(original_shape).astype(f_arr.dtype, copy=False)

    def _apply_diffusion_adjoint(self, f: np.ndarray):
        """Adjoint of the diffusion convolution (convolution with the flipped kernel)."""
        if self._diffusion_kernel_adjoint is None:
            return f
        f_arr = np.asarray(f)
        original_shape = f_arr.shape
        f_3d = np.atleast_3d(f_arr).astype(np.float32, copy=False)
        diffused_adj = _diffusion_convolve(f_3d, self._diffusion_kernel_adjoint)
        if original_shape == ():
            return float(diffused_adj.reshape(-1)[0])
        return diffused_adj.reshape(original_shape).astype(f_arr.dtype, copy=False)

    # =================================Utilities==========================================================================
    def plotMap(
        self, fig=None, ax=None, lb=0, ub=1, n_pts=512, block=True, **plot_kwargs
    ):

        f_test = np.linspace(lb, ub, n_pts)
        mapped_f_test = self.map(f_test)

        if ax == None:
            fig, ax = plt.subplots()

        ax.plot(f_test, mapped_f_test, **plot_kwargs)
        ax.set_xlabel("Optical dose")
        ax.set_ylabel("Material response (mapped dose)")

        if block == False:
            fig.show()  # does not block. This function does not accept block argument.
        else:
            if "label" in plot_kwargs:
                ax.legend()
            plt.show(block=True)

        return fig, ax

    def plotDmapDf(
        self, fig=None, ax=None, lb=0, ub=1, n_pts=512, block=True, **plot_kwargs
    ):

        f_test = np.linspace(lb, ub, n_pts)
        mapped_f_test = self.dmapdf(f_test)

        if ax == None:
            fig, ax = plt.subplots()

        ax.plot(f_test, mapped_f_test, **plot_kwargs)
        ax.set_xlabel("Optical dose")
        ax.set_ylabel("1st derivative of material response (mapped dose)")

        if block == False:
            fig.show()  # does not block. This function does not accept block argument.
        else:
            if "label" in plot_kwargs:
                ax.legend()
            plt.show(block=True)

        return fig, ax

    def plotMapInv(
        self, fig=None, ax=None, lb=0, ub=1, n_pts=512, block=True, **plot_kwargs
    ):

        map_test = np.linspace(lb, ub, n_pts)
        f_test = self.map_inv(map_test)

        if ax == None:
            fig, ax = plt.subplots()

        ax.plot(map_test, f_test, **plot_kwargs)
        ax.set_xlabel("Material response (mapped dose)")
        ax.set_ylabel("Optical dose")

        if block == False:
            fig.show()  # does not block. This function does not accept block argument.
        else:
            if "label" in plot_kwargs:
                ax.legend()
            plt.show(block=True)

        return fig, ax

    def checkResponseTarget(self, f_T: np.ndarray):
        # Check if the response target is reachable with non-negative real inputs, and if it contains inf or nan.
        # Get target range
        f_T_min = np.amin(f_T)
        f_T_max = np.amax(f_T)

        validity = True

        # Check upper limit of response function (only for logistic function)
        if self.form == "gen_log_fun":
            if f_T_max > self.params["K"]:
                warnings.warn(
                    "Maximum response target is greater than right asymptotic value of response function."
                )
                validity = False

        # Check lower limit of response function (for all functional forms), up to 1% tolerance
        if (f_T_min < self.map(0)) and ~(np.isclose(f_T_min, self.map(0), atol=0.01 * f_T_max)):  # type: ignore
            warnings.warn(
                "Minimum response target is lower than response at zero optical dose."
            )
            validity = False

        # Check for boundedness
        if np.isinf(f_T).any():
            warnings.warn("Response target contains infinite value(s).")
            validity = False

        # Check for numeric values
        if np.isnan(f_T).any():
            warnings.warn("Response target contains nan value(s).")
            validity = False

        return validity

    def __repr__(self):
        return str(self.params)


def blur_ker(mm_in_px, D, tmax, rotspd, psf_size=19, fwhm_z=0.190, fwhm_xy=0.120,
             optical=False):
    """
    Diffusion (and optionally optical) point-spread function (PSF) for VAM
    diffusion correction.

    Ported from the diffusion-correction build of VAMToolbox.  The diffusion part
    is the resin diffusion-equation Green's function summed over every vial
    rotation during the print.  When ``optical`` is True it is additionally
    convolved with the projector's optical blur (anisotropic Gaussian).

    Parameters
    ----------
    mm_in_px : float   voxel pitch (mm per voxel) of the OPTIMIZE grid.
    D : float          resin diffusion coefficient (mm^2 / s).
    tmax : float       total print time (s).
    rotspd : float     vial rotation speed (deg / s).
    psf_size : int     PSF cube side in voxels (odd; large enough not to clip).
    fwhm_z, fwhm_xy : float
                       optical-PSF FWHM (mm) along / perpendicular to the vial
                       axis (measured values; defaults are the lab system's).
    optical : bool     include the optical PSF (default False -> diffusion-only,
                       a sum of isotropic Gaussians; set True once the projector
                       optical blur is characterized for your system).

    Returns
    -------
    np.ndarray   3D PSF, normalized to sum = 1.
    """
    dt = 360.0 / rotspd  # seconds per full vial rotation
    ntot = int(tmax // dt)
    px = mm_in_px

    n = int(psf_size)
    x, y, z = np.mgrid[0:n, 0:n, 0:n]
    x = x * px - np.mean(x * px)
    y = y * px - np.mean(y * px)
    z = z * px - np.mean(z * px)
    r = np.sqrt(x**2 + y**2 + z**2)

    # diffusion part: diffusion-equation Green's function, summed over each vial
    # rotation that occurs during the print.
    dker = (1 / ((4 * np.pi * D * (dt / 2)) ** 1.5)) * np.exp(
        -(r**2) / (4 * D * (dt / 2))
    ) * px**3
    dker = dker / np.sum(dker)
    for k in range(1, max(ntot, 1)):
        t = dt * k
        ddker = (1 / ((4 * np.pi * D * t) ** 1.5)) * np.exp(-(r**2) / (4 * D * t)) * px**3
        ddker = ddker / np.sum(ddker)
        dker = dker + ddker
    dker = dker / np.sum(dker)

    if not optical:
        return dker.astype(np.float32)

    # optical part of the PSF (anisotropic Gaussian), convolved into the diffusion PSF
    from scipy.signal import convolve as _convolve  # signal.convolve has mode='same'
    sigm_z = fwhm_z / 2.355
    sigm_xy = fwhm_xy / 2.355
    dkeropt = np.exp(
        -(x**2) / (2 * sigm_z**2)
        - (y**2) / (2 * sigm_xy**2)
        - (z**2) / (2 * sigm_xy**2)
    )
    dkeropt = dkeropt / np.sum(dkeropt)
    dker2 = _convolve(dker, dkeropt, mode="same")
    dker2 = dker2 / np.sum(dker2)
    return dker2.astype(np.float32)


def correct_blurring(kernel, target, n_iter=3, progress_cb=None, normalize=True, conv_fn=None):
    """
    Pre-deconvolve a target dose volume to compensate for resin diffusion +
    optical blur, following Orth et al., "Deconvolution Volumetric Additive
    Manufacturing" (Nat. Commun. 2023, https://doi.org/10.1038/s41467-023-39886-4).

    This is the published VAM deconvolution method: a ONE-TIME pre-processing
    step on the target, NOT an in-loop forward model.  Run it once before
    optimize(), then optimize the returned grey-scale target with BCLP.  The
    result is sharper than the original geometry so that, after the resin/optics
    blur the projected dose, fine features reach the cure threshold at the same
    time as bulk features (eliminating their systematic under-curing).

    Because the correction lives entirely in the target, the optimizer needs no
    per-iteration convolution and no z-halos — it runs with normal slabbing.

    Note: the corrected target is GREY-SCALE (continuous dose), so it must be
    optimized with BCLP.  OSMO partitions the target into binary gel/void index
    sets (``target > 0`` / ``target == 0``) and cannot use a grey-scale target.

    Uses a modified Richardson-Lucy deconvolution (a standard RL sharpening step
    followed by a dose-equalizing normalization step, Eqs. 6-7 of the paper),
    which keeps the result non-negative — required for physically realizable
    (non-negative) light projection.

    Parameters
    ----------
    kernel : np.ndarray   3D combined PSF from blur_ker().
    target : np.ndarray   intended target geometry (binary or grey, any range).
    n_iter : int          RL iterations (paper uses 2-10; 3 is a good default).
    progress_cb : callable(i, n), optional   per-iteration progress hook.
    normalize : bool      True (default) -> scale output to [0, 1] by its own max.
                 Set False to return the RAW deconvolution (bulk ~1, fine features
                 overshoot >1); used by the z-streamed driver so it can apply a
                 single GLOBAL max across all slabs (per-slab maxes would seam).
    conv_fn : callable(vol)->vol, optional   override the convolution (e.g. a GPU
                 separable conv from separable_decompose()).  Default: FFT conv with
                 ``kernel`` (torch/scipy on host, cupy on device).

    Returns
    -------
    np.ndarray   float32 grey-scale corrected target, same shape as ``target``;
                 [0, 1] if ``normalize`` else raw (non-negative).
    """
    # Array-module agnostic: numpy on host, cupy when `target` is a GPU array — so
    # the whole RL deconvolution (convolutions + pointwise math) runs on-device.
    xp = _array_module(target)
    k = xp.asarray(kernel, dtype=xp.float32)
    npad = int(1 + k.shape[0])
    _conv = conv_fn if conv_fn is not None else (lambda v: _diffusion_convolve(v, k))
    # pad so the convolutions don't wrap features at the volume boundary
    I = xp.pad(xp.asarray(target, dtype=xp.float32), npad, mode="constant")
    I += np.float32(1e-4)                 # small background: avoids divide-by-zero
    I /= I.max()                          # 0-d array divide (works on numpy & cupy)
    In = I.copy()
    n_iter = int(n_iter)
    # in-place updates keep at most ~4 full-volume float32 buffers live (I, In, tmp,
    # and the convolution's output) instead of allocating a fresh temporary per op.
    for i in range(n_iter):
        # standard Richardson-Lucy step (sharpening): In *= conv(I / conv(In))
        tmp = _conv(In)
        xp.divide(I, tmp, out=tmp)
        tmp = _conv(tmp)
        In *= tmp
        del tmp
        # dose-equalizing normalization step: In *= I / conv(In)
        tmp = _conv(In)
        xp.divide(I, tmp, out=tmp)
        In *= tmp
        del tmp
        if progress_cb is not None:
            try:
                progress_cb(i + 1, n_iter)
            except Exception:
                pass
    # un-pad, clip tiny FFT-convolution negatives
    sl = tuple(slice(npad, -npad) for _ in range(In.ndim))
    out = xp.ascontiguousarray(In[sl], dtype=xp.float32)
    xp.clip(out, 0.0, None, out=out)
    if normalize:
        m = float(out.max())
        if m > 0:
            out /= m
    return out


def _deconv_halo(kz, n_iter):
    """z-overlap halo for streamed deconvolution (slices each side of a slab).  Compact
    PSF -> fast boundary decay; even 1*half matches the whole-volume result to <=1 uint8
    level (verified), so a modest multiple is safely seam-free while minimizing the
    halo recompute that dominates small GPU slabs."""
    half = int(kz) // 2
    return half * (1 + (int(n_iter) + 2) // 3)


def correct_blurring_streamed(kernel, target, n_iter=3, working_bytes=6_000_000_000,
                              progress_cb=None):
    """Memory-bounded z-streamed version of correct_blurring() for large volumes.

    The Richardson-Lucy math is intrinsically float, but holding the whole volume
    plus its working buffers in float32 spikes to ~5x the volume (tens of GB on a
    billion-voxel part).  Here only one z-slab (+ an overlap halo) is held in float
    at a time, so peak RAM stays ~`working_bytes`; the result is returned as uint8
    [0,255] (recover dose in [0,1] as uint8/255).

    TWO passes so the quantization scale is GLOBAL (no per-slab seam): pass 1 finds
    the global max of the raw deconvolution, pass 2 writes uint8 normalized by that
    single max.  The overlap halo equals the exact kernel reach over n_iter
    iterations, so each slab's interior is numerically identical to the whole-volume
    result (verified seam-free).
    """
    k = np.asarray(kernel, dtype=np.float32)
    half = k.shape[2] // 2
    target = np.asarray(target)
    nX, nY, nZ = target.shape
    # Overlap halo.  The strict dependency radius of n_iter RL updates is ~3*half*n_iter,
    # but the PSF is compact and normalized so boundary influence decays fast — even a
    # 1*half halo matches the whole-volume result to <=1 uint8 level.  Use a modest
    # multiple (with a gentle n_iter term) so streaming stays efficient at large XY
    # while staying seam-free.
    H = _deconv_halo(k.shape[2], n_iter)
    slice_bytes = max(1, nX * nY * 4)
    # ~6x slab-float headroom (working buffers + FFT workspace); fit interior to budget
    interior = int(working_bytes / (slice_bytes * 6)) - 2 * H

    # GPU path: run each slab's deconvolution on-device with a SEPARABLE convolution
    # (the PSF decomposes into a few 1D Gaussians — ~120x less compute than the 19^3
    # conv, and no FFT padding, so it's fast AND memory-light at any XY width — even
    # 1407^2 / billion-voxel parts fit a 12 GB card).  CPU stays on the torch/scipy FFT
    # path (separable is slower than MKL FFT on the CPU).  Gate only on whether the
    # resident arrays + a couple of separable temps fit VRAM.
    use_gpu = _gpu_ok()
    gterms = None
    if use_gpu:
        try:
            free_b, _tot = _cp.cuda.Device().mem_info
            # Cap the GPU slab so the on-device deconv (padded I/In/tmp + separable
            # temps + cupyx working set, ~30 bytes per XY-voxel per z) fits ~0.72 of
            # free VRAM.  Smaller GPU slabs than the host budget is fine — separable
            # conv is cheap, so the extra halo recompute costs little.
            npad = k.shape[0] + 1
            fit_z = int(0.72 * free_b / (30.0 * nX * nY)) - 2 * npad
            interior = max(8, min(interior, fit_z - 2 * H))
            if fit_z - 2 * H < 8:                                  # XY too wide even for one slab
                use_gpu = False
            else:
                terms = separable_decompose(k)                    # once per run
                gterms = [(lam, _cp.asarray(u), _cp.asarray(v), _cp.asarray(w))
                          for lam, u, v, w in terms]
        except Exception:
            use_gpu = False

    def _deconv_raw(sub):
        """Raw (un-normalized) deconvolution of one slab — separable conv on the GPU if
        available (transfer once, all math on-device), else the torch/scipy CPU FFT."""
        if use_gpu:
            try:
                sg = _cp.asarray(np.ascontiguousarray(sub, dtype=np.float32))
                rg = correct_blurring(k, sg, n_iter=n_iter, normalize=False,
                                      conv_fn=lambda v: _sep_convolve(v, gterms))
                r = _cp.asnumpy(rg)
                del sg, rg
                return r
            except Exception:
                pass                                  # VRAM OOM etc. -> CPU fallback
            finally:
                _gpu_free()                            # release pool every slab
        return correct_blurring(k, sub, n_iter=n_iter, normalize=False)

    def _slab_raw(z0, z1):
        a0, a1 = max(0, z0 - H), min(nZ, z1 + H)
        sub = np.ascontiguousarray(target[:, :, a0:a1])
        raw = _deconv_raw(sub)
        lo = z0 - a0
        return raw[:, :, lo:lo + (z1 - z0)]

    # small enough (or budget can't fit even one minimal slab): one shot
    if interior < 1 or nZ <= interior + 2 * H:
        raw = _deconv_raw(target)
        gmax = float(raw.max()) or 1.0
        if progress_cb:
            progress_cb(1, 1)
        return np.rint(np.clip(raw / gmax, 0.0, 1.0) * 255.0).astype(np.uint8)

    n_slabs = (nZ + interior - 1) // interior
    # pass 1: global max over slab interiors
    gmax = 0.0
    for s in range(n_slabs):
        z0 = s * interior
        z1 = min(z0 + interior, nZ)
        gmax = max(gmax, float(_slab_raw(z0, z1).max()))
        if progress_cb:
            progress_cb(s + 1, 2 * n_slabs)
    gmax = gmax or 1.0
    # pass 2: write uint8 normalized by the single global max
    out = np.empty((nX, nY, nZ), dtype=np.uint8)
    for s in range(n_slabs):
        z0 = s * interior
        z1 = min(z0 + interior, nZ)
        part = _slab_raw(z0, z1)
        out[:, :, z0:z1] = np.rint(np.clip(part / gmax, 0.0, 1.0) * 255.0).astype(np.uint8)
        if progress_cb:
            progress_cb(n_slabs + s + 1, 2 * n_slabs)
    return out
