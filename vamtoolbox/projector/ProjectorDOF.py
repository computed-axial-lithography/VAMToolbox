"""
Measured depth-of-focus (DOF) projector for VAMToolbox.

`ProjectorDOF` (bottom of this file) is what `projectorconstructor` returns for
`ProjectionGeometry(ray_type="dof")`: a parallel-beam projector that blurs every voxel,
transverse to the beam, by the measured projector PSF at its distance from the focal plane.
Divergence and vial refraction stay with the fan-beam rebin step (`throw_ratio`), as for
the "parallel" projector, so nothing is counted twice.

The general engine (`HelicalDOFProjector`) also supports helical motion, a cone beam and,
via `MultiFocalProjector`, several time-multiplexed focal planes; those are research options
not wired into the pipeline yet. Measured data: Nov 2026 HeliCAL PSF scan (ISO 11146 caustic,
waist 73 um D4sigma, NA 0.0275, DOF ~7 mm in air for FWHM <= 100 um).
"""
from __future__ import annotations

import math
import os
import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------------
# Depth-dependent PSF
# ----------------------------------------------------------------------------------
class DepthPSF:
    """Radially-symmetric PSF vs defocus, from a measured table or a Gaussian-beam fit.

    Table format (npz): dz_air_mm [nZ], r_um [nR], prof [nZ, nR] (energy-normalised
    radial intensity, i.e. sum over 2*pi*r*dr == 1), plus optional d4s_um [nZ].
    """

    def __init__(self, table_path=None, n_resin=1.51, sign=+1.0,
                 gauss_w0_um=35.0, gauss_m2=8.1, wavelength_um=0.405, mode="measured",
                 hyper_d0_um=None, hyper_theta_mrad=None):
        """mode: "measured" (table, extended past its ends by the fitted cone),
        "gauss" (Gaussian beam w0, M^2), or "hyperbola" (Gaussian profile whose D4sigma is
        sqrt(d0^2 + (theta*z)^2); e.g. a stopped-down lens: d0 from pixel + diffraction,
        theta = 2*NA in mrad)."""
        self.n = float(n_resin)
        self.sign = float(sign)
        self.mode = mode
        self.w0 = gauss_w0_um
        self.m2 = gauss_m2
        self.lam = wavelength_um
        self.hd0, self.hth = hyper_d0_um, hyper_theta_mrad
        self.table = None
        if table_path is not None:
            t = np.load(table_path)
            self.table = {k: t[k] for k in t.files}
        if self.mode == "measured" and self.table is None:
            raise ValueError("mode='measured' needs a table_path")
        if self.mode == "hyperbola" and (self.hd0 is None or self.hth is None):
            raise ValueError("mode='hyperbola' needs hyper_d0_um and hyper_theta_mrad")
        if self.table is not None:
            # caustic fit for extrapolating past the measured range (geometric cone)
            if "fit_d0_um" in self.table:
                self.fd0 = float(self.table["fit_d0_um"]); self.fth = float(self.table["fit_theta_mrad"])
            else:
                zz, dd = self.table["dz_air_mm"], self.table["d4s_um"]
                far = np.abs(zz) > 0.6 * np.abs(zz).max()
                self.fth = float(np.median(np.sqrt(np.clip(dd[far] ** 2 - dd.min() ** 2, 0, None)) / np.abs(zz[far])))
                self.fd0 = float(dd.min())

    def _fit_d4s(self, z_air):
        return np.sqrt(self.fd0 ** 2 + (self.fth * np.asarray(z_air, dtype=float)) ** 2)

    def dz_air(self, dz_resin_mm):
        return self.sign * np.asarray(dz_resin_mm) / self.n

    def d4sigma_um(self, dz_resin_mm):
        z = self.dz_air(dz_resin_mm)
        if self.mode == "hyperbola":
            return np.sqrt(self.hd0 ** 2 + (self.hth * np.asarray(z, dtype=float)) ** 2)
        if self.table is not None and "d4s_um" in self.table:
            zz = self.table["dz_air_mm"]
            inside = np.interp(z, zz, self.table["d4s_um"])
            # past the measured range: scale the table's end value along the fitted cone
            lo = self.table["d4s_um"][0] * self._fit_d4s(z) / self._fit_d4s(zz[0])
            hi = self.table["d4s_um"][-1] * self._fit_d4s(z) / self._fit_d4s(zz[-1])
            return np.where(z < zz[0], lo, np.where(z > zz[-1], hi, inside))
        zr_um = math.pi * self.w0 ** 2 / (self.m2 * self.lam)
        return 2 * self.w0 * np.sqrt(1 + (z * 1e3 / zr_um) ** 2)

    def radial_profile(self, dz_resin_mm, r_um):
        """Energy-normalised radial intensity at this defocus, sampled at r_um."""
        z = float(self.dz_air(dz_resin_mm))
        if self.mode == "measured":
            T = self.table
            zz, rr, P = T["dz_air_mm"], T["r_um"], T["prof"]
            if z < zz[0] or z > zz[-1]:
                # Far defocus is geometric and self-similar: stretch the end profile by the
                # cone's growth, keeping energy (p -> p(r/s)/s^2).
                ze = zz[0] if z < zz[0] else zz[-1]
                pe = P[0] if z < zz[0] else P[-1]
                sc = float(self._fit_d4s(z) / self._fit_d4s(ze))
                return np.interp(np.asarray(r_um) / sc, rr, pe, right=0.0) / sc ** 2
            i = int(np.clip(np.searchsorted(zz, z) - 1, 0, len(zz) - 2))
            a = (z - zz[i]) / (zz[i + 1] - zz[i])
            p = (1 - a) * P[i] + a * P[i + 1]
            return np.interp(r_um, rr, p, right=0.0)
        # Gaussian beam with the same second moment
        w = self.d4sigma_um(dz_resin_mm) / 2.0  # 1/e^2 radius
        return np.exp(-2 * np.asarray(r_um) ** 2 / w ** 2) * 2 / (math.pi * w ** 2)

    def kernel2d(self, dz_resin_mm, voxel_um, max_half=None, supersample=5):
        """(z, t) kernel on the voxel grid, sum == 1. Integrates over each voxel."""
        d4s = float(self.d4sigma_um(dz_resin_mm))
        half = int(math.ceil(0.9 * d4s / voxel_um)) + 1      # ~1.8 x 1/e^2 radius
        if self.mode == "measured":
            zz = self.table["dz_air_mm"]; z = float(self.dz_air(dz_resin_mm))
            ze = min(max(z, zz[0]), zz[-1])
            stretch = max(1.0, float(self._fit_d4s(z) / self._fit_d4s(ze)))
            rmax = self.table["r_um"][-1] * stretch
            half = max(half, int(math.ceil(rmax / voxel_um * 0.5)))
            half = min(half, int(math.ceil(rmax / voxel_um)))
        if max_half is not None:
            half = min(half, max_half)
        half = max(half, 1)
        s = supersample
        g = (np.arange(-half * s, half * s + s) + 0.5) / s - 0.5  # sub-sample centres
        g = g[: (2 * half + 1) * s]
        Z, T = np.meshgrid(g, g, indexing="ij")
        R = np.hypot(Z, T) * voxel_um
        k = self.radial_profile(dz_resin_mm, R)
        k = k.reshape(2 * half + 1, s, 2 * half + 1, s).sum(axis=(1, 3))
        k = np.clip(k, 0, None)
        if k.sum() <= 0:
            k = np.zeros((2 * half + 1, 2 * half + 1)); k[half, half] = 1
        return k / k.sum()


# ----------------------------------------------------------------------------------
# Projector
# ----------------------------------------------------------------------------------
class HelicalDOFProjector:
    def __init__(self, vol_shape, voxel_mm, angles_deg, psf: DepthPSF | None,
                 pitch_mm=0.0, window_rows=None, z_start_mm=None,
                 focal_offset_mm=0.0, depth_bin_mm=0.5, alpha_per_mm=0.0,
                 vial_radius_mm=None, device="cpu", dtype=torch.float32,
                 angles_per_chunk=8, max_kernel_half=None, source_dist_mm=None,
                 inverse_square=False):
        self.nx, self.ny, self.nz = [int(v) for v in vol_shape]
        assert self.nx == self.ny, "square xy grid expected"
        self.vox = float(voxel_mm)
        self.angles = np.asarray(angles_deg, dtype=np.float64)
        self.pitch = float(pitch_mm)
        self.helical = self.pitch > 0
        self.rows = int(window_rows) if (self.helical and window_rows) else self.nz
        self.z_start = (self.rows / 2 * self.vox) if z_start_mm is None else float(z_start_mm)
        self.f_off = float(focal_offset_mm)
        self.psf = psf
        self.dev = torch.device(device)
        self.dtype = dtype
        self.chunk = int(angles_per_chunk)
        self.alpha = float(alpha_per_mm)
        self.R = (self.nx * self.vox / 2) if vial_radius_mm is None else float(vial_radius_mm)
        # cone beam: virtual source at p = -S (S = source-to-rotation-axis distance, resin-equivalent)
        self.S = None if source_dist_mm in (None, 0, np.inf) else float(source_dist_mm)
        self.inv_sq = bool(inverse_square)

        # depth axis p (along beam, from rotation axis, + = away from projector)
        n = self.nx
        self.p_mm = (np.arange(n) - (n - 1) / 2) * self.vox
        self.t_mm = self.p_mm.copy()
        # depth bins
        if psf is None:
            self.bin_of_p = np.zeros(n, dtype=np.int64)
            self.n_bins = 1
            self.bin_centres = np.array([0.0])
        else:
            dz = self.p_mm - self.f_off
            edges = np.arange(dz.min() - 1e-9, dz.max() + depth_bin_mm, depth_bin_mm)
            b = np.clip(np.digitize(dz, edges) - 1, 0, len(edges) - 2)
            used = np.unique(b)
            remap = {u: i for i, u in enumerate(used)}
            self.bin_of_p = np.array([remap[v] for v in b])
            self.n_bins = len(used)
            self.bin_centres = np.array([dz[self.bin_of_p == i].mean() for i in range(self.n_bins)])
        # Beer-Lambert weight along p (light enters at p = -R)
        w = np.ones(n)
        if self.alpha > 0:
            w = np.exp(-self.alpha * np.clip(self.p_mm + self.R, 0, None))
        self.p_weight = torch.tensor(w, dtype=dtype, device=self.dev)
        # bin membership as a (n_bins, n) matrix for a single matmul reduction
        M = np.zeros((self.n_bins, n), dtype=np.float32)
        M[self.bin_of_p, np.arange(n)] = 1.0
        self.bin_mat = torch.tensor(M, dtype=dtype, device=self.dev)
        self._build_kernels(max_kernel_half)

    # -------------------------------------------------------------- kernels
    def _build_kernels(self, max_half):
        if self.psf is None:
            self.kpad = (0, 0)
            self.Kf = None
            return
        vox_um = self.vox * 1e3
        ks = [self.psf.kernel2d(dz, vox_um, max_half=max_half) for dz in self.bin_centres]
        if self.rows == 1:  # 2-D slice of an extruded object: collapse to the line-spread fn
            ks = [k.sum(axis=0, keepdims=True) for k in ks]
        hz = max(k.shape[0] for k in ks) // 2
        ht = max(k.shape[1] for k in ks) // 2
        self.kpad = (hz, ht)
        Hp, Wp = self.rows + 2 * hz, self.nx + 2 * ht
        self.fft_shape = (Hp, Wp)
        K = np.zeros((self.n_bins, Hp, Wp), dtype=np.float64)
        for i, k in enumerate(ks):
            kz, kt = k.shape[0] // 2, k.shape[1] // 2
            # place kernel centred at (0,0) with wrap-around
            for dz in range(-kz, kz + 1):
                for dt in range(-kt, kt + 1):
                    K[i, dz % Hp, dt % Wp] = k[dz + kz, dt + kt]
        self.Kf = torch.fft.rfft2(torch.tensor(K, dtype=self.dtype, device=self.dev))
        self.kernels = ks

    # -------------------------------------------------------------- geometry
    def z_centre_mm(self, theta_deg):
        return self.z_start + self.pitch * theta_deg / 360.0

    def _grid(self, thetas_deg):
        """Sampling grid (A, rows, P, T, 3) in normalised [-1,1] coords of the volume."""
        n, A = self.nx, len(thetas_deg)
        th = torch.tensor(np.deg2rad(thetas_deg), dtype=self.dtype, device=self.dev)
        p = torch.tensor(self.p_mm, dtype=self.dtype, device=self.dev)
        t = torch.tensor(self.t_mm, dtype=self.dtype, device=self.dev)
        P, T = torch.meshgrid(p, t, indexing="ij")                       # (n, n)
        mag = (self.S + P) / self.S if self.S else torch.ones_like(P)     # ray spread vs depth
        T = T * mag
        c, s = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
        X = P[None] * c - T[None] * s                                    # beam dir (cos, sin)
        Y = P[None] * s + T[None] * c
        half = (n - 1) * self.vox / 2   # align_corners=True: voxel centres 0..n-1 -> [-1, 1]
        gx = X / half
        gy = Y / half
        if self.helical:
            zc = torch.tensor(self.z_centre_mm(np.asarray(thetas_deg)), dtype=self.dtype, device=self.dev)
        else:
            zc = torch.full((A,), self.nz * self.vox / 2, dtype=self.dtype, device=self.dev)
        rows = torch.arange(self.rows, dtype=self.dtype, device=self.dev)
        zmm = zc[:, None] + (rows[None] - (self.rows - 1) / 2) * self.vox  # (A, rows)
        v = (rows - (self.rows - 1) / 2) * self.vox                      # detector row offset at axis
        zmm = zc[:, None, None] + v[None, :, None] * mag[:, 0][None, None, :]   # (A, rows, P)

        zvox = zmm / self.vox - 0.5
        gz = 2 * zvox / max(self.nz - 1, 1) - 1 if self.nz > 1 else torch.zeros_like(zvox)
        G = torch.stack(torch.broadcast_tensors(
            gx[:, None], gy[:, None], gz[..., None]), dim=-1)              # (A, rows, n, n, 3)
        return G

    def _ray_weight(self):
        """(rows, P, T) weight for the cone beam: oblique path length per depth step.

        The 1/r^2 irradiance fall-off needs NO explicit weight: diverging rays are spaced
        mag*voxel apart at depth p, so the ray density through a voxel already falls as
        1/mag (2-D fan) or 1/mag^2 (3-D cone). inverse_square=True would double count it."""
        n = self.nx
        p = torch.tensor(self.p_mm, dtype=self.dtype, device=self.dev)
        u = torch.tensor(self.t_mm, dtype=self.dtype, device=self.dev)
        v = (torch.arange(self.rows, dtype=self.dtype, device=self.dev) - (self.rows - 1) / 2) * self.vox
        if not self.S:
            return None
        mag = (self.S + p) / self.S
        w = torch.ones((self.rows, n, n), dtype=self.dtype, device=self.dev)
        if self.inv_sq:
            w = w / mag[None, :, None] ** 2
        path = torch.sqrt(1 + (u[None, None, :] / self.S) ** 2 + (v[:, None, None] / self.S) ** 2)
        return w * path

    # -------------------------------------------------------------- core op
    def _forward_chunk(self, vol5, thetas):
        """vol5: (1,1,nz,ny,nx) tensor. Returns (A, rows, T)."""
        A = len(thetas)
        G = self._grid(thetas)
        inp = vol5.expand(A, 1, *vol5.shape[2:]) if A > 1 else vol5
        samp = F.grid_sample(inp, G, mode="bilinear", padding_mode="zeros", align_corners=True)
        samp = samp[:, 0]                                                  # (A, rows, P, T)
        samp = samp * self.p_weight[None, None, :, None]
        if self.S:
            if not hasattr(self, "_rw"):
                self._rw = self._ray_weight()
            samp = samp * self._rw[None]
        binned = torch.einsum("bp,arpt->abrt", self.bin_mat, samp)       # (A, bins, rows, T)
        if self.Kf is None:
            return binned.sum(1) * self.vox
        hz, ht = self.kpad
        x = F.pad(binned, (ht, ht, hz, hz))
        X = torch.fft.rfft2(x)
        Y = (X * self.Kf[None]).sum(1)
        y = torch.fft.irfft2(Y, s=self.fft_shape)
        return y[:, hz:hz + self.rows, ht:ht + self.nx] * self.vox

    def _to_vol5(self, volume):
        v = torch.as_tensor(np.ascontiguousarray(np.transpose(volume, (2, 1, 0))),
                            dtype=self.dtype, device=self.dev)            # (nz, ny, nx)
        return v[None, None]

    def forward(self, volume):
        vol5 = self._to_vol5(volume)
        out = []
        with torch.no_grad():
            for i in range(0, len(self.angles), self.chunk):
                out.append(self._forward_chunk(vol5, self.angles[i:i + self.chunk]).cpu())
        return torch.cat(out).numpy()                                      # (A, rows, nx)

    def backward(self, sinogram):
        sino = torch.as_tensor(np.asarray(sinogram), dtype=self.dtype)
        vol5 = torch.zeros((1, 1, self.nz, self.ny, self.nx), dtype=self.dtype,
                           device=self.dev, requires_grad=True)
        acc = torch.zeros_like(vol5)
        for i in range(0, len(self.angles), self.chunk):
            th = self.angles[i:i + self.chunk]
            y = self._forward_chunk(vol5, th)
            g, = torch.autograd.grad(y, vol5, grad_outputs=sino[i:i + self.chunk].to(self.dev))
            acc += g
        return np.transpose(acc[0, 0].cpu().numpy(), (2, 1, 0))           # (nx, ny, nz)

    # convenience: depth of a voxel from the focal plane for a given angle
    def depth_mm(self, x_mm, y_mm, theta_deg):
        th = np.deg2rad(theta_deg)
        return x_mm * np.cos(th) + y_mm * np.sin(th) - self.f_off


def dot_test(P: HelicalDOFProjector, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((P.nx, P.ny, P.nz)).astype(np.float32)
    y = rng.random((len(P.angles), P.rows, P.nx)).astype(np.float32)
    Ax = P.forward(x)
    ATy = P.backward(y)
    a, b = float((Ax.astype(np.float64) * y).sum()), float((x.astype(np.float64) * ATy).sum())
    return a, b, abs(a - b) / abs(a)


DEFAULT_PSF_TABLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "staticresources", "measured_psf_table.npz")


# ----------------------------------------------------------------------------------
# Measured HeliCAL optics (Nov 2026 caustic + TwoWave projector geometry)
# ----------------------------------------------------------------------------------
MEASURED = dict(
    pixel_um_at_focus=50.0,          # TwoWave: 128 x 80 mm picture on its 2560 x 1600 DMD (HeliCAL project report, 2025)
    virtual_source_air_mm=345.0,     # projector lens to focal plane, measured on the PSF setup (throw ratio 2.70; report 2.63)
    waist_d4s_um=72.9,               # Nov 2026 caustic, ISO 11146 hyperbola fit
    M2=7.8,
    NA=0.0275,                       # second-moment (1/e^2) half-angle; 90%-energy cone gives 0.024-0.026
    dof_air_mm_fwhm100=6.95,         # FWHM <= 100 um span (asymmetric: -2.0 / +4.9 mm)
    n_resin=1.51,                    # LVUDMA
)


def measured_setup(vol_shape, voxel_mm, angles_deg, table_path=None, focal_offset_mm=0.0, pitch_mm=0.0,
                   window_rows=None, z_start_mm=None, n_resin=1.51, caustic_sign=+1.0,
                   cone=True, depth_bin_mm=0.5, **kw):
    """Projector configured from the measured optics. Resin-equivalent source distance
    S = n * L_air - focal_offset (paraxial refraction at a flat/index-matched boundary)."""
    psf = DepthPSF(table_path or DEFAULT_PSF_TABLE, n_resin=n_resin, sign=caustic_sign, mode="measured")
    S = (n_resin * MEASURED["virtual_source_air_mm"] - focal_offset_mm) if cone else None
    return HelicalDOFProjector(vol_shape, voxel_mm, angles_deg, psf, pitch_mm=pitch_mm,
                               window_rows=window_rows, z_start_mm=z_start_mm,
                               focal_offset_mm=focal_offset_mm, depth_bin_mm=depth_bin_mm,
                               source_dist_mm=S, **kw)



# ----------------------------------------------------------------------------------
# Several focal planes per angle (time-multiplexed sub-frames); doses add.
# ----------------------------------------------------------------------------------
class MultiFocalProjector:
    """One sinogram per focal plane, stacked on a new leading axis; backward sums the doses."""

    def __init__(self, projectors):
        self.projectors = list(projectors)

    def forward(self, volume):
        return np.stack([P.forward(volume) for P in self.projectors])

    def backward(self, sinograms):
        return sum(P.backward(sinograms[k]) for k, P in enumerate(self.projectors))


# ----------------------------------------------------------------------------------
# VAMToolbox projector interface (target (nX, nY[, nZ]) <-> sinogram (nX, nA[, nZ]))
# ----------------------------------------------------------------------------------
def _dof_settings(proj_geo):
    d = dict(getattr(proj_geo, "dof", None) or {})
    vox_mm = d.get("voxel_mm") or (float(proj_geo.projector_pixel_size) * 10.0)
    return d, float(vox_mm)


def _device(d):
    if d.get("device"):
        return d["device"]
    return "cuda" if torch.cuda.is_available() else "cpu"


def dof_psf(proj_geo):
    d, _ = _dof_settings(proj_geo)
    return DepthPSF(d.get("psf_table") or DEFAULT_PSF_TABLE, n_resin=d.get("n_resin", 1.51),
                    sign=d.get("caustic_sign", 1.0), mode="measured")


def dof_z_halo(proj_geo, nX):
    """Slab overlap (voxels) so the PSF's vertical extent sees its neighbours (optimizeSlabbed)."""
    d, vox = _dof_settings(proj_geo)
    psf = dof_psf(proj_geo)
    depth = nX * vox / 2 + abs(d.get("focal_offset_mm", 0.0))
    return psf.kernel2d(depth, vox * 1e3).shape[0] // 2


class ProjectorDOF:
    """Parallel beam + measured depth-of-focus blur, in VAMToolbox's conventions.

    Scaled like the ASTRA parallel projectors (line integrals in voxel units), clipped to the
    inscribed circle, Beer-Lambert absorption from `absorption_coeff` / `container_radius`.
    Settings come from `proj_geo.dof` (dict): voxel_mm, focal_offset_mm, n_resin, caustic_sign,
    psf_table, depth_bin_mm, device.
    """

    def __init__(self, target_geo, proj_geo):
        import vamtoolbox
        self._clip = vamtoolbox.util.data.clipToCircle
        arr = target_geo.array
        self.is2d = arr.ndim == 2
        nX, nY = arr.shape[0], arr.shape[1]
        if nX != nY:
            raise ValueError("ray_type='dof' needs a square xy grid")
        nZ = 1 if (self.is2d or arr.ndim < 3) else arr.shape[2]
        d, vox = _dof_settings(proj_geo)
        self.vox = vox
        alpha = (float(proj_geo.absorption_coeff) / 10.0) if proj_geo.absorption_coeff else 0.0
        R = float(proj_geo.container_radius) * 10.0 if getattr(proj_geo, "container_radius", None) else None
        self.P = HelicalDOFProjector(
            (nX, nY, nZ), vox, np.asarray(proj_geo.angles, dtype=np.float64), dof_psf(proj_geo),
            focal_offset_mm=d.get("focal_offset_mm", 0.0), depth_bin_mm=d.get("depth_bin_mm", 0.5),
            alpha_per_mm=alpha, vial_radius_mm=R, device=_device(d),
            angles_per_chunk=d.get("angles_per_chunk", 16))

    def forward(self, target):
        x = self._clip(np.asarray(target, dtype=np.float32))
        two_d = x.ndim == 2
        x3 = x[:, :, None] if two_d else x
        s = self.P.forward(x3) / self.vox                      # (nA, nZ, nX), voxel units
        s = np.transpose(s, (2, 0, 1))                          # (nX, nA, nZ)
        return s[:, :, 0] if two_d else s

    def backward(self, sinogram):
        b = np.asarray(sinogram, dtype=np.float32)
        two_d = b.ndim == 2
        b3 = b[:, :, None] if two_d else b
        x = self.P.backward(np.transpose(b3, (1, 2, 0))) / self.vox   # (nX, nY, nZ)
        x = x[:, :, 0] if two_d else x
        return self._clip(x)
