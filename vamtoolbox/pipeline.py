"""
High-level, GUI-facing API for VAMToolbox print preparation.

This wraps the whole tomographic-VAM pipeline (voxelize -> optimize -> fan-beam
rebin -> projector image/video) behind a simple, serializable configuration object
and a stateful, inspectable pipeline class.  It is the layer a GUI should call: no
environment variables, no global state to manage, plain parameters in and structured
results out, with progress callbacks and cancellation.

Quick start
-----------
    import vamtoolbox
    from vamtoolbox.pipeline import PrintConfig, VAMPipeline

    cfg = PrintConfig(stl_path="thinker.stl", method="BCLP",
                      diffusion=True, absorption=True, n_iterations=10)

    def on_progress(stage, frac, msg):
        print(f"[{stage}] {frac*100:4.0f}%  {msg}")

    pipe = VAMPipeline(cfg, on_progress=on_progress)
    pipe.apply_hardware()              # auto GPU/RAM/CPU tuning (optional)
    print(pipe.estimate_optimize_time()["pretty"])   # ETA for a progress bar
    result = pipe.run()                # voxelize + optimize + rebin
    pipe.save_video("out.mp4")

Step-by-step (for a GUI that previews between stages):
    pipe.voxelize();  show(pipe.target.array)
    pipe.optimize();  show(pipe.reconstruction.array)
    pipe.rebin();     show(pipe.rebinned.array)
"""
from dataclasses import dataclass, field, asdict
import time

import numpy as np

import vamtoolbox


class PipelineCancelled(Exception):
    """Raised internally when a caller cancels a running pipeline."""


# ───────────────────────────── configuration ──────────────────────────────────
@dataclass
class PrintConfig:
    """All parameters for one print job.  Plain types only -> JSON-serializable
    (see to_dict/from_dict), so a GUI can save/load presets trivially."""

    # — input target (give an STL path; or pass an array to voxelize()/run()) —
    stl_path: str = None

    # — geometry / resolution —
    part_height_mm: float = 25.4           # tallest extent of the part
    voxel_pitch_um: float = 80.0           # print voxel pitch -> full resolution
    resolution_scale: float = 1.0          # optimize at this fraction of full res (<1 = faster)
    n_angles: int = 360
    vial_radius_mm: float = 48.8           # inner radius of the resin vial

    # — optimizer —
    method: str = "OSMO"                   # "OSMO" | "BCLP"
    n_iterations: int = 10
    d_high: float = 0.85                   # in-target dose lower bound
    d_low: float = 0.65                    # out-of-target dose upper bound
    learning_rate: float = 0.005
    eps: float = 0.1                       # BCLP band tolerance ±eps around target
    weight: float = 1.0                    # BCLP Lp weighting

    # — corrections —
    absorption: bool = True                # Beer-Lambert attenuation
    absorption_coeff_cm: float = None      # None -> computed from photoinitiator below
    diffusion: bool = False                # light/heat diffusion blur (BCLP only)
    diffusion_coeff: float = 1e-4          # mm^2/s
    print_time_s: float = 10.0
    rotation_deg_s: float = 24.0

    # — performance / memory —
    use_cuda: bool = None                  # None -> auto-detect
    slab: str = "auto"                     # "auto" | "off" | "<int>" (force z-slab size)
    low_memory: bool = False               # buffer-reusing BCLP variant
    rebin_jobs: int = -1                   # rebin CPU workers (-1 all, 1 serial)
    verbose: bool = False                  # True -> per-iteration optimizer logging
    save_img_path: str = None              # if set + BCLP, save the verbose figure frames here

    # — resin / optics (absorption coeff, fan-beam rebin, projector video) —
    resin_ri: float = 1.51
    pi_extinction: float = 36.0            # photoinitiator molar extinction (M^-1 cm^-1)
    pi_concentration_mm: float = 4.0       # photoinitiator concentration (mM)
    proj_u_px: int = 1080
    proj_v_px: int = 1920
    mm_per_pix: float = 76.0 / 1080
    throw_ratio: float = float("inf")      # inf = telecentric/collimated
    vial_print_height_mm: float = 93.6

    # ---- derived quantities ----
    @property
    def res_full(self) -> int:
        """Full (print) resolution in z-slices."""
        return max(4, round(self.part_height_mm / (self.voxel_pitch_um / 1000.0)))

    @property
    def res_opt(self) -> int:
        """Resolution actually optimized (after resolution_scale)."""
        return max(4, round(self.res_full * self.resolution_scale))

    @property
    def pixel_size_cm(self) -> float:
        """Voxel pitch at the optimize resolution, in cm (projector units)."""
        return (self.part_height_mm / self.res_opt) / 10.0

    @property
    def effective_absorption_coeff_cm(self) -> float:
        """mu (cm^-1): explicit override, else Beer-Lambert from photoinitiator."""
        if self.absorption_coeff_cm is not None:
            return self.absorption_coeff_cm
        return self.pi_extinction * (self.pi_concentration_mm * 1e-3) * np.log(10)

    # ---- (de)serialization for GUI presets ----
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PrintConfig":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def validate(self):
        """Raise ValueError with a GUI-friendly message for impossible configs."""
        if self.method not in ("OSMO", "BCLP"):
            raise ValueError(f"method must be 'OSMO' or 'BCLP', got {self.method!r}")
        if self.diffusion and self.method != "BCLP":
            raise ValueError("diffusion correction requires method='BCLP'")
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be >= 1")
        if not (0.0 < self.resolution_scale <= 1.0):
            raise ValueError("resolution_scale must be in (0, 1]")
        if self.part_height_mm <= 0 or self.voxel_pitch_um <= 0:
            raise ValueError("part_height_mm and voxel_pitch_um must be positive")


# ───────────────────────────── result container ───────────────────────────────
@dataclass
class PrintResult:
    """Structured output of a pipeline run (arrays are plain numpy)."""
    sinogram: np.ndarray = None            # optimized parallel-beam sinogram
    reconstruction: np.ndarray = None      # predicted dose
    rebinned_sinogram: np.ndarray = None   # fan-beam-corrected (printer-ready)
    timing: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)


# ───────────────────────────── pipeline ───────────────────────────────────────
class VAMPipeline:
    """Stateful print pipeline.  Run end-to-end with run(), or step through
    voxelize()/optimize()/rebin() and inspect intermediate results between calls."""

    def __init__(self, config: PrintConfig, on_progress=None):
        self.config = config
        self.on_progress = on_progress
        self.target = None              # geometry.TargetGeometry
        self.sinogram = None            # geometry.Sinogram (parallel)
        self.reconstruction = None      # geometry.Reconstruction
        self.rebinned = None            # geometry.Sinogram (fan-beam)
        self.timing = {}
        self._cancelled = False
        self._rebin_params = None

    # -- progress + cancellation --
    def _emit(self, stage, fraction, message=""):
        if self._cancelled:
            raise PipelineCancelled(stage)
        if self.on_progress is not None:
            try:
                self.on_progress(stage, float(fraction), str(message))
            except PipelineCancelled:
                raise
            except Exception:
                pass

    def cancel(self):
        """Request cancellation; takes effect at the next progress checkpoint."""
        self._cancelled = True

    # -- hardware --
    def detect_hardware(self) -> dict:
        return vamtoolbox.util.hardware.detect_system()

    def apply_hardware(self) -> dict:
        """Detect GPU/RAM/CPU and fill in unset performance settings."""
        self._emit("hardware", 0.0, "detecting hardware")
        info, rec = vamtoolbox.util.hardware.autoconfigure(apply=True, verbose=False)
        if self.config.use_cuda is None:
            self.config.use_cuda = rec["use_cuda"]
        if self.config.rebin_jobs == -1:
            self.config.rebin_jobs = rec["rebin_jobs"]
        self._emit("hardware", 1.0, "cuda=%s rebin_jobs=%s" % (
            self.config.use_cuda, self.config.rebin_jobs))
        return {"info": info, "recommendation": rec}

    def _cuda(self) -> bool:
        if self.config.use_cuda is None:
            return bool(self.detect_hardware()["cuda"])
        return bool(self.config.use_cuda)

    def estimate_optimize_time(self) -> dict:
        """ETA for the optimize stage (drive a GUI progress bar / warn the user)."""
        n_vox = self.config.res_opt ** 3
        backend = "gpu" if self._cuda() else "sparse"
        return vamtoolbox.util.timing.estimateOptimizeTime(
            n_vox, self.config.n_iterations, backend, self.config.n_angles)

    # -- stage 1: voxelize --
    def voxelize(self, target_array=None):
        cfg = self.config
        self._emit("voxelize", 0.0, "voxelizing")
        t = time.perf_counter()
        if target_array is not None:
            arr = np.ascontiguousarray(target_array)
            self.target = vamtoolbox.geometry.TargetGeometry(
                target=arr, resolution=arr.shape[2])
        else:
            if not cfg.stl_path:
                raise ValueError("PrintConfig.stl_path is empty and no target_array given")
            self.target = vamtoolbox.geometry.TargetGeometry(
                stlfilename=cfg.stl_path, resolution=cfg.res_opt)
        self.target.insert = None
        self.timing["voxelize"] = time.perf_counter() - t
        self._emit("voxelize", 1.0, f"target {tuple(self.target.array.shape)}")
        return self.target

    # -- internal builders --
    def _build_proj_geo(self):
        cfg = self.config
        angles = np.linspace(0, 360 - 360 / cfg.n_angles, cfg.n_angles)
        pg = vamtoolbox.geometry.ProjectionGeometry(
            angles=angles, ray_type="parallel", CUDA=self._cuda(),
            absorption_coeff=(cfg.effective_absorption_coeff_cm if cfg.absorption else None),
            container_radius=cfg.vial_radius_mm / 10.0,
            projector_pixel_size=cfg.pixel_size_cm)
        pg.sparse = not self._cuda()
        return pg

    def _build_options(self):
        cfg = self.config
        # per-iteration progress -> GUI; raising PipelineCancelled aborts the run
        def _iter_cb(i, n, loss, msg=None):
            # `msg` lets the slabbed optimizer label progress "slab k/N · iter i/n"
            # instead of a single doubled "iter i/(slabs*n)" that looks like extra iterations.
            try:
                self.final_loss = float(loss)        # remembered for the Output page
                self.loss_history.append([int(i), float(loss)])   # convergence graph
                self._emit("optimize", i / max(n, 1), msg or f"iter {i}/{n} · dose error {float(loss):.4g}")
            except Exception:
                self._emit("optimize", i / max(n, 1), msg or f"iter {i}/{n}")

        if cfg.method == "BCLP":
            model = vamtoolbox.response.ResponseModel(type="analytical", form="identity")
            if cfg.diffusion:
                pitch_mm = cfg.part_height_mm / cfg.res_opt
                dker = vamtoolbox.response.blur_ker(
                    pitch_mm, cfg.diffusion_coeff, cfg.print_time_s, cfg.rotation_deg_s)
                model = vamtoolbox.response.ResponseModel(
                    type="analytical", form="identity", diffusion_kernel=dker)
            # When verbose + a save path is given, run BCLP's "plot" mode so its
            # EvolvingPlot figure (target/dose/response/error + loss + histogram) is
            # rendered to PNG frames the GUI can show live.
            _bclp_verbose = ("plot" if (cfg.verbose and cfg.save_img_path) else ("time" if cfg.verbose else "off"))
            return vamtoolbox.optimize.Options(
                method="BCLP", response_model=model, eps=cfg.eps, weight=cfg.weight,
                n_iter=cfg.n_iterations, d_h=cfg.d_high, d_l=cfg.d_low,
                learning_rate=cfg.learning_rate, verbose=_bclp_verbose, save_img_path=cfg.save_img_path,
                lowmem=cfg.low_memory, iter_callback=_iter_cb)
        _osmo_verbose = ("plot" if (cfg.verbose and cfg.save_img_path) else ("time" if cfg.verbose else "off"))
        return vamtoolbox.optimize.Options(
            method="OSMO", n_iter=cfg.n_iterations, d_h=cfg.d_high, d_l=cfg.d_low,
            learning_rate=cfg.learning_rate, filter="hanning",
            verbose=_osmo_verbose, save_img_path=cfg.save_img_path, iter_callback=_iter_cb)

    def _resolve_slab(self, shape):
        cfg = self.config
        mode = str(cfg.slab).strip().lower()
        if mode in ("off", "0", "none", ""):
            return 0
        if mode != "auto":
            return int(mode)
        # auto: estimate working set vs available RAM.
        nX, nY, nZ = shape
        # Arrays whose footprint scales with the slab depth.  The CPU sparse path
        # also holds forward/backward DENSE result blocks (nA·nX·z and nX·nY·z),
        # so count ~2 extra slab-scaling arrays for it.
        n_arrays = (7 if cfg.method == "BCLP" else 4) + (2 if cfg.diffusion else 0)
        if not cfg.use_cuda:
            n_arrays += 2
        per_slice_gb = n_arrays * nX * nY * 4 / 1e9
        try:
            ram = vamtoolbox.util.hardware.detect_system()["ram_avail_gb"]
        except Exception:
            ram = 12.0
        if ram != ram:                                   # nan-safe
            ram = 12.0
        # Fixed overheads that do NOT scale with the slab and so must be RESERVED
        # out of the budget: the resident target grid, and (CPU sparse) the
        # z-independent 2D system matrix (~360·nX·nY·8 bytes — multi-GB at large XY).
        fixed_gb = nX * nY * nZ / 1e9                     # target uint8 grid
        if not cfg.use_cuda:
            fixed_gb += 360.0 * nX * nY * 8 / 1e9         # sparse system matrix
        budget = max(2.0, ram * 0.55 - fixed_gb)
        full_need = per_slice_gb * nZ
        if full_need <= budget:
            return 0
        return int(min(nZ, max(16, (budget / full_need) * nZ)))

    # -- stage 2: optimize --
    def optimize(self):
        cfg = self.config
        cfg.validate()
        self.loss_history = []                    # (iter, loss) for the convergence graph
        if self.target is None:
            self.voxelize()
        vamtoolbox.geometry.REBIN_N_JOBS = cfg.rebin_jobs
        proj_geo = self._build_proj_geo()
        options = self._build_options()
        slab_z = self._resolve_slab(self.target.array.shape)

        self._emit("optimize", 0.0, f"{cfg.method}{' +diffusion' if cfg.diffusion else ''}"
                   f"{' (slab %d)' % slab_z if slab_z else ''}")
        t = time.perf_counter()
        try:
            if slab_z and slab_z < self.target.array.shape[2]:
                sino, recon, _ = vamtoolbox.optimize.optimizeSlabbed(
                    self.target, proj_geo, options, z_slab=slab_z)
            else:
                sino, recon, _ = vamtoolbox.optimize.optimize(
                    target_geo=self.target, proj_geo=proj_geo, options=options)
        except Exception as exc:                          # GUI-friendly re-raise
            msg = str(exc)
            if "container" in msg.lower() or "radius" in msg.lower():
                raise ValueError(
                    "Part is wider than the vial for the absorption model. "
                    "Increase vial_radius_mm or disable absorption.") from exc
            raise
        self.timing["optimize"] = time.perf_counter() - t
        self.sinogram, self.reconstruction = sino, recon
        self._emit("optimize", 1.0, f"{self.timing['optimize']:.1f}s")
        return sino, recon

    # -- stage 3: fan-beam rebin (printer geometry) --
    def _rebin_params_compute(self):
        if self._rebin_params is None:
            cfg = self.config
            self._rebin_params = vamtoolbox.geometry.compute_rebin_params(
                vial_id_mm=cfg.vial_radius_mm * 2, vial_print_height_mm=cfg.vial_print_height_mm,
                mm_per_pix=cfg.mm_per_pix, proj_u_px=cfg.proj_u_px, proj_v_px=cfg.proj_v_px,
                throw_ratio=cfg.throw_ratio)
        return self._rebin_params

    def rebin(self):
        if self.sinogram is None:
            raise RuntimeError("call optimize() before rebin()")
        cfg = self.config
        sino = self.sinogram
        # If optimized at reduced resolution, upsample (R, Z) back to print
        # resolution before rebinning, so the printer sequence is full-res.
        if cfg.res_opt < cfg.res_full:
            from scipy.ndimage import zoom
            inv = cfg.res_full / cfg.res_opt
            self._emit("rebin", 0.0, "upsampling to print resolution")
            up = zoom(sino.array, (inv, 1.0, inv), order=1).astype(np.float32)
            sino = vamtoolbox.geometry.Sinogram(up, self.sinogram.proj_geo)
        rp = self._rebin_params_compute()
        # Scale the sinogram to PROJECTOR sampling before rebinning.  The rebin's
        # refraction geometry is defined in projector pixels (vial spans vial_width_px),
        # so whatever voxel pitch the part was optimized at, the diameter axis must be
        # resampled so the vial matches — otherwise a down-scaled part rebins mis-sized.
        pitch_mm = cfg.part_height_mm / max(cfg.res_full, 1)        # actual optimize pitch (mm)
        cur_vial_px = (cfg.vial_radius_mm * 2.0) / max(pitch_mm, 1e-6)   # vial width in current sino px
        if cur_vial_px > 0:
            f = rp["vial_width_px"] / cur_vial_px                   # = pitch / mm_per_pix
            if abs(f - 1.0) > 0.02:
                from scipy.ndimage import zoom
                self._emit("rebin", 0.2, f"scaling sinogram to projector ({f:.2f}x)")
                # Scale BOTH diameter (axis 0) and height (axis 2) by f: rebinFanBeam
                # carries the V/height axis through unchanged, and the projector has
                # square pixels (mm_per_pix), so height must use the same factor as the
                # width or the part's aspect ratio (and physical height) come out wrong.
                up = zoom(sino.array, (f, 1.0, f), order=1).astype(np.float32)
                sino = vamtoolbox.geometry.Sinogram(up, sino.proj_geo)
        self._emit("rebin", 0.3, "vial-correction rebin")
        t = time.perf_counter()
        self.rebinned = vamtoolbox.geometry.rebinFanBeam(
            sinogram=sino, vial_width=rp["vial_width_px"],
            N_screen=rp["N_screen"], n_write=cfg.resin_ri, throw_ratio=cfg.throw_ratio)
        self.timing["rebin"] = time.perf_counter() - t
        self._emit("rebin", 1.0, f"{tuple(self.rebinned.array.shape)}")
        return self.rebinned

    # -- outputs --
    def save_video(self, save_path, rot_vel=54, num_loops=1):
        """Render the printer projection sequence (uses the rebinned sinogram)."""
        sino = self.rebinned or self.sinogram
        if sino is None:
            raise RuntimeError("nothing to render; run optimize()/rebin() first")
        cfg = self.config
        rp = self._rebin_params_compute()
        self._emit("video", 0.0, "rendering projection video")
        img_cfg = vamtoolbox.imagesequence.ImageConfig(
            (cfg.proj_u_px, cfg.proj_v_px), intensity_scale=1, size_scale=rp["size_scale"],
            array_num=1, array_offset=0, invert_v=False, v_offset=0,
            normalization_percentile=99.9)
        imgset = vamtoolbox.imagesequence.ImageSeq(img_cfg, sinogram=sino)
        imgset.saveAsVideo(save_path=save_path, rot_vel=rot_vel, num_loops=num_loops, preview=False)
        self._emit("video", 1.0, save_path)
        return save_path

    def compute_quality(self) -> dict:
        """Gel/void dose separation metrics (handy for a GUI quality readout)."""
        if self.reconstruction is None:
            return {}
        d = np.asarray(self.reconstruction.array)
        g = np.asarray(self.target.array) > 0
        gel = float(d[g].mean()) if g.any() else float("nan")
        void = float(d[~g].mean()) if (~g).any() else float("nan")
        return {"gel_mean": gel, "void_mean": void, "contrast": gel - void}

    # -- one-shot --
    def run(self, target_array=None, do_rebin=True) -> PrintResult:
        self.voxelize(target_array)
        self.optimize()
        if do_rebin:
            self.rebin()
        self._emit("done", 1.0, "complete")
        return PrintResult(
            sinogram=self.sinogram.array,
            reconstruction=None if self.reconstruction is None else self.reconstruction.array,
            rebinned_sinogram=None if self.rebinned is None else self.rebinned.array,
            timing=dict(self.timing),
            config=self.config.to_dict(),
            quality=self.compute_quality())


# ───────────────────────────── convenience ────────────────────────────────────
def detect_hardware() -> dict:
    """Probe CPU/RAM/GPU (re-exported for convenience)."""
    return vamtoolbox.util.hardware.detect_system()


def run_print(config: PrintConfig, target_array=None, on_progress=None,
              auto_hardware=True, do_rebin=True) -> PrintResult:
    """One-call helper: optionally auto-tune to the machine, then run end-to-end."""
    pipe = VAMPipeline(config, on_progress=on_progress)
    if auto_hardware:
        pipe.apply_hardware()
    return pipe.run(target_array=target_array, do_rebin=do_rebin)
