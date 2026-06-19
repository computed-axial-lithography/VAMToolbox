import matplotlib.pyplot as plt
import numpy as np

import vamtoolbox


# TODO: make options a dataclass
class Options:

    __default_FBP = {"offset": False}
    __default_CAL = {
        "learning_rate": 0.01,
        "momentum": 0,
        "positivity": 0,
        "sigmoid": 0.01,
    }
    __default_PM = {"rho_1": 1, "rho_2": 1, "p": 1}
    __default_OSMO = {"inhibition": 0}
    __default_BCLP = {
        "response_model": "default",
        "eps": 0.1,
        "weight": 1,
        "p": 2,
        "q": 1,
        "learning_rate": 0.01,
        "optim_alg": "grad_des",
        "g0": None,
    }

    def __init__(
        self,
        method: str = "CAL",
        n_iter: int = 50,
        d_h: float = 0.8,
        d_l: float = 0.7,
        filter: str = "ram-lak",
        units: str = "normalized",
        blb=0,
        bub=None,
        **kwargs
    ):
        """
        Parameters
        ----------

        method : str
            Type of VAM method
                - "FBP"
                - "CAL"
                - "PM"
                - "OSMO"
                - "BCLP"

        n_iter : int
            number of iterations to perform

        d_h : float
            in-target dose constraint

        d_l : float
            out-of-target dose constraint

        filter : str
            filter for initialization ("ram-lak", "shepp-logan", "cosine", "hamming", "hanning", None)

        blb : double
            lower bound of sinogram pixel value. If None is given, no lower limit.

        bub : double
            upper bound of sinogram pixel value. If None is given, no upper limit.

        learning_rate : float, optional (CAL) (BCLP)
            step size in approximate gradient descent

        momentum : float, optional (CAL)
            descent momentum for faster convergence

        positivity : float, optional (CAL)
            positivity constraint enforced at each iteration

        sigmoid : float, optional (CAL)
            sigmoid thresholding strength

        rho_1 : float, optional (PM)

        rho_2 : float, optional (PM)

        p : int, optional (PM)

        inhibition : float, optional (OSMO)

        response model : ResponseModel, optional (BCLP)
            ResponseModel object to capture material response

        eps : float, optional (BCLP)
            band tolerance is +-eps around target value. Scalar or same array size as target array.

        weights : float, optional (BCLP)
            weightings in Lp minimization. Scalar (no local emphasis) or same array size as target array.

        p : float, optional (BCLP)
            p as in Lp norm. Not required to be integer in BCLP.

        q : float, optional (BCLP)
            Cost function is a Lp norm (scalar) raised to q-th power. Changing q does not affect minimizer on solution landscape but affect convergence behavior.

        g0 : sinogram, optional (BCLP)
            Initial guess of sinogram solution. Can be obtained from saved result or

        """
        self.method = method
        self.n_iter = n_iter
        self.d_h = d_h
        self.d_l = d_l
        self.filter = filter
        self.units = units
        self.blb = blb
        self.bub = bub

        self.__default_FBP.update(kwargs)
        self.__default_CAL.update(kwargs)
        self.__default_PM.update(kwargs)
        self.__default_OSMO.update(kwargs)
        self.__default_BCLP.update(
            kwargs
        )  # TODO: The class definition of dict "__default_BCLP" should not be edited in place here.
        self.__dict__.update(kwargs)  # Store all the extra variables

        self.verbose = self.__dict__.get("verbose", False)
        self.save_img_path = self.__dict__.get("save_img_path", None)
        self.bit_depth = self.__dict__.get("bit_depth", None)
        self.exit_param = self.__dict__.get("exit_param", None)

        if method == "FBP":
            self.offset = self.__default_FBP["offset"]

        if method == "CAL":
            self.learning_rate = self.__default_CAL["learning_rate"]
            self.momentum = self.__default_CAL["momentum"]
            self.positivity = self.__default_CAL["positivity"]
            self.sigmoid = self.__default_CAL["sigmoid"]

        if method == "PM":
            self.rho_1 = self.__default_PM["rho_1"]
            self.rho_2 = self.__default_PM["rho_2"]
            self.p = self.__default_PM["p"]

        if method == "OSMO":
            self.inhibition = self.__default_OSMO["inhibition"]

        if method == "BCLP":
            if self.__default_BCLP["response_model"] == "default":
                # Initialize a response model by default, only upon __init__ of Options class.
                # This avoids putting the ResponseModel object inside the class definition of Options (and hence avoid import problems and unnesscary init of default response model)
                self.response_model = vamtoolbox.response.ResponseModel()
            else:
                # If a response model is given, use the provided one instead.
                self.response_model = self.__default_BCLP["response_model"]  # type: ignore

            self.eps = self.__default_BCLP["eps"]
            self.weight = self.__default_BCLP["weight"]
            self.p = self.__default_BCLP["p"]  # type: ignore
            self.q = self.__default_BCLP["q"]  # type: ignore
            self.learning_rate = self.__default_BCLP["learning_rate"]  # type: ignore
            self.optim_alg = self.__default_BCLP["optim_alg"]
            self.g0 = self.__default_BCLP["g0"]
            self.test_alternate_handling = self.__dict__.get(
                "test_alternate_handling", False
            )  # flag for testing alternate handling. Default: False. This will override original weight setting. Will be removed for actual release

    def __str__(self):
        return str(self.__dict__)


def optimize(
    target_geo: vamtoolbox.geometry.TargetGeometry,
    proj_geo: vamtoolbox.geometry.ProjectionGeometry,
    options: Options,
    output="packaged",
):
    """
    Performs VAM optimization using the selected optimizer in options

    Parameters
    ----------
    target_geo : geometry.TargetGeometry object

    proj_geo : geometry.ProjectionGeometry object

    options : optimize.Options object

    Returns
    -------
    geometry.Sinogram object

    geometry.Reconstruction object

    """

    if options.units != "normalized" or proj_geo.absorption_coeff is not None:
        proj_geo.calcAbsorptionMask(target_geo)

    if options.method == "FBP":
        return vamtoolbox.optimizer.FBP.minimizeFBP(target_geo, proj_geo, options)

    elif options.method == "CAL":
        return vamtoolbox.optimizer.CAL.minimizeCAL(target_geo, proj_geo, options)

    elif options.method == "PM":
        return vamtoolbox.optimizer.PM.minimizePM(target_geo, proj_geo, options)

    elif options.method == "OSMO":
        return vamtoolbox.optimizer.OSMO.minimizeOSMO(target_geo, proj_geo, options)

    elif options.method == "BCLP":
        if getattr(options, "lowmem", False):
            return vamtoolbox.optimizer.BCLP_lowmem.minimizeBCLPLowMem(
                target_geo, proj_geo, options, output
            )
        return vamtoolbox.optimizer.BCLP.minimizeBCLP(
            target_geo, proj_geo, options, output
        )


def optimizeSlabbed(
    target_geo,
    proj_geo,
    options,
    z_slab=128,
    z_halo=None,
    verbose=True,
):
    """
    Memory-bounded VAM optimization: optimize the volume in z-slabs (the parallel-
    beam projection and the per-voxel dose updates are z-separable), then apply a
    shared global normalization so all slabs print consistently.  Peak RAM is
    bounded by ONE slab instead of the whole volume, so large/high-res parts fit
    in 16-32 GB.  Same (sinogram, reconstruction, error) return as optimize().

    z_slab : int   z-slices per slab (smaller -> less RAM, more per-slab overhead).
    z_halo : int   overlap added each side of a slab so a 3D diffusion kernel sees
                   correct neighbours at slab boundaries.  Auto: half the diffusion
                   kernel z-extent, else 0.
    """
    arr = target_geo.array
    nX, nY, nZ = arr.shape
    nA = proj_geo.angles.size

    if z_halo is None:
        _rm = getattr(options, "response_model", None)
        _ker = getattr(_rm, "diffusion_kernel", None)
        z_halo = (_ker.shape[2] // 2) if _ker is not None else 0

    # A grey-scale (diffusion-deconvolved) target is stored as uint8 [0,255] with a
    # fixed dose_scale; normalize each slab to [0,1] float for the optimizer.  The
    # scale is global (not per-slab), so every slab is normalized identically.
    _dose_scale = float(getattr(target_geo, "dose_scale", 1.0))

    def _subgeo(z_a, z_b):
        sub = np.ascontiguousarray(arr[:, :, z_a:z_b])
        if _dose_scale != 1.0:
            sub = sub.astype(np.float32) / np.float32(_dose_scale)
        stg = vamtoolbox.geometry.TargetGeometry(target=sub, resolution=z_b - z_a)
        stg.insert = None
        spg = vamtoolbox.geometry.ProjectionGeometry(
            angles=proj_geo.angles, ray_type=proj_geo.ray_type, CUDA=proj_geo.CUDA,
            absorption_coeff=proj_geo.absorption_coeff,
            container_radius=proj_geo.container_radius,
            projector_pixel_size=proj_geo.projector_pixel_size,
            inclination_angle=getattr(proj_geo, "inclination_angle", None))
        spg.sparse = getattr(proj_geo, "sparse", False)
        return stg, spg

    out = np.zeros((nX, nA, nZ), dtype=np.float32)
    n_slabs = int(np.ceil(nZ / z_slab))
    # The per-iteration callback restarts at iter 1 for every slab.  Wrap it so the GUI
    # sees ONE monotonic progress over all slabs (else the bar appears to "jump back"
    # e.g. iter 4 -> iter 2 when slab 2 starts).
    _base_cb = getattr(options, "iter_callback", None)
    for si, z0 in enumerate(range(0, nZ, z_slab)):
        z1 = min(z0 + z_slab, nZ)
        h0 = max(0, z0 - z_halo)
        h1 = min(nZ, z1 + z_halo)
        if verbose:
            print(f"  [slab {si + 1}/{n_slabs}] z {z0}:{z1} (+halo {z0 - h0}/{h1 - z1})")
        stg, spg = _subgeo(h0, h1)
        if _base_cb is not None:
            def _slab_cb(i, n, loss, _si=si):
                cum_i, cum_n = _si * n + i, n_slabs * n   # monotonic bar across slabs
                try:
                    _base_cb(cum_i, cum_n, loss,
                             f"slab {_si + 1}/{n_slabs} · iter {i}/{n} · dose error {float(loss):.4g}")
                except TypeError:
                    _base_cb(cum_i, cum_n, loss)   # callback without the message arg
            options.iter_callback = _slab_cb
        s_sino, _, _ = optimize(stg, spg, options)
        lo = z0 - h0
        out[:, :, z0:z1] = s_sino.array[:, :, lo:lo + (z1 - z0)]
    if _base_cb is not None:
        options.iter_callback = _base_cb     # restore the original callback

    # Shared global normalization: dose = backprojection(sinogram) is linear, so
    # rescale each z-slice so every slice's material-region dose hits one level.
    if verbose:
        print("  [slab] equalizing per-z dose (shared global normalization) ...")
    # build the full-volume absorption mask (per-slab optimize only built slab masks)
    if proj_geo.absorption_coeff is not None:
        proj_geo.calcAbsorptionMask(target_geo)
    A = vamtoolbox.projectorconstructor.projectorconstructor(target_geo, proj_geo)
    dose = A.backward(out)
    # Per-z material-region mean dose, computed in z-chunks (vectorized, no slow
    # per-slice Python loop, transient bounded to a chunk).
    mat = arr > 0
    cnt = mat.sum(axis=(0, 1)).astype(np.float64)
    ssum = np.zeros(nZ, dtype=np.float64)
    _ch = 64
    for z0 in range(0, nZ, _ch):
        z1 = min(z0 + _ch, nZ)
        ssum[z0:z1] = (dose[:, :, z0:z1] * mat[:, :, z0:z1]).sum(axis=(0, 1))
    ref = np.where(cnt > 0, ssum / np.maximum(cnt, 1.0), np.nan)
    good = np.isfinite(ref) & (ref > 1e-9)
    if good.any():
        lvl = float(np.median(ref[good]))
        k = np.ones(nZ, dtype=np.float64)
        k[good] = lvl / ref[good]
        kf = k[None, None, :].astype(np.float32)
        out *= kf
        dose = (dose * kf).astype(np.float32)
    sino = vamtoolbox.geometry.Sinogram(out, proj_geo, options)
    recon = vamtoolbox.geometry.Reconstruction(dose, proj_geo, options)
    return sino, recon, None
