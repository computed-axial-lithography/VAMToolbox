import time

import numpy as np

# VAM toolbox imports
import vamtoolbox


def minimizeOSMO(target_geo, proj_geo, options):
    """
    Sinogram optimization via object-space model optimization (OSMO).

    Parameters
    ----------
    target_geo : geometry.TargetGeometry object

    proj_geo : geometry.ProjectionGeometry object

    options : Options object

    Returns
    -------
    geometry.Sinogram
        optimized sinogram
    geometry.Reconstruction
        reconstruction from non-filtered backprojection of the
        optimized reconstructions

    References
    ----------
    [1] Object-space optimization of tomographic reconstructions for additive manufacturing
    https://doi.org/10.1016/j.addma.2021.102367

    """

    def stepVoid(x, prev_xm):
        """
        Update model to reduce dose in void regions.

        Parameters
        ----------
        x:
            current reconstruction
        prev_xm:
            previous model
        Returns
        -------
        x_update:
            updated reconstruction
        xm:
            updated model
        """
        xm = prev_xm

        void_diff = x[target_geo.void_inds] - options.d_l
        void_diff[void_diff < 0] = 0
        if k_i > 0.0:
            # PIHE: integral term removes the steady-state residual that plain
            # proportional OSMO leaves (some voxels plateau short of d_l / d_h).
            np.add(int_void, void_diff, out=int_void)   # in-place: no rebinding of the closure variable
            xm[target_geo.void_inds] = xm[target_geo.void_inds] - (k_p * void_diff + k_i * int_void)
        else:
            xm[target_geo.void_inds] = xm[target_geo.void_inds] - k_p * void_diff

        b = A.forward(xm)
        b = b / np.amax(b)
        if options.bit_depth is not None:
            b = vamtoolbox.util.data.discretize(
                b, options.bit_depth, [np.min(b), np.max(b)]
            )
        b[b < min_proj_val] = min_proj_val

        x_update = A.backward(b)
        x_update = x_update / np.amax(x_update)
        x_update[x_update < 0] = 0

        return x_update, b, xm

    def stepGel(x, prev_xm):
        """
        Update model to increase dose in gel regions.

        Parameters
        ----------
        x:
            current reconstruction
        prev_xm:
            previous model
        Returns
        -------
        x_update:
            updated reconstruction
        xm:
            updated model
        """

        xm = prev_xm

        gel_diff = options.d_h - x[target_geo.gel_inds]
        gel_diff[gel_diff < 0] = 0
        if k_i > 0.0:
            np.add(int_gel, gel_diff, out=int_gel)
            xm[target_geo.gel_inds] = xm[target_geo.gel_inds] + (k_p * gel_diff + k_i * int_gel)
        else:
            xm[target_geo.gel_inds] = xm[target_geo.gel_inds] + k_p * gel_diff

        b = A.forward(xm)
        b = b / np.amax(b)
        if options.bit_depth is not None:
            b = vamtoolbox.util.data.discretize(
                b, options.bit_depth, [np.min(b), np.max(b)]
            )
        b[b < min_proj_val] = min_proj_val

        x_update = A.backward(b)
        x_update = x_update / np.amax(x_update)
        x_update[x_update < 0] = 0

        return x_update, b, xm

    A = vamtoolbox.projectorconstructor.projectorconstructor(target_geo, proj_geo)
    _error = np.zeros(options.n_iter)
    _error[:] = np.nan
    iter_times = np.zeros(options.n_iter)

    min_proj_val = options.inhibition
    # PIHE gains (Webber et al. 2024: k_p = 1, k_i = 0.5).  k_i = 0 -> classic OSMO.
    k_p = float(getattr(options, "k_p", 1.0))
    k_i = float(getattr(options, "k_i", 0.0))
    hist_eq = bool(getattr(options, "hist_eq", False))

    dp = None
    if options.verbose == "plot":
        try:
            dp = vamtoolbox.display.EvolvingPlot(target_geo, options.n_iter, getattr(options, "save_img_path", None))
        except Exception as _e:
            print("[OSMO] verbose plot unavailable:", _e); dp = None

    # the first model is just the target
    x_model = np.copy(target_geo.array)

    # begin timing
    t0 = time.perf_counter()

    target_filtered = vamtoolbox.util.data.filterTargetOSMO(
        target_geo.array, options.filter
    )
    x_model = np.real(target_filtered)

    # the initial sinogram is just the forward projection of the model
    b = A.forward(x_model)
    b = np.clip(b, 0, None)
    x = A.backward(b)
    x = x / np.amax(x)
    # PIHE integral accumulators (one entry per void / gel voxel); unused when k_i = 0
    int_void = np.zeros_like(x[target_geo.void_inds]) if k_i > 0.0 else None
    int_gel = np.zeros_like(x[target_geo.gel_inds]) if k_i > 0.0 else None

    _error[0] = vamtoolbox.metrics.calcVER(target_geo, x)
    iter_times[0] = time.perf_counter() - t0

    for curr_iter in range(options.n_iter):
        if curr_iter == 0:
            continue
        # for each iteration the model is updated first in the void regions
        x, _, x_model = stepVoid(x, x_model)
        # then in the gel regions
        x, b, x_model = stepGel(x, x_model)

        _error[curr_iter] = vamtoolbox.metrics.calcVER(target_geo, x)

        iter_times[curr_iter] = time.perf_counter() - t0
        # Optional progress hook for GUIs / external callers (no-op if unset).
        if getattr(options, "iter_callback", None) is not None:
            options.iter_callback(curr_iter + 1, options.n_iter, float(_error[curr_iter]))
        if options.verbose == "time" or options.verbose == "plot":
            print(
                "Iteration %4.0f at time: %6.1f s" % (curr_iter, iter_times[curr_iter])
            )
        if options.verbose == "plot" and dp is not None:
            try: dp.update(_error, x)
            except Exception as _e: print("[OSMO] verbose plot update failed:", _e); dp = None

        if options.exit_param is not None:
            if _error[curr_iter] <= options.exit_param:
                b_opt = b
                break

    if hist_eq:
        # PIHE histogram equalization (Webber 2024 Eqs. 5-6): drop the dim tail
        # below T-3s, rescale so T+3s maps to full scale, clip.  Uses the projector's
        # 8-bit range far better than max-normalization (a few hot pixels otherwise
        # compress everything else), so the print runs at higher mean intensity.
        vals = b[b > 0]
        if vals.size:
            T, s = float(vals.mean()), float(vals.std())
            b = np.where(b < T - 3.0 * s, 0.0, b)
            b = np.clip(b / max(T + 3.0 * s, 1e-12), 0.0, 1.0)
            x = A.backward(b)                       # recon consistent with the clipped sinogram
            x = x / np.amax(x)
            x[x < 0] = 0
    if options.bit_depth is not None:
        b_opt = vamtoolbox.util.data.discretize(b, options.bit_depth, [0.0, np.max(b)])
    else:
        b_opt = b

    if options.verbose == "time" or options.verbose == "plot":
        print("Iteration %4.0f at time: %6.1f s" % (curr_iter, iter_times[curr_iter]))
    if options.verbose == "plot" and dp is not None:
        dp.ioff()
    # plt.close()
    return (
        vamtoolbox.geometry.Sinogram(b_opt, proj_geo, options),
        vamtoolbox.geometry.Reconstruction(x, proj_geo, options),
        _error,
    )
