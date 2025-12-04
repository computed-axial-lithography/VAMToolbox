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
        xm[target_geo.void_inds] = xm[target_geo.void_inds] - void_diff

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
        xm[target_geo.gel_inds] = xm[target_geo.gel_inds] + gel_diff

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

    if options.verbose == "plot":
        dp = vamtoolbox.display.EvolvingPlot(target_geo, options.n_iter)

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
        if options.verbose == "time" or options.verbose == "plot":
            print(
                "Iteration %4.0f at time: %6.1f s" % (curr_iter, iter_times[curr_iter])
            )
        if options.verbose == "plot":
            dp.update(_error, x)

        if options.exit_param is not None:
            if _error[curr_iter] <= options.exit_param:
                b_opt = b
                break

    if options.bit_depth is not None:
        b_opt = vamtoolbox.util.data.discretize(b, options.bit_depth, [0.0, np.max(b)])
    else:
        b_opt = b

    if options.verbose == "time" or options.verbose == "plot":
        print("Iteration %4.0f at time: %6.1f s" % (curr_iter, iter_times[curr_iter]))
    if options.verbose == "plot":
        dp.ioff()
    # plt.close()
    return (
        vamtoolbox.geometry.Sinogram(b_opt, proj_geo, options),
        vamtoolbox.geometry.Reconstruction(x, proj_geo, options),
        _error,
    )
