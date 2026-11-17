from typing import Protocol

import numpy as np

from vamtoolbox import geometry


class CALopticalparams:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class Projector(Protocol):
    """Protocol for projector classes returned by `projectorconstructor`."""

    def forward(self, x: np.ndarray) -> np.ndarray: ...

    def backward(self, b: np.ndarray) -> np.ndarray: ...


def projectorconstructor(
    target_geo: geometry.TargetGeometry,
    proj_geo: geometry.ProjectionGeometry,
    optical_params=None,
) -> Projector:
    """
    Constructor to create the projector based on the target size and projector type selected

    Parameters
    ----------
    target_geo : geometry.TargetGeometry

    proj_geo : geometry.ProjectionGeometry

    Returns
    -------
    Projector object
        the projector object has methods forward() and backward()

    Examples
    --------
    >>> A = projectorconstructor(target_geo, proj_geo)
    >>> b = A.forward(x) # returns the forward projection (sinogram)
    >>> x_ = A.backward(b) # returns the backward projection (reconstruction)

    """

    # check arguments
    assert hasattr(target_geo, "nX") and hasattr(target_geo, "n_dim"), \
        "target_geo must have geometry attributes (nX, nY, nZ, n_dim)"
    assert isinstance(
        proj_geo, geometry.ProjectionGeometry
    ), "proj_geo should be of type: geometry.ProjectionGeometry"
    
    
    if isinstance(optical_params, CALopticalparams):
        # TODO: utilize optical_params in projector construction
        pass

    # Measured depth-of-focus projector: parallel beam + the projector's measured,
    # depth-dependent PSF (torch; CUDA when available). Settings in proj_geo.dof.
    if proj_geo.ray_type == "dof":
        from vamtoolbox.projector.ProjectorDOF import ProjectorDOF
        return ProjectorDOF(target_geo, proj_geo)

    # Mesh-based GPU ray-density projector (Taichi, backend-agnostic).
    # Must be checked before the CUDA/CPU split because Taichi selects its
    # own backend internally.
    if proj_geo.ray_type == "ray_density":
        from vamtoolbox.projector.ProjectorRayDensityGPU import ProjectorRayDensityGPU
        A = ProjectorRayDensityGPU(target_geo, proj_geo)
        if target_geo.zero_dose is not None:
            proj_geo.calcZeroDoseSinogram(A, target_geo)
        return A

    if target_geo.insert is not None:
        if proj_geo.attenuation_field is not None:
            # with attenuation field in place, replace values where insert is with infinite attenuation
            proj_geo.attenuation_field[
                np.where(target_geo.insert == 1, True, False)
            ] = np.inf
        else:
            # create new attenuation field array the size of the insert array with infinite attenuation where the insert is
            proj_geo.attenuation_field = np.where(target_geo.insert == 1, np.inf, 0)

    # if GPU projection
    A: Projector
    if proj_geo.CUDA is True:
        if proj_geo.ray_type == "algebraic":
            from vamtoolbox.projector.pyTorchAlgebraicPropagation import (
                PyTorchAlgebraicPropagator,
            )

            A = PyTorchAlgebraicPropagator(target_geo, proj_geo)
        elif (
            proj_geo.ray_type == "ray_trace"
        ):  # PyTorchRayTracingPropagator automatically uses GPU if it is present and fallback to CPU if not found.
            from vamtoolbox.projector.pyTorchRayTrace import PyTorchRayTracingPropagator

            A = PyTorchRayTracingPropagator(target_geo, proj_geo)
        else:
            # if absorption or occlusion
            if proj_geo.attenuation_field is not None:
                if target_geo.n_dim == 2:
                    raise NotImplementedError(
                        "2D attenuation CUDA projector not yet implemented."
                    )
                else:
                    raise NotImplementedError(
                        "3D attenuation CUDA projector not yet implemented."
                    )
                    # from vamtoolbox.projector.Projector3DParallelCUDA import Projector3DParallelCUDATigre
                    # A = Projector3DParallelCUDATigre(target_geo,proj_geo)

            else:
                if target_geo.n_dim == 2:
                    try:
                        from vamtoolbox.projector.Projector2DParallelCUDA import (
                            Projector2DParallelCUDAAstra,
                        )
                        A = Projector2DParallelCUDAAstra(target_geo, proj_geo)
                    except (ImportError, AttributeError):
                        from vamtoolbox.projector.Projector2DParallel import (
                            Projector2DParallelSkimage,
                        )
                        A = Projector2DParallelSkimage(target_geo, proj_geo)

                else:
                    inclined = getattr(proj_geo, "inclination_angle", None) not in (None, 0)
                    try:
                        if inclined:
                            from vamtoolbox.projector.Projector3DParallelCUDA import (
                                Projector3DParallelCUDAAstra,
                            )
                            A = Projector3DParallelCUDAAstra(target_geo, proj_geo)
                        else:
                            # z-chunked GPU projector: bounds VRAM for tall/large volumes
                            from vamtoolbox.projector.Projector3DParallelCUDA import (
                                Projector3DParallelCUDAAstraChunked,
                            )
                            A = Projector3DParallelCUDAAstraChunked(target_geo, proj_geo)
                    except (ImportError, AttributeError):
                        from vamtoolbox.projector.Projector3DParallel import (
                            Projector3DParallelSkimage,
                        )
                        A = Projector3DParallelSkimage(target_geo, proj_geo)

    # if CPU projection
    else:
        if proj_geo.ray_type == "algebraic":
            from vamtoolbox.projector.algebraicPropagation import AlgebraicPropagator

            A = AlgebraicPropagator(target_geo, proj_geo)
        elif (
            proj_geo.ray_type == "ray_trace"
        ):  # PyTorchRayTracingPropagator automatically uses GPU if it is present and fallback to CPU if not found.
            from vamtoolbox.projector.pyTorchRayTrace import PyTorchRayTracingPropagator

            A = PyTorchRayTracingPropagator(target_geo, proj_geo)
        else:
            # if absorption or occlusion
            if proj_geo.attenuation_field is not None:
                if target_geo.n_dim == 2:
                    from vamtoolbox.projector.Projector2DParallel import (
                        Projector2DParallelPython,
                    )

                    A = Projector2DParallelPython(target_geo, proj_geo)
                else:
                    from vamtoolbox.projector.Projector3DParallel import (
                        Projector3DParallelPython,
                    )

                    A = Projector3DParallelPython(target_geo, proj_geo)

            else:
                if target_geo.n_dim == 2:
                    try:
                        from vamtoolbox.projector.Projector2DParallel import (
                            Projector2DParallelAstra,
                        )
                        A = Projector2DParallelAstra(target_geo, proj_geo)
                    except (ImportError, AttributeError):
                        from vamtoolbox.projector.Projector2DParallel import (
                            Projector2DParallelSkimage,
                        )
                        A = Projector2DParallelSkimage(target_geo, proj_geo)

                else:
                    # Default CPU 3D parallel projector: precomputed astra-built
                    # sparse system matrix (~5x faster than skimage radon, and it
                    # matches the astra-CUDA/GPU convention).  Falls back to skimage
                    # for cases the sparse projector doesn't cover, or if it can't
                    # be built.  Set proj_geo.sparse = False to force skimage.
                    sparse_pref = getattr(proj_geo, "sparse", None)
                    use_sparse = sparse_pref is not False and (
                        proj_geo.ray_type == "parallel"
                        and getattr(proj_geo, "inclination_angle", None) in (None, 0)
                        and getattr(proj_geo, "attenuation_field", None) is None
                    )
                    A = None
                    if use_sparse:
                        try:
                            from vamtoolbox.projector.Projector3DParallel import (
                                Projector3DParallelSparse,
                            )
                            A = Projector3DParallelSparse(target_geo, proj_geo)
                        except Exception as e:  # astra missing / build failure
                            print(f"  [projectorconstructor] sparse projector "
                                  f"unavailable ({e}); falling back to skimage.")
                            A = None
                    if A is None:
                        try:
                            from vamtoolbox.projector.Projector3DParallel import (
                                Projector3DParallelSkimage,
                            )
                            A = Projector3DParallelSkimage(target_geo, proj_geo)
                        except (ImportError, AttributeError):
                            from vamtoolbox.projector.Projector3DParallel import (
                                Projector3DParallelAstra,
                            )
                            A = Projector3DParallelAstra(target_geo, proj_geo)

    if target_geo.zero_dose is not None:
        proj_geo.calcZeroDoseSinogram(A, target_geo)

    return A
