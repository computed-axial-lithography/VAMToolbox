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
    # FIXME: What is this weirdness
    assert isinstance(
        target_geo, geometry.TargetGeometry
    ), "target_geo should be of type: geometry.TargetGeometry"
    assert isinstance(
        proj_geo, geometry.ProjectionGeometry
    ), "proj_geo should be of type: geometry.ProjectionGeometry"
    
    
    if isinstance(optical_params, CALopticalparams):
        # TODO: utilize optical_params in projector construction
        pass

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
                    from vamtoolbox.projector.Projector2DParallelCUDA import (
                        Projector2DParallelCUDAAstra,
                    )

                    A = Projector2DParallelCUDAAstra(target_geo, proj_geo)

                else:
                    from vamtoolbox.projector.Projector3DParallelCUDA import (
                        Projector3DParallelCUDAAstra,
                    )

                    A = Projector3DParallelCUDAAstra(target_geo, proj_geo)

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
                    from vamtoolbox.projector.Projector2DParallel import (
                        Projector2DParallelAstra,
                    )

                    A = Projector2DParallelAstra(target_geo, proj_geo)

                else:
                    from vamtoolbox.projector.Projector3DParallel import (
                        Projector3DParallelAstra,
                    )

                    A = Projector3DParallelAstra(target_geo, proj_geo)

                    # from vamtoolbox.projector.Projector3DParallel import Projector3DParallelPython
                    # A = Projector3DParallelPython(target_geo,proj_geo)

    if target_geo.zero_dose is not None:
        proj_geo.calcZeroDoseSinogram(A, target_geo)

    return A
