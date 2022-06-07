import numpy as np

from vamtoolbox import geometry


def projectorconstructor(target_geo : geometry.TargetGeometry, proj_geo : geometry.ProjectionGeometry, optical_params=None):

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
    assert isinstance(target_geo,geometry.TargetGeometry), "target_geo should be of type: geometry.TargetGeometry"
    assert isinstance(proj_geo,geometry.ProjectionGeometry), "proj_geo should be of type: geometry.ProjectionGeometry"
    if optical_params is not None:
        isinstance(optical_params,CALopticalparams)

    if target_geo.insert is not None:
        if proj_geo.attenuation_field is not None:
            # with attenuation field in place, replace values where insert is with infinite attenuation
            proj_geo.attenuation_field[np.where(target_geo.insert == 1,True,False)] = np.inf
        else:
            # create new attenuation field array the size of the insert array with infinite attenuation where the insert is
            proj_geo.attenuation_field = np.where(target_geo.insert == 1, np.inf,0)

    # if GPU projection
    if proj_geo.CUDA is True:

        # if absorption or occlusion
        if proj_geo.attenuation_field is not None:
            if target_geo.n_dim == 2:
                raise NotImplementedError("2D attenuation CUDA projector not yet implemented.")
            else:
                raise NotImplementedError("3D attenuation CUDA projector not yet implemented.")
                # from vamtoolbox.projector.Projector3DParallelCUDA import Projector3DParallelCUDATigre
                # A = Projector3DParallelCUDATigre(target_geo,proj_geo)

        else:
            if target_geo.n_dim == 2:
                from vamtoolbox.projector.Projector2DParallelCUDA import Projector2DParallelCUDAAstra
                A = Projector2DParallelCUDAAstra(target_geo,proj_geo)

            else:
                from vamtoolbox.projector.Projector3DParallelCUDA import Projector3DParallelCUDAAstra
                A = Projector3DParallelCUDAAstra(target_geo,proj_geo)

    # if CPU projection
    else:

        # if absorption or occlusion
        if proj_geo.attenuation_field is not None:
            if target_geo.n_dim == 2:
                from vamtoolbox.projector.Projector2DParallel import Projector2DParallelPython
                A = Projector2DParallelPython(target_geo,proj_geo)
            else:
                from vamtoolbox.projector.Projector3DParallel import Projector3DParallelPython
                A = Projector3DParallelPython(target_geo,proj_geo)
                
        else:
            if target_geo.n_dim == 2:
                from vamtoolbox.projector.Projector2DParallel import Projector2DParallelAstra
                A = Projector2DParallelAstra(target_geo,proj_geo)

            else:
                from vamtoolbox.projector.Projector3DParallel import Projector3DParallelAstra
                A = Projector3DParallelAstra(target_geo,proj_geo)

                # from vamtoolbox.projector.Projector3DParallel import Projector3DParallelPython
                # A = Projector3DParallelPython(target_geo,proj_geo)

    if target_geo.zero_dose is not None:
        proj_geo.calcZeroDoseSinogram(A,target_geo)



    return A

