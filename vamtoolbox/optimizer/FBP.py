import numpy as np
import time

# VAM toolbox imports
import vamtoolbox


def minimizeFBP(target_geo,proj_geo,options):
    """
    Filtered backprojection (no optimization, only filtered) 

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

    """



    A = vamtoolbox.projectorconstructor.projectorconstructor(target_geo,proj_geo)
    _error = np.zeros(options.n_iter)
    _error[:] = np.NaN
    iter_times = np.zeros(options.n_iter)

    if options.verbose == 'plot':
        dp = vamtoolbox.display.EvolvingPlot(target_geo,options.n_iter)

    # begin timing
    t0 = time.perf_counter()

    # initialize sinogram with selected filter
    b = A.forward(target_geo.array)
    b = vamtoolbox.util.data.filterSinogram(b,options.filter)
      
    if options.offset == True:
        # offset by the absolute value of the minimum value
        b_min = np.min(b,axis=0)
        b = b + np.broadcast_to(np.abs(b_min),b.shape)

    else:
        # truncate negative values
        b = np.clip(b,a_min=0,a_max=None) 

    b = b/np.max(b)
    if options.bit_depth is not None:
        b = vamtoolbox.util.data.discretize(b,options.bit_depth,[0.0,np.max(b)])
        
    x = A.backward(b)
    x = x/np.max(x)     

    # Calculate current error
    _error[0] = vamtoolbox.metrics.calcVER(target_geo,x)
    
    iter_times[0] = time.perf_counter() - t0

    if options.verbose == 'time' or options.verbose == 'plot':
        print('Iteration %4.0f at time: %6.1f s'%(0,iter_times[0]))
    if options.verbose == 'plot':
        dp.update(_error,x)

    return vamtoolbox.geometry.Sinogram(b,proj_geo,options), vamtoolbox.geometry.Reconstruction(x,proj_geo,options), _error