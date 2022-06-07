import numpy as np
import time
import matplotlib.pyplot as plt

# VAM toolbox imports
import vamtoolbox

def minimizeCAL(target_geo,proj_geo,options):

    """
    Gradient-descent sinogram optimization. 

    Parameters
    ----------
    target_geo : vamtoolbox.geometry.TargetGeometry object

    proj_geo : vamtoolbox.geometry.ProjectionGeometry object

    options : Options object

    Returns
    -------
    vamtoolbox.geometry.Sinogram     
        optimized sinogram
    vamtoolbox.geometry.Reconstruction
        reconstruction from non-filtered backprojection of the 
        optimized reconstructions

    References
    ----------
    [1] Volumetric additive manufacturing via tomographic reconstruction 
    https://science.sciencemag.org/content/363/6431/1075
    
    [2] Computed Axial Lithography (CAL): Toward Single Step 3D Printing of Arbitrary Geometries 
    https://arxiv.org/abs/1705.05893
    """
    def thresholdReconstruction(reconstruction):
        """
        Thresholds the reconstruction using the threshold value
        that minimizes the difference between the number of voxels in 
        the target and the number of voxels in the thresholded reconstruction.

        Parameters
        ----------
        reconstruction : ndarray

        Returns
        -------
        thresholded_reconstruction : ndarray
            thresholded reconstruction 

        """
        threshold_low = np.amin(reconstruction)
        threshold_high = np.amax(reconstruction)

        test_thresholds = np.linspace(threshold_low,threshold_high,100)
        # if attenuation is not None:
        #     sum_target_voxels = np.sum(target_geo.array) - np.sum(attenuation)
        # else:
        #     sum_target_voxels = np.sum(target_geo.array)
        sum_target_voxels = np.sum(target_geo.array)

        voxel_num_diff = np.zeros(len(test_thresholds))

        score = np.zeros(len(test_thresholds))
        for i in range(0,len(test_thresholds)):
            thresholded = np.where(reconstruction > test_thresholds[i], 1, 0)
            
            if target_geo.n_dim == 2:
                total_gel_not_in_target = np.sum(thresholded[target_geo.void_inds[0],target_geo.void_inds[1]])
                total_gel_in_target = np.sum(thresholded[target_geo.gel_inds[0],target_geo.gel_inds[1]])
            else:
                total_gel_not_in_target = np.sum(thresholded[target_geo.void_inds[0],target_geo.void_inds[1],target_geo.void_inds[2]])
                total_gel_in_target = np.sum(thresholded[target_geo.gel_inds[0],target_geo.gel_inds[1],target_geo.gel_inds[2]])

            score[i] = total_gel_in_target/len(target_geo.gel_inds[0]) - total_gel_not_in_target/len(target_geo.void_inds[0])


            # voxel_num_diff[i] = np.abs(np.sum(reconstruction>=test_thresholds[i]) - sum_target_voxels)

        opt_threshold_ind = np.argmax(score)
        threshold = test_thresholds[opt_threshold_ind]

        # opt_threshold_ind = np.argmin(voxel_num_diff)
        # threshold = test_thresholds[opt_threshold_ind]

        #D-mu
        recon_sub_mu = reconstruction - threshold

        # Sigmoid Function Implementation to calculate new dose/sigmoid param determines sharpness
        thresholded_reconstruction = 1 / (1 + np.exp(-(float(options.sigmoid) * recon_sub_mu)))

        return thresholded_reconstruction, threshold

    A = vamtoolbox.projectorconstructor.projectorconstructor(target_geo,proj_geo)
    _error = np.zeros(options.n_iter)
    _error[:] = np.NaN
    iter_times = np.zeros(options.n_iter)

    if options.verbose == 'plot':
        dp = vamtoolbox.display.EvolvingPlot(target_geo,options.n_iter)

    t0 = time.perf_counter()

    # Initialize sinogram with selected filter
    b = A.forward(target_geo.array)
    b = vamtoolbox.util.data.filterSinogram(b,options.filter)
    b = np.clip(b,a_min=0,a_max=None) 
    b0 = b
    delta_b_prev = np.zeros_like(b)
    x0 = A.backward(b0)
    x0 = x0/np.max(x0)

    x = x0
    _error[0] = vamtoolbox.metrics.calcVER(target_geo,x0)

    iter_times[0] = time.perf_counter() - t0


    prev_proj_error = np.zeros(b.shape)

    # if attenuation is not None:
    #     opt_target = target_geo.array - attenuation
    # else:
    #     opt_target = target_geo.array
    opt_target = target_geo.array
    for curr_iter in range(options.n_iter):
        if curr_iter == 0:
            continue


        # Threshold current reconstruction
        if options.d_h is not None:
            x_thresh = vamtoolbox.util.data.sigmoid(x-options.d_h,options.sigmoid)
        else:
            x_thresh, _ = thresholdReconstruction(x)
        

        delta_x = x_thresh - opt_target
        
        # Transform error into projection space
        delta_b = A.forward(delta_x)

    
        # Update projections
        grad = delta_b*(1-options.momentum) + delta_b_prev*options.momentum/(1-options.momentum**(curr_iter+1))   # apply momentum weighting on gradient
        b = b - options.learning_rate*grad # Update b (take a step in the direction of the negative gradient)
        b = np.clip(b,a_min=0,a_max=None)

        if options.bit_depth is not None:
            # b = vamtoolbox.util.data.histogramEqualization(b,options.bit_depth)
            b = vamtoolbox.util.data.discretize(b,options.bit_depth,[0.0,np.max(b)])


        # Reconstruct using current projections
        x = A.backward(b)
        x = x/np.amax(x)

        delta_b_prev = delta_b
        x_opt = x
        b_opt = b

        # Calculate current error
        _error[curr_iter] = vamtoolbox.metrics.calcVER(target_geo,x)
        
        iter_times[curr_iter] = time.perf_counter() - t0

        if options.verbose == 'time' or options.verbose == 'plot':
            print('Iteration %4.0f at time: %6.1f s'%(curr_iter,iter_times[curr_iter]))
        if options.verbose == 'plot':
            dp.update(_error,x)

        if options.exit_param is not None:
            if _error[curr_iter] <= options.exit_param:
                break

    
    b_opt[b_opt<0] = 0 # final positivity constraint applied in case there are any negatives 
    b_opt = b_opt/np.amax(b_opt)

    # if options.bit_depth is not None:
    #     b_opt = vamtoolbox.util.data.discretize(b_opt,options.bit_depth,[0.0,np.max(b)])


    x_opt = A.backward(b_opt) # reconstruct the optimal positivity constrained projections
    x_opt = x_opt/np.amax(x_opt)
    if options.verbose == 'time' or options.verbose == 'plot':
        print('Iteration %4.0f at time: %6.1f s'%(curr_iter,iter_times[curr_iter]))
    if options.verbose == 'plot':
        dp.ioff()

    return vamtoolbox.geometry.Sinogram(b_opt,proj_geo,options), vamtoolbox.geometry.Reconstruction(x_opt,proj_geo,options), _error
    