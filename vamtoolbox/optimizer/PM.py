import numpy as np
import time
from scipy import optimize

# VAM toolbox imports
import vamtoolbox

class LogError:
    def __init__(self,options):
        self.options = options
        self.curr_iter = 0
        self.error_iter = 0
        self.loss = []
        self.error = np.zeros(options.n_iter+1)
        self.error[:] = np.NaN
        self.iter_times = np.zeros(options.n_iter+1)
        self.t0 = time.perf_counter()

    def getIterTime(self,i):
        self.iter_times[i] = time.perf_counter() - self.t0


def minimizePM(target_geo,proj_geo,_options):
    """
    Quasi-Newton projection optimization via L-BFGS-B algorithm. 

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
    [1] High fidelity volumetric additive manufacturing 
    https://doi.org/10.1016/j.addma.2021.102299

    """
    def getRegions(dose):

        R1 = np.zeros_like(dose)
        R2 = np.zeros_like(dose)
        V1 = np.zeros_like(dose)
        V2 = np.zeros_like(dose)

        region_shape = np.shape(dose)

        if len(region_shape) == 3:
            for iZ in range(region_shape[2]):
                R1[...,iZ] = dose[...,iZ]>=_options.d_h
                R2[...,iZ] = dose[...,iZ]<=_options.d_l
        else:
            R1 = dose>=_options.d_h
            R2 = dose<=_options.d_l

        V1 = np.logical_and( core, np.logical_not(R1) )
        V2 = np.logical_and( background, np.logical_not(R2) )

        return R1, R2, V1, V2

    def loss(b_iter):
        global log_error
        if _options.bit_depth is not None:
            b_iter = vamtoolbox.util.data.discretize(b_iter,_options.bit_depth,[0.0,np.max(b_iter)])
        b_iter = np.reshape(b_iter,b0_shape)

        global dose_3D_iter
        dose_3D_iter = A.backward(b_iter)
        dose_3D_iter = dose_3D_iter*np.pi/180

        log_error.error_iter = vamtoolbox.metrics.calcVER(target_geo,dose_3D_iter)
        R1, R2, V1, V2 = getRegions(dose_3D_iter)

        V1_loc_iter = V1
        V2_loc_iter = V2

        loss_integrand = _options.rho_1*np.power(dose_3D_iter-_options.d_l, _options.p)*V2_loc_iter + _options.rho_2*np.power(_options.d_h-dose_3D_iter,_options.p)*V1_loc_iter 

        loss_iter = np.sum(loss_integrand).astype('double')
        log_error.loss.append(loss_iter)
        return loss_iter

    def lossGradient(b_iter):

        b_iter = np.reshape(b_iter,b0_shape)
        global dose_3D_iter
        # dose_3D_iter = A.backward(b_iter)
        # dose_3D_iter = dose_3D_iter*np.pi/180

        R1, R2, V1, V2 = getRegions(dose_3D_iter)

        V1_loc_iter = V1
        V2_loc_iter = V2

        grad_dose_iter = _options.rho_1*np.double(V2_loc_iter)*_options.p*np.power(dose_3D_iter-_options.d_l,_options.p-1) - _options.rho_2*np.double(V1_loc_iter)*_options.p*np.power(_options.d_h-dose_3D_iter,_options.p-1)

        grad_iter = A.forward(grad_dose_iter)

        grad_iter = np.reshape(grad_iter,np.product(b0_shape)).astype('double')

        factor = np.pi/180

        return grad_iter*factor

    def callback(x):
        global log_error
        global dose_3D_iter
        log_error.getIterTime(log_error.curr_iter+1)
        log_error.error[log_error.curr_iter+1] = log_error.error_iter
        if _options.verbose == 'time' or _options.verbose == 'plot':
            print('Iteration %4.0f at time: %6.1f s'%(log_error.curr_iter+1,log_error.iter_times[log_error.curr_iter+1]))

        if _options.verbose == 'plot':
            dp.update(log_error.error,dose_3D_iter)


        log_error.curr_iter += 1



    A = vamtoolbox.projectorconstructor.projectorconstructor(target_geo,proj_geo)

    if _options.verbose == 'plot':
        dp = vamtoolbox.display.EvolvingPlot(target_geo,_options.n_iter+1)



    core = np.zeros_like(target_geo.array,dtype=int)
    core[target_geo.gel_inds] = 1
    background = np.zeros_like(target_geo.array,dtype=int)
    background[target_geo.void_inds] = 1


    # begin timing
    t0 = time.perf_counter()

    
    # initialize sinogram with selected filter
    b0 = A.forward(target_geo.array)
    b0_shape = b0.shape
    b0 = vamtoolbox.util.data.filterSinogram(b0,_options.filter)
    b0 = np.clip(b0,a_min=0,a_max=None) 
    b0 = b0/np.max(b0)

    # set optimization constraints/bounds for each element of the sinogram
    bl, bh = 0, 1
    if b0.ndim == 3:
        bounds = [[[(bl,bh) for k in range(b0_shape[2]) ] for j in range(b0_shape[1]) ]  for i in range(b0_shape[0]) ]
    else:
        bounds = [[(bl,bh) for j in range(b0_shape[1]) ]  for i in range(b0_shape[0]) ]

    global log_error
    log_error = LogError(_options)

    x0 = A.backward(b0)
    x0 = x0/np.max(x0)
    log_error.error[0] = vamtoolbox.metrics.calcVER(target_geo,x0)
    
    log_error.iter_times[0] = time.perf_counter() - t0

    b0 = np.reshape(b0, np.product(b0_shape))
    bounds = np.reshape(bounds, (np.product(b0_shape), 2))

    lbfgs_options = {}
    lbfgs_options['maxiter'] = _options.n_iter
    lbfgs_options['ftol'] = 1e-12#np.finfo(float).eps
    # lbfgs_options['gtol'] = np.finfo(float).eps
    lbfgs_options['disp'] = False
    lbfgs_options['maxcor'] = 10
    lbfgs_options['maxls'] = 30

    t0 = time.perf_counter()
    result = optimize.minimize(fun=loss, 
                                x0=b0, 
                                callback=callback,
                                method='L-BFGS-B', 
                                jac=lossGradient, 
                                bounds=bounds, 
                                options=lbfgs_options)
    log_error.loss = np.array(log_error.loss)
    b_opt = np.reshape(result.x, b0_shape)
    if _options.bit_depth is not None:
        b_opt = vamtoolbox.util.data.discretize(b_opt,_options.bit_depth,[0.0,np.max(b_opt)])
    x_opt = A.backward(b_opt)
    x_opt = x_opt*np.pi/180
    if _options.verbose == 'plot':
        dp.ioff()
        
    return vamtoolbox.geometry.Sinogram(b_opt,proj_geo,_options), vamtoolbox.geometry.Reconstruction(x_opt,proj_geo,_options), log_error.error