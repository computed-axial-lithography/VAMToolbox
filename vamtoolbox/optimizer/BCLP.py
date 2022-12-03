from distutils.log import warn
import numpy as np
import time
from scipy import optimize
import warnings
import matplotlib.pyplot as plt
import vamtoolbox

class LogPerf:
    #Object to store performance metric values and options.
    def __init__(self,options):
        self.options = options
        self.curr_iter = 0
        self.loss = np.zeros(options.n_iter+1)*np.nan #including the init
        
        self.l0 = np.zeros(options.n_iter+1)*np.nan #L0 norm evaluated with equal weights, q=1, eps as in optimization
        self.l1 = np.zeros(options.n_iter+1)*np.nan #L1 norm evaluated with equal weights, q=1, eps as in optimization
        self.l2 = np.zeros(options.n_iter+1)*np.nan #L2 norm evaluated with equal weights, q=1, eps as in optimization
        self.lf = np.zeros(options.n_iter+1)*np.nan #Linf norm evaluated with equal weights, q=1, eps as in optimization

        self.l0_eps0 = np.zeros(options.n_iter+1)*np.nan #L0 norm evaluated with equal weights, q=1, eps=0
        self.l1_eps0 = np.zeros(options.n_iter+1)*np.nan #L1 norm evaluated with equal weights, q=1, eps=0
        self.l2_eps0 = np.zeros(options.n_iter+1)*np.nan #L2 norm evaluated with equal weights, q=1, eps=0
        self.lf_eps0 = np.zeros(options.n_iter+1)*np.nan #Linf norm evaluated with equal weights, q=1, eps=0
        
        self.ver = np.zeros(options.n_iter+1)*np.nan

        self.iter_times = np.zeros(options.n_iter+1)
        self.t0 = np.nan

    def startTiming(self):
        self.t0 = time.perf_counter()

    def recordIterTime(self):
        self.iter_times[self.curr_iter] = time.perf_counter() - self.t0

    def show(self, normalation_flags = (True, True, True, True)):
        #Plot cost function and all other metrics
        fig, axs = plt.subplots(1,3) #Create figure
        fig.set_size_inches(12, 8) #Resize figure
        error_plot = vamtoolbox.displaygrayscale.MultiErrorPlot(self.loss,ax=axs[0],fig=fig, title= 'Weighted $L_{p}$ norm cost function', xlabel= 'Iteration', ylabel='Cost')
        error_lp_plot = vamtoolbox.displaygrayscale.MultiErrorPlot(
            self.l0,
			self.l1,
			self.l2,
			self.lf,
			ax=axs[1],
			fig=fig,
			title= r'Other $L_{p}$ norms with original tolerance $\varepsilon$',
			xlabel= 'Iteration',
			ylabel='Normalized norm value',
			legends = ['$L_{0}$','$L_{1}$','$L_{2}$','$L_{\infty}$'],
			normalization_flags= normalation_flags
			)

        error_lp_eps0_plot = vamtoolbox.displaygrayscale.MultiErrorPlot(
            self.l0_eps0,
			self.l1_eps0,
			self.l2_eps0,
			self.lf_eps0,
			ax=axs[2],
			fig=fig,
			title= r'Other $L_{p}$ norms with zero tolerance $\varepsilon = 0$',
			xlabel= 'Iteration',
			ylabel='Normalized norm value',
			legends = ['$L_{0}$','$L_{1}$','$L_{2}$','$L_{\infty}$'],
			normalization_flags= normalation_flags
			)
        
        plt.show()
        return fig, axs, error_plot, error_lp_plot, error_lp_eps0_plot

class BCLPNorm:
    """
    This class provide loss function and gradient methods to external optimizer.
    It also temporarily stores the state of the variables (including error) implicitly during the optimization.
    
    Naming convention:
    Ax=b in algebraic reconstruction corresponds to Pf=g in this code and in the paper.
    """
    
    def __init__(self, target_geo, proj_geo, options, g0 = None):
        self.target_geo = target_geo
        self.proj_geo = proj_geo
        self.tomogram_scale = 1/proj_geo.n_angles #This scale will be applied to reconstruction after P.backward()

        #Initialize performance logger
        self.logs = LogPerf(options)
        self.logs.startTiming()

        #Unpack options
        # self.__dict__.update(options.__dict__) #unpack the option variables to itself

        #Renamed variables has a separate copy
        self.response_model = options.response_model
        self.eps = options.eps
        self.weight = options.weight
        self.p = options.p
        self.q = options.q
        self.learning_rate = options.learning_rate
        self.optim_alg = options.optim_alg
        self.glb = options.blb
        self.gub = options.bub
        self.bit_depth = options.bit_depth
        self.dvol = 1 #differential volume in integration
        self.verbose = options.verbose
        self.save_img_path = options.save_img_path

        #State variables, in the order of computation
        self.dose = None
        self.mapped_dose = None
        self.mapped_dose_error_from_f_T = None #error from f_T. Without subtracting epsilon
        self.mapped_dose_error_from_band = None #error from f_T. Without subtracting epsilon
        self.v = None #indicator function of constraint violation

        #Iteration index of each variable.
        self.dose_iter = -1
        self.mapped_dose_iter = -1
        self.mapped_dose_error_from_f_T_iter = -1
        self.mapped_dose_error_from_band_iter = -1
        self.v_iter = -1
        self.loss_iter = -1
        self.loss_grad_iter = -1

        if self.verbose == 'plot':
            self.dp = vamtoolbox.displaygrayscale.EvolvingPlot(target_geo,self.logs.options.n_iter+1, self.save_img_path)
        elif self.verbose == 'plot_demo':
            self.dp = vamtoolbox.displaygrayscale.EvolvingPlotDemo(target_geo,self.logs.options.n_iter+1, self.save_img_path)


        #Setup projector
        self.P = vamtoolbox.projectorconstructor.projectorconstructor(self.target_geo, self.proj_geo)

        # self.custom_fig1, self.custom_ax1 = plt.subplots()
        # self.custom_fig2, self.custom_ax2 = plt.subplots()

        #Check if input f_T in range of response function over non-negative real
        if self.response_model.checkResponseTarget(self.target_geo.array) is False:
            warnings.warn("Target is either out of range of response function over non-negative real dose input, or contains inf/nan.")

        #Initialize sinogram, if it is not provided.
        if g0 is None:
            self.g0 = self.P.forward(self.response_model.map_inv(self.target_geo.array))
            self.g0_shape = self.g0.shape
            self.g0 = vamtoolbox.util.data.filterSinogram(self.g0, options.filter)
            self.g0 = self.imposeSinogramConstraints(self.g0)
        else:
            self.g0 = g0

        # #Evaluate performance of initialization --> Turns out it should not be run here. It will be executed in the first iteration anyway.
        # self.updateVariables(self.g0) #update loss and gradient.
        # self.callback(self.g0) #Record other metrics and initialization time as iter 0


    def updateVariables(self, g_iter):
        """
        This function checks if the state variables are up-to-date, and update them if not.
        The function ensures the variables are (1) computed in a specified order (to avoid accessing old copy), and (2) computed only once.
        Therefore regardless of when external function query loss or loss gradient, there is no ambiguity of when each variable is updated.
        In the following code, any particular line can safely assume the variables above it are up-to-date.
        """

        if self.dose_iter != self.logs.curr_iter:
            g_iter = self.checkSinogramShape(g_iter, desired_shape = "cylindrical")
            self.dose  = self.P.backward(g_iter)*self.tomogram_scale
            self.dose_iter = self.logs.curr_iter

        if self.mapped_dose_iter != self.logs.curr_iter:
            self.mapped_dose = self.response_model.map(self.dose)
            self.mapped_dose_iter = self.logs.curr_iter

        if self.mapped_dose_error_from_f_T_iter != self.logs.curr_iter:
            self.mapped_dose_error_from_f_T = self.mapped_dose - self.target_geo.array
            self.mapped_dose_error_from_f_T_iter = self.logs.curr_iter

        if self.mapped_dose_error_from_band_iter != self.logs.curr_iter:
            #"mapped_dose_error_from_band" is by definition positive in the constraint violation zone, and either zero or negative outside.
            self.mapped_dose_error_from_band = np.abs(self.mapped_dose_error_from_f_T) - self.eps
            self.mapped_dose_error_from_band_iter = self.logs.curr_iter

        if self.v_iter != self.logs.curr_iter:
            self.v = (self.mapped_dose_error_from_band > 0)
            self.v_iter = self.logs.curr_iter

        if self.loss_iter != self.logs.curr_iter:
            self.loss = self.computeLoss()
            self.loss_iter = self.logs.curr_iter

        if self.loss_grad_iter != self.logs.curr_iter:
            self.loss_grad = self.computeLossGradient()
            self.loss_grad_iter = self.logs.curr_iter


    def computeLoss(self):
        """
        Computation of the loss function
        """

        #The absolute sign around "mapped_dose_error_from_band" (as in the definition of Lp norm) is preserved here, although simiplification in the paper hides it.
        #This absolute sign is to avoid letting the negative values of mapped_dose_error_from_band to create NaNs in **p operation when p<1. The NaNs would avoid the sum to be properly evaluated.
        #These originally negative voxels would not contribute to the loss_integrand due to selection by v. 
        loss_integrand = self.v*self.weight*np.abs(self.mapped_dose_error_from_band)**self.p

        # loss = (np.sum(loss_integrand).astype('double')*self.dvol)**(self.q/self.p) #multiply by a constant differential volume, self.dvol. #We may not need the astype('double')
        loss = (np.sum(loss_integrand)*self.dvol)**(self.q/self.p) #multiply by a constant differential volume, self.dvol
        
        self.logs.loss[self.logs.curr_iter] = loss

        return loss



    def computeLossGradient(self):
        """
        Computation of the loss gradient. Output gradient as a flattened array.
        """

        #The absolute sign around "mapped_dose_error_from_band" (as in the definition of Lp norm) is preserved here, although simiplification in the paper hides it.
        #This absolute sign is to avoid letting the negative values of mapped_dose_error_from_band to create NaNs in **(p-1) operation when p<2. The NaNs would start to spread throughout g_iter can cause errors.
        #These originally negative voxels would not contribute to the operand due to selection by v. 
        operand = self.v * self.weight * (np.abs(self.mapped_dose_error_from_band)**(self.p-1)) * np.sign(self.mapped_dose_error_from_f_T) * self.response_model.dmapdf(self.dose)

        #Computation of gradient happens between the two iterations. Now the index just incremented, but we need to access the loss evaluated in last iteration
        loss_grad =  ( self.q * self.loss**((self.q - self.p)/self.q) ) * self.P.forward(operand) 

        loss_grad = self.checkSinogramShape(loss_grad, desired_shape= "flattened")

        return loss_grad



    def getLoss(self, g_iter):
        self.updateVariables(g_iter)
        return self.loss

    def getLossGrad(self, g_iter):
        self.updateVariables(g_iter)
        return self.loss_grad


    def evaluateNormMetrics(self):
        #All following norms are evaluated with eps = original value and eps = 0
        #l0 and l0_eps0

        # l0 = np.sum(self.v).astype('double')*self.dvol #multiply by a constant differential volume, self.dvol
        l0 = np.sum(self.v)*self.dvol #multiply by a constant differential volume, self.dvol
        self.logs.l0[self.logs.curr_iter] = l0
        
        v_eps0 = (np.abs(self.mapped_dose_error_from_f_T) > 0)
        l0_eps0 = np.sum(v_eps0)*self.dvol #multiply by a constant differential volume, self.dvol
        self.logs.l0_eps0[self.logs.curr_iter] = l0_eps0

        #l1 and l1_eps0
        l1 = np.sum(self.v*np.abs(self.mapped_dose_error_from_band))*self.dvol
        self.logs.l1[self.logs.curr_iter] = l1

        l1_eps0 = np.sum(np.abs(self.mapped_dose_error_from_f_T))*self.dvol  #multiplication with v_eps0 is suppressed for performance. Such binary mask is unnecessary in this case. 
        self.logs.l1_eps0[self.logs.curr_iter] = l1_eps0

        #l2 and l2_eps0 
        l2 = np.sqrt(np.sum(self.v*(self.mapped_dose_error_from_band**2))*self.dvol)
        self.logs.l2[self.logs.curr_iter] = l2

        l2_eps0 = np.sqrt(np.sum(self.mapped_dose_error_from_f_T**2)*self.dvol)  #multiplication with v_eps0 is suppressed for performance. Such binary mask is unnecessary in this case.
        self.logs.l2_eps0[self.logs.curr_iter] = l2_eps0

        #lf and lf_eps0 
        #The existance of self.v here is important. Without it, we can't select only the constraint violation only.
        #In the case where all voxels have negative errors (constraint satisfaction), these values could possibly show up either as negative (without abs) and positive (with abs)
        lf = np.amax(self.v*np.abs(self.mapped_dose_error_from_band))  
        self.logs.lf[self.logs.curr_iter] = lf
        
        lf_eps0 = np.amax(v_eps0 * np.abs(self.mapped_dose_error_from_f_T))  
        self.logs.lf_eps0[self.logs.curr_iter] = lf_eps0
        

    def callback(self, g_iter):
        #This function evaluate other metrics and record iter time
        #Evaluate other metrics
        self.updateVariables(g_iter)
        self.evaluateNormMetrics()

        #record iter time
        self.logs.recordIterTime()
        if self.verbose == 'time' or self.verbose == 'plot' or self.verbose == 'plot_demo':
            print(f'Iteration {self.logs.curr_iter: 4.0f} at time: { self.logs.iter_times[self.logs.curr_iter]: 6.1f} s')
        
        if self.verbose == 'plot':
            self.dp.update(self.logs.loss, self.dose, self.mapped_dose, [self.logs.l0, self.logs.l1, self.logs.l2, self.logs.lf, self.logs.l0_eps0, self.logs.l1_eps0, self.logs.l2_eps0, self.logs.lf_eps0])
        elif self.verbose == 'plot_demo':
            self.dp.update(self.logs.loss, self.dose, self.mapped_dose)

        self.logs.curr_iter += 1


    def gradientDescent(self): 
        #TODO: using scipy optimize method format such that it could be called by "minimize" front end
        g_iter = self.checkSinogramShape(self.g0, desired_shape = 'flattened')

        #Impose constraint on latest g here
        for iter in range(self.logs.options.n_iter): 
            #The first iteration here evaluate the initial solution performance, and generate the next g_iter (for iteration 1).
            #Variables and performance are evaluated at the beginning of each loop iteration (by construction). g_iter generated is for the next iteration.
            #This process is repeated for n_iter times.
            g_iter = g_iter - self.learning_rate * self.getLossGrad(g_iter)
            g_iter = self.imposeSinogramConstraints(g_iter)
            
            #Evaluate performance
            self.getLoss(g_iter)
            self.callback(g_iter)
            
        self.updateVariables(g_iter) #Update the variables according to the last generated g_iter
        self.evaluateNormMetrics() #compute the metrics for the last generated g_iter

        if self.verbose == 'plot':
            self.dp.update(self.logs.loss, self.dose, self.mapped_dose, [self.logs.l0, self.logs.l1, self.logs.l2, self.logs.lf, self.logs.l0_eps0, self.logs.l1_eps0, self.logs.l2_eps0, self.logs.lf_eps0])
        elif self.verbose == 'plot_demo':
            self.dp.update(self.logs.loss, self.dose, self.mapped_dose)

        return g_iter


    def imposeSinogramConstraints(self, g):
        if self.bit_depth is not None:
            g = vamtoolbox.util.data.discretize(g, self.bit_depth, [0.0, np.amax(g)])

        g = np.clip(g, a_min=self.glb, a_max=self.gub) 

        return g

    def checkSinogramShape(self, g, desired_shape = 'flattened'):
        #This function check the shape of sinogram array and reshape it if needed. 
        if desired_shape == 'flattened':
            if len(g.shape)> 1:
                g = np.reshape(g, np.product(g.shape))
        
        elif desired_shape == 'cylindrical':
            if len(g.shape) == 1:
                g = np.reshape(g, self.g0_shape)
        
        return g
        


def minimizeBCLP(target_geo, proj_geo, options, output = "packaged"):
    """
    Band constraint Lp norm minimization. 

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

    """
    
    #Initialize norm class

    bclp = BCLPNorm(target_geo, proj_geo, options)
    g_opt = bclp.gradientDescent()
    g_opt = bclp.checkSinogramShape(g_opt, desired_shape = "cylindrical")

    if bclp.verbose == 'plot':
        bclp.dp.ioff()



    if output == "packaged":
        return vamtoolbox.geometry.Sinogram(g_opt, proj_geo, options), vamtoolbox.geometry.Reconstruction(bclp.dose, proj_geo, options), bclp.logs.loss
    elif output == "full":
        return vamtoolbox.geometry.Sinogram(g_opt, proj_geo, options), vamtoolbox.geometry.Reconstruction(bclp.dose, proj_geo, options), vamtoolbox.geometry.Reconstruction(bclp.mapped_dose, proj_geo, options), bclp.logs
    elif output == "components":
        return g_opt, bclp.dose, bclp.mapped_dose, bclp.logs

    #Future improvement: Parse options and run other optimization methods in Scipy minimize 
    # lbfgs_options = {}
    # lbfgs_options['maxiter'] = options.n_iter
    # lbfgs_options['ftol'] = 1e-12#np.finfo(float).eps
    # # lbfgs_options['gtol'] = np.finfo(float).eps
    # lbfgs_options['disp'] = False
    # lbfgs_options['maxcor'] = 10
    # lbfgs_options['maxls'] = 30

    # result = optimize.minimize(fun=loss, 
    #                         x0=b0, 
    #                         callback=callback,
    #                         method='L-BFGS-B', 
    #                         jac=lossGradient, 
    #                         bounds=bounds, 
    #                         options=lbfgs_options)