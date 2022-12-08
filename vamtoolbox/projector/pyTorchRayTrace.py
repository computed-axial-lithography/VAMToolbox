import torch
import numpy as np
import matplotlib.pyplot as plt
import logging

class PyTorchRayTracingPropagator():
    """
    pyTorch implementation of custom ray tracing operation
    This operator is the A in Ax = b, or the P in Pf = g, where x or f is the real space object (flattened into 1D array) and b or g is the sinogram (flattened into 1D array).
    Here naming convention Pf = g is used to avoid collision with x which commonly denote position of ray in ray tracing literature.

    Since storing all curved ray path is prohibitively memory-intensive, this tracing operation is designed to have as low memory requirement as possible.
    Supposing a GPU option but fallback to CPU computation if GPU is not found.
    Currently occulsion is not supported yet.

    Internally, 2D and 3D problem is handled identically. Conditional handling only happens in I/O.
    #Implement ray_setup, coordinate systems
    """
    
    # pyTorch is used in inference mode with @torch.inference_mode().
    # Alternatively, @torch.no_grad() and torch.autograd.set_grad_enabled(False) can be used.

    @torch.inference_mode()
    def __init__(self, target_geo, proj_geo, output_torch_tensor = False) -> None:
        self.logger = logging.getLogger(__name__)

        self.target_geo = target_geo
        self.proj_geo = proj_geo
        self.output_torch_tensor = output_torch_tensor #Select if the output should be torch tensor or numpy array
        
        self.domain_n_dim = len(np.squeeze(self.target_geo.array).shape) #Dimensionality of the domain is set to be same as target 
        
        #Check if GPU is detected. Fallback to CPU if GPU is not found.
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        logging.info(f'Ray tracing computation is performed on: {repr(self.device)}')

        if isinstance(self.proj_geo.ray_trace_ray_config, RayState): #if state of a set of rays is already provided, use that as initial state.
            self.ray_state = self.proj_geo.ray_trace_ray_config
        else: #otherwise build initial ray state based on option 'parallel', 'cone'
            self.ray_state = RayState(device = self.device)
            self.ray_state.setupRays(self.proj_geo.ray_trace_ray_config,
                                    self.target_geo.constructCoordVec(),
                                    self.proj_geo.angles,
                                    self.proj_geo.inclination_angle,
                                    ray_density = self.proj_geo.ray_density,
                                    ray_width_fun = None
                                    )
        
        self.solver = RayTraceSolver(self.device, self.proj_geo.index_model, self.proj_geo.ray_trace_method, self.proj_geo.eikonal_parametrization, self.proj_geo.ray_trace_ode_solver)

    @torch.inference_mode()
    def forward(self, f):
        #Convert the input to a torch tensor if it is not
        if ~isinstance(f, torch.Tensor): #attempt to convert input to tensor
            f = torch.as_tensor(f, device = self.device)

        #Execution loop
        #Check ray exit condition, update exit flags
        all_rays_exited = False
        while (not all_rays_exited):
            #Compute v

            #Step forward
            self.solver.step

            #Check intersection here
            #Step back in case of intersection
            #Handle intersection events 
            #Deposition 
            
            #Update ray exit condition
            pass
        
    @torch.inference_mode()
    def backward(self):
        #Check if input is tensor
        pass

    @torch.inference_mode()
    def perAngleTrace(self):
        pass
    
    @torch.inference_mode()
    def perZTrace(self):
        pass

    @torch.inference_mode()
    def perRayTrace(self):
        pass


class RayTraceSolver():
    """
    ODE Solver to iterate over ray operations
    
    Store defined step size, selected stepping scheme

    index and its gradient field

    """
    @torch.inference_mode()
    def __init__(self, device, index_model, ray_trace_method, eikonal_parametrization, ode_solver, max_num_step = 750, num_step_per_exit_check = 10) -> None:
        self.logger = logging.getLogger(__name__) #Inputting the same name just output the same logger. No need to pass logger around.
        self.device = device
        self.index_model = index_model
        self.ray_trace_method = ray_trace_method
        self.eikonal_parametrization = eikonal_parametrization
        self.ode_solver = ode_solver

        self.max_num_step = max_num_step
        self.num_step_per_exit_check = num_step_per_exit_check
        
        #Parse ray_trace_method
        if self.ray_trace_method == 'snells' or self.ray_trace_method == 'hybrid':
            self.surface_intersection_check = True
        else:
            self.surface_intersection_check = False

        if self.ray_trace_method == 'eikonal' or self.ray_trace_method == 'hybrid':
            #Eikonal equation parametrization 
            if self.eikonal_parametrization == 'canonical':
                self.dx_dstep = self._dx_dsigma
                self.dv_dstep = self._dv_dsigma
            elif self.eikonal_parametrization == 'physical_path_length':
                #Physical path length
                self.dx_dstep = self._dx_ds
                self.dv_dstep = self._dv_ds
            elif self.eikonal_parametrization == 'optical_path_length':
                #Optical path length
                self.dx_dstep = self._dx_dopl
                self.dv_dstep = self._dv_dopl
            else:
                raise Exception(f'eikonal_parametrization = {self.eikonal_parametrization} and is not one of the available options.')



        #Parse ODE solver
        if self.ode_solver == 'forward_symplectic_euler':
            self.step_init = self._no_step_init
            self.step = self._forwardSymplecticEuler

        elif self.ode_solver == 'forward_euler':
            self.step_init = self._no_step_init
            self.step = self._forwardEuler

        elif self.ode_solver == 'leapfrog': 
            #leapfrog is identical to velocity verlet. Both differs from symplectic euler only by a half step in velocity in the beginning and the end.
            #However, they have higher order error performance
            self.step_init = self._leapfrog_init
            self.step = self._forwardSymplecticEuler

        elif self.ode_solver == 'rk4':
            pass #self.step = self._rk4
        else:
            raise Exception(f'ode_solver = {self.ode_solver} and is not one of the available options.')


    @torch.inference_mode()
    def solveUntilExit(self, ray_state, step_size, callback = None, tracker_on = False, track_every = 5):
        #Initialize ray tracker, where consecutive ray positions are recorded. Default to track every 5 rays
        if tracker_on:
            ray_tracker = torch.nan*torch.zeros((((ray_state.num_rays-1)//track_every)+1,3,self.max_num_step) ,device = self.device, dtype= ray_state.x_0.dtype) #ray_state.x_i[::track_every, :] has number of rows equal (ray_state.num_rays-1)//track_every)+1
        else:
            ray_tracker = None

        #Initialize stepping (only applies to certain methods, e.g. leapfrog)
        ray_state = self.step_init(ray_state, step_size)

        self.step_counter = 0
        exit_ray_count = 0
        all_ray_exited = False
        while (self.step_counter < self.max_num_step) and not all_ray_exited:
            
            if self.step_counter%self.num_step_per_exit_check == 0: #Check exit condition
                ray_state = self.exitCheck(ray_state) #Note: The operating set is always inverted ray_state.exited
                exit_ray_count = torch.sum(ray_state.exited, dim = 0) #.int()
                all_ray_exited = (exit_ray_count == ray_state.num_rays)

            ray_state = self.step(ray_state, step_size) #step forward. Where x_i and v_i will be replaced by x_ip1 and v_ip1 respectively

            if self.surface_intersection_check:
                self.discreteSurfaceIntersectionCheck(ray_state)

            if tracker_on:
                ray_tracker[:,:,self.step_counter] = ray_state.x_i[::track_every, :]

            if self.step_counter%self.num_step_per_exit_check == 0:
                self.logger.debug(f'completed {self.step_counter}th-step. Ray exited: {exit_ray_count}/{ray_state.num_rays}')
                
            self.step_counter += 1

        
        if not all_ray_exited:
            self.logger.error(f'Some rays have not exited before max_num_step ({self.max_num_step}) is reached. Rays exited: {exit_ray_count}/{ray_state.num_rays}.')

        return ray_state, ray_tracker

    #=======================Eikonal equation parametrizations======================
    #Canonical parameter, as sigma in "Adjoint Nonlinear Ray Tracing, Teh, et al. 2022"
    @torch.inference_mode()
    def _dx_dsigma(self, _, v, *arg):
        return v

    @torch.inference_mode()
    def _dv_dsigma(self, x, _, known_n = None):
        if known_n is None:
            n = self.index_model.n(x)[:,None]
        else:
            n = known_n

        return n*self.index_model.grad_n(x) #broadcasted to all xyz components

    #Physical path length, as s in "Eikonal Rendering: Efficient Light Transport in Refractive Objects, Ihrke, et al. 2007"
    @torch.inference_mode()
    def _dx_ds(self, x, v, known_n = None):
        if known_n is None:
            n = self.index_model.n(x)[:,None]
        else:
            n = known_n
        return v/n

    @torch.inference_mode()
    def _dv_ds(self, x, _, *arg):
        return self.index_model.grad_n(x) 

    #Optical path length, opl as t in "Eikonal Rendering: Efficient Light Transport in Refractive Objects, Ihrke, et al. 2007"
    @torch.inference_mode()
    def _dx_dopl(self, x, v, known_n = None):
        if known_n is None:
            n = self.index_model.n(x)[:,None]
        else:
            n = known_n
        return v*(n**(-2))

    @torch.inference_mode()
    def _dv_dopl(self, x, _, known_n = None):
        if known_n is None:
            n = self.index_model.n(x)[:,None]
        else:
            n = known_n
        return self.index_model.grad_n(x)/n
        
    #=======================ODE solvers======================
    @torch.inference_mode()
    def _forwardSymplecticEuler(self, ray_state, step_size): #step forward the RayState
        #push np1 to be n
        ray_state.x_i = ray_state.x_ip1
        ray_state.v_i = ray_state.v_ip1

        #Compute new np1 based on n (the old np1)
        #Compute v_ip1 using x_i
        ray_state.v_ip1 = ray_state.v_i + self.dv_dstep(ray_state.x_i, ray_state.v_i)*step_size

        #Then compute x_ip1 using v_ip1
        ray_state.x_ip1 = ray_state.x_i + self.dx_dstep(ray_state.x_i, ray_state.v_ip1)*step_size

        return ray_state

    @torch.inference_mode()
    def _forwardEuler(self, ray_state, step_size): #step forward the RayState
        #push np1 to be n
        ray_state.x_i = ray_state.x_ip1
        ray_state.v_i = ray_state.v_ip1

        #Compute new np1 based on n (the old np1)
        #Compute v_ip1 using x_i
        ray_state.v_ip1 = ray_state.v_i + self.dv_dstep(ray_state.x_i, ray_state.v_i)*step_size

        #Then compute x_ip1 using v_i
        ray_state.x_ip1 = ray_state.x_i + self.dx_dstep(ray_state.x_i, ray_state.v_i)*step_size

        return ray_state
    
    '''
    @torch.inference_mode()
    def _velocityVerlet(self, ray_state, step_size):  #Same as leapfrog
        
        #(Deprecated)
        #As written below, this form takes one more dv_dstep calculation than symplectic euler.
        #However, it doesn't have to be the case if there is a half step shift in v prior to the regular stepping.
        #That would result a verlet method (essentially a leapfrog) with same computational cost as symplectic euler, but with higher order accuracy.

        #push np1 to be n
        ray_state.x_i = ray_state.x_ip1
        ray_state.v_i = ray_state.v_ip1

        #Compute new np1 based on n (the old np1)
        #Compute v_ip1 using x_i (half step forward)
        ray_state.v_ip1 = ray_state.v_i + self.dv_dstep(ray_state.x_i)*step_size/2

        #Then compute x_ip1 using v_ip1 (full step forward)
        ray_state.x_ip1 = ray_state.x_i + self.dx_dstep(ray_state.v_ip1)*step_size

        ray_state.v_ip1 += self.dv_dstep(ray_state.x_ip1)*step_size/2 #Another half step for v, using updated x
        return ray_state
       
        return self._forwardSymplecticEuler(ray_state, step_size)
    '''

    @torch.inference_mode()
    def _leapfrog_init(self, ray_state, step_size):
        #Step backward ray_state.v_ip1 by half-step. After entering _forwardSymplecticEuler, v_ip1 will be half-step forward relative to x, when x is updated with v.
        #Initially x_0 = x_i = x_ip1, and v_0 = v_i = v_ip1, so it doesn't matter which input we use.
        ray_state.v_ip1 = ray_state.v_ip1 - self.dv_dstep(ray_state.x_ip1, ray_state.v_ip1)*step_size/2.0
        return ray_state

    @torch.inference_mode()
    def _no_step_init(self, ray_state, _):
        return ray_state

    #=======================Ray-voxel interactions======================
    @torch.inference_mode()
    def deposit(self): #deposit projected values along the ray
        #Check python line integral, ray-box intersection
        
        pass

    @torch.inference_mode()
    def integrate(self): #integrate real space quantities along the ray
        #Check python line integral, ray-box intersection
        
        pass

    @torch.inference_mode()
    def discreteSurfaceIntersectionCheck(self):
        pass
    
    @torch.inference_mode()
    def exitCheck(self, ray_state):
        '''
        When the rays are outside the bounding box AND pointing away. As long as this condition is satisfied in one of the dimension, the rest of the straight ray will not intersect the domain.
                
        #for (position < min-tol & direction < 0) or (position > max+tol & direction > 0)
        
        #exit = exit_x | exit_y | exit_z
        '''
        
        #Check for position if it is outside domain bounds (as defined by center of edge voxels). Check the coordinate PER DIMENSION, <min-tol or >max+tol
        #Distance tolerance is one voxel away from the bound (so the rest of the edge voxels are included, plus another half voxel in extra)
        exited_in_positive_direction = (ray_state.x_ip1 > self.index_model.xv_yv_zv_max + self.index_model.voxel_size) & (ray_state.v_ip1 > 0.0) #In case of 2D problems, voxel size along z is inf so the ray is always within the z-bounds
        exited_in_negative_direction = (ray_state.x_ip1 < self.index_model.xv_yv_zv_min - self.index_model.voxel_size) & (ray_state.v_ip1 < 0.0) #Therefore in 2D problems, the rays can only escape from x,y-bounds
        
        #Perform logical or in the spatial dimension axis
        exited_in_positive_direction = exited_in_positive_direction[:,0] | exited_in_positive_direction[:,1] | exited_in_positive_direction[:,2]
        exited_in_negative_direction = exited_in_negative_direction[:,0] | exited_in_negative_direction[:,1] | exited_in_negative_direction[:,2]
        # #Alternatively check logical OR with sum
        # exited_in_positive_direction = torch.sum(exited_in_positive_direction, dim = 1, keepdim = False).bool()
        # exited_in_negative_direction = torch.sum(exited_in_negative_direction, dim = 1, keepdim = False).bool()

        ray_state.exited =  exited_in_positive_direction | exited_in_negative_direction

        #Optionally check if intensity close to zero (e.g. due to occulsion)

        return ray_state
        


        
    
class RayState():
    '''
    State of a set of rays. The set can be all rays, rays emitted per angle, per z-slice, or per angle-z-slice
    This class is used as a dataclass to store current state of a set of rays, and where these rays originally belong to (x,theta,z) or (phi, theta, psi).
    For parallelization and generality, x_0 and v_0 has shape of (n_rays,n_dim).
    
    '''

    _dtype_to_tensor_type = {torch.float16 : torch.HalfTensor, torch.float32: torch.FloatTensor, torch.float64: torch.DoubleTensor} #Class attribute: mapping dtype to tensor type.
    
    @torch.inference_mode()
    def __init__(self, device, tensor_dtype = torch.float16, x_0 : torch.Tensor = None, v_0: torch.Tensor = None, sino_shape : tuple = None, sino_coord : list = None) -> None:
        self.device = device
        if tensor_dtype is not None:
            self.tensor_dtype = tensor_dtype
        else:
            self.tensor_dtype = torch.float16
        torch.set_default_tensor_type(self._dtype_to_tensor_type[self.tensor_dtype]) #A conversion between dtype and tensor type is needed since they are different objects and 'set_default_tensor_type' only accepts the latter.

        #Option to directly prescribe contents when ray positions and directions are generated externally (not using the provided methods)
        # self.x_0 = torch.as_tensor(x_0, device = self.device, dtype = self.tensor_dtype)
        # self.v_0 = torch.as_tensor(v_0, device = self.device, dtype = self.tensor_dtype)
        # self.sino_shape = sino_shape
        # self.sino_coord = sino_coord            
        # self.num_rays = self.x_0.size
        # self.s
        # self.int_factor
        # self.width 
        # self.integral
        # self.exited

    @torch.inference_mode()
    def setupRays(self, ray_trace_ray_config, target_coord_vec_list, azimuthal_angles_deg, inclination_angle_deg = 0, ray_density = 1, ray_width_fun = None):
        '''
        By default, ray_density = 1. 
        It means per azimuthal projection angle, there are max(nX, nY)*nZ number of rays, where nX,nY,nZ = target.shape.
        When ray_density =/= 1, number of rays = max(nX, nY)*nZ*(ray_density^2) because the density increase/decrease per dimension.
        '''
        #Preprocess inputs
        self.ray_trace_ray_config = ray_trace_ray_config #'parallel', 'cone' 

        #Convert all angles to radian
        azimuthal_angles_rad = azimuthal_angles_deg*np.pi/180.0
        if (inclination_angle_deg is None):
            inclination_angle_deg = 0 #In case none is passed down
        inclination_angle_rad = inclination_angle_deg * np.pi/180.0

        if ray_density is None:
            ray_density = 1

        #Generate initial ray position and direction according to their configuration
        if self.ray_trace_ray_config == 'parallel':
            self.setupRaysParallel(target_coord_vec_list, azimuthal_angles_rad, inclination_angle_rad, ray_density)
        elif self.ray_trace_ray_config == 'cone':
            raise Exception('Ray setup for cone beam is not yet implemented')
        else:
            raise Exception(f"Ray config: {str(self.ray_trace_ray_config)} is not one of the supported string: 'parallel', 'cone'.")
        
        #Initialize the iterate position and direction
        self.resetRaysIterateToInitial() #Initialize x_i and x_ip1 to be x_0. Same for v.

        #Initialize other properties of the rays
        self.s = torch.zeros((self.num_rays, 1), device = self.device, dtype = self.tensor_dtype) #distance travelled by the ray from initial position
        self.int_factor = torch.zeros((self.num_rays, 1), device = self.device, dtype = self.tensor_dtype) #current intensity relative to the intensity at x_0
        self.width = torch.ones((self.num_rays, 1), device = self.device, dtype = self.tensor_dtype) #constant if FOV is in depth of focus. Converging or Diverging if not. Function of s.
        self.integral = torch.zeros((self.num_rays, 1), device = self.device, dtype = self.tensor_dtype) #accumulating integral along the path
        self.exited = torch.zeros((self.num_rays, 1), device = self.device, dtype = torch.bool) #boolean value. Tracing of corresponding ray stops when its exited flag is true.

    @torch.inference_mode()
    def setupRaysParallel(self, target_coord_vec_list, azimuthal_angles_rad, inclination_angle_rad, ray_density):
        #The number of rays is assumed to be proportional to the size of real space grid/array
        real_nX = target_coord_vec_list[0].size 
        real_nY = target_coord_vec_list[1].size 
        real_nZ = target_coord_vec_list[2].size
        
        self.sino_shape = (round(max(real_nX,real_nY) * ray_density), azimuthal_angles_rad.size, max(round(real_nZ * ray_density),1)) #At least compute 1 z layer
        self.num_rays = self.sino_shape[0]*self.sino_shape[1]*self.sino_shape[2]

        #The following assumes the patterning volume is inscribed inside the simulation cube
        #===========Coordinates local to projection plane===================
        #Determine first coordinate of sinogram
        sino_n0_min = min(np.amin(target_coord_vec_list[0]),np.amin(target_coord_vec_list[1]))
        sino_n0_max = max(np.amax(target_coord_vec_list[0]),np.amax(target_coord_vec_list[1]))
        sino_n0 = np.linspace(sino_n0_min, sino_n0_max, self.sino_shape[0])
        
        #Determine second coordinate of sinogram
        sino_n1_rad = azimuthal_angles_rad

        #Determine third coordinate of sinogram.
        #Currently physical size of the projection is assumed to be equal to z height.
        #With moderate elevation, this assumption might be limiting patternable volume when bounding box has xy dimension >> z dimension. 
        sino_n2_min = np.amin(target_coord_vec_list[2])
        sino_n2_max = np.amax(target_coord_vec_list[2])
        sino_n2 = np.linspace(sino_n2_min, sino_n2_max, self.sino_shape[2])

        self.sino_coord = [sino_n0, sino_n1_rad, sino_n2]
        
        #There are a number of ways to determine the radial distance between center of grid to the projection plane where the rays starts. Longer distance requires more computation.
        #(1, smallest) Ellipsoidal (superset of spherical) patternable volume inscribed by the bounding box
        #The max distance along the 3 axes. Recall that sino_n0_max is the longer dimension in xy
        # proj_plane_radial_offset = max(sino_n0_max, sino_n2_max)

        #(2, middle) Elliptic cylindrical (superset of cylindrical) patternable volume inscribed by the bounding box.
        #When xy dimensions of the patternable volume is inscribed by the bounding box, while all of the z-extend of the array is assumed to be patternable.
        #Recall that sino_n0_max is the longer dimension in xy 
        #Correction: Maybe it is still like case 1, considering a axially long part, positive elevation and the bottom parts could be misesed.
        proj_plane_radial_offset = np.sqrt(sino_n0_max**2 + sino_n2_max**2)

        #(3, largest) The whole bounding box is patternable. Note: The existing vamtoolbox always cut to cylinder and discard the corner of the box.
        #When patternable volume is circumscribes the bounding box (the corners of the box is patternable/relevant), the radial distance between ray init position and the grid center is the diagonal of the 3D box.
        # proj_plane_radial_offset = np.sqrt(np.amax(target_coord_vec_list[0])**2 + np.amax(target_coord_vec_list[1])**2 + np.amax(target_coord_vec_list[2])**2)

        if CPU_create := False:
            #Create with CPU (which usually has access to more memory)
            G0, G1, G2 = np.meshgrid(sino_n0, sino_n1_rad, sino_n2, indexing='ij')
            G0 = G0.ravel()
            G1 = G1.ravel()
            G2 = G2.ravel()
            
            self.x_0 = np.ndarray((G0.size,3))
            #Center of projection plane relative to grid center. Refers to documentations for derivation.
            self.x_0[:,0] = proj_plane_radial_offset*np.cos(G1)*np.cos(inclination_angle_rad)
            self.x_0[:,1] = proj_plane_radial_offset*np.sin(G1)*np.cos(inclination_angle_rad)
            self.x_0[:,2] = proj_plane_radial_offset*np.sin(inclination_angle_rad)

            #Adding vectors from center of projection plane to pixel. Refers to documentations for derivation.
            self.x_0[:,0]+= -G0*np.sin(G1) - G2*np.cos(G1)*np.sin(inclination_angle_rad)
            self.x_0[:,1]+= G0*np.cos(G1) - G2*np.sin(G1)*np.sin(inclination_angle_rad)
            self.x_0[:,2]+= G2*np.cos(inclination_angle_rad)

            self.v_0 = np.ndarray((G0.size,3))
            self.v_0[:,0] = -np.cos(G1)*np.cos(inclination_angle_rad)
            self.v_0[:,1] = -np.sin(G1)*np.cos(inclination_angle_rad)
            self.v_0[:,2] = -np.sin(inclination_angle_rad)

            self.x_0 = torch.as_tensor(self.x_0, device = self.device)
            self.v_0 = torch.as_tensor(self.v_0, device = self.device)

        else: 
            #Create with GPU (which is faster if GPU memory is sufficient)
            inclination_angle_rad = torch.as_tensor(inclination_angle_rad, device = self.device, dtype = self.tensor_dtype)

            sino_n0 = torch.as_tensor(sino_n0, device = self.device, dtype = self.tensor_dtype)
            sino_n1_rad = torch.as_tensor(sino_n1_rad, device = self.device, dtype = self.tensor_dtype)
            sino_n2 = torch.as_tensor(sino_n2, device = self.device, dtype = self.tensor_dtype)
            G0, G1, G2 = torch.meshgrid(sino_n0, sino_n1_rad, sino_n2, indexing = 'ij')
            G0 = torch.ravel(G0)
            G1 = torch.ravel(G1)
            G2 = torch.ravel(G2)


            self.x_0 = torch.empty((self.num_rays,3), device = self.device, dtype = self.tensor_dtype) #using same date type as sino_n0, which is inferred from its numpy version
            #Center of projection plane relative to grid center. Refers to documentations for derivation.
            self.x_0[:,0] = proj_plane_radial_offset*torch.cos(G1)*torch.cos(inclination_angle_rad)
            self.x_0[:,1] = proj_plane_radial_offset*torch.sin(G1)*torch.cos(inclination_angle_rad)
            self.x_0[:,2] = proj_plane_radial_offset*torch.sin(inclination_angle_rad)

            #Adding vectors from center of projection plane to pixel. Refers to documentations for derivation.
            self.x_0[:,0]+= -G0*torch.sin(G1) - G2*torch.cos(G1)*torch.sin(inclination_angle_rad)
            self.x_0[:,1]+= G0*torch.cos(G1) - G2*torch.sin(G1)*torch.sin(inclination_angle_rad)
            self.x_0[:,2]+= G2*torch.cos(inclination_angle_rad)

            self.v_0 = torch.empty((self.num_rays,3), device = self.device, dtype = self.tensor_dtype)
            self.v_0[:,0] = -torch.cos(G1)*torch.cos(inclination_angle_rad)
            self.v_0[:,1] = -torch.sin(G1)*torch.cos(inclination_angle_rad)
            self.v_0[:,2] = -torch.sin(inclination_angle_rad)
    
    
    def resetRaysIterateToInitial(self):
        self.x_i = self.x_0
        self.x_ip1 = self.x_0 #torch.zeros_like(self.x_0)
        self.v_i = self.v_0
        self.v_ip1 = self.v_0 #torch.zeros_like(self.v_0)


    @torch.inference_mode()
    def plot_ray_init_position(self, angles_deg, color = 'black'):

        angles_rad = angles_deg*np.pi/180.0

        #Find the closest match
        angular_indices = np.round(np.interp(angles_rad, self.sino_coord[1], range(self.sino_coord[1].size), left=None, right=None, period=None))
        angles_selected_rad = np.array(self.sino_coord[1][angular_indices.astype(np.int32)], ndmin=1)

        G0, G1, G2 = np.meshgrid(self.sino_coord[0], self.sino_coord[1], self.sino_coord[2], indexing='ij')
        
        relevant_points = np.zeros_like(G1, dtype = np.bool_)
        #Find the selected points in the meshgrid
        for idx in range(angles_selected_rad.size):
            relevant_points = relevant_points | np.isclose(angles_selected_rad[idx],G1)

        relevant_points = np.ravel(relevant_points)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        x_0 = self.x_0.cpu()
        v_0 = self.v_0.cpu()
        #This only plot positions
        # ax.scatter(x_0[relevant_points,0], x_0[relevant_points,1], x_0[relevant_points,2], marker='o')
        #This plot both position and direction with arrows.
        ax.quiver(x_0[relevant_points,0],
                x_0[relevant_points,1],
                x_0[relevant_points,2],
                v_0[relevant_points,0],
                v_0[relevant_points,1],
                v_0[relevant_points,2],
                length=0.15,
                color = color)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()


    class RaySelector():
        def __init__(self, ray_state) -> None:
            self.device = ray_state.device
            self.num_rays = ray_state.num_rays
            self.sino_coord = ray_state.sino_coord

            self.selectParallelAngles = self.select

            self.selected_idx = torch.zeros((self.num_rays, 1), device = self.device, dtype = torch.bool)

        def selectCoord(self, angles, mode = 'and'):
            #mode: 'and', 'or', 'xor'. This is the relationship BETWEEN the sinogram coordinates.
            #AND will only select rays that simultaneously satisfy requirements for sino_n0, sino_n1, sino_n2
            #OR will select rays that satisfy any requirements for sino_n0, sino_n1, sino_n2
            #So is XOR and NOT


            
            return self.selected_idx

        def selectInverse(self):
            #invert current selection
            pass

        def _selectSino0(self, sino0_coord):
            return #idx

        def _selectSino1(self, sino0_coord):
            return #idx

        def _selectSino2(self, sino0_coord):
            return #idx

    # class RayTracker(RaySelector):
    #     def __init__(self, ray_state, angles_rad, ) -> None:
    #         ray_state.sino_coord

    #         #Find the closest match ray

    #         #Determine their indices in vector x

    #         #Store the indices

    #     def plotRays(self):

    #         #Color each rays from each angle.
    #         fig = plt.figure()
    #         ax = plt.axes(projection='3d')

    #         for ind in [ind for ind in range(x_np_record.shape[0]) if (ind)%1==0]:
    #             xline = x_np_record[ind,0,:]
    #             yline = x_np_record[ind,1,:]
    #             zline = x_np_record[ind,2,:]
    #             ax.plot3D(xline, yline, zline, 'blue', linewidth=0.2)

    #         ax.set_xlabel('x')
    #         ax.set_ylabel('y')
    #         ax.set_zlabel('z')
    #         ax.set_aspect('equal', 'box')