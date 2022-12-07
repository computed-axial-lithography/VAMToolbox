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
        
        self.solver = RayTraceSolver(self.device, self.proj_geo.index_model, self.proj_geo.ray_trace_method, self.proj_geo.ray_trace_ode_solver)

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
    def __init__(self, device, index_model, ray_trace_method, ode_solver, max_num_step = 750) -> None:
        self.logger = logging.getLogger(__name__) #Inputting the same name just output the same logger. No need to pass logger around.
        self.device = device
        self.index_model = index_model
        self.ray_trace_method = ray_trace_method
        self.ode_solver = ode_solver

        self.max_num_step = max_num_step
        self.step_counter = 0
        
        #Parse ray_trace_method
        if self.ray_trace_method == 'snells' or self.ray_trace_method == 'hybrid':
            self.surface_intersection_check = True
        else:
            self.surface_intersection_check = False

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

        #Equation form
        self.dx_dstep = self._dx_dsigma
        self.dv_dstep = self._dv_dsigma


    @torch.inference_mode()
    def solveUntilExit(self, ray_state, step_size, callback = None, tracker_on = False, track_every = 5):
        #Initialize ray tracker, where consecutive ray positions are recorded. Default to track every 5 rays
        if tracker_on:
            ray_tracker = torch.zeros(((ray_state.num_rays//track_every),3,self.max_num_step) ,device = self.device, dtype= ray_state.x_0.dtype)
        else:
            ray_tracker = None

        #Initialize stepping (only applies to certain methods, e.g. leapfrog)
        ray_state = self.step_init(ray_state, step_size)

        while self.step_counter < self.max_num_step:
            
            #Check exit condition
            ray_state = self.step(ray_state, step_size)

            if self.surface_intersection_check:
                self.discreteSurfaceIntersectionCheck(ray_state)


            if tracker_on:
                ray_tracker[:,:,self.step_counter] = ray_state.x_n[::track_every, :]


            if self.step_counter%10 == 0:
                self.logger.debug(f'completed {self.step_counter}th-step')
            self.step_counter += 1

        # termination_type = None
        return ray_state, ray_tracker

    #=======================ODE solvers======================
    @torch.inference_mode()
    def _forwardSymplecticEuler(self, ray_state, step_size): #step forward the RayState
        #push np1 to be n
        ray_state.x_n = ray_state.x_np1
        ray_state.v_n = ray_state.v_np1

        #Compute new np1 based on n (the old np1)
        #Compute v_np1 using x_n
        ray_state.v_np1 = ray_state.v_n + self.dv_dstep(ray_state.x_n)*step_size

        #Then compute x_np1 using v_np1
        ray_state.x_np1 = ray_state.x_n + self.dx_dstep(ray_state.v_np1)*step_size

        return ray_state

    @torch.inference_mode()
    def _forwardEuler(self, ray_state, step_size): #step forward the RayState
        #push np1 to be n
        ray_state.x_n = ray_state.x_np1
        ray_state.v_n = ray_state.v_np1

        #Compute new np1 based on n (the old np1)
        #Compute v_np1 using x_n
        ray_state.v_np1 = ray_state.v_n + self.dv_dstep(ray_state.x_n)*step_size

        #Then compute x_np1 using v_n
        ray_state.x_np1 = ray_state.x_n + self.dx_dstep(ray_state.v_n)*step_size

        return ray_state
    
    '''
    @torch.inference_mode()
    def _velocityVerlet(self, ray_state, step_size):  #Same as leapfrog
        
        #(Deprecated)
        #As written below, this form takes one more dv_dstep calculation than symplectic euler.
        #However, it doesn't have to be the case if there is a half step shift in v prior to the regular stepping.
        #That would result a verlet method (essentially a leapfrog) with same computational cost as symplectic euler, but with higher order accuracy.

        #push np1 to be n
        ray_state.x_n = ray_state.x_np1
        ray_state.v_n = ray_state.v_np1

        #Compute new np1 based on n (the old np1)
        #Compute v_np1 using x_n (half step forward)
        ray_state.v_np1 = ray_state.v_n + self.dv_dstep(ray_state.x_n)*step_size/2

        #Then compute x_np1 using v_np1 (full step forward)
        ray_state.x_np1 = ray_state.x_n + self.dx_dstep(ray_state.v_np1)*step_size

        ray_state.v_np1 += self.dv_dstep(ray_state.x_np1)*step_size/2 #Another half step for v, using updated x
        return ray_state
       
        return self._forwardSymplecticEuler(ray_state, step_size)
    '''

    @torch.inference_mode()
    def _leapfrog_init(self, ray_state, step_size):
        #Step backward ray_state.v_np1 by half-step. After entering _forwardSymplecticEuler, v_np1 will be half-step forward relative to x, when x is updated with v.
        ray_state.v_np1 = ray_state.v_np1 - self.dv_dstep(ray_state.x_n)*step_size/2.0
        return ray_state

    @torch.inference_mode()
    def _no_step_init(self, ray_state, _):
        return ray_state

    #=======================Equation form======================
    @torch.inference_mode()
    def _dx_dsigma(self, v):
        return v

    @torch.inference_mode()
    def _dv_dsigma(self, x):
        # return self.index_model.n(x)*self.index_model.grad_n(x) #broadcasted to all xyz components
        n = self.index_model.n(x)[:,None]
        grad_n = self.index_model.grad_n(x)
        return n*grad_n
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
    def exitCheck(self):
        #When the rays are outside the bounding box AND pointing awayw
        #Distance tolerance

        #Check the coordinate PER DIMENSION, <min-tol or >max+tol
        #then check sign of v for such dimension
        #for (position < min-tol & direction < 0) or (position > max+tol & direction > 0)
        
        #exit = exit_x | exit_y | exit_z

        #Check if intensity close to zero (e.g. due to occulsion)

        pass
        


        
    
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
        self.x_n = self.x_0
        self.x_np1 = self.x_0 #torch.zeros_like(self.x_0)
        self.v_n = self.v_0
        self.v_np1 = self.v_0 #torch.zeros_like(self.v_0)

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
        
        self.sino_shape = (round(max(real_nX,real_nY) * ray_density), azimuthal_angles_rad.size, round(real_nZ * ray_density))
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


