import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import torch

class IndexModel:
    _default_analytical = {'R': 1.0, 'length_unit': 'fractional_domain_radius', 'p': 2.0, 'n_sur' : 1.0} #default arguments for **kwargs
    _default_interpolation = {}

    @torch.inference_mode()
    def __init__(self, coord_vec : np.ndarray, type : str = 'analytical', form :str = 'luneburg_lens', **kwargs):
        '''
        Analytical and interpolated decription of index distribution of the simulation domain.
        Internally implemented with pyTorch tensor for GPU computation in ray tracing.
        Input/output compatibility with numpy array is to be added.

        Naming: n(x) and grad_n(x) is the index and spatial gradient of index as a function of position x.
        The functions returns the values as a tensor with the same shape as the first 3 dimension of the input.

        Parameters
        ----------
        type : str ('analytical', 'interpolation')
            Select analytical function evaluation or interpolate on pre-built interpolant arrays. 
            Interpolation method handles edge cases of input explicitly and hence is more robust.

        form : str ('homogeneous', 'luneburg_lens', 'maxwell_lens', 'eaton_lens', 'freeform')

        R : float, optional
            Radius of the luneburg/maxwell/eaton lens
            Default equals to the radius of the grid (lens occupy almost all of simulation volume), regardless of length_unit parameter.
            
        length_unit: str, optional
            unit used in specification of R only. The grid size is always assumed to be the same as target_geo.target/
            'fractional_domain_radius', 'physical_coordinate'
            Default: 'fractional_domain_radius', where 1 represent a lens filling the domain with boundaries touching
            'physical_coordinate' uses the same unit as 

        p : float, power of luneburg lens
            Default 2, which is used in classical definition of luneburg lens.

        n_sur : float, index of the surroundings (at the wavelength of interest)
            The absolute index of the lenses or other analytical index distribution will be adjusted accordingly to achieve equivalent refraction.
            Also, this sets the extrapolation value when index is queried outside the grid
        '''
        self.logger = logging.getLogger(__name__)
        
        #Save inputs as attributes
        self.type = type
        self.form = form

        #Merge all kwargs into params
        self.params = self._default_analytical.copy() #Shallow copy avoid editing dict '_default_lenses' in place 
        self.params.update(self._default_interpolation) #Add relevant parameters
        self.params.update(kwargs) #up-to-date parameters. Default dict is not updated

        # Compute derived attributes R_physical
        if self.params['length_unit'] == 'fractional_domain_radius':
            R_domain_0 = (np.amax(coord_vec[0]) - np.amin(coord_vec[0]))/2
            R_domain_1 = (np.amax(coord_vec[1]) - np.amin(coord_vec[1]))/2
            R_domain_2 = (np.amax(coord_vec[2]) - np.amin(coord_vec[2]))/2
            if coord_vec[2].size == 1: #only fit in xy box in 2D case
                R_domain_min = min(R_domain_0, R_domain_1)
            else: #fit in xyz box in 3D case
                R_domain_min = min(R_domain_0, R_domain_1, R_domain_2)
            self.params['R_physical'] = self.params['R']*R_domain_min

        elif self.params['length_unit'] == 'physical':
            self.params['R_physical'] = self.params['R']

        else:
            raise Exception('"length_unit" should be either "fractional_domain_radius" or "physical"')


        #Construct meshgrid in tensor
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.logger.info(f'Medium computation is performed on: {repr(self.device)}')

        #Set up grid as tensor
        self.xv = torch.as_tensor(coord_vec[0], device = self.device)
        self.yv = torch.as_tensor(coord_vec[1], device = self.device)
        self.zv = torch.as_tensor(coord_vec[2], device = self.device)
        self.Xg, self.Yg, self.Zg = torch.meshgrid(self.xv, self.yv, self.zv, indexing = 'ij') #This grid is to be used for interpolation, simulation
        
        #Max min of grid coordinate
        self.xv_yv_zv_max = torch.tensor([torch.amax(self.xv), torch.amax(self.yv), torch.amax(self.zv)], device = self.device)
        self.xv_yv_zv_min = torch.tensor([torch.amin(self.xv), torch.amin(self.yv), torch.amin(self.zv)], device = self.device)
        if self.zv.numel() == 1:
            self.voxel_size = torch.tensor([self.xv[1]-self.xv[0], self.yv[1]-self.yv[0], 0], device = self.device) #Sampling rate along z is 0
        else:
            self.voxel_size = torch.tensor([self.xv[1]-self.xv[0], self.yv[1]-self.yv[0], self.zv[1]-self.zv[0]], device = self.device) #Only valid when self.zv.size > 1
        #==========================Setup according to type and form================================
        if self.type == 'analytical':
            # self.Xg, self.Yg, self.Zg = None, None, None
            # self.params = self._default_analytical.copy() #Shallow copy avoid editing dict '_default_lenses' in place 
            # self.params.update(kwargs) #up-to-date parameters. Default dict is not updated

            if self.form == 'homogeneous':
                self.n = self._n_homo
                self.grad_n = self._grad_n_homo
                
            elif self.form == 'luneburg_lens':  
                self.n = self._n_lune
                self.grad_n = self._grad_n_lune

            elif self.form == 'maxwell_lens':
                self.n = self._n_maxwell
                self.grad_n = self._grad_n_maxwell

            elif self.form == 'eaton_lens':
                self.n = self._n_eaton
                self.grad_n = self._grad_n_eaton

            else:
                raise Exception('Form: Other analytical functions are not supported yet.')

        elif self.type == 'interpolation':
            #Interpolation method stores three 3D arrays as interpolant and query them upon each mapping call.
            #Stored arrays : (1) index 'n_x' and (2)gradient of n 'grad_n_x'.
            #They are sampled at the grid defined by coord_vec.
    
            #function alias
            self.n = self._n_interp
            self.grad_n = self._grad_n_interp

            #build or import interpolant dataset 
            if self.form == 'homogeneous':
                #build interpolant arrays
                self.interp_x_sample = torch.vstack(torch.ravel(self.Xg), torch.ravel(self.Yg), torch.ravel(self.Zg)).T #1D tensor is row vector. Stack and then transpose yields shape (samples, 3)
                self.interp_n_x_sample = self._n_homo(self.interp_x_sample).reshape(self.Xg.shape)
                grad_n_shape = list(self.Xg.shape)
                grad_n_shape.append(3) #in-place modification. Gradient has additional dimension of 3 at each sampling position
                self.interp_grad_n_x_sample = torch.reshape(self._grad_n_homo(self.interp_x_sample), tuple(grad_n_shape)) #Save as a 4D tensor


            elif self.form == 'luneburg_lens':
                #build interpolant arrays
                self.interp_x_sample = torch.vstack((torch.ravel(self.Xg), torch.ravel(self.Yg), torch.ravel(self.Zg))).T #1D tensor is row vector. Stack and then transpose yields shape (samples, 3)
                self.interp_n_x_sample = self._n_lune(self.interp_x_sample).reshape(self.Xg.shape) #Save as a 3D tensor
                grad_n_shape = list(self.Xg.shape)
                grad_n_shape.append(3) #in-place modification. Gradient has additional dimension of 3 at each sampling position
                self.interp_grad_n_x_sample = torch.reshape(self._grad_n_lune(self.interp_x_sample), tuple(grad_n_shape)) #Save as a 4D tensor

            elif self.form == 'maxwell_lens':
                #build interpolant arrays
                self.interp_x_sample = torch.vstack((torch.ravel(self.Xg), torch.ravel(self.Yg), torch.ravel(self.Zg))).T #1D tensor is row vector. Stack and then transpose yields shape (samples, 3)
                self.interp_n_x_sample = self._n_maxwell(self.interp_x_sample).reshape(self.Xg.shape)
                grad_n_shape = list(self.Xg.shape)
                grad_n_shape.append(3) #in-place modification. Gradient has additional dimension of 3 at each sampling position
                self.interp_grad_n_x_sample = torch.reshape(self._grad_n_maxwell(self.interp_x_sample), tuple(grad_n_shape)) #Save as a 4D tensor

            elif self.form == 'eaton_lens':
                #build interpolant arrays
                self.interp_x_sample = torch.vstack((torch.ravel(self.Xg), torch.ravel(self.Yg), torch.ravel(self.Zg))).T #1D tensor is row vector. Stack and then transpose yields shape (samples, 3)
                self.interp_n_x_sample = self._n_eaton(self.interp_x_sample).reshape(self.Xg.shape)
                grad_n_shape = list(self.Xg.shape)
                grad_n_shape.append(3) #in-place modification. Gradient has additional dimension of 3 at each sampling position
                self.interp_grad_n_x_sample = torch.reshape(self._grad_n_eaton(self.interp_x_sample), tuple(grad_n_shape)) #Save as a 4D tensor


            elif self.form == 'freeform': #Directly import data instead of generating.
                self.interp_x_sample = self.params.get('interp_x_sample',None) #Input data points are designated with sample subscript
                self.interp_n_x_sample = self.params.get('interp_n_x_sample',None)  
                self.interp_grad_n_x_sample = self.params.get('interp_grad_n_x_sample',None) 

                #Check inputs
                if (len(self.interp_x_sample.shape) <= 2) or (len(self.interp_n_x_sample) <= 2):
                    raise Exception('Imported "interp_x_sample" and "interp_n_x_sample" should be 3D (even in 2D the thrid dim should have size 1).')
                if (self.interp_x_sample.shape) != (self.interp_n_x_sample):
                    raise Exception('Size mismatch between "interp_x_sample" and "interp_n_x_sample".')

                if self.interp_grad_n_x_sample is None:
                    #If not provided, use finite difference to approximate the gradient
                    pass
                else:
                    #check input and store input gradient field
                    #Gradient field should have 1 more dimension (for spatial derviatives)
                    if not (len(self.interp_grad_n_x_sample.shape) == len(self.interp_n_x_sample.shape) + 1):
                        raise Exception('Index gradient field should have extactly one more dimension than index field.')

            else:
                raise Exception('Other interpolation functions are not supported yet.')                
            
            #================Setup pyTorch interpolation grid_sample inputs
            #Two requirements of grid_sample function: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample
            #(1)Volumetric data stored in a 5D grid
            #(2)Sampled position normalized to be within [-1, +1] in all dimensions of the grid

            #For index
            self.interp_n_x_5D = torch.permute(self.interp_n_x_sample, (2,1,0)) #Swapping the x and z axis because grid_sample takes input tensor in (N,Ch,Z,Y,X) order
            self.interp_n_x_5D = self.interp_n_x_5D[None, None, :,:,:] #Inser N and Ch axis
            #Note that the query points[0],[1],[2] are still arranged in x,y,z order
            self.logger.debug(f"interp_n_x_5D has shape of {self.interp_n_x_5D.shape}")
            
            self.grid_span = self.xv_yv_zv_max - self.xv_yv_zv_min
            self.position_normalization_factor = 2.0/self.grid_span #normalize physical location by 3D volume half span, such that values are between [-1, +1]
            if torch.isnan(self.position_normalization_factor[2]): #In 2D case, the z span is 0. The above line evaluate as 2.0/0
                self.position_normalization_factor[2] = 0.0

            #For index gradient
            self.interp_grad_n_x_5D = torch.permute(self.interp_grad_n_x_sample, (3, 2, 1, 0)) #shape = [Ch, Z, Y, X] where Ch is the components of the gradient in x,y,z
            self.interp_grad_n_x_5D = self.interp_grad_n_x_5D[None, :, :, :, :] #Insert N axis
            self.logger.debug(f"interp_grad_n_x_5D has shape of {self.interp_grad_n_x_5D.shape}")
            
        else:
            raise Exception('"type" should be either "analytical" or "interpolation".')

    #=================================Init functions================================================


    #=================================Analytic: Homogeneous medium================================================
    @torch.inference_mode()
    def _n_homo(self, x : torch.Tensor):
        return torch.ones_like(x) #automatically create on same device

    @torch.inference_mode()
    def _grad_n_homo(self, x : torch.Tensor):
        shape = tuple(list(x.shape).append(3))
        return torch.zeros(shape, device = x.device)

    #=================================Analytic: Luneburg lens=====================================================
    @torch.inference_mode()
    def _n_lune(self, x : torch.Tensor = None, r_known : torch.Tensor = None):
        '''
        Parameters:

        x : torch.Tensor, query position vector size(n_query_pts, 3)
        
        r_known : 
        
        To achieve the lensing effect, the refractive index has to change relative to its surroundings.
        With p = 2, the center of the lens has index sqrt(2) times that of surroundings.
        Expression of n can be found in "Path Tracing Estimators for Refractive Radiative Transfer, Pediredla et. al., 2020, page 10"
        '''
        if  (x is None) and (r_known is None):
            raise Exception('Either x or r must be provided.')
        elif r_known is None:
            r = torch.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2) #radial position, r of vector x, 1D tensor, numel = x.size[0]
        else:
            r = r_known

        n = torch.ones_like(r) #1D tensor, numel = x.size[0]

        #Inside the lens
        n[r < self.params['R_physical']] = self.params['n_sur'] * (2- (r[r < self.params['R_physical']]/self.params['R_physical'])**self.params['p'])**(1/self.params['p'])
        #Outside or at the boundary of the lens.
        if self.params['n_sur'] != 1.0: #if n_sur is just 1, the query points outside lens is already populated with 1 so no operation needed
            n[r >= self.params['R_physical']] = self.params['n_sur']
        
        return n

    @torch.inference_mode()
    def _grad_n_lune(self, x : torch.Tensor):
        r = torch.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2) #radial position, r of vector x

        n = self._n_lune(r_known = r)

        grad_n = torch.zeros_like(x)

        n_0_over_r_to_P = (self.params['n_sur']/self.params['R_physical'])**self.params['p'] #compute the constant multiplier together first for performance
        idx_r_in_lens = r < self.params['R_physical']
        #Inside the lens. The following two lines are equivalent. The middle two tensors are 1-D so a new axis has to be inserted for broadcasting. None == np.newaxis in torch and numpy.
        # grad_n[idx_r_in_lens, :] = - n_0_over_r_to_P * torch.unsqueeze((n[idx_r_in_lens]**(1 - self.params['p'])) * (r[idx_r_in_lens]**(self.params['p']-2)), dim = 1) * x[idx_r_in_lens,:]
        grad_n[idx_r_in_lens, :] = - n_0_over_r_to_P * ((n[idx_r_in_lens]**(1 - self.params['p'])) * (r[idx_r_in_lens]**(self.params['p']-2)))[:,None] * x[idx_r_in_lens,:]

        #Outside or at the boundary of the lens
        #grad_n[r >= self.params['R_physical'],:] = 0

        return grad_n
    #=================================Analytic: Maxwell lens============================================================
    @torch.inference_mode()
    def _n_maxwell(self):
        raise Exception('To be implemented.')

    @torch.inference_mode()
    def _grad_n_maxwell(self):
        raise Exception('To be implemented.')
    #=================================Analytic: Eaton lens============================================================
    @torch.inference_mode()
    def _n_eaton(self):
        raise Exception('To be implemented.')

    @torch.inference_mode()
    def _grad_n_eaton(self):
        raise Exception('To be implemented.')
    #=================================Interpolation==========================================================================
    @torch.inference_mode()
    def _n_interp(self, x : torch.Tensor):
        '''
        Obtain index value via interpolating on provided or pre-sampled data points
        '''
        x = self.normalizePosition(x) #Normalize x such that the computation domain is between [-1,+1] in all dimensions

        #self.interp_n_x_5D has already its axes permuted such that the last 3 dimensions are Z,Y,X in the class __init__
        #The query points x should still have x,y,z components in its [:,0], [:,1], [:,2] respectively

        n = torch.nn.functional.grid_sample(input = self.interp_n_x_5D, #shape = [N, Ch, D_in, H_in, W_in], where D_in, H_in, W_in are ZYX dimension respectively
                                        grid = x[None, None, None, :,:], #shape = [N,D_out, H_out, W_out,3], where N = D_out = H_out = 1, and W_out = number of samples
                                        mode='bilinear',
                                        padding_mode= 'border',
                                        align_corners = True # True: extrema refers to center of corner voxels. False: extrema refers to corner of corner voxels
                                        )

        return n[0,0,0,0,:] #n originally has shape [N, C, D_out, H_out, W_out], where W_out = number of samples

    @torch.inference_mode()
    def _grad_n_interp(self, x : torch.Tensor):
        '''
        Obtain value of index gradient via interpolating on provided or pre-sampled data points
        '''
        x = self.normalizePosition(x) #Normalize x such that the computation domain is between [-1,+1] in all dimensions

        #self.interp_grad_n_x_5D has already its axes permuted such that the last 4 dimensions are Ch,Z,Y,X in the class __init__
        #The query points x should still have x,y,z components in its [:,0], [:,1], [:,2] respectively

        grad_n = torch.nn.functional.grid_sample(input = self.interp_grad_n_x_5D, #shape = [N, Ch, D_in, H_in, W_in], where Ch is channel, and D_in, H_in, W_in are ZYX dimension respectively
                                        grid = x[None, None, None, :,:], #shape = [N,D_out, H_out, W_out,3], where N = D_out = H_out = 1, and W_out = number of samples
                                        mode='bilinear',
                                        padding_mode= 'border',
                                        align_corners = True # True: extrema refers to center of corner voxels. False: extrema refers to corner of corner voxels
                                        )
        #grad_n originally has shape [N, C, D_out, H_out, W_out], where W_out = number of samples
        return torch.squeeze(grad_n[0,:,0,0,:]).T #remove singleton dimension and transpose. Resultant shape should be [number of samples, Ch], where Ch = 3

    @torch.inference_mode()
    def normalizePosition(self, x):
        #Need to check if the meshgrid has index increasing from the negative physical location to positive physical location
        return x*self.position_normalization_factor

    #=================================Utilities==========================================================================
    def plotIndex(self, fig = None, ax = None, block=False, show = True):
        '''
        Plot a 2D slice of index. Currently only for real part. Future extenstion: for both its real and imaginary parts
        '''
        if 'interp_x_sample' not in self.__dict__:
            x_sample = torch.vstack((torch.ravel(self.Xg), torch.ravel(self.Yg), torch.ravel(self.Zg))).T #1D tensor is row vector. Stack and then transpose yields shape (samples, 3)
        else:
            x_sample = self.interp_x_sample

        n = self.n(x_sample).reshape(self.Xg.shape)
        # n_slice = self.interp_n_x_sample[:,:,self.interp_n_x_sample.shape[2]//2].cpu().numpy()
        n_slice = n[:,:,n.shape[2]//2].cpu().numpy()

        vmin, vmax = np.amin(n_slice), np.amax(n_slice)

        if ax == None:
            fig, ax =  plt.subplots()

        mappable = ax.imshow(n_slice, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title('Index distribution')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(mappable)

        if block == False:
            plt.ion()
        if show == True:
            plt.show()

        return fig, ax


    def plotGradNMag(self, fig = None, ax = None, block=False, show = True):
        '''
        Plot a 2D slice of index gradient. Currently only for real part. Future extenstion: for both its real and imaginary parts
        '''
        if 'interp_x_sample' not in self.__dict__:
            x_sample = torch.vstack((torch.ravel(self.Xg), torch.ravel(self.Yg), torch.ravel(self.Zg))).T #1D tensor is row vector. Stack and then transpose yields shape (samples, 3)
        else:
            x_sample = self.interp_x_sample

        grad_n = self.grad_n(x_sample)
        grad_n = grad_n.reshape((self.Xg.shape[0],self.Xg.shape[1],self.Xg.shape[2],3))
        grad_n_slice = grad_n[:,:,grad_n.shape[2]//2,:].cpu().numpy()

        grad_n_slice_mag = np.sqrt(grad_n_slice[:,:,0]**2 + grad_n_slice[:,:,1]**2 + grad_n_slice[:,:,2]**2)

        if ax == None:
            fig, ax =  plt.subplots(nrows = 2, ncols = 2)
        plt_mag = ax[0,0].imshow(grad_n_slice_mag, cmap='gray', vmin= np.amin(grad_n_slice_mag), vmax=np.amax(grad_n_slice_mag))
        plt_x = ax[0,1].imshow(grad_n_slice[:,:,0], cmap='bwr', vmin= np.amin(grad_n_slice[:,:,0]), vmax=np.amax(grad_n_slice[:,:,0]))
        plt_y = ax[1,0].imshow(grad_n_slice[:,:,1], cmap='bwr', vmin= np.amin(grad_n_slice[:,:,1]), vmax=np.amax(grad_n_slice[:,:,1]))  
        plt_z = ax[1,1].imshow(grad_n_slice[:,:,2], cmap='bwr', vmin= np.amin(grad_n_slice[:,:,2]), vmax=np.amax(grad_n_slice[:,:,2]))

        ax[0,0].set_title('Magnitude of index gradient')
        ax[0,0].set_xlabel('x')
        ax[0,0].set_ylabel('y')
        plt.colorbar(plt_mag)

        ax[0,1].set_title('x component of index gradient')
        ax[0,1].set_xlabel('x')
        ax[0,1].set_ylabel('y')
        plt.colorbar(plt_x)

        ax[1,0].set_title('y component of index gradient')
        ax[1,0].set_xlabel('x')
        ax[1,0].set_ylabel('y')
        plt.colorbar(plt_y)

        ax[1,1].set_title('z component of index gradient')
        ax[1,1].set_xlabel('x')
        ax[1,1].set_ylabel('y')
        plt.colorbar(plt_z)

        if block == False:
            plt.ion()
        if show == True:
            plt.show()
        
        return fig, ax

    def plotGradNArrow(self, fig = None, ax = None, lb = 0, ub = 1, n_pts=512, block=False, show = True):
        '''
        Plot a 2D slice of index gradient in sampled arrow plot. Currently only for real part. Future extenstion: for both its real and imaginary parts
        '''

        # grad_n_slice = self.interp_grad_n_x_sample[:,:,self.interp_grad_n_x_sample.shape[2]//2,:].cpu().numpy()

        # grad_n_slice_mag = np.sqrt(grad_n_slice[:,0]**2 + grad_n_slice[:,1]**2 + grad_n_slice[:,2]**2)

        # # vmin, vmax = np.amin(n_slice), np.amax(n_slice)

        # if ax == None:
        #     fig, ax =  plt.subplots()
        # # ax.quiver(self.interp_x_sample[relevant_points,0],
        # # x0[relevant_points,1],
        # # x0[relevant_points,2],
        # # v0[relevant_points,0],
        # # v0[relevant_points,1],
        # # v0[relevant_points,2],
        # # length=0.15,
        # # color = color)
        # # mappable = ax.imshow(n_slice, cmap='gray', vmin=vmin, vmax=vmax)
        # ax.set_title('Index distribution gradient')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # # plt.colorbar(mappable)

        # if block == False:
        #     plt.ion()
        # if show == True:
        #     plt.show()
        
        return fig, ax

    def plotRandomlySampledIndex(self, pts = 500, fig = None, ax = None, block=False, show = True):
        '''
        Create a scatter plot of index at random positions. Color value is proportional to index.
        '''
        random_position = torch.rand(pts,3, device = self.device)

        random_position[:,0] = (random_position[:,0]-0.5)*2*torch.amax(self.xv)
        random_position[:,1] = (random_position[:,1]-0.5)*2*torch.amax(self.yv)
        random_position[:,2] = (random_position[:,2]-0.5)*2*torch.amax(self.zv)

        self.plotIndexAtPosition(x = random_position, fig = fig, ax = ax, block = block, show = show)


    def plotIndexAtPosition(self, x, fig = None, ax = None, block=False, show = True, cmap = 'viridis', marker = 'o'):
        '''
        Create a scatter plot of index at specified positions x. Color value is proportional to index.
        '''
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x)
        
        n = self.n(x)
        n = n.cpu().numpy()
        x = x.cpu().numpy()

        if ax == None:
            # fig, ax =  plt.subplots()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        mappable = ax.scatter(x[:,0], x[:,1], x[:,2], c = n, cmap = cmap, marker = marker)
        ax.set_title('Index distribution')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.colorbar(mappable)

        if block == False:
            plt.ion()
        if show == True:
            plt.show()



## Test

if __name__ == '__main__':
    input()
