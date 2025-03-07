import torch
import numpy as np
import matplotlib.pyplot as plt
import vamtoolbox as vam

'''Old implementation with flat portion truncated such that data fits within [-r_o,r_o] in y-axis'''
# class WrapTransformation():

#     @torch.inference_mode()
#     def __init__(self,r_i,tau,spatial_sampling_rate,angles_per_circumference,N_z):


#         self.device = torch.device('cuda')


#         self.r_i = r_i # inside radius of wrapped domain
#         self.tau = tau # thickness of domain
#         self.r_o = self.r_i + tau # outside radius of wrapped domain
#         self.r_mid = (self.r_i + self.r_o)/2 # middle radius of wrapped domain
#         self.y_arc = self.r_o # length of flat section in wrapped domain (arc length)
#         self.y_arc_radians = self.y_arc/self.r_mid # length of flat section in wrapped domain (radians)
#         self.spatial_sampling_rate = spatial_sampling_rate # 
#         self.angles_per_circumference = angles_per_circumference # 

#         self.quad_I_radians = [0,self.y_arc_radians]
#         self.quad_II_III_radians = [self.y_arc_radians,self.y_arc_radians+torch.pi]
#         self.quad_IV_radians = [self.y_arc_radians+torch.pi, 2*self.y_arc_radians+torch.pi]
#         self.quad_I = [0,self.y_arc]
#         self.quad_II_III = [self.y_arc,self.y_arc+torch.pi*self.r_mid]
#         self.quad_IV = [self.y_arc+torch.pi*self.r_mid, 2*self.y_arc+torch.pi*self.r_mid]

#         # below are numbers of voxels in each dimension of the wrapped and unwrapped domain kernels
#         # the kernel is defined as a domain which has boundaries such that 1 circumference of the geometry is sampled
#         # this is relevant when the length of the target geometry is longer than 1 circumference of the R2R configuration
#         self.N_x_w, self.N_y_w = int(2*self.r_o*self.spatial_sampling_rate), int(2*self.r_o*self.spatial_sampling_rate)
#         self.N_l_uw, self.N_rho_uw = int(angles_per_circumference), int(self.spatial_sampling_rate*(self.r_o-self.r_i))
#         self.N_z = N_z

#         # coordinate vectors
#         self.x_v_w = torch.linspace(-self.r_o,self.r_o,self.N_x_w,device=self.device) # units of length
#         self.y_v_w = torch.linspace(-self.r_o,self.r_o,self.N_y_w,device=self.device) # units of length
#         self.l_v_uw = torch.linspace(0,2*self.y_arc+self.r_mid*torch.pi,self.N_l_uw,device=self.device) # units of arc length
#         self.rho_v_uw = torch.linspace(self.r_i,self.r_o,self.N_rho_uw,device=self.device) # units of length
#         self.z_v = torch.linspace(-(self.N_z-1)/(2*self.spatial_sampling_rate), (self.N_z-1)/(2*self.spatial_sampling_rate), self.N_z,device=self.device) # units of length

#         # below are the coordinate grids defining the wrapped and unwrapped domain kernels
#         self.x_g_w, self.y_g_w, self.z_g_w = self._xygridWrappedDomain()
#         self.x_g_uw, self.y_g_uw, self.z_g_uw = self._xygridUnwrappedDomain(normalized=True)
#         self.rho_g_w, self.l_g_w = self._rholgridWrappedDomain()
#         self.rho_g_uw, self.l_g_uw, _ = self._rholgridUnwrappedDomain()

#     @torch.inference_mode()
#     def _xygridWrappedDomain(self,normalized=False):

#         x_g, y_g, z_g = torch.meshgrid(
#             self.x_v_w,
#             self.y_v_w,
#             self.z_v,
#             indexing='ij',
#         )


#         if normalized:
#             x_g = x_g/self.r_o
#             y_g = y_g/self.r_o
#             z_g = z_g/torch.amax(z_g)

#         # fig, axs = plt.subplots(2,1)
#         # axs[0].imshow(x_g.cpu().numpy(),cmap='viridis')
#         # axs[1].imshow(y_g.cpu().numpy(),cmap='viridis')
#         # fig.savefig('xywrapped.jpg',dpi=300)
        
#         return x_g, y_g, z_g

#     @torch.inference_mode()
#     def _xygridUnwrappedDomain(self,normalized=False):

#         """
#         II      | I
#                 |
#         --------|--------
#                 |
#         III     | IV
#         """
        
#         rho_g, l_g, z_g = self._rholgridUnwrappedDomain()
        
#         l_g_radians = l_g/self.r_mid

#         x_g = torch.zeros_like(rho_g)
#         y_g = torch.zeros_like(rho_g)

#         #quadrant I
#         mask_I = torch.logical_and((l_g>=self.quad_I[0]),(l_g<self.quad_I[1]))
#         x_g[mask_I] = -rho_g[mask_I]
#         y_g[mask_I] = self.r_mid*(self.quad_I_radians[1]-l_g_radians[mask_I])

#         #quadrant IV III
#         mask_IV_III = torch.logical_and((l_g>=self.quad_II_III[0]),(l_g<self.quad_II_III[1]))
#         x_g[mask_IV_III] = -rho_g[mask_IV_III]*torch.sin(l_g_radians[mask_IV_III]+(torch.pi/2-self.y_arc_radians))
#         y_g[mask_IV_III] = rho_g[mask_IV_III]*torch.cos(l_g_radians[mask_IV_III]+(torch.pi/2-self.y_arc_radians))

#         #quadrant II
#         mask_II = torch.logical_and((l_g>=self.quad_IV[0]),(l_g<=self.quad_IV[1]))
#         x_g[mask_II] = rho_g[mask_II]
#         y_g[mask_II] = self.r_mid*(l_g_radians[mask_II]-self.quad_IV_radians[0])


#         if normalized:
#             x_g = x_g/self.r_o
#             y_g = y_g/self.r_o
#             if torch.amax(z_g) != 0:
#                 z_g = z_g/torch.amax(z_g)
                
#         # fig, axs = plt.subplots(2,1)
#         # axs[0].imshow(x_g[:,:,0].cpu().numpy(),cmap='viridis')
#         # axs[1].imshow(y_g[:,:,0].cpu().numpy(),cmap='viridis')
#         # plt.show()
#         # fig.savefig('xyunwrapped.jpg',dpi=300)
            

#         return x_g, y_g, z_g
    

#     @torch.inference_mode()
#     def _rholgridUnwrappedDomain(self,):

#         rho_g, l_g, z_g = torch.meshgrid(
#             self.rho_v_uw,
#             self.l_v_uw,
#             self.z_v,
#             indexing='ij',
#         )

#         return rho_g, l_g, z_g
   
#     @torch.inference_mode()
#     def _rholgridWrappedDomain(self,):
#         """
#         II      | I
#                 |
#         --------|--------
#                 |
#         III     | IV
#         """

#         x_g, y_g, _ = self._xygridWrappedDomain()

#         rho_g = torch.sqrt(x_g**2 + y_g**2)

#         mask_I_II = y_g>=0
#         rho_g[mask_I_II] = torch.abs(x_g[mask_I_II])

        
#         mask_I = torch.logical_and((y_g>=0),(x_g>=0))
#         mask_II = torch.logical_and((y_g>=0),(x_g<=0))
#         mask_III = torch.logical_and((y_g<=0),(x_g<=0))
#         mask_IV = torch.logical_and((y_g<=0),(x_g>=0))


#         mask_III_IV = y_g>0
#         l_g = torch.arctan2(y_g,x_g)*self.r_mid

#         l_g[mask_I] = y_g[mask_I]
#         l_g[mask_II] = -y_g[mask_II] - torch.pi*self.r_mid

#         l_g += self.quad_IV[0]

#         # fig, axs = plt.subplots(2,1)
#         # axs[0].imshow(rho_g.cpu().numpy(),cmap='viridis')
#         # axs[1].imshow(l_g.cpu().numpy(),cmap='viridis')
#         # fig.savefig('rholwrapped.jpg',dpi=300)

#         return rho_g, l_g
    
#     @torch.inference_mode()
#     def wrappedToUnwrapped(self,x:torch.Tensor):
        
#         # if x.shape[2] < batch_size:
#         #     batch_size = x.shape[2]
        
#         samples_z_g_uw = torch.linspace(-1,1,x.shape[2],device=self.device)
#         samples_z_g_uw = samples_z_g_uw.broadcast_to((self.x_g_uw.shape[0],self.x_g_uw.shape[1],x.shape[2]))
#         # samples_z_g_uw = self.z_g_uw
#         samples_x_g_uw = self.x_g_uw.broadcast_to((self.x_g_uw.shape[0],self.x_g_uw.shape[1],x.shape[2]))
#         samples_y_g_uw = self.y_g_uw.broadcast_to((self.x_g_uw.shape[0],self.x_g_uw.shape[1],x.shape[2]))
        
        


#         # grid to sample onto to
#         sample_grid = torch.stack((samples_x_g_uw,samples_y_g_uw,samples_z_g_uw),3)[None,:,:,:,:] # shaping grid to be (N,D,H,W,3) = (1,1,1,n_samples,2)
        
#         # preallocate output so that batches can be inserted sequentially
#         # output = torch.zeros((1,1,samples_x_g_uw.shape[0],samples_x_g_uw.shape[1],samples_x_g_uw.shape[2]),device=self.device) # output has shape (1,1,1,1,n_samples)
#         # n_batches = int(np.floor(x.shape[2]/batch_size))

#         # reshape input for grid_sample
#         x_4D = x.permute((2,1,0))[None,None,:,:,:] # reshape from (nX,nY,nZ) to shape (N,C,D,H,W) = (1,1,nZ,nY,nX)
#         output = torch.nn.functional.grid_sample(x_4D,sample_grid,mode='bilinear',padding_mode='zeros',align_corners=True)[0,0,:,:,:]

#         # for i in range(n_batches):
#         #     _i_start = i*batch_size
#         #     _i_end = (i+1)*batch_size
            
#         #     output[:,:,:,:,_i_start:_i_end] = torch.nn.functional.grid_sample(x_4D[:,:,:,:,_i_start:_i_end],sample_grid,mode='bilinear',padding_mode='zeros',align_corners=True)


#         return output
#         # if x.shape[2] < batch_size:
#         #     batch_size = x.shape[2]
#         # samples_x_g_uw = self.x_g_uw.ravel()
#         # samples_y_g_uw = self.y_g_uw.ravel()

#         # # grid to sample onto to
#         # sample_grid = torch.stack((samples_x_g_uw,samples_y_g_uw),1)[None,None,:,:] # shaping grid to be (N,H,W,2) = (1,1,n_samples,2)
#         # sample_grid = sample_grid.broadcast_to((batch_size,1,samples_x_g_uw.size(dim=0),2)) # broadcast to shape (batch_size,1,n_samples,2)

        
#         # # preallocate output so that batches can be inserted sequentially
#         # output = torch.zeros((x.shape[2],1,1,samples_x_g_uw.size(dim=0)),device=self.device) # output has shape (nZ,1,1,n_samples)
#         # n_batches = int(np.floor(x.shape[2]/batch_size))

#         # # reshape input for grid_sample
#         # x_4D = x.permute((2,1,0))[:,None,:,:] # reshape from (nX,nY,nZ) to shape (N,C,H,W) = (nZ,1,nY,nX)
#         # # x_4D = x.reshape((x.shape[2],x.shape[1],x.shape[0]))[:,None,:,:] # reshape from (nX,nY,nZ) to shape (N,C,H,W) = (nZ,1,nY,nX)

#         # for i in range(n_batches):
#         #     _i_start = i*batch_size
#         #     _i_end = (i+1)*batch_size
            
#         #     output[_i_start:_i_end,:,:,:] = torch.nn.functional.grid_sample(x_4D[_i_start:_i_end,:,:,:],sample_grid,mode='bilinear',padding_mode='zeros',align_corners=True)

#         # # reshape output from (nZ,1,1,n_samples) to (nrho,nl,nZ)
#         # # output = output.squeeze().reshape((x.shape[2],self.x_g_uw.shape[1],self.x_g_uw.shape[0])).permute((2,1,0))
#         # output = output.squeeze().reshape((self.x_g_uw.shape[0],self.x_g_uw.shape[1],x.shape[2]))
#         # # output = output.squeeze().reshape((x.shape[2],self.x_g_uw.shape[1],self.x_g_uw.shape[0]))
#         # return output

#     @torch.inference_mode()
#     def unwrappedToWrapped(self,x):

#         # grid to sample onto to
#         sample_grid = torch.stack((normalized_samples_theta_g_w,normalized_samples_r_g_w[0:batch_size,:,:]),3)
        
#         # sample_grid = sample_grid.unsqueeze(0)

#         # print(sample_grid.shape)

#         output = torch.nn.functional.grid_sample(target_tensor_input,sample_grid,mode='bilinear',padding_mode='zeros',align_corners=True)

#     @torch.inference_mode()
#     def attenuationGrid(self,alpha):
        
#         # neutral axis
#         strain = -(self.rho_g_w - self.r_mid)/self.rho_g_w

#         attenuation = torch.zeros_like(self.rho_g_w,device=self.device)
#         attenuation[self.y_g_w>0] = alpha
#         attenuation[self.y_g_w<=0] = alpha*(1+strain[self.y_g_w<=0])

#         attenuation[self.rho_g_w<self.r_i] = 0
#         attenuation[self.rho_g_w>self.r_o] = 0

#         # fig, axs = plt.subplots(2,1)
#         # axs[0].imshow(strain[:,:,0].cpu().numpy(),cmap='viridis')
#         # axs[1].imshow(attenuation[:,:,0].cpu().numpy(),cmap='viridis')
#         # # fig.savefig('rholwrapped.jpg',dpi=300)
#         # plt.show()
#         return attenuation


class SpiralTransformation():
    @torch.inference_mode()
    def __init__(self,tau,r_i,r_o,spatial_sampling_rate_rho,spatial_sampling_rate_l,N_x,N_y,N_z):
        self.device = torch.device('cuda')


        self.r_i = r_i
        self.r_o = r_o
        self.tau = tau # thickness of domain
        self.b = self.tau/(2*torch.pi)
        self.number_turns = (self.r_o - self.r_i)/self.tau + 1
        self.start_angle = 0
        self.end_angle = self.number_turns*2*torch.pi + self.start_angle
        self.spatial_sampling_rate_rho = spatial_sampling_rate_rho # 
        self.spatial_sampling_rate_l = spatial_sampling_rate_l


        # below are numbers of voxels in each dimension of the wrapped and unwrapped domain kernels
        # the kernel is defined as a domain which has boundaries such that 1 circumference of the geometry is sampled
        # this is relevant when the length of the target geometry is longer than 1 circumference of the R2R configuration
        theta_v = torch.linspace(self.start_angle,self.end_angle,1000,device=self.device)
        f = torch.sqrt((self.r_i+self.b*theta_v)**2+(self.b)**2)
        self.l = torch.trapz(f,theta_v)
        self.rho = self.tau

        # self.N_x_s, self.N_y_s = int(2*self.r_o*self.spatial_sampling_rate_rho), int(2*self.r_o*self.spatial_sampling_rate_rho)
        self.N_x_s, self.N_y_s = N_x, N_y
        self.N_l_f, self.N_rho_f = int(self.l*self.spatial_sampling_rate_l), int(self.rho*self.spatial_sampling_rate_rho)
        self.N_z = N_z

        # coordinate vectors
        self.x_v_s = torch.linspace(-self.r_o,self.r_o,self.N_x_s,device=self.device) # units of length
        self.y_v_s = torch.linspace(-self.r_o,self.r_o,self.N_y_s,device=self.device) # units of length
        self.theta_v_f = torch.linspace(self.start_angle,self.end_angle,self.N_l_f,device=self.device) # units of arc length
        self.l_v_f = torch.linspace(0,self.l,self.N_l_f,device=self.device) # units of arc length
        self.rho_v_f = torch.linspace(-self.tau,0,self.N_rho_f,device=self.device) # units of length
        self.z_v = torch.linspace(-(self.N_z-1)/(2*self.spatial_sampling_rate_rho), (self.N_z-1)/(2*self.spatial_sampling_rate_rho), self.N_z,device=self.device) # units of length

        self.normalization_parameters = torch.tensor([1/self.r_o, 1/self.r_o, 1/self.z_v.max()])
        if torch.isinf(self.normalization_parameters[2]): #In 2D case, the z span is 0. The above line evaluate as 2.0/0 which yields inf
            self.normalization_parameters[2] = 0.0 #If inf, clamp the z position down back to 0
    

        self.x_g_f, self.y_g_f = self._xygridFlatDomain(normalized=True)
        # self.rho_g_w, self.l_g_w = self._rholgridWrappedDomain()
    

    @torch.inference_mode()
    def _xygridSprialDomain(self,normalized=False):
        x_g, y_g = torch.meshgrid(
            self.x_v_s,
            self.y_v_s,
            indexing='ij',
        )

        if normalized:
            x_g = x_g*self.normalization_parameters[0]
            y_g = y_g*self.normalization_parameters[1]

        return x_g[:,:,None], y_g[:,:,None]
    
    @torch.inference_mode()
    def _rholgridFlatDomain(self,):

        rho_g, l_g = torch.meshgrid(
            self.rho_v_f,
            self.l_v_f,
            indexing='ij',
        )

        return rho_g[:,:,None], l_g[:,:,None]
    
    @torch.inference_mode()
    def _rhothetagridFlatDomain(self,):

        rho_g, theta_g = torch.meshgrid(
            self.rho_v_f,
            self.theta_v_f,
            indexing='ij',
        )

        rho_g = rho_g + self.r_i + self.b*theta_g # rho_g = [0,tau]

        return rho_g[:,:,None], theta_g[:,:,None]
    
    @torch.inference_mode()
    def _xygridFlatDomain(self,normalized=False):

        rho_g, theta_g = self._rhothetagridFlatDomain()

        x_g = rho_g*torch.cos(theta_g)
        y_g = rho_g*torch.sin(theta_g)

        # fig, axs = plt.subplots(2,1)
        # axs[0].imshow(x_g[:,:,0].cpu().numpy(),cmap='viridis')
        # axs[1].imshow(y_g[:,:,0].cpu().numpy(),cmap='viridis')
        # plt.show()

        if normalized:
            x_g = x_g*self.normalization_parameters[0]
            y_g = y_g*self.normalization_parameters[1]

        return x_g, y_g
    
    @torch.inference_mode()
    def spiralToFlat(self,x:torch.Tensor):

        if not torch.is_tensor(x):
            x = torch.from_numpy(x).to(device=self.device)


        samples_z_g_f = torch.linspace(-1,1,x.shape[2],device=self.device)
        samples_z_g_f = samples_z_g_f.broadcast_to((self.x_g_f.shape[0],self.x_g_f.shape[1],x.shape[2]))
        samples_x_g_f = self.x_g_f.broadcast_to((self.x_g_f.shape[0],self.x_g_f.shape[1],x.shape[2]))
        samples_y_g_f = self.y_g_f.broadcast_to((self.x_g_f.shape[0],self.x_g_f.shape[1],x.shape[2]))
        
        
        # grid to sample onto to
        sample_grid = torch.stack((samples_x_g_f,samples_y_g_f,samples_z_g_f),3)[None,:,:,:,:] # shaping grid to be (N,D,H,W,3) = (1,1,1,n_samples,2)
        
        # preallocate output so that batches can be inserted sequentially
        # output = torch.zeros((1,1,samples_x_g_f.shape[0],samples_x_g_f.shape[1],samples_x_g_f.shape[2]),device=self.device) # output has shape (1,1,1,1,n_samples)
        # n_batches = int(np.floor(x.shape[2]/batch_size))

        # reshape input for grid_sample
        x_4D = x.permute((2,1,0))[None,None,:,:,:] # reshape from (nX,nY,nZ) to shape (N,C,D,H,W) = (1,1,nZ,nY,nX)
        # output = torch.nn.functional.grid_sample(x_4D,sample_grid,mode='bilinear',padding_mode='zeros',align_corners=True)[0,0,:,:,:]
        output = torch.nn.functional.grid_sample(x_4D,sample_grid,mode='bilinear',padding_mode='border',align_corners=True)[0,0,:,:,:]

        return output
    
    @torch.inference_mode()
    def ArchimedesSpiral(self,):

        def checkerboard(nX,nY,sq):
            coords = np.ogrid[0:nX, 0:nY]
            idx = (coords[0] // sq + coords[1] // sq) % 2
            vals = np.array([0.0, 1.0],dtype=np.float32)
            img = vals[idx]
            return torch.from_numpy(img[:,:,None]).to(device=self.device)


        x_g, y_g = self._xygridSprialDomain()

        r_g = torch.sqrt(x_g**2 + y_g**2)
        theta_g = torch.atan2(y_g,x_g)

        rho_g, theta_g = torch.meshgrid(
            self.rho_v_f,
            self.theta_v_f,
            indexing='ij',
        )

        rho_g = rho_g + self.r_i + self.b*theta_g # rho_g = [0,tau]

        x_arch_spiral = rho_g*torch.cos(theta_g)
        y_arch_spiral = rho_g*torch.sin(theta_g)
        
        x_arch_spiral = x_arch_spiral[x_arch_spiral.shape[0]//2,:]
        y_arch_spiral = y_arch_spiral[y_arch_spiral.shape[0]//2,:]
        # x = torch.from_numpy(array).to(device=self.device)


        # x = torch.zeros((self.N_x_s,self.N_y_s,1),device=self.device)
        # for i in range(np.ceil(self.number_turns).astype(int)):
        #     x[r_g>self.r_i+i*self.tau] = i + 1
        x = checkerboard(self.N_x_s,self.N_y_s,50)

        x[r_g<self.r_i] = 0

        x[r_g>self.r_o] = 0

        x = x.expand((self.N_x_s,self.N_y_s,self.N_z))
        # r_g = r_g.expand((self.N_x_s,self.N_y_s,self.N_z))

        return x, x_arch_spiral.cpu().numpy(), y_arch_spiral.cpu().numpy()


'''New implementation with flat portion complete pi/2*r_mid arc length such that data fits within [-y_arc,y_arc] in y-axis'''
class WrapTransformation():

    @torch.inference_mode()
    def __init__(self,r_i,tau,spatial_sampling_rate_rho,spatial_sampling_rate_l,N_z):


        self.device = torch.device('cuda')


        self.r_i = r_i # inside radius of wrapped domain
        self.tau = tau # thickness of domain
        self.r_o = self.r_i + tau # outside radius of wrapped domain
        self.r_mid = (self.r_i + self.r_o)/2 # middle radius of wrapped domain
        self.y_arc = self.r_i*torch.pi/2 # length of flat section in wrapped domain (arc length)
        # self.y_arc = self.r_o*torch.pi/2 # length of flat section in wrapped domain (arc length)
        self.y_arc_radians = torch.pi/2 # length of flat section in wrapped domain (radians)
        self.spatial_sampling_rate_rho = spatial_sampling_rate_rho # 
        self.spatial_sampling_rate_l = spatial_sampling_rate_l


        """
        IV      | I
                |
        --------|--------
                |
        III     | II
        """
        

        self.quad_I_radians = [0,self.y_arc_radians]
        self.quad_II_III_radians = [self.y_arc_radians,3*self.y_arc_radians]
        self.quad_IV_radians = [3*self.y_arc_radians, 4*self.y_arc_radians]
        self.quad_I = [0,self.y_arc]
        self.quad_II_III = [self.y_arc,3*self.y_arc]
        self.quad_IV = [3*self.y_arc, 4*self.y_arc]

        # below are numbers of voxels in each dimension of the wrapped and unwrapped domain kernels
        # the kernel is defined as a domain which has boundaries such that 1 circumference of the geometry is sampled
        # this is relevant when the length of the target geometry is longer than 1 circumference of the R2R configuration
        self.l = 4*self.y_arc
        self.rho = self.tau

        self.N_x_w, self.N_y_w = int(2*self.r_o*self.spatial_sampling_rate_rho), int(2*self.y_arc*self.spatial_sampling_rate_rho)
        self.N_l_uw, self.N_rho_uw = int(self.l*self.spatial_sampling_rate_l), int(self.rho*self.spatial_sampling_rate_rho)
        self.N_z = N_z

        # coordinate vectors
        self.x_v_w = torch.linspace(-self.r_o,self.r_o,self.N_x_w,device=self.device) # units of length
        self.y_v_w = torch.linspace(-self.y_arc,self.y_arc,self.N_y_w,device=self.device) # units of length
        self.l_v_uw = torch.linspace(0,self.l,self.N_l_uw,device=self.device) # units of arc length
        self.rho_v_uw = torch.linspace(self.r_i,self.r_o,self.N_rho_uw,device=self.device) # units of length
        self.z_v = torch.linspace(-(self.N_z-1)/(2*self.spatial_sampling_rate_rho), (self.N_z-1)/(2*self.spatial_sampling_rate_rho), self.N_z,device=self.device) # units of length

        self.normalization_parameters = torch.tensor([1/self.r_o, 1/self.y_arc, 1/self.z_v.max()])
        if torch.isinf(self.normalization_parameters[2]): #In 2D case, the z span is 0. The above line evaluate as 2.0/0 which yields inf
            self.normalization_parameters[2] = 0.0 #If inf, clamp the z position down back to 0

        # below are the coordinate grids defining the wrapped and unwrapped domain kernels
        # self.x_g_w, self.y_g_w, self.z_g_w = self._xygridWrappedDomain()
        self.x_g_uw, self.y_g_uw = self._xygridUnwrappedDomain(normalized=True)
        self.rho_g_w, self.l_g_w = self._rholgridWrappedDomain()
        # self.rho_g_uw, self.l_g_uw, _ = self._rholgridUnwrappedDomain()

    @torch.inference_mode()
    def _xygridWrappedDomain(self,normalized=False):

        x_g, y_g = torch.meshgrid(
            self.x_v_w,
            self.y_v_w,
            indexing='ij',
        )


        if normalized:
            x_g = x_g*self.normalization_parameters[0]
            y_g = y_g*self.normalization_parameters[1]

        # fig, axs = plt.subplots(2,1)
        # axs[0].imshow(x_g[:,:,0].cpu().numpy(),cmap='viridis')
        # axs[1].imshow(y_g[:,:,0].cpu().numpy(),cmap='viridis')
        # plt.show()

        # fig.savefig('xywrapped.jpg',dpi=300)
        
        return x_g[:,:,None], y_g[:,:,None]


    @torch.inference_mode()
    def _rholgridUnwrappedDomain(self,):

        rho_g, l_g = torch.meshgrid(
            self.rho_v_uw,
            self.l_v_uw,
            indexing='ij',
        )

        return rho_g[:,:,None], l_g[:,:,None]

    @torch.inference_mode()
    def _xygridUnwrappedDomain(self,normalized=False):

        """
        IV      | I
                |
        --------|--------
                |
        III     | II
        """
        
        rho_g, l_g = self._rholgridUnwrappedDomain()
        
        l_g_radians = l_g/self.r_i

        x_g = torch.zeros_like(rho_g)
        y_g = torch.zeros_like(rho_g)

        #quadrant I
        mask_I = torch.logical_and((l_g>=self.quad_I[0]),(l_g<self.quad_I[1]))
        x_g[mask_I] = rho_g[mask_I]
        y_g[mask_I] = self.r_i*(self.quad_I_radians[1]-l_g_radians[mask_I])

        #quadrant II III
        mask_II_III = torch.logical_and((l_g>=self.quad_II_III[0]),(l_g<self.quad_II_III[1]))
        x_g[mask_II_III] = rho_g[mask_II_III]*torch.sin(l_g_radians[mask_II_III]+(torch.pi/2-self.y_arc_radians))
        y_g[mask_II_III] = rho_g[mask_II_III]*torch.cos(l_g_radians[mask_II_III]+(torch.pi/2-self.y_arc_radians))

        #quadrant II
        mask_IV = torch.logical_and((l_g>=self.quad_IV[0]),(l_g<=self.quad_IV[1]))
        x_g[mask_IV] = -rho_g[mask_IV]
        y_g[mask_IV] = self.r_i*(l_g_radians[mask_IV]-self.quad_IV_radians[0])

        # fig, axs = plt.subplots(2,1)
        # axs[0].imshow(x_g[:,:,0].cpu().numpy(),cmap='viridis')
        # axs[1].imshow(y_g[:,:,0].cpu().numpy(),cmap='viridis')
        # plt.show()
        # fig.savefig('xyunwrapped.jpg',dpi=300)

        if normalized:
            x_g = x_g*self.normalization_parameters[0]
            y_g = y_g*self.normalization_parameters[1]
                
            

        return x_g, y_g
    

   
    @torch.inference_mode()
    def _rholgridWrappedDomain(self,):

        """
        IV      | I
                |
        --------|--------
                |
        III     | II
        """
        

        x_g, y_g = self._xygridWrappedDomain()

        rho_g = torch.sqrt(x_g**2 + y_g**2)

        # mask_I_II = y_g>=0
        rho_g[y_g>=0] = torch.abs(x_g[y_g>=0])


        l_g = torch.arctan2(y_g,x_g)*self.r_i

        l_g[(y_g>=0)&(x_g>=0)] = y_g[(y_g>=0)&(x_g>=0)]
        l_g[(y_g>=0)&(x_g<=0)] = -y_g[(y_g>=0)&(x_g<=0)] - torch.pi*self.r_i

        l_g += self.quad_IV[0]


        # fig, axs = plt.subplots(1,1)
        # l_g[rho_g<self.r_i] = torch.nan
        # l_g[rho_g>self.r_o] = torch.nan
        # # axs[0].imshow(rho_g[:,:,0].cpu().numpy(),cmap='viridis')
        # # axs[1].imshow(l_g[:,:,0].cpu().numpy(),cmap='viridis')
        # extent=[0,self.N_y_w/self.spatial_sampling_rate_rho,self.N_x_w/self.spatial_sampling_rate_rho,0]
        # im=axs.imshow(l_g[:,:,0].cpu().numpy(),cmap='viridis',extent=extent)
        # axs.set_xlabel('Y [cm]')
        # axs.set_ylabel('X [cm]')
        # cbar = vam.util.matplotlib.addColorbar(im)
        # cbar.ax.set_ylabel(r'L [cm] or $\theta r_{mid}$ [cm]')
        # # fig.savefig(r'test_output\rholwrapped.jpg',dpi=600)
        # plt.show()

        return rho_g, l_g
    
    @torch.inference_mode()
    def wrappedToUnwrapped(self,x:torch.Tensor):
        
        # if x.shape[2] < batch_size:
        #     batch_size = x.shape[2]
        
        samples_z_g_uw = torch.linspace(-1,1,x.shape[2],device=self.device)
        samples_z_g_uw = samples_z_g_uw.broadcast_to((self.x_g_uw.shape[0],self.x_g_uw.shape[1],x.shape[2]))
        samples_x_g_uw = self.x_g_uw.broadcast_to((self.x_g_uw.shape[0],self.x_g_uw.shape[1],x.shape[2]))
        samples_y_g_uw = self.y_g_uw.broadcast_to((self.x_g_uw.shape[0],self.x_g_uw.shape[1],x.shape[2]))
        
        


        # grid to sample onto to
        sample_grid = torch.stack((samples_x_g_uw,samples_y_g_uw,samples_z_g_uw),3)[None,:,:,:,:] # shaping grid to be (N,D,H,W,3) = (1,1,1,n_samples,2)
        
        # preallocate output so that batches can be inserted sequentially
        # output = torch.zeros((1,1,samples_x_g_uw.shape[0],samples_x_g_uw.shape[1],samples_x_g_uw.shape[2]),device=self.device) # output has shape (1,1,1,1,n_samples)
        # n_batches = int(np.floor(x.shape[2]/batch_size))

        # reshape input for grid_sample
        x_4D = x.permute((2,1,0))[None,None,:,:,:] # reshape from (nX,nY,nZ) to shape (N,C,D,H,W) = (1,1,nZ,nY,nX)
        # output = torch.nn.functional.grid_sample(x_4D,sample_grid,mode='bilinear',padding_mode='zeros',align_corners=True)[0,0,:,:,:]
        output = torch.nn.functional.grid_sample(x_4D,sample_grid,mode='bilinear',padding_mode='border',align_corners=True)[0,0,:,:,:]

        # for i in range(n_batches):
        #     _i_start = i*batch_size
        #     _i_end = (i+1)*batch_size
            
        #     output[:,:,:,:,_i_start:_i_end] = torch.nn.functional.grid_sample(x_4D[:,:,:,:,_i_start:_i_end],sample_grid,mode='bilinear',padding_mode='zeros',align_corners=True)


        return output
        # if x.shape[2] < batch_size:
        #     batch_size = x.shape[2]
        # samples_x_g_uw = self.x_g_uw.ravel()
        # samples_y_g_uw = self.y_g_uw.ravel()

        # # grid to sample onto to
        # sample_grid = torch.stack((samples_x_g_uw,samples_y_g_uw),1)[None,None,:,:] # shaping grid to be (N,H,W,2) = (1,1,n_samples,2)
        # sample_grid = sample_grid.broadcast_to((batch_size,1,samples_x_g_uw.size(dim=0),2)) # broadcast to shape (batch_size,1,n_samples,2)

        
        # # preallocate output so that batches can be inserted sequentially
        # output = torch.zeros((x.shape[2],1,1,samples_x_g_uw.size(dim=0)),device=self.device) # output has shape (nZ,1,1,n_samples)
        # n_batches = int(np.floor(x.shape[2]/batch_size))

        # # reshape input for grid_sample
        # x_4D = x.permute((2,1,0))[:,None,:,:] # reshape from (nX,nY,nZ) to shape (N,C,H,W) = (nZ,1,nY,nX)
        # # x_4D = x.reshape((x.shape[2],x.shape[1],x.shape[0]))[:,None,:,:] # reshape from (nX,nY,nZ) to shape (N,C,H,W) = (nZ,1,nY,nX)

        # for i in range(n_batches):
        #     _i_start = i*batch_size
        #     _i_end = (i+1)*batch_size
            
        #     output[_i_start:_i_end,:,:,:] = torch.nn.functional.grid_sample(x_4D[_i_start:_i_end,:,:,:],sample_grid,mode='bilinear',padding_mode='zeros',align_corners=True)

        # # reshape output from (nZ,1,1,n_samples) to (nrho,nl,nZ)
        # # output = output.squeeze().reshape((x.shape[2],self.x_g_uw.shape[1],self.x_g_uw.shape[0])).permute((2,1,0))
        # output = output.squeeze().reshape((self.x_g_uw.shape[0],self.x_g_uw.shape[1],x.shape[2]))
        # # output = output.squeeze().reshape((x.shape[2],self.x_g_uw.shape[1],self.x_g_uw.shape[0]))
        # return output

    @torch.inference_mode()
    def unwrappedToWrapped(self,x):

        # grid to sample onto to
        sample_grid = torch.stack((normalized_samples_theta_g_w,normalized_samples_r_g_w[0:batch_size,:,:]),3)
        
        # sample_grid = sample_grid.unsqueeze(0)

        # print(sample_grid.shape)

        output = torch.nn.functional.grid_sample(target_tensor_input,sample_grid,mode='bilinear',padding_mode='zeros',align_corners=True)

    # @torch.inference_mode()
    # def attenuationGrid(self,alpha):
        
    #     # neutral axis
    #     strain = -(self.rho_g_w - self.r_mid)/self.rho_g_w

    #     attenuation = torch.zeros_like(self.rho_g_w,device=self.device)
    #     attenuation[self.y_g_w>0] = alpha
    #     attenuation[self.y_g_w<=0] = alpha*(1+strain[self.y_g_w<=0])

    #     attenuation[self.rho_g_w<self.r_i] = 0
    #     attenuation[self.rho_g_w>self.r_o] = 0

    #     # fig, axs = plt.subplots(2,1)
    #     # axs[0].imshow(strain[:,:,0].cpu().numpy(),cmap='viridis')
    #     # axs[1].imshow(attenuation[:,:,0].cpu().numpy(),cmap='viridis')
    #     # # fig.savefig('rholwrapped.jpg',dpi=300)
    #     # plt.show()
    #     return attenuation


if __name__ == "__main__":
    
    gpu = torch.device('cuda')


    tau = 0.3 # cm
    r_i = 0.75 # cm
    r_o = r_i + 5*tau

    deg_inc = 0.5
    spatial_sampling_rate_rho = 1/(0.00108*1.33)
    spatial_sampling_rate_l = 1/(deg_inc*np.pi/180*r_i)
    anisotropy_factor = spatial_sampling_rate_rho/spatial_sampling_rate_l

    N_x = 1000
    N_y = 1000
    N_z = 3
    def checkerboard(nX,nY,sq):
        coords = np.ogrid[0:nX, 0:nY]
        idx = (coords[0] // sq + coords[1] // sq) % 2
        vals = np.array([0.0, 1.0])
        img = vals[idx]
        return img[:,:,None]

    transform = SpiralTransformation(tau,r_i,r_o,spatial_sampling_rate_rho=spatial_sampling_rate_rho,spatial_sampling_rate_l=spatial_sampling_rate_l,N_x=N_x,N_y=N_y,N_z=N_z)
    target_geo = vam.geometry.TargetGeometry(imagefilename=r"C:\Users\Administrator\Documents\members\Joe\LDCT-VAM-JOE\vamtoolbox\staticresources\flower.png",pixels=N_x, binarize_image=False)
    x_array = checkerboard(N_x,N_y,10)
    x_spiral = transform.ArchimedesSpiral(target_geo.array)
    # fig, axs = plt.subplots(1,1)
    # axs.imshow(x_spiral[:,:,0].cpu().numpy(),cmap='viridis')
    # plt.show()

    
    x_flat = transform.spiralToFlat(x_spiral)

    fig, axs = plt.subplots(2,1)
    axs[0].imshow(x_spiral[:,:,0].cpu().numpy(),cmap='viridis')
    axs[1].imshow(x_flat[:,:,0].cpu().numpy(),cmap='viridis')
    plt.show()
    input()
    # fig, axs = plt.subplots(2,1)
    # axs[0].imshow(a[:,0,:].T,cmap='viridis')
    # axs[1].imshow(target,cmap='viridis')

    # fig.show()





    s = torch.tensor([torch.pi/180*1*(r_i+r_o)/2])
    stride = torch.round(s)

    transform = WrapTransformation(r_i,tau,spatial_sampling_rate,angles_per_circumference,N_z)
    transform.attenuationGrid(alpha=0.5)

    # x = transform.rho_g_w.detach().clone()
    # x[torch.logical_and(transform.rho_g_w<=r_o,transform.rho_g_w>=r_i)] = 1
    # x[transform.rho_g_w<r_i] = 0
    # x[transform.rho_g_w>r_o] = 0

    x = torch.zeros((N_x,N_y,N_z),device=gpu)
    width = 40
    for i in range(N_y//width):
        i_start = 2*i*width
        i_end = i_start + width
        val = -N_y//(width*2) + i
        x[i_start:i_end,:,0] = i+1
    # x = x*torch.linspace(0.4,1,N_y,device=gpu)[None,:,None].broadcast_to((N_x,N_y,N_z))
    # x[:,:,1] = 1
    x_uw = transform.wrappedToUnwrapped(x)

    print(f'wrapped shape: {transform.x_g_w.shape} x-y size: {transform.N_x_w*transform.N_y_w}')
    print(f'unwrapped shape: {transform.l_g_uw.shape} l-rho size: {transform.N_l_uw*transform.N_rho_uw}')

    fig, axs = plt.subplots(2,1)
    axs[0].imshow(x[:,:,0].cpu().numpy(),cmap='viridis')
    axs[1].imshow(x_uw[:,:,0].cpu().numpy(),cmap='viridis')

    # fig.savefig('transformationtest.jpg',dpi=300)
    plt.show()

