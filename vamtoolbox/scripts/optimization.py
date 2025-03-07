import torch
import vamtoolbox as vam
import numpy as np
import logging
import matplotlib.pyplot as plt
import scipy
import os
import sys
import PIL.Image
import time
import skimage.draw


logging.basicConfig() #initialize the logging module
logger = logging.getLogger('vamtoolbox') #Set the logging level in the whole vamtoolbox
logger.setLevel(logging.DEBUG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('Computing on device: {}'.format(device))
logger.info(torch.cuda.device_count())

class configr2r:
    def __init__(self,name,alpha_mat,n_sur,n_mat,tau,r_i,l_max,deg_inc,gradient=1.0,hours_set=0):
        
        self.name = name
        self.alpha_mat = alpha_mat
        self.n_sur = n_sur
        self.n_mat = n_mat
        self.tau = tau
        self.r_i = r_i
        self.r_o = self.r_i + self.tau

        self.l_max = l_max
        self.gradient = gradient

        self.hours_set = hours_set

        self.deg_inc = deg_inc
        self.f_rho = 1/(0.00108*1.33)
        self.f_l = 1/(self.deg_inc*np.pi/180*self.r_i)

        self.anisotropy_factor = self.f_rho/self.f_l
        
        self.N_l = int(self.f_l*self.l_max)
        self.N_rho = int(self.f_rho*(self.r_o-self.r_i))
        self.N_z = int(self.f_rho*self.tau)
    
    def __repr__(self,):
        '''Returns config string with name and all parameters'''
        # return (f'{self.name}_alpha_mat={self.alpha_mat}_n_sur={self.n_sur}_n_mat={self.n_mat}_r_i={self.r_i}_tau={self.tau}_l_max={self.l_max}_gradient={self.gradient}_deg_inc={self.deg_inc}_hrs_set={self.hours_set}')
        return (f'{self.name}_alpha_mat={self.alpha_mat}_n_sur={self.n_sur}_n_mat={self.n_mat}_r_i={self.r_i}_tau={self.tau}_l_max={self.l_max}_gradient={self.gradient}_deg_inc={self.deg_inc}')
    
    def printFull(self,):
        print(f'{self.name}')
        print(f'alpha_mat = {self.alpha_mat} 1/cm')
        print(f'n_sur = {self.n_sur} n_mat = {self.n_mat}') 
        print(f'tau = {self.tau} cm')
        print(f'r_i = {self.r_i} cm')
        print(f'f_rho = {self.f_rho} voxels/cm f_l = {self.f_l} voxels/cm')
        print(f'deg_inc = {self.deg_inc} °')
        print(f'l_max = {self.l_max} cm')
        print(f'hours_set = {self.hours_set} hr')

    def matrixStr(self,):
        '''Returns config string with only propagation relevant parameters'''
        return (f'alpha_mat={self.alpha_mat}_n_sur={self.n_sur}_n_mat={self.n_mat}_r_i={self.r_i}_tau={self.tau}_deg_inc={self.deg_inc}')
        # return (f'alpha_mat={self.alpha_mat}_n_sur={self.n_sur}_n_mat={self.n_mat}_reflection_r_i={self.r_i}_tau={self.tau}_deg_inc={self.deg_inc}')
        # print(f"Spatial sampling rate rho = {spatial_sampling_rate_rho}")
        # print(f"Spatial sampling rate L = {spatial_sampling_rate_l}")
        # print(f"Anistropy factor = {anisotropy_factor}")
        # print(f"Target shape = {N_rho},{N_l},{N_z}")



################ CREATE TARGET GEOMETRY AND SETUP CONFIGURATION ####################
# Choose slice_2D = True to run optimization on a single layer of the 3D geometry/stack of layers
geometry = "aperiodic_target"
slice_2D = True

match geometry:

    case "sheet_gyroid":
    # case "tpms_gyroid_sheet5":
        config = configr2r(geometry,2.2, 1.33, 1.49, 0.3, 0.76, 5.0, 0.5, [-0.5])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='gyroid-sheet')
        # a = a[:,:,0]
        # a = a[:,:,None]

    case "skeletal_gyroid":
        config = configr2r(geometry,2.2, 1.33, 1.49, 0.3, 0.76, 5.0, 0.5, [-1.2])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='gyroid-skel')
        # a = a[:,:,0]
        # a = a[:,:,None]

    case "aperiodic_target":
        config = configr2r(geometry,2.2, 1.33, 1.49, 0.3, 0.76, 5.0, 0.5, 1.0)
        config.N_z = 1
        image = PIL.Image.open(r"vamtoolbox\data\aperiodic_target.png").convert('L')
        image = image.resize((config.N_l,config.N_rho))
        a = np.array(image,dtype=np.float32)
        a /= np.max(a)
        a[a<0.5] = 0
        a = a[:,:,None]

if slice_2D:
    a = a[:,:,0]
    a = a[:,:,None]

################ SETUP CONFIGURATION ####################
print(config.printFull())
directory = rf"vamtoolbox\data\optimization_output\{config}"

################ SETUP SAVING PATH ####################
if not os.path.exists(directory):
    os.makedirs(directory)

if os.path.exists(rf"{directory}\sinogram.npy"):
    # array_num = 3
    base_u_offset = -100

    sinogram = np.load(rf"{directory}\sinogram.npy")
    # sinogram = sinogram[41:sinogram.shape[0]-41,:,:]
    fresnel_mask = vam.medium.fresnelMask(config.n_mat,config.n_sur,config.r_i,config.tau,N_r=sinogram.shape[0],N_z=sinogram.shape[2],pixel_size=10.8*1.33*1e-4,maximum=2)
    sinogram = sinogram*fresnel_mask[:,None,:]

    dmd_nonhomogeneous_mask = np.load(rf"C:\Users\Administrator\Documents\members\Joe\dmdIntensityMask\intensity_mask.npy")

    avg = np.average(sinogram[sinogram>0])
    perc_99 = np.percentile(sinogram,99)
    print(avg)
    print(perc_99)
    z_length = sinogram.shape[2]
    # array_offset = np.stack((np.linspace(0,0,array_num,dtype=int),np.linspace(-1,1,array_num)*z_length*(array_num-1)/2),axis=1)
    image_config = vam.imagesequence.ImageConfig(image_dims=(1920,1080),
                                                u_offset=base_u_offset,
                                                array_num=array_num,
                                                array_offset=array_offset,
                                                intensity_scale=1,
                                                normalization_percentile=99.5,
                                                bit_depth=8,
                                                mask=dmd_nonhomogeneous_mask,
                                                )
    image_seq = vam.imagesequence.ImageSeq(image_config,sinogram)

    image_seq.saveAsVideo(rf"D:\Joe\test_images\{config}_imageseq_offset={array_offset[:,0]+base_u_offset}_masked.mp4",rot_vel=3.01,num_loops=1,mode='prescribed',angle_increment_per_image=0.5,preview=False)
    np.save(rf"D:\Joe\test_images\{config}_imageseq_offset={array_offset[:,0]+base_u_offset}_masked.npy",image_seq.images)


    if continuous:= True:
        num_images_per_period = int(config.tau/config.r_i *180/np.pi / config.deg_inc) #assuming length of repeating unit cell is equal to tau
        image_range = [np.floor(len(image_seq.images)//2-num_images_per_period//2).astype(int),np.ceil(len(image_seq.images)//2+num_images_per_period//2).astype(int)+1]
        np.save(rf"D:\Joe\test_images\{config}_imageseq_offset={array_offset[:,0]+base_u_offset}_masked_continuous.npy",image_seq.images[image_range[0]:image_range[1]])

    sys.exit()


target_geo = vam.geometry.TargetGeometryR2R(target=a,r_i=config.r_i,tau=config.tau,spatial_sampling_rate_rho=config.f_rho,spatial_sampling_rate_l=config.f_l)
coord_vec = target_geo.constructCoordVec()


np.save(rf'{directory}\target.npy',target_geo.array)

# Plot target
fig, axs = plt.subplots(1,1)
im_recon = axs.imshow(target_geo.array[:,:,target_geo.array.shape[2]//2],extent=[0,target_geo.nL_total/target_geo.transform.spatial_sampling_rate_l,target_geo.transform.r_o,target_geo.transform.r_i],cmap='gray')
im_recon.set_clim(0,1)
axs.set_xlabel(r'$L$ [cm]')
axs.set_ylabel(r'$\rho$ [cm]')
vam.util.matplotlib.addColorbar(im_recon)
fig.tight_layout()
fig.savefig(rf'{directory}\target.png',dpi=600)
plt.show()

############# SETUP RAYTRACING MEDIUM FOR PROJECTION ####################
attenuation_model = vam.medium.AttenuationModel(coord_vec, type = 'interpolation', form='homogeneous_r2r', alpha_internal=config.alpha_mat, rho_grid=target_geo.transform.rho_g_w, r_mid=target_geo.transform.r_mid, tau=target_geo.transform.tau)
absorption_model = vam.medium.AttenuationModel(coord_vec, type = 'interpolation', form='homogeneous_r2r', alpha_internal=config.alpha_mat, rho_grid=target_geo.transform.rho_g_w, r_mid=target_geo.transform.r_mid, tau=target_geo.transform.tau)
index_model = vam.medium.IndexModel(coord_vec, type = 'interpolation', form='homogeneous_r2r', n_sur=config.n_sur, n=config.n_mat, rho_grid=target_geo.transform.rho_g_w, r_mid=target_geo.transform.r_mid, tau=target_geo.transform.tau)
response_model = vam.response.ResponseModel(B=10)

matrix_sino2uw_filename = rf'vamtoolbox\data\matrix_sino2uw_{config.matrixStr()}.pt'
angles = np.array([269.9])

if os.path.exists(matrix_sino2uw_filename):
    P_matrix_sino2uw = torch.load(matrix_sino2uw_filename,map_location=device)
    proj_geo_al = vam.geometry.ProjectionGeometry(angles,
                                            ray_type='algebraic_r2r',
                                            CUDA=True,
                                            loading_path_for_matrix = matrix_sino2uw_filename
                                            )
    
    P_al = vam.projector.pyTorchAlgebraicPropagationR2R.PyTorchAlgebraicR2RPropagator(target_geo, proj_geo_al,output_torch_tensor=False)

else:
    proj_geo_rt = vam.geometry.ProjectionGeometry(angles,
                                                ray_type='ray_trace',
                                                CUDA=True,
                                                index_model = index_model,
                                                attenuation_model = attenuation_model,
                                                absorption_model = absorption_model,
                                                ray_trace_method = 'eikonal',
                                                eikonal_parametrization = 'physical_path_length',
                                                ray_trace_ode_solver = 'forward_symplectic_euler',
                                                ray_trace_ray_config = 'r2r',
                                                inclination_angle = 0.0,
                                                ray_density = 1
                                                )

    #Direct call high level function
    P_rt = vam.projector.pyTorchRayTraceR2R.PyTorchRayTracingPropagator(target_geo, proj_geo_rt, output_torch_tensor=True)

    # test_sino = torch.zeros(P_rt.ray_state.sino_shape,device=device)
    # test_sino[20:30,0,0] = 1
    test_sino = torch.ones(P_rt.ray_state.sino_shape,device=device)
    # test_sino[0::40,0,:] = 1
    
    test_recon = P_rt.backward(test_sino.ravel())
    # test_recon = test_recon.reshape(target_geo.transform.x_g_w.shape).cpu().numpy()
    test_recon = test_recon.reshape(target_geo.transform.x_g_uw.shape).cpu().numpy()
    # test_recon = test_recon.reshape(target_geo.array.shape).cpu().numpy()

    fig, axs = plt.subplots(1,1)
    axs.imshow(test_recon[:,:,0])
    plt.show()

    P_matrix_sino2uw = P_rt.buildPropagationMatrix()
    torch.save(P_matrix_sino2uw,matrix_sino2uw_filename)
    sys.exit()


########## CREATE INITIALIZATION FOR OPTIMIZATION ################
g0 = P_al.forward(response_model.map_inv(target_geo.array))
g0 /= P_al.n_fold_blocks*100

# Plot initialization sinogram
fig, axs = plt.subplots(1,1)
im_sino = axs.imshow(g0[:,:,target_geo.array.shape[2]//2])
axs.set_xlabel(r'$\theta$ []')
axs.set_ylabel(r'$\rho$ [cm]')
vam.util.matplotlib.addColorbar(im_sino)
fig.savefig(rf'{directory}\initializationsinogram.png',dpi=600)


############## SETUP OPTIMIZATION PARAMETERS ####################
weight = np.ones_like(target_geo.array)
eps_map = np.ones_like(target_geo.array)
eps_map[target_geo.array > 0] = 0.01
eps_map[target_geo.array == 0] = 0.05

weight[target_geo.array>0] = 3
# weight[target_geo.array==0] = 3

optimizer_params = vam.optimize.Options(method='BCLP',
                                        n_iter=100,
                                        filter='ram-lak',
                                        verbose='plot',
                                        p=3,
                                        learning_rate = 0.01,
                                        regularize_l = 0.04,
                                        # p=2,
                                        # learning_rate = 0.0005,
                                        # regularize_l = 0.05,
                                        momentum = 0.5,
                                        response_model = response_model,
                                        # eps=0.01,
                                        eps=eps_map,
                                        g0=g0,
                                        weight=weight,
                                        # bit_depth=8,
                                        # save_img_path=rf'test_output\{config}'
                                        ) #Working



######################## EXECUTE OPTIMIZATION #####################################
opt_sino, opt_recon, opt_response, logs = vam.optimize.optimize(target_geo, proj_geo_al,optimizer_params, output = 'components')


np.save(rf'{directory}\sinogram.npy',opt_sino)
np.save(rf'{directory}\reconstruction.npy',opt_recon)
np.save(rf'{directory}\response.npy',opt_response)


######################### SAVE FIGURES FROM OPTIMIZATION ############################3
plt.savefig(rf'{directory}\optimizationplot.png',dpi=600)


fig, axs = plt.subplots(1,1)
im_sino = axs.imshow(opt_sino[:,:,target_geo.array.shape[2]//2])
axs.set_xlabel(r'$\theta$ []')
axs.set_ylabel(r'$\rho$ []')
vam.util.matplotlib.addColorbar(im_sino)
fig.tight_layout()
fig.savefig(rf'{directory}\sinogram.png',dpi=600)

# plt.show()
fig, axs = plt.subplots(1,1)
im_recon = axs.imshow(opt_recon[:,:,target_geo.array.shape[2]//2],extent=[0,target_geo.nL_total/target_geo.transform.spatial_sampling_rate_l,target_geo.transform.r_o,target_geo.transform.r_i])
# im_recon.set_clim(0,1)
axs.set_xlabel(r'$L$ [cm]')
axs.set_ylabel(r'$\rho$ [cm]')
vam.util.matplotlib.addColorbar(im_recon)
fig.tight_layout()
fig.savefig(rf'{directory}\reconstruction.png',dpi=600)

fig, axs = plt.subplots(1,1)
im_recon = axs.imshow(opt_recon[:,0:200,target_geo.array.shape[2]//2],extent=[0,200/target_geo.transform.spatial_sampling_rate_l,target_geo.transform.r_o,target_geo.transform.r_i])
# im_recon.set_clim(0,1)
axs.set_xlabel(r'$L$ [cm]')
axs.set_ylabel(r'$\rho$ [cm]')
vam.util.matplotlib.addColorbar(im_recon)
fig.tight_layout()
fig.savefig(rf'{directory}\reconstruction_zoom.png',dpi=600)
plt.show()
