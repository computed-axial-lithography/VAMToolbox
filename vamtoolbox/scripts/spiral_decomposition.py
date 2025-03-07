import torch
import numpy as np
import matplotlib.pyplot as plt
import vamtoolbox as vam


plt.style.use(r'vamtoolbox\scripts\style.mplstyle')

gpu = torch.device('cuda')


tau = 0.3 # cm
r_i = 0.75 # cm
r_o = r_i + 4*tau

deg_inc = 0.5
spatial_sampling_rate_rho = 1/(0.00108*1.33)
spatial_sampling_rate_l = 1/(deg_inc*np.pi/180*r_i)
anisotropy_factor = spatial_sampling_rate_rho/spatial_sampling_rate_l

N_x = 1000
N_y = 1000
N_z = 1


transform = vam.projector.transformationR2R.SpiralTransformation(tau,r_i,r_o,spatial_sampling_rate_rho=spatial_sampling_rate_rho,spatial_sampling_rate_l=spatial_sampling_rate_l,N_x=N_x,N_y=N_y,N_z=N_z)
target_geo = vam.geometry.TargetGeometry(imagefilename=r"vamtoolbox\data\spiral_network.png",pixels=N_x, binarize_image=False)


# 2D test image 
x_spiral, x_arch_spiral, y_arch_spiral = transform.ArchimedesSpiral()
# x_flat = transform.spiralToFlat(x_spiral)


x_flat = transform.spiralToFlat(target_geo.array)



fig, axs = plt.subplots(2,1,figsize=[10,8])

# draw sprial overlay on 2D image
axs[0].imshow(target_geo.array[:,:,0],extent=[-r_o,r_o,-r_o,r_o],cmap='gray')
# axs[0].imshow(x_spiral[:,:,0].cpu().numpy(),extent=[-r_o,r_o,-r_o,r_o],cmap='gray')
axs[0].plot(x_arch_spiral,y_arch_spiral,color='r',linewidth=3)
limit = r_o + tau/2
axs[0].set_xlim([-limit,limit])
axs[0].set_ylim([-limit,limit])

# flattened/unwrapped object to optimize
axs[1].imshow(x_flat[:,:,0].cpu().numpy(),extent=[0,transform.l.cpu().numpy(),0,tau],cmap='gray')
axs[1].axhline(tau/2,color='r',linewidth=3)
axs[1].set_aspect(6)
fig.tight_layout()
plt.show()



