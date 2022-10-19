import vamtoolbox as vam
import numpy as np

n_pixel = 501

target_geo = vam.geometry.TargetGeometry(imagefilename=vam.resources.load("reschart.png"),pixels=501)
# target_geo = vam.geometry.TargetGeometry(imagefilename=r"C:\Users\ccli\OneDrive - Facebook\Desktop\logo-Meta.png", pixels=n_pixel)
target_geo.array = np.logical_not(target_geo.array)
for row in range(target_geo.array.shape[0]):
    if np.sum(target_geo.array[row, :]) == 501:
        target_geo.array[row, :] = 0

num_angles = 360
angles = np.linspace(0, 360 - 360 / num_angles, num_angles)
proj_geo = vam.geometry.ProjectionGeometry(angles, ray_type='parallel', CUDA=True)

optimizer_params = vam.optimize.Options(method='OSMO', n_iter=10, d_h=0.85, d_l=0.5, filter='hamming', verbose='plot')
opt_sino, opt_recon, error = vam.optimize.optimize(target_geo, proj_geo, optimizer_params)
opt_recon.show()
opt_sino.show()
