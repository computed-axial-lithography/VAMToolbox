import vamtoolbox as vam
import numpy as np

target_geo = vam.geometry.TargetGeometry(
    imagefilename=vam.resources.load("reschart.png"), pixels=501
)

num_angles = 720
angles = np.linspace(0, 360 - 360 / num_angles, num_angles)
proj_geo = vam.geometry.ProjectionGeometry(angles, ray_type="parallel", CUDA=True)

# FBP
optimizer_params = vam.optimize.Options(
    method="FBP", filter="shepp-logan", verbose="plot"
)
opt_sino, opt_recon, error = vam.optimize.optimize(
    target_geo, proj_geo, optimizer_params
)
opt_recon.show()

# CAL
optimizer_params = vam.optimize.Options(
    method="CAL", filter="shepp-logan", n_iter=100, d_h=0.8, d_l=0.7, verbose="plot"
)
opt_sino, opt_recon, error = vam.optimize.optimize(
    target_geo, proj_geo, optimizer_params
)
opt_recon.show()

# PM
optimizer_params = vam.optimize.Options(
    method="PM", filter="ram-lak", n_iter=100, d_h=0.8, d_l=0.7, verbose="plot"
)
opt_sino, opt_recon, error = vam.optimize.optimize(
    target_geo, proj_geo, optimizer_params
)
opt_recon.show()

# OSMO
optimizer_params = vam.optimize.Options(
    method="OSMO", filter="shepp-logan", n_iter=100, d_h=0.8, d_l=0.7, verbose="plot"
)
opt_sino, opt_recon, error = vam.optimize.optimize(
    target_geo, proj_geo, optimizer_params
)
opt_recon.show()
