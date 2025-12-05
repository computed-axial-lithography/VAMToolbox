import vamtoolbox as vam
import numpy as np

target_geo = vam.geometry.TargetGeometry(
    stlfilename=vam.resources.load("trifurcatedvasculature.stl"), resolution=200
)
target_geo.show()

num_angles = 360
angles = np.linspace(0, 360 - 360 / num_angles, num_angles)
proj_geo = vam.geometry.ProjectionGeometry(angles, ray_type="parallel", CUDA=True)

optimizer_params = vam.optimize.Options(
    method="OSMO", n_iter=20, d_h=0.85, d_l=0.6, filter="hamming", verbose="plot"
)
opt_sino, opt_recon, error = vam.optimize.optimize(
    target_geo, proj_geo, optimizer_params
)
opt_recon.show()
opt_sino.show()

import vedo
import vedo.applications

vol = vedo.Volume(opt_recon.array, mode=0)
vedo.applications.RayCastPlotter(vol, bg="black").show(viewup="x")
