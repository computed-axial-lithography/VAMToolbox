import vamtoolbox as vam
import numpy as np

target_geo = vam.geometry.TargetGeometry(
    stlfilename=vam.resources.load("bear.stl"), resolution=250, rot_angles=[90, 0, 0]
)
# target_geo.show()

num_angles = 360
angles = np.linspace(0, 360 - 360 / num_angles, num_angles)

# no absorption
proj_geo = vam.geometry.ProjectionGeometry(angles, ray_type="parallel", CUDA=True)

optimizer_params = vam.optimize.Options(
    method="OSMO",
    n_iter=40,
    d_h=0.85,
    d_l=0.6,
    filter="hamming",
    units="normalized",
    verbose="plot",
)
opt_sino, opt_recon, error = vam.optimize.optimize(
    target_geo, proj_geo, optimizer_params
)
opt_recon.show(savepath="noabs_recon.jpg", dpi=300)
opt_sino.show(savepath="noabs_sino", dpi=300)

# absorption
proj_geo = vam.geometry.ProjectionGeometry(
    angles,
    ray_type="parallel",
    CUDA=True,
    projector_pixel_size=100e-4,
    absorption_coeff=1,
    container_radius=1,
    n_rot=3,
    rot_vel=24,
)

optimizer_params = vam.optimize.Options(
    method="OSMO",
    n_iter=40,
    d_h=0.85,
    d_l=0.6,
    filter="hamming",
    units="dose",
    verbose="plot",
)
opt_sino, opt_recon, error = vam.optimize.optimize(
    target_geo, proj_geo, optimizer_params
)
opt_recon.show(savepath="abs_recon.jpg", dpi=300)
opt_sino.show(savepath="abs_sino.jpg", dpi=300)
