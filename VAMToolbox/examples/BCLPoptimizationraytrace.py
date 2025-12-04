import vamtoolbox as vam
import numpy as np
import logging

logging.basicConfig()  # initialize the logging module
logger = logging.getLogger(
    "vamtoolbox"
)  # Set the logging level in the whole vamtoolbox
logger.setLevel(
    logging.DEBUG
)  # the logging level can be changed to display info of different importance

target_geo = vam.geometry.TargetGeometry(
    imagefilename=vam.resources.load("flower.png"), pixels=512, binarize_image=False
)

num_angles = 360
angles = np.linspace(0, 180 - 180 / num_angles, num_angles)

# Setup projection geometry
coord_vec = target_geo.constructCoordVec()
index_model = vam.medium.IndexModel(coord_vec, type="interpolation", form="homogeneous")
attenuation_model = vam.medium.AttenuationModel(
    coord_vec, type="analytical", form="homogeneous_cylinder", R=1, alpha_internal=1e-3
)
absorption_model = attenuation_model
response_model = vam.response.ResponseModel(B=25)

proj_geo = vam.geometry.ProjectionGeometry(
    angles,
    ray_type="ray_trace",
    CUDA=True,
    index_model=index_model,
    attenuation_model=attenuation_model,
    absorption_model=absorption_model,
    ray_trace_method="eikonal",
    eikonal_parametrization="physical_path_length",
    ray_trace_ode_solver="forward_symplectic_euler",
    ray_trace_ray_config="parallel",
    inclination_angle=0.0,
    ray_density=1,
)

optimizer_params = vam.optimize.Options(
    method="BCLP",
    n_iter=30,
    filter="ram-lak",
    verbose="plot",
    p=2,
    learning_rate=10,
    response_model=response_model,
    eps=0.1,
)

opt_sino, opt_recon, opt_response, logs = vam.optimize.optimize(
    target_geo, proj_geo, optimizer_params, output="full"
)

opt_recon.show()
opt_sino.show()
logs.show()
