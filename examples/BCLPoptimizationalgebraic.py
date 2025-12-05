import vamtoolbox as vam
import numpy as np
import logging

# This example works for 2D targets. For 3D targets, the size of the algebraic representation (P_matrix) is typically too large for practical usage.
# For 3D optimization, we assume the propagations of each z-layer are the same (shift-invariant) so a P_matrix built for 2D propagation can be used for all z-layers in 3D.
# User would need to build the P_matrix in proper size in xy, and the alegbraic propagator will accept this 2D matrix for 3D propagation.

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

proj_geo_rt = vam.geometry.ProjectionGeometry(
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

# Direct call high level function
P_rt = vam.projector.pyTorchRayTrace.PyTorchRayTracingPropagator(
    target_geo, proj_geo_rt, output_torch_tensor=False
)

P_matrix = P_rt.buildPropagationMatrix()
save_path = (
    r"D:\archive\projects\LDCT\P_matrix for examples (360 angles over 360, 512x512).npz"
)
import scipy.sparse

scipy.sparse.save_npz(save_path, P_matrix)

proj_geo_al = vam.geometry.ProjectionGeometry(
    angles, ray_type="algebraic", CUDA=False, loading_path_for_matrix=save_path
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
    target_geo, proj_geo_al, optimizer_params, output="full"
)

opt_recon.show()
opt_sino.show()
logs.show()
