import vamtoolbox as vam
import numpy as np

target_geo = vam.geometry.TargetGeometry(
    imagefilename=vam.resources.load("flower.png"), pixels=512, binarize_image=False
)

num_angles = 360
angles = np.linspace(0, 180 - 180 / num_angles, num_angles)
proj_geo = vam.geometry.ProjectionGeometry(angles, ray_type="parallel", CUDA=True)

response_model = vam.response.ResponseModel(B=25)

optimizer_params = vam.optimize.Options(
    method="BCLP",
    n_iter=30,
    filter="ram-lak",
    verbose="plot",
    p=2,
    learning_rate=0.00001,
    response_model=response_model,
    eps=0.1,
)

opt_sino, opt_recon, opt_response, logs = vam.optimize.optimize(
    target_geo, proj_geo, optimizer_params, output="full"
)

opt_recon.show()
opt_sino.show()
logs.show()
