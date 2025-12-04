import vamtoolbox as vam

# Medium models define the optical properties of the printing domain
# Three different types of medium models are implemented
# 1. AttenuationModel is used to simulate the light intensity during propagation.
# 2. AbsorptionModel is used to compute the portion of light that is absorbed by the active component. Implementation-wise, it is an alias of AttenuationModel meaning options are identical. The AbsorptionModel and AttenuationModel are often the same when there is only one absorbing species and it solely contributes to light attenuation.
# 3. IndexModel is used to simulate light refraction during propagation.
# NOTE: Medium models are only implemented with ray tracing propagation.
# See the documentation of each medium model for detailed description

target_geo = vam.geometry.TargetGeometry(
    imagefilename=vam.resources.load("flower.png"), pixels=512, binarize_image=False
)

# Setup projection geometry
coord_vec = target_geo.constructCoordVec()


index_model = vam.medium.IndexModel(
    coord_vec, type="interpolation", form="homogeneous", n_sur=1.5
)
index_model.plotIndex(block=False)
index_model = vam.medium.IndexModel(
    coord_vec,
    type="interpolation",
    form="luneburg_lens",
    R=1.2,
    length_unit_of_R="fractional_domain_radius",
)
index_model.plotIndex(block=False)

attenuation_model = vam.medium.AttenuationModel(
    coord_vec, type="analytical", form="homogeneous_cylinder", R=1, alpha_internal=0.2
)
attenuation_model.plotAlpha(block=True)

absorption_model = attenuation_model
absorption_model.plotAlpha(block=True)
