import vamtoolbox as vam

target_geo = vam.geometry.TargetGeometry(
    stlfilename=vam.resources.load("trifurcatedvasculature.stl"), resolution=250
)

target_geo.show()

import vedo

vol = vedo.Volume(target_geo.array).legosurface(vmin=0.5, vmax=1.5)
vol.show(viewup="x")


voxelizer = vam.voxelize.Voxelizer()
voxelizer.addMeshes({vam.resources.load("trifurcatedvasculature.stl"): "print_body"})
array = voxelizer.voxelize(
    "print_body", layer_thickness=0.1, voxel_value=1.0, voxel_dtype="uint8"
)
