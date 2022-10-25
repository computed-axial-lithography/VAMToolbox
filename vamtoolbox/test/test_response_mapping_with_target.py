import vamtoolbox as vam
import numpy as np

#2D test
# target_geo = vam.geometry.TargetGeometry(imagefilename=vam.resources.load("reschart.png"),pixels=501)
# target_geo.show()

# rm = vam.material.ResponseModel()
# print(rm.checkResponseTarget(target_geo.array))
# map_inv = rm.map_inv(target_geo.array)

# target_geo_inv = target_geo
# target_geo_inv.array = map_inv
# target_geo_inv.show()


#3D test

target_geo = vam.geometry.TargetGeometry(stlfilename=vam.resources.load("trifurcatedvasculature.stl"),resolution=200)
target_geo.show()

rm = vam.material.ResponseModel()
print(rm.checkResponseTarget(target_geo.array))
map_inv = rm.map_inv(target_geo.array)

target_geo_inv = target_geo
target_geo_inv.array = map_inv
target_geo_inv.show()