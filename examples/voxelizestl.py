import vamtoolbox as vam
target_geo = vam.geometry.TargetGeometry(stlfilename=vam.resources.load("trifurcatedvasculature.stl"),resolution=250)

target_geo.show()

import vedo
vol = vedo.Volume(target_geo.array).legosurface(vmin=0.5,vmax=1.5)
vol.show(viewup="x")