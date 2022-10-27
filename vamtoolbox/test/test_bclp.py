import vamtoolbox as vam
import numpy as np

#2D targets
# target_geo = vam.geometry.TargetGeometry(imagefilename=vam.resources.load("reschart.png"),pixels=501)
target_geo = vam.geometry.TargetGeometry(imagefilename=r"C:\Users\ccli\Downloads\boss.jpeg",pixels=501, binarize_image=False)
# target_geo = vam.geometry.TargetGeometry(imagefilename=r"C:\Users\ccli\Downloads\grayscale bars.png",pixels=501, binarize_image=False)
# target_geo = vam.geometry.TargetGeometry(imagefilename=r"C:\Users\ccli\Downloads\thisisfine_nobg.png",pixels=501, binarize_image=False)

#3D targets
# target_geo = vam.geometry.TargetGeometry(stlfilename=vam.resources.load("trifurcatedvasculature.stl"),resolution=300)
# target_geo = vam.geometry.TargetGeometry(stlfilename=vam.resources.load("thinker.stl"),resolution=300)

target_geo.show()

num_angles = 360
angles = np.linspace(0, 360 - 360 / num_angles, num_angles)
proj_geo = vam.geometry.ProjectionGeometry(angles,ray_type='parallel',CUDA=True)

#Identity response model
# response_model = vam.material.ResponseModel(form = "identity")
#Linear response with negative offset
# response_model = vam.material.ResponseModel(form = "linear", M = 2, C = -1)
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=30,filter='ram-lak',verbose='plot', learning_rate = 0.02, response_model = response_model, eps=0.05)
optimizer_params = vam.optimize.Options(method='BCLP',n_iter=30,filter='ram-lak',verbose='plot', learning_rate = 0.02, response_model = "default", eps=0.05)
opt_sino, opt_recon, error = vam.optimize.optimize(target_geo, proj_geo,optimizer_params)
# opt_recon.show()
opt_sino.show()

#Display 3D reconstruction
# import vedo
# import vedo.applications
# vol = vedo.Volume(opt_recon.array,mode=0)
# vedo.applications.RayCastPlotter(vol,bg='black').show(viewup="x")

