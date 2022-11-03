import vamtoolbox as vam
import numpy as np
import vedo
import vedo.applications

#2D targets
# target_geo = vam.geometry.TargetGeometry(imagefilename=vam.resources.load("reschart.png"),pixels=501)
target_geo = vam.geometry.TargetGeometry(imagefilename=r"C:\Users\ccli\Downloads\boss.jpeg", pixels=501, binarize_image=False, clip_to_circle= True)
# target_geo = vam.geometry.TargetGeometry(imagefilename=r"C:\Users\ccli\Downloads\grayscale bars.png",pixels=501, binarize_image=False)
# target_geo = vam.geometry.TargetGeometry(imagefilename=r"C:\Users\ccli\Downloads\thisisfine_nobg.png",pixels=501, binarize_image=False)
# target_geo.show()

#3D targets
# target_geo = vam.geometry.TargetGeometry(stlfilename=vam.resources.load("trifurcatedvasculature.stl"),resolution=300)
# target_geo = vam.geometry.TargetGeometry(stlfilename=vam.resources.load("thinker.stl"),resolution=300)
# target_geo = vam.geometry.TargetGeometry(stlfilename=r"C:\Users\ccli\Documents\internship project - junction\Drawings\FRMMA-Conformed-new.STL",resolution=200)
# vol = vedo.Volume(target_geo.array,mode=0)

# vedo.applications.RayCastPlotter(vol,bg='black').show(viewup="x")

num_angles = 360
angles = np.linspace(0, 180 - 180 / num_angles, num_angles)
proj_geo = vam.geometry.ProjectionGeometry(angles,ray_type='parallel',CUDA=True)

#Identity response model
# response_model = vam.material.ResponseModel(form = "identity")
#Linear response with negative offset
# response_model = vam.material.ResponseModel(form = "linear", M = 2, C = -1)
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=50,filter='ram-lak_freq',verbose='plot', learning_rate = 0.02, response_model = response_model, eps=0.05)
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=30, filter='ram-lak_freq',verbose='plot', p=2, learning_rate = 0.01, response_model = "default", eps=0.05) #Working

# ========================================= q = 1 ==============================================================================
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=50, filter='ram-lak_freq',verbose='plot', p=2, learning_rate = 0.01, response_model = "default", eps=0.05) #Working
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=50, filter='ram-lak_freq',verbose='plot', p=1, learning_rate = 1e-5, response_model = "default", eps=0.05) #It works stably up to learning_rate = 1e-5 
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=50, filter='ram-lak_freq',verbose='plot', p=0.5, learning_rate = 1e-10, response_model = "default", eps=0.5) # It works stably up to learning_rate = 1e-10
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=50, filter='ram-lak_freq',verbose='plot', p=10, learning_rate = 3, response_model = "default", eps=0.05) #Works
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=50, filter='ram-lak_freq',verbose='plot', p=50, learning_rate = 15, response_model = "default", eps=0.05) #It works stably up to learning_rate = 15 

# ========================================= q = p ==============================================================================
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=50, filter='ram-lak_freq',verbose='plot', p=0.5, q=0.5, learning_rate = 1e-3, response_model = "default", eps=0.5) # stable until learning_rate = 1e-10
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=50, filter='ram-lak_freq',verbose='plot', p=1, q=1, learning_rate = 1e-5, response_model = "default", eps=0.05) #stable until learning_rate = 1e-5 
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=50, filter='ram-lak_freq',verbose='plot', p=2, q=2, learning_rate = 1e-4, response_model = "default", eps=0.05) #stable until learning_rate = 1e-4
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=50, filter='ram-lak_freq',verbose='plot', p=10, q=10, learning_rate = 0.01, response_model = "default", eps=0.05) #stable until learning_rate = 1e-2
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=50, filter='ram-lak_freq',verbose='plot', p=50, q=50, learning_rate = 0.1, response_model = "default", eps=0.05) #stable until learning_rate = 1e-1 
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=50, filter='ram-lak_freq',verbose='plot', p=100, q=100, learning_rate = 10, response_model = "default", eps=0.05) #stable until learning_rate = 1e1, other norms also stable at learning_rate = 1e0 

# ========================================= q = p , eps = 0 ==============================================================================
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=50, filter='ram-lak_freq',verbose='plot', p=0.5, q=0.5, learning_rate = 1e-5, response_model = "default", eps=0) #  not converging in this setting 
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=50, filter='ram-lak_freq',verbose='plot', p=1, q=1, learning_rate = 1e-5, response_model = "default", eps=0) # 
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=50, filter='ram-lak_freq',verbose='plot', p=2, q=2, learning_rate = 1e-4, response_model = "default", eps=0) #
optimizer_params = vam.optimize.Options(method='BCLP',n_iter=50, filter='ram-lak_freq',verbose='plot', p=10, q=10, learning_rate = 1e-2, response_model = "default", eps=0) # works well at 1e-2
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=50, filter='ram-lak_freq',verbose='plot', p=50, q=50, learning_rate = 0.1, response_model = "default", eps=0) #
# optimizer_params = vam.optimize.Options(method='BCLP',n_iter=50, filter='ram-lak_freq',verbose='plot', p=100, q=100, learning_rate = 10, response_model = "default", eps=0) # 

print(optimizer_params)
opt_sino, opt_recon, opt_response, logs = vam.optimize.optimize(target_geo, proj_geo,optimizer_params, output = 'full')
print(f'Starting loss is: {logs.loss[0]} , and final loss is : {logs.loss[-1]}')
print(f'Loss = {logs.loss}')

# opt_recon.show()
# opt_sino.show()
logs.show()
input()
# Display 3D reconstruction
# vol = vedo.Volume(opt_recon.array,mode=0)
# vedo.applications.RayCastPlotter(vol,bg='black').show(viewup="x")

