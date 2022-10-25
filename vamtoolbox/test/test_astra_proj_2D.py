import vamtoolbox as vam
import numpy as np
import matplotlib.pyplot as plt

target_geo = vam.geometry.TargetGeometry(imagefilename=vam.resources.load("reschart.png"),pixels=501)
# target_geo.show()

num_angles = 512
angles = np.linspace(0, 360 - 360 / num_angles, num_angles)
proj_geo = vam.geometry.ProjectionGeometry(angles,ray_type='parallel',CUDA=True)

optimizer_params = vam.optimize.Options(method='BCLP',n_iter=30,filter='ram-lak',verbose='plot', learning_rate = 0.1)

P = vam.projectorconstructor.projectorconstructor(target_geo, proj_geo)

g0 = P.forward(target_geo.array)

g0[:] = 1

g0[g0.shape[0]//2,g0.shape[1]//2] = 1

f = P.backward(g0)

fig, ax = plt.subplots()

ax.imshow(f, vmin = np.amin(f), vmax = np.amax(f))

plt.show()

#Result of this test showed that the value of the tomogram is direct sum of projection at all angles.
#Projection at each angle is simply smeared across the real space. Hence a unity projection at angle i will create unity value along the projection line.
#The normalization factor applied onto the tomogram should be 1/n_angles because of the assumption that each projection is allocated even amount of time.
