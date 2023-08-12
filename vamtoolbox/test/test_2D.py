import vamtoolbox as vam
import numpy as np
import matplotlib.pyplot as plt



def fourQuadrantGrating(n_pixel,periods=5,bit_levels=[1,2,4,8],range=[0,1]):
    X, Y = np.meshgrid(np.linspace(-np.sqrt(2),np.sqrt(2),n_pixel),np.linspace(-np.sqrt(2),np.sqrt(2),n_pixel))
    X[np.abs(X)>1] = np.nan
    Y[np.abs(Y)>1] = np.nan

    average_value = np.mean(range)
    # average_value = 0
    PV_amplitude = range[1] - range[0]

    composite = np.zeros((n_pixel,n_pixel))
    quadrants = [[1,1],[1,-1],[-1,-1],[-1,1]]
    for i, (bit_level, quadrant) in enumerate(zip(bit_levels,quadrants)):
        _X = np.where(np.sign(X+np.spacing(1.0))==quadrant[0],1,0)
        _Y = np.where(np.sign(Y+np.spacing(1.0))==quadrant[1],1,0)
        if i % 2 == 0:
            quad = (PV_amplitude/2*np.sin(2*np.pi*Y*periods) + average_value)*(1+PV_amplitude/(2**bit_level-1))
        else:
            quad = (PV_amplitude/2*np.sin(2*np.pi*X*periods) + average_value)*(1+PV_amplitude/(2**bit_level-1))
        
        

        quad = vam.util.data.discretize(quad,bit_depth=bit_level,range=range,output_dtype=np.float64)

        quad[_X!=1] = 0
        quad[_Y!=1] = 0
       
        composite += quad
    return composite
    # return composite
# # target_geo = vam.geometry.TargetGeometry(imagefilename=vam.resources.load("reschart.png"),pixels=501)
# target_geo = vam.geometry.TargetGeometry(imagefilename=vam.resources.load("flower.png"),pixels=501,binarize_image=False)
# target_geo.show()
# # target_geo = vam.geometry.TargetGeometry(imagefilename=r"C:\Users\ccli\OneDrive - Facebook\Desktop\logo-Meta.png", pixels=n_pixel)
# target_geo.array = np.logical_not(target_geo.array)
# for row in range(target_geo.array.shape[0]):
#     if np.sum(target_geo.array[row, :]) == 501:
#         target_geo.array[row, :] = 0


# num_angles = 360
# angles = np.linspace(0, 360 - 360 / num_angles, num_angles)
# proj_geo = vam.geometry.ProjectionGeometry(angles, ray_type='parallel', CUDA=True)

# optimizer_params = vam.optimize.Options(method='OSMO', n_iter=10, d_h=0.85, d_l=0.5, filter='hamming', verbose='plot')
# opt_sino, opt_recon, error = vam.optimize.optimize(target_geo, proj_geo, optimizer_params)
# opt_recon.show()
# opt_sino.show()



if __name__ == "__main__":
    n_pixel = 501
    target = fourQuadrantGrating(n_pixel)

    fig, ax = plt.subplots(1,1)
    ax.imshow(target,cmap="gray")
    # fig.savefig("fourquadrantgrating.png",dpi=300)
    plt.show()
