from scipy import io
import vamtoolbox

def saveMatData(filepath : str,target_geo : vamtoolbox.geometry.TargetGeometry=None,sinogram :  vamtoolbox.geometry.Sinogram=None,reconstruction :  vamtoolbox.geometry.Reconstruction=None):
    """
    Saves VAM toolbox (Python) objects to Matlab formats

    Parameters
    ----------
    filepath : str
        filepath to save to ending in .mat

    target_geo : geometry.TargetGeometry

    sinogram : geometry.Sinogram

    reconstruction : geometry.Reconstruction 


    Usage
    -----
    >>> opt_sino, opt_recon = optimize.optimize(target_geo, proj_geo,optimizer_params)
    >>> saveMatData('sample.mat',sinogram=opt_sino,reconstruction=opt_recon)

    """

    save_dict = dict()

    if isinstance(sinogram, vamtoolbox.geometry.Sinogram):
        save_dict['opt_proj'] = sinogram.array
    elif sinogram is not None:
        raise Exception("sinogram argument must be of type geometry.Sinogram")
    
    if isinstance(reconstruction, vamtoolbox.geometry.Reconstruction):
        save_dict['opt_recon'] = reconstruction.array
    elif reconstruction is not None:
        raise Exception("reconstruction argument must be of type geometry.Reconstruction")

    if isinstance(target_geo, vamtoolbox.geometry.TargetGeometry):
        save_dict['target'] = target_geo.array
    elif target_geo is not None:
        raise Exception("target_geo argument must be of type geometry.TargetGeometry")
    
    io.savemat(filepath,save_dict)



