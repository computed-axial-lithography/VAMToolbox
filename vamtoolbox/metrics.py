import numpy as np
import vamtoolbox

def calcCV(target,recon,layerwise=False):
    """
    Calculate coefficient of variance in target region

    Parameters
    ----------
        target as the reference to which the reconstruction is compared

    recon : ndarray
        reconstruction to evaluate

    layerwise : bool, optional
        if True, the CV is computed per z-slice

    Returns
    -------

    float or ndarray
        if layerwise == True, ndarray will have size equal to the # of z slices

    """


    gel_inds, void_inds = target.gel_inds, target.void_inds
    
    if layerwise == False:
        gel_mean = np.mean(recon[gel_inds])
        gel_std_dev = np.std(recon[gel_inds])
        CV = gel_std_dev/gel_mean
        return CV

    else:
        nX,nY,nZ = recon.shape
        CV_arr = np.zeros(nZ)
        for z in range (nZ):
            gel_mean = np.mean(recon[gel_inds])
            gel_std_dev = np.std(recon[gel_inds])
            CV = gel_std_dev/gel_mean
            CV_arr[z] = CV
        return CV_arr



def calcVER(target,recon,layerwise=False):
    """
    Calculate volumetric error rate

    Parameters
    ----------
        target as the reference to which the reconstruction is compared

    recon : ndarray
        reconstruction to evaluate

    layerwise : bool, optional
        if True, the VER is computed per z-slice

    Returns
    -------

    float or ndarray
        if layerwise == True, ndarray will have size equal to the # of z slices

    """
    gel_inds, void_inds = target.gel_inds, target.void_inds
    num_gel_void = np.sum(gel_inds[:]) + np.sum(void_inds[:])
    
    if layerwise == False:
        min_gel_dose = np.min(recon[gel_inds])
        void_doses = recon[void_inds]
        n_pix_overlap = np.sum(void_doses >= min_gel_dose)
        VER = n_pix_overlap/num_gel_void
        return VER 

    else:
        nX,nY,nZ = recon.shape
        VER_arr = np.zeros(nZ)
        for z in range (nZ):
            recon_slice = recon[:,:,z]
            min_gel_dose = np.min(recon_slice[gel_inds])
            void_doses = recon_slice[void_inds]
            n_pix_overlap = np.sum(void_doses >= min_gel_dose)
            VER_arr[z] = n_pix_overlap/num_gel_void
        return VER_arr


def calcPW(target,recon,layerwise=False):
    """
    Calculate process window

    Parameters
    ----------
        target as the reference to which the reconstruction is compared

    recon : ndarray
        reconstruction to evaluate

    layerwise : bool, optional
        if True, the PW is computed per z-slice

    Returns
    -------

    float or ndarray
        if layerwise == True, ndarray will have size equal to the # of z slices

    """
    gel_inds, void_inds = target.gel_inds, target.void_inds

    if layerwise == False:
        min_gel_dose = np.min(recon[gel_inds])
        max_void_dose = np.max(recon[void_inds])
        PW = max_void_dose - min_gel_dose
        return PW
        
    else:
        nX,nY,nZ = recon.shape
        PW_arr = np.zeros(nZ)
        for z in range (nZ):
            recon_slice = recon[:,:,z]
            min_gel_dose = np.min(recon_slice[gel_inds])
            max_void_dose = np.max(recon_slice[void_inds])
            PW_arr[z] = max_void_dose - min_gel_dose
        return PW_arr


def calcIPDR (target,recon,layerwise=False):
    """
    Calculate in-part dose range

    Parameters
    ----------
        target as the reference to which the reconstruction is compared

    recon : ndarray
        reconstruction to evaluate

    layerwise : bool, optional
        if True, the IPDR is computed per z-slice

    Returns
    -------

    float or ndarray
        if layerwise == True, ndarray will have size equal to the # of z slices

    """
    gel_inds, void_inds = target.gel_inds, target.void_inds

    if layerwise == False:
        min_gel_dose = np.min(recon[gel_inds])
        max_gel_dose = np.max(recon[gel_inds])
        IPDR = max_gel_dose - min_gel_dose
        return IPDR
        
    else:
        nX,nY,nZ = recon.shape
        IPDR_arr = np.zeros(nZ)
        for z in range (nZ):
            recon_slice = recon[:,:,z]
            min_gel_dose = np.min(recon_slice[gel_inds])
            max_gel_dose = np.max(recon_slice[gel_inds])
            IPDR_arr[z]= max_gel_dose - min_gel_dose
        return IPDR_arr



