"""
Copyright May 11 2023 Antony Orth and coauthors

"Deconvolution Volumetric Additive Manufacturing"
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.signal import convolve


def blur_ker(mm_in_px, D, tmax, rotspd):  #This function calculates the optical psf + diffusion kernel based on the vial rotation rate and expected print time
    
    dt=360/rotspd
    ntot=int(tmax//dt)
    
    #diffusion kernel setup
    px=mm_in_px
    x,y,z=np.mgrid[0:19,0:19,0:19]  #grid for the PSF is 19x19x19 voxels.  SHould be made as small as possible without clipping PSF.

    x=x*px
    y=y*px
    z=z*px
    
    x=x-np.mean(x)
    y=y-np.mean(y)
    z=z-np.mean(z)

    r=np.sqrt(x**2 + y**2 + z**2)
    
    
    fwhm_z=0.190 #measured fwhm along vial axis
    fwhm_xy=0.120 # measured fwhm perpendicular to vial axis
    sigm_z=fwhm_z/2.355
    sigm_xy=fwhm_xy/2.355
    
    dkeropt=np.exp(-(x**2)/(2*sigm_z**2)-(y**2)/(2*sigm_xy**2)-(z**2)/(2*sigm_xy**2)) # Optical part of the combined PSF
    dkeropt=dkeropt/np.sum(dkeropt) #Normalization
    
    dker=(1/((4*np.pi*D*(dt/2))**1.5))*np.exp(-(r**2)/(4*D*(dt/2)))*px*px*px # Diffusion part of the combined PSF
    dker=dker/np.sum(dker)
    
    # Summing diffusion kernels for each rotation of the vial
    for n in range(1,ntot):
        t=dt*n
        ddker=(1/((4*np.pi*D*t)**1.5))*np.exp(-(r**2)/(4*D*t))*px*px*px  # Diffusion kernel due to rotation n
        ddker=ddker/np.sum(ddker)
        dker=dker+ddker
    
    dker=dker/np.sum(dker) # Normalization
    dker2=convolve(dker,dkeropt,mode='same') # Convolution of the diffusion and optical parts of the combined PSF
    dker2=dker2/np.sum(dker2) # Normalization
    
    return dker2 #return the combined diffusion-optical PSF

# dker = combined diffusion-optical PSF calculated by blur_ker function above
# I = input target geometry (3D voxel array)
# n = number of iterations

def correct_blurring(dker, I, n):
    
    
    npad=int(1+np.shape(dker)[0])
    I=np.pad(I,(npad,),mode='constant')
    I=I+0.0001 # add small background to prevent division by 0
    I=I/np.max(I)
    In=copy.copy(I)
    
    #Modified Richardson Lucy deconvolution
    for i in range(n):
        
        In=In*convolve(I/convolve(In,dker,mode='same'),dker,mode='same') #Normal RL iteration
        In=In*I/convolve(In,dker,mode='same') #RL iteration modification to help equalize applied dose
    
    return In[npad:-npad,npad:-npad,npad:-npad] #corrected target dose 3D voxel array


if __name__ == "__main__":
    ### Example use###

    #Parameters 
    px=0.05  #Pixel size
    Dcoeff=0.000151 #Diffusion coefficient (resin dependent)
    print_time=60 #Estimated print time
    print_spd=36 #Vial rotation speed in deg/s

    original_target=np.load('gyroidcylinder.ply') #Original target volume.  3D voxel array; 1=solid, 0=no curing.

    visaxis = 0 #axis over which to visualize result (for display purposes only)

    ######

    print('Calculating diffusion kernel')
    dker=blur_ker(px,Dcoeff,print_time,print_spd)  #Calculation of combined diffusion-optical PSF (kernel)
    print('Calculating initial blurred dose')
    blurred_dose_uncorrected=convolve(original_target,dker,mode='same')
    print('Calculating corrected target dose')
    corrected_target=correct_blurring(dker,original_target,3)  #Modified RL dconvolution for n=3 iterations
    print('Calculating diffused corrected target dose')
    blurred_dose_corrected=convolve(corrected_target,dker,mode='same')


    ##Visualization


    plt.figure()
    plt.imshow(np.sum(original_target,axis=visaxis))  #Sum projection of original target dose
    plt.title('Sum projection of original target dose')
    plt.figure()
    plt.imshow(np.sum(corrected_target,axis=visaxis))  #Sum projection of corrected target dose
    plt.title('Sum projection of corrected target dose')
    plt.figure()
    plt.imshow(np.sum(blurred_dose_uncorrected,axis=visaxis)) #Sum projection of blurred original target dose
    plt.title('Sum projection of blurred original target dose')
    plt.figure()
    plt.imshow(np.sum(blurred_dose_corrected,axis=visaxis)) #Sum projection of blurred corrected target dose
    plt.title('Sum projection of blurred corrected target dose')


        
        
    
 