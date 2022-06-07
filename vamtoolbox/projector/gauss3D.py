import cupy as cp
import cupyx.scipy.ndimage
# cp.cuda.set_allocator(None)
# cp.cuda.set_pinned_memory_allocator(None)
import numpy as np
import scipy.ndimage
import pyfftw
# try:
#     import cupy.fft as fft_method
# except:
#     import mkl_fft as fft_method

from mkl_fft import _numpy_fft
import mkl_fft
import matplotlib.pyplot as plt
import  matplotlib.colors
import display_functions as disp
from time import perf_counter
"""
GPU accelerated fft2
https://www.idtools.com.au/gpu-accelerated-fft-compatible-numpy/
"""



"""
EXAMPLE OF PARAMETER DEFINITIONS

# optical params only have an effect if used with P_type = 'gauss'
optical_params = {
    'NA':                   0.1,    # numerical aperture
    'n':                    1.48,   # refractive index of monomer
    'focal_offset':         30e-6,      # focal plane offset from axis of rotation (m)
    'alpha':                0,      # absorption coefficient (1/m); NOTE still deciding how absorption should be incorporated for Gauss beam (in Gauss kernel or in backprojector)
    'w0':                   3e-6,  # beam waist radius (m)
    'voxel_size':           1e-6   # physical dimension of voxel (m); NOTE none of the other projectors have physical units yet, still working on it
}

optical_params = {
    'wavelength': 405e-9,
    'NA':  0.2,
    'n':    1.48,
    'DMD_pixel_size': 10.6e-6,
    'magnification': 45.0/125.0,
    'focal_offset': 0e-6,
    'alpha': 0,
    'voxel_size': voxel_size
}
"""



class gauss3D():
    def __init__(self,target,proj_params,optical_params):
        """
        Parameters
        ----------
        target : ndarray
            target array or array with equivalent size
        proj_params : dict
            parameters defining projector configuration
        optical_params : dict
            optical parameters for defining the Gaussian beam
        """
        self.NA = optical_params['NA']
        self.n = optical_params['n']
        self.offset = optical_params['focal_offset']
        self.alpha = optical_params['alpha']
        self.w_pixel = np.sqrt(2/np.pi)*optical_params['DMD_pixel_size']
        self.w_psf = 1.62*optical_params['wavelength']*optical_params['n']/(np.pi*optical_params['NA']*np.sqrt(2*np.log(2)))
        self.w0 = np.sqrt(self.w_psf**2 + (self.w_pixel**2) * (optical_params['magnification']**2))
        self.voxel_size = optical_params['voxel_size']
        self.target = target
        self.angles = proj_params['angles']
        self.angles_rad = np.deg2rad(self.angles)


        self.nProp, self.nT, self.nZ = self.target.shape
        self.mempool = cp.get_default_memory_pool()

        self.kernel, self.FFT_kernel = self.createGaussKernel()
        
            
        
    def createGaussKernel(self):
        """
        Builds Gaussian kernel and Fourier-transformed kernel according 
        to the optical parameters specified in optical_params

        Returns
        -------
        kernel : ndarray
            single pixel Gaussian beam approximation with same size as 
            target
        FFT_kernel : ndarray
            kernel Fourier-transformed along dimensions orthogonal to 
            propagation
        """
        


        t = np.linspace(-self.nT*self.voxel_size/2,
                        self.nT*self.voxel_size/2,
                        self.nT)
        prop = np.linspace(-self.nProp*self.voxel_size/2,
                            self.nProp*self.voxel_size/2,
                            self.nProp)
        z = np.linspace(-self.nZ*self.voxel_size/2,
                        self.nZ*self.voxel_size/2,
                        self.nZ)

        Prop, T, Z = np.meshgrid(prop,t,z,indexing='ij')
        
        Prop = Prop - self.offset
        R = np.sqrt(T**2 + Z**2)
        zr = self.n*self.w0/self.NA
        # TODO need to decide how to incorporate absorption
        # should it be in the kernel? Then we have to assume spatial invariance which
        # is incorrect for cylindrical vial
        # should it be element-wise multiplication in the backprojector? 
        
        wz = self.w0*np.sqrt(1+(Prop/zr)**2)
        I0 = 2/(np.pi*self.w0**2)
        kernel = I0 * (self.w0/wz)**2 * np.exp(-2*R**2/wz**2) * np.exp(-Prop*self.alpha)


        # kernel = kernel/kernel.sum(axis=1)[:,:,None]
        max_ind = np.argmax(kernel.sum(axis=(1,2)))
        kernel = cp.asarray(kernel/np.sum(kernel[max_ind,:,:]))
        # disp.view_slices(kernel,2,'kerentl')
        # kernel[kernel < 1/np.exp(3)] = 0
        FFT_kernel = cp.fft.fftn(kernel,axes=(1,2))
        self.mempool.free_all_blocks()


        return kernel, FFT_kernel

    def setupInterpCoords(self,dimension,angle=None):

        x = cp.linspace(-self.nT/2,self.nT/2,self.nT,dtype=cp.float16)
        y = cp.linspace(-self.nProp/2,self.nProp/2,self.nProp,dtype=cp.float16)
        z = cp.linspace(-self.nZ/2,self.nZ/2,self.nZ,dtype=cp.float16)

        YY, XX, ZZ = cp.meshgrid(y, x, z, indexing='ij')
        limits = [(-self.nProp/2, self.nProp/2), (-self.nT/2, self.nT/2), (-self.nZ/2, self.nZ/2)]

        
        t = YY*cp.sin(angle) - XX*cp.cos(angle)
        prop = YY*cp.cos(angle) + XX*cp.sin(angle)

        propi_coor = cp.ravel((self.nProp - 1) * (prop - limits[0][0])/(limits[0][1] - limits[0][0]))
        ti_coor = cp.ravel((self.nT - 1) * (t - limits[1][0])/(limits[1][1] - limits[1][0]))
        ZZi_coor = cp.ravel((self.nZ - 1) * (ZZ - limits[2][0])/(limits[2][1] - limits[2][0]))
        coords_ret = cp.vstack((propi_coor,ti_coor,ZZi_coor))

        propi_coor = None
        ti_coor = None
        ZZi_coor = None

        return cp.asarray(coords_ret)

    def convolveWithGaussKernelFP(self,target):

        FFT_target = cp.fft.fft2(target,axes=(1,2))

        convolved_target = cp.real(cp.fft.ifftshift(cp.fft.ifft2(cp.multiply(FFT_target,self.FFT_kernel),axes=(1,2))))

        return cp.asarray(convolved_target)

    def convolveWithGaussKernelBP(self,projection):
        """
        Convolves a 1D or 2D projection with a 1D or 2D Gauss kernel defined at each 
        propagation distance to generate the backprojection for the angle
        of the projection.
            1D
            f*_theta(x,y) = fft(P_theta(t)) * fft(G(t,prop))

            2D
            f*_theta(x,y,z) = fft(P_theta(t,z)) * fft(G(t,prop,z))

        Parameters
        ----------
        projection : ndarray
            1D projection in t (lateral) or 2D projection in t (lateral)
            and z (height) axes at a single projection angle

        Returns
        -------
        convolved_backproj : ndarray
            convolved backprojection in spatial domain for the same angle
            of the input projection
        """

        # Fourier transform projection
        FFT_projection = cp.fft.fft2(projection,axes=(1,2))
        
        convolved_backproj = cp.real(cp.fft.ifftshift(cp.fft.ifft2(cp.multiply(FFT_projection[cp.newaxis,:,:],self.FFT_kernel)), axes=(1,2)))
        
        return convolved_backproj
    


    def gaussFP(self,target):
        targetcp = cp.asarray(target)
        projections = cp.zeros((self.nT,len(self.angles_rad),self.nZ),dtype=cp.float32)
    
        
        for curr_angle_ind in range(len(self.angles_rad)):
            time1 = perf_counter()  

            coordscp = self.setupInterpCoords(3,self.angles_rad[curr_angle_ind])

            # print('coords time: %f' % (perf_counter() - time1))  
            # interpolated = scipy.ndimage.map_coordinates(curr_backproj,coords,order=1,mode='constant',cval=0)
            # reconstruction += np.reshape(interpolated,reconstruction.shape)
            time2 = perf_counter()
            interpolated = cupyx.scipy.ndimage.map_coordinates(targetcp,coordscp,order=1,mode='constant',cval=0)
            # print('interp time: %f' % (perf_counter() - time2))

            rotated = cp.reshape(interpolated,target.shape)
            time3 = perf_counter()
            rotated = self.convolveWithGaussKernelFP(rotated)
            # print('conv time: %f' % (perf_counter() - time3))

            projections[:, curr_angle_ind, :] = cp.sum(rotated,axis=0)
            # print(curr_angle_ind)

        projections = cp.asnumpy(projections)

        return projections

    def gaussBP(self,projections):

        reconstruction = cp.zeros(self.target.shape,dtype=cp.float32)
        projections = cp.array(projections,dtype=cp.float32)

        for curr_angle_ind in range(len(self.angles_rad)):
            # convolve 2D projection with each slice (in propagation axis) 
            # of precomputed Gaussian kernel
            curr_backproj = self.convolveWithGaussKernelBP(projections[:,curr_angle_ind])

            coords = self.setupInterpCoords(3,self.angles_rad[curr_angle_ind])
            # interpolated = scipy.ndimage.map_coordinates(curr_backproj,coords,order=1,mode='constant',cval=0)
            # reconstruction += np.reshape(interpolated,reconstruction.shape)
            interpolated = cupyx.scipy.ndimage.map_coordinates(curr_backproj,coords,order=1,mode='constant',cval=0)
            reconstruction += cp.reshape(interpolated,reconstruction.shape)

        reconstruction = cp.asnumpy(reconstruction)
        
        # disp.view_slices(reconstruction,2,'recon')              


        return reconstruction * np.pi / (2* len(self.angles))
    
    def forwardProject(self,target):
        if np.any(target<0):
            neg_target = -np.clip(target,a_min=None,a_max=0)
            pos_target = np.clip(target,a_min=0,a_max=None)               
            neg_projections = self.gaussFP(neg_target)
            pos_projections = self.gaussFP(pos_target)
            
            projections = pos_projections + (-neg_projections)

        else:
            projections = self.gaussFP(target)

        return projections

    def backProject(self,projections):
        reconstruction = self.gaussBP(projections)

        return reconstruction
 