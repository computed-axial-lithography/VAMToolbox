import functools
from logging import warning
import dill
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy import interpolate

import vamtoolbox


def defaultKwargs(**default_kwargs):
    def actualDecorator(fn):
        @functools.wraps(fn)
        def g(*args, **kwargs):
            default_kwargs.update(kwargs)
            return fn(*args, **defaultKwargs)
        return g
    return actualDecorator

class ProjectionGeometry:
    def __init__(self, angles,ray_type, CUDA=False, **kwargs):
        """
        Parameters:
        ----------
        angles : np.ndarray
            vector of angles at which to forward/backward project

        ray_type : str
            ray type of projection geometry e.g. "parallel","cone"

        CUDA : boolean, optional
            activates CUDA-GPU accelerated projectors

        projector_pixel_size : float, optional
            size of a pixel in the focal plane (cm)

        absorption_coeff : float, optional
            photopolymer absorption coeff (cm^-1)

        container_radius : float, optional
            photopolymer container radius, MUST BE SPECIFIED ALONG WITH absorption_coeff (cm)

        attenuation_field : np.ndarray, optional

        occlusion : np.ndarray, optional

        inclination_angle : float, optional
            laminography configuration angle above the plane of normal tomography configuration
       
        """
        
        self.angles = angles
        self.n_angles = np.size(self.angles)
        self.ray_type = ray_type
        self.CUDA = CUDA
        self.projector_pixel_size = None if 'projector_pixel_size' not in kwargs else kwargs['projector_pixel_size']
        self.absorption_coeff = None if 'absorption_coeff' not in kwargs else kwargs['absorption_coeff']
        self.container_radius = None if 'container_radius' not in kwargs else kwargs['container_radius']
        self.attenuation_field = None if 'attenuation_field' not in kwargs else kwargs['attenuation_field']
        self.occlusion = None if 'occlusion' not in kwargs else kwargs['occlusion']
        self.inclination_angle = None if 'inclination_angle' not in kwargs else kwargs['inclination_angle']
        self.zero_dose_sino = None if 'zero_dose_sino' not in kwargs else kwargs['zero_dose_sino']

    def calcZeroDoseSinogram(self,A,target_geo):
        b = A.forward(target_geo.zero_dose)
        self.zero_dose_sino = np.where(b!=0, True, False)


    def calcAbsorptionMask(self,target_geo):
        if self.container_radius is None or self.projector_pixel_size is None:
            raise Exception("container_radius and projector_pixel_size must be specified in ProjectorGeometry if absoption_coeff is used to calculate an absorption mask.")

        x = target_geo.array

        # r is reconstruction grid radius
        # R is container radius

        r = x.shape[0]/2*self.projector_pixel_size
        R = self.container_radius

        if R < r:
            raise Exception("container radius is smaller than the simulation radius. container radius must be larger than simulation radius for valid reconstruction.")

        circle_y, circle_x = np.meshgrid(np.linspace(-r,r,x.shape[0]),
                                        np.linspace(-r,r,x.shape[1]))
        
        # this is not mathematically valid but it works sufficiently until a projector kernel with ray-based absorption can be written
        z = R - np.sqrt(circle_x**2 + circle_y**2)
           
        self.absorption_mask = np.exp(-self.absorption_coeff*z)
        self.absorption_mask[z < R-r] = 0 
        
        if x.ndim == 3:
            self.absorption_mask = np.broadcast_to(self.absorption_mask[...,np.newaxis],x.shape)

class Volume:
    def __init__(self,array : np.ndarray, proj_geo : ProjectionGeometry = None,options = None,**kwargs):
        
        self.array = array
        self.proj_geo = proj_geo
        self.file_extension = None if 'file_extension' not in kwargs else kwargs['file_extension']
        self.vol_type = None if 'vol_type' not in kwargs else kwargs['vol_type']

        self.n_dim = self.array.ndim
        if self.vol_type == 'recon' or self.vol_type == 'target':
            if self.n_dim == 2:
                self.nY, self.nX = self.array.shape
                self.nZ = 0
                self.resolution = None
            elif self.n_dim == 3:
                self.nY, self.nX, self.nZ = self.array.shape
                self.resolution = self.nZ

        elif self.vol_type == 'sino':
            if self.n_dim == 2:
                self.nR, self.nTheta = self.array.shape
                self.nZ = 0
                self.resolution = None
            elif self.n_dim == 3:
                self.nR, self.nTheta, self.nZ = self.array.shape
                self.resolution = self.nZ

    def segmentZ(self,slices):
        """
        Segment volume object by chosen z slices. Modifies the array attribute of the volume object.

        Parameters
        ----------
        slices : int or list
            index of the slice or slices to keep

        Examples
        --------
        Keep z slices between and including 1 and 10

        >>> vol.segmentZ([1,10])

        Keep single z slice at index 10, converts volume object to 2D

        >>> vol.segmentZ(10)
        """
        
        if isinstance(slices,int) or (isinstance(slices,list) and len(slices)==1):
            self.array = self.array[:,:,slices]
            self.n_dim = 2
            self.nZ = 0
            self.resolution = None

        if isinstance(slices,list):
            self.array = self.array[:,:,slices[0]:slices[1]]
            self.nZ = slices[1] - slices[0] + 1
            self.resolution = self.nZ

    def save(self,name : str):
        """Save geometry object"""
        file = open(name + self.file_extension,'wb')
        dill.dump(self,file)
        file.close()

    def show(self,savepath=None,dpi='figure',transparent=False,**kwargs):
        """
        Parameters
        ----------
        savepath : str, optional

        dpi : int, optional
            image dots per inch from `matplotlib.pyplot.savefig <https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.savefig.html>`_
        
        transparent : bool, optional
            sets transparency of the axes patch `matplotlib.pyplot.savefig <https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.savefig.html>`_

        **kwargs
            accepts `matplotlib.pyplot.imshow <https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.imshow.html>`_ keyword arguments
        """
        kwargs['cmap'] = 'CMRmap' if 'cmap' not in kwargs else kwargs['cmap']
        kwargs['interpolation'] = 'antialiased' if 'interpolation' not in kwargs else kwargs['interpolation']
        if self.n_dim == 2:
            vamtoolbox.display.SlicePlot(self.array,self.vol_type,**kwargs)

        elif self.n_dim == 3:
            # must keep instance of slicer for mouse wheel scrolling to work
            self.viewer = vamtoolbox.display.VolumeSlicer(self.array,self.vol_type,**kwargs)
        
        if savepath is not None:
            plt.savefig(savepath,dpi=dpi,transparent=transparent)

        plt.show()

class TargetGeometry(Volume):
    def __init__(self,target=None,stlfilename=None,resolution=None,imagefilename=None,pixels=None,rot_angles=[0,0,0],bodies='all',options=None):
        """
        Parameters
        ----------

        target : np.ndarray or str

        resolution : int, optional

        stlfilename : str, optional

        rot_angles : np.ndarray

        bodies : str or dict

        Examples
        --------
        Raw matrix target

        >>> t = TargetGeometry(target=np.ones((3,3,3)))

        Image (binary e.g. white and black) file target

        >>> t = TargetGeometry(imagefilename="example.png",pixels=300)

        STL file target to voxelize

        >>> t = TargetGeometry(stlfilename="example.stl",resolution=100,rot_angles=[90,0,0])

        """
        self.insert = None
        if target is not None:
            array = target

        # image as target
        elif imagefilename is not None and stlfilename is None:
            # open and convert image to grayscale (single channel 2D matrix)
            image = Image.open(imagefilename).convert('L')

            if image.size[0] != image.size[1]:
                # pad non-square image into square image
                sq_size = np.max(image.size)
                delta_w = sq_size - image.size[0]
                delta_h = sq_size - image.size[1]
                padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
                image = ImageOps.expand(image, padding)

            # resize to requested size
            if pixels is not None:
                image = image.resize(size=(pixels,pixels))
            image = np.array(image).astype(np.float)
            # normalize image to 0-1 range
            image = image/np.max(image)
            # binarize image
            array = np.where(image>=0.5,1.0,0.0)

            if bodies != "all":
                print("Warning: zero dose and insert bodies are not implemented in 2D yet.")
                self.zero_dose = None
                self.insert = None
            else:
                self.zero_dose = None
                self.insert = None


        # stl file as target to voxelized
        elif stlfilename is not None:
            self.stlfilename = stlfilename
            array, insert, zero_dose = vamtoolbox.voxelize.voxelizeTarget(stlfilename, resolution, bodies, rot_angles)
            self.zero_dose = zero_dose
            self.insert = insert

        
        self.gel_inds, self.void_inds = getInds(array)
        super().__init__(array=array,
        options=options,
        file_extension=".target",
        vol_type="target"
        )

    def segmentZ(self,slices):
        """
        Segment target geometry by chosen z slices. Modifies the array and insert attributes of the target geometry.

        Parameters
        ----------
        slices : int or list
            index of the slice or slices to keep

        Examples
        --------
        Keep z slices between and including 1 and 10

        >>> target_geo.segmentZ([1,10])

        Keep single z slice at index 10, converts target_geo to 2D
        
        >>> target_geo.segmentZ(10)
        """
        if isinstance(slices,int) or (isinstance(slices,list) and len(slices)==1):
            self.array = self.array[:,:,slices]
            if self.insert is not None:
                self.insert = self.insert[:,:,slices]
            self.n_dim = 2
            self.nZ = 0
            self.resolution = None

        if isinstance(slices,list):
            self.array = self.array[:,:,slices[0]:slices[1]]
            if self.insert is not None:
                self.insert = self.insert[:,:,slices[0]:slices[1]]
            self.nZ = slices[1] - slices[0] + 1
            self.resolution = self.nZ


    def show(self,show_bodies=False,savepath=None,dpi='figure',transparent=False,**kwargs):
        kwargs['cmap'] = 'gray' if 'cmap' not in kwargs else kwargs['cmap']
        kwargs['interpolation'] = 'none' if 'interpolation' not in kwargs else kwargs['interpolation']


        if self.n_dim == 2:
            if show_bodies == True:
                vamtoolbox.display.SlicePlot(self.array,self.vol_type,show_bodies=True,**kwargs)
            else:
                vamtoolbox.display.SlicePlot(self,self.vol_type,**kwargs)

        elif self.n_dim == 3:
            if show_bodies == True:
                # must keep instance of slicer for mouse wheel scrolling to work
                self.viewer = vamtoolbox.display.VolumeSlicer(self,self.vol_type,show_bodies=True,**kwargs)
            else:
                # must keep instance of slicer for mouse wheel scrolling to work
                self.viewer = vamtoolbox.display.VolumeSlicer(self.array,self.vol_type,**kwargs)
            
        if savepath is not None:
            plt.savefig(savepath,dpi=dpi,transparent=transparent)
        plt.show()

class Sinogram(Volume):
    def __init__(self,sinogram : np.ndarray, proj_geo : ProjectionGeometry, options=None):
        """
        Parameters
        ----------
        sinogram : np.ndarray

        proj_geo : geometry.ProjectionGeometry

        options : dict, optional

        """
        super().__init__(array=sinogram,
        proj_geo=proj_geo,
        options=options,
        file_extension=".sino",
        vol_type="sino"
        )

class Reconstruction(Volume):
    def __init__(self,reconstruction : np.ndarray, proj_geo : ProjectionGeometry, options=None):
        """
        Parameters
        ----------
        reconstruction : np.ndarray

        proj_geo : geometry.ProjectionGeometry

        options : dict, optional
        
        """
        super().__init__(array=reconstruction,
        proj_geo=proj_geo,
        options=options,
        file_extension=".recon",
        vol_type="recon"
        )



def loadVolume(file_name : str):
    """
    Load saved vamtoolbox.geometry.Volume object

    Parameters
    ----------
    file_name : str
        filepath to Volume object e.g. "C:\\\\A\\\\sinogram.sino"

    Returns
    -------
    vamtoolbox.geometry.Volume
    """
    file = open(file_name,'rb')
    data_pickle = file.read()
    file.close()
    A = dill.loads(data_pickle)
    return A

def getCircleMask(target : np.ndarray):
    """
    Generates a boolean mask of the inscribed circle of a square array

    Parameters
    ----------
    target : np.ndarray
        square array to create a boolean mask

    Returns
    -------
    circle_mask
        boolean mask where inscribed circle is True, outside the circle is False
    """
    # Define void and gel indices for error and thresholding operations
    if np.ndim(target) == 2:
        circle_y, circle_x = np.meshgrid(np.linspace(-1,1,target.shape[0]),
                                        np.linspace(-1,1,target.shape[1]))

    else:
        circle_y, circle_x, _ = np.meshgrid(np.linspace(-1,1,target.shape[0]),
                                            np.linspace(-1,1,target.shape[1]),
                                            np.linspace(-1,1,target.shape[2]))
    R = (circle_x ** 2 + circle_y ** 2)
    circle_mask = np.array(R <= 1 ** 2, dtype=bool)

    return circle_mask

def getInds(target : np.ndarray):
    """
    Gets gel and void indices of the boolean target array

    Parameters
    ----------
    target : np.ndarray
        binary target array

    Returns
    -------
    gel_inds, void_inds : np.ndarray
        boolean arrays where the target is 1 (gel_inds) and where the target is 0 (void_inds)
    """
    circle_mask = getCircleMask(target)

    gel_inds = np.logical_and(target>0,circle_mask)
    void_inds = np.logical_and(target==0,circle_mask)

    return gel_inds, void_inds

def rebinFanBeam(sinogram,vial_width,N_screen,n_write,throw_ratio):
    """
    Rebins a parallel ray projection geometry (telecentric) to a converging fan beam projection geometry that can be used when the photopolymer vial is NOT indexed matched to its surrounding, i.e. when the projector light is directly incident on the outer wall of the vial at an air-vial interface. 

    Parameters
    ----------
    sinogram : geometry.Sinogram
        sinogram generated for parallel ray geometry that is to be rebinned for use in a non telecentric VAM geometry

    vial_width : int
        Apparent vial width in projector pixels.  Apparent vial width is only equal to true vial width if projection is telecentric.

    N_screen : tuple
        (N_U,N_v), (# of pixels in u-axis, # of pixels in v-axis) of the projected screen

    n_write : float
        refractive index at the wavelength used for writing (projector wavelength)
    
    throw_ratio : float
        Throw ratio of projector

    Returns
    -------
    Rebinned sinogram in geometry.Sinogram object


    Based on code by @author: Antony Orth

    Antony Orth, Kathleen L. Sampson, Kayley Ting, Jonathan Boisvert, and Chantal Paquet, "Correcting ray distortion in tomographic additive manufacturing," Opt. Express 29, 11037-11054 (2021) 

    Please use the above citation if used in your work.

    *Note*
    The resampling process can be thought of as a resampling from the parallel beam case to the non-parallel beam case (ie. virtual projector to physical projector), where refraction alters the tranjectory of the rays in the vial.
    The basic idea is to consider that the physical projector should sample Radon space (virtual projector space) at the appropriate coordinates in the virtual projector space.
    In other words, each pixel on the physical projector at each instant in time, corresponds to a particular position in Radon space.  The correspondence is calculated with the equations for xv and thetav in the paper above.
    The desired object is Radon transformed (corresponding to the virtual projector space) and then resampled in the altered space that is accessible by the physical projector.

    In the paper above, the process is described as a resampling from the physical projector space to the virtual projector space.  However, it makes more sense to think of it the other way around (virtual to physical).  This may be addressed 
    by a correction to the paper above in the near future (as of 18 Jan 2022).
    """



    def rebin(b,xp,angles,x_samp,theta_samp,dxv_dxp,T=None):
        """
        Function that calls the scipy interpolate function that performs the actual resampling

        Parameters
        ----------
        b : np.ndarray
            sinogram for the case of no refraction and telecentric projection (parallel beam case)
        xp : np.ndarray
            Projector pixel coordinates
        angles : np.ndarray
            Vial rotation angles to sample.
        x_sample : np.ndarray
            The ray coordinate that is actually sampled in the vial (tiled version of xv)
        theta_samp : np.ndarray
            the ray angle in the vial (tiled version of thetav)
        dxv_dxp : np.ndarray
            Change in differential area of radon space sampled by the virtual projector compared to the physical projector
        T : np.ndarray, optional
            Fresnel transmission coefficients

        Returns
        -------
        b_rebinned : np.ndarray
            Resampled sinogram, including corrections for non-uniform fresnel transmission and change in differential area
        """

        b_rebinned=interpolate.interpn((xp,angles), b, (x_samp,theta_samp), method='linear', bounds_error=False, fill_value=0) #resampling happens here
        b_rebinned=b_rebinned*dxv_dxp #Correcting for change in differential area in radon space.  Very small correction, could probably be ignored in most cases.

        if T is not None:
            T_inv=1/T
            #Correction for Fresnel transmission loss
            b_rebinned = T_inv * b_rebinned
            
        return b_rebinned


    
    angles = sinogram.proj_geo.angles
    N_angles = angles.size
    N_z, N_r = sinogram.array.shape[2], sinogram.array.shape[0]
    N_U,N_V = N_screen
    n1=1 #refractive index of air
    n2=n_write # measured refractive index at the projection beam wavelength
    vial_width=np.int(vial_width) # Apparent vial width in the field of view of the projector.  Obtained by projecting projector columns and noting the first and last columns to intersect the vial vall.
    
    throw_ratio_pix=throw_ratio*N_U #Throw ratio x number of pixels in the horizontal direction.  Change this depending on projector width in pixels.
    
    Rv=(vial_width/2)*np.sqrt(1+((vial_width)/(2*throw_ratio_pix))**2) # Actual radius of vial in units of pixels
    
    xp=np.linspace(-vial_width/2,vial_width/2,vial_width) #Projector x-coordinate
    phi=np.arctan(xp/throw_ratio_pix) # Extra divergence angle caused by non-telecentricity of projector across the projector's field of view
    
    xps=(xp-xp*np.sqrt(1-(1+(xp/throw_ratio_pix)**2)*(1-(Rv/throw_ratio_pix)**2)))/(1+(xp/throw_ratio_pix)**2) # Location at which a ray from projector pixel xp intersects the vial
    
    theta10=np.arcsin(xps/Rv) #Angle that the normal vector of the vial makes with the optical axis at xps
    thetai=np.arcsin(xps/Rv)+phi # angle of incidence of a light ray from projector pixel xp, incident at on the vial at xps
    thetat=np.arcsin((n1/n2)*np.sin(thetai)) # angle of transmission after refraction at air/vial interface
    thetav=theta10-thetat # Deviation from optical axis of transmitted ray after refraction 
    thetavD=(180/np.pi)*thetav # As above, expressed in degrees
    
    #Fresnel coefficients
    Ts=1-((n1*np.cos(thetai)-n2*np.cos(thetat))/((n1*np.cos(thetai)+n2*np.cos(thetat))))**2
    Tp=1-((n1*np.cos(thetat)-n2*np.cos(thetai))/((n1*np.cos(thetat)+n2*np.cos(thetai))))**2
    T=(Ts+Tp)/2 #averaging for s- and p-polarized light
    T[T<0]=0  #Just in case
    T_b = np.transpose(np.tile(T,(N_angles,1)))
    
    
    # Calculating change in differential area element due to variable change
    xv=(xps*np.cos(thetav))-(np.sin(thetav)*np.sqrt((Rv**2)-(xps**2)))
    dxv_dxp=np.gradient(xv)
    dxv_dxp[T<0]=0 #just in case of a pathological situation
    dxv_dxp[dxv_dxp<0]=0 #ignore pixels where the sign of the differential area flips
    dxv_dxp_tiled=np.transpose(np.tile(dxv_dxp,(N_angles,1)))   
    
    #Constructing the arrays (theta_samp = thetav) that contains the angles and ray coordinates (x_samp) at which the build volume is sampled by the projector.
    theta_samp, x_samp, = np.meshgrid(angles, xv)
    thetaDelt=np.transpose(np.tile(thetavD,(N_angles,1)))
    theta_samp=theta_samp+thetaDelt  #Remember, each ray in the vial is rotated by thetaDelt with respect to the optical axis
    
    #This part deals with angles wrapping around past 360 degrees
    min_theta, max_theta = sinogram.proj_geo.angles[0], sinogram.proj_geo.angles[-1]
    diff_theta = abs(sinogram.proj_geo.angles[1] - sinogram.proj_geo.angles[0])
    theta_samp[theta_samp>max_theta]=theta_samp[theta_samp>max_theta]-360
    theta_samp[theta_samp<min_theta]=360+theta_samp[theta_samp<min_theta]
    theta_samp[theta_samp>max_theta-diff_theta]=min_theta
    theta_samp[(theta_samp<=max_theta-diff_theta) & (theta_samp>max_theta)]=max_theta
    
    #Padding the frames array so that its width is the same as the vial
    pd=np.int((vial_width-N_r)/2)
    if pd>0:
        sinogram.array=np.pad(sinogram.array,((pd,pd),(0,0),(0,0)),mode='constant')
    if np.shape(sinogram.array)[0] < vial_width:
        sinogram.array=np.pad(sinogram.array,((0,1),(0,0),(0,0)),mode='constant')
    
    sinogram_rs=np.zeros_like(sinogram.array) #Initializing array that will contain the resampled projections
    
    #Calculating the resampled projections for each slice of the object (frames) in a loop.
    for z_i in range(N_z):
        sinogram_rs[...,z_i]=rebin(sinogram.array[...,z_i],xp,angles,x_samp,theta_samp,dxv_dxp_tiled,T_b) 
        
    return Sinogram(sinogram_rs,sinogram.proj_geo)



