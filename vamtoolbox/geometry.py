import functools
from logging import warning
from typing import Literal, TypeAlias

import dill  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from PIL import Image, ImageOps
from scipy import interpolate, sparse

import vamtoolbox

try:
    from joblib import Parallel as _Parallel, delayed as _jdelayed, effective_n_jobs as _eff_n_jobs
except ImportError:
    _Parallel = None

# Worker count for the parallel fan-beam rebin.  -1 = all cores (fastest, but pegs
# the whole CPU for ~tens of seconds on big sinograms); set to a positive int to cap
# the spike, or 1 to force the serial path.  Override via env VAM_REBIN_JOBS or by
# setting vamtoolbox.geometry.REBIN_N_JOBS at runtime.
import os as _os
REBIN_N_JOBS = int(_os.environ.get("VAM_REBIN_JOBS", "-1"))


def _rebin_chunk(b_chunk, xp, angles, x_samp, theta_samp, dxv_dxp, T_inv):
    """Worker for parallel rebinFanBeam: resample a contiguous block of z-slices.
    b_chunk: (N_r, N_angles, k) -> (N_r, N_angles, k).  Each z-slice is independent.
    """
    out = np.empty_like(b_chunk)
    for i in range(b_chunk.shape[2]):
        rb = interpolate.interpn(
            (xp, angles), b_chunk[:, :, i], (x_samp, theta_samp),
            method="linear", bounds_error=False, fill_value=0,
        )
        out[:, :, i] = T_inv * (rb * dxv_dxp)
    return out


def defaultKwargs(**default_kwargs):
    def actualDecorator(fn):
        @functools.wraps(fn)
        def g(*args, **kwargs):
            default_kwargs.update(kwargs)
            return fn(*args, **defaultKwargs)

        return g

    return actualDecorator


RayType: TypeAlias = Literal["parallel", "cone", "algebraic", "ray_trace"]


class ProjectionGeometry:
    def __init__(
        self, angles: np.ndarray, ray_type: RayType, CUDA: bool | None = False, **kwargs
    ):
        """
        Parameters:
        ----------
        angles : np.ndarray
            vector of angles at which to forward/backward project

        ray_type : str
            ray type of projection geometry e.g. "parallel", "cone", "algebraic", "ray_trace"

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

        index_model : class IndexModel, optional, only used when ray tracing is enabled
            This object provide analytical or interpolational description of real part of refractive index of the simulation volume in ray tracing.
            index_model is configured prior to initialization of proj_geo, using class vamtoolbox.medium.IndexModel.

        attenuation_model : class AttenuationModel, optional, only used when ray tracing is enabled
            This object provide analytical or interpolational description of total attenuation coefficient of the simulation volume in ray tracing.
            attenuation_model is confifgured prior to initialization of proj_geo, using class vamtoolbox.medium.AttenuationModel.

        absorption_model : class AbsorptionModel, optional, only used when ray tracing is enabled.
            This object provide analytical or interpolational description of the absorption coefficient of the photochemically active component (e.g. photointiatior) of simulation volume in ray tracing.
            absorption_model is confifgured prior to initialization of proj_geo, using class vamtoolbox.medium.AbsorptionModel (alias of AttenuationModel).

        ray_trace_method : str, Required for ray tracing propagation (when ray_type == 'ray_trace')
            'eikonal', 'snells', 'hybrid'. Default: 'eikonal'

        eikonal_parametrization : str, Required for when (ray_trace_method == 'eikonal') or (ray_trace_method == 'hybrid')
            'canonical', 'physical_path_length', 'optical_path_length'

        ray_trace_ode_solver : str, Required for ray tracing propagation (when ray_type == 'ray_trace')
            'forward_symplectic_euler', 'forward_euler', 'leapfrog', 'rk4'

        ray_trace_ray_config : str, optional
            'parallel', 'cone', user_defined RayState. Default: 'parallel'
            user_defined RayState is an object storing the initial position and direction of the set of rays.

        loading_path_for_matrix : str, Required for algebraic propagation (when ray_type == 'algebraic')
            For algebraic propagation only.
            File path to algebraic propagation matrix. File type .npz (scipy sparse matrix format)

        """

        self.angles = angles
        self.n_angles = np.size(self.angles)
        self.ray_type = ray_type
        self.CUDA = CUDA
        self.projector_pixel_size = (
            None
            if "projector_pixel_size" not in kwargs
            else kwargs["projector_pixel_size"]
        )
        self.absorption_coeff = (
            None if "absorption_coeff" not in kwargs else kwargs["absorption_coeff"]
        )
        self.container_radius = (
            None if "container_radius" not in kwargs else kwargs["container_radius"]
        )
        self.attenuation_field = (
            None if "attenuation_field" not in kwargs else kwargs["attenuation_field"]
        )
        self.occlusion = None if "occlusion" not in kwargs else kwargs["occlusion"]
        self.inclination_angle = (
            None if "inclination_angle" not in kwargs else kwargs["inclination_angle"]
        )
        self.zero_dose_sino = (
            None if "zero_dose_sino" not in kwargs else kwargs["zero_dose_sino"]
        )
        self.index_model = (
            None if "index_model" not in kwargs else kwargs["index_model"]
        )
        self.attenuation_model = (
            None if "attenuation_model" not in kwargs else kwargs["attenuation_model"]
        )
        self.absorption_model = (
            None if "absorption_model" not in kwargs else kwargs["absorption_model"]
        )
        self.ray_trace_method = (
            None if "ray_trace_method" not in kwargs else kwargs["ray_trace_method"]
        )
        self.eikonal_parametrization = (
            None
            if "eikonal_parametrization" not in kwargs
            else kwargs["eikonal_parametrization"]
        )
        self.ray_trace_ode_solver = (
            None
            if "ray_trace_ode_solver" not in kwargs
            else kwargs["ray_trace_ode_solver"]
        )
        self.ray_trace_ray_config = (
            None
            if "ray_trace_ray_config" not in kwargs
            else kwargs["ray_trace_ray_config"]
        )
        self.tensor_dtype = (
            None if "tensor_dtype" not in kwargs else kwargs["tensor_dtype"]
        )
        self.ray_density = (
            None if "ray_density" not in kwargs else kwargs["ray_density"]
        )
        self.loading_path_for_matrix = (
            True
            if "loading_path_for_matrix" not in kwargs
            else kwargs["loading_path_for_matrix"]
        )

    def calcZeroDoseSinogram(self, A, target_geo):
        b = A.forward(target_geo.zero_dose)
        self.zero_dose_sino = np.where(b != 0, True, False)

    def calcAbsorptionMask(self, target_geo):
        if self.container_radius is None or self.projector_pixel_size is None:
            raise Exception(
                "container_radius and projector_pixel_size must be specified in ProjectorGeometry if absoption_coeff is used to calculate an absorption mask."
            )

        x = target_geo.array

        # r is reconstruction grid radius
        # R is container radius

        r = x.shape[0] / 2 * self.projector_pixel_size
        R = self.container_radius

        if R < r:
            raise Exception(
                "container radius is smaller than the simulation radius. container radius must be larger than simulation radius for valid reconstruction."
            )

        circle_y, circle_x = np.meshgrid(
            np.linspace(-r, r, x.shape[0]), np.linspace(-r, r, x.shape[1])
        )

        # Radial approximation: depth from vial wall = R - radial_distance_from_centre.
        # Exact path length depends on projection angle; this symmetric average is
        # sufficient and is applied consistently to forward and back-projection.
        radial_dist = np.sqrt(circle_x**2 + circle_y**2)
        z = R - radial_dist  # depth from wall (0 at wall, R at centre)

        # float32 keeps the projector forward/backward from upcasting the whole
        # volume to float64 (halves RAM for large-volume OSMO).
        self.absorption_mask = np.exp(-self.absorption_coeff * z).astype(np.float32)
        self.absorption_mask[radial_dist > r] = 0  # zero outside reconstruction circle

        # Print absorption profile so the user can verify the correction is gradual.
        depths_cm = np.array([0.0, R * 0.25, R * 0.5, R * 0.75, R])
        labels = ["wall", "R/4", "R/2", "3R/4", "centre"]
        print(
            f"  Absorption mask profile  (mu = {self.absorption_coeff:.4f} cm^-1, R = {R:.3f} cm)"
        )
        for d, lbl in zip(depths_cm, labels):
            print(
                f"    {lbl:8s} (z={d:.2f} cm) : {100 * np.exp(-self.absorption_coeff * d):.1f}% intensity"
            )

        if x.ndim == 3:
            self.absorption_mask = np.broadcast_to(
                self.absorption_mask[..., np.newaxis], x.shape
            )


class Volume:
    def __init__(
        self,
        array: np.ndarray,
        proj_geo: ProjectionGeometry | None = None,
        **kwargs,
    ):

        self.array = array
        self.proj_geo = proj_geo
        self.file_extension: str | None = (
            None if "file_extension" not in kwargs else kwargs["file_extension"]
        )
        self.vol_type = None if "vol_type" not in kwargs else kwargs["vol_type"]
        self.spatial_sampling_rate = (
            None
            if "spatial_sampling_rate" not in kwargs
            else kwargs["spatial_sampling_rate"]
        )

        self.n_dim = self.array.ndim
        # self.n_dim = len(np.squeeze(self.array).shape) #robust against singleton dimensions
        if self.vol_type == "recon" or self.vol_type == "target":
            if self.n_dim == 2:
                self.nY, self.nX = self.array.shape
                self.nZ = 0
                self.resolution = None
            elif self.n_dim == 3:
                self.nY, self.nX, self.nZ = self.array.shape
                self.resolution = self.nZ

        elif self.vol_type == "sino":
            if self.n_dim == 2:
                self.nR, self.nTheta = self.array.shape
                self.nZ = 0
                self.resolution = None
            elif self.n_dim == 3:
                self.nR, self.nTheta, self.nZ = self.array.shape
                self.resolution = self.nZ

    def segmentZ(self, slices):
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

        if isinstance(slices, int) or (isinstance(slices, list) and len(slices) == 1):
            self.array = self.array[:, :, slices]
            self.n_dim = 2
            self.nZ = 0
            self.resolution = None

        if isinstance(slices, list):
            self.array = self.array[:, :, slices[0] : slices[1]]
            self.nZ = slices[1] - slices[0] + 1
            self.resolution = self.nZ

    def save(self, name: str):
        """Save geometry object"""
        if self.file_extension:
            name = name + self.file_extension

        file = open(name, "wb")
        dill.dump(self, file)
        file.close()

    def show(self, savepath=None, dpi="figure", transparent=False, **kwargs):
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
        kwargs["cmap"] = "CMRmap" if "cmap" not in kwargs else kwargs["cmap"]
        kwargs["interpolation"] = (
            "antialiased" if "interpolation" not in kwargs else kwargs["interpolation"]
        )
        if self.n_dim == 2:
            vamtoolbox.display.SlicePlot(self.array, self.vol_type, **kwargs)

        elif self.n_dim == 3:
            # must keep instance of slicer for mouse wheel scrolling to work
            self.viewer = vamtoolbox.display.VolumeSlicer(
                self.array, self.vol_type, **kwargs
            )

        if savepath is not None:
            plt.savefig(savepath, dpi=dpi, transparent=transparent)

        plt.show()

    def constructCoordGrid(self, spatial_sampling_rate=None, device=None):
        """
        Get coordinate grid centered around the object (target/recon) in physical length unit using spatial_sampling_rate.
        Unit of spatial_samplign_rate is voxel/cm
        """
        if self.vol_type == "sino":
            print(
                "Coordinate system of sinogram is defined by propagator, using both target_geo and proj_geo."
            )
            return None

        ###=============== This part accommodate the cases where sampling rate to be either predefined, supplied, or neither.
        if spatial_sampling_rate is not None:
            self.spatial_sampling_rate = spatial_sampling_rate  # Allow the input to override the original sampling rate

        if (
            self.spatial_sampling_rate is None
        ):  # if the provided and the original are both None, assume sampling rate is 1
            self.spatial_sampling_rate = (
                500  # the assumed 500 voxel/cm correspond to 20 micron per voxel
            )
        ###===============

        # get the coordinate vectors as numpy arrays
        coord_vec_list = self.constructCoordVec(self.spatial_sampling_rate, device=None)

        # Construct grid using meshgrid 'ij' indexing. Order of coordinate axes are x,y,z (instead of y,x,z)
        """ Old implementation. The length of the output list varies.
        if self.n_dim == 2:
            xv = coord_vec_list[0]
            yv = coord_vec_list[1]
            xg, yg = np.meshgrid(xv, yv, indexing = 'ij')
            self.coord_grid_list = [xg, yg]

        elif self.n_dim == 3:
            xv = coord_vec_list[0]
            yv = coord_vec_list[1]
            zv = coord_vec_list[2]
            xg, yg, zg = np.meshgrid(xv, yv, zv, indexing = 'ij')
            self.coord_grid_list = [xg, yg, zg]
        """
        # New implementation. The length of the output list stay constant. The extra vec/grid in 2D case can simply be ignored.
        xg, yg, zg = np.meshgrid(
            coord_vec_list[0], coord_vec_list[1], coord_vec_list[2], indexing="ij"
        )

        # If device is specified, the vectors are provided as tensor. Otherwise, numpy array are provided.
        # Providing tensor directly at this level facilitate data sharing and avoid storing duplicates unnecessarily.
        # if device is not None:
        #     xg = torch.as_tensor(xg, device=device)
        #     yg = torch.as_tensor(yg, device=device)
        #     zg = torch.as_tensor(zg, device=device)

        self.coord_grid_list = [xg, yg, zg]
        return self.coord_grid_list

    def constructCoordVec(self, spatial_sampling_rate=None, device=None):
        """
        Get coordinate vectors centered around the object (target/recon) in physical length unit using spatial_sampling_rate.
        Unit of spatial_samplign_rate is voxel/cm
        """
        if self.vol_type == "sino":
            print(
                "Coordinate system of sinogram is defined by propagator, using both target_geo and proj_geo."
            )
            return None

        ###=============== This part accommodate the cases where sampling rate to be either predefined, supplied, or neither.
        if spatial_sampling_rate is not None:
            self.spatial_sampling_rate = spatial_sampling_rate  # Allow the input to override the original sampling rate

        if (
            self.spatial_sampling_rate is None
        ):  # if the provided and the original are both None, assume sampling rate is 1
            self.spatial_sampling_rate = (
                500  # the assumed 500 voxel/cm correspond to 20 micron per voxel
            )
        ###===============
        """ Old implementation. The length of the output list varies.
        #Construct vectors
        if self.n_dim == 2:
            xv = np.linspace(-(self.nX-1)/(2*self.spatial_sampling_rate), (self.nX-1)/(2*self.spatial_sampling_rate), self.nX)
            yv = np.linspace(-(self.nY-1)/(2*self.spatial_sampling_rate), (self.nY-1)/(2*self.spatial_sampling_rate), self.nY)
            self.coord_vec_list = [xv, yv]

        elif self.n_dim == 3:
            xv = np.linspace(-(self.nX-1)/(2*self.spatial_sampling_rate), (self.nX-1)/(2*self.spatial_sampling_rate), self.nX)
            yv = np.linspace(-(self.nY-1)/(2*self.spatial_sampling_rate), (self.nY-1)/(2*self.spatial_sampling_rate), self.nY)
            zv = np.linspace(-(self.nZ-1)/(2*self.spatial_sampling_rate), (self.nZ-1)/(2*self.spatial_sampling_rate), self.nZ)
            self.coord_vec_list = [xv, yv, zv]
        """
        # New implementation. The length of the output list stay constant. The extra vec/grid in 2D case can simply be ignored.
        xv = np.linspace(
            -(self.nX - 1) / (2 * self.spatial_sampling_rate),
            (self.nX - 1) / (2 * self.spatial_sampling_rate),
            self.nX,
        )
        yv = np.linspace(
            -(self.nY - 1) / (2 * self.spatial_sampling_rate),
            (self.nY - 1) / (2 * self.spatial_sampling_rate),
            self.nY,
        )
        if self.n_dim == 2:
            zv = np.atleast_1d(
                0.0
            )  # we can't use the same expression as in 3D case because self.nZ is defined as 0 =/= 1 for 2D case.
        elif self.n_dim == 3:
            zv = np.linspace(
                -(self.nZ - 1) / (2 * self.spatial_sampling_rate),
                (self.nZ - 1) / (2 * self.spatial_sampling_rate),
                self.nZ,
            )

        # If device is specified, the vectors are provided as tensor. Otherwise, numpy array are provided.
        # Providing tensor directly at this level facilitate data sharing and avoid storing duplicates unnecessarily.
        # if device is not None:
        #     xv = torch.as_tensor(xv, device=device)
        #     yv = torch.as_tensor(yv, device=device)
        #     zv = torch.as_tensor(zv, device=device)

        self.coord_vec_list = [xv, yv, zv]
        return self.coord_vec_list


class TargetGeometry(Volume):
    def __init__(
        self,
        target=None,
        stlfilename=None,
        resolution=None,
        imagefilename=None,
        pixels=None,
        rot_angles=[0, 0, 0],
        bodies="all",
        binarize_image=True,
        clip_to_circle=True,
        options=None,
    ):
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
        self.zero_dose = None

        if target is not None:
            array = np.atleast_3d(
                target
            )  # Adapt to new practice of treating both 2D and 3D targets in 3D array.

        # image as target
        elif imagefilename is not None and stlfilename is None:
            # open and convert image to grayscale (single channel 2D matrix)
            image = Image.open(imagefilename).convert("L")

            if image.size[0] != image.size[1]:
                # pad non-square image into square image
                sq_size = np.max(image.size)
                delta_w = sq_size - image.size[0]
                delta_h = sq_size - image.size[1]
                padding = (
                    delta_w // 2,
                    delta_h // 2,
                    delta_w - (delta_w // 2),
                    delta_h - (delta_h // 2),
                )
                image = ImageOps.expand(image, padding)

            # resize to requested size
            if pixels is not None:
                image = image.resize(size=(pixels, pixels))
            image = np.array(image).astype(np.float32)
            # normalize image to 0-1 range
            image = image / np.max(image)
            # binarize image
            if binarize_image == True:
                array = np.where(image >= 0.5, 1.0, 0.0)
            else:
                array = image

            if bodies != "all":
                print(
                    "Warning: zero dose and insert bodies are not implemented in 2D yet."
                )
                self.zero_dose = None
                self.insert = None
            else:
                self.zero_dose = None
                self.insert = None

            array = np.atleast_3d(
                array
            )  # Adapt to new practice of treating both 2D and 3D targets in 3D array.

        # stl file as target to voxelized
        elif stlfilename is not None:
            self.stlfilename = stlfilename
            array, insert, zero_dose = vamtoolbox.voxelize.voxelizeTargetOpenGL(
                stlfilename, resolution, bodies, rot_angles
            )
            self.zero_dose = zero_dose
            self.insert = insert

        self.gel_inds, self.void_inds = getInds(array)
        if clip_to_circle:
            array = vamtoolbox.util.data.clipToCircle(array)
        super().__init__(
            array=array, options=options, file_extension=".target", vol_type="target"
        )

    def segmentZ(self, slices):
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
        if isinstance(slices, int) or (isinstance(slices, list) and len(slices) == 1):
            self.array = self.array[:, :, slices]
            if self.insert is not None:
                self.insert = self.insert[:, :, slices]
            self.n_dim = 2
            self.nZ = 0
            self.resolution = None

        if isinstance(slices, list):
            self.array = self.array[:, :, slices[0] : slices[1]]
            if self.insert is not None:
                self.insert = self.insert[:, :, slices[0] : slices[1]]
            self.nZ = slices[1] - slices[0] + 1
            self.resolution = self.nZ

    def show(
        self,
        show_bodies=False,
        savepath=None,
        dpi="figure",
        transparent=False,
        **kwargs,
    ):
        kwargs["cmap"] = "gray" if "cmap" not in kwargs else kwargs["cmap"]
        kwargs["interpolation"] = (
            "none" if "interpolation" not in kwargs else kwargs["interpolation"]
        )

        if self.n_dim == 2:
            if show_bodies == True:
                vamtoolbox.display.SlicePlot(
                    self.array, self.vol_type, show_bodies=True, **kwargs
                )
            else:
                vamtoolbox.display.SlicePlot(self, self.vol_type, **kwargs)

        elif self.n_dim == 3:
            if show_bodies == True:
                # must keep instance of slicer for mouse wheel scrolling to work
                self.viewer = vamtoolbox.display.VolumeSlicer(
                    self, self.vol_type, show_bodies=True, **kwargs
                )
            else:
                # must keep instance of slicer for mouse wheel scrolling to work
                self.viewer = vamtoolbox.display.VolumeSlicer(
                    self.array, self.vol_type, **kwargs
                )

        if savepath is not None:
            plt.savefig(savepath, dpi=dpi, transparent=transparent)
        plt.show()


class Sinogram(Volume):
    def __init__(
        self, sinogram: np.ndarray, proj_geo: ProjectionGeometry, options=None
    ):
        """
        Parameters
        ----------
        sinogram : np.ndarray

        proj_geo : geometry.ProjectionGeometry

        options : dict, optional

        """
        super().__init__(
            array=sinogram,
            proj_geo=proj_geo,
            options=options,
            file_extension=".sino",
            vol_type="sino",
        )


class Reconstruction(Volume):
    def __init__(
        self, reconstruction: np.ndarray, proj_geo: ProjectionGeometry, options=None
    ):
        """
        Parameters
        ----------
        reconstruction : np.ndarray

        proj_geo : geometry.ProjectionGeometry

        options : dict, optional

        """
        super().__init__(
            array=reconstruction,
            proj_geo=proj_geo,
            options=options,
            file_extension=".recon",
            vol_type="recon",
        )


def loadVolume(file_name: str):
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
    file = open(file_name, "rb")
    data_pickle = file.read()
    file.close()
    A = dill.loads(data_pickle)
    return A


def getCircleMask(target: np.ndarray):
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
    # The inscribed-circle mask is purely radial in X-Y and INDEPENDENT of z, so build the
    # 2D mask and broadcast over z.  The old 3D np.meshgrid built three FULL float64 grids
    # (shape[0]*shape[1]*shape[2]*8 bytes EACH, plus R) — ~90 GB for a 2.8e9-voxel grid,
    # including a z meshgrid that was discarded — which OOM'd the backend on big parts.
    circle_y, circle_x = np.meshgrid(
        np.linspace(-1, 1, target.shape[0]), np.linspace(-1, 1, target.shape[1])
    )
    mask2d = np.array(circle_x**2 + circle_y**2 <= 1**2, dtype=bool)
    if np.ndim(target) == 2:
        return mask2d
    return np.broadcast_to(mask2d[:, :, np.newaxis], target.shape)


def getInds(target: np.ndarray):
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

    gel_inds = np.logical_and(target > 0, circle_mask)
    void_inds = np.logical_and(target == 0, circle_mask)

    return gel_inds, void_inds


def compute_rebin_params(
    vial_id_mm,
    vial_print_height_mm,
    mm_per_pix,
    proj_u_px,
    proj_v_px,
    throw_ratio=1.0,
):
    """
    Compute rebinFanBeam parameters and print a geometry summary for a given vial
    and optical setup.  Call once at script start to understand your achievable print volume.

    Parameters
    ----------
    vial_id_mm : float
        Vial inner diameter (mm).
    vial_print_height_mm : float
        Usable print height inside the vial (mm).  Subtract base glass thickness and
        meniscus clearance from the nominal vial height before passing here.
    mm_per_pix : float
        Physical size of one projector pixel at the vial plane (mm/px).
        Calibrate by projecting a ruler or known feature onto the vial plane.
    proj_u_px : int
        Projector pixel count in the U (vial-diameter) direction.
    proj_v_px : int
        Projector pixel count in the V (vial-height) direction.
    throw_ratio : float
        Projector throw ratio (throw distance / projected image width).

    Returns
    -------
    dict with keys
        vial_width_px  - pass directly as ``vial_width`` to rebinFanBeam
        N_screen       - pass directly as ``N_screen`` to rebinFanBeam
        size_scale     - pass as ``size_scale`` to ImageConfig so the rebinned
                         sinogram fills the projector U axis exactly
        max_diam_mm    - maximum printable part diameter (mm)
        max_height_mm  - maximum printable part height (mm)
        resolution     - recommended Z resolution (slices that span max_height_mm)
        mm_per_pix_for_full_vial - mm_per_pix required so the full vial inner
                         diameter fills the projector U axis (optical adjustment target)
    """
    fov_u_mm = proj_u_px * mm_per_pix
    fov_v_mm = proj_v_px * mm_per_pix
    vial_u_px = vial_id_mm / mm_per_pix

    max_diam_mm = min(vial_id_mm, fov_u_mm)
    max_height_mm = min(vial_print_height_mm, fov_v_mm)
    resolution = int(round(max_height_mm / mm_per_pix))

    # size_scale maps the vial_width_px sinogram onto the projector U axis.
    # < 1 when the vial is wider than the FOV (shrink to fit the canvas).
    # > 1 when the vial is narrower than the FOV (zoom in to fill).
    # = 1 when the vial inner diameter exactly fills the projector U axis (optimal).
    vial_width_px = int(round(vial_u_px))
    if vial_width_px % 2 != 0:
        vial_width_px += 1  # keep even so rebinFanBeam padding is always symmetric
    size_scale = proj_u_px / max(vial_u_px, 1.0)

    mm_per_pix_for_full_vial = vial_id_mm / proj_u_px

    sep = "=" * 65
    print(sep)
    print("  VAM vial geometry analysis")
    print(sep)
    print(f"  Optical setup")
    print(f"    mm per pixel (calibrated) : {mm_per_pix:.5f} mm/px")
    print(f"    Projector U (diam axis)   : {proj_u_px} px  = {fov_u_mm:.1f} mm")
    print(f"    Projector V (height axis) : {proj_v_px} px  = {fov_v_mm:.1f} mm")
    print(f"  Vial")
    print(f"    Inner diameter (ID)       : {vial_id_mm:.2f} mm  = {vial_u_px:.0f} px")
    print(f"    Usable print height       : {vial_print_height_mm:.2f} mm")
    print(f"  Maximum printable part")
    print(
        f"    Diameter                  : {max_diam_mm:.2f} mm  ({int(round(max_diam_mm / mm_per_pix))} px)"
    )
    print(f"    Height                    : {max_height_mm:.2f} mm  ({resolution} px)")
    print(f"  rebinFanBeam parameters")
    print(f"    vial_width                : {vial_width_px} px")
    print(f"    size_scale for ImageConfig: {size_scale:.4f}")
    if abs(size_scale - 1.0) > 0.01:
        if vial_u_px > proj_u_px:
            print(f"")
            print(f"  *** OPTICS NOTE ***")
            print(
                f"  Vial ID ({vial_id_mm:.1f} mm) exceeds the projector FOV ({fov_u_mm:.1f} mm)."
            )
            print(
                f"  The printed part diameter is limited to {max_diam_mm:.1f} mm by the optics."
            )
            print(f"  To maximize part size and set size_scale = 1.0, adjust the optics so")
            print(
                f"  the vial ID spans the full {proj_u_px} px (set mm_per_pix = "
                f"{mm_per_pix_for_full_vial:.5f} mm/px)."
            )
        else:
            print(f"")
            print(f"  NOTE: Vial ID ({vial_id_mm:.1f} mm) is smaller than the projector FOV")
            print(
                f"  ({fov_u_mm:.1f} mm).  size_scale > 1 zooms in to fill the projector width."
            )
    print(sep)

    return {
        "vial_width_px": vial_width_px,
        "N_screen": (proj_u_px, proj_v_px),
        "size_scale": size_scale,
        "max_diam_mm": max_diam_mm,
        "max_height_mm": max_height_mm,
        "resolution": resolution,
        "mm_per_pix_for_full_vial": mm_per_pix_for_full_vial,
    }


def compute_fanbeam_extent(N_r, vial_width, N_screen, n_write, throw_ratio):
    """
    Determine how far outward the fan-beam rebinning stretches a sinogram of
    width N_r pixels when projected through a cylindrical vial.

    Because n_write > 1, each projector pixel at position xp illuminates a point
    *closer to centre* inside the vial (xv < xp).  To reach the edge of the part
    (at +/-N_r/2 in sinogram space) the projector must use pixels further out than
    +/-N_r/2.  This function finds that outermost projector pixel (xp_edge) and
    reports whether it falls within the vial_width frame.

    If xp_edge > vial_width/2 the outer edges of the part cannot be illuminated by
    any projector pixel in the vial frame - they will be silently zero in the
    rebinned sinogram.

    Parameters
    ----------
    N_r         : int   radial width of the original (pre-rebin) sinogram in pixels
    vial_width  : int   vial inner diameter in projector pixels (= rebinFanBeam vial_width)
    N_screen    : tuple (N_U, N_V) projector pixel count passed to rebinFanBeam
    n_write     : float refractive index of the resin at the writing wavelength
    throw_ratio : float projector throw ratio

    Returns
    -------
    dict
        xp_edge_px       - projector pixel (from centre) where the part boundary maps to
        vial_half_px     - vial half-width (= vial_width/2); the available frame limit
        stretch_factor   - xp_edge / (N_r/2); > 1 means the image is stretched outward
        frame_fill_frac  - xp_edge / vial_half; > 1 means part edges exceed the frame
        fits_in_frame    - True when the entire part can be projected without clipping
        clipped_frac     - fraction of the half-width that is outside the frame (0 if fits)
    """
    N_U = N_screen[0]
    n1 = 1.0
    n2 = float(n_write)

    throw_ratio_pix = throw_ratio * N_U
    vial_half = vial_width / 2.0
    part_half = N_r / 2.0

    xp = np.linspace(-vial_half, vial_half, int(vial_width))
    phi = np.arctan(xp / throw_ratio_pix)

    # Position where a ray from pixel xp hits the vial wall
    discriminant = 1.0 - (1.0 + (xp / throw_ratio_pix) ** 2) * (
        1.0 - (vial_half / throw_ratio_pix) ** 2
    )
    discriminant = np.maximum(discriminant, 0.0)  # guard sqrt of negative
    Rv = vial_half * np.sqrt(1.0 + (vial_half / throw_ratio_pix) ** 2)
    xps = (xp - xp * np.sqrt(discriminant)) / (1.0 + (xp / throw_ratio_pix) ** 2)

    theta10 = np.arcsin(np.clip(xps / Rv, -1.0, 1.0))
    thetai = theta10 + phi
    sin_t = np.clip((n1 / n2) * np.sin(thetai), -1.0, 1.0)
    thetat = np.arcsin(sin_t)
    thetav = theta10 - thetat

    xv = xps * np.cos(thetav) - np.sin(thetav) * np.sqrt(
        np.maximum(Rv**2 - xps**2, 0.0)
    )

    # Use only the positive half (symmetric problem)
    mask = xp >= 0
    xp_pos = xp[mask]
    xv_pos = xv[mask]

    # xv is the virtual (parallel-ray) coordinate illuminated by physical pixel xp.
    # Find the physical pixel whose virtual coordinate equals the part boundary.
    max_reachable_xv = np.max(xv_pos)

    if part_half <= 0:
        xp_edge = 0.0
    elif part_half > max_reachable_xv:
        # Even the outermost pixel can't reach the part edge - it's already clipped
        xp_edge = vial_half * (part_half / max_reachable_xv)  # extrapolated
    else:
        xp_edge = float(np.interp(part_half, xv_pos, xp_pos))

    stretch_factor = xp_edge / max(part_half, 1e-9)
    frame_fill_frac = xp_edge / max(vial_half, 1e-9)
    fits = frame_fill_frac <= 1.0
    clipped_frac = max(0.0, frame_fill_frac - 1.0)

    return {
        "xp_edge_px": xp_edge,
        "vial_half_px": vial_half,
        "stretch_factor": stretch_factor,
        "frame_fill_frac": frame_fill_frac,
        "fits_in_frame": fits,
        "clipped_frac": clipped_frac,
    }


def rebinFanBeam(sinogram, vial_width, N_screen, n_write, throw_ratio):
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

    def rebin(b, xp, angles, x_samp, theta_samp, dxv_dxp, T=None):
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

        b_rebinned = interpolate.interpn(
            (xp, angles),
            b,
            (x_samp, theta_samp),
            method="linear",
            bounds_error=False,
            fill_value=0,
        )  # resampling happens here
        b_rebinned = (
            b_rebinned * dxv_dxp
        )  # Correcting for change in differential area in radon space.  Very small correction, could probably be ignored in most cases.

        if T is not None:
            # Guard against T=0 at grazing-incidence edge pixels (theta_i -> 90 deg).
            # 1/max(T, 1e-10) prevents division-by-zero/NaN; the T-threshold mask
            # applied after the loop removes the over-amplified edge rows.
            T_inv = np.where(T > 0, 1.0 / np.maximum(T, 1e-10), 0.0)
            # Correction for Fresnel transmission loss
            b_rebinned = T_inv * b_rebinned

        return b_rebinned

    angles = sinogram.proj_geo.angles
    N_angles = angles.size
    N_z, N_r = sinogram.array.shape[2], sinogram.array.shape[0]
    N_U, N_V = N_screen
    n1 = 1  # refractive index of air
    n2 = n_write  # measured refractive index at the projection beam wavelength
    vial_width = int(
        vial_width
    )  # Apparent vial width in the field of view of the projector.  Obtained by projecting projector columns and noting the first and last columns to intersect the vial vall.

    throw_ratio_pix = (
        throw_ratio * N_U
    )  # Throw ratio x number of pixels in the horizontal direction.  Change this depending on projector width in pixels.

    Rv = (vial_width / 2) * np.sqrt(
        1 + ((vial_width) / (2 * throw_ratio_pix)) ** 2
    )  # Actual radius of vial in units of pixels

    xp = np.linspace(
        -vial_width / 2, vial_width / 2, vial_width
    )  # Projector x-coordinate
    phi = np.arctan(
        xp / throw_ratio_pix
    )  # Extra divergence angle caused by non-telecentricity of projector across the projector's field of view

    xps = (
        xp
        - xp
        * np.sqrt(
            1 - (1 + (xp / throw_ratio_pix) ** 2) * (1 - (Rv / throw_ratio_pix) ** 2)
        )
    ) / (
        1 + (xp / throw_ratio_pix) ** 2
    )  # Location at which a ray from projector pixel xp intersects the vial

    theta10 = np.arcsin(
        xps / Rv
    )  # Angle that the normal vector of the vial makes with the optical axis at xps
    thetai = (
        np.arcsin(xps / Rv) + phi
    )  # angle of incidence of a light ray from projector pixel xp, incident at on the vial at xps
    thetat = np.arcsin(
        (n1 / n2) * np.sin(thetai)
    )  # angle of transmission after refraction at air/vial interface
    thetav = (
        theta10 - thetat
    )  # Deviation from optical axis of transmitted ray after refraction
    thetavD = (180 / np.pi) * thetav  # As above, expressed in degrees

    # Fresnel coefficients
    Ts = (
        1
        - (
            (n1 * np.cos(thetai) - n2 * np.cos(thetat))
            / ((n1 * np.cos(thetai) + n2 * np.cos(thetat)))
        )
        ** 2
    )
    Tp = (
        1
        - (
            (n1 * np.cos(thetat) - n2 * np.cos(thetai))
            / ((n1 * np.cos(thetat) + n2 * np.cos(thetai)))
        )
        ** 2
    )
    T = (Ts + Tp) / 2  # averaging for s- and p-polarized light
    T[T < 0] = 0  # Just in case
    T_b = np.transpose(np.tile(T, (N_angles, 1)))

    # Calculating change in differential area element due to variable change
    xv = (xps * np.cos(thetav)) - (np.sin(thetav) * np.sqrt((Rv**2) - (xps**2)))
    dxv_dxp = np.gradient(xv)
    dxv_dxp[T < 0] = 0  # just in case of a pathological situation
    dxv_dxp[dxv_dxp < 0] = (
        0  # ignore pixels where the sign of the differential area flips
    )
    dxv_dxp_tiled = np.transpose(np.tile(dxv_dxp, (N_angles, 1)))

    # Correctable-region diagnostic --------------------------------------------
    # xv peaks at the correctable boundary; beyond it dxv_dxp <= 0 and those
    # physical pixels contribute nothing.  If N_r/2 > max(xv), the outermost
    # sinogram pixels are unreachable at the widest projection angles.
    xv_max = float(np.max(xv))
    vial_half = vial_width / 2.0
    actual_frac = xv_max / vial_half
    sinogram_half = N_r / 2.0
    margin_px = xv_max - sinogram_half
    tr_str = "inf (collimated)" if np.isinf(throw_ratio) else f"{throw_ratio:.2f}"
    print(f"  rebinFanBeam correctable region  (n={n_write:.3f}, throw={tr_str})")
    print(
        f"    max reachable xv    : {xv_max:7.2f} px  ({actual_frac:.4f} x vial half-width)"
    )
    print(f"    sinogram half-width : {sinogram_half:7.1f} px  (N_r/2)")
    print(f"    margin              : {margin_px:+7.2f} px")
    if margin_px < 0:
        print(
            f"    *** CUT-OFF: outer {abs(margin_px):.1f} px of sinogram are unreachable - "
            f"edge features will be missing at widest projection angles ***"
        )
    elif margin_px < 5:
        print(
            f"    WARNING: margin is very tight ({margin_px:.1f} px) - slight edge cut-off likely."
        )
    # end diagnostic -----------------------------------------------------------

    # Constructing the arrays (theta_samp = thetav) that contains the angles and ray coordinates (x_samp) at which the build volume is sampled by the projector.
    (
        theta_samp,
        x_samp,
    ) = np.meshgrid(angles, xv)
    thetaDelt = np.transpose(np.tile(thetavD, (N_angles, 1)))
    theta_samp = (
        theta_samp + thetaDelt
    )  # Remember, each ray in the vial is rotated by thetaDelt with respect to the optical axis

    # Wrap theta_samp back into [0, 360) then clamp to [min_theta, max_theta].
    # The original code clamped anything > max_theta-diff_theta to min_theta, which
    # incorrectly forced valid angles like 359 deg to 0 deg whenever thetavD pushed a
    # near-edge pixel's angle above 358 deg.  That made edge pixels use the wrong
    # projection angle, producing a ghost line on the opposite side of the frame
    # as the rotation approached the 0/360 deg seam.
    min_theta, max_theta = sinogram.proj_geo.angles[0], sinogram.proj_geo.angles[-1]
    theta_samp[theta_samp > 360] = theta_samp[theta_samp > 360] - 360
    theta_samp[theta_samp < 0] = theta_samp[theta_samp < 0] + 360
    theta_samp = np.clip(theta_samp, min_theta, max_theta)

    # Pad sinogram radial axis to vial_width, keeping content centered.
    # If N_r is odd, add one zero-column first so both N_r and vial_width are even
    # and the centering pad is always exactly symmetric.
    if N_r % 2 != 0:
        sinogram.array = np.pad(
            sinogram.array, ((0, 1), (0, 0), (0, 0)), mode="constant"
        )
        N_r += 1
    total_pad = vial_width - N_r
    if total_pad > 0:
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        sinogram.array = np.pad(
            sinogram.array, ((pad_left, pad_right), (0, 0), (0, 0)), mode="constant"
        )
    elif total_pad < 0:
        # The part is WIDER than the apparent vial (its radial extent exceeds the
        # projector's correctable region) — crop (cut off) the out-of-region edges,
        # centered, so the rebin grid (xp, width=vial_width) matches the sinogram.
        # The cut-off region can't be reached by the physical projector anyway.
        crop = -total_pad
        crop_left = crop // 2
        crop_right = crop - crop_left
        warning(
            f"Part radial extent ({N_r} px) exceeds the vial's correctable region "
            f"({vial_width} px); cropped {crop} px (±{crop_left}) before rebin.")
        sinogram.array = sinogram.array[crop_left:N_r - crop_right, :, :]
        N_r = vial_width

    sinogram_rs = np.zeros_like(
        sinogram.array
    )  # Initializing array that will contain the resampled projections

    # Precompute T_inv once (the old inner rebin() recomputed it for every z-slice).
    T_inv = np.where(T_b > 0, 1.0 / np.maximum(T_b, 1e-10), 0.0)

    # Resample each z-slice (frame).  The slices are independent, so dispatch
    # contiguous z-blocks across CPU workers when joblib is available.  Only worth
    # it for large sinograms: below ~384 slices the one-time loky worker-spawn
    # (paid here on GPU runs, where the pool isn't already warm) outweighs the gain.
    if _Parallel is not None and N_z >= 384 and REBIN_N_JOBS != 1:
        n_chunks = int(min(N_z, max(1, _eff_n_jobs(REBIN_N_JOBS)) * 2))
        bounds = [b for b in np.array_split(np.arange(N_z), n_chunks) if len(b) > 0]
        blocks = _Parallel(n_jobs=REBIN_N_JOBS)(
            _jdelayed(_rebin_chunk)(
                sinogram.array[:, :, zs[0]:zs[-1] + 1],
                xp, angles, x_samp, theta_samp, dxv_dxp_tiled, T_inv,
            )
            for zs in bounds
        )
        i0 = 0
        for blk in blocks:
            sinogram_rs[:, :, i0:i0 + blk.shape[2]] = blk
            i0 += blk.shape[2]
    else:
        for z_i in range(N_z):
            sinogram_rs[..., z_i] = _rebin_chunk(
                sinogram.array[:, :, z_i:z_i + 1],
                xp, angles, x_samp, theta_samp, dxv_dxp_tiled, T_inv,
            )[..., 0]

    # Zero out near-tangent edge pixels where Fresnel T -> 0 (T_inv -> inf).
    # For telecentric projection dxv_dxp > 0 everywhere, so the dxv_dxp > 0 mask
    # is vacuous - it doesn't remove any rows.  The actual problem is T_inv >= 5x
    # at the last 1-2 rows on each side (xp ~ +/-vial_half, grazing incidence).
    # Threshold on T directly: any row with T < 0.2 has T_inv > 5x amplification
    # and will produce visible bright-line artifacts at the output frame edges.
    T_EDGE_THRESHOLD = 0.2
    correctable = T >= T_EDGE_THRESHOLD
    sinogram_rs[~correctable, :, :] = 0

    return Sinogram(sinogram_rs, sinogram.proj_geo)
