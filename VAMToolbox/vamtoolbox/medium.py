# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the GNU GPLv3 license.

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


class MediumModel:  # Abstract class. Superclass of IndexModel and AttenuationModel

    @torch.inference_mode()
    def __init__(self, coord_vec: np.ndarray):
        """
        This superclass implement the coordinate grid and interpolation functions that are common to both IndexModel and AttenuationModel.
        The subclasses of Medium Model, namely IndexModel and AttenuationModel, decribe the refractive index and attenuation coefficients of the propagation medium.

        Internally implemented with pyTorch tensor for GPU computation and compatibility with pyTorchRayTrace.
        Input/output compatibility with numpy array is to be added.

        Parameters
        ----------
        coord_vec : list of coordinate vectors [xv, yv, zv]
        """

        # Construct meshgrid in tensor
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # ==========================Setup coordinate grid according ======================================================
        # Set up grid as tensor
        self.xv = torch.as_tensor(coord_vec[0], device=self.device)
        self.yv = torch.as_tensor(coord_vec[1], device=self.device)
        self.zv = torch.as_tensor(coord_vec[2], device=self.device)
        self.xg, self.yg, self.zg = torch.meshgrid(
            self.xv, self.yv, self.zv, indexing="ij"
        )  # This grid is to be used for interpolation, simulation

        # Max min of grid coordinate
        self.xv_yv_zv_max = torch.tensor(
            [torch.amax(self.xv), torch.amax(self.yv), torch.amax(self.zv)],
            device=self.device,
        )
        self.xv_yv_zv_min = torch.tensor(
            [torch.amin(self.xv), torch.amin(self.yv), torch.amin(self.zv)],
            device=self.device,
        )
        if self.zv.numel() == 1:
            self.voxel_size = torch.tensor(
                [self.xv[1] - self.xv[0], self.yv[1] - self.yv[0], torch.inf],
                device=self.device,
            )  # Sampling rate [voxel/cm] along z is 0, therefore voxel size is inf to stay consistent
        else:
            self.voxel_size = torch.tensor(
                [
                    self.xv[1] - self.xv[0],
                    self.yv[1] - self.yv[0],
                    self.zv[1] - self.zv[0],
                ],
                device=self.device,
            )  # Only valid when self.zv.size > 1

        self.grid_span = self.xv_yv_zv_max - self.xv_yv_zv_min
        self.position_normalization_factor = (
            2.0 / self.grid_span
        )  # normalize physical location by 3D volume half span, such that values are between [-1, +1]

        # In 2D case, the z span is 0. The above line evaluate as 2.0/0 which yields inf
        if torch.isinf(self.position_normalization_factor[2]):
            # If inf, clamp the z position down back to 0
            self.position_normalization_factor[2] = 0.0

        # Define this attribute in MediumModel. To be populated in subclasses.
        self.presampled_scalar_field_3D: Tensor | None = None
        # Define this attribute in MediumModel. To be populated in subclasses.
        self.presampled_vector_field_4D: Tensor | None = None
        # Default extrapolation settings. Can be overriden in subclass.
        self.interpolation_padding_mode = "border"

    # =================================Interpolation functions==========================================================================
    @torch.inference_mode()
    def _formatPresampledScalarField(self):
        """
        Format presampled_scalar_field_3D into presampled_scalar_field_5D and store the latter as object attribute.
        """
        # Check imported index field
        if len(self.presampled_scalar_field_3D.shape) <= 2:
            raise Exception(
                'Imported scalar field ("n_x" or "alpha_x") should be 3D (even in 2D the thrid dim should have size 1).'
            )
        if (self.xg.shape) != (self.presampled_scalar_field_3D.shape):
            raise Exception(
                f'Size mismatch between the constructed coordinate grid ({self.xg.shape}) and imported scalar field ("n_x" or "alpha_x") ({self.presampled_scalar_field_3D.shape}).'
            )

        # Setup pyTorch interpolation grid_sample inputs
        # Two requirements of grid_sample function: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample
        # (1)Volumetric data stored in a 5D grid
        # (2)Sampled position normalized to be within [-1, +1] in all dimensions of the grid

        self.presampled_scalar_field_5D = torch.permute(
            self.presampled_scalar_field_3D, (2, 1, 0)
        )  # Swapping the x and z axis because grid_sample takes input tensor in (N,Ch,Z,Y,X) order
        self.presampled_scalar_field_5D = self.presampled_scalar_field_5D[
            None, None, :, :, :
        ]  # Inser N and Ch axis
        # Note that the query points[0],[1],[2] are still arranged in x,y,z order
        self.logger.debug(
            f"presampled_scalar_field_5D has shape of {self.presampled_scalar_field_5D.shape}"
        )

    @torch.inference_mode()
    def _formatPresampledVectorField(self):
        """
        Format presampled_vector_field_4D into presampled_vector_field_5D and store the latter as object attribute.
        """
        # Check imported index gradient (vector) field
        if self.presampled_vector_field_4D is None:
            # If not provided, use finite difference to approximate the gradient
            self.logger.info(
                "Index gradient is not explicitly provided. Approximating index gradient with central finite difference"
            )
            self.presampled_vector_field_4D = self.centralFiniteDifference(
                self.presampled_scalar_field_3D, self.voxel_size
            )
        else:
            # If provided, check input. Gradient field should have 1 more dimension (for spatial derviatives)
            if not (
                self.presampled_vector_field_4D.shape
                == (
                    self.presampled_scalar_field_3D.shape[0],
                    self.presampled_scalar_field_3D.shape[1],
                    self.presampled_scalar_field_3D.shape[2],
                    3,
                )
            ):
                raise Exception(
                    "Index gradient field should have extactly one more dimension (of size 3) than index field."
                )

        # Setup pyTorch interpolation grid_sample inputs
        # Two requirements of grid_sample function: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample
        # (1)Volumetric data stored in a 5D grid
        # (2)Sampled position normalized to be within [-1, +1] in all dimensions of the grid

        self.presampled_vector_field_5D = torch.permute(
            self.presampled_vector_field_4D, (3, 2, 1, 0)
        )  # shape = [Ch, Z, Y, X] where Ch is the components of the gradient in x,y,z
        self.presampled_vector_field_5D = self.presampled_vector_field_5D[
            None, :, :, :, :
        ]  # Insert N axis
        self.logger.debug(
            f"presampled_vector_field_5D has shape of {self.presampled_vector_field_5D.shape}"
        )

    @torch.inference_mode()
    def _scalar_field_interp(self, x: Tensor):
        """
        Obtain scalar field value (e.g. refractive index, attenuation coefficient) via interpolating on provided or pre-sampled data points
        """
        x = self.normalizePosition(
            x
        )  # Normalize x such that the computation domain is between [-1,+1] in all dimensions

        # self.presampled_scalar_field_5D has already its axes permuted such that the last 3 dimensions are Z,Y,X in the class __init__
        # The query points x should still have x,y,z components in its [:,0], [:,1], [:,2] respectively

        scalar_val = torch.nn.functional.grid_sample(
            input=self.presampled_scalar_field_5D,  # shape = [N, Ch, D_in, H_in, W_in], where D_in, H_in, W_in are ZYX dimension respectively
            grid=x[
                None, None, None, :, :
            ],  # shape = [N,D_out, H_out, W_out,3], where N = D_out = H_out = 1, and W_out = number of samples
            mode="bilinear",
            padding_mode=self.interpolation_padding_mode,
            align_corners=True,  # True: extrema refers to center of corner voxels. False: extrema refers to corner of corner voxels
        )

        return scalar_val[
            0, 0, 0, 0, :
        ]  # scalar_val originally has shape [N, C, D_out, H_out, W_out], where W_out = number of samples

    @torch.inference_mode()
    def _vector_field_interp(self, x: Tensor):
        """
        Obtain vector field value (e.g. gradient of refractive index) via interpolating on provided or pre-sampled data points
        """
        x = self.normalizePosition(
            x
        )  # Normalize x such that the computation domain is between [-1,+1] in all dimensions

        # self.presampled_vector_field_5D has already its axes permuted such that the last 4 dimensions are Ch,Z,Y,X in the class __init__
        # The query points x should still have x,y,z components in its [:,0], [:,1], [:,2] respectively

        vector_val = torch.nn.functional.grid_sample(
            input=self.presampled_vector_field_5D,  # shape = [N, Ch, D_in, H_in, W_in], where Ch is channel, and D_in, H_in, W_in are ZYX dimension respectively
            grid=x[
                None, None, None, :, :
            ],  # shape = [N,D_out, H_out, W_out,3], where N = D_out = H_out = 1, and W_out = number of samples
            mode="bilinear",
            padding_mode=self.interpolation_padding_mode,
            align_corners=True,  # True: extrema refers to center of corner voxels. False: extrema refers to corner of corner voxels
        )
        # vector_val originally has shape [N, C, D_out, H_out, W_out], where W_out = number of samples
        return torch.squeeze(
            vector_val[0, :, 0, 0, :]
        ).T  # remove singleton dimension and transpose. Resultant shape should be [number of samples, Ch], where Ch = {0,1,2} < 3

    @torch.inference_mode()
    def normalizePosition(self, x):
        # Need to check if the meshgrid has index increasing from the negative physical location to positive physical location
        return x * self.position_normalization_factor

    def getPositionVectorsAtGridPoints(self):
        """Get the position vectors (x) with shape [num_of_points,3] at the stored grid points."""
        return torch.vstack(
            (torch.ravel(self.xg), torch.ravel(self.yg), torch.ravel(self.zg))
        ).T  # 1D tensor is row vector. Stack and then transpose yields shape (samples, 3)

    @staticmethod
    @torch.inference_mode()
    def centralFiniteDifference(presampled_scalar_field_5D, voxel_size):
        # This function evaluate the first derivative of a scalar field with central finite difference.
        # The boundary elements of gradient are set to zero because the scalar field is assumed constant outside the domain.

        dn_dx = torch.zeros_like(presampled_scalar_field_5D)
        dn_dx[1:-1, :, :] = torch.narrow(
            presampled_scalar_field_5D, 0, 2, presampled_scalar_field_5D.shape[0] - 2
        ) - torch.narrow(
            presampled_scalar_field_5D, 0, 0, presampled_scalar_field_5D.shape[0] - 2
        )  # Last argument of narrow dictates the size of output along that 'dim' dimension
        dn_dx = dn_dx / (
            2 * voxel_size[0]
        )  # The trim-first version of the array minus the trim-last version, divided by 2*voxel size

        dn_dy = torch.zeros_like(presampled_scalar_field_5D)
        dn_dy[:, 1:-1, :] = torch.narrow(
            presampled_scalar_field_5D, 1, 2, presampled_scalar_field_5D.shape[1] - 2
        ) - torch.narrow(
            presampled_scalar_field_5D, 1, 0, presampled_scalar_field_5D.shape[1] - 2
        )  # Last argument of narrow dictates the size of output along that 'dim' dimension
        dn_dy = dn_dy / (
            2 * voxel_size[1]
        )  # The trim-first version of the array minus the trim-last version, divided by 2*voxel size

        dn_dz = torch.zeros_like(presampled_scalar_field_5D)
        if (presampled_scalar_field_5D.shape[2] == 1) or torch.isnan(
            voxel_size[2]
        ):  # 2D problem.
            pass
        else:  # 3D problem
            dn_dz[:, :, 1:-1] = torch.narrow(
                presampled_scalar_field_5D,
                2,
                2,
                presampled_scalar_field_5D.shape[2] - 2,
            ) - torch.narrow(
                presampled_scalar_field_5D,
                2,
                0,
                presampled_scalar_field_5D.shape[2] - 2,
            )  # Last argument of narrow dictates the size of output along that 'dim' dimension
            dn_dz = dn_dz / (
                2 * voxel_size[2]
            )  # The trim-first version of the array minus the trim-last version, divided by 2*voxel size

        return torch.cat(
            (dn_dx[:, :, :, None], dn_dy[:, :, :, None], dn_dz[:, :, :, None]), dim=3
        )  # add new axis at the end and concatenate along that new axis


class IndexModel(MediumModel):
    _default_analytical = {
        "R": 1.0,
        "length_unit_of_R": "fractional_domain_radius",
        "p": 2.0,
        "n_sur": 1.0,
    }  # default arguments for **kwargs
    _default_interpolation = {"interpolation_padding_mode": "border"}

    @torch.inference_mode()
    def __init__(
        self,
        coord_vec: np.ndarray,
        type: str = "analytical",
        form: str = "homogeneous",
        **kwargs,
    ):
        """
        Analytical and interpolated decription of index distribution of the simulation domain.
        Internally implemented with pyTorch tensor for GPU computation and compatibility with pyTorchRayTrace.
        Input/output compatibility with numpy array is to be added.

        Naming: n(x) and grad_n(x) is the index and spatial gradient of index as a function of position x.
        Input x should have shape [num_of_sample, 3]. And the output of n and grad_n would have shape [num_of_sample] and [num_of_sample, 3] respectively.

        Although in particular total attenuation coefficient is directly related to the imaginary part (kappa) of complex refractive index (n_comp = n_real + i*kappa),
        index in this module refers only to the real part of the complex index. Reference:https://en.wikipedia.org/wiki/Refractive_index#Complex_refractive_index
        The total attenuation coefficient (alpha in AttenuationModel) equals 4*pi*kappa/wavelength.

        The attenuation part is handled separately from n_real for performance and organizational reasons.
        Performance:
            In all cases, computation of gradient of kappa is not required (so it is not included in computation/sampling of grad_n).
            In many cases, the attenuation is homogeneous within certain boundary and interpolation gives trivial results (even when interpolation of n_real is required).
        Organization:
            Generally attenuation effects can have their own description which is almost independent of n_real.
            The sampling of a scalar field quantity in general (atteunation, absorption or scattering coefficient) is described in AttenuationModel class.

        Parameters
        ----------
        type : str ('analytical', 'interpolation')
            Select analytical function evaluation or interpolate on pre-built interpolant arrays.
            Interpolation method handles edge cases of input explicitly and hence is more robust.

        form : str ('homogeneous', 'luneburg_lens', 'maxwell_lens', 'eaton_lens', 'freeform')

        R : float, optional
            Radius of the luneburg/maxwell/eaton lens
            Default equals to the radius of the grid (lens occupy almost all of simulation volume), regardless of length_unit parameter.

        length_unit_of_R: str, optional, ('fractional_domain_radius', 'physical_coordinate')
            This unit is only used in specification of R (radius of analytical lens). The size of computational grid is always assumed to be the same as input coordinate vectors.
            Default: 'fractional_domain_radius', where 1 represent a lens filling the domain with boundaries touching
            'physical_coordinate' uses the same unit as the input coordinate vectors (coord_vec, the first argument).

        p : float, power of luneburg lens
            Default 2, which is used in classical definition of luneburg lens.

        n_sur : float, index of the surroundings (at the wavelength of interest)
            The absolute index of the lenses or other analytical index distribution will be adjusted accordingly to achieve equivalent refraction.
            Also, this sets the extrapolation value when index is queried outside the grid
        """

        super().__init__(coord_vec)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Medium computation is performed on: {repr(self.device)}")

        # Save inputs as attributes
        self.type = type
        self.form = form

        # Merge all kwargs into params
        self.params = (
            self._default_analytical.copy()
        )  # Shallow copy avoid editing dict '_default_lenses' in place
        self.params.update(self._default_interpolation)  # Add relevant parameters
        self.params.update(kwargs)  # up-to-date parameters. Default dict is not updated

        # Compute derived attributes R_physical
        if self.params["length_unit_of_R"] == "fractional_domain_radius":
            R_domain_0 = (np.amax(coord_vec[0]) - np.amin(coord_vec[0])) / 2
            R_domain_1 = (np.amax(coord_vec[1]) - np.amin(coord_vec[1])) / 2
            R_domain_2 = (np.amax(coord_vec[2]) - np.amin(coord_vec[2])) / 2
            if coord_vec[2].size == 1:  # only fit in xy box in 2D case
                R_domain_min = min(R_domain_0, R_domain_1)
            else:  # fit in xyz box in 3D case
                R_domain_min = min(R_domain_0, R_domain_1, R_domain_2)
            self.params["R_physical"] = self.params["R"] * R_domain_min

        elif self.params["length_unit_of_R"] == "physical_coordinate":
            self.params["R_physical"] = self.params["R"]

        else:
            raise Exception(
                '"length_unit" should be either "fractional_domain_radius" or "physical_coordinate"'
            )

        # ==========================Setup index query functions according to type and form================================
        if self.type == "analytical":
            if self.form == "homogeneous":
                self.n = self._n_homo
                self.grad_n = self._grad_n_homo

            elif self.form == "luneburg_lens":
                self.n = self._n_lune
                self.grad_n = self._grad_n_lune

            elif self.form == "maxwell_lens":
                self.n = self._n_maxwell
                self.grad_n = self._grad_n_maxwell

            elif self.form == "eaton_lens":
                self.n = self._n_eaton
                self.grad_n = self._grad_n_eaton

            else:
                raise Exception(
                    "Form: Other analytical functions are not supported yet."
                )

        elif self.type == "interpolation":
            # Interpolation method stores two 3D arrays as interpolant and query them upon each mapping call.
            # Stored arrays : (1) index (a scalar field) and (2)gradient of n (a vector field).
            # They are sampled at the grid defined by coord_vec (presampled_x, which is also stored).

            # function alias
            self.n = self._scalar_field_interp
            self.grad_n = self._vector_field_interp
            # This setting is used in _scalar_field_interp and _vector_field_interp.
            self.interpolation_padding_mode = self.params["interpolation_padding_mode"]  # type: ignore
            self.presampled_x = self.getPositionVectorsAtGridPoints()

            # build or import interpolant dataset
            if self.form == "homogeneous":
                # build interpolant arrays
                self.presampled_scalar_field_3D = self._n_homo(
                    self.presampled_x
                ).reshape(self.xg.shape)
                grad_n_shape = list(self.xg.shape)
                # in-place modification. (This function call return None.) Gradient has additional dimension of 3 at each sampling position
                grad_n_shape.append(3)
                # Save as a 4D tensor
                self.presampled_vector_field_4D = torch.reshape(
                    self._grad_n_homo(self.presampled_x), tuple(grad_n_shape)
                )

            elif self.form == "luneburg_lens":
                # build interpolant arrays
                self.presampled_scalar_field_3D = self._n_lune(
                    self.presampled_x
                ).reshape(
                    self.xg.shape
                )  # Save as a 3D tensor
                grad_n_shape = list(self.xg.shape)
                grad_n_shape.append(
                    3
                )  # in-place modification. (This function call return None.) Gradient has additional dimension of 3 at each sampling position
                self.presampled_vector_field_4D = torch.reshape(
                    self._grad_n_lune(self.presampled_x), tuple(grad_n_shape)
                )  # Save as a 4D tensor

            elif self.form == "maxwell_lens":
                # build interpolant arrays
                self.presampled_scalar_field_3D = self._n_maxwell(
                    self.presampled_x
                ).reshape(self.xg.shape)
                grad_n_shape = list(self.xg.shape)
                grad_n_shape.append(
                    3
                )  # in-place modification. (This function call return None.) Gradient has additional dimension of 3 at each sampling position
                self.presampled_vector_field_4D = torch.reshape(
                    self._grad_n_maxwell(self.presampled_x), tuple(grad_n_shape)
                )  # Save as a 4D tensor

            elif self.form == "eaton_lens":
                # build interpolant arrays
                self.presampled_scalar_field_3D = self._n_eaton(
                    self.presampled_x
                ).reshape(self.xg.shape)
                grad_n_shape = list(self.xg.shape)
                grad_n_shape.append(
                    3
                )  # in-place modification. (This function call return None.) Gradient has additional dimension of 3 at each sampling position
                self.presampled_vector_field_4D = torch.reshape(
                    self._grad_n_eaton(self.presampled_x), tuple(grad_n_shape)
                )  # Save as a 4D tensor

            elif self.form == "freeform":  # Directly import data instead of generating.
                # Input data points are designated with sample subscript
                self.presampled_scalar_field_3D = self.params.get("n_x", None)  # type: ignore
                self.presampled_vector_field_4D = self.params.get("grad_n_x", None)  # type: ignore

            else:
                raise Exception("Other interpolation functions are not supported yet.")

            self._formatPresampledScalarField()  # Format presampled_scalar_field_3D into presampled_scalar_field_5D and store the latter as object attribute.
            self._formatPresampledVectorField()  # Format presampled_vector_field_4D into presampled_vector_field_5D and store the latter as object attribute.
        else:
            raise Exception('"type" should be either "analytical" or "interpolation".')

    # =================================Analytic: Homogeneous medium================================================
    @torch.inference_mode()
    def _n_homo(self, x: Tensor) -> Tensor:
        # return self.params['n_sur']*torch.ones(x.shape[0], device=x.device) #automatically create on same device
        # automatically create on same device
        return self.params["n_sur"] * torch.ones_like(x[:, 0])  # type: ignore

    @torch.inference_mode()
    def _grad_n_homo(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    # =================================Analytic: Luneburg lens=====================================================
    @torch.inference_mode()
    def _n_lune(self, x: Tensor | None = None, r_known: Tensor | None = None):
        """
        Parameters:

        x : Tensor, query position vector size(n_query_pts, 3)

        r_known :

        To achieve the lensing effect, the refractive index has to change relative to its surroundings.
        With p = 2, the center of the lens has index sqrt(2) times that of surroundings.
        Expression of n can be found in "Path Tracing Estimators for Refractive Radiative Transfer, Pediredla et. al., 2020, page 10"
        """
        if r_known is None:
            if x is None:
                raise Exception("Either x or r must be provided.")
            # radial position, r of vector x, 1D tensor, numel = x.size[0]
            r = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)
        else:
            r = r_known

        n = torch.ones_like(r)  # 1D tensor, numel = x.size[0]

        # Inside the lens
        r_physical: float = self.params["R_physical"]  # type: ignore
        p: float = self.params["p"]  # type: ignore
        n_sur: float = self.params["n_sur"]  # type: ignore
        n[r < r_physical] = n_sur * (2 - (r[r < r_physical] / r_physical) ** p) ** (1 / p) # fmt: skip

        # Outside or at the boundary of the lens.
        # if n_sur is just 1, the query points outside lens is already populated with 1 so no operation needed
        if self.params["n_sur"] != 1.0:
            n[r >= r_physical] = n_sur

        return n

    @torch.inference_mode()
    def _grad_n_lune(self, x: Tensor) -> Tensor:
        # radial position, r of vector x
        r = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)

        n = self._n_lune(r_known=r)

        grad_n = torch.zeros_like(x)
        n_sur: float = self.params["n_sur"]  # type: ignore
        r_physical: float = self.params["R_physical"]  # type: ignore
        p: float = self.params["p"]  # type: ignore

        # compute the constant multiplier together first for performance
        n_0_over_r_to_P = (n_sur / r_physical) ** p
        idx_r_in_lens = r < r_physical
        # Inside the lens. The following two lines are equivalent. The middle two tensors are 1-D so a new axis has to be inserted for broadcasting. None == np.newaxis in torch and numpy.
        # grad_n[idx_r_in_lens, :] = - n_0_over_r_to_P * torch.unsqueeze((n[idx_r_in_lens]**(1 - self.params['p'])) * (r[idx_r_in_lens]**(self.params['p']-2)), dim = 1) * x[idx_r_in_lens,:]
        grad_n[idx_r_in_lens, :] = (
            -n_0_over_r_to_P
            * ((n[idx_r_in_lens] ** (1 - p)) * (r[idx_r_in_lens] ** (p - 2)))[:, None]
            * x[idx_r_in_lens, :]
        )

        # Outside or at the boundary of the lens
        # grad_n[r >= self.params['R_physical'],:] = 0

        return grad_n

    # =================================Analytic: Maxwell lens============================================================
    @torch.inference_mode()
    def _n_maxwell(self, x: Tensor) -> Tensor:
        raise Exception("To be implemented.")

    @torch.inference_mode()
    def _grad_n_maxwell(self, x: Tensor) -> Tensor:
        raise Exception("To be implemented.")

    # =================================Analytic: Eaton lens============================================================
    @torch.inference_mode()
    def _n_eaton(self, x: Tensor) -> Tensor:
        raise Exception("To be implemented.")

    @torch.inference_mode()
    def _grad_n_eaton(self, x: Tensor) -> Tensor:
        raise Exception("To be implemented.")

    # =================================Utilities==========================================================================
    def plotIndex(self, fig=None, ax=None, block=False):
        """
        Plot a 2D slice of index. Currently only for real part. Future extenstion: for both its real and imaginary parts
        """
        if "presampled_x" not in self.__dict__:
            x_sample = self.getPositionVectorsAtGridPoints()
        else:
            x_sample = self.presampled_x

        n = self.n(x_sample).reshape(self.xg.shape)
        # n_slice = self.presampled_n_x[:,:,self.presampled_n_x.shape[2]//2].cpu().numpy()
        n_slice = n[:, :, n.shape[2] // 2].cpu().numpy()

        vmin, vmax = np.amin(n_slice), np.amax(n_slice)

        if ax == None:
            fig, ax = plt.subplots()

        mappable = ax.imshow(n_slice, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title("Index distribution")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(mappable)

        if block == False:
            fig.show()  # does not block. This function does not accept block argument.
        else:
            plt.show(block=True)

        return fig, ax

    def plotGradNMag(self, fig=None, ax=None, block=False):
        """
        Plot a 2D slice of index gradient. Currently only for real part. Future extenstion: for both its real and imaginary parts
        """
        if "presampled_x" not in self.__dict__:
            x_sample = self.getPositionVectorsAtGridPoints()
        else:
            x_sample = self.presampled_x

        grad_n = self.grad_n(x_sample)
        grad_n = grad_n.reshape(
            (self.xg.shape[0], self.xg.shape[1], self.xg.shape[2], 3)
        )
        grad_n_slice = grad_n[:, :, grad_n.shape[2] // 2, :].cpu().numpy()

        grad_n_slice_mag = np.sqrt(
            grad_n_slice[:, :, 0] ** 2
            + grad_n_slice[:, :, 1] ** 2
            + grad_n_slice[:, :, 2] ** 2
        )

        if ax == None:
            fig, ax = plt.subplots(nrows=2, ncols=2)
        plt_mag = ax[0, 0].imshow(
            grad_n_slice_mag,
            cmap="gray",
            vmin=np.amin(grad_n_slice_mag),
            vmax=np.amax(grad_n_slice_mag),
        )
        plt_x = ax[0, 1].imshow(
            grad_n_slice[:, :, 0],
            cmap="bwr",
            vmin=np.amin(grad_n_slice[:, :, 0]),
            vmax=np.amax(grad_n_slice[:, :, 0]),
        )
        plt_y = ax[1, 0].imshow(
            grad_n_slice[:, :, 1],
            cmap="bwr",
            vmin=np.amin(grad_n_slice[:, :, 1]),
            vmax=np.amax(grad_n_slice[:, :, 1]),
        )
        plt_z = ax[1, 1].imshow(
            grad_n_slice[:, :, 2],
            cmap="bwr",
            vmin=np.amin(grad_n_slice[:, :, 2]),
            vmax=np.amax(grad_n_slice[:, :, 2]),
        )

        ax[0, 0].set_title("Magnitude of index gradient")
        ax[0, 0].set_xlabel("x")
        ax[0, 0].set_ylabel("y")
        plt.colorbar(plt_mag)

        ax[0, 1].set_title("x component of index gradient")
        ax[0, 1].set_xlabel("x")
        ax[0, 1].set_ylabel("y")
        plt.colorbar(plt_x)

        ax[1, 0].set_title("y component of index gradient")
        ax[1, 0].set_xlabel("x")
        ax[1, 0].set_ylabel("y")
        plt.colorbar(plt_y)

        ax[1, 1].set_title("z component of index gradient")
        ax[1, 1].set_xlabel("x")
        ax[1, 1].set_ylabel("y")
        plt.colorbar(plt_z)

        if block == False:
            fig.show()  # does not block. This function does not accept block argument.
        else:
            plt.show(block=True)

        return fig, ax

    # def plotGradNArrow(self, fig = None, ax = None, lb = 0, ub = 1, n_pts=512, block=False):
    #     '''
    #     Plot a 2D slice of index gradient in sampled arrow plot. Currently only for real part. Future extenstion: for both its real and imaginary parts
    #     '''

    # grad_n_slice = self.presampled_vector_field_4D[:,:,self.presampled_vector_field_4D.shape[2]//2,:].cpu().numpy()

    # grad_n_slice_mag = np.sqrt(grad_n_slice[:,0]**2 + grad_n_slice[:,1]**2 + grad_n_slice[:,2]**2)

    # # vmin, vmax = np.amin(n_slice), np.amax(n_slice)

    # if ax == None:
    #     fig, ax =  plt.subplots()
    # # ax.quiver(self.presampled_x[relevant_points,0],
    # # x0[relevant_points,1],
    # # x0[relevant_points,2],
    # # v0[relevant_points,0],
    # # v0[relevant_points,1],
    # # v0[relevant_points,2],
    # # length=0.15,
    # # color = color)
    # # mappable = ax.imshow(n_slice, cmap='gray', vmin=vmin, vmax=vmax)
    # ax.set_title('Index distribution gradient')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # # plt.colorbar(mappable)

    # if block == False:
    #     fig.show() #does not block. This function does not accept block argument.
    # else:
    #     plt.show(block=True)

    # return fig, ax

    def plotRandomlySampledIndex(self, pts=500, fig=None, ax=None, block=False):
        """
        Create a scatter plot of index at random positions. Color value is proportional to index.
        """
        random_position = torch.rand(pts, 3, device=self.device)

        random_position[:, 0] = (random_position[:, 0] - 0.5) * 2 * torch.amax(self.xv)
        random_position[:, 1] = (random_position[:, 1] - 0.5) * 2 * torch.amax(self.yv)
        random_position[:, 2] = (random_position[:, 2] - 0.5) * 2 * torch.amax(self.zv)

        self.plotIndexAtPosition(x=random_position, fig=fig, ax=ax, block=block)

    def plotIndexAtPosition(
        self, x, fig=None, ax=None, block=False, cmap="viridis", marker="o"
    ):
        """
        Create a scatter plot of index at specified positions x. Color value is proportional to index.
        """
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x)

        n = self.n(x)
        n = n.cpu().numpy()
        x = x.cpu().numpy()

        if ax == None:
            # fig, ax =  plt.subplots()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        mappable = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=n, cmap=cmap, marker=marker)
        ax.set_title("Index distribution")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        plt.colorbar(mappable)

        if block == False:
            fig.show()  # does not block. This function does not accept block argument.
        else:
            plt.show(block=True)


class AttenuationModel(MediumModel):
    _default_analytical = {
        "R": 1.0,
        "length_unit_of_R": "fractional_domain_radius",
        "alpha_internal": 1e-3,
    }  # default arguments for **kwargs
    _default_interpolation = {"interpolation_padding_mode": "zeros"}

    @torch.inference_mode()
    def __init__(
        self,
        coord_vec: np.ndarray,
        type: str = "analytical",
        form: str = "homogeneous_cylinder",
        **kwargs,
    ):
        """
        Analytical and interpolated decription of attenuation coefficient (and similar quantities) of the simulation domain.
        The class can describe any one of the following:
        1. (total/component) attenuation coefficient,
        2. (total/component) absorption coefficient, or
        3. (total/component) scattering coefficient.

        Typical use of this class: Users can describe the total attenuation of the medium with one class instance and the absorption of the active component with another instance.

        The generic scalar quantity (a function of position x) is referred to as 'alpha' in function name and function arguments.
        Alpha has unit of [1/(length unit)], where length units is the unit used in coord_vec.

        Input x should have shape [num_of_sample, 3]. And the output of alpha would have shape [num_of_sample].
        Outside to the simulation bounding box, alpha is always assumed zero because the external geometry is not defined.

        Internally implemented with pyTorch tensor for GPU computation and compatibility with pyTorchRayTrace.
        Input/output compatibility with numpy array is to be added.

        Parameters
        ----------
        type : str ('analytical', 'interpolation')
            Select analytical function evaluation or interpolate on pre-built interpolant arrays.
            Interpolation method handles edge cases of input explicitly and hence is more robust.

        form : str ('homogeneous_cylinder', 'homogeneous_ball', 'freeform')

        R : float, optional
            Radius of the cylinder or ball
            Default equals to the radius of the grid (cylinder/ball occupy almost all of simulation volume), regardless of length_unit parameter.

        length_unit: str, optional, ('fractional_domain_radius', 'physical_coordinate')
            This unit is only used in specification of R (radius of cylinder/ball). The size of computational grid is always assumed to be the same as input coordinate vectors.
            Default: 'fractional_domain_radius', where 1 represent a cylinder/ball filling the domain with boundaries touching
            'physical_coordinate' uses the same unit as the input coordinate vectors (coord_vec, the first argument).

        alpha_internal : float, [1/(grid length unit)], which is commonly [1/cm].
            Value of alpha inside the homogeneous cylinder or ball (at the wavelength of interest)

        """

        super().__init__(coord_vec)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Medium computation is performed on: {repr(self.device)}")

        # Save inputs as attributes
        self.type = type
        self.form = form

        # Merge all kwargs into params
        self.params = (
            self._default_analytical.copy()
        )  # Shallow copy avoid editing dict '_default_lenses' in place
        self.params.update(self._default_interpolation)  # Add relevant parameters
        self.params.update(kwargs)  # up-to-date parameters. Default dict is not updated

        # Compute derived attributes R_physical
        if self.params["length_unit_of_R"] == "fractional_domain_radius":
            R_domain_0 = (np.amax(coord_vec[0]) - np.amin(coord_vec[0])) / 2
            R_domain_1 = (np.amax(coord_vec[1]) - np.amin(coord_vec[1])) / 2
            R_domain_2 = (np.amax(coord_vec[2]) - np.amin(coord_vec[2])) / 2
            if coord_vec[2].size == 1:  # only fit in xy box in 2D case
                R_domain_min = min(R_domain_0, R_domain_1)
            else:  # fit in xyz box in 3D case
                R_domain_min = min(R_domain_0, R_domain_1, R_domain_2)
            self.params["R_physical"] = self.params["R"] * R_domain_min

        elif self.params["length_unit_of_R"] == "physical_coordinate":
            self.params["R_physical"] = self.params["R"]

        else:
            raise Exception(
                '"length_unit" should be either "fractional_domain_radius" or "physical_coordinate"'
            )

        # ==========================Setup index query functions according to type and form================================
        if self.type == "analytical":
            if self.form == "homogeneous_cylinder":
                self.alpha = self._alpha_homo_cylinder

            elif self.form == "homogeneous_ball":
                self.alpha = self._alpha_homo_ball

            else:
                raise Exception(
                    "Form: Other analytical functions are not supported yet."
                )

        elif self.type == "interpolation":
            # Interpolation method stores a 3D array as interpolant and query them upon each mapping call.
            # Stored arrays : alpha (a scalar field).
            # It is sampled at the grid defined by coord_vec (presampled_x, which is also stored).

            # function alias
            self.alpha = self._scalar_field_interp
            # This setting is used in _scalar_field_interp and _vector_field_interp.
            self.interpolation_padding_mode = self.params["interpolation_padding_mode"]  # type: ignore
            self.presampled_x = self.getPositionVectorsAtGridPoints()

            # build or import interpolant dataset
            if self.form == "homogeneous_cylinder":
                # build interpolant arrays
                self.presampled_scalar_field_3D = self._alpha_homo_cylinder(
                    self.presampled_x
                ).reshape(self.xg.shape)

            elif self.form == "homogeneous_ball":
                # build interpolant arrays
                self.presampled_scalar_field_3D = self._alpha_homo_ball(
                    self.presampled_x
                ).reshape(
                    self.xg.shape
                )  # Save as a 3D tensor

            elif self.form == "freeform":  # Directly import data instead of generating.
                # Input data points are designated with sample subscript
                self.presampled_scalar_field_3D = self.params.get("alpha_x", None)  # type: ignore

            else:
                raise Exception("Other interpolation functions are not supported yet.")

            self._formatPresampledScalarField()  # Format presampled_scalar_field_3D into presampled_scalar_field_5D and store the latter as object attribute.
        else:
            raise Exception('"type" should be either "analytical" or "interpolation".')

    # =================================Analytic: Homogeneous cylinder================================================
    @torch.inference_mode()
    def _alpha_homo_cylinder(self, x: Tensor):
        """Constant absorption coefficient in a cylinder centered at origin. Radius determined by the class initialization variable R."""
        r = torch.linalg.norm(
            x[:, 0:2], dim=1
        )  # radial position on xy plane = torch.sqrt(x[:,0]**2 + x[:,1]**2), 1D tensor, numel = x.size[0], Note: x[:,0:2] exclude x[:,2]
        alpha = torch.zeros_like(r)  # 1D tensor, numel = x.size[0]
        # Inside the cylinder
        alpha[r < self.params["R_physical"]] = self.params["alpha_internal"]  # type: ignore
        return alpha

    # =================================Analytic: Homogeneous ball================================================
    @torch.inference_mode()
    def _alpha_homo_ball(self, x: Tensor):
        """Constant absorption coefficient in a ball centered at origin. Radius determined by the class initialization variable R."""
        r = torch.linalg.norm(
            x, dim=1
        )  # radial position in xyz = torch.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2), 1D tensor, numel = x.size[0]
        alpha = torch.zeros_like(r)  # 1D tensor, numel = x.size[0]
        # Inside the cylinder
        alpha[r < self.params["R_physical"]] = self.params["alpha_internal"]  # type: ignore
        return alpha

    # =================================Utilities==========================================================================
    def plotAlpha(self, fig=None, ax=None, block=False):
        """
        Plot a 2D slice of index. Currently only for real part. Future extenstion: for both its real and imaginary parts
        """
        if "presampled_x" not in self.__dict__:
            x_sample = self.getPositionVectorsAtGridPoints()
        else:
            x_sample = self.presampled_x

        alpha = self.alpha(x_sample).reshape(self.xg.shape)
        # n_slice = self.presampled_n_x[:,:,self.presampled_n_x.shape[2]//2].cpu().numpy()
        alpha_slice = alpha[:, :, alpha.shape[2] // 2].cpu().numpy()

        vmin, vmax = np.amin(alpha_slice), np.amax(alpha_slice)

        if ax == None:
            fig, ax = plt.subplots()

        mappable = ax.imshow(alpha_slice, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title("Alpha distribution")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(mappable)

        if block == False:
            fig.show()  # does not block. This function does not accept block argument.
        else:
            plt.show(block=True)

        return fig, ax

    def plotRandomlySampledAlpha(self, pts=500, fig=None, ax=None, block=False):
        """
        Create a scatter plot of index at random positions. Color value is proportional to alpha.
        """
        random_position = torch.rand(pts, 3, device=self.device)

        random_position[:, 0] = (random_position[:, 0] - 0.5) * 2 * torch.amax(self.xv)
        random_position[:, 1] = (random_position[:, 1] - 0.5) * 2 * torch.amax(self.yv)
        random_position[:, 2] = (random_position[:, 2] - 0.5) * 2 * torch.amax(self.zv)

        self.plotAlphaAtPosition(x=random_position, fig=fig, ax=ax, block=block)

    def plotAlphaAtPosition(
        self, x, fig=None, ax=None, block=False, cmap="viridis", marker="o"
    ):
        """
        Create a scatter plot of index at specified positions x. Color value is proportional to index.
        """
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x)

        alpha = self.alpha(x)
        alpha = alpha.cpu().numpy()
        x = x.cpu().numpy()

        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        mappable = ax.scatter(
            x[:, 0], x[:, 1], x[:, 2], c=alpha, cmap=cmap, marker=marker
        )
        ax.set_title("Alpha distribution")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        plt.colorbar(mappable)

        if block == False:
            fig.show()  # does not block. This function does not accept block argument.
        else:
            plt.show(block=True)


class AbsorptionModel(AttenuationModel):
    def __init__(self, *args, **kwargs):
        """Class alias. Identical to AttenuationModel."""
        super().__init__(*args, **kwargs)


class ScatteringModel(AttenuationModel):
    def __init__(self, *args, **kwargs):
        """Subclass of AttenuationModel. Reserved for future development."""
        super().__init__(*args, **kwargs)
