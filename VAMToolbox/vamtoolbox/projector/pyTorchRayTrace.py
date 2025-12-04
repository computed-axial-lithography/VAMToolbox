# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the GNU GPLv3 license.

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import torch

from vamtoolbox.util.data import filterSinogram  # only required for inverse_backward


class PyTorchRayTracingPropagator:
    """
    pyTorch implementation of custom ray tracing operation
    This operator is the A in Ax = b, or the P in Pf = g, where x or f is the real space object (flattened into 1D array) and b or g is the sinogram (flattened into 1D array).
    Here naming convention Pf = g is used to avoid collision with x which commonly denote position of ray in ray tracing literature.

    Since storing all curved ray path is prohibitively memory-intensive, this tracing operation is designed to have as low memory requirement as possible.
    Compute on GPU by default but fallback to CPU computation if GPU is not found.
    Currently occulsion is not supported yet.

    Internally, 2D and 3D problem is handled identically. Conditional handling only happens in I/O.
    #Implement ray_setup, coordinate systems
    """

    # pyTorch is used in inference mode with @torch.inference_mode() decorator.
    # Alternatively, @torch.no_grad() and torch.autograd.set_grad_enabled(False) can be used.
    # Currently @torch.inference_mode() decorator is applied to most functions but it is only strictly required on the outermost functions.

    @torch.inference_mode()
    def __init__(self, target_geo, proj_geo, output_torch_tensor=False) -> None:
        self.logger = logging.getLogger(__name__)

        self.target_geo = target_geo
        self.proj_geo = proj_geo
        self.output_torch_tensor = output_torch_tensor  # Select if the output should be torch tensor or numpy array

        self.domain_n_dim = len(
            np.squeeze(self.target_geo.array).shape
        )  # Dimensionality of the domain is set to be same as target
        self.tracing_step_size = (
            self.proj_geo.index_model.voxel_size[0].cpu().numpy() / 2
        )  # Assuming the step size (in case of physical step distance) is about half the voxel.
        self.max_num_step = self.target_geo.array.shape[0] * (
            2 * 1.5
        )  # 2 is the assumed (voxel size/step size) ratio, 1.5 is safety margin

        # Check if GPU is detected. Fallback to CPU if GPU is not found.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.logger.info(f"Ray tracing computation is performed on: {repr(self.device)}")

        if isinstance(
            self.proj_geo.ray_trace_ray_config, RayState
        ):  # if state of a set of rays is already provided, use that as initial state.
            self.ray_state = self.proj_geo.ray_trace_ray_config
        else:  # otherwise build initial ray state based on option 'parallel', 'cone'
            self.ray_state = RayState(device=self.device)
            self.ray_state.setupRays(
                self.proj_geo.ray_trace_ray_config,
                self.target_geo.constructCoordVec(),
                self.proj_geo.angles,
                self.proj_geo.inclination_angle,
                ray_density=self.proj_geo.ray_density,
                ray_width_fun=None,
            )

        # Assume a model if index_model, attenuation_model, absorption_model is None. Need to import vamtoolbox.medium
        # self.target_geo.constructCoordVec() should be used as the master coordinate system

        self.solver = RayTraceSolver(
            self.device,
            self.proj_geo.index_model,
            self.proj_geo.attenuation_model,
            self.proj_geo.absorption_model,
            self.proj_geo.ray_trace_method,
            self.proj_geo.eikonal_parametrization,
            self.proj_geo.ray_trace_ode_solver,
            max_num_step=self.max_num_step,
        )

    @torch.inference_mode()
    def forward(self, f):
        # Convert the input to a torch tensor if it is not
        if ~isinstance(f, torch.Tensor):  # attempt to convert input to tensor
            f = torch.as_tensor(f, device=self.device)

        self.ray_state.resetRaysIterateToInitial()

        self.ray_state, _ = self.solver.integrateEnergyUntilExit(
            ray_state=self.ray_state,
            step_size=self.tracing_step_size,
            real_space_distribution=f,
            tracker_on=False,
            track_every=1,
        )

        # Convert output to nparray if needed
        if self.output_torch_tensor:
            return self.ray_state.integral
        else:
            return self.ray_state.integral.cpu().numpy()

    @torch.inference_mode()
    def backward(self, ray_energy_0):
        # Check if input is tensor
        if ~isinstance(ray_energy_0, torch.Tensor):  # attempt to convert input to tensor
            ray_energy_0 = torch.as_tensor(
                ray_energy_0, device=self.device, dtype=self.ray_state.tensor_dtype
            )
        # Optional: if input is sparse, use ray selector to filter out rays corresponding to zero intensity pixels

        self.ray_state.resetRaysIterateToInitial()
        self.ray_state.ray_energy_0 = ray_energy_0

        deposition_grid, self.ray_state, _ = self.solver.depositEnergyUntilExit(
            ray_state=self.ray_state,
            step_size=self.tracing_step_size,
            tracker_on=False,
            track_every=1,
        )

        # Convert output to nparray if needed
        if self.output_torch_tensor:
            return deposition_grid
        else:
            # maintain backward compatibility with optimizers written in numpy, which assumes 2D real space quantities to have no third dimension
            return np.squeeze(deposition_grid.cpu().numpy())

    @torch.inference_mode()
    def inverseBackward(self, f, filter_option="ram-lak", **kwargs):
        """This function compute the inverse of backpropagation. Similar to forward propagation, this function maps from a spatial quantity to a sinogram quantity."""
        # Convert the input to a torch tensor if it is not
        if ~isinstance(f, torch.Tensor):  # attempt to convert input to tensor
            f = torch.as_tensor(f, device=self.device)

        absorption_at_target_grid = self.proj_geo.absorption_model.alpha(
            self.proj_geo.index_model.getPositionVectorsAtGridPoints()
        ).reshape(
            self.target_geo.array.shape
        )  # Multiply with the alpha values exactly at the grid points.
        f /= absorption_at_target_grid**2.0
        f[torch.isnan(f) | torch.isinf(f)] = 0.0

        g0 = self.ray_state.reshape(
            self.forward(f)
        )  # reshape into cylindrical grid for filtering operation
        if isinstance(
            g0, torch.Tensor
        ):  # if forward returned a tensor, convert to numpy array
            g0 = g0.cpu().numpy()

        g0 = filterSinogram(
            g0, filter_option
        ).ravel()  # pyTorchRayTracingPropagator always return 1D vector
        g0 *= np.pi / (2 * self.proj_geo.n_angles)

        if self.output_torch_tensor:
            return torch.as_tensor(g0, device=self.device)
        else:
            return g0

    @torch.inference_mode()
    def buildPropagationMatrix(self):
        """
        This function builds the forward propagation matrix that corresponds to the forward propagation operation in discrete form.
        This operator is the A in Ax = b, or the P in Pf = g, where x or f is the real space object (flattened into 1D array) and b or g is the sinogram (flattened into 1D array).
        This propagation matrix is returned as a pytorch COO tensor if "output_torch_tensor" attribute is True, and as a scipy sparse COO array otherwise.
        """
        self.ray_state.resetRaysIterateToInitial()
        propagation_matrix_coo, _, _ = self.solver.recordEnergyUntilExit(
            ray_state=self.ray_state,
            step_size=self.tracing_step_size,
            tracker_on=False,
            track_every=1,
        )

        propagation_matrix_coo.coalesce()

        # Output as a torch sparse tensor
        if self.output_torch_tensor:
            return propagation_matrix_coo
        else:  # Convert output to class scipy.sparse.coo_array if needed
            indices = propagation_matrix_coo.indices().cpu().numpy()
            values = propagation_matrix_coo.values().cpu().numpy()
            return scipy.sparse.coo_array(
                (values, (indices[0, :], indices[1, :])),
                shape=(
                    propagation_matrix_coo.size(dim=0),
                    propagation_matrix_coo.size(dim=1),
                ),
            )

    @torch.inference_mode()
    def perAngleTrace(self):
        pass

    @torch.inference_mode()
    def perZTrace(self):
        pass

    @torch.inference_mode()
    def perRayTrace(self):
        pass


class RayTraceSolver:
    """
    ODE Solver to iterate over ray operations

    Store defined step size, selected stepping scheme

    index and its gradient field

    """

    @torch.inference_mode()
    def __init__(
        self,
        device,
        index_model,
        attenuation_model,
        absorption_model,
        ray_trace_method,
        eikonal_parametrization,
        ode_solver,
        max_num_step=750,
        num_step_per_exit_check=10,
    ) -> None:
        self.logger = logging.getLogger(
            __name__
        )  # Inputting the same name just output the same logger. No need to pass logger around.
        self.device = device
        self.index_model = index_model
        self.attenuation_model = attenuation_model
        self.absorption_model = absorption_model
        self.ray_trace_method = ray_trace_method
        self.eikonal_parametrization = eikonal_parametrization
        self.ode_solver = ode_solver

        self.max_num_step = max_num_step
        self.num_step_per_exit_check = num_step_per_exit_check

        # Parse ray_trace_method
        if self.ray_trace_method == "snells" or self.ray_trace_method == "hybrid":
            self.surface_intersection_check = True
        else:
            self.surface_intersection_check = False

        if self.ray_trace_method == "eikonal" or self.ray_trace_method == "hybrid":
            # Eikonal equation parametrization
            if self.eikonal_parametrization == "canonical":
                # Canaonical variable in Hamiltonian equation. Reference: Adjoint Nonlinear Ray Tracing, Arjun Teh, Matthew O'Toole, Ioannis Gkioulekas ACM Trans. Graph., Vol. 41, No. 4, Article 126. Publication date: July 2022.
                self.dx_dstep = self._dx_dsigma
                self.dv_dstep = self._dv_dsigma
            elif self.eikonal_parametrization == "physical_path_length":
                # Physical path length
                self.dx_dstep = self._dx_ds
                self.dv_dstep = self._dv_ds
            elif self.eikonal_parametrization == "optical_path_length":
                # Optical path length
                self.dx_dstep = self._dx_dopl
                self.dv_dstep = self._dv_dopl
            else:
                raise Exception(
                    f"eikonal_parametrization = {self.eikonal_parametrization} and is not one of the available options."
                )

        # Parse ODE solver
        if self.ode_solver == "forward_symplectic_euler":
            self.step_init = self._no_step_init
            self.step = self._forwardSymplecticEuler

        elif self.ode_solver == "forward_euler":
            self.step_init = self._no_step_init
            self.step = self._forwardEuler

        elif self.ode_solver == "leapfrog":
            # leapfrog is identical to velocity verlet. Both differs from symplectic euler only by a half step in velocity in the beginning and the end.
            # However, they have higher order error performance
            self.step_init = self._leapfrog_init
            self.step = self._forwardSymplecticEuler

        elif (
            self.ode_solver == "rk4"
        ):  # Placeholder for future implementation of Fourth Order Runge-Kutta.
            pass  # self.step = self._rk4
        else:
            raise Exception(
                f"ode_solver = {self.ode_solver} and is not one of the available options."
            )

    @torch.inference_mode()
    def solveUntilExit(
        self, ray_state, step_size, callback=None, tracker_on=False, track_every=5
    ):
        # Initialize ray tracker, where consecutive ray positions are recorded. Default to track every 5 rays
        if tracker_on:
            ray_tracker = (
                torch.nan
                * torch.zeros(
                    (((ray_state.num_rays - 1) // track_every) + 1, 3, self.max_num_step),
                    device=self.device,
                    dtype=ray_state.x_0.dtype,
                )
            )  # ray_state.x_i[::track_every, :] has number of rows equal (ray_state.num_rays-1)//track_every)+1
            tracker_ind = torch.zeros_like(ray_state.active)
            tracker_ind[::track_every] = True
        else:
            ray_tracker = None

        # Initialize stepping (only applies to certain methods, e.g. leapfrog)
        ray_state = self.step_init(ray_state, step_size)

        self.step_counter = 0
        start_time = time.perf_counter()
        ray_state.active = (
            ray_state.active & ~ray_state.exited
        )  # only non exited rays are considered
        initial_active_ray_count = torch.sum(ray_state.active, dim=0)  # initially active
        active_ray_count = torch.sum(ray_state.active, dim=0)  # current active ray count
        active_ray_exist = active_ray_count > 0

        while (self.step_counter < self.max_num_step) and active_ray_exist:
            ray_state = self.step(
                ray_state, step_size
            )  # step forward. Where x_i and v_i will be replaced by x_ip1 and v_ip1 respectively

            # if self.surface_intersection_check: #Placeholder: Exception handling for discrete surfaces (This feature is to be implemented)
            #     self.discreteSurfaceIntersectionCheck(ray_state)

            if tracker_on:
                tracker_active = ray_state.active[
                    ::track_every
                ]  # tracker_active.numel = ray_tracker.shape[0]
                active_and_tracked = (
                    ray_state.active & tracker_ind
                )  # active_and_tracked.numel = x_i.shape[0]
                ray_tracker[tracker_active, :, self.step_counter] = ray_state.x_i[
                    active_and_tracked, :
                ]

            if (
                self.step_counter % self.num_step_per_exit_check == 0
            ):  # Check exit condition
                ray_state = self.exitCheck(
                    ray_state
                )  # Note: The operating set is always inverted ray_state.exited
                ray_state.active = (
                    ray_state.active & ~ray_state.exited
                )  # deactivate exited rays
                active_ray_count = torch.sum(
                    ray_state.active, dim=0
                )  # current active ray count
                active_ray_exist = active_ray_count > 0
                self.logger.debug(
                    f"Completed {self.step_counter}th-step at time {time.perf_counter() - start_time}. Ray active/init active/all: {active_ray_count}/{initial_active_ray_count}/{ray_state.num_rays}"
                )

            self.step_counter += 1

        self.total_stepping = (
            step_size * self.step_counter
        )  # Total stepped value in parametrization variable

        if active_ray_exist:
            self.logger.error(
                f"Some rays are still active when max_num_step ({self.max_num_step}) is reached. Rays active: {active_ray_count}/{initial_active_ray_count}."
            )

        return ray_state, ray_tracker

    @torch.inference_mode()
    def depositEnergyUntilExit(
        self, ray_state, step_size, callback=None, tracker_on=False, track_every=5
    ):
        # Initialize ray tracker, where consecutive ray positions are recorded. Default to track every 5 rays
        if tracker_on:
            ray_tracker = (
                torch.nan
                * torch.zeros(
                    (((ray_state.num_rays - 1) // track_every) + 1, 3, self.max_num_step),
                    device=self.device,
                    dtype=ray_state.x_0.dtype,
                )
            )  # ray_state.x_i[::track_every, :] has number of rows equal (ray_state.num_rays-1)//track_every)+1
            tracker_ind = torch.zeros_like(ray_state.active)
            tracker_ind[::track_every] = True
        else:
            ray_tracker = None

        # Initialize stepping (only applies to certain methods, e.g. leapfrog)
        ray_state = self.step_init(ray_state, step_size)

        # Initialize varaibles
        self.step_counter = 0  # remember how many step taken after solving
        start_time = time.perf_counter()
        ray_state.active = (
            ray_state.active & ~ray_state.exited
        )  # only non exited rays are considered
        initial_active_ray_count = torch.sum(ray_state.active, dim=0)  # initially active
        active_ray_count = torch.sum(ray_state.active, dim=0)  # current active ray count
        active_ray_exist = active_ray_count > 0
        desposition_grid = torch.zeros_like(
            self.index_model.xg
        )  # This grid is where the energy accumulates

        while (self.step_counter < self.max_num_step) and active_ray_exist:
            ray_state = self.step(
                ray_state, step_size
            )  # step forward. Where x_i and v_i will be replaced by x_ip1 and v_ip1 respectively

            # if self.surface_intersection_check: #Placeholder: Exception handling for discrete surfaces (This feature is to be implemented)
            #     self.discreteSurfaceIntersectionCheck(ray_state)
            desposition_grid = self.deposit(ray_state, step_size, desposition_grid)

            if tracker_on:
                tracker_active = ray_state.active[
                    ::track_every
                ]  # tracker_active.numel = ray_tracker.shape[0]
                active_and_tracked = (
                    ray_state.active & tracker_ind
                )  # active_and_tracked.numel = x_i.shape[0]
                ray_tracker[tracker_active, :, self.step_counter] = ray_state.x_i[
                    active_and_tracked, :
                ]

            if (
                self.step_counter % self.num_step_per_exit_check == 0
            ):  # Check exit condition
                ray_state = self.exitCheck(
                    ray_state
                )  # Note: The operating set is always inverted ray_state.exited
                ray_state.active = (
                    ray_state.active & ~ray_state.exited
                )  # deactivate exited rays
                active_ray_count = torch.sum(
                    ray_state.active, dim=0
                )  # current active ray count
                active_ray_exist = active_ray_count > 0
                self.logger.debug(
                    f"Completed {self.step_counter}th-step at time {time.perf_counter() - start_time}. Ray active/init active/all: {active_ray_count}/{initial_active_ray_count}/{ray_state.num_rays}"
                )

            self.step_counter += 1

        self.total_stepping = (
            step_size * self.step_counter
        )  # Total stepped value in parametrization variable

        # Take into account spatial varying absorption coefficient. The resulted quantity has an additional (length unit^-1). (e.g. It can convert sum of intensities into volumetric dose.)
        # Although this step can be done in the innermost level, it is performed at this higher level for performance.
        desposition_grid *= self.absorption_model.alpha(
            self.index_model.getPositionVectorsAtGridPoints()
        ).reshape(
            desposition_grid.shape
        )  # Multiply with the alpha values exactly at the grid points.

        if active_ray_exist:
            self.logger.error(
                f"Some rays are still active when max_num_step ({self.max_num_step}) is reached. Rays active: {active_ray_count}/{initial_active_ray_count}."
            )

        return desposition_grid, ray_state, ray_tracker

    @torch.inference_mode()
    def integrateEnergyUntilExit(
        self,
        ray_state,
        step_size,
        real_space_distribution,
        callback=None,
        tracker_on=False,
        track_every=5,
    ):
        # Initialize ray tracker, where consecutive ray positions are recorded. Default to track every 5 rays
        if tracker_on:
            ray_tracker = (
                torch.nan
                * torch.zeros(
                    (((ray_state.num_rays - 1) // track_every) + 1, 3, self.max_num_step),
                    device=self.device,
                    dtype=ray_state.x_0.dtype,
                )
            )  # ray_state.x_i[::track_every, :] has number of rows equal (ray_state.num_rays-1)//track_every)+1
            tracker_ind = torch.zeros_like(ray_state.active)
            tracker_ind[::track_every] = True
        else:
            ray_tracker = None

        # Initialize stepping (only applies to certain methods, e.g. leapfrog)
        ray_state = self.step_init(ray_state, step_size)

        # Initialize varaibles
        self.step_counter = 0  # remember how many step taken after solving
        start_time = time.perf_counter()
        ray_state.active = (
            ray_state.active & ~ray_state.exited
        )  # only non exited rays are considered
        initial_active_ray_count = torch.sum(ray_state.active, dim=0)  # initially active
        active_ray_count = torch.sum(ray_state.active, dim=0)  # current active ray count
        active_ray_exist = active_ray_count > 0
        ray_state.integral = torch.zeros_like(
            ray_state.integral
        )  # Reset the integral values before integrating. This step is redundent when used when integrateEnergyUntilExit() is called by PyTorchRayTracingPropagator.forward() since it is resetted outside.
        real_space_distribution = torch.atleast_3d(real_space_distribution)

        # Take into account spatial varying absorption coefficient. The resulted quantity has an additional (length unit^-1). (e.g. It can convert sum of intensities into volumetric dose.)
        # Although this step can be done in the innermost level, it is performed at this higher level for performance.
        real_space_distribution *= self.absorption_model.alpha(
            self.index_model.getPositionVectorsAtGridPoints()
        ).reshape(
            real_space_distribution.shape
        )  # Multiply with the alpha values exactly at the grid points.

        while (self.step_counter < self.max_num_step) and active_ray_exist:
            ray_state = self.step(
                ray_state, step_size
            )  # step forward. Where x_i and v_i will be replaced by x_ip1 and v_ip1 respectively

            # if self.surface_intersection_check: #Placeholder: Exception handling for discrete surfaces (This feature is to be implemented)
            #     self.discreteSurfaceIntersectionCheck(ray_state)
            ray_state = self.integrate(
                ray_state, step_size, real_space_distribution
            )  # ++++++++++++++++++++++++++++++++

            if tracker_on:
                tracker_active = ray_state.active[
                    ::track_every
                ]  # tracker_active.numel = ray_tracker.shape[0]
                active_and_tracked = (
                    ray_state.active & tracker_ind
                )  # active_and_tracked.numel = x_i.shape[0]
                ray_tracker[tracker_active, :, self.step_counter] = ray_state.x_i[
                    active_and_tracked, :
                ]

            if (
                self.step_counter % self.num_step_per_exit_check == 0
            ):  # Check exit condition
                ray_state = self.exitCheck(
                    ray_state
                )  # Note: The operating set is always inverted ray_state.exited
                ray_state.active = (
                    ray_state.active & ~ray_state.exited
                )  # deactivate exited rays
                active_ray_count = torch.sum(
                    ray_state.active, dim=0
                )  # current active ray count
                active_ray_exist = active_ray_count > 0
                self.logger.debug(
                    f"Completed {self.step_counter}th-step at time {time.perf_counter() - start_time}. Ray active/init active/all: {active_ray_count}/{initial_active_ray_count}/{ray_state.num_rays}"
                )

            self.step_counter += 1

        self.total_stepping = (
            step_size * self.step_counter
        )  # Total stepped value in parametrization variable

        if active_ray_exist:
            self.logger.error(
                f"Some rays are still active when max_num_step ({self.max_num_step}) is reached. Rays active: {active_ray_count}/{initial_active_ray_count}."
            )

        return ray_state, ray_tracker

    @torch.inference_mode()
    def recordEnergyUntilExit(
        self, ray_state, step_size, tracker_on=False, track_every=5
    ):
        """This function record the energy deposited to voxels by rays of unity energy. The result is recorded as a pyTorch sparse COO tensor."""
        # Initialize ray tracker, where consecutive ray positions are recorded. Default to track every 5 rays
        if tracker_on:
            ray_tracker = (
                torch.nan
                * torch.zeros(
                    (((ray_state.num_rays - 1) // track_every) + 1, 3, self.max_num_step),
                    device=self.device,
                    dtype=ray_state.x_0.dtype,
                )
            )  # ray_state.x_i[::track_every, :] has number of rows equal (ray_state.num_rays-1)//track_every)+1
            tracker_ind = torch.zeros_like(ray_state.active)
            tracker_ind[::track_every] = True
        else:
            ray_tracker = None

        # ray_state should be resetted in buildPropagationMatrix(). Repeated here as a failsafe when recordEnergyUntilExit() is called outside of buildPropagationMatrix().
        ray_state.resetRaysIterateToInitial()

        # Initialize stepping (only applies to certain methods, e.g. leapfrog)
        ray_state = self.step_init(ray_state, step_size)

        # Initialize varaibles
        self.step_counter = 0  # remember how many step taken after solving
        start_time = time.perf_counter()
        ray_state.active = (
            ray_state.active & ~ray_state.exited
        )  # only non exited rays are considered
        initial_active_ray_count = torch.sum(ray_state.active, dim=0)  # initially active
        active_ray_count = torch.sum(ray_state.active, dim=0)  # current active ray count
        active_ray_exist = active_ray_count > 0

        # Get the sparse tensor size
        n_rows = ray_state.num_rays  # currently use number of rays as the number of sinogram elements. Supersampling (where num_rays > number of sinogram elements) is to be supported in the future.
        n_cols = self.index_model.xg.numel()  # number of voxel elements
        propagation_matrix_shape = (n_rows, n_cols)
        propagation_matrix = torch.sparse_coo_tensor(
            size=propagation_matrix_shape, device=self.device
        )  # initialize empty propagation matrix

        while (self.step_counter < self.max_num_step) and active_ray_exist:
            ray_state = self.step(
                ray_state, step_size
            )  # step forward. Where x_i and v_i will be replaced by x_ip1 and v_ip1 respectively

            # if self.surface_intersection_check: #Placeholder: Exception handling for discrete surfaces (This feature is to be implemented)
            #     self.discreteSurfaceIntersectionCheck(ray_state)
            propagation_matrix += self.record(
                ray_state, step_size, propagation_matrix_shape
            )  # record function returns the ray contribution in current step only
            if self.step_counter % 50 == 0:
                propagation_matrix = (
                    propagation_matrix.coalesce()
                )  # coalesce to remove redundent indices and values

            if tracker_on:
                tracker_active = ray_state.active[
                    ::track_every
                ]  # tracker_active.numel = ray_tracker.shape[0]
                active_and_tracked = (
                    ray_state.active & tracker_ind
                )  # active_and_tracked.numel = x_i.shape[0]
                ray_tracker[tracker_active, :, self.step_counter] = ray_state.x_i[
                    active_and_tracked, :
                ]

            if (
                self.step_counter % self.num_step_per_exit_check == 0
            ):  # Check exit condition
                ray_state = self.exitCheck(
                    ray_state
                )  # Note: The operating set is always inverted ray_state.exited
                ray_state.active = (
                    ray_state.active & ~ray_state.exited
                )  # deactivate exited rays
                active_ray_count = torch.sum(
                    ray_state.active, dim=0
                )  # current active ray count
                active_ray_exist = active_ray_count > 0
                self.logger.debug(
                    f"Completed {self.step_counter}th-step at time {time.perf_counter() - start_time}. Ray active/init active/all: {active_ray_count}/{initial_active_ray_count}/{ray_state.num_rays}"
                )

            self.step_counter += 1

        self.total_stepping = (
            step_size * self.step_counter
        )  # Total stepped value in parametrization variable

        # propagation_matrix = propagation_matrix.coalesce() #coalesce to remove redundent indices and values

        # Multiply the recorded energy values by absorption coefficient at each voxel location
        absorption_vector = self.absorption_model.alpha(
            self.index_model.getPositionVectorsAtGridPoints()
        )  # in vector form
        # absorption_matrix = torch.diag(absorption_vector) #dense diagonal matrix. Each element represent local active species absorption value.
        diagonal_indices = torch.arange(absorption_vector.numel(), device=self.device)
        absorption_matrix = torch.sparse_coo_tensor(
            indices=torch.vstack((diagonal_indices, diagonal_indices)),
            values=absorption_vector,
            device=self.device,
            dtype=propagation_matrix.dtype,
        )  # sparse diagonal tensor. Each element represent local active species absorption value.

        propagation_matrix = torch.mm(
            propagation_matrix, absorption_matrix
        )  # matrix multiplication.

        if active_ray_exist:
            self.logger.error(
                f"Some rays are still active when max_num_step ({self.max_num_step}) is reached. Rays active: {active_ray_count}/{initial_active_ray_count}."
            )

        return propagation_matrix, ray_state, ray_tracker

    # =======================Eikonal equation parametrizations======================
    # Canonical parameter, as sigma in "Adjoint Nonlinear Ray Tracing, Teh, et al. 2022"
    @torch.inference_mode()
    def _dx_dsigma(self, _, v, *arg):
        return v

    @torch.inference_mode()
    def _dv_dsigma(self, x, _, known_n=None):
        if known_n is None:
            n = self.index_model.n(x)[:, None]
        else:
            n = known_n

        return n * self.index_model.grad_n(x)  # broadcasted to all xyz components

    # Physical path length, as s in "Eikonal Rendering: Efficient Light Transport in Refractive Objects, Ihrke, et al. 2007"
    @torch.inference_mode()
    def _dx_ds(self, x, v, known_n=None):
        if known_n is None:
            n = self.index_model.n(x)[:, None]
        else:
            n = known_n
        return v / n

    @torch.inference_mode()
    def _dv_ds(self, x, _, *arg):
        return self.index_model.grad_n(x)

    # Optical path length, opl as t in "Eikonal Rendering: Efficient Light Transport in Refractive Objects, Ihrke, et al. 2007"
    @torch.inference_mode()
    def _dx_dopl(self, x, v, known_n=None):
        if known_n is None:
            n = self.index_model.n(x)[:, None]
        else:
            n = known_n
        return v * (n ** (-2))

    @torch.inference_mode()
    def _dv_dopl(self, x, _, known_n=None):
        if known_n is None:
            n = self.index_model.n(x)[:, None]
        else:
            n = known_n
        return self.index_model.grad_n(x) / n

    # =======================ODE solvers======================
    @torch.inference_mode()
    def _forwardSymplecticEuler(self, ray_state, step_size):  # step forward the RayState
        # push np1 to be n
        ray_state.x_i[ray_state.active] = ray_state.x_ip1[ray_state.active]
        ray_state.v_i[ray_state.active] = ray_state.v_ip1[ray_state.active]

        # Compute new np1 based on n (the old np1)
        # Compute v_ip1 using x_i
        ray_state.v_ip1[ray_state.active] = (
            ray_state.v_i[ray_state.active]
            + self.dv_dstep(
                ray_state.x_i[ray_state.active], ray_state.v_i[ray_state.active]
            ).to(ray_state.v_i.dtype)
            * step_size
        )

        # Then compute x_ip1 using v_ip1
        dx_active = (
            self.dx_dstep(
                ray_state.x_i[ray_state.active], ray_state.v_ip1[ray_state.active]
            ).to(ray_state.x_i.dtype)
            * step_size
        )
        ray_state.x_ip1[ray_state.active] = ray_state.x_i[ray_state.active] + dx_active

        ds_active = torch.linalg.norm(dx_active, dim=1)  # physical distance travelled
        ray_state.s[ray_state.active] += (
            ds_active  # update total physical distance travelled
        )
        ray_state.attenuance[ray_state.active] += (
            self.attenuation_model.alpha(ray_state.x_i[ray_state.active]) * ds_active
        )  # attenuance is integrated along the path. Explicit integral of attenuance avoid errors from repeated multiplication.
        ray_state.ray_energy[ray_state.active] = ray_state.ray_energy_0[
            ray_state.active
        ] * torch.exp(
            -ray_state.attenuance[ray_state.active]
        )  # current energy is the initial energy exponentially decayed by attenuance
        return ray_state

    @torch.inference_mode()
    def _forwardEuler(self, ray_state, step_size):  # step forward the RayState
        # push np1 to be n
        ray_state.x_i[ray_state.active] = ray_state.x_ip1[ray_state.active]
        ray_state.v_i[ray_state.active] = ray_state.v_ip1[ray_state.active]

        # Compute new np1 based on n (the old np1)
        # Compute v_ip1 using x_i
        ray_state.v_ip1[ray_state.active] = (
            ray_state.v_i[ray_state.active]
            + self.dv_dstep(
                ray_state.x_i[ray_state.active], ray_state.v_i[ray_state.active]
            ).to(ray_state.v_i.dtype)
            * step_size
        )

        # Then compute x_ip1 using v_i
        dx_active = (
            self.dx_dstep(
                ray_state.x_i[ray_state.active], ray_state.v_i[ray_state.active]
            ).to(ray_state.x_i.dtype)
            * step_size
        )
        ray_state.x_ip1[ray_state.active] = ray_state.x_i[ray_state.active] + dx_active

        ds_active = torch.linalg.norm(dx_active, dim=1)  # physical distance travelled
        ray_state.s[ray_state.active] += (
            ds_active  # update total physical distance travelled
        )
        ray_state.attenuance[ray_state.active] += (
            self.attenuation_model.alpha(ray_state.x_i[ray_state.active]) * ds_active
        )  # attenuance is integrated along the path. Explicit integral of attenuance avoid errors from repeated multiplication.
        ray_state.ray_energy[ray_state.active] = ray_state.ray_energy_0[
            ray_state.active
        ] * torch.exp(
            -ray_state.attenuance[ray_state.active]
        )  # current energy is the initial energy exponentially decayed by attenuance
        return ray_state

    @torch.inference_mode()
    def _leapfrog_init(self, ray_state, step_size):
        # Step backward ray_state.v_ip1 by half-step. After entering _forwardSymplecticEuler, v_ip1 will be half-step forward relative to x, when x is updated with v.
        # Initially x_0 = x_i = x_ip1, and v_0 = v_i = v_ip1, so it doesn't matter which input we use.
        ray_state.v_ip1[ray_state.active] = (
            ray_state.v_ip1[ray_state.active]
            - self.dv_dstep(
                ray_state.x_ip1[ray_state.active], ray_state.v_ip1[ray_state.active]
            )
            * step_size
            / 2.0
        )
        return ray_state

    @torch.inference_mode()
    def _no_step_init(self, ray_state, _):
        return ray_state

    # =======================Ray-voxel interactions======================
    @torch.inference_mode()
    def deposit(
        self, ray_state, step_size, desposition_grid
    ):  # deposit projected values along the ray
        # Preprocess inputs for deposition, which is computed in a normalized and recentered unit of length, in terms of array indices
        step_size_idx_unit = (
            step_size / self.index_model.voxel_size[0]
        )  # step size in index units, used to determine deposition weights of each ray segement
        x_arr_idx = self.expressPositionInArrayIndices(
            ray_state.x_ip1[ray_state.active, :]
        )  # Express position x in grid indices. This only contains the active set.
        voxel_idx_x = self.getAdjacentVoxelIndicesAtLocation(
            x_arr_idx
        )  # This also only contains the active set.

        adj_voxel_count = (
            4 if (self.index_model.zv.numel() == 1) else 8
        )  # for a 2D problem 4 adjacent voxels, for 3D problem 8 adjacent voxels.

        for adjacent_voxel_number in range(adj_voxel_count):
            # For performance, the accumulation happens in the innermost layer, only to relevant elements.
            desposition_grid = self.depositEnergyOnAdjacentVoxel(
                ray_state.ray_energy[ray_state.active],
                step_size_idx_unit,
                x_arr_idx,
                voxel_idx_x,
                adjacent_voxel_number,
                desposition_grid,
            )
            # All arguments of depositEnergyOnAdjacentVoxel are now downselected by ray_state.active. It will not act on inactive set of rays.
        return desposition_grid

    @torch.inference_mode()
    def integrate(
        self, ray_state, step_size, real_space_distribution
    ):  # integrate real space quantities along the ray
        # Preprocess inputs for integration, which is computed in a normalized and recentered unit of length, in terms of array indices
        step_size_idx_unit = (
            step_size / self.index_model.voxel_size[0]
        )  # step size in index units, used to determine deposition weights of each ray segement
        x_arr_idx = self.expressPositionInArrayIndices(
            ray_state.x_ip1[ray_state.active, :]
        )  # Express position x in grid indices. This only contains the active set.
        voxel_idx_x = self.getAdjacentVoxelIndicesAtLocation(
            x_arr_idx
        )  # This also only contains the active set.

        adj_voxel_count = (
            4 if (self.index_model.zv.numel() == 1) else 8
        )  # for a 2D problem 4 adjacent voxels, for 3D problem 8 adjacent voxels.

        for adjacent_voxel_number in range(adj_voxel_count):
            # For performance, the accumulation happens in the innermost layer, only to relevant elements.
            ray_state.integral[ray_state.active] = self.integrateEnergyFromAdjacentVoxel(
                ray_state.ray_energy[ray_state.active],
                step_size_idx_unit,
                x_arr_idx,
                voxel_idx_x,
                adjacent_voxel_number,
                real_space_distribution,
                ray_state.integral[ray_state.active],
            )
            # All arguments of integrateEnergyFromAdjacentVoxel are now downselected by ray_state.active. It will not act on inactive set of rays.
        return ray_state

    @torch.inference_mode()
    def record(
        self, ray_state, step_size, propagation_matrix_shape
    ):  # deposit projected values along the ray and record them as matrix entries
        # Preprocess inputs for deposition and recording, which is computed in a normalized and recentered unit of length, in terms of array indices
        step_size_idx_unit = (
            step_size / self.index_model.voxel_size[0]
        )  # step size in index units, used to determine deposition weights of each ray segement
        x_arr_idx = self.expressPositionInArrayIndices(
            ray_state.x_ip1[ray_state.active, :]
        )  # Express position x in grid indices. This only contains the active set.
        voxel_idx_x = self.getAdjacentVoxelIndicesAtLocation(
            x_arr_idx
        )  # This also only contains the active set.

        adj_voxel_count = (
            4 if (self.index_model.zv.numel() == 1) else 8
        )  # for a 2D problem 4 adjacent voxels, for 3D problem 8 adjacent voxels.

        delta_propagation_matrix = torch.sparse_coo_tensor(
            size=propagation_matrix_shape, device=self.device
        )
        active_ray_idx = ray_state.active.argwhere()[
            :, 0
        ]  # Pick the first index of all non-zero elements. This will be used to indicate row position in the propagation matrix.
        for adjacent_voxel_number in range(adj_voxel_count):
            delta_propagation_matrix += self.recordEnergyOnAdjacentVoxel(
                ray_state.ray_energy[ray_state.active],
                step_size_idx_unit,
                x_arr_idx,
                voxel_idx_x,
                adjacent_voxel_number,
                propagation_matrix_shape,
                active_ray_idx,
            )
            # All arguments of recordEnergyOnAdjacentVoxel are now downselected by ray_state.active. It will not act on inactive set of rays.
        return delta_propagation_matrix
        # return delta_propagation_matrix.coalesce() #coalesce to prevent redundent indices build up

    @torch.inference_mode()
    def depositEnergyOnAdjacentVoxel(
        self,
        ray_energy,
        step_size_idx_unit,
        x_arr_idx,
        voxel_idx_x,
        adjacent_voxel_number: int,
        desposition_grid,
    ):
        """
        In 3D problems, this function handles the dose deposition of one particular voxel (lower/upper corner) out of 8 voxels adjacent to positions at x.
        (In 2D problems, this function handles one out of 4 adjacent voxels.)

        The adjacent_voxel_number is integer from 0-7 for 3D problem and 0-3 for 2D problem

        This function works for any number of rays, as long as the shape[0] of all ray-numbered inputs are consistent.
        Internally, it only process rays that have a valid adjacent voxel position.
        """

        # Check if the number of active rays matches
        assert x_arr_idx.shape[0] == ray_energy.shape[0], (
            f"First dimension of x_arr_idx ({x_arr_idx.shape[0]}) and ray_energy ({ray_energy.shape[0]}) must match."
        )
        assert x_arr_idx.shape[0] == voxel_idx_x.shape[0], (
            f"First dimension of x_arr_idx ({x_arr_idx.shape[0]}) and voxel_idx_x ({voxel_idx_x.shape[0]}) must match."
        )

        # These 3 binary indices index the last dimension of voxel_idx_x which contains the floor and ceiling voxel (idx) adjacent to position x
        above_x = int(adjacent_voxel_number % 2)  # first bit
        above_y = int((adjacent_voxel_number // 2) % 2)  # second bit
        above_z = int(
            (adjacent_voxel_number // 4) % 2
        )  # third bit. For 2D case, this bit is constantly 0

        voxel_idx_select = torch.cat(
            (
                voxel_idx_x[:, 0, above_x][:, None],
                voxel_idx_x[:, 1, above_y][:, None],
                voxel_idx_x[:, 2, above_z][:, None],
            ),
            dim=1,
        )

        # Valid VOXEL INDEX: filter out combination that falls out of grid, if any indices are out of the grid, ignore it
        valid_idx = (voxel_idx_select[:, 0] >= 0) & (
            voxel_idx_select[:, 0] < self.index_model.xv.numel()
        )  # Check x index
        valid_idx = (
            valid_idx
            & (voxel_idx_select[:, 1] >= 0)
            & (voxel_idx_select[:, 1] < self.index_model.yv.numel())
        )  # Check y index
        valid_idx = (
            valid_idx
            & (voxel_idx_select[:, 2] >= 0)
            & (voxel_idx_select[:, 2] < self.index_model.zv.numel())
        )  # Check z index

        if torch.any(valid_idx):  # if valid_idx has at least one non-zero element
            # Interpolation coefficient is computed as Trilinear interpolation = Lagrange interpolation of order 1 = Linear shape function
            interp_coef = torch.prod(
                1 - torch.abs(x_arr_idx[valid_idx, :] - voxel_idx_select[valid_idx, :]),
                dim=1,
            )

            # Deposition values are comprise of step size (in voxel unit) * interp_coef * ray_energy
            energy_interaction_at_valid_idx = (
                step_size_idx_unit * interp_coef * ray_energy[valid_idx]
            )

            # Now we have deposition index and deposition values. Let's add to our deposition_grid
            # Deposition is more complicated than integration because multiple rays might deposite energy into the same element, causing racing condition in assignment.
            # Therefore we cannot simply assign/add to the deposition_grid via advance indexing (option 1 below).
            # Option 3 is the fastest and correct method. The experimented methods are displayed here for reference.

            if _direct_assignment := False:
                # (1) Direct assignement
                # This is the operation that would results in racing condition in the assignment process due to duplicates of insertion indices.
                # Although this advance indexing does not raise exceptions, the numerical results are WRONG.
                # This is because the multi-accessed voxels will only take values from one particular ray segment instead of from all segments that contribution to them.
                desposition_grid[
                    voxel_idx_select[valid_idx, 0],
                    voxel_idx_select[valid_idx, 1],
                    voxel_idx_select[valid_idx, 2],
                ] += energy_interaction_at_valid_idx
            elif _as_linear_operator := False:
                # (2) Duplicate-summation done by sparse tensor multiplication. (Each step takes ~10 times long as (1))
                # In this method, the assignment operation is written as a sparse linear operator.
                # Summation is done naturally through the matrix vector multiplication process.
                desposition_grid += self.linearCombinationRayEnergyToGrid(
                    desposition_grid.shape,
                    voxel_idx_select[valid_idx, :],
                    energy_interaction_at_valid_idx,
                )
            elif _as_uncoalesced_sparse_tensor := True:
                # (3) Duplicate-summation done by coalescence of sparse tensor. (Each step takes similar time as (1))
                # In this method, the deposition grid itself is represented by a uncoalesced sparse tensor.
                # Uncoalesced sparse tensor permits multiple elements to coexist with the same index
                # Summation is done implicitly in the pyTorch implementation of uncoalesced sparse COO tensor.
                desposition_grid += self.coalesceRayEnergyToGrid(
                    desposition_grid.shape,
                    voxel_idx_select[valid_idx, :],
                    energy_interaction_at_valid_idx,
                )
            else:
                # (4) Sequential python for loop to assign value one by one. (Each step takes 300x time as (1))
                # Because the loop operate at the interpreter level, this is the slowest and makes common problem size infeasible to solve.
                loop_idx = valid_idx.argwhere()[
                    :, 0
                ]  # Pick the first index of all non-zero elements
                for ray in loop_idx:  # nonzero returns a 2D array by default
                    desposition_grid[
                        voxel_idx_select[ray, 0],
                        voxel_idx_select[ray, 1],
                        voxel_idx_select[ray, 2],
                    ] += energy_interaction_at_valid_idx[ray]
        return desposition_grid

    @torch.inference_mode()
    def integrateEnergyFromAdjacentVoxel(
        self,
        ray_energy,
        step_size_idx_unit,
        x_arr_idx,
        voxel_idx_x,
        adjacent_voxel_number: int,
        real_space_distribution,
        ray_integral,
    ):
        """
        In 3D problems, this function handles the dose deposition of one particular voxel (lower/upper corner) out of 8 voxels adjacent to positions at x.
        (In 2D problems, this function handles one out of 4 adjacent voxels.)

        The adjacent_voxel_number is integer from 0-7 for 3D problem and 0-3 for 2D problem

        This function works for any number of rays, as long as the shape[0] of all ray-numbered inputs are consistent.
        Internally, it only process rays that have a valid adjacent voxel position.
        """

        # Check if the number of active rays matches
        assert x_arr_idx.shape[0] == ray_energy.shape[0], (
            f"First dimension of x_arr_idx ({x_arr_idx.shape[0]}) and ray_energy ({ray_energy.shape[0]}) must match."
        )
        assert x_arr_idx.shape[0] == voxel_idx_x.shape[0], (
            f"First dimension of x_arr_idx ({x_arr_idx.shape[0]}) and voxel_idx_x ({voxel_idx_x.shape[0]}) must match."
        )
        assert x_arr_idx.shape[0] == ray_integral.shape[0], (
            f"First dimension of x_arr_idx ({x_arr_idx.shape[0]}) and ray_integral ({ray_integral.shape[0]}) must match."
        )

        # These 3 binary indices index the last dimension of voxel_idx_x which contains the floor and ceiling voxel (idx) adjacent to position x
        above_x = int(adjacent_voxel_number % 2)  # first bit
        above_y = int((adjacent_voxel_number // 2) % 2)  # second bit
        above_z = int(
            (adjacent_voxel_number // 4) % 2
        )  # third bit. For 2D case, this bit is constantly 0

        voxel_idx_select = torch.cat(
            (
                voxel_idx_x[:, 0, above_x][:, None],
                voxel_idx_x[:, 1, above_y][:, None],
                voxel_idx_x[:, 2, above_z][:, None],
            ),
            dim=1,
        )

        # Valid VOXEL INDEX: filter out combination that falls out of grid, if any indices are out of the grid, igore it
        valid_idx = (voxel_idx_select[:, 0] >= 0) & (
            voxel_idx_select[:, 0] < self.index_model.xv.numel()
        )  # Check x index
        valid_idx = (
            valid_idx
            & (voxel_idx_select[:, 1] >= 0)
            & (voxel_idx_select[:, 1] < self.index_model.yv.numel())
        )  # Check y index
        valid_idx = (
            valid_idx
            & (voxel_idx_select[:, 2] >= 0)
            & (voxel_idx_select[:, 2] < self.index_model.zv.numel())
        )  # Check z index

        if torch.any(valid_idx):  # if valid_idx has at least one non-zero element
            # Interpolation coefficient is computed as Trilinear interpolation = Lagrange interpolation of order 1 = Linear shape function
            interp_coef = torch.prod(
                1 - torch.abs(x_arr_idx[valid_idx, :] - voxel_idx_select[valid_idx, :]),
                dim=1,
            )

            # Interaction strength are comprise of step size (in voxel unit) * interp_coef * ray_energy
            energy_interaction_at_valid_idx = (
                step_size_idx_unit * interp_coef * ray_energy[valid_idx]
            )

            # Now we have deposition index and deposition values. Let's collect the energy into ray_integral
            ray_integral[valid_idx] += (
                real_space_distribution[
                    voxel_idx_select[valid_idx, 0],
                    voxel_idx_select[valid_idx, 1],
                    voxel_idx_select[valid_idx, 2],
                ]
                * energy_interaction_at_valid_idx
            )

        return ray_integral  # Note this only contains the active set. And inside the active set, only those ray segments which has valid voxel correspondence have been updated.

    @torch.inference_mode()
    def recordEnergyOnAdjacentVoxel(
        self,
        ray_energy,
        step_size_idx_unit,
        x_arr_idx,
        voxel_idx_x,
        adjacent_voxel_number: int,
        propagation_matrix_shape,
        active_ray_idx,
    ):
        """
        In 3D problems, this function handles the dose deposition of one particular voxel (lower/upper corner) out of 8 voxels adjacent to positions at x.
        (In 2D problems, this function handles one out of 4 adjacent voxels.)

        The adjacent_voxel_number is integer from 0-7 for 3D problem and 0-3 for 2D problem

        This function works for any number of rays, as long as the shape[0] of all ray-numbered inputs are consistent.
        Internally, it only process rays that have a valid adjacent voxel position.

        voxel_idx_x is multi-dimensional index in real space domain
        active_ray_idx is 1-dimensional index of rays
        """

        # Check if the number of active rays matches
        assert x_arr_idx.shape[0] == ray_energy.shape[0], (
            f"First dimension of x_arr_idx ({x_arr_idx.shape[0]}) and ray_energy ({ray_energy.shape[0]}) must match."
        )
        assert x_arr_idx.shape[0] == voxel_idx_x.shape[0], (
            f"First dimension of x_arr_idx ({x_arr_idx.shape[0]}) and voxel_idx_x ({voxel_idx_x.shape[0]}) must match."
        )
        assert x_arr_idx.shape[0] == active_ray_idx.shape[0], (
            f"First dimension of x_arr_idx ({x_arr_idx.shape[0]}) and active_ray_idx ({active_ray_idx.shape[0]}) must match."
        )

        # These 3 binary indices index the last dimension of voxel_idx_x which contains the floor and ceiling voxel (idx) adjacent to position x
        above_x = int(adjacent_voxel_number % 2)  # first bit
        above_y = int((adjacent_voxel_number // 2) % 2)  # second bit
        above_z = int(
            (adjacent_voxel_number // 4) % 2
        )  # third bit. For 2D case, this bit is constantly 0

        voxel_idx_select = torch.cat(
            (
                voxel_idx_x[:, 0, above_x][:, None],
                voxel_idx_x[:, 1, above_y][:, None],
                voxel_idx_x[:, 2, above_z][:, None],
            ),
            dim=1,
        )

        # Valid VOXEL INDEX: filter out combination that falls out of grid, if any indices are out of the grid, igore it
        # valid_idx is index in the tensor of the active rays and associated quantities.
        valid_idx = (voxel_idx_select[:, 0] >= 0) & (
            voxel_idx_select[:, 0] < self.index_model.xv.numel()
        )  # Check x index
        valid_idx = (
            valid_idx
            & (voxel_idx_select[:, 1] >= 0)
            & (voxel_idx_select[:, 1] < self.index_model.yv.numel())
        )  # Check y index
        valid_idx = (
            valid_idx
            & (voxel_idx_select[:, 2] >= 0)
            & (voxel_idx_select[:, 2] < self.index_model.zv.numel())
        )  # Check z index

        if torch.any(valid_idx):  # if valid_idx has at least one non-zero element
            # Interpolation coefficient is computed as Trilinear interpolation = Lagrange interpolation of order 1 = Linear shape function
            interp_coef = torch.prod(
                1 - torch.abs(x_arr_idx[valid_idx, :] - voxel_idx_select[valid_idx, :]),
                dim=1,
            )

            # Deposition values are comprise of step size (in voxel unit) * interp_coef * ray_energy
            energy_interaction_at_valid_idx = (
                step_size_idx_unit * interp_coef * ray_energy[valid_idx]
            )

            # Matrix row indices corresponds to rays/sinogram voxels
            matrix_row_idx = active_ray_idx[
                valid_idx
            ]  # Among all active rays, pick those with a valid adjacent voxel

            # Matrix column indices corresponds to real space voxel
            # ravel index. In C order the last index cycle the fastest
            x_ind_stride = self.index_model.yv.numel() * self.index_model.zv.numel()
            y_ind_stride = self.index_model.zv.numel()
            matrix_col_idx = (
                voxel_idx_select[valid_idx, 0] * x_ind_stride
                + voxel_idx_select[valid_idx, 1] * y_ind_stride
                + voxel_idx_select[valid_idx, 2]
            )

            # Record values in a sparse COO tensor
            delta_propagation_matrix = torch.sparse_coo_tensor(
                indices=torch.vstack((matrix_row_idx, matrix_col_idx)),
                values=energy_interaction_at_valid_idx,
                size=propagation_matrix_shape,
                device=self.device,
            )
        else:
            delta_propagation_matrix = torch.sparse_coo_tensor(
                size=propagation_matrix_shape, device=self.device
            )  # Return empty sparse tensor if there is none of the rays access valid space voxel index.

        return delta_propagation_matrix

    @torch.inference_mode()
    def expressPositionInArrayIndices(self, x):
        """
        Express position x in grid indices
        """
        return (
            (x - (-self.index_model.grid_span / 2)) * (1 / self.index_model.voxel_size)
        )  # Normalize position with voxel size so that the position is indicative of the tensor index

    @torch.inference_mode()
    def getAdjacentVoxelIndicesAtLocation(self, x_arr_idx: torch.Tensor):
        """
        Get adjacent voxel indices. Output indices are presented by binary permutation of lower and upper indices in every dimension
        Note: Not all rays are required. Able to only process the activate ray segments.
        """
        x_lower_voxel_idx = torch.floor(x_arr_idx).to(
            torch.long
        )  # tensor index must be of type long
        x_upper_voxel_idx = torch.ceil(x_arr_idx).to(
            torch.long
        )  # tensor index must be of type long

        # The dimensions of voxel_idx_x array is [num_active_rays, dim (length of 3), lower_or_upper (length of 2)]
        voxel_idx_x = torch.cat(
            (x_lower_voxel_idx[:, :, None], x_upper_voxel_idx[:, :, None]), dim=2
        )  # last dimension store lower idx first then upper idx

        return voxel_idx_x

    @torch.inference_mode()
    def linearCombinationRayEnergyToGrid(
        self,
        desposition_grid_shape,
        voxel_idx_select_at_valid_idx,
        energy_interaction_at_valid_idx,
    ):
        """
        This function builds a sparse tensor for depositing ray energy onto the grid.
        The tensor adds up the contribution of multiple rays towards the same elements.
        This avoid the racing condition of grid elements in direct insertion.
        The first dimension matches the numel() of the deposition grid and the second dimension matches the number of rays that have a valid adjacent voxel index.
        """
        num_row = torch.prod(torch.tensor(desposition_grid_shape))
        num_col = energy_interaction_at_valid_idx.numel()
        # The following statement express the raveled multi-index into single index, following 'C' order where the last element varies the fastest.
        row_index_sparse = (
            voxel_idx_select_at_valid_idx[:, 0]
            * (desposition_grid_shape[1] * desposition_grid_shape[2])
            + voxel_idx_select_at_valid_idx[:, 1] * (desposition_grid_shape[2])
            + voxel_idx_select_at_valid_idx[:, 2]
        )
        col_index_sparse = torch.arange(
            energy_interaction_at_valid_idx.numel(), device=self.device
        )
        # index.shape[0] should equal the number of dimension of the sparse tensor.
        # index.shape[1] should equal the number of elements to be inserted into the COO sparse tensor
        index_sparse = torch.cat(
            (row_index_sparse[None, :], col_index_sparse[None, :]), dim=0
        )
        insertion_tensor = torch.sparse_coo_tensor(
            indices=index_sparse,
            values=torch.ones_like(energy_interaction_at_valid_idx),
            size=(num_row, num_col),
        )
        insertion_tensor = insertion_tensor.to_sparse_csr()
        # return (insertion_tensor@energy_interaction_at_valid_idx).reshape(desposition_grid_shape)
        return torch.mv(insertion_tensor, energy_interaction_at_valid_idx).reshape(
            desposition_grid_shape
        )  # torch.mv marginally faster than @

    @torch.inference_mode()
    def coalesceRayEnergyToGrid(
        self,
        desposition_grid_shape,
        voxel_idx_select_at_valid_idx,
        energy_interaction_at_valid_idx,
    ):
        delta_desposition_grid = torch.sparse_coo_tensor(
            indices=voxel_idx_select_at_valid_idx.T,
            values=energy_interaction_at_valid_idx,
            size=desposition_grid_shape,
        )
        return delta_desposition_grid

    # ==============================================

    @torch.inference_mode()
    def discreteSurfaceIntersectionCheck(self):
        pass

    @torch.inference_mode()
    def exitCheck(self, ray_state):
        """
        When the rays are outside the bounding box AND pointing away. As long as this condition is satisfied in one of the dimension, the rest of the straight ray will not intersect the domain.

        #for (position < min-tol & direction < 0) or (position > max+tol & direction > 0)

        #exit = exit_x | exit_y | exit_z
        """
        # When most rays are active most of the time, checking exit for only the active rays turns out to be empricically slower than checking all rays.
        # The logical comparison of the entire array is much faster than indexing the array which may require creating a copy.
        # When the active rays are sparse (e.g. only tracing 1 ray at a time to build algebraic representation of propagator), there might be performance benefit.
        if _only_check_active_rays := False:
            # Check for position if it is outside domain bounds (as defined by center of edge voxels). Check the coordinate PER DIMENSION, <min-tol or >max+tol
            # Distance tolerance is one voxel away from the bound (so the rest of the edge voxels are included, plus another half voxel in extra)
            exited_in_positive_direction = (
                (
                    ray_state.x_ip1[ray_state.active, :]
                    > self.index_model.xv_yv_zv_max + self.index_model.voxel_size
                )
                & (ray_state.v_ip1[ray_state.active, :] > 0.0)
            )  # In case of 2D problems, voxel size along z is inf so the ray is always within the z-bounds
            exited_in_negative_direction = (
                ray_state.x_ip1[ray_state.active, :]
                < self.index_model.xv_yv_zv_min - self.index_model.voxel_size
            ) & (
                ray_state.v_ip1[ray_state.active, :] < 0.0
            )  # Therefore in 2D problems, the rays can only escape from x,y-bounds

            # Perform logical or in the spatial dimension axis
            exited_in_positive_direction = (
                exited_in_positive_direction[:, 0]
                | exited_in_positive_direction[:, 1]
                | exited_in_positive_direction[:, 2]
            )
            exited_in_negative_direction = (
                exited_in_negative_direction[:, 0]
                | exited_in_negative_direction[:, 1]
                | exited_in_negative_direction[:, 2]
            )

            ray_state.exited[ray_state.active] = (
                exited_in_positive_direction | exited_in_negative_direction
            )

        else:
            # Check for position if it is outside domain bounds (as defined by center of edge voxels). Check the coordinate PER DIMENSION, <min-tol or >max+tol
            # Distance tolerance is one voxel away from the bound (so the rest of the edge voxels are included, plus another half voxel in extra)
            exited_in_positive_direction = (
                (
                    ray_state.x_ip1
                    > self.index_model.xv_yv_zv_max + self.index_model.voxel_size
                )
                & (ray_state.v_ip1 > 0.0)
            )  # In case of 2D problems, voxel size along z is inf so the ray is always within the z-bounds
            exited_in_negative_direction = (
                ray_state.x_ip1
                < self.index_model.xv_yv_zv_min - self.index_model.voxel_size
            ) & (
                ray_state.v_ip1 < 0.0
            )  # Therefore in 2D problems, the rays can only escape from x,y-bounds

            # Perform logical or in the spatial dimension axis
            exited_in_positive_direction = (
                exited_in_positive_direction[:, 0]
                | exited_in_positive_direction[:, 1]
                | exited_in_positive_direction[:, 2]
            )
            exited_in_negative_direction = (
                exited_in_negative_direction[:, 0]
                | exited_in_negative_direction[:, 1]
                | exited_in_negative_direction[:, 2]
            )
            # #Alternatively check logical OR with sum
            # exited_in_positive_direction = torch.sum(exited_in_positive_direction, dim = 1, keepdim = False).bool()
            # exited_in_negative_direction = torch.sum(exited_in_negative_direction, dim = 1, keepdim = False).bool()

            ray_state.exited = exited_in_positive_direction | exited_in_negative_direction

        # Optionally check if intensity close to zero (e.g. due to occulsion)

        return ray_state


class RayState:
    """
    State of a set of rays. The set can be all rays, rays emitted per angle, per z-slice, or per angle-z-slice
    This class is used as a dataclass to store current state of a set of rays, and where these rays originally belong to (x,theta,z) or (phi, theta, psi).
    For parallelization and generality, x_0 and v_0 has shape of (n_rays,n_dim).

    """

    _dtype_to_tensor_type = {
        torch.float16: torch.HalfTensor,
        torch.float32: torch.FloatTensor,
        torch.float64: torch.DoubleTensor,
    }  # Class attribute: mapping dtype to tensor type.

    @torch.inference_mode()
    def __init__(
        self,
        device,
        tensor_dtype=torch.float32,
        x_0: torch.Tensor = None,
        v_0: torch.Tensor = None,
        sino_shape: tuple = None,
        sino_coord: list = None,
    ) -> None:
        self.device = device
        if tensor_dtype is not None:
            self.tensor_dtype = tensor_dtype
        else:
            self.tensor_dtype = torch.float32
        torch.set_default_tensor_type(
            self._dtype_to_tensor_type[self.tensor_dtype]
        )  # A conversion between dtype and tensor type is needed since they are different objects and 'set_default_tensor_type' only accepts the latter.

        # Option to directly prescribe contents when ray positions and directions are generated externally (not using the provided methods)
        # self.x_0 = torch.as_tensor(x_0, device = self.device, dtype = self.tensor_dtype)
        # self.v_0 = torch.as_tensor(v_0, device = self.device, dtype = self.tensor_dtype)
        # self.sino_shape = sino_shape
        # self.sino_coord = sino_coord
        # self.num_rays = self.x_0.size
        # self.s
        # self.int_factor
        # self.width
        # self.integral
        # self.exited

    @torch.inference_mode()
    def setupRays(
        self,
        ray_trace_ray_config,
        target_coord_vec_list,
        azimuthal_angles_deg,
        inclination_angle_deg=0,
        ray_density=1,
        ray_width_fun=None,
    ):
        """
        By default, ray_density = 1.
        It means per azimuthal projection angle, there are max(nX, nY)*nZ number of rays, where nX,nY,nZ = target.shape.
        When ray_density =/= 1, number of rays = max(nX, nY)*nZ*(ray_density^2) because the density increase/decrease per dimension.
        """
        # Preprocess inputs
        self.ray_trace_ray_config = ray_trace_ray_config  #'parallel', 'cone'
        self.azimuthal_angles_deg = azimuthal_angles_deg
        self.inclination_angle_deg = inclination_angle_deg
        self.ray_density = ray_density

        # Convert all angles to radian
        self.azimuthal_angles_rad = self.azimuthal_angles_deg * np.pi / 180.0
        if self.inclination_angle_deg is None:
            self.inclination_angle_deg = 0  # In case none is passed down
        self.inclination_angle_rad = self.inclination_angle_deg * np.pi / 180.0

        if self.ray_density is None:
            self.ray_density = 1

        # Generate initial ray position and direction according to their configuration
        if self.ray_trace_ray_config == "parallel":
            self.setupRaysParallel(target_coord_vec_list)
        elif self.ray_trace_ray_config == "cone":
            raise Exception("Ray setup for cone beam is not yet implemented")
        else:
            raise Exception(
                f"Ray config: {str(self.ray_trace_ray_config)} is not one of the supported string: 'parallel', 'cone'."
            )

        # Initialize the iterate position and direction
        self.resetRaysIterateToInitial()  # Initialize x_i and x_ip1 to be x_0. Same for v.

    @torch.inference_mode()
    def setupRaysParallel(self, target_coord_vec_list):
        # The number of rays is assumed to be proportional to the size of real space grid/array
        real_nX = target_coord_vec_list[0].size
        real_nY = target_coord_vec_list[1].size
        real_nZ = target_coord_vec_list[2].size

        self.sino_shape_rd1 = (
            max(real_nX, real_nY),
            self.azimuthal_angles_rad.size,
            real_nZ,
        )  # This is the desired shape when ray density = 1. Saved for binning purpose.

        if real_nZ == 1:  # 2D case
            self.sino_shape = (
                round(max(real_nX, real_nY) * self.ray_density),
                self.azimuthal_angles_rad.size,
                real_nZ,
            )  # no need to supersample or undersample in z
        else:  # 3D case
            self.sino_shape = (
                round(max(real_nX, real_nY) * self.ray_density),
                self.azimuthal_angles_rad.size,
                max(round(real_nZ * self.ray_density), 1),
            )  # At least compute 1 z layer
        self.num_rays = self.sino_shape[0] * self.sino_shape[1] * self.sino_shape[2]

        # The following assumes the patterning volume is inscribed inside the simulation cube
        # ===========Coordinates local to projection plane===================
        # Determine first coordinate of sinogram
        sino_n0_min = min(
            np.amin(target_coord_vec_list[0]), np.amin(target_coord_vec_list[1])
        )
        sino_n0_max = max(
            np.amax(target_coord_vec_list[0]), np.amax(target_coord_vec_list[1])
        )
        sino_n0 = np.linspace(sino_n0_min, sino_n0_max, self.sino_shape[0])

        # Determine second coordinate of sinogram
        sino_n1_rad = self.azimuthal_angles_rad

        # Determine third coordinate of sinogram.
        # Currently physical size of the projection is assumed to be equal to z height.
        # With moderate elevation, this assumption might be limiting patternable volume when bounding box has xy dimension >> z dimension.
        sino_n2_min = np.amin(target_coord_vec_list[2])
        sino_n2_max = np.amax(target_coord_vec_list[2])
        sino_n2 = np.linspace(sino_n2_min, sino_n2_max, self.sino_shape[2])

        self.sino_coord = [sino_n0, sino_n1_rad, sino_n2]

        # There are a number of ways to determine the radial distance between center of grid to the projection plane where the rays starts. Longer distance requires more computation.
        # (1, smallest) Ellipsoidal (superset of spherical) patternable volume inscribed by the bounding box
        # The max distance along the 3 axes. Recall that sino_n0_max is the longer dimension in xy
        # proj_plane_radial_offset = max(sino_n0_max, sino_n2_max)

        # (2, middle) Elliptic cylindrical (superset of cylindrical) patternable volume inscribed by the bounding box.
        # When xy dimensions of the patternable volume is inscribed by the bounding box, while all of the z-extend of the array is assumed to be patternable.
        # Recall that sino_n0_max is the longer dimension in xy
        # Correction: Maybe it is still like case 1, considering a axially long part, positive elevation and the bottom parts could be misesed.
        proj_plane_radial_offset = np.sqrt(sino_n0_max**2 + sino_n2_max**2)

        # (3, largest) The whole bounding box is patternable. NOTE: The existing vamtoolbox always cut to cylinder and discard the corner of the box.
        # When patternable volume is circumscribes the bounding box (the corners of the box is patternable/relevant), the radial distance between ray init position and the grid center is the diagonal of the 3D box.
        # proj_plane_radial_offset = np.sqrt(np.amax(target_coord_vec_list[0])**2 + np.amax(target_coord_vec_list[1])**2 + np.amax(target_coord_vec_list[2])**2)

        if _CPU_create := False:
            # Create with CPU (which usually has access to more memory)
            G0, G1, G2 = np.meshgrid(sino_n0, sino_n1_rad, sino_n2, indexing="ij")
            G0 = G0.ravel()
            G1 = G1.ravel()
            G2 = G2.ravel()

            self.x_0 = np.ndarray((G0.size, 3))
            # Center of projection plane relative to grid center. Refers to documentations for derivation.
            self.x_0[:, 0] = (
                proj_plane_radial_offset * np.cos(G1) * np.cos(self.inclination_angle_rad)
            )
            self.x_0[:, 1] = (
                proj_plane_radial_offset * np.sin(G1) * np.cos(self.inclination_angle_rad)
            )
            self.x_0[:, 2] = proj_plane_radial_offset * np.sin(self.inclination_angle_rad)

            # Adding vectors from center of projection plane to pixel. Refers to documentations for derivation.
            self.x_0[:, 0] += -G0 * np.sin(G1) - G2 * np.cos(G1) * np.sin(
                self.inclination_angle_rad
            )
            self.x_0[:, 1] += G0 * np.cos(G1) - G2 * np.sin(G1) * np.sin(
                self.inclination_angle_rad
            )
            self.x_0[:, 2] += G2 * np.cos(self.inclination_angle_rad)

            self.v_0 = np.ndarray((G0.size, 3))
            self.v_0[:, 0] = -np.cos(G1) * np.cos(self.inclination_angle_rad)
            self.v_0[:, 1] = -np.sin(G1) * np.cos(self.inclination_angle_rad)
            self.v_0[:, 2] = -np.sin(self.inclination_angle_rad)

            self.x_0 = torch.as_tensor(self.x_0, device=self.device)
            self.v_0 = torch.as_tensor(self.v_0, device=self.device)

        else:
            # Create with GPU (which is faster if GPU memory is sufficient)
            inclination_angle_rad_tensor = torch.as_tensor(
                self.inclination_angle_rad, device=self.device, dtype=self.tensor_dtype
            )

            sino_n0 = torch.as_tensor(
                sino_n0, device=self.device, dtype=self.tensor_dtype
            )
            sino_n1_rad = torch.as_tensor(
                sino_n1_rad, device=self.device, dtype=self.tensor_dtype
            )
            sino_n2 = torch.as_tensor(
                sino_n2, device=self.device, dtype=self.tensor_dtype
            )
            G0, G1, G2 = torch.meshgrid(sino_n0, sino_n1_rad, sino_n2, indexing="ij")
            G0 = torch.ravel(G0)
            G1 = torch.ravel(G1)
            G2 = torch.ravel(G2)

            self.x_0 = torch.empty(
                (self.num_rays, 3), device=self.device, dtype=self.tensor_dtype
            )  # using same date type as sino_n0, which is inferred from its numpy version
            # Center of projection plane relative to grid center. Refers to documentations for derivation.
            self.x_0[:, 0] = (
                proj_plane_radial_offset
                * torch.cos(G1)
                * torch.cos(inclination_angle_rad_tensor)
            )
            self.x_0[:, 1] = (
                proj_plane_radial_offset
                * torch.sin(G1)
                * torch.cos(inclination_angle_rad_tensor)
            )
            self.x_0[:, 2] = proj_plane_radial_offset * torch.sin(
                inclination_angle_rad_tensor
            )

            # Adding vectors from center of projection plane to pixel. Refers to documentations for derivation.
            self.x_0[:, 0] += -G0 * torch.sin(G1) - G2 * torch.cos(G1) * torch.sin(
                inclination_angle_rad_tensor
            )
            self.x_0[:, 1] += G0 * torch.cos(G1) - G2 * torch.sin(G1) * torch.sin(
                inclination_angle_rad_tensor
            )
            self.x_0[:, 2] += G2 * torch.cos(inclination_angle_rad_tensor)

            self.v_0 = torch.empty(
                (self.num_rays, 3), device=self.device, dtype=self.tensor_dtype
            )
            self.v_0[:, 0] = -torch.cos(G1) * torch.cos(inclination_angle_rad_tensor)
            self.v_0[:, 1] = -torch.sin(G1) * torch.cos(inclination_angle_rad_tensor)
            self.v_0[:, 2] = -torch.sin(inclination_angle_rad_tensor)

    def resetRaysIterateToInitial(self):
        # Setting current ray state back to original.
        self.x_i = self.x_0.detach().clone()  # .detach() removes computation path, detach tensor from autograd graph. .clone() creates a copy.
        self.x_ip1 = self.x_0.detach().clone()  # This is the proper way to copy tensor without sharing storage: https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor/62496418
        self.v_i = self.v_0.detach().clone()  # Without .detach().clone(), subsequent in-place updates to x_i and x_ip1 would also update x_0. Similarly, v_i, v_ip1 would affect v_0.
        self.v_ip1 = self.v_0.detach().clone()

        # Initialize other properties of the rays
        self.s = torch.zeros(
            (self.num_rays), device=self.device, dtype=self.tensor_dtype
        )  # distance travelled by the ray from initial position
        self.ray_energy_0 = torch.ones(
            (self.num_rays), device=self.device, dtype=self.tensor_dtype
        )  # energy carried by the ray.
        self.ray_energy_0 *= (
            (self.ray_density ** (-1))
            if self.sino_coord[2].size == 1
            else self.ray_density ** (-2)
        )  # With supersampling, there are more rays and each ray carry less energy.
        self.ray_energy = self.ray_energy_0.detach().clone()
        self.attenuance = torch.zeros(
            (self.num_rays), device=self.device, dtype=self.tensor_dtype
        )  # exponent on e to model attenuation or gain. This "e^-attenuatance" factor is NOT subsumed in ray_energy due to potential loss of accuracy when it is multipled repeatably.
        # self.width = torch.ones((self.num_rays), device = self.device, dtype = self.tensor_dtype) #constant if FOV is in depth of focus. Converging or Diverging if not. Function of s.
        self.integral = torch.zeros(
            (self.num_rays), device=self.device, dtype=self.tensor_dtype
        )  # accumulating integral along the path
        self.exited = torch.zeros(
            (self.num_rays), device=self.device, dtype=torch.bool
        )  # boolean value. Tracing of corresponding ray stops when its exited flag is true.
        self.active = torch.ones(
            (self.num_rays), device=self.device, dtype=torch.bool
        )  # boolean value. Active set. This set is usually a subset of ~self.exited.

    def allocateSinogramEnergyToRay(self, sinogram):
        pass

    def allocateRayEnergyToSinogram(self):
        # Binning process
        return  # sinogram

    def reshape(self, ray_quantity):
        return ray_quantity.reshape(self.sino_shape)

    @torch.inference_mode()
    def plot_ray_init_position(self, angles_deg, color="black"):
        angles_rad = angles_deg * np.pi / 180.0

        # Find the closest match
        angular_indices = np.round(
            np.interp(
                angles_rad,
                self.sino_coord[1],
                range(self.sino_coord[1].size),
                left=None,
                right=None,
                period=None,
            )
        )
        angles_selected_rad = np.array(
            self.sino_coord[1][angular_indices.astype(np.int32)], ndmin=1
        )

        G0, G1, G2 = np.meshgrid(
            self.sino_coord[0], self.sino_coord[1], self.sino_coord[2], indexing="ij"
        )

        relevant_points = np.zeros_like(G1, dtype=np.bool_)
        # Find the selected points in the meshgrid
        for idx in range(angles_selected_rad.size):
            relevant_points = relevant_points | np.isclose(angles_selected_rad[idx], G1)

        relevant_points = np.ravel(relevant_points)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        x_0 = self.x_0.cpu()
        v_0 = self.v_0.cpu()
        # This only plot positions
        # ax.scatter(x_0[relevant_points,0], x_0[relevant_points,1], x_0[relevant_points,2], marker='o')
        # This plot both position and direction with arrows.
        ax.quiver(
            x_0[relevant_points, 0],
            x_0[relevant_points, 1],
            x_0[relevant_points, 2],
            v_0[relevant_points, 0],
            v_0[relevant_points, 1],
            v_0[relevant_points, 2],
            length=0.15,
            color=color,
        )
        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")

        plt.show()

    class RaySelector:
        def __init__(self, ray_state) -> None:
            self.device = ray_state.device
            self.num_rays = ray_state.num_rays
            self.sino_coord = ray_state.sino_coord

            self.selectParallelAngles = self.select

            self.selected_idx = torch.zeros(
                (self.num_rays, 1), device=self.device, dtype=torch.bool
            )

    def selectCoord(self, angles, mode="and"):
        # mode: 'and', 'or', 'xor'. This is the relationship BETWEEN the sinogram coordinates.
        # AND will only select rays that simultaneously satisfy requirements for sino_n0, sino_n1, sino_n2
        # OR will select rays that satisfy any requirements for sino_n0, sino_n1, sino_n2
        # So is XOR and NOT

        return self.selected_idx

    def selectInverse(self):
        # invert current selection
        pass

    def _selectSino0(self, sino0_coord):
        return  # idx

    def _selectSino1(self, sino0_coord):
        return  # idx

    def _selectSino2(self, sino0_coord):
        return  # idx

    # class RayTracker(RaySelector):
    #     def __init__(self, ray_state, angles_rad, ) -> None:
    #         ray_state.sino_coord

    #         #Find the closest match ray

    #         #Determine their indices in vector x

    #         #Store the indices

    #     def plotRays(self):

    #         #Color each rays from each angle.
    #         fig = plt.figure()
    #         ax = plt.axes(projection='3d')

    #         for ind in [ind for ind in range(x_np_record.shape[0]) if (ind)%1==0]:
    #             xline = x_np_record[ind,0,:]
    #             yline = x_np_record[ind,1,:]
    #             zline = x_np_record[ind,2,:]
    #             ax.plot3D(xline, yline, zline, 'blue', linewidth=0.2)

    #         ax.set_xlabel('x')
    #         ax.set_ylabel('y')
    #         ax.set_zlabel('z')
    #         ax.set_aspect('equal', 'box')
