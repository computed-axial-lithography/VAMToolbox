import numpy as np
from leapctype import *
from vamtoolbox.projector.leap_geometry import convert_geometry_to_leap_format

def forward_project(volume, geometry, params):
    """
    Performs forward projection using LEAP.

    Parameters:
        volume (np.ndarray): 3D array [Z, Y, X].
        geometry (object): User-defined geometry object (angles, radii, pitch, etc).
        params (dict): Includes geometry_type, detector settings, voxel size, etc.

    Returns:
        sinogram (np.ndarray): Forward projection result [views, rows, cols].
    """
    leapct = tomographicModels()

    print("[leap3D] Using Leap 3D forward projector")

    # Convert high-level geometry into LEAP-compatible dictionary
    geom = convert_geometry_to_leap_format(geometry, params)
    geom_type = geom["geometry_type"]


    # MODULAR-BEAM (supports helical motion)
    if geom_type == "modular":
        leapct.set_modularbeam(
            geom["source_positions"],
            geom["detector_centers"],
            geom["u_vectors"],
            geom["v_vectors"],
            geom["num_rows"],
            geom["num_cols"],
            geom["pixel_width"]
        )

        # Optional helical pitch for helical trajectories
        if geom.get("pitch", 0.0) != 0.0:
            leapct.set_normalizedHelicalPitch(geom["pitch"])

    # PARALLEL-BEAM
    elif geom_type == "parallel":
        leapct.set_parallelbeam(
            geom["angles"],
            geom["num_rows"],
            geom["num_cols"],
            geom["pixel_width"]
        )

    # CONE-BEAM (not fully implemented in your geometry builder yet)
    elif geom_type == "cone":
        raise NotImplementedError(
            "Cone-beam geometry is not fully implemented in this LEAP wrapper."
        )

    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")

    # Set volume geometry in LEAP: expects (X, Y, Z)
    leapct.set_volume(*volume.shape[::-1])

    # Allocate sinogram & set volume copy
    g = leapct.allocateProjections()
    f = np.copy(volume)

    # Perform forward projection
    leapct.project(g, f)

    return g



def back_project(sinogram, geometry, params):
    """
    Performs backprojection (FBP or iterative) using LEAP.

    Parameters:
        sinogram (np.ndarray): Input projection data.
        geometry (object): User-defined geometry setup.
        params (dict): Reconstruction settings. May include:
            - use_iterative_filter (bool)
            - tv_delta (float)
            - volume_shape (tuple)
            - geometry_type (str)

    Returns:
        volume (np.ndarray): Reconstructed 3D volume [Z, Y, X].
    """

    leapct = tomographicModels()
    geom = convert_geometry_to_leap_format(geometry, params)
    geom_type = geom["geometry_type"]

    # --------------------------------------------------------
    # Geometry setup (same logic as forward projector)
    # --------------------------------------------------------
    if geom_type == "modular":
        leapct.set_modularbeam(
            geom["source_positions"],
            geom["detector_centers"],
            geom["u_vectors"],
            geom["v_vectors"],
            geom["num_rows"],
            geom["num_cols"],
            geom["pixel_width"]
        )

        if geom.get("pitch", 0.0) != 0.0:
            leapct.set_normalizedHelicalPitch(geom["pitch"])

    elif geom_type == "parallel":
        leapct.set_parallelbeam(
            geom["angles"],
            geom["num_rows"],
            geom["num_cols"],
            geom["pixel_width"]
        )

    elif geom_type == "cone":
        raise NotImplementedError(
            "Cone-beam reconstruction is not implemented in this wrapper."
        )

    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")

    # --------------------------------------------------------
    # Volume setup: LEAP expects (X, Y, Z)
    # --------------------------------------------------------
    volume_shape = params["volume_shape"]  # (Z, Y, X)
    leapct.set_volume(*volume_shape[::-1])

    g = sinogram
    f = np.zeros(volume_shape, dtype=np.float32)

    # --------------------------------------------------------
    # Reconstruction choice: FBP or Iterative (ASD-POCS + TV)
    # --------------------------------------------------------
    if params.get("use_iterative_filter", False):
        filters = filterSequence(1.0)
        delta = params.get("tv_delta", 0.02 / 20.0)
        filters.append(TV(leapct, delta=delta))

        # ASD-POCS settings: (ng, nf, relax, filters)
        leapct.ASDPOCS(g, f, 10, 10, 1, filters)

    else:
        leapct.FBP(g, f)

    return f
