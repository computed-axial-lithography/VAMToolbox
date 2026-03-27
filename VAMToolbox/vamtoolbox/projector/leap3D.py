import numpy as np
from leapctype import *
from .leap_geometry import _build_cone_geometry, convert_geometry_to_leap_format

def forward(volume, geometry, params):
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

    geom = convert_geometry_to_leap_format(geometry, params)
    geom_type = geom["geometry_type"]

    num_angles = len(geom["angles"]) if "angles" in geom else geom["source_positions"].shape[0]
    num_rows = int(geom["num_rows"])
    num_cols = int(geom["num_cols"])
    pixel_width = float(geom["pixel_width"])
    pixel_height = float(params.get("pixel_height", pixel_width))
    center_row = (num_rows - 1) / 2.0
    center_col = (num_cols - 1) / 2.0
    phis_deg = np.ascontiguousarray(np.degrees(geom.get("angles", np.arange(num_angles))), dtype=np.float32)
    sod = float(getattr(geometry, "source_radius", 200.0)) # source to object distance, measured in mm; this can also be viewed as the source to center of rotation distance
    sdd = sod + float(getattr(geometry, "detector_distance", 400.0)) # source to detector distance, measured in mm

    volume = np.rot90(volume, k=1, axes=(0, 2))
    vol = np.transpose(volume, (2, 1, 0)).copy()
    vol = np.ascontiguousarray(volume, dtype=np.float32)

    if geom_type == "modular":
        leapct.set_modularbeam(
            num_angles,
            num_rows,
            num_cols,
            pixel_width, 
            pixel_width,
            np.ascontiguousarray(geom["source_positions"], dtype=np.float32),
            np.ascontiguousarray(geom["detector_centers"], dtype=np.float32),
            np.ascontiguousarray(geom["rowVectors"], dtype=np.float32),
            np.ascontiguousarray(geom["colVectors"], dtype=np.float32),
        )

        if geom.get("pitch", 0.0) != 0.0:
            leapct.set_normalizedHelicalPitch(geom["pitch"])

    elif geom_type == "parallel":
        leapct.set_parallelbeam(
            num_angles,
            num_rows,
            num_cols,
            pixel_height,
            pixel_width,
            center_row,
            center_col,
            phis_deg,
        )

    elif geom_type == "cone":
        geom = _build_cone_geometry(geometry, params)
        num_angles = geom["source_positions"].shape[0]
        pixel_width = float(geom["pixel_width"])
        pixel_height = float(geom.get("pixel_height", pixel_width))
        center_row = (num_rows - 1) / 2.0
        center_col = (num_cols - 1) / 2.0
        phis_deg = np.ascontiguousarray(np.degrees(geometry.angles), dtype=np.float32)
        if geom.get("pitch", 0.0) != 0.0:
           pitch = leapct.set_normalizedHelicalPitch(geom["pitch"])

        leapct.set_conebeam(
            num_angles,
            num_rows,
            num_cols,
            pixel_height,
            pixel_width,
            center_row,
            center_col,
            phis_deg, # phis (float32 numpy array):  a numpy array for specifying the angles of each projection, measured in degrees
            sod,
            sdd,
            0.0, # tau (float): center of rotation offset
            pitch if "pitch" in geom else 0.0, # helicalPitch (float): the helical pitch (mm/radians)
            0.0, # tiltAngle (float) the rotation of the detector around the optical axis (degrees)
        )

    else:
        raise ValueError(f"Unsupported geometry_type: {geom_type}")

    leapct.set_volume(*vol.shape[::-1])

    g = np.zeros((num_angles, num_rows, num_cols), dtype=np.float32, order="C")
    leapct.project(g, vol)
    return g

def back(sinogram, geometry, params):
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

    num_angles = len(geom["angles"]) if "angles" in geom else geom["source_positions"].shape[0]
    num_rows = int(geom["num_rows"])
    num_cols = int(geom["num_cols"])
    pixel_width = float(geom["pixel_width"])
    pixel_height = float(params.get("pixel_height", pixel_width))
    center_row = (num_rows - 1) / 2.0
    center_col = (num_cols - 1) / 2.0
    phis_deg = np.ascontiguousarray(np.degrees(geom.get("angles", np.arange(num_angles))), dtype=np.float32)
    sod = float(getattr(geometry, "source_radius", 100.0))
    sdd = sod + float(getattr(geometry, "detector_distance", 200.0))

    if geom_type == "modular":
        leapct.set_modularbeam(
            num_angles,
            num_rows,
            num_cols,
            pixel_width,
            pixel_width,
            np.ascontiguousarray(geom["source_positions"], dtype=np.float32),
            np.ascontiguousarray(geom["detector_centers"], dtype=np.float32),
            np.ascontiguousarray(geom["rowVectors"], dtype=np.float32),
            np.ascontiguousarray(geom["colVectors"], dtype=np.float32),
        )

        if geom.get("pitch", 0.0) != 0.0:
            leapct.set_normalizedHelicalPitch(geom["pitch"])

    elif geom_type == "parallel":
        leapct.set_parallelbeam(
            num_angles,
            num_rows,
            num_cols,
            pixel_height,
            pixel_width,
            center_row,
            center_col,
            phis_deg,
        )

    elif geom_type == "cone":
        geom = _build_cone_geometry(geometry, params)
        num_angles = geom["source_positions"].shape[0]
        pixel_width = float(geom["pixel_width"])
        pixel_height = float(geom.get("pixel_height", pixel_width))
        center_row = (num_rows - 1) / 2.0
        center_col = (num_cols - 1) / 2.0
        phis_deg = np.ascontiguousarray(np.degrees(geometry.angles), dtype=np.float32)

        leapct.set_conebeam(
            num_angles,
            num_rows,
            num_cols,
            pixel_height,
            pixel_width,
            center_row,
            center_col,
            phis_deg,
            sod,
            sdd,
            0.0,
            geom.get("pitch", 0.0),
            0.0,
        )

    # Volume setup: LEAP expects (X, Y, Z)
    volume_shape = params["volume_shape"]  # (Z, Y, X)
    leapct.set_volume(*volume_shape[::-1])

    g = sinogram
    f = np.zeros(volume_shape, dtype=np.float32)

    # Reconstruction choice: FBP or Iterative (ASD-POCS + TV)
    if params.get("use_iterative_filter", False):
        filters = filterSequence(1.0)
        delta = params.get("tv_delta", 0.02 / 20.0)
        filters.append(TV(leapct, delta=delta))

        # ASD-POCS settings: (ng, nf, relax, filters)
        leapct.ASDPOCS(g, f, 10, 10, 1, filters)

    else:
        leapct.FBP(g, f)
    
    f = np.transpose(f, (2, 1, 0)).copy()

    # f = np.rot90(f, k=1, axes=(0, 2))
    print(f"[back] leap_shape: {leap_shape} → output: {f.shape}")

    return f
