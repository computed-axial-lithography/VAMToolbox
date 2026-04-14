import numpy as np
from leapctype import *


def _setup_leap(leapct, geometry, params):
    """
    Configure LEAP projector geometry and volume.
    All spatial units are mm.

    VAMToolbox volume shape: (nx, ny, nz)  — Z is the axial/helical direction
    LEAP volume array shape: (nz, ny, nx)  — conversion via transpose(2,1,0)

    Returns (nx, ny, nz).
    """
    num_angles = len(geometry.angles)
    num_rows   = int(params["num_rows"])
    num_cols   = int(params["num_cols"])
    pixel_h    = float(params.get("pixel_height", params["pixel_width"]))
    pixel_w    = float(params["pixel_width"])
    sod        = float(geometry.source_radius)
    sdd        = sod + float(geometry.detector_distance)

    # Use absolute angles so LEAP's helical z always increases
    phis_deg = np.ascontiguousarray(
        np.abs(np.degrees(geometry.angles)), dtype=np.float32
    )

    # helical_pitch is mm/revolution; LEAP set_conebeam wants mm/radian
    pitch_mm_per_rev = float(getattr(geometry, "helical_pitch", 0.0))
    pitch_mm_per_rad = pitch_mm_per_rev / (2.0 * np.pi)

    leapct.set_conebeam(
        num_angles, num_rows, num_cols,
        pixel_h, pixel_w,
        (num_rows - 1) / 2.0,   # center_row
        (num_cols - 1) / 2.0,   # center_col
        phis_deg,
        sod, sdd,
        0.0,                     # tau  (center-of-rotation offset)
        pitch_mm_per_rad,        # helicalPitch (mm/radian)
        0.0,                     # tiltAngle
    )

    nx, ny, nz = params["volume_shape"]   # VAMToolbox (nx, ny, nz)
    voxel_size = float(params.get("voxel_size", 1.0))

    # LEAP z starts at 0 and goes to z_total = pitch × max_phi.
    # Set offsetZ = +z_total/2 so the volume centre aligns with the scan midpoint.
    # (offsetZ is the physical z of the volume centre; volume spans [0, z_total])
    max_phi_rad = float(phis_deg[-1]) * np.pi / 180.0
    z_total     = pitch_mm_per_rad * max_phi_rad
    offset_z    = z_total / 2.0

    leapct.set_volume(nx, ny, nz, voxelWidth=voxel_size, offsetZ=offset_z)

    return nx, ny, nz


def forward(volume, geometry, params):
    """
    Forward projection: VAMToolbox volume → sinogram.

    volume : (nx, ny, nz)  float32
    returns: (num_angles, num_rows, num_cols)  float32
    """
    leapct = tomographicModels()
    nx, ny, nz = _setup_leap(leapct, geometry, params)

    # VAMToolbox (nx, ny, nz) → LEAP (nz, ny, nx)
    vol = np.ascontiguousarray(np.transpose(volume, (2, 1, 0)), dtype=np.float32)

    num_angles = len(geometry.angles)
    num_rows   = int(params["num_rows"])
    num_cols   = int(params["num_cols"])
    g = np.zeros((num_angles, num_rows, num_cols), dtype=np.float32)

    leapct.project(g, vol)
    return g


def back(sinogram, geometry, params):
    """
    Backprojection (adjoint of forward): sinogram → VAMToolbox volume.

    sinogram : (num_angles, num_rows, num_cols)  float32
    returns  : (nx, ny, nz)  float32
    """
    leapct = tomographicModels()
    nx, ny, nz = _setup_leap(leapct, geometry, params)

    # LEAP volume shape is (nz, ny, nx)
    f = np.zeros((nz, ny, nx), dtype=np.float32)
    leapct.backproject(sinogram, f)

    # LEAP (nz, ny, nx) → VAMToolbox (nx, ny, nz)
    return np.transpose(f, (2, 1, 0))


class LEAPProjectorWrapper:
    def __init__(self, geometry, params):
        self.geometry = geometry
        self.params   = params

    def forward(self, volume):
        return forward(volume, self.geometry, self.params)

    def backward(self, sinogram):
        return back(sinogram, self.geometry, self.params)
