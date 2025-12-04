import numpy as np

def convert_geometry_to_leap_format(geometry, params):
    """
    Computes all geometric parameters required by LEAP and returns them as a dictionary.
    These values are used to call LEAP's tomographicModels.set_*() methods for different
    geometry types such as modular, parallel, and cone.

    Parameters:
        geometry (object): Your geometry definition (angles, radii, pitch, etc.).
        params (dict): Dictionary of parameters including pixel sizes, volume shape, etc.

    Returns:
        dict: Dictionary containing all needed inputs for LEAP geometry setup.
    """
    geom_type = params.get('geometry_type', 'modular')

    if geom_type == 'modular':
        return build_modular_geometry(geometry, params)
    elif geom_type == 'parallel':
        return build_parallel_geometry(geometry, params)
    elif geom_type == 'cone':
        return build_cone_geometry(geometry, params)
    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")

def build_modular_geometry(geometry, params):
    """
    Build geometry for LEAP's modular-beam projector.
    This version matches LEAP's C++ conventions more closely.

    Parameters:
        geometry (object):
            - angles (radians)
            - source_radius
            - detector_distance  (distance from source to detector plane)
            - helical_pitch      (axial distance per 360° rotation)

        params (dict):
            - num_rows
            - num_cols
            - detector_pixel_width
            - voxel_size

    Returns:
        dict with:
            - source_positions: (N,3)
            - detector_centers: (N,3)
            - u_vectors: (N,3)
            - v_vectors: (N,3)
            - pitch: normalized pitch per rotation
    """

    angles = geometry.angles
    num_proj = len(angles)

    R = geometry.source_radius                 # radius of source arc (gantry radius)
    D = geometry.detector_distance             # source → detector distance
    pitch = getattr(geometry, "helical_pitch", 0.0)

    # LEAP expects *normalized pitch* = pitch / (2π)
    pitch_per_radian = pitch / (2 * np.pi)

    source_positions = []
    detector_centers = []
    u_vectors = []
    v_vectors = []

    for i, theta in enumerate(angles):

        # -----------------------------
        # 1) SOURCE POSITION (helical)
        # -----------------------------
        # LEAP uses gantry rotation around z-axis
        # pitch grows linearly with angle
        z = pitch_per_radian * theta

        sx = R * np.cos(theta)
        sy = R * np.sin(theta)
        sz = z
        source = np.array([sx, sy, sz])

        # -----------------------------
        # 2) DETECTOR CENTER POSITION
        # -----------------------------
        # Detector lies along the *opposite direction* from rotation center
        # pointing outward along the x-y plane.
        #
        # C++ equivalent:
        # dc = source_position + (detector_distance * (-cosθ, -sinθ, 0))
        #
        dc = np.array([
            sx - D * np.cos(theta),
            sy - D * np.sin(theta),
            sz
        ])

        # -----------------------------
        # 3) DETECTOR AXES (u and v)
        # -----------------------------
        # v-axis is ALWAYS the rotation axis (z-direction)
        v = np.array([0, 0, 1])

        # u-axis is tangent to rotation path
        # Equivalent to derivative along rotation angle
        # u = (-sinθ, cosθ, 0)
        u = np.array([-np.sin(theta), np.cos(theta), 0])
        u = u / np.linalg.norm(u)

        # Append
        source_positions.append(source)
        detector_centers.append(dc)
        u_vectors.append(u)
        v_vectors.append(v)

    return {
        "geometry_type": "modular",
        "num_proj": num_proj,
        "source_positions": np.array(source_positions),
        "detector_centers": np.array(detector_centers),
        "u_vectors": np.array(u_vectors),
        "v_vectors": np.array(v_vectors),
        "pixel_width": params.get("detector_pixel_width", params["voxel_size"]),
        "num_rows": params["num_rows"],
        "num_cols": params["num_cols"],
        "pitch": pitch_per_radian,  # LEAP expects normalized pitch
    }


def build_parallel_geometry(geometry, params):
    """
    Returns geometry parameters for LEAP's parallel-beam configuration.

    Parameters:
        geometry (object): Contains projection angles.
        params (dict): Pixel and detector specifications.

    Returns:
        dict: {
            'geometry_type': 'parallel',
            'angles': list or np.ndarray,
            'num_rows': int,
            'num_cols': int,
            'pixel_width': float
        }
    """
    return {
        'geometry_type': 'parallel',
        'angles': geometry.angles,
        'num_rows': params['num_rows'],
        'num_cols': params['num_cols'],
        'pixel_width': params['detector_pixel_width']
    }

def build_cone_geometry(geometry, params):
    """
    Returns geometry parameters for LEAP's cone-beam configuration.

    Parameters:
        geometry (object): Contains projection angles and system distances.
        params (dict): Pixel and detector specifications.

    Returns:
        dict: {
            'geometry_type': 'cone',
            'angles': list or np.ndarray,
            'source_to_detector': float,
            'num_rows': int,
            'num_cols': int,
            'pixel_width': float
        }
    """
    return {
        'geometry_type': 'cone',
        'angles': geometry.angles,
        'source_to_detector': geometry.source_to_detector,
        'num_rows': params['num_rows'],
        'num_cols': params['num_cols'],
        'pixel_width': params['detector_pixel_width']
    }
