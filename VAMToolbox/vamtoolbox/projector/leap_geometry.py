import numpy as np

def convert_geometry_to_leap_format(geometry, params):
    """
    Convert high-level LEAPGeometry object into arrays required by LEAP.

    geometry:
        .angles (radians)
        .source_radius
        .detector_distance
        .helical_pitch

    params:
        "geometry_type"
        "num_rows"
        "num_cols"
        "pixel_width"
    """

    geom_type = params.get("geometry_type", "modular")
    angles = np.asarray(geometry.angles).ravel()
    n = len(angles)

    src_R = float(geometry.source_radius)
    det_R = float(geometry.detector_distance)
    pitch = float(getattr(geometry, "helical_pitch", 0.0))

    # allocate arrays
    source_positions = np.zeros((n, 3), dtype=np.float32)
    detector_centers = np.zeros((n, 3), dtype=np.float32)
    rowVectors = np.zeros((n, 3), dtype=np.float32)
    colVectors = np.zeros((n, 3), dtype=np.float32)

    for i, theta in enumerate(angles):

        z = pitch * theta / (2*np.pi)  # helical height

        # source
        sx = src_R * np.cos(theta)
        sy = src_R * np.sin(theta)
        source_positions[i] = [sx, sy, z]

        # detector center (opposite direction)
        dx = -det_R * np.cos(theta)
        dy = -det_R * np.sin(theta)
        detector_centers[i] = [dx, dy, z]

        # detector axes
        # horizontal axis (u direction): tangent
        col = np.array([-np.sin(theta), np.cos(theta), 0], dtype=np.float32)
        col /= np.linalg.norm(col)
        colVectors[i] = col

        # vertical axis (v direction)
        rowVectors[i] = [0, 0, 1]

    return {
        "geometry_type": geom_type,
        "angles": angles,
        "source_positions": source_positions,
        "detector_centers": detector_centers,
        "rowVectors": rowVectors,
        "colVectors": colVectors,
        "num_rows": params["num_rows"],
        "num_cols": params["num_cols"],
        "pixel_width": params["pixel_width"],
        "pitch": pitch
    }


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

        # SOURCE POSITION (helical)
        # LEAP uses gantry rotation around z-axis
        # pitch grows linearly with angle
        z = pitch_per_radian * theta

        sx = R * np.cos(theta)
        sy = R * np.sin(theta)
        sz = z
        source = np.array([sx, sy, sz])

        # DETECTOR CENTER POSITION
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

        # DETECTOR AXES (u and v)
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

def _build_cone_geometry(geometry, params):
    """
    Build cone-beam geometry for LEAP.
    This assumes a circular cone-beam trajectory (like standard CT),
    optionally with helical motion.
    """

    angles = np.asarray(geometry.angles).ravel()
    n_views = len(angles)

    src_R = float(geometry.source_radius)
    det_R = float(geometry.detector_distance)
    pitch = float(getattr(geometry, "helical_pitch", 0.0))

    # allocate
    source_positions = np.zeros((n_views, 3), dtype=np.float32)
    detector_centers = np.zeros((n_views, 3), dtype=np.float32)
    rowVectors = np.zeros((n_views, 3), dtype=np.float32)
    colVectors = np.zeros((n_views, 3), dtype=np.float32)

    for i, theta in enumerate(angles):

        # helical z
        z = pitch * theta / (2 * np.pi)

        # source circular path
        sx = src_R * np.cos(theta)
        sy = src_R * np.sin(theta)

        # detector opposite side
        dx = -det_R * np.cos(theta)
        dy = -det_R * np.sin(theta)

        source_positions[i] = [sx, sy, z]
        detector_centers[i] = [dx, dy, z]

        # detector coordinate system
        # column (u direction): tangent
        col = np.array([-np.sin(theta), np.cos(theta), 0], dtype=np.float32)
        col /= np.linalg.norm(col)
        colVectors[i] = col

        # row (v direction): vertical axis
        rowVectors[i] = [0.0, 0.0, 1.0]

    geom = {
        "geometry_type": "cone",
        "source_positions": source_positions,
        "detector_centers": detector_centers,
        "rowVectors": rowVectors,
        "colVectors": colVectors,
        "num_rows": int(params["num_rows"]),
        "num_cols": int(params["num_cols"]),
        "pixel_width": float(params["pixel_width"]),
        "pixel_height": float(params.get("pixel_height", params["pixel_width"])),
    }
    return geom

