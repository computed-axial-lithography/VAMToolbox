"""
Voxelization backends for projections_clean.py.
Each function turns an STL into a TargetGeometry (or a duck-typed SimpleTarget for direct mode). The dispatcher `voxelize()` picks one by name.
"""

import numpy as np
import trimesh
import vamtoolbox.geometry
import vamtoolbox.voxelize


class SimpleTarget:
    """Stand-in for TargetGeometry exposing only `.array`.

    Skips the circle-clip + gel/void indexing TargetGeometry does, which
    are unnecessary in direct (no-optimization) mode.
    """
    def __init__(self, array):
        self.array = array


def voxelize_pyvista(stl_path, resolution):
    return vamtoolbox.geometry.TargetGeometry(
        stlfilename=stl_path, resolution=resolution
    )


def voxelize_trimesh(stl_path, mm_per_pix):
    mesh = trimesh.load(stl_path)
    mesh.apply_translation(-mesh.centroid)
    xmin, ymin, _ = mesh.bounds[0]
    xmax, ymax, _ = mesh.bounds[1]

    arr = np.asarray(mesh.voxelized(mm_per_pix).fill().matrix, dtype=np.float32)

    d = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)
    N = int(np.ceil(d / mm_per_pix))
    px, py = max(0, N - arr.shape[0]), max(0, N - arr.shape[1])
    arr = np.pad(arr, ((px // 2, px - px // 2),
                       (py // 2, py - py // 2),
                       (0, 0)))
    return vamtoolbox.geometry.TargetGeometry(target=arr)


def voxelize_voxelizer(stl_path, mm_per_pix):
    v = vamtoolbox.voxelize.Voxelizer()
    v.addMeshes({stl_path: "print"})
    arr = v.voxelize(
        body_name="print",
        layer_thickness=mm_per_pix,
        voxel_value=1.0,
        voxel_dtype="float32",
        square_xy=True,
    )
    return vamtoolbox.geometry.TargetGeometry(target=arr)


def voxelize_trimesh_ray(stl_path, mm_per_pix,
                          z_pad_lo=0, z_pad_hi=0, direct_mode=False):
    """Ray-cast voxelization. Low memory, watertight-safe."""
    mesh = trimesh.load(stl_path)
    # bbox center, not centroid — asymmetric meshes have centroid != bbox center
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2)
    xmin, ymin, zmin = mesh.bounds[0]
    xmax, ymax, zmax = mesh.bounds[1]

    nx = int(np.ceil((xmax - xmin) / mm_per_pix))
    ny = int(np.ceil((ymax - ymin) / mm_per_pix))
    nz = int(np.ceil((zmax - zmin) / mm_per_pix))

    xs = xmin + (np.arange(nx) + 0.5) * mm_per_pix
    ys = ymin + (np.arange(ny) + 0.5) * mm_per_pix
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    origins = np.stack(
        [X.ravel(), Y.ravel(), np.full(X.size, zmin - 1.0)], axis=1
    )
    directions = np.tile([0.0, 0.0, 1.0], (origins.shape[0], 1))

    # shoot one ray per (x,y) column, collect z-hits, fill between pairs
    locs, ray_idx, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=True
    )
    arr = np.zeros((nx, ny, nz), dtype=np.float32)
    order = np.argsort(ray_idx, kind="stable")
    ray_idx = ray_idx[order]
    z_hits = locs[order, 2]

    starts = np.searchsorted(ray_idx, np.arange(nx * ny), side="left")
    ends = np.searchsorted(ray_idx, np.arange(nx * ny), side="right")
    for col, (s, e) in enumerate(zip(starts, ends)):
        zs = np.sort(z_hits[s:e])
        if zs.size < 2:
            continue
        ix, iy = divmod(col, ny)
        for k in range(0, zs.size - 1, 2):
            z0 = int(np.floor((zs[k] - zmin) / mm_per_pix))
            z1 = int(np.ceil((zs[k + 1] - zmin) / mm_per_pix))
            arr[ix, iy, max(0, z0):min(nz, z1)] = 1.0

    # square xy + caller-supplied z padding
    N = max(nx, ny)
    px, py = N - nx, N - ny
    arr = np.pad(arr, ((px // 2, px - px // 2),
                       (py // 2, py - py // 2),
                       (z_pad_lo, z_pad_hi)))

    if direct_mode:
        return SimpleTarget(arr)
    return vamtoolbox.geometry.TargetGeometry(target=arr)


def voxelize(backend, stl_path, mm_per_pix, **kwargs):
    """Dispatcher.

    backend     : 'pyvista' | 'trimesh' | 'voxelizer' | 'trimesh_ray'
    stl_path    : path to STL file
    mm_per_pix  : voxel size (mm); unused by 'pyvista'

    Backend-specific kwargs:
        pyvista       : resolution (required)
        trimesh_ray   : z_pad_lo, z_pad_hi, direct_mode (optional)
    """
    if backend == "pyvista":
        return voxelize_pyvista(stl_path, kwargs["resolution"])
    if backend == "trimesh":
        return voxelize_trimesh(stl_path, mm_per_pix)
    if backend == "voxelizer":
        return voxelize_voxelizer(stl_path, mm_per_pix)
    if backend == "trimesh_ray":
        return voxelize_trimesh_ray(
            stl_path, mm_per_pix,
            z_pad_lo=kwargs.get("z_pad_lo", 0),
            z_pad_hi=kwargs.get("z_pad_hi", 0),
            direct_mode=kwargs.get("direct_mode", False),
        )
    raise ValueError(f"Unknown voxelize_backend: {backend}")
