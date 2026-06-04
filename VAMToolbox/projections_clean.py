import sys
import os
import time
import tkinter as tk
from tkinter import filedialog
import numpy as np
import trimesh
import vamtoolbox.geometry
from vamtoolbox.optimize import optimize
from vamtoolbox.util.export_projection import export_sinogram_to_images
from leapctype import *
from cbi_toolbox import splineradon as spl
from deconv_corr import blur_ker, correct_blurring
from voxelize_backends import voxelize

# Refractive indices (RI) for various resin materials
RI = dict()
RI["LVUDMA"] = 1.51
RI["HVUDMA"] = 1.51
RI["PEGDA700"] = 1.5
RI["UW"] = 1.5
RI["GELMA"] = 1.34
RI["242N"] = 1.5

# Choose projection backend: "leap" for LEAP projector, "astra" for ASTRA toolbox, "cbi" for CBI toolbox
projection_backend = "leap"
cbi_mode = "fps_opt"               # "opt", "fps_opt", or "spim"

# Z-padding fraction (each side). Only applied in non-helical mode —
# in helical mode pre-pad just lengthens the scan trajectory (black frames + truncated top).
Z_PAD_FRAC = 0.30

# Number of revolutions for helical scan. Set to 0 for standard (non-helical) projection.
# 240mm object + 2×80mm padding = 400mm; pitch=20mm/rev → 20 revs
N_REV = 0

# Physical projector vertical resolution (rows per frame). Used as detector num_rows in helical mode.
PROJECTOR_ROWS = 1600

# Total volume height (mm) for helical mode. Object is padded symmetrically with zeros to reach this height. Set to 0 to disable (use object's own height).
HELICAL_VOLUME_HEIGHT_MM = 310.0   # 240mm object + 80mm padding each side


class FileSpecs:
    def __init__(
        self,
        dir,
        stl_name,
        resin,
        sino_name,
        rot_vel,
        height,
        intensity_scales,
        size_scale,
        print_bodies="all",
        array_num=1,
        array_offset=0,
        invert_v=False,
        v_offset=0,
    ):
        self.dir = dir
        self.stl_name = self.dir + stl_name
        self.sino_name = self.dir + sino_name
        self.rot_vel = rot_vel
        self.height = height
        self.print_bodies = print_bodies
        self.resin = resin
        self.refractive_index = RI[self.resin]
        self.video_name = (
            self.dir + self.resin + "_" + sino_name + "_%ddegps" % self.rot_vel
        )
        self.intensity_scales = intensity_scales
        self.size_scale = size_scale
        self.array_num = array_num
        self.array_offset = array_offset
        self.invert_v = invert_v
        self.v_offset = v_offset

# File specifications
files = list()

# User-selected STL
def _prompt_stl_path_with_dialog():
    """Open a native file picker for STL selection; returns empty string if cancelled."""
    try:
        root = tk.Tk()
        root.withdraw()
        root.update()
        path = filedialog.askopenfilename(
            title="Select STL file",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
        )
        root.destroy()
        return path or ""
    except Exception as e:
        print(f"File dialog failed ({e}); falling back to manual path entry.")
        return input("Enter path to STL file (press Enter to skip): ").strip()

def append_user_selected_filespec(
    default_rot_vel=54,
    default_height=89.52, # building: 89.52
    default_intensity_scales=None,
    default_size_scale=1,
    default_resin="LVUDMA",
):

# GUI) for an STL path and append a FileSpecs using its directory

    if default_intensity_scales is None:
        default_intensity_scales = [1, 2]

    user_path = _prompt_stl_path_with_dialog().strip()

    if not user_path:
        print("No STL selected; skipping user-defined file.")
        return

    stl_path = os.path.abspath(os.path.expanduser(user_path))

    if not os.path.isfile(stl_path):
        raise FileNotFoundError(f"STL file not found: {stl_path}")

    dir_path = os.path.dirname(stl_path)
    stl_name = os.path.basename(stl_path)
    sino_name = os.path.splitext(stl_name)[0]

    # Ensure trailing separator because FileSpecs concatenates dir + filename
    dir_with_sep = dir_path if dir_path.endswith(os.sep) else dir_path + os.sep

    files.append(
        FileSpecs(
            dir=dir_with_sep,
            stl_name=stl_name,
            sino_name=sino_name,
            rot_vel=default_rot_vel,
            height=default_height,
            intensity_scales=default_intensity_scales,
            size_scale=default_size_scale,
            resin=default_resin,
        )
    )
    print(f"Added STL: {stl_path}")


append_user_selected_filespec()

for file in files:
    
    print(f"Using backend: {projection_backend}")

    try:
        sino = vamtoolbox.geometry.loadVolume(file.sino_name + ".sino")
        print("Loaded sinogram: %s" % (file.sino_name + ".sino"))

    except Exception as e:
        print(
            "Failed loading sinogram: %s. Resorting to optimization of stl."
            % (file.sino_name + ".sino")
        )

        # Setup target geometry
        mm_per_pix = 50 / 1000
        res = round(file.height / mm_per_pix)
        print("Resolution: %d" % res)

        # target_geo = vamtoolbox.geometry.TargetGeometry(stlfilename=file.stl_name,resolution=res,bodies={'print':[2],'insert':[1]})
        
        print("file height:", file.height)

        # "optimize" = run OSMO optimization, "direct" = forward projection only
        projection_mode = "optimize"

        # Voxelization backend: "pyvista" (legacy, memory-heavy),
        # "trimesh" (fast CPU, fails on dense meshes with many faces),
        # "voxelizer" (VAMToolbox OpenGL slicer, best for angular/complex meshes)
        voxelize_backend = "voxelizer"

        if voxelize_backend == "trimesh_ray":
            # z-padding policy lives here, not in the backend
            nz_est = round(file.height / mm_per_pix)
            if N_REV == 0:
                z_pad_lo = z_pad_hi = round(Z_PAD_FRAC * nz_est)
            elif HELICAL_VOLUME_HEIGHT_MM > 0:
                target_nz = round(HELICAL_VOLUME_HEIGHT_MM / mm_per_pix)
                extra = max(0, target_nz - nz_est)
                z_pad_lo = extra // 2
                z_pad_hi = extra - z_pad_lo
            else:
                z_pad_lo = z_pad_hi = 0
            target_geo = voxelize(
                "trimesh_ray", file.stl_name, mm_per_pix,
                z_pad_lo=z_pad_lo, z_pad_hi=z_pad_hi,
                direct_mode=(projection_mode == "direct"),
            )
        else:
            target_geo = voxelize(
                voxelize_backend, file.stl_name, mm_per_pix, resolution=res,
            )
        
        # Deconvolution correction 
        APPLY_DECONV = True

        if APPLY_DECONV:
            print("Calculating PSF kernel...")
            dker = blur_ker(mm_per_pix, 0.000151, 60, file.rot_vel)

            print("Running deconvolution...")
            corrected_arr = correct_blurring(
                dker, target_geo.array.astype(float), n=3
            )
            target_geo.array = np.clip(corrected_arr, 0, 1).astype(np.float32)
            print(f"Deconv done. shape: {target_geo.array.shape}")

        # print("target shape:", target_geo.array.shape)  
        # target_geo.save(file.sino_name)

        # Setup projection geometry
        sod = 200.0                     # source to object distance (mm)
        sdd = 600.0                     # source to detector distance (mm)
        magnification = sdd / sod       # = 3.0
        detector_pixel_mm = mm_per_pix * magnification  # 0.15 mm — physical detector pixel

        # PITCH SETTING
        # Option A: physical pitch from number of revolutions (recommended)
        # Set N_REV = 0 (top of file) for standard (non-helical) projection
        N_rev = N_REV
        # Use actual mesh height (from voxelized array), not file.height
        actual_height_mm = target_geo.array.shape[2] * mm_per_pix
        helical_pitch_mm = actual_height_mm / N_rev if N_rev > 0 else 0.0
        print(f"[helical] actual mesh height: {actual_height_mm:.1f}mm, "
              f"pitch: {helical_pitch_mm:.1f}mm/rev, N_rev: {N_rev}")

        # # Option B (old): pitch in pixels, converted via voxel size
        # # Set to 0 for standard projection (post-processed by crop_video.py)
        # # Set to pixels/rev for helical projection (images played directly)
        # helical_pitch_pix = 1600
        # pixels_per_rev = helical_pitch_pix
        # helical_pitch_mm = helical_pitch_pix * mm_per_pix

        if helical_pitch_mm > 0:
            # Helical mode
            N_angles = N_rev * 360
            angles = np.linspace(0, N_rev * 360 - 360 / N_angles, N_angles)
            projector_image_size = (2560, 1600)   # one band per frame, play directly
        else:
            # Standard mode: 360 projections, each shows full object — crop_video.py slides
            N_angles = 360
            angles = np.linspace(0, 359, N_angles)
            projector_image_size = (2560, 3000)   # tall image for crop_video.py window

        # Configure optimization

        options = vamtoolbox.optimize.Options(
            method="OSMO",
            n_iter=50,
            d_h=0.85,
            d_l=0.65,
            learning_rate=0.005,
            filter="hanning",
            verbose="plot",
        )

        # Run optimization
        if file.print_bodies == "all":
            target_geo.insert = None  # TODO find way to integrate flag for attenuation computation even if stl is multi body


        if projection_backend == "leap":
            from vamtoolbox.projector import leap3D

            # Volume from STL
            volume = target_geo.array
            # volume = np.transpose(volume, (2, 1, 0))  # ZYX -> XYZ (for LEAP)


            # Define a simple geometry container for LEAP
            class LEAPGeometry:
                def __init__(self, angles, source_radius, detector_distance, helical_pitch=0.0):
                    # angles should be in radians
                    self.angles = angles
                    self.source_radius = source_radius
                    self.detector_distance = detector_distance
                    self.helical_pitch = helical_pitch  # axial distance per 360° rotation
                    self.absorption_coeff = None  # Add missing attribute

            # Create LEAP geometry instance
            proj_geo = LEAPGeometry(
                angles=np.radians(angles),
                source_radius=sod,
                detector_distance=sdd - sod,          # = 400 mm
                helical_pitch=helical_pitch_mm,
            )

            # opt_scale reduces detector resolution for faster optimization
            # Pixel size scales inversely so the detector still covers the full object
            opt_scale = 1
            if helical_pitch_mm > 0:
                # Helical: detector size matches the physical projector. Source/detector
                # together scroll through the volume; object slides within each frame.
                num_rows = round(PROJECTOR_ROWS * opt_scale)
            else:
                # Standard: detector rows cover full object height + 5% margin
                num_rows = round(volume.shape[2] * opt_scale * 1.05)  # ≈ 940 rows
            # Detector cols: covers full object diameter (nx), scaled
            num_cols = round(volume.shape[0] * opt_scale)  # ≈ 191 cols
            # Pixel must be larger so fewer pixels still span the whole FOV
            pixel_size_opt = detector_pixel_mm / opt_scale  # = 0.30 mm

            params = {
                "geometry_type": "cone",
                "num_rows": num_rows,
                "num_cols": num_cols,
                "voxel_size": mm_per_pix,
                "volume_shape": volume.shape,
                "pixel_width": pixel_size_opt,
                "pixel_height": pixel_size_opt,
            }
            
            projector = leap3D.LEAPProjectorWrapper(proj_geo, params)

        elif projection_backend == "astra":            
            proj_geo = vamtoolbox.geometry.ProjectionGeometry(angles=angles, ray_type="parallel", CUDA=True)

            projector = vamtoolbox.projectorconstructor.projectorconstructor(target_geo, proj_geo)

        elif projection_backend == "cbi":
            import copy
            from vamtoolbox.projector import cbi3D
            from scipy.ndimage import zoom as _zoom

            cbi_scale = 0.3   # downsample; adjust: 0.1=38px, 0.3=114px, 0.5=191px

            # Pre-compute zoomed arrays from originals (no padding)
            arr = _zoom(target_geo.array, cbi_scale).astype(np.float32)
            zoomed_attrs = {}
            for _attr in ("void_inds", "gel_inds"):
                if getattr(target_geo, _attr, None) is not None:
                    zoomed_attrs[_attr] = _zoom(getattr(target_geo, _attr).astype(np.float32), cbi_scale) > 0.5

            # Temporarily null large arrays so deepcopy is fast
            _saved = {a: getattr(target_geo, a, None) for a in ["array", "void_inds", "gel_inds"]}
            for a in _saved: setattr(target_geo, a, None)
            target_geo_cbi = copy.deepcopy(target_geo)
            for a, v in _saved.items(): setattr(target_geo, a, v)  # restore originals

            # Assign small downsampled arrays to cbi copy
            target_geo_cbi.array = arr
            for _attr, m in zoomed_attrs.items():
                setattr(target_geo_cbi, _attr, m)
            print(f"CBI: downsampled+padded volume to {target_geo_cbi.array.shape}")

            proj_geo = vamtoolbox.geometry.ProjectionGeometry(angles=angles, ray_type="parallel", CUDA=True)

            vial_diameter_mm = 100
            radius = int((vial_diameter_mm / 2) / mm_per_pix * cbi_scale)

            # cbi_focal_offset = 0                      # center
            # cbi_focal_offset = int(radius * 0.5)   # 50% radius
            cbi_focal_offset = int(radius)         # edge

            zx_size = target_geo_cbi.array.shape[0]
            # opt() requires PSF axial >= ZX size; fps_opt sums PSF along Z so 31 is fine
            psf_axial = (zx_size if zx_size % 2 == 1 else zx_size + 1) if cbi_mode == "opt" else 31
            projector = cbi3D.CBIProjectorWrapper(
                angles_deg=angles,
                npix_axial=psf_axial,
                npix_lateral=31,
                circle=False,
                mode=cbi_mode,
                focal_offset=cbi_focal_offset,
            )
            print(f"CBI mode: {cbi_mode}, focal_offset: {cbi_focal_offset}, radius: {radius}")
            target_geo = target_geo_cbi  # use downsampled copy for optimizer

        else:
            raise ValueError("Unsupported projection backend. Choose 'leap', 'astra', or 'cbi'.")

        if projection_mode == "direct":
            print("\n--- Direct Forward Projection (no optimization) ---")
            proj_start = time.time()
            sino = projector.forward(target_geo.array)
            sino = np.clip(sino, 0, None)
            proj_end = time.time()
            print(f"--- Forward Projection Runtime: {proj_end - proj_start:.2f} seconds ---")
            print(f"sino shape: {sino.shape}, max: {sino.max():.4f}, min: {sino.min():.4f}")
            recon = target_geo
            error = None
        else:
            # Time only the optimizer execution
            print("\n--- Starting Optimizer ---")
            optimizer_start = time.time()
            sino, recon, error = optimize(target_geo, proj_geo, options, projector)
            optimizer_end = time.time()
            print(f"--- Optimizer Runtime: {optimizer_end - optimizer_start:.2f} seconds ---\n")

        # if projection_backend == "cbi":
        #     from cbi_toolbox import splineradon as spl
        #     clean_sino = spl.radon(
        #         recon.array if hasattr(recon, 'array') else recon,
        #         theta=angles,
        #         circle=False,
        #     )
        #     clean_sino = np.clip(clean_sino, 0, None)
        #     sino = clean_sino

        # recon = target_geo # reuse
        if hasattr(recon, 'show'):
            recon.show()

        if hasattr(sino, "save"):
            sino.save(file.sino_name)
        else:
            np.save(file.sino_name + ".npy", sino)
            print(f"Saved ndarray sinogram to {file.sino_name}.npy")

    print("Rebinning of original sinogram.")

    # Rebin for vial refraction correction
    sino = sino if isinstance(sino, np.ndarray) else getattr(sino, "array", None)
    if sino is None:
        raise TypeError("Sinogram is neither ndarray nor has an 'array' attribute.")
    print(sino.shape)

    N_angles = sino.shape[0]


# Export helical projector images

projector_output_dir = os.path.join(file.dir, "projection_output")

image_size = projector_image_size
print("sino.shape:", sino.shape)
print("image_size:", image_size)

# Per-angle max to check which frames have content
per_angle_max = np.array([sino[k].max() for k in range(sino.shape[0])])
first_nonzero = np.argmax(per_angle_max > 0)
last_nonzero = sino.shape[0] - 1 - np.argmax(per_angle_max[::-1] > 0)
print(f"[sino diagnostic] per-angle max: first 10 = {per_angle_max[:10]}")
print(f"[sino diagnostic] first nonzero frame: {first_nonzero}, last: {last_nonzero}, "
      f"total with content: {np.count_nonzero(per_angle_max)}/{sino.shape[0]}")

# Trim z-padding rows from sinogram so object fills full frame
# In helical mode: trim based on actual helical padding amounts
# In standard mode: trim based on Z_PAD_FRAC
if N_REV > 0 and HELICAL_VOLUME_HEIGHT_MM > 0:
    # # Helical mode: calculate trim from actual padding amounts
    # # We padded the object from nz to target_nz pixels
    # # The trim needs to match the padding amounts proportionally
    # actual_height_mm = target_geo.array.shape[2] * mm_per_pix  # Should be HELICAL_VOLUME_HEIGHT_MM
    # target_nz = round(HELICAL_VOLUME_HEIGHT_MM / mm_per_pix)
    # object_nz = actual_height_mm / mm_per_pix if actual_height_mm < HELICAL_VOLUME_HEIGHT_MM else None
    
    # # Only trim if we have room to trim (sino.shape[1] should equal PROJECTOR_ROWS in helical)
    # # In helical, rows represent the detector height (1600px), so minimal/no trimming needed
    # trim_rows = 0
    # print(f"[helical trim] sino.shape[1]={sino.shape[1]}, PROJECTOR_ROWS={PROJECTOR_ROWS}, "
    #       f"HELICAL_VOLUME_HEIGHT_MM={HELICAL_VOLUME_HEIGHT_MM}, trim_rows={trim_rows}")
    pass
else:
    # Standard mode: use Z_PAD_FRAC-based calculation
    pad_frac = Z_PAD_FRAC / (1 + 2 * Z_PAD_FRAC)
    trim_rows = round(pad_frac * sino.shape[1])
    print(f"[standard trim] Z_PAD_FRAC={Z_PAD_FRAC}, pad_frac={pad_frac}, trim_rows={trim_rows}")

# if trim_rows > 0:
#     sino = sino[:, trim_rows:-trim_rows, :]
#     print(f"[trim] removed {trim_rows} padding rows from each side, "
#           f"sino now: {sino.shape}")

# if trim_rows > 0:
#     sino = sino[:, trim_rows:-trim_rows, :]
#     print(f"[trim] removed {trim_rows} padding rows from each side, "
#           f"sino now: {sino.shape}")

# Flip z-axis (rows) so base of object is at top of image (crop_video.py scrolls from top down)
sino = sino[:, ::-1, :]
print(f"[after flip] sino.shape: {sino.shape}")

# Diagnostic: show object position after flip
nz_per_row_after_flip = np.count_nonzero(sino, axis=(0, 2))
if np.max(nz_per_row_after_flip) > 0:
    first_row_with_content = np.argmax(nz_per_row_after_flip > 0)
    last_row_with_content = sino.shape[1] - 1 - np.argmax(nz_per_row_after_flip[::-1] > 0)
    print(f"[after flip] Object rows: {first_row_with_content} to {last_row_with_content}, "
          f"span={last_row_with_content - first_row_with_content + 1} rows")
    print(f"[after flip] First frame sample max: {np.max(sino[0])}, "
          f"Last frame sample max: {np.max(sino[-1])}")
print()  # blank line for readability

# export_sinogram_to_images expects (T, rows, cols) with T in axis 0
# CBI sino is (T, Y=height, P=diameter) — swap rows/cols to (T, P, Y) so that
# after internal .T in _insertImage, S_v=Y (tall) and S_u=P (narrow)
if projection_backend == "cbi":
    sino = sino.transpose(0, 2, 1)  # (T, Y, P) → (T, P, Y)

# CBI (no rotate): S_v = sino.shape[2]=Y → v, S_u = sino.shape[1]=P → u
# LEAP (rotate 90°): S_v = sino.shape[0], S_u = sino.shape[2]
if projection_backend == "cbi":
    size_scale = min(image_size[0] / sino.shape[1], image_size[1] / sino.shape[2])
else:
    size_scale = min(image_size[0] / sino.shape[2], image_size[1] / sino.shape[1])

export_sinogram_to_images(
    sinogram=sino,
    output_dir=projector_output_dir,
    image_size=image_size,
    bit_depth=8,
    normalization_percentile=99.9,
    rotate_angle=90.0,
    invert_u=False,
    invert_v=False,
    v_offset=0,
    size_scale=size_scale,
)
