import sys
import os
import time
import tkinter as tk
from tkinter import filedialog
import numpy as np
import vamtoolbox.geometry
from vamtoolbox.optimize import optimize
from vamtoolbox.util.export_projection import export_sinogram_to_images
from leapctype import *

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
cbi_mode = "fps_opt"       # "fps_opt" or "spim"
# cbi_focal_offset = 0    # 0=center, int(radius*0.5)=50%, int(radius)=edge

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
    default_rot_vel=60,
    default_height=89.52,
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
        target_geo = vamtoolbox.geometry.TargetGeometry(
            stlfilename=file.stl_name, resolution=res
        )

        # target_geo.show(show_bodies=True)
        print("target_geo shape:", target_geo.array.shape)  
        # target_geo.save(file.sino_name)

        # Setup projection geometry
        sod = 200.0                     # source to object distance (mm)
        sdd = 600.0                     # source to detector distance (mm)
        magnification = sdd / sod       # = 3.0
        detector_pixel_mm = mm_per_pix * magnification  # 0.15 mm — physical detector pixel

        # PITCH SETTING (pixels/rev)
        # Set to 0 for standard projection (post-processed by crop_video.py)
        # Set to pixels/rev for helical projection (images played directly)

        helical_pitch_pix = 1600 
        
        pixels_per_rev = helical_pitch_pix
        helical_pitch_mm = helical_pitch_pix * mm_per_pix

        if helical_pitch_mm > 0:
            # Helical mode: N_rev rotations needed to cover full object height
            N_rev = int(np.ceil(res * mm_per_pix / helical_pitch_mm))
            N_angles = N_rev * 360
            angles = np.linspace(0, N_rev * 360 - 360 / N_angles, N_angles)
            projector_image_size = (2560, 1600)   # one band per frame, play directly
        else:
            # Standard mode: 360 projections, each shows full object — crop_video.py slides
            N_angles = 360
            angles = np.linspace(0, 359, N_angles)
            projector_image_size = (2560, 4800)   # tall image for crop_video.py window

        # Configure optimization

        options = vamtoolbox.optimize.Options(
            method="OSMO",
            n_iter=40,
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
                # helical_pitch=helical_pitch_mm,
                helical_pitch=file.height/3,
            )

            # opt_scale reduces detector resolution for faster optimization
            # Pixel size scales inversely so the detector still covers the full object
            opt_scale = 0.5
            if helical_pitch_mm > 0:
                # Helical: detector rows cover one pitch-band of height
                pitch_voxels = helical_pitch_mm / mm_per_pix  # e.g. 596.8 voxels
                num_rows = round(pitch_voxels * opt_scale * 1.05)  # 5% margin
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
            from vamtoolbox.projector import cbi3D

            proj_geo = vamtoolbox.geometry.ProjectionGeometry(angles=angles, ray_type="parallel", CUDA=True)
            
            vial_diameter_mm = 100
            radius = int((vial_diameter_mm / 2) / mm_per_pix)

            cbi_focal_offset = 0                      # center
            # cbi_focal_offset = int(radius * 0.5)   # 50% radius
            # cbi_focal_offset = int(radius)         # edge

            projector = cbi3D.CBIProjectorWrapper(
                angles_deg=angles,
                npix_axial=31,     
                npix_lateral=31,
                circle=False,
                mode=cbi_mode,
                focal_offset=cbi_focal_offset,
            )
            print(f"CBI mode: {cbi_mode}, focal_offset: {cbi_focal_offset}, radius: {radius}")

        else:
            raise ValueError("Unsupported projection backend. Choose 'leap', 'astra', or 'cbi'.")

        # Time only the optimizer execution
        print("\n--- Starting Optimizer ---")
        optimizer_start = time.time()
        sino, recon, error = optimize(target_geo, proj_geo, options, projector)
        optimizer_end = time.time()
        print(f"--- Optimizer Runtime: {optimizer_end - optimizer_start:.2f} seconds ---\n")

        # recon = target_geo # reuse
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

# Flip z-axis (rows) so base of object is at top of image (crop_video.py scrolls from top down)
sino = sino[:, ::-1, :]

# After 90° rotation: sino rows (axis 1) → v/height, sino cols (axis 2) → u/width
# Scale must fit both: cols×sf ≤ image_size[0] (width) AND rows×sf ≤ image_size[1] (height)
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
