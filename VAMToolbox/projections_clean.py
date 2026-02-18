import sys
import os
import time
import tkinter as tk
from tkinter import filedialog
import cv2

import matplotlib.pyplot as plt
import numpy as np

import vamtoolbox.geometry
import vamtoolbox.imagesequence
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

# Choose projection backend: "leap" for LEAP projector, "astra" for ASTRA toolbox
projection_backend = "leap" 

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
    default_rot_vel=20,
    default_height=30,
    default_intensity_scales=None,
    default_size_scale=1,
    default_resin="LVUDMA",
):
    """Prompt (GUI) for an STL path and append a FileSpecs using its directory."""

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
        
        target_geo = vamtoolbox.geometry.TargetGeometry(
            stlfilename=file.stl_name, resolution=res
        )

        target_geo.show(show_bodies=True)

        # target_geo.save(file.sino_name)

        # Setup projection geometry
        N_angles = 360
        angles = np.linspace(0, 360 - 360 / N_angles, N_angles)
        # Reverse rotation direction (clockwise)
        angles = -angles

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
                angles=np.radians(angles),    # degrees → radians
                source_radius=100.0,          # TODO: tune to match your setup
                detector_distance=200.0,      # TODO: tune to match your setup
                helical_pitch=file.height     # axial distance per full rotation (example)
            )

            # Parameter dictionary for LEAP conversion
            params = {
                "geometry_type": "cone",             # helical needs cone/cone-parallel
                "num_rows": 1000,                    # detector rows 400
                "num_cols": 2000,                  # detector columns 512
                "voxel_size": 0.05,
                "volume_shape": volume.shape,
                "pixel_width": 0.05,         # usually same as voxel_size
                "pixel_height": 0.05,
            }

            # Wrapper class to make LEAP compatible with optimizer interface
            class LEAPProjectorWrapper:
                def __init__(self, leap_module, geometry, params):
                    self.leap = leap_module
                    self.geometry = geometry
                    self.params = params
                
                def forward(self, volume):
                    return self.leap.forward(volume, self.geometry, self.params)
                
                def backward(self, sinogram):
                    return self.leap.back(sinogram, self.geometry, self.params)
            
            projector = LEAPProjectorWrapper(leap3D, proj_geo, params)




        elif projection_backend == "astra":            
            proj_geo = vamtoolbox.geometry.ProjectionGeometry(
                angles=angles, ray_type="parallel", CUDA=False
            )
            projector = vamtoolbox.projectorconstructor.projectorconstructor(target_geo, proj_geo)


        else:
            raise ValueError("Unsupported projection backend. Choose 'leap' or 'astra'.")

        # Time only the optimizer execution
        print("\n--- Starting Optimizer ---")
        optimizer_start = time.time()
        sino, recon, error = optimize(target_geo, proj_geo, options, projector)
        optimizer_end = time.time()
        print(f"--- Optimizer Runtime: {optimizer_end - optimizer_start:.2f} seconds ---\n")

        recon = target_geo # reuse
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

# Convert physical pitch (mm / rev) → pixels / frame
mm_per_pixel = 0.05                     # must match ImageConfig scale
pixels_per_rev = file.height / mm_per_pixel
helical_pitch_pixels = pixels_per_rev / sino.shape[0]

export_sinogram_to_images(
    sinogram=sino,
    output_dir=projector_output_dir,
    image_size=(3840, 2160),            # projector resolution
    # image_size=(2000, 1000),                
    bit_depth=8,
    normalization_percentile=99.9,
    rotate_angle=90.0,             
    invert_u=False,
    invert_v=file.invert_v,
    helical_pitch_pixels=helical_pitch_pixels,
    # helical_pitch_pixels=0,
    # start_v_offset=0,
    start_v_offset=200,
)

# Crop + Make Helical Video
print("Creating cropped helical video...")

input_dir = projector_output_dir  
output_video = os.path.join(input_dir, "helical_crop_video.mp4")

# CROP SETTINGS
crop_height = 100  # height of the cropped band (pixels)
start_row = (2160 - crop_height) // 2    # vertical start row of the crop (top of band)

# Collect sorted image list
file_list = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
if not file_list:
    raise RuntimeError("No PNG files found in projection stack output folder.")

# Read the first image to get image dimensions
first_image = cv2.imread(os.path.join(input_dir, file_list[0]), cv2.IMREAD_GRAYSCALE)
height, width = first_image.shape

# Set up video writer (MP4 format)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, 30, (width, crop_height), isColor=False)

# Loop through each PNG image and crop
for fname in file_list:
    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cropped = img[start_row:start_row + crop_height, :]
    out.write(cropped)

out.release()
print(f"Saved cropped helical video to: {output_video}")