import sys
import os
import time
import tkinter as tk
from tkinter import filedialog
import cv2


import matplotlib.pyplot as plt
import numpy as np
import trimesh
import trimesh.viewer

import vamtoolbox.geometry
import vamtoolbox.imagesequence
from vamtoolbox.optimize import optimize
from vamtoolbox.util.export_projection import export_sinogram_to_images
from leapctype import *


RI = dict()
RI["LVUDMA"] = 1.51
RI["HVUDMA"] = 1.51
RI["PEGDA700"] = 1.5
RI["UW"] = 1.5
RI["GELMA"] = 1.34
RI["242N"] = 1.5

start_all = time.time()

projection_backend = "astra" # "leap" or "astra"

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


# ## Helical projection geometry

# import numpy as np

# def helical_projection_geometry(
#     sinogram,
#     k,                          # projection index (angle index)
#     image_shape=(2160, 3840),   # projector resolution
#     helical_pitch=10.0,         # mm per 360°
#     z_max=10.0,                 # object height (mm)
#     v_mode="center",            # how to choose detector column
#     ):
#     """
#     Geometry-only mapping:
#     (theta, z) -> screen (X, Y)
#     using helical pitch.
#     """

#     H, W = image_shape
#     img = np.zeros((H, W), dtype=np.float32)

#     N_angles, N_rows, N_cols = sinogram.shape

#     # --- screen center ---
#     Xc, Yc = W // 2, H // 2
#     r_max = min(Xc, Yc) - 2

#     # --- angle for this projection ---
#     theta_k = 2 * np.pi * k / N_angles

#     # --- physical z from helical pitch ---
#     z_k = (helical_pitch / (2 * np.pi)) * theta_k

#     if z_k < 0 or z_k > z_max:
#         return img  # nothing to expose

#     # --- sinogram row (u) from z ---
#     u = int((z_k / z_max) * (N_rows - 1))

#     # --- detector column (v) ---
#     if v_mode == "center":
#         v = N_cols // 2
#     else:
#         raise NotImplementedError("Only center v is implemented.")

#     # --- screen radius from z ---
#     r = r_max * (z_k / z_max)

#     # --- screen coordinates ---
#     X = int(Xc + r * np.cos(theta_k))
#     Y = int(Yc + r * np.sin(theta_k))

#     if 0 <= X < W and 0 <= Y < H:
#         value = sinogram[k, u, v]

#         sigma = 3.0                 # PSF width (pixels)
#         radius = int(3 * sigma)     # truncate Gaussian at 3σ

#         for dy in range(-radius, radius + 1):
#             for dx in range(-radius, radius + 1):
#                 yy = Y + dy
#                 xx = X + dx

#                 if 0 <= yy < H and 0 <= xx < W:
#                     r2 = dx*dx + dy*dy
#                     weight = np.exp(-r2 / (2 * sigma * sigma))
#                     img[yy, xx] += value * weight

#     return img



# File specifications
files = list()


# #  Bars
# files.append(FileSpecs(dir=r"G:\Shared drives\taylorlab\CAL Projects\SpaceCAL\Metal Printing\Parts\\",
# stl_name=r"1000mc.stl",
# sino_name="beam1000mc",
# rot_vel=54,
# height=5,
# intensity_scales=[1,2,3,4,5],
# size_scale=1,
# array_num=2,
# array_offset=20,
# resin="PEGDA700"))

#  Thinker
# files.append(FileSpecs(dir=r"C:\Users\wadde\Documents\VAMToolbox\Parts\\",
# stl_name=r"thinker.stl",
# sino_name="thinker",
# rot_vel=36,
# height=10/2,
# intensity_scales=[2],
# size_scale=2,
# resin="PEGDA700"))

# files.append(FileSpecs(dir=r"G:\Shared drives\taylorlab\CAL Projects\SpaceCAL\Metal Printing\Parts\\",
# stl_name=r"lithium.stl",
# sino_name="lithium",
# rot_vel=36,
# height=25.4/2,
# intensity_scales=[2,3,7],
# size_scale=2,
# resin="PEGDA700"))

# files.append(FileSpecs(dir=r"C:\Users\wadde\Documents\VAMToolbox\Parts\\",
# stl_name=r"DogBone2mmV1.stl",
# sino_name="DogBone2mmV2",
# rot_vel=54,
# height=18,
# intensity_scales=[2,3],
# size_scale=1,
# array_num=2,
# array_offset=700,
# resin="PEGDA700"))

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
    default_height=10 / 2,
    default_intensity_scales=None,
    default_size_scale=2,
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

#  Thinker
# files.append(FileSpecs(dir=r"C:\Users\wadde\\Documents\VAMToolbox\Parts\MISSE\\",
# stl_name=r"Cylinder.stl",
# sino_name="cylinder",
# rot_vel=54,
# height=7.62,
# intensity_scales=[1,2],
# size_scale=1,
# resin="LVUDMA"))

# files.append(FileSpecs(dir=r"G:\Shared drives\3DHL + OptiCAL + TFT + SpaceCAL\SpaceCAL\Metal Printing\Parts\\",
# stl_name=r"1mm-0gap.stl",
# sino_name="camp90",
# rot_vel=54,
# height=90 ,
# intensity_scales=[2,3,7],
# size_scale=1,
# resin="LVUDMA"))

#  Thinker
# files.append(FileSpecs(dir=r"C:\Users\wadde\Documents\VAMToolbox\Parts\\",
# stl_name=r"1mm-0gap.stl",
# sino_name="teeth3",
# rot_vel=36,
# height=11/2,
# intensity_scales=[5],
# size_scale=2,
# resin="LVUDMA"))


# files.append(FileSpecs(dir=r"C:\Users\wadde\Documents\VAMToolbox\Parts\\",
# stl_name=r"benchyv2.stl",
# sino_name="benchyv3",
# rot_vel=54,
# height=18,
# intensity_scales=[4],
# size_scale=1,
# array_num=2,
# array_offset=700,
# resin="PEGDA700"))

# files.append(FileSpecs(dir=r"C:\Users\wadde\Documents\VAMToolbox\Parts\\",
# stl_name=r"shuttlev5.stl",
# sino_name="shuttlev6",
# rot_vel=54,
# height=27,
# intensity_scales=[2,3],
# size_scale=1,
# array_num=2,
# array_offset=680,
# resin="PEGDA700"))


# #  Truss
# files.append(FileSpecs(dir=r"G:\Shared drives\taylorlab\CAL Projects\SpaceCAL\Metal Printing\Parts\\",
# stl_name=r"truss.stl",
# sino_name="truss",
# rot_vel=36,
# height=15/2,
# intensity_scales=[1,2,3],
# size_scale=2,
# resin="PEGDA700"))

# files.append(FileSpecs(dir=r"G:\Shared drives\taylorlab\CAL Projects\SpaceCAL\Metal Printing\Parts\\",
# stl_name=r"cube2.stl",
# sino_name="cube2",
# rot_vel=36,
# height=12/2,
# intensity_scales=[5,7,11],
# size_scale=2,
# resin="PEGDA700"))


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

        # target_geo.show(show_bodies=True)

        # target_geo.save(file.sino_name)

        # Setup projection geometry
        N_angles = 360
        angles = np.linspace(0, 360 - 360 / N_angles, N_angles)

        # Configure optimization

        options = vamtoolbox.optimize.Options(
            method="OSMO",
            n_iter=96,
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
            # print("Leap projector imported!")

            # Volume from STL
            volume = target_geo.array

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
                "num_rows": 400,                    # detector rows 400
                "num_cols": 512,                  # detector columns 512
                "voxel_size": 0.1,
                "volume_shape": volume.shape,
                "pixel_width": 0.1,         # usually same as voxel_size
                "pixel_height": 0.1,
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
            from vamtoolbox.projector import projector3D as projector
            # print("Astra projector imported!")

            proj_geo = vamtoolbox.geometry.ProjectionGeometry(
                angles=angles, ray_type="parallel", CUDA=False
            )

            # sino = projector.forward_project(target_geo.volume, proj_geo)

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
    # sino_rebinned = vamtoolbox.geometry.rebinFanBeam(sinogram=sino,vial_width=1030,N_screen=(1280,800),n_write=1.51,throw_ratio=2.62825) #2.62825
    # print(sino_rebinned.array.shape)

    # sino = sino_rebinned
    # vamtoolbox.Display.showVolumeSlicer(sino.sinogram,type="proj")
    # plt.show()

    # sino_rebinned.save(file.sino_name_rebinned)

    # sino.show()
    # sino_rebinned.show()


    # # generate the projection sequence video
    # for intensity_scale in file.intensity_scales:
    #     # Save imageseq and video file
    #     config = vamtoolbox.imagesequence.ImageConfig(
    #         (2560, 1600),
    #         intensity_scale=intensity_scale,
    #         size_scale=file.size_scale,
    #         array_num=file.array_num,
    #         array_offset=file.array_offset,
    #         invert_v=file.invert_v,
    #         v_offset=0,
    #         normalization_percentile=99.9,
    #     )
    #     imgset = vamtoolbox.imagesequence.ImageSeq(config, sinogram=sino)

    #     imgset.saveAsVideo(
    #         save_path=f"{file.video_name}_intensity{intensity_scale}xarray{file.array_num}{file.video_name}_projection_sequence.mp4",
    #         rot_vel=36,
    #         num_loops=5,  # 30
    #         preview=True,
    #     )
    #     print("Saved video: %s" % (file.video_name + ".mp4"))


    N_angles = sino.shape[0]


    # # ---- video writer setup ----
    # H, W =  2160, 3840
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # video_path = f"{file.video_name}__helical_path_visualization.mp4"

    # out = cv2.VideoWriter(
    #     video_path,
    #     fourcc,
    #     file.rot_vel,   
    #     (W, H),
    #     isColor=False
    # )

    # N_angles = sino.shape[0]

    # for k in range(N_angles):
    #     img_k = helical_projection_geometry(
    #         sinogram=sino,
    #         k=k,
    #         image_shape=(H, W),
    #         helical_pitch=file.height,
    #         z_max=file.height,
    #     )

    #     # normalize to 8-bit
    #     frame = img_k
    #     frame = frame - frame.min()
    #     if frame.max() > 0:
    #         frame = frame / frame.max()
    #     frame = (frame * 255).astype(np.uint8)

    #     out.write(frame)

    # out.release()
    # print(f"Saved helical video: {video_path}")


    # # ============================================================================
    # # Backprojection to reconstruct 3D volume from sinogram
    # # ============================================================================
    if projection_backend == "leap":
    #     print("Starting LEAP backprojection...")
        
    #     # Reconstruction parameters
    #     recon_params = {
    #         "geometry_type": "cone",
    #         "num_rows": params["num_rows"],
    #         "num_cols": params["num_cols"],
    #         "voxel_size": params["voxel_size"],
    #         "volume_shape": volume.shape,  # Same shape as original volume
    #         "pixel_width": params["pixel_width"],
    #         "pixel_height": params["pixel_height"],
    #         "use_iterative_filter": False,  # Set to True for TV regularization
    #         "tv_delta": 0.02 / 20.0,  # Only used if use_iterative_filter=True
    #     }
        
        # Perform backprojection using the wrapper
        # reconstructed_volume = projector.backward(sino)
    
        # Extract the numpy array from the Reconstruction object
        reconstructed_volume = recon.array if hasattr(recon, 'array') else recon
        
        print(f"Reconstructed volume shape: {reconstructed_volume.shape}")
        
        # Save the reconstructed volume
        recon_path = file.sino_name + "_reconstructed.npy"
        np.save(recon_path, reconstructed_volume)
        print(f"Saved reconstructed volume to {recon_path}")
        
        # Save reconstructed volume as video (Z-axis slices)
        print("Creating reconstruction video...")
        Z, Y, X = reconstructed_volume.shape
        
        # Normalize volume to 0-255
        vol_normalized = reconstructed_volume.copy()
        vol_normalized = vol_normalized - vol_normalized.min()
        if vol_normalized.max() > 0:
            vol_normalized = vol_normalized / vol_normalized.max()
        vol_normalized = (vol_normalized * 255).astype(np.uint8)
        
        # Create video of Z-slices
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = f"{file.video_name}_reconstruction_z_slices.mp4"
        out = cv2.VideoWriter(video_path, fourcc, 30, (X, Y), isColor=False)
        
        # Write each Z-slice as a frame
        for z in range(Z):
            frame = vol_normalized[z, :, :]
            out.write(frame)
        
        out.release()
        print(f"Saved reconstruction video: {video_path}")
        
        # Optional: Create multi-view video (XY, XZ, YZ slices side by side)
        canvas_width = X + Y
        canvas_height = max(Y, Z)
        video_path_multi = f"{file.video_name}_reconstruction_multiview.mp4"
        out_multi = cv2.VideoWriter(video_path_multi, fourcc, 30, (canvas_width, canvas_height), isColor=False)
        
        max_frames = max(Z, Y, X)
        for i in range(max_frames):
            canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
            
            # XY slice (varying Z)
            z_idx = min(i * Z // max_frames, Z - 1)
            xy_slice = vol_normalized[z_idx, :, :]
            canvas[:Y, :X] = xy_slice
            
            # YZ slice (varying X)
            x_idx = min(i * X // max_frames, X - 1)
            yz_slice = vol_normalized[:, :, x_idx]
            yz_resized = cv2.resize(yz_slice, (Y, canvas_height))
            canvas[:, X:X+Y] = yz_resized
            
            out_multi.write(canvas)
        
        out_multi.release()
        print(f"Saved multi-view reconstruction video: {video_path_multi}")


# vamtoolbox.util.export_projection.export_sinogram_to_images(
#     sinogram=sino,
#     output_dir=file.dir + "/projection_stack",
# )

# ------------------------------------------------------------
# Export helical projector images
# ------------------------------------------------------------

projector_output_dir = os.path.join(file.dir, "projection_stack_helical")

# Convert physical pitch (mm / rev) → pixels / frame
mm_per_pixel = 0.1                     # must match ImageConfig scale
pixels_per_rev = file.height / mm_per_pixel
helical_pitch_pixels = pixels_per_rev / sino.shape[0]



export_sinogram_to_images(
    sinogram=sino,
    output_dir=projector_output_dir,
    image_size=(3840, 2160),            # projector resolution
    bit_depth=8,
    normalization_percentile=99.9,
    rotate_angle=0.0,
    invert_u=False,
    invert_v=file.invert_v,
    helical_pitch_pixels=helical_pitch_pixels,
    start_v_offset=0,
)
