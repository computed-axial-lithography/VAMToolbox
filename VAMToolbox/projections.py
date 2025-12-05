import sys
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import trimesh
import trimesh.viewer

import vamtoolbox.geometry
import vamtoolbox.imagesequence
import vamtoolbox.optimize
import vamtoolbox.util.export_projection
from leapctype import *


RI = dict()
RI["LVUDMA"] = 1.51
RI["HVUDMA"] = 1.51
RI["PEGDA700"] = 1.5
RI["UW"] = 1.5
RI["GELMA"] = 1.34
RI["242N"] = 1.5

start_all = time.time()

projection_backend = "leap" # "leap" or "astra"

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

#  Thinker
files.append(
    FileSpecs(
        dir=r"/Users/wendylin/Desktop/Helical/models/thinker/",
        stl_name=r"thinker.stl",
        sino_name="thinker",
        rot_vel=20,
        height=10 / 2,
        intensity_scales=[1, 2],
        size_scale=2,
        resin="LVUDMA",
    )
)

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
       
       ### for mac ###
        # if target_geo.insert is None:
        #     proj_geo = vamtoolbox.geometry.ProjectionGeometry(
        #         angles=angles, ray_type="parallel", CUDA=True
        #     )
        # else:
        #     print("Disabled CUDA")
        #     proj_geo = vamtoolbox.geometry.ProjectionGeometry(
        #         angles=angles, ray_type="parallel", CUDA=False
        #     )

        if projection_backend == "leap":
            from vamtoolbox.projector import leap3D as projector
            print("Leap projector imported!")

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

            # Create LEAP geometry instance
            proj_geo = LEAPGeometry(
                angles=np.radians(angles),    # degrees → radians
                source_radius=100.0,          # TODO: tune to match your setup
                detector_distance=200.0,      # TODO: tune to match your setup
                helical_pitch=file.height     # axial distance per full rotation (example)
            )

            # Parameter dictionary for LEAP conversion
            params = {
                "geometry_type": "modular",          # using modular-beam helical
                "num_rows": 400,
                "num_cols": 512,
                "voxel_size": 0.1,
                "volume_shape": volume.shape,
                "pixel_width": 0.1,         # usually same as voxel_size
            }

            sino_array = projector.forward_project(volume, proj_geo, params)

            sino = sino_array



        elif projection_backend == "astra":
            from vamtoolbox.projector import projector3D as projector
            print("Astra projector imported!")

            proj_geo = vamtoolbox.geometry.ProjectionGeometry(
                angles=angles, ray_type="parallel", CUDA=False
            )

            sino = projector.forward_project(target_geo.volume, proj_geo)

        else:
            raise ValueError("Unsupported projection backend. Choose 'leap' or 'astra'.")



        recon = target_geo # reuse
        recon.show()
        sino.save(file.sino_name)

    print("Rebinning of original sinogram.")

    # Rebin for vial refraction correction
    print(sino.shape)
    # sino_rebinned = vamtoolbox.geometry.rebinFanBeam(sinogram=sino,vial_width=1030,N_screen=(1280,800),n_write=1.51,throw_ratio=2.62825) #2.62825
    # print(sino_rebinned.array.shape)

    # sino = sino_rebinned
    # vamtoolbox.Display.showVolumeSlicer(sino.sinogram,type="proj")
    # plt.show()

    # sino_rebinned.save(file.sino_name_rebinned)

    # sino.show()
    # sino_rebinned.show()

    for intensity_scale in file.intensity_scales:
        # Save imageseq and video file
        config = vamtoolbox.imagesequence.ImageConfig(
            (2560, 1600),
            intensity_scale=intensity_scale,
            size_scale=file.size_scale,
            array_num=file.array_num,
            array_offset=file.array_offset,
            invert_v=file.invert_v,
            v_offset=0,
            normalization_percentile=99.9,
        )
        imgset = vamtoolbox.imagesequence.ImageSeq(config, sinogram=sino)

        imgset.saveAsVideo(
            save_path=f"{file.video_name}_intensity{intensity_scale}xarray{file.array_num}final.mp4",
            rot_vel=36,
            num_loops=5,  # 30
            preview=True,
        )
        print("Saved video: %s" % (file.video_name + ".mp4"))
        time.sleep(2)

print("Total runtime: %.2f seconds" % (time.time() - start_all))

vamtoolbox.util.export_projection.export_sinogram_to_images(sinogram=sino, output_dir=file.dir + "/projection_stack")
print(f"Using backend: {projection_backend}")