import sys
import os
import time
import numpy as np
from leapctype import *
import vamtoolbox.geometry
try:
    import tkinter as tk
    from tkinter import filedialog
    _TK_AVAILABLE = True
except Exception:
    _TK_AVAILABLE = False
leapct = tomographicModels()
# Make sure you add: .../LEAP/src to your python path

'''
This script is nearly identical to d01_standard_geometries.py except it demonstrates LEAP's
helical cone-beam functionality.  LEAP has an implementation of helical FBP for the GPU only
Of course, just like all geometries in LEAP, one can use any iterative reconstruction.
For details of the helical FBP algorithm, please see the LEAP technical manual here:
https://github.com/LLNL/LEAP/blob/main/documentation/LEAP.pdf
'''


# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numTurns = 10
numAngles = 2*2*int(360*numCols/1024)*numTurns
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = numCols//4

# Set the scanner geometry
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0*numTurns), 1100, 1400)
#leapct.set_coneparallel(numAngles, numRows, numCols, pixelSize, pixelSize*1100.0/1400.0, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0*numTurns), 1100, 1400)
#leapct.set_curvedDetector()

# Set the helical pitch.
leapct.set_normalizedHelicalPitch(0.5)
#leapct.set_normalizedHelicalPitch(1.0)

# Set the volume parameters
leapct.set_default_volume()
#leapct.set_volume(numCols, numCols, numRows)

# Trouble-Shooting Functions
leapct.print_parameters()
#leapct.sketch_system()

# Set the backprojector model, 'SF' (the default setting), is more accurate, but 'VD' is faster
#leapct.set_projector('VD')

# Allocate space for the projections and the volume
g = leapct.allocateProjections()
f = leapct.allocateVolume()

# Load STL file and convert to volume using vamtoolbox (same method as projections.py)
# Try a GUI file chooser when available; otherwise use env var or default.
stl_path = None
if _TK_AVAILABLE:
    try:
        root = tk.Tk()
        root.withdraw()
        stl_path = filedialog.askopenfilename(title='Select STL file', filetypes=[('STL files','*.stl'),('All files','*.*')])
        root.destroy()
    except Exception:
        stl_path = None

if not stl_path:
    stl_path = os.environ.get('LEAP_STL_PATH', r'/Users/wendylin/Desktop/Helical/models/thinker/thinker.stl')
    if not stl_path:
        raise FileNotFoundError('No STL file selected and LEAP_STL_PATH is not set.')

# Calculate resolution based on volume voxel size (matching projections.py approach)
mm_per_voxel = leapct.get_voxelWidth()  # Get LEAP voxel size in mm
target_height_mm = leapct.get_numZ() * mm_per_voxel  # Total volume height
res = leapct.get_numZ()  # Use LEAP's Z resolution directly

print(f'Loading STL: {stl_path}')
print(f'Resolution: {res}, voxel size: {mm_per_voxel} mm')

# Load STL using vamtoolbox (same method as projections.py)
target_geo = vamtoolbox.geometry.TargetGeometry(stlfilename=stl_path, resolution=res)

# Copy voxelized volume into LEAP array
# VAMToolbox TargetGeometry stores the voxel data in `array` (Volume.array).
if hasattr(target_geo, 'array') and target_geo.array is not None:
    stl_volume = target_geo.array
    print(f'STL volume shape: {stl_volume.shape}, LEAP volume shape: {f.shape}')

    # If shapes don't match, try permutations of axes to find a matching orientation
    if stl_volume.shape != f.shape:
        from itertools import permutations

        matched = False
        for perm in permutations((0, 1, 2)):
            perm_vol = np.transpose(stl_volume, perm)
            if perm_vol.shape == f.shape:
                stl_volume = perm_vol
                matched = True
                print(f'Applied axis permutation {perm} to match LEAP volume ordering')
                break

        if not matched:
            # No exact match; proceed with center-crop/pad using the best overlapping orientation (no transpose)
            print('No exact axis-permutation match found; proceeding with centered crop/pad')

    # Center the STL volume in LEAP volume (crop or pad) to match dimensions
    if stl_volume.shape == f.shape:
        f[:, :, :] = stl_volume
    else:
        min_x = min(stl_volume.shape[0], f.shape[0])
        min_y = min(stl_volume.shape[1], f.shape[1])
        min_z = min(stl_volume.shape[2], f.shape[2])

        x_offset_stl = (stl_volume.shape[0] - min_x) // 2
        y_offset_stl = (stl_volume.shape[1] - min_y) // 2
        z_offset_stl = (stl_volume.shape[2] - min_z) // 2

        x_offset_leap = (f.shape[0] - min_x) // 2
        y_offset_leap = (f.shape[1] - min_y) // 2
        z_offset_leap = (f.shape[2] - min_z) // 2

        f[x_offset_leap:x_offset_leap+min_x,
          y_offset_leap:y_offset_leap+min_y,
          z_offset_leap:z_offset_leap+min_z] = stl_volume[
              x_offset_stl:x_offset_stl+min_x,
              y_offset_stl:y_offset_stl+min_y,
              z_offset_stl:z_offset_stl+min_z]

    print('STL volume loaded successfully')
else:
    print('Warning: Failed to load STL (no `array`), using FORBILD phantom as fallback')
    leapct.set_FORBILD(f, True)

#leapct.display(f)

print("== Finished voxelization, starting forward projection ==")

# "Simulate" projection data
startTime = time.time()
leapct.project(g,f)
print('Forward Projection Elapsed Time: ' + str(time.time()-startTime))
#leapct.display(g)

# Add noise to the data (just for demonstration purposes)
I_0 = 50000.0
#g[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)

# Reset the volume array to zero, otherwise iterative reconstruction algorithm will start their iterations
# with the true result which is cheating
f[:] = 0.0

# Reconstruct the data
startTime = time.time()
#leapct.backproject(g,f)
leapct.FBP(g,f)
#leapct.print_cost = True
filters = filterSequence(1.0e0)
filters.append(TV(leapct, delta=0.02/20.0))
#leapct.ASDPOCS(g,f,10,10,1,filters)
#leapct.SART(g,f,10,10)
#leapct.OSEM(g,f,10,10)
#leapct.LS(g,f,50,'SQS')
#leapct.RWLS(g,f,100,filters,None,'SQS')
#leapct.RDLS(g,f,50,filters,1.0,True,1)
#leapct.MLTR(g,f,10,10,filters)
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))


# Post Reconstruction Smoothing (optional)
#startTime = time.time()
#leapct.diffuse(f,0.02/20.0,4)
#leapct.MedianFilter(f)
#leapct.BlurFilter(f,2.0)
#print('Post-Processing Elapsed Time: ' + str(time.time()-startTime))

# Display the result with napari
leapct.display(f)
