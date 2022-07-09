import os
from turtle import Turtle
import pyvista
import numpy as np

import vamtoolbox.display
import vamtoolbox.geometry


# try:
# 	import pyvoxsurfa # pip install pyvoxsurf
# 	voxelize_type = 'pyvoxsurf'
# except:
# 	warnings.warn('Unable to import pyvoxsurf. Resorting to trimesh voxelization. This will be slower for high resolutions >100.')

class MeshBounds:
	def __init__(self,mesh,voxel_pitch=None,parent_shape=None):
		
		self.x_min, self.x_max = mesh.bounds[0][0], mesh.bounds[1][0]
		self.y_min, self.y_max = mesh.bounds[0][1], mesh.bounds[1][1]
		self.z_min, self.z_max = mesh.bounds[0][2], mesh.bounds[1][2]
		
		if voxel_pitch is not None:

			self.z_min_ind, self.z_max_ind = np.round(np.array([self.z_min,self.z_max]) / voxel_pitch).astype(int) + parent_shape[2]//2
			self.y_min_ind, self.y_max_ind = np.round(np.array([self.y_min,self.y_max]) / voxel_pitch).astype(int) + parent_shape[0]//2
			self.x_min_ind, self.x_max_ind = np.round(np.array([self.x_min,self.x_max]) / voxel_pitch).astype(int) + parent_shape[0]//2

			nX = np.round(abs(self.x_max - self.x_min)/voxel_pitch + 1).astype(int)
			nY = np.round(abs(self.y_max - self.y_min)/voxel_pitch + 1).astype(int)
			nZ = np.round(abs(self.z_max - self.z_min)/voxel_pitch + 1).astype(int)
			x = np.linspace(self.x_min,self.x_max,nX)
			y = np.linspace(self.y_min,self.y_max,nY)
			z = np.linspace(self.z_min,self.z_max,nZ)

			X,Y,Z = np.meshgrid(x,y,z)
			points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
			self.inds = np.round((np.array(points) + mesh.extents/2)/voxel_pitch).astype(int)
			# inds = trimesh.voxel.ops.points_to_indices(points,pitch=voxel_pitch,origin=mesh.extents/2)

			# self.z_min_ind -= 1
			# self.y_min_ind -= 1
			# self.x_min_ind -= 1

			# self.slice = np.s_[self.x_min_ind:self.x_max_ind,
			# 				self.y_min_ind:self.y_max_ind,
			# 				self.z_min_ind:self.z_max_ind]
			# self.slice = np.mgrid[self.x_min_ind:self.x_max_ind,
			# 				self.y_min_ind:self.y_max_ind,
			# 				self.z_min_ind:self.z_max_ind]
			

def voxelizeTarget(input_path, resolution, bodies='all', rot_angles=[0,0,0]): # when an input argument is set to a value in the fuction definition this is its default value
	"""
	Takes a mesh of surface points from the input .stl file, voxelizes the mesh,
	and places the array inside a square array (nx,ny,nz) where nx = ny

	Parameters
	----------
	input_path : str
		path to .stl file
	resolution : int
		number of layers in height to voxelize the target
	bodies : str or dict
		specifier of which bodies in the stl (if multibody) are to be printed, default is 'all' meaning that all mesh bodies will be included in the voxel array
	rot_angles : list, optional
		angles in degrees around (x,y,z) axes which to rotate the 
		target geometry

	Returns
	-------
	voxels : np.ndarray
		voxelized target

	Examples
	--------
	>>> voxelizeTarget(")
	"""

	if not os.path.exists(input_path):
		raise Exception("Input .stl file does not exist")

	if type(bodies) == int:
		bodies = [bodies]
	
	# check file type e.g. stl, obj, etc, and get mesh from corresponding reader
	# the parent mesh may have an arbitrary number of independent bodies
	reader = pyvista.get_reader(input_path)
	parent_mesh = reader.read()


	# rotate mesh if specified
	if np.any(rot_angles):
		parent_mesh = rotate(parent_mesh, rot_angles)

	# read mesh boundaries so that the maximum z dimension can be scaled according to input voxel resolution
	x_min, x_max, y_min, y_max, z_min, z_max = parent_mesh.bounds
	density = abs(z_max-z_min)/resolution
	# the diameter of the inscribed circle should be at least equal in length to the diagonal of the bounding box in x-y
	d = np.sqrt((x_max-x_min)**2 + (y_max-y_min)**2)
	r_max_1 = np.sqrt(x_max**2 + y_max**2)
	r_max_2 = np.sqrt(x_min**2 + y_min**2)
	r_max = r_max_1 if r_max_1 > r_max_2 else r_max_2
	r = np.arange(-r_max, r_max, density,dtype=np.float32) # coordinate array in radius

	# r = np.arange(-d/2, d/2, density,dtype=np.float32) # coordinate array in radius
	z = np.arange(z_min, z_max, density,dtype=np.float32) # coordinate array in z
	X, Y, Z = np.meshgrid(r, r, z) # coordinate grids to be used to query the mesh for inside/outside-ness 


	x_ind, y_ind, z_ind = np.meshgrid(np.arange(0,r.size,1,dtype=int),np.arange(0,r.size,1,dtype=int),np.arange(0,z.size,1,dtype=int))

	# For DEBUGGING: This should be the same order as selection.points
	# parent_voxels_point = np.squeeze(np.dstack((X.T.ravel(),Y.T.ravel(),Z.T.ravel())))

	parent_voxels_ind = np.squeeze(np.dstack((x_ind.T.ravel(),y_ind.T.ravel(),z_ind.T.ravel())))
	
	# Create unstructured grid from the structured grid
	grid = pyvista.StructuredGrid(X,Y,Z)
	del X, Y, Z # these are large grids when the voxel resolution is high
	ugrid = pyvista.UnstructuredGrid(grid) # this requires a lot of memory

	array_voxels = np.zeros(x_ind.shape,dtype=int) # empty array which will hold the voxelized mesh

	# when the mesh has multiple indpendent bodies, a body may be selected to be the "insert"
	# the insert is an object that is preexisting in the print container
	# if 'all' == True the function assumes all print bodies will be voxelized and considered the same connected voxel array
	if bodies != 'all':
		if 'insert' in bodies.keys():
			insert_voxels = np.zeros_like(array_voxels)
		else:
			insert_voxels = None

		if 'zero_dose' in bodies.keys():
			zero_dose_voxels = np.zeros_like(array_voxels)
		else:
			zero_dose_voxels = None

	else:
		insert_voxels = None
		zero_dose_voxels = None

	mesh_bodies = parent_mesh.split_bodies()

	for k in range(mesh_bodies.n_blocks):
	
		body = mesh_bodies.pop(0) # gets the next mesh body in the parent mesh body list
		child_mesh = body.extract_surface()

		# https://github.com/pyvista/pyvista-support/issues/141
		# get part of the mesh within the mesh's bounding surface
		selection = ugrid.select_enclosed_points(child_mesh.extract_surface(),
												tolerance=0.0,
												check_surface=False)
		mask = selection.point_data['SelectedPoints'].view(np.bool)
		ind = parent_voxels_ind[mask]
		
		if bodies != 'all':
			if k + 1 in bodies['print']:
				array_voxels[ind[:,0],ind[:,1],ind[:,2]] = 1
			if 'insert' in bodies.keys():
				if k + 1 in bodies['insert']:
					insert_voxels[ind[:,0],ind[:,1],ind[:,2]] = 1
			if 'zero_dose' in bodies.keys():
				if k + 1 in bodies['zero_dose']:
					zero_dose_voxels[ind[:,0],ind[:,1],ind[:,2]] = 1
		else:
			array_voxels[ind[:,0],ind[:,1],ind[:,2]] = 1
			# vamtoolbox.display.showVolumeSlicer(array_voxels,vol_type='target',slice_step=1)


	return array_voxels, insert_voxels, zero_dose_voxels






def rotate(mesh, rot_angles):
	"""
	Rotates mesh before voxelization

	Parameters
	----------
	mesh : pyvista mesh object
		input .stl mesh read by pyvista
	rot_angles : list
		angles in degrees around (x,y,z) axes which to rotate the target geometry
		
	Returns
	-------
	mesh : pyvista mesh object
		rotated mesh
	"""
	if rot_angles[0] != 0:
		mesh.rotate_x(rot_angles[0],point=mesh.center,inplace=True)

	elif rot_angles[1] != 0:
		mesh.rotate_y(rot_angles[1],point=mesh.center,inplace=True)

	elif rot_angles[2] != 0:
		mesh.rotate_z(rot_angles[2],point=mesh.center,inplace=True)

	return mesh












	# # load stl file with trimesh library 
	# parent_mesh = trimesh.load(input_path)


	# # rotate mesh if specified
	# if np.any(rot_angles):
	# 	parent_mesh = rotate_mesh(parent_mesh, rot_angles)


	# # Find the max and min coordinates of the mesh to form a bounding box
	# parent_mesh_bounds = MeshBounds(parent_mesh)

	# # calculate voxelization pitch as a function of the height in z-axis
	# voxel_pitch = abs(parent_mesh_bounds.z_max - parent_mesh_bounds.z_min)/(resolution-1)
	# nZ = np.round(resolution).astype(int)
	# nX = abs(parent_mesh_bounds.x_max - parent_mesh_bounds.x_min)/voxel_pitch
	# nY = abs(parent_mesh_bounds.y_max - parent_mesh_bounds.y_min)/voxel_pitch
	# nR = np.round(np.sqrt(nX**2 + nY**2)).astype(int)
	# if np.mod(nR,2) != 0:
	# 	nR = nR + 1

	# # try to split mesh into multiple bodies if they exist
	# child_meshes = parent_mesh.split()
	# num_meshes = len(child_meshes)

	# # create empty voxel grid for all voxelized meshes to be inserted into
	# parent_voxels = np.zeros((nR,nR,nZ))
	# for k, child_mesh in enumerate(child_meshes):
		
	# 	# get child mesh bounds for insertion into global voxel grid
	# 	child_mesh_bounds = MeshBounds(child_mesh,voxel_pitch=voxel_pitch,parent_shape=parent_voxels.shape)
		
	# 	# create voxel surface
	# 	# child_mesh_discretized = trimesh.voxel.creation.voxelize_ray(child_mesh,pitch=voxel_pitch,per_cell=[3,3])
	# 	child_mesh_discretized = trimesh.voxel.creation.voxelize_subdivide(child_mesh,pitch=abs(parent_mesh_bounds.z_max - parent_mesh_bounds.z_min)/(resolution-1),max_iter=100)

	# 	# fill voxel surface
	# 	child_voxels = trimesh.voxel.morphology.fill(child_mesh_discretized.encoding,method='holes')
	# 	child_voxels = np.array(child_voxels.dense).astype(int) * (k + 1)
	# 	vamtoolbox.display.showVolumeSlicer(vamtoolbox.geometry.Volume(child_voxels,vol_type="recon"),slice_step=1)

	# 	tmp_parent_voxels = parent_voxels.copy()
	# 	tmp_parent_voxels[child_mesh_bounds.inds[:,0],child_mesh_bounds.inds[:,1],child_mesh_bounds.inds[:,2]] = child_voxels
	# 	parent_voxels += tmp_parent_voxels













	# if voxelize_type == 'pyvoxsurf':
	# 	# voxelize mesh with number of slices in z equal to resolution input
	# 	voxels = pyvoxsurf.voxelize(mesh.vertices,mesh.faces,bounds,resolution,"Robust")

	# elif voxelize_type == 'trimesh':

	# 	# create voxel surface
	# 	# mesh_discretized = mesh.voxelized(voxel_pitch,method="ray")
	# 	mesh_discretized = trimesh.voxel.creation.voxelize_subdivide(mesh,pitch=voxel_pitch,max_iter=20)

	# 	# fill voxel surface
	# 	voxels_encoding = trimesh.voxel.morphology.fill(mesh_discretized.encoding,method='holes')
		
	# 	voxels = voxels_encoding.dense.astype('uint8')

	# pad target so that it has dimensions nR x nR x nZ 
	# voxels = pad_target_to_square(voxels)
	# TODO add cubic padding for complex projector geometries





def pad_target_to_square(input_voxel_array):
	"""
	Places input array inside a square array (nx,ny,nz) where nx = ny

	Parameters
	----------
	input_voxel_array : ndarray
		target voxel array

	Returns
	-------
	voxels : ndarray
		voxelized target
	"""
	nX, nY, nZ = input_voxel_array.shape

	# Largest dimension of projections is when the diagonal of the cubic target matrix is perpendicular to the projection angle 
	nR = np.round(np.sqrt(nX**2 + nY**2))
	if np.mod(nR,2) != 0:
		nR = nR + 1

	pad_x_before = (nR-nX)//2
	pad_y_before = (nR-nY)//2
	pad_x_after = pad_x_before
	pad_y_after = pad_y_before

	if 2*pad_x_before + nX != nR:
		pad_x_after = pad_x_after + 1
	if 2*pad_y_before + nY != nR:
		pad_y_after = pad_y_after + 1


	pad_x_before = int(pad_x_before)
	pad_y_before = int(pad_y_before)
	pad_x_after = int(pad_x_after)
	pad_y_after = int(pad_y_after)

	square_pad_voxels = np.pad(input_voxel_array, ((pad_x_before,pad_x_after),(pad_y_before,pad_y_after),(0,0)), 'constant')

	return square_pad_voxels

def rotate_mesh(mesh, rot_angles):
	"""
	Rotates mesh before voxelization

	Parameters
	----------
	mesh : Trimesh mesh object
		input .stl mesh read by Trimesh
	rot_angles : Nx3 array
		angles in degrees around (x,y,z) axes which to rotate the target geometry
		
	Returns
	-------
	mesh : Trimesh mesh object
		rotated mesh
	"""
	if rot_angles[0] != 0:
		rot_matrix_x = trimesh.transformations.rotation_matrix(np.radians(rot_angles[0]),(1,0,0))
	else:
		rot_matrix_x = np.identity(4)

	if rot_angles[1] != 0:
		rot_matrix_y = trimesh.transformations.rotation_matrix(np.radians(rot_angles[1]),(0,1,0))
	else:
		rot_matrix_y = np.identity(4)

	if rot_angles[2] != 0:
		rot_matrix_z = trimesh.transformations.rotation_matrix(np.radians(rot_angles[2]),(0,0,1))
	else:
		rot_matrix_z = np.identity(4)

	rot_matrix = np.dot(rot_matrix_x, np.dot(rot_matrix_y,rot_matrix_z))
	mesh.apply_transform(rot_matrix)


	return mesh

if __name__ == '__main__':
	# testing voxelization
	voxels = voxelizeTarget("STLs/CAL_Bear_reduced.stl",100,rot_angles=[90,0,0])
	print('Target size (nX,nY,nZ): ',voxels.shape[0],',',voxels.shape[1],',',voxels.shape[2])
	disp.view_vol(voxels,'Target','voxels')
	# disp.view_slices(voxels,2,'voxels')
	# np.save('CALBear400.npy',voxels.astype('bool'))