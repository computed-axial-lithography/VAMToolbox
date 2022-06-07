try:
    import astra
except:
    ImportError("Astra toolbox is either not installed or installed incorrectly.")
import numpy as np
from skimage.util import dtype


class astra3Dinclined():
    def __init__(self,target,params):
        self.pT = target.shape[0]
        self.nX, self.nY, self.nZ = target.shape
        self.angles_rad = np.deg2rad(params['angles'])
        inclination_angle = np.deg2rad(params['inclination_angle'])
        self.cuda_available = astra.astra.use_cuda()
        
        # From samples/python/s005_3d_geometry.py
        # create_proj_geom('parallel3d_vec', det_row_count, det_col_count, V):

        # :param det_row_count: Number of detector pixel rows.
        # :type det_row_count: int
        # :param det_col_count: Number of detector pixel columns.
        # :type det_col_count: int
        # :param V: Vector array.
        # :type V: numpy.ndarray
        # :returns: A parallel projection geometry.

        assert self.cuda_available

        
        # We generate the same geometry as the circular one above. 
        self.angles_vector = np.zeros((len(self.angles_rad), 12))
        for i in range(len(self.angles_rad)):
            # ray direction
            self.angles_vector[i,0] = np.sin(self.angles_rad[i])
            self.angles_vector[i,1] = -np.cos(self.angles_rad[i])
            self.angles_vector[i,2] = inclination_angle

            # center of detector
            self.angles_vector[i,3:6] = 0

            # vector from detector pixel (0,0) to (0,1)
            self.angles_vector[i,6] = np.cos(self.angles_rad[i])
            self.angles_vector[i,7] = np.sin(self.angles_rad[i])
            self.angles_vector[i,8] = 0

            # vector from detector pixel (0,0) to (1,0)
            self.angles_vector[i,9] = np.sin(self.angles_rad[i])*np.cos(inclination_angle)
            self.angles_vector[i,10] = -np.cos(self.angles_rad[i])*np.cos(inclination_angle)
            self.angles_vector[i,11] = np.cos(inclination_angle)

        # if gpu is available setup 3D projector
        self.vol_geom = astra.create_vol_geom(self.nY, self.nX, self.nZ)
        # inclined_nZ = int(np.ceil(self.nZ/np.cos(inclination_angle)))
        self.proj_geo = astra.create_proj_geom('parallel3d_vec', self.nZ, self.pT, self.angles_vector)


    def forwardProject(self,target):
        assert self.cuda_available
        
        target = np.transpose(target)

        _, projections = astra.create_sino3d_gpu(target, self.proj_geo, self.vol_geom)
        projections = np.transpose(projections,(2, 1, 0))

        return projections

    def backProject(self,projections):
        assert self.cuda_available
        
        projections = np.transpose(projections,(2, 1, 0))
        rec_id, reconstruction = astra.creators.create_backprojection3d_gpu(projections,self.proj_geo,self.vol_geom)
        reconstruction = np.transpose(reconstruction)
        astra.data3d.delete(rec_id)

        return reconstruction       