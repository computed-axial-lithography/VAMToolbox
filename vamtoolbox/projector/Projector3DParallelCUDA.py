try:
    import astra
except:
    ImportError("Astra toolbox is either not installed or installed incorrectly.")

try:
    import tigre
except:
    ImportError("Tigre toolbox is either not installed or installed incorrectly.")

import numpy as np
import vamtoolbox

class Projector3DParallelCUDAAstra():
    def __init__(self,target_geo,proj_geo):
        self.target_geo = target_geo
        self.proj_geo = proj_geo
        self.nT = target_geo.nX
        self.angles_rad = np.deg2rad(proj_geo.angles)
        
        if self.proj_geo.absorption_coeff is not None:
            self.proj_geo.absorption_mask = np.transpose(self.proj_geo.absorption_mask)

        self.vol_geom = astra.create_vol_geom(target_geo.nX, target_geo.nY, target_geo.nZ)

        if proj_geo.inclination_angle is None or proj_geo.inclination_angle == 0:
            self.proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, target_geo.nZ , self.nT, self.angles_rad)
        
        else:
            self.angles_vector = vamtoolbox.projector.genVectorsAstra.genVectorsAstra(proj_geo.angles,proj_geo.inclination_angle)
            self.proj_geom = astra.create_proj_geom('parallel3d_vec', target_geo.nZ , self.nT, self.angles_vector)

    def forward(self,target):
        """Forward projector operation (b = Ax)"""
        x = np.transpose(vamtoolbox.util.data.clipToCircle(target))
        if self.proj_geo.absorption_coeff is not None:
            x = self.proj_geo.absorption_mask*x

        b_id, tmp_b = astra.create_sino3d_gpu(x, self.proj_geom, self.vol_geom)

        b = np.transpose(tmp_b,(2,1,0))

        astra.data3d.delete(b_id)

        return b

    def backward(self,sinogram):
        """Backward projector operation (x = A^Tb)"""
        b = sinogram
        if self.proj_geo.zero_dose_sino is not None:
            b[self.proj_geo.zero_dose_sino] = 0.0

        tmp_b = np.transpose(b,(2,1,0))

        x_id, x = astra.creators.create_backprojection3d_gpu(tmp_b, self.proj_geom, self.vol_geom)   

        if self.proj_geo.absorption_coeff is not None:
            x = self.proj_geo.absorption_mask*x
        x = np.transpose(x)
        
        astra.data3d.delete(x_id)

        return vamtoolbox.util.data.clipToCircle(x)


class Projector3DParallelCUDATigre():
    def __init__(self,target_geo,proj_geo,optical_params=None):
        
        self.angles_rad = np.deg2rad(proj_geo.angles)

        try:
            self.attenuation = np.swapaxes(proj_geo.attenuation.astype(np.float32),2,0)
            self.attenuation = np.ascontiguousarray(self.attenuation)
        except:
            self.attenuation = None

        # setup fixed coordinate grid for backprojection and dimensions of projections  
        self.radius = target_geo.nY//2
        self.y, self.x = np.mgrid[:target_geo.nY, :target_geo.nY] - self.radius
        self.center = target_geo.nY//2
        self.proj_t = np.arange(target_geo.nY) - target_geo.nY//2

        self.geo = tigre.geometry(mode='parallel', nVoxel=np.array([target_geo.nZ,target_geo.nY,target_geo.nX]))
        self.geo.dDetector = np.array([1, 1])               # size of each pixel            (mm)
        self.geo.sDetector = self.geo.dDetector * self.geo.nDetector
        self.geo.accuracy = 1
        self.geo.vialRadius = 1
        self.geo.maxIntensity = 1


        
    def forward(self,target):

        x = vamtoolbox.util.data.clipToCircle(target.astype(np.float32))
        x = np.swapaxes(x,2,0)
        x = np.ascontiguousarray(x)

        b = tigre.Ax(x,self.geo,self.angles_rad,projection_type='interpolated',img_att=self.attenuation)
        b = np.transpose(b,(2,0,1))


        return b

    def backward(self,projections):
        b = projections.astype(np.float32)

        b = np.ascontiguousarray(np.transpose(b,(1,2,0)))
        if self.attenuation is not None:
            tmp_attenuation = np.ascontiguousarray(np.swapaxes(self.attenuation,1,2))
            x = tigre.Atb(projections,self.geo,self.angles_rad,img_att=tmp_attenuation)
        else:
            x = tigre.Atb(projections,self.geo,self.angles_rad)

        # print(tmp_attenuation.shape)



        x = np.swapaxes(x,0,2)

        return vamtoolbox.util.data.clipToCircle(x)