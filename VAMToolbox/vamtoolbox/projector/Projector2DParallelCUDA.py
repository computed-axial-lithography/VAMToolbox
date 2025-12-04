import astra  # type: ignore
import numpy as np

import vamtoolbox


class Projector2DParallelCUDAAstra:
    def __init__(self, target_geo, proj_geo):
        self.target_geo = target_geo
        self.proj_geo = proj_geo

        self.nT = target_geo.nX

        self.angles_rad = np.deg2rad(proj_geo.angles)

        self.proj_geom = astra.create_proj_geom("parallel", 1.0, self.nT, self.angles_rad)
        self.vol_geom = astra.create_vol_geom(target_geo.nY, target_geo.nX)
        self.proj_id = astra.create_projector("line", self.proj_geom, self.vol_geom)

    def forward(self, target):
        """Forward projector operation (b = Ax)"""
        x = vamtoolbox.util.data.clipToCircle(target)

        b_id, tmp_b = astra.create_sino(x, self.proj_id, gpuIndex=0)

        b = np.transpose(tmp_b)

        astra.data2d.delete(b_id)

        return b

    def backward(self, sinogram):
        """Backward projector operation (x = A^Tb)"""
        b = sinogram
        if self.proj_geo.zero_dose_sino is not None:
            b[self.proj_geo.zero_dose_sino] = 0.0

        tmp_b = np.transpose(b)

        x_id, x = astra.creators.create_backprojection(tmp_b, self.proj_id)

        astra.data2d.delete(x_id)

        return vamtoolbox.util.data.clipToCircle(x)
