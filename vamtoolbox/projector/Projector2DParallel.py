try:
    import astra
    astra_available = True
except:
    ImportError("Astra toolbox is either not installed or installed incorrectly.")
    astra_available = False
import numpy as np
from skimage.transform._warps import warp
from functools import partial
import matplotlib.pyplot as plt
import vamtoolbox

class Projector2DParallelAstra():
    def __init__(self,target_geo,proj_geo):
        
        self.target_geo = target_geo
        self.proj_geo = proj_geo
        
        self.nT = target_geo.nX
        
        self.angles_rad = np.deg2rad(proj_geo.angles)

        if astra_available is True:
            self.proj_geom = astra.create_proj_geom('parallel', 1.0, self.nT, self.angles_rad)
            self.vol_geom = astra.create_vol_geom(target_geo.nY, target_geo.nX)
            self.proj_id = astra.create_projector('line', self.proj_geom, self.vol_geom)
        

    def forward(self,target):
        """Foward projector operation (b = Ax)"""
        x = vamtoolbox.util.data.clipToCircle(target)

        _, tmp_b = astra.create_sino(x, self.proj_id)

        b = np.transpose(tmp_b)

        return b

    def backward(self,sinogram):
        """Backward projector operation (x = A^Tb)"""
        b = sinogram
        if self.proj_geo.zero_dose_sino is not None:
            b[self.proj_geo.zero_dose_sino] = 0.0

        tmp_b = np.transpose(b)
        
        _, x = astra.creators.create_backprojection(tmp_b,self.proj_id)   



        return vamtoolbox.util.data.clipToCircle(x)

class Projector2DParallelPython():
    def __init__(self, target_geo, proj_geo):
        self.target_geo = target_geo
        self.proj_geo = proj_geo
        if proj_geo.occlusion is not None:
            self.occlusion = proj_geo.occlusion
        else:
            self.occlusion = None
        self.angles = proj_geo.angles

        # setup fixed coordinate grid for backprojection and dimensions of sinogram  
        self.radius = target_geo.nY//2
        self.y, self.x = np.mgrid[:target_geo.nY, :target_geo.nY] - self.radius
        self.center = target_geo.nY//2
        self.proj_t = np.arange(target_geo.nY) - target_geo.nY//2

        # Occlusion sinogram is computed in init because it will remain the same for a 
        # given instance of the class; all forward/backprojections with the instance 
        # assume that the occlusion does not change
        
        if self.occlusion is not None:
            self.occ_sinogram = self.generateOccSinogram()


    def generateOccSinogram(self):
        """
        Create sinogram containing minimum values of 's' within the occlusion map

        Returns
        ---------------
        occ_sinogram : nd_array
        Npixels x Nangles

        """

        occ_sinogram = np.zeros((self.target_geo.nY,self.angles.shape[0]))
        for i, angle in enumerate(np.deg2rad(self.angles)):
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            R = np.array([[cos_a, sin_a, -self.center * (cos_a + sin_a - 1)],
                        [-sin_a, cos_a, -self.center * (cos_a - sin_a - 1)],
                        [0, 0, 1]])

            
            t = self.x * cos_a - self.y * sin_a
            s = self.x * sin_a + self.y * cos_a

            t_bins = np.sort(self.proj_t)
            t_discrete_inds = np.digitize(t,bins=t_bins) -1

            t_discrete = self.proj_t[t_discrete_inds]

            s_occ = s[self.occlusion]
            t_occ = t_discrete[self.occlusion]
            t_inds = np.unique(t_discrete_inds[self.occlusion])
            t_s = np.stack((t_occ,s_occ),axis=-1)
            t_s = t_s[t_s[:, 0].argsort()]
           
            # if i == 30:
            #     fig, ax = plt.subplots(1,1)
            #     ax.imshow(t)
            #     fig, ax = plt.subplots(1,1)
            #     ax.imshow(t_discrete)
            #     plt.show()
            s_grouped_by_t_discrete = np.split(t_s[:,1], np.unique(t_s[:,0], return_index=True)[1][1:])
            s_min_occ = [np.min(sub_t) for sub_t in s_grouped_by_t_discrete]
            s_min = np.full(self.proj_t.shape,fill_value=np.nan)
            s_min[t_inds] = s_min_occ
            occ_sinogram[:, i] = s_min

            # if i == self.angles.shape[0] - 1:
            #     fig, ax = plt.subplots(1,1)
            #     ax.imshow(occ_sinogram)
            #     plt.show()


            # warp doesn't work because of smoothing near edges of occlusion
            # instead create t and s grids like in backprojection
            # to build occ_sinogram, 
            # 1. mask the s grid with the occlusion

            # 2. bin the s values into bins that correspond to the t axis array (number of pixels in proj_t)
            # 3. find min of each bin
            # 4. store s_min vs. t as occ_sinogram for current theta




            ################# OLD WAY #################
            # rotated_occlusion = warp(self.occlusion, R, clip=True,preserve_range=True)
            # s_occ = np.where(rotated_occlusion>0,self.y,np.NaN)

            # # disp.view_plot(s_occ,'S')
            
            # occ_sinogram[:, i] = np.nanmin(s_occ,axis=0)

        return occ_sinogram

    def forward(self,target):
        """
        Computes forward Radon transform of the target space object accounting for
        reduced projection contribution due to occlusion shadowing

        Inputs
        ---------------
        target : nd_array
        Npixels x Npixels array that contains the target space object

        Returns
        ---------------
        projection : nd_array
        Npixels x Nangles array of forward Radon transform with occlusion shadowing

        """
        x = vamtoolbox.util.data.clipToCircle(target)
        b = np.zeros((self.target_geo.nY,self.angles.shape[0]))

        for i, angle in enumerate(np.deg2rad(self.angles)):
        
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            R = np.array([[cos_a, sin_a, -self.center * (cos_a + sin_a - 1)],
                        [-sin_a, cos_a, -self.center * (cos_a - sin_a - 1)],
                        [0, 0, 1]])

            rotated = warp(x, R, clip=True,preserve_range=True)
            
            if self.occlusion is not None:
                curr_occ = self.occ_sinogram[:,i]

                occ_shadow = self.y > curr_occ[np.newaxis,:]

                rotated = np.multiply(rotated, np.logical_not(occ_shadow))

            b[:, i] = rotated.sum(0)

        return b
    
    def backward(self,b,clipping=True):
        """
        Computes inverse Radon transform of projection accounting for reduced dose
        deposition due to occlusion shadowing

        Inputs
        ---------------
        projection : nd_array
        Npixels x Nangles array that contains the projection space sinogram of the target

        Returns
        ---------------
        projection : nd_array
        Npixels x Npixels array of inverse Radon transform with occlusion shadowing

        """

        x = np.zeros(self.target_geo.target.shape)
        if self.proj_geo.zero_dose_sino is not None:
            b[self.proj_geo.zero_dose_sino] = 0.0

        for i, (curr_proj, angle) in enumerate(zip(b.T, np.deg2rad(self.angles))):
            
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            t = self.x * cos_a - self.y * sin_a
            s = self.x * sin_a + self.y * cos_a

            interpolant = partial(np.interp, xp=self.proj_t, fp=curr_proj, left=0, right=0)
            curr_backproj = interpolant(t)

            if self.occlusion is not None:
                curr_occ = self.getOccShadow(i,angle,t,s)
                x += np.multiply(curr_backproj,np.logical_not(curr_occ))

            else:
                x += curr_backproj

            # plt.imshow(np.multiply(curr_backproj,np.logical_not(curr_occ)),cmap='CMRmap')
            # plt.show()

            if clipping:
                x = vamtoolbox.util.data.clipToCircle(x)

        return x

    def getOccShadow(self,i,angle,t,s):

        curr_occ = self.occ_sinogram[:,i]
        interpolant = partial(np.interp, xp=self.proj_t, fp=curr_occ, left=np.NaN, right=np.NaN)
        
        # return s > np.floor(interpolant(t))
        
        t_bins = np.sort(self.proj_t)
        t_discrete_inds = np.digitize(t,bins=t_bins) - 1

        t_discrete = self.proj_t[t_discrete_inds]
        return s > interpolant(t_discrete)

    def calcVisibility(self):
        tmp = np.zeros((self.target_geo.nY,self.target_geo.nX,self.angles.shape[0]))
        vis = np.zeros(self.target_geo.target.shape)
        projection = np.ones((self.target_geo.nY,self.angles.shape[0]))

        for i, (curr_proj, angle) in enumerate(zip(projection.T, np.deg2rad(self.angles))):
            
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            t = self.x * cos_a - self.y * sin_a
            s = self.x * sin_a + self.y * cos_a

            interpolant = partial(np.interp, xp=self.proj_t, fp=curr_proj*self.angles[i], left=0, right=0)
            curr_backproj = interpolant(t)

            curr_occ = self.getOccShadow(i,angle,t,s)


            tmp[..., i] = np.multiply(curr_backproj,np.logical_not(curr_occ))

        for k in range(self.target_geo.nY):
            for j in range(self.target_geo.nX):
                q = np.unique(tmp[k,j,:]%(self.angles.shape[0]//2))
                
                vis[k,j] = q.shape[0]

        vis = np.multiply(vis,self.target_geo.target)
        vis = vis/(self.angles.shape[0]//2)
        vis = np.where(vis >= 1, 1, vis)

        return vamtoolbox.util.data.clipToCircle(vis)