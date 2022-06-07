try:
    import astra
except:
    ImportError("Astra toolbox is either not installed or installed incorrectly.")
import numpy as np
from scipy import interpolate

class astra3Dabsorption():
    def __init__(self,target,params):
        self.pT = target.shape[0]
        self.nX, self.nY, self.nZ = target.shape
        self.angles_rad = np.deg2rad(params['angles'])
        self.voxel_size = params['voxel_size']

        self.y, self.x = np.mgrid[:target.shape[0], :target.shape[0]] - target.shape[0]//2
        self.y_mm, self.x_mm = self.y*self.voxel_size, self.x*self.voxel_size

        self.abs_coeff = params['absorption']
        self.container_radius = params['radius']
        self.I0 = params['max_intensity']/100 # converting from W/cm^2 to W/mm^2
        if 'print_time' in params and 'rot_vel' in params:
            self.print_time = params['print_time']
            self.rot_vel = params['rot_vel']
            self.s_per_rot = 360/self.rot_vel
            self.n_rot = np.floor(self.print_time/self.s_per_rot)
            self.q_eff_Navag_planck_freq = params['quantum_eff']*1/(6.022e23*6.626e-34*7.358e14)
        else:
            self.print_time = 0
            self.rot_vel = 0

        

        self.cuda_available = astra.astra.use_cuda()
        
        if self.cuda_available:
            # if gpu is available setup 3D projector
            self.vol_geom = astra.create_vol_geom(self.nY, self.nX, self.nZ)
            self.proj_geo = astra.create_proj_geom('parallel3d', 1.0, 1.0, self.nZ , self.pT, self.angles_rad)
        else:
            # if not setup a 2D projector for slice-by-slice operation
            self.proj_geo = astra.create_proj_geom('parallel', 1.0, self.pT, self.angles_rad)
            self.vol_geom = astra.create_vol_geom(self.nY, self.nX)
            self.proj_id = astra.create_projector('line', self.proj_geo, self.vol_geom)

    def forwardProject(self,target):
        if self.cuda_available:
            target = np.transpose(target)

            _, projections = astra.create_sino3d_gpu(target, self.proj_geo, self.vol_geom)
            projections = np.transpose(projections,(2, 1, 0))

        else:
            for z_i in range(self.nZ):
                _, tmp_proj = astra.create_sino(target[:,:,z_i], self.proj_id)
                projections[:,:,z_i] = np.transpose(tmp_proj)

        return projections

    def backProject(self,projections):
        if self.cuda_available:
            projections = np.transpose(projections,(2, 1, 0))
            reconstruction = np.zeros((self.nY,self.nX,self.nZ))


            partial_reconstruction_time = self.print_time - self.n_rot*self.s_per_rot
            interpolant = interpolate.interp1d(np.linspace(0,self.s_per_rot,len(self.angles_rad)),self.angles_rad,kind='nearest')
            partial_reconstruction_angle = interpolant(partial_reconstruction_time)

            angle_iter = np.nditer(self.angles_rad,flags=['f_index'])
            for angle in angle_iter:
                proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, self.nZ , self.pT, np.array([angle]))

                _, recon_view = astra.creators.create_backprojection3d_gpu(projections[:,[angle_iter.index],:],proj_geom,self.vol_geom)
                recon_view = np.transpose(recon_view)

                # import matplotlib.pyplot as plt
                
                # fig, axs = plt.subplots(1,2)
                # axs[0].imshow(recon_view[:,:,10])
                

                exp_abs_view = self.getExpabs(angle)
                Iabs_view = self.I0*self.abs_coeff*np.multiply(recon_view/np.max(projections),exp_abs_view[:,:,np.newaxis])  # 

                
                # Rinit_view = Iabs_view*self.q_eff_Navag_planck_freq



                reconstruction += Iabs_view

                if angle == partial_reconstruction_angle:
                    partial_reconstruction = reconstruction
                # axs[1].imshow(recon_view[:,:,10])
                # plt.show()

            
            reconstruction = reconstruction*self.s_per_rot/len(self.angles_rad) + partial_reconstruction*self.s_per_rot/len(self.angles_rad)*partial_reconstruction_time/self.s_per_rot



            

        else:
            for z_i in range(self.nZ):
                tmp_proj = np.transpose(projections[:,:,z_i])
                _, reconstruction[:, :, z_i] = astra.creators.create_backprojection(tmp_proj,self.proj_id)

        return self.truncateCircle(reconstruction)



    def getExpabs(self,angle_rad):
        t = self.y_mm * np.cos(-angle_rad) - self.x_mm * np.sin(-angle_rad)
        t_perp = self.y_mm * np.sin(-angle_rad) + self.x_mm * np.cos(-angle_rad)

        w = np.real(np.sqrt(self.container_radius**2 - t**2)) - t_perp  # propagation distance inside container
        w[np.absolute(t) > self.container_radius] =  np.inf  # set all values outside container radius to infinity
        relative_intensity = np.exp(-w*self.abs_coeff);   # absorption coefficient in units 1/distance
        # import matplotlib.pyplot as plt
        # plt.imshow(relative_intensity)
        # plt.show()

        return relative_intensity
    
    def truncateCircle(self,target):
        
        container_circle = (self.x_mm ** 2 + self.y_mm ** 2) > (self.container_radius) ** 2

        out_reconstruction_circle = (self.x ** 2 + self.y ** 2) > (target.shape[0]//2) ** 2
        target[np.logical_and(out_reconstruction_circle,container_circle),:] = 0.

        return target

if __name__ == '__main__':


    def createCubeCage(target_bounds,strut_size,side_length,voxel_size):
        # target = np.ones((int(target_bounds/voxel_size),int(target_bounds/voxel_size),int(target_bounds/voxel_size)))
        target = np.ones((int(target_bounds/voxel_size),int(target_bounds/voxel_size),20))

        y, x, z = np.mgrid[:target.shape[0],:target.shape[1],:target.shape[2]] 
        y, x, z = y-target.shape[0]//2, x-target.shape[1]//2, z-target.shape[2]//2,
        y, x, z = y*voxel_size, x*voxel_size, z*voxel_size

        x1 = np.where(np.abs(x)<side_length/2-strut_size,1,0)
        y1 = np.where(np.abs(y)<side_length/2-strut_size,1,0)
        z1 = np.where(np.abs(z)<side_length/2-strut_size,1,0)
        xy = np.logical_and(x1,y1)
        xz = np.logical_and(x1,z1)
        yz = np.logical_and(y1,z1)
        a1 = np.where(np.abs(x)>=side_length/2,1,0)
        a2 = np.where(np.abs(y)>=side_length/2,1,0)
        a3 = np.where(np.abs(z)>=side_length/2,1,0)
        a4 = np.logical_or(np.logical_or(a1,a2),a3)

        target[np.where(xy)] = 0
        target[np.where(xz)] = 0
        target[np.where(yz)] = 0

        target[np.where(a4)] = 0        


        return target




    voxel_size = 0.015
    target_bounds = 4
 
    strut_size = 0.4
    side_length = 2

    
    target = createCubeCage(target_bounds,strut_size,side_length,voxel_size)
    # import matplotlib.pyplot as plt
    
    # fig, ax = plt.subplots()
    # ax.imshow(target[:,:,target.shape[2]//2])
    # plt.show()

    nangles = 180
    params = {
        'angles': np.linspace(0,360-360/nangles,nangles),
        'absorption': 250/1000,  # [1/mm]
        'radius': 3, # [mm]
        'max_intensity': 25, # [W/cm^2]
        'quantum_eff': 0.7,
        'print_time': 60, # [s]
        'rot_vel': 24, # [deg/s]
        'voxel_size': voxel_size, # [mm]
    }


    A = astra3Dabsorption(target,params)

    proj = A.forwardProject(target)

    recon = A.backProject(proj)

    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    ax.imshow(recon[:,:,10])
    plt.show()
