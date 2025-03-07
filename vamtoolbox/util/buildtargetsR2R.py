import numpy as np
import skimage.draw
import skimage.morphology 
import scipy.ndimage, scipy.fft, scipy.signal
import matplotlib.pyplot as plt

def hexlattice(shape,rho_tiling,diameter,anisotropy_factor=1,offset=(0,0),gradient=1.0):
    offset = np.array([[offset[0],offset[1]]])
    rhol_shape = (shape[0],shape[1])
    x = diameter
    a = np.zeros(rhol_shape)

    anisotropy_multiplier = np.array([[1,1/anisotropy_factor]])
    rho_spacing = rhol_shape[0]/rho_tiling
    l_spacing = rho_spacing*np.cos(np.pi/6)

    spacing = np.array([rho_spacing,l_spacing])*anisotropy_multiplier[0,:]
    tiling = np.array([rho_tiling,int(np.ceil(rhol_shape[1]/spacing[1]))])
    
    # gradient_multiplier = np.stack((np.ones(l_tiling), np.linspace(1,gradient,num=l_tiling)),axis=1)
    gradient_multiplier = np.linspace(1,gradient,num=tiling[1])
    _points = np.array([[-x/2,-x/4],[-x/2,x/4],[0,x/2],[x/2,x/4],[x/2,-x/4],[0,-x/2]])*anisotropy_multiplier
    for i in range(tiling[0]+1):
        for j in range(tiling[1]):
            if j % 2 == 1:
                patching_offset = np.array([[((-i-np.sin(np.pi/6))*spacing[0]),-j*spacing[1]]])
            else:
                patching_offset = np.array([[-i*spacing[0],-j*spacing[1]]])
            
            # patching_offset[0,:] = patching_offset[0,:]*gradient_multiplier[j]
            # points = np.array([[-x/2,0],[-x/4,x/2],[x/4,x/2],[x/2,0],[x/4,-x/2],[-x/4,-x/2]]) - offset

            points = (_points*gradient_multiplier[j] - patching_offset) - offset
            
            mask = skimage.draw.polygon2mask(rhol_shape,points)

            # fig, axs = plt.subplots(1,1,figsize=(10,1))
            # im_recon = axs.imshow(mask)
            # im_recon.set_clim(0,1)
            # axs.set_xlabel(r'$L$ [cm]')
            # axs.set_ylabel(r'$\rho$ [cm]')
            # vam.util.matplotlib.addColorbar(im_recon)
            # plt.show()
            a += mask
    
    a = np.logical_not(a).astype('float')

    return np.broadcast_to(a[:,:,None],shape)

def simpleCirclePattern(shape,rho_tiling,diameter,anisotropy_factor=1,offset=(0,0),gradient=1.0):
    rhol_shape = (shape[0],shape[1])
    a = np.zeros(rhol_shape)
    radius = diameter/2

    anisotropy_multiplier = np.array([1,1/anisotropy_factor])
    rho_spacing = rhol_shape[0]/rho_tiling
    l_spacing = rho_spacing

    spacing = np.array([rho_spacing,l_spacing])*anisotropy_multiplier
    tiling = np.array([rho_tiling,int(np.ceil(rhol_shape[1]/spacing[1]))])
    
    center = np.array([0,0])
    gradient_multiplier = np.linspace(1,gradient,num=tiling[1]+1)
    for i in range(tiling[0]+1):
        for j in range(tiling[1]+1):
            if i % 2 == 0:
                patching_offset = np.array([-i*spacing[0],-j*spacing[1]-radius*anisotropy_multiplier[1]])
                _center = center - patching_offset - offset
            else:
                patching_offset = np.array([-i*spacing[0],-j*spacing[1]])
                _center = center - patching_offset - offset

            _r_center = _center[0]
            _c_center = _center[1]
            
            
            _r_radius = radius*gradient_multiplier[j] 
            _c_radius = radius*gradient_multiplier[j]*anisotropy_multiplier[1]
            coord = skimage.draw.ellipse(_r_center,_c_center,_r_radius,_c_radius,shape=rhol_shape)
            a[coord] = 1

    a = np.logical_not(a).astype('float')


    fig, axs = plt.subplots(1,1,figsize=(10,1))
    im_recon = axs.imshow(a)
    im_recon.set_clim(0,1)
    axs.set_xlabel(r'$L$ [cm]')
    axs.set_ylabel(r'$\rho$ [cm]')
    # vam.util.matplotlib.addColorbar(im_recon)
    plt.show()
    

    return np.broadcast_to(a[:,:,None],shape)


def pill(shape,f_rho,f_l,diameter,length,pitch_l=0.5,num=11):
    rhol_shape = (shape[0],shape[1])
    a = np.zeros(rhol_shape)
    radius = diameter/2
    
    _x_st_el = np.linspace(-radius,radius,int(f_rho*diameter))
    _y_st_el = np.linspace(-radius,radius,int(f_l*diameter))
    _z_st_el = np.linspace(-radius,radius,int(f_rho*diameter))

    X_st_el,Y_st_el,Z_st_el = np.meshgrid(_x_st_el,_y_st_el,_z_st_el,indexing='ij')

    R_st_el = np.sqrt(X_st_el**2+Y_st_el**2+Z_st_el**2)

    ellipsoid_struct_el = np.zeros_like(X_st_el)
    ellipsoid_struct_el[R_st_el<=radius] = 1

    # fig, axs = plt.subplots(1,1)
    # im_recon = axs.imshow(ellipsoid_struct_el[:,:,ellipsoid_struct_el.shape[2]//2])
    # # im_recon.set_clim(0,1)
    # axs.set_xlabel(r'$L$ [cm]')
    # axs.set_ylabel(r'$\rho$ [cm]')
    # # vam.util.matplotlib.addColorbar(im_recon)
    # plt.show()

    x_dim = shape[0]/f_rho
    y_dim = shape[1]/f_l
    z_dim = shape[2]/f_rho

    _x = np.linspace(0,x_dim,shape[0])
    _y = np.linspace(0,y_dim,shape[1])
    _z = np.linspace(-z_dim/2,z_dim/2,shape[2])

    X,Y,Z = np.meshgrid(_x,_y,_z,indexing='ij')

    centers = np.linspace(-pitch_l*np.floor(num/2),pitch_l*np.floor(num/2),num)
    center_indices = (centers*f_l + shape[1]//2).astype(int)
    half_length_voxels = int(length*f_l/2)
    start_indices = center_indices-half_length_voxels
    end_indices = center_indices+half_length_voxels

    line = np.zeros_like(X)
    for start, end in zip(start_indices,end_indices):

        line[shape[0]//2,start:end,shape[2]//2] = 1

    pill = scipy.signal.fftconvolve(line,ellipsoid_struct_el,mode='same') >= 1

    # fig, axs = plt.subplots(1,1,figsize=(10,1))
    # im_recon = axs.imshow(pill[:,:,shape[2]//2])
    # # im_recon.set_clim(0,1)
    # axs.set_xlabel(r'$L$ [cm]')
    # axs.set_ylabel(r'$\rho$ [cm]')
    # # vam.util.matplotlib.addColorbar(im_recon)
    # plt.show()
    

    return pill.astype('float')

def pointsPattern(shape,rho_tiling,diameter,connecting_beam_thickness,anisotropy_factor=1,offset=(0,0),gradient=1.0):
    rhol_shape = (shape[0],shape[1])
    a = np.zeros(rhol_shape)
    _output = np.zeros(shape)
    radius = diameter/2

    anisotropy_multiplier = np.array([1,1/anisotropy_factor])
    rho_spacing = rhol_shape[0]/rho_tiling
    l_spacing = rho_spacing

    spacing = np.array([rho_spacing,l_spacing])*anisotropy_multiplier
    tiling = np.array([rho_tiling,int(np.ceil(rhol_shape[1]/spacing[1]))])
    
    center = np.array([0,0])
    gradient_multiplier = np.linspace(1,gradient,num=tiling[1]+1)
    
    i = 1
    for j in range(tiling[1]+1):
        if i % 2 == 0:
            patching_offset = np.array([-i*spacing[0],-j*spacing[1]-radius*anisotropy_multiplier[1]])
            _center = center - patching_offset - offset
        else:
            patching_offset = np.array([-i*spacing[0],-j*spacing[1]])
            _center = center - patching_offset - offset

        _r_center = _center[0]
        _c_center = _center[1]
        
        
        _r_radius = radius*gradient_multiplier[j] 
        _c_radius = radius*gradient_multiplier[j]*anisotropy_multiplier[1]
        coord_r, coord_c = skimage.draw.ellipse(_r_center,_c_center,_r_radius,_c_radius,shape=rhol_shape)
        _output[coord_r,coord_c,:] = 1

        left_edge = int(np.floor(np.max((0,_c_center-_c_radius))))
        right_edge = int(np.ceil(np.min((_c_center+_c_radius,shape[1]))))
        if j==0:
            top_edge_connecting_beam = int(np.ceil(_r_center+_r_radius))
            bottom_edge_connecting_beam = int(np.ceil(_r_center-_r_radius))
        _output[0:top_edge_connecting_beam,left_edge:right_edge,0:int(connecting_beam_thickness)] = 1
        _output[0:top_edge_connecting_beam,left_edge:right_edge,-int(connecting_beam_thickness):-1] = 1
        

    _output[bottom_edge_connecting_beam:top_edge_connecting_beam,:,0:int(connecting_beam_thickness)] = 1
    _output[bottom_edge_connecting_beam:top_edge_connecting_beam,:,-int(connecting_beam_thickness):-1] = 1

    
    # fig, axs = plt.subplots(1,1,figsize=(10,1))
    # im_recon = axs.imshow(a)
    # im_recon.set_clim(0,1)
    # axs.set_xlabel(r'$L$ [cm]')
    # axs.set_ylabel(r'$\rho$ [cm]')
    # # vam.util.matplotlib.addColorbar(im_recon)
    # plt.show()
    

    # return np.broadcast_to(a[:,:,None],shape)
    return _output

def simpleLattice(shape,tiling,width,spatial_sampling_rate,anisotropy_factor=1,offset=(0,0)):
    offset = np.array([[offset[0],offset[1]]])
    rhol_shape = (shape[0],shape[1])
    a = np.zeros(rhol_shape)
    half_width = (int(width*spatial_sampling_rate[0]/2), int(width*spatial_sampling_rate[1]/2))
    if tiling[0] >= 1:
        rho_spacing = int(rhol_shape[0]/tiling[0])
        for i in range(tiling[0]):
            a[i*rho_spacing-half_width[0]:i*rho_spacing+half_width[0],:] = 1

    if tiling[1] >= 1:
        l_spacing = int(rhol_shape[1]/tiling[1])
        for j in range(tiling[1]+1):
            a[:,j*l_spacing-half_width[1]:j*l_spacing+half_width[1]] = 1

    return np.broadcast_to(a[:,:,None],shape)
    
'''
def hexlattice(rhol_shape,tiling,diameter,offset):
    offset = np.array([[offset[0],offset[1]]])
    
    x = diameter
    a = np.zeros(rhol_shape)
    for i in range(tiling[0]+1):
        for j in range(tiling[1]+1):
            if j % 2 == 1:
                patching_offset = np.array([[x/2,0]])
            else:
                patching_offset = np.array([[0,0]])
            tile_offset = np.array([[-i/tiling[0]*rhol_shape[0],-j/tiling[1]*rhol_shape[1]]])
            
            # points = np.array([[-x/2,0],[-x/4,x/2],[x/4,x/2],[x/2,0],[x/4,-x/2],[-x/4,-x/2]]) - offset

            points = np.array([[-x/2,-x/4],[-x/2,x/4],[0,x/2],[x/2,x/4],[x/2,-x/4],[0,-x/2]]) - offset - tile_offset - patching_offset

            mask = skimage.draw.polygon2mask((rhol_shape),points)
            a += mask

    # plt.show()
    fig, axs = plt.subplots(1,1,figsize=(10,1))
    # extent = [0,target_geo.nL_total/target_geo.transform.spatial_sampling_rate_l,target_geo.transform.r_i,target_geo.transform.r_o]
    im_recon = axs.imshow(a)
    im_recon.set_clim(0,1)
    axs.set_xlabel(r'$L$ [cm]')
    axs.set_ylabel(r'$\rho$ [cm]')
    vam.util.matplotlib.addColorbar(im_recon)
    plt.show()
    # fig.savefig(rf'{test_string}_target.png',dpi=600)
    return a
'''

def triangleLattice(shape, rho_tiling,height,anisotropy_factor=1,gradient=1.0):

    rhol_shape = (shape[0],shape[1])
    a = np.zeros(rhol_shape)
    

    anisotropy_multiplier = np.array([1,1/anisotropy_factor])
    rho_spacing = rhol_shape[0]/rho_tiling
    l_spacing = rho_spacing*np.cos(np.pi/3)

    spacing = np.array([rho_spacing,l_spacing])*anisotropy_multiplier
    tiling = np.array([rho_tiling,int(np.ceil(rhol_shape[1]/spacing[1]))])
    
    gradient_multiplier = np.linspace(1,gradient,num=tiling[1]+1)
    _points = np.array([[-height/2, 0], [height/2, -height/2*np.cos(np.pi/6)], [height/2,height/2*np.cos(np.pi/6)]])*anisotropy_multiplier

    for i in range(tiling[0]+1):
        for j in range(tiling[1]+1):

            patching_offset = np.array([[-i*spacing[0],-j*spacing[1]]])

            # Define the coordinates of the triangle vertices
            if (i + j) % 2 == 0:
                # Upward triangle
                vertices = _points
            else:
                # Downward triangle
                vertices = _points*np.array([[-1,1],[-1,1],[-1,1]])
            
            points = vertices*gradient_multiplier[j] - patching_offset
            
            # Draw the triangle using polygon2mask
            mask = skimage.draw.polygon2mask(rhol_shape, points)

            # fig, axs = plt.subplots(1,1,figsize=(10,1))
            # im_recon = axs.imshow(a)
            # # im_recon.set_clim(0,1)
            # axs.set_xlabel(r'$L$ [cm]')
            # axs.set_ylabel(r'$\rho$ [cm]')
            # plt.show()
            a += mask
    
    a = np.logical_not(a).astype('float')

    return np.broadcast_to(a[:,:,None],shape)

def gratings(shape,ori,freq):
    rhol_shape = (shape[0],shape[1])
    a = np.zeros(rhol_shape)
    rho_v = np.linspace(0,1,rhol_shape[0])
    l_v = np.linspace(0,rhol_shape[1]/rhol_shape[0],rhol_shape[1])
    
    ori *= np.pi/180
    rho, l = np.meshgrid(rho_v,l_v,indexing='ij')
    rho_r   =  np.cos(ori)*rho + np.sin(ori)*l
    l_r   = -np.sin(ori)*rho + np.cos(ori)*l

    a = (np.sin(2*np.pi*freq*rho_r) + 1)/2
        
    return np.broadcast_to(a[:,:,None],shape)

def tpms(shape,f_rho,f_l,unit_cell_size=0.2,t=[-1],anisotropy_factor=1,type='schwarzp'):
    
    x_dim = shape[0]/f_rho
    y_dim = shape[1]/f_l
    z_dim = shape[2]/f_rho

    _x = np.linspace(0,x_dim,shape[0])
    _y = np.linspace(0,y_dim,shape[1])
    _z = np.linspace(-z_dim/2,z_dim/2,shape[2])
    X,Y,Z = np.meshgrid(_x,_y,_z,indexing='ij')

    
    
    # dx = np.linspace(0,xyz[0],shape[0])
    # dy = np.linspace(0,xyz[1],shape[1])
    # dz = np.linspace(0,xyz[2],shape[2])
    # X,Y,Z = np.meshgrid(dx,dy,dz,indexing='ij')
    l = np.array([1,1/anisotropy_factor,1])
    # omega = 2*np.pi/l
    unit_cell_pitch = 1/unit_cell_size
    k = 2*np.pi*unit_cell_pitch

    Cos_x = np.cos(k*X)
    Cos_y = np.cos(k*Y)
    Cos_z = np.cos(k*Z)
    Cos_2x = np.cos(2*k*X)
    Cos_2y = np.cos(2*k*Y)
    Cos_2z = np.cos(2*k*Z)
    Sin_x = np.sin(k*X)
    Sin_y = np.sin(k*Y)
    Sin_z = np.sin(k*Z)
    
    if len(t) == 2:
        T = np.linspace(t[0],t[1],shape[1])
        T = np.broadcast_to(T[None,:],(shape[0],shape[1]))
        T = np.broadcast_to(T[:,:,None],(shape[0],shape[1],shape[2]))
    else:
        T = np.ones_like(X)*t
    
    if type=='schwarzp':
        # implicit = np.cos(omega[0]*X) + np.cos(omega[1]*Y) + np.cos(omega[2]*Z)
        implicit = Cos_x + Cos_y + Cos_z - t
        a = np.where(np.abs(implicit-c)<=thickness,1,0)
    
    elif type=='gyroid-sheet':
        # implicit = np.cos(omega[0]*X)*np.sin(omega[1]*Y) + np.cos(omega[1]*Y)*np.sin(omega[2]*Z) + np.cos(omega[2]*Z)*np.sin(omega[0]*X)
        # a = np.where(np.abs(implicit-c)<=thickness,0,1)
        implicit = (Cos_x*Sin_y + Cos_y*Sin_z + Cos_z*Sin_x)**2 - T**2
        a = np.where(implicit<=0,1,0)
    
    elif type=='gyroid-skel':
        # implicit = np.cos(omega[0]*X)*np.sin(omega[1]*Y) + np.cos(omega[1]*Y)*np.sin(omega[2]*Z) + np.cos(omega[2]*Z)*np.sin(omega[0]*X)
        # a = np.where(np.abs(implicit-c)<=thickness,0,1)
        implicit = Cos_x*Sin_y + Cos_y*Sin_z + Cos_z*Sin_x - T
        a = np.where(implicit<=0,1,0)

    elif type=='schwarzp-skel':
        # implicit = (1-Y/np.amax(Y))*(np.cos(X) + np.cos(Y) + np.cos(Z)) - (np.cos(X)*np.cos(Y) + np.cos(Y)*np.cos(Z) + np.cos(Z)*np.cos(X))
        # implicit[implicit<1.5] = 0
        # implicit[implicit>=1.5] = 1

        implicit = Cos_x + Cos_y + Cos_z - T
        a = np.where(np.abs(implicit)<=0.4,1,0)
        # a = np.where(implicit<=T,0,0)

    elif type == 'iwp-skel':
        implicit = 2*(Cos_x*Cos_y + Cos_y*Cos_z + Cos_z*Cos_x) - (Cos_2x + Cos_2y + Cos_2z) + T
        a = np.where(implicit<=0,1,0) 

    elif type == 'diamond-skel':
        implicit = (Cos_x*Cos_y*Cos_z) - (Sin_x*Sin_y*Sin_z) + T
        a = np.where(implicit<=0,1,0) 
    elif type == 'diamond-sheet':
        implicit = ((Cos_x*Cos_y*Cos_z) - (Sin_x*Sin_y*Sin_z))**2 - T**2
        a = np.where(implicit<=0,1,0) 

    elif type=='1':
        implicit = (1.8-Y/np.amax(Y))*((np.sin(X)*np.cos(Y)) + (np.sin(Y)*np.cos(Z)) + (np.sin(Z)*np.cos(X)))
        implicit[implicit<1] = 0
        implicit[implicit>=1] = 1

    elif type=='Gyr-skel':
        implicit = (np.cos(X)*np.sin(Y) + np.cos(Y)*np.sin(Z) + np.cos(Z)*np.sin(X)) - (np.cos(2*X)*np.cos(2*Y) + np.cos(2*Y)*np.cos(2*Z) + np.cos(2*Z)*np.cos(2*X))

    return a


def pattern(array,pattern_axis,number):
    # patterned_array = np.repeat(array,repeats=number,axis=pattern_axis)
    if pattern_axis == 0:
        patterned_array = np.tile(array,reps=(number,1,1))
    elif pattern_axis == 1:
        patterned_array = np.tile(array,reps=(1,number,1))
    elif pattern_axis == 2:
        patterned_array = np.tile(array,reps=(1,1,number))
    return patterned_array


# def microfluidicssine(shape,f_rho,f_l,period=0.5,anisotropy_factor=1):
    
#     x_dim = shape[0]/f_rho
#     y_dim = shape[1]/f_l
#     z_dim = shape[2]/f_rho

#     _x = np.linspace(0,x_dim,shape[0])
#     _y = np.linspace(0,y_dim,shape[1])
#     _z = np.linspace(-z_dim/2,z_dim/2,shape[2])
#     X,Y,Z = np.meshgrid(_x,_y,_z,indexing='ij')

#     output = np.ones_like(X)
    

    
#     # dx = np.linspace(0,xyz[0],shape[0])
#     # dy = np.linspace(0,xyz[1],shape[1])
#     # dz = np.linspace(0,xyz[2],shape[2])
#     # X,Y,Z = np.meshgrid(dx,dy,dz,indexing='ij')
#     l = np.array([1,1/anisotropy_factor,1])
#     # omega = 2*np.pi/l
#     unit_cell_pitch = 1/unit_cell_size
#     k = 2*np.pi*unit_cell_pitch