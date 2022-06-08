from ctypes import ArgumentError
import numpy as np
import dill
import cv2
from scipy import ndimage
from PIL import Image
import os
import matplotlib.pyplot as plt
import imageio

import vamtoolbox

class ImageConfig:
    def __init__(self,image_dims,**kwargs):
        """
        Parameters
        ----------
        image_dims : tuple
            (image width, image height)
        
        rotate_angle : float, optional
            rotation angle in degrees clockwise

        u_offset : int, optional
            offset in pixels from center in U-axis (+) right (-) left

        v_offset : int, optional
            offset in pixels from center in V-axis (+) up (-) down

        invert_u : bool, optional
            invert U-axis

        invert_v : bool, optional
            invert V-axis

        array_num : int, optional
            number of multiplied sinograms 

        array_offset : int, optional
            pitch of arrayed sinograms in pixels

        intensity_scale : float, optional
            intensity scale factor

        size_scale : float, optional
            size scale factor
        """
        self.N_u, self.N_v = image_dims

        if self.N_u % 4 != 0 or self.N_v % 4 != 0:
            raise Exception("Image dimensions should be divisible by 4.")

        self.rotate_angle = 0.0 if 'rotate_angle' not in kwargs else kwargs['rotate_angle']
        self.u_offset = 0 if 'u_offset' not in kwargs else kwargs['u_offset']
        self.v_offset = 0 if 'v_offset' not in kwargs else kwargs['v_offset']

        self.invert_u = False if 'invert_u' not in kwargs else kwargs['invert_u']
        self.invert_v = False if 'invert_v' not in kwargs else kwargs['invert_v']

        self.array_num = 1 if 'array_num' not in kwargs else kwargs['array_num']
        self.array_offset = 0 if 'array_offset' not in kwargs else kwargs['array_offset']

        self.intensity_scale = 1.0 if 'intensity_scale' not in kwargs else kwargs['intensity_scale']
        self.size_scale = 1.0 if 'size_scale' not in kwargs else kwargs['size_scale']



class ImageSeq:
    def __init__(self,image_config,sinogram):
        """
        Parameters
        ----------
        image_config : imagesequence.ImageConfig object
            configuration object that contains the options how the sinogram should be placed in the image(s) for projection
            
        sinogram : geometry.Sinogram object
            sinogram object to be converted to image set
        """
        if isinstance(sinogram,np.ndarray):
            pass
        elif isinstance(sinogram,vamtoolbox.geometry.Sinogram):
            sinogram = sinogram.array
        else:
            raise ArgumentError("sinogram not specified.")

        self.file_extension = ".imgseq"
        self.image_config = image_config
        
        
        mod_sinogram = np.copy(sinogram)


        if self.image_config.invert_u == True:
            mod_sinogram = _invertU(mod_sinogram)

        if self.image_config.invert_v == True:
            mod_sinogram = _invertV(mod_sinogram)

        if self.image_config.rotate_angle != 0.0:
            mod_sinogram = _rotate(mod_sinogram,self.image_config.rotate_angle)

        if self.image_config.size_scale != 1.0:
            mod_sinogram = _scaleSize(mod_sinogram,self.image_config.size_scale)
        
        max_intensity = np.max(mod_sinogram)
        if  max_intensity > 0 and max_intensity <= 1:
            mod_sinogram = mod_sinogram*255

        if self.image_config.intensity_scale != 1.0:
            mod_sinogram = _scaleIntensity(mod_sinogram,self.image_config.intensity_scale)

        mod_sinogram = _cropToBounds(mod_sinogram)

        mod_sinogram = mod_sinogram.astype(np.uint8)


        images = list()
        N_angles = mod_sinogram.shape[1]

        for j in range(N_angles):

            image_out = np.zeros((self.image_config.N_v,self.image_config.N_u),dtype=np.uint8)
            
            if self.image_config.array_num != 1:
                image = _arrayInsertImage(mod_sinogram[:,j,:].T,image_out,self.image_config)
                images.append(image)

            else:
                image = _insertImage(mod_sinogram[:,j,:].T,image_out,self.image_config)
                images.append(image)

        self.images = images


    def __getstate__(self):
        return self.__dict__

    def __setstate__(self,d):
        self.__dict__ = d

    def save(self,name:str):
        """save imagesequence.ImageSeq as 'name.imgseq'"""
        file = open(name + self.file_extension,'wb')
        dill.dump(self,file)
        file.close()

    def preview(self,):
        """Preview an animated image sequence"""
        vamtoolbox.dlp.players.preview(self)

    #TODO make show method with scrollable images like geometry.show()
    # def show(self,savepath=None,dpi='figure',**kwargs):
    #     """
    #     Parameters
    #     ----------
    #     savepath : str, optional

    #     dpi : int, optional
    #         image dots per inch from `matplotlib.pyplot.savefig <https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.savefig.html>`_

    #     **kwargs
    #         accepts `matplotlib.pyplot.imshow <https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.imshow.html>`_ keyword arguments
    #     """
    #     kwargs['cmap'] = 'CMRmap' if 'cmap' not in kwargs else kwargs['cmap']
    #     kwargs['interpolation'] = 'antialiased' if 'interpolation' not in kwargs else kwargs['interpolation']

    #     # must keep instance of slicer for mouse wheel scrolling to work
    #     self.viewer = vamtoolbox.display.VolumeSlicer(self.array,self.vol_type,**kwargs)
        
    #     if savepath is not None:
    #         plt.savefig(savepath,dpi=dpi)

    #     plt.show()

    def saveAsVideo(self,save_path: str,rot_vel: float,duration: float):

        if self.images is None:
            raise Exception("Problem encountered creating images in ImageSeq initialization")

        # codec = cv2.VideoWriter_fourcc("I","4","2","0")
        # codec = cv2.VideoWriter_fourcc('X','2','6','4')



        N_images_per_rot = len(self.images)
        fps = N_images_per_rot/(360/rot_vel)
        video_out = imageio.get_writer(save_path,fps=fps)
        # video_out = cv2.VideoWriter(save_path, codec, fps, (int(self.image_config.N_u), int(self.image_config.N_v)))


        N_total_images = np.round(N_images_per_rot/360*rot_vel*duration)

        k = 0
        while k < N_total_images:
                
            for image in self.images:
                # video_out.write(image)
                video_out.append_data(image)
                k += 1
                if k % 5 == 0:
                    print("Writing video frame %4.0d/%4.0d..."%(k,N_total_images))
            
        # video_out.release()
        video_out.close()

    def saveAsImages(self,save_dir: str,image_prefix: str ="image",image_type: str =".png"):
        
        for k, image in enumerate(self.images):
            save_path = os.path.join(save_dir, image_prefix+"%s"%str(k).zfill(4)+image_type)
            im = Image.fromarray(image)
            im.save(save_path,subsampling=0, quality=100)
            print("Saving image %s/%s"%(str(k).zfill(4),str(len(self.images)-1).zfill(4)))


def loadImageSeq(file_name:str):
    file = open(file_name,'rb')
    data_pickle = file.read()
    file.close()
    A = dill.loads(data_pickle)

    return A


def _insertImage(image,image_out,image_config,**kwargs):

    v_offset = image_config.v_offset if 'v_offset' not in kwargs else kwargs['v_offset']

    N_u = image_config.N_u
    N_v = image_config.N_v

    S_u = image.shape[1]
    S_v = image.shape[0]
  
    u1, u2 = int(N_u/2 + image_config.u_offset - S_u/2), int(N_u/2 + image_config.u_offset + S_u/2)
    v1, v2 = int(N_v/2 - v_offset - S_v/2), int( N_v/2 - v_offset + S_v/2)

    if u1 < 0 or u2 > image_config.N_u:
        raise Exception("Image could not be inserted because it is either too large in the u-dimension or the offset causes it to extend out of the input screen size")
    if v1 < 0 or v2 > image_config.N_v:
        raise Exception("Image could not be inserted because it is either too large in the v-dimension or the offset causes it to extend out of the input screen size")
        
    image_out[v1:v2,u1:u2] = image


    return image_out

def _arrayInsertImage(image,image_out,image_config):  

    for k in range(image_config.array_num):
        if image_config.array_num % 2 == 0:
            # if array number is even distribute evenly over image height centered at midheight
            a = k + 1
            array_dir = int(np.ceil(a/2)*(-1)**(k)) # sequence 1, -1, 2, -2, 3, -3,...
            tmp_v_offset = int(image_config.array_offset*array_dir/2 + image_config.v_offset)
        else:
            # if array number is odd distribute around the middle element of the array of images
            array_dir = int(0 + np.ceil(k/2)*(-1)**(k+1)) # sequence 0, 1, -1, 2, -2, 3, -3,...
            tmp_v_offset = int(image_config.array_offset*array_dir + image_config.v_offset)
    
        image_out = _insertImage(image,image_out,image_config,v_offset=tmp_v_offset)

    return image_out

def _invertU(sinogram):

    mod_sinogram = np.flip(sinogram,axis=0)
    return mod_sinogram

def _invertV(sinogram):
    mod_sinogram = np.flip(sinogram,axis=2)
    return mod_sinogram

def _scaleSize(sinogram,scale_factor):
    new_height = int(sinogram.shape[0]*scale_factor)
    new_width = int(sinogram.shape[2]*scale_factor)
    mod_sinogram = np.zeros((new_height,sinogram.shape[1],new_width))

    for i in range(sinogram.shape[1]):
        mod_sinogram[:,i,:] = cv2.resize(sinogram[:,i,:],(new_width,new_height),interpolation=cv2.INTER_LINEAR)

    return mod_sinogram

def _scaleIntensity(sinogram,intensity_scalar):
    mod_sinogram = np.minimum(sinogram*intensity_scalar,255)
    
    return mod_sinogram
    

def _rotate(sinogram,angle_deg):
    mod_sinogram = ndimage.rotate(sinogram,angle_deg,axes=(0,2),reshape=True,order=1)

    return mod_sinogram


def _cropToBounds(sinogram):
    collapsed_sinogram = np.squeeze(np.sum(sinogram,1))

    collapsed_u_sinogram = np.sum(collapsed_sinogram,1)
    collapsed_v_sinogram = np.sum(collapsed_sinogram,0)

    indices_u = np.argwhere(collapsed_u_sinogram != 0.0)
    indices_v = np.argwhere(collapsed_v_sinogram != 0.0)
    first_u, last_u = int(indices_u[0]), int(indices_u[-1])
    first_v, last_v = int(indices_v[0]), int(indices_v[-1])

    # if first_u == 0 or last_u == sinogram.shape[0]:
    #     mod_sinogram = sinogram
    # elif first_v == 0 or last_v == sinogram.shape[2]:
    #     mod_sinogram = sinogram
    # else:
    #     mod_sinogram = sinogram[first_v:last_v,:,first_u:last_u]
    mod_sinogram = sinogram[first_u:last_u,:,first_v:last_v]
    
    return mod_sinogram


# def array(view_box,center,mod_proj_shape,array_num,array_shift):
#     j = 0
#     for i in range(array_num):
#         if i == 0:
#             k = 0                
#         elif (i+1)%2 == 0:
#             k = 1
#             j+=1
#         else:
#             k = -1

#         T = center[1] - mod_proj_shape[2]//2 - k*j*array_shift
#         L = center[0] - mod_proj_shape[0]//2

#         view_box.append(((L,T),(mod_proj_shape[2],mod_proj_shape[0])))
#     return view_box
