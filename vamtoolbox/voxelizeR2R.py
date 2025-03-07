
import os
import time

import numpy as np
import pyglet
import trimesh
from PIL import Image
import tqdm
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo

EPSILON = 0.0001

def orthoMatrix(left, right, bottom, top, zNear, zFar, dtype):
    '''
    Return the following matrix
    |       2                               -(right+left)   |
    |  ----------       0           0       -------------   |
    |  right-left                             right-left    |
    |                                                       |
    |                   2                   -(top+bottom)   |
    |      0       ----------       0       -------------   |
    |              top-bottom                 top-bottom    |
    |                                                       |
    |                              -2       -(zFar+zNear)   |
    |      0            0      ----------   -------------   |
    |                          zFar-zNear     zFar-zNear    |
    |                                                       |
    |                                                       |
    |      0            0           0             1         |
    '''
    M = np.identity(4, dtype=dtype)
    M[0,0] = 2 / (right - left)
    M[1,1] = 2 / (top - bottom)
    M[2,2] = -2 / (zFar - zNear)
    M[0,3] = -(right + left) / (right - left)
    M[1,3] = -(top + bottom) / (top - bottom)
    M[2,3] = -(zFar + zNear) / (zFar - zNear)
    return M.T

def translationMatrix(direction, dtype):
    """Return matrix to translate by direction vector.

    If direction is [x, y, z], return the following matrix
    
    |   1   0   0   x   |
    |                   |
    |   0   1   0   y   |
    |                   |
    |   0   0   1   z   |
    |                   |
    |   0   0   0   1   |

    """
    M = np.identity(4, dtype=dtype)
    M[:3, 3] = direction[:3]
    return M.T

class Bounds():
    xmin = np.nan
    xmax = np.nan
    ymin = np.nan
    ymax = np.nan
    zmin = np.nan
    zmax = np.nan
    length_x = np.nan
    length_y = np.nan
    length_z = np.nan  

class BodyMesh():
    def __init__(self,mesh:trimesh.Trimesh):
        self.mesh = mesh
      
        self.bounds = Bounds()
        self.bounds.xmin = self.mesh.bounds[0][0]
        self.bounds.xmax = self.mesh.bounds[1][0]
        self.bounds.ymin = self.mesh.bounds[0][1]
        self.bounds.ymax = self.mesh.bounds[1][1]
        self.bounds.zmin = self.mesh.bounds[0][2]
        self.bounds.zmax = self.mesh.bounds[1][2]


        self.bounds.length_x = np.abs(self.bounds.xmax - self.bounds.xmin)
        self.bounds.length_y = np.abs(self.bounds.ymax - self.bounds.ymin)
        self.bounds.length_z = np.abs(self.bounds.zmax - self.bounds.zmin)

        self.num_of_verts = self.mesh.triangles.shape[0] * 3

class Voxelizer():
    def __init__(self,):
        self.meshes = {}
        self.voxel_arrays = {}
        self.global_bounds = Bounds()

    def addMeshes(self,stl_struct:dict):
        """
        Add mesh files to be voxelized. After adding meshes, the global bounding box of all meshes is calculated and used for subsequent voxelization such that all voxel arrays have the same physical bounds.

        If a mesh is added in a later call of addMeshes(), the global bounding box is updated. If the new mesh has a larger bounding box, previously voxelized meshes should be re-voxelized.

        Parameters
        ----------
        stl_struct : dict
            dict with file name, body name pairs for each mesh file. The body name is only an identifier for the user and accepts arbitrary strings.

            Example: {'mymesh.stl': 'print_body'}
        """
        for filename, body_name in stl_struct.items():
            mesh = trimesh.load_mesh(filename)
            self.meshes[body_name] = BodyMesh(mesh)
            self.voxel_arrays[body_name] = None

        # after adding all meshes, update the global max and min bounds
        self._updateBounds()

    def voxelize(self,body_name:str,xy_voxel_size:float,voxel_value:float,voxel_dtype:str='uint8',z_voxel_size=None,square_xy:bool=True,store_voxel_array:bool=False,slice_save_path:str=None):
        """
        Parameters
        ----------
        body_name : str
            name of the mesh to be voxelized

        xy_voxel_size : float
            side length of square voxel cross section in x-y plane in voxelized array in same units as the mesh file

        voxel_value : float
            value of voxels in voxelized array

            Only homogeneous voxel values are implemented
		
		voxel_dtype : str, optional
            datatype of voxel array
            
        z_voxel_size : float, optional
            thickness of each slice in voxelized array in same units as the mesh file, defaults to z_voxel_size = xy_voxel_size. This allows for non-isotropic voxel size in the z-axis

		square_xy : bool, optional
            if the resulting voxel array should have equal number of voxels in x and y dimensions (i.e. square in x-y)
            
        store_voxel_array : bool, optional
            store the voxel array in the Voxelizer object. Retrieve with voxelizer_object.voxel_arrays[body_name]

        slice_save_path : str, optional
            file path to directory in which to save .png images of each slice

        Returns
        -------
        voxel_array : np.ndarray
            voxelized array of the selected mesh
        """

        slicer = OpenGLSlicer()
        slicer.begin()

        # slice the selected mesh
        voxel_array = slicer.slice(self.global_bounds,self.meshes[body_name],xy_voxel_size,z_voxel_size,square_xy,slice_save_path=slice_save_path,value=voxel_value,dtype=voxel_dtype)
        
        if store_voxel_array == True:
            self.voxel_arrays[body_name] = voxel_array

        # close the OpenGL window
        slicer.quit()

        return voxel_array
    
    def _updateBounds(self,):
        """Identify max and min bounds of every mesh and set global bounding box size from these"""
        for _, _mesh in self.meshes.items():  
            self.global_bounds.xmin = np.nanmin([_mesh.bounds.xmin,self.global_bounds.xmin])
            self.global_bounds.xmax = np.nanmax([_mesh.bounds.xmax,self.global_bounds.xmax])
            self.global_bounds.ymin = np.nanmin([_mesh.bounds.ymin,self.global_bounds.ymin])
            self.global_bounds.ymax = np.nanmax([_mesh.bounds.ymax,self.global_bounds.ymax])
            self.global_bounds.zmin = np.nanmin([_mesh.bounds.zmin,self.global_bounds.zmin])
            self.global_bounds.zmax = np.nanmax([_mesh.bounds.zmax,self.global_bounds.zmax])

        self.global_bounds.length_x = np.abs(self.global_bounds.xmax - self.global_bounds.xmin)
        self.global_bounds.length_y = np.abs(self.global_bounds.ymax - self.global_bounds.ymin)
        self.global_bounds.length_z = np.abs(self.global_bounds.zmax - self.global_bounds.zmin)

class ShaderProgram():

    _glsl_vert = '''
    #version 330 core
    layout (location = 0) in vec3 vert;

    uniform mat4 model;
    uniform mat4 proj;

    void main() {
        gl_Position = proj * model * vec4(vert, 1.0f);
    }
    '''
    _glsl_frag = '''
    #version 330 core
    out vec4 FragColor;

    void main()
    {
        FragColor = vec4(1.0);
    }
    '''
    def __init__(self,):
        self._compileProgram()

    def _compileShaders(self,):
        self.vert_shader = shaders.compileShader(self._glsl_vert, GL_VERTEX_SHADER)
        self.frag_shader = shaders.compileShader(self._glsl_frag, GL_FRAGMENT_SHADER)

    def _compileProgram(self,):
        self._compileShaders()
        self.id = shaders.compileProgram(self.vert_shader,self.frag_shader)

    def use(self):
        glUseProgram(self.id)
        
    def delete(self):
        glDeleteProgram(self.id)
        
    def setInt(self, name, value):
        glUniform1i(glGetUniformLocation(self.id, name), int(value))
        
    def setMat4(self, name, arr):
        glUniformMatrix4fv(glGetUniformLocation(self.id, name), 1, GL_FALSE, arr)
        
    def get_uniform_location(self, name):
        return glGetUniformLocation(self.id, name)

class OpenGLSlicer():

    def __init__(self,):
        self.VAO, self.vertVBO, self.maskVAO, self.maskVBO = 0, 0, 0, 0
        self.slice_fbo, self.slice_tex, self.slice_buf = 0, 0, 0
        self.width, self.height = None, None

        self.shader = None

    def begin(self,):
        """Begin the pyglet window which will render each slice"""
        self.window = pyglet.window.Window()
        self.window.set_vsync(False)
        self.window.set_mouse_visible(False)
        self.window.switch_to()

    def quit(self,):
        """Close the pyglet slice rendering window"""
        self.window.close()

    def _drawWindow(self,):
        """Draw slice to window"""
        self.window.switch_to()
        self.window.flip()
        self.window.clear()

    def slice(self,bounds:Bounds,mesh:BodyMesh,xy_voxel_size:float,z_voxel_size:None,square_xy:bool=True,slice_save_path:str=None,value=1.0,dtype:str='uint8'):
        """
        Parameters
        ----------
        bounds : Bounds

        mesh : BodyMesh
            mesh object to be voxelized

        layer_thickness : float
            thickness of each slice in voxelized array in same units as the mesh file
        
        xy_voxel_size : float
            side length of square voxel cross section in x-y plane in voxelized array in same units as the mesh file
            
        z_voxel_size : float, optional
            thickness of each slice in voxelized array in same units as the mesh file, defaults to z_voxel_size = xy_voxel_size. This allows for non-isotropic voxel size in the z-axis
            
        square_xy : bool, optional
            if the resulting voxel array should have equal number of voxels in x and y dimensions (i.e. square in x-y)
        
        slice_save_path : str, optional
            file path to directory in which to save .png images of each slice
        
        value : np.uint8, optional
            value of voxels in voxelized array

            Only homogeneous voxel values are implemented

        Returns
        -------
        voxel_array : np.ndarray
            voxelized array of the selected mesh
        """
        # check if z_voxel_size is specified, if not it should be equal to xy_voxel_size for isotropic voxels
        if z_voxel_size == None:
            z_voxel_size = xy_voxel_size

        # update bounds
        self.bounds = bounds



        # preallocate voxel_array for the slicer
        length_x_voxels = int(self.bounds.length_x/xy_voxel_size)
        length_y_voxels = int(self.bounds.length_y/xy_voxel_size)
        length_z_voxels = int(np.floor(self.bounds.length_z/z_voxel_size))
        self.slicer_bounds = Bounds()

        if square_xy:
            # calculate diagonal such that the bounds will fit within the circle inscribed in the square grid
            bound_corner_vectors = np.array([[self.bounds.xmin,self.bounds.ymin],
                                      [self.bounds.xmin,self.bounds.ymax],
                                      [self.bounds.xmax,self.bounds.ymin],
                                      [self.bounds.xmax,self.bounds.ymax]])
            norms = np.linalg.norm(bound_corner_vectors,axis=1)
            norm_max = np.max(norms,axis=0)
            diagonal_length = 2*norm_max
            diagonal_length_voxels = int(diagonal_length/xy_voxel_size)
            slicer_length_x = diagonal_length_voxels
            slicer_length_y = diagonal_length_voxels
            self.slicer_bounds.xmin = -norm_max
            self.slicer_bounds.xmax = norm_max
            self.slicer_bounds.ymin = -norm_max
            self.slicer_bounds.ymax = norm_max
        else:
            slicer_length_x = length_x_voxels
            slicer_length_y = length_y_voxels
            self.slicer_bounds.xmin = bounds.xmin
            self.slicer_bounds.xmax = bounds.xmax
            self.slicer_bounds.ymin = bounds.ymin
            self.slicer_bounds.ymax = bounds.ymax

        voxel_array = np.zeros((slicer_length_y,slicer_length_x,length_z_voxels),dtype=dtype)


        # prepare the OpenGL window for rendering at the selected x-y grid size i.e. the window size in pixels is the same as the length in x-y voxels
        self.prepareSlice(slicer_length_x,slicer_length_y)

        self._makeMasks(mesh)
        
        # setup OpenGL shader
        self.shader = ShaderProgram()

        # each "slice" is a view mesh cross section at the center of each voxel thickness i.e. layer_thickness/2. First slice is at layer_thickness/2
        translation = z_voxel_size/2


        for i in tqdm.tqdm(range(length_z_voxels)):

            self._draw(translation-EPSILON,mesh)

            if slice_save_path is not None:
                array = self._renderSlice(translation-EPSILON,mesh,os.path.join(slice_save_path, f'out{i+1:04d}.png'))
            else:
                array = self._renderSlice(translation-EPSILON,mesh)

            # on the interior of the cross section (where there is solid), record this as the input voxel value
            array[array == 255] = value

            # insert the cross section at the corresponding layer of the input 
            voxel_array[:,:,i] = array
            
            # increment slicing plane by one layer thickness
            translation += z_voxel_size
            
            self._drawWindow()


        return voxel_array

    def _makeMasks(self,body_mesh:BodyMesh):
        # make VAO for drawing our mesh
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        # trimesh
        vertVBO = vbo.VBO(data=body_mesh.mesh.triangles.astype(
            GLfloat).tobytes(), usage='GL_STATIC_DRAW', target='GL_ARRAY_BUFFER')
        
        vertVBO.bind()
        vertVBO.copy_data()
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                            3 * sizeof(GLfloat), vertVBO)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

        # a mask vertex array for stencil buffer to subtract
        # uses global bounds of all meshes 
        maskVert = np.array(
            [[self.slicer_bounds.xmin, self.slicer_bounds.ymin, 0],
            [self.slicer_bounds.xmax, self.slicer_bounds.ymin, 0],
            [self.slicer_bounds.xmax, self.slicer_bounds.ymax, 0],

            [self.slicer_bounds.xmin, self.slicer_bounds.ymin, 0],
            [self.slicer_bounds.xmax, self.slicer_bounds.ymax, 0],
            [self.slicer_bounds.xmin, self.slicer_bounds.ymax, 0]], dtype=GLfloat
        )

        # make VAO for drawing mask
        self.maskVAO = glGenVertexArrays(1)
        glBindVertexArray(self.maskVAO)
        maskVBO = vbo.VBO(data=maskVert.tobytes(),
                        usage='GL_STATIC_DRAW', target='GL_ARRAY_BUFFER')
        maskVBO.bind()
        maskVBO.copy_data()
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                            3 * sizeof(GLfloat), maskVBO)
        glEnableVertexAttribArray(0)
        maskVBO.unbind()
        glBindVertexArray(0)

    def prepareSlice(self,width,height):
        self.width = width
        self.height = height
        self.slice_fbo = glGenFramebuffers(1)
        self.slice_tex = glGenTextures(1)
        self.slice_buf = glGenRenderbuffers(1)

        glBindTexture(GL_TEXTURE_2D, self.slice_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width,
                    self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _setModelLocation(self,translation):
        proj = orthoMatrix(self.slicer_bounds.xmin, self.slicer_bounds.xmax,
                        self.slicer_bounds.ymin, self.slicer_bounds.ymax,
                        -self.bounds.zmax, self.bounds.zmax, GLfloat)

        self.shader.setMat4("proj", proj)

        model = translationMatrix([0, 0, 0+translation], GLfloat)
        self.shader.setMat4("model", model)

    def _draw(self,translation,body_mesh:BodyMesh):
    
        glEnable(GL_STENCIL_TEST)
        glClearColor(0., 0., 0., 1.)
        glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        glBindVertexArray(self.VAO)
        self.shader.use()

        self._setModelLocation(translation)

        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)
        glStencilFunc(GL_ALWAYS, 0, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_INCR)
        glDrawArrays(GL_TRIANGLES, 0, body_mesh.num_of_verts)

        glCullFace(GL_BACK)
        glStencilOp(GL_KEEP, GL_KEEP, GL_DECR)
        glDrawArrays(GL_TRIANGLES, 0, body_mesh.num_of_verts)
        glDisable(GL_CULL_FACE)

        glClear(GL_COLOR_BUFFER_BIT)
        glBindVertexArray(self.maskVAO)
        glStencilFunc(GL_NOTEQUAL, 0, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glDisable(GL_STENCIL_TEST)

    def _renderSlice(self, translation, body_mesh:BodyMesh, filename=None):
        glEnable(GL_STENCIL_TEST)
        glViewport(0, 0, self.width, self.height)
        glBindFramebuffer(GL_FRAMEBUFFER, self.slice_fbo)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.slice_tex, 0)
        glBindRenderbuffer(GL_RENDERBUFFER, self.slice_buf)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_STENCIL,
                            self.width, self.height)
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self.slice_buf)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        glBindVertexArray(self.VAO)
        self.shader.use()

        self._setModelLocation(translation)

        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)
        glStencilFunc(GL_ALWAYS, 0, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_INCR)
        glDrawArrays(GL_TRIANGLES, 0, body_mesh.num_of_verts)

        glCullFace(GL_BACK)
        glStencilOp(GL_KEEP, GL_KEEP, GL_DECR)
        glDrawArrays(GL_TRIANGLES, 0, body_mesh.num_of_verts)
        glDisable(GL_CULL_FACE)

        glClear(GL_COLOR_BUFFER_BIT)
        glBindVertexArray(self.maskVAO)
        glStencilFunc(GL_NOTEQUAL, 0, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glDisable(GL_STENCIL_TEST)

        data = glReadPixels(0, 0, self.width, self.height,
                            GL_RED, GL_UNSIGNED_BYTE)
        
        data_numpy = np.frombuffer(data,dtype=np.uint8).reshape((self.height,self.width))

        if filename is not None:
            image = Image.frombytes('L', (self.width, self.height), data, 'raw', 'L', 0, -1)
            image.save(filename)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDisable(GL_STENCIL_TEST)
        glViewport(0, 0, self.width, self.height)

        return np.copy(data_numpy)