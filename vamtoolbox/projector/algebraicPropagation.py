from scipy import sparse
import time
import numpy as np
#CPU implementation.
#subclass scipy linear operator. wrap their matrix-vector multiplication with * , transpose and ".forward" ".backward" member functions.
#The operator automatically identify the z-size mismatch (if there is any, and apply to all the layers.)


class AlgebraicPropagator(sparse.linalg.LinearOperator):
    """
    CPU implementation of algebraic propagator as a subclass of scipy sparse linear operator
    This operator is the A in Ax = b, or the P in Pf = g, where x or f is the real space object (flattened into 1D array) and b or g is the sinogram (flattened into 1D array).
    
    Notes: Due to the demanding memory requirement in this algebraic propagation approach, it is often to only create projection matrix in 2D and apply it to both 2D and 3D problems.
    In this simplified case of decoupling a 3D problem into many 2D problems, the imported sparse matrix is only required to encode projection process of one z-layer. 

    Therefore, the number of real space voxels only has to be equal to the integer multiples of the number of columns in imported sparse matrix (.npz).
    This requirement reads as (x.size % A.shape[1]) == 0.  
    For example,  x.size = 256 * A.shape[1] when there are 256 z-layers but A only encode projection of one layer. 

    As in other projectors, the size of the sinogram is determined by the projection process itself.
    Forward propagation using this method produce sinogram that has the size of (number of rows of sparse propagation matrix * number of z-layer extension) 
    Continuing from above example, it reads b.size = A.shape[0]*256
    
    Backpropagation process will check for (b.size % A.shape[0]) == 0

    #TODO: replace _rmatvec with _adjoint
    # use ravel() to perform z-tiled multiplication instead of loops
    """
    def __init__(self, target_geo, proj_geo) -> None:

        try:
            start_time = time.perf_counter()
            print('Loading sparse matrix...')
            self.propagation_matrix = sparse.load_npz(proj_geo.loading_path_for_matrix)
        except OSError: #override the error message for clarity
            raise Exception(f'Sparse matrix file at "loading_path_for_matrix"={self.loading_path_for_matrix} does not exist or cannot be read.')
        print(f'Loading finished in {time.perf_counter()-start_time}')

        self.propagation_matrix = sparse.csr_array(self.propagation_matrix, copy=False) #Converting to new csr_array format if it is either in other sparse formats

        self.internal_matrix_shape = self.propagation_matrix.shape #The internal matrix shape can be smaller (for 2D problems) than the effective operator shape (for 3D problems).
        self.n_row_internal, self.n_col_internal = self.internal_matrix_shape

        #Get array shape of real space target.
        n_col_eff = target_geo.array.size #Number of voxel in target_geo

        #Check if the shape of imported sparse matrix is compatible to target_geo and proj_geo
        #Compatibile means that either the imported sparse matrix is created exactly for the dimension of the input/output of the propagation,
        # or it can be extended along z to produce propagation result. The latter case being matrix created for 2D problem being used to project in 3D, assuming slice independency.  
        if (n_col_eff % self.n_col_internal) != 0: #The target_geo is not a z-extension of the col
            raise Exception(f'The imported sparse matrix has {self.n_col_internal} columns, and it can neither match or be tiled to match total number of voxel ({n_col_eff})of real space target.')

        self.z_tiling = target_geo.array.size // self.n_col_internal # minimum 1
        n_row_eff = self.internal_matrix_shape[0] * self.z_tiling

        #Define the effective linear operator shape
        super().__init__(shape = (n_row_eff, n_col_eff), dtype = self.propagation_matrix.dtype) #Supply the properties of the sparse matrix to superclass for output size checking.

        self.forward_cache = np.zeros(n_row_eff)
        self.backward_cache = np.zeros(n_col_eff)


    def _matvec(self, x):
        #Forward propagation. return b = Ax or eqivalently g = Pf
        if x.ndim > 1:
            x = x.flatten() #raval (returns a view) and flatten (returns a copy) both works. Theoretically raval is faster but in test flatten is faster, potentially due to further slicing in _matvec
    
        for z_section in range(self.z_tiling): #handle each z-slice every loop
            #ravel, reshape and flatten are by default in C order, the last index (z dimension) varies the quickest.
            self.forward_cache[z_section::self.z_tiling] = self.propagation_matrix.dot(x[z_section::self.z_tiling])

        ''' #To be tested
        x = x.reshape((x.shape[0]//self.z_tiling,self.z_tiling)) # x.shape[0]//self.z_tiling = self.n_col_internal
        return self.propagation_matrix.dot(x).flatten() #or ravel
        '''
        return self.forward_cache



    def _rmatvec(self, b):
        #Backpropagation. return x = (A^T)b or eqivalently f = (P^T)g
        if b.ndim > 1:
            b = b.flatten() #raval (returns a view) and flatten (returns a copy) both works. Theoretically raval is faster but in test flatten is faster, potentially due to further slicing in _rmatvec
    
        AT = self.propagation_matrix.transpose(copy=False)
        for z_section in range(self.z_tiling): #handle each z-slice every loop
            #ravel, reshape and flatten are by default in C order, the last index (z dimension) varies the quickest.
            self.backward_cache[z_section::self.z_tiling] = AT.dot(b[z_section::self.z_tiling])
            #Although mathematically, np.dot(b.T,A).T is equivalent but np will convert the sparse array to dense array so it will consume unreasonable amount of memory. It should not be faster either.  

        ''' #To be tested
        b = b.reshape((b.shape[0]//self.z_tiling,self.z_tiling)) # b.shape[0]//self.z_tiling = self.n_row_internal
        return AT.dot(b).flatten() #or ravel
        '''
        return self.backward_cache

    def forward(self, x):  #Wrapper around matvec(x)
        return self.matvec(x)

    def backward(self, b): #Wrapper around rmatvec(b)
        return self.rmatvec(b)
    

    def inverseBackward(self, x, method='lsqr', atol=1e-6, btol=1e-6, iter_lim=50, show=True, x0=None):
        '''
        #Currently only works for one 2D slice.

        (Approximate) inverse of backpropagation operator
        In x = (A^T)b, for a given x, find approximate solution b .
        In f = (P^T)g, for a given f, find approximate solution g.

        #Options
        LSQR: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        LSMR: https://docs.scipy.org/doc/scipy-1.9.1/reference/generated/scipy.sparse.linalg.lsmr.html#scipy.sparse.linalg.lsmr

        #TODO: Get it working for multiple layers. Also test if optimization works for linear model and zero initialization.
        '''
        print('Computing inverse of backpropagation...')
        if x.ndim > 1:
            x = x.flatten() #raval (returns a view) and flatten (returns a copy) both works. Theoretically raval is faster but in test flatten is faster, potentially due to further slicing in _matvec
        
        AT = self.propagation_matrix.transpose(copy=False)
        if method =='lsqr':
            b, istop, itn = sparse.linalg.lsqr(AT, x, atol=atol, btol=btol, iter_lim=iter_lim, show=show, x0=x0)[:3]
        elif method == 'lsmr':
            b, istop, itn = sparse.linalg.lsmr(AT, x, atol=atol, btol=btol, maxiter=iter_lim, show=show, x0=x0)[:3]
        elif method == 'zeros':
            b = np.zeros(self.internal_matrix_shape[0] * self.z_tiling)
        else:
            raise Exception('Specificed method is not supported.')

        return b

