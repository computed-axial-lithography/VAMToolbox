import time
import torch
import numpy as np
import torch.sparse
import scipy.sparse
import os
import logging
import matplotlib.pyplot as plt

class PyTorchAlgebraicR2RPropagator:

    """
    CPU/GPU implementation with pyTorch.
    The operator automatically identify the z-size mismatch (if there is any, and apply to all the layers.)
    
    This operator is the A in Ax = b, or the P in Pf = g, where x or f is the real space object (flattened into 1D array) and b or g is the sinogram (flattened into 1D array).
    
    Notes: Due to the demanding memory requirement in this algebraic propagation approach, it is often to only create projection matrix in 2D and apply it to both 2D and 3D problems.
    In this simplified case of decoupling a 3D problem into many 2D problems, the imported sparse matrix is only required to encode projection process of one z-layer. 

    Therefore, the number of real space voxels only has to be equal to the integer multiples of the number of columns in imported sparse matrix (.py or .npz).
    This requirement reads as (x.size % A.shape[1]) == 0.  
    For example,  x.size = 256 * A.shape[1] when there are 256 z-layers but A only encode projection of one layer. 

    As in other projectors, the size of the sinogram is determined by the projection process itself.
    Forward propagation using this method produce sinogram that has the size of (number of rows of sparse propagation matrix * number of z-layer tiling) 
    Continuing from above example, it reads b.size = A.shape[0]*256
    
    Vice versa, Backpropagation process will check for (b.size % A.shape[0]) == 0

    Current implementation store sparse matrix in CSR format for fast matrix-vector multiplication.
    #TODO: Enable performance mode to optionally allow explicit storage of adjoint of the matrix also in CSR (instead of CSC).
    """
    @torch.inference_mode()
    def __init__(self, target_geo, proj_geo, output_torch_tensor = False, dtype = torch.float32):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f'pyTorch algebraic propagation will be performed on: {repr(self.device)}')
        self.output_torch_tensor = output_torch_tensor
        self.dtype = dtype #Data type of value in sparse tensor

        self.target_geo = target_geo
        self.proj_geo = proj_geo 

        #Load the matrix from disk in COO sparse format
        file_path = proj_geo.loading_path_for_matrix
        file_extension = os.path.splitext(file_path)[-1].lower()
        if file_extension == '.npz':
            try:
                self.logger.info('Loading sparse matrix from NPZ file...')
                start_time = time.perf_counter()
                npz_matrix = scipy.sparse.load_npz(file_path)
                self.logger.info(f'Loading finished in {time.perf_counter()-start_time}')
            except OSError:
                raise Exception(f'Sparse matrix file at "{file_path}" does not exist or cannot be read.')
            
            start_time = time.perf_counter()
            self.logger.debug('Converting sparse matrix to PyTorch tensor...')

            npz_matrix = scipy.sparse.coo_matrix(npz_matrix, copy=False) #Converting to new coo_matrix format if it is either in other sparse formats
            # Convert the propagation matrix to a sparse tensor in COO format
            values = torch.as_tensor(npz_matrix.data, dtype = self.dtype, device = self.device)
            # indices = torch.LongTensor(np.vstack((npz_matrix.row, npz_matrix.col)), device = self.device) #stack in numpy
            row_indices = torch.as_tensor(npz_matrix.row, dtype = torch.int64, device = self.device) #indices need to be int64: https://pytorch.org/docs/stable/sparse.html#construction
            col_indices = torch.as_tensor(npz_matrix.col, dtype = torch.int64, device = self.device)
            indices = torch.stack((row_indices, col_indices), dim = 0) #stack in torch
            self.propagation_matrix = torch.sparse_coo_tensor(indices, values, size = npz_matrix.shape, device = self.device) #Construct sparse matrix using COO indices
            self.propagation_matrix = self.propagation_matrix.to_sparse_csr() #Converting to csr_matrix format
            del npz_matrix, values, row_indices, col_indices, indices
            self.logger.debug(f'Conversion finished in {time.perf_counter()-start_time}')

        elif file_extension == '.pt':
            try:
                self.logger.info('Loading sparse matrix from PyTorch tensor file...')
                start_time = time.perf_counter()
                self.propagation_matrix = torch.load(file_path, map_location = self.device).to(self.dtype)
                self.logger.info(f'Loading finished in {time.perf_counter()-start_time}')
            except OSError:
                raise Exception(f'PyTorch tensor file at "{file_path}" does not exist or cannot be read.')
            self.propagation_matrix = self.propagation_matrix.to_sparse_csr() #Converting to new coo_matrix format if it is either in other sparse formats

        else:
            raise Exception("Invalid file format. Supported formats: .npz (SciPy) or .pt (PyTorch)")
        
        #Transpose of the propagation matrix to perform adjoint operation (backpropagation)
        self.propagation_matrix_H = torch.transpose(self.propagation_matrix,0,1) #does not create a new tensor but only a new view (effectively CSC)
        if performance_mode:=False: #Determine the format 
            self.propagation_matrix_H = self.propagation_matrix_H.to_sparse_csr() #Create another copy in CSR format for maximum performance

        #Check if the shape of imported sparse matrix is compatible to target_geo and proj_geo
        #Compatibile means that either the imported sparse matrix is created exactly for the dimension of the input/output of the propagation,
        # or it can be extended along z to produce propagation result. The latter case being matrix created for 2D problem being used to project in 3D, assuming slice independency.  
        self.internal_matrix_shape = self.propagation_matrix.shape
        self.n_row_internal, self.n_col_internal = self.internal_matrix_shape

        # Determine z_tiling and effective matrix shape
        self.n_col_eff = self.target_geo.nRho*self.target_geo.nL*self.target_geo.nZ #Number of voxel in moving window
        if (self.n_col_eff % self.n_col_internal) != 0:
            raise Exception(f'The imported sparse matrix has {self.n_col_internal} columns, and it cannot match or be tiled to match the total number of voxels ({self.n_col_eff}) in the real space moving window.')

        self.z_tiling = self.n_col_eff // self.n_col_internal 
        self.n_row_eff = self.internal_matrix_shape[0] * self.z_tiling
        self.shape = (self.n_row_eff, self.n_col_eff) #shape of the effective linear operator

        # Setup torch.nn.Unfold and torch.nn.Fold functions for unfolding and folding operations
        # kernel_size is the shape of the sliding sub image of the target (n_rho_uw,n_l_uw)
        # padding is the number of zeros to pad the target in each dimension (0,n_l_uw-1)
        # stride is the step size of the sliding sub image (1)
        # output_size is the shape of the reconstructed target including all overlapping sub images. It should be the original target shape
        kernel_size = (self.target_geo.nRho,self.target_geo.nL)
        padding = (0,self.target_geo.nL-1)
        output_size = (self.target_geo.nRho_total,self.target_geo.nL_total)
        stride = (1,self.target_geo.folding_stride)
        fold_params = dict(kernel_size=kernel_size,dilation=1,padding=padding,stride=stride)
        self.unfold = torch.nn.Unfold(**fold_params)
        self.fold = torch.nn.Fold(output_size=output_size,**fold_params)

        # calculate the number of blocks in the unfolding process
        # this depends on the kernel size (the moving window), target array size, and the stride of the kernel
        product = torch.zeros((2),device=self.device)
        for d in range(2):
            product[d] = torch.ceil(torch.tensor([(output_size[d] + 2*padding[d] - (kernel_size[d]-1)-1)/stride[d] + 1 ]))
        self.n_fold_blocks = int(torch.prod(product))
        self.z_fold_batch_size = 1
        
    @torch.inference_mode()
    def forward(self, x):
        if ~isinstance(x, torch.Tensor): #Convert the input to a torch tensor if it is not
            x = torch.as_tensor(x, device = self.device, dtype = self.dtype)

        b = torch.zeros((self.n_row_internal,self.n_fold_blocks,self.target_geo.nZ),device=self.device)
        
        # # Add axes for unfold function, input should be shape=(1,1,)
        # x = x[None,None,:,:,:]

        # '''==================WORKING===============Looping implementation'''
        # # loop over z. This allows longer targets to be unfolded and stored in memory for propagation matrix multiplication (as opposed to tiling also in z)
        # for i_z in range(self.z_tiling):
        #     # unfold x along theta axis. this is a rolling window that slides across x, stepping one angle at a time. 
        #     x_unfolded = self.unfold(x[:,:,:,:,i_z]) # has shape (1,n_r*n_theta,L)

        #     # fig, axs = plt.subplots(1,1)
        #     # axs.imshow(x_unfolded.squeeze()[:,100].reshape((self.target_geo.nRho,self.target_geo.nL)).cpu().numpy(),cmap='viridis')
        #     # plt.show()

        #     b[:,:,i_z] = torch.sparse.mm(self.propagation_matrix, x_unfolded.squeeze())

        '''==================WORKING===============Batched looping implementation'''
        x = x[None,:,:,:]
        # loop over z. This allows longer targets to be folded and stored in memory for each z (as opposed to tiling and folding also in z)
        n_z_batches = int(np.ceil(self.target_geo.nZ/self.z_fold_batch_size))
        for i_z in range(n_z_batches):
            _z_start = i_z*self.z_fold_batch_size
            _z_end = (i_z+1)*self.z_fold_batch_size
            if _z_end > self.target_geo.nZ:
                _z_end = self.target_geo.nZ
                _internal_z_batch_size = _z_end-_z_start
            else:
                _internal_z_batch_size = self.z_fold_batch_size

            # unfold x along theta axis. this is a rolling window that slides across x, stepping one angle at a time. 
            x_unfolded = self.unfold(x[:,:,:,_z_start:_z_end].permute((3,0,1,2))) # has shape (1,n_r*n_theta,L)

            sub_b = torch.sparse.mm(self.propagation_matrix, x_unfolded.permute((1,2,0)).reshape((-1,self.n_fold_blocks*_internal_z_batch_size)))

            b[:,:,_z_start:_z_end] = sub_b[:,:,None].reshape((self.n_row_internal,self.n_fold_blocks,_internal_z_batch_size))

        # b = b.flatten()
        if self.output_torch_tensor:
            return b
        else:
            return b.cpu().numpy()

    @torch.inference_mode()
    def backward(self, b):
        if ~isinstance(b, torch.Tensor): #Convert the input to a torch tensor if it is not
            b = torch.as_tensor(b, device = self.device, dtype = self.dtype)

        # '''original implementation'''
        # b = b.reshape((-1, self.z_tiling)).to(self.dtype)  # Reshape b for batched matrix multiplication. Make sure data type is correct.
        # x = torch.sparse.mm(self.propagation_matrix_H, b).flatten()
        

        x = torch.zeros(self.target_geo.array.shape,device=self.device,dtype=self.dtype) # preallocate x since we are looping over z
        
        # '''==================WORKING===============Looping implementation'''
        # # loop over z. This allows longer targets to be folded and stored in memory for each z (as opposed to tiling and folding also in z)
        # for i_z in range(self.z_tiling):
        #     x_sub_images = torch.sparse.mm(self.propagation_matrix_H, b[:,:,i_z]) # has shape (nRho*nL,n_theta)
            
        #     # fold x along l axis. this is a rolling window that slides across x, stepping one angle at a time. 
        #     x[:,:,i_z] = self.fold(x_sub_images[None,:,:]).squeeze() # has shape (batches,channels,n_r,n_theta) before squeeze and (n_r,n_theta) after squeeze
        #     # fig, axs = plt.subplots(1,1)
        #     # axs.imshow(x.cpu().numpy(),cmap='viridis')
        #     # plt.show()

        '''==================WORKING====================Batched looping implementation'''
        # loop over z. This allows longer targets to be folded and stored in memory for each z (as opposed to tiling and folding also in z)
        n_z_batches = int(np.ceil(self.target_geo.nZ/self.z_fold_batch_size))
        for i_z in range(n_z_batches):
            _z_start = i_z*self.z_fold_batch_size
            _z_end = (i_z+1)*self.z_fold_batch_size
            if _z_end > self.target_geo.nZ:
                _z_end = self.target_geo.nZ
                _internal_z_batch_size = _z_end-_z_start
            else:
                _internal_z_batch_size = self.z_fold_batch_size
            
            x_sub_images = torch.sparse.mm(self.propagation_matrix_H, b[:,:,_z_start:_z_end].reshape((-1,self.n_fold_blocks*_internal_z_batch_size))) # has shape (nRho*nL,n_theta)
            x_sub_images = x_sub_images.reshape((-1,self.n_fold_blocks,_internal_z_batch_size)).permute((2,0,1))
            
            # fold x along l axis. this is a rolling window that slides across x, stepping one angle at a time. 
            x[:,:,_z_start:_z_end] = self.fold(x_sub_images)[:,0,:,:].permute((1,2,0)) # has shape (batches,channels,n_r,n_theta) before squeeze and (n_r,n_theta) after squeeze


        if self.output_torch_tensor:
            return x
        else:
            return x.cpu().numpy()