# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the GNU GPLv3 license.

import logging
import os
import time

import scipy.sparse
import torch
import torch.sparse


class PyTorchAlgebraicPropagator:
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
    def __init__(
        self, target_geo, proj_geo, output_torch_tensor=False, dtype=torch.float32
    ):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(
            f"pyTorch algebraic propagation will be performed on: {repr(self.device)}"
        )
        self.output_torch_tensor = output_torch_tensor
        self.dtype = dtype  # Data type of value in sparse tensor

        # Load the matrix from disk in COO sparse format
        file_path = proj_geo.loading_path_for_matrix
        file_extension = os.path.splitext(file_path)[-1].lower()
        if file_extension == ".npz":
            try:
                self.logger.info("Loading sparse matrix from NPZ file...")
                start_time = time.perf_counter()
                npz_matrix = scipy.sparse.load_npz(file_path)
                self.logger.info(
                    f"Loading finished in {time.perf_counter() - start_time}"
                )
            except OSError:
                raise Exception(
                    f'Sparse matrix file at "{file_path}" does not exist or cannot be read.'
                )

            start_time = time.perf_counter()
            self.logger.debug("Converting sparse matrix to PyTorch tensor...")

            npz_matrix = scipy.sparse.coo_matrix(
                npz_matrix, copy=False
            )  # Converting to new coo_matrix format if it is either in other sparse formats
            # Convert the propagation matrix to a sparse tensor in COO format
            values = torch.as_tensor(
                npz_matrix.data, dtype=self.dtype, device=self.device
            )
            # indices = torch.LongTensor(np.vstack((npz_matrix.row, npz_matrix.col)), device = self.device) #stack in numpy
            row_indices = torch.as_tensor(
                npz_matrix.row, dtype=torch.int64, device=self.device
            )  # indices need to be int64: https://pytorch.org/docs/stable/sparse.html#construction
            col_indices = torch.as_tensor(
                npz_matrix.col, dtype=torch.int64, device=self.device
            )
            indices = torch.stack((row_indices, col_indices), dim=0)  # stack in torch
            self.propagation_matrix = torch.sparse_coo_tensor(
                indices, values, size=npz_matrix.shape, device=self.device
            )  # Construct sparse matrix using COO indices
            self.propagation_matrix = (
                self.propagation_matrix.to_sparse_csr()
            )  # Converting to csr_matrix format
            del npz_matrix, values, row_indices, col_indices, indices
            self.logger.debug(
                f"Conversion finished in {time.perf_counter() - start_time}"
            )

        elif file_extension == ".pt":
            try:
                self.logger.info("Loading sparse matrix from PyTorch tensor file...")
                start_time = time.perf_counter()
                self.propagation_matrix = torch.load(
                    file_path, map_location=self.device
                ).to(self.dtype)
                self.logger.info(
                    f"Loading finished in {time.perf_counter() - start_time}"
                )
            except OSError:
                raise Exception(
                    f'PyTorch tensor file at "{file_path}" does not exist or cannot be read.'
                )
            self.propagation_matrix = self.propagation_matrix.to_sparse_csr()  # Converting to new coo_matrix format if it is either in other sparse formats

        else:
            raise Exception(
                "Invalid file format. Supported formats: .npz (SciPy) or .pt (PyTorch)"
            )

        # Transpose of the propagation matrix to perform adjoint operation (backpropagation)
        self.propagation_matrix_H = torch.transpose(
            self.propagation_matrix, 0, 1
        )  # does not create a new tensor but only a new view (effectively CSC)
        # FIXME: Where is performance mode coming from?
        if _performance_mode := False:  # Determine the format
            self.propagation_matrix_H = (
                self.propagation_matrix_H.to_sparse_csr()
            )  # Create another copy in CSR format for maximum performance

        # Check if the shape of imported sparse matrix is compatible to target_geo and proj_geo
        # Compatibile means that either the imported sparse matrix is created exactly for the dimension of the input/output of the propagation,
        # or it can be extended along z to produce propagation result. The latter case being matrix created for 2D problem being used to project in 3D, assuming slice independency.
        self.internal_matrix_shape = self.propagation_matrix.shape
        self.n_row_internal, self.n_col_internal = self.internal_matrix_shape

        # Determine z_tiling and effective matrix shape
        self.n_col_eff = target_geo.array.size  # Number of voxel in target_geo
        if (self.n_col_eff % self.n_col_internal) != 0:
            raise Exception(
                f"The imported sparse matrix has {self.n_col_internal} columns, and it cannot match or be tiled to match the total number of voxels ({self.n_col_eff}) in the real space target."
            )

        self.z_tiling = target_geo.array.size // self.n_col_internal
        self.n_row_eff = self.internal_matrix_shape[0] * self.z_tiling
        self.shape = (
            self.n_row_eff,
            self.n_col_eff,
        )  # shape of the effective linear operator

    @torch.inference_mode()
    def forward(self, x):
        if not isinstance(
            x, torch.Tensor
        ):  # Convert the input to a torch tensor if it is not
            x = torch.as_tensor(x, device=self.device, dtype=self.dtype)

        # if x.ndimension() > 1:
        #     x = x.flatten()

        x = x.reshape((-1, self.z_tiling)).to(
            self.dtype
        )  # Reshape x for batched matrix multiplication. Make sure data type is correct.
        b = torch.sparse.mm(self.propagation_matrix, x).flatten()
        if self.output_torch_tensor:
            return b
        else:
            return b.cpu().numpy()

    @torch.inference_mode()
    def backward(self, b):
        if not isinstance(
            b, torch.Tensor
        ):  # Convert the input to a torch tensor if it is not
            b = torch.as_tensor(b, device=self.device, dtype=self.dtype)

        # if b.ndimension() > 1:
        #     b = b.flatten()

        b = b.reshape((-1, self.z_tiling)).to(
            self.dtype
        )  # Reshape b for batched matrix multiplication. Make sure data type is correct.
        x = torch.sparse.mm(self.propagation_matrix_H, b).flatten()
        if self.output_torch_tensor:
            return x
        else:
            return x.cpu().numpy()

    # def inverseBackward(self, x, method='lsqr', atol=1e-6, btol=1e-6, iter_lim=50, show=True, x0=None):
    #     '''
    #     #Currently only works for one 2D slice.

    #     (Approximate) inverse of backpropagation operator
    #     In x = (A^T)b, for a given x, find approximate solution b .
    #     In f = (P^T)g, for a given f, find approximate solution g.

    #     #Options
    #     LSQR: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
    #     LSMR: https://docs.scipy.org/doc/scipy-1.9.1/reference/generated/scipy.sparse.linalg.lsmr.html#scipy.sparse.linalg.lsmr

    #     #TODO: Get it working for multiple layers. Also test if optimization works for linear model and zero initialization.
    #     '''
    #     print('Computing inverse of backpropagation...')
    #     if x.ndim > 1:
    #         x = x.flatten() #raval (returns a view) and flatten (returns a copy) both works. Theoretically raval is faster but in test flatten is faster, potentially due to further slicing in _matvec

    #     AT = self.propagation_matrix.transpose(copy=False)
    #     if method =='lsqr':
    #         b, istop, itn = sparse.linalg.lsqr(AT, x, atol=atol, btol=btol, iter_lim=iter_lim, show=show, x0=x0)[:3]
    #     elif method == 'lsmr':
    #         b, istop, itn = sparse.linalg.lsmr(AT, x, atol=atol, btol=btol, maxiter=iter_lim, show=show, x0=x0)[:3]
    #     elif method == 'zeros':
    #         b = np.zeros(self.internal_matrix_shape[0] * self.z_tiling)
    #     else:
    #         raise Exception('Specificed method is not supported.')

    #     return b
    #     return b
