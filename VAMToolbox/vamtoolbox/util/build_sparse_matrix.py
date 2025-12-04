# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the GNU GPLv3 license.

import time

import numpy as np
from scipy import sparse

import vamtoolbox


def buildMatrix(
    target_geo,
    proj_geo,
    sparse_format="csr",
    save_path: str | None = None,
    block_insert: bool | None = False,
):
    """
    save_path : str, optional
        For algebraic propagation only.
        File path to store the generated algebraic propagation matrix. File type .npz (scipy sparse matrix format)
    """
    # construct the implicit projector (propagator)
    A = vamtoolbox.projectorconstructor.projectorconstructor(target_geo, proj_geo)

    # get shape of the target and sinogram
    _target_shape = target_geo.array.shape
    space_domain_array = np.zeros_like(target_geo.array)
    proj_domain_array = A.forward(space_domain_array)
    sino_shape = proj_domain_array.shape

    space_domain_flat_array = space_domain_array.flatten()
    space_domain_flat_array[:] = 0
    proj_domain_flat_array = proj_domain_array.flatten()
    proj_domain_flat_array[:] = 0

    n_matrix_col = space_domain_flat_array.size
    n_matrix_row = proj_domain_flat_array.size

    P = sparse.lil_matrix(
        (n_matrix_row, n_matrix_col)
    )  # propagation (projection) matrix.
    # For building and saving, we use the sparse matrix format (instead of sparse array) for compatibility with GPU multiplication operations.

    print(f"Building sparse matrix with {n_matrix_row} rows and {n_matrix_col} columns.")

    # Measure shift-variant impulse response
    # store impulse response as row of sparse propagation (projection) matrix
    # TODO: implement block insertion
    if block_insert:
        raise Exception("Block insertsion is not yet implemented.")
    else:
        start_time = time.perf_counter()
        for row in range(n_matrix_row):
            proj_domain_flat_array[row] = 1
            proj_domain_flat_array[row - 1] = 0

            space_domain_array = A.backward(proj_domain_flat_array.reshape(sino_shape))
            P[row, :] = space_domain_array.flatten()
            print(
                f"Completed row: {row} = ({row*100/n_matrix_row :2.1f}%) at {time.perf_counter()-start_time : .1f}s"
            )

    start_time = time.perf_counter()
    print(f"Converting to {sparse_format} format.")
    P = P.asformat(sparse_format)
    print(f"Conversion done in {time.perf_counter() - start_time}")

    # Saving to file path, if specified
    if save_path is not None:
        start_time = time.perf_counter()
        print(f"Saving to {save_path}")
        sparse.save_npz(save_path, P)  # type: ignore
        print(f"Saving done in {time.perf_counter() - start_time}s.")

    return P


# if __name__ == "__main__":
#     input()
