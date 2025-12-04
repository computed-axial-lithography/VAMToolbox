import os
import numpy as np
from vamtoolbox.imagesequence import ImageConfig, ImageSeq

def export_sinogram_to_images(sinogram, output_dir, image_size=(512, 512),
                               bit_depth=8, normalization_percentile=99,
                               rotate_angle=0.0, invert_u=False, invert_v=False):
    """
    Converts a sinogram into a sequence of 2D projection images and saves them.

    Parameters:
        sinogram (np.ndarray): Shape (detector_rows, num_angles, detector_cols)
        output_dir (str): Directory where images will be saved
        image_size (tuple): (width, height) of output images
        bit_depth (int): Bit depth (default 8)
        normalization_percentile (float): Percentile for normalization
        rotate_angle (float): Rotation of projection images (in degrees)
        invert_u (bool): Flip image horizontally
        invert_v (bool): Flip image vertically

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    config = ImageConfig(
        image_dims=image_size,
        bit_depth=bit_depth,
        normalization_percentile=normalization_percentile,
        rotate_angle=rotate_angle,
        invert_u=invert_u,
        invert_v=invert_v
    )

    image_seq = ImageSeq(config, sinogram)
    image_seq.saveAsImages(output_dir, image_prefix="proj_", image_type=".png")

    print(f"[VAM] Projection images saved to: {output_dir}")
