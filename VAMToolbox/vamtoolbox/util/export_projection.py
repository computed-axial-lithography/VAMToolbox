import os
import numpy as np
from vamtoolbox.imagesequence import ImageConfig, ImageSeq

def export_sinogram_to_images(sinogram, output_dir, image_size=(512, 512),
                               bit_depth=8, normalization_percentile=99,
                               rotate_angle=0.0, invert_u=False, invert_v=False, v_offset=0, size_scale=1.0):
    
    """
    Export a sinogram as a TRUE helical projector image sequence.

    Helicity is introduced by applying an angle-dependent vertical offset
    to each projection image.

    Parameters
    ----------
    sinogram : np.ndarray
        Shape (num_angles, detector_rows, detector_cols)

    output_dir : str
        Output directory for PNG images

    image_size : tuple
        (width, height) of projector images

    bit_depth : int
        Bit depth of output images

    normalization_percentile : float
        Percentile used for intensity normalization

    rotate_angle : float
        In-plane rotation of projector images (degrees)

    invert_u : bool
        Horizontal flip

    invert_v : bool
        Vertical flip

    v_offset : int
        Vertical offset (pixels)

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)

    num_angles = sinogram.shape[0]
    scale_factor = size_scale

    for k in range(num_angles):


        config = ImageConfig(
            image_dims=image_size,
            bit_depth=bit_depth,
            normalization_percentile=normalization_percentile,
            rotate_angle=rotate_angle,
            invert_u=invert_u,
            invert_v=invert_v,
            v_offset=v_offset,
            size_scale=scale_factor,
        )

        # LEAP: (angles, rows, cols) -> VAM/ImageSeq: (rows, angles, cols)
        sino_vam = np.transpose(sinogram, (1, 0, 2))   # (400, 360, 512)
        # Extract single-angle projection
        sino_k = sino_vam[:, k:k+1, :]  

        image_seq = ImageSeq(config, sino_k)

        image_seq.saveAsImages(
            output_dir,
            image_prefix=f"proj_{k:04d}_",
            image_type=".png",
        )
        if k == 0:
            sino_vam = np.transpose(sinogram, (1, 0, 2))
            sino_k = sino_vam[:, 0:1, :]
            print(f"sino_k shape (before rotate): {sino_k.shape}")
            
            from scipy import ndimage
            rotated = ndimage.rotate(sino_k, 90, axes=(0, 2), reshape=True, order=1)
            print(f"rotated shape (after rotate):  {rotated.shape}")
            
            from vamtoolbox.imagesequence import _scaleSize
            dbg_scale_factor = image_size[1] / sinogram.shape[1]
            scaled = _scaleSize(rotated, dbg_scale_factor)
            print(f"scaled shape (after scale):    {scaled.shape}")
            print(f"scale_factor used:             {dbg_scale_factor}")
            
            # After .T in _insertImage
            image = scaled[:, 0, :].T
            print(f"image inserted S_v (height):   {image.shape[0]}  (should be 4800)")
            print(f"image inserted S_u (width):    {image.shape[1]}  (should be ~2560)")

    print(f"[VAM] Helical projection images saved to: {output_dir}")
