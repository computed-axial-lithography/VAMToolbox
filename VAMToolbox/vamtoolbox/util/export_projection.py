import os
import numpy as np
from vamtoolbox.imagesequence import ImageConfig, ImageSeq

def export_sinogram_to_images(sinogram, output_dir, image_size=(512, 512),
                               bit_depth=8, normalization_percentile=99,
                               rotate_angle=0.0, invert_u=False, invert_v=False, helical_pitch_pixels=2.0, start_v_offset=0):
    
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

    helical_pitch_pixels : float
        Vertical shift (in pixels) per projection angle

    start_v_offset : int
        Initial vertical offset (pixels)

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)

    num_angles = sinogram.shape[0]

    for k in range(num_angles):

        # HELICAL MOTION
        v_offset = int(start_v_offset + k * helical_pitch_pixels)
        config = ImageConfig(
            image_dims=image_size,
            bit_depth=bit_depth,
            normalization_percentile=normalization_percentile,
            rotate_angle=rotate_angle,
            invert_u=invert_u,
            invert_v=invert_v,
            v_offset=v_offset,
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

    print(f"[VAM] Helical projection images saved to: {output_dir}")
