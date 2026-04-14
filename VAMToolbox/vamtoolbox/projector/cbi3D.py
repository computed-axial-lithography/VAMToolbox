import numpy as np
from cbi_toolbox import splineradon as spl
from cbi_toolbox.simu import optics, imaging
from cbi_toolbox.reconstruct import fpsopt


class CBIProjectorWrapper:
    """
    Forward:  imaging.fps_opt      — blurred projection with PSF
    Backward: fpsopt.deconvolve_sinogram + spl.iradon — matches CBI demo pipeline

    CBI axis conventions:
        Volume:   (Z, X, Y)
        Sinogram: (T, P, Y)

    VAM axis conventions:
        Volume:   (Z, Y, X)
        Sinogram: (T, Z, X)
    """

    def __init__(
        self,
        angles_deg,
        numerical_aperture = 0.1,   
        npix_axial=None,
        npix_lateral=None,
        pixelscale=50e-6,
        deconv_lambda=200,
        deconv_mode="laplacian",
        circle=False,
        mode="fps_opt",          # "fps_opt" 或 "spim"
        focal_offset=0,          # for spim mode, focal plane offset from center (pixels)
        slit_opening=2e-3,       # for spim mode, slit opening in metres
    ):
        """
        Parameters
        ----------
        angles_deg : array-like
            Projection angles in degrees.
        numerical_aperture : float
            NA of the optical system, controls PSF size.
        npix_axial : int, optional
            PSF axial size in pixels. Defaults to volume Z size + 1.
        npix_lateral : int, optional
            PSF lateral size in pixels. Defaults to volume X size + 1.
        pixelscale : float
            Physical pixel size in metres, by default 3e-7 (300 nm).
        deconv_lambda : float
            Regularization strength for sinogram deconvolution, by default 20.
        deconv_mode : str
            Regularizer for deconvolution, 'laplacian' or 'constant'.
        circle : bool
            If True, only pixels inside the inscribed circle are used.
        mode : str
            Projection mode, 'fps_opt' for standard PSF-based projection, 'spim' for SPIM-like slit illumination.
        focal_offset : int
            Focal plane offset from center in pixels.
            Works for both 'fps_opt' (shifts PSF) and 'spim' (shifts illu).
            0 = center, int(radius*0.5) = 50%, int(radius) = edge.
        """
        self.angles_deg = np.asarray(angles_deg)
        self.numerical_aperture = numerical_aperture
        self.npix_axial = npix_axial
        self.npix_lateral = npix_lateral
        self.pixelscale = pixelscale
        self.deconv_lambda = deconv_lambda
        self.deconv_mode = deconv_mode
        self.circle = circle
        self._psf = None 
        self.mode = mode
        self.focal_offset = focal_offset
        self.slit_opening = slit_opening
        self._illu = None

    def _get_psf(self, volume_shape):
        """Build PSF"""
        npix_axial = self.npix_axial or volume_shape[0] + 1
        npix_lateral = self.npix_lateral or volume_shape[1] + 1

        if self._psf is None or self._psf.shape != (npix_axial, npix_lateral, npix_lateral):
            self._psf = optics.gaussian_psf(
                numerical_aperture=self.numerical_aperture,
                npix_axial=npix_axial,
                npix_lateral=npix_lateral,
                pixelscale=self.pixelscale,
            )
        if self.focal_offset != 0:
            return np.roll(self._psf, self.focal_offset, axis=0)
        return self._psf

    def _get_illu(self, volume_shape):
        """Build SPIM light sheet illumination"""
        if self._illu is None:
            self._illu = optics.openspim_illumination(
                slit_opening=self.slit_opening,
                npix_fov=volume_shape[1],
                simu_size=4 * volume_shape[1],
                rel_thresh=1e-3,
            )
        return np.roll(self._illu, self.focal_offset, axis=0)

    def forward(self, volume):
        # save original sizes for backward
        self._z_size = volume.shape[0]  
        self._x_size = volume.shape[2]  

        volume_cbi = volume

        psf = self._get_psf(volume_cbi.shape)

        if self.mode == "fps_opt":
            sinogram_cbi = imaging.fps_opt(volume_cbi, psf, theta=self.angles_deg)
            return sinogram_cbi.real.transpose(0, 2, 1)  # (T, P, Y) → (T, Y, P)

        elif self.mode == "opt":
            # opt() requires ZX square; focal_offset via np.roll(psf, offset, axis=0) is effective here
            if volume_cbi.shape[0] != volume_cbi.shape[1]:
                raise ValueError(
                    f"opt mode requires square ZX, got {volume_cbi.shape[0]} x {volume_cbi.shape[1]}. "
                    "Ensure XY padding makes volume ZX square."
                )
            sinogram_cbi = imaging.opt(volume_cbi, psf, theta=self.angles_deg)
            return sinogram_cbi.real.transpose(0, 2, 1)  # (T, P, Y) → (T, Y, P)

        elif self.mode == "spim":
            illu = self._get_illu(volume_cbi.shape)
            image_cbi = imaging.spim(volume_cbi, psf, illu)
            return image_cbi.real    # (Z, X, Y)

        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'opt', 'fps_opt', or 'spim'.")


    def forward_with_offsets(self, volume, offsets):
        """
        Run forward projection for multiple focal offsets.

        Parameters
        ----------
        volume : ndarray
            Input volume.
        offsets : dict
            {name: focal_offset_pixels}, e.g. {"center": 0, "50pct": 500, "edge": 1000}

        Returns
        -------
        dict
            {name: sinogram} for each offset.
        """
        results = {}
        original_offset = self.focal_offset
        for name, offset in offsets.items():
            self.focal_offset = offset
            self._psf = None   # clear cache so PSF is rebuilt with new offset
            results[name] = self.forward(volume)
        self.focal_offset = original_offset
        self._psf = None
        return results

    def backward(self, sinogram):
        from scipy.ndimage import zoom

        psf = self._psf
        if psf is None:
            raise RuntimeError("PSF not initialised — call forward() before backward().")

        sinogram = sinogram.transpose(0, 2, 1)  # (T, Y, P) → (T, P, Y) for CBI

        sinogram_deconv = fpsopt.deconvolve_sinogram(
            sinogram, psf,
            l=self.deconv_lambda,
            mode=self.deconv_mode,
        ).real

        # replace any NaN/inf from unstable deconvolution
        sinogram_deconv = np.nan_to_num(sinogram_deconv, nan=0.0, posinf=0.0, neginf=0.0)

        # if deconv result is all zeros, fall back to raw sinogram
        if np.all(sinogram_deconv == 0):
            sinogram_deconv = sinogram

        volume_cbi = spl.iradon(sinogram_deconv, self.angles_deg, circle=False, use_cuda=False)
        # iradon output: (90, 90, 600)

        # resize Z and Y axes back to original size, X stays the same
        z_scale = self._z_size / volume_cbi.shape[0]   # 128/90
        y_scale = self._z_size / volume_cbi.shape[1]   # 128/90
        x_scale = self._x_size / volume_cbi.shape[2]   # 600/600 = 1.0
        volume_resized = zoom(volume_cbi, (z_scale, y_scale, x_scale))
        # output: (128, 128, 600)

        return volume_resized