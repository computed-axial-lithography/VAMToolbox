from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.fft import fft, fft2, ifft, ifft2, next_fast_len


def sigmoid(x, g):
    y = 1 / (1 + np.exp(-x * (1 / g)))
    return y.astype(float)


def clipToCircle(x: np.ndarray):
    """
    Sets all data outside the inscribed circle to zero

    Parameters
    ----------
    x : np.ndarray
        square array to be modified

    Returns
    -------
    x
        input array with all data outside the inscribed circle set to zero

    """
    circle_y, circle_x = np.meshgrid(
        np.linspace(-1, 1, x.shape[0]), np.linspace(-1, 1, x.shape[1])
    )
    if x.ndim == 2:
        x[circle_x**2 + circle_y**2 > 1] = 0
    else:
        circle = np.broadcast_to(
            (circle_x**2 + circle_y**2 > 1)[..., np.newaxis], x.shape
        )
        x[circle] = 0

    return x


def filterTargetOSMO(x: np.ndarray, filter_name: str):
    """
    Parameters
    ----------
    x : np.ndarray

    filter_name : str
        type of filter to apply to target, options: "ram-lak", "shepp-logan", "cosine", "hamming", "hanning", "none"

    Returns
    -------
    x_filtered : np.ndarray
        direct output of filtering in frequency space
    """
    filter_types = ("ram-lak", "shepp-logan", "cosine", "hamming", "hanning", None)
    if filter_name not in filter_types:
        raise ValueError("Unknown filter: %s" % filter_name)

    if filter_name is None or filter_name.lower() == "none":
        return x

    if x.ndim == 2:
        n_y, n_x = x.shape
    else:
        n_y, n_x, _ = x.shape

    f_y, f_x = np.meshgrid(
        np.linspace(-1, 1, n_y), np.linspace(-1, 1, n_x), indexing="ij"
    )

    f = np.sqrt(f_x**2 + f_y**2)
    fourier_filter = np.real(f)  # ram-lak filter
    if filter_name == "ram-lak":
        pass
    elif filter_name == "shepp-logan":
        omega = np.pi / 2 * f
        fourier_filter[omega != 0] *= (
            np.sin(omega[omega != 0]) / omega[omega != 0]
        )  # Start from first element to avoid divide by zero
    elif filter_name == "cosine":
        omega = np.pi / 2 * f
        cosine_filter = np.cos(omega)
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        hamming_window = np.abs(np.hamming(n_x))
        hamming_window_2D = np.sqrt(np.outer(hamming_window, hamming_window))
        # hamming_ = scipy.fft.fftshift(fft2(hamming_window_2D,axes=(0,1)),axes=(0,1))
        fourier_filter *= np.abs(hamming_window_2D)
    elif filter_name == "hanning":
        hanning_window = np.abs(np.hanning(n_x))
        hanning_window_2D = np.sqrt(np.outer(hanning_window, hanning_window))
        # hanning_ = scipy.fft.fftshift(fft2(hanning_window_2D,axes=(0,1)),axes=(0,1))
        fourier_filter *= np.abs(hanning_window_2D)
    elif filter_name is None:
        fourier_filter[:] = 1

    x_FT = scipy.fft.fftshift(fft2(x, axes=(0, 1)), axes=(0, 1))
    if x.ndim == 2:
        x_filtered = ifft2(
            scipy.fft.ifftshift(np.multiply(x_FT, fourier_filter), axes=(0, 1)),
            axes=(0, 1),
        )
    else:
        x_filtered = ifft2(
            scipy.fft.ifftshift(
                np.multiply(x_FT, fourier_filter[:, :, np.newaxis]), axes=(0, 1)
            ),
            axes=(0, 1),
        )  # [:,:,np.newaxis]

    return x_filtered.astype(float)


def filterTargetBCLP(real_space_array: np.ndarray, filter_name: str):
    """
    Parameters
    ----------
    real_space_array : np.ndarray

    filter_name : str
        type of filter to apply to target, options: "ram-lak", "shepp-logan", "cosine", "hamming", "hanning", "none"

    Returns
    -------
    x_filtered : np.ndarray
        direct output of filtering in frequency space
    """
    filter_types = ("ram-lak", "shepp-logan", "cosine", "hamming", "hanning", None)
    if filter_name not in filter_types:
        raise ValueError("Unknown filter: %s" % filter_name)

    if filter_name is None or filter_name.lower() == "none":
        return real_space_array

    n_x, n_y = real_space_array.shape[0], real_space_array.shape[1]

    assert (
        n_x == n_y
    ), "The first two dimensions of the real space array must have the same number of elements."

    f_x, f_y = np.meshgrid(
        scipy.fft.fftfreq(n_x), scipy.fft.fftfreq(n_y), indexing="ij"
    )
    f_radial = np.sqrt(f_x**2 + f_y**2)
    fourier_filter = scipy.fft.fftshift(f_radial)  # center-aligned after this line

    # Modify in center-algined view
    if filter_name == "ram-lak":
        pass
    elif filter_name == "shepp-logan":
        omega = np.pi / 2 * f_radial
        fourier_filter[omega != 0] *= (
            np.sin(omega[omega != 0]) / omega[omega != 0]
        )  # Start from first element to avoid divide by zero
    elif filter_name == "cosine":
        omega = np.pi / 2 * f_radial
        cosine_filter = np.cos(omega)
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        hamming_window = np.abs(np.hamming(n_x))
        hamming_window_2D = np.sqrt(np.outer(hamming_window, hamming_window))
        # hamming_ = scipy.fft.fftshift(fft2(hamming_window_2D,axes=(0,1)),axes=(0,1))
        fourier_filter *= np.abs(hamming_window_2D)
    elif filter_name == "hanning":
        hanning_window = np.abs(np.hanning(n_x))
        hanning_window_2D = np.sqrt(np.outer(hanning_window, hanning_window))
        # hanning_ = scipy.fft.fftshift(fft2(hanning_window_2D,axes=(0,1)),axes=(0,1))
        fourier_filter *= np.abs(hanning_window_2D)
    elif filter_name is None:
        fourier_filter[:] = 1

    # TODO:To be simplified
    real_space_array_FT = scipy.fft.fftshift(
        fft2(real_space_array, axes=(0, 1)), axes=(0, 1)
    )
    if (
        real_space_array.ndim == 2
    ):  # This case is being depractated but temporarily retained for backward compatibility
        real_space_array_filtered = ifft2(
            scipy.fft.ifftshift(
                np.multiply(real_space_array_FT, fourier_filter), axes=(0, 1)
            ),
            axes=(0, 1),
        )
    else:
        real_space_array_filtered = ifft2(
            scipy.fft.ifftshift(
                np.multiply(real_space_array_FT, fourier_filter[:, :, np.newaxis]),
                axes=(0, 1),
            ),
            axes=(0, 1),
        )  # [:,:,np.newaxis]

    return real_space_array_filtered.astype(float)


def filterTarget(real_space_array: np.ndarray, filter_name: str):
    """
    Parameters
    ----------
    real_space_array : np.ndarray

    filter_name : str
        type of filter to apply to target, options: "ram-lak", "shepp-logan", "cosine", "hamming", "hanning", "none"

    Returns
    -------
    x_filtered : np.ndarray
        direct output of filtering in frequency space
    """
    # Preprocess input array
    n_x, n_y = real_space_array.shape[0], real_space_array.shape[1]
    assert (
        n_x == n_y
    ), "The first two dimensions of the real space array must have the same number of elements."
    n_x_padded = max(
        64, int(2 ** np.ceil(np.log2(n_x)))
    )  # Find the next size being power of 2

    # Pad the input array to a xy size being power of 2
    pad_before = int(np.ceil((n_x_padded - n_x) / 2))
    pad_after = int(np.floor((n_x_padded - n_x) / 2))
    if real_space_array.ndim == 2:
        pad_width = (
            (pad_before, pad_after),
            (pad_before, pad_after),
        )  # ndim = 2 case is retained for backward compatibility
    else:
        pad_width = (  # type: ignore
            (pad_before, pad_after),
            (pad_before, pad_after),
            (0, 0),
        )  # ((before_1, after_1), ... (before_N, after_N)) unique pad widths for each axis.
    real_space_array_padded = np.pad(
        real_space_array, pad_width, mode="constant", constant_values=0
    )

    # Construct 2D filter from 1D
    fourier_filter_1D = np.squeeze(
        _get_fourier_filter(n_x_padded, filter_name)
    )  # Get 1D filter
    fourier_filter_1D_freq = scipy.fft.fftfreq(n_x_padded)

    fourier_filter_1D_sorted = fourier_filter_1D[
        fourier_filter_1D_freq.argsort()
    ]  # sort for interpolation later
    fourier_filter_1D_freq_sorted = np.sort(fourier_filter_1D_freq)

    # Frequency in 2D
    f_x, f_y = np.meshgrid(
        scipy.fft.fftfreq(n_x_padded), scipy.fft.fftfreq(n_x_padded), indexing="ij"
    )
    f_radial = np.sqrt(f_x**2 + f_y**2)
    fourier_filter_2D = np.interp(
        f_radial,
        fourier_filter_1D_freq_sorted,
        fourier_filter_1D_sorted,
        left=0,
        right=0,
    )  # interpolation the filter on 2D. Result is still corner-algined
    if (
        real_space_array.ndim > 2
    ):  # This should always be the case. ndim=2 is being depractated but temporarily retained for backward compatibility
        fourier_filter_2D = fourier_filter_2D[:, :, np.newaxis]

    real_space_array_FT = fft2(real_space_array_padded, axes=(0, 1))
    real_space_array_filtered = ifft2(
        np.multiply(real_space_array_FT, fourier_filter_2D), axes=(0, 1)
    ).astype(float)

    if (
        pad_after == 0
    ):  # Syntax prevented us to output full array with array[0:-0, 0:-0] which instead gives empty array
        return real_space_array_filtered[
            pad_before:, pad_before:
        ]  # output to the end of array in each dimension
    else:
        return real_space_array_filtered[
            pad_before:-pad_after, pad_before:-pad_after
        ]  # np array indexing exclude the ending index, so the element exactly at -pad_after is excluded


"""
Filtering functions are derived from scikit-image radon_transform.py
https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/radon_transform.py

"""

type FilterType = Literal[
    "ram-lak",
    "shepp-logan",
    "cosine",
    "hamming",
    "hanning",
    "ram-lak_freq",
    "ram-lak_scikit",
]


def filterSinogram(sinogram: np.ndarray, filter_name: FilterType | None):
    """
    Filters a set of sinogram for a 2D or 3D target

    Parameters
    ----------
    sinogram : np.ndarray
        input sinogram
    filter_name : str
        type of filter to apply to sinogram, options: "ram-lak", "shepp-logan", "cosine", "hamming", "hanning", "ram-lak_freq", "none"
    Returns
    -------
    sinogram_filt : np.ndarray
        filtered sinogram
    """
    filter_types = (
        "ram-lak",
        "shepp-logan",
        "cosine",
        "hamming",
        "hanning",
        "ram-lak_freq",
        "ram-lak_scikit",
        None,
    )
    if filter_name is None or filter_name.lower() == "none":
        return sinogram

    if filter_name.lower() not in filter_types:
        raise ValueError("Unknown filter: %s" % filter_name)

    if sinogram.ndim == 2:
        pX, pY = sinogram.shape
        sinogram_filt = np.zeros([pX, pY], dtype=float)

        # Resize image to next power of two (but no less than 64) for
        # Fourier analysis; speeds up Fourier and lessens artifacts
        projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * pX))))
        pad_width = ((0, projection_size_padded - pX), (0, 0))
        fourier_filter = _get_fourier_filter(projection_size_padded, filter_name)

        img = np.pad(sinogram, pad_width, mode="constant", constant_values=0)
        # Apply filter in Fourier domain
        filt_img = fft(img, axis=0) * fourier_filter
        sinogram_filt = np.real(ifft(filt_img, axis=0)[:pX, :])
    elif sinogram.ndim == 3:

        pX, pY, nZ = sinogram.shape
        sinogram_filt = np.zeros([pX, pY, nZ], dtype=float)

        # Resize image to next power of two (but no less than 64) for
        # Fourier analysis; speeds up Fourier and lessens artifacts
        projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * pX))))
        pad_width = ((0, projection_size_padded - pX), (0, 0))
        fourier_filter = _get_fourier_filter(projection_size_padded, filter_name)

        for z_i in range(nZ):
            img = np.pad(
                sinogram[:, :, z_i], pad_width, mode="constant", constant_values=0
            )
            # Apply filter in Fourier domain
            filt_img = fft(img, axis=0) * fourier_filter
            sinogram_filt[:, :, z_i] = np.real(ifft(filt_img, axis=0)[:pX, :])
    else:
        raise Exception(
            "The sinogram provided does not have compatible number of dimension (2 or 3)."
        )

    return sinogram_filt


def _get_fourier_filter(size: int, filter_name: str):
    """Construct the Fourier filter.
    This computation lessens artifacts and removes a small bias as
    explained in [1], Chap 3. Equation 61.

    Parameters
    ----------
    size : int
        filter size. Must be even.
    filter_name : str
        Filter used in frequency domain filtering. Filters available:
        ram-lak, shepp-logan, cosine, hamming, hanning. Assign None to use
        no filter.

    Returns
    -------
    fourier_filter : np.ndarray
        The computed Fourier filter.
    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.
    """
    n = np.concatenate(
        (
            np.arange(1, size / 2 + 1, 2, dtype=int),
            np.arange(size / 2 - 1, 0, -2, dtype=int),
        )
    )
    # n = scipy.fft.fftfreq(size)[1::2]*size #equivalent way to write the odd spatial coordinate (this version is signed)
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    # Computing the ram-lak filter from the fourier transform of its
    # frequency domain representation lessens artifacts and removes a
    # small bias as explained in [1], Chap 3. Equation 61
    fourier_filter = 2 * np.real(fft(f))  # ram-lak filter
    if filter_name == "ram-lak":
        pass
    elif filter_name == "shepp-logan":
        # Start from first element to avoid divide by zero
        omega = np.pi * scipy.fft.fftfreq(size)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter_name == "cosine":
        freq = np.linspace(0, np.pi, size, endpoint=False)
        cosine_filter = scipy.fft.fftshift(np.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        fourier_filter *= scipy.fft.fftshift(np.hamming(size))
    elif filter_name == "hanning":
        fourier_filter *= scipy.fft.fftshift(np.hanning(size))
    elif filter_name == "ram-lak_freq":
        fourier_filter /= (
            np.pi / 4
        ) ** 2  # fudge factor 1/(pi/4)^2 is needed on top of the original filter to reconstruct values close to original. So far this yields the best performance.
        # fourier_filter = np.abs(scipy.fft.fftfreq(size)*4) #Build filter from ground up. Somehow we still need a factor of 4 to reconstruct original values. So far this yields the second best performance.
    elif filter_name == "ram-lak_scikit":
        fourier_filter *= (
            np.pi / 2
        )  # This include the final multiplier the scikit-image iradon has. One potential source of this factor is the definition of DFT matrix in FFT.
        # Ref: https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/radon_transform.py
    elif filter_name is None or filter_name.lower() == "none":
        fourier_filter[:] = 1

    return fourier_filter[:, np.newaxis]


# def histogramEqualization(x:np.ndarray, bit_depth:int,output_dtype:np.dtype = np.float32):
#     # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

#     number_bins = 2**bit_depth
#     bins = np.linspace(0,1,number_bins)
#     # flatten image
#     flat = x.flatten()

#     # get image histogram
#     # x_histogram, bins = np.histogram(flat[np.where((flat != 0))[0]],
#     #                                     bins=bins,
#     #                                     density=True)
#     x_histogram, bins = np.histogram(flat,
#                                         bins=bins,
#                                         density=True)

#     fig, ax = plt.subplots(1,1)
#     ax.plot(bins[:-1],x_histogram)
#     plt.show()
#     # get x histogram
#     # x_histogram, bins = np.histogram(x.flatten(), bins=bins, density=True)
#     cdf = x_histogram.cumsum() # cumulative distribution function
#     cdf = cdf / cdf[-1] # normalize

#     # use linear interpolation of cdf to find new pixel values
#     x_equalized = np.interp(x.flatten(), bins[:-1], cdf)

#     return x_equalized.reshape(x.shape)


def histogramEqualization(
    x: np.ndarray, bit_depth: int, output_dtype: np.typing.DTypeLike = float
):
    max_x = np.amax(x)
    min01 = np.percentile(x, 1)
    # min01 = 0.01*np.amin(proj)
    max99 = np.percentile(x, 99)
    # max99 = 0.99*maxproj
    # fig, axs = plt.subplots(2,1)
    # axs[0].imshow(proj[:,:,20])

    # x_stretch = (proj)*(maxproj/(max99 - min01))
    x_stretch = (x) * (max_x / (max99 - min01))

    # axs[1].imshow(x_stretch[:,:,20])
    # plt.show()
    # x_stretch = np.where(x_stretch>=max99,max99,x_stretch)
    # x_stretch = np.where(x_stretch<=min01,min01,x_stretch)
    x_stretch = np.where(x_stretch >= max99, 1, x_stretch)
    x_stretch = np.where(x_stretch <= min01, 0, x_stretch)

    return x_stretch


def discretize(
    x: np.ndarray,
    bit_depth: int,
    range: list,
    output_dtype: np.typing.DTypeLike = float,
):
    """
    Digitizes a variable to requested bit depth and output data type

    Parameters
    ----------
    x : nd.ndarray
        array to digitize
    bit_depth : int
        bit depth of output, 2^bit_depth number of bins
    range : list
        [min,max] values to discretize within
    output_dtype : np.dtype (optional)
        data type of resulting digitized array

    """

    assert len(range) == 2, "range argument should have 2 elements [min,max]"
    bins = np.linspace(range[0], range[1], 2**bit_depth, endpoint=True)

    discrete_x_inds = np.digitize(x, bins=bins) - 1
    discrete_x = bins[discrete_x_inds].astype(output_dtype)

    return discrete_x
