import numpy as np
from scipy.fft import fft2
import matplotlib.pyplot as plt

try:
    import scipy.fft
    from scipy.fft import next_fast_len
    fftmodule = scipy.fft
except ImportError:
    import numpy.fft
    fftmodule = numpy.fft
    from scipy.fftpack import next_fast_len

if fftmodule is np.fft:
    # fallback from scipy.fft to scipy.fftpack instead of numpy.fft
    # (fftpack preserves single precision while numpy.fft does not)
    from scipy.fftpack import fft, ifft
else:
    fft = fftmodule.fft
    ifft = fftmodule.ifft
    fft2 = fftmodule.fft2
    ifft2 = fftmodule.ifft2


def sigmoid(x,g):
    y = 1/(1+np.exp(-x*(1/g)))
    return y.astype(np.float)


    
def clipToCircle(x : np.ndarray):
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
    circle_y, circle_x = np.meshgrid(np.linspace(-1,1,x.shape[0]),
                                    np.linspace(-1,1,x.shape[1]))
    if x.ndim == 2:
        x[circle_x**2 + circle_y**2 > 1] = 0 
    else:
        circle = np.broadcast_to((circle_x**2 + circle_y**2 > 1)[...,np.newaxis],x.shape)
        x[circle] = 0

    return x


def filterTargetOSMO(x : np.ndarray,filter_name : str):
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
    filter_types = ('ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hanning', None)
    if filter_name not in filter_types:
        raise ValueError("Unknown filter: %s" % filter_name)

    if filter_name is None or filter_name.lower() == "none":
        return x

    if x.ndim == 2:
        n_y,n_x = x.shape
    else:
        n_y,n_x,_ = x.shape


    f_y,f_x = np.meshgrid(np.linspace(-1,1,n_y),np.linspace(-1,1,n_x),indexing='ij')
    
    f = np.sqrt(f_x**2 + f_y**2)
    fourier_filter = np.real(f) # ram-lak filter
    if filter_name == "ram-lak":
        pass
    elif filter_name == "shepp-logan":
        omega = np.pi/2 * f
        fourier_filter[omega != 0] *= np.sin(omega[omega != 0]) / omega[omega != 0] # Start from first element to avoid divide by zero
    elif filter_name == "cosine":
        omega = np.pi/2 * f
        cosine_filter = np.cos(omega)
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        hamming_window = np.abs(np.hamming(n_x))
        hamming_window_2D = np.sqrt(np.outer(hamming_window,hamming_window))
        # hamming_ = fftmodule.fftshift(fft2(hamming_window_2D,axes=(0,1)),axes=(0,1))
        fourier_filter *= np.abs(hamming_window_2D)
    elif filter_name == "hanning":
        hanning_window = np.abs(np.hanning(n_x))
        hanning_window_2D = np.sqrt(np.outer(hanning_window,hanning_window))
        # hanning_ = fftmodule.fftshift(fft2(hanning_window_2D,axes=(0,1)),axes=(0,1))
        fourier_filter *= np.abs(hanning_window_2D)
    elif filter_name is None:
        fourier_filter[:] = 1

    x_FT = fftmodule.fftshift(fft2(x,axes=(0,1)),axes=(0,1))
    if x.ndim == 2:
        x_filtered = ifft2(fftmodule.ifftshift(np.multiply(x_FT,fourier_filter),axes=(0,1)),axes=(0,1))
    else:
        x_filtered = ifft2(fftmodule.ifftshift(np.multiply(x_FT,fourier_filter[:,:,np.newaxis]),axes=(0,1)),axes=(0,1)) #[:,:,np.newaxis]

    return x_filtered.astype(float)



"""
Filtering functions are derived from scikit-image radon_transform.py
https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/radon_transform.py

"""


def filterSinogram(sinogram : np.ndarray,filter_name : str):
    """
    Filters a set of sinogram for a 2D or 3D target

    Parameters
    ----------
    sinogram : np.ndarray
        input sinogram
    filter_name : str
        type of filter to apply to sinogram, options: "ram-lak", "shepp-logan", "cosine", "hamming", "hanning", "none"
    Returns
    -------
    sinogram_filt : np.ndarray
        filtered sinogram
    """
    filter_types = ('ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hanning', 'none',None)
    if filter_name.lower() not in filter_types:
        raise ValueError("Unknown filter: %s" % filter_name)

    if filter_name is None or filter_name.lower() == "none":
        return sinogram

    if sinogram.ndim == 2:
        pX, pY,  = sinogram.shape
        sinogram_filt = np.zeros([pX,pY],dtype=float)


        # Resize image to next power of two (but no less than 64) for
        # Fourier analysis; speeds up Fourier and lessens artifacts
        projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * pX))))
        pad_width = ((0, projection_size_padded - pX), (0, 0))
        fourier_filter = _get_fourier_filter(projection_size_padded, filter_name)

        img = np.pad(sinogram, pad_width, mode='constant', constant_values=0)
        # Apply filter in Fourier domain
        filt_img = fft(img, axis=0) * fourier_filter
        sinogram_filt = np.real(ifft(filt_img, axis=0)[:pX, :])
    else:

        pX, pY, nZ = sinogram.shape
        sinogram_filt = np.zeros([pX,pY,nZ],dtype=float)


        # Resize image to next power of two (but no less than 64) for
        # Fourier analysis; speeds up Fourier and lessens artifacts
        projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * pX))))
        pad_width = ((0, projection_size_padded - pX), (0, 0))
        fourier_filter = _get_fourier_filter(projection_size_padded, filter_name)

        for z_i in range(nZ):
            img = np.pad(sinogram[:,:,z_i], pad_width, mode='constant', constant_values=0)
            # Apply filter in Fourier domain
            filt_img = fft(img, axis=0) * fourier_filter
            sinogram_filt[:,:,z_i] = np.real(ifft(filt_img, axis=0)[:pX, :])


    return sinogram_filt


def _get_fourier_filter(size : int, filter_name : str):
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
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=np.int),np.arange(size / 2 - 1, 0, -2, dtype=np.int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    # Computing the ram-lak filter from the fourier transform of its
    # frequency domain representation lessens artifacts and removes a
    # small bias as explained in [1], Chap 3. Equation 61
    fourier_filter = 2 * np.real(fft(f))         # ram-lak filter
    if filter_name == "ram-lak":
        pass
    elif filter_name == "shepp-logan":
        # Start from first element to avoid divide by zero
        omega = np.pi * fftmodule.fftfreq(size)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter_name == "cosine":
        freq = np.linspace(0, np.pi, size, endpoint=False)
        cosine_filter = fftmodule.fftshift(np.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        fourier_filter *= fftmodule.fftshift(np.hamming(size))
    elif filter_name == "hanning":
        fourier_filter *= fftmodule.fftshift(np.hanning(size))
    elif filter_name is None or filter_name.lower() is "none":
        fourier_filter[:] = 1

    return fourier_filter[:, np.newaxis]



# def histogramEqualization(x:np.ndarray, bit_depth:int,output_dtype:np.dtype = np.float):
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


def histogramEqualization(x:np.ndarray, bit_depth:int,output_dtype:np.dtype = np.float):
    max_x = np.amax(x)
    min01 = np.percentile(x,1)
    # min01 = 0.01*np.amin(proj)
    max99 = np.percentile(x,99)
    # max99 = 0.99*maxproj
    # fig, axs = plt.subplots(2,1)
    # axs[0].imshow(proj[:,:,20])


    # x_stretch = (proj)*(maxproj/(max99 - min01))
    x_stretch = (x)*(max_x/(max99 - min01))

    # axs[1].imshow(x_stretch[:,:,20])
    # plt.show()
    # x_stretch = np.where(x_stretch>=max99,max99,x_stretch)
    # x_stretch = np.where(x_stretch<=min01,min01,x_stretch)
    x_stretch = np.where(x_stretch>=max99,1,x_stretch)
    x_stretch = np.where(x_stretch<=min01,0,x_stretch)        

    return x_stretch


def discretize(x:np.ndarray,bit_depth:int,range:list,output_dtype:np.dtype=np.float):
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
    bins = np.linspace(range[0],range[1],2**bit_depth)
    
    discrete_x_inds = np.digitize(x,bins=bins) - 1
    discrete_x = bins[discrete_x_inds].astype(output_dtype)

    return discrete_x