import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit
import scipy.stats

import numpy as np
plt.style.use(r'vamtoolbox\scripts\style.mplstyle')
VOXELS_TO_MM = 1/0.0461 #voxels/mm


def gaussian(x, mu, sigma, a):
    """
    Defines the Gaussian function.

    Args:
      x: The input values.
      mu: The mean of the Gaussian distribution.
      sigma: The standard deviation of the Gaussian distribution.
      a: amplitude

    Returns:
      The Gaussian function evaluated at x.
    """
    return a * scipy.stats.norm.pdf(x,loc=mu,scale=sigma)
    # return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def fit_gaussian(x_data,y_data):
    """
    Fits a Gaussian distribution to the given data.

    Args:
      data: The input data.

    Returns:
      A tuple containing:
        - mu: The estimated mean of the Gaussian distribution.
        - sigma: The estimated standard deviation of the Gaussian distribution.
        - two_sigma: The value of 2*sigma.
    """
    try:
        # Initial guess for the mean and standard deviation
        mu_guess = 0
        sigma_guess = 1
        a_guess = np.amax(y_data)

        # Fit the Gaussian function to the data
        popt, _ = curve_fit(gaussian, x_data, y_data, p0=[mu_guess, sigma_guess, a_guess])
        mu, sigma, a = popt

        residuals = y_data - gaussian(x_data,*popt)

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data-np.mean(y_data))**2)

        r_squared = 1 - (ss_res / ss_tot) 

        # Calculate 2*sigma
        two_sigma = 2 * sigma

        return mu, sigma, two_sigma, r_squared

    except RuntimeError:
        print("Gaussian fit failed. Check data and initial guesses.")
        return None, None, None


class HistogramData():

    def __init__(self,filepath):
        self.filepath = filepath
        self._loaded_data = np.loadtxt(filepath,skiprows=1,usecols=[0,1,2,3],delimiter=';')
        self.hist_class = self._loaded_data[:,0]
        self.hist_values = self._loaded_data[:,1]
        self.hist_bin_start = self._loaded_data[:,2]/VOXELS_TO_MM
        self.hist_bin_end = self._loaded_data[:,3]/VOXELS_TO_MM

        self.hist_bin_centers = (self.hist_bin_start + self.hist_bin_end)/2

    def plot(self,fig=None,ax=None,bound=1,plottype='histogram',label=None):
        if ax == None:
            fig, ax = plt.subplots(1,1,figsize=(2.5,2))


        # Create a custom colormap
        colors = [(0, 'blue'), (0.5, 'white'), (1, 'red')]  # Blue to white to red
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

        # Normalize data for colormapping
        if bound != None:
            normalized_data = (self.hist_bin_centers - (-bound)) / (2*bound)
        else:
            normalized_data = (self.hist_bin_centers - self.hist_bin_centers.min()) / (self.hist_bin_centers.max() - self.hist_bin_centers.min())
        normalized_data = np.clip(normalized_data,0,1)
        
        # Create the plot
        if plottype == 'histogram':
            _, bins, patches = ax.hist(self.hist_bin_centers,weights=self.hist_values,bins=self.hist_bin_centers.size)

            # Color each patch according to the normalized bin center
            for i, patch in enumerate(patches):
                patch.set_facecolor(cmap(normalized_data[i]))
            ax.hist(self.hist_bin_centers,weights=self.hist_values,bins=self.hist_bin_centers.size,histtype='step',color='black')
            ax.set_xticks([-bound,-bound/2,0,bound/2,bound])
        elif plottype == 'line':
            ax.plot(self.hist_bin_centers, self.hist_values,label=label) 

        # Fill the area under the curve with color
        ax.set_xlim([-bound,bound])
        
        fig.tight_layout()

    def fitGaussian(self,):
        return fit_gaussian(self.hist_bin_centers,self.hist_values)

datas = []

gyroid_sheet = HistogramData(r'vamtoolbox\data\CC\Gyroidsheet\Histogram.csv')
gyroid_sheet.plot(bound=1)
datas.append(gyroid_sheet)
plt.savefig(r"vamtoolbox\data\CC\sheet_gyroid_histogram.png",dpi=500)

gyroid_skel = HistogramData(r'vamtoolbox\data\CC\Gyroidskel\Histogram.csv')
gyroid_skel.plot(bound=1)
datas.append(gyroid_skel)
plt.savefig(r"vamtoolbox\data\CC\skeletal_gyroid_histogram.png",dpi=500)

schwarzp = HistogramData(r'vamtoolbox\data\CC\Schwarzp\Histogram.csv')
schwarzp.plot(bound=1)
datas.append(schwarzp)
plt.savefig(r"vamtoolbox\data\CC\sheet_schwarz_p_histogram.png",dpi=500)

diamond_skel = HistogramData(r'vamtoolbox\data\CC\Diamondskel\Histogram.csv')
diamond_skel.plot(bound=1)
datas.append(diamond_skel)
plt.savefig(r"vamtoolbox\data\CC\skeletal_diamond_histogram.png",dpi=500)

octettruss = HistogramData(r'vamtoolbox\data\CC\Octettruss\Histogram.csv')
octettruss.plot(bound=1)
datas.append(octettruss)
plt.savefig(r"vamtoolbox\data\CC\octet_truss_histogram.png",dpi=500)


# Save results to a file
save_txt_file = r"vamtoolbox\data\CC\statistics.txt"
with open(save_txt_file, "w") as f:

    for data in datas:
        mu, sigma, two_sigma, r_squared = data.fitGaussian()
        f.write(f"{data.filepath}\n")
        f.write(f"Mean (mu) [mm]: {mu:.4f}\n")
        f.write(f"Standard Deviation (sigma) [mm]: {sigma:.4f}\n")
        f.write(f"2*sigma [mm]: {two_sigma:.4f}\n")
        f.write(f"R^2: {r_squared:.4f}\n\n\n")




fig, ax = plt.subplots(1,1)
gyroid_sheet.plot(fig=fig,ax=ax,bound=1,plottype='line',label='sheet gyroid')
gyroid_skel.plot(fig=fig,ax=ax,bound=1,plottype='line',label='skeletal gyroid')
schwarzp.plot(fig=fig,ax=ax,bound=1,plottype='line',label='sheet Schwarz P')
diamond_skel.plot(fig=fig,ax=ax,bound=1,plottype='line',label='skeletal diamond')
octettruss.plot(fig=fig,ax=ax,bound=1,plottype='line',label='octet truss')
ax.legend()
ax.set_xlabel('Signed distance [mm]')
ax.set_ylabel('Counts')
fig.tight_layout()
plt.savefig(r"vamtoolbox\data\CC\combined_line_plot.png",dpi=300)
plt.show()


