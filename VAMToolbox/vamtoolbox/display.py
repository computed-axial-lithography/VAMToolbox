import time

import matplotlib
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import vedo  # type: ignore
from matplotlib import axes, cm, colors
from scipy.ndimage import rotate

import vamtoolbox

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]


class CursorFormatter:
    def __init__(self, im, slice_index=None):
        self.im = im
        self.slice_index = slice_index

    def cursorString(self, x, y, body_str):
        if self.slice_index is not None:
            if self.slice_index == 2:
                cursor_string = "x=%d, y=%d, body=%s" % (y, x, body_str)
            elif self.slice_index == 1:
                cursor_string = "z=%d, y=%d, body=%s" % (y, x, body_str)
        else:
            cursor_string = "x=%d, y=%d, body=%s" % (y, x, body_str)
        return cursor_string

    def __call__(self, x, y):
        x = int(x + 0.5)
        y = int(y + 0.5)

        val = self.im.get_array()[y, x]

        if np.all(val[0:2] == val[0]):
            if val[0] == 0.0:
                body_str = "void"
            else:
                # if all values are the same then it is part of the target
                body_str = "target"
        elif val[0] == 1.0:
            # if red
            body_str = "insert"
        elif val[1] == 1.0:
            # if green
            body_str = "zero dose"

        cursor_string = self.cursorString(x, y, body_str)

        return cursor_string


class IndexTracker:
    def __init__(
        self,
        array,
        vol_type,
        slice_index,
        label,
        slice_step,
        ax=None,
        fig=None,
        **kwargs
    ):

        if isinstance(array, np.ndarray):
            self.array = array.astype(np.float32)
            v_min, v_max = np.min(array), np.max(array)
        elif isinstance(array, vamtoolbox.geometry.TargetGeometry):
            self.array = array.array.astype(np.float32)
            v_min, v_max = np.min(array.array), np.max(array.array)

        self.vol_type = vol_type
        self.slice_index = slice_index
        if self.slice_index == 1:
            self.array = np.swapaxes(self.array, 0, 2)

        self.ax = ax

        self.slice_step = slice_step
        kwargs["cmap"] = "CMRmap" if "cmap" not in kwargs else kwargs["cmap"]
        kwargs["interpolation"] = (
            "antialiased" if "interpolation" not in kwargs else kwargs["interpolation"]
        )

        # if title is not None:
        # 	ax.set_title('%s\nUse scroll wheel to navigate volume' %title)
        shape = self.array.shape
        self.n0, self.n1, self.n2 = shape[0], shape[1], shape[2]
        if slice_index == 0:
            self.ind = self.n0 // 2
            self.im = self.ax.imshow(
                self.array[self.ind, :, :], vmin=0, vmax=v_max, **kwargs
            )
        elif slice_index == 1:
            self.ind = self.n1 // 2
            self.im = self.ax.imshow(
                self.array[:, self.ind, :], vmin=0, vmax=v_max, **kwargs
            )
        elif slice_index == 2:
            self.ind = self.n2 // 2
            self.im = self.ax.imshow(
                self.array[:, :, self.ind], vmin=0, vmax=v_max, **kwargs
            )
        vamtoolbox.util.matplotlib.addColorbar(self.im)
        self.ax.set_xlabel(label[0])
        self.ax.set_ylabel(label[1])
        self.ax.xaxis.set_ticks_position("top")
        self.ax.xaxis.set_label_position("top")
        self.update()
        fig.canvas.mpl_connect("scroll_event", self.onscroll)

    def onscroll(self, event):

        # print("%s %s" % (event.button, event.step))
        if event.inaxes in [self.ax]:
            if self.slice_index == 0:
                num_slices = self.n0
            elif self.slice_index == 1:
                num_slices = self.n1
            elif self.slice_index == 2:
                num_slices = self.n2

            if event.button == "up":
                self.ind = (self.ind + self.slice_step) % num_slices
            else:
                self.ind = (self.ind - self.slice_step) % num_slices
            self.update()

    def update(self):
        if self.slice_index == 0:
            self.im.set_data(self.array[self.ind, :, :])
        elif self.slice_index == 1:
            self.im.set_data(self.array[:, self.ind, :])
        elif self.slice_index == 2:
            self.im.set_data(self.array[:, :, self.ind])

        if self.slice_index == 1 and (
            self.vol_type == "recon" or self.vol_type == "target"
        ):
            self.ax.set_title(r"X$_\mathrm{i}$: %s" % (self.ind))
        elif self.slice_index == 1 and self.vol_type == "sino":
            self.ax.set_title(r"Angle$_\mathrm{i}$: %s" % (self.ind))
        elif self.slice_index == 2:
            self.ax.set_title(r"Z$_\mathrm{i}$: %s" % (self.ind))

        self.im.axes.figure.canvas.draw()


class BodiesIndexTracker:
    def __init__(
        self,
        array,
        vol_type,
        slice_index,
        label,
        slice_step,
        ax=None,
        fig=None,
        **kwargs
    ):

        if isinstance(array, np.ndarray):
            self.array = array.astype(np.float32)
            v_min, v_max = np.min(array), np.max(array)
        elif isinstance(array, vamtoolbox.geometry.TargetGeometry):
            self.array = array.array.astype(np.float32)
            v_min, v_max = np.min(array.array), np.max(array.array)

        self.array = np.stack(
            (self.array, self.array, self.array, np.ones_like(self.array)), axis=-1
        )

        if array.insert is not None:
            color_insert = [1, 0, 0, 1]  # red
            self.array[array.insert.astype(np.bool_), :] = color_insert
        if array.zero_dose is not None:
            color_zero_dose = [0, 1, 0, 1]  # green
            self.array[array.zero_dose.astype(np.bool_), :] = color_zero_dose

        self.vol_type = vol_type
        self.slice_index = slice_index
        if self.slice_index == 1:
            self.array = np.swapaxes(self.array, 0, 2)

        self.ax = ax

        self.slice_step = slice_step
        kwargs["cmap"] = "CMRmap" if "cmap" not in kwargs else kwargs["cmap"]
        kwargs["interpolation"] = (
            "antialiased" if "interpolation" not in kwargs else kwargs["interpolation"]
        )

        # if title is not None:
        # 	ax.set_title('%s\nUse scroll wheel to navigate volume' %title)
        shape = self.array.shape
        self.n0, self.n1, self.n2 = shape[0], shape[1], shape[2]
        if slice_index == 0:
            self.ind = self.n0 // 2
            self.im = self.ax.imshow(
                self.array[self.ind, :, :], vmin=0, vmax=v_max, **kwargs
            )
        elif slice_index == 1:
            self.ind = self.n1 // 2
            self.im = self.ax.imshow(
                self.array[:, self.ind, :], vmin=0, vmax=v_max, **kwargs
            )
        elif slice_index == 2:
            self.ind = self.n2 // 2
            self.im = self.ax.imshow(
                self.array[:, :, self.ind], vmin=0, vmax=v_max, **kwargs
            )
        vamtoolbox.util.matplotlib.addColorbar(self.im)
        self.ax.format_coord = CursorFormatter(self.im, slice_index)
        self.ax.set_xlabel(label[0])
        self.ax.set_ylabel(label[1])
        self.ax.xaxis.set_ticks_position("top")
        self.ax.xaxis.set_label_position("top")
        self.update()
        fig.canvas.mpl_connect("scroll_event", self.onscroll)

    def onscroll(self, event):

        # print("%s %s" % (event.button, event.step))
        if event.inaxes in [self.ax]:
            if self.slice_index == 0:
                num_slices = self.n0
            elif self.slice_index == 1:
                num_slices = self.n1
            elif self.slice_index == 2:
                num_slices = self.n2

            if event.button == "up":
                self.ind = (self.ind + self.slice_step) % num_slices
            else:
                self.ind = (self.ind - self.slice_step) % num_slices
            self.update()

    def update(self):
        if self.slice_index == 0:
            self.im.set_data(self.array[self.ind, :, :])
        elif self.slice_index == 1:
            self.im.set_data(self.array[:, self.ind, :])
        elif self.slice_index == 2:
            self.im.set_data(self.array[:, :, self.ind])

        if self.slice_index == 1 and (
            self.vol_type == "recon" or self.vol_type == "target"
        ):
            self.ax.set_title(r"X$_\mathrm{i}$: %s" % (self.ind))
        elif self.slice_index == 1 and self.vol_type == "sino":
            self.ax.set_title(r"Angle$_\mathrm{i}$: %s" % (self.ind))
        elif self.slice_index == 2:
            self.ax.set_title(r"Z$_\mathrm{i}$: %s" % (self.ind))

        self.im.axes.figure.canvas.draw()


class VolumeSlicer:
    def __init__(
        self,
        array,
        vol_type,
        slice_step=1,
        show_bodies=False,
        ax=None,
        fig=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        vol : vamtoolbox.geometry.Volume
                the data object to plotted

        slice_step : int, optional
                array slice increment on a mouse wheel scroll event

        ax : matplotlib.axes.Axes, optional

        fig : matplotlib.figure.Figure, optional
        """
        if ax is None and fig is None:
            fig, axs = plt.subplots(1, 2, figsize=(8, 5))
        self.axs = axs
        self.fig = fig
        self.array = array
        self.vol_type = vol_type
        self.slice_inds = [2, 1]

        if self.vol_type == "recon" or self.vol_type == "target":
            labels = [["Y [voxel]", "X [voxel]"], ["Y [voxel]", "Z [voxel]"]]
        elif self.vol_type == "sino":
            labels = [
                [r"$\mathrm{\theta}^\circ$", "R [voxel]"],
                ["R [voxel]", "Z [voxel]"],
            ]

        self.scroll_trackers = list()
        for ax, slice_ind, label in zip(self.axs, self.slice_inds, labels):
            if show_bodies == True:
                self.scroll_trackers.append(
                    BodiesIndexTracker(
                        self.array,
                        vol_type=self.vol_type,
                        slice_index=slice_ind,
                        label=label,
                        slice_step=slice_step,
                        ax=ax,
                        fig=self.fig,
                        **kwargs
                    )
                )
            else:
                self.scroll_trackers.append(
                    IndexTracker(
                        self.array,
                        vol_type=self.vol_type,
                        slice_index=slice_ind,
                        label=label,
                        slice_step=slice_step,
                        ax=ax,
                        fig=self.fig,
                        **kwargs
                    )
                )

        plt.tight_layout()


def showVolumeSlicer(array, vol_type, slice_step=1, **kwargs):

    x = VolumeSlicer(array, vol_type, slice_step=slice_step, **kwargs)
    plt.show()
    return x


def showSinoSlicer(array, **kwargs):
    x = VolumeSlicer(array, vol_type="sino", **kwargs)
    plt.show()
    return x


class SlicePlot:
    def __init__(self, array, vol_type, show_bodies=False, ax=None, fig=None, **kwargs):
        if ax is None and fig is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        self.ax = ax
        self.fig = fig

        if isinstance(array, np.ndarray):
            self.array = array.astype(np.float32)
            v_min, v_max = np.min(array), np.max(array)
        elif isinstance(array, vamtoolbox.geometry.TargetGeometry):
            self.array = array.array.astype(np.float32)
            v_min, v_max = np.min(array.array), np.max(array.array)

        if show_bodies == True:
            self.array = np.stack(
                (self.array, self.array, self.array, np.ones_like(self.array)), axis=-1
            )

            if array.insert is not None:
                color_insert = [1, 0, 0, 1]  # red
                self.array[array.insert.astype(np.bool_), :] = color_insert
            if array.zero_dose is not None:
                color_zero_dose = [0, 1, 0, 1]  # green
                self.array[array.zero_dose.astype(np.bool_), :] = color_zero_dose

        if self.array.ndim == 2:
            self.im = self.ax.imshow(self.array, vmin=0, vmax=1, **kwargs)
            # self.im = self.ax.imshow(self.array,**kwargs)

        else:
            if show_bodies == True:
                self.im = self.ax.imshow(
                    self.array[:, :, int(array.shape[-1] / 2), :],
                    vmin=0,
                    vmax=1,
                    **kwargs
                )
            else:
                self.im = self.ax.imshow(
                    self.array[:, :, int(array.shape[-1] / 2)], vmin=0, vmax=1, **kwargs
                )

        if vol_type == "recon" or vol_type == "target":
            xlabel = "Y [voxel]"
            ylabel = "X [voxel]"
        elif vol_type == "sino":
            xlabel = r"$\mathrm{\theta}^\circ$"
            ylabel = "R [voxel]"
        if show_bodies == True:
            self.ax.format_coord = CursorFormatter(self.im)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.xaxis.set_ticks_position("top")
        self.ax.xaxis.set_label_position("top")

        vamtoolbox.util.matplotlib.addColorbar(self.im)

        # self.ax.axis('off')
        plt.tight_layout()

    def update(self, vol):
        if vol.ndim == 2:
            self.im.set_data(vol)
        else:
            self.im.set_data(vol[..., int(vol.shape[-1] / 2)])


class ErrorPlot:
    def __init__(self, *args, ax=None, fig=None):
        if ax is None and fig is None:
            fig, ax = plt.subplots(1, 1)
        self.ax = ax
        self.fig = fig
        colors = ["red", "blue", "lime", "darkviolet", "cyan", "olive"]
        for ii, error in enumerate(args):

            n_iter = np.size(error)

            (self.line,) = self.ax.plot(range(n_iter), error)
            self.line.set(linestyle="-", color=colors[ii])

        self.ax.set(
            title="Error",
            xlabel="Iterations",
            ylabel="VER",
            xlim=[0, n_iter - 1],
            xticks=np.arange(0, n_iter, step=np.max([int(n_iter / 10), 1])),
            xticklabels=np.arange(0, n_iter, step=np.max([int(n_iter / 10), 1])),
        )
        self.ax.grid()
        # self.ax.set_ylim(bottom=0)

    def update(self, error):
        self.ax.set_yscale("log")
        self.line.set_ydata(error)
        self.ax.relim()
        self.ax.autoscale(axis="y")
        # self.ax.set_ylim(bottom=0)


def showErrorPlot(*args, **kwargs):
    x = ErrorPlot(*args, **kwargs)
    return x


class HistogramPlot:
    def __init__(self, x, target, scale="linear", ax=None, fig=None):
        if ax is None and fig is None:
            fig, ax = plt.subplots(1, 1)
        self.ax = ax
        self.fig = fig
        self.scale = scale
        self.hist_bins = np.linspace(0, 1, 100)
        self.target = target
        void_dose = x[self.target.void_inds]
        gel_dose = x[self.target.gel_inds]

        self.ax.hist(
            void_dose,
            bins=self.hist_bins,
            color="red",
            alpha=0.5,
            edgecolor="black",
            label="out-of-part dose",
        )
        self.ax.hist(
            gel_dose,
            bins=self.hist_bins,
            color="blue",
            alpha=0.5,
            edgecolor="black",
            label="in-part dose",
        )
        self.ax.set(
            title="Dose distribution", xlabel="Normalized Dose", ylabel="Frequency"
        )
        self.ax.legend()

        if self.scale == "log":
            self.ax.set_yscale("log")

        plt.tight_layout()

    def update(self, x):

        void_dose = x[self.target.void_inds]
        gel_dose = x[self.target.gel_inds]

        self.ax.clear()
        self.ax.hist(
            void_dose,
            bins=self.hist_bins,
            color="red",
            alpha=0.5,
            edgecolor="black",
            label="out-of-part dose",
        )
        self.ax.hist(
            gel_dose,
            bins=self.hist_bins,
            color="blue",
            alpha=0.5,
            edgecolor="black",
            label="in-part dose",
        )
        self.ax.legend()
        self.ax.set(
            title="Dose distribution", xlabel="Normalized Dose", ylabel="Frequency"
        )
        if self.scale == "log":
            self.ax.set_yscale("log")

    def save(self, savepath, **kwargs):
        """
        Save the current plot

        Parameters
        ----------
        savepath : str

        **kwargs
                kwargs from matplotlib.pyplot.savefig
        """
        dpi = 300 if "dpi" not in kwargs else kwargs["dpi"]
        plt.savefig(savepath, **kwargs)


def showHistogramPlot(*args, **kwargs):
    x = HistogramPlot(*args, **kwargs)
    return x


class EvolvingPlot:
    def __init__(self, target_geo, n_iter):
        plt.ion()
        self.target = target_geo.array
        self.n_dim = target_geo.n_dim
        self.n_iter = n_iter

        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 10))
        self.target_plot = SlicePlot(
            self.target,
            vol_type="target",
            ax=self.axs[0, 0],
            fig=self.fig,
            cmap="gray",
            interpolation="none",
        )
        self.error_plot = ErrorPlot(
            np.full(self.n_iter, np.nan), ax=self.axs[1, 0], fig=self.fig
        )
        self.recon_plot = SlicePlot(
            np.zeros_like(self.target),
            vol_type="recon",
            ax=self.axs[0, 1],
            fig=self.fig,
            cmap="CMRmap",
        )
        self.hist_plot = HistogramPlot(
            np.zeros_like(self.target),
            target_geo,
            ax=self.axs[1, 1],
            fig=self.fig,
            scale="log",
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.draw()
        plt.show()
        plt.pause(0.01)

    def update(self, error, x=None):
        self.error_plot.update(error)
        self.recon_plot.update(x)
        self.hist_plot.update(x)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def ioff(self):
        plt.ioff()


# class Visualize:
# 	def __init__(self,TargetGeometry,ProjectionGeometry):
# 		self.ProjectionGeometry = ProjectionGeometry
# 		self.reconstructed = np.zeros((ProjectionGeometry.projections.shape[0],ProjectionGeometry.projections.shape[0]))
# 		self.target_obj = TargetGeometry
# 		self.A = projectorconstructor(self.target_obj,self.ProjectionGeometry)

# 		self.fig, self.axs = plt.subplots(1,1,figsize=(7,6))
# 		self.image = self.axs.imshow(self.reconstructed,cmap='CMRmap')
# 		self.axs.axis('off')
# 		self.axs.set_xticks([])
# 		self.axs.set_yticks([])

# 	def show(self,):
# 		plt.ion()
# 		plt.show()
# 		self.fig.canvas.flush_events()
# 		plt.draw()
# 		if self.ProjectionGeometry.projections.ndim == 3:
# 			projection = self.ProjectionGeometry.projections[:,:,self.ProjectionGeometry.projections.shape[2]//2]
# 		else:
# 			projection = self.ProjectionGeometry.projections
# 		cmax = np.max(self.A.backward(projection))/1.5
# 		self.image.set_clim(vmin=0,vmax=cmax)
# 		angles = self.ProjectionGeometry.angles
# 		angle_step = np.abs(angles[0] - angles[1])

# 		projection_copy = np.zeros_like(projection)
# 		projection_copy [:,0] = 1
# 		occ_shadow = self.A.backward(projection_copy)
# 		# plt.imshow(occ_shadow)
# 		# plt.show()
# 		for i, (curr_proj, angle) in enumerate(zip(projection.T, angles)):
# 			projection_copy[:,i] = curr_proj

# 			self.reconstructed = self.A.backward(projection_copy)

# 			self.reconstructed = rotate(self.reconstructed,np.max(angles)-angle,reshape=False,order=2)

# 			self.reconstructed = self.reconstructed * occ_shadow
# 			# self.reconstructed = rotate(self.reconstructed,angles[-i],reshape=False,order=2).T


# 			# self.reconstructed += np.broadcast_to(curr_proj,self.reconstructed.shape).T
# 			# self.reconstructed = rotate(self.reconstructed,angle_step,reshape=False,order=2)


# 			self.image.set_data(self.reconstructed)
# 			# self.image.set_clim(vmax=np.max(self.reconstructed))
# 			self.fig.canvas.flush_events()
# 			plt.draw()
# 			plt.savefig("%d.png"%i,dpi=300, bbox_inches = 'tight', pad_inches = 0)


# 		plt.ioff()
# 		plt.show()

# def update(self,x,b,i):

# 	if i == 0:
# 		self.showTarget(self.target,ax=self.axs[0])

# 	# self.showError(ax=self.axs[1])
# 	# self.showHistogram(ax=self.axs[3])

# 	# if i == self.n_iter -1:
# 	# 	self.showProj(ax=self.axs[4])
# 	# 	self.showRecon(ax=self.axs[5])
# 	self.fig.canvas.draw()
# 	self.fig.canvas.flush_events()
# 	# return


def errorTolerancePlot(x, target_geo, dh, tol, savepath=None):

    void = np.zeros_like(target_geo.array)
    void[target_geo.void_inds] = 1
    gel = np.zeros_like(target_geo.array)
    gel[target_geo.void_inds] = 1

    cmap_under = matplotlib.colors.ListedColormap(["black", "bisque"])
    cmap_over = matplotlib.colors.ListedColormap(["black", "paleturquoise"])

    th_over = np.array(x >= dh - tol, dtype=np.bool)
    th_over = np.array(np.logical_and(th_over, void), dtype=int)

    th_under = np.array(x <= dh + tol, dtype=np.bool)
    th_under = np.array(np.logical_and(th_under, gel), dtype=int)

    fig, axs = plt.subplots(1, 3, figsize=(15, 12))

    # target
    axs[0].imshow(target_geo.array, cmap="gray", interpolation="none")
    axs[0].axis("off")

    # overexposed
    axs[1].imshow(th_over, cmap=cmap_over)
    axs[1].axis("off")

    # underexposed
    axs[2].imshow(th_under, cmap=cmap_under)
    axs[2].axis("off")

    plt.tight_layout()
    if savepath is not None:
        saveFigure(savepath)


def saveFigure(savepath):
    try:
        plt.savefig(savepath, dpi=500)
    except:
        print("Could not save figure!")
