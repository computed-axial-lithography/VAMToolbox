import numpy as np

try:
    import tigre
except:
    ImportError("TIGRE toolbox is either not installed or installed incorrectly.")
import matplotlib.pyplot as plt

# import display_functions as disp


class tigre3D:
    def __init__(self, target, params):
        self.target = target
        try:
            self.attenuation = np.swapaxes(
                params["attenuation"].astype(np.float32), 2, 0
            )
            self.attenuation = np.ascontiguousarray(self.attenuation)
        except:
            self.attenuation = None
        self.pT = target.shape[0]
        self.nX, self.nY, self.nZ = target.shape
        self.angles_rad = np.deg2rad(params["angles"])
        # setup fixed coordinate grid for backprojection and dimensions of projections
        self.radius = target.shape[0] // 2
        self.y, self.x = np.mgrid[: target.shape[0], : target.shape[0]] - self.radius
        self.center = target.shape[0] // 2
        self.proj_t = np.arange(target.shape[0]) - target.shape[0] // 2

        self.geo = tigre.geometry(
            mode="parallel", nVoxel=np.array([self.nZ, self.nY, self.nX])
        )
        self.geo.dDetector = np.array([1, 1])  # size of each pixel            (mm)
        self.geo.sDetector = self.geo.dDetector * self.geo.nDetector
        self.geo.accuracy = 1
        self.geo.vialRadius = 1
        self.geo.maxIntensity = 1

    def forwardProject(self, target):

        target = target.astype(np.float32)
        target = np.swapaxes(target, 2, 0)
        target = np.ascontiguousarray(target)

        projections = tigre.Ax(
            target,
            self.geo,
            self.angles_rad,
            projection_type="interpolated",
            img_att=self.attenuation,
        )
        projections = np.transpose(projections, (2, 0, 1))

        return projections

    def backProject(self, projections):
        projections = projections.astype(np.float32)

        projections = np.ascontiguousarray(np.transpose(projections, (1, 2, 0)))
        if self.attenuation is not None:
            tmp_attenuation = np.ascontiguousarray(np.swapaxes(self.attenuation, 1, 2))
            reconstruction = tigre.Atb(
                projections, self.geo, self.angles_rad, img_att=tmp_attenuation
            )
        else:
            reconstruction = tigre.Atb(projections, self.geo, self.angles_rad)

        # print(tmp_attenuation.shape)

        reconstruction = np.swapaxes(reconstruction, 0, 2)

        return self.truncateCircle(reconstruction)

    def truncateCircle(self, recon_input):

        out_reconstruction_circle = (self.x**2 + self.y**2) > self.radius**2
        recon_input[out_reconstruction_circle, :] = 0
        return recon_input


if __name__ == "__main__":

    target = np.zeros((300, 300, 100))

    target[50:130, 50:130, :] = 1
    target[110:130, 110:130, :] = 0
    target[50:70, 110:130, :] = 0
    occl = np.zeros_like(target)
    occl[70:110, 70:110, :] = np.inf
    occl[70:85, 95:110, :] = 0
    occl[95:110, 95:110, :] = 0

    disp.view_slices(target, 2, "slices")
    disp.view_slices(occl, 2, "slices")

    params = {"angles": np.linspace(0, 360 - (360 / 180), 100), "attenuation": occl}

    C = tigre3D(target, params)
    projection = C.forwardProject(target)
    disp.view_slices(projection, 2, "slices")
    reconstruction = C.backProject(projection)
    disp.view_slices(reconstruction, 2, "slices")
