import os
from ctypes import ArgumentError

import cv2
import dill  # type: ignore
import numpy as np
from PIL import Image
from scipy import ndimage

import vamtoolbox
from vamtoolbox.geometry import Sinogram


class ImageConfig:
    def __init__(self, image_dims: tuple[int, int], **kwargs):
        """
        Parameters
        ----------
        image_dims : tuple
            (image width, image height)

        rotate_angle : float, optional
            rotation angle in degrees clockwise

        u_offset : int, optional
            offset in pixels from center in U-axis (+) right (-) left

        v_offset : int, optional
            offset in pixels from center in V-axis (+) up (-) down

        invert_u : bool, optional
            invert U-axis

        invert_v : bool, optional
            invert V-axis

        array_num : int, optional
            number of multiplied sinograms

        array_offset : int, optional
            pitch of arrayed sinograms in pixels

        normalization_percentile : float, optional
            normalize the intensity values with this percentile

        bit_depth : int, optional
            bit depth of the resulting image sequence

        intensity_scale : float, optional
            intensity scale factor

        size_scale : float, optional
            size scale factor
        """
        self.N_u, self.N_v = image_dims

        if self.N_u % 4 != 0 or self.N_v % 4 != 0:
            raise Exception("Image dimensions should be divisible by 4.")

        self.rotate_angle = (
            0.0 if "rotate_angle" not in kwargs else kwargs["rotate_angle"]
        )
        self.u_offset = 0 if "u_offset" not in kwargs else kwargs["u_offset"]
        self.v_offset = 0 if "v_offset" not in kwargs else kwargs["v_offset"]

        self.invert_u = False if "invert_u" not in kwargs else kwargs["invert_u"]
        self.invert_v = False if "invert_v" not in kwargs else kwargs["invert_v"]

        self.array_num = 1 if "array_num" not in kwargs else kwargs["array_num"]
        self.array_offset = 0 if "array_offset" not in kwargs else kwargs["array_offset"]

        self.normalization_percentile = (
            None
            if "normalization_percentile" not in kwargs
            else kwargs["normalization_percentile"]
        )
        self.intensity_scale = (
            1.0 if "intensity_scale" not in kwargs else kwargs["intensity_scale"]
        )
        self.bit_depth = 8 if "bit_depth" not in kwargs else kwargs["bit_depth"]

        self.size_scale = 1.0 if "size_scale" not in kwargs else kwargs["size_scale"]


class ImageSeq:
    def __init__(self, image_config: ImageConfig, sinogram: np.ndarray | Sinogram):
        """
        Parameters
        ----------
        image_config : imagesequence.ImageConfig object
            configuration object that contains the options how the sinogram should be placed in the image(s) for projection

        sinogram : geometry.Sinogram object
            sinogram object to be converted to image set

        ----------
        The stored sinogram is formatted by the following sequence of operations:
        - Normalization to specified 'normalization_percentile' (if not None)
        - Scale sinogram values with 'intensity_scale'
        - truncate to specified bit_depth (maximum value = 2**bit_depth-1). 255 for 8-bit, 1023 for 10-bit
        """
        if isinstance(sinogram, np.ndarray):
            pass
        elif isinstance(sinogram, Sinogram):
            sinogram = sinogram.array
        else:
            raise ArgumentError("sinogram not specified.")

        self.file_extension = ".imgseq"
        self.image_config = image_config

        mod_sinogram = np.copy(sinogram)

        if self.image_config.invert_u:
            mod_sinogram = _invertU(mod_sinogram)

        if self.image_config.invert_v:
            mod_sinogram = _invertV(mod_sinogram)

        if self.image_config.rotate_angle != 0.0:
            mod_sinogram = _rotate(mod_sinogram, self.image_config.rotate_angle)

        if self.image_config.size_scale != 1.0:
            mod_sinogram = _scaleSize(mod_sinogram, self.image_config.size_scale)

        max_output_value = 2**self.image_config.bit_depth - 1

        if self.image_config.normalization_percentile is not None:
            normalization_value = np.percentile(
                mod_sinogram, self.image_config.normalization_percentile
            )
            mod_sinogram = mod_sinogram / normalization_value * max_output_value

        if self.image_config.intensity_scale != 1.0:
            mod_sinogram = _scaleIntensity(
                mod_sinogram, self.image_config.intensity_scale
            )

        dtype: np.typing.DTypeLike
        if self.image_config.bit_depth <= 8:
            dtype = np.uint8
        elif self.image_config.bit_depth <= 16:
            dtype = np.uint16
        elif self.image_config.bit_depth <= 32:
            dtype = np.uint32
        else:
            raise ValueError("Bit depth higher than 32-bit is not supported.")

        # The truncation is trivial if normalization_percentile is specified.
        mod_sinogram = _truncateIntensity(mod_sinogram, max_output_value).astype(dtype)

        images = list()
        N_angles = mod_sinogram.shape[1]

        for j in range(N_angles):
            # FIXME: is np.uint8 truncating here?
            image_out = np.zeros(
                (self.image_config.N_v, self.image_config.N_u), dtype=np.uint8
            )

            if self.image_config.array_num != 1:
                image = _arrayInsertImage(
                    mod_sinogram[:, j, :].T, image_out, self.image_config
                )
                images.append(image)

            else:
                # FIXME: v_offset was missing here
                image = _insertImage(
                    mod_sinogram[:, j, :].T, image_out, self.image_config, 0
                )
                images.append(image)

        self.images = images

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

    def save(self, name: str):
        """save imagesequence.ImageSeq as 'name.imgseq'"""
        file = open(name + self.file_extension, "wb")
        dill.dump(self, file)
        file.close()

    def preview(
        self,
    ):
        """Preview an animated image sequence"""
        vamtoolbox.dlp.players.preview(self)

    def saveAsVideo(
        self,
        save_path: str,
        rot_vel: float,
        num_loops: float = 1,
        mode: str = "conventional",
        angle_increment_per_image: float | None = None,
        preview: bool = False,
    ):
        """
        Parameters
        ----------
        save_path : str
            filename of output video e.g. "video.mp4"

        rot_vel : float
            rotation velocity (deg/s)

        num_loops : float, optional
            number of times to loop the images in playback. In conventional mode, num_loops is equivalent to the number of rotations because the image set is assumed to span a full rotation.

        mode : str, optional
            'conventional' mode: angle_increment_per_image is derived from number of images, assuming the image set span a full rotation.
                            Video duration is proportional to num_loops. The argument angle_increment_per_image is ignored in this mode.
            'prescribed' mode: angle_increment_per_image is prescribed. Assuming the video plays back the image set exactly once, regardless whether the image set span less or more than one rotation.
                            The argument num_loops is ignored in this mode.

        angle_increment_per_image : float, optional
            spacing of images (deg)

        preview : bool, optional
            preview the video while the function exports the video

        """
        if self.images is None:
            raise Exception(
                "Problem encountered creating images in ImageSeq initialization"
            )

        if preview:
            cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)

        if mode == "conventional":
            assert angle_increment_per_image is None, (
                "angle_increment_per_image must be None in conventional mode because it is derived from number of images in sinogram"
            )
            angle_increment_per_image = 360 / len(self.images)
            num_image_per_rot = 360.0 / angle_increment_per_image
            num_total_images = int(np.round(num_image_per_rot * num_loops))
        elif mode == "prescribed":
            assert angle_increment_per_image is not None, (
                "angle_increment_per_image must be None in conventional mode because it is derived from number of images in sinogram"
            )
            num_total_images = int(len(self.images) * num_loops)
        else:
            raise Exception(
                'mode argument is not valid. Either "conventional" or "prescribed"'
            )

        image_time = angle_increment_per_image / rot_vel
        fps = 1 / image_time

        codec = cv2.VideoWriter.fourcc(*"avc1")  # type: ignore
        video_out = cv2.VideoWriter(
            save_path,
            codec,
            fps,
            (int(self.image_config.N_u), int(self.image_config.N_v)),
            isColor=False,  # FIXME: This wasn't here before
        )

        k = 0
        while k < num_total_images:
            image = self.images[k % len(self.images)]
            video_out.write(image)
            if preview:
                cv2.imshow("Preview", image)
                cv2.waitKey(1)
            k += 1
            if (k == 1) or (k % 5 == 0) or (k == num_total_images):
                print(f"Writing video frame {k:4d}/{num_total_images:4d}...")

        video_out.release()

    def saveAsImages(
        self, save_dir: str, image_prefix: str = "image", image_type: str = ".png"
    ):
        for k, image in enumerate(self.images):
            save_path = os.path.join(
                save_dir, image_prefix + "%s" % str(k).zfill(4) + image_type
            )
            im = Image.fromarray(image)
            im.save(save_path, subsampling=0, quality=100)
            print(
                "Saving image %s/%s"
                % (str(k).zfill(4), str(len(self.images) - 1).zfill(4))
            )


def loadImageSeq(file_name: str):
    file = open(file_name, "rb")
    data_pickle = file.read()
    file.close()
    A = dill.loads(data_pickle)

    return A


def _insertImage(
    image: np.ndarray,
    image_out: np.ndarray,
    image_config: ImageConfig,
    v_offset,
    **kwargs,
):
    N_u = image_config.N_u
    N_v = image_config.N_v

    S_u = image.shape[1]
    S_v = image.shape[0]

    u1, u2 = (
        int(N_u / 2 - image_config.u_offset - S_u / 2),
        int(N_u / 2 - image_config.u_offset + S_u / 2),
    )
    v1, v2 = (
        int(N_v / 2 - image_config.v_offset - v_offset - S_v / 2),
        int(N_v / 2 - image_config.v_offset - v_offset + S_v / 2),
    )

    if u1 < 0 or u2 > image_config.N_u:
        raise Exception(
            "Image could not be inserted because it is either too large in the u-dimension or the offset causes it to extend out of the input screen size"
        )
    if v1 < 0 or v2 > image_config.N_v:
        raise Exception(
            "Image could not be inserted because it is either too large in the v-dimension or the offset causes it to extend out of the input screen size"
        )

    image_out[v1:v2, u1:u2] = image

    return image_out


def _arrayInsertImage(
    image: np.ndarray, image_out: np.ndarray, image_config: ImageConfig
):
    for k in range(image_config.array_num):
        if image_config.array_num % 2 == 0:
            # if array number is even distribute evenly over image height centered at midheight
            a = k + 1
            array_dir = int(
                np.ceil(a / 2) * (-1) ** (k)
            )  # sequence 1, -1, 2, -2, 3, -3,...
            tmp_v_offset = int(
                image_config.array_offset * array_dir / 2 + image_config.v_offset
            )
        else:
            # if array number is odd distribute around the middle element of the array of images
            array_dir = int(
                0 + np.ceil(k / 2) * (-1) ** (k + 1)
            )  # sequence 0, 1, -1, 2, -2, 3, -3,...
            tmp_v_offset = int(
                image_config.array_offset * array_dir + image_config.v_offset
            )

        image_out = _insertImage(image, image_out, image_config, v_offset=tmp_v_offset)

    return image_out


def _invertU(sinogram):
    mod_sinogram = np.flip(sinogram, axis=0)
    return mod_sinogram


def _invertV(sinogram):
    mod_sinogram = np.flip(sinogram, axis=2)
    return mod_sinogram


def _scaleSize(sinogram, scale_factor):
    new_height = int(sinogram.shape[0] * scale_factor)
    new_width = int(sinogram.shape[2] * scale_factor)
    mod_sinogram = np.zeros((new_height, sinogram.shape[1], new_width))

    for i in range(sinogram.shape[1]):
        mod_sinogram[:, i, :] = cv2.resize(
            sinogram[:, i, :], (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

    return mod_sinogram


def _scaleIntensity(sinogram, intensity_scalar):
    mod_sinogram = sinogram * intensity_scalar
    return mod_sinogram


def _truncateIntensity(sinogram, maximum_intensity):
    mod_sinogram = np.minimum(sinogram, maximum_intensity)
    return mod_sinogram


def _rotate(sinogram, angle_deg):
    mod_sinogram = ndimage.rotate(sinogram, angle_deg, axes=(0, 2), reshape=True, order=1)

    return mod_sinogram
