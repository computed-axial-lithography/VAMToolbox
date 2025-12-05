import vamtoolbox as vam

sino = vam.geometry.loadVolume(file_name=vam.resources.load("sino0.sino"))
iconfig0 = vam.imagesequence.ImageConfig(
    image_dims=(1920, 1080), array_num=2, array_offset=450
)
iconfig1 = vam.imagesequence.ImageConfig(
    image_dims=(1920, 1080), rotate_angle=45, size_scale=2
)

image_seq = vam.imagesequence.ImageSeq(image_config=iconfig0, sinogram=sino)
image_seq.preview()

image_seq = vam.imagesequence.ImageSeq(image_config=iconfig1, sinogram=sino)
image_seq.preview()
