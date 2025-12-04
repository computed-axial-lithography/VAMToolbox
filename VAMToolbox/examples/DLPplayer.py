import vamtoolbox as vam

if __name__ == "__main__":
    # Precomputed folder of images
    vam.dlp.players.player(
        images_dir=vam.resources.load("seq0imagesdir"), rot_vel=24, windowed=True
    )

    # Sinogram object
    sino = vam.geometry.loadVolume(file_name=vam.resources.load("sino0.sino"))
    iconfig = vam.imagesequence.ImageConfig(
        image_dims=(1920, 1080), array_num=2, array_offset=450
    )
    vam.dlp.players.player(
        sinogram=sino, image_config=iconfig, rot_vel=24, windowed=True
    )

    # Image sequence object
    sino = vam.geometry.loadVolume(file_name=vam.resources.load("sino0.sino"))
    iconfig = vam.imagesequence.ImageConfig(
        image_dims=(1920, 1080), array_num=2, array_offset=450
    )
    image_seq = vam.imagesequence.ImageSeq(image_config=iconfig, sinogram=sino)
    vam.dlp.players.player(
        image_seq=image_seq,
        pause_bg_color=(0, 255, 0),
        duration=10,
        rot_vel=24,
        start_index=100,
        windowed=True,
    )

    # Video
    vam.dlp.players.player(
        video=vam.resources.load("video0.mp4"), rot_vel=12, windowed=True
    )
