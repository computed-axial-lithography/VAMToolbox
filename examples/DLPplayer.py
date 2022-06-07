if __name__ == "__main__":
    import vamtoolbox as vam

    # Precomputed folder of images
    vam.dlp.players.player(images_dir=r"E:\myfiles",rot_vel=24,windowed=True)
    
    # Precomputed image sequence object
    image_seq = vam.imagesequence.loadImageSeq(r"E:\myfiles\seq1.imgseq")
    vam.dlp.players.player(image_seq=image_seq,rot_vel=24,windowed=True)
    
    # Precomputed sinogram object
    sino = vam.geometry.loadVolume(r"E:\myfiles\sino1.sino")
    iconfig=vam.imagesequence.ImageConfig(image_dims=(1920,1080),array_num=2,array_offset=450)
    vam.dlp.players.player(sinogram=sino,image_config=iconfig,rot_vel=24,windowed=True)

    # Video
    vam.dlp.players.player(video=r"E:\myfiles\video1.mp4",rot_vel=12,windowed=True)

    # Display control settings
    vam.dlp.players.player(image_seq=r"E:\myfiles\seq1.imgseq",rot_vel=24,start_index=100,windowed=True)