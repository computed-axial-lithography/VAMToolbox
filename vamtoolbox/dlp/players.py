import warnings
import numpy as np
import time
import glob
import pyglet
import pyglet.gl
import pyglet.window.key as key
import multiprocessing as mp
import PIL.Image

import vamtoolbox


class VideoPlayer:
    def __init__(self, path):
        self._paused = False

        self.media = pyglet.media.load(path)
        self.format = self.media.video_format

        self._player = pyglet.media.Player()

        self._player.queue(self.media)

        self._player.play()

    def pauseVideo(self):
        self._player.pause()
        self._paused = True

    def resumeVideo(self):
        self._player.play()
        self._paused = False


class SequencePlayer(pyglet.sprite.Sprite):
    """
    [1]: https://github.com/pyglet/pyglet/issues/152#issuecomment-597480199
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._paused = False

    def pauseSequence(self):
        if self._paused or not hasattr(self, "_animation"):
            return
        pyglet.clock.unschedule(self._animate)
        self._paused = True

    def resumeSequence(self):
        if not self._paused or not hasattr(self, "_animation"):
            return
        frame = self._animation.frames[self._frame_index]
        self._texture = frame.image.get_texture()
        self._next_dt = frame.duration
        if self._next_dt:
            pyglet.clock.schedule_once(self._animate, self._next_dt)
        self._paused = False

    @property
    def frame_index(self):
        return self._frame_index

    @frame_index.setter
    def frame_index(self, index):
        # Bound to available number of frames
        self._frame_index = max(0, min(index, len(self._animation.frames) - 1))

    def onAnimationEnd(self):
        pass


class _Process(pyglet.window.Window):
    def __init__(self, *args, **kwargs):

        assert kwargs["rot_vel"] is not None, "rot_vel must be specified"
        self.rot_vel = kwargs["rot_vel"]
        self.start_index = 0 if "start_index" not in kwargs else kwargs["start_index"]
        self.debug_fps = False if "debug_fps" not in kwargs else kwargs["debug_fps"]
        self.windowed = False if "windowed" not in kwargs else kwargs["windowed"]
        self.screen_num = -1 if "screen_num" not in kwargs else kwargs["screen_num"]
        self.sinogram = None if "sinogram" not in kwargs else kwargs["sinogram"]
        self.image_config = (
            None if "image_config" not in kwargs else kwargs["image_config"]
        )
        self.image_seq = None if "image_seq" not in kwargs else kwargs["image_seq"]
        self.images_dir = None if "images_dir" not in kwargs else kwargs["images_dir"]
        self.video = None if "video" not in kwargs else kwargs["video"]
        self.duration = None if "duration" not in kwargs else kwargs["duration"]
        self.pause_bg_color = (
            None if "pause_bg_color" not in kwargs else kwargs["pause_bg_color"]
        )

    def run(self):

        if self.video is not None:
            self.video_player = VideoPlayer(self.video)

            width, height = (
                self.video_player.format.width,
                self.video_player.format.height,
            )

            self.resume = self.video_player.resumeVideo
            self.pause = self.video_player.pauseVideo
            self._paused = self.video_player._paused

        else:
            if self.sinogram is not None:
                assert (
                    self.image_config is not None
                ), "image_config must be specified along with sinogram geometry"
                if isinstance(self.sinogram, vamtoolbox.geometry.Sinogram):
                    self.image_seq = vamtoolbox.imagesequence.ImageSeq(
                        self.image_config, sinogram=self.sinogram
                    )
                else:
                    Exception("sinogram is not of type vamtoolbox.geometry.Sinogram")

                sprites = list()
                for k, image in enumerate(self.image_seq.images):
                    print(
                        "Loading image %s/%s"
                        % (
                            str(k).zfill(4),
                            str(len(self.image_seq.images) - 1).zfill(4),
                        )
                    )
                    image = np.ascontiguousarray(np.flipud(image))
                    aii = vamtoolbox.dlp.arrayimage.ArrayInterfaceImage(image)
                    pyglet_image = aii.texture
                    sprites.append(pyglet_image)
                height, width = image.shape

            elif self.image_seq is not None:
                assert isinstance(
                    self.image_seq, vamtoolbox.imagesequence.ImageSeq
                ), "image_seq must be of type imagesequence.ImageSeq"

                sprites = list()
                for k, image in enumerate(self.image_seq.images):
                    print(
                        "Loading image %s/%s"
                        % (
                            str(k).zfill(4),
                            str(len(self.image_seq.images) - 1).zfill(4),
                        )
                    )
                    image = np.ascontiguousarray(np.flipud(image))
                    aii = vamtoolbox.dlp.arrayimage.ArrayInterfaceImage(image)
                    pyglet_image = aii.texture
                    sprites.append(pyglet_image)
                height, width = image.shape

            elif self.images_dir is not None:
                images_glob = glob.glob(self.images_dir + "\\*.png")
                sample_image = PIL.Image.open(images_glob[0])

                sprites = list()
                for filename in images_glob:
                    print("Loading image %s" % filename)
                    sprites.append(pyglet.image.load(filename))

                height, width = np.array(sample_image).shape

                ############# Alternate method for real time update of the image ############
                # start=time.perf_counter()
                # for i, image in enumerate(sprites):
                #     self.window.dispatch_events()
                #     image.blit(0, 0, 0)
                #     self.window.flip()

                # end = time.perf_counter()
                # total_time = end-start
                # print(total_time)
                # print("fps",i/total_time)

            self.N_images_per_rot = len(sprites)
            dt = float(360.0 / self.N_images_per_rot / self.rot_vel)
            self.sequence = pyglet.image.Animation.from_image_sequence(
                sprites, dt, True
            )
            # self.sequence_player = Animation(self.sequence) # custom sprite for animation
            self.sequence_player = SequencePlayer(self.sequence)
            # self.sprite = pyglet.sprite.Sprite(self.sequence)
            self.sequence_player._frame_index = self.start_index
            self.resume = self.sequence_player.resumeSequence
            self.pause = self.sequence_player.pauseSequence
            self._paused = self.sequence_player._paused

        display = pyglet.canvas.Display()
        screens = display.get_screens()
        selected_screen = screens[self.screen_num]

        if self.windowed == True:
            super().__init__(
                width, height, visible=False, screen=screens[self.screen_num]
            )
            # self.set_fullscreen(screen=screens[self.screen_num])
        else:
            if width != selected_screen.width or height != selected_screen.height:
                warnings.warn(
                    "Selected screen width or height != image width or height! Image may be decentered from display. Check input width and height of image config."
                )
            super().__init__(
                width,
                height,
                visible=False,
                style=pyglet.window.Window.WINDOW_STYLE_BORDERLESS,
                screen=screens[self.screen_num],
            )
            self.set_fullscreen(screen=screens[self.screen_num])
        if self.debug_fps == True:
            self.fps_display = pyglet.window.FPSDisplay(self)

        self.set_mouse_visible(False)
        self.set_visible(True)

        if self.pause_bg_color is not None:
            idle_image = vamtoolbox.dlp.arrayimage.idleImage(
                (height, width), self.pause_bg_color
            )
            self.idle_sprite = pyglet.sprite.Sprite(idle_image)

        print(
            "Beginning display... Press SPACE to play or pause/resume. Press ESC to exit."
        )

        # pyglet.clock.schedule_interval(self.on_draw,dt)
        self._started = False
        self._started_timer = False
        self._paused = False
        self._paused_time = 0.0
        pyglet.app.run()

    def on_draw(self):
        self.clear()

        # can be run here, on_show assumes the window remains maximized forever
        # run once to get the start time immediately before the first frame
        if self._started == True:
            if self._started_timer == False:
                self._start_time = time.perf_counter()
                self._started_timer = True

            # calculate total played time, accounting for the total paused time
            self._played_time = (
                time.perf_counter() - self._paused_time - self._start_time
            )

            # kill player if played time > duration
            if self.duration is not None:
                if self._played_time >= self.duration:
                    pyglet.app.exit()

        if self._started == True:
            if self._paused == False:
                if hasattr(self, "sequence_player"):
                    self.sequence_player.draw()
                elif hasattr(self, "video_player"):
                    try:
                        self.video_player._player.texture.blit(0, 0)
                    except:
                        pyglet.app.exit()

            elif self._paused == True and self.pause_bg_color is not None:
                self.idle_sprite.draw()

            else:
                pass

        if self.debug_fps == True:
            self.fps_display.draw()

    def on_key_press(self, symbol, modifiers):

        if symbol == key.SPACE:
            if self._started == True:
                if self._paused == True:
                    # resume
                    print("resume")

                    self._end_paused_time = (
                        time.perf_counter() - self._start_paused_time
                    )
                    self._paused_time = self._paused_time + self._end_paused_time

                    self.resume()
                    self._paused = False
                else:
                    # pause
                    if hasattr(self, "sequence_player"):
                        print(
                            "paused at index: %d/%d"
                            % (self.sequence_player._frame_index, self.N_images_per_rot)
                        )
                    elif hasattr(self, "video_player"):
                        print("paused")
                    self._start_paused_time = time.perf_counter()
                    self.pause()
                    self._paused = True

            else:
                self._started = True
                print("starting")

        return super().on_key_press(symbol, modifiers)


def _worker(*args, **kwargs):

    dlp = _Process(*args, **kwargs)
    dlp.run()


def player(*args, **kwargs):
    """
    Parameters
    ----------
    rot_vel : float
        rotation velocity in degrees per second

    start_index : int, optional
        starting index of image sequence, default is 0

    image_seq : imagesequence.ImageSeq, optional
        imagesequence.ImageSeq object

    sinogram : geometry.Sinogram, optional
        geometry.Sinogram object

    image_config : imagesequence.ImageConfig, optional
        imagesequence.ImageConfig object defining sinogram to image transformation

    images_dir : str, optional
        file directory to saved images

    screen_num : int, optional
        number of the screen to display onto, default -1 (last screen)

    windowed : bool, optional
        bordered window, default False

    duration : float, optional
        duration of sequence or video playback, default None (infinite playback)

    pause_bg_color : tuple, optional
        color to be shown when playback is paused, default (0,0,0) (black background)

    debug_fps : bool, optional
        display estimated fps on the displayed window, default False




    Examples
    --------

    Press spacebar to start playback. During playback, press spacebar to pause or resume playback.

    Specifying ImageSeq object

    >>> I = ImageSeq()
    >>> player(image_seq=I,rot_vel=12)

    Specifying Sinogram object

    >>> sino = loadVolume("C:\\mysinogram.sino")
    >>> player(sinogram=sino,rot_vel=12)

    Specifying images directory

    >>> dir = "C:\\images"
    >>> player(rot_vel=12,images_dir=dir)

    Specifying video file

    >>> path = "C:\\video.mp4"
    >>> player(rot_vel=12,video=path)

    """

    x = mp.Process(target=_worker, args=args, kwargs=kwargs)
    x.start()

    return x


def preview(image_seq):
    player(image_seq=image_seq, rot_vel=30, windowed=True, screen_num=0)
    return
