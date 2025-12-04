import numpy as np
import pyglet
import pyglet.gl
import pyglet.window.key as key


from vamtoolbox.dlp.arrayimage import ArrayInterfaceImage


class SetupWindow(pyglet.window.Window):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------

        screen_num : int, optional
            number of the screen to display onto, default -1 (last screen)

        screen_orientation : str, optional
            orientation of screen with respect to world (assumes rotation axis is vertical with respect to world)

        windowed : bool, optional
            bordered window, default False

        N_screen : tuple, optional
            window size in pixels (N_U,N_V) where N_U is # of pixels in horizontal direction and N_V is # of pixels in vertical direction
        """

        self.screen_orientation = (
            "horizontal"
            if "screen_orientation" not in kwargs
            else kwargs["screen_orientation"]
        )
        self.windowed = False if "windowed" not in kwargs else kwargs["windowed"]
        self.N_screen = None if "N_screen" not in kwargs else kwargs["N_screen"]
        self.screen_num = -1 if "screen_num" not in kwargs else kwargs["screen_num"]

        display = pyglet.canvas.Display()
        screens = display.get_screens()
        selected_screen = screens[self.screen_num]

        if self.windowed == True:

            assert (
                self.N_screen != None
            ), "Screen size (N_screen) must be specified in windowed mode. N_screen = (N_U,N_V)"
            self.N_U, self.N_V = self.N_screen
            super().__init__(
                self.N_U, self.N_V, visible=False, screen=screens[self.screen_num]
            )
            # self.set_fullscreen(screen=screens[self.screen_num])
        else:
            self.N_U, self.N_V = selected_screen.width, selected_screen.height
            super().__init__(
                self.N_U,
                self.N_V,
                visible=False,
                style=pyglet.window.Window.WINDOW_STYLE_BORDERLESS,
                screen=screens[self.screen_num],
            )
            self.set_fullscreen(screen=screens[self.screen_num])

        self.array = np.zeros((self.N_V, self.N_U), dtype=np.uint8)
        self.convertArray()
        self.set_mouse_visible(False)
        self.set_visible(True)

    def _run(self):
        pyglet.clock.schedule_interval(self.update, 1.0 / 30.0)
        pyglet.app.run()

    def update(self, _):
        pass

    def convertArray(self):
        aii = ArrayInterfaceImage(self.array)
        aii.dirty()
        pyglet_array = aii.texture
        self.sprite = pyglet.sprite.Sprite(pyglet_array)

    def on_draw(self):
        self.clear()
        self.sprite.draw()


class AxisAlignment(SetupWindow):
    def __init__(
        self,
        half_line_thickness=10,
        half_line_separation=200,
        center_offset=0,
        **kwargs
    ):
        """
        Parameters
        ----------
        half_line_thickness : int, optional
            alignment line half thickness in pixels

        half_line_separation : int, optional
            alignment lines half separation in pixels

        center_offset : int, optional
            alignment lines origin offset from center of image

        Examples
        --------
        >>> AxisAlignment(half_line_thickness=10,half_line_separation=200,center_offset=0,screen_orientation="horizontal")

        """

        super().__init__(**kwargs)
        self.half_line_separation = half_line_separation
        self.half_line_thickness = half_line_thickness
        self.center_offset = center_offset
        self.key_handler = key.KeyStateHandler()
        self.push_handlers(self.key_handler)
        self._run()

    def update(self, _):
        if self.key_handler[key.UP]:
            self.half_line_thickness += 1
        elif self.key_handler[key.DOWN]:
            self.half_line_thickness -= 1
        elif self.key_handler[key.COMMA]:
            self.center_offset -= 1
        elif self.key_handler[key.PERIOD]:
            self.center_offset += 1
        elif self.key_handler[key.LEFT]:
            self.half_line_separation += 1
        elif self.key_handler[key.RIGHT]:
            self.half_line_separation -= 1

        if self.half_line_separation < 1:
            self.half_line_separation = 1
        if self.half_line_thickness < 1:
            self.half_line_thickness = 1

        inside_separation = self.half_line_separation * 2 - self.half_line_thickness * 2
        print("Line separation (inside edge) = %d pixels" % inside_separation)
        print(
            "half line thickness = %d, half line separation (center to center) = %d, center offset = %d\n"
            % (self.half_line_thickness, self.half_line_separation, self.center_offset)
        )

        self.constructLines()
        self.convertArray()

    def constructLines(self):

        self.array = np.zeros((self.N_V, self.N_U), dtype=np.uint8)
        if self.screen_orientation == "horizontal":
            self.array[
                :,
                self.N_U // 2
                - self.half_line_thickness
                + self.center_offset : self.N_U // 2
                + self.half_line_thickness
                + self.center_offset,
            ] = 255
            self.array[
                :,
                self.N_U // 2
                - self.half_line_separation
                - self.half_line_thickness
                + self.center_offset : self.N_U // 2
                - self.half_line_separation
                + self.half_line_thickness
                + self.center_offset,
            ] = 255
            self.array[
                :,
                self.N_U // 2
                + self.half_line_separation
                - self.half_line_thickness
                + self.center_offset : self.N_U // 2
                + self.half_line_separation
                + self.half_line_thickness
                + self.center_offset,
            ] = 255
        elif self.screen_orientation == "vertical":
            self.array[
                self.N_V // 2
                - self.half_line_thickness
                + self.center_offset : self.N_V // 2
                + self.half_line_thickness
                + self.center_offset,
                :,
            ] = 255
            self.array[
                self.N_V // 2
                - self.half_line_separation
                - self.half_line_thickness
                + self.center_offset : self.N_V // 2
                - self.half_line_separation
                + self.half_line_thickness
                + self.center_offset,
                :,
            ] = 255
            self.array[
                self.N_V // 2
                + self.half_line_separation
                - self.half_line_thickness
                + self.center_offset : self.N_V // 2
                + self.half_line_separation
                + self.half_line_thickness
                + self.center_offset,
                :,
            ] = 255


class Focus(SetupWindow):
    def __init__(self, slices: int = 20, **kwargs):
        """Siemens star for adusting focus of optical system

        Parameters
        ----------
        slices : int, optional
            number of slices in the Siemens star


        Examples
        --------
        >>> Focus(slices=10,windowed=True,screen_orientation="horizontal",N_screen=(1000,1000))

        """
        self.slices = slices
        super().__init__(**kwargs)
        self.constructSiemen()
        self._run()

    def constructSiemen(self):

        U, V, R = _createGrid(self.array.shape)
        theta = np.arctan2(V, U)

        self.array = 1 + np.sin(self.slices * theta)
        self.array = np.logical_and((self.array > 1), (R < 0.95)).astype(np.uint8) * 255

        self.convertArray()


# class IntensityCalibration(SetupWindow):


def _createGrid(shape, offset=None):
    if offset is not None:
        u_offset, v_offset = offset
    else:
        u_offset, v_offset = (0, 0)

    dim_v, dim_u = shape
    aspect_ratio = dim_u / dim_v

    U, V = np.meshgrid(
        np.linspace(-1 * aspect_ratio, 1 * aspect_ratio, dim_u),
        np.linspace(-1, 1, dim_v),
    )
    V = V - v_offset
    V = V - u_offset
    R = np.sqrt((V - u_offset) ** 2 + (U - v_offset) ** 2)

    return U, V, R


if __name__ == "__main__":

    # AxisAlignment(half_line_thickness=1,half_line_separation=200,windowed=True,screen_orientation="horizontal",N_screen=(1000,1000))

    Focus(
        slices=16, windowed=True, screen_orientation="horizontal", N_screen=(1000, 1000)
    )
