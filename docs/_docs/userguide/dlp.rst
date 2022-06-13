.. _userguide_dlp:

########################
Digital light processing
########################

In tomographic VAM printers, typically a digital light processing (DLP) device is used to project the computed images into the volume of photosensitive material. DLP video projectors and standalone DMD controllers are often used :cite:`Kelly2019a,Bernal2019,Loterie2020a,Bhattacharya2021,Orth2021,Rackson2022,Kollep2022,Wang2022,Schwartz2022,Toombs2022`. In some cases, these devices have video display ports and are used a monitor or extended monitor on the user's computer. The :py:mod:`vamtoolbox.dlp.players` module makes displaying images or image sequences on the connected DLP device easy.

**********
DLP player
**********

.. raw:: html

    <video width="500px" controls="true" autoplay="true" loop="true">
        <source src="../../_static/trifurcated_vasculature_video.mp4" type="video/webm">
        Example player from video file source.
    </video>

The :py:class:`vamtoolbox.dlp.players.player` is a tool which accepts several types of image sequence formats and video and displays the image sequence or video on the DLP device. 

.. warning::
    The :py:meth:`vamtoolbox.dlp.players.player` must be run from within 

    .. code-block:: python
        if __name__ == "__main__":

    because the player spawns a subprocess with the `multiprocessing standard library <https://docs.python.org/3/library/multiprocessing.html>`. 

Options
=======

There are several options that can be changed to control the display of videos and image sequences. These options are specified in the :py:def:`vamtoolbox.dlp.players.player` initialization. 

The starting image index can also be specified with the ``start_index`` keyword argument. See :ref:`examples_DLPplayer`. This is useful if some rotation alignment is required at the beginning of exposure.

The background color shown during paused playback can be changed with the ``pause_bg_color`` keyword argument. The color is specified with a 3-element tuple in the RGB format [0-255] range. For instance, to set the solid background color to be white while paused (this clears the current image and displays only the solid color), the keyword argument would be ``pause_bg_color=(255,255,255)``. 

.. note::
    The default action (no ``pause_bg_color`` specified or ``pause_bg_color=None``) is to leave the image unchanged during paused playback. This means the static image is displayed while paused. 

The window in which the image sequence or video is shown can be set to be full screen or bordered windowed-mode with the ``windowed`` keyword argument. If specified as ``windowed=True``, the size of the window will be the size of the input image sequence or video. The default is borderless fullscreen mode. 

The duration of playback can be specified with the ``duration`` keyword argument. If specified, the playback will stop automatically after the specified time has elapsed **after pressing the spacebar to start the playback**. Paused playback time does not count towards the total elapsed time. See :ref:`playback_interaction`.

.. _playback_interaction:

Playback interaction
====================

To start the playback of the image sequence or video, press the spacebar.

While the image sequence or video is playing, the user can press the spacebar to pause or resume playback. If the input was either a directory of image files, image sequence object, or a sinogram object, when the spacebar is pressed, the terminal will print which image index playback was paused on. 



Image sequence
==============
The :py:mod:`vamtoolbox.imagesequence` module contains the :py:class:`vamtoolbox.imagesequence.ImageSeq` class and helper functions insert a sinogram into a sequence of images for display on DLP device. A :py:class:`vamtoolbox.imagesequence.ImageSeq` object can be saved with the :py:meth:`vamtoolbox.imagesequence.ImageSeq.save` method or the image sequence itself can be saved as a video (:py:meth:`vamtoolbox.imagesequence.ImageSeq.saveAsVideo`) or sequence of image files (:py:meth:`vamtoolbox.imagesequence.ImageSeq.saveAsImages`).


Image configuration
-------------------
A :py:class:`vamtoolbox.imagesequence.ImageConfig` object contains the settings which describe how the sinogram is inserted into the image that is to be displayed on the DLP device. 



***************
Setup utilities
***************

The :py:mod:`vamtoolbox.dlp.setup` module has several utility functions to assist in the initial setup and calibration of the VAM printer. 

.. note:: 
   See the :ref:`examples_DLPsetup` example for demonstrations about how to use each setup utility.

Axis alignment
==============
:py:class:`vamtoolbox.dlp.setup.AxisAlignment` is a class that allows the user to align the rotation axis of the VAM printer to the "central" axis of the projector device. 

Focus
=====
:py:class:`vamtoolbox.dlp.setup.Focus` is a class that will display a Siemen's star (or spoke target) to assist in focusing the optical system inside the resin container. 
