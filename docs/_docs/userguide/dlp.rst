.. _userguide_dlp:

########################
Digital light processing
########################

In tomographic VAM printers, typically a digital light processing (DLP) device is used to project the computed images into the volume of photosensitive material. DLP video projectors and standalone DMD controllers are often used :cite:`Kelly2019a,Bernal2019,Loterie2020a,Bhattacharya2021,Orth2021,Rackson2022,Kollep2022,Wang2022,Schwartz2022`. In some cases, these devices have video display ports and are used a monitor or extended monitor on the user's computer. The :py:mod:`VAMToolbox.DLP.DLP` module makes displaying images or image sequences on the connected DLP device easy.

**********
DLP player
**********

.. raw:: html

    <video width="500px" controls="true" autoplay="true" loop="true">
        <source src="../../_static/trifurcated_vasculature_video.mp4" type="video/webm">
        Example player from video file source.
    </video>

The :py:class:`VAMToolbox.DLP.DLP.player` is a tool which accepts several types of image sequence formats and displays the image sequence on the DLP device. 

Playback interaction
====================

While the image sequence or video is playing, the user can press the spacebar to pause or resume playback. If the input was either a directory of image files, image sequence object, or a sinogram object, when the spacebar is pressed, the terminal will print which image index playback was paused on. 

.. note:: 
   The starting image index can also be specified on :py:class:`VAMToolbox.DLP.DLP.player`` initialization with the ``start_index`` keyword argument. See :ref:`examples_DLPplayer`.



Image sequence
==============
The :py:mod:`VAMToolbox.imagesequence` module contains the :py:class:`VAMToolbox.imagesequence.ImageSeq` class and helper functions insert a sinogram into a sequence of images for display on DLP device. A :py:class:`VAMToolbox.imagesequence.ImageSeq` object can be saved with the :py:meth:`VAMToolbox.imagesequence.ImageSeq.save` method or it the image sequence itself can be saved as a video (:py:meth:`VAMToolbox.imagesequence.ImageSeq.saveAsVideo`) or sequence of image files (:py:meth:`VAMToolbox.imagesequence.ImageSeq.saveAsVideo`).


Image configuration
-------------------
A :py:class:`VAMToolbox.imagesequence.ImageConfig` object contains the settings which describe how the sinogram is inserted into the image that is to be displayed on the DLP device. 



***************
Setup utilities
***************

The :py:mod:`VAMToolbox.DLP.Setup` module has several utility functions to assist in the initial setup and calibration of the VAM printer. 

.. note:: 
   See :ref:`examples_DLPsetup` for examples about how to use each setup utility.

Axis alignment
==============
:py:class:`VAMToolbox.DLP.Setup.AxisAlignment` is a class that allows the user to align the rotation axis of the VAM printer to the "central" axis of the projector device. 

Focus
=====
:py:class:`VAMToolbox.DLP.Setup.Focus` is a class that will display a Siemen's star (or spoke target) to assist in focusing the optical system inside the resin container. 
