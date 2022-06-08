.. _examples_DLPplayer:

##########
DLP player
##########

***********
Walkthrough
***********

Precomputed folder of images
============================

If the user has a folder of image files saved from an :py:class:`vamtoolbox.imagesequence.ImageSeq` object or created through other means, e.g. third-party library, the directory can be specified in the ``images_dir`` keyword argument.

.. literalinclude:: ../../../examples/DLPplayer.py
    :lines: 1-5

Precomputed sinogram object
===========================

The player also accepts a :py:class:`vamtoolbox.geometry.Sinogram` object as input in the ``sinogram`` keyword argument. If this method is chosen, the ``image_config`` keyword argument must also be specified. See :ref:`userguide_dlp` for information on the :py:class:`vamtoolbox.imagesequence.ImageConfig` object.

.. literalinclude:: ../../../examples/DLPplayer.py
    :lines: 7-10

Image sequence object
=====================

The player also accepts a :py:class:`vamtoolbox.imagesequence.ImageSeq` object as input in the ``image_seq`` keyword argument.


.. literalinclude:: ../../../examples/DLPplayer.py
    :lines: 12-16

Video file
==========

The player accepts a regular video file.

.. literalinclude:: ../../../examples/DLPplayer.py
    :lines: 18-19

.. tip::
   The spacebar can be used to pause/resume playback of the image sequence or video. 

************
Example file
************

.. literalinclude:: ../../../examples/DLPplayer.py
    :caption: examples/DLPplayer.py