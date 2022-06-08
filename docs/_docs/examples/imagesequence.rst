.. _examples_imagesequence:

########################
Creating image sequences
########################

***********
Walkthrough
***********

Load sinogram
=============

If the user has a folder of image files saved from an :py:class:`vamtoolbox.imagesequence.ImageSeq` object or created through other means, e.g. third-party library, the directory can be specified in the ``images_dir`` keyword argument.

.. literalinclude:: ../../../examples/imagesequence.py
    :lines: 1-4

Create image sequence configurations
====================================

The image sequence configuration contains the properties that determine how the sinogram is to be displayed on a screen or digital light projection device. See :py:class:`vamtoolbox.imagesequence.ImageConfig` to see the options and how they affect the image sequence. Here, two different :py:class:`vamtoolbox.imagesequence.ImageConfig` objects are created for demonstration.

.. literalinclude:: ../../../examples/imagesequence.py
    :lines: 7-10

Creating image sequence objects
===============================

With the :py:class:`vamtoolbox.imagesequence.ImageConfig` and a :py:class:`vamtoolbox.geometry.Sinogram` the :py:class:`vamtoolbox.imagesequence.ImageSeq` object can be created. The :py:meth:`vamtoolbox.imagesequence.ImageSeq.preview` method of the object is used to show a preview of image sequence.

.. literalinclude:: ../../../examples/imagesequence.py
    :lines: 12-16

************
Example file
************

.. literalinclude:: ../../../examples/imagesequence.py
    :caption: examples/imagesequence.py