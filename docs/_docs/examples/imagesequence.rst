.. _examples_imagesequence:

########################
Creating image sequences
########################

***********
Walkthrough
***********

Load sinogram
=============

The image sequence is created from a sinogram. First, load (or optimize) a sinogram.

.. literalinclude:: ../../../examples/imagesequence.py
    :lines: 1-3

Create image sequence configurations
====================================

The image sequence configuration contains the properties that determine how the sinogram is to be displayed on a screen or digital light projection device. See :py:class:`~vamtoolbox.imagesequence.ImageConfig` to see the options and how they affect the image sequence. Here, two different :py:class:`~vamtoolbox.imagesequence.ImageConfig` objects are created for demonstration.

.. literalinclude:: ../../../examples/imagesequence.py
    :lines: 4-5

Creating image sequence objects
===============================

With the :py:class:`~vamtoolbox.imagesequence.ImageConfig` and a :py:class:`~vamtoolbox.geometry.Sinogram` the :py:class:`~vamtoolbox.imagesequence.ImageSeq` object can be created. The :py:meth:`~vamtoolbox.imagesequence.ImageSeq.preview` method of the object is used to show a preview of image sequence.

.. literalinclude:: ../../../examples/imagesequence.py
    :lines: 7-11

************
Example file
************

.. literalinclude:: ../../../examples/imagesequence.py
    :caption: examples/imagesequence.py