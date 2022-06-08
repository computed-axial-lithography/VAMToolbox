.. _userguide_FBP:


##############################
Filtered back projection (FBP)
##############################


To be added

.. image:: /_images/examples/userguide/FBPnooffset.png

*********************************************
Offset with most negative element of sinogram
*********************************************

An alternative to truncation of the negative values is to add to the sinogram the most negative value of the filtered sinogram. Since this is simply an offset of the sinogram, this results in a *perfect** reconstruction of a binary target. However, the caveat is that there is a very small process window between the in and out-of-part regions. This optimization technique may be useful if the material has very high contrast or sharp dose threshold response but otherwise the small process window results in high background exposure and consequently prints that are difficult to develop.

.. note::
    See :py:func:`vamtoolbox.metrics.calcPW` for details on the process window in a VAM reconstruction.

.. image:: /_images/examples/userguide/FBPoffset.png