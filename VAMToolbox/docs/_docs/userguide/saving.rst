.. _userguide_saving:

############################
Save/Load VAMToolbox objects
############################

***********
Saving data
***********

VAMToolbox objects, including :py:class:`~vamtoolbox.geometry.TargetGeometry`, :py:class:`~vamtoolbox.geometry.Sinogram`, and :py:class:`~vamtoolbox.geometry.Reconstruction` can be saved with the class method :py:meth:`vamtoolbox.geometry.Volume.save`, and the :py:class:`~vamtoolbox.imagesequence.ImageSeq` object can be saved with the class method :py:meth:`~vamtoolbox.imagesequence.ImageSeq.save`.

.. code-block:: python

   # target_geo is a geometry.TargetGeometry object
   # save the object as mytarget.target
   # ".target" is the default file extension
   target_geo.save("C:\User\myfiles\mytarget")

   # sino is a geometry.Sinogram object
   # save the sinogram as mysinogram.sino
   # ".sino" is the default file extension
   sino.save("C:\User\myfiles\mysinogram")

   # recon is a geometry.Reconstruction object
   # save the reconstruction as myreconstruction.recon
   # ".recon" is the default file extension
   recon.save("C:\User\myfiles\myreconstruction")

   # imgseq is a imagesequence.ImageSeq object
   # save the imagesequence as myimagesequence.imgseq
   # ".imgseq" is the default file extension
   imgseg.save("C:\User\myfiles\myimagesequence")

************
Loading data
************

To load :py:class:`~vamtoolbox.geometry.Volume` objects, a convenience function is provided in the :py:meth:`~vamtoolbox.geometry.loadVolume`.

.. code-block:: python
   
   # to load saved Volume objects, use the method in the geometry module
   # loadVolume()
   target_geo_load = vam.geometry.loadVolume("")
   sino_load = vam.geometry.loadVolume("C:\User\myfiles\mysinogram.sino")
   recon_load = vam.geometry.loadVolume("C:\User\myfiles\myreconstruction.recon")

Similarly, to load :py:class:`~vamtoolbox.imagesequence.ImageSeq` objects, a convenience function is provided in the :py:meth:`~vamtoolbox.imagesequence.loadImageSeq`.

.. code-block:: python

   # to load saved ImageSeq objects, use the method in the imagesequence module
   # loadImageSeq()
   imgseq_load = vam.imagesequence.loadImageSeq("C:\User\myfiles\myimagesequence.imgseq")


