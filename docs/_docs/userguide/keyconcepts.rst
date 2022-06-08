.. _userguide_keyconcepts:

############
Key concepts
############

The VAMToolbox is composed of geometries, projectors, optimization, and digital light processing (DLP). 

********
Geometry
********

The :py:mod:`vamtoolbox.geometry` module contains classes to organize flow of data within the toolbox. 

* :py:class:`vamtoolbox.geometry.TargetGeometry` is used to hold the voxelized target. It is passed into the optimization function.
* :py:class:`vamtoolbox.geometry.ProjectionGeometry` holds data about the type of projector used, the projection angles, the inclination angle for laminographic geometries, etc. It is passed into the optimization function. 
* :py:class:`vamtoolbox.geometry.Sinogram` 
* :py:class:`vamtoolbox.geometry.Reconstruction`

**********
Projectors
**********

:py:mod:`vamtoolbox.Projector`

************
Optimization
************

:py:mod:`vamtoolbox.Optimizer`

************************
Digital light processing
************************

:py:mod:`vamtoolbox.dlp`

*****
Other
*****

:py:mod:`vamtoolbox.metrics`