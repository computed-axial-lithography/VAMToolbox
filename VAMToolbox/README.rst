.. image:: /docs/_static/logos/logo_bone.png
   :height: 200px
----

.. |conda| image:: https://anaconda.org/vamtoolbox/vamtoolbox/badges/version.svg
   :target: https://anaconda.org/vamtoolbox/vamtoolbox

.. |rtd| image:: https://readthedocs.org/projects/vamtoolbox/badge/?version=latest
   :target: https://vamtoolbox.readthedocs.io/en/latest/?badge=latest

.. |zen| image:: https://zenodo.org/badge/500715593.svg
   :target: https://zenodo.org/badge/latestdoi/500715593

+----------------------+-----------+
| Deployment           | |conda|   |
+----------------------+-----------+
| Documentation        | |rtd|     |
+----------------------+-----------+
| Citation             | |zen|     |
+----------------------+-----------+

VAMToolbox is a Python library to support the generation of the light projections and the control of a DLP projector for tomographic volumetric additive manufacturing. It provides visualization, various optimization techniques, and flexible projection geometries to assist in the creation of sinograms and reconstructions for simulated VAM.

VAMToolbox 2.0.0 Release Notes:
------------
This major release includes a number of new features and improvements. The major changes are listed below. For more details, please refer to the documentation.

1. General loss function to formulate optimization of grayscale response profile with high tunability

   Added a formulation of the general optimization problem called Band-Contraint-Lp-norm (BCLP) minimization. This loss function formulation generalizes three existing optimization schemes and is capable to optimize for grayscale target values.
   Added material response model for capturing non-linear relationships between optical dose and the desired response (such as conversion). This allows us to optimize response profile instead of dose profile.
   Refer to the "Band-Contraint-Lp-norm" section for details of BCLP and material response.

2. Ray tracing propagator

   Added a ray tracing propagator to model light attenuation and refraction in medium with gradient refractive index. This ray tracer is written with pyTorch and compatible with python CUDA libraries like cuPy. This ray tracing is performed on GPU.
   Added a models of spatial variant attenuation coefficient, absorption coefficient and refractive index to support ray tracing operations and optimization. 
   Refer to the "Ray tracing propagator and algebraic propagator" section for details of ray tracing.

3. OpenGL voxelizer

   Major performance improvement over pure python voxelizers. This is important for voxelizing large objects at high resolution. Please refer to `voxelizestl <https://github.com/computed-axial-lithography/VAMToolbox/blob/main/examples/voxelizestl.py>`_ for example usage. This is also available as a standalone package `OpenGL Voxelizer <https://github.com/computed-axial-lithography/OpenGL-voxelizer>`_.


Band-Contraint-Lp-norm (BCLP) minimization (contribution from `LDCT-VAM <https://github.com/facebookresearch/LDCT-VAM>`_)
------------
This new loss function unified the optimization for both real-valued (grayscale) and binary targets. It is a generalization of three existing projection optimization schemes.
The target tomogram can now be specified in physical unit of response (such as degree-of-conversion, elastic modulus, or refractive index).
BCLP uses a material response model to capture the non-linear relationship between the response and optical dose. 
This response model is consistently implemented throughout initialization, optimization and evaluation. 
The physical unit of projection parameters (sinogram), optical dose and material response are all preserved during optimization for experimental calibration purposes.
One major benefit of the BCLP formulation is that it allows numerous optimization features to be implemented in a unified framework.
The BCLP loss function provides control over local response tolerance, local weighting and global error sparsity.
For details of the BCLP formulation, please refer to our arXiv publication "Tomographic projection optimization for volumetric additive manufacturing with general band constraint Lp-norm minimization".


Ray tracing propagator and algebraic propagator (contribution from `LDCT-VAM <https://github.com/facebookresearch/LDCT-VAM>`_)
------------
The light propagation model is one of the most critical elements in projection optimization. A ray tracing propagator is coded in pyTorch to models light attenuation, absorption and refraction in medium with gradient refractive index.
The light attenuation, absorption and refraction is simulated based on a spatial description of the simulation medium.
Additionally, the ray tracer can generate an algebraic representation of the propagation such that various algebraic techniques in tomography can be applied.
LCDT-VAM provides algebraic propagators (one in scipy and one in pyTorch) to compute light propagation via matrix-vector multiplication.
The memory-intensive algebraic represntation is only practical for 2D problems or 3D shift-invariant problems (shift-invariant in z direction, along the rotation axis).
However, when the propagation can be performed algebraically, the computation is much faster than the ray tracing propagator.
For futher details of the algebraic representation, refers to the supplementary of the BCLP publication above.


Installation
------------

To install VAMToolbox, enter the command below in the `Anaconda <https://www.anaconda.com/products/distribution>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ prompt::

   conda install vamtoolbox -c vamtoolbox -c conda-forge -c astra-toolbox

*NOTE: This command is different than what Anaconda.org suggests. This is because to properly install the dependencies you must tell conda to search in the astra-toolbox and vamtoolbox channels (in addition to the conda-forge channel, this is a default channel but is added to be explicit).*

*NOTE: This toolbox is currently only compatible with Windows OS.*

For more information, refer to the `installation documentation <https://vamtoolbox.readthedocs.io/en/latest/_docs/gettingstarted.html>`_.


Resources
---------
View the `documentation <https://vamtoolbox.readthedocs.io/en/latest/_docs/intro.html>`_ site.


License
------------
This repository is licensed under GNU General Public License v3. Please see LICENSE.txt for details.
