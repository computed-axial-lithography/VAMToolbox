.. image:: /docs/_static/logos/logo_bone.png
   :height: 200px
----

.. |conda| image:: https://anaconda.org/vamtoolbox/vamtoolbox/badges/version.svg
   :target: https://anaconda.org/vamtoolbox/vamtoolbox

.. |rtd| image:: https://readthedocs.org/projects/vamtoolbox/badge/?version=latest
   :target: https://vamtoolbox.readthedocs.io/en/latest/?badge=latest

.. |zen| image:: https://zenodo.org/badge/500715593.svg
   :target: https://zenodo.org/badge/latestdoi/500715593


Features of LCDT-VAM
------------
`VAMToolbox <https://github.com/computed-axial-lithography/VAMToolbox>`_ is a python library to support the generation of the light projections and the control of a DLP projector for tomographic volumetric additive manufacturing.
It provides visualization, various optimization techniques, and flexible projection geometries to assist in the creation of sinograms and reconstructions for simulated VAM.

The local-dose-controlled tomography (LDCT) project forks the original VAMToolbox and provides the following features and improvements:
1. Added a formulation of the general optimization problem called Band-Contraint-Lp-norm (BCLP) minimization. This loss function formulation generalizes three existing optimization schemes and is capable to optimize for grayscale target values.
2. Added material response model for capturing the non-linear relationship between optical dose and the desired response (such as conversion). This allows the optimization to optimize response profile instead of dose profile.
3. Added a ray tracing propagator to model light attenuation and refraction in medium with gradient refractive index. This ray tracer is written with pyTorch and performs computation on GPU.
4. Added a models of spatial variant attenuation coefficient, absorption coefficient and refractive index to support ray tracing operations and optimization. 

These new features are to be merged to VAMToolbox 2.0 and maintained by the corresponding community initiated at UC Berkeley. 


Band-Contraint-Lp-norm (BCLP) minimization
------------
This new loss function unified the optimization for both real-valued (grayscale) and binary targets. It is a generalization of three existing projection optimization schemes.
The target tomogram can now be specified in physical unit of response (such as degree-of-conversion, elastic modulus, or refractive index).
BCLP uses a material response model to capture the non-linear relationship between the response and optical dose. 
This response model is consistently implemented throughout initialization, optimization and evaluation. 
The physical unit of projection parameters (sinogram), optical dose and material response are all preserved during optimization for experimental calibration purposes.
One major benefit of the BCLP formulation is that it allows numerous optimization features to be implemented in a unified framework.
The BCLP loss function provides control over local response tolerance, local weighting and global error sparsity.
For details of the BCLP formulation, please refer to our arXiv publication "Tomographic projection optimization for volumetric additive manufacturing with general band constraint Lp-norm minimization".


Ray tracing propagator and algebraic propagator
------------
The light propagation model is one of the most critical elements in projection optimization. A ray tracing propagator is coded in pyTorch to models light attenuation, absorption and refraction in medium with gradient refractive index.
The light attenuation, absorption and refraction is simulated based on a spatial description of the simulation medium.
Additionally, the ray tracer can generate an algebraic representation of the propagation such that various algebraic techniques in tomography can be applied.
LCDT-VAM provides algebraic propagators (one in scipy and one in pyTorch) to compute light propagation via matrix-vector multiplication.
The memory-intensive algebraic represntation is only practical for 2D problems or 3D shift-invariant problems (shift-invariant in z direction, along the rotation axis).
However, when the propagation can be performed algebraically, the computation is much faster than the ray tracing propagator.
For futher details of the algebraic representation, refers to the supplementary of the BCLP publication above.

Prerequisites
------------
- Anaconda or miniconda installation
- Windows OS (This toolbox is not tested on other platforms)
- CUDA-compatible GPU (required when using any features related to ray tracing)


Installation
------------
To install LCDT-VAM, please create a compatible anaconda environment and install the package using pip.

Steps:

1. Clone the repository with git or download the zip file. Navigate into the projector folder LDCT-VAM.

2. Locate the file "ldct_vam_env.yaml" in the "conda" subfolder.
   Execute the following command in anaconda to create an environment with the required dependencies:

   conda env create -f ldct_vam_env.yaml

   Note that if your current working directory is not this subfolder, you need to specify the full file path.
   The created environment will be named "ldct310". As the name suggest, this environment use python 3.10.

3. After creating the environment, activate the environment by running:

   conda activate ldct310

4. In the environment, navigate to the root directory of the repository by:

   cd "path to the root directory of the repository"

   This should be the directory where setup.py is located.

5. Finally, install the toolbox with pip by running:

   pip install -e .

   The flag -e means that this installation is editable. It means changes to the package files comes into effect everytime the python interpreter is restarted.
   It allows you to modify the python package when needed.

Usage
------------
To use the toolbox, just run your python script in the created conda environment.
The package can be imported by:
   import vamtoolbox


License
------------
This repository is licensed under GNU General Public License v3. Please see LICENSE.txt for details.


Publication
------------
"Tomographic projection optimization for volumetric additive manufacturing with general band constraint Lp-norm minimization", arXiv,
Chi Chung Li (@alvinccli), Joseph Toombs (@jttoombs), Hayden K. Taylor, Thomas J. Wallin

Resource on VAMToolbox
------------
+----------------------+-----------+
| Deployment           | |conda|   |
+----------------------+-----------+
| Documentation        | |rtd|     |
+----------------------+-----------+
| Citation             | |zen|     |
+----------------------+-----------+
