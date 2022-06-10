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

**WARNING!**
------------

This project is under active development!


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
