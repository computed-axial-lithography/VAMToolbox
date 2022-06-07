.. _setup:

#####
Setup
#####

************
Installation
************


Conda
=====
The VAMToolbox is available at the VAMToolbox channel in the conda package manager. Run the following inside a conda environment:

.. code:: console
    
    conda install VAMToolbox


Astra Toolbox
=============

If there are problems with the installation of Astra Toolbox go to the `downloads <https://www.astra-toolbox.com/downloads/index.html#downloads>`_ page and install the latest Python version by following the `installation instructions <https://www.astra-toolbox.com/docs/install.html#installation-instructions>`_.


Uninstallation
==============

Use the Anaconda prompt for one of the following steps.

To remove the toolbox from the base environment (where ``myenv`` is replaced by the name of the environment in which VAMToolbox is installed): 

.. code:: console

    conda remove -n myenv VAMToolbox

To remove the toolbox from the current environment: 

.. code:: console

    conda remove VAMToolbox

