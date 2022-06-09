.. _setup:

#####
Setup
#####

************
Installation
************

A prerequiste to install VAMToolbox is to install the `Anaconda <https://www.anaconda.com/products/distribution>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ package manager.


Conda
=====
The VAMToolbox is available at the `VAMToolbox channel <https://anaconda.org/VAMToolbox>`_ in the conda package manager. Run the following inside a conda environment:

.. code:: console
    
    conda install vamtoolbox -c vamtoolbox -c conda-forge -c astra-toolbox

.. warning:: 
    
    This command is different than what Anaconda.org suggests. This is because to properly install the dependencies you must tell conda to search in the astra-toolbox and vamtoolbox channels (in addition to the conda-forge channel, this is a default channel but is added to be explicit).


Astra Toolbox
=============

If there are problems with the installation of Astra Toolbox go to the `downloads <https://www.astra-toolbox.com/downloads/index.html#downloads>`_ page and install the latest Python version by following the `installation instructions <https://www.astra-toolbox.com/docs/install.html#installation-instructions>`_.


***********************
Updating or downgrading
***********************

For updating, type into the Anaconda prompt:

.. code:: console

    conda update vamtoolbox -c vamtoolbox -c conda-forge -c astra-toolbox

For downgrading, type into the Anaconda prompt replacing ``x.x.x`` with the version to downgrade to:

.. code:: console

    conda install vamtoolbox==x.x.x -c vamtoolbox -c conda-forge -c astra-toolbox

.. note::

    The same channel specifications apply here, like in installation.



**************
Uninstallation
**************

Use the Anaconda prompt for one of the following steps.

To remove the toolbox from the base environment (where ``myenv`` is replaced by the name of the environment in which VAMToolbox is installed): 

.. code:: console

    conda remove -n myenv vamtoolbox --all

To remove the toolbox from the current environment: 

.. code:: console

    conda remove vamtoolbox --all

