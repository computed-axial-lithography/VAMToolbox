.. _examples_optimizationcomparison :

#######################
Optimization comparison
#######################

***********
Walkthrough
***********

Import VAMToolbox and numpy. Create the target Create the :py:class:`vamtoolbox.geometry.TargetGeometry` object and the :py:class:`vamtoolbox.geometry.ProjectionGeometry` object.

.. literalinclude:: ../../../examples/optimizationcomparison.py
    :lines: 1-9

Filtered back projection
========================

.. literalinclude:: ../../../examples/optimizationcomparison.py
    :lines: 10-13

CAL 
====
:cite:p:`Kelly2019a`

.. literalinclude:: ../../../examples/optimizationcomparison.py
    :lines: 15-18

Penalty minimization 
====================
:cite:p:`Bhattacharya2021`

.. literalinclude:: ../../../examples/optimizationcomparison.py
    :lines: 20-23

Object space model optimization
===============================
:cite:p:`Rackson2021`
 
.. literalinclude:: ../../../examples/optimizationcomparison.py
    :lines: 25-28


.. image:: /_images/examples/optimizers/allopts.png



************
Example file
************

.. literalinclude:: ../../../examples/optimizationcomparison.py
    :caption: examples/optimizationcomparison.py