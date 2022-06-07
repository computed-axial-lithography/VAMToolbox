.. _examples_optimizationcomparison :

#######################
Optimization comparison
#######################

***********
Walkthrough
***********

Import VAMToolbox modules and numpy. Create the target Create the :py:class:`VAMToolbox.geometry.TargetGeometry` object and the :py:class:`VAMToolbox.geometry.ProjectionGeometry` object.

.. literalinclude:: ../../../examples/optimizationcomparison.py
    :lines: 1-10

Filtered back projection
========================

.. literalinclude:: ../../../examples/optimizationcomparison.py
    :lines: 12-15

CAL 
====
:cite:p:`Kelly2019a`

.. literalinclude:: ../../../examples/optimizationcomparison.py
    :lines: 17-20

Penalty minimization 
====================
:cite:p:`Bhattacharya2021`

.. literalinclude:: ../../../examples/optimizationcomparison.py
    :lines: 22-25

Object space model optimization
===============================
 :cite:p:`Rackson2021`
 
.. literalinclude:: ../../../examples/optimizationcomparison.py
    :lines: 27-30


.. image:: /_images/examples/optimizers/allopts.png



************
Example file
************

.. literalinclude:: ../../../examples/optimizationcomparison.py
    :caption: examples/optimizationcomparison.py