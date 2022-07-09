.. _examples_2Doptimization :

###############
2D optimization
###############

***********
Walkthrough
***********

Create the :py:class:`~vamtoolbox.geometry.TargetGeometry` object. In this case, the ``imagefilename`` keyword argument is specified to create the object from an image file. The ``pixels`` keyword argument specifies the width of the desired *square* 2D target array. Show the target with the :py:meth:`~vamtoolbox.geometry.Volume.show` method.

.. literalinclude:: ../../../examples/2Doptimization.py
    :lines: 1-5

.. image:: /_images/examples/2Doptimization/target.png

Create the :py:class:`~vamtoolbox.geometry.ProjectionGeometry` object. First, the ``angles`` array is created by using `numpy.linspace <https://numpy.org/doc/stable/reference/generated/numpy.linspace.html>`_ to create 1D array of evenly spaced angles at which to perform projection. 

.. literalinclude:: ../../../examples/2Doptimization.py
    :lines: 7-9

Create an :py:class:`vamtoolbox.optimize.Options` object and run optimization. The :py:class:`~vamtoolbox.optimize.Options` object holds the parameters used by the :py:func:`~vamtoolbox.optimize.optimize` function.

.. literalinclude:: ../../../examples/2Doptimization.py
    :lines: 11-12

Show the optimized reconstruction and sinogram with the :py:meth:`~vamtoolbox.geometry.Volume.show` method.

.. literalinclude:: ../../../examples/2Doptimization.py
    :lines: 13-14  

.. image:: /_images/examples/2Doptimization/recon.png

.. image:: /_images/examples/2Doptimization/sino.png

************
Example file
************

.. literalinclude:: ../../../examples/2Doptimization.py
    :caption: examples/2Doptimization.py