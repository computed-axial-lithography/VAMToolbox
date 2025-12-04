.. _examples_3Doptimization:

###############
3D optimization
###############

***********
Walkthrough
***********

Create a :py:class:`~vamtoolbox.geometry.TargetGeometry` from a .stl file by specifying the ``stlfilename`` and ``resolution`` i.e. the number of slices in the z-axis and show the target with the :py:meth:`~vamtoolbox.geometry.Volume.show` method.

.. literalinclude:: ../../../examples/3Doptimization.py
    :lines: 1-5

.. image:: /_images/examples/3Doptimization/target.png


Create the :py:class:`~vamtoolbox.geometry.ProjectionGeometry` object. First, the ``angles`` array is created by using `numpy.linspace <https://numpy.org/doc/stable/reference/generated/numpy.linspace.html>`_ to create 1D array of evenly spaced angles at which to perform projection. 

.. literalinclude:: ../../../examples/3Doptimization.py
    :lines: 7-9

Create an :py:class:`vamtoolbox.optimize.Options` object and run optimization. The :py:class:`~vamtoolbox.optimize.Options` object holds the parameters used by the :py:func:`~vamtoolbox.optimize.optimize` function.

.. literalinclude:: ../../../examples/3Doptimization.py
    :lines: 11-12

.. image:: /_images/examples/3Doptimization/recon.png

.. image:: /_images/examples/3Doptimization/sino.png

.. tip:: Hover the mouse pointer over either slice and scroll to slice through the 3D voxel array at different z and x indices.

Alternatively, the `vedo <https://vedo.embl.es/autodocs/content/vedo/index.html>`_ plotting package may be used to display the optimized reconstruction array (``opt_recon.array``). The `RayCastPlotter application <https://vedo.embl.es/autodocs/content/vedo/applications.html#vedo.applications.RayCastPlotter>`_ works well for customizable 3D display of the reconstruction array. 

.. literalinclude:: ../../../examples/3Doptimization
    :lines: 16-19

.. image:: /_images/examples/3Doptimization/vedo_recon.png


************
Example file
************

.. literalinclude:: ../../../examples/3Doptimization.py
    :caption: examples/3Doptimization.py