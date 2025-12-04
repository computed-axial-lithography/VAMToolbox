.. _examples_3Dzerodoseconstraint:

#########################################
3D optimization with zero dose constraint
#########################################

***********
Walkthrough
***********

.. image:: /_images/examples/3Dzerodoseconstraint/bodies.png
    :height: 300px

Create a :py:class:`~vamtoolbox.geometry.TargetGeometry` from a .stl file by specifying the ``stlfilename`` and ``resolution`` i.e. the number of slices in the z-axis. When a zero dose constraint is desired, use the ``bodies`` kwarg to specify which bodies are to be printed (i.e. a dictionary specifying which bodies are to be printed and which are the zero dose constraint/s). In this case, the body to printed is body 1 and the body which is the zero dose constraint is body 2.

.. literalinclude:: ../../../examples/3Dzerodoseconstraint.py
    :lines: 1-4

Finally, show the target with the :py:meth:`~vamtoolbox.geometry.Volume.show` method and use the ``show_bodies`` kwarg to show the insert bodies in a green color.

.. literalinclude:: ../../../examples/3Dzerodoseconstraint.py
    :lines: 5

.. image:: /_images/examples/3Dzerodoseconstraint/target.png


Create the :py:class:`~vamtoolbox.geometry.ProjectionGeometry` object. First, the ``angles`` array is created by using `numpy.linspace <https://numpy.org/doc/stable/reference/generated/numpy.linspace.html>`_ to create 1D array of evenly spaced angles at which to perform projection. 

.. literalinclude:: ../../../examples/3Dzerodoseconstraint.py
    :lines: 7-9

Create an :py:class:`vamtoolbox.optimize.Options` object and run optimization. The :py:class:`~vamtoolbox.optimize.Options` object holds the parameters used by the :py:func:`~vamtoolbox.optimize.optimize` function.

.. literalinclude:: ../../../examples/3Dzerodoseconstraint.py
    :lines: 11-14

.. image:: /_images/examples/3Dzerodoseconstraint/recon.png

.. image:: /_images/examples/3Dzerodoseconstraint/sino.png

.. tip:: Hover the mouse pointer over either slice and scroll to slice through the 3D voxel array at different z and x indices.



************
Example file
************

.. literalinclude:: ../../../examples/3Dzerodoseconstraint.py
    :caption: examples/3Dzerodoseconstraint.py