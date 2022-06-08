.. _examples_3Doverprint:

#########################
3D overprint optimization
#########################

***********
Walkthrough
***********

.. image:: /_images/examples/3Doverprint/overprintbodies.png
    :height: 300px

Create a :py:class:`vamtoolbox.geometry.TargetGeometry` from a .stl file by specifying the ``stlfilename`` and ``resolution`` i.e. the number of slices in the z-axis. When overprint optimization is desired, use the ``bodies`` kwarg to specify which bodies are to be printed (i.e. a dictionary specifying which bodies are to be printed and which are the insert/s). In this case, the body to printed is body 2 and the body which is the insert is body 1.

.. note::
    The 3D CAD model should be created with separate bodies e.g. when extruding a new feature, you can choose to create a new body instead of creating a union with the existing body and the new feature. This has been tested with Autodesk Inventor and Solidworks. Some 3D solid modeling software may not have this capability. 

.. literalinclude:: ../../../examples/3Doverprint.py
    :lines: 1-4


Finally, show the target with the :py:meth:`vamtoolbox.geometry.Volume.show` method and use the ``show_bodies`` kwarg to show the insert bodies in a red color.

.. literalinclude:: ../../../examples/3Doverprint.py
    :lines: 5

.. image:: /_images/examples/3Doverprint/target.png


Create the :py:class:`vamtoolbox.geometry.ProjectionGeometry` object. First, the ``angles`` array is created by using `numpy.linspace <https://numpy.org/doc/stable/reference/generated/numpy.linspace.html>`_ to create 1D array of evenly spaced angles at which to perform projection. 

.. literalinclude:: ../../../examples/3Doverprint.py
    :lines: 7-9

Create an :py:class:`vamtoolbox.optimize.Options` object and run optimization. The :py:class:`vamtoolbox.optimize.Options` object holds the parameters used by the :py:func:`vamtoolbox.optimize.optimize` function.

.. literalinclude:: ../../../examples/3Doverprint.py
    :lines: 11-14

.. image:: /_images/examples/3Doverprint/recon.png

.. image:: /_images/examples/3Doverprint/sino.png

.. tip:: Hover the mouse pointer over either slice and scroll to slice through the 3D voxel array at different z and x indices.



************
Example file
************

.. literalinclude:: ../../../examples/3Doverprint.py
    :caption: examples/3Doverprint.py