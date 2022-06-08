.. _examples_voxelizestl:

###############
Voxelize a .stl
###############

***********
Walkthrough
***********

Create a :py:class:`vamtoolbox.geometry.TargetGeometry` from a .stl file by specifying the ``stlfilename`` and ``resolution`` i.e. the number of slices in the z-axis.

.. literalinclude:: ../../../examples/voxelizestl.py
    :lines: 1-4

Show the voxel array of the :py:class:`vamtoolbox.geometry.TargetGeometry` instance (:py:attr:`TargetGeometry.array`) with :py:meth:`TargetGeometry.show`.


.. literalinclude:: ../../../examples/voxelizestl.py
    :lines: 6

.. tip:: Hover the mouse pointer over either slice and scroll to slice through the 3D voxel array at different z and x indices.

.. image:: /_images/examples/voxelizestl/toolbox.png


Alternatively, the `vedo <https://vedo.embl.es/autodocs/content/vedo/index.html>`_ plotting package may be used to display the voxel array.

.. literalinclude:: ../../../examples/voxelizestl.py
    :lines: 8-10

.. image:: /_images/examples/voxelizestl/vedo.png

************
Example file
************

.. literalinclude:: ../../../examples/voxelizestl.py
    :caption: examples/voxelizestl.py