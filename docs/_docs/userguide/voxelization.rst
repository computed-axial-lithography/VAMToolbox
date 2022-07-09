.. _userguide_voxelization:

############
Voxelization
############

**************
Model position
**************

The position of the .stl model in the x and y axes determine where in the voxel domain the voxelized model will be created. The distance from the x-y plane (i.e. the position along the z-axis) does **not** affect the voxelized model position. This is because it is assumed that the tomographic rotation axis or the axis that the simulated detector/projector rotates around is along coaxial with the z-axis. For standard parallel ray projector geometry, the z-coordinate of the model does not affect the resulting sinogram, however, the x-y-coordinate affects the sinogram if attenuation is present. For cone beam projector geometry, the z-coordinate does affect the sinogram even without attenuation present. The VAMToolbox does enable cone beam projector geometries (via Astra toolbox) but it is not yet well equipped handle advanced cone beam geometries. Therefore, to make the voxelization generalizable, it is assumed that the x-y-coordinate affects the voxelization but the z-coordinate does not.

To illustrate, two schematics of a cylinder are shown. 

* Left: the model is displaced away from the z-axis (e.g. in the x-y plane) 
* Right: the model is located on the z-axis (e.g. the axis of revolution of the cylinder is coaxial with the global z-axis)

.. image:: /_images/userguide/onoffaxismodels.png

When the on-axis model is voxelized when a :py:class:`~vamtoolbox.geometry.TargetGeometry` is created, the voxelized cylinder is created in the center of the :py:attr:`~vamtoolbox.geometry.TargetGeometry.array`.


.. image:: /_images/userguide/onaxiscylinder_voxels.png

When the off-axis model is voxelized, the voxelized cylinder is created according to the models position in relation to its global origin. Here, that means the cylinder is created in the first quadrant (i.e. +x, +y) because it was modeled in a CAD software in the first quadrant. Note that the voxelized cylinder appears smaller but it is actually the same size as the on-axis model but the voxel domain is larger. 

.. image:: /_images/userguide/offaxiscylinder_voxels.png

.. warning:: 

    When an off-axis model is voxelized, it requires a voxel domain sufficiently large to enclose the non-zero voxels within the *inscribed cylinder*. The inscribed cylinder is an imaginary region that the tomographic projector sweeps out over a complete rotation. Anything outside of the inscribed cylinder cannot be accurately reconstructed due to missing sinogram data in certain angular ranges. For large off-axis offsets, the requirement that the non-zero voxels must reside within the inscribed cylinder can lead to very large voxel domains and consequently very large memory requirements to store the voxel array.

    .. image:: /_images/userguide/inscribedcylinder.png


**********************
Multibody voxelization
**********************

To be added