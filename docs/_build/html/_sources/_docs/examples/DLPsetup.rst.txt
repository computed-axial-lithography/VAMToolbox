.. _examples_DLPsetup:

#########
DLP setup
#########

***********
Walkthrough
***********

Focus
=====

Show a Siemen's star for focus adjustment with :py:class:`VAMToolbox.DLP.Setup.Focus`. The number of slices in the star can be changed with the ``slices`` argument.

.. literalinclude:: ../../../examples/DLPsetup.py
    :lines: 1-4

.. image:: /_images/examples/DLPsetup/siemens_star.png



Axis alignment
==============

Show 3 lines for axis alignment with :py:class:`VAMToolbox.DLP.Setup.AxisAlignment`. The line thickness, line separation, and line u-axis center offset can be changed with the arguments. 

.. literalinclude:: ../../../examples/DLPsetup.py
    :lines: 6

.. tip:: Use the up/down arrow keys to change line thickness, left/right arrow keys to change line separation, and comma/period keys to change the line center offset while the displaying on the DLP device.

.. image:: /_images/examples/DLPsetup/axis_alignment.png


************
Example file
************

.. literalinclude:: ../../../examples/DLPsetup.py
    :caption: examples/DLPsetup.py