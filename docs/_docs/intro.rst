.. _intro:

############
Introduction
############

**********
Background
**********

Volumetric additive manufacturing (VAM) is a form of resin-based additive manufacturing or 3D printing which builds objects volumetrically, or all at once, as opposed to layer-by-layer like conventional stereolithography :cite:`Shusteff2017b,Kelly2019a,Loterie2020a`. Tomographic VAM (also known as computed axial lithography (CAL)) is a subset of VAM which functions like inverse X-ray computed tomography and intensity modulated radiation therapy :cite:`Kelly2019a,Bernal2019,Loterie2020a,Cook2020,Bhattacharya2021,Rackson2021,Orth2021,Schwartz2022,Rackson2022,Wang2022,Kollep2022,Toombs2022`. A photosensitive material is exposed by digital light projections from many azimuthal angles such that that the cumulative light dose surpasses a dose threshold and solidifies or makes the material insoluble in developer solution in a prescribed volume. 

Computational optimization or filtering of the digital light projections is required to minimize the background exposure of the material surrounding the prescribed target volume. Increasing the light dose contrast means that the printed object can be more easily removed from the surrounding material. Several optimization algorithms have been developed, from traditional Ram-Lak filtering to second order methods :cite:`Loterie2020a,Kelly2019a,Rackson2021,Bhattacharya2021`. These are described in more detail in :ref:`userguide_optimization`.

********
Citation
********

If you use this package in your research, please cite it as ... TBD ...


If you use any of the following optimization algorithms in your research, please cite the corresponding research:

* **CAL**: :cite:p:`Kelly2019a`
* **PM**: :cite:p:`Bhattacharya2021`
* **OSMO**: :cite:p:`Rackson2021`

************
Bibliography
************

.. bibliography::
    :style: unsrt


.. _`[Kelly2019a]`: https://doi.org/10.1126/science.aau7114
.. _`[Bhattacharya2021]`: https://doi.org/10.1016/j.addma.2021.102299
.. _`[Rackson2021]`: https://doi.org/10.1016/j.addma.2021.102367 
