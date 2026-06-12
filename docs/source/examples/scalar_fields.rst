Scalar Fields
=============

Attach, manipulate, and move scalar fields alongside coordinates.

Creating with scalar fields
---------------------------
There are multiple ways of creating scalar fields.
First is a dictionary can be passed upon initialisation of the object:

.. code-block:: python

    import numpy as np
    from pchandler.core import PointCloudData

    N = 10
    xyz = np.random.rand(N, 3)

    fields = {
       "intensity": np.random.rand(N),      # scalar field
       "quality": np.linspace(0, 1, N),     # another scalar field
    }

    pcd = PointCloudData(
       xyz=xyz,
       scalar_fields=fields
    )

Also, it's possible to add them directly to the scalar field manager.

.. code-block:: python

    from pchandler.scalar_fields import ScalarField, RGBFields

    # Via the "create_field()" method of the ScalarFieldsManager / "scalar_fields" property
    pcd.scalar_fields.create_field('new_field', np.random.rand(N))

    # Adding an existing ScalarField object
    new_displacement_field = ScalarField(np.random.rand(N) * 50, name='displacement')

    pcd.scalar_fields.add_field(new_displacement_field)

    # Or like adding a key-value pair to a dictionary
    pcd.scalar_fields['displacements'] = np.random.rand(N)

    pcd.scalar_fields['rgb'] = np.random.randint(0, 255, (N, 3)).astype(np.uint8)


Dedicated Scalar Fields
-----------------------

There are a number of special scalar fields where they have a dedicated namespace:

- RGB (see :py:data:`pchandler.constants.RGB_NAMES`)
- Normals (see :py:data:`pchandler.constants.NORMAL_NAMES`)
- Intensity (see :py:data:`pchandler.constants.INTENSITY_NAMES`)
- Reflectance (see :py:data:`pchandler.constants.REFLECTANCE_NAMES`)

These can then be accessed directly from the PointCloudData base object like this

.. code-block:: python

    pcd.rgb
    pcd.r
    pcd.normals # All normals
    pcd.ny      # Normal components on the y axis
    pcd.intensity
    pcd.reflectance

Setting RGB and normals
-----------------------
Likewise, these values can also be updated at the root level
.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   N = 5
   pcd = PointCloudData(np.random.rand(N, 3), numerical_optimization_shift=None)

   # RGB can be float in [0, 1] or uint8 in [0, 255]
   pcd.rgb = (np.random.rand(N, 3)).astype(float)

   # Normals: Nx3 unit vectors (you may want to normalize your inputs first)
   normals = np.random.randn(N, 3).astype(float)
   pcd.normals = normals


.. note::
    | All normal values will be scaled to unit vectors
    | ``np.allclose(np.linalg.norm(pcd.normals, axis=1), 1)``


Access and modify scalar fields
-------------------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   N = 8
   pcd = PointCloudData(
       np.random.rand(N, 3),
       scalar_fields={
            "custom_field": np.random.rand(N),
            "intensity": np.random.rand(N),
            "field_2": np.random.rand(N)
       }
   )

   # Write or replace
   pcd.scalar_fields['custom_field'] = np.linspace(0.0, 1.0, N)
   print(f"Number of fields before: {len(pcd.scalar_fields)}")

   # Removal can be done by setting to None or the ``remove_field()`` method
   pcd.scalar_fields.remove_field('custom_field')
   pcd.intensity = None
   pcd.scalar_fields['field_2'] = None

   print(f"Number of fields after: {len(pcd.scalar_fields)}")



Sampling preserves fields
-------------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   N = 100
   pcd = PointCloudData(
       np.random.rand(N, 3),
       scalar_fields={
           "intensity": np.random.rand(N),
           "class_id": np.random.randint(0, 5, size=N),
       },
       numerical_optimization_shift=None,
   )

   mask = pcd.xyz[:, 2] > 0.5
   subset = pcd.sample(mask)

   # All included fields are sampled consistently
   assert len(subset) == mask.sum()

   print(subset.intensity)
