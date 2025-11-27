Scalar Fields
=============

Attach, manipulate, and move scalar fields alongside coordinates.

Creating with scalar fields
---------------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   N = 10
   xyz = np.random.rand(N, 3)

   # Provide fields via dict: scalars or Nx3 triplets
   fields = {
       "intensity": np.random.rand(N),      # scalar field
       "quality": np.linspace(0, 1, N),     # another scalar field
   }

   pcd = PointCloudData(
       xyz=xyz,
       scalar_fields=fields,
       numerical_optimization_shift=None,
   )

Setting RGB and normals
-----------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   N = 5
   pcd = PointCloudData(np.random.rand(N, 3), numerical_optimization_shift=None)

   # RGB can be float in [0, 1] or uint8 in [0, 255]
   pcd.rgb = (np.random.rand(N, 3)).astype(float)

   # Normals: Nx3 unit vectors (you may want to normalize your inputs first)
   normals = np.random.randn(N, 3).astype(float)
   normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
   pcd.normals = normals

Access and modify scalar fields
-------------------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   N = 8
   pcd = PointCloudData(
       np.random.rand(N, 3),
       scalar_fields={"intensity": np.random.rand(N)},
       numerical_optimization_shift=None,
   )

   # Read
   intens = pcd.intensity            # property for a common scalar field name

   # Write or replace
   pcd.intensity = np.linspace(0.0, 1.0, N)

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
