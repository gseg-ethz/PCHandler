Numpy Functionality
===================


PointCloudData integrates naturally with NumPy arrays for vectorized operations.

Basic construction and array-style usage
----------------------------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   xyz = np.random.rand(5, 3)
   pcd = PointCloudData(xyz=xyz, numerical_optimization_shift=None)

   # Length equals number of points
   n = len(pcd)

   # Access underlying coordinates
   coords = pcd.xyz.copy()        # same coordinate data

   assert np.all(pcd == coords)
   print(np.mean(pcd, axis=0))

Vectorized computations
-----------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   pcd = PointCloudData(
       xyz=np.array([[0, 0, 0],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]], dtype=float),
       numerical_optimization_shift=None,
   )

   # Distances from origin
   d = np.linalg.norm(pcd.xyz, axis=1)

   # Center the point cloud
   pcd -= np.mean(pcd, axis=0)
   pcd2 = pcd + d[:, None]

   assert not np.allclose(pcd.xyz, pcd2.xyz)

   assert isinstance(pcd, PointCloudData)
   print(pcd.xyz)
