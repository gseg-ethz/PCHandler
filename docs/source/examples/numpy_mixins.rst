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
   arr = pcd.arr           # alias for positions array
   coords = pcd.xyz        # same coordinate data

   assert np.all(arr == coords)

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

   # Center the cloud (create a modified copy)
   centered = pcd.copy(pcd.xyz - pcd.xyz.mean(axis=0))


