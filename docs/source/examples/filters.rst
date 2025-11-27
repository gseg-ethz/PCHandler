Filters
=======

This page demonstrates practical filtering using NumPy masks together with
PCHandler’s sampling utilities.

Axis-aligned box filter
-----------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   pcd = PointCloudData(np.random.rand(10_000, 3), numerical_optimization_shift=None)

   # Keep points inside [0.2, 0.8] along each axis
   lo, hi = 0.2, 0.8
   mask = (
       (pcd.xyz[:, 0] >= lo) & (pcd.xyz[:, 0] <= hi) &
       (pcd.xyz[:, 1] >= lo) & (pcd.xyz[:, 1] <= hi) &
       (pcd.xyz[:, 2] >= lo) & (pcd.xyz[:, 2] <= hi)
   )

   cropped = pcd.sample(mask)

Spherical region filter
-----------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   pcd = PointCloudData(np.random.rand(50_000, 3), numerical_optimization_shift=None)

   center = np.array([0.5, 0.5, 0.5])
   radius = 0.25

   d2 = np.sum((pcd.xyz - center) ** 2, axis=1)
   mask = d2 <= (radius ** 2)

   inside = pcd.sample(mask)

Outlier removal by simple z-score
---------------------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   pcd = PointCloudData(np.random.randn(30_000, 3), numerical_optimization_shift=None)

   # Simple per-axis z-score filter
   mean = pcd.xyz.mean(axis=0)
   std = pcd.xyz.std(axis=0) + 1e-12
   z = (pcd.xyz - mean) / std

   thresh = 3.0
   mask = np.all(np.abs(z) < thresh, axis=1)

   filtered = pcd.sample(mask)
