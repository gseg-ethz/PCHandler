3D Transformations
==================

Below are common workflows for transforming point-cloud coordinates with NumPy.
All scalar fields follow their corresponding points when you use PCHandler’s
copy/sample/reduce utilities.

Translate and scale
-------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   pcd = PointCloudData(np.random.rand(100, 3), numerical_optimization_shift=None)

   # Translate by a vector
   t = np.array([10.0, -2.0, 3.5])
   translated = pcd.copy(pcd.xyz + t)

   # Uniform scale
   s = 2.0
   scaled = pcd.copy(pcd.xyz * s)

Rigid transform with a rotation matrix
--------------------------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   def rot_z(theta):
       c, s = np.cos(theta), np.sin(theta)
       return np.array([[ c, -s, 0],
                        [ s,  c, 0],
                        [ 0,  0, 1]], dtype=float)

   pcd = PointCloudData(np.random.rand(100, 3), numerical_optimization_shift=None)

   R = rot_z(np.deg2rad(30.0))
   t = np.array([1.0, 0.0, 0.0])

   # Apply x' = R x + t
   transformed = pcd.copy((pcd.xyz @ R.T) + t)

Merging transformed clouds
--------------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   a = PointCloudData(np.random.rand(200, 3), numerical_optimization_shift=None)
   b = a.copy(a.xyz + np.array([5.0, 0.0, 0.0]))

   merged = PointCloudData.merge(a, b)
