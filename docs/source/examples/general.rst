General
=======

Sampling, reduction and extraction
----------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   pcd = PointCloudData(
       xyz=np.random.rand(100, 3),
       numerical_optimization_shift=None,
   )

   # Boolean mask: keep points with x > 0.5
   mask = pcd.xyz[:, 0] > 0.5

   # Sample returns a new object
   sample = pcd.sample(mask)

   # Reduce modifies in place
   pcd.reduce(mask)

   # Extract removes the selected points from the object
   extracted = pcd.extract(mask)


Merging clouds
--------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   a = PointCloudData(np.random.rand(50, 3), numerical_optimization_shift=None)
   b = PointCloudData(np.random.rand(75, 3) + 1.0, numerical_optimization_shift=None)

   merged = PointCloudData.merge(a, b)



Create a point cloud from NumPy
-------------------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   # Minimal example positions (N x 3)
   xyz = np.array([
       [0.0, 0.0, 0.0],
       [1.0, 0.0, 0.0],
       [0.0, 1.0, 0.0],
   ], dtype=float)

   # Both of these are supported
   pcd = PointCloudData(xyz=xyz, numerical_optimization_shift=None)
   pcd2 = PointCloudData(arr=xyz, numerical_optimization_shift=None)  # alias for xyz

Roundtrip with Open3D (optional)
--------------------------------

If Open3D is installed, you can convert to/from Open3D point clouds.

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   xyz = np.random.rand(1000, 3).astype(float)
   pcd = PointCloudData(xyz=xyz, numerical_optimization_shift=None)

   try:
       # Convert to legacy Open3D geometry
       o3d_pcd = pcd.to_o3d(as_tensor=False)

       # ... e.g. write using open3d if desired:
       # import open3d as o3d
       # o3d.io.write_point_cloud("cloud.ply", o3d_pcd)

       # Convert back
       pcd_back = PointCloudData.from_o3d(o3d_pcd)
   except ModuleNotFoundError:
       # Open3D isn't available; skip this example at runtime
       pass

Tensor Open3D roundtrip (optional)
----------------------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   xyz = (np.random.rand(500, 3) * 10).astype(np.float32)
   pcd = PointCloudData(xyz=xyz, numerical_optimization_shift=None)

   try:
       # Convert to tensor-based Open3D geometry
       o3d_tensor_pcd = pcd.to_o3d(as_tensor=True)

       # Back to PointCloudData
       pcd_back = PointCloudData.from_o3d(o3d_tensor_pcd)
   except ModuleNotFoundError:
       pass

Interoperability with py4dgeo (optional)
----------------------------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   xyz = np.random.randn(200, 3).astype(float)
   pcd = PointCloudData(xyz=xyz, numerical_optimization_shift=None)

   try:
       epoch = pcd.to_py4dgeo()

       # Back to PCHandler
       pcd_again = PointCloudData.from_py4dgeo(epoch)
   except ModuleNotFoundError:
       # py4dgeo isn't available; skip
       pass
