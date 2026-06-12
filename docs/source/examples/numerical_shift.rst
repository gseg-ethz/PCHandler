Numerical Optimization Shift
============================

The numerical optimization shift parameter is useful for managing very large
coordinate magnitudes while maintaining good numerical stability. You can pass
it at construction time and turn it off (set to ``None``) when you need the
raw values (e.g., for export).

Construct with or without a shift
---------------------------------

.. code-block:: python

    import numpy as np
    from pchandler.core import PointCloudData
    from pchandler.geometry.optimal_shift import OptimizedShift

    xyz = (np.random.rand(1000, 3) * 1e3).astype(float)

    # Explicitly disable shifting
    pcd = PointCloudData(xyz=xyz, numerical_optimization_shift=None)

    pcd = PointCloudData(xyz=xyz, numerical_optimization_shift=OptimizedShift([500, 200, 1000]))

A warning will be thrown if the shift is not feasible.

.. code-block:: python

    xyz = (np.random.rand(1000, 3) * 1e6).astype(float)
    pcd = PointCloudData(xyz=xyz,
                         numerical_optimization_shift=OptimizedShift([500, 200, 1000]))

Exporting with Open3D ignores the shift
---------------------------------------

When exporting, a copy is made without the shift so that downstream libraries
see the unshifted coordinates.

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   xyz = (np.random.rand(500, 3) + 60_000).astype(float)
   pcd = PointCloudData(xyz=xyz)

   try:
       # If a shift were active, to_o3d would transparently export unshifted coords
       o3d_pcd = pcd.to_o3d(as_tensor=False)
   except ModuleNotFoundError:
       pass

   assert not np.allclose(xyz, pcd) # PCD has been shifted so shouldn't match
   assert np.allclose(xyz, o3d_pcd.points)  # But should match to the original coordinates
