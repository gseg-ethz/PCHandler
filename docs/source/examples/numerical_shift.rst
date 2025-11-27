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

   xyz = (np.random.rand(1000, 3) * 1e6).astype(float)

   # Explicitly disable shifting
   pcd = PointCloudData(xyz=xyz, numerical_optimization_shift=None)

Exporting with Open3D ignores the shift
---------------------------------------

When exporting, a copy is made without the shift so that downstream libraries
see the unshifted coordinates.

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData

   xyz = (np.random.rand(500, 3) * 1e4).astype(float)
   pcd = PointCloudData(xyz=xyz, numerical_optimization_shift=None)

   try:
       # If a shift were active, to_o3d would transparently export unshifted coords
       o3d_pcd = pcd.to_o3d(as_tensor=False)
   except ModuleNotFoundError:
       pass
