Filters
=======

This page demonstrates practical filtering using the high-level classes from
``pchandler.filters``. These filters produce boolean masks and provide the
convenience methods ``sample``, ``reduce``, and ``extract`` via the
:class:`pchandler.core.PointCloudFilter` base class.

Quick reference
---------------

- ``f.mask(pcd)`` → boolean mask (NumPy array of shape ``(N,)``)
- ``f.sample(pcd)`` → returns a new :class:`pchandler.core.PointCloudData` with points where mask is True
- ``f.reduce(pcd)`` → in-place reduction of ``pcd`` to points where mask is True
- ``f.extract(pcd)`` → returns selected points; the original ``pcd`` is reduced to the complement


Axis-aligned box filtering (Cartesian)
--------------------------------------

Use :class:`pchandler.filters.BoxFilter` to keep points inside a 3D axis-aligned box.
The box can be evaluated in the point cloud’s local or global frame.

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData
   from pchandler.filters import BoxFilter

   # Synthetic point cloud in [0, 1]^3
   pcd = PointCloudData(np.random.rand(50_000, 3), numerical_optimization_shift=None)

   # Define a box: [0.2, 0.8] along each axis
   minimum = np.array([0.2, 0.2, 0.2], dtype=float)
   maximum = np.array([0.8, 0.8, 0.8], dtype=float)
   box = BoxFilter(minimum=minimum, maximum=maximum)

   # 1) Get a sampled copy containing only inside points
   cropped = box.sample(pcd)

   # 2) In-place reduction (pcd is modified)
   box.reduce(pcd)

   # 3) Extract inside points into a new object; 'pcd' becomes the outside points
   # Note: after this call, 'pcd' is reduced to points outside the box.
   pcd = PointCloudData(np.random.rand(50_000, 3), numerical_optimization_shift=None)
   inside = box.extract(pcd)


Range filtering (spherical radius)
----------------------------------

Use :class:`pchandler.filters.RangeFilter` to keep points by radial distance.
For spherical quantities (radius, horizontal, vertical angles) make sure to
define a scan origin (SOCS) so spherical coordinates are meaningful.

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData
   from pchandler.filters import RangeFilter

   # Place points roughly within a sphere around the origin
   rng = np.random.default_rng(0)
   xyz = rng.normal(size=(80_000, 3)).astype(float)

   # Provide an SOCS origin so spherical coordinates (r, hz, v) are available
   pcd = PointCloudData(xyz, socs_origin=np.zeros(3), numerical_optimization_shift=None)

   # Keep points with 0.5 m <= radius <= 1.5 m
   rf = RangeFilter(low=0.5, high=1.5)

   # Sample a copy using the filter
   shell = rf.sample(pcd)

   # You can also work with the raw mask if you need to combine filters
   mask_r = rf.mask(pcd)           # boolean array, shape (N,)


Field-of-view (FoV) clipping (spherical angles)
-----------------------------------------------

Use :class:`pchandler.filters.FoVFilter` to select points by sensor field of view.
This filter evaluates horizontal and vertical angles from spherical coordinates.

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData
   from pchandler.filters import FoVFilter
   from pchandler.geometry import FoV  # FoV(top, bottom, left, right) in radians

   # Example cloud with defined SOCS for spherical coordinates
   pcd = PointCloudData(np.random.rand(60_000, 3) - 0.5,
                        socs_origin=np.zeros(3),
                        numerical_optimization_shift=None)

   # Define a rectangular FoV in spherical angles:
   # vertical in [-30°, +10°], horizontal in [-60°, +60°]
   fov = FoV(
       top=np.deg2rad(10.0),
       bottom=np.deg2rad(120.0),
       left=np.deg2rad(-60.0),
       right=np.deg2rad(60.0),
   )
   fov_filter = FoVFilter(fov=fov)

   in_fov = fov_filter.sample(pcd)  # copy containing only points within the FoV


Combining multiple filters
--------------------------

You can intersect, union, or subtract filters by composing their masks before
sampling or reducing.

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData
   from pchandler.filters import BoxFilter, RangeFilter

   pcd = PointCloudData(np.random.rand(100_000, 3), socs_origin=np.zeros(3), numerical_optimization_shift=None)

   # Box in Cartesian space
   box = BoxFilter(minimum=np.array([0.1, 0.1, 0.1]), maximum=np.array([0.9, 0.9, 0.9]))

   # Radial shell in spherical space
   shell = RangeFilter(low=0.4, high=1.0)

   # Intersection: inside box AND within radial shell
   m = box.mask(pcd) & shell.mask(pcd)
   subset = pcd.sample(m)

   # Difference: inside box but NOT within the shell
   outside_shell = pcd.sample(box.mask(pcd) & ~shell.mask(pcd))


Custom filters
--------------

You can implement a custom filter by subclassing
:class:`pchandler.core.PointCloudFilter` and overriding ``mask(pcd)``.
The resulting filter will automatically support ``sample``, ``reduce``, and ``extract``.

.. code-block:: python

   import numpy as np
   from typing import Any
   from pchandler.core import PointCloudData
   from pchandler.filters.core import PointCloudFilter

   class ZScoreOutlierFilter(PointCloudFilter):
       def __init__(self, thresh: float = 3.0):
           self.thresh = float(thresh)

       def mask(self, pcd: PointCloudData) -> np.ndarray:
           # keep points whose per-axis z-score is within +/- thresh
           mean = pcd.xyz.mean(axis=0)
           std = pcd.xyz.std(axis=0) + 1e-12
           z = (pcd.xyz - mean) / std
           return np.all(np.abs(z) < self.thresh, axis=1)

   # Usage
   pcd = PointCloudData(np.random.randn(30_000, 3), numerical_optimization_shift=None)
   zf = ZScoreOutlierFilter(thresh=3.0)
   filtered = zf.sample(pcd)
