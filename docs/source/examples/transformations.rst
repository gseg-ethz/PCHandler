3D Transformations
==================

This page shows how to apply translations, rotations, and scales using the
``Transform`` class from ``pchandler.geometry.transforms``. A ``Transform``
is a 4x4 affine operator that can be chained and applied succinctly to
:class:`pchandler.core.PointCloudData`.

By using the transform class, writing code can now follow the mathematical form ``y = Tx``
and not having to handle matrix transpositions.

Key points
----------

- Import with: ``from pchandler.geometry.transforms import Transform``
- Build transforms via helpers:
  - ``Transform.from_translation(vector)``
  - ``Transform.from_rotation(R3x3)``
  - ``Transform.from_scale(scalar_or_vec3)``
  - ``Transform.from_affine(A4x4)``
  - ``Transform.generate(rotation=..., translation=..., scale=...)``
- Chain transforms with ``@=`` (matrix multiply in-place).
- Apply to point clouds with ``T @ pcd`` to get a transformed copy.


Translate and scale
-------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData
   from pchandler.geometry.transforms import Transform

   pcd = PointCloudData(np.random.rand(100, 3), numerical_optimization_shift=None)

   # Build a translation transform
   t = np.array([10.0, -2.0, 3.5], dtype=float)
   T_translate = Transform.from_translation(t)

   # Build a uniform scale transform
   s = 2.0
   T_scale = Transform.from_scale(s)

   # Option A: apply individually
   translated = T_translate @ pcd
   scaled = T_scale @ pcd

   # Option B: chain and apply once (scale, then translate)
   T = Transform.from_scale(s)
   T @= Transform.from_translation(t)
   moved = T @ pcd


Rigid transform with a rotation matrix
--------------------------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData
   from pchandler.geometry.transforms import Transform

   def rot_z(theta):
       c, s = np.cos(theta), np.sin(theta)
       return np.array([[ c, -s, 0.0],
                        [ s,  c, 0.0],
                        [ 0.0, 0.0, 1.0]], dtype=float)

   pcd = PointCloudData(np.random.rand(100, 3), numerical_optimization_shift=None)

   R = rot_z(np.deg2rad(30.0))
   t = np.array([1.0, 0.0, 0.0], dtype=float)

   # Build a single affine transform with rotation and translation
   T_rt = Transform.generate(rotation=R, translation=t)

   transformed = T_rt @ pcd  # returns a transformed copy


Using a 4x4 affine matrix directly
----------------------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData
   from pchandler.geometry.transforms import Transform

   pcd = PointCloudData(np.random.rand(200, 3), numerical_optimization_shift=None)

   # Example 4x4 matrix (row-major): rotation + translation
   A = np.eye(4, dtype=float)
   A[:3, :3] = np.array([[0.0, -1.0, 0.0],
                         [1.0,  0.0, 0.0],
                         [0.0,  0.0, 1.0]], dtype=float)
   A[:3, 3] = np.array([2.0, 0.0, 0.0], dtype=float)

   T = Transform.from_affine(A)
   pcd_affine = T @ pcd


Chaining multiple transforms
----------------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData
   from pchandler.geometry.transforms import Transform

   pcd = PointCloudData(np.random.rand(50, 3), numerical_optimization_shift=None)

   R = np.eye(3)
   t1 = np.array([0.5, 0.0, 0.0])
   t2 = np.array([0.0, 1.0, 0.0])

   T = Transform.from_rotation(R)
   T @= Transform.from_translation(t1)   # first translate by t1
   T @= Transform.from_translation(t2)   # then translate by t2

   pcd_moved = T @ pcd


Merging transformed clouds
--------------------------

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData
   from pchandler.geometry.transforms import Transform

   a = PointCloudData(np.random.rand(200, 3), numerical_optimization_shift=None)

   # Shift 'a' by +5 along X using a transform
   T = Transform.from_translation(np.array([5.0, 0.0, 0.0], dtype=float))
   b = T @ a

   merged = PointCloudData.merge(a, b)


Notes on ordering
-----------------

- ``T @= U`` composes transforms so that when later applied as ``(T @ pcd)``,
  the effect of ``U`` happens after the transforms already in ``T``.
- ``Transform.generate(rotation=R, translation=t, scale=s)`` constructs a
  single affine with those components combined.