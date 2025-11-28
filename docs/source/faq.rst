FAQ / Limitations / Troubleshooting
===================================

FAQ
---

Q. How do I get the original coordinates that were loaded into the point cloud?

One method is to add the numerical optimization shift offset to the coordinates of the point cloud:

.. code-block:: python

  original = pcd.xyz + pcd.numerical_optimization_shift

Another is to remove the optimal shift from the point cloud:

.. code-block:: python

    # Simple assign the optimal shift to None and the coordinates will be updated
    pcd.numerical_optimization_shift = None


Limitations
-----------

- Basic handling of Transformations
- Some assumptions still exist in automatic CSV/TXT file loading
- E57 is only supported for coordinate data and a single point cloud (no intensity or scalar field values
- Spherical coordinates are based on the assumption the scan center is at 0, 0, 0.
  This still handles the numerical optimization shift offset but a point cloud with existing offset may produce incorrect spherical angles.
- Not yet support kdTrees natively

Contribution / Requests
-----------------------

To do ...
