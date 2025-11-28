General
=======

Sampling, reduction and extraction
----------------------------------

.. code-block:: python

    import numpy as np
    from pchandler.core import PointCloudData

    pcd = PointCloudData(
       xyz=np.random.rand(100, 3),
       numerical_optimization_shift=None,
    )

    pcd2 = pcd.copy(deep=True)

    # Boolean mask: keep points with x > 0.5
    mask = pcd.xyz[:, 0] > 0.5

    # Sample returns a new object
    sample = pcd.sample(mask)

    # Reduce modifies in place
    pcd.reduce(mask)

    # Extract removes the selected points from the object
    extracted = pcd2.extract(mask)

.. note::
    The above three methods only accept a vector that corresponds to indexes or boolean mask.
    If you want to sample with other advanced numpy indexing, you may have to do so directly on the array. You will only get a numpy array in return without scalar field info.
    ``xy = pcd.arr[mask, 0:2]``

Merging clouds
--------------
It's also easy to merge multiple point clouds together

.. code-block:: python

    import numpy as np
    from pchandler.core import PointCloudData

    a = PointCloudData(np.random.rand(50, 3), numerical_optimization_shift=None)
    b = PointCloudData(np.random.rand(75, 3) + 1.0, numerical_optimization_shift=None)

    merged = PointCloudData.merge(a, b)

Or as a list of point clouds

.. code-block:: python

    import numpy as np
    from pchandler.core import PointCloudData

    a = PointCloudData(np.random.rand(50, 3), numerical_optimization_shift=None)
    b = PointCloudData(np.random.rand(75, 3) + 1.0, numerical_optimization_shift=None)
    c = PointCloudData(np.random.rand(100,3), numerical_optimization_shift=None)

    all_pcds = [a, b, c]

    merged = PointCloudData.merge(*all_pcds)

Spherical Coordinates
---------------------

Easily accessed from a cached property:

.. code-block::

    spherical_coordinates = merged.spher
    hz_angles = merged.hz
    v_angles = merged.v
    radius = merged.r

.. note::

    The same access can be gained from the cartesian coordinates via the ``x``, ``y``, ``z`` properties.

    .. code-block::

        x = merged.x
        y = merged.y
        z = merged.z