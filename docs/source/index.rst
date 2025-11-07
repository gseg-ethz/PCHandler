PCHandler
=========

"""
``pchandler``

The ``pchandler`` module provides a comprehensive set of tools for handling, manipulating, and analyzing
3D point cloud data. Its components are modularly designed, covering geometry processing, field-of-view
(FoV) management, data input/output, and utility functions. The package is optimized for flexibility,
efficiency, and extensibility, supporting both CPU and GPU acceleration for scalable workflows.

Components:
-----------
1. ``pchandler.geometry``:
   - Core functionality for managing 3D point cloud data.
   - Provides the `PointCloudData` class for handling coordinates, colors, normals, and scalar fields.
   - Includes methods for filtering, sampling, range operations, voxel downsampling, and integration with Open3D.
   - Supports spherical coordinate transformations and FoV-based operations for spatial analysis.

   Example Usage:

.. code-block:: python

    from pchandler.geometry import PointCloudData
    import numpy as np

    # Create a point cloud from random data
    points = np.random.rand(1000, 3)
    pcd = PointCloudData(points)

    # Filter points within a specific range
    pcd.filter_range(low=0.2, high=0.8)
    print(f"Filtered point count: {pcd.nbPoints}")


2. ``pchandler.geometry.spherical.fov``:
   - Utilities for managing angular regions in 3D space.
   - Includes the FoV class for defining and manipulating fields of view, supporting conversions between angular units.
   - The FoVTree class allows hierarchical organization of FoVs for spatial partitioning.

   Example Usage:

.. code-block:: python

    from pchandler.geometry.spherical.fov import FoV
    # Define a field of view
    fov = FoV(horizontal_min=0, horizontal_max=90, elevation_min=-30, elevation_max=30, unit="deg")

    # Split the FoV into quadrants
    quadrants = fov.quadrants()
    print("FoV quadrants:", quadrants)

3. ``pchandler.data_io``:
   - Provides utilities for reading and writing point cloud data in various formats, including PLY, LAS/LAZ, and CSV.
   - Includes support for managing colors, normals, and scalar fields during import/export.

   Example Usage:

.. code-block:: python

    from pchandler.data_io import load_ply, save_ply
    from pathlib import Path

    # Load a PLY file
    pcd = load_ply(Path("example.ply"))

    # Save the point cloud to another file
    save_ply(Path("output.ply"), pcd)

4. ``pchandler.geometry.util``:
   - Provides utility functions for common operations, such as angle unit conversions.
   - Includes the AngleUnit enum and a robust convert_angles function.

   Example Usage:

.. code-block:: python

    from pchandler.util import convert_angles, AngleUnit
    import numpy as np

    # Convert angles from degrees to radians
    degrees = np.array([0, 45, 90, 180])
    radians = convert_angles(degrees, source_unit=AngleUnit.DEGREE, target_unit=AngleUnit.RAD)
    print("Radians:", radians)

Key Features:
-------------
1. **Point Cloud Geometry**:
   - Manage and manipulate point cloud data with attributes like coordinates, colors, normals, and scalar fields.
   - Support for operations such as filtering, voxel downsampling, and outlier removal.

2. **Field-of-View (FoV) Management**:
   - Define and manipulate rectangular angular regions (FoVs) in 3D space.
   - Hierarchical FoV trees for spatial partitioning and efficient data processing.
   - Functions for splitting, merging, tiling, and converting between angular units.

3. **Data I/O**:
   - Read and write various point cloud file formats, including PLY, LAS/LAZ, and ASCII.
   - Efficient handling of large datasets with GPU acceleration (when supported).

4. **Utilities**:
   - Convert between angle units (radians, degrees, gons).
   - Provide numerical constants and helper functions to facilitate computations.

Dependencies:
-------------
- **Core Libraries**:
  - ``numpy``: For numerical computations and array manipulation.
  - ``geopandas``, `shapely`: For geometry processing and spatial operations.
  - ``alphashape``: For generating alpha shapes to compute point cloud outlines.

- **Optional GPU Support**:
  - ``cudf``, ``cuspatial``, ``cuml``: For accelerated processing of point cloud data.
  - Requires a CUDA-enabled GPU for GPU-based functionalities.

- **Point Cloud I/O**:
  - ``plyfile``: For reading/writing PLY files.
  - ``laspy``: For handling LAS/LAZ file formats.

- **Other Utilities**:
  - ``joblib``: For parallel processing.
  - ``open3d``: For interfacing with the Open3D library for visualization and advanced operations.

This modular design ensures that `pchandler` is both extensible and scalable, making it suitable for a wide range of
applications in 3D data analysis, GIS, and computer vision.
"""


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   about
   getting_started
   examples
   API



