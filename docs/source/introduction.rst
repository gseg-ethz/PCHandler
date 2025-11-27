Introduction
============

The ``pchandler`` module provides a comprehensive set of tools for handling, manipulating, and analyzing
3D point cloud data. Its components are modularly designed, covering geometry processing, field-of-view
(FoV) management, data input/output, and utility functions. The package is optimized for flexibility,
efficiency, and extensibility, supporting both CPU and GPU acceleration for scalable workflows.

Components:
-----------
1. ``pchandler.core.py``

  Contains the main PointCloudData object for working with point clouds and interfaces with the other components.

2. ``pchandler.geometry``:

  - Core coordinaate classes from which PointCloudData inherits from the ``CartesianCoordinates``
  - Numerical optimal shift functionality to enable more efficient processing with float 32 data types
  - Splitter functions to separate the point cloud based on geometry
  - **(Under Development)** Easy 3D Transformation classes and handling to track coordinate system transformations or scan registrations

  .. code-block:: python

    import numpy as np
    from pchandler import PointCloudData
    PointCloudData.__bases__[0]
    # -> (<class 'CartesianCoordinates'>)


  Furthermore, it has addition dedicated spherical coordinate based classes for managing angular regions in 3D space:

    - ``FoV`` and ``FoVTree`` for working with Field-of-Views from the scan origin
    - ``Angle`` and ``AngleArray`` for a flexible class for simple handling of various angular units

  .. code-block:: python

    from pchandler.geometry.spherical.fov import FoV
    from pchandler.geometry.spherical.angle import Angle

    # Define a field of view
    fov = FoV(left=Angle(0, 'deg'), right=Angle(3.14, 'rad'), top=Angle(0, 'gon'), bottom=np.pi)

    # Split the FoV into quadrants
    quadrants = fov.quadrants()


3. ``pchandler.data_io``:

  - Provides utilities for reading and writing point cloud data in various formats, including PLY, LAS/LAZ, CSV and minor E57 support.
  - Includes support for managing colors, normals, and scalar fields during import/export.

  .. code-block:: python

      from pchandler.data_io import Ply
      from pathlib import Path

      # Load a PLY file
      pcd = Ply.load(Path("example.ply"))

      # Save the point cloud to another file
      Ply.save(pcd, Path("output.ply"))

  A generic file loader also exists that automatically attempts to load the file based on the file extension:

  .. code-block:: python

    from pchandler import load_file
    pcd = load_file("PointCloud.e57")

4. ``pchandler.filters``:
   Provides the following types of filters.
   - Includes the AngleUnit enum and a robust convert_angles function.

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
   :maxdepth: 2





