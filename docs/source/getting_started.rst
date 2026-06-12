Getting Started
=======================
How to guide on installing PCHandler

Dependencies
------------
Core Libraries

- `NumPy <https://numpy.org/>`_ — Fast N-dimensional arrays and numerical operations that power core point-cloud computations.
- `GeoPandas <https://geopandas.org/>`_ — High-level geospatial data structures used for GIS-style processing and analysis.
- `Shapely <https://shapely.readthedocs.io/>`_ — Geometric predicates and operations for working with 2D geometry (buffers, intersections, etc.).
- `alphashape <https://alphashape.readthedocs.io/>`_ — Alpha-shape computation to derive concave hulls/outlines from point sets.

Point Cloud I/O

- `plyfile <https://github.com/dranjan/python-plyfile>`_ — Read/write PLY files (ASCII/Binary) with attribute preservation.
- `laspy <https://laspy.readthedocs.io/>`_ — Read/write LAS/LAZ lidar point cloud formats, including point attributes.

Visualization / 3D Operations

- `Open3D <https://www.open3d.org/>`_ — Visualization and selected 3D geometry utilities for point clouds and meshes.
- `py4dgeo <https://py4dgeo.readthedocs.io/en/latest/>`_ - Library containing other geomonitoring algorithms from Heidelberg University.

Utilities

- `joblib <https://joblib.readthedocs.io/>`_ — Simple parallelism and caching for speeding up CPU-bound workflows.

Optional GPU Acceleration

- `cuDF <https://docs.rapids.ai/api/cudf/stable/>`_ — GPU DataFrame operations to accelerate tabular point attributes and transforms.
- `cuSpatial <https://docs.rapids.ai/api/cuspatial/stable/>`_ — GPU-accelerated spatial/trajectory operations for large-scale geospatial workloads.
- `cuML <https://docs.rapids.ai/api/cuml/stable/>`_ — GPU-accelerated machine learning algorithms useful for clustering, outlier detection, and similar tasks.


Install from GitHub
-------------------

.. code-block:: bash

   # (optional) create and activate a virtual environment
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate

   # clone and install
   git clone https://github.com/your-org/pchandler.git
   cd pchandler
   python -m pip install -e .  # editable install for development


Quick Example
-------------

To ensure PCHandler has been properly installed, you can try the following code:

.. code-block:: python

   import numpy as np
   from pchandler import PointCloudData

    offset = [10_000_000, -500_000, 20_000]

   # Create an (N, 3) array of XYZ coordinates
    points = np.random.rand(10, 3) * 1000 - 500 + offset

   # Initialize the point cloud
   pcd = PointCloudData(points)

   # (Optional) confirm basic properties
   print(f"{points.shape=}")  # (100, 3)
   print(f"{pcd.xyz=}")
   print(f"{pcd.numerical_optimization_shift=}")
