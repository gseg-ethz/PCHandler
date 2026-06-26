# Introduction

[![License: BSD-3](https://img.shields.io/badge/License-BSD_3-yellow.svg)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/pchandler/badge/)](https://pchandler.readthedocs.io/)


The ``pchandler`` module provides a comprehensive set of tools for handling, manipulating, and analyzing
3D point cloud data. Its components are modularly designed, covering geometry processing, field-of-view
(FoV) management, data input/output, and utility functions. The package is optimized for flexibility,
efficiency, and extensibility, supporting both CPU and GPU acceleration for scalable workflows.

## Key Features

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

## Install from GitHub

```shell
# (optional) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# clone and install
git clone https://github.com/your-org/pchandler.git
cd pchandler
python -m pip install -U pip
python -m pip install -e .  # editable install for development
```

Quick Example
-------------

To ensure PCHandler has been properly installed, you can try the following code:

```python
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
```

## Components
### 1. ``pchandler.core``

  Contains the main PointCloudData object for working with point clouds and interfaces with the other components.

### 2. ``pchandler.geometry``

  - Core coordinaate classes from which PointCloudData inherits from the ``CartesianCoordinates``
  - Numerical optimal shift functionality to enable more efficient processing with float 32 data types
  - Splitter functions to separate the point cloud based on geometry
  - **(Under Development)** Easy 3D Transformation classes and handling to track coordinate system transformations or scan registrations

```python
import numpy as np
from pchandler import PointCloudData
PointCloudData.__bases__[0]
# -> (<class 'CartesianCoordinates'>)
```

Furthermore, it has addition dedicated spherical coordinate based classes for managing angular regions in 3D space:
- `FoV` and ``FoVTree`` for working with Field-of-Views from the scan origin
- ``Angle`` and ``AngleArray`` for a flexible class for simple handling of various angular units

```python
from pchandler.geometry.spherical.fov import FoV
from pchandler.geometry.spherical.angle import Angle

# Define a field of view
fov = FoV(left=Angle(0, 'deg'), right=Angle(3.14, 'rad'), top=Angle(0, 'gon'), bottom=np.pi)

# Split the FoV into quadrants
quadrants = fov.quadrants()
```

### 3. ``pchandler.data_io``

- Provides utilities for reading and writing point cloud data in various formats, including PLY, LAS/LAZ, CSV and minor E57 support.
- Includes support for managing colors, normals, and scalar fields during import/export.

```python
from pchandler.data_io import Ply
from pathlib import Path

# Load a PLY file
pcd = Ply.load(Path("example.ply"))

# Save the point cloud to another file
Ply.save(pcd, Path("output.ply"))
```
A generic file loader also exists that automatically attempts to load the file based on the file extension:

```python
from pchandler import load_file
pcd = load_file("PointCloud.e57")
```

4. ``pchandler.filters``:
   Provides the following types of filters.
   - Includes the AngleUnit enum and a robust convert_angles function.

```python
    from pchandler.util import convert_angles, AngleUnit
    import numpy as np

    # Convert angles from degrees to radians
    degrees = np.array([0, 45, 90, 180])
    radians = convert_angles(degrees, source_unit=AngleUnit.DEGREE, target_unit=AngleUnit.RAD)
    print("Radians:", radians)
```


## Dependencies
Core Libraries

- [NumPy](https://numpy.org/) — Fast N-dimensional arrays and numerical operations that power core point-cloud computations.
- [GeoPandas](https://geopandas.org/) — High-level geospatial data structures used for GIS-style processing and analysis.
- [Shapely](https://shapely.readthedocs.io/) — Geometric predicates and operations for working with 2D geometry (buffers, intersections, etc.).
- [alphashape](https://alphashape.readthedocs.io/) — Alpha-shape computation to derive concave hulls/outlines from point sets.

Point Cloud I/O

- [plyfile](https://github.com/dranjan/python-plyfile) — Read/write PLY files (ASCII/Binary) with attribute preservation.
- [laspy](https://laspy.readthedocs.io/) — Read/write LAS/LAZ lidar point cloud formats, including point attributes.

Visualization / 3D Operations

- [Open3D](https://www.open3d.org/) — Visualization and selected 3D geometry utilities for point clouds and meshes.
- [py4dgeo](https://py4dgeo.readthedocs.io/en/latest/) - Library containing other geomonitoring algorithms from Heidelberg University.

Utilities

- [joblib](https://joblib.readthedocs.io/) — Simple parallelism and caching for speeding up CPU-bound workflows.

Optional GPU Acceleration

- [cuDF](https://docs.rapids.ai/api/cudf/stable/) — GPU DataFrame operations to accelerate tabular point attributes and transforms.
- [cuSpatial](https://docs.rapids.ai/api/cuspatial/stable/) — GPU-accelerated spatial/trajectory operations for large-scale geospatial workloads.
- [cuML](https://docs.rapids.ai/api/cuml/stable/) — GPU-accelerated machine learning algorithms useful for clustering, outlier detection, and similar tasks.

## Verifying Releases

Download a wheel from a release and verify its Sigstore provenance attestation:

```bash
gh attestation verify pchandler-2.1.0-py3-none-any.whl --repo gseg-ethz/PCHandler
```

See [RELEASE.md](RELEASE.md) for claim fields, rollback procedure, and environment details.
