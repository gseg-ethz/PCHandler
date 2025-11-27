File Handlers
=============

This page shows a few practical ways to create point clouds and exchange data
with other libraries.

This page demonstrates loading and saving point clouds using the CSV-like and PLY handlers from ``pchandler.data_io``.

CSV-like formats (CsvHandler)
-----------------------------

Supported extensions: ``.txt``, ``.csv``, ``.xyz``, ``.asc``, ``.ascii``, ``.pts``

Load
^^^^

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData
   from pchandler.data_io.csv import CsvHandler

   # Example: load a CSV with header/comment lines and auto-delimited columns.
   # The handler will sniff the file structure and locate XYZ plus any extra fields.
   pcd = CsvHandler.load(
       "points.csv",
       # Select scalar fields to load (names must match columns, excluding x,y,z)
       # None -> load all fields, [] -> ignore scalar fields
       scalar_fields=None,
       # If file columns are named like "scalar_intensity", "scalar_quality", remove the prefix:
       remove_prefix=True,
       prefix="scalar_",
       # Header name row index (default -1 for last header line)
       column_names_row=-1,
       # Comment marker used in file header
       comment="//",
       # Override delimiter if needed; None lets the handler auto-detect
       delimiter=None,
       # You can pass PointCloudData kwargs as well:
       numerical_optimization_shift=None,
   )

   # Access XYZ and any loaded scalar fields
   xyz = pcd.xyz
   intensity = pcd.scalar_fields.get("intensity", None)

Save
^^^^

.. code-block:: python

   from pchandler.data_io.csv import CsvHandler

   # Create or modify a point cloud
   # pcd = PointCloudData(...)

   # Save with a specific delimiter and prefixed scalar field names
   CsvHandler.save(
       pcd,
       "points_out.csv",
       # Restrict which scalar fields to write; None -> write all available fields
       scalar_fields=None,
       # Add a prefix to scalar field column names in the file
       add_prefix=True,
       prefix="scalar_",
       # Keep original dtypes for scalar fields where possible
       revert_sf_types=False,
       # Choose delimiter (e.g., "," or " ")
       delimiter=",",
   )

PLY format (PlyHandler)
-----------------------

Supported extension: ``.ply``

Load
^^^^

.. code-block:: python

   from pchandler.data_io.ply import PlyHandler

   pcd = PlyHandler.load(
       "points.ply",
       # Choose which scalar fields to load; None -> all, [] -> none
       scalar_fields=None,
       # Remove "scalar_" (or a custom) prefix when importing field names
       remove_prefix=True,
       prefix="scalar_",
       # You can pass PointCloudData kwargs as well:
       numerical_optimization_shift=None,
   )

Save
^^^^

.. code-block:: python

   from pchandler.data_io.ply import PlyHandler

   PlyHandler.save(
       pcd,
       "points_out.ply",
       # Restrict scalar fields; None -> write all
       scalar_fields=None,
       # Add a prefix to scalar field names when writing
       add_prefix=False,
       prefix="scalar_",
       # Revert scalar field types to original representations where possible
       revert_sf_types=False,
       # Write ASCII PLY (False -> binary, which is smaller/faster)
       as_ascii=False,
   )

Roundtrip examples
------------------

CSV roundtrip
^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData
   from pchandler.data_io.csv import CsvHandler

   # Create a simple cloud with one scalar field
   N = 5
   pcd = PointCloudData(
       xyz=np.random.rand(N, 3),
       scalar_fields={"intensity": np.linspace(0, 1, N)},
       numerical_optimization_shift=None,
   )

   CsvHandler.save(pcd, "example.csv", add_prefix=True, prefix="scalar_")
   pcd2 = CsvHandler.load("example.csv", remove_prefix=True, prefix="scalar_")

   # Scalar fields and coordinates are preserved
   assert len(pcd2) == N

PLY roundtrip
^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from pchandler.core import PointCloudData
   from pchandler.data_io.ply import PlyHandler

   N = 5
   pcd = PointCloudData(
       xyz=np.random.rand(N, 3),
       scalar_fields={"quality": np.random.rand(N)},
       numerical_optimization_shift=None,
   )

   PlyHandler.save(pcd, "example.ply", add_prefix=False)
   pcd2 = PlyHandler.load("example.ply", remove_prefix=False)

   assert len(pcd2) == N
