File Handlers
=============

This page demonstrates loading and saving point clouds using the CSV-like and PLY handlers from ``pchandler.data_io``.

CSV-like formats (CsvHandler)
-----------------------------
Although this file handler is named CSV, it is more of a generic text delimited file format handler.

Supported extensions:

- ``.txt``
- ``.csv``
- ``.xyz``
- ``.asc``
- ``.ascii``
- ``.pts``

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
       scalar_fields=None, # Allows the selection of scalar fields or is needed when file has no header
       remove_prefix=True,  # In some cases, like cloud compare, a prefix is added to all scalar fields
       prefix="scalar_",
       column_names_row=-1, # refers to the row index in the header/comments which contains the column names
       comment="//",
       delimiter=None,  # Setting delimiter to None, the handler will attempt to "sniff" the file and determine it
       numerical_optimization_shift=None, # Other parameters for the PointCloudData keyword arguments can be passed
   )

   xyz = pcd.xyz
   intensity = pcd.intensity

Save
^^^^

.. code-block:: python

   from pchandler.data_io.csv import CsvHandler

   # Save with a specific delimiter and prefixed scalar field names
   CsvHandler.save(
       pcd,
       "points_out.csv",
       scalar_fields=None,  # Select which scalar fields to save (defaults to all fields)
       add_prefix=True,
       prefix="scalar_",
       revert_sf_types=False,   # Keep original dtypes for scalar fields where possible
       delimiter=",",
   )

PLY format (PlyHandler)
----------------------------------

Supported extension: ``.ply``

Load
^^^^

.. code-block:: python

   from pchandler.data_io.ply import PlyHandler

   pcd = PlyHandler.load(
       "points.ply",
       scalar_fields=None,
       remove_prefix=True,
       prefix="scalar_",
       numerical_optimization_shift=None,
   )

Save
^^^^

.. code-block:: python

   from pchandler.data_io.ply import PlyHandler

   PlyHandler.save(
       pcd,
       "points_out.ply",
       scalar_fields=None,
       add_prefix=False,
       prefix="scalar_",
       revert_sf_types=False,
       as_ascii=False,  # Write ASCII PLY (False -> binary, which is smaller/faster)
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
