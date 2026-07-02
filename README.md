# PChandler — modified branch for tls2dseg

This branch is a modified/personal development branch (development/tomislav) of an outdated copy of `pchandler` (v1.0.0 or even earlier). It is used as a dependency for `tls2dseg`.

The main project README and stable documentation are maintained on the `main` branch. This branch may contain experimental changes, API deviations, or temporary modifications needed by `tls2dseg`.

Use this branch with caution unless you specifically need the `tls2dseg` integration.

## Differences from main

The diff is large because the main branch of pchandler has undergone substantial restructuring since this branch was created (and was upgraded for a version to v2). The main modifications introduced in this branch relative to the old one are listed below.

* **E57 loading and coordinate-system handling**

  * Extended the E57 loader in `src/pchandler/data_io.py`.
  * Added optional arguments to `load_e57()`:

    * `stay_global`
    * `save_global_info`
  * Updated `_load_all_e57_scans()` and `_load_single_e57()` accordingly.
  * Added support for storing the point cloud transformation matrix during E57 loading.
  * Added a docstring to `load_e57()`.
  * Added `find_tmat_in_directory()` for locating transformation matrices in a directory.

* **PointCloudData metadata**

  * Modified `src/pchandler/geometry/core.py`.
  * Added coordinate-system-related attributes to `PointCloudData`:
    * `tmat_socs2prcs`
    * `xyz_is_prcs`
  * Updated `__post_init__()` and `_validate_internal_state()` accordingly.
  * Updated `PointCloudData.sample()` to preserve these attributes.

* **Coordinate-system transformations**

  * Modified `src/pchandler/geometry/transforms.py`.
  * Added:

    * `toggle_socs2prcs()`
    * `toggle_prcs2socs()`
  * Updated `PointCloudData.transform()` so that `tmat_socs2prcs` is transformed consistently when a transformation is applied to the point cloud.
  * Added `lazy_global_shift_change`.

* **Field-of-view tiling**

  * Modified `fov.py`.
  * Added `tile_with_overlap()` to support overlapping FoV tiles.
  * Added `save_pcd_tiles()` in `util.py`.

* **Voxel downsampling**

  * Modified `src/pchandler/geometry/filters/downsample.py`.
  * Implemented `"nearest"` mode for voxel downsampling.
  * Implemented weighted majority voting for voxel downsampling.

* **Scalar field handling**

  * Removed the warning for scalar field merges.
  * In the `ScalarField` dataclass, changed the default value of `override_forced_dtype_conversion` to `True`.

## License and Citation

This branch remains licensed under the same license as the main project. See `LICENSE`.

For citation purposes, please refer to the `main` branch of the original `pchandler` repository and its corresponding citation information, since this branch is a modified derivative used for `tls2dseg` integration.





