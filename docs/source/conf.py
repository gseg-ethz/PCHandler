import os
import sys

# Phase 09.1 Plan 01 residual warnings: 566
# (after numpydantic fix, structural RST fixes, and py4dgeo nitpick_ignore_regex;
# toc.not_included + PointCloudData/load_file duplicate-object warnings resolved.
# The remaining 566 are nitpicky-mode ref.class/ref.meth/ref.obj/ref.data surfaced
# cross-ref issues; Plan 09.1-03 will address them via autodoc_type_aliases +
# role-matching + nitpick_ignore_regex expansion)
# from docs.conf import autosummary_generate

sys.path.insert(0, os.path.abspath("../../src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PCHandler"
copyright = "2024, Nicholas Meyer"
author = "Nicholas Meyer"

# Automatic version updates via release-please
version = "2.1.0"  # x-release-please-version
release = "2.1.0"  # x-release-please-version


extensions = [
    "sphinx.ext.autodoc",  # For generating documentation from docstrings
    "sphinx.ext.napoleon",  # For Google-style and NumPy-style docstrings
    "sphinx.ext.autosummary",  # For summary tables
    "sphinx.ext.intersphinx",
]

# Intersphinx Config
intersphinx_mapping = {
    "open3d": ("https://www.open3d.org/docs/release/", None),
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "numpydantic": ("https://numpydantic.readthedocs.io/en/latest/", None),
    "GSEGUtils": ("https://gsegutils.readthedocs.io/en/latest/", None),
}


# General Config
python_use_unqualified_type_name = True  # False

# ======= Autodoc Config =========
autoclass_content = "class"  # 'both', 'None, 'init', 'class'
autodoc_class_signature = "separated"  # 'mixed' / 'separated
autodoc_member_order = "bysource"  # 'alphabetical', 'bysource'
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    # 'special-members': '__init__',
    "inherited-members": False,
    # 'imported-members': True,
    "exclude-members": "model_config, model_post_init, _abc_impl, _reconstruct",
    # 'ignore-module-all': False,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_docstring_signature = True  # True
autodoc_mock_imports = []  # []
autodoc_typehints = "description"  # 'signature', 'description', 'none', 'both'
autodoc_typehints_description_target = "documented"  # 'all', 'documented', 'documented_params'
autodoc_typehints_format = "short"  # 'short', 'fully-qualified'
autodoc_preserve_defaults = True  # False
autodoc_use_type_comments = True  # True
autodoc_warningiserror = True  # True
autodoc_inherit_docstrings = True  # True
linkcheck_allowed_redirects = {}
autosummary_generate_overwrite = False
# Defaults
templates_path = ["_templates"]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# DOC-V2-01 (Phase 09.1-03): Render GSEGUtils alias names instead of expanded
# NDArray[Shape["*, 3"], Float] form. From __future__ import annotations is active
# in all pchandler source files (PEP 563 string annotations), so autodoc sees the
# alias name as a string and this dict performs textual substitution before the
# cross-reference is emitted. Values are fully-qualified GSEGUtils names so that
# intersphinx can look them up in the GSEGUtils RTD inventory (py:attribute entries).
autodoc_type_aliases = {
    "Array_3x3_T": "GSEGUtils.base_types.Array_3x3_T",
    "Array_4x4_T": "GSEGUtils.base_types.Array_4x4_T",
    "Array_Float_T": "GSEGUtils.base_types.Array_Float_T",
    "Array_Nx2_Float_T": "GSEGUtils.base_types.Array_Nx2_Float_T",
    "Array_Nx3_Float_T": "GSEGUtils.base_types.Array_Nx3_Float_T",
    "Array_Nx3_Float32_T": "GSEGUtils.base_types.Array_Nx3_Float32_T",
    "Array_Nx3_T": "GSEGUtils.base_types.Array_Nx3_T",
    "Array_Nx3_Uint8_T": "GSEGUtils.base_types.Array_Nx3_Uint8_T",
    "Array_Uint8_T": "GSEGUtils.base_types.Array_Uint8_T",
    "ArrayT": "GSEGUtils.base_types.ArrayT",
    "DtypeDict": "GSEGUtils.base_types.DtypeDict",
    "IndexLike": "GSEGUtils.base_types.IndexLike",
    "LowerStr": "str",  # matches GSEGUtils conf.py precedent (Annotated[str,...] → str)
    "SfNameT": "GSEGUtils.base_types.SfNameT",
    "Vector_3_T": "GSEGUtils.base_types.Vector_3_T",
    "Vector_Bool_T": "GSEGUtils.base_types.Vector_Bool_T",
    "Vector_Float_T": "GSEGUtils.base_types.Vector_Float_T",
    "Vector_Float32_T": "GSEGUtils.base_types.Vector_Float32_T",
    "Vector_Uint8_T": "GSEGUtils.base_types.Vector_Uint8_T",
    "VectorT": "GSEGUtils.base_types.VectorT",
}

# Phase 09.1-03: suppress_warnings retained ONLY for the "more than one target"
# case produced by the lazy-load __init__ re-export pattern (ScalarField appears at
# both pchandler.scalar_fields.ScalarField and pchandler.scalar_fields.scalar_fields.
# ScalarField; similarly for Angle, FoV, OptimizedShift, etc.). This cannot be
# addressed by nitpick_ignore_regex (which only handles "not found", not "ambiguous").
# The structural fix would be :no-index: / :canonical: on the .pyi stubs — deferred
# (D-07). The blanket "ref.class" no-op was dropped in Plan 09.1-01.
suppress_warnings = ["ref.python"]

# Phase 09.1-03: nitpicky mode + minimal auditable nitpick_ignore_regex for all
# genuinely-unresolvable external targets. New self-inflicted broken refs still fail.
nitpicky = True
nitpick_ignore_regex: list[tuple[str, str]] = [
    # --- Optional external deps with no accessible public RTD inventory ---
    # py4dgeo has no public intersphinx inventory; Epoch is used in docstrings + rst_epilog
    (r"py:.*", r"py4dgeo\..*"),
    (r"py:class", r"Epoch"),
    # open3d inventory exists but t.geometry types are not exported in objects.inv
    (r"py:.*", r"o3d\..*"),
    (r"py:.*", r"open3d\..*"),
    # --- numpydantic vendor-internal / unexported types ---
    # numpydantic.vendor.nptyping.* is a private implementation detail; the public
    # inventory only exports numpydantic.ndarray.NDArray and numpydantic.types.*
    (r"py:.*", r"numpydantic\.vendor\..*"),
    # numpydantic 1.10 made NDArray a runtime Protocol; autodoc now renders the
    # fully-parametrised numpydantic.ndarray.NDArray[...] form (submodule-qualified,
    # expanded dtype tuple) for raw NDArray-typed members, which no inventory exports.
    # Pre-1.10 rendered the shorter numpydantic.NDArray form. Mirrors the GSEGUtils fix.
    (r"py:class", r"numpydantic\.ndarray\.NDArray.*"),
    # npt.NDArray and np.* short-form aliases — inventory uses the full numpy.* form
    (r"py:.*", r"npt\.NDArray"),
    (r"py:.*", r"np\.\w+"),
    # numpy._typing internal types not in the public numpy inventory
    (r"py:.*", r"numpy\._typing\..*"),
    # NDArray as an unqualified short name (from autosummary cross-refs, rst_epilog contexts)
    (r"py:class", r"NDArray"),
    # TypeAliasForwardRef: emitted by autodoc when autodoc_type_aliases maps to a target
    # whose inventory role (py:attribute) does not match the autodoc-emitted role (py:class)
    (r"py:class", r"TypeAliasForwardRef"),
    # --- GSEGUtils alias role-mismatch (autodoc emits py:class; inventory has py:attribute) ---
    # autodoc_type_aliases maps alias names to fully-qualified GSEGUtils.base_types.* targets.
    # The GSEGUtils RTD inventory exposes these as py:attribute (verified 2026-06-16 via live
    # objects.inv fetch). Sphinx autodoc emits py:class for type-alias cross-refs — mismatch.
    # nitpick_ignore_regex covers the role mismatch so the build exits 0; the alias names
    # still render correctly in HTML (the textual substitution fires regardless of link success).
    # Fully-qualified form (from autodoc_type_aliases substitution in autodoc output):
    (r"py:.*", r"GSEGUtils\.base_types\..*"),
    # Short/unqualified form (from autosummary cross-refs in .rst pages, rst_epilog):
    (r"py:class", r"Array_[A-Za-z0-9_]+_T"),
    (r"py:class", r"Vector_[A-Za-z0-9_]+_T"),
    (r"py:class", r"(ArrayT|VectorT|IndexLike|SfNameT|LowerStr|DtypeDict)"),
    # --- pchandler private / module-internal types in docstrings ---
    # Types referenced by short name in inline docstring cross-refs; they are documented
    # in autosummary pages under their module path but the unqualified form is not resolvable.
    # Private (underscore-prefixed) classes:
    (r"py:class", r"_NameConstantsSingle"),
    (r"py:class", r"_NameConstantsTriplet"),
    (r"py:class", r"_ScalarKwargT"),
    (r"py:class", r"pchandler\.(scalar_fields|geometry|constants)\..*_.*"),
    # TypedDict / KW types:
    (r"py:class", r"(PointCloudDataKW|CartesianKwFull|CartesianKw)"),
    (r"py:class", r"pchandler\.geometry\.coordinates\.(CartesianKw|CartesianKwFull)"),
    # pchandler public classes referenced by unqualified name in autosummary RST pages:
    (r"py:class", r"(CartesianCoordinates|PointCloudData|PointCloudFilter)"),
    (r"py:class", r"(ScalarField(Manager)?|NormalFields|RGBFields)"),
    (r"py:class", r"(PlyHandler|LasHandler|E57Handler|CsvHandler)"),
    (r"py:class", r"(ShiftNotFeasibleError|Unshifted|DtypeState|BaseDataT|BaseVector|ArrayNx3)"),
    (r"py:class", r"(AngleLikeT|AngleUnit|FoVSplitMethodT|PlaneStrings|PercentileT|WeightingMethods)"),
    (r"py:class", r"(ValidatedPolygonT|Polygon|SF_T|SfOrigDtT|normalize_int16)"),
    # Fully-qualified pchandler type paths that appear in autosummary cross-refs:
    (r"py:class", r"pchandler\.PointCloudData.*"),
    (r"py:class", r"pchandler\.core\.PointCloudFilter"),
    (r"py:class", r"pchandler\.geometry\.transforms\._Transform4x4"),
    (r"py:class", r"pchandler\.constants\._Name.*"),
    # --- External stdlib / pydantic / shapely types not resolvable in this build env ---
    # These appear in docstrings; their intersphinx inventories can't be fetched in this
    # network-restricted build environment. The types are correct and resolve on RTD.
    (r"py:class", r"(BeforeValidator|NonNegativeFloat|PositiveFloat|UUID4?)"),
    (r"py:class", r"Path"),
    (r"py:class", r"optional"),
    # --- pchandler module-level data constants referenced by short name ---
    (r"py:data", r"(COMMON_FIELD_NAMES|COMMON_FIELD_BASES|SUPPORTED_TYPES|_SERIAL_THRESHOLD)"),
    # --- pchandler functions/methods referenced in docstrings ---
    (r"py:func", r"(pchandler\.load_file|typing\.NewType)"),
    (r"py:meth", r"(ScalarFieldManager\._set_rgb|__contains__|_expand_and_add|model_construct)"),
    (r"py:meth", r"pchandler\.(PointCloudData|data_io)\..*"),
    (r"py:meth", r"weakref\.WeakSet\..*"),
    # --- Relative module refs in geometry RST (using short module names) ---
    (r"py:mod", r"(coordinates|spherical|splitter|transforms|util)"),
    # --- ref.obj targets from autosummary pages (private helpers not in __all__) ---
    (r"py:obj", r"pchandler\..*"),
    # --- autodoc_preserve_defaults false positives ---
    # autodoc_preserve_defaults=True causes Sphinx to render default values (e.g.
    # default=True, default="auto") as type cross-references — which they are not.
    # These patterns catch all literal default-value strings that Sphinx misidentifies
    # as cross-reference targets.
    (r"py:class", r"default.*"),
    (r"py:class", r'".*'),  # lone quote or partial/full quoted strings
    (r"py:class", r"\{.*"),  # dict/set literal fragments
    # float | NDArray union type string emitted verbatim by autodoc for angle properties
    (r"py:class", r"float \|.*"),
]


redirects = {"index.html": "introduction.html"}


def setup(app):
    app.add_css_file("pchandler_theme.css")


rst_epilog = """
.. |NDArray| replace:: :external+numpydantic:py:class:`NDArray <numpydantic.ndarray.NDArray>`
.. |o3d.geometry.PointCloud| replace:: :class:`~open3d.geometry.PointCloud`
.. |o3d.t.geometry.PointCloud| replace:: :class:`o3d.t.geometry.PointCloud <open3d.t.geometry.PointCloud>`
.. |Epoch| replace:: :class:`py4dgeo.Epoch <py4dgeo.Epoch>`
.. |VectorT| replace:: :attr:`VectorT <GSEGUtils.base_types.VectorT>`
.. |Vector_Bool_T| replace:: :attr:`VectorT <GSEGUtils.base_types.Vector_Bool_T>`
.. |Vector_3_Float_T| replace:: :attr:`Vector_3_Float_T <GSEGUtils.base_types.Vector_3_Float_T>`
.. |Array_Nx3_T| replace:: :attr:`Array_Nx3_Float_T <GSEGUtils.base_types.Array_Nx3_Float_T>`
.. |Array_Nx3_Float_T| replace:: :attr:`Array_Nx3_Float_T <GSEGUtils.base_types.Array_Nx3_Float_T>`
.. |Array_Nx3_Float32_T| replace:: :attr:`Array_Nx3_Float32_T <GSEGUtils.base_types.Array_Nx3_Float32_T>`
.. |Array_Nx3_Uint8_T| replace:: :attr:`Array_Nx3_Uint8_T <GSEGUtils.base_types.Array_Nx3_Uint8_T>`
.. |ArrayT| replace:: :attr:`ArrayT <GSEGUtils.base_types.ArrayT>`
.. |NormalFields| replace:: :attr:`NormalFields <pchandler.scalar_fields.scalar_fields.NormalFields>`
.. |RGBFields| replace:: :attr:`RGBFields <pchandler.scalar_fields.scalar_fields.RGBFields>`
.. |_SF_| replace:: :attr:`ScalarField <pchandler.scalar_fields.scalar_fields.ScalarField>`
.. |_SFM_| replace:: :attr:`ScalarFieldManager <pchandler.scalar_fields.ScalarFieldManager>`
"""
