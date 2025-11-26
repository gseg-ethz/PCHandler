import os
import sys

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
version = "1.0.1"  # x-release-please-version
release = "1.0.1"  # x-release-please-version


extensions = [
    "sphinx.ext.autodoc",  # For generating documentation from docstrings
    "sphinx.ext.napoleon",  # For Google-style and NumPy-style docstrings
    "sphinx.ext.autosummary",  # For summary tables
    "sphinx.ext.intersphinx",
]

# TODO fix the GSEGUtils once it's implemented to ReadTheDocs
# Intersphinx Config
intersphinx_mapping = {'open3d': ('https://www.open3d.org/docs/release/', None),
                       'python': ('https://docs.python.org/3/', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'pydantic': ('https://docs.pydantic.dev/latest/', None),
                       'numpydantic': ('https://numpydantic.readthedocs.io/en/latest/', None),
                       'GSEGUtils': ('http://localhost:53413/GSEGUtils/docs/build/html/', '../libs/objects.inv'),}


# General Config
python_use_unqualified_type_name = True                # False

# ======= Autodoc Config =========
autoclass_content = 'class'                              # 'both', 'None, 'init', 'class'
autodoc_class_signature = 'separated'                       # 'mixed' / 'separated
autodoc_member_order = 'bysource'                       # 'alphabetical', 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    # 'special-members': '__init__',
    'inherited-members': False,
    # 'imported-members': True,
    'exclude-members': 'model_config model_post_init _abc_impl _reconstruct',
    # 'ignore-module-all': False,
    'member-order': 'bysource',
    'show-inheritance': True,
}
autodoc_docstring_signature = True                      # True
autodoc_mock_imports = []                               # []
autodoc_typehints = 'description'                       # 'signature', 'description', 'none', 'both'
autodoc_typehints_description_target = 'documented'     # 'all', 'documented', 'documented_params'
autodoc_typehints_format = 'short'                      # 'short', 'fully-qualified'
autodoc_preserve_defaults = True                        # False
autodoc_use_type_comments = True                        # True
autodoc_warningiserror = True                           # True
autodoc_inherit_docstrings = True                       # True
linkcheck_allowed_redirects = {}
autosummary_generate_overwrite = False
# Defaults
templates_path = ["_templates"]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


def setup(app):
    app.add_css_file('pchandler_theme.css')

rst_epilog = """
.. |NDArray| replace:: :external+numpydantic:py:class:`NDArray <numpydantic.NDArray>`
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
