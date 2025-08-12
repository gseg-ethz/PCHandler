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

# Intersphinx Config
intersphinx_mapping = {'open3d': ('https://www.open3d.org/docs/release/', None),
                       'python': ('https://docs.python.org/3/', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'pydantic': ('https://docs.pydantic.dev/latest/', None),
                       'numpydantic': ('https://numpydantic.readthedocs.io/en/latest/', None)}

# General Config
python_use_unqualified_type_name = True                # False

# ======= Autodoc Config =========
autoclass_content = 'both'                              # 'both'
autodoc_class_signature = 'separated'                       # 'mixed' / 'separated
autodoc_member_order = 'bysource'                       # 'alphabetical'
autodoc_default_options = {             # {}
    'exclude-members': 'model_config'
}
autodoc_docstring_signature = True                      # True
autodoc_mock_imports = []                               # []
autodoc_typehints = 'both'                       # 'signature'
autodoc_typehints_description_target = 'all'     # 'all', 'documented', 'documented_params'
autodoc_type_aliases = {}                               # {}
autodoc_typehints_format = 'short'                      # 'short'
autodoc_preserve_defaults = True                        # False
autodoc_use_type_comments = True                        # True
autodoc_warningiserror = True                           # True
autodoc_inherit_docstrings = True                       # True

linkcheck_allowed_redirects = {}

autosummary_generate = True


# Defaults
templates_path = ["_templates"]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


def setup(app):
    app.add_css_file('pchandler_theme.css')

rst_epilog = """
.. |NDArray| replace:: :external+numpydantic:py:class:`NDArray <numpydantic.NDArray>`
"""
