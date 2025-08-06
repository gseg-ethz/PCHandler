import os
import sys


# from docs.conf import autosummary_generate

sys.path.insert(0, os.path.abspath('../../src'))

import pchandler

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PCHandler'
copyright = '2025, Nicholas Meyer'
author = 'Nicholas Meyer'
version = release = '2.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # For generating documentation from docstrings
    'sphinx.ext.napoleon',  # For Google-style and NumPy-style docstrings
    'sphinx.ext.autosummary',  # For summary tables
]

napoleon_numpy_docstring = True
autosummary_generate = True
autodoc_member_order = 'groupwise'
templates_path = ['_templates']
exclude_patterns = []

autodoc_typehints = 'description'

autodoc_typehints_format = 'short'
python_use_unqualified_type_names = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
