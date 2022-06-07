# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "../VAMToolbox"))
sys.path.insert(0, target_dir)

print(target_dir)
# sys.path.insert(0, os.path.abspath(os.path.join('..', 'VAMToolbox')))


# -- Project information -----------------------------------------------------

project = 'VAMToolbox'
copyright = '2022, Joseph Toombs, Chi Chung Li, Charlie Rackson, Indrasen Bhattacharya, Vishal Bansal'
author = 'Joseph Toombs, Chi Chung Li, Charlie Rackson, Indrasen Bhattacharya, Vishal Bansal'

# The full version, including alpha/beta/rc tags
release = '0.1.0beta'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinxcontrib.napoleon',
    'sphinx_panels',
    'sphinx_copybutton',
    # 'sphinxcontrib.video',
    'autoapi.extension',
    'sphinxcontrib.bibtex',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'autoapi.extension',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# # Bibtex
# import pybtex.plugin
# from pybtex.style.formatting.unsrt import Style as UnsrtStyle
# from pybtex.style.template import words

# extensions = ['sphinxcontrib.bibtex']
exclude_patterns = ['_build']
bibtex_reference_style = 'author_year'
bibtex_bibfiles = ['refs.bib']



# class NoWebRefStyle(UnsrtStyle):
#     def format_web_refs(self, e):
#         # the following is just one simple way to return an empty node
#         return words['']

# pybtex.plugin.register_plugin('pybtex.style.formatting', 'nowebref', NoWebRefStyle)
# bibtex_default_style = 'nowebref'


# AutoAPI
autoapi_type = 'python'
autoapi_dirs = ['../VAMToolbox']
# autoapi_add_toctree_entry = False




# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']