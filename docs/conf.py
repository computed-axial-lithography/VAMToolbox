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
import codecs
import mock
MOCK_MODULES = ['numpy','scipy','matplotlib','matplotlib.pyplot','scipy.interpolate','skimage','cv2','PIL','imageio','dill']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()


current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "../vamtoolbox"))
sys.path.insert(0, target_dir)


import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

# sys.path.insert(0, os.path.abspath(os.path.join('..', 'VAMToolbox')))


# -- Project information -----------------------------------------------------

project = 'VAMToolbox'
copyright = '2022, Joseph Toombs, Chi Chung Li, Charlie Rackson, Indrasen Bhattacharya, Vishal Bansal'
author = 'Joseph Toombs, Chi Chung Li, Charlie Rackson, Indrasen Bhattacharya, Vishal Bansal'

# The full version, including alpha/beta/rc tags
release = get_version("../vamtoolbox/__init__.py")


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_panels',
    'sphinx_copybutton',
    'autoapi.extension',
    'sphinxcontrib.bibtex',
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
autoapi_dirs = ['../vamtoolbox']
# autoapi_add_toctree_entry = False




# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'
html_logo = "_static/logos/logo_bone.png"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/computed-axial-lithography/VAMToolbox",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

