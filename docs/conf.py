# Configuration file for the Sphinx documentation builder.

# -- Project information

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

sys.path.insert(0, os.path.abspath('../')) # Source code dir relative to this file
#sys.path.append(os.path.abspath('../'))
#sys.path.append(os.path.abspath('../seda/spectral_indices/'))

project = 'SEDA'
copyright = '2024, Genaro Suárez'
author = 'Genaro Suárez'

release = '1.0'
#version = '1.0'

# -- General configuration
extensions = ['sphinx_rtd_theme',
              'sphinx.ext.autodoc', # Core library for html generation from docstrings
              'sphinx.ext.autosummary',  # Create neat summary tables for modules/classes/methods etc
              #'sphinx.ext.intersphinx',  # Link to other project's documentation (see mapping below)
              'sphinx.ext.doctest',
              'sphinx.ext.todo',
              'sphinx.ext.mathjax',    
              'sphinx.ext.napoleon', # preprocessor that converts docstrings to correct reStructuredText before autodoc processes them.
              'sphinx.ext.viewcode', # Add a link to the Python source code for classes, functions etc.
              'nbsphinx', # Integrate Jupyter Notebooks and Sphinx
              ]

autosummary_generate = True  # Turn on sphinx.ext.autosummary

# Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc! (.inv file)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# replace "view page source" with "edit on github" in Read The Docs theme
html_context = {
  'display_github': True,  
  'github_user': 'suarezgenaro',
  'github_repo': 'seda',
  'github_version': 'main/docs/',
}

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
#html_theme = 'bizstyle'
#html_theme = 'classic'

html_logo = 'SEDA_logo_nobg.png'
