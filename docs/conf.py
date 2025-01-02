# Configuration file for the Sphinx documentation builder.

# -- Project information

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys


## Get the directory of the current file (conf.py)
#conf_dir = os.path.dirname(os.path.abspath(__file__))
## Add the submodule directory to sys.path
#submodule_dir = os.path.join(conf_dir, '../seda/synthetic_photometry_test/')
#sys.path.insert(0, submodule_dir)


sys.path.insert(0, os.path.abspath('../'))

project = 'SEDA'
copyright = '2024, Genaro Suárez'
author = 'Genaro Suárez'

release = '1.0'
#version = '1.0'

# -- General configuration
extensions = ['sphinx_rtd_theme',
              'sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.todo',
              'sphinx.ext.mathjax',    
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'nbsphinx',
              ]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

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
