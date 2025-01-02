# Configuration file for the Sphinx documentation builder.

# -- Project information

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
sys.path.insert(0, os.path.abspath('../'))


# TEST
# Get the absolute path of the submodule directory
submodule_path = os.path.abspath('../seda/spectral_indices')
# Add the submodule directory to sys.path
sys.path.append(submodule_path)


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
