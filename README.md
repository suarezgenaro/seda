SEDA: Spectral Energy Distribution Analyzer
===========================================

<p align="center">
    <img src="https://github.com/suarezgenaro/seda/blob/main/docs/SEDA_logo.png" title="SEDA logo" alt="Spitzer IRS spectra of ultracool objects" width="400">
</p>

SEDA is an open-source Python package for forward modeling and empirical analysis of spectral energy distributions for ultracool objects, including brown dwarfs, directly imaged exoplanets, and low-mass stars. The code compares observed spectrophotometric data to atmospheric models by using a Bayesian framework to sample posteriors. Alternatively, the code minimizes the chi-square statistic to find the best model fits. SEDA also includes several tools useful for the analysis of spectral energy distributions and a variety of functions to visualize atmospheric models and results.

Check out the documentation at https://seda.readthedocs.io

Installation and Build
----------------------

SEDA uses a `pyproject.toml` (PEP 621) configuration with setuptools. To install, use the following commands in a terminal:

```zsh
# Upgrade build tooling
python -m pip install --upgrade pip build
```

Developer notes
---------------
```zsh

# Install in editable mode
python -m pip install -e .

# Install with tools needed to build the docs 
python -m pip install -e .[docs]

# Build source distribution and wheel
python -m build
```

To run the docs locally (optional):

```zsh
python -m pip install -e .[docs]
pushd docs
make html
open _build/html/index.html
popd
```
