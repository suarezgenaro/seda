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
python -m pip install .
```

For developer notes and advanced installation (editable installs, building wheels, building docs), see docs/installation.rst or read it online at https://seda.readthedocs.io/en/latest/installation.html
