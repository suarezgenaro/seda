Installation and Dependencies
=============================

Installation
------------

We recommend to make a conda environment to install :math:`\texttt{SEDA}` using Python 3.11 (most recent security version):

.. code-block:: console

    $ conda create -n env_seda python=3.11
    $ conda activate env_seda

Installation of :math:`\texttt{SEDA}` via GitHub:

.. code-block:: console

    $ git clone https://github.com/suarezgenaro/seda.git
    $ cd seda
    $ pip install -e .

Dependencies
------------

:math:`\texttt{SEDA}` uses several python packages. All dependencies are automatically installed when you install SEDA using ``pip install -e .`` as shown above. The dependencies are defined in ``pyproject.toml`` and include:

* `astropy <http://www.astropy.org/>`_
* `corner <http://corner.readthedocs.io/en/latest/>`_
* `dynesty <https://dynesty.readthedocs.io/en/stable/>`_
* `lmfit <https://pypi.org/project/lmfit/>`_
* `matplotlib <http://matplotlib.org/>`_
* `numpy <http://www.numpy.org/>`_
* `prettytable <https://pypi.org/project/prettytable/>`_
* `scipy <https://www.scipy.org/>`_
* `spectres <https://spectres.readthedocs.io/en/latest/>`_
* `specutils <https://pypi.org/project/specutils/>`_
* `tqdm <https://pypi.org/project/tqdm/>`_
* `xarray <https://docs.xarray.dev/en/stable/>`_

:math:`\texttt{SEDA}` has been tested in Python 3.9, 3.10, and 3.11 security versions as of Nov. 2024.

:math:`\texttt{SEDA}` has been tested in Linux (Ubuntu 22.04), Windows, and macOS.
