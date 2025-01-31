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
    $ python setup.py install

Dependencies
------------

:math:`\texttt{SEDA}` uses several python packages, as indicated below.

Included in the `Ananconda <https://docs.continuum.io/>`_ installation:

* `astropy <http://www.astropy.org/>`_
* `matplotlib <http://matplotlib.org/>`_
* `numpy <http://www.numpy.org/>`_
* `scipy <https://www.scipy.org/>`_
* `specutils <https://pypi.org/project/specutils/>`_

External packages that must be installed separately (using ``pip``):

* `corner <http://corner.readthedocs.io/en/latest/>`_
* `dynesty <https://dynesty.readthedocs.io/en/stable/>`_
* `fnmatch <https://docs.python.org/3/library/fnmatch.html>`_
* `lmfit <https://pypi.org/project/lmfit/>`_
* `prettytable <https://pypi.org/project/prettytable/>`_
* `spectres <https://spectres.readthedocs.io/en/latest/>`_
* `tqdm <https://pypi.org/project/tqdm/>`_
* `xarray <https://docs.xarray.dev/en/stable/>`_

These packages can be installed using ``pip``:

.. code-block:: console

    $ pip install astropy matplotlib numpy scipy specutils
    $ pip install corner dynesty fnmatch lmfit prettytable spectres tqdm xarray

:math:`\texttt{SEDA}` has been tested in Python 3.9, 3.10, and 3.11 security versions as of Nov. 2024.

:math:`\texttt{SEDA}` has been tested in Linux (Ubuntu 22.04), Windows, and macOS.
