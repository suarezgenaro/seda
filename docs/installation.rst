Installation and Dependencies
=============================

Installation
------------

We recommend to make a conda environment to install :math:`\texttt{SEDA}` using Python 3.14 (latest stable version):

.. code-block:: console

    $ conda create -n env_seda python=3.14
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

:math:`\texttt{SEDA}` has been tested in Python versions 3.9-3.14.

:math:`\texttt{SEDA}` has been tested on Linux, Windows, and macOS.


Developer Installation
----------------------

To set up a development environment:

.. code-block:: console

    $ git fork https://github.com/suarezgenaro/seda.git
    $ git clone <your-fork-url>
    $ cd seda
    $ pip install -e .[docs]
    $ pre-commit install

Run the test suite:

.. code-block:: console

    $ pytest


Build the Documentation
-----------------------

Build HTML docs:

.. code-block:: console

    $ sphinx-build -b html docs docs/_build/html

Open the generated documentation in your web browser.
