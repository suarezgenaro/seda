Installation and Dependencies
=============================

Installation
------------

We recommend to make a conda environment to install :math:`\texttt{SEDA}`:

.. code-block:: console

    $ conda create -n env_seda
    $ conda activate env_seda

Installation of :math:`\texttt{SEDA}` via GitHub:

.. code-block:: console

    $ git clone https://github.com/suarezgenaro/seda.git
    $ cd seda
    $ python -m pip install .

All required dependencies are installed automatically.


Uninstallation
--------------

To uninstall :math:`\texttt{SEDA}`:

.. code-block:: console

    $ python -m pip uninstall seda


Dependencies
------------

:math:`\texttt{SEDA}` uses several python packages. All dependencies are automatically installed when SEDA is installed. The dependencies are defined in ``pyproject.toml`` and include:

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

:math:`\texttt{SEDA}` has been tested in Python versions 3.9--3.14 and on Linux, Windows, and macOS.


Developer Installation
----------------------

To install SEDA in editable mode for development:

.. code-block:: console

    $ git clone https://github.com/suarezgenaro/seda.git
    $ cd seda
    $ python -m pip install -e ".[docs]"
    $ pre-commit install

The ``-e`` option installs SEDA in editable mode, so changes made to the source code are immediately available in the installed package.

Run the test suite with:

.. code-block:: console

    $ pytest


Contributing
------------

Contributions to SEDA are welcome. To contribute to the code:

1. Fork the `SEDA repository <https://github.com/suarezgenaro/seda>`_ on GitHub.

2. Clone your fork:

   .. code-block:: console

      $ git clone https://github.com/<your-username>/seda.git
      $ cd seda

3. Create a new branch for your changes:

   .. code-block:: console

      $ git checkout -b <branch-name>

4. Install SEDA following the `Developer Installation`_ instructions.


Build the Documentation
-----------------------

Build the HTML documentation with:

.. code-block:: console

    $ sphinx-build -b html docs docs/_build/html


Open ``docs/_build/html/index.html`` in your web browser to view the generated documentation.
