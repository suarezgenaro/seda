Installation
============

Installation of :math:`\texttt{SEDA}` via GitHub:

.. code-block:: console

    $ git clone https://github.com/suarezgenaro/seda.git
    $ cd seda
    $ python setup.py install

Dependencies
============
The :math:`\texttt{SEDA}` code uses the following packages:

Included in the `Ananconda <https://docs.continuum.io/>`_ installation:

* `astropy <http://www.astropy.org/>`_
* `matplotlib <http://matplotlib.org/>`_
* `numpy <http://www.numpy.org/>`_
* `scipy <https://www.scipy.org/>`_

External packages that must be installed separately (using ``pip``):

* `corner <http://corner.readthedocs.io/en/latest/>`_
* `dynesty <https://dynesty.readthedocs.io/en/stable/>`_
* `lmfit <https://pypi.org/project/lmfit/>`_
* `spectres <https://spectres.readthedocs.io/en/latest/>`_
* `xarray <https://docs.xarray.dev/en/stable/>`_

These packages can be installed using ``pip``:

.. code-block:: console

    $ pip install astropy matplotlib numy scipy
    $ pip install corner dynesty lmfit spectres xarray
