.. _overview:

Overview
========

:math:`\texttt{SEDA}` (Spectral Energy Distribution Analyzer) is an open-source Python package for forward modeling analysis of brown dwarfs, directly imaged exoplanets, and low-mass stars. 

.. _seda_overview:

Principal Modules
-----------------
  - :meth:`~seda.input_parameters`: to define all input parameters, namely :meth:`~seda.input_parameters.InputData`, :meth:`~seda.input_parameters.ModelOptions`, :meth:`~seda.input_parameters.Chi2Options`, and :meth:`~seda.input_parameters.BayesOptions`.
  - :meth:`~seda.chi2_fit`: to find the best model fits from :ref:`models` using `LMFIT <https://lmfit.github.io/lmfit-py/>`_ non-linear least-square minimization python package by `Newville et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014zndo.....11813N/abstract>`_.
  - :meth:`~seda.bayes_fit`: to estimate Bayesian posteriors using `dynesty <https://dynesty.readthedocs.io/en/stable/index.html>`_ dynamic nested sampling package by `Speagle (2020) <https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S/abstract>`_.
  - :meth:`~seda.plots`: plotting functions.
  - :meth:`~seda.utils`: additional useful functions (see :ref:`seda_tools`).

Code Workflow
-------------
  - Download any :ref:`models`.
  - Load input data (:meth:`~seda.input_parameters.InputData`).
  - Load model options (:meth:`~seda.input_parameters.ModelOptions`).

  - Option 1: chi-square minimization (see tutorial :ref:`tutorials_chi2`).

    - Load chi-square fit options (:meth:`~seda.input_parameters.Chi2Options`).
    - Run chi-square minimization module (:meth:`~seda.chi2_fit`).

  - Option 2: Bayesian sampling  (see tutorial :ref:`tutorials_bayes`).

    - Load Bayesian fit options (:meth:`~seda.input_parameters.BayesOptions`).
    - Run Bayesian framework module (:meth:`~seda.bayes_fit`).

  - Visualize the results (:meth:`~seda.plots`)

.. _seda_tools:

Main Tools
----------
:math:`\texttt{SEDA}` includes several useful functions that can be used separately:
  - :meth:`~seda.synthetic_photometry.synthetic_photometry.synthetic_photometry`: to calculate synthetic phothometry from spectra for any `SVO filter IDs <http://svo2.cab.inta-csic.es/theory/fps/>`_ (see tutorial :ref:`tutorials_synthetic_photometry`).
  - :meth:`~seda.spectral_indices.spectral_indices`: to measure the depth of key feature in mid-infrared spectra namely silicates (:meth:`~seda.spectral_indices.spectral_indices.silicate_index`) (see tutorial :ref:`tutorials_spectral_indices`).
  - :meth:`~seda.utils.convolve_spectrum`: to convolve spectra to a desired resolution at a given wavelength.
  - :meth:`~seda.synthetic_photometry.synthetic_photometry.convert_flux`: to convert fluxes from wavelength units (erg/s/cm2/s) to frequency units (Jy) or vice versa.
  - :meth:`~seda.synthetic_photometry.synthetic_photometry.flux_to_mag`: to convert fluxes into magnitudes for any `SVO filter IDs <http://svo2.cab.inta-csic.es/theory/fps/>`_.
  - :meth:`~seda.synthetic_photometry.synthetic_photometry.mag_to_flux`: to convert magnitudes into fluxes for any `SVO filter IDs <http://svo2.cab.inta-csic.es/theory/fps/>`_.
  - :meth:`~seda.utils.read_model_spectrum`: to read a synthetic spectrum from :ref:`models`.
  - :meth:`~seda.utils.generate_model_spectrum`: to generate a synthetic spectrum with any parameters (within the grid coverage) from the indicated atmospheric models using interpolation techniques.
  - :meth:`~seda.utils.app_to_abs_flux`: to convert apparent fluxes into absolute fluxes considering a distance.
  - :meth:`~seda.utils.separate_params`: to extract parameters from model spectra file names.
  - :meth:`~seda.plots.plot_model_coverage` and :meth:`~seda.plots.plot_model_resolution`: to visualize parameters' coverage and resolution of :ref:`models` (see tutorial :ref:`tutorials_model_examination`).

.. _models:

Available Atmospheric Models
----------------------------

:math:`\texttt{SEDA}` can use several modern and widely used atmospheric models, as indicated below. Check out the tutorial :ref:`tutorials_model_examination` to see the parameters' coverage and resolution of the models.

Sonora Diamondback Models
+++++++++++++++++++++++++

Cloudy (silicate clouds) atmospheric models assuming chemical equilibrium but considering the effect of both clouds and metallicity by `Morley et al (2024) <https://ui.adsabs.harvard.edu/abs/2024arXiv240200758M/abstract>`_. Download the `Sonora Diamondback models <https://ui.adsabs.harvard.edu/abs/2024arXiv240200758M/abstract>`_.

Parameter coverage:

.. code-block:: console

  - wavelength = [0.3, 250] um
  - Teff = [900, 2400] K in steps of 100 K
  - logg = [3.5, 5.5] in steps of 0.5
  - [M/H] = [-0.5, 0.5] (cgs) in steps of 0.5
  - fsed = 1, 2, 3, 4, 8, nc


Sonora Elf Owl Models
+++++++++++++++++++++

Models with atmospheric mixing and chemical disequilibrium with varying metallicity and C/O by `Mukherjee et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024ApJ...963...73M/abstract>`_. Download the Sonora Elf Owl models for `L-type <https://zenodo.org/records/10385987>`_, `T-type <https://zenodo.org/records/10385821>`_, and `Y-type <https://zenodo.org/records/10381250>`_.

Parameter coverage:

.. code-block:: console

  - wavelength = [0.6, 15] um
  - Teff = [275, 2400] K in steps: 25 K for 275-600 K, 50 K for 600-1000 K, and 100 K for 1000-2400 K
  - logg = [3.25, 5.50] in steps of 0.25 dex plus logg=3.0 for Teff=[275-2000], logKzz=8, [M/H]=1.0, and C/O=1.0.
  - logKzz = 2, 4, 7, 8, and 9 (Kzz in cm2/s)
  - [M/H] = -1.0, -0.5, +0.0, +0.5, +0.7, and +1.0 (cgs)
  - C/O = 0.5, 1.0, 1.5, 2.5 (relative to solar C/O, assumed as 0.458) (these are the values in the filenames). It corresponds to C/O=[0.22, 1.12] with values of 0.22, 0.458, 0.687, and 1.12 (e.g. 0.5 in the filename means 0.5*0.458=0.22)

Lacy & Burrows (2023) Models
++++++++++++++++++++++++++++

Cloudy (water clouds) atmospheric models with equilibrium and non-equilibrium chemistry for Y-dwarf atmospheres by `Lacy & Burrows (2023) <https://ui.adsabs.harvard.edu/abs/2023ApJ...950....8L/abstract>`_. Download the `LB23 models <https://zenodo.org/records/7779180>`_. The extended models are shared on request to the authors. The models include four grids: 

  - ClearEQ: cloudless models with equilibrium chemistry
  - ClearNEQ: cloudless models with non-equilibrium chemistry
  - CloudyEQ: cloudy models with equilibrium chemistry
  - CloudyNEQ: cloudy models with non-equilibrium chemistry

Parameter coverage in common for all grids:

.. code-block:: console

  - wavelength = [0.5, 300] um with 30,000 frequency points evenly spaced in ln(frequency)
  - R~4340 (average resolving power)

Parameter coverage for cloudless models:

.. code-block:: console

  - Teff = [200, 600] K in steps of 25 K
  - logg = [3.50, 5.00] in steps of 0.25 (g in cgs)
  - [M/H] = -0.5, 0.0, and 0.5 (Z/Z_sun = 0.316, 1.0, 3.16)
  - logKzz = 6 for non-equilibrium models
  
Parameter coverage for cloudy models (there are some additional cloudy atmospheres extending to lower surface gravities and warmer temperatures in some combinations where convergence was easy): 

.. code-block:: console

  - Teff = [200, 400] K (200-350 for Z/Z_sun=3.16) in steps of 25 K 
  - logg = [3.75, 5.00] in steps of 0.25 (g in cgs)
  - [M/H] = -0.5, 0.0, and 0.5 (Z/Z_sun = 0.316, 1.0, 3.16), but some Z/Z_sun=3.16 are missing
  - logKzz = 6 for non-equilibrium models
  
Extended models (additions to models in the paper). This grid replaces the original one ("The original spectra had an inconsistent wavelength grid and was missing CO2, so new ones are really a replacement.")

.. code-block:: console
  
  - Teff up to 800 K
  - Hmix (mixing length) = 1.0, 0.1, and 0.01

Sonora Cholla Models
++++++++++++++++++++

Cloudless models with non-equilibrium chemistry due to different eddy diffusion parameters by `Karalidi et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJ...923..269K/abstract>`_. Download the `Sonora Cholla models <https://zenodo.org/records/4450269>`_.

Parameter coverage:

.. code-block:: console

  - wavelength = [1, 250] um for Teff>=850 K (plus some with Teff=750 K)
  - wavelength = [0.3, 250] um for Teff<800 K (plus 950K_1780g_logkzz2.spec)
  - Teff = [500, 1300] K in steps of 50 K
  - logg = [3.00, 5.50] in steps of 0.25 (g in cgs)
  - log Kzz=2, 4, and 7

Sonora Bobcat Models
++++++++++++++++++++

Cloudless models in chemical equilibrium by `Marley et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJ...920...85M/abstract>`_. Download the `Sonora Bobcat models <https://zenodo.org/records/5063476>`_.

Parameter coverage:

.. code-block:: console
  
  - wavelength = [0.4, 50] um
  - Teff = [200, 2400] K in steps: 25 K for 200-600 K, 50 K for 600-1000 K, and 100 K for 1000-2400 K
  - logg = [3.25, 5.50] in steps of 0.25 (g in cgs)
  - M/H=-0.5, 0.0, and 0.5
  - C/O = 0.5, 1.0 (solar C/O), and 1.5 for solar metallicity models
  - R = [6000, 200000] (the resolving power varies with wavelength but is otherwise the same for all spectra)

ATMO 2020 Models
++++++++++++++++

Cloudless atmospheric models with chemical and non-chemical equilibrium by `Phillips et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020A%26A...637A..38P/abstract>`_. Download the `ATMO 2020 models <https://noctis.erc-atmo.eu/fsdownload/zyU96xA6o/phillips2020>`_. The models include three grids:
  
  - ATMO2020_CEQ: cloudless models with equilibrium chemistry.
  - ATMO2020_NEQ_weak: cloudless models with non-equilibrium chemistry due to weak vertical mixing (logKzz=4).
  - ATMO2020_NEQ_strong: cloudless models with non-equilibrium chemistry due to strong vertical mixing (logKzz=6).

Parameter coverage:

.. code-block:: console
  
  - wavelength = [0.2, 2000] um
  - Teff = [200, 2400] K in steps varying from 25 K to 100 K
  - logg = [2.5, 5.5] in steps of 0.5 (g in cgs)
  - logKzz = 0 (ATMO2020_CEQ), 4 (ATMO2020_NEQ_weak), and 6 (ATMO2020_NEQ_strong)

BT-Settl Models
+++++++++++++++

Cloudy models with non-equilibrium chemistry by `Allard et al. (2012) <https://ui.adsabs.harvard.edu/abs/2012RSPTA.370.2765A/abstract>`_. Download the `BT-Settl models <http://phoenix.ens-lyon.fr/simulator/>`_.

Parameter coverage:

.. code-block:: console
  
  - wavelength = [1.e-4, 1000] um
  - Teff = [200, 7000] K (Teff<=450 K for only logg<=3.5) in steps varying from 20 K to 100 K
  - logg = [2.0, 5.5] in steps of 0.5 (g in cgs)
  - R = [100000, 500000] (the resolving power varies with wavelength)

Saumon & Marley (2008) Models
+++++++++++++++++++++++++++++

Cloudy models with equilibrium chemistry by `Saumon & Marley (2008) <https://ui.adsabs.harvard.edu/abs/2008ApJ...689.1327S>`_. SM08 models are shared on request to the authors.

Parameter coverage:

.. code-block:: console

  - wavelength = [0.4, 50] um
  - Teff = [800, 2400] K in steps of 100 K
  - logg = [3.0, 5.5] in steps of 0.5 (g in cgs)
  - fsed = 1, 2, 3, 4
  - R = [100000, 700000] (the resolving power varies with wavelength)
