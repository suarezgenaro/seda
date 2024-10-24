Overview
========

Introduction
------------
:math:`\texttt{SEDA}`

Atmospheric Models
------------------

- **Sonora Diamondback**:

  Cloudy (silicate clouds) atmospheric models assuming chemical equilibrium but considering the effect of both clouds and metallicity by `Morley et al (2024) <https://ui.adsabs.harvard.edu/abs/2024arXiv240200758M/abstract>`_.

  Parameter coverage:

  .. code-block:: console

    **Parameter coverage:**
    wavelength = [0.3, 250] um
    Teff = [900, 2400] K in steps of 100 K
    logg = [3.5, 5.5] in steps of 0.5
    [M/H] = [-0.5, 0.5] (cgs) in steps of 0.5
    fsed = 1, 2, 3, 4, 8, nc

- **Sonora Elf Owl**:

  Models with atmospheric mixing and chemical disequilibrium with varying metallicity and C/O by `Mukherjee et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024ApJ...963...73M/abstract>`_.

  Parameter coverage:

  .. code-block:: console

    - wavelength = [0.6, 15] um
    - Teff = [275, 2400] K in steps: 25 K for 275-600 K, 50 K for 600-1000 K, and 100 K for 1000-2400 K
    - logg = [3.25, 5.50] in steps of 0.25 dex
    - logKzz = 2, 4, 7, 8, and 9 (Kzz in cm2/s)
    - [M/H] = [-1.0, 1.0] (cgs) with values of -1.0, -0.5, +0.0, +0.5, +0.7, and +1.0
    - C/O = [0.5, 2.5] with steps of 0.5 (relative to solar C/O, assumed as 0.458) (these are the values in the filenames). It corresponds to C/O=[0.22, 1.12] with values of 0.22, 0.458, 0.687, and 1.12 (e.g. 0.5 in the filename means 0.5*0.458=0.22)

- `Sonora Diamondback <https://ui.adsabs.harvard.edu/abs/2024arXiv240200758M/abstract>`_ by `Morley et al (2024) <https://ui.adsabs.harvard.edu/abs/2024arXiv240200758M/abstract>`_.
- **Sonora Elf Owl** for `L-type <https://zenodo.org/records/10385987>`_, `T-type <https://zenodo.org/records/10385821>`_, and `Y-type <https://zenodo.org/records/10381250>`_ by `Mukherjee et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024ApJ...963...73M/abstract>`_.
- `LB23 <https://zenodo.org/records/7779180>`_ by `Lacy & Burrows (2023) <https://ui.adsabs.harvard.edu/abs/2023ApJ...950....8L/abstract>`_.
- `Sonora Cholla <https://zenodo.org/records/4450269>`_ by `Karalidi et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJ...923..269K/abstract>`_.
- `Sonora Bobcat <https://zenodo.org/records/5063476>`_ by `Marley et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJ...920...85M/abstract>`_.
- `ATMO 2020 <https://noctis.erc-atmo.eu/fsdownload/zyU96xA6o/phillips2020>`_ by `Phillips et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020A%26A...637A..38P/abstract>`_.
- `BT-Settl <http://phoenix.ens-lyon.fr/simulator/>`_ by `Allard et al. (2012) <https://ui.adsabs.harvard.edu/abs/2012RSPTA.370.2765A/abstract>`_.
- `Saumon & Marley (2008) <https://ui.adsabs.harvard.edu/abs/2008ApJ...689.1327S>`_ (private communication with the authors for data).
