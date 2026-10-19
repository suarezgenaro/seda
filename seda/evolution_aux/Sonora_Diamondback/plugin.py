from pathlib import Path

import numpy as np
from astropy.constants import M_jup, M_sun

_MODEL_DIR = Path(__file__).parent

# The bundled Sonora Diamondback tables store mass in M_sun. Their radius is
# already in R_jup, despite the R_sun label in the published file header.
_MSUN_TO_MJUP = (1.0 * M_sun / M_jup).decompose().value

def _read_evolutionary_model(filename):
    """
    Read a Sonora Diamondback ``*_mass`` evolutionary table and return a dictionary of grid arrays.

    Returns mass in M_jup, age in Gyr, radius in R_jup, Teff in Kelvin, logL as
    log10(L/Lsun), and logg as log10(g[cm/s^2]).
    """

    table_path = _MODEL_DIR / filename
    with open(table_path) as evo_file:
        rows = [line.split() for line in evo_file]
    data = np.array([row for row in rows if len(row) == 6], dtype=float)
    if data.size == 0:
        raise ValueError(f'No six-column data rows were found in "{table_path}". '
                         f'Pass a Sonora Diamondback *_mass evolutionary table.')

    out = {'mass': data[:, 0] * _MSUN_TO_MJUP, 'age': data[:, 1], 'logL': data[:, 2],
           'Teff': data[:, 3], 'logg': data[:, 4], 'radius': data[:, 5]}

    return out
