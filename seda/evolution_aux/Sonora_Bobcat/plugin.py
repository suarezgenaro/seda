from pathlib import Path

import numpy as np
from astropy.constants import M_jup, M_sun, R_jup, R_sun

_MODEL_DIR = Path(__file__).parent

# The bundled Sonora Bobcat tables store mass in M_sun and radius in R_sun.
_MSUN_TO_MJUP = (1.0 * M_sun / M_jup).decompose().value
_RSUN_TO_RJUP = (1.0 * R_sun / R_jup).decompose().value

def _read_evolutionary_model(filename):
    """
    Read a Sonora Bobcat ``*_mass`` evolutionary table and return a dictionary of grid arrays.

    Returns mass in M_jup, age in Gyr, radius in R_jup, Teff in Kelvin, logL as
    log10(L/Lsun), and logg as log10(g[cm/s^2]).

    The seventh column (log I) is parsed out and not returned.
    """

    table_path = _MODEL_DIR / filename
    with open(table_path) as evo_file:
        rows = [line.split() for line in evo_file]
    data = np.array([row for row in rows if len(row) == 7], dtype=float)
    if data.size == 0:
        raise ValueError(f'No seven-column data rows were found in "{table_path}". '
                         f'Pass a Sonora Bobcat *_mass evolutionary table.')

    out = {'mass': data[:, 0] * _MSUN_TO_MJUP, 'age': data[:, 1], 'logL': data[:, 2],
           'Teff': data[:, 3], 'logg': data[:, 4], 'radius': data[:, 5] * _RSUN_TO_RJUP}

    return out
