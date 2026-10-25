from pathlib import Path

import numpy as np
from astropy.constants import M_jup, M_sun, R_jup, R_sun

_MODEL_DIR = Path(__file__).parent

# The bundled ATMO tables store mass in M_sun and radius in R_sun.
_MSUN_TO_MJUP = (1.0 * M_sun / M_jup).decompose().value
_RSUN_TO_RJUP = (1.0 * R_sun / R_jup).decompose().value

def _read_evolutionary_model(filename):
    """
    Read a combined ATMO evolutionary table and return a dictionary of grid arrays.

    Returns mass in M_jup, age in Gyr, radius in R_jup, Teff in Kelvin, logL as
    log10(L/Lsun), and logg as log10(g[cm/s^2]).

    Lines that are not six-field numeric rows (e.g. # headers) are skipped.
    """
    table_path = _MODEL_DIR / filename

    with open(table_path) as evo_file:
        rows = [line.split() for line in evo_file]

    # keep only six-column numeric rows; skip # header lines
    data = np.array([row for row in rows if len(row) == 6], dtype=float)
    if data.size == 0:
        raise ValueError(
            f'No six-column data rows were found in "{table_path}". '
            f'Expected mass, age, Teff, logL, radius, logg.'
        )

    # map file columns to SEDA grid keys, converting to the returned units
    # (the bundled ATMO tables store mass in M_sun and radius in R_sun)
    out = {
        'mass':   data[:, 0] * _MSUN_TO_MJUP,
        'age':    data[:, 1],
        'Teff':   data[:, 2],
        'logL':   data[:, 3],
        'radius': data[:, 4] * _RSUN_TO_RJUP,
        'logg':   data[:, 5],
    }

    return out
