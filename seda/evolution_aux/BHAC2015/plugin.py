from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.constants import M_jup, M_sun, R_jup, R_sun

_MODEL_DIR = Path(__file__).parent

# The bundled BHAC15 table stores mass in M_sun, radius in R_sun, and age as
# log10(yr). The radiative-core columns Mrad and Rrad are also solar.
_MSUN_TO_MJUP = (1.0 * M_sun / M_jup).decompose().value
_RSUN_TO_RJUP = (1.0 * R_sun / R_jup).decompose().value


def _read_evolutionary_model(filename):
    """
    Read a BHAC15 tracks+structure table and return a dictionary of grid arrays.

    Returns mass in M_jup, age in Gyr, radius in R_jup, Teff in Kelvin, logL as
    log10(L/Lsun), and logg as log10(g[cm/s^2]). The radiative-core mass and
    radius are returned in M_jup and R_jup; the remaining structure columns are
    logarithmic or dimensionless and are returned as published.

    Lines that are not thirteen-field numeric rows (e.g. header comments) are skipped.
    """
    table_path = _MODEL_DIR / filename

    with open(table_path) as evo_file:
        rows = [line.split() for line in evo_file]

    data = np.array([row for row in rows if len(row) == 13], dtype=float)
    if data.size == 0:
        raise ValueError(
            f'No thirteen-column data rows were found in "{table_path}". '
            f'Expected mass, log t, Teff, logL, logg, radius, and structure columns.'
        )

    return {
        'mass':     data[:, 0] * _MSUN_TO_MJUP,
        'age':      ((10.0 ** data[:, 1]) * u.yr).to(u.Gyr).value,
        'Teff':     data[:, 2],
        'logL':     data[:, 3],
        'logg':     data[:, 4],
        'radius':   data[:, 5] * _RSUN_TO_RJUP,
#        'logLi':    data[:, 6],
#        'logTc':    data[:, 7],
#        'logRho_c': data[:, 8],
#        'Mrad':     data[:, 9] * _MSUN_TO_MJUP,
#        'Rrad':     data[:, 10] * _RSUN_TO_RJUP,
#        'k2conv':   data[:, 11],
#        'k2rad':    data[:, 12],
    }
