from pathlib import Path

import numpy as np
from astropy.constants import R_jup, R_sun

from seda.evolution_aux.units import age_from_gyr, to_user_units

_MODEL_DIR = Path(__file__).parent

# Bundled-table column units. Must match config.json. phy_params converts
# mass, age, and radius to M_jup, Gyr, and R_jup at the user boundary.
# Age stays log10(yr) inside the table so interpolation is unchanged.
_FILE_UNITS = {
    'mass': 'M_sun',
    'age': 'log10(yr)',
    'radius': 'R_sun',
}


def _read_evolutionary_model(filename):
    """
    Read a BHAC15 tracks+structure table and return a dictionary of grid arrays.

    Lines that are not thirteen-field numeric rows (e.g. header comments) are skipped..
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
        'mass':     data[:, 0],
        'age':      data[:, 1],
        'Teff':     data[:, 2],
        'logL':     data[:, 3],
        'logg':     data[:, 4],
        'radius':   data[:, 5],
        'logLi':    data[:, 6],
        'logTc':    data[:, 7],
        'logRho_c': data[:, 8],
        'Mrad':     data[:, 9],
        'Rrad':     data[:, 10],
        'k2conv':   data[:, 11],
        'k2rad':    data[:, 12],
    }


def _convert_inputs(Lbol, eLbol, R, eR):
    """
    Convert user inputs to grid interpolation-axis units.

    Users pass Lbol in L_sun and R in R_jup. Returns
    the ``logL`` and ``radius`` axes used by the evolutionary grid.
    """
    R_rsun = (R * R_jup).to(R_sun).value
    eR_rsun = (eR * R_jup).to(R_sun).value
    logL = np.log10(Lbol)
    e_logL = eLbol / (Lbol * np.log(10))

    return {
        'logL': logL, 'e_logL': e_logL,
        'radius': R_rsun, 'e_radius': eR_rsun,
    }


def _age_to_grid(age_gyr):
    """Convert a user age in Gyr to this table's log10(yr) age coordinate."""

    return age_from_gyr(age_gyr, _FILE_UNITS['age'])


def _to_user_units(param, values):
    """Convert an interpolated column to M_jup, Gyr, or R_jup when applicable."""

    return to_user_units(param, values, _FILE_UNITS)
