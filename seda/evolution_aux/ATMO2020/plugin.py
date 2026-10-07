from pathlib import Path
import numpy as np
from astropy.constants import R_jup, R_sun

from seda.evolution_aux.units import age_from_gyr, to_user_units

# Bundled-table column units. Must match config.json. phy_params converts
# mass, age, and radius to M_jup, Gyr, and R_jup at the user boundary.
_FILE_UNITS = {
    'mass': 'M_sun',
    'age': 'Gyr',
    'radius': 'R_sun',
}

def _read_evolutionary_model(filename):
    """
    Read a combined ATMO evolutionary table and return a dictionary of grid arrays.

    Lines that are not six-field numeric rows (e.g. # headers) are skipped.
    """
    _MODEL_DIR = Path(__file__).parent
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

    # map file columns to SEDA grid keys
    out = {
        'mass':   data[:, 0],
        'age':    data[:, 1],
        'Teff':   data[:, 2],
        'logL':   data[:, 3],
        'radius': data[:, 4],
        'logg':   data[:, 5],
    }

    return out

def _convert_inputs(Lbol, eLbol, R, eR):
    """
    Convert user inputs to grid interpolation-axis units.

    Users pass Lbol in L_sun and R in R_jup. Returns
    the ``logL`` and ``radius`` axes used by the evolutionary grid.
    """

    # convert radius to the native grid unit (R_sun for ATMO2020 example)
    R_rsun = (R * R_jup).to(R_sun).value
    eR_rsun = (eR * R_jup).to(R_sun).value

    # evol_params interpolates in logL, not linear L
    logL = np.log10(Lbol)
    e_logL = eLbol / (Lbol * np.log(10))  # Gaussian propagation e_L -> e_logL for Monte Carlo sampling

    # keys must match the interpolation axes declared in the grid
    return {
        'logL': logL, 'e_logL': e_logL,
        'radius': R_rsun, 'e_radius': eR_rsun,
    }


def _age_to_grid(age_gyr):
    """Convert a user age in Gyr to this table's age coordinate."""

    return age_from_gyr(age_gyr, _FILE_UNITS['age'])


def _to_user_units(param, values):
    """Convert an interpolated column to M_jup, Gyr, or R_jup when applicable."""

    return to_user_units(param, values, _FILE_UNITS)
