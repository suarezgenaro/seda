from pathlib import Path

import numpy as np
from astropy.constants import R_jup, R_sun

from seda.evolution_aux.units import age_from_gyr, to_user_units

_MODEL_DIR = Path(__file__).parent

# Bundled-table column units. Must match config.json. phy_params converts
# mass, age, and radius to M_jup, Gyr, and R_jup at the user boundary.
_FILE_UNITS = {
    'mass': 'M_sun',
    'age': 'Gyr',
    'radius': 'R_sun',
}


def _read_evolutionary_model(filename):
    """
    Read a Sonora Red Diamondback ``*_mass`` evolutionary table.

    Mass and radius stay in the bundled M_sun and R_sun. Age stays in Gyr.
    """

    table_path = _MODEL_DIR / filename
    with open(table_path) as evo_file:
        rows = [line.split() for line in evo_file]

    # six-column numeric rows only; skip header and mass-block labels
    data = np.array([row for row in rows if len(row) == 6], dtype=float)
    if data.size == 0:
        raise ValueError(
            f'No six-column data rows were found in "{table_path}". '
            f'Pass a Sonora Red Diamondback *_mass evolutionary table.'
        )

    return {
        'mass': data[:, 0],
        'age': data[:, 1],
        'logL': data[:, 2],
        'Teff': data[:, 3],
        'logg': data[:, 4],
        'radius': data[:, 5],
    }


def _convert_inputs(Lbol, eLbol, R, eR):
    """
    Convert user inputs to grid interpolation-axis units.

    Users pass Lbol in L_sun and R in R_jup. Returns the ``logL`` and
    ``radius`` axes used by the evolutionary grid (log10 L/Lsun, R_sun).
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
    """Convert a user age in Gyr to this table's age coordinate."""

    return age_from_gyr(age_gyr, _FILE_UNITS['age'])


def _to_user_units(param, values):
    """Convert an interpolated column to M_jup, Gyr, or R_jup when applicable."""

    return to_user_units(param, values, _FILE_UNITS)
