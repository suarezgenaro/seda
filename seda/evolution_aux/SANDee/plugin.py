from pathlib import Path

import numpy as np

from seda.evolution_aux.units import age_from_gyr, to_user_units

_MODEL_DIR = Path(__file__).parent

# Bundled-table column units. Must match config.json. phy_params converts
# mass, age, and radius to M_jup, Gyr, and R_jup at the user boundary.
_FILE_UNITS = {
    'mass': 'M_jup',
    'age': 'Gyr',
    'radius': 'R_jup',
}


def _read_evolutionary_model(filename):
    """
    Read a SANDee combined mass table derived from MESA history files.

    Tables are whitespace-delimited with a one-line header and columns already
    converted to M_jup, Gyr, and R_jup.
    """

    table_path = _MODEL_DIR / filename
    try:
        data = np.loadtxt(table_path, comments='#')
    except Exception as exc:
        raise ValueError(
            f'Failed to read SANDee table "{table_path}": {exc}'
        ) from exc

    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(
            f'Expected six-column numeric rows in "{table_path}", '
            f'got shape {getattr(data, "shape", None)}.'
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
    ``radius`` axes used by the evolutionary grid (log10 L/Lsun, R_jup).
    """

    logL = np.log10(Lbol)
    e_logL = eLbol / (Lbol * np.log(10))
    return {
        'logL': logL, 'e_logL': e_logL,
        'radius': R, 'e_radius': eR,
    }


def _age_to_grid(age_gyr):
    """Convert a user age in Gyr to this table's age coordinate."""

    return age_from_gyr(age_gyr, _FILE_UNITS['age'])


def _to_user_units(param, values):
    """Convert an interpolated column to M_jup, Gyr, or R_jup when applicable."""

    return to_user_units(param, values, _FILE_UNITS)
