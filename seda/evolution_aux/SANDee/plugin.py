from pathlib import Path

import numpy as np

_MODEL_DIR = Path(__file__).parent


def _read_evolutionary_model(filename):
    """
    Read a SANDee combined mass table and return a dictionary of grid arrays.

    Returns mass in M_jup, age in Gyr, radius in R_jup, Teff in Kelvin, logL as
    log10(L/Lsun), and logg as log10(g[cm/s^2]).

    Bundled tables are whitespace-delimited with a one-line header. They are
    built from the authors' MESA histories via build_tables.py.
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
