"""
Rebuild the bundled SANDee evolutionary tables from the authors' MESA output.

The SANDee grid is distributed as raw MESA output (one ``LOGS/history.data`` per
chemistry and initial mass) rather than as flat evolutionary tables, so the
tables shipped with SEDA are derived here. This script documents that
derivation and regenerates them.

Provenance of each output column, following the convention of the authors' own
``isogen.py``:

===============  ==============================  ============================
Output column    MESA history column             Conversion
===============  ==============================  ============================
``mass_Mjup``    ``star_mass``                   M_sun -> M_jup
``age_Gyr``      ``star_age``                    yr -> Gyr
``logL_Lsun``    ``log_L``                       none
``Teff_K``       ``log_Teff``                    10**x
``logg_cgs``     ``log_g``                       none
``radius_Rjup``  ``log_R``                       10**x, then R_sun -> R_jup
===============  ==============================  ============================

MESA's ``log_R`` and the Stefan-Boltzmann radius implied by ``log_L`` and
``log_Teff`` agree to ~1e-15, and ``log_g`` agrees with G*M/R**2 to the printed
precision, so reading the native columns is equivalent to deriving them.

Ages below ``--age-floor`` are dropped. The default of 1 Gyr matches the range
over which Gerasimov et al. (2024) Section 3.1 state the initial-mass sampling
was validated: adjacent tracks were kept within 0.12 dex in luminosity and
120 K in Teff "at all ages between 1.0 Gyr and 13.5 Gyr". Below 1 Gyr the
spacing of neighbouring tracks is not guaranteed, and the MESA runs also
required the bulk of their convergence accommodations during pre-main-sequence
relaxation.

Usage:
    python build_tables.py --zip /path/to/SANDee.zip
    python build_tables.py --zip /path/to/SANDee.zip --age-floor 1e-4 --outdir /tmp/check
"""

import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
from astropy.constants import M_jup, M_sun, R_jup, R_sun

_MODEL_DIR = Path(__file__).parent
_Msun_to_Mjup = (1.0 * M_sun / M_jup).decompose().value
_Rsun_to_Rjup = (1.0 * R_sun / R_jup).decompose().value

# MESA history columns consumed, in output order
_MESA_COLUMNS = ('star_mass', 'star_age', 'log_L', 'log_Teff', 'log_g', 'log_R')

_HEADER = '# mass_Mjup age_Gyr logL_Lsun Teff_K logg_cgs radius_Rjup'
_HISTORY_PATH = re.compile(r'^SANDee/MESA/(SAND_z[^/]+)/([^/]+)/LOGS/history\.data$')

DEFAULT_AGE_FLOOR_GYR = 1.0


def _read_history(handle, source):
    """Return the needed columns of one MESA ``history.data`` as a dict of arrays."""

    # MESA writes two header blocks; the run metadata occupies the first 5 lines
    table = np.loadtxt(handle, skiprows=5, dtype=str)
    if table.ndim != 2 or len(table) < 2:
        raise ValueError(f'No data rows in MESA history "{source}".')

    names = table[0]
    values = table[1:].astype(float)

    columns = {}
    for name in _MESA_COLUMNS:
        match = names == name
        if not match.any():
            raise ValueError(f'MESA history "{source}" has no {name!r} column.')
        columns[name] = values[:, match].ravel()
    return columns


def _convert(columns):
    """Convert one MESA track to the bundled table's columns and units."""

    return np.column_stack((
        columns['star_mass'] * _Msun_to_Mjup,
        columns['star_age'] / 1e9,
        columns['log_L'],
        10.0 ** columns['log_Teff'],
        columns['log_g'],
        10.0 ** columns['log_R'] * _Rsun_to_Rjup,
    ))


def build_tables(zip_path, outdir, age_floor_gyr=DEFAULT_AGE_FLOOR_GYR, verbose=True):
    """Write one ``<chemistry>_mass.txt`` per SANDee chemistry; return the paths written."""

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    age_floor_yr = age_floor_gyr * 1e9

    tracks = {}
    n_dropped = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(archive.namelist())
        for member in members:
            found = _HISTORY_PATH.match(member)
            if found is None:
                continue
            chemistry, mass_dir = found.groups()
            try:
                float(mass_dir)
            except ValueError:  # e.g. the per-chemistry TABLES directory
                continue

            with archive.open(member) as handle:
                columns = _read_history(handle, member)

            keep = columns['star_age'] >= age_floor_yr
            n_dropped += int((~keep).sum())
            if not keep.any():
                continue
            tracks.setdefault(chemistry, []).append(
                _convert({name: values[keep] for name, values in columns.items()})
            )

    if not tracks:
        raise ValueError(
            f'No SANDee MESA histories found in "{zip_path}". '
            f'Expected members like "SANDee/MESA/SAND_z0.1_a0.0/0.06/LOGS/history.data".'
        )

    written = []
    for chemistry in sorted(tracks):
        data = np.vstack(tracks[chemistry])
        data = data[np.lexsort((data[:, 1], data[:, 0]))]  # by mass, then age
        path = outdir / f'{chemistry}_mass.txt'
        with open(path, 'w') as table_file:
            table_file.write(_HEADER + '\n')
            for row in data:
                table_file.write(' '.join('%.8g' % value for value in row) + '\n')
        written.append(path)
        if verbose:
            print(f'{path.name}: {len(data)} rows, {len(tracks[chemistry])} tracks')

    if verbose:
        print(
            f'\nWrote {len(written)} tables to {outdir} '
            f'({n_dropped} rows dropped below {age_floor_gyr:g} Gyr).'
        )
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument(
        '--zip', required=True, dest='zip_path',
        help='Path to SANDee.zip from https://doi.org/10.5281/zenodo.11582126',
    )
    parser.add_argument(
        '--outdir', default=str(_MODEL_DIR),
        help='Directory to write the tables into (default: this model directory)',
    )
    parser.add_argument(
        '--age-floor', type=float, default=DEFAULT_AGE_FLOOR_GYR, dest='age_floor_gyr',
        help=f'Drop ages below this value in Gyr (default: {DEFAULT_AGE_FLOOR_GYR:g})',
    )
    args = parser.parse_args()
    build_tables(args.zip_path, args.outdir, age_floor_gyr=args.age_floor_gyr)


if __name__ == '__main__':
    main()
