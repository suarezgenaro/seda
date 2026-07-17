"""Load empirical color reference sequences for color_anomaly."""

from __future__ import annotations

import json
import re
import shlex
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).parent

_FAHERTY_COLOR_COLS = {
    'J-H': 'J-H_avg',
    'J-K': 'J-K_avg',
    'J-W1': 'J-W1_avg',
    'J-W2': 'J-W2_avg',
    'H-K': 'H-K_avg',
    'H-W1': 'H-W1_avg',
    'H-W2': 'H-W2_avg',
    'K-W1': 'K-W1_avg',
    'K-W2': 'K-W2_avg',
    'W1-W2': 'W1-W2_avg',
}

_SPT_OFFSETS = {'M': 0, 'L': 10, 'T': 20, 'Y': 30}


@lru_cache(maxsize=1)
def _load_config() -> dict:
    with (_DATA_DIR / 'config.json').open(encoding='utf-8') as cfg_file:
        return json.load(cfg_file)


def available_colors() -> tuple[str, ...]:
    """Return supported color names."""
    return tuple(_load_config()['colors'].keys())


def normalize_color_name(color_name: str) -> str:
    """Map aliases (e.g. J_H, j-h) to strict names (J-H)."""
    key = color_name.strip().replace('_', '-')
    canonical = {c.upper(): c for c in available_colors()}
    upper = key.upper()
    if upper not in canonical:
        valid = ', '.join(available_colors())
        raise ValueError(
            f"color_name={color_name!r} is not recognized. "
            f"Valid options: {valid}."
        )
    return canonical[upper]


def parse_spt_float(spt) -> float:
    """
    Convert spectral type to a numeric subtype (M0=0 ... Y0=30+).

    Fractional subtypes are preserved (e.g. ``'L3.7'`` -> ``13.7``).
    """
    if isinstance(spt, (int, np.integer)):
        return float(spt)
    if isinstance(spt, (float, np.floating)):
        return float(spt)

    text = str(spt).strip()
    match = re.match(r'^([MLTY])\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        number = float(match.group(2))
        return _SPT_OFFSETS[letter] + number

    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"spt={spt!r} could not be parsed. "
            "Use a string like 'L5' or 'T4', or a numeric float subtype."
        ) from exc


def spt_label(spt_int: int) -> str:
    """Format integer subtype as a letter class string (e.g. 15 -> L5)."""
    if spt_int < 10:
        return f'M{spt_int}'
    if spt_int < 20:
        return f'L{spt_int - 10}'
    if spt_int < 30:
        return f'T{spt_int - 20}'
    return f'Y{spt_int - 30}'


def _ultracool_sheet_to_standard_int(spt_val: float) -> int:
    """
    Convert Ultracool Sheet ``spt_adop_flt`` to standard numeric subtype.

    For values >= 70, the standard subtype is ``round(spt) - 70``.
    Smaller values are treated as already-standard subtypes (0-39).
    """
    v = float(spt_val)
    if v >= 70:
        return int(round(v - 70))
    return int(round(v))


def _matches_age_group(youth_evidence: str, age_group: str | None) -> bool:
    if age_group is None:
        return True

    youth = (youth_evidence or '').strip()
    if '?' in youth:
        return False

    cfg = _load_config()
    if age_group == 'young':
        return any(tok in youth for tok in cfg['young_tokens'])
    if age_group == 'old':
        return youth in cfg['old_values']
    raise ValueError(
        f"age_group={age_group!r} is not recognized. "
        "Valid options: None, 'young', 'old'."
    )


def _parse_ultracool_rows() -> tuple[list[str], list[dict[str, str]]]:
    path = _DATA_DIR / 'ultracool_sheet.dat'
    with path.open(encoding='utf-8') as data_file:
        header_line = data_file.readline()
        if not header_line.startswith('#'):
            raise ValueError(f'Expected header line in "{path}".')
        columns = header_line.lstrip('#').split()

        rows = []
        for line in data_file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = shlex.split(line)
            if len(parts) < len(columns):
                continue
            rows.append(dict(zip(columns, parts)))
    return columns, rows


@lru_cache(maxsize=1)
def _parse_faherty_table1_rows() -> tuple[list[str], list[dict[str, str]]]:
    """Parse the bundled Faherty et al. (2016) Table 1 sample."""
    path = _DATA_DIR / 'faherty16_table1_sample.dat'
    with path.open(encoding='utf-8') as data_file:
        header_line = data_file.readline()
        if not header_line.startswith('#'):
            raise ValueError(f'Expected header line in "{path}".')
        columns = header_line.lstrip('#').split()

        rows = []
        for line in data_file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = shlex.split(line)
            if len(parts) != len(columns):
                continue
            rows.append(dict(zip(columns, parts)))
    return columns, rows


def _is_low_gravity(label: str, tokens: list[str]) -> bool:
    """
    True if a Table 1 optical/infrared gravity label indicates youth.

    Labels containing '?' (e.g. 'gamma?') mark an uncertain classification
    and are excluded, matching the Ultracool Sheet '?' convention.
    """
    label = (label or '').strip().strip('"')
    if not label or '?' in label:
        return False
    return any(tok in label for tok in tokens)


def _spt_bin_suarez(spt_val: float) -> int:
    """
    Assign an integer SpT bin using half-open intervals
    ``[spt_int - 0.5, spt_int + 0.5)``, matching the binning used in
    Suárez et al. (2023) (e.g. L4.5 falls in the L5 bin, not L4).

    This differs from plain ``round()``, which uses round-half-to-even
    and would put 14.5 in the L4 bin instead of L5.
    """
    return int(np.floor(spt_val + 0.5))


@lru_cache(maxsize=None)
def _faherty_young_reference_bins(color_name: str) -> dict[int, list[float]]:
    """
    Per-integer-bin colors from all Table 1 objects with valid photometry.

    Bin membership follows Suárez et al. (2023)-style half-open intervals
    (``_spt_bin_suarez``), so half-integer ``spt_flt_assumed`` values (e.g.
    14.5) are grouped with the nearer-higher integer bin (L5), not rounded
    down to L4.
    """

    cfg = _load_config()
    t1_cfg = cfg['faherty16_table1']
    photometry = cfg['faherty16_table1_photometry']
    _, rows = _parse_faherty_table1_rows()
    bins: dict[int, list[float]] = {}

    for row in rows:
        spt_val = _finite_mag(row.get(t1_cfg['spt_column'], ''))
        if spt_val is None:
            continue
        spt_int = _spt_bin_suarez(spt_val)

        color_val = _object_color(row, color_name, photometry, max_mag_err=None)
        if color_val is None:
            continue

        bins.setdefault(spt_int, []).append(color_val)

    return bins


@lru_cache(maxsize=None)
def _faherty_young_reference(
    color_name: str,
    reference_stat: str,
) -> dict[int, float]:
    """
    Build {rounded_spt_int: reference_color} from all Faherty et al. (2016)
    Table 1 objects in the bundled sample. Unlike Tables 15-16, this is
    recomputed from bundled photometry. Bins with fewer than min_bin_count
    objects are omitted here; see sparse-bin and interpolation fallbacks.
    """
    cfg = _load_config()
    bins = _faherty_young_reference_bins(color_name)
    reducer = np.mean if reference_stat == 'mean' else np.median
    min_count = cfg['faherty16_table1']['min_bin_count']
    out: dict[int, float] = {}
    for spt, values in bins.items():
        if len(values) >= min_count:
            out[spt] = float(reducer(values))
    return out


@lru_cache(maxsize=1)
def _load_faherty_table() -> dict[int, dict[str, float]]:
    """Return {spt_flt: {color: mean}} from Tables 15-16."""
    path = _DATA_DIR / 'faherty16_tables15_16.dat'
    header_cols = None
    table: dict[int, dict[str, float]] = {}

    with path.open(encoding='utf-8') as data_file:
        for line in data_file:
            line = line.strip()
            if not line or line.startswith('#'):
                if line.startswith('#'):
                    header_cols = line.lstrip('#').split()
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            spt_flt = int(float(parts[1]))
            row: dict[str, float] = {}
            for color, avg_col in _FAHERTY_COLOR_COLS.items():
                avg_idx = header_cols.index(avg_col)
                row[color] = float(parts[avg_idx])
            table[spt_flt] = row
    return table


def _finite_mag(value: str) -> float | None:
    value = (value or '').strip().strip('"')
    if not value or value in ('""', '-', '...'):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not np.isfinite(out):
        return None
    return out


def _object_color(
    row: dict[str, str],
    color_name: str,
    photometry: dict[str, dict[str, str]],
    max_mag_err: float | None = 0.1,
) -> float | None:
    cfg = _load_config()
    bands = cfg['colors'][color_name]

    mag1 = _finite_mag(row.get(photometry[bands['band1']]['mag'], ''))
    mag2 = _finite_mag(row.get(photometry[bands['band2']]['mag'], ''))
    err1 = _finite_mag(row.get(photometry[bands['band1']]['err'], ''))
    err2 = _finite_mag(row.get(photometry[bands['band2']]['err'], ''))

    if mag1 is None or mag2 is None:
        return None

    if max_mag_err is not None:
        if err1 is None or err2 is None:
            return None
        if err1 > max_mag_err or err2 > max_mag_err:
            return None

    return mag1 - mag2


@lru_cache(maxsize=None)
def _ultracool_reference(
    color_name: str,
    age_group: str | None,
    reference_stat: str,
) -> dict[int, float]:
    """Build {rounded_spt_int: reference_color} for one filter configuration."""
    cfg = _load_config()
    _, rows = _parse_ultracool_rows()
    bins: dict[int, list[float]] = {}

    for row in rows:
        youth = row.get(cfg['ultracool']['youth_column'], '')
        if not _matches_age_group(youth, age_group):
            continue

        spt_val = _finite_mag(row.get(cfg['ultracool']['spt_column'], ''))
        if spt_val is None:
            continue
        spt_int = _ultracool_sheet_to_standard_int(spt_val)

        color_val = _object_color(
            row, color_name, cfg['photometry'], cfg['ultracool']['max_mag_err'],
        )
        if color_val is None:
            continue

        bins.setdefault(spt_int, []).append(color_val)

    if reference_stat == 'mean':
        reducer = np.mean
    elif reference_stat == 'median':
        reducer = np.median
    else:
        raise ValueError(
            f"reference_stat={reference_stat!r} is not recognized. "
            "Valid options: 'mean', 'median'."
        )

    min_count = cfg['ultracool']['min_bin_count']
    out: dict[int, float] = {}
    for spt, values in bins.items():
        if len(values) >= min_count:
            out[spt] = float(reducer(values))
    return out


def _faherty_young_reference_interpolated(
    grid: dict[int, float],
    spt_int: int,
    color: str,
    reference_stat: str,
) -> float | None:
    """
    Linearly interpolate a young Table 1 reference when an integer bin is
    missing but bracketing bins are available (e.g. L6 from L5 and L7).
    """
    available = sorted(grid)
    below = [k for k in available if k < spt_int]
    above = [k for k in available if k > spt_int]
    if not below or not above:
        return None

    spt_lo = below[-1]
    spt_hi = above[0]
    frac = (spt_int - spt_lo) / (spt_hi - spt_lo)
    ref_lo = grid[spt_lo]
    ref_hi = grid[spt_hi]
    ref = (1.0 - frac) * ref_lo + frac * ref_hi

    warnings.warn(
        f"Faherty et al. (2016) Table 1 'young' reference for {color} at "
        f"{spt_label(spt_int)} has insufficient objects in that bin; "
        f"interpolating between {spt_label(spt_lo)} "
        f"({reference_stat}={ref_lo:.4f} mag) and {spt_label(spt_hi)} "
        f"({reference_stat}={ref_hi:.4f} mag) with weight {frac:.2f} toward "
        f"{spt_label(spt_hi)} (interpolated {reference_stat}={ref:.4f} mag).",
        UserWarning,
        stacklevel=4,
    )
    return ref


def _faherty_young_reference_sparse(
    bins: dict[int, list[float]],
    spt_int: int,
    color: str,
    reference_stat: str,
    min_count: int,
) -> float | None:
    """
    Use a sparse integer bin (1 <= n < min_bin_count objects) when no
    bracketing interpolation is available (e.g. L8 with one object).
    """
    values = bins.get(spt_int)
    if not values or len(values) >= min_count:
        return None

    reducer = np.mean if reference_stat == 'mean' else np.median
    ref = float(reducer(values))
    n_obj = len(values)
    obj_word = 'object' if n_obj == 1 else 'objects'
    warnings.warn(
        f"Faherty et al. (2016) Table 1 'young' reference for {color} at "
        f"{spt_label(spt_int)} has only {n_obj} {obj_word} with valid "
        f"photometry (fewer than the usual minimum of {min_count}); using "
        f"the bin {reference_stat}={ref:.4f} mag from the available "
        f"{obj_word}.",
        UserWarning,
        stacklevel=4,
    )
    return ref


def _reference_color_at_int(
    color: str,
    spt_int: int,
    table: str,
    age_group: str | None,
    reference_stat: str,
) -> float:
    """Return the reference color at a single integer SpT bin."""
    cfg = _load_config()

    if table == 'faherty16':
        spt_min = cfg['faherty16']['spt_min']
        spt_max = cfg['faherty16']['spt_max']
        if spt_int < spt_min or spt_int > spt_max:
            raise ValueError(
                f"table='faherty16' covers {spt_label(spt_min)}-{spt_label(spt_max)} "
                f"(numeric {spt_min}-{spt_max}) only. "
                f"Requested {spt_label(spt_int)} (numeric {spt_int}). "
                "Use table='ultracool' for other types."
            )

        if age_group in (None, 'old'):
            faherty = _load_faherty_table()
            if spt_int not in faherty or color not in faherty[spt_int]:
                raise ValueError(
                    f"No Faherty et al. (2016) reference for {color} at "
                    f"{spt_label(spt_int)}."
                )
            return faherty[spt_int][color]

        # age_group == 'young': recomputed from all Table 1 objects.
        min_count = cfg['faherty16_table1']['min_bin_count']
        bins = _faherty_young_reference_bins(color)
        grid = _faherty_young_reference(color, reference_stat)
        if spt_int in grid:
            return grid[spt_int]

        interpolated = _faherty_young_reference_interpolated(
            grid, spt_int, color, reference_stat,
        )
        if interpolated is not None:
            return interpolated

        sparse = _faherty_young_reference_sparse(
            bins, spt_int, color, reference_stat, min_count,
        )
        if sparse is not None:
            return sparse

        raise ValueError(
            f"Insufficient Faherty et al. (2016) Table 1 "
            f"objects to define a 'young' reference for {color} at "
            f"{spt_label(spt_int)} (numeric {spt_int}). At least "
            f"{min_count} objects with valid photometry are required "
            "per integer subtype bin, and no bracketing bins were "
            "available for interpolation."
        )

    if table == 'ultracool':
        grid = _ultracool_reference(color, age_group, reference_stat)
        if spt_int not in grid:
            age_msg = f", age_group={age_group!r}" if age_group else ''
            min_count = cfg['ultracool']['min_bin_count']
            raise ValueError(
                f"Insufficient Ultracool Sheet data to define a reference for "
                f"{color} at {spt_label(spt_int)} (numeric {spt_int}){age_msg}. "
                f"At least {min_count} objects with valid photometry are required "
                "per integer subtype bin."
            )
        return grid[spt_int]

    raise ValueError(
        f"table={table!r} is not recognized. "
        "Valid options: 'faherty16', 'ultracool'."
    )


def reference_color(
    color_name: str,
    spt,
    table: str = 'faherty16',
    age_group: str | None = None,
    reference_stat: str = 'mean',
) -> float:
    """
    Return the reference color (mag) for a spectral type and table.

    Integer SpT values use that bin directly. Fractional SpT values linearly
    interpolate between the above and below integer references.

    """
    color = normalize_color_name(color_name)
    spt_float = parse_spt_float(spt)

    if reference_stat not in ('mean', 'median'):
        raise ValueError(
            f"reference_stat={reference_stat!r} is not recognized. "
            "Valid options: 'mean', 'median'."
        )

    if table == 'faherty16':
        if age_group not in (None, 'old', 'young'):
            raise ValueError(
                f"age_group={age_group!r} is not supported for "
                "table='faherty16'. Valid options: None, 'old', 'young'."
            )
        if age_group in (None, 'old') and reference_stat != 'mean':
            raise ValueError(
                "table='faherty16' with age_group=None or 'old' only "
                "supports reference_stat='mean' (published Table 15-16 "
                "averages). age_group='young' also supports 'median' "
                "(recomputed from Table 1 photometry)."
            )

    # Exact integer SpT: look up that bin only.
    if abs(spt_float - round(spt_float)) < 1e-12:
        return _reference_color_at_int(
            color, int(round(spt_float)), table, age_group, reference_stat,
        )

    # Fractional SpT: linear interpolation between neighboring integer bins.
    spt_lo = int(np.floor(spt_float))
    spt_hi = int(np.ceil(spt_float))
    frac = spt_float - spt_lo
    ref_lo = _reference_color_at_int(
        color, spt_lo, table, age_group, reference_stat,
    )
    ref_hi = _reference_color_at_int(
        color, spt_hi, table, age_group, reference_stat,
    )
    return (1.0 - frac) * ref_lo + frac * ref_hi
