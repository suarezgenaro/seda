"""Load empirical color reference sequences for color_anomaly."""

from __future__ import annotations

import json
import re
import shlex
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).parent

_FAHERTY_COLOR_COLS = {
    'J-H': ('J-H_avg', 'sigma_J-H'),
    'J-K': ('J-K_avg', 'sigma_J-K'),
    'J-W1': ('J-W1_avg', 'sigma_J-W1'),
    'J-W2': ('J-W2_avg', 'sigma_J-W2'),
    'H-K': ('H-K_avg', 'sigma_H-K'),
    'H-W1': ('H-W1_avg', 'sigma_H-W1'),
    'H-W2': ('H-W2_avg', 'sigma_H-W2'),
    'K-W1': ('K-W1_avg', 'sigma_K-W1'),
    'K-W2': ('K-W2_avg', 'sigma_K-W2'),
    'W1-W2': ('W1-W2_avg', 'sigma_W1-W2'),
}

_SPT_OFFSETS = {'M': 0, 'L': 10, 'T': 20, 'Y': 30}


@lru_cache(maxsize=1)
def _load_config() -> dict:
    with (_DATA_DIR / 'config.json').open(encoding='utf-8') as cfg_file:
        return json.load(cfg_file)


def available_colors() -> tuple[str, ...]:
    """Return supported color names (canonical hyphenated form)."""
    return tuple(_load_config()['colors'].keys())


def normalize_color_name(color_name: str) -> str:
    """Map aliases (e.g. J_H, j-h) to canonical names (J-H)."""
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


def parse_spt(spt) -> int:
    """
    Parse spectral type to integer numeric subtype (M0=0 ... Y0=30+).

    Fractional types are rounded to the nearest integer subtype.
    """
    if isinstance(spt, (int, np.integer)):
        return int(round(int(spt)))
    if isinstance(spt, (float, np.floating)):
        return int(round(float(spt)))

    text = str(spt).strip()
    match = re.match(r'^([MLTY])\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        number = float(match.group(2))
        return int(round(_SPT_OFFSETS[letter] + number))

    try:
        return int(round(float(text)))
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

    The sheet uses Kirkpatrick-style values (e.g. M7=77, L5=85, T5=95, Y0=100).
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
def _load_faherty_table() -> dict[int, dict[str, tuple[float, float]]]:
    """Return {spt_flt: {color: (mean, sigma)}}."""
    path = _DATA_DIR / 'faherty16_tables15_16.dat'
    header_cols = None
    table: dict[int, dict[str, tuple[float, float]]] = {}

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
            row: dict[str, tuple[float, float]] = {}
            for color, (avg_col, sig_col) in _FAHERTY_COLOR_COLS.items():
                avg_idx = header_cols.index(avg_col)
                sig_idx = header_cols.index(sig_col)
                row[color] = (float(parts[avg_idx]), float(parts[sig_idx]))
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


def _object_color(row: dict[str, str], color_name: str) -> float | None:
    cfg = _load_config()
    bands = cfg['colors'][color_name]
    photo = cfg['photometry']

    mag1 = _finite_mag(row.get(photo[bands['band1']]['mag'], ''))
    mag2 = _finite_mag(row.get(photo[bands['band2']]['mag'], ''))
    err1 = _finite_mag(row.get(photo[bands['band1']]['err'], ''))
    err2 = _finite_mag(row.get(photo[bands['band2']]['err'], ''))

    if mag1 is None or mag2 is None or err1 is None or err2 is None:
        return None

    max_err = cfg['ultracool']['max_mag_err']
    if err1 > max_err or err2 > max_err:
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

        color_val = _object_color(row, color_name)
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


def reference_color(
    color_name: str,
    spt,
    table: str = 'faherty16',
    age_group: str | None = None,
    reference_stat: str = 'mean',
) -> float:
    """
    Return the reference color (mag) for a spectral type and table backend.

    See color_anomaly docstring for assumptions and citations.
    """
    color = normalize_color_name(color_name)
    spt_int = parse_spt(spt)
    cfg = _load_config()

    if reference_stat not in ('mean', 'median'):
        raise ValueError(
            f"reference_stat={reference_stat!r} is not recognized. "
            "Valid options: 'mean', 'median'."
        )

    if table == 'faherty16':
        if age_group is not None:
            raise ValueError(
                "age_group is not supported for table='faherty16'. "
                "Faherty et al. (2016) Tables 15-16 are field/normal sequences."
            )
        if reference_stat != 'mean':
            raise ValueError(
                "table='faherty16' only supports reference_stat='mean' "
                "(published Table 15-16 averages)."
            )

        faherty = _load_faherty_table()
        spt_min = cfg['faherty16']['spt_min']
        spt_max = cfg['faherty16']['spt_max']
        if spt_int < spt_min or spt_int > spt_max:
            raise ValueError(
                f"table='faherty16' covers {spt_label(spt_min)}-{spt_label(spt_max)} "
                f"(numeric {spt_min}-{spt_max}) only. "
                f"Requested {spt_label(spt_int)} (numeric {spt_int}). "
                "Use table='ultracool' for other types."
            )
        if spt_int not in faherty or color not in faherty[spt_int]:
            raise ValueError(
                f"No Faherty et al. (2016) reference for {color} at "
                f"{spt_label(spt_int)}."
            )
        return faherty[spt_int][color][0]

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


def reference_scatter(
    color_name: str,
    spt,
    table: str = 'faherty16',
) -> float | None:
    """Return published reference scatter (sigma) when available (faherty16 only)."""
    color = normalize_color_name(color_name)
    if table != 'faherty16':
        return None
    spt_int = parse_spt(spt)
    faherty = _load_faherty_table()
    if spt_int not in faherty or color not in faherty[spt_int]:
        return None
    return faherty[spt_int][color][1]
