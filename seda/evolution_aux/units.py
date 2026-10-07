"""Conversions between evolutionary-table file units and phy_params user units.

``isochrone_params`` and ``evol_params`` always use:

- age in Gyr
- mass in M_jup
- radius in R_jup

The bundled tables are not rewritten. Interpolation stays in the
coordinates stored in each model's ``config.json`` (for example
``log10(yr)`` age on BHAC2015). Convert inputs before interpolation and
outputs after it. Logarithmic age is converted sample by sample so the
Monte Carlo summary is in Gyr. ``Teff`` stays in Kelvin.
"""

import numpy as np
from astropy import units as u
from astropy.constants import M_jup, M_sun, R_jup, R_sun

# Units presented by phy_params for these columns, for every model.
USER_UNITS = {
	'mass': 'M_jup',
	'age': 'Gyr',
	'radius': 'R_jup',
}


def _as_float_array(values):
	return np.asarray(values, dtype=float)


def mass_to_mjup(mass, native_unit):
	"""Convert a mass from ``native_unit`` to M_jup."""

	mass = _as_float_array(mass)
	if native_unit == 'M_jup':
		return mass
	if native_unit == 'M_sun':
		return (mass * M_sun).to(M_jup).value
	raise ValueError(
		f'Unsupported evolutionary mass unit {native_unit!r}. '
		f'Expected "M_sun" or "M_jup".'
	)


def radius_to_rjup(radius, native_unit):
	"""Convert a radius from ``native_unit`` to R_jup."""

	radius = _as_float_array(radius)
	if native_unit == 'R_jup':
		return radius
	if native_unit == 'R_sun':
		return (radius * R_sun).to(R_jup).value
	raise ValueError(
		f'Unsupported evolutionary radius unit {native_unit!r}. '
		f'Expected "R_sun" or "R_jup".'
	)


def age_to_gyr(age, native_unit):
	"""Convert a file age coordinate to Gyr."""

	age = _as_float_array(age)
	if native_unit == 'Gyr':
		return age
	if native_unit == 'log10(yr)':
		with np.errstate(invalid='ignore', over='ignore'):
			years = np.power(10.0, age)
		return (years * u.yr).to(u.Gyr).value
	raise ValueError(
		f'Unsupported evolutionary age unit {native_unit!r}. '
		f'Expected "Gyr" or "log10(yr)".'
	)


def age_from_gyr(age_gyr, native_unit):
	"""Convert an age in Gyr to a file age coordinate.

	Non-positive ages are undefined in ``log10(yr)`` and become NaN.
	"""

	age_gyr = _as_float_array(age_gyr)
	if native_unit == 'Gyr':
		return age_gyr
	if native_unit == 'log10(yr)':
		years = (age_gyr * u.Gyr).to(u.yr).value
		with np.errstate(invalid='ignore', divide='ignore'):
			log_age = np.log10(years)
		return np.where(age_gyr > 0, log_age, np.nan)
	raise ValueError(
		f'Unsupported evolutionary age unit {native_unit!r}. '
		f'Expected "Gyr" or "log10(yr)".'
	)


def to_user_units(param, values, file_units):
	"""Convert one interpolated column to phy_params user units.

	``mass``, ``age``, and ``radius`` are converted using ``file_units``.
	Every other column is returned unchanged.
	"""

	values = _as_float_array(values)
	if param == 'mass':
		return mass_to_mjup(values, file_units['mass'])
	if param == 'age':
		return age_to_gyr(values, file_units['age'])
	if param == 'radius':
		return radius_to_rjup(values, file_units['radius'])
	return values
