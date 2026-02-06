import pytest
import seda
import numpy as np

# ----------------------------
# Test data
# ----------------------------

# test data for spectral type from str <-> float
SPT_CASES = [
	('L0', 10),
	('L5.5', 15.5),
	('T0', 20),
	('Y0', 30),
]

# test cases for spt <-> teff
SPT_TO_TEFF_CASES = [
	('L0', 'str', 'F15', 2249),
	('T5', 'str', 'F15', 1033),
	(15, 'float', 'K21', 1613),
	(20, 'float', 'K21', 1255),
]

# test data for parallax -> distance
PARALLAX_CASES = [
	(100, 0.01, 10.0, 0.001),
	(50, 0.5, 20.0, 0.2),
	(10, 0.1, 100.0, 1.0),
]

# test data for apparent flux to absolute flux conversion and vice versa
APP_TO_ABS_FLUX_CASES = [
	# flux, eflux, distance(pc), edistance(pc)
	(1.0, 0.1, 10.0, 0.5),
	(2.5, 0.2, 5.0, 0.1),
	(0.1, 0.01, 20.0, 1.0),
]

APP_TO_mag_abs_CASES = [
	# m_app, em_app, distance(pc), edistance(pc)
	(10.0, 0.05, 10.0, 0.5),
	(15.0, 0.10, 20.0, 1.0),
	(18.5, 0.02, 5.0, 0.2),
]

# ----------------------------
# Tests
# ----------------------------

@pytest.mark.parametrize('spt_str, spt_float', SPT_CASES)
def test_spt_float_to_str(spt_str, spt_float):
	"""Verify conversion of spectral type from float into string"""

	spt = seda.utils.spt_float_to_str(spt_float)
	assert spt == spt_str, f'{spt_float} should convert to {spt_str} instead of {spt}'

@pytest.mark.parametrize('spt_str, spt_float', SPT_CASES)
def test_spt_str_to_float(spt_str, spt_float):
	"""Verify conversion of spectral type from str into float"""

	spt = seda.utils.spt_str_to_float(spt_str)
	assert spt == spt_float, f'{spt_str} should convert to {spt_float} instead of {spt}'

@pytest.mark.parametrize("_, spt_float", SPT_CASES)
def test_spt_round_trip(_, spt_float):
	"""Verify round-trip conversion of spectral type"""

	spt_str = seda.utils.spt_float_to_str(spt_float)
	spt_float_rec = seda.utils.spt_str_to_float(spt_str) 
	assert spt_float_rec == spt_float, (
		f'Round-trip failed: {spt_float} -> {spt_str} -> {spt_float_rec}'
	)

@pytest.mark.parametrize('parallax, eparallax, expected_dist, expected_edist', PARALLAX_CASES)
def test_parallax_to_distance(parallax, eparallax, expected_dist, expected_edist):
	"""Test parallax conversion into distance"""
	distance, edistance = seda.utils.parallax_to_distance(parallax, eparallax)
	
	# use pytest.approx to protects against tiny floating-point differences
	assert distance == pytest.approx(expected_dist), (
		f'Expected distance {expected_dist} but got {distance}'
	)
	assert edistance == pytest.approx(expected_edist), (
		f'Expected distance error {expected_edist} but got {edistance}'
	)

@pytest.mark.parametrize('spt_input, spt_type, ref, expected_teff', SPT_TO_TEFF_CASES)
def test_spt_to_teff(spt_input, spt_type, ref, expected_teff):
	"""Test spt to teff conversion"""

	teff = seda.utils.spt_to_teff(spt_input, spt_type, ref=ref)
	
	# use a teff tolerance of 1 K because expected values are rounded to units
	assert teff == pytest.approx(expected_teff, abs=1), (
		f'spt_to_teff({spt_input!r}, {spt_type!r}, ref={ref!r}) returned {teff}, '
		f'expected ~{expected_teff}'
	)

@pytest.mark.parametrize('spt_input, spt_type, ref, _', SPT_TO_TEFF_CASES)
def test_teff_to_spt(spt_input, spt_type, ref, _):
	"""Test spt to teff round-trip conversion"""

	teff = seda.utils.spt_to_teff(spt_input, spt_type, ref=ref)[0]
	spt_back = seda.utils.teff_to_spt(teff, ref=ref)

	# verify the return spectral type ...
	if spt_type == 'float': # input spectrum as a float
		# compare to canonical string derived from float input
		expected_str = seda.utils.spt_float_to_str(float(spt_input))
		assert spt_back == expected_str, (
			f'Round-trip failed: {spt_input} ({expected_str}) -> {teff} -> {spt_back}'
		)
	else: # input spectrum as a string
		print(spt_input, teff, spt_back)
		assert spt_back == spt_input, (
			f'Round-trip failed: {spt_input} -> {teff} -> {spt_back}'
		)

@pytest.mark.parametrize('flux, eflux, distance, edistance', APP_TO_ABS_FLUX_CASES)
def test_app_to_abs_flux(flux, eflux, distance, edistance):
	"""
	Test apparent flux to absolute flux conversion using the definition:
		F_abs = F_app * (d / 10)^2
	and standard uncertainty propagation.
	"""

	result = seda.utils.app_to_abs_flux(
		flux,
		distance,
		eflux=eflux,
		edistance=edistance,
	)

	abs_flux = result['flux_abs']
	eabs_flux = result['eflux_abs']

	# expected values from the definition
	expected_abs = flux * (distance / 10.0) ** 2

	expected_eabs = np.sqrt(
		((distance / 10.0) ** 2 * eflux) ** 2
		+ ((2.0 * flux * distance / 100.0) * edistance) ** 2
	)

	assert abs_flux == pytest.approx(expected_abs), (
		f"Expected absolute flux {expected_abs}, got {abs_flux}"
	)

	assert eabs_flux == pytest.approx(expected_eabs), (
		f"Expected absolute flux uncertainty {expected_eabs}, got {eabs_flux}"
	)

@pytest.mark.parametrize('flux, eflux, distance, edistance', APP_TO_ABS_FLUX_CASES)
def test_app_to_abs_flux_round_trip(flux, eflux, distance, edistance):
	"""Test round-trip of apparent flux to absolute flux conversion"""

	forward = seda.utils.app_to_abs_flux(
		flux,
		distance,
		eflux=eflux,
		edistance=edistance,
	)

	backward = seda.utils.app_to_abs_flux(
		forward['flux_abs'],
		distance,
		eflux=forward['eflux_abs'],
		edistance=edistance,
		reverse=True,
	)

	# flux must be recovered
	assert backward['flux_app'] == pytest.approx(flux), (
		f"Round-trip failed: {flux} -> {forward['flux_abs']} -> {backward['flux_app']}"
	)

	# uncertainty checks
	assert backward['eflux_app'] > 0, 'Returned uncertainty must be positive'
	assert backward['eflux_app'] >= eflux, (
		'Round-trip uncertainty should not decrease'
	)

@pytest.mark.parametrize('mag, emag, distance, edistance', APP_TO_mag_abs_CASES)
def test_app_to_abs_mag(mag, emag, distance, edistance):
	"""
	Test apparent -> absolute magnitude conversion using the equation:
		M = m - 5 * log10(d / 10)
	with standard uncertainty propagation.
	"""

	result = seda.utils.app_to_abs_mag(
		mag,
		distance,
		emagnitude=emag,
		edistance=edistance,
	)

	mag_abs = result['mag_abs']
	emag_abs = result['emag_abs']

	# expected values from the definition
	mag_abs_exp = mag - 5.0 * np.log10(distance / 10.0)
	emag_abs_exp = np.sqrt(emag**2 + (5.0/np.log(10.0)*edistance/distance)**2)

	assert mag_abs == pytest.approx(mag_abs_exp), (
		f"Expected absolute magnitude {mag_abs_exp}, got {mag_abs}"
	)

	assert emag_abs == pytest.approx(emag_abs_exp), (
		f"Expected absolute magnitude uncertainty {emag_abs_exp}, got {emag_abs}"
	)
