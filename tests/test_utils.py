import seda
import pytest
import numpy as np

from tests.conftest import load_model_spectra_catalog
from unittest.mock import patch
from astropy.table import Table, MaskedColumn

def test_convolve_spectrum_identity():
	"""
	Verify that the convolution of a spectrum to a higher 
	resolution than the original one returns the same spectrum.
	"""
	wl = np.linspace(1.0, 2.0, 2000)  # microns
	flux = np.exp(-((wl - 1.5) / 0.05) ** 2)

	result = seda.utils.convolve_spectrum(
		wl=wl,
		flux=flux,
		res=1e6,  # extremely high resolution -> no change
	)

	assert np.allclose(result['wl_conv'], wl)
	assert np.allclose(result['flux_conv'], flux, rtol=1e-4), (
		"High-resolution convolution should preserve flux"
	)

def test_convolve_spectrum_flux_conservation():
	"""
	Verify that the spectrum convolution 
	preserve integrated flux (within tolerance).
	"""
	wl = np.linspace(1.0, 2.0, 3000)
	flux = np.ones_like(wl)

	result = seda.utils.convolve_spectrum(
		wl=wl,
		flux=flux,
		res=500,
	)

	try:
		trapz = np.trapezoid   # NumPy >= 1.20 (and 2.x)
	except AttributeError:
		trapz = np.trapz	   # NumPy < 1.20
	
	input_integral = trapz(flux, wl)
	output_integral = trapz(result['flux_conv'], result['wl_conv'])

	assert np.isclose(output_integral, input_integral, rtol=1e-3), (
		"Integrated flux not conserved after convolution"
	)

def test_convolve_spectrum_uncertainty():
	"""
	Verify flux uncertainties exist, remain 
	finite, and not increase unphysically
	"""
	wl = np.linspace(1.0, 2.0, 1000)
	flux = np.sin(wl)
	eflux = np.full_like(flux, 0.1)

	result = seda.utils.convolve_spectrum(
		wl=wl,
		flux=flux,
		eflux=eflux,
		res=1000,
	)
	
	assert 'eflux_conv' in result, (
		'No flux uncertainties for convolved spectrum'
	)
	assert np.all(np.isfinite(result['eflux_conv'])), (
		'Convolution should produce finite flux uncertainties'
	)
	assert np.all(result['eflux_conv'] <= np.max(eflux) * 1.01), (
		'Convolution should not increase uncertainties unphysically'
	)

@pytest.mark.parametrize('model, spectrum_file', load_model_spectra_catalog())
def test_convolve_spectrum_real_data(model, spectrum_file):
	"""
	Test that convolution runs end-to-end on real spectra without crashing.
	"""

	# read each example model spectrum
	spectrum = seda.models.read_model_spectrum(spectrum_file, model)
	wl = spectrum['wl_model']
	flux = spectrum['flux_model']

	result = seda.utils.convolve_spectrum(
		wl=wl,
		flux=flux,
		res=100,
	)

	# basic checks
	assert 'wl_conv' in result
	assert 'flux_conv' in result
	
	# shapes match
	assert len(result['wl_conv']) == len(result['flux_conv'])
	
	# flux is finite
	assert np.all(np.isfinite(result['flux_conv']))

def test_read_SVO_params():
	"""Verify that reading parameters from the SVO table return the expected values"""

	filters = ['2MASS/2MASS.J', 'Spitzer/IRAC.I1', 'WISE/WISE.W3', 'JWST/MIRI.F1000W']
	wavelength_eff = np.array([12350.0, 35074.83347149, 115608.0, 98793.44853281])
	width_eff = np.array([1624.3190191, 6836.18018646, 55055.230027, 17036.67237661])
	zero_point = np.array([1594.0, 274.53023427, 31.674, 38.50820313])

	# make masked fake SVO table to mimic the original one
	fake_svo_table = Table([
		MaskedColumn(filters, name='filterID'),
		MaskedColumn(wavelength_eff, name='WavelengthEff'),
		MaskedColumn(width_eff, name='WidthEff'),
		MaskedColumn(zero_point, name='ZeroPoint')
	])

	filters_req = ['2MASS/2MASS.J', 'JWST/MIRI.F1000W']  # subset of table
	params_req = ['WavelengthEff', 'ZeroPoint']		  # subset of columns

	# select the desired parameters from the fake table
	with patch('seda.utils.read_SVO_table', return_value=fake_svo_table):
		result = seda.utils.read_SVO_params(
			filters=filters_req,
			params=params_req,
		)

	# check that all requested columns are returned
	assert set(result) == set(params_req), (
		f'Returned columns {list(result.keys())} do not match requested {params_req}'
	)

	# find indices of requested filters in the fake table
	indices = []
	for filt in filters_req:
		# find where the filter matches in the table
		match_index = np.where(fake_svo_table['filterID'] == filt)[0][0]
		indices.append(match_index)

	# check that values match
	assert np.allclose(result['WavelengthEff'], wavelength_eff[indices]), (
		f"WavelengthEff mismatch: got {result['WavelengthEff']}, expected {wavelength_eff[indices]}"
	)
	assert np.allclose(result['ZeroPoint'], zero_point[indices]), (
			f"ZeroPoint mismatch: got {result['ZeroPoint']}, expected {zero_point[indices]}"
	)
