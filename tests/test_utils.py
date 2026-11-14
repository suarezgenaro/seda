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

	assert result['convolution_method'] == 'constant_fwhm_fft'

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


@pytest.mark.parametrize('sampling, expected_method', [
	('log', 'log_uniform_fft'),
	('linear', 'wavelength_dependent')
])
def test_convolve_spectrum_constant_R(sampling, expected_method):
	"""
	Verify both constant-R convolution methods.
	"""
	if sampling == 'log':
		wl = np.geomspace(1.0, 2.0, 1000)
	else:
		wl = np.linspace(1.0, 2.0, 1000)

	flux = np.ones_like(wl)
	eflux = np.full_like(wl, 0.1)

	result = seda.utils.convolve_spectrum(
		wl=wl,
		flux=flux,
		eflux=eflux,
		res=100,
		resolution_mode='constant_R'
	)

	assert result['convolution_method'] == expected_method
	assert np.allclose(result['wl_conv'], wl)
	assert np.all(np.isfinite(result['flux_conv']))
	assert np.all(np.isfinite(result['eflux_conv']))

	# constant input flux should remain constant away from FFT boundaries
	assert np.allclose(result['flux_conv'][50:-50], 1.0, rtol=1e-3)
	assert np.all(result['eflux_conv'] >= 0)


def test_convolve_spectrum_invalid_resolution_mode():
	"""
	Verify that invalid resolution modes raise an error.
	"""
	wl = np.linspace(1.0, 2.0, 100)
	flux = np.ones_like(wl)

	with pytest.raises(ValueError):
		seda.utils.convolve_spectrum(
			wl=wl,
			flux=flux,
			res=100,
			resolution_mode='invalid'
	)


def test_convolve_spectrum_save(tmp_path):
	"""
	Verify that the convolved spectrum is correctly saved to NetCDF.
	"""
	import xarray as xr

	wl = np.linspace(1.0, 2.0, 1000)
	flux = np.sin(wl)
	eflux = np.full_like(wl, 0.1)
	out_file = tmp_path / 'convolved_spectrum.nc'

	result = seda.utils.convolve_spectrum(
		wl=wl,
		flux=flux,
		eflux=eflux,
		res=100,
		out_file=out_file
	)

	with xr.open_dataset(out_file) as ds:
		assert 'wl' in ds.coords
		assert 'flux' in ds.data_vars
		assert 'eflux' in ds.data_vars
		assert np.allclose(ds['wl'].values, result['wl_conv'])
		assert np.allclose(ds['flux'].values, result['flux_conv'])
		assert np.allclose(ds['eflux'].values, result['eflux_conv'])

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
