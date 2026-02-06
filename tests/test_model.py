import seda
import glob
import os
import numpy as np
import pytest
from astropy.io import ascii

# ----------------------------
# Constants / Test data
# ----------------------------
# List of (model_name, filename, expected_params)
TEST_CASES = [
	('Sonora_Diamondback', 't1000g316f4_m0.0_co1.0.spec',
	 {'Teff': 1000., 'logg': 4.5, 'Z': 0.0, 'fsed': 4.}),
	('Sonora_Elf_Owl', 'spectra_logzz_4.0_teff_1200.0_grav_1000.0_mh_0.0_co_0.5.nc',
	 {'Teff': 1200., 'logg': 5., 'logKzz': 4., 'Z': 0., 'CtoO': 0.5}),
	('LB23', 'T700_g5.00_Z1.000_CDIFF1e6_HMIX1.000.21',
	 {'Teff': 700., 'logg': 5., 'Z': 0., 'logKzz': 6., 'Hmix': 1.}),	
	('Sonora_Cholla', '1000K_1000g_logkzz2.spec',
	 {'Teff': 1000., 'logg': 5., 'logKzz': 2.}),
	('Sonora_Bobcat', 'sp_t1000g1000nc_m0.0',
	 {'Teff': 1000., 'logg': 5., 'Z': 0., 'CtoO': 1.}),
	('ATMO2020', 'spec_T1000_lg4.0_NEQ_weak.txt',
	 {'Teff': 1000., 'logg': 4., 'logKzz': 4.}),
	('BT-Settl', 'lte010-4.0-0.0a+0.0.BT-Settl.spec.7',
	 {'Teff': 1000., 'logg': 4.}),
	('SM08', 'sp_t1000g1000f1',
	 {'Teff': 1000., 'logg': 5., 'fsed': 1.}),
]

# ----------------------------
# Test functions
# ----------------------------
@pytest.mark.parametrize('model, filename, expected', TEST_CASES)
def test_separate_params(model, filename, expected):
	# separate parameters
	result = seda.models.separate_params(model, filename)
	params = result['params']

	# check filename captured
	assert result['spectra_name'] == [filename], (
		f"Expected spectrum name [{filename}] " 
		f"but got {result['spectra_name']}"
	)

	# ensure each parameter array has exactly one element
	for key in expected:
		assert params[key].size == 1, f'{key} should have exactly one value'

	# check numerical values
	for key, value in expected.items():
		assert params[key][0] == value, f'{key} expected {value}, got {params[key][0]}'

def test_coverage_pickle_files_exist():
	# available models
	available_models = seda.models.Models().available_models

	# base path to the seda package
	path_seda = os.path.dirname(os.path.dirname(seda.__file__))
	# path to pickle files
	coverage_dir = 'seda/models_aux/model_coverage'
	full_pattern = os.path.join(path_seda, coverage_dir, '*.pickle')

	# pickle files in model_coverage folder
	coverage_models_full = sorted(glob.glob(full_pattern))

	# models with pickle files
	coverage_models = []
	for coverage_model in coverage_models_full:
		file_name = os.path.basename(coverage_model)
		model_name = file_name.split('_free_parameters.pickle')[0]
		coverage_models.append(model_name)

	# find missing coverage files
	missing = set(available_models) - set(coverage_models)

	# assert all models have coverage
	assert not missing, f'Coverage pickle files in model_coverage missing for models: {sorted(missing)}'

def load_model_spectra_catalog():
	"""Read catalog of example spectra corresponding to models"""

	# base path to the seda package
	path_seda = os.path.dirname(os.path.dirname(seda.__file__))
	# path to example model spectra
	spectra_dir = 'seda/models_aux/model_spectra'
	# file with model names and corresponding example spectra names
	table_file = os.path.join(path_seda, spectra_dir, 'README')
	table = ascii.read(table_file)

	# make a list of 2-tuples of strings for pytest.mark.parametrize
	catalog = []
	for model in table:
		catalog.append((model[0], os.path.join(path_seda, spectra_dir, model[1])))
	
	return catalog

@pytest.mark.parametrize('model, spectrum_file', load_model_spectra_catalog())
def test_read_model_spectrum(model, spectrum_file):
	"""Test that all model spectra can be read without errors"""
	path_seda = os.path.dirname(os.path.dirname(seda.__file__))
	spectra_dir = 'seda/models_aux/model_spectra'

	# read each example model spectrum
	spectrum = seda.models.read_model_spectrum(spectrum_file, model)

	# basic checks
	assert isinstance(spectrum, dict), (
		f'read_model_spectrum should return a dict')
	assert 'wl_model' in spectrum, (
		f"'wl_model' missing for spectrum {os.path.basename(spectrum_file)} by {model} models")
	assert 'flux_model' in spectrum, (
		f"'flux_model' missing for spectrum {os.path.basename(spectrum_file)} by {model} models")
	assert 'flux_model_Jy' in spectrum, (
		f"'flux_model_Jy' missing for spectrum {os.path.basename(spectrum_file)} by {model} models")
	assert len(spectrum['wl_model']) > 0, (
		f"No data read for spectrum {os.path.basename(spectrum_file)} by {model} models")

@pytest.mark.parametrize('model', seda.models.Models().available_models)
def test_select_model_spectra(model):
	# base path to the seda package
	path_seda = os.path.dirname(os.path.dirname(seda.__file__))
	# path to example model spectra
	spectra_dir = 'seda/models_aux/model_spectra'
	model_dir = os.path.join(path_seda, spectra_dir)

	# model-specific filename pattern
	pattern = seda.models.Models(model).filename_pattern

	# select spectra
	result = seda.utils.select_model_spectra(model, model_dir, filename_pattern=pattern)
	assert isinstance(result, dict), (
		f'select_model_spectra should return a dict for model {model}')

	# must find at least one spectrum
	assert len(result['spectra_name']) > 0, (
		f"No spectra found for model {model} using pattern '{pattern}'"
	)

@pytest.mark.parametrize('model, spectrum_file', load_model_spectra_catalog())
def test_generate_model_spectrum(model, spectrum_file):
	"""
	Generate a model spectrum at the exact parameters of an example 
	spectrum and verify that it matches the reference spectrum.
	"""

	# read reference spectrum
	spec_ref = seda.models.read_model_spectrum(spectrum_file, model)
	wl_ref = spec_ref['wl_model']
	flux_ref = spec_ref['flux_model']

	# extract parameters from filename
	params = seda.models.separate_params(model, os.path.basename(spectrum_file))['params']

	# generate spectrum at the same parameters
	path_seda = os.path.dirname(os.path.dirname(seda.__file__))
	spectra_dir = 'seda/models_aux/model_spectra'
	dir_sep = os.sep # directory separator for the current operating system
	model_dir = os.path.join(path_seda, spectra_dir)+dir_sep
	spectrum_gen = seda.utils.generate_model_spectrum(
		params=params, 
		model=model, 
		model_dir=model_dir
	)
	wl_gen = spectrum_gen['wavelength']
	flux_gen = spectrum_gen['flux']

	# compare wavelength grids
	assert np.allclose(wl_gen, wl_ref), (
		f'Wavelength mismatch for model {model}: '
		f'generated vs reference differs'
	)
	# compare flux arrays
	assert np.allclose(flux_gen, flux_ref, rtol=1e-6), (
		f'Flux mismatch for model {model}: '
		f'generated vs reference differs'
	)
