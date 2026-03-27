import seda
import glob
import os
import numpy as np
import pytest
from astropy.io import ascii
from pathlib import Path
from importlib import resources

from tests.conftest import load_model_spectra_catalog

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

	# ensure each parameter is a numpy array
	for key,value in params.items():
		assert isinstance(value, np.ndarray), f"Parameter '{key}' should be a numpy array, got {type(value)}"

	# ensure each parameter array has exactly one element
	for key,value in params.items():
		assert len(value) == 1, f'{key} should have exactly one value, found {len(value)}'

	# check numerical values
	for key, value in expected.items():
		assert params[key][0] == value, f'{key} expected {value}, got {params[key][0]}'

def test_coverage_pickle_files_exist():
	"""Check that each model has its coverage pickle file."""

	# available models
	available_models = seda.models.Models().available_models

	# base path to models_aux
	base = Path(resources.files("seda.models_aux"))

	missing = []

	# loop over models and check coverage.pickle
	for model in available_models:
		model_dir = base / model

		# expected file name (same for all models)
		pickle_file = model_dir / "coverage.pickle"

		if not pickle_file.exists():
			missing.append(model)

	# assert all models have coverage
	assert not missing, (
		f"Missing coverage.pickle for models: {sorted(missing)}"
	)

@pytest.mark.parametrize('model, spectrum_file', load_model_spectra_catalog())
def test_read_model_spectrum(model, spectrum_file):
	"""Test that all model spectra can be read without errors"""

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

@pytest.mark.parametrize("model", seda.models.Models().available_models)
def test_select_model_spectra(model):
	"""Check that example spectra can be selected for each available model."""

	# path to the folder for this specific model inside models_aux
	model_dir = str(Path(resources.files("seda.models_aux")) / model)

	# filename pattern from model config
	pattern = seda.models.Models(model).filename_pattern

	# select spectra using utility function
	result = seda.utils.select_model_spectra(model, model_dir, filename_pattern=pattern)

	# result should be a dictionary
	assert isinstance(result, dict), f"select_model_spectra should return a dict for model {model}"

	# check that 'spectra_name' key exists
	assert "spectra_name" in result, f"'spectra_name' key missing for model {model}"

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

	# construct path to the model folder
	model_dir = str(Path(resources.files("seda.models_aux")) / model)

	# generate spectrum at the same parameters
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
