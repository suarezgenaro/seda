import pickle
import numpy as np
import os
import fnmatch
import json
import xarray
import importlib.util
from astropy import units as u
from astropy.constants import R_jup, R_sun
from astropy.io import ascii
from pathlib import Path
from importlib import resources
from sys import exit

##########################
class Models:
	'''
	Description:
	------------
		See available atmospheric models and get basic parameters from a desired model grid.

	Parameters:
	-----------
	- model : str, optional.
		Atmospheric models for which basic information will be read. 
		See available models with ``seda.models.Models().available_models``.

	Attributes:
	-----------
	- available_models (list) : Atmospheric models available on SEDA.
	- ref (str) : Reference to ``model`` (if provided).
	- name (str) : Name of ``model`` (if provided).
	- bibcode (str) : bibcode identifier for ``model`` (if provided).
	- ADS (str) : ADS links to ``model`` (if provided) reference.
	- download (str) : link to download ``model`` (if provided).
	- filename_pattern (str) : common pattern in all spectra filenames in ``model`` (if provided). 
		It is used to avoid other potential files in the same directory with model spectra.
	- filename_trim (list) : start and end indices of filenames to trim, selecting only the relevant part for display.
	- free_params (list) : free parameters in ``model`` (if provided).
	- params (dict) : values (including repetitions) for each free parameter in ``model`` (if provided).
	- params_unique (dict) : unique (no repetitions) values for each free parameter in ``model`` (if provided).

	Returns:
	--------
	NoneType

	Example:
	--------
	>>> import seda
	>>> 
	>>> # see available atmospheric models
	>>> seda.Models().available_models
	    ['BT-Settl',
	     'ATMO2020',
	     'Sonora_Elf_Owl',
	     'SM08',
	     'Sonora_Bobcat',
	     'Sonora_Diamondback',
	     'Sonora_Cholla',
	     'LB23']
	>>> 
	>>> # see link to the reference paper
	>>> seda.Models('Sonora_Elf_Owl').ADS
	    'https://ui.adsabs.harvard.edu/abs/2024ApJ...963...73M/abstract'
	>>> 
	>>> # see free parameters in one of the models
	>>> seda.Models('Sonora_Elf_Owl').free_params
	    ['Teff', 'logg', 'logKzz', 'Z', 'CtoO']

	Author: Genaro Suárez
	'''

	def __init__(self, model=None):

		# path to model folders
		self.path_models_aux = resources.files('seda.models_aux')

		# read info from json files
		model_configs = self._load_model_configs()

		# set available atmospheric models
		self.available_models = list(model_configs.keys())

		# if a specific model is requested
		if model:
			if model not in model_configs:
				raise Exception(f'"{model}" models are not recognized. Available models: \n          {self.available_models}')

			config = model_configs[model]

			# assign all non-comment fields as attributes
			for key, value in config.items():
				if not key.startswith('_comment'):
					setattr(self, key, value)

			# set attributes related to coverage of the free parameters
			self.model_ranges()

	def _load_model_configs(self):
		"""
		Scan model folders, ensure config + plugin exist,
		and return a dict with config and available models

		Author: Genaro Suárez
		Data: 2026-03-25
		"""

		model_configs = {}

		# loop over model directories
		for model_dir in self.path_models_aux.iterdir():
			# skip non-directories and internal folders
			if not model_dir.is_dir() or model_dir.name.startswith('_'):
				continue

			config_path = model_dir / 'config.json'
			plugin_path = model_dir / 'plugin.py'

			# require both config and plugin files
			if not config_path.exists() or not plugin_path.exists():
				continue

			# read model name from config
			with open(config_path) as f:
				config = json.load(f)

			model_name = config['model']
			model_configs[model_name] = config

		return model_configs

	def model_ranges(self):
		'''
		Read coverage of model free parameters.

		Author: Genaro Suárez
		'''

		# path to model folders
		path_models_aux = resources.files('seda.models_aux')

		# open the pickle file, if any, with model coverage
		pickle_file = f'{path_models_aux}/{self.model}/coverage.pickle'
		if not os.path.exists(pickle_file): # if the pickle file exists
			raise Exception(f'"{pickle_file}" file with model coverage is missing')
			
		#if os.path.exists(pickle_file): # if the pickle file exists
		else:
			with open(pickle_file, 'rb') as file:
				model_coverage = pickle.load(file)

			# dictionary to save all values for each free parameter
			params = {}
			for param in model_coverage['params']:
				params[param] = model_coverage['params'][param] # unique values for each free parameter
			self.params = params

			# dictionary to save unique values for each free parameter
			params_unique = {}
			for param in model_coverage['params']:
				params_unique[param] = np.unique(model_coverage['params'][param]) # unique values for each free parameter
			self.params_unique = params_unique

	def get_parameters(self, return_type="dict", include_values=True):
		"""Return all user-facing attributes (exclude __dunder__ names)."""

		attrs = {}
		for key, value in self.__dict__.items():
			if key.startswith('__'):
				continue
			if return_type == 'dict' and include_values:
				attrs[key] = value
			else:
				attrs[key] = None
		if return_type == 'list':
			return list(attrs.keys())
		return attrs

##########################
class EvolutionaryModels:
	'''
	Description:
	------------
		See available evolutionary models and get basic parameters from a desired grid.

	Parameters:
	-----------
	- model : str, optional
		Evolutionary models. If provided, the model metadata from its ``config.json``
		(e.g. ``ref``, ``bibcode``, ``ADS``, ``download``, ``columns``, ``units``) is
		assigned as attributes. If omitted, only ``available_models`` is set.

	Attributes:
	-----------
	- available_models (list) : Evolutionary models available on SEDA.
	- available_tables (list) : Bundled table basenames for ``model`` (if provided).
	- params (dict) : Per-table ``[min, max]`` for each grid column returned by the plugin (if provided).

	Example:
	--------
	>>> import seda
	>>>
	>>> # see available evolutionary models
	>>> seda.models.EvolutionaryModels().available_models
	    ['ATMO2020', 'BHAC2015', 'Sonora_Bobcat', 'Sonora_Diamondback']
	>>>
	>>> # see the reference for a given evolutionary model
	>>> seda.models.EvolutionaryModels('Sonora_Bobcat').ref
	    'Marley et al. (2021)'
	>>>
	>>> # min/max of each grid column for every bundled table
	>>> seda.models.EvolutionaryModels('Sonora_Bobcat').params['nc+0.0_co1.0_mass']['mass']
	    [0.0005, 0.08]


	Author: Theo Olsen

	Date: 2026-06-04
	'''
#basically the same as Models class, but for evolutionary models
	def __init__(self, model=None):

		# path to evolutionary model folders
		self.path_evolution_aux = resources.files('seda.evolution_aux')

		# read info from json files
		model_configs = self._load_model_configs()

		# set available evolutionary models
		self.available_models = list(model_configs.keys())

		# if a specific model is requested
		if model:
			if model not in model_configs:
				raise Exception(f'Evolutionary models "{model}" are not recognized. '
				                f'Available evolutionary models: \n          {self.available_models}')

			config = model_configs[model]

			# assign all non-comment fields as attributes
			for key, value in config.items():
				if not key.startswith('_comment'):
					setattr(self, key, value)

			self.model = model

	def _load_model_configs(self):
		"""
		Scan evolutionary model folders, ensure config + plugin exist,
		and return a dict with config keyed by available models.
		"""

		model_configs = {}

		# loop over evolutionary model directories
		for model_dir in self.path_evolution_aux.iterdir():
			# skip non-directories and internal folders
			if not model_dir.is_dir() or model_dir.name.startswith('_'):
				continue

			config_path = model_dir / 'config.json'
			plugin_path = model_dir / 'plugin.py'

			# require both config and plugin files
			if not config_path.exists() or not plugin_path.exists():
				continue

			# read model name from config
			with open(config_path) as f:
				config = json.load(f)

			model_name = config['model']
			model_configs[model_name] = config

		return model_configs

	@property
	def available_tables(self):
		"""Basenames of evolutionary table files bundled for this model."""
		if not hasattr(self, 'model'):
			raise Exception('Pass a model name to EvolutionaryModels to list available tables.')
		return _list_evolutionary_tables(self.model)

	@property
	def params(self):
		"""Per-table min/max for each grid column in native units (see ``config.json`` ``units``)."""
		if not hasattr(self, 'model'):
			raise Exception('Pass a model name to EvolutionaryModels to read parameter ranges.')
		if not hasattr(self, '_params_cache'):
			self._params_cache = _compute_evolutionary_param_ranges(self.model)
		return self._params_cache

##########################
def _list_evolutionary_tables(model):
	'''
	Return sorted basenames of evolutionary table files in ``evolution_aux/<model>/``.
	'''

	if model not in EvolutionaryModels().available_models:
		raise Exception(f'Evolutionary models "{model}" are not recognized. '
		                f'Available evolutionary models: \n          {EvolutionaryModels().available_models}')

	config, _ = _load_evolutionary_model(model)
	pattern = config.get('filename_pattern', '*')
	model_dir = _EVOL_BASE_PATH / model
	skip = {'config.json', 'plugin.py'}

	tables = sorted(
		f.name for f in model_dir.iterdir()
		if f.is_file() and f.name not in skip and fnmatch.fnmatch(f.name, pattern)
	)
	return tables

##########################
def resolve_evolutionary_table(model, filename=None):
	'''
	Description:
	------------
		Resolve the basename of a bundled evolutionary table for ``model``.

	Parameters:
	-----------
	- model : str
		Evolutionary models. See available models in
		``seda.models.EvolutionaryModels().available_models``.
	- filename : str, optional
		Basename of a table file inside ``evolution_aux/<model>/``.
		If omitted and exactly one table exists, it is selected automatically.
		If omitted and multiple tables exist, available basenames are printed
		and a ``ValueError`` is raised.

	Returns:
	--------
	- str
		Basename of the selected evolutionary table file.

	Author: Theo Olsen

	Date: 06/04/2026
	'''

	tables = _list_evolutionary_tables(model)

	if not tables:
		raise FileNotFoundError(
			f'No evolutionary table files found for "{model}" in evolution_aux/{model}/.'
		)

	if filename is None:   #automatically return only avaialble table
		if len(tables) == 1:
			return tables[0]
		print(f'Available evolutionary tables for "{model}":')
		for name in tables:
			print(f'  - {name}')
		raise ValueError(
			f'Multiple evolutionary tables are available for "{model}". '
			f'Pass filename=<basename> to select one.'
		)

	if filename not in tables:
		print(f'Available evolutionary tables for "{model}":')
		for name in tables:
			print(f'  - {name}')
		raise ValueError(
			f'filename={filename!r} is not recognized for "{model}". '
			f'Choose one of the available table basenames listed above.'
		)

	return filename

##########################
def separate_params(model, spectra_name, save_results=False, out_file=None):
	'''
	Description:
	------------
		Extract parameters from the file names for model spectra.

	Parameters:
	-----------
	- model : str
		Atmospheric models. See available models in ``input_parameters.ModelOptions``.  
	- spectra_name : array or list
		Model spectra names (without full path).
	- save_results : {``True``, ``False``}, optional (default ``False``)
		Save (``True``) or do not save (``False``) the output as a pickle file named '``model``\_free\_parameters.pickle'.
	- out_file : str, optional
		File name to save the results as a pickle file (it can include a path e.g. my_path/free\_params.pickle).
		Default name is '``model``\_free_parameters.pickle' and is stored at the notebook location.

	Returns:
	--------
	Dictionary with parameters for each model spectrum.
		- ``spectra_name`` : model spectra names.
		- ``params``: model free parameters for the spectra.

	Example:
	--------
	>>> import seda
	>>>
	>>> model = 'Sonora_Elf_Owl'
	>>> spectra_name = np.array(['spectra_logzz_4.0_teff_750.0_grav_178.0_mh_0.0_co_1.0.nc', 
	>>>                          'spectra_logzz_2.0_teff_800.0_grav_316.0_mh_0.0_co_1.0.nc'])
	>>> seda.models.separate_params(spectra_name=spectra_name, model=model)
	    {'spectra_name': array(['spectra_logzz_4.0_teff_750.0_grav_178.0_mh_0.0_co_1.0.nc',
	            'spectra_logzz_2.0_teff_800.0_grav_316.0_mh_0.0_co_1.0.nc'],
	           dtype='<U56'),
	     'params': {'Teff': array([750., 800.]),
	      'logg': array([4.25, 4.5 ]),
	      'logKzz': array([4., 2.]),
	      'Z': array([0., 0.]),
	      'CtoO': array([1., 1.])}}

	Author: 
		Genaro Suárez

	Date: 
		Created: 2021-05-12
		Last Modified: 2026-03-25
	'''

	# if there is one input spectrum with its name given as a string, convert it into a list
	if isinstance(spectra_name, str): spectra_name = [spectra_name]

	# load model config and plugin
	config, plugin = _load_model(model)

	# call the plugin to get the raw parameters
	params = plugin._separate_params(spectra_name)

	# sort params in the same order as free_params in the JSON file
	free_params = Models(model).free_params

	params = reorder_dict(params, free_params)

	# output dictionary
	out = {'spectra_name': spectra_name, 'params': params}

	# save output dictionary
	if save_results:
		if out_file is None: out_file = f'{model}_coverage.pickle'
		with open(out_file, 'wb') as file:
			pickle.dump(out, file)
			print(f'{out_file} saved successfully')

	return out

##########################
def read_model_spectrum(spectrum_name_full, model, model_wl_range=None):
	'''
	Description:
	------------
		Read a desired model spectrum.

	Parameters:
	-----------
	- model : str
		Atmospheric models. See available models in ``input_parameters.ModelOptions``.  
	- spectrum_name_full: str
		Spectrum file name with full path.
	- model_wl_range : float array, optional
		Minimum and maximum wavelength (in microns) to cut the model spectrum.

	Returns:
	--------
	Dictionary with model spectrum:
		- ``'wl_model'`` : wavelengths in microns
		- ``'flux_model'`` : fluxes in erg/s/cm2/A
		- ``'flux_model_Jy'`` : fluxes in Jy

	Author: 
		Genaro Suárez

	Date: 
		Created: 2021-05-12
		Last Modified: 2026-03-25
	'''

	# verify the input model is available
	if model not in Models().available_models: raise Exception(f'Models "{model}" are not recognized. Available models: \n          {Models().available_models}')

	# load plugin (and config if needed)
	_, plugin = _load_model(model)

	# get the model spectrum
	spectrum = plugin._read_model_spectrum(spectrum_name_full)

	# extract main arrays first
	wl_model = spectrum['wl_model']
	flux_model = spectrum['flux_model']

	# ensure wl is sorted
	sort_index = np.argsort(wl_model)
	wl_model = wl_model[sort_index]
	flux_model = flux_model[sort_index]

	# cut the model spectra to the indicated range
	if model_wl_range is not None:
		mask = (wl_model>=model_wl_range[0]) & (wl_model<=model_wl_range[1])
		wl_model = wl_model[mask]
		flux_model = flux_model[mask]

	# obtain fluxes in Jy
	flux_model_Jy = (flux_model*u.erg/u.s/u.cm**2/(u.nm*0.1)).to(u.Jy, equivalencies=u.spectral_density(wl_model*u.micron)).value

	out = {'wl_model': wl_model, 'flux_model': flux_model, 'flux_model_Jy': flux_model_Jy}

	return out

##########################
# read a pre-stored convolved model spectrum
# it is a netCDF file with xarray produced by convolve_spectrum
def read_model_spectrum_conv(spectrum_name_full, model_wl_range=None):

	# read convolved spectrum
	spectrum = xarray.open_dataset(spectrum_name_full)
	wl_model = spectrum['wl'].data # um
	flux_model = spectrum['flux'].data # erg/s/cm2/A

	# cut the model spectra to the indicated range
	if model_wl_range is not None:
		mask = (wl_model>=model_wl_range[0]) & (wl_model<=model_wl_range[1])
		wl_model = wl_model[mask]
		flux_model = flux_model[mask]

	# obtain fluxes in Jy
	flux_model_Jy = (flux_model*u.erg/u.s/u.cm**2/(u.nm*0.1)).to(u.Jy, equivalencies=u.spectral_density(wl_model*u.micron)).value

	out = {'wl_model': wl_model, 'flux_model': flux_model, 'flux_model_Jy': flux_model_Jy}

	return out

##########################
def read_PT_profile(filename, model):
	'''
	Description:
	------------
		Read a PT profile from atmospheric models

	Parameters:
	-----------
	- model : str
		Atmospheric models. See available models in ``input_parameters.ModelOptions``.  
	- filename: str
		Spectrum file name with full path.

	Returns:
	--------
	Dictionary with model spectrum:
		- ``'pressure'`` : pressure in bars
		- ``'temperature'`` : temperature in K

	Example:
	--------
	>>> import seda
	>>> 
	>>> # desired models and PT profile file
	>>> model = 'Sonora_Diamondback'
	>>> filename = 'my_path/Sonora_Diamondback/pressure-temperature_profiles/t1000g100f1_m-0.5_co1.0.pt' # change my_path accordingly
	>>> 
	>>> # read PT profile
	>>> out = seda.read_PT_profile(filename=filename, model=model)
	>>> P = out['pressure'] # pressure in bar
	>>> T = out['temperature'] # temperature in K

	Author: Genaro Suárez
	'''
	
	# read PT profile
	if (model == 'Sonora_Diamondback'):
		spec_model = ascii.read(filename, data_start=2, format='no_header')
		P_model = spec_model['col2'] # bar
		T_model = spec_model['col3'] # K

	else:
		raise Exception(f'"{model}" models are not recognized.')

	# output dictionary
	out = {'pressure': P_model, 'temperature': T_model}

	return out

##########################
def read_evolutionary_model(filename, model):
	'''
	Description:
	------------
		Read an evolutionary model table and return its grid arrays.

	Parameters:
	-----------
	- filename : str
		Basename of an evolutionary table file inside ``evolution_aux/<model>/``.
		Pass ``None`` to auto-select when only one table is bundled.
	- model : str
		Evolutionary models. See available models in
		``seda.models.EvolutionaryModels().available_models``.

	Returns:
	--------
	Dictionary with evolutionary grid columns parsed by the model plugin.
	Each model defines its own columns; all tables must provide ``logL`` and
	``radius`` for interpolation. See ``config.json`` ``units`` for column units.

	Example:
	--------
	>>> import seda
	>>>
	>>> model = 'Sonora_Bobcat'
	>>> filename = 'nc+0.0_co1.0_mass'
	>>>
	>>> out = seda.read_evolutionary_model(filename=filename, model=model)
	>>> mass = out['mass']

	Author: Theo Olsen

	Date: 2026-06-04
	'''
	if model not in EvolutionaryModels().available_models:
		raise Exception(f'Evolutionary models "{model}" are not recognized. '
		                f'Available evolutionary models: \n          {EvolutionaryModels().available_models}')

	basename = resolve_evolutionary_table(model, filename)

	_, plugin = _load_evolutionary_model(model)
	out = plugin._read_evolutionary_model(basename)

	for key in ('logL', 'radius'):
		if key not in out:
			raise ValueError(
				f'{model}/plugin.py must return "{key}" in the grid dictionary '
				f'for evolutionary interpolation.'
			)

	return out

##########################
def _param_ranges_from_grid(grid):
	"""Return ``{column: [min, max]}`` for each array in an evolutionary grid dict."""

	return {
		col: [float(np.min(grid[col])), float(np.max(grid[col]))]
		for col in grid
	}

##########################
def _compute_evolutionary_param_ranges(model):
	"""Min/max of every grid column for each bundled evolutionary table."""

	ranges = {}
	for filename in _list_evolutionary_tables(model):
		grid = read_evolutionary_model(filename=filename, model=model)
		ranges[filename] = _param_ranges_from_grid(grid)
	return ranges

##########################
def _split_evolutionary_tracks(grid):
	"""Split a bundled evolutionary table into constant-mass cooling tracks."""

	mass = np.asarray(grid['mass'], dtype=float)
	tracks = []
	for mass_val in np.unique(mass):
		mask = mass == mass_val
		track = {
			col: np.asarray(grid[col], dtype=float)[mask]
			for col in grid
		}
		order = np.argsort(track['age'])
		track = {col: track[col][order] for col in track}
		tracks.append(track)
	return tracks


def _build_isochrone(tracks, age):
	"""Build a constant-age isochrone by interpolating each mass track to ``age``."""

	age = float(age)
	isochrone = {col: [] for col in tracks[0]}
	for track in tracks:
		ages = track['age']
		if age < ages[0] or age > ages[-1]:
			continue
		for col in track:
			val = np.interp(age, ages, track[col])
			isochrone[col].append(val)

	if not isochrone['age']:
		return {col: np.array([], dtype=float) for col in isochrone}

	isochrone = {col: np.asarray(vals, dtype=float) for col, vals in isochrone.items()}
	order = np.argsort(isochrone['logL'])
	return {col: isochrone[col][order] for col in isochrone}


def _interp_on_isochrone(isochrone, logL_targets):
	"""Interpolate isochrone columns at target log10(L/Lsun) values."""

	logL_targets = np.atleast_1d(np.asarray(logL_targets, dtype=float))
	logL_grid = np.asarray(isochrone['logL'], dtype=float)

	if logL_grid.size < 2:
		return {
			col: np.full(logL_targets.shape, np.nan, dtype=float)
			for col in isochrone if col != 'logL'
		}

	sort_idx = np.argsort(logL_grid)
	logL_sorted = logL_grid[sort_idx]

	out = {}
	for col, values in isochrone.items():
		if col == 'logL':
			continue
		val_sorted = np.asarray(values, dtype=float)[sort_idx]
		out[col] = np.interp(
			logL_targets, logL_sorted, val_sorted,
			left=np.nan, right=np.nan,
		)
	return out


def _evolutionary_radius_unit(model):
	"""Return the native radius unit string from an evolutionary model config."""

	config, _ = _load_evolutionary_model(model)
	return config.get('units', {}).get('radius', 'R_sun')


def _convert_radius_to_rjup(radius, model):
	"""Convert evolutionary-grid radius values to R_jup."""

	radius = np.asarray(radius, dtype=float)
	unit = _evolutionary_radius_unit(model)
	if unit == 'R_sun':
		return (radius * R_sun).to(R_jup).value
	if unit == 'R_jup':
		return radius
	raise ValueError(
		f'Unsupported evolutionary radius unit {unit!r} for model {model!r}.'
	)


def _precompute_isochrone_cache(tracks, grid):
	"""Pre-build isochrones at every tabulated age in an evolutionary grid."""

	tabulated_ages = np.unique(np.asarray(grid['age'], dtype=float))
	return {
		float(age): _build_isochrone(tracks, float(age))
		for age in tabulated_ages
	}


def _nan_on_isochrone(isochrone, logL_targets):
	"""Return NaN interpolation results for every non-logL isochrone column."""

	logL_targets = np.atleast_1d(np.asarray(logL_targets, dtype=float))
	return {
		col: np.full(logL_targets.shape, np.nan, dtype=float)
		for col in isochrone if col != 'logL'
	}


def _lookup_on_isochrone_grid(isochrone_cache, tabulated_ages, age, logL_targets):
	"""Look up evolutionary parameters at ``(age, logL)``, interpolating in age."""

	logL_targets = np.atleast_1d(np.asarray(logL_targets, dtype=float))
	tabulated_ages = np.asarray(tabulated_ages, dtype=float)
	age = float(age)
	age_min = float(tabulated_ages[0])
	age_max = float(tabulated_ages[-1])

	if age < age_min or age > age_max:
		return _nan_on_isochrone(isochrone_cache[age_min], logL_targets)

	hi = int(np.searchsorted(tabulated_ages, age))
	if hi >= len(tabulated_ages):
		return _interp_on_isochrone(isochrone_cache[age_max], logL_targets)

	age_lo = float(tabulated_ages[hi - 1])
	age_hi = float(tabulated_ages[hi])
	if age_lo == age_hi or age == age_lo:
		return _interp_on_isochrone(isochrone_cache[age_lo], logL_targets)
	if age == age_hi:
		return _interp_on_isochrone(isochrone_cache[age_hi], logL_targets)

	weight = (age - age_lo) / (age_hi - age_lo)
	interp_lo = _interp_on_isochrone(isochrone_cache[age_lo], logL_targets)
	interp_hi = _interp_on_isochrone(isochrone_cache[age_hi], logL_targets)

	out = {}
	for col in interp_lo:
		out[col] = (1.0 - weight) * interp_lo[col] + weight * interp_hi[col]
	return out

##########################
# short name for plot legends for model spectra
def spectra_name_short(model, spectra_name):

	if isinstance(spectra_name, str): spectra_name = [spectra_name]
	if isinstance(spectra_name, np.ndarray): spectra_name = spectra_name.tolist()
	if isinstance(spectra_name, float): spectra_name = [spectra_name]

	short_name = []

	trim = getattr(Models(model), "filename_trim", None)
	
	for spectrum_name in spectra_name:
		if trim:
			start, end = trim
			short_name.append(spectrum_name[start:end])
		else:
			raise ValueError(f"No 'filename_trim' parameter in 'config.json' for '{model}' models")

	return short_name

##########################
_BASE_PATH = Path(__file__).parent / 'models_aux'
_PLUGIN_CACHE = {}

def _load_model(model):

	# return cached model if already loaded
	if model in _PLUGIN_CACHE:
		return _PLUGIN_CACHE[model]

	# define model folder and load JSON config
	model_dir = _BASE_PATH / model
	config_path = model_dir / 'config.json'

	# verify the json file exists
	if not config_path.exists():
		raise FileNotFoundError(f"No config.json for model '{model}'")

	with open(config_path) as f:
		config = json.load(f)

	# load plugin.py dynamically as a module
	plugin_path = model_dir / 'plugin.py'
	# verify that plugin.py exists
	if not plugin_path.exists():
		raise NotImplementedError(
			f"Model '{model}' has no plugin.py"
		)

	# create a module spec for the plugin file
	spec = importlib.util.spec_from_file_location(model, plugin_path)
	# create a module object from that spec
	plugin = importlib.util.module_from_spec(spec)
	# execute the module code to load it into Python
	spec.loader.exec_module(plugin)

	# validate required functions
	for func in ['_read_model_spectrum', '_separate_params']:
		if not hasattr(plugin, func):
			raise AttributeError(
				f"{model}/plugin.py must define '{func}'"
			)

	# cache and return
	_PLUGIN_CACHE[model] = (config, plugin)
	return config, plugin

##########################
_EVOL_BASE_PATH = Path(__file__).parent / 'evolution_aux'
_EVOL_PLUGIN_CACHE = {}

def _load_evolutionary_model(model):

	# return cached model if already loaded
	if model in _EVOL_PLUGIN_CACHE:
		return _EVOL_PLUGIN_CACHE[model]

	# define model folder and load JSON config
	model_dir = _EVOL_BASE_PATH / model
	config_path = model_dir / 'config.json'

	# verify the json file exists
	if not config_path.exists():
		raise FileNotFoundError(f"No config.json for evolutionary model '{model}'")

	with open(config_path) as f:
		config = json.load(f)

	# load plugin.py dynamically as a module
	plugin_path = model_dir / 'plugin.py'
	# verify that plugin.py exists
	if not plugin_path.exists():
		raise NotImplementedError(
			f"Evolutionary model '{model}' has no plugin.py"
		)

	# create a module spec for the plugin file
	spec = importlib.util.spec_from_file_location(f'{model}_evolution', plugin_path)
	# create a module object from that spec
	plugin = importlib.util.module_from_spec(spec)
	# execute the module code to load it into Python
	spec.loader.exec_module(plugin)

	# validate required functions in model folder
	for func in ['_read_evolutionary_model', '_convert_inputs']:
		if not hasattr(plugin, func):
			raise AttributeError(
				f"{model}/plugin.py (evolution_aux) must define '{func}'"
			)

	# cache and return
	_EVOL_PLUGIN_CACHE[model] = (config, plugin)
	return config, plugin

#+++++++++++++++++++++++++++
# reorder dictionary keys according to a list with the order for the keys
def reorder_dict(data_dict, order_list):

	reordered_dict = {}
	for key in order_list:
	    if key in data_dict:
	        reordered_dict[key] = data_dict[key]
	    else:
	        raise Exception(f'{key} param is not provided')

	return reordered_dict
