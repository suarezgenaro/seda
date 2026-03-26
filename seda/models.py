import pickle
import numpy as np
import os
import fnmatch
import json
import xarray
import importlib.util
from astropy import units as u
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
		## path to the this module
		#path_models = os.path.dirname(__file__)+'/'

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
# short name for plot legends for model spectra
def spectra_name_short(model, spectra_name):
	
	short_name = []
	for spectrum_name in spectra_name:
		if model=='Sonora_Diamondback': short_name.append(spectrum_name[:-5])
		if model=='Sonora_Elf_Owl': short_name.append(spectrum_name[8:-3])
		if model=='Sonora_Cholla': short_name.append(spectrum_name[:-5])
		if model=='Sonora_Bobcat': short_name.append(spectrum_name[3:])
		if model=='LB23': short_name.append(spectrum_name[:-3])
		if model=='ATMO2020': short_name.append(spectrum_name.split('spec_')[1][:-4])
		if model=='BT-Settl': short_name.append(spectrum_name[:-16])
		if model=='SM08': short_name.append(spectrum_name[3:])

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
