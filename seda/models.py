import pickle
import numpy as np
import os
import fnmatch
import json
import xarray
from astropy import units as u
from astropy.io import ascii
from specutils.utils.wcs_utils import vac_to_air
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
		See available models with ``seda.Models().available_models``.

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

		# path to this module
		path_models = os.path.dirname(__file__)+'/'

		# available atmospheric models
		# open all the json files in models_aux/model_specifics
		path_jsons = path_models+'models_aux/model_specifics/'
		
		json_files = fnmatch.filter(os.listdir(path_jsons), '*.json')

		# read available models
		available_models = []
		for json_file in json_files:
			if json_file!='template_models.json': # avoid template
				available_models.append(json.load(open(path_jsons+json_file))['model'])

		# set attribute
		self.available_models = available_models

		# read attributes from the json file for desired models
		if model is not None:
			if model not in self.available_models: raise Exception(f'"{model}" models are not recognized. Available models: \n          {self.available_models}')
			
			self.model = model
	
			# read relevant info for available atmospheric models from the json files in models_aux/model_specifics
			for json_file in json_files:
				if json.load(open(path_jsons+json_file))['model']==model: models_json = json.load(open(path_jsons+json_file))

			# define attributes
			self.ref = models_json['ref']
			self.name = models_json['name']
			self.bibcode = models_json['bibcode']
			self.ADS = models_json['ADS']
			self.download = models_json['download']
			self.filename_pattern = models_json['filename_pattern']
			self.free_params = models_json['free_params']

			# set attributes related to coverage of the free parameters
			self.model_ranges()

	def model_ranges(self):
		'''
		Read coverage of model free parameters.

		Author: Genaro Suárez
		'''

		# path to the this module
		path_models = os.path.dirname(__file__)+'/'

		# open the pickle file, if any, with model coverage
		pickle_file = f'{path_models}models_aux/model_coverage/{self.model}_free_parameters.pickle'
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
	>>> seda.separate_params(spectra_name=spectra_name, model=model)
	    {'spectra_name': array(['spectra_logzz_4.0_teff_750.0_grav_178.0_mh_0.0_co_1.0.nc',
	                            'spectra_logzz_2.0_teff_800.0_grav_316.0_mh_0.0_co_1.0.nc'],
	    'Teff': array([750., 800.]),
	    'logg': array([4.25042   , 4.49968708]),
	    'logKzz': array([4., 2.]),
	    'Z': array([0., 0.]),
	    'CtoO': array([1., 1.])}

	Author: Genaro Suárez
	'''

	# if there is one input spectrum with its name given as a string, convert it into a list
	if isinstance(spectra_name, str): spectra_name = [spectra_name]

	out = {'spectra_name': spectra_name} # start dictionary with some parameters
	out['params'] = {}

	# get parameters from model spectra names
	# consider a way that also works when adding an additional string at the end of the name
	# which is the case when convolved spectra are store and then read
	if (model == 'Exo-REM'):
		Teff_fit = np.zeros(len(spectra_name))
		logg_fit = np.zeros(len(spectra_name))
		Z_fit = np.zeros(len(spectra_name))
		CtoO_fit = np.zeros(len(spectra_name))
		for i in range(len(spectra_name)):
			# Teff
			Teff_fit[i] = float(spectra_name[i].split('_')[2][:-1]) # K
			# logg
			logg_fit[i] = float(spectra_name[i].split('_')[3][4:]) # g in cgs
			# Z
			Z_fit[i] = np.round(np.log10(float(spectra_name[i].split('_')[4][3:])),2) # 
			# CtoO
			CtoO_fit[i] = float(spectra_name[i].split('_')[5].split('.dat')[0][2:])
		out['params']['Teff']= Teff_fit
		out['params']['logg']= logg_fit
		out['params']['Z']= Z_fit
		out['params']['CtoO']= CtoO_fit

	if (model == 'Sonora_Diamondback'):
		Teff_fit = np.zeros(len(spectra_name))
		logg_fit = np.zeros(len(spectra_name))
		Z_fit = np.zeros(len(spectra_name))
		fsed_fit = np.zeros(len(spectra_name))
		for i in range(len(spectra_name)):
			# Teff 
			Teff_fit[i] = float(spectra_name[i].split('g')[0][1:]) # in K
			# logg
			logg_fit[i] = round(np.log10(float(spectra_name[i].split('g')[1].split('_')[0][:-2])),1) + 2 # g in cgs
			# Z
			Z_fit[i] = float(spectra_name[i].split('_')[1][1:])
			# fsed
			fsed = spectra_name[i].split('_')[0][-1:]
			if fsed=='c': fsed_fit[i] = 99 # 99 to indicate nc (no clouds)
			if fsed!='c': fsed_fit[i] = float(fsed)
		out['params']['Teff']= Teff_fit
		out['params']['logg']= logg_fit
		out['params']['Z']= Z_fit
		out['params']['fsed']= fsed_fit
	if (model == 'Sonora_Elf_Owl'):
		Teff_fit = np.zeros(len(spectra_name))
		logg_fit = np.zeros(len(spectra_name))
		logKzz_fit = np.zeros(len(spectra_name))
		Z_fit = np.zeros(len(spectra_name))
		CtoO_fit = np.zeros(len(spectra_name))
		for i in range(len(spectra_name)):
			# Teff 
			Teff_fit[i] = float(spectra_name[i].split('_')[4]) # in K
			# logg
			logg_fit[i] = round_logg_point25(np.log10(float(spectra_name[i].split('_')[6])) + 2) # g in cgs
			# logKzz
			logKzz_fit[i] = float(spectra_name[i].split('_')[2]) # Kzz in cgs
			# Z
			Z_fit[i] = float(spectra_name[i].split('_')[8]) # in cgs
			# C/O
			CtoO_fit[i] = float(spectra_name[i].split('_')[10][:-3])
		out['params']['Teff']= Teff_fit
		out['params']['logg']= logg_fit
		out['params']['logKzz']= logKzz_fit
		out['params']['Z']= Z_fit
		out['params']['CtoO']= CtoO_fit
	if (model == 'LB23'):
		Teff_fit = np.zeros(len(spectra_name))
		logg_fit = np.zeros(len(spectra_name))
		Z_fit = np.zeros(len(spectra_name))
		logKzz_fit = np.zeros(len(spectra_name))
		Hmix_fit = np.zeros(len(spectra_name))
		for i in range(len(spectra_name)):
			# Teff 
			Teff_fit[i] = float(spectra_name[i].split('_')[0][1:]) # K
			# logg
			logg_fit[i] = float(spectra_name[i].split('_')[1][1:]) # logg
			# Z (metallicity)
			Z_fit[i] = np.round(np.log10(float(spectra_name[i].split('_')[2][1:])),1)
			# Kzz (radiative zone)
			logKzz_fit[i] = np.log10(float(spectra_name[i].split('CDIFF')[1].split('_')[0])) # in cgs units
			# Hmix
			Hmix_fit[i] = float(spectra_name[i].split('HMIX')[1][:5])
		out['params']['Teff']= Teff_fit
		out['params']['logg']= logg_fit
		out['params']['Z']= Z_fit
		out['params']['logKzz']= logKzz_fit
		out['params']['Hmix']= Hmix_fit
	if (model == 'Sonora_Cholla'):
		Teff_fit = np.zeros(len(spectra_name))
		logg_fit = np.zeros(len(spectra_name))
		logKzz_fit = np.zeros(len(spectra_name))
		for i in range(len(spectra_name)):
			# Teff 
			Teff_fit[i] = float(spectra_name[i].split('_')[0][:-1]) # K
			# logg
			logg_fit[i] = round_logg_point25(np.log10(float(spectra_name[i].split('_')[1][:-1])) + 2) # g in cm/s2
			# logKzz
			logKzz_fit[i] = float(spectra_name[i].split('_')[2].split('.')[0][-1]) # Kzz in cm2/s
		out['params']['Teff']= Teff_fit
		out['params']['logg']= logg_fit
		out['params']['logKzz']= logKzz_fit
	if (model == 'Sonora_Bobcat'):
		Teff_fit = np.zeros(len(spectra_name))
		logg_fit = np.zeros(len(spectra_name))
		Z_fit = np.zeros(len(spectra_name))
		CtoO_fit = np.zeros(len(spectra_name))
		for i in range(len(spectra_name)):
			# Teff 
			Teff_fit[i] = float(spectra_name[i].split('_')[1].split('g')[0][1:]) # K
			# logg
			logg_fit[i] = round_logg_point25(np.log10(float(spectra_name[i].split('_')[1].split('g')[1][:-2])) + 2) # g in cm/s2
			# Z
			Z_fit[i] = float(spectra_name[i].split('_')[2][1:])
			# C/O
			if (len(spectra_name[i].split('_'))==4): # when the spectrum file name includes the C/O
				CtoO_fit[i] = float(spectra_name[i].split('_')[3][2:])
			if (len(spectra_name[i].split('_'))==3): # when the spectrum file name does not include the C/O
				CtoO_fit[i] = 1.0
		out['params']['Teff']= Teff_fit
		out['params']['logg']= logg_fit
		out['params']['Z']= Z_fit
		out['params']['CtoO']= CtoO_fit
	if (model == 'ATMO2020'):
		Teff_fit = np.zeros(len(spectra_name))
		logg_fit = np.zeros(len(spectra_name))
		logKzz_fit = np.zeros(len(spectra_name))
		for i in range(len(spectra_name)):
			# Teff 
			Teff_fit[i] = float(spectra_name[i].split('spec_')[1].split('_')[0][1:])
			# logg
			logg_fit[i] = float(spectra_name[i].split('spec_')[1].split('_')[1][2:])
			# logKzz
			if (spectra_name[i].split('spec_')[1].split('lg')[1][4:-4]=='NEQ_weak'): # when the grid is NEQ_weak
				logKzz_fit[i] = 4
			if (spectra_name[i].split('spec_')[1].split('lg')[1][4:-4]=='NEQ_strong'): # when the grid is NEQ_strong
				logKzz_fit[i] = 6
		out['params']['Teff']= Teff_fit
		out['params']['logg']= logg_fit
		out['params']['logKzz']= logKzz_fit
	if (model == 'BT-Settl'):
		Teff_fit = np.zeros(len(spectra_name))
		logg_fit = np.zeros(len(spectra_name))
		for i in range(len(spectra_name)):
			# Teff 
			Teff_fit[i] = float(spectra_name[i].split('-')[0][3:]) * 100 # K
			# logg
			logg_fit[i] = float(spectra_name[i].split('-')[1]) # g in cm/s^2
		out['params']['Teff']= Teff_fit
		out['params']['logg']= logg_fit
	if (model == 'SM08'):
		Teff_fit = np.zeros(len(spectra_name))
		logg_fit = np.zeros(len(spectra_name))
		fsed_fit = np.zeros(len(spectra_name))
		for i in range(len(spectra_name)):
			# Teff 
			Teff_fit[i] = float(spectra_name[i].split('_')[1].split('g')[0][1:])
			# logg
			logg_fit[i] = np.round(np.log10(float(spectra_name[i].split('_')[1].split('g')[1].split('f')[0])), 1) + 2 # g in cm/s^2
			# fsed
			fsed_fit[i] = float(spectra_name[i].split('_')[1].split('g')[1].split('f')[1])
		out['params']['Teff']= Teff_fit
		out['params']['logg']= logg_fit
		out['params']['fsed']= fsed_fit

	# save output dictionary
	if save_results:
		if out_file is None: out_file = f'{model}_free_parameters.pickle'
		with open(out_file, 'wb') as file:
			pickle.dump(out, file)

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

	Author: Genaro Suárez
	'''

	# verify the input model is available
	if model not in Models().available_models: raise Exception(f'Models "{model}" are not recognized. Available models: \n          {Models().available_models}')

	# read model spectra files
	if (model == 'Exo-REM'):
		# reading how to convert fluxes in wavenumber units to fluxes in wavelength units
		print('exit')
	if (model == 'Sonora_Diamondback'):
		spec_model = ascii.read(spectrum_name_full, data_start=3, format='no_header')
		wl_model = spec_model['col1'] * u.micron # um (in vacuum?)
		wl_model = vac_to_air(wl_model).value # um in the air
		flux_model = spec_model['col2'] * u.W/u.m**2/u.m # W/m2/m
		flux_model = flux_model.to(u.erg/u.s/u.cm**2/(u.nm*0.1)).value # erg/s/cm2/A
	if (model == 'Sonora_Elf_Owl'):
		spec_model = xarray.open_dataset(spectrum_name_full) # Sonora Elf Owl model spectra have NetCDF Data Format data
		wl_model = spec_model['wavelength'].data * u.micron # um
		wl_model = wl_model.value
		flux_model = spec_model['flux'].data * u.erg/u.s/u.cm**2/u.cm # erg/s/cm2/cm
		flux_model = flux_model.to(u.erg/u.s/u.cm**2/(u.nm*0.1)).value # erg/s/cm2/A
	if (model == 'LB23'):
		spec_model = ascii.read(spectrum_name_full)
		wl_model = spec_model['LAMBDA(mic)'] # micron
		flux_model = spec_model['FLAM'] # erg/s/cm2/A
		# convert scientific notation from 'D' to 'E'
		wl_LB23 = np.zeros(wl_model.size)
		flux_LB23 = np.zeros(wl_model.size)
		for j in range(wl_LB23.size):
			wl_LB23[j] = float(wl_model[j].replace('D', 'E'))
			flux_LB23[j] = float(flux_model[j].replace('D', 'E'))
		wl_model = wl_LB23 # um
		flux_model = flux_LB23 # erg/s/cm2/A
	if (model == 'Sonora_Cholla'):
		spec_model = ascii.read(spectrum_name_full, data_start=2, format='no_header')
		wl_model = spec_model['col1'] * u.micron # um (in vacuum?)
		wl_model = vac_to_air(wl_model).value # um in the air
		flux_model = spec_model['col2'] * u.W/u.m**2/u.m # W/m2/m
		flux_model = flux_model.to(u.erg/u.s/u.cm**2/(u.nm*0.1)).value # erg/s/cm2/A
	if (model == 'Sonora_Bobcat'):
		spec_model = ascii.read(spectrum_name_full, data_start=2, format='no_header')
		wl_model = spec_model['col1'] * u.micron # um (in vacuum?)
		wl_model = vac_to_air(wl_model).value # um in the air
		flux_model = spec_model['col2'] * u.erg/u.s/u.cm**2/u.Hz # erg/s/cm2/Hz
		flux_model = flux_model.to(u.erg/u.s/u.cm**2/(u.nm*0.1), equivalencies=u.spectral_density( wl_model * u.micron)).value # erg/s/cm2/A
	if (model == 'ATMO2020'):
		spec_model = ascii.read(spectrum_name_full, format='no_header')
		wl_model = spec_model['col1'] * u.micron # um (in vacuum)
		wl_model = vac_to_air(wl_model).value # um in the air
		flux_model = spec_model['col2'] * u.W/u.m**2/u.micron # W/m2/micron
		flux_model = flux_model.to(u.erg/u.s/u.cm**2/(u.nm*0.1)).value # erg/s/cm2/A
	if (model == 'BT-Settl'):
		spec_model = ascii.read(spectrum_name_full, format='no_header')
		wl_model = (spec_model['col1']*(u.nm*0.1)).to(u.micron) # um (in vacuum)
		wl_model = vac_to_air(wl_model).value # um in the air
		flux_model = spec_model['col2'] * u.erg/u.s/u.cm**2/u.Hz # erg/s/cm2/Hz (to an unknown distance). 10**(F_lam + DF) to convert to erg/s/cm2/A
		DF= -8.0
		flux_model = 10**(flux_model.value + DF) # erg/s/cm2/A
	if (model == 'SM08'):
		spec_model = ascii.read(spectrum_name_full, data_start=2, format='no_header')
		wl_model = spec_model['col1'] * u.micron # um (in air the alkali lines and in vacuum the rest of the spectra)
		#wl_model = vac_to_air(wl_model).value # um in the air
		wl_model = wl_model.value # um
		flux_model = spec_model['col2'] * u.erg/u.s/u.cm**2/u.Hz # erg/s/cm2/Hz (to an unknown distance)
		flux_model = flux_model.to(u.erg/u.s/u.cm**2/(u.nm*0.1), equivalencies=u.spectral_density(wl_model*u.micron)).value # erg/s/cm2/A

	# sort the array. For BT-Settl is recommended by Allard in her webpage and some models are sorted from higher to smaller wavelengths.
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
# round logg to steps of 0.25
def round_logg_point25(logg):
	logg = round(logg*4.) / 4. 
	return logg

