import numpy as np
from astropy import units as u
from .utils import *
from .synthetic_photometry.synthetic_photometry import *
from .models import *
from sys import exit

print('\n    SEDA package imported')

#+++++++++++++++++++++++++++
class InputData:
	'''
	Description:
	------------
		Define input data.
	
	Parameters:
	-----------
	- fit_spectra : {``True``, ``False``}, optional (default ``True``)
		Include (``True``) or do not include (``False``) spectra.
	- fit_photometry : {``True``, ``False``}, optional (default ``False``)
		Include (``True``) or do not include (``False``) photometry.
	- wl_spectra : float array or list, optional (required if ``fit_spectra``)
		Wavelength in um of the spectra for model comparisons.
		For multiple spectra, provide them as a list (e.g., ``wl_spectra = [wl_spectrum1, wl_spectrum2]``).
	- flux_spectra : float array or list, optional (required if ``fit_spectra``)
		Fluxes for the input spectra in units indicated by ``flux_unit``.
		Use a list for multiple spectra (similar to ``wl_spectra``).
	- eflux_spectra : float array or list, optional (required if ``fit_spectra``)
		Fluxes uncertainties in erg/cm^2/s/A of the input spectra. 
		Use a list for multiple spectra (similar to ``wl_spectra``).
	- flux_unit : str, optional (default ``'erg/s/cm2/A'``)
		Units of ``flux``: ``'Jy'``, ``'erg/s/cm2/A'``, or ``erg/s/cm2/um``.
	- mag_phot : float array, optional (required if ``fit_photometry``)
		Magnitudes for the fit
	- emag_phot : float array, optional (required if ``fit_photometry``)
		Magnitude uncertainties for the fit. Magnitudes with uncertainties equal to zero are excluded from the fit.
	- filter_phot : float array, optional (required if ``fit_photometry``)
		Filters associated to the input magnitudes following SVO filter IDs 
		http://svo2.cab.inta-csic.es/theory/fps/
	- res : float, list or array, optional (required if ``fit_spectra``)
		Resolving power (R=lambda/delta(lambda) at ``lam_res``) of input spectra to smooth model spectra.
		For multiple input spectra, ``res`` should be a list or array with a value for each spectrum.
	- lam_res : float, list or array, optional
		Wavelength of reference at which ``res`` is given (because resolution may change with wavelength).
		For multiple input spectra, ``lam_res`` should be a list or array with a value for each spectrum.
		Default is the integer closest to the median wavelength for each input spectrum.
	- distance : float, optional
		Target distance (in pc) used to derive radii from scaling factors for models.
	- edistance : float, optional
		Distance error (in pc).

	Returns:
	--------
	- Dictionary with all (provided and default) input data parameters.

	Example:
	--------
	>>> import seda
	>>> 
	>>> # input spectrum wl_input, flux_input, eflux_input
	>>> wl_spectra = wl_input # in um
	>>> flux_spectra = flux_input # in erg/cm^2/s/A
	>>> eflux_spectra = eflux_input # in erg/cm^2/s/A
	>>> res = 100 # input spectrum resolution
	>>> my_data = seda.InputData(wl_spectra=wl_spectra, flux_spectra=flux_spectra, 
	>>>                          eflux_spectra=eflux_spectra, res=res)
	    Input data loaded successfully

	Author: Genaro Suárez
	'''

	def __init__(self, fit_spectra=True, fit_photometry=False, wl_spectra=None, 
	    flux_spectra=None, eflux_spectra=None, flux_unit=None, 
	    res=None, lam_res=None, mag_phot=None, emag_phot=None, filter_phot=None, 
	    distance=None, edistance=None):	

		self.fit_spectra = fit_spectra
		self.fit_photometry = fit_photometry
		self.mag_phot = mag_phot
		self.emag_phot = emag_phot
		self.filter_phot = filter_phot
		self.distance = distance
		self.edistance = edistance

		# number of input spectra
		if isinstance(wl_spectra, list): # when multiple spectra are provided
			N_spectra = len(wl_spectra) # number of input spectra
		else: # when only one spectrum is provided
			N_spectra = 1 # number of input spectra

		self.N_spectra = N_spectra

		# verify that all the mandatory parameters are given
		# reshape some input parameters and define non-provided parameters in terms of other parameters
		if fit_spectra:

			# when one spectrum is given, convert the spectrum into a list
			# for wavelenghts
			if wl_spectra is None: # if not provided
				raise Exception(f'Parameter "wl_spectra" is missing.')
			else: # if provided
				if not isinstance(wl_spectra, list): wl_spectra = [wl_spectra]
			# for fluxes
			if flux_spectra is None: # if not provided
				raise Exception(f'Parameter "flux_spectra" is missing.')
			else: # if provided
				if not isinstance(flux_spectra, list): flux_spectra = [flux_spectra]
			# for flux errors
			if eflux_spectra is None: # if not provided
				raise Exception(f'Parameter "eflux_spectra" is missing.')
			else: # if provided
				if not isinstance(eflux_spectra, list): eflux_spectra = [eflux_spectra]

			# if res is a scalar, convert it into an array
			if res is None: # if not provided
				raise Exception(f'Parameter "res" is missing.')
			else: # if provided
				if isinstance(res, (float, int)): res = np.array([res])

			# if lam_res is a scalar, convert it into an array
			if isinstance(lam_res, (float, int)): lam_res = np.array([lam_res])

			# set lam_res if not provided
			if lam_res is None: 
				lam_res = []
				for wl_spectrum in wl_spectra:
					lam_res.append(set_lam_res(wl_spectrum))

		# handle input parameters
		# convert input spectra to numpy arrays, if they are astropy
		for i in range(N_spectra):
			wl_spectra[i] = astropy_to_numpy(wl_spectra[i])
			flux_spectra[i] = astropy_to_numpy(flux_spectra[i])
			eflux_spectra[i] = astropy_to_numpy(eflux_spectra[i])
		# remove NaN values
		for i in range(N_spectra):
			mask_nonan = (~np.isnan(wl_spectra[i])) & (~np.isnan(flux_spectra[i])) & (~np.isnan(eflux_spectra[i]))
			wl_spectra[i] = wl_spectra[i][mask_nonan]
			flux_spectra[i] = flux_spectra[i][mask_nonan]
			eflux_spectra[i] = eflux_spectra[i][mask_nonan]
		# remove negative fluxes
		for i in range(N_spectra):
			mask_noneg = flux_spectra[i]>0
			wl_spectra[i] = wl_spectra[i][mask_noneg]
			flux_spectra[i] = flux_spectra[i][mask_noneg]
			eflux_spectra[i] = eflux_spectra[i][mask_noneg]

		# set default flux_unit
		if flux_unit is None: flux_unit='erg/s/cm2/A'
		# convert input fluxes to erg/s/cm2/A, if need
		if flux_unit=='Jy': # if fluxes are provided in Jy
			for i in range(N_spectra):
				out_convert_flux = convert_flux(flux=flux_spectra[i], eflux=eflux_spectra[i], 
				                                wl=wl_spectra[i], unit_in='Jy', unit_out='erg/s/cm2/A')
				flux_spectra[i] = out_convert_flux['flux_out']
				eflux_spectra[i] = out_convert_flux['eflux_out']
		if flux_unit=='erg/s/cm2/um': # if fluxes are provided in erg/s/cm2/um
			for i in range(N_spectra):
				flux_spectra[i] = (flux_spectra[i]*u.erg/u.s/u.cm**2/u.micron).to(u.erg/u.s/u.cm**2/u.angstrom).value # erg/s/cm2/A
				eflux_spectra[i] = (eflux_spectra[i]*u.erg/u.s/u.cm**2/u.micron).to(u.erg/u.s/u.cm**2/u.angstrom).value # erg/s/cm2/A

		self.res = res
		self.lam_res = lam_res
		self.wl_spectra = wl_spectra
		self.flux_spectra = flux_spectra
		self.eflux_spectra = eflux_spectra

		print('\n   Input data loaded successfully')

#+++++++++++++++++++++++++++
class ModelOptions:
	'''
	Description:
	------------
		Define model options.

	Parameters:
	-----------
	- model : str
		Label for any of the available atmospheric models.
		See more info in ``seda.Models``.
	- model_dir : str, list, or array
		Path to the directory (str, list, or array) or directories (as a list or array) containing the model spectra (e.g., ``model_dir = ['path_1', 'path_2']``). 
		Avoid using paths with null spaces. 
	- params_ranges : dictionary, optional
		Minimum and maximum values for any free model parameters to select a model grid subset.
		E.g., ``params_ranges = {'Teff': [1000, 1200], 'logg': [4., 5.]}`` to consider spectra within those Teff and logg ranges.
		If a parameter range is not provided, the full range in ``model_dir`` is considered.
	- path_save_spectra_conv: str, optional
		Directory path to store convolved model spectra. 
		If not provided (default), the convolved spectra will not be saved. 
		If the directory does not exist, it will be created. Otherwise, the spectra will be added to the existing folder.
		The convolved spectra will keep the same original names along with the ``res`` and ``lam_res`` parameters, e.g. 'original_spectrum_name_R100at1um.nc' for ``res=100`` and ``lam_res=1``.
		They will be saved as netCDF with xarray (it produces lighter files compared to normal ASCII files).
	- skip_convolution : {``True``, ``False``}, optional (default ``False``)
		Convolution of model spectra (the slowest process in the code) can (``True``) or cannot (``False``) be avoided. 
		Once the code has be run and the convolved spectra were stored in ``path_save_spectra_conv``, the convolved grid can be reused for other input data with the same resolution as the convolved spectra.
		If 'True', ``model_dir`` should include the previously convolved spectra for ``res`` at ``lam_res`` in ``input_parameters.InputData``. 

	Returns:
	--------
	Dictionary with all (provided and default) model option parameters plus the following parameters:
		- ...

	Example:
	--------
	>>> import seda
	>>> 
	>>> # models
	>>> model = 'Sonora_Elf_Owl'
	>>> model_dir = ['my_path/output_700.0_800.0/', 
	>>>              'my_path/output_850.0_950.0/'] # folders to seek model spectra
	>>> 
	>>> # some parameter ranges to read a model grid subset
	>>> params_ranges = {'Teff': [700, 900, 'logg': [4.0, 5.0]
	>>> 
	>>> # load model options
	>>> my_model = seda.ModelOptions(model=model, model_dir=model_dir, 
	>>>                              params_ranges=params_ranges)
	    Model options loaded successfully

	Author: Genaro Suárez
	'''

	def __init__(self, model, model_dir, params_ranges=None,
	             path_save_spectra_conv=None, skip_convolution=False):

		self.model = model
		if model not in Models().available_models: raise Exception(f'Models "{model}" are not recognized. Available models are: \n          {models_valid}')
		self.params_ranges = params_ranges
		self.path_save_spectra_conv = path_save_spectra_conv
		self.skip_convolution = skip_convolution

		# when only one directory with models is given
		if not isinstance(model_dir, (list, np.ndarray)): model_dir = [model_dir]
		self.model_dir = model_dir

		print('\n   Model options loaded successfully')

#+++++++++++++++++++++++++++
class Chi2Options:
	'''
	Description:
	------------
		Define chi-square fit options.

	Parameters:
	-----------
	- my_data : dictionary
		Output dictionary by ``input_parameters.InputData`` with input data.
	- my_model : dictionary
		Output dictionary by ``input_parameters.ModelOptions`` with model options.
	- fit_wl_range : float array, optional
		Minimum and maximum wavelengths (in microns) where each input spectrum will be compared to the models. E.g., ``fit_wl_range = np.array([fit_wl_min1, fit_wl_max1], [fit_wl_min2, fit_wl_max2])``. 
		This parameter is used if ``fit_spectra`` but ignored if only ``fit_photometry``. 
		Default values are the minimum and the maximum wavelengths of each input spectrum.
	- model_wl_range : array or list, optional
		Minimum and maximum wavelength (in microns) to cut model spectra to keep only wavelengths of interest.
		Default values are the minimum and maximum wavelengths covered by the input spectra with a padding to avoid the point below.
		CAVEAT: the selected wavelength range of model spectra must cover the spectrophotometry used in the fit and a bit more (to avoid errors when resampling synthetic spectra using spectres).
	- extinction_free_param : {``'yes'``, ``'no'``}, optional (default ``'no'``)
		Extinction as a free parameter: 
			- ``'no'``: null extinction is assumed and it will not change.
			- ``'yes'``: null extinction is assumed and it varies to minimize chi square.
	- scaling_free_param : {``'yes'``, ``'no'``}, optional (default ``'yes'``)
		Scaling as a free parameter: 
			- ``'yes'``: to find the scaling that minimizes chi square for each model
			- ``'no'``: to fix ``scaling`` if radius and distance are known
	- scaling: float, optional (required if ``scaling_free_param='no'``)
		Fixed scaling factor ((R/d)^2, R: object's radius, d: distance to the object) to be applied to model spectra
	- avoid_IR_excess : {``'yes'``, ``'no'``}, optional (default ``'no'``)
		Wavelengths longer than ``IR_excess_limit`` will (``'yes'``) or will not (``'no'``) be avoided in the fit in case infrared excesses are expected. 
	- IR_excess_limit : float, optional (default 3 um).
		Shortest wavelength at which IR excesses are expected.
	- save_results : {``True``, ``False``}, optional (default ``True``)
		Save (``True``) or do not save (``False``) ``seda.chi2_fit`` results
	- chi2_pickle_file : str, optional
		Filename for the output dictionary stored as a pickle file, if ``save_results``.
		Default name is '``model``\_chi2\_minimization.pickle'.
	- chi2_table_file : str, optional
		Filename for an output ascii table (if ``save_results``) with relevant information from the fit.
		Default name is '``model``\_chi2\_minimization.dat'.

	Returns:
	--------
	- Dictionary with all (provided and default) chi-square fit option parameters.

	Example:
	--------
	>>> import seda
	>>> 
	>>> # input spectrum wl_input, flux_input, eflux_input
	>>> fit_wl_range = np.array([value1, value2]) # to make the fit between value1 and value2
	>>> my_chi2 = seda.Chi2FitOptions(my_data=my_data, my_model=my_model, 
	>>>                               fit_wl_range=fit_wl_range)
	    Chi2 fit options loaded successfully

	Author: Genaro Suárez
	'''

	def __init__(self, my_data, my_model, 
		fit_wl_range=None, model_wl_range=None, extinction_free_param='no', 
		scaling_free_param='yes', scaling=None, 
		avoid_IR_excess='no', IR_excess_limit=3, save_results=True,
		chi2_pickle_file=None, chi2_table_file=None):

		self.save_results = save_results
		self.scaling_free_param = scaling_free_param
		self.scaling = scaling
		self.extinction_free_param = extinction_free_param
		self.avoid_IR_excess = avoid_IR_excess
		self.IR_excess_limit = IR_excess_limit

		# read parameters from InputData
		self.fit_spectra = my_data.fit_spectra
		self.fit_photometry = my_data.fit_photometry
		self.mag_phot = my_data.mag_phot
		self.emag_phot = my_data.emag_phot
		self.filter_phot = my_data.filter_phot
		self.distance = my_data.distance
		self.edistance = my_data.edistance
		self.N_spectra = my_data.N_spectra
		self.res = my_data.res
		self.lam_res = my_data.lam_res
		self.wl_spectra = my_data.wl_spectra
		self.flux_spectra = my_data.flux_spectra
		self.eflux_spectra = my_data.eflux_spectra

		# read parameters from ModelOptions
		self.model = my_model.model
		self.params_ranges = my_model.params_ranges
		self.path_save_spectra_conv = my_model.path_save_spectra_conv
		self.skip_convolution = my_model.skip_convolution
		self.model_dir = my_model.model_dir

		# extract parameters for convenience
		N_spectra = my_data.N_spectra
		wl_spectra = my_data.wl_spectra
		res = my_data.res
		lam_res = my_data.lam_res
		model = my_model.model
		model_dir = my_model.model_dir
		params_ranges = my_model.params_ranges
		skip_convolution = my_model.skip_convolution

		# number of data points in the input spectra
		out_input_data_stats = input_data_stats(wl_spectra=wl_spectra)
		wl_spectra_min = out_input_data_stats['wl_spectra_min']
		wl_spectra_max = out_input_data_stats['wl_spectra_max']
		N_datapoints = out_input_data_stats['N_datapoints']

		# handle fit_wl_range
		fit_wl_range = set_fit_wl_range(fit_wl_range=fit_wl_range, N_spectra=N_spectra, wl_spectra=wl_spectra)

		# handle model_wl_range
		model_wl_range = set_model_wl_range(model_wl_range=model_wl_range, wl_spectra_min=wl_spectra_min, wl_spectra_max=wl_spectra_max)

		self.fit_wl_range = fit_wl_range
		self.model_wl_range = model_wl_range
		self.wl_spectra_min = wl_spectra_min
		self.wl_spectra_max = wl_spectra_max
		self.N_datapoints = N_datapoints

		# file name to save the chi2 results as a pickle
		if chi2_pickle_file is None: self.chi2_pickle_file = f'{model}_chi2_minimization.pickle'
		else: self.chi2_pickle_file = chi2_pickle_file
		if chi2_table_file is None: self.chi2_table_file = f'{model}_chi2_minimization.dat'
		else: self.chi2_table_file = chi2_table_file


#		# initialize lists for resampled, convolved model spectra for all input spectra
#		wl_array_model_conv_resam = []
#		flux_array_model_conv_resam = []
#		for k in range(N_spectra): # for each input observed spectrum
#			print(f'\nFor input spectrum {k+1} of {N_spectra}')
#
#			# select files with spectra within the indicated ranges
#			if not skip_convolution: # read and convolve original model spectra
#				out_select_model_spectra = select_model_spectra(model=model, model_dir=model_dir, params_ranges=params_ranges)
#			else: # read model spectra already convolved to the data resolution
#				# set filename_pattern to look for model spectra with the corresponding resolution
#				filename_pattern = Models(model).filename_pattern+f'_R{res[k]}at{lam_res[k]}um.nc'
#				out_select_model_spectra = select_model_spectra(model=model, model_dir=model_dir, params_ranges=params_ranges, filename_pattern=filename_pattern)
#			spectra_name_full = out_select_model_spectra['spectra_name_full']
#			spectra_name = out_select_model_spectra['spectra_name']
#			N_model_spectra = len(spectra_name) # number of selected model spectra
#
#			# create a tqdm progress bar
#			if not skip_convolution: # read original model spectra
#				model_bar = tqdm(total=N_model_spectra, desc='Reading, convolving, and resampling model spectra')
#			else:
#				model_bar = tqdm(total=N_model_spectra, desc='Reading and resampling model spectra')
#
#			# list to save a resampled and convolved model spectrum for the input spectra
#			wl_array_model_conv_resam_each = []
#			flux_array_model_conv_resam_each = []
#			for i in range(N_model_spectra): # for each model spectrum
#				# update progress bar
#				model_bar.update(1)
#		
#				# read and convolve model spectra, if needed
#				if not skip_convolution: # read original model spectra
#					out_read_model_spectrum = read_model_spectrum(spectrum_name_full=spectra_name_full[i], model=model, model_wl_range=model_wl_range)
#				else: # read model spectra already convolved to the data resolution
#					out_read_model_spectrum = read_model_spectrum_conv(spectrum_name_full=spectra_name_full[i], model_wl_range=model_wl_range)
#				wl_model = out_read_model_spectrum['wl_model'] # um
#				flux_model = out_read_model_spectrum['flux_model'] # erg/s/cm2/A
#		
#				# convolve the model spectrum to the indicated resolution
#				wl_array_model_conv_each = [] # to save each convolved spectrum 
#				flux_array_model_conv_each = [] # to save each convolved spectrum 
#				if not skip_convolution: # avoid convolution, even if convolve was left as True, when skip_convolution is True
#					# convolve models in the full wavelength range of each input spectrum plus a padding
#					convolve_wl_range = [0.9*wl_spectra[k].min(), 1.1*wl_spectra[k].max()]
#			
#					# convolve model spectrum according to the resolution and fit range of each input spectrum
#					if path_save_spectra_conv is None: # do not save the convolved spectrum
#						out_convolve_spectrum = convolve_spectrum(wl=wl_model, flux=flux_model, lam_res=lam_res[k], res=res[k], 
#						                                          disp_wl_range=fit_wl_range[k], convolve_wl_range=convolve_wl_range)
#					else: # save convolved spectrum
#						if not os.path.exists(path_save_spectra_conv): os.makedirs(path_save_spectra_conv) # make directory (if not existing) to store convolved spectra
#						out_file = path_save_spectra_conv+spectra_name[i]+f'_R{res[k]}at{lam_res[k]}um.nc'
#						out_convolve_spectrum = convolve_spectrum(wl=wl_model, flux=flux_model, lam_res=lam_res[k], res=res[k], 
#						                                          disp_wl_range=fit_wl_range[k], convolve_wl_range=convolve_wl_range, out_file=out_file)
#					
#					out_convolve_spectrum = convolve_spectrum(wl=wl_model, flux=flux_model, res=res, lam_res=lam_res)
#					wl_model = out_convolve_spectrum['wl_conv']
#					flux_model = out_convolve_spectrum['flux_conv']
#
#				# resample the convolved model spectrum to the wavelength data points in the observed spectra
#				mask = (wl_spectra[k]>wl_model.min()) & (wl_spectra[k]<wl_model.max()) # input wavelengths within model wavelength coverage
#				flux_array_model_conv_resam_each.append(spectres(wl_spectra[k][mask], wl_model, flux_model)) # resampled fluxes
#				wl_array_model_conv_resam_each.append(wl_spectra[k][mask]) # wavelengths for resampled fluxes
#
#			# nested list with all resampled and convolved model spectra
#			wl_array_model_conv_resam.append(wl_array_model_conv_resam_each)
#			flux_array_model_conv_resam.append(flux_array_model_conv_resam_each)
#						
#		# close progress bar
#		model_bar.close()
#
#		self.wl_array_model_conv_resam = wl_array_model_conv_resam
#		self.flux_array_model_conv_resam = flux_array_model_conv_resam
#
#		# when reading pre-computed convolved model spectra, remove the extended name added when saving the convolved models
#		# spectra_name in memory are the ones for the last input spectrum, but they have the same root name for all input spectra
#		spectra_name_ori = []
#		for spectrum in spectra_name: # for each selected model spectrum
#			name = spectrum.split('_R')[0]
#			if name not in spectra_name_ori: # to avoid repeating the same model spectrum when inputting multiple observed spectra
#				spectra_name_ori.append(name)
#		spectra_name_full_ori = [model_dir[0]+spectrum for spectrum in spectra_name_ori] # spectra files with full path
#		# renaming
#		spectra_name = np.array(spectra_name_ori)
#		spectra_name_full = np.array(spectra_name_full_ori)
#		N_model_spectra = len(spectra_name)
#
#		self.spectra_name = spectra_name
#		self.spectra_name_full = spectra_name_full
#		self.N_model_spectra = N_model_spectra

		print('\n   Chi-square fit options loaded successfully')

#+++++++++++++++++++++++++++
class BayesOptions:
	'''
	Description:
	------------
		Define Bayesian sampling options.

	Parameters:
	-----------
	- my_data : dictionary
		Output dictionary by ``input_parameters.InputData`` with input data.
	- my_model : dictionary
		Output dictionary by ``input_parameters.ModelOptions`` with model options.
	- fit_wl_range : float array, optional
		Minimum and maximum wavelengths (in microns) where model spectra will be compared to the data. 
		This parameter is used if ``fit_spectra`` but ignored if only ``fit_photometry``. 
		Default values are the minimum and the maximum wavelengths of each input spectrum. E.g., ``fit_wl_range = np.array([bayes_wl_min, bayes_wl_max])``. 
	- model_wl_range : array or list, optional
		Minimum and maximum wavelength (in microns) to cut model spectra (to make the code faster). 
		Default values are the same as ``fit_wl_range`` with a padding to avoid the point below.
		CAVEAT: the selected wavelength range of model spectra must cover the spectrophotometry used in the fit and a bit more (to avoid errors when resampling synthetic spectra using spectres)
	- R_range: float array, optional (used in ``bayes_fit``)
		Minimum and maximum radius values to sample the posterior for radius. It requires the parameter ``distance`` in `input_parameters.InputData`.
	- 'chi2_pickle_file' : dictionary, optional
		Pickle file with a dictionary with the results from the chi-square minimization by ``chi2_fit.chi2``.
		If given, the grid nodes around the best fit will be used as ``params_ranges`` to select a grid subset and as sampling priors.
		Otherwise, ``params_ranges`` provided in ``ModelOptions`` will be considered.
	- grid: dictionary, optional
		Model grid (``'wavelength'`` and ``'flux'``) generated by ``utils.read_grid`` to interpolate model spectra. 
		If not provided (default), then the grid is read (``model`` and ``model_dir`` must be provided). 
		If provided, the code will skip reading the grid, which will save some time (a few minutes).
	- dynamic_sampling: {``True``, ``False``}, optional (default ``True``). 
		Consider dynamic (``True``) or static (``False``) nested sampling. Read ``dynesty`` documentation for more info. 
		Dynamic nested sampling is slower (~20-30%) than static nested sampling. 
	- nlive: float, optional (default 500). 
		Number of nested sampling live points. 
	- save_results : {``True``, ``False``}, optional (default ``True``)
		Save (``True``) or do not save (``False``) ``seda.bayes_fit`` results
	- bayes_pickle_file : str, optional
		Filename for the output dictionary stored as a pickle file, if ``save_results``.
		Default name is '``model``\_bayesian\_sampling.pickle'.

	Returns:
	--------
	- Dictionary with all (provided and default) Bayes fit option parameters.

	Example:
	--------
	>>> import seda
	>>> 
	>>> fit_wl_range = np.array([value1, value2]) # to make the fit between value1 and value2
	>>> my_bayes = seda.Chi2FitOptions(my_data=my_data, my_model=my_model, 
	>>>                               fit_wl_range=fit_wl_range)
	    Bayes fit options loaded successfully

	Author: Genaro Suárez
	'''

	def __init__(self, my_data, my_model, fit_wl_range=None, model_wl_range=None, 
	             R_range=None, chi2_pickle_file=None, bayes_pickle_file=None,
		         grid=None, dynamic_sampling=True, nlive=500, save_results=True):

		self.R_range = R_range
		self.chi2_pickle_file = chi2_pickle_file
		self.dynamic_sampling = dynamic_sampling
		self.nlive = nlive
		self.save_results = save_results

		# read parameters from InputData
		self.fit_spectra = my_data.fit_spectra
		self.fit_photometry = my_data.fit_photometry
		self.mag_phot = my_data.mag_phot
		self.emag_phot = my_data.emag_phot
		self.filter_phot = my_data.filter_phot
		self.distance = my_data.distance
		self.edistance = my_data.edistance
		self.N_spectra = my_data.N_spectra
		self.res = my_data.res
		self.lam_res = my_data.lam_res
		self.wl_spectra = my_data.wl_spectra
		self.flux_spectra = my_data.flux_spectra
		self.eflux_spectra = my_data.eflux_spectra

		# read parameters from ModelOptions
		self.model = my_model.model
		self.model_dir = my_model.model_dir
		self.params_ranges = my_model.params_ranges
		self.path_save_spectra_conv = my_model.path_save_spectra_conv
		self.skip_convolution = my_model.skip_convolution

		# extract parameters for convenience
		N_spectra = my_data.N_spectra
		wl_spectra = my_data.wl_spectra
		distance = my_data.distance
		res = my_data.res
		lam_res = my_data.lam_res
		model = my_model.model
		model_dir = my_model.model_dir
		params_ranges = my_model.params_ranges
		path_save_spectra_conv = my_model.path_save_spectra_conv
		skip_convolution = my_model.skip_convolution

		# some statistic for input spectra
		out_input_data_stats = input_data_stats(wl_spectra=wl_spectra)
		wl_spectra_min = out_input_data_stats['wl_spectra_min'] # minimum wavelength in the input spectra
		wl_spectra_max = out_input_data_stats['wl_spectra_max'] # maximum wavelength in the input spectra
		N_datapoints = out_input_data_stats['N_datapoints'] # number of data points in all input spectra

		# handle fit_wl_range
		fit_wl_range = set_fit_wl_range(fit_wl_range=fit_wl_range, N_spectra=N_spectra, wl_spectra=wl_spectra)

		# handle model_wl_range
		model_wl_range = set_model_wl_range(model_wl_range=model_wl_range, wl_spectra_min=wl_spectra_min, wl_spectra_max=wl_spectra_max)

		self.fit_wl_range = fit_wl_range
		self.model_wl_range = model_wl_range
		self.wl_spectra_min = out_input_data_stats['wl_spectra_min']
		self.wl_spectra_max = out_input_data_stats['wl_spectra_max']
		self.N_datapoints = out_input_data_stats['N_datapoints']

		# file name to save the bayes results as a pickle
		if bayes_pickle_file is None: self.bayes_pickle_file = f'{model}_bayesian_sampling.pickle'
		else: self.bayes_pickle_file = bayes_pickle_file

		# parameter ranges to select the grid and then used for the posterior priors
		if chi2_pickle_file:
			# params for the best chi-square fit
			params = best_chi2_fits(chi2_pickle_file, N_best_fits=1)['params']

			# grid values around the best fit (which is also a grid node)
			params_models = Models(model).params_unique # round median values for model grid parameters
			params_ranges = {}
			for param in params_models: # for each free parameter in the grid
				params_ranges[param] = find_two_around_node(params_models[param], params[param])
			
		# define the dimensionality of our problem
		if distance is not None: # sample radius distribution
			ndim = len(Models(model).free_params) + 1
		else:
			ndim = len(Models(model).free_params)
		self.ndim = ndim

		# read model grid
		if grid is None: 
			# initialize list to save the resampled, convolved grid appropriate for each input spectrum
			grid = []
			for i in range(N_spectra): # for each input observed spectrum
				# convolve the grid to the given res and lam_res for each input spectrum
				# resample the convolved grid to the input wavelengths within the fit range for each input spectrum
				print(f'\nFor input spectrum {i+1} of {N_spectra}')
				if not skip_convolution: # read and convolve original model spectra
					grid_each = read_grid(model=model, model_dir=model_dir, params_ranges=params_ranges, 
					                 convolve=True, res=res[i], lam_res=lam_res[i], 
					                 fit_wl_range=fit_wl_range[i], wl_resample=wl_spectra[i],
					                 path_save_spectra_conv=path_save_spectra_conv)
				else: # read model spectra already convolved to the data resolution
					# set filename_pattern to look for model spectra with the corresponding resolution
					filename_pattern = Models(model).filename_pattern+f'_R{res[i]}at{lam_res[i]}um.nc'
					grid_each = read_grid(model=model, model_dir=model_dir, params_ranges=params_ranges, 
					                 res=res[i], lam_res=lam_res[i], 
					                 fit_wl_range=fit_wl_range[i], wl_resample=wl_spectra[i], 
					                 skip_convolution=skip_convolution, filename_pattern=filename_pattern)
					
				# add resampled grid for each input spectrum to the same list
				grid.append(grid_each)
		self.grid = grid

		# unique values for each model free parameter
		params_unique = grid[0]['params_unique']
		self.params_unique = params_unique

		# define priors from input parameters or grid ranges
		params_priors = {}
		for param in params_unique:
			params_priors[param] = np.array([params_unique[param].min(), params_unique[param].max()])

		if distance is not None:
			if R_range is None: raise Exception('Please provide the "R_range" parameter')
			params_priors['R'] = R_range

		self.params_priors = params_priors

		print('\n   Bayes fit options loaded successfully')
