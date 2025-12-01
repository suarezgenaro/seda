import numpy as np
from astropy import units as u
import os
import re
from tqdm.auto import tqdm 
from spectres import spectres
from . import utils
from . import models
from .synthetic_photometry import synthetic_photometry
from sys import exit

#++++++++++++++++++++++++++++++++
# message when importing the code
#++++++++++++++++++++++++++++++++
# read code version
dir_path = os.path.dirname(os.path.realpath(__file__))
version_string = open(os.path.join(dir_path, '_version.py')).read()
VERS = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VERS, version_string, re.M)
if mo:
	__version__ = mo.group(1)
else:
	raise RuntimeError("Unable to find version string in %s." % (version_string,))

print(f'\n    SEDA v{__version__} package imported')

#++++++++++++++++++++++++++++++++
# module-level constants
#++++++++++++++++++++++++++++++++
allowed_weights = {"none", "dataset", "width"}

#++++++++++++++++++++++++++++++++
# main classes
#++++++++++++++++++++++++++++++++
class InputData:
	'''
	Description:
	------------
		Define input data for fitting (spectra and/or photometry).
	
	Parameters:
	-----------
	- fit_spectra : {``True``, ``False``}, optional (default ``True``)
		Include (``True``) or do not include (``False``) spectra.
	- fit_photometry : {``True``, ``False``}, optional (default ``False``)
		Include (``True``) or do not include (``False``) photometry.
	- wl_spectra : float array or list, optional (required if ``fit_spectra``)
		Wavelength in micron of the spectra for model comparisons.
		For multiple spectra, provide them as a list (e.g., ``wl_spectra = [wl_spectrum1, wl_spectrum2]``).
	- flux_spectra : float array or list, optional (required if ``fit_spectra``)
		Fluxes for the input spectra in units indicated by ``flux_unit``.
		Use a list for multiple spectra (similar to ``wl_spectra``).
	- eflux_spectra : float array or list, optional (required if ``fit_spectra``)
		Fluxes uncertainties in units indicated by ``flux_unit``.
		Use a list for multiple spectra (similar to ``wl_spectra``).
	- flux_unit : str, optional (default ``'erg/s/cm2/A'``)
		Units of ``flux_spectra`` and ``eflux_spectra``: ``'Jy'``, ``'erg/s/cm2/A'``, or ``erg/s/cm2/um``.
	- phot : float array, optional (required if ``fit_photometry``)
		Magnitudes or photometric fluxes for the fit
	- ephot : float array, optional (required if ``fit_photometry``)
		Magnitude or photometric flux uncertainties for the fit. Magnitudes or fluxes with uncertainties equal to zero are excluded from the fit.
	- filters : float array, optional (required if ``fit_photometry``)
		Filters associated to the input photometry following SVO filter IDs 
		http://svo2.cab.inta-csic.es/theory/fps/
	- phot_unit : str, optional (default ``'mag'``)
		Units of ``phot`` and ``ephot``: ``'mag'``, ``'Jy'``, ``'erg/s/cm2/A'``, or ``erg/s/cm2/um``.
	- res : float, list or array, optional (required if ``fit_spectra``)
		Resolving power (R=lambda/delta(lambda) at ``lam_res``) of input spectra to smooth model spectra.
		For multiple input spectra, ``res`` should be a list or array with a value for each spectrum.
	- lam_res : float, list or array, optional
		Wavelength of reference at which ``res`` is given (because resolution may change with wavelength).
		For multiple input spectra, ``lam_res`` should be a list or array with a value for each spectrum.
		Default is the integer closest to the median wavelength for each input spectrum.
		If lam_res is provided, the values are also rounded to the nearest integer.
		This will facilitate managing (saving and reading) convolved model spectra in ``seda.ModelOptions``.
	- distance : float, optional
		Target distance (in pc) used to derive radii from scaling factors for models.
	- edistance : float, optional
		Distance error (in pc).

	Attributes
	----------
	Input parameters.
		All input parameters are stored as attributes with their default values if not specified.
		Units:
		- ``flux_spectra``, ``eflux_spectra``, ``Phot``, and ``ephot`` attributes are stored in erg/s/cm2/A.
	Other parameters
		N_spectra : int
			Number of input spectra (if ``fit_spectra``).
		lambda_eff_SVO : float array
			Effective wavelengths from the SVO for filters (if ``fit_photometry``).
		width_eff_SVO : float array
			Effective widths from the SVO for filters (if ``fit_photometry``).
	
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
	    res=None, lam_res=None, phot=None, ephot=None, filters=None, phot_unit=None,
	    distance=None, edistance=None):	

		self.fit_spectra = fit_spectra
		self.fit_photometry = fit_photometry
		self.distance = distance
		self.edistance = edistance
		self.flux_unit = flux_unit
		self.phot_unit = phot_unit

		# verify that all the mandatory parameters are given
		# reshape some input parameters and define non-provided parameters in terms of other parameters
		if fit_spectra:
			# number of input spectra
			if isinstance(wl_spectra, list): # when multiple spectra are provided
				N_spectra = len(wl_spectra) # number of input spectra
			else: # when only one spectrum is provided
				N_spectra = 1 # number of input spectra

			self.N_spectra = N_spectra

			# when one spectrum is given, convert the spectrum into a list
			# for wavelengths
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

			# handle input parameters
			for i in range(N_spectra):
				# convert input spectra to numpy arrays, if they are astropy
				wl_spectra[i] = utils.astropy_to_numpy(wl_spectra[i])
				flux_spectra[i] = utils.astropy_to_numpy(flux_spectra[i])
				eflux_spectra[i] = utils.astropy_to_numpy(eflux_spectra[i])
		
				# remove NaN values
				mask_nonan = (~np.isnan(wl_spectra[i])) & (~np.isnan(flux_spectra[i])) & (~np.isnan(eflux_spectra[i]))
				wl_spectra[i] = wl_spectra[i][mask_nonan]
				flux_spectra[i] = flux_spectra[i][mask_nonan]
				eflux_spectra[i] = eflux_spectra[i][mask_nonan]

				# remove negative fluxes
				mask_noneg = flux_spectra[i]>0
				wl_spectra[i] = wl_spectra[i][mask_noneg]
				flux_spectra[i] = flux_spectra[i][mask_noneg]
				eflux_spectra[i] = eflux_spectra[i][mask_noneg]

				# ensure that wavelength is arranged in ascending order
				sort_ind = np.argsort(wl_spectra[i])
				wl_spectra[i] = wl_spectra[i][sort_ind ]
				flux_spectra[i] = flux_spectra[i][sort_ind ]
				eflux_spectra[i] = eflux_spectra[i][sort_ind ]

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
					lam_res.append(utils.set_lam_res(wl_spectrum))
			# convert lam_res to the nearest integer
			if isinstance(lam_res, list):
				for i,lam_res_each in enumerate(lam_res): # for each lam_res value
					lam_res[i] = round(lam_res_each)
			else:
				lam_res = np.round(lam_res).astype(int)

			# verify the wl_spectra, flux_spectra, eflux_spectra, and res have the same dimension
			if len(flux_spectra)!=N_spectra: raise Exception(f'"flux_spectra" has fluxes for {len(flux_spectra)} spectra but {N_spectra} spectra are expected from "wl_spectra"')
			if len(eflux_spectra)!=N_spectra: raise Exception(f'"eflux_spectra" has efluxes for {len(eflux_spectra)} spectra but {N_spectra} spectra are expected from "wl_spectra"')
			if len(res)!=N_spectra: raise Exception(f'"res" has resolutions for {len(res)} spectra but {N_spectra} spectra are expected from "wl_spectra"')

			# set default flux_unit
			if flux_unit is None: flux_unit='erg/s/cm2/A'
			# convert input fluxes to erg/s/cm2/A, if needed
			if flux_unit=='Jy': # if fluxes are provided in Jy
				for i in range(N_spectra):
					out_convert_flux = synthetic_photometry.convert_flux(flux=flux_spectra[i], eflux=eflux_spectra[i], 
					                                                     wl=wl_spectra[i], unit_in='Jy', unit_out='erg/s/cm2/A')
					flux_spectra[i] = out_convert_flux['flux_out']
					eflux_spectra[i] = out_convert_flux['eflux_out']
			if flux_unit=='erg/s/cm2/um': # if fluxes are provided in erg/s/cm2/um
				for i in range(N_spectra):
					flux_spectra[i] = (flux_spectra[i]*u.erg/u.s/u.cm**2/u.micron).to(u.erg/u.s/u.cm**2/(u.nm*0.1)).value # erg/s/cm2/A
					eflux_spectra[i] = (eflux_spectra[i]*u.erg/u.s/u.cm**2/u.micron).to(u.erg/u.s/u.cm**2/(u.nm*0.1)).value # erg/s/cm2/A

		# add spectroscopic input data as attributes
		self.res = res
		self.lam_res = lam_res
		self.wl_spectra = wl_spectra
		self.flux_spectra = flux_spectra
		self.eflux_spectra = eflux_spectra

		# handle input photometry
		if fit_photometry:
			# if input photometry arrays are lists or astropy varaibles, convert them into numpy array
			phot = utils.var_to_numpy(phot)
			phot = utils.astropy_to_numpy(phot)
			ephot = utils.var_to_numpy(ephot)
			ephot = utils.astropy_to_numpy(ephot)
			filters = utils.var_to_numpy(filters)
			filters = utils.astropy_to_numpy(filters)

			# remove magnitudes with null magnitude errors (usually upper limits)
			mask_nonull = ephot!=0
			print(f'Null photometric errors for {filters[~mask_nonull]} magnitudes, so they will be discarded.')
			phot = phot[mask_nonull]
			ephot = ephot[mask_nonull]
			filters = filters[mask_nonull]

			# obtain relevant parameters from SVO for input filters
			params = ['WavelengthEff', 'WidthEff']
			out = utils.read_SVO_params(filters=filters, params=params)
			lambda_eff_SVO = u.Quantity(out['WavelengthEff'].data, u.nm*0.1).to(u.micron).value # in um
			width_eff_SVO = u.Quantity(out['WidthEff'].data, u.nm*0.1).to(u.micron).value # in um

			# set default phot_unit
			if phot_unit is None: phot_unit='mag'
			# convert input photometry to erg/s/cm2/A, if needed
			if phot_unit=='mag': # if photometry given in magnitudes
				out_mag_to_flux = synthetic_photometry.mag_to_flux(mag=phot, emag=ephot, filters=filters, 
				                                                   flux_unit='erg/s/cm2/A')
				phot = out_mag_to_flux['flux']
				ephot = out_mag_to_flux['eflux']
			if phot_unit=='Jy': # if photometry is given in Jy
				# effective wavelength of the filters is needed
				out_convert_flux = synthetic_photometry.convert_flux(flux=phot, eflux=ephot, wl=lambda_eff_SVO, 
				                                                     unit_in='Jy', unit_out='erg/s/cm2/A')
				phot = out_convert_flux['flux_out']
				ephot = out_convert_flux['eflux_out']
			if phot_unit=='erg/s/cm2/um': # if photometry is given in erg/s/cm2/um
				phot = (phot*u.erg/u.s/u.cm**2/u.micron).to(u.erg/u.s/u.cm**2/(u.nm*0.1)).value # erg/s/cm2/A
				ephot = (ephot*u.erg/u.s/u.cm**2/u.micron).to(u.erg/u.s/u.cm**2/(u.nm*0.1)).value # erg/s/cm2/A

			# add some exclusive photometric parameters as attributes
			self.lambda_eff_SVO = lambda_eff_SVO
			self.width_eff_SVO = width_eff_SVO

		# add some input photometric parameters as attributes
		self.phot = phot
		self.ephot = ephot
		self.filters = filters

		print('\n   Input data loaded successfully:')
		if fit_spectra: print(f'      {N_spectra} spectra')
		if fit_photometry: print(f'      {len(filters)} magnitudes')

#+++++++++++++++++++++++++++
class ModelOptions:
	'''
	Description:
	------------
		Define model options.

	Parameters:
	-----------
	- model : str, optional (required if a model spectrum is not provided via `wl_model` and `flux_model` or for nested sampling)
		Label for any of the available atmospheric models.
		See more info in ``seda.models.Models``.
	- model_dir : str, list, or array, optional (required if a model spectrum is not provided via `wl_model` and `flux_model` or for nested sampling)
		Path to the directory (str, list, or array) or directories (as a list or array) containing the model spectra (e.g., ``model_dir = ['path_1', 'path_2']``). 
		Avoid using paths with null spaces. 
	- params_ranges : dictionary, optional
		Minimum and maximum values for any free model parameters to select a model grid subset.
		E.g., ``params_ranges = {'Teff': [1000, 1200], 'logg': [4., 5.]}`` to consider spectra within those Teff and logg ranges.
		If a parameter range is not provided, the full range in ``model_dir`` is considered.
	- wl_model : array, optional (required if `model_dir` is not provided)
		Wavelengths in micron of model spectrum.
	- flux_model : array, optional (required if `model_dir` is not provided)
		Fluxes in erg/s/cm2/A of model spectrum.
	- path_save_spectra_conv: str, optional
		Directory path to store convolved model spectra. 
		If not provided (default), the convolved spectra will not be saved. 
		If the directory does not exist, it will be created. Otherwise, the spectra will be added to the existing folder.
		The convolved spectra will keep the same original names along with the ``res`` and ``lam_res`` parameters, e.g. 'original_spectrum_name_R100at1um.nc' for ``res=100`` and ``lam_res=1``.
		They will be saved as netCDF with xarray (it produces lighter files compared to normal ASCII files).
	- skip_convolution : {``True``, ``False``}, optional (default ``False``)
		Convolution of model spectra (the slowest process when fitting spectra) can (``True``) or cannot (``False``) be avoided. 
		Once the code has be run and the convolved spectra were stored in ``path_save_spectra_conv``, the convolved grid can be reused for other input data with the same resolution as the convolved spectra.
		If 'True', ``model_dir`` should include the previously convolved spectra for ``res`` at ``lam_res`` in ``input_parameters.InputData``. 
	- path_save_syn_phot: str, optional
		Directory path to store the synthetic fluxes (in erg/s/cm2/A).
		If not provided (default), the synthetic photometry will not be saved. 
		If the directory does not exist, it will be created. Otherwise, the photometry will be added to the existing folder.
		The synthetic photometry for different filters derived from the same model spectrum will be saved in a single ASCII table, named after the model with the suffix "_syn_phot.dat".
		If a synthetic photometry file for a given model spectrum already exists, it will be updated to include photometry for any new filters as needed.
	- skip_syn_phot : {``True``, ``False``}, optional (default ``False``)
		Synthetic photometry calculation (the lowest process when fitting photometry) can (``True``) or cannot (``False``) be avoided. 
		Once the code has be run and the synthetic photometry was stored in ``path_save_syn_phot``, 
		the synthetic values can be reused for other input photometric SEDs that use the same set of filters.
		If 'True', ``model_dir`` should correspond to the directory with the synthetic photometry for ``filters`` in ``input_parameters.InputData``. 

	Attributes
	----------
	Input parameters.
		All input parameters are stored as attributes with their default values if not specified.
	Other parameters:

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

	def __init__(self, model=None, model_dir=None, params_ranges=None,
		         wl_model=None, flux_model=None,
		         path_save_spectra_conv=None, skip_convolution=False,
		         path_save_syn_phot=None, skip_syn_phot=False):

		# verify necessary input parameters are provided
		if (model is None) & (model_dir is None): # when no model and model_dir parameters are provided
			if wl_model is None: raise Exception(f'"wl_model" must be provided')
			if flux_model is None: raise Exception(f'"flux_model" must be provided')
		if (wl_model is None) & (flux_model is None): # when a model spectrum is not explicitly provided
			if model is None: raise Exception(f'"model" must be provided')
			if model_dir is None: raise Exception(f'"model_dir" must be provided')
			if model not in models.Models().available_models: raise Exception(f'Models "{model}" are not recognized. Available models:'
			                                                                  f'\n{models.Models().available_models}')

		# when only one directory with models is given
		if not isinstance(model_dir, (list, np.ndarray)): model_dir = [model_dir]

		self.model = model
		self.model_dir = model_dir
		self.params_ranges = params_ranges
		self.wl_model = wl_model
		self.flux_model = flux_model
		self.path_save_spectra_conv = path_save_spectra_conv
		self.skip_convolution = skip_convolution
		self.path_save_syn_phot = path_save_syn_phot
		self.skip_syn_phot = skip_syn_phot

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
	- fit_wl_range : float array or list, optional
		Minimum and maximum wavelengths (in micron) where each input spectrum will be compared to the models. E.g., ``fit_wl_range = np.array([[fit_wl_min1, fit_wl_max1], [fit_wl_min2, fit_wl_max2]])``. 
		This parameter is used if ``fit_spectra`` but ignored if only ``fit_photometry``. 
		Default values are the minimum and the maximum wavelengths of each input spectrum.
	- disp_wl_range : float array, optional
		Minimum and maximum wavelengths (in micron) to compute the median wavelength dispersion of model spectra to convolve them.
		It can take a set of values for each input spectrum e.g. ``disp_wl_range = np.array([[disp_wl_min1, disp_wl_max1], [disp_wl_min2, disp_wl_max2]])``.
		Default values are ``fit_wl_range``.
	- model_wl_range : array or list, optional
		Minimum and maximum wavelength (in microns) to cut model spectra to keep only wavelengths of interest.
		Default values are the minimum and maximum wavelengths covered by the input spectra with a padding to avoid the point below.
		CAVEAT: the selected wavelength range of model spectra must cover the spectrophotometry used in the fit and a bit more (to avoid errors when resampling synthetic spectra using spectres).
	- fit_phot_range : float array or list, optional
		Minimum and maximum wavelengths (in micron) where photometry will be compared to the models. E.g., ``fit_phot_range = np.array([fit_phot_min1, fit_phot_max1])``. 
		This parameter is used if ``fit_photometry`` but ignored if only ``fit_spectra``. 
		Default values are the minimum and the maximum of the filter effective wavelengths from SVO.
	- weight_label : str, optional (default ``dataset``)
		Weight applied to each input data point (photometric and spectroscopic) when minimizing the chi-square statistics.
		This parameters is different to the weighting that comes from the observed flux uncertainties in the chi-square statistics definition.
		``weight_label`` allows assigning different relative weight to photometry and spectra (when ``fit_spectra`` and ``fit_photometry`` are enable),
		ensuring that photometric measurements do not have a negligible contribution to the chi-square, considering that photometric SEDs typically contain far fewer data points than spectroscopic SEDs.
		Available options are:
		- ``'dataset'`` (default): each dataset, whether photometric or spectroscopic, is assigned a weight equal to the inverse of its total number of data points. All data contribute the same, despite having very different numbers of points.
		- ``'width'``: each point is weighted by wavelength resolution (wavelength step) for spectra or filter effective width for photometry. Broader filters receive larger weights, but the overall contribution of photometry may still differ from that of spectra.
		- ``'none'``: apply the same weight to all input data points, which is equivalent to using no weighting at all (beyond the uncertainty-based weighting). Large datasets such as high-resolution spectra will dominate.
	- extinction_free_param : {``True``, ``False``}, optional (default ``False``)
		Extinction as a free parameter: 
			- ``'False'``: null extinction is assumed and it will not change.
			- ``'True'``: null extinction is assumed and it varies to minimize chi-square.
	- scaling_free_param : {``True``, ``False``}, optional (default ``True``)
		Scaling as a free parameter: 
			- ``'True'``: to find the scaling that minimizes chi square for each model.
			- ``'False'``: to fix ``scaling`` if radius and distance are known.
	- scaling: float, optional (required if ``scaling_free_param='False'``)
		Fixed scaling factor ((R/d)^2, R: object's radius, d: distance to the object) to be applied to model spectra.
	- avoid_IR_excess : {``True``, ``False``}, optional (default ``False``)
		Wavelengths longer than ``IR_excess_limit`` will (``'True'``) or will not (``'False'``) be avoided in the fit in case infrared excesses are expected. 
	- IR_excess_limit : float, optional (default 3 um).
		Shortest wavelength at which IR excesses are expected.
	- save_results : {``True``, ``False``}, optional (default ``True``)
		Save (``True``) or do not save (``False``) ``seda.chi2_fit`` results.
	- chi2_pickle_file : str, optional
		Filename for the output dictionary stored as a pickle file, if ``save_results``.
		Default name is '``model``\_chi2\_minimization.pickle'.
	- chi2_table_file : str, optional
		Filename for an output ascii table (if ``save_results``) with relevant information from the fit.
		Default name is '``model``\_chi2\_minimization.dat'.

	Attributes
	----------
	Input parameters either from this class or attributes from the ``InputData`` and ``ModelOptions`` classes.
		All input parameters are stored as attributes with their default values if not specified.
	Other parameters
		spectra_name : numpy array
			File names of selected model spectra
		spectra_name_full : numpy array
			File names of selected model spectra with full absolute path.
		if ``fit_spectra``:
			wl_spectra_min : float
				Minimum wavelength from the input spectra.
			wl_spectra_max : float
				Maximum wavelength from the input spectra.
			N_datapoints : int
				Total number of data points in all input spectra.
			N_model_spectra : int
				Number of model spectra selected within ``fit_wl_range``.
			wl_array_obs_fit : list
				Nested list input spectra wavelengths within ``fit_wl_range`` for each selected model spectrum.
				This is convenient because model spectra do not necessary all have the same wavelength coverage.
			flux_array_obs_fit : list
				Nested list with input spectra fluxes (in erg/s/cm2/A) within ``fit_wl_range`` for each selected model spectrum.
			eflux_array_obs_fit  : list
				Nested list with input spectra flux uncertainties (in erg/s/cm2/A) within ``fit_wl_range`` for each selected model spectrum.
			wl_array_model_conv_resam : list
				Nested list with wavelengths for the selected model spectra convolved using ``res`` and ``lam_res`` and resampled (based on the input wavelengths) within ``model_wl_range``.
			flux_array_model_conv_resam : list
				Nested list with fluxes (in erg/s/cm2/A) for the selected model spectra convolved using ``res`` and ``lam_res`` and resampled (based on the input wavelengths) within ``model_wl_range``.
			wl_array_model_conv_resam_fit : list
				Same as ``wl_array_model_conv_resam`` but within ``fit_wl_range``.
			flux_array_model_conv_resam_fit :
				Same as ``flux_array_model_conv_resam`` but within ``fit_wl_range``.
			weight_spec_fit : list
				Nested list with the weights assigned to each spectroscopic data point within the fit range considering the equation chi2 = weight * (data-model)^2 / edata^2.
		if ``fit_photometry``:
			phot_fit : numpy array
				Input photometry ``phot`` (in erg/s/cm2/A) within ``fit_phot_range``.
			ephot_fit : numpy array
				Input photometry errors ``ephot`` (in erg/s/cm2/A) within ``fit_phot_range``.
			filters_fit : numpy array
				Input filters ``filters`` within ``fit_phot_range``.
			lambda_eff_SVO_fit : numpy array
				Effective wavelengths from SVO for ``filters_fit``.
			width_eff_SVO_fit : numpy array
				Effective widths from SVO for ``filters_fit``.
			flux_syn_array_model_fit : list
				Nested list with synthetic fluxes (in erg/s/cm2/A) for ``filters_fit`` for each selected model spectrum.
			lambda_eff_array_model_fit : numpy array
				Effective wavelengths (in micron) from the spectrum for ``filters_fit`` for each selected model spectrum.
			width_eff_array_model_fit : numpy array
				Effective width (in micron) from the spectrum for ``filters_fit`` for each selected model spectrum.
			weight_phot_fit : list
				Nested list with the weights assigned to each photometric data point within the fit range considering the equation chi2 = weight * (data-model)^2 / edata^2.

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
#		The available options are:
#		- ``"dataset"``  
#		  Each dataset (photometric or spectroscopic) is assigned a weight equal  
#		  to the inverse of the total number of its data points.  
#		  All datasets contribute equally, regardless of size.


	def __init__(self, my_data, my_model, 
		fit_wl_range=None, disp_wl_range=None, model_wl_range=None, fit_phot_range=None, 
		weight_label='dataset', extinction_free_param=False, scaling_free_param=True, 
		scaling=None, avoid_IR_excess=False, IR_excess_limit=3, save_results=True,
		chi2_pickle_file=None, chi2_table_file=None):

		ini_time_mychi2 = utils.time.time() # to estimate the time elapsed running chi2

		dir_sep = os.sep # directory separator for the current operating system

		# associate input parameters as attributes
		self.fit_wl_range = fit_wl_range
		self.disp_wl_range = disp_wl_range
		self.model_wl_range = model_wl_range
		self.fit_phot_range = fit_phot_range
		self.save_results = save_results
		self.scaling_free_param = scaling_free_param
		self.scaling = scaling
		self.extinction_free_param = extinction_free_param
		self.avoid_IR_excess = avoid_IR_excess
		self.IR_excess_limit = IR_excess_limit

		# parameters from InputData as attributes
		self.fit_spectra = my_data.fit_spectra
		self.fit_photometry = my_data.fit_photometry
		self.flux_unit = my_data.flux_unit
		self.phot_unit = my_data.phot_unit
		self.phot = my_data.phot
		self.ephot = my_data.ephot
		self.filters = my_data.filters
		self.distance = my_data.distance
		self.edistance = my_data.edistance
		self.res = my_data.res
		self.lam_res = my_data.lam_res
		self.wl_spectra = my_data.wl_spectra
		self.flux_spectra = my_data.flux_spectra
		self.eflux_spectra = my_data.eflux_spectra
		if my_data.fit_spectra: self.N_spectra = my_data.N_spectra
		if my_data.fit_photometry: 
			self.lambda_eff_SVO = my_data.lambda_eff_SVO # in um
			self.width_eff_SVO = my_data.width_eff_SVO # in um

		# parameters from ModelOptions as attributes
		self.model = my_model.model
		self.model_dir = my_model.model_dir
		self.params_ranges = my_model.params_ranges
		self.wl_model = my_model.wl_model
		self.flux_model = my_model.flux_model
		self.path_save_spectra_conv = my_model.path_save_spectra_conv
		self.skip_convolution = my_model.skip_convolution
		self.path_save_syn_phot = my_model.path_save_syn_phot
		self.skip_syn_phot = my_model.skip_syn_phot

		# extract parameters for convenience
		fit_spectra = my_data.fit_spectra
		fit_photometry = my_data.fit_photometry
		wl_spectra = my_data.wl_spectra
		flux_spectra = my_data.flux_spectra
		eflux_spectra = my_data.eflux_spectra
		res = my_data.res
		lam_res = my_data.lam_res
		phot = my_data.phot
		ephot = my_data.ephot
		filters = my_data.filters
		model = my_model.model
		model_dir = my_model.model_dir
		wl_model = my_model.wl_model
		flux_model = my_model.flux_model
		params_ranges = my_model.params_ranges
		path_save_spectra_conv = my_model.path_save_spectra_conv
		skip_convolution = my_model.skip_convolution
		path_save_syn_phot = my_model.path_save_syn_phot
		skip_syn_phot = my_model.skip_syn_phot
		if fit_spectra: N_spectra = my_data.N_spectra
		if fit_photometry: 
			lambda_eff_SVO = my_data.lambda_eff_SVO # in um
			width_eff_SVO = my_data.width_eff_SVO # in um

		# handle input parameters
		if weight_label not in allowed_weights: 
			raise Exception(f'"{weight_label}" is not recognized for the parameter "weight_label". '
			                f'Available options: {sorted(allowed_weights)}.')
		self.weight_label = weight_label

		# handle input spectra
		if fit_spectra:
			# number of data points in the input spectra
			out_input_data_stats = utils.input_data_stats(wl_spectra=wl_spectra)
			wl_spectra_min = out_input_data_stats['wl_spectra_min']
			wl_spectra_max = out_input_data_stats['wl_spectra_max']
			N_datapoints = out_input_data_stats['N_datapoints']

			# handle fit_wl_range
			fit_wl_range = utils.set_fit_wl_range(fit_wl_range=fit_wl_range, N_spectra=N_spectra, wl_spectra=wl_spectra)
			# handle disp_wl_range
			disp_wl_range = utils.set_disp_wl_range(disp_wl_range=disp_wl_range, N_spectra=N_spectra, wl_spectra=wl_spectra)
			# handle model_wl_range
			model_wl_range = utils.set_model_wl_range(model_wl_range=model_wl_range, wl_spectra_min=wl_spectra_min, wl_spectra_max=wl_spectra_max)
	
			# define parameters as attributes (the ones already as attributes will be replaced by the updated parameter values)
			self.fit_wl_range = fit_wl_range
			self.disp_wl_range = disp_wl_range
			self.model_wl_range = model_wl_range
			self.wl_spectra_min = wl_spectra_min
			self.wl_spectra_max = wl_spectra_max
			self.N_datapoints = N_datapoints

		# handle input photometry
		if fit_photometry:
			# handle fit_phot_range
			fit_phot_range = utils.set_fit_phot_range(fit_phot_range=fit_phot_range, filters=filters)
			self.fit_phot_range = fit_phot_range

		# file name to save the chi2 results as a pickle
		if chi2_pickle_file is None: self.chi2_pickle_file = f'{model}_chi2_minimization.pickle'
		else: self.chi2_pickle_file = chi2_pickle_file
		if chi2_table_file is None: self.chi2_table_file = f'{model}_chi2_minimization.dat'
		else: self.chi2_table_file = chi2_table_file

		# when an input model spectrum is explicitly provided, convolve and resample it
		if (wl_model is not None) & (flux_model is not None):
			# convolve the model spectrum separately according to each input spectrum

			# initialize lists to save a resampled and convolved model spectrum for each input spectrum
			# and for all model spectra (one in this case), but to keep the format for the case of multiple model spectra
			wl_array_model_conv_resam = []
			flux_array_model_conv_resam = []
			# initialize lists to save a resampled and convolved model spectrum for each input spectrum
			wl_array_model_conv_resam_each = []
			flux_array_model_conv_resam_each = []
			for k in range(N_spectra): # for each input observed spectrum
				# convolve model
				# define convolving range as the full wavelength range of each input spectrum plus a padding
				convolve_wl_range = utils.add_pad(wl_spectra[k].min(), wl_spectra[k].max())
				out_convolve_spectrum = utils.convolve_spectrum(wl=wl_model, flux=flux_model, lam_res=lam_res[k], res=res[k], 
				                                                disp_wl_range=disp_wl_range[k], convolve_wl_range=convolve_wl_range)
				# replace model spectrum by the convolved one
				wl_model_conv = out_convolve_spectrum['wl_conv']
				flux_model_conv = out_convolve_spectrum['flux_conv']

				# resample the convolved model spectrum to the wavelength data points in the observed spectra
				mask = (wl_spectra[k]>wl_model_conv.min()) & (wl_spectra[k]<wl_model_conv.max()) # input wavelengths within model wavelength coverage
				flux_array_model_conv_resam_each.append(spectres(wl_spectra[k][mask], wl_model_conv, flux_model_conv)) # resampled fluxes
				wl_array_model_conv_resam_each.append(wl_spectra[k][mask]) # wavelengths for resampled fluxes

			# nested list with all resampled model spectra
			wl_array_model_conv_resam.append(wl_array_model_conv_resam_each)
			flux_array_model_conv_resam.append(flux_array_model_conv_resam_each)

			N_model_spectra = 1 # number of model spectra

		# when directories with spectra files are provided, read, convolve (if needed), and resample model spectra
		if (model is not None) & (model_dir is not None):
			# read the model spectra names in the input folders and meeting the indicated parameter ranges 
			if not skip_convolution and not skip_syn_phot:
				filename_pattern = models.Models(model).filename_pattern # to be used to select files with original names
			else:
				filename_pattern = models.Models(model).filename_pattern+'*' # to be used to select files with original names plus the suffix added when convolved spectra were saved

			out_select_model_spectra = utils.select_model_spectra(model=model, model_dir=model_dir, params_ranges=params_ranges, filename_pattern=filename_pattern)
			spectra_name_full = out_select_model_spectra['spectra_name_full']
			spectra_name = out_select_model_spectra['spectra_name']

			if skip_convolution or skip_syn_phot:
				# for the case of pre-convolved models or pre-computed synthetic photometry, 
				# keep only the base name (without the suffix added after the original file names)
				for i,spectrum_name in enumerate(spectra_name):
					# spectrum file name without the extension added for the convolved version
					if fit_spectra: basename = spectrum_name.split('_R')[0]
					if fit_photometry: basename = spectrum_name.split('_syn_phot')[0]
					spectra_name[i] = basename
					spectra_name_full[i] = os.path.dirname(spectra_name_full[i])+dir_sep+spectra_name[i] # with full path

				# remove duplicate spectrum file names due to having the same spectrum stored multiple times for different resolutions at different wavelengths
				spectra_name = np.unique(spectra_name)
				spectra_name_full = np.unique(spectra_name_full)
	
			N_model_spectra = len(spectra_name) # number of selected model spectra
	
			# convolve and/or resample model spectra, as need it
			if fit_spectra:
				# create a tqdm progress bar
				if not skip_convolution: # read original model spectra
					model_bar = tqdm(total=N_model_spectra, desc='Reading, convolving, and resampling model spectra')
				else:
					model_bar = tqdm(total=N_model_spectra, desc='Reading and resampling model spectra')
			
				# initialize lists to save a resampled and convolved model spectrum for each input spectrum
				wl_array_model_conv_resam = []
				flux_array_model_conv_resam = []

				# read, convolve, and resample model spectra accordingly for each input spectrum
				if not skip_convolution: # do not avoid convolution
					for i,spectrum_name in enumerate(spectra_name): # for each model spectrum
						# update progress bar
						model_bar.update(1)

						# read model spectrum with original resolution in the full model_wl_range
						out_read_model_spectrum = models.read_model_spectrum(spectrum_name_full=spectra_name_full[i], model=model, model_wl_range=model_wl_range)
						wl_model = out_read_model_spectrum['wl_model'] # um
						flux_model = out_read_model_spectrum['flux_model'] # erg/s/cm2/A

						# convolve the model spectrum separately according to each input spectrum
						wl_array_model_conv_resam_each = []
						flux_array_model_conv_resam_each = []
						for k in range(N_spectra): # for each input observed spectrum
							# define convolving range as the full wavelength range of each input spectrum plus a padding
							convolve_wl_range = utils.add_pad(wl_spectra[k].min(), wl_spectra[k].max())

							# convolve model spectrum according to the resolution of each input spectrum
							if path_save_spectra_conv is None: # do not save the convolved spectrum
								out_convolve_spectrum = utils.convolve_spectrum(wl=wl_model, flux=flux_model, lam_res=lam_res[k], res=res[k], 
								                                                disp_wl_range=disp_wl_range[k], convolve_wl_range=convolve_wl_range)
							else: # save convolved spectrum
								if not os.path.exists(path_save_spectra_conv): os.makedirs(path_save_spectra_conv) # make directory (if not existing) to store convolved spectra
								out_file = path_save_spectra_conv+spectra_name[i]+f'_R{res[k]}at{lam_res[k]}um.nc'
								out_convolve_spectrum = utils.convolve_spectrum(wl=wl_model, flux=flux_model, lam_res=lam_res[k], res=res[k], 
								                                                disp_wl_range=disp_wl_range[k], convolve_wl_range=convolve_wl_range, out_file=out_file)
							# replace model spectrum by the convolved one
							wl_model_conv = out_convolve_spectrum['wl_conv']
							flux_model_conv = out_convolve_spectrum['flux_conv']

							# resample the convolved model spectrum to the wavelength data points in the observed spectra
							mask = (wl_spectra[k]>wl_model_conv.min()) & (wl_spectra[k]<wl_model_conv.max()) # input wavelengths within model wavelength coverage
							flux_array_model_conv_resam_each.append(spectres(wl_spectra[k][mask], wl_model_conv, flux_model_conv)) # resampled fluxes
							wl_array_model_conv_resam_each.append(wl_spectra[k][mask]) # wavelengths for resampled fluxes

						# nested list with all resampled model spectra
						wl_array_model_conv_resam.append(wl_array_model_conv_resam_each)
						flux_array_model_conv_resam.append(flux_array_model_conv_resam_each)

				else: # skip convolution
					for i,spectrum_name in enumerate(spectra_name): # for each model spectrum
						# update progress bar
						model_bar.update(1)

						# read and resample model spectra accordingly for each input spectrum
						wl_array_model_conv_resam_each = []
						flux_array_model_conv_resam_each = []
						for k in range(N_spectra): # for each input observed spectrum
							# read pre-convolved model spectrum for each input spectrum in its full wavelength range plus a padding
							model_wl_range_each = utils.add_pad(wl_spectra[k].min(), wl_spectra[k].max())
							spectrum_name_full = spectra_name_full[i]+f'_R{res[k]}at{lam_res[k]}um.nc'
							try: out_read_model_spectrum = models.read_model_spectrum_conv(spectrum_name_full=spectrum_name_full, model_wl_range=model_wl_range_each)
							except: raise Exception(f'Convolved model spectrum {spectrum_name_full} does not exists.')
							wl_model = out_read_model_spectrum['wl_model'] # um
							flux_model = out_read_model_spectrum['flux_model'] # erg/s/cm2/A

							# resample the convolved model spectrum to the wavelength data points in the observed spectra
							mask = (wl_spectra[k]>wl_model.min()) & (wl_spectra[k]<wl_model.max()) # input wavelengths within model wavelength coverage
							flux_array_model_conv_resam_each.append(spectres(wl_spectra[k][mask], wl_model, flux_model)) # resampled fluxes
							wl_array_model_conv_resam_each.append(wl_spectra[k][mask]) # wavelengths for resampled fluxes

						# nested list with all resampled model spectra
						wl_array_model_conv_resam.append(wl_array_model_conv_resam_each)
						flux_array_model_conv_resam.append(flux_array_model_conv_resam_each)
			
				# close progress bar
				model_bar.close()

			# derive synthetic photometry from model spectra 
			if fit_photometry:
				# select filters within the wavelength range to fit the photometry
				mask_fit = (lambda_eff_SVO>=fit_phot_range.min()) & (lambda_eff_SVO<=fit_phot_range.max())
				phot_fit = phot[mask_fit]
				ephot_fit = ephot[mask_fit]
				filters_fit = filters[mask_fit]
				lambda_eff_SVO_fit = lambda_eff_SVO[mask_fit]
				width_eff_SVO_fit = width_eff_SVO[mask_fit]

				print(f'\n      {len(filters_fit)} of {len(filters)} input valid magnitudes within the fit range "fit_phot_range"')

				# create a tqdm progress bar
				if not skip_syn_phot: # read original model spectra to then derive synthetic photometry
					model_bar = tqdm(total=N_model_spectra, desc='Deriving synthetic photometry from model spectra')
				else:
					model_bar = tqdm(total=N_model_spectra, desc='Reading pre-computed synthetic photometry')

				# initialize lists to save information for the synthetic photometry within the fit range from each input spectrum
				flux_syn_array_model_fit = []
				lambda_eff_array_model_fit = []
				width_eff_array_model_fit = []

				# read and derive synthetic photometry
				if not skip_syn_phot: # do not avoid synthetic photometry calculation
					for i,spectrum_name in enumerate(spectra_name): # for each model spectrum
						# update progress bar
						model_bar.update(1)

						# read model spectrum with original resolution in the full model_wl_range_each
						out_read_model_spectrum = models.read_model_spectrum(spectrum_name_full=spectra_name_full[i], model=model, model_wl_range=model_wl_range)
						wl_model = out_read_model_spectrum['wl_model'] # um
						flux_model = out_read_model_spectrum['flux_model'] # erg/s/cm2/A

						# derive synthetic photometry
						out_syn_phot = synthetic_photometry.synthetic_photometry(wl=wl_model, flux=flux_model, flux_unit='erg/s/cm2/A', filters=filters_fit)

						# store filters' parameters in the lists
						flux_syn_array_model_fit.append(out_syn_phot['syn_flux(erg/s/cm2/A)']) # erg/s/cm2/A
						lambda_eff_array_model_fit.append(out_syn_phot['lambda_eff(um)']) # um
						width_eff_array_model_fit.append(out_syn_phot['width_eff(um)']) # um

						# store synthetic photometric
						if path_save_syn_phot is not None:
							file_name = path_save_syn_phot+spectra_name[i]+'_syn_phot.dat'
							if not os.path.exists(file_name): # file with synthetic photometry does not exist yet
								# make a dictionary with the parameters to be saved
								dict_syn_phot = {}
								keys = ['filters', 'syn_flux(erg/s/cm2/A)', 'lambda_eff(um)', 'width_eff(um)'] # parameters of interest
								for key in keys:
									if key=='filters':
										dict_syn_phot[key] = out_syn_phot[key]
									else:
										dict_syn_phot[key] = np.round(out_syn_phot[key], 6) # keep six decimals
								# sort dictionary with respect to filter name
								sort_ind = np.argsort(dict_syn_phot['filters'])
								for key in dict_syn_phot.keys():
									dict_syn_phot[key] = dict_syn_phot[key][sort_ind]

								# save the dictionary as prettytable table
								if not os.path.exists(path_save_syn_phot): os.makedirs(path_save_syn_phot) # make directory (if not existing) to store synthetic photometry
								save_prettytable(my_dict=dict_syn_phot, table_name=file_name)

							else: # file already exist
								# open file to see whether the flux for a given filter is already stored
								dict_syn_phot = read_prettytable(file_name)

								for j, filt in enumerate(filters_fit): # for each filter used to derived synthetic photometry
									if filt not in dict_syn_phot['filters']: # filter with synthetic photometry is not in the table
										for key in dict_syn_phot.keys(): # for each parameter in the table
											if key=='filters':
												dict_syn_phot[key] = np.append(dict_syn_phot[key], out_syn_phot[key][j])
											else:
												dict_syn_phot[key] = np.append(dict_syn_phot[key], np.round(out_syn_phot[key][j], 6)) # keep six decimals

								# sort dictionary with respect to filter name
								sort_ind = np.argsort(dict_syn_phot['filters'])
								for key in dict_syn_phot.keys():
									dict_syn_phot[key] = dict_syn_phot[key][sort_ind]

								# update the existing file with new synthetic photometry
								save_prettytable(my_dict=dict_syn_phot, table_name=file_name)

				else: # read pre-computed synthetic photometry
					for i,spectrum_name_full in enumerate(spectra_name_full): # for each model spectrum
						# update progress bar
						model_bar.update(1)

						out_syn_phot = read_prettytable(filename=spectrum_name_full+'_syn_phot.dat')

						# get from the table only parameters for the input filters within the fit range
						flux_syn_each = []
						lambda_eff_each = []
						width_eff_each = []
						for filt in filters_fit:
							if filt in out_syn_phot['filters']: # filter is in the table with synthetic photometry
								ind = out_syn_phot['filters']==filt # index in the table for filter in the iteration
								# store filters' parameters in the lists
								flux_syn_each.append(out_syn_phot['syn_flux(erg/s/cm2/A)'][ind][0])
								lambda_eff_each.append(out_syn_phot['lambda_eff(um)'][ind][0])
								width_eff_each.append(out_syn_phot['width_eff(um)'][ind][0])

							else:
								raise Exception(f'There is not synthetic photometry for filter "{filt}" and model "{spectra_name[i]}".')

						# store filters' parameters in the lists
						flux_syn_array_model_fit.append(np.array(flux_syn_each))
						lambda_eff_array_model_fit.append(np.array(lambda_eff_each))
						width_eff_array_model_fit.append(np.array(width_eff_each))

				# close progress bar
				model_bar.close()

		# prepare input spectra and model spectra for the chi-square minimization
		if fit_spectra:
			# cut input spectra to the wavelength region for the fit or model coverage, whatever is narrower
			wl_spectra_fit = []
			flux_spectra_fit = []
			eflux_spectra_fit = []
			for i in range(N_model_spectra): # for each model spectrum 
				wl_spectra_fit_each = []
				flux_spectra_fit_each = []
				eflux_spectra_fit_each = []
				for k in range(N_spectra): # for each input observed spectrum
					mask_fit = (wl_spectra[k] >= max(fit_wl_range[k][0], wl_array_model_conv_resam[i][k].min())) & \
					           (wl_spectra[k] <= min(fit_wl_range[k][1], wl_array_model_conv_resam[i][k].max()))
					wl_spectra_fit_each.append(wl_spectra[k][mask_fit])
					flux_spectra_fit_each.append(flux_spectra[k][mask_fit])
					eflux_spectra_fit_each.append(eflux_spectra[k][mask_fit])
				# nested list with input spectra in the fit range
				# it will have a set of input spectra for each model spectrum, 
				# which is convenient in case all model spectra do not necessary have the same wavelength coverage
				wl_spectra_fit.append(wl_spectra_fit_each)
				flux_spectra_fit.append(flux_spectra_fit_each)
				eflux_spectra_fit.append(eflux_spectra_fit_each)
	
			# cut model spectra to the wavelength region for the fit
			wl_array_model_conv_resam_fit = []
			flux_array_model_conv_resam_fit = []
			for i in range(N_model_spectra): # for each model spectrum
				wl_array_model_conv_resam_fit_each = []
				flux_array_model_conv_resam_fit_each = []
				for k in range(N_spectra): # for each input observed spectrum
					# mask to select model wavelength points within the fit range
					# if a model spectrum has a narrower coverage, so that will be the selected range for the fit
					mask_fit = (wl_array_model_conv_resam[i][k]>=fit_wl_range[k][0]) & \
					           (wl_array_model_conv_resam[i][k]<=fit_wl_range[k][1])
					flux_array_model_conv_resam_fit_each.append(flux_array_model_conv_resam[i][k][mask_fit])
					wl_array_model_conv_resam_fit_each.append(wl_array_model_conv_resam[i][k][mask_fit])
				# nested list with all resampled and convolved model spectra in the fit range
				wl_array_model_conv_resam_fit.append(wl_array_model_conv_resam_fit_each)
				flux_array_model_conv_resam_fit.append(flux_array_model_conv_resam_fit_each)

		# set weights for the fit
		out_compute_weights = compute_weights(fit_spectra=fit_spectra, fit_photometry=fit_photometry, weight_label=weight_label, 
		                                      wl_spectra_fit=wl_spectra_fit[0], width_eff_SVO_fit=width_eff_SVO_fit)
		weight_spec_fit = out_compute_weights['weight_spec_fit']
		weight_phot_fit = out_compute_weights['weight_phot_fit']

		# assign some attributes
		self.N_model_spectra = N_model_spectra
		if (model is not None) & (model_dir is not None):
			self.spectra_name = spectra_name
			self.spectra_name_full = spectra_name_full
		if fit_spectra:
			self.wl_array_obs_fit = wl_spectra_fit
			self.flux_array_obs_fit = flux_spectra_fit
			self.eflux_array_obs_fit = eflux_spectra_fit
			self.wl_array_model_conv_resam = wl_array_model_conv_resam
			self.flux_array_model_conv_resam = flux_array_model_conv_resam
			self.wl_array_model_conv_resam_fit = wl_array_model_conv_resam_fit
			self.flux_array_model_conv_resam_fit = flux_array_model_conv_resam_fit
			self.weight_spec_fit = weight_spec_fit
		if fit_photometry:
			self.flux_syn_array_model_fit = flux_syn_array_model_fit
			self.lambda_eff_array_model_fit = lambda_eff_array_model_fit
			self.width_eff_array_model_fit = width_eff_array_model_fit
			self.phot_fit = phot_fit
			self.ephot_fit = ephot_fit
			self.filters_fit = filters_fit
			self.lambda_eff_SVO_fit = lambda_eff_SVO_fit
			self.width_eff_SVO_fit = width_eff_SVO_fit
			self.weight_phot_fit = weight_phot_fit

		print('\n   Chi-square fit options loaded successfully')
		fin_time_mychi2 = utils.time.time()
		utils.print_time(fin_time_mychi2-ini_time_mychi2)

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
	- disp_wl_range : float array, optional
		Minimum and maximum wavelengths (in micron) to compute the median wavelength dispersion of model spectra to convolve them.
		It can take a set of values for each input spectrum e.g. ``disp_wl_range = np.array([[disp_wl_min1, disp_wl_max1], [disp_wl_min2, disp_wl_max2]])``.
		Default values are ``fit_wl_range``.
	- model_wl_range : array or list, optional
		Minimum and maximum wavelength (in microns) to cut model spectra (to make the code faster). 
		Default values are the same as ``fit_wl_range`` with a padding to avoid the point below.
		CAVEAT: the selected wavelength range of model spectra must cover the spectrophotometry used in the fit and a bit more (to avoid errors when resampling synthetic spectra using spectres)
	- fit_phot_range : float array or list, optional
		Minimum and maximum wavelengths (in micron) where photometry will be compared to the models. E.g., ``fit_phot_range = np.array([fit_phot_min1, fit_phot_max1])``. 
		This parameter is used if ``fit_photometry`` but ignored if only ``fit_spectra``. 
		Default values are the minimum and the maximum of the filter effective wavelengths from SVO.
	- weight_label : str, optional (default ``dataset``)
		Weight applied to each input data point (photometric and spectroscopic) when minimizing the chi-square statistics.
		This parameters is different to the weighting that comes from the observed flux uncertainties in the chi-square statistics definition.
		``weight_label`` allows assigning different relative weight to photometry and spectra (when ``fit_spectra`` and ``fit_photometry`` are enable),
		ensuring that photometric measurements do not have a negligible contribution to the chi-square, considering that photometric SEDs typically contain far fewer data points than spectroscopic SEDs.
		Available options are:
			- ``'dataset'`` (default): each dataset, whether photometric or spectroscopic, is assigned a weight equal to the inverse of its total number of data points. All data contribute the same, despite having very different numbers of points.
			- ``'width'``: each point is weighted by wavelength resolution (wavelength step) for spectra or filter effective width for photometry. Broader filters receive larger weights, but the overall contribution of photometry may still differ from that of spectra.
			- ``'none'``: apply the same weight to all input data points, which is equivalent to using no weighting at all (beyond the uncertainty-based weighting). Large datasets such as high-resolution spectra will dominate.
	- R_range: float array, optional (used in ``bayes_fit``)
		Minimum and maximum radius values to sample the posterior for radius. It requires the parameter ``distance`` in `input_parameters.InputData`.
	- 'chi2_pickle_file' : dictionary, optional
		Pickle file with a dictionary with the results from the chi-square minimization by ``chi2_fit.chi2``.
		If given, the grid nodes around the best fit will be used as ``params_ranges`` to select a grid subset and as sampling priors.
		Otherwise, ``params_ranges`` provided in ``ModelOptions`` will be considered.
	- grid: dictionary, optional
		Model grid (``'wavelength'`` and ``'flux'``) generated by ``seda.utils.read_grid`` to interpolate model spectra. 
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

	Attributes
	----------
	Input parameters either from this class or attributes from the ``InputData`` and ``ModelOptions`` classes.
		All input parameters are stored as attributes with their default values if not specified.
	Other parameters
		grid : list
			Nested list of dictionaries containing the selected model spectra grid and their associated unique free-parameter values.
			If ``fit_spectra``, a dictionary including the selected model spectra convolved using ``res`` and ``lam_res`` and resampled (based on the input wavelengths) within ``fit_wl_range``.
			If ``fit_photometry``, a dictionary containing synthetic model fluxes (in erg/s/cm2/A) for ``filters_fit`` computed from the selected model spectra.
		ndim : int
			Number of free parameters in the model fit equal to the number of free parameters plus radius (if a ``distance`` is provided).
		filename_pattern : list
			String as a list with a common pattern in model spectra filenames.
		params_unique : dictionary
			Dictionary with unique values for each free parameters in the selected models.
		params_priors : dictionary
			Dictionary with the prior ranges for each model free parameter.
		if ``fit_spectra``:
			wl_spectra_min : float
				Minimum wavelength from the input spectra.
			wl_spectra_max : float
				Maximum wavelength from the input spectra.
			N_datapoints : int
				Total number of data points in all input spectra.
			wl_spectra_fit : list
				Nested list input spectra wavelengths within ``fit_wl_range`` for each selected model spectrum.
				This is convenient because model spectra do not necessary all have the same wavelength coverage.
			flux_spectra_fit : list
				Nested list with input spectra fluxes (in erg/s/cm2/A) within ``fit_wl_range`` for each selected model spectrum.
			eflux_spectra_fit : list
				Nested list with input spectra flux uncertainties (in erg/s/cm2/A) within ``fit_wl_range`` for each selected model spectrum.
			weight_spec_fit : list
				Nested list with the weights assigned to each spectroscopic data point within the fit range considering the equation chi2 = weight * (data-model)^2 / edata^2.
		if ``fit_photometry``:
			phot_fit : numpy array
				Input photometry ``phot`` (in erg/s/cm2/A) within ``fit_phot_range``.
			ephot_fit : numpy array
				Input photometry errors ``ephot`` (in erg/s/cm2/A) within ``fit_phot_range``.
			filters_fit : numpy array
				Input filters ``filters`` within ``fit_phot_range``.
			lambda_eff_SVO_fit : numpy array
				Effective wavelengths from SVO for ``filters_fit``.
			width_eff_SVO_fit : numpy array
				Effective widths from SVO for ``filters_fit``.
			weight_phot_fit : list
				Nested list with the weights assigned to each photometric data point within the fit range considering the equation chi2 = weight * (data-model)^2 / edata^2.

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

	def __init__(self, my_data, my_model, 
	             fit_wl_range=None, disp_wl_range=None, model_wl_range=None, fit_phot_range=None,
	             weight_label='dataset', R_range=None, chi2_pickle_file=None, bayes_pickle_file=None,
		         grid=None, dynamic_sampling=True, nlive=500, save_results=True):

		# associate input parameters as attributes
		self.fit_wl_range = fit_wl_range
		self.disp_wl_range = disp_wl_range
		self.model_wl_range = model_wl_range
		self.fit_phot_range = fit_phot_range
		self.R_range = R_range
		self.chi2_pickle_file = chi2_pickle_file
		self.bayes_pickle_file = bayes_pickle_file
		self.grid = grid
		self.dynamic_sampling = dynamic_sampling
		self.nlive = nlive
		self.save_results = save_results

		# read parameters from InputData
		self.fit_spectra = my_data.fit_spectra
		self.fit_photometry = my_data.fit_photometry
		self.flux_unit = my_data.flux_unit
		self.phot_unit = my_data.phot_unit
		self.phot = my_data.phot
		self.ephot = my_data.ephot
		self.filters = my_data.filters
		self.distance = my_data.distance
		self.edistance = my_data.edistance
		self.res = my_data.res
		self.lam_res = my_data.lam_res
		self.wl_spectra = my_data.wl_spectra
		self.flux_spectra = my_data.flux_spectra
		self.eflux_spectra = my_data.eflux_spectra
		if my_data.fit_spectra: self.N_spectra = my_data.N_spectra
		if my_data.fit_photometry: 
			self.lambda_eff_SVO = my_data.lambda_eff_SVO # in um
			self.width_eff_SVO = my_data.width_eff_SVO # in um

		# read parameters from ModelOptions
		self.model = my_model.model
		self.model_dir = my_model.model_dir
		self.params_ranges = my_model.params_ranges
		self.path_save_spectra_conv = my_model.path_save_spectra_conv
		self.skip_convolution = my_model.skip_convolution

		# extract parameters for convenience
		fit_spectra = my_data.fit_spectra
		fit_photometry = my_data.fit_photometry
		wl_spectra = my_data.wl_spectra
		flux_spectra = my_data.flux_spectra
		eflux_spectra = my_data.eflux_spectra
		distance = my_data.distance
		res = my_data.res
		lam_res = my_data.lam_res
		phot = my_data.phot
		ephot = my_data.ephot
		filters = my_data.filters
		model = my_model.model
		model_dir = my_model.model_dir
		params_ranges = my_model.params_ranges
		path_save_spectra_conv = my_model.path_save_spectra_conv
		skip_convolution = my_model.skip_convolution
		if fit_spectra: N_spectra = my_data.N_spectra
		if fit_photometry: 
			lambda_eff_SVO = my_data.lambda_eff_SVO # in um
			width_eff_SVO = my_data.width_eff_SVO # in um

		# handle input parameters
		if weight_label not in allowed_weights: 
			raise Exception(f'"{weight_label}" is not recognized for the parameter "weight_label". '
			                f'Available options: {sorted(allowed_weights)}.')
		self.weight_label = weight_label

		# handle input spectra
		if fit_spectra:
			# some statistic for input spectra
			out_input_data_stats = utils.input_data_stats(wl_spectra=wl_spectra)
			wl_spectra_min = out_input_data_stats['wl_spectra_min'] # minimum wavelength in the input spectra
			wl_spectra_max = out_input_data_stats['wl_spectra_max'] # maximum wavelength in the input spectra
			N_datapoints = out_input_data_stats['N_datapoints'] # number of data points in all input spectra

			# handle fit_wl_range
			fit_wl_range = utils.set_fit_wl_range(fit_wl_range=fit_wl_range, N_spectra=N_spectra, wl_spectra=wl_spectra)
			# handle disp_wl_range
			disp_wl_range = utils.set_disp_wl_range(disp_wl_range=disp_wl_range, N_spectra=N_spectra, wl_spectra=wl_spectra)
			# handle model_wl_range
			model_wl_range = utils.set_model_wl_range(model_wl_range=model_wl_range, wl_spectra_min=wl_spectra_min, wl_spectra_max=wl_spectra_max)

			self.fit_wl_range = fit_wl_range
			self.disp_wl_range = disp_wl_range
			self.model_wl_range = model_wl_range
			self.wl_spectra_min = out_input_data_stats['wl_spectra_min']
			self.wl_spectra_max = out_input_data_stats['wl_spectra_max']
			self.N_datapoints = out_input_data_stats['N_datapoints']

		# handle input photometry
		if fit_photometry:
			# handle fit_phot_range
			fit_phot_range = utils.set_fit_phot_range(fit_phot_range=fit_phot_range, filters=filters)

			self.fit_phot_range = fit_phot_range

		# file name to save the bayes results as a pickle
		if bayes_pickle_file is None: self.bayes_pickle_file = f'{model}_bayesian_sampling.pickle'
		else: self.bayes_pickle_file = bayes_pickle_file

		# parameter ranges to select the grid and then used for the posterior priors
		if chi2_pickle_file:
			# params for the best chi-square fit
			params = utils.best_chi2_fits(chi2_pickle_file, N_best_fits=1)['params']

			# grid values around the best fit (which is also a grid node)
			params_models = models.Models(model).params_unique
			params_ranges = {}
			for param in params_models: # for each free parameter in the grid
				params_ranges[param] = utils.find_two_around_node(params_models[param], params[param])
			
		# define the dimensionality of our problem
		if distance is not None: # sample radius distribution
			ndim = len(models.Models(model).free_params) + 1
		else:
			ndim = len(models.Models(model).free_params)
		self.ndim = ndim
 
		# read model grid
		if grid is None: 
			# initialize list
			if fit_spectra:
				grid_spec = [] #  for the resampled, convolved grid appropriate for each input spectrum
				filename_pattern = [] # for the filename pattern adequate for each input spectrum
				for i in range(N_spectra): # for each input observed spectrum
					# convolve the grid to the given res and lam_res for each input spectrum
					# resample the convolved grid to the input wavelengths within the fit range for each input spectrum
					print(f'\nFor input spectrum {i+1} of {N_spectra}')
					if not skip_convolution: # read and convolve original model spectra
						# set filename_pattern to look for model spectra
						filename_pattern.append(models.Models(model).filename_pattern)
						grid_each = utils.read_grid(model=model, model_dir=model_dir, params_ranges=params_ranges, 
						                            convolve=True, res=res[i], lam_res=lam_res[i], fit_wl_range=fit_wl_range[i], 
						                            wl_resample=wl_spectra[i], disp_wl_range=disp_wl_range[i],
						                            path_save_spectra_conv=path_save_spectra_conv)
					else: # read model spectra already convolved to the data resolution
						# set filename_pattern to look for model spectra with the corresponding resolution
						filename_pattern.append(models.Models(model).filename_pattern+f'_R{res[i]}at{lam_res[i]}um.nc')
						grid_each = utils.read_grid(model=model, model_dir=model_dir, params_ranges=params_ranges, 
						                            res=res[i], lam_res=lam_res[i], 
						                            fit_wl_range=fit_wl_range[i], wl_resample=wl_spectra[i], 
						                            skip_convolution=skip_convolution, filename_pattern=filename_pattern[i])
					# add resampled grid for each input spectrum to the same list
					grid_spec.append(grid_each)

			if fit_photometry:
				# select filters within the wavelength range to fit the photometry
				mask_fit = (lambda_eff_SVO>=fit_phot_range.min()) & (lambda_eff_SVO<=fit_phot_range.max())
				phot_fit = phot[mask_fit]
				ephot_fit = ephot[mask_fit]
				filters_fit = filters[mask_fit]
				lambda_eff_SVO_fit = lambda_eff_SVO[mask_fit]
				width_eff_SVO_fit = width_eff_SVO[mask_fit]
				print(f'\n      {len(filters_fit)} of {len(filters)} input valid magnitudes within the fit range "fit_phot_range"')

				# set filename_pattern to look for model spectra
				filename_pattern = [models.Models(model).filename_pattern]
				grid_phot = utils.read_grid_phot(model=model, model_dir=model_dir, params_ranges=params_ranges, filters=filters_fit)
				grid_phot = [grid_phot] # grid as list to follow structure from then fit_spectra

		# define grid depending whether spectra and/or photometry were provided
		if fit_spectra and not fit_photometry:
			grid = grid_spec
		if not fit_spectra and fit_photometry:
			grid = grid_phot 
		if fit_spectra and fit_photometry:
			grid = grid_spec + grid_phot 

		self.grid = grid
		self.filename_pattern = filename_pattern

		# unique values for each model free parameter
		params_unique = grid[0]['params_unique']
		self.params_unique = params_unique

		# define priors from input parameters or grid ranges
		params_priors = {}
		for param in params_unique:
			params_priors[param] = np.array([params_unique[param].min(), params_unique[param].max()])

		if distance is not None:
			if R_range is None: 
				raise Exception('Please provide the "R_range" parameter to sample the radius '
			                    'or remove the "distance" parameter in seda.InputData')
			params_priors['R'] = R_range

		self.params_priors = params_priors

		# cut input spectra to the wavelength region or model coverage for the fit
		if fit_spectra:
			wl_spectra_fit = []
			flux_spectra_fit = []
			eflux_spectra_fit = []
			for i in range(N_spectra): # for each input spectrum
				mask_fit = (wl_spectra[i] >= max(fit_wl_range[i][0], grid[i]['wavelength'].min())) & \
				           (wl_spectra[i] <= min(fit_wl_range[i][1], grid[i]['wavelength'].max()))
	
				wl_spectra_fit.append(wl_spectra[i][mask_fit])
				flux_spectra_fit.append(flux_spectra[i][mask_fit])
				eflux_spectra_fit.append(eflux_spectra[i][mask_fit])

		# set weights for the fit
		out_compute_weights = compute_weights(fit_spectra=fit_spectra, fit_photometry=fit_photometry, weight_label=weight_label, 
		                                      wl_spectra_fit=wl_spectra_fit, width_eff_SVO_fit=width_eff_SVO_fit)
		weight_spec_fit = out_compute_weights['weight_spec_fit']
		weight_phot_fit = out_compute_weights['weight_phot_fit']

		# assign some attributes
		if fit_spectra:
			self.wl_spectra_fit = wl_spectra_fit
			self.flux_spectra_fit = flux_spectra_fit
			self.eflux_spectra_fit = eflux_spectra_fit
			self.weight_spec_fit = weight_spec_fit

		if fit_photometry:
			self.phot_fit = phot_fit
			self.ephot_fit = ephot_fit
			self.filters_fit = filters_fit
			self.lambda_eff_SVO_fit = lambda_eff_SVO_fit
			self.width_eff_SVO_fit = width_eff_SVO_fit
			self.weight_phot_fit = weight_phot_fit

		print('\n   Bayes fit options loaded successfully')

#+++++++++++++++++++++++++++++++++
# useful functions for the classes
#+++++++++++++++++++++++++++++++++
# set the weights based on the input parameter weight_label
def compute_weights(fit_spectra, fit_photometry, weight_label, wl_spectra_fit, width_eff_SVO_fit):
	# set weights for the fit
	if fit_spectra and fit_photometry: # only when both spectra and photometry are provided
		weight_spec_fit = []
		if weight_label=='width': 
			# for spectra
			for k in range(len(wl_spectra_fit)): # for each input observed spectrum
				delta_lam = wl_spectra_fit[k][1:] - wl_spectra_fit[k][:-1] # dispersion of data points
				delta_lam = np.insert(delta_lam, delta_lam.size, delta_lam[-1]) # add an element equal to the last row to keep the same shape as the wl_ref_convolve_resample array
				weight_spec_fit.append(delta_lam)
			# for photometry
			weight_phot_fit = width_eff_SVO_fit
		if weight_label=='dataset': 
			# for spectra
			for k in range(len(wl_spectra_fit)): # for each input observed spectrum
				weight_spec = np.repeat(1./len(wl_spectra_fit[k]), len(wl_spectra_fit[k]))
				weight_spec_fit.append(weight_spec )
			# for photometry
			weight_phot_fit = np.repeat(1./len(width_eff_SVO_fit), len(width_eff_SVO_fit))
		if weight_label=='none': 
			# for spectra
			for k in range(len(wl_spectra_fit)): # for each input observed spectrum
				weight_spec = np.repeat(1, len(wl_spectra_fit[k]))
				weight_spec_fit.append(weight_spec )
			# for photometry
			weight_phot_fit = np.repeat(1, len(width_eff_SVO_fit))
	else: # only spectra or photometry are provided
		# assign the same weight to each data point
		if fit_spectra: # only when both spectra and photometry are provided
			weight_spec_fit = []
			for k in range(len(wl_spectra_fit)): # for each input observed spectrum
				delta_lam = np.repeat(1, len(wl_spectra_fit[k]))
				weight_spec_fit.append(delta_lam)
		if fit_photometry: # only when both spectra and photometry are provided
			weight_phot_fit = np.repeat(1, len(width_eff_SVO_fit))

	# output dictionary with the weights for both spectra and photometry
	out = {'weight_spec_fit': weight_spec_fit, 'weight_phot_fit': weight_phot_fit}

	return out
