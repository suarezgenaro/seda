import numpy as np
from .utils import *
from .synthetic_photometry.synthetic_photometry import *
from sys import exit

print('\n    SEDA package imported')

# Define input parameters to run SEDA
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
		Fluxes in erg/cm^2/s/A of the input spectra.
		Use a list for multiple spectra (similar to ``wl_spectra``).
	- eflux_spectra : float array or list, optional (required if ``fit_spectra``)
		Fluxes uncertainties in erg/cm^2/s/A of the input spectra. 
		Use a list for multiple spectra (similar to ``wl_spectra``).
	- flux_unit : str, optional (default ``'erg/s/cm2/A'``)
		Units of ``flux``: ``'Jy'`` or ``'erg/s/cm2/A'``.
	- mag_phot : float array, optional (required if ``fit_photometry``)
		Magnitudes for the fit
	- emag_phot : float array, optional (required if ``fit_photometry``)
		Magnitude uncertainties for the fit. Magnitudes with uncertainties equal to zero are excluded from the fit.
	- filter_phot : float array, optional (required if ``fit_photometry``)
		Filters associated to the input magnitudes following SVO filter IDs 
		http://svo2.cab.inta-csic.es/theory/fps/
	- res : float, list or array, optional (required if ``fit_spectra``)
		Resolving power (R=lambda/delta(lambda)) at ``lam_res`` of input spectra to smooth model spectra.
		For multiple input spectra, ``res`` should be a list or array with a value for each spectrum.
	- lam_res : float, list or array, optional (required if ``fit_spectra``)
		Wavelength of reference at which ``res`` is given.
		For multiple input spectra, ``lam_res`` should be a list or array with a value for each spectrum.
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
	>>> lam_res = 2.0 # (um) reference wavelength for res
	>>> my_data = seda.InputData(wl_spectra=wl_spectra, flux_spectra=flux_spectra, 
	>>>                          eflux_spectra=eflux_spectra, res=res, lam_res=lam_res)
	    Input data loaded successfully

	Author: Genaro Suárez
	'''

	def __init__(self, fit_spectra=True, fit_photometry=False, wl_spectra=None, 
	    flux_spectra=None, eflux_spectra=None, flux_unit='erg/s/cm2/A', 
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

		# reshape some input parameters and define non-provided parameters in terms of other parameters
		if fit_spectra:
			# if res and lam_res are scalars, convert them into array
			if isinstance(res, (float, int)): res = np.array([res])
			if isinstance(lam_res, (float, int)): lam_res = np.array([lam_res])

			# when one spectrum is given, convert the spectrum into a list
			if not isinstance(wl_spectra, list): wl_spectra = [wl_spectra]
			if not isinstance(flux_spectra, list): flux_spectra = [flux_spectra]
			if not isinstance(eflux_spectra, list): eflux_spectra = [eflux_spectra]

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
		# convert input fluxes to erg/s/cm2/A if provided in Jy
		if flux_unit=='Jy':
			for i in range(N_spectra):
				out_convert_flux = convert_flux(flux=flux_spectra[i], eflux=eflux_spectra[i], 
				                                wl=wl_spectra[i], unit_in='Jy', unit_out='erg/s/cm2/A')
				flux_spectra[i] = out_convert_flux['flux_out']
				eflux_spectra[i] = out_convert_flux['eflux_out']

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
	- model : {``'Sonora_Diamondback'``, ``'Sonora_Elf_Owl'``, ``'LB23'``, ``'Sonora_Cholla'``, ``'Sonora_Bobcat'``, ``'ATMO2020'``, ``'BT-Settl'``, ``'SM08'``}
		Atmospheric models used in the comparison. Available models: 
			- ``'Sonora_Diamondback'`` : cloudy (silicate clouds) atmospheric models assuming chemical equilibrium but considering the effect of both clouds and metallicity by Morley et al. (2024).
				Parameter coverage: 
					- wavelength = [0.3, 250] um
					- Teff = [900, 2400] K in steps of 100 K
					- logg = [3.5, 5.5] in steps of 0.5
					- [M/H] = [-0.5, 0.5] (cgs) in steps of 0.5
					- fsed = 1, 2, 3, 4, 8, nc
			- ``'Sonora_Elf_Owl'`` : models with atmospheric mixing and chemical disequilibrium with varying metallicity and C/O by Mukherjee et al. (2024).
				Parameter coverage: 
					- wavelength = [0.6, 15] um
					- Teff = [275, 2400] K in steps: 25 K for 275-600 K, 50 K for 600-1000 K, and 100 K for 1000-2400 K
					- logg = [3.25, 5.50] in steps of 0.25 dex plus logg=3.0 for Teff=[275-2000], logKzz=8, [M/H]=1.0, and C/O=1.0.
					- logKzz = 2, 4, 7, 8, and 9 (Kzz in cm2/s)
					- [M/H] = -1.0, -0.5, +0.0, +0.5, +0.7, and +1.0 (cgs)
					- C/O = 0.5, 1.0, 1.5, and 2.5 (relative to solar C/O, assumed as 0.458) (these are the values in the filenames). It corresponds to C/O=[0.22, 1.12] with values of 0.22, 0.458, 0.687, and 1.12 (e.g. 0.5 in the filename means 0.5*0.458=0.22)
			- ``'LB23'`` : cloudy (water clouds) atmospheric models with equilibrium and non-equilibrium chemistry for Y-dwarf atmospheres by Lacy & Burrows (2023).
				Parameter coverage in common for all grids:
					- wavelength = [0.5, 300] um with 30,000 frequency points evenly spaced in ln(frequency)
					- R~4340 (average resolving power)
				Parameter coverage for cloudless models:
					- Teff = [200, 600] K in steps of 25 K
					- logg = [3.50, 5.00] in steps of 0.25 (g in cgs)
					- [M/H] = -0.5, 0.0, and 0.5 (Z/Z_sun = 0.316, 1.0, 3.16)
					- logKzz = 6 for non-equilibrium models
				Parameter coverage for cloudy models (there are some additional cloudy atmospheres extending to lower surface gravities and warmer temperatures in some combinations where convergence was easy): 
					- Teff = [200, 400] K (200-350 for Z/Z_sun=3.16) in steps of 25 K 
					- logg = [3.75, 5.00] in steps of 0.25 (g in cgs)
					- [M/H] = -0.5, 0.0, and 0.5 (Z/Z_sun = 0.316, 1.0, 3.16), but some Z/Z_sun=3.16 are missing
					- logKzz = 6 for non-equilibrium models
				Extended models (additions to models in the paper)
					- Teff up to 800 K
					- Hmix (mixing length) = 1.0, 0.1, and 0.01
					- This grid replaces the original one ("The original spectra had an inconsistent wavelength grid and was missing CO2, so new ones are really a replacement.")
			- ``'Sonora_Cholla'`` : cloudless models with non-equilibrium chemistry due to different eddy diffusion parameters by Karalidi et al. (2021).
				Parameter coverage: 
					- wavelength = [1, 250] um for Teff>=850 K (plus some with Teff=750 K)
					- wavelength = [0.3, 250] um for Teff<800 K (plus 950K_1780g_logkzz2.spec)
					- Teff = [500, 1300] K in steps of 50 K
					- logg = [3.00, 5.50] in steps of 0.25 (g in cgs)
					- log Kzz=2, 4, and 7
			- ``'Sonora_Bobcat'`` : cloudless models in chemical equilibrium by Marley et al. (2021).
				Parameter coverage: 
					- wavelength = [0.4, 50] um
					- Teff = [200, 2400] K in steps: 25 K for 200-600 K, 50 K for 600-1000 K, and 100 K for 1000-2400 K
					- logg = [3.25, 5.50] in steps of 0.25 (g in cgs)
					- M/H=-0.5, 0.0, and 0.5
					- C/O = 0.5, 1.0 (solar C/O), and 1.5 for solar metallicity models
					- R = [6000, 200000] (the resolving power varies with wavelength but is otherwise the same for all spectra)
			- ``'ATMO2020'`` : cloudless atmospheric models with chemical and non-chemical equilibrium by Phillips et al. (2020).
				ATMO2020 includes three grid:
						- 'ATMO2020_CEQ': cloudless models with equilibrium chemistry
						- 'ATMO2020_NEQ_weak': cloudless models with non-equilibrium chemistry due to weak vertical mixing (logKzz=4).
						- 'ATMO2020_NEQ_strong': cloudless models with non-equilibrium chemistry due to strong vertical mixing (logKzz=6).
				Parameter coverage: 
					- wavelength = [0.2, 2000] um
					- Teff = [200, 2400] K in steps varying from 25 K to 100 K
					- logg = [2.5, 5.5] in steps of 0.5 (g in cgs)
					- logKzz = 0 (ATMO2020_CEQ), 4 (ATMO2020_NEQ_weak), and 6 (ATMO2020_NEQ_strong)
			- ``'BT-Settl'`` : cloudy models with non-equilibrium chemistry by Allard et al. (2012).
				Parameter coverage: 
					- wavelength = [1.e-4, 1000] um
					- Teff = [200, 7000] K (Teff<=450 K for only logg<=3.5) in steps varying from 20 K to 100 K
					- logg = [2.0, 5.5] in steps of 0.5 (g in cgs)
					- R = [100000, 500000] (the resolving power varies with wavelength)
			- ``'SM08'`` : cloudy models with equilibrium chemistry by Saumon & Marley (2008).
				Parameter coverage: 
					- wavelength = [0.4, 50] um
					- Teff = [800, 2400] K in steps of 100 K
					- logg = [3.0, 5.5] in steps of 0.5 (g in cgs)
					- fsed = 1, 2, 3, 4
					- R = [100000, 700000] (the resolving power varies with wavelength)
	- model_dir : str, list, or array
		Path to the directory (str, list, or array) or directories (as a list or array) containing the model spectra (e.g., ``model_dir = ['path_1', 'path_2']``). 
		Avoid using paths with null spaces. 
	- Teff_range : float array, optional
		Minimum and maximum Teff values to select a model grid subset (e.g., ``Teff_range = np.array([Teff_min, Teff_max])``).
		If not provided, the full Teff range in ``model_dir`` is considered.
	- logg_range : float array, optional
		Minimum and maximum logg values to select a model grid subset.
		If not provided, the full logg range in ``model_dir`` is considered.
	- Z_range : float array, optional
		Minimum and maximum metallicity values to select a model grid subset.
		If not provided, the full Z range in ``model_dir`` is considered, if available in ``model``.
	- logKzz_range : float array, optional
		Minimum and maximum diffusion parameter values to select a model grid subset.
		If not provided, the full logKzz range in ``model_dir`` is considered, if available in ``model``.
	- CtoO_range : float array, optional
		Minimum and maximum C/O ratio values to select a model grid subset.
		If not provided, the full C/O ratio range in ``model_dir`` is considered, if available in ``model``.
	- fsed_range : float array, optional
		Minimum and maximum cloudiness parameter values to select a model grid subset.
		If not provided, the full fsed range in ``model_dir`` is considered, if available in ``model``.
	- R_range: float array, optional (used in ``bayes_fit``)
		Minimum and maximum radius values to sample the posterior for radius. It requires the parameter ``distance`` in `input_parameters.InputData`.
	- path_save_spectra_conv: str, optional
		Directory path to store convolved model spectra. 
		If not provided (default), the convolved spectra will not be saved. 
		If the directory does not exist, it will be created. Otherwise, the spectra will be added to the existing folder.
		The convolved spectra will keep the same original names along with the ``res`` and ``lam_res`` parameters, e.g. 'original_spectrum_name_R100at1um.nc' for ``res=100`` and ``lam_res=1``.
		They will be saved as netCDF with xarray (it produces lighter files compared to normal ASCII files).
	- skip_convolution : {``True``, ``False``}, optional (default ``True``)
		Convolution of model spectra (the slowest process in the code) can (``True``) or cannot (``False``) be avoided. 
		Once the code has be run and the convolved spectra were stored in ``path_save_spectra_conv``, the convolved grid can be reused for other input data with the same resolution as the convolved spectra.
		If 'True', ``model_dir`` should include the previously convolved spectra for ``res`` at ``lam_res`` in ``input_parameters.InputData``. 

	Returns:
	--------
	- Dictionary with all (provided and default) model option parameters plus the following parameters:
		- N_modelpoints : int
			Maximum number of data points in the model spectra.

	Example:
	--------
	>>> import seda
	>>> 
	>>> model = 'Sonora_Elf_Owl'
	>>> model_dir = ['my_path/output_700.0_800.0/', 
	>>>              'my_path/output_850.0_950.0/'] # folders to seek model spectra
	>>> Teff_range = np.array((700, 900)) # Teff range
	>>> logg_range = np.array((4.0, 5.0)) # logg range
	>>> my_model = seda.ModelOptions(model=model, model_dir=model_dir, 
	>>>                              logg_range=logg_range, Teff_range=Teff_range)
	    Model options loaded successfully

	Author: Genaro Suárez
	'''

	def __init__(self, model, model_dir, R_range=None, Teff_range=None, logg_range=None, 
	             Z_range=None, logKzz_range=None, CtoO_range=None, fsed_range=None, 
	             path_save_spectra_conv=None, skip_convolution=False):

		self.model = model
		models_valid = list(available_models().keys())
		if model not in models_valid: raise Exception(f'Models "{model}" are not recognized. Available models are: \n          {models_valid}')
		self.R_range = R_range
		self.Teff_range = Teff_range
		self.logg_range = logg_range
		self.Z_range = Z_range
		self.logKzz_range = logKzz_range
		self.CtoO_range = CtoO_range
		self.fsed_range = fsed_range
		self.path_save_spectra_conv = path_save_spectra_conv
		self.skip_convolution = skip_convolution

		# when only one directory with models is given
		if not isinstance(model_dir, list): model_dir = [model_dir]
		self.model_dir = model_dir

		# get number of model data points
		self.N_modelpoints = model_points(model)

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
	- skip_convolution : {``'yes'``, ``'no'``}, optional (default ``'no'``)
		Convolution of model spectra (the slowest process in the code) can (``'yes'``) or cannot (``'no'``) be avoided. ``skip_convolution='yes'`` only if ``fit_photometry`` and ``fit_spectra=False``. Predetermined synthetic magnitudes in the desired filters are required. 
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
#		self.skip_convolution = skip_convolution
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
		self.R_range = my_model.R_range
		self.Teff_range = my_model.Teff_range
		self.logg_range = my_model.logg_range
		self.Z_range = my_model.Z_range
		self.logKzz_range = my_model.logKzz_range
		self.CtoO_range = my_model.CtoO_range
		self.fsed_range = my_model.fsed_range
#		self.save_convolved_spectra = my_model.save_convolved_spectra
		self.path_save_spectra_conv = my_model.path_save_spectra_conv
		self.skip_convolution = my_model.skip_convolution
		self.model_dir = my_model.model_dir
		self.N_modelpoints = my_model.N_modelpoints

		# extract parameters for convenience
		N_spectra = my_data.N_spectra
		wl_spectra = my_data.wl_spectra
		model = my_model.model

#		# define fit_wl_range when not provided
#		if fit_wl_range is None:
#			fit_wl_range = np.zeros((N_spectra, 2)) # Nx2 array, N:number of spectra and 2 for the minimum and maximum values for each spectrum
#			for i in range(N_spectra):
#				fit_wl_range[i,:] = np.array((wl_spectra[i].min(), wl_spectra[i].max()))
#		else: # fit_wl_range is provided
#			if len(fit_wl_range.shape)==1: fit_wl_range = fit_wl_range.reshape((1, 2)) # reshape fit_wl_range array
#
#		# model_wl_range
#		if model_wl_range is None:
#			model_wl_range = np.array((fit_wl_range.min()-0.1*fit_wl_range.min(), 
#									   fit_wl_range.max()+0.1*fit_wl_range.max()
#									   )) # add a pad to have enough spectral coverage in models for the fits
#
#		# when model_wl_range is given and is equal or narrower than fit_wl_range
#		# add padding to model_wl_range to avoid problems with the spectres routine
#		# first find the minimum and maximum wavelength from the input spectra
#		min_tmp1 = min(wl_spectra[0])
#		for i in range(N_spectra):
#			min_tmp2 = min(wl_spectra[i])
#			if min_tmp2<min_tmp1: 
#				wl_spectra_min = min_tmp2
#				min_tmp1 = min_tmp2
#			else: 
#				wl_spectra_min = min_tmp1
#		max_tmp1 = max(wl_spectra[0])
#		for i in range(N_spectra):
#			max_tmp2 = max(wl_spectra[i])
#			if max_tmp2>max_tmp1:
#				wl_spectra_max = max_tmp2
#				max_tmp1 = max_tmp2
#			else:
#				wl_spectra_max = max_tmp1
#		
#		if (model_wl_range.min()>=wl_spectra_min):
#			model_wl_range[0] = wl_spectra_min-0.1*wl_spectra_min # add padding to shorter wavelengths
#		if (model_wl_range.max()<=wl_spectra_max):
#			model_wl_range[1] = wl_spectra_max+0.1*wl_spectra_max # add padding to longer wavelengths
#
#		# count the total number of data points in all input spectra
#		N_datapoints = 0
#		for i in range(N_spectra):
#			N_datapoints  = N_datapoints + wl_spectra[i].size

		# number of data points in the input spectra
		out_input_data_stats = input_data_stats(wl_spectra=wl_spectra, N_spectra=N_spectra)
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
	- 'chi2_pickle_file' : dictionary, optional
		Pickle file with a dictionary with the results from the chi-square minimization by ``chi2_fit.chi2``.
		If given, the parameters from the best chi-square fits will be used to define priors to estimate posteriors. 
		Otherwise, the grid parameters and radius are constrained by the input ranges. 
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
	             chi2_pickle_file=None, bayes_pickle_file=None,
		         grid=None, dynamic_sampling=True, nlive=500, save_results=True):

		#self.logKzz_range = logKzz_range
		#self.Z_range = Z_range
		#self.CtoO_range = CtoO_range
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
		self.R_range = my_model.R_range
		self.Teff_range = my_model.Teff_range
		self.logg_range = my_model.logg_range
		self.Z_range = my_model.Z_range
		self.logKzz_range = my_model.logKzz_range
		self.CtoO_range = my_model.CtoO_range
		self.fsed_range = my_model.fsed_range
		self.model_dir = my_model.model_dir
		self.N_modelpoints = my_model.N_modelpoints

		# extract parameters for convenience
		N_spectra = my_data.N_spectra
		wl_spectra = my_data.wl_spectra
		distance = my_data.distance
		res = my_data.res
		lam_res = my_data.lam_res
		model = my_model.model
		model_dir = my_model.model_dir
		R_range = my_model.R_range
		Teff_range = my_model.Teff_range
		logg_range = my_model.logg_range
		Z_range = my_model.Z_range
		logKzz_range = my_model.logKzz_range
		CtoO_range = my_model.CtoO_range
		fsed_range = my_model.fsed_range

		# number of data points in the input spectra
		out_input_data_stats = input_data_stats(wl_spectra=wl_spectra, N_spectra=N_spectra)
		wl_spectra_min = out_input_data_stats['wl_spectra_min']
		wl_spectra_max = out_input_data_stats['wl_spectra_max']
		N_datapoints = out_input_data_stats['N_datapoints']

		# handle fit_wl_range
		fit_wl_range = set_fit_wl_range(fit_wl_range=fit_wl_range, N_spectra=N_spectra, wl_spectra=wl_spectra)

		# handle model_wl_range
		model_wl_range = set_model_wl_range(model_wl_range=model_wl_range, wl_spectra_min=wl_spectra_min, wl_spectra_max=wl_spectra_max)

		self.fit_wl_range = fit_wl_range
		self.model_wl_range = model_wl_range
		self.wl_spectra_min = out_input_data_stats['wl_spectra_min']
		self.wl_spectra_max = out_input_data_stats['wl_spectra_max']
		self.N_datapoints = out_input_data_stats['N_datapoints']

		# file name to save the chi2 results as a pickle
		if bayes_pickle_file is None: self.bayes_pickle_file = f'{model}_bayesian_sampling.pickle'
		else: self.bayes_pickle_file = bayes_pickle_file

		# parameter ranges for the posteriors
		if chi2_pickle_file:
			out_param_ranges_sampling = param_ranges_sampling(chi2_pickle_file)
			Teff_range_prior = out_param_ranges_sampling['Teff_range_prior']
			logg_range_prior = out_param_ranges_sampling['logg_range_prior']
			logKzz_range_prior = out_param_ranges_sampling['logKzz_range_prior']
			Z_range_prior = out_param_ranges_sampling['Z_range_prior']
			CtoO_range_prior = out_param_ranges_sampling['CtoO_range_prior']
		else:
			# I AM HERE: SAMPLING DOESN'T WORK WITH ANY OF THE PARAMS DEFINITIONS BELOW. IT SAYS THERE IS SOMETHING OUT OF BOUNDS
			# define priors from input parameters or grid ranges
			out_grid_ranges = grid_ranges(model) # read grid ranges
			if Teff_range is not None: Teff_range_prior = Teff_range
			else: Teff_range_prior = np.array([out_grid_ranges['Teff'].min(), out_grid_ranges['Teff'].max()])
			if logg_range is not None: logg_range_prior = logg_range
			else: logg_range_prior = np.array([out_grid_ranges['logg'].min(), out_grid_ranges['logg'].max()])
			if logKzz_range is not None: logKzz_range_prior = logKzz_range
			else: logKzz_range_prior = np.array([out_grid_ranges['logKzz'].min(), out_grid_ranges['logKzz'].max()])
			if Z_range is not None: Z_range_prior = Z_range
			else: Z_range_prior = np.array([out_grid_ranges['Z'].min(), out_grid_ranges['Z'].max()])
			if CtoO_range is not None: CtoO_range_prior = CtoO_range
			else: CtoO_range_prior = np.array([out_grid_ranges['CtoO'].min(), out_grid_ranges['CtoO'].max()])

			# TEST: give a padding on both edges of each parameter
#			Teff_range_prior = np.array([1.05*Teff_range_prior[0], 0.95*Teff_range_prior[1]])
#			logg_range_prior = np.array([1.05*logg_range_prior[0], 0.95*logg_range_prior[1]])
#			logKzz_range_prior = np.array([1.05*logKzz_range_prior[0], 0.95*logKzz_range_prior[1]])
#			Z_range_prior = np.array([1.05*Z_range_prior[0], 0.95*Z_range_prior[1]])
#			CtoO_range_prior = np.array([1.05*CtoO_range_prior[0], 0.95*CtoO_range_prior[1]])

#			# TEST: define ranges as lists as when chi2_pickle_file is True
#			Teff_range_prior = [Teff_range[0], Teff_range[1]]
#			logg_range_prior = [logg_range[0], logg_range[1]]
#			out_grid_ranges = grid_ranges(model) # read grid ranges
#			logKzz_range_prior = [out_grid_ranges['logKzz'].min(), out_grid_ranges['logKzz'].max()]
#			Z_range_prior = [out_grid_ranges['Z'].min(), out_grid_ranges['Z'].max()]
#			CtoO_range_prior = [out_grid_ranges['CtoO'].min(), out_grid_ranges['CtoO'].max()]

#			Teff_range_prior = [525.0, 625.0]
#			logg_range_prior = [3.4981880270062007, 3.9981880270062007]
#			logKzz_range_prior = [2.5, 5.5]
#			Z_range_prior = [-1.0, -0.75]
#			CtoO_range_prior = [1.0, 2.0]
#
##			SOLUTION: CUT THE LOGG RANGE FROM THE RANGE 3.0-5.5 to 3.3-5.5. UNDERSTAND WHY 3.0 DOESN'T WORK BUT 3.3 DOES!!!!!!!!!!!!!!!!!
#			Teff_range_prior = [300, 700]

#			logg_range_prior = [3.3, 5.5]

		#if distance_label=='yes':
		if distance is not None:
			if R_range is None: raise Exception('Please provide the "R_range" parameter')
			R_range_prior = R_range
			self.R_range_prior = R_range_prior

		self.Teff_range_prior = Teff_range_prior
		self.logg_range_prior = logg_range_prior
		self.logKzz_range_prior = logKzz_range_prior
		self.Z_range_prior = Z_range_prior
		self.CtoO_range_prior = CtoO_range_prior

		# define the dimensionality of our problem.
		if distance is not None: # sample radius distribution
			ndim = 6
		else:
			ndim = 5
		self.ndim = ndim

		# read model grid
		if grid is None: 
			# initialize list to save the resampled, convolved grid appropriate for each input spectrum
			grid = []
			for i in range(N_spectra): # for each input observed spectrum
				if model=='Sonora_Elf_Owl':
					# convolve the grid to the given res and lam_res for each input spectrum
					# resample the convolved grid to the input wavelengths within the fit range for each input spectrum
					print(f'\nFor input observed spectrum #{i+1}')
					grid_each = read_grid(model=model, model_dir=model_dir, Teff_range=Teff_range, logg_range=logg_range, 
					                 Z_range=Z_range, logKzz_range=logKzz_range, CtoO_range=CtoO_range, 
					                 fsed_range=fsed_range, convolve=True, res=res[i], lam_res=lam_res[i], 
					                 fit_wl_range=fit_wl_range[i], wl_resample=wl_spectra[i])
				# add resampled grid for each input spectrum to the same list
				grid.append(grid_each)
		self.grid = grid

		print('\n   Bayes fit options loaded successfully')

#+++++++++++++++++++++++++++
# count the total number of data points in all input spectra
def input_data_stats(wl_spectra, N_spectra):

	# count the total number of data points in all input spectra
	N_datapoints = 0
	for i in range(N_spectra):
		N_datapoints  = N_datapoints + wl_spectra[i].size

	# minimum and maximum wavelength from the input spectra
	min_tmp1 = min(wl_spectra[0])
	for i in range(N_spectra):
		min_tmp2 = min(wl_spectra[i])
		if min_tmp2<min_tmp1: 
			wl_spectra_min = min_tmp2
			min_tmp1 = min_tmp2
		else: 
			wl_spectra_min = min_tmp1
	max_tmp1 = max(wl_spectra[0])
	for i in range(N_spectra):
		max_tmp2 = max(wl_spectra[i])
		if max_tmp2>max_tmp1:
			wl_spectra_max = max_tmp2
			max_tmp1 = max_tmp2
		else:
			wl_spectra_max = max_tmp1

	out = {'N_datapoints': N_datapoints, 'wl_spectra_min': wl_spectra_min, 'wl_spectra_max': wl_spectra_max}

	return out
