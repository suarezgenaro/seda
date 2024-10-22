import numpy as np
from sys import exit

print('\n    SEDA package imported')

# Define input parameters to run SEDA
#+++++++++++++++++++++++++++
class InputData:
	'''
	Description:
	------------
		Define input data for SEDA
	
	Parameters:
	-----------
	- fit_spectra : {``'yes'``, ``'no'``}, optional
		Include (``'yes'``; default) or do not include (``'no'``) spectra.
	- fit_photometry : {``'yes'``, ``'no'``}, optional
		Include (``'yes'``) or do not include (``'no'``; default) photometry.
	- wl_spectra : float array, optional (required when ``fit_spectra=='yes'``)
		Wavelength in um of the spectrum or set of spectra for the fits. 
		When providing more than one spectrum, verify that there is no overlap between the spectra. 
		Provide the multiple spectra as a list (e.g., ``wl_spectra = []``, ``wl_spectra.append(spectrum_1)``, etc.).
		The input list must have the spectra from shorter to longer wavelength coverage
	- flux_spectra : float array, optional
		Fluxes in erg/cm^2/s/A of the input spectrum or spectra.
		Input list for multiple spectra (equivalent to wl_spectra).
	- eflux_spectra : float array, optional
		Fluxes uncertainties in erg/cm^2/s/A of the input spectrum or spectra. 
		Input multiple spectra as a list (equivalent to wl_spectra). 
	- mag_phot : float array, optional (required when ``fit_photometry=='yes'``)
		Magnitudes for the fit
	- emag_phot : float array, optional (required when ``fit_photometry=='yes'``)
		Magnitude uncertainties for the fit. Magnitudes with uncertainties equal to zero are excluded from the fit.
	- filter_phot : float array, optional (required when ``fit_photometry=='yes'``)
		Filters associated to the input magnitudes following SVO filter IDs 
		http://svo2.cab.inta-csic.es/theory/fps/
	- R : float, optional
		Resolution at ``lam_R`` of input spectra (default R=100) to smooth model spectra.
	- lam_R : float, optional
		Wavelength reference (default 2 um) for ``R``.
	- distance : float, optional
		Target distance (in pc) used to derive radius from the scaling factor
	- edistance : float, optional
		Distance error (in pc)

	Example:
	--------
	>>> import seda
	>>> 
	>>> # input spectrum wl_input, flux_input, eflux_input
	>>> wl_spectra = wl_input # in um
	>>> flux_spectra = flux_input # in erg/cm^2/s/A
	>>> eflux_spectra = eflux_input # in erg/cm^2/s/A
	>>> R = 100 # input spectrum resolution
	>>> lam_R = 2.0 # (um) wavelength reference for R
	>>> my_data = seda.InputData(wl_spectra=wl_spectra, flux_spectra=flux_spectra, eflux_spectra=eflux_spectra, R=R, lam_R=lam_R)
		Input data loaded successfully
	'''

	def __init__(self, 
		fit_spectra=True, fit_photometry=False, 
		wl_spectra=None, flux_spectra=None, eflux_spectra=None, 
		mag_phot=None, emag_phot=None, filter_phot=None,
		R=100, lam_R=2, distance=None, edistance=None):	

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
			# if R and lam_R are scalars, convert them into array
			if not isinstance(R, np.ndarray): R = np.array([R])
			if not isinstance(lam_R, np.ndarray): lam_R = np.array([lam_R])

			# when one spectrum is given, convert the spectrum into a list
			if not isinstance(wl_spectra, list): wl_spectra = [wl_spectra]
			if not isinstance(flux_spectra, list): flux_spectra = [flux_spectra]
			if not isinstance(eflux_spectra, list): eflux_spectra = [eflux_spectra]

		# handle input parameters
		# remove NaN values
		for i in range(N_spectra):
			mask_nonan = (~np.isnan(wl_spectra[i])) & (~np.isnan(flux_spectra[i])) & (~np.isnan(eflux_spectra[i]))
			wl_spectra[i] = wl_spectra[i][mask_nonan]
			flux_spectra[i] = flux_spectra[i][mask_nonan]
			eflux_spectra[i] = eflux_spectra[i][mask_nonan]

		self.R = R
		self.lam_R = lam_R
		self.wl_spectra = wl_spectra
		self.flux_spectra = flux_spectra
		self.eflux_spectra = eflux_spectra

		print('\nInput data loaded successfully')

#+++++++++++++++++++++++++++
class ModelOptions:
	'''
	Description:
	------------
		Define model options for SEDA

	Parameters:
	-----------
	- model : {``'Sonora_Diamondback'``, ``'Sonora_Elf_Owl'``, ``'LB23'``, ``'Sonora_Cholla'``, ``'Sonora_Bobcat'``, ``'ATMO2020'``, ``'BT-Settl'``, ``'SM08'``}
		Atmospheric models used in the comparison. Available models: 
			- ``'Sonora_Diamondback'`` : cloudy (silicate clouds) atmospheric models assuming chemical equilibrium but considering the effect of both clouds and metallicity by Morley et al. (2024). Paper: https://ui.adsabs.harvard.edu/abs/2024arXiv240200758M/abstract. Data: https://zenodo.org/records/12735103
				Parameter coverage: 
					- wavelength = [0.3, 250] um
					- Teff = [900, 2400] K in steps of 100 K
					- logg = [3.5, 5.5] in steps of 0.5
					- [M/H] = [-0.5, 0.5] (cgs) in steps of 0.5
					- fsed = 1, 2, 3, 4, 8, nc
			- ``'Sonora_Elf_Owl'`` : models with atmospheric mixing and chemical disequilibrium with varying metallicity and C/O by Mukherjee et al. (2024). Paper: https://ui.adsabs.harvard.edu/abs/2024ApJ...963...73M/abstract. Data for L-type models: https://zenodo.org/records/10385987. Data for T-type models: https://zenodo.org/records/10385821. Data for Y-type models: https://zenodo.org/records/10381250.
				Parameter coverage: 
					- wavelength = [0.6, 15] um
					- Teff = [275, 2400] K in steps: 25 K for 275-600 K, 50 K for 600-1000 K, and 100 K for 1000-2400 K
					- logg = [3.25, 5.50] in steps of 0.25 dex
					- logKzz = 2, 4, 7, 8, and 9 (Kzz in cm2/s)
					- [M/H] = [-1.0, 1.0] (cgs) with values of -1.0, -0.5, +0.0, +0.5, +0.7, and +1.0
					- C/O = [0.5, 2.5] with steps of 0.5 (relative to solar C/O, assumed as 0.458) (these are the values in the filenames). It corresponds to C/O=[0.22, 1.12] with values of 0.22, 0.458, 0.687, and 1.12 (e.g. 0.5 in the filename means 0.5*0.458=0.22)
			- ``'LB23'`` : cloudy (water clouds) atmospheric models with equilibrium and non-equilibrium chemistry for Y-dwarf atmospheres by Lacy & Burrows (2023). Paper: https://ui.adsabs.harvard.edu/abs/2023ApJ...950....8L/abstract. Data: https://zenodo.org/records/7779180. Extended models: Private communication with the authors.
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
			- ``'Sonora_Cholla'`` : cloudless models with non-equilibrium chemistry due to different eddy diffusion parameters by Karalidi et al. (2021). Paper: https://ui.adsabs.harvard.edu/abs/2021ApJ...923..269K/abstract. Data: https://zenodo.org/records/4450269.
				Parameter coverage: 
					- wavelength = [1, 250] um for Teff>=850 K (plus some with Teff=750 K)
					- wavelength = [0.3, 250] um for Teff<800 K (plus 950K_1780g_logkzz2.spec)
					- Teff = [500, 1300] K in steps of 50 K
					- logg = [3.00, 5.50] in steps of 0.25 (g in cgs)
					- log Kzz=2, 4, and 7
			- ``'Sonora_Bobcat'`` : cloudless models in chemical equilibrium by Marley et al. (2021). Paper: https://ui.adsabs.harvard.edu/abs/2021ApJ...920...85M/abstract. Data: https://zenodo.org/records/5063476.
				Parameter coverage: 
					- wavelength = [0.4, 50] um
					- Teff = [200, 2400] K in steps: 25 K for 200-600 K, 50 K for 600-1000 K, and 100 K for 1000-2400 K
					- logg = [3.25, 5.50] in steps of 0.25 (g in cgs)
					- M/H=-0.5, 0.0, and 0.5
					- C/O = 0.5, 1.0 (solar C/O), and 1.5 for solar metallicity models
					- R = [6000, 200000] (the resolving power varies with wavelength but is otherwise the same for all spectra)
			- ``'ATMO2020'`` : cloudless atmospheric models with chemical and non-chemical equilibrium by Phillips et al. (2020). Paper: https://ui.adsabs.harvard.edu/abs/2020A%26A...637A..38P/abstract. Data: https://noctis.erc-atmo.eu/fsdownload/zyU96xA6o/phillips2020, http://perso.ens-lyon.fr/isabelle.baraffe/ATMO2020/.
				ATMO2020 includes three grid:
						- 'ATMO2020_CEQ': cloudless models with equilibrium chemistry
						- 'ATMO2020_NEQ_weak': cloudless models with non-equilibrium chemistry due to weak vertical mixing (logKzz=4).
						- 'ATMO2020_NEQ_strong': cloudless models with non-equilibrium chemistry due to strong vertical mixing (logKzz=6).
				Parameter coverage: 
					- wavelength = [0.2, 2000] um
					- Teff = [200, 2400] K in steps varying from 25 K to 100 K
					- logg = [2.5, 5.5] in steps of 0.5 (g in cgs)
					- logKzz = 0 (ATMO2020_CEQ), 4 (ATMO2020_NEQ_weak), and 6 (ATMO2020_NEQ_strong)
			- ``'BT-Settl'`` : cloudy models with non-equilibrium chemistry by Allard et al. (2012). Paper: https://ui.adsabs.harvard.edu/abs/2012RSPTA.370.2765A/abstract. Data: http://phoenix.ens-lyon.fr/simulator/.
				Parameter coverage: 
					- wavelength = [1.e-4, 100] um
					- Teff = [200, 4200] K (Teff<=450 K for only logg<=3.5) in steps varying from 25 K to 100 K
					- logg = [2.0, 5.5] in steps of 0.5 (g in cgs)
					- R = [100000, 500000] (the resolving power varies with wavelength)
			- ``'SM08'`` : cloudy models with equilibrium chemistry by Saumon & Marley (2008). Paper: https://ui.adsabs.harvard.edu/abs/2008ApJ...689.1327S. Data: Private communication with the authors.
				Parameter coverage: 
					- wavelength = [0.4, 50] um
					- Teff = [800, 2400] K in steps of 100 K
					- logg = [3.0, 5.5] in steps of 0.5 (g in cgs)
					- fsed = 1, 2, 3, 4
					- R = [100000, 700000] (the resolving power varies with wavelength)
	- model_dir : str or list
		Path to the directory (str or list) or directories (as a list) containing the model spectra (e.g., ``model_dir = ['path_1', 'path_2']``). 
		Avoid using paths with null spaces. 
	- Teff_range : float array
		Minimum and maximum Teff values to select a model grid subset (e.g., ``Teff_range = np.array([Teff_min, Teff_max])``)
	- logg_range : float array
		Minimum and maximum logg values to select a model grid subset
	- R_range: float array, optional (used in ``bayes_fit``)
		Minimum and maximum radius values to sample the posterior for radius. It requires the parameter ``distance`` in `input_parameters.InputData`

	Example:
	--------
	>>> import seda
	>>> 
	>>> model = 'Sonora_Elf_Owl'
	>>> model_dir = ['my_path/output_575.0_650.0/', 'my_path/output_700.0_800.0/'] # folders to look for model spectra
	>>> Teff_range = np.array((600, 800)) # Teff range
	>>> logg_range = np.array((4.0, 5.0)) # logg range
	>>> my_model = seda.ModelOptions(model=model, model_dir=model_dir, logg_range=logg_range, Teff_range=Teff_range)
		Model grid options loaded successfully
	'''

	def __init__(self, model, model_dir, Teff_range, logg_range, R_range=None):

		self.model = model
		models_valid = ['Sonora_Diamondback', 'Sonora_Elf_Owl', 'LB23_all', 'Sonora_Cholla', 
						'Sonora_Bobcat', 'ATMO2020_all', 'BT-Settl', 'SM08']
		if model not in models_valid:
			print(f'Models {model} are not recognized')
			print(f'   the options are {models_valid}')
			exit()
		self.Teff_range = Teff_range
		self.logg_range = logg_range
		self.R_range = R_range

		# when only one directory with models is given
		if not isinstance(model_dir, list): model_dir = [model_dir]
		self.model_dir = model_dir

		print('\nModel grid options loaded successfully')

#+++++++++++++++++++++++++++
class Chi2FitOptions:
	'''
	Description:
	------------
		Define chi2 fit options for SEDA

	Parameters:
	-----------
	- chi2_wl_range : float array, optional
		Minimum and maximum wavelengths in microns where model spectra will be compared to the data. 
		This parameter is used when ``fit_spectra='yes'`` but ignored when only ``fit_photometry='yes'``. 
		Default values are the minimum and the maximum wavelengths of each input spectrum. E.g., ``chi2_wl_range = np.array([chi2_wl_min, chi2_wl_max]``)
	- model_wl_range : float array, optional
		Minimum and maximum wavelength to cut model spectra (to make the code faster). 
		Default values are the same as ``chi2_wl_range`` with a padding to avoid the point below.
		CAVEAT: the selected wavelength range of model spectra must cover the spectrophotometry used in the fit and a bit more (to avoid errors when resampling synthetic spectra using spectres)
	- extinction_free_param : {``'yes'``, ``'no'``}, optional
		Extinction as a free parameter: 
			- ``'no'``: (default) null extinction is assumed and it will not change.
			- ``'yes'``: null extinction is assumed and it varies to minimize chi square.
	- scaling_free_param : {``'yes'``, ``'no'``}, optional
		Scaling as a free parameter: 
			- ``'yes'``: (default) to find the scaling that minimizes chi square for each model
			- ``'no'``: to fix ``scaling`` if radius and distance are known
	- scaling: float, optional (required if ``scaling_free_param='no'``)
		Fixed scaling factor ((R/d)^2, R: object's radius, d: distance to the object) to be applied to model spectra
	- skip_convolution : {``'yes'``, ``'no'``}, optional
		Convolution of model spectra (the slowest process in the code) can (``'yes'``) or cannot (``'no'``; default) be avoided. ``skip_convolution='yes'`` only if ``fit_photometry='yes'`` and ``fit_spectra='no'``. Predetermined synthetic magnitudes in the desired filters are required. 
	- avoid_IR_excess : {``'yes'``, ``'no'``}, optional
		Wavelengths longer than ``IR_excess_limit`` will (``'yes'``) or will not (``'no'``; default) be avoided in the fit in case infrared excesses are expected. 
	- IR_excess_limit : float, optional
		Shortest wavelength at which IR excesses are expected (default 3 um).
	- save_results: {``'yes'``, ``'no'``}, optional
		Save (``'yes'``; default) or do not save (``'no'``)	``seda.chi2_fit`` results
	'''

	def __init__(self, my_data, my_model, 
		chi2_wl_range=None, model_wl_range=None, extinction_free_param='no', 
		scaling_free_param='yes', scaling=None, skip_convolution='no', 
		avoid_IR_excess='no', IR_excess_limit=3, save_results='yes'):

		self.save_results = save_results
		self.scaling_free_param = scaling_free_param
		self.scaling = scaling
		self.extinction_free_param = extinction_free_param
		self.skip_convolution = skip_convolution
		self.avoid_IR_excess = avoid_IR_excess
		self.IR_excess_limit = IR_excess_limit

		# read parameters from other classes
		N_spectra = my_data.N_spectra
		wl_spectra = my_data.wl_spectra
		model = my_model.model

		# define chi2_wl_range when not provided
		if chi2_wl_range is None:
			chi2_wl_range = np.zeros((N_spectra, 2)) # Nx2 array, N:number of spectra and 2 for the minimum and maximum values for each spectrum
			for i in range(N_spectra):
				chi2_wl_range[i,:] = np.array((wl_spectra[i].min(), wl_spectra[i].max()))
		else: # chi2_wl_range is provided
			if len(chi2_wl_range.shape)==1: chi2_wl_range = chi2_wl_range.reshape((1, 2)) # reshape chi2_wl_range array

		# model_wl_range
		if model_wl_range is None:
			model_wl_range = np.array((chi2_wl_range.min()-0.1*chi2_wl_range.min(), 
									   chi2_wl_range.max()+0.1*chi2_wl_range.max()
									   )) # add a pad to have enough spectral coverage in models for the fits

		# when model_wl_range is given and is equal or narrower than chi2_wl_range
		# add padding to model_wl_range to avoid problems with the spectres routine
		# first find the minimum and maximum wavelength from the input spectra
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
		
		if (model_wl_range.min()>=wl_spectra_min):
			model_wl_range[0] = wl_spectra_min-0.1*wl_spectra_min # add padding to shorter wavelengths
		if (model_wl_range.max()<=wl_spectra_max):
			model_wl_range[1] = wl_spectra_max+0.1*wl_spectra_max # add padding to longer wavelengths

		# count the total number of data points in all input spectra
		N_datapoints = 0
		for i in range(N_spectra):
			N_datapoints  = N_datapoints + wl_spectra[i].size

		self.chi2_wl_range = chi2_wl_range
		self.model_wl_range = model_wl_range
		self.wl_spectra_min = wl_spectra_min
		self.wl_spectra_max = wl_spectra_max
		self.N_datapoints = N_datapoints

		# file name to save the chi2 results as a pickle
		self.pickle_file = f'{model}_chi2_minimization.pickle'

		print('\nChi2 fit options loaded successfully')
