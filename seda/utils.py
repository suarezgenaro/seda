import numpy as np
import time
import os
import xarray
import pickle
from spectres import spectres
from astropy import units as u
from astropy.io import ascii
from astropy.table import Column, MaskedColumn
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm
from specutils.utils.wcs_utils import vac_to_air
from sys import exit

##########################
def convolve_spectrum(wl, flux, res, lam_res, eflux=None, disp_wl_range=None, convolve_wl_range=None):
	'''
	Description:
	------------
		Convolve a spectrum to a desired resolution at a given wavelength.

	Parameters:
	-----------
	- wl : float array
		Wavelength (any length units) for the spectrum.
	- flux : float array
		Fluxes (any flux units) for the spectrum.
	- eflux : float array, optional
		Flux uncertainties (any flux units) for the spectrum.
	- res : float
		Spectral resolution at ``lam_res`` of input spectra to smooth model spectra.
	- lam_res : float
		Wavelength of reference at which ``res`` is given.
	- disp_wl_range : float array, optional
		Minimum and maximum wavelengths (same units as ``wl``) to compute the median wavelength dispersion of the input spectrum.
		Default values are the minimum and maximum wavelengths in ``wl``.
	- convolve_wl_range : float array, optional
		Minimum and maximum wavelengths (same units as ``wl``) to convolve the input spectrum.
		Default values are the minimum and maximum wavelengths in ``wl``.

	Returns:
	--------
	Dictionary with the convolved spectrum: 
		- ``'wl_conv'`` : wavelengths for the convolved spectrum (equal to input ``wl`` within ``convolve_wl_range``). 
		- ``'flux_conv'`` : convolved fluxes.
		- ``'eflux_conv'`` : convolved flux errors (if ``eflux`` is provided).

	Example:
	--------
	>>> import seda
	>>> from astropy.io import ascii
	>>> 
	>>> # read sample spectrum
	>>> file = '../docs/notebooks/data/IRTF_SpeX_0415-0935.dat'
	>>> spectrum = ascii.read(file)
	>>> wl = spectrum['wl(um)'] # um 
	>>> flux = spectrum['flux(erg/s/cm2/A)'] # erg/s/cm2/A
	>>> eflux = spectrum['eflux(erg/s/cm2/A)'] # erg/s/cm2/A
	>>> # desired resolution
	>>> res, lam_res = 50, 1 # resolution of 50 at 2 um
	>>>
	>>> out_convolve_spectrum = seda.convolve_spectrum(wl=wl, flux=flux, eflux=eflux, 
	>>>                                                res=res, lam_res=lam_res)

	Author: Genaro Suárez
	'''

	# convert input spectrum into numpy arrays if astropy
	wl = astropy_to_numpy(wl)
	flux = astropy_to_numpy(flux)
	eflux = astropy_to_numpy(eflux)

	if (disp_wl_range is None): disp_wl_range = np.array((wl.min(), wl.max())) # define disp_wl_range if not provided
	if (convolve_wl_range is None): convolve_wl_range = np.array((wl.min(), wl.max())) # define convolve_wl_range if not provided

	wl_bin = abs(wl[1:] - wl[:-1]) # wavelength dispersion of the spectrum
	wl_bin = np.append(wl_bin, wl_bin[-1]) # add an element equal to the last row to have same data points

	# define a Gaussian for convolution
	mask_fit = (wl>=disp_wl_range[0]) & (wl<=disp_wl_range[1]) # mask to obtain the median wavelength dispersion
	stddev = (lam_res/res)*(1./np.median(wl_bin[mask_fit]))/ (2.*np.sqrt(2*np.log(2))) # stddev is given in pixels
	if stddev<1: 
		print('   Warning: the input spectrum may have a resolution smaller than the desired one.')
		print('            the spectrum will be convolved but will be essentially the same.')

	gauss = Gaussian1DKernel(stddev=stddev)

	mask_conv = (wl>=convolve_wl_range[0]) & (wl<=convolve_wl_range[1]) # range to convolve the spectrum

	flux_conv = convolve(flux[mask_conv], gauss) # convolve only the selected wavelength range 
	wl_conv = wl[mask_conv] # corresponding wavelength data points for convolved fluxes
	if (eflux is not None): eflux_conv = convolve(eflux[mask_conv], gauss)

	out = {'wl_conv': wl_conv, 'flux_conv': flux_conv}
	if (eflux is not None): out['eflux_conv'] = eflux_conv

	return out

##########################
def scale_synthetic_spectrum(wl, flux, distance, radius):
	'''
	Description:
	------------
		scale model spectrum when distance and radius are known.

	Parameters:
	-----------
	- wl : float array
		Wavelength (any length units) for the spectrum.
	- flux : float array
		Fluxes (any flux units) for the spectrum.
	- radius: float
		Radius in Rjup.
	- distance: float
		Distance in pc.

	Returns:
	--------
	Scaled fluxes.

	'''

	distance_km = distance*u.parsec.to(u.km) # distance in km
	radius_km = radius*u.R_jup.to(u.km) # radius in km
	scaling = (radius_km/distance_km)**2 # scaling = (R/d)^2
	flux_scaled = scaling*flux

	return flux_scaled

##########################
def print_time(time):
	'''
	Description:
	------------
		Print a time in suitable units.

	Parameters:
	-----------
	- time: float
		Time in seconds.

	Returns:
	--------
	Print time in seconds, minutes, or hours as appropriate.

	Author: Genaro Suárez
	'''

	if time<60: ftime, unit = np.round(time), 's' # s
	elif time<3600: ftime, unit = np.round(time/60.,1), 'min' # s
	else: ftime, unit = np.round(time/3600.,1), 'hr' # s

	print(f'      elapsed time: {ftime} {unit}')

##########################
def model_points(model):
	'''
	Description:
	------------
		Maximum number of data points in the model spectra.

	Parameters:
	-----------
	- model : str
		Atmospheric models. See available models in ``input_parameters.ModelOptions``.  

	Returns:
	--------
	- N_modelpoints: int
		Maximum number of data points in the model spectra.

	Author: Genaro Suárez
	'''

	if (model == 'Sonora_Diamondback'):	N_modelpoints = 385466 # number of rows in model spectra (all spectra have the same length)
	if (model == 'Sonora_Elf_Owl'):	N_modelpoints = 193132 # number of rows in model spectra (all spectra have the same length)
	if (model == 'LB23'): N_modelpoints = 30000 # maximum number of rows in model spectra
	if (model == 'Sonora_Cholla'): N_modelpoints = 110979 # maximum number of rows in spectra of the grid
	if (model == 'Sonora_Bobcat'): N_modelpoints = 362000 # maximum number of rows in spectra of the grid
	if (model == 'ATMO2020'): N_modelpoints = 5000 # maximum number of rows of the ATMO2020 model spectra
	if (model == 'BT-Settl'): N_modelpoints = 1291340 # maximum number of rows of the BT-Settl model spectra
	if (model == 'SM08'): N_modelpoints = 184663 # rows of the SM08 model spectra

	return N_modelpoints

##########################
def read_model_spectrum(spectra_name_full, model, model_wl_range=None):
	'''
	Description:
	------------
		Read a desired model spectrum.

	Parameters:
	-----------
	- model : str
		Atmospheric models. See available models in ``input_parameters.ModelOptions``.  
	- spectra_name_full: str
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

	# read model spectra files
	if (model == 'Sonora_Diamondback'):
		spec_model = ascii.read(spectra_name_full, data_start=3, format='no_header')
		wl_model = spec_model['col1'] * u.micron # um (in vacuum?)
		wl_model = vac_to_air(wl_model).value # um in the air
		flux_model = spec_model['col2'] * u.W/u.m**2/u.m # W/m2/m
		flux_model = flux_model.to(u.erg/u.s/u.cm**2/u.angstrom).value # erg/s/cm2/A
	if (model == 'Sonora_Elf_Owl'):
		spec_model = xarray.open_dataset(spectra_name_full) # Sonora Elf Owl model spectra have NetCDF Data Format data
		wl_model = spec_model['wavelength'].data * u.micron # um
		wl_model = wl_model.value
		flux_model = spec_model['flux'].data * u.erg/u.s/u.cm**2/u.cm # erg/s/cm2/cm
		flux_model = flux_model.to(u.erg/u.s/u.cm**2/u.angstrom).value # erg/s/cm2/A
	if (model == 'LB23'):
		spec_model = ascii.read(spectra_name_full)
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
		spec_model = ascii.read(spectra_name_full, data_start=2, format='no_header')
		wl_model = spec_model['col1'] * u.micron # um (in vacuum?)
		wl_model = vac_to_air(wl_model).value # um in the air
		flux_model = spec_model['col2'] * u.W/u.m**2/u.m # W/m2/m
		flux_model = flux_model.to(u.erg/u.s/u.cm**2/u.angstrom).value # erg/s/cm2/A
	if (model == 'Sonora_Bobcat'):
		spec_model = ascii.read(spectra_name_full, data_start=2, format='no_header')
		wl_model = spec_model['col1'] * u.micron # um (in vacuum?)
		wl_model = vac_to_air(wl_model).value # um in the air
		flux_model = spec_model['col2'] * u.erg/u.s/u.cm**2/u.Hz # erg/s/cm2/Hz
		flux_model = flux_model.to(u.erg/u.s/u.cm**2/u.angstrom, equivalencies=u.spectral_density( wl_model * u.micron)).value # erg/s/cm2/A
	if (model == 'ATMO2020'):
		spec_model = ascii.read(spectra_name_full, format='no_header')
		wl_model = spec_model['col1'] * u.micron # um (in vacuum)
		wl_model = vac_to_air(wl_model).value # um in the air
		flux_model = spec_model['col2'] * u.W/u.m**2/u.micron # W/m2/micron
		flux_model = flux_model.to(u.erg/u.s/u.cm**2/u.angstrom).value # erg/s/cm2/A
	if (model == 'BT-Settl'):
		spec_model = ascii.read(spectra_name_full, format='no_header')
		wl_model = (spec_model['col1']*u.angstrom).to(u.micron) # um (in vacuum)
		wl_model = vac_to_air(wl_model).value # um in the air
		flux_model = spec_model['col2'] * u.erg/u.s/u.cm**2/u.Hz # erg/s/cm2/Hz (to an unknown distance). 10**(F_lam + DF) to convert to erg/s/cm2/A
		DF= -8.0
		flux_model = 10**(flux_model.value + DF) # erg/s/cm2/A
	if (model == 'SM08'):
		spec_model = ascii.read(spectra_name_full, data_start=2, format='no_header')
		wl_model = spec_model['col1'] * u.micron # um (in air the alkali lines and in vacuum the rest of the spectra)
		#wl_model = vac_to_air(wl_model).value # um in the air
		wl_model = wl_model.value # um
		flux_model = spec_model['col2'] * u.erg/u.s/u.cm**2/u.Hz # erg/s/cm2/Hz (to an unknown distance)
		flux_model = flux_model.to(u.erg/u.s/u.cm**2/u.angstrom, equivalencies=u.spectral_density(wl_model*u.micron)).value # erg/s/cm2/A

	# sort the array. For BT-Settl is recommended by Allard in her webpage and some models are sorted from higher to smaller wavelengths.
	sort_index = np.argsort(wl_model)
	wl_model = wl_model[sort_index]
	flux_model = flux_model[sort_index]

	# cut the model spectra to the indicated range
	if model_wl_range is not None:
		ind = np.where((wl_model>=(model_wl_range[0])) & (wl_model<=model_wl_range[1]))
		wl_model = wl_model[ind]
		flux_model = flux_model[ind]

	# obtain fluxes in Jy
	flux_model_Jy = (flux_model*u.erg/u.s/u.cm**2/u.angstrom).to(u.Jy, equivalencies=u.spectral_density(wl_model*u.micron)).value

	out = {'wl_model': wl_model, 'flux_model': flux_model, 'flux_model_Jy': flux_model_Jy}

	return out

##########################
def best_chi2_fits(chi2_pickle_file, N_best_fits=1):
	'''
	Description:
	------------
		Read best-fitting model spectra from the chi-square minimization.

	Parameters:
	-----------
	- 'chi2_pickle_file' : dictionary
		Pickle file with a dictionary with the results from the chi-square minimization by ``chi2_fit.chi2``.
	- N_best_fits : int, optional (default 1)
		Number of best model fits to be read.

	Returns:
	--------
	Dictionary with model spectra:
		- ``'spectra_name_best'`` : name of model spectrum
		- ``'chi2_red_fit_best'`` : reduced chi-square
		- ``'wl_model'`` : wavelength (in um) of original model spectra.
		- ``'flux_model'`` : fluxes (in erg/s/cm2/A) of original model spectra.
		- ``'wl_model_conv'`` : wavelength (in um) of convolved model spectra using ``res`` and ``lam_res`` in the input dictionary.
		- ``'flux_model_conv'`` : fluxes (in erg/s/cm2/A) of convolved model spectra.
		- ``'parameters'`` : physical parameters for each spectrum as provided by ``utils.separate_params``, namely ``Teff``, ``logg``, ``Z``, ``logKzz``, ``fsed``, and ``CtoO``, if provided by ``model``.

	Author: Genaro Suárez
	'''

	# open results from the chi square analysis
	with open(chi2_pickle_file, 'rb') as file:
		out_chi2 = pickle.load(file)
	model = out_chi2['model']
	res = out_chi2['res']#[0] # resolution for the first input spectrum
	lam_res = out_chi2['lam_res']#[0] # wavelength reference for the first input spectrum
	N_modelpoints = out_chi2['N_modelpoints']
	fit_wl_range = out_chi2['fit_wl_range'][0] # for the first input spectrum
	spectra_name_full = out_chi2['spectra_name_full']
	spectra_name = out_chi2['spectra_name']
	chi2_red_fit = out_chi2['chi2_red_fit']
	scaling_fit = out_chi2['scaling_fit']

	# select the best fits given by N_best_fits
	sort_ind = np.argsort(chi2_red_fit)
	chi2_red_fit_best = chi2_red_fit[sort_ind][:N_best_fits]
	scaling_fit_best = scaling_fit[sort_ind][:N_best_fits]
	spectra_name_full_best = spectra_name_full[sort_ind][:N_best_fits]
	spectra_name_best = spectra_name[sort_ind][:N_best_fits]

	# read parameters from file name for the best fits
	out_separate_params = separate_params(model=model, spectra_name=spectra_name_best)

	# read best fits
	wl_model = np.zeros((N_modelpoints, N_best_fits))
	flux_model = np.zeros((N_modelpoints, N_best_fits))
	for i in range(N_best_fits):
		out_read_model_spectrum = read_model_spectrum(spectra_name_full_best[i], model)
		wl_model[:,i] = out_read_model_spectrum['wl_model']
		flux_model[:,i] = scaling_fit_best[i] * out_read_model_spectrum['flux_model'] # scaled fluxes

	# convolve spectrum
	wl_model_conv = np.zeros((N_modelpoints, N_best_fits))
	flux_model_conv = np.zeros((N_modelpoints, N_best_fits))
	for i in range(N_best_fits):
		out_convolve_spectrum = convolve_spectrum(wl=wl_model[:,i], flux=flux_model[:,i], lam_res=lam_res, res=res, disp_wl_range=fit_wl_range)
		wl_model_conv[:,i] = out_convolve_spectrum['wl_conv']
		flux_model_conv[:,i] = out_convolve_spectrum['flux_conv']

	out = {'spectra_name_best': spectra_name_best, 'chi2_red_fit_best': chi2_red_fit_best, 'wl_model': wl_model, 
		   'flux_model': flux_model, 'wl_model_conv': wl_model_conv, 'flux_model_conv': flux_model_conv}
	out['parameters'] = out_separate_params

	return out

##################################################
def generate_model_spectrum(Teff, logg, logKzz, Z, CtoO, grid=None, model=None, model_dir=None, Teff_range=None, logg_range=None, save_spectrum=False):
	'''
	Description:
	------------
		Generate a Sonora Elf Owl model spectrum with any combination of parameters within the grid coverage (see ``input_parameters.ModelOptions``).

	Parameters:
	-----------
	- Teff : float 
		Effective temperature (Teff) of the desired spectrum.
	- logg : float 
		Surface gravity (logg) of the desired spectrum.
	- logKzz : float 
		Diffusion parameter (logKzz) of the desired spectrum.
	- Z : float
		Metallicity of the desired spectrum.
	- CtoO : float 
		C/O ration of the desired spectrum.
	- grid : dictionary, optional
		Model grid (``'wavelength'`` and ``'flux'``) generated by ``utils.read_grid`` for interpolations.
		If not provided (default), then the grid is read (``model``, ``model_dir``, ``Teff_range`` and ``logg_range`` must be provided). 
		If provided, the code will skip reading the grid, which will save some time (a few minutes).
	- model : str, optional
		Atmospheric models. See available models in ``input_parameters.ModelOptions``.  
		Required if ``grid`` is not provided.
	- model_dir : str or list, optional
		Path to the directory (str or list) or directories (as a list) containing the model spectra (e.g., ``model_dir = ['path_1', 'path_2']``). 
		Required if ``grid`` is not provided.
	- Teff_range : float array, optional (required if ``grid`` is not provided)
		Minimum and maximum Teff values to select a model grid subset (e.g., ``Teff_range = np.array([Teff_min, Teff_max])``).
	- logg_range : float array, optional (required if ``grid`` is not provided)
		Minimum and maximum logg values to select a model grid subset.
	- save_spectrum : {``True``, ``False``}, optional (default ``False``)
		Save (``'yes'``) or do not save (``'no'``) the generated spectrum as an ascii table.

	Returns:
	--------
	Dictionary with generated model spectrum:
		- ``'wavelength'``: wavelengths in microns for the generated spectrum.
		- ``'flux'``: fluxes in erg/s/cm2/A for the generated spectrum.
		- ``'params'``: input parameters ``Teff``, ``logg``, ``logKzz``, ``Z``, and ``CtoO`` used to generate the spectrum.

	Example:
	--------
	>>> import seda
	>>>
	>>> # models
	>>> model = 'Sonora_Elf_Owl'
	>>> model_dir = ['my_path/Sonora_Elf_Owl/spectra/output_700.0_800.0/',
	>>>              'my_path/Sonora_Elf_Owl/spectra/output_850.0_950.0/']
	>>> 
	>>> # parameters to generate a model spectrum
	>>> Teff, logg, logKzz, Z, CtoO = 765, 4.15, 5.2, 0.2, 1.2
	>>> 
	>>> # Teff and logg ranges to read model grid
	>>> Teff_range = np.array([750, 800])
	>>> logg_range = np.array([4.0, 4.5])
	>>>
	>>> # generate model spectrum
	>>> out = seda.generate_model_spectrum(model=model, model_dir=model_dir, Teff=Teff, 
	>>>                                    logg=logg, logKzz=logKzz, Z=Z, CtoO=CtoO, 
	>>>                                    Teff_range=Teff_range, logg_range=logg_range)

	Author: Genaro Suárez
	'''

	# read model grid if not provided
	if grid is None:
		# verify needed input parameters are provided
		if Teff_range is None: raise Exception('missing Teff range to read a model grid subset')
		if logg_range is None: raise Exception('missing logg range to read a model grid subset')
		if Teff<Teff_range.min() or Teff>Teff_range.max(): raise Exception('Teff value is out of the Teff range') # show a warning if a Teff value is out of the loaded model grid.
		if logg<logg_range.min() or logg>logg_range.max(): raise Exception('logg value is out of the logg range') # show a warning if a logg value is out of the loaded model grid.
		if model is None: raise Exception('missing \'model\' variable')
		if model_dir is None: raise Exception('missing \'model_dir\' variable')

		out_grid_ranges = grid_ranges(model) # read parameters' ranges

		if logKzz<out_grid_ranges['logKzz'].min() or logKzz>out_grid_ranges['logKzz'].max(): raise Exception('logKzz value is out of the grid coverage')
		if Z<out_grid_ranges['Z'].min() or Z>out_grid_ranges['Z'].max(): raise Exception('Z value is out of the grid coverage')
		if CtoO<out_grid_ranges['CtoO'].min() or CtoO>out_grid_ranges['CtoO'].max(): raise Exception('C/O value is out of the grid coverage')

		grid = read_grid(model=model, model_dir=model_dir, Teff_range=Teff_range, logg_range=logg_range)

	# define an interpolating function from the grid
	flux_grid = grid['flux']
	wl_grid = grid['wavelength']
	Teff_grid = grid['Teff']
	logg_grid = grid['logg']
	logKzz_grid = grid['logKzz']
	Z_grid = grid['Z']
	CtoO_grid = grid['CtoO']
	interp_flux = RegularGridInterpolator((Teff_grid, logg_grid, logKzz_grid, Z_grid, CtoO_grid), flux_grid)
	interp_wl = RegularGridInterpolator((Teff_grid, logg_grid, logKzz_grid, Z_grid, CtoO_grid), wl_grid)

	params = ([Teff, logg, logKzz, Z, CtoO])
	spectra_flux = interp_flux(params)[0,:] # to return a 1D array
	spectra_wl = interp_wl(params)[0,:] # to return a 1D array

	# reverse array to sort wavelength from shortest to longest
	ind_sort = np.argsort(spectra_wl)
	spectra_wl = spectra_wl[ind_sort]
	spectra_flux = spectra_flux[ind_sort]

	# store generated spectrum
	if save_spectrum:
		out = open(f'Elf_Owl_Teff{Teff}_logg{logg}_logKzz{logKzz}_Z{Z}_CtoO{CtoO}.dat', 'w')
		out.write('# wavelength(um) flux(erg/s/cm2/A)  \n')
		for i in range(len(spectra_wl)):
			out.write('%11.7f %17.6E \n' %(spectra_wl[i], spectra_flux[i]))
		out.close()

	out = {'wavelength': spectra_wl, 'flux': spectra_flux, 'params': params}

	return out

##################################################
def read_grid(model, model_dir, Teff_range, logg_range, convolve=False, model_wl_range=None, fit_wl_range=None, res=100, lam_res=2, wl_resample=None):
	'''
	Description:
	------------
		Read a model grid constrained by input parameters.

	Parameters:
	-----------
	- model : str
		Atmospheric models. See available models in ``input_parameters.ModelOptions``.  
	- model_dir : str or list
		Path to the directory (str or list) or directories (as a list) containing the model spectra (e.g., ``model_dir = ['path_1', 'path_2']``). 
	- Teff_range : float array
		Minimum and maximum Teff values to select a model grid subset (e.g., ``Teff_range = np.array([Teff_min, Teff_max])``).
	- logg_range : float array
		Minimum and maximum logg values to select a model grid subset.
	- convolve: {``True``, ``False``}, optional (default ``False``)
		Convolve (``'yes'``) or do not convolve (``'no'``) the model grid spectra to the indicated ``res`` at ``lam_res``.
	- res : float, optional (required if ``convolve``).
		Spectral resolution at ``lam_res`` to smooth model spectra.
	- lam_res : float, optional (required if ``convolve``).
		Wavelength of reference for ``res``.
	- model_wl_range : float array (optional)
		Minimum and maximum wavelengths in microns to cut model spectra.
	- fit_wl_range : float array, optional
		Minimum and maximum wavelengths in microns to resample model spectra to the observed spectrum.
	- res : float, optional (default res=100)
		Spectral resolution at ``lam_res`` of input spectra to smooth model spectra.
	- lam_res : float, optional (default 2 um) 
		Wavelength of reference at which ``res`` is given.
	- wl_resample : float array or list, optional
		wavelength data points to resample the grid

	Returns:
	--------
	Dictionary with the model grid:
		- ``'wavelength'`` : wavelengths in microns for the model spectra in the grid.
		- ``'flux'`` : fluxes in erg/s/cm2/A for the model spectra in the grid.
		- ``'Teff'`` : Effective temperature at each grid point.
		- ``'logg'`` : Surface gravity (logg) at each grid point.
		- ``'logKzz'`` : Diffusion parameter (logKzz) at each grid point.
		- ``'Z'`` : Metallicity at each grid point.
		- ``'CtoO'`` : C/O ratio at each grid point.

	Example:
	--------
	>>> import seda
	>>>
	>>> # models
	>>> model = 'Sonora_Elf_Owl'
	>>> model_dir = ['my_path/Sonora_Elf_Owl/spectra/output_700.0_800.0/',
	>>>              'my_path/Sonora_Elf_Owl/spectra/output_850.0_950.0/']
	>>> 
	>>> # Teff and logg ranges to read the model grid
	>>> Teff_range = np.array([750, 800])
	>>> logg_range = np.array([4.0, 4.5])
	>>>
	>>> # read the grid
	>>> out_read_grid = seda.read_grid(model=model, model_dir=model_dir, 
	>>>                                Teff_range=Teff_range, logg_range=logg_range)

	Author: Genaro Suárez
	'''

	ini_time_grid = time.time() # to estimate the time elapsed reading the grid

	# read models in the input folders
	out_select_model_spectra = select_model_spectra(model=model, model_dir=model_dir, Teff_range=Teff_range, logg_range=logg_range)
	spectra_name_full = out_select_model_spectra['spectra_name_full']
	spectra_name = out_select_model_spectra['spectra_name']

	# set model_wl_range in case wl_resample is given
	if wl_resample is not None:
		model_wl_range = set_model_wl_range(model_wl_range=model_wl_range, fit_wl_range=fit_wl_range)

	# all parameters's steps in the grid
	out_grid_ranges = grid_ranges(model)
	
	Teff_grid = out_grid_ranges['Teff']
	logg_grid = out_grid_ranges['logg']
	logKzz_grid = out_grid_ranges['logKzz']
	Z_grid = out_grid_ranges['Z']
	CtoO_grid = out_grid_ranges['CtoO']

	# parameters constraints to read only a part of the grid
	mask_Teff = (Teff_grid>=Teff_range[0]) & (Teff_grid<=Teff_range[1])
	mask_logg = (logg_grid>=logg_range[0]) & (logg_grid<=logg_range[1])
	
	# replace Teff and logg array
	Teff_grid = Teff_grid[mask_Teff]
	logg_grid = logg_grid[mask_logg]

#	# define arrays to save the model grid
#	if wl_resample is not None: 
#		flux_grid = np.zeros((len(Teff_grid), len(logg_grid), len(logKzz_grid), len(Z_grid), len(CtoO_grid), len(wl_resample))) # to save the flux at each grid point
#		wl_grid = np.zeros((len(Teff_grid), len(logg_grid), len(logKzz_grid), len(Z_grid), len(CtoO_grid), len(wl_resample))) # to save the wavelength at each grid point
#		
#	else:
#		N_modelpoints = model_points(model)
#		flux_grid = np.zeros((len(Teff_grid), len(logg_grid), len(logKzz_grid), len(Z_grid), len(CtoO_grid), N_modelpoints)) # to save the flux at each grid point
#		wl_grid = np.zeros((len(Teff_grid), len(logg_grid), len(logKzz_grid), len(Z_grid), len(CtoO_grid), N_modelpoints)) # to save the wavelength at each grid point
	
	# create a tqdm progress bar
	if convolve:
		if wl_resample is not None:
			desc = 'Reading, convolving, and resampling model grid'
		else:
			desc = 'Reading and convolving model grid'
	else:
		if wl_resample is not None:
			desc = 'Reading and resampling model grid'
		else:
			desc = 'Reading model grid'
	grid_bar = tqdm(total=len(spectra_name), desc=desc)

	# read the grid in the constrained ranges
	k = 0
	for i_Teff in range(len(Teff_grid)): # iterate Teff
		for i_logg in range(len(logg_grid)): # iterate logg
			for i_logKzz in range(len(logKzz_grid)): # iterate logKzz
				for i_Z in range(len(Z_grid)): # iterate Z
					for i_CtoO in range(len(CtoO_grid)): # iterate C/O ration
						# update the progress bar
						grid_bar.update(1)

						# read the spectrum for each parameter combination
						# file name uses g instead of logg
						if (logg_grid[i_logg]==3.25): g_grid = 17.0
						if (logg_grid[i_logg]==3.50): g_grid = 31.0
						if (logg_grid[i_logg]==3.75): g_grid = 56.0
						if (logg_grid[i_logg]==4.00): g_grid = 100.0
						if (logg_grid[i_logg]==4.25): g_grid = 178.0
						if (logg_grid[i_logg]==4.50): g_grid = 316.0
						if (logg_grid[i_logg]==4.75): g_grid = 562.0
						if (logg_grid[i_logg]==5.00): g_grid = 1000.0
						if (logg_grid[i_logg]==5.25): g_grid = 1780.0
						if (logg_grid[i_logg]==5.50): g_grid = 3160.0

						# name of spectrum with parameters in the iteration
						spectrum_name = f'spectra_logzz_{logKzz_grid[i_logKzz]}_teff_{Teff_grid[i_Teff]}_grav_{g_grid}_mh_{Z_grid[i_Z]}_co_{CtoO_grid[i_CtoO]}.nc'
						# look into the selected spectra to find the full path for the spectrum above
						spectrum_name_full = [x for x in spectra_name_full if x.endswith(spectrum_name)][0]

						if spectrum_name_full: # if there is a spectrum with the parameters in the iteration
							# read spectrum from each combination of parameters
							out_read_model_spectrum = read_model_spectrum(spectra_name_full=spectrum_name_full, model=model, model_wl_range=model_wl_range)
							wl_model = out_read_model_spectrum['wl_model'] # in um
							flux_model = out_read_model_spectrum['flux_model'] # in erg/s/cm2/A
							
							# convolve model spectrum to the resolution of the input observed spectra
							if convolve:
								out_convolve_spectrum = convolve_spectrum(wl=wl_model, flux=flux_model, res=res, lam_res=lam_res)
								wl_model = out_convolve_spectrum['wl_conv']
								flux_model = out_convolve_spectrum['flux_conv']

							# resample convolved model spectrum to the wavelength data points in the observed spectra
							if wl_resample is not None:
								flux_model = spectres(wl_resample, wl_model, flux_model)
								wl_model = wl_resample

							# save flux at each combination
							if i_Teff==0 and i_logg==0 and i_logKzz==0 and i_Z==0 and i_CtoO==0:
								# define arrays to save the model grid
								flux_grid = np.zeros((len(Teff_grid), len(logg_grid), len(logKzz_grid), len(Z_grid), len(CtoO_grid), len(wl_model))) # to save the flux at each grid point
								wl_grid = np.zeros((len(Teff_grid), len(logg_grid), len(logKzz_grid), len(Z_grid), len(CtoO_grid), len(wl_model))) # to save the wavelength at each grid point

								# save first read model spectrum
								flux_grid[i_Teff, i_logg, i_logKzz, i_Z, i_CtoO, :len(wl_model)] = flux_model
								wl_grid[i_Teff, i_logg, i_logKzz, i_Z, i_CtoO, :len(wl_model)] = wl_model
							else:
								flux_grid[i_Teff, i_logg, i_logKzz, i_Z, i_CtoO, :len(wl_model)] = flux_model
								wl_grid[i_Teff, i_logg, i_logKzz, i_Z, i_CtoO, :len(wl_model)] = wl_model
						
							#print(len(wl_model))
							## measure the resolution of each model spectrum
							## wavelength dispersion of synthetic spectra
							#Dellam_model = wl_model[1:] - wl_model[:-1] # dispersion of model spectra
							#Dellam_model = np.insert(Dellam_model, Dellam_model.size, Dellam_model[-1]) # add an element equal to the last row to keep the same shape as the wl_model array
							#mask_1 = (wl_model>=1) & (wl_model<=2)
							#mask_2 = (wl_model>=9) & (wl_model<=10)
							#print('   ',1.5/np.median(Dellam_model[mask_1]), 9.5/np.median(Dellam_model[mask_2]))
						else:
						    print(f'There is no spectrum: {spectrum_name}') # no spectrum with that name
						k += 1
	# close the progress bar
	grid_bar.close()

	fin_time_grid = time.time()
	print_time(fin_time_grid-ini_time_grid)

	out = {'wavelength': wl_grid, 'flux': flux_grid, 'Teff': Teff_grid, 'logg': logg_grid, 'logKzz': logKzz_grid, 'Z': Z_grid, 'CtoO': CtoO_grid}

	return out

##########################
def grid_ranges(model):
	'''
	Description:
	------------
		Read coverage of parameters in a model grid.

	Parameters:
	-----------
	- model : str
		Atmospheric models. See available models in ``input_parameters.ModelOptions``.  

	Returns:
	--------
	Dictionary with model parameter coverage:
		- ``'Teff'`` : effective temperature.
		- ``'logg'`` : surface gravity (logg).
		- ``'logKzz'`` : (if provided by ``model``) diffusion parameter (logKzz).
		- ``'Z'`` : (if provided by ``model``) metallicity at each grid point.
		- ``'CtoO'`` : (if provided by ``model``) C/O ratio at each grid point.

	Example:
	--------
	>>> import seda
	>>>
	>>> # models
	>>> model = 'Sonora_Elf_Owl'
	>>>
	>>> # read model parameters
	>>> out_grid_ranges = seda.grid_ranges(model)

	Author: Genaro Suárez
	'''

	if (model=='Sonora_Elf_Owl'):
		# Teff
		Teff_range1 = np.arange(275., 600.+25, 25)
		Teff_range2 = np.arange(650., 1000.+50, 50)
		Teff_range3 = np.arange(1100., 2400.+100, 100)
		Teff = np.concatenate((Teff_range1, Teff_range2, Teff_range3)) # K
		# logg
		logg = np.arange(3.25, 5.50+0.25, 0.25) # dex (g in cgs)
		#logKzz
		logKzz = np.array((2.0, 4.0, 7.0, 8.0, 9.0)) # dex (Kzz in cgs)
		# Z or [M/H]
		Z = np.array((-1.0, -0.5, 0.0, 0.5, 0.7, 1.0)) # cgs
		# C/O ratio
		CtoO = np.array((0.5, 1.0, 1.5, 2.5)) # relative to solar C/O

#	if (model=='ATMO2020'):


	out = {'Teff': Teff, 'logg': logg, 'logKzz': logKzz, 'Z': Z, 'CtoO': CtoO}

	return out

##########################
def param_ranges_sampling(chi2_pickle_file, N_best_fits=3):
	'''
	Description:
	------------
		Tolerance around the parameters for the best-fitting spectra to define the parameter ranges to estimate posteriors.

	Parameters:
	-----------
	- 'chi2_pickle_file' : dictionary
		Pickle file with a dictionary with the results from the chi-square minimization by ``chi2_fit.chi2``.
	- N_best_fits : int, optional (default 3)
		Number of best model fits to be read.

	Returns:
	--------
	Dictionary with parameter ranges around the parameters of the best model fits. It considers half of the biggest step for each parameter around the median parameters from the best ``N_best_fits`` model fits.
		``'Teff_range'`` : effective temperature range.
		``'logg_range'`` : surface gravity (logg) range.
		``'logKzz_range'`` : (if provided by ``model``) diffusion parameter (logKzz) range.
		``'Z_range'`` : (if provided by ``model``) metallicity range.
		``'CtoO_range'`` : (if provided by ``model``) C/O ratio range.

	Example:
	--------
	>>> import seda
	>>>
	>>> # pickle file
	>>> chi2_pickle_file = model+'_chi2_minimization.pickle'
	>>> # read parameters' tolerance around the best model fits
	>>> out_param_ranges_sampling = seda.param_ranges_sampling(chi2_pickle_file=chi2_pickle_file)

	Author: Genaro Suárez
	'''

	# open results from the chi square analysis
	with open(chi2_pickle_file, 'rb') as file:
		out_chi2 = pickle.load(file)

	model = out_chi2['model']

	if (model=='Sonora_Elf_Owl'):
		ind_best_fit = np.argsort(out_chi2['chi2_red_fit'])[:N_best_fits] # index of the three best-fitting spectra
		# median parameter values from the best-fitting models
		Teff_chi2 = np.median(out_chi2['Teff'][ind_best_fit])
		logg_chi2 = np.median(out_chi2['logg'][ind_best_fit])
		logKzz_chi2 = np.median(out_chi2['logKzz'][ind_best_fit])
		Z_chi2 = np.median(out_chi2['Z'][ind_best_fit])
		CtoO_chi2 = np.median(out_chi2['CtoO'][ind_best_fit])

		# whole grid parameter ranges to avoid trying to generate a spectrum out of the grid
		out_grid_ranges = grid_ranges(model)

		# define the search tolerance for each parameter
		Teff_search = (out_grid_ranges['Teff'][1:]-out_grid_ranges['Teff'][:-1]).max() / 2. # K (half of the biggest Teff step)
		logg_search = (out_grid_ranges['logg'][1:]-out_grid_ranges['logg'][:-1]).max() / 2. # dex (half of the biggest logg step)
		logKzz_search = (out_grid_ranges['logKzz'][1:]-out_grid_ranges['logKzz'][:-1]).max() / 2. # dex (half of the biggest logKzz step)
		Z_search = (out_grid_ranges['Z'][1:]-out_grid_ranges['Z'][:-1]).max() / 2. # cgs (half of the biggest Z step)
		CtoO_search = (out_grid_ranges['CtoO'][1:]-out_grid_ranges['CtoO'][:-1]).max() / 2. # relative to solar C/O (half of the biggest C/O step)
	
		# define the parameter ranges around the median from the best model fits
		Teff_range_prior = [max(Teff_chi2-Teff_search, min(out_grid_ranges['Teff'])), 
								min(Teff_chi2+Teff_search, max(out_grid_ranges['Teff']))]
		logg_range_prior = [max(logg_chi2-logg_search, min(out_grid_ranges['logg'])), 
								min(logg_chi2+logg_search, max(out_grid_ranges['logg']))]
		logKzz_range_prior = [max(logKzz_chi2-logKzz_search, min(out_grid_ranges['logKzz'])), 
								min(logKzz_chi2+logKzz_search, max(out_grid_ranges['logKzz']))]
		Z_range_prior = [max(Z_chi2-Z_search, min(out_grid_ranges['Z'])), 
								min(Z_chi2+Z_search, max(out_grid_ranges['Z']))]
		CtoO_range_prior = [max(CtoO_chi2-CtoO_search, min(out_grid_ranges['CtoO'])), 
								min(CtoO_chi2+CtoO_search, max(out_grid_ranges['CtoO']))]

		out = {'Teff_range_prior': Teff_range_prior, 'logg_range_prior': logg_range_prior, 'logKzz_range_prior': logKzz_range_prior, 'Z_range_prior': Z_range_prior, 'CtoO_range_prior': CtoO_range_prior}

	return out

##########################
def best_bayesian_fit(bayes_pickle_file, grid=None, save_spectrum=False):
	'''
	Description:
	------------
		Generate model spectrum with the posterior parameters.

	Parameters:
	-----------
	- 'bayes_pickle_file' : dictionary
		Dictionary with ``bayes_fit.bayes`` output: 
			- Output dictionary by ``input_parameters.BayesOptions`` 
			- Dynesty output.
	- grid : dictionary, optional
		Model grid (``'wavelength'`` and ``'flux'``) generated by ``utils.read_grid`` for interpolations.
		If not provided (default), then the grid is read (``model``, ``model_dir``, ``Teff_range`` and ``logg_range`` must be provided). 
		If provided, the code will skip reading the grid, which will save some time (a few minutes).
	- save_spectrum : {``True``, ``False``}, optional (default ``False``)
		Save (``True``) or do not save (``False``) best model fit from the nested sampling.

	Returns:
	--------
	- '``model``\_best\_fit\_bayesian\_sampling.dat' : ascii table
		Table with best model fit (if ``save_spectrum``).
	- dictionary
		Dictionary with the best model fit from the nested sampling:
			- ``'wl_spectra'`` : wavelength in um of the input spectra.
			- ``'flux_spectra'`` : fluxes in erg/cm^2/s/A of the input spectra.
			- ``'eflux_spectra'`` : flux uncertainties in erg/cm^2/s/A of the input spectra.
			- ``'Teff_med'`` : median effective temperature (in K) from the posterior.
			- ``'logg_med'`` : median surface gravity (logg) from the posterior.
			- ``'logKzz_med'`` : (if provided by ``model``) median diffusion parameter from the posterior.
			- ``'Z_med'`` : (if provided by ``model``) median metallicity from the posterior.
			- ``'CtoO_med'`` : (if provided by ``model``) median C/O ratio from the posterior.
			- ``'R_med'`` : (if ``distance`` is provided) median radio from the posterior.
			- ``'wl_mod'`` : wavelength in um of the best model fit.
			- ``'flux_mod'`` : fluxes in erg/cm^2/s/A of the best model fit.
			- ``'wl_mod_conv'`` : wavelength in um of the best convolved model spectrum to the resolution of the input spectra.
			- ``'flux_mod_conv'`` : fluxes in erg/cm^2/s/A of the best convolved model spectrum.
			- ``'wl_mod_conv_resam'`` : wavelength in um of the best convolved, resampled model spectrum to the input spectra data points.
			- ``'flux_mod_conv_resam'`` : fluxes in erg/cm^2/s/A of the best convolved, resampled model spectrum.
			- ``'wl_mod_scaled'`` : (if ``distance`` was provided in ``input_parameters.InputData``) wavelength in um of the best scaled model spectrum.
			- ``'flux_mod_scaled'`` : (if ``distance`` was provided in ``input_parameters.InputData``) fluxes in erg/cm^2/s/A of the best scaled model spectrum.
			- ``'wl_mod_conv_scaled'`` : (if ``distance`` was provided in ``input_parameters.InputData``) wavelength in um of the best scaled, convolved model spectrum.
			- ``'flux_mod_conv_scaled'`` : (if ``distance`` was provided in ``input_parameters.InputData``) fluxes in erg/cm^2/s/A of the best scaled, convolved model spectrum.
			- ``'wl_mod_conv_scaled_resam'`` : (if ``distance`` was provided in ``input_parameters.InputData``) wavelength in um of the best scaled, convolved, resampled model spectrum.
			- ``'flux_mod_conv_scaled_resam'`` : (if ``distance`` was provided in ``input_parameters.InputData``) fluxes in erg/cm^2/s/A of the best scaled, convolved, resampled model spectrum.

	Author: Genaro Suárez
	'''

	# open results from sampling
	with open(bayes_pickle_file, 'rb') as file:
		out_bayes = pickle.load(file)
	model = out_bayes['my_bayes'].model
	model_dir = out_bayes['my_bayes'].model_dir
	res = out_bayes['my_bayes'].res
	lam_res = out_bayes['my_bayes'].lam_res
	distance = out_bayes['my_bayes'].distance
	wl_spectra_min = out_bayes['my_bayes'].wl_spectra_min
	wl_spectra_max = out_bayes['my_bayes'].wl_spectra_max
	wl_spectra = out_bayes['my_bayes'].wl_spectra
	flux_spectra = out_bayes['my_bayes'].flux_spectra
	eflux_spectra = out_bayes['my_bayes'].eflux_spectra
	N_spectra = out_bayes['my_bayes'].N_spectra

	# compute median values for all parameters
	if model=='Sonora_Elf_Owl':
		Teff_med = np.median(out_bayes['out_dynesty'].samples[:,0]) # Teff values
		logg_med = np.median(out_bayes['out_dynesty'].samples[:,1]) # logg values
		logKzz_med = np.median(out_bayes['out_dynesty'].samples[:,2]) # logKzz values
		Z_med = np.median(out_bayes['out_dynesty'].samples[:,3]) # Z values
		CtoO_med = np.median(out_bayes['out_dynesty'].samples[:,4]) # CtoO values
		if distance is not None: R_med = np.median(out_bayes['out_dynesty'].samples[:,5]) # R values

	# generate spectrum with the median parameter values
	if grid is None:
		# read grid around the median values	
		# first define Teff and logg ranges around the median values
		out_grid_ranges = grid_ranges(model)
		
		# find the grid point before and after a median value
		def find_two_nearest(array, value):
			diff = array - value
			if any(array[diff<0]): near_1 = diff[diff<0].max()+value
			else: near_1 = array.min()
			if any(array[diff>0]): near_2 = diff[diff>0].min()+value
			else: near_2 = array.max()

			return np.array([near_1, near_2])

		Teff_range = find_two_nearest(out_grid_ranges['Teff'], Teff_med)
		logg_range = find_two_nearest(out_grid_ranges['logg'], logg_med)
		
		# read grid
		grid = read_grid(model=model, model_dir=model_dir, Teff_range=Teff_range, logg_range=logg_range)

	# generate synthetic spectrum
	syn_spectrum = generate_model_spectrum(Teff=Teff_med, logg=logg_med, logKzz=logKzz_med, Z=Z_med, CtoO=CtoO_med, grid=grid)
	wl = syn_spectrum['wavelength']
	flux = syn_spectrum['flux']

	# convolve synthetic spectrum
	# (add padding on both edges to avoid issues we using spectres)
	out_convolve_spectrum = convolve_spectrum(wl=wl, flux=flux, lam_res=lam_res, res=res, 
	                                          disp_wl_range=np.array([wl_spectra_min, wl_spectra_max]), 
	                                          convolve_wl_range=np.array([0.99*wl_spectra_min, 1.01*wl_spectra_max]))
	wl_conv = out_convolve_spectrum['wl_conv']
	flux_conv = out_convolve_spectrum['flux_conv']

	# scale synthetic spectrum
	if distance is not None:
		flux_scaled = scale_synthetic_spectrum(wl=wl, flux=flux, distance=distance, radius=R_med)
		flux_conv_scaled = scale_synthetic_spectrum(wl=wl_conv, flux=flux_conv, distance=distance, radius=R_med)
		wl_scaled = wl
		wl_conv_scaled = wl_conv

	# resample synthetic spectra to the observed wavelengths
	flux_conv_resam = [] # initialize list to store resampled, convolved model spectra for each input spectrum
	flux_conv_scaled_resam = [] # initialize list to store resampled, scaled, convolved model spectra for each input spectrum
	for k in range(N_spectra): # for each input observed spectrum
		flux_conv_resam.append(spectres(wl_spectra[k], wl_conv, flux_conv))
		if distance is not None:
			flux_conv_scaled_resam.append(spectres(wl_spectra[k], wl_conv_scaled, flux_conv_scaled))
	wl_conv_resam = wl_spectra
	wl_conv_scaled_resam = wl_spectra

	# save synthetic spectrum
	if save_spectrum:
		#Teff_file = round(Teff_med, 1)
		#logg_file = round(logg_med, 2)
		#logKzz_file = round(logKzz_med,1)
		#Z_file = round(Z_med, 1)
		#CtoO_file = round(CtoO_med, 1)
		#out = open(f'{model}_Teff{Teff_file}_logg{logg_file}_logKzz{logKzz_file}_Z{Z_file}_CtoO{CtoO_file}.dat', 'w')
		out = open(f'{model}_best_fit_bayesian_sampling.dat', 'w')
		out.write('# wavelength(um) flux(erg/s/cm2/A)  \n')
		for i in range(len(wl_conv)):
			out.write('%11.7f %17.6E \n' %(wl_conv[i], flux_conv[i]))
		out.close()

	# output dictionary
	out = {'wl_spectra': wl_spectra, 'flux_spectra': flux_spectra, 'eflux_spectra': eflux_spectra, 'wl_mod': wl, 'flux_mod': flux, 'wl_mod_conv': wl_conv, 'flux_mod_conv': flux_conv, 
	       'wl_mod_conv_resam': wl_conv_resam, 'flux_mod_conv_resam': flux_conv_resam, 'Teff_med': Teff_med, 'logg_med': logg_med, 'logKzz_med': logKzz_med, 
	       'Z_med': Z_med, 'CtoO_med': CtoO_med}
	if distance is not None:
		out['R_med'] = R_med
		out['wl_mod_scaled'] = wl_scaled
		out['flux_mod_scaled'] = flux_scaled
		out['wl_mod_conv_scaled'] = wl_conv_scaled
		out['flux_mod_conv_scaled'] = flux_conv_scaled
		out['wl_mod_conv_scaled_resam'] = wl_conv_scaled_resam
		out['flux_mod_conv_scaled_resam'] = flux_conv_scaled_resam

	return out

##########################
def select_model_spectra(model, model_dir, Teff_range=None, logg_range=None, Z_range=None, 
                         logKzz_range=None, CtoO_range=None, fsed_range=None):
	'''
	Description:
	------------
		Select model spectra from the indicated models and meeting given parameters ranges.

	Parameters:
	-----------
	- model : str
		Atmospheric models. See available models in ``input_parameters.ModelOptions``.  
	- model_dir : str or list
		Path to the directory (str or list) or directories (as a list) containing the model spectra (e.g., ``model_dir = ['path_1', 'path_2']``). 
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
		
	Returns:
	--------
	Dictionary with the parameters:
		- ``spectra_name``: selected model spectra names.
		- ``spectra_name_full``: selected model spectra names with full path.

	Example:
	--------
	>>> import seda
	>>> 
	>>> model = 'Sonora_Elf_Owl'
	>>> model_dir = ['my_path/output_575.0_650.0/', 
	>>>              'my_path/output_700.0_800.0/'] # folders to seek model spectra
	>>> Teff_range = np.array((700, 900)) # Teff range
	>>> logg_range = np.array((4.0, 5.0)) # logg range
	>>> out = seda.select_model_spectra(model=model, model_dir=model_dir,
	>>>                                 Teff_range=Teff_range, logg_range=logg_range)

	Author: Genaro Suárez
	'''

	# make sure model_dir is a list
	if isinstance(model_dir, str): model_dir = [model_dir]

	# to store files in model_dir
	files = [] # with full path
	files_short = [] # only spectra names
	for i in range(len(model_dir)):
		files_model_dir = os.listdir(model_dir[i])
		files_model_dir.sort() # just to sort the files wrt their names
		for file in files_model_dir:
			files.append(model_dir[i]+file)
			files_short.append(file)

	# read Teff and logg from each model spectrum
	out_separate_params = separate_params(model=model, spectra_name=files_short)

	# select spectra within the desired Teff and logg ranges
	spectra_name_full = [] # full name with path
	spectra_name = [] # only spectra names
	for i in range(len(files)):
		mask = True
		if ('Teff' in out_separate_params) and (Teff_range is not None):
			if not Teff_range[0] <= out_separate_params['Teff'][i] <= Teff_range[1]: mask = False
		if ('logg' in out_separate_params) and (logg_range is not None):
			if not logg_range[0] <= out_separate_params['logg'][i] <= logg_range[1]: mask = False
		if ('Z' in out_separate_params) and (Z_range is not None):
			if not Z_range[0] <= out_separate_params['Z'][i] <= Z_range[1]: mask = False
		if ('logKzz' in out_separate_params) and (logKzz_range is not None):
			if not logKzz_range[0] <= out_separate_params['logKzz'][i] <= logKzz_range[1]: mask = False
		if ('CtoO' in out_separate_params) and (CtoO_range is not None):
			if not CtoO_range[0] <= out_separate_params['CtoO'][i] <= CtoO_range[1]: mask = False
		if ('fsed' in out_separate_params) and (fsed_range is not None):
			if not fsed_range[0] <= out_separate_params['fsed'][i] <= fsed_range[1]: mask = False

		if mask:
			spectra_name_full.append(files[i]) # keep only spectra within the input parameter ranges
			spectra_name.append(files_short[i]) # keep only spectra within the input parameter ranges


	#--------------
	# TEST to fit only a few model spectra
	#spectra_name = spectra_name[:2]
	#spectra_name_full = spectra_name_full[:2]
	#--------------

	if len(spectra_name_full)==0: raise Exception('No model spectra within the indicated parameter ranges') # show up an error when there are no models in the indicated ranges
	else: 
		print(f'\n      {len(spectra_name)} model spectra selected with:')
		if ('Teff' in out_separate_params) and (Teff_range is not None): print(f'         Teff=[{Teff_range[0]}, {Teff_range[1]}]')
		if ('logg' in out_separate_params) and (logg_range is not None): print(f'         logg=[{logg_range[0]}, {logg_range[1]}]')
		if ('Z' in out_separate_params) and (Z_range is not None): print(f'         Z=[{Z_range[0]}, {Z_range[1]}]')
		if ('logKzz' in out_separate_params) and (logKzz_range is not None): print(f'         logKzz=[{logKzz_range[0]}, {logKzz_range[1]}]')
		if ('CtoO' in out_separate_params) and (CtoO_range is not None): print(f'         CtoO=[{CtoO_range[0]}, {CtoO_range[1]}]')
		if ('fsed' in out_separate_params) and (fsed_range is not None): print(f'         fsed=[{fsed_range[0]}, {fsed_range[1]}]')

	out = {'spectra_name_full': np.array(spectra_name_full), 'spectra_name': np.array(spectra_name)}

	return out

##########################
def separate_params(model, spectra_name):
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

	Returns:
	--------
	Dictionary with parameters for each model spectrum.
		- ``Teff``: effective temperature (in K) for each model spectrum.
		- ``logg``: surface gravity (log g) for each model spectrum.
		- ``Z``: (if provided by ``model``) metallicity for each model spectrum.
		- ``logKzz``: (if provided by ``model``) diffusion parameter for each model spectrum. 
		- ``fsed``: (if provided by ``model``) cloudiness parameter for each model spectrum (models with nc or no clouds are assigned a value of 99).
		- ``CtoO``: (if provided by ``model``) C/O ratio for each model spectrum.
		- ``spectra_name`` : model spectra names.

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

	out = {'spectra_name': spectra_name} # start dictionary with some parameters

	# get parameters from model spectra names
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
		out['Teff']= Teff_fit
		out['logg']= logg_fit
		out['Z']= Z_fit
		out['fsed']= fsed_fit
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
		out['Teff']= Teff_fit
		out['logg']= logg_fit
		out['logKzz']= logKzz_fit
		out['Z']= Z_fit
		out['CtoO']= CtoO_fit
	if (model == 'LB23'):
		Teff_fit = np.zeros(len(spectra_name))
		logg_fit = np.zeros(len(spectra_name))
		Z_fit = np.zeros(len(spectra_name))
		logKzz_fit = np.zeros(len(spectra_name))
		for i in range(len(spectra_name)):
			# Teff 
			Teff_fit[i] = float(spectra_name[i].split('_')[0][1:]) # K
			# logg
			logg_fit[i] = float(spectra_name[i].split('_')[1][1:]) # logg
			# Z (metallicity)
			Z_fit[i] = float(spectra_name[i].split('_')[2][1:])
			# Kzz (radiative zone)
			logKzz_fit[i] = np.log10(float(spectra_name[i].split('CDIFF')[1].split('_')[0])) # in cgs units
		out['Teff']= Teff_fit
		out['logg']= logg_fit
		out['logKzz']= logKzz_fit
		out['Z']= Z_fit
	if (model == 'Sonora_Cholla'):
		Teff_fit = np.zeros(len(spectra_name))
		logg_fit = np.zeros(len(spectra_name))
		logKzz_fit = np.zeros(len(spectra_name))
		for i in range(len(spectra_name)):
			# Teff 
			Teff_fit[i] = float(spectra_name[i].split('_')[0][:-1]) 
			# logg
			logg_fit[i] = round(np.log10(float(spectra_name[i].split('_')[1][:-1])),1) + 2
			# logKzz
			logKzz_fit[i] = float(spectra_name[i].split('_')[2].split('.')[0][-1])
		out['Teff']= Teff_fit
		out['logg']= logg_fit
		out['logKzz']= logKzz_fit
	if (model == 'Sonora_Bobcat'):
		Teff_fit = np.zeros(len(spectra_name))
		logg_fit = np.zeros(len(spectra_name))
		Z_fit = np.zeros(len(spectra_name))
		CtoO_fit = np.zeros(len(spectra_name))
		for i in range(len(spectra_name)):
			# Teff 
			Teff_fit[i] = float(spectra_name[i].split('_')[1].split('g')[0][1:])
			# logg
			logg_fit[i] = round(np.log10(float(spectra_name[i].split('_')[1].split('g')[1][:-2])),2) + 2
			# Z
			Z_fit[i] = float(spectra_name[i].split('_')[2][1:])
			# C/O
			if (len(spectra_name[i].split('_'))==4): # when the spectrum file name includes the C/O
				CtoO_fit[i] = float(spectra_name[i].split('_')[3][2:])
			if (len(spectra_name[i].split('_'))==3): # when the spectrum file name does not include the C/O
				CtoO_fit[i] = 1.0
		out['Teff']= Teff_fit
		out['logg']= logg_fit
		out['Z']= Z_fit
		out['CtoO']= CtoO_fit
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
		out['Teff']= Teff_fit
		out['logg']= logg_fit
		out['logKzz']= logKzz_fit
	if (model == 'BT-Settl'):
		Teff_fit = np.zeros(len(spectra_name))
		logg_fit = np.zeros(len(spectra_name))
		for i in range(len(spectra_name)):
			# Teff 
			Teff_fit[i] = float(spectra_name[i].split('-')[0][3:]) * 100 # K
			# logg
			logg_fit[i] = float(spectra_name[i].split('-')[1]) # g in cm/s^2
		out['Teff']= Teff_fit
		out['logg']= logg_fit
	if (model == 'SM08'):
		Teff_fit = np.zeros(len(spectra_name))
		logg_fit = np.zeros(len(spectra_name))
		fsed_fit = np.zeros(len(spectra_name))
		for i in range(len(spectra_name)):
			# Teff 
			Teff_fit[i] = float(spectra_name[i].split('_')[1].split('g')[0][1:])
			# logg
			logg_fit[i] = np.log10(float(spectra_name[i].split('_')[1].split('g')[1].split('f')[0])) + 2 # g in cm/s^2
			# fsed
			fsed_fit[i] = float(spectra_name[i].split('_')[1].split('g')[1].split('f')[1])
		out['Teff']= Teff_fit
		out['logg']= logg_fit
		out['fsed']= fsed_fit

	return out

#+++++++++++++++++++++++++++
# set wavelength range for the model comparison via chi-square or Bayes techniques
def set_fit_wl_range(fit_wl_range, N_spectra, wl_spectra):

	# define fit_wl_range when not provided
	if fit_wl_range is None:
		fit_wl_range = np.zeros((N_spectra, 2)) # Nx2 array, N:number of spectra and 2 for the minimum and maximum values for each spectrum
		for i in range(N_spectra):
			fit_wl_range[i,:] = np.array((wl_spectra[i].min(), wl_spectra[i].max()))
	else: # fit_wl_range is provided
		if len(fit_wl_range.shape)==1: fit_wl_range = fit_wl_range.reshape((1, 2)) # reshape fit_wl_range array

	return fit_wl_range

#+++++++++++++++++++++++++++
# set wavelength range to cut models for comparisons via chi-square or Bayes techniques
#def set_model_wl_range(model_wl_range, fit_wl_range, N_spectra,  wl_spectra):
def set_model_wl_range(model_wl_range, fit_wl_range):

	# define model_wl_range (if not provided) in terms of fit_wl_range
	if model_wl_range is None:
		model_wl_range = np.array((0.9*fit_wl_range.min(), 1.1*fit_wl_range.max())) # add padding to have enough spectral coverage in models

	# when model_wl_range is given and is equal or narrower than fit_wl_range
	# add padding to model_wl_range to avoid problems with the spectres routine
	# first find the minimum and maximum wavelength from the input spectra
#	min_tmp1 = min(wl_spectra[0])
#	for i in range(N_spectra):
#		min_tmp2 = min(wl_spectra[i])
#		if min_tmp2<min_tmp1: 
#			wl_spectra_min = min_tmp2
#			min_tmp1 = min_tmp2
#		else: 
#			wl_spectra_min = min_tmp1
#	max_tmp1 = max(wl_spectra[0])
#	for i in range(N_spectra):
#		max_tmp2 = max(wl_spectra[i])
#		if max_tmp2>max_tmp1:
#			wl_spectra_max = max_tmp2
#			max_tmp1 = max_tmp2
#		else:
#			wl_spectra_max = max_tmp1
#
#	if (model_wl_range.min()>=wl_spectra_min):
#		model_wl_range[0] = wl_spectra_min-0.1*wl_spectra_min # add padding to shorter wavelengths
#	if (model_wl_range.max()<=wl_spectra_max):
#		model_wl_range[1] = wl_spectra_max+0.1*wl_spectra_max # add padding to longer wavelengths

	# it may need an update to work with multiple spectra
	if (model_wl_range.min()>=fit_wl_range.min()):
		model_wl_range[0] = 0.9*fit_wl_range.min() # add padding to shorter wavelengths
	if (model_wl_range.max()<=fit_wl_range.max()):
		model_wl_range[1] = 1.1*fit_wl_range.max() # add padding to longer wavelengths

	return model_wl_range

##########################
def app_to_abs_flux(flux, distance, eflux=0, edistance=0):
	'''
	Description:
	------------
		Convert apparent fluxes into absolute fluxes.

	Parameters:
	-----------
	- flux : float array or float
		Fluxes (in any flux units).
	- distance : float
		Target distance (in pc).
	- eflux : float array or float, optional
		Flux uncertainties (in any flux units).
	- edistance : float, optional
		Distance error (in pc).

	Returns:
	--------
	- Dictionary with absolute fluxes and input parameters:
		- ``'flux_abs'`` : absolute fluxes in the same units as the input fluxes.
		- ``'eflux_abs'`` : (if ``eflux`` or ``edistance`` is provided) absolute flux uncertainties.
		- ``'flux_app'`` : input apparent fluxes.
		- ``'eflux_app'`` : (if provided) input apparent flux errors.
		- ``'distance'`` : input distance.
		- ``'edistance'`` : (if provided) input distance error.

	Example:
	--------
	>>> import seda
	>>>
	>>> # input parameters
	>>> d = 5.00 # pc
	>>> ed = 0.06 # pc
	>>> flux = 2.10 # mJy
	>>> eflux = 0.01 # mJy
	>>> 
	>>> # convert fluxes
	>>> seda.app_to_abs_flux(flux=flux, distance=d, eflux=eflux, edistance=ed)
	    {'flux_app': 2.1,
	    'flux_abs': 0.525,
	    'eflux_app': 0.01,
	    'eflux_abs': 0.01284562182223967,
	    'distance': 5.0,
	    'edistance': 0.06}

	Author: Genaro Suárez
	'''

	# use numpy arrays
	flux = astropy_to_numpy(flux)
	eflux = astropy_to_numpy(eflux)

	# absolute fluxes
	flux_abs = (distance/10.)**2 * flux 

	# absolute flux errors
	eflux_abs = flux_abs*np.sqrt((2.*edistance/distance)**2 + (eflux/flux)**2)

	# output dictionary
	out = {'flux_app': flux, 'flux_abs': flux_abs}

	if isinstance(eflux, np.ndarray): # if eflux is an array
		out['eflux_app'] = eflux
		out['eflux_abs'] = eflux_abs
	elif eflux!=0: # if flux is a float non-equal to zero
		out['eflux_app'] = eflux
		out['eflux_abs'] = eflux_abs

	out['distance'] = distance
	if edistance!=0:
		out['edistance'] = edistance
		out['eflux_abs'] = eflux_abs # if eflux_abs was stored above it would replace it without issues
	    
	return out

##########################
# convert an astropy array into a numpy array
def astropy_to_numpy(x):
	# if the variable is an astropy Column
	if isinstance(x, Column):
		if isinstance(x, MaskedColumn): # if MaskedColumn
			x = x.filled(np.nan) # fill masked values with nan
			x = x.data
		else: # if Column
			x = x.data
	# if the variable is an astropy Quantity (with units)
	if isinstance(x, u.Quantity): x = x.value

	return x

##########################
# round logg to steps of 0.25
def round_logg_point25(logg):
	logg = round(logg*4.) / 4. 
	return logg
