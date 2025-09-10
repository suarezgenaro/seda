import numpy as np
import time
import os
import fnmatch
import xarray
import pickle
from prettytable import PrettyTable
#import itertools
from spectres import spectres
from astropy import units as u
from astropy.io import ascii
from astropy.table import Column, MaskedColumn, Table
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm
from sys import exit
from .models import *

##########################
def convolve_spectrum(wl, flux, res, eflux=None, lam_res=None, disp_wl_range=None, convolve_wl_range=None, out_file=None):
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
	- res : float
		Spectral resolution (at ``lam_res``) desired to smooth the input spectrum.
	- eflux : float array, optional
		Flux uncertainties (any flux units) for the spectrum.
	- lam_res : float, optional
		Wavelength of reference at which ``res`` is given.
		Default is the integer closest to the median wavelength of the spectrum.
	- disp_wl_range : float array, optional
		Minimum and maximum wavelengths (same units as ``wl``) to compute the median wavelength dispersion of the input spectrum.
		Default values are the minimum and maximum wavelengths in ``wl``.
	- convolve_wl_range : float array, optional
		Minimum and maximum wavelengths (same units as ``wl``) to convolve the input spectrum.
		Default values are the minimum and maximum wavelengths in ``wl``.
	- out_file : str, optional
		File name to save the convolved spectrum as netCDF with xarray (it produces lighter files compared to normal ASCII files).
		The file name can include a path e.g. my_path/convolved_spectrum.nc
		If not provided, the convolved spectrum will not be saved.

	Returns:
	--------
	- Dictionary
		Dictionary with the convolved spectrum: 
			- ``'wl_conv'`` : wavelengths for the convolved spectrum (equal to input ``wl`` within ``convolve_wl_range``). 
			- ``'flux_conv'`` : convolved fluxes.
			- ``'eflux_conv'`` : convolved flux errors (if ``eflux`` is provided).
	- ``out_file`` file
		netCDF file with the convolved spectrum, if ``out_file`` is provided. 
		Wavelengths and fluxes for the convolved spectrum have the same units as the input spectrum.
		Note: the wavelength data points are the same as in the input spectrum, so wavelength steps do not reflect the resolution of the convolved spectrum.

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
	>>> res = 50 # resolution of 50
	>>>
	>>> out_convolve_spectrum = seda.convolve_spectrum(wl=wl, flux=flux, 
	>>>                                                eflux=eflux, res=res)

	Author: Genaro Suárez

	Date: 2020-03
	'''

	# convert input spectrum into numpy arrays if astropy
	wl = astropy_to_numpy(wl)
	flux = astropy_to_numpy(flux)
	eflux = astropy_to_numpy(eflux)

	# remove NaN values
	if eflux is None: mask_nonan = (~np.isnan(wl)) & (~np.isnan(flux))
	else: mask_nonan = (~np.isnan(wl)) & (~np.isnan(flux)) & (~np.isnan(eflux))
	wl = wl[mask_nonan]
	flux = flux[mask_nonan]
	if eflux is not None: eflux = eflux[mask_nonan]

	# stop if res=None (it can occur when other functions use this function)
	if res is None: raise Exception(f'"res=None" is not allowed. It must take a value.')

	# set lam_res if not provided
	if lam_res is None: lam_res = set_lam_res(wl)

	# set disp_wl_range and convolve_wl_range if not provided
	if (disp_wl_range is None): disp_wl_range = np.array((wl.min(), wl.max())) # define disp_wl_range if not provided
	if (convolve_wl_range is None): convolve_wl_range = np.array((wl.min(), wl.max())) # define convolve_wl_range if not provided

	wl_bin = abs(wl[1:] - wl[:-1]) # wavelength dispersion of the spectrum
	wl_bin = np.append(wl_bin, wl_bin[-1]) # add an element equal to the last row to have same data points

	# define a Gaussian for convolution
	mask_fit = (wl>=disp_wl_range[0]) & (wl<=disp_wl_range[1]) # mask to obtain the median wavelength dispersion
	stddev = (lam_res/res)*(1./np.median(wl_bin[mask_fit])) / (2.*np.sqrt(2*np.log(2))) # stddev is given in pixels
	#if stddev<1: 
	if (lam_res/res)*(1./np.median(wl_bin[mask_fit]))<1:
		print('   Warning: the input spectrum may have a resolution smaller than the desired one.')
		print('            the spectrum will be convolved but will be essentially the same.')

	gauss = Gaussian1DKernel(stddev=stddev)

	mask_conv = (wl>=convolve_wl_range[0]) & (wl<=convolve_wl_range[1]) # range to convolve the spectrum

	flux_conv = convolve(flux[mask_conv], gauss) # convolve only the selected wavelength range 
	wl_conv = wl[mask_conv] # corresponding wavelength data points for convolved fluxes
	if eflux is not None: eflux_conv = convolve(eflux[mask_conv], gauss)

	out = {'wl_conv': wl_conv, 'flux_conv': flux_conv}
	if (eflux is not None): out['eflux_conv'] = eflux_conv

	# save convolved spectrum as netCDF
	if out_file is not None:
		# make xarray
		if eflux is None: ds = xarray.Dataset({'wl': wl_conv}, coords={'flux': flux_conv})
		else: ds = xarray.Dataset({'wl': wl_conv}, coords={'flux': flux_conv, 'eflux': eflux_conv})
		# store xarray as netCDF
		ds.to_netcdf(out_file)

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
def best_chi2_fits(output_chi2, N_best_fits=1, model_dir_ori=None, ori_res=False):
	'''
	Description:
	------------
		Read best-fitting model spectra from the chi-square minimization.

	Parameters:
	-----------
	- output_chi2 : dictionary or str
		Output dictionary with the results from the chi-square minimization by ``chi2``.
		It can be either the name of the pickle file or simply the output dictionary.
	- N_best_fits : int, optional (default 1)
		Number of best model fits to be read.
	- ori_res : {``True``, ``False``}, optional (default ``False``)
		Read (``True``) or do not read (``False``) model spectra with the original resolution.
	- model_dir_ori : str, list, or array
		Path to the directory (str, list, or array) or directories (as a list or array) containing the model spectra with the original resolution.
		This parameter is needed if ``ori_res=True`` and `seda.chi2` was run skipping the model spectra convolution (if `skip_convolution=True``).

	Returns:
	--------
	Dictionary with best model fits:
		- ``'spectra_name_best'`` : model spectra names.
		- ``'fit_spectra'``: label indicating whether spectra were fitted.
		- ``'fit_photometry'``: label indicating whether photometry was fitted.
		- ``'chi2_red_fit_best'`` : reduced chi-square.
		- ``'chi2_red_wl_fit_best'`` : reduced chi-square as a function of wavelength.
		- ``'flux_residuals_best'`` : flux residuals in linear scale between model and input spectra.
		- ``'logflux_residuals_best'`` : flux residuals in log scale between model and input spectra.
		- ``'wl_model_best'`` : (if ``ori_res``) wavelength (in um) of original model spectra.
		- ``'flux_model_best'`` : (if ``ori_res``) fluxes (in erg/s/cm2/A) of original model spectra.
		- ``'wl_array_model_conv_resam_best'`` : (if ``fit_spectra``) wavelength (in um) of resampled, convolved model spectra in the input spectra ranges.
		- ``'flux_array_model_conv_resam_best'`` : (if ``fit_spectra``) fluxes (in erg/s/cm2/A) of resampled, convolved model spectra in the input spectra ranges.
		- ``'wl_array_model_conv_resam_fit_best'`` : (if ``fit_spectra``) wavelength (in um) of resampled, convolved model spectra within the fit range.
		- ``'flux_array_model_conv_resam_fit_best'`` : (if ``fit_spectra``) fluxes (in erg/s/cm2/A) of resampled, convolved model spectra within the fit range.
		- ``'flux_syn_array_model_fit'`` : (if ``fit_photometry``) synthetic photometry (in erg/s/cm2/A) for the filters within the fit range.
		- ``'lambda_eff_array_model_fit'`` : (if ``fit_photometry``) effective wavelength (in um) from model spectra for filters within the fit range.
		- ``'widtheff_array_model_fit'`` : (if ``fit_photometry``) effective width (in um) from model spectra for filters within the fit range.
		- ``'params'`` : model free parameters

	Author: Genaro Suárez

	Date: 2024-10, 2025-09-07
	'''

	# open results from the chi square analysis
	try: # if given as a pickle file
		with open(output_chi2, 'rb') as file:
			output_chi2 = pickle.load(file)
	except: # if given as the output of chi2_fit
		pass
	fit_spectra = output_chi2['my_chi2'].fit_spectra
	fit_photometry = output_chi2['my_chi2'].fit_photometry
	model = output_chi2['my_chi2'].model
	res = output_chi2['my_chi2'].res#[0] # resolution for the first input spectrum
	lam_res = output_chi2['my_chi2'].lam_res#[0] # lam_resolution for the first input spectrum
	N_model_spectra = output_chi2['N_model_spectra']
	spectra_name_full = output_chi2['spectra_name_full']
	spectra_name = output_chi2['spectra_name']
	chi2_red_fit = output_chi2['chi2_red_fit']
	chi2_red_wl_fit = output_chi2['chi2_red_wl_fit'] # list of reduced chi-square for each model compared
	scaling_fit = output_chi2['scaling_fit']
	flux_residuals = output_chi2['flux_residuals']
	logflux_residuals = output_chi2['logflux_residuals']
	skip_convolution = output_chi2['my_chi2'].skip_convolution
	if fit_spectra:
		wl_array_model_conv_resam = output_chi2['wl_array_model_conv_resam'] # wavelengths of resampled, convolved model spectra in the input spectra wavelength ranges
		flux_array_model_conv_resam = output_chi2['flux_array_model_conv_resam'] # fluxes of resampled, convolved model spectra in the input spectra wavelength ranges
		wl_array_model_conv_resam_fit = output_chi2['wl_array_model_conv_resam_fit'] # wavelengths of resampled, convolved model spectra within the fit ranges
		flux_array_model_conv_resam_fit = output_chi2['flux_array_model_conv_resam_fit'] # fluxes of resampled, convolved model spectra within the fit ranges
	if fit_photometry:
		flux_syn_array_model_fit = output_chi2['flux_syn_array_model_fit']
		lambda_eff_array_model_fit = output_chi2['lambda_eff_array_model_fit']
		width_eff_array_model_fit = output_chi2['width_eff_array_model_fit']

	# select the best fits given by N_best_fits
	if (N_best_fits > N_model_spectra): raise Exception(f'not enough model spectra (only {N_model_spectra}) in the fit to select "N_best_fits={N_best_fits}" best fits')
	sort_ind = np.argsort(chi2_red_fit)
	chi2_red_fit_best = chi2_red_fit[sort_ind][:N_best_fits]
	chi2_red_wl_fit_best = [chi2_red_wl_fit[i] for i in sort_ind][:N_best_fits]
	scaling_fit_best = scaling_fit[sort_ind][:N_best_fits]
	spectra_name_full_best = spectra_name_full[sort_ind][:N_best_fits]
	spectra_name_best = spectra_name[sort_ind][:N_best_fits]
	flux_residuals_best = [flux_residuals[i] for i in sort_ind][:N_best_fits]
	logflux_residuals_best = [logflux_residuals[i] for i in sort_ind][:N_best_fits]
	if fit_spectra:
		wl_array_model_conv_resam_best = [wl_array_model_conv_resam[i] for i in sort_ind][:N_best_fits]
		flux_array_model_conv_resam_best = [flux_array_model_conv_resam[i] for i in sort_ind][:N_best_fits]
		wl_array_model_conv_resam_fit_best = [wl_array_model_conv_resam_fit[i] for i in sort_ind][:N_best_fits]
		flux_array_model_conv_resam_fit_best = [flux_array_model_conv_resam_fit[i] for i in sort_ind][:N_best_fits]
	if fit_photometry:
		flux_syn_array_model_fit_best = [flux_syn_array_model_fit[i] for i in sort_ind][:N_best_fits]
		lambda_eff_array_model_fit_best = [lambda_eff_array_model_fit[i] for i in sort_ind][:N_best_fits]
		width_eff_array_model_fit_best = [width_eff_array_model_fit[i] for i in sort_ind][:N_best_fits]

	# read parameters from file name for the best fits
	out_separate_params = separate_params(model=model, spectra_name=spectra_name_best)

	# read best fits with the original resolution
	if ori_res:
		wl_model_best_lst = []
		flux_model_best_lst = []
		if not skip_convolution: # when the convolution was not skipped
			for i in range(N_best_fits):
				out_read_model_spectrum = read_model_spectrum(spectrum_name_full=spectra_name_full_best[i], model=model)
				wl_model_best_lst.append(out_read_model_spectrum['wl_model'])
				flux_model_best_lst.append(scaling_fit_best[i] * out_read_model_spectrum['flux_model']) # scaled fluxes
		else: # when the convolution was skipped
			if model_dir_ori is None: raise Exception(f'parameter "model_dir_ori" is needed to read model spectra with the original resolution')
			else:
				out_select_model_spectra = select_model_spectra(model_dir=model_dir_ori, model=model) # all spectra in model_dir_ori
				for i in range(N_best_fits):
					spectrum_name = spectra_name_best[i].split('_R')[0] # convolved spectrum name without additions to match original resolution name
					if spectrum_name in out_select_model_spectra['spectra_name']: # if the i best fit is in the model_dir_ori folder 
						spectrum_name_full = out_select_model_spectra['spectra_name_full'][out_select_model_spectra['spectra_name']==spectrum_name][0]
						out_read_model_spectrum = read_model_spectrum(spectrum_name_full=spectrum_name_full, model=model)
						wl_model_best_lst.append(out_read_model_spectrum['wl_model'])
						flux_model_best_lst.append(scaling_fit_best[i] * out_read_model_spectrum['flux_model']) # scaled fluxes
					else: raise Exception(f'{spectrum_name} is not in {model_dir_ori}')

		# convert list with models to a numpy array
		# maximum number of data points
		max_tmp1 = 0
		for i in range(N_best_fits): # for each model spectrum
			max_tmp2 = len(wl_model_best_lst[i])
			if max_tmp2>max_tmp1:
				N_modelpoints_max = max_tmp2
				max_tmp1 = max_tmp2
			else:
				N_modelpoints_max = max_tmp1
	
		wl_model_best = np.zeros((N_best_fits, N_modelpoints_max))
		flux_model_best = np.zeros((N_best_fits, N_modelpoints_max))
		for i in range(N_best_fits): # for each model spectrum
			wl_model_best[i,:] = wl_model_best_lst[i]
			flux_model_best[i,:] = flux_model_best_lst[i]

#	# convolve spectrum
#	wl_model_conv = np.zeros((N_modelpoints, N_best_fits))
#	flux_model_conv = np.zeros((N_modelpoints, N_best_fits))
#	for i in range(N_best_fits):
#		out_convolve_spectrum = convolve_spectrum(wl=wl_model[:,i], flux=flux_model[:,i], lam_res=lam_res, res=res, disp_wl_range=fit_wl_range)
#		wl_model_conv[:,i] = out_convolve_spectrum['wl_conv']
#		flux_model_conv[:,i] = out_convolve_spectrum['flux_conv']
#		#if not skip_convolution:
#		#	out_convolve_spectrum = convolve_spectrum(wl=wl_model[:,i], flux=flux_model[:,i], lam_res=lam_res, res=res, disp_wl_range=fit_wl_range)
#		#	wl_model_conv[:,i] = out_convolve_spectrum['wl_conv']
#		#	flux_model_conv[:,i] = out_convolve_spectrum['flux_conv']
#		#else:
#		#	out_read_model_spectrum = read_model_spectrum_conv(spectrum_name_full=spectra_name_full_best[i])
#		#	wl_model_conv[:,i] = out_read_model_spectrum['wl_model']
#		#	flux_model_conv[:,i] = out_read_model_spectrum['flux_model']

	out = {'spectra_name_best': spectra_name_best, 'fit_spectra': fit_spectra, 'fit_photometry': fit_photometry,
	       'chi2_red_fit_best': chi2_red_fit_best, 'chi2_red_wl_fit_best': chi2_red_wl_fit_best, 
		   #'flux_model': flux_model, 'wl_model_conv': wl_model_conv, 'flux_model_conv': flux_model_conv}
		   #'wl_model_conv': wl_array_model_conv_resam_best, 'flux_model_conv': flux_array_model_conv_resam_best,
	       'flux_residuals_best': flux_residuals_best, 'logflux_residuals_best': logflux_residuals_best}
	if fit_spectra:
		out['wl_array_model_conv_resam_best'] = wl_array_model_conv_resam_best
		out['flux_array_model_conv_resam_best'] = flux_array_model_conv_resam_best
		out['wl_array_model_conv_resam_fit_best'] = wl_array_model_conv_resam_fit_best
		out['flux_array_model_conv_resam_fit_best'] = flux_array_model_conv_resam_fit_best
	if fit_photometry:
		out['flux_syn_array_model_fit_best'] = flux_syn_array_model_fit_best
		out['lambda_eff_array_model_fit_best'] = lambda_eff_array_model_fit_best
		out['width_eff_array_model_fit_best'] = width_eff_array_model_fit_best
	#if not skip_convolution or model_dir_ori is not None:
	if ori_res:
		out['wl_model_best'] = wl_model_best
		out['flux_model_best'] = flux_model_best
	out['params'] = out_separate_params['params']

	return out

##################################################
def generate_model_spectrum(params, model, grid=None, model_dir=None, save_spectrum=False):
	'''
	Description:
	------------
		Generate a synthetic spectrum with any combination of free parameters within the coverage of the desired atmospheric models.

	Parameters:
	-----------
	- params : dictionary
		Value for each free parameter in the models to generate a synthetic spectrum with the desired parameter values.
		E.g., ``params = {'Teff': 1010, 'logg': 4.2, 'Z': 0.1, 'fsed': 2.2}`` for a model grid with those free parameters.
	- model : str
		Atmospheric models to generate the synthetic spectrum. 
		See available models in ``seda.Models().available_models``.  
	- grid : dictionary, optional
		Model grid (``'wavelength'`` and ``'flux'``) generated by ``seda.read_grid`` for interpolations.
		If not provided (default), then the grid is read (``model``, ``model_dir``, ``Teff_range`` and ``logg_range`` must be provided). 
		If provided, the code will skip reading the grid, which will save some time (a few minutes).
	- model_dir : str or list, optional
		Path to the directory (str or list) or directories (as a list) containing the model spectra (e.g., ``model_dir = ['path_1', 'path_2']``). 
		Required if ``grid`` is not provided.
	- save_spectrum : {``True``, ``False``}, optional (default ``False``)
		Save (``True``) or do not save (``False``) the generated spectrum as an ascii table.
		Default name is '``model``\_``params``\_.dat'.

	Returns:
	--------
	Dictionary with generated model spectrum:
		- ``'wavelength'``: wavelengths in microns for the generated spectrum.
		- ``'flux'``: fluxes in erg/s/cm2/A for the generated spectrum.
		- ``'params'``: input parameters for the generated synthetic spectrum.

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
	>>> params = {'Teff': 765, 'logg': 4.15, 'logKzz': 5.2, 'Z': 0.2, 'CtoO': 1.2}
	>>> 
	>>> # generate model spectrum
	>>> out = seda.generate_model_spectrum(model=model, model_dir=model_dir, 
	>>>                                    params=params)

	Author: Genaro Suárez

	Date: 2025-03-14
	'''

	# verify there is an input value for each free parameter in the grid
	# parameter ranges covered by the model spectra in the input model_dir
	if grid is None: # when no grid is provided, read the parameter ranges from the spectra in the input folders
		params_models = select_model_spectra(model=model, model_dir=model_dir)['params']
	else: # when a grid is provided, read the parameter ranges from the grid dictionary
		params_models = grid['params_unique']

	for param in params_models:
		if param not in params: raise Exception(f'Provide "{param}" value in "params" since it is a free parameter in "{model}".')

	# verify that "params" are within the model grid coverage
	for param in params:
		if params[param]<params_models[param].min() or params[param]>params_models[param].max():
			raise Exception(f'"{param}={params[param]}" value in "params" is out of the range covered by the "{model}" spectra in "model_dir", which is [{params_models[param].min()}, {params_models[param].max()}]')

	# sort input params in the same order as the dictionary with free parameters returned by `separate_params`
	params = reorder_dict(params, Models(model).free_params)

	# read model grid if not provided
	if grid is None:
		# grid values around the desired parameter values
		params_ranges = {}
		for param in params: # for each free parameter in the grid
			params_ranges[param] = find_two_nearest(params_models[param], params[param])

		grid = read_grid(model=model, model_dir=model_dir, params_ranges=params_ranges)

	# define an interpolating function from the grid
	flux_grid = grid['flux']
	wl_grid = grid['wavelength']
	params_unique = grid['params_unique']
	interp_flux = RegularGridInterpolator((params_unique.values()), flux_grid)
	interp_wl = RegularGridInterpolator((params_unique.values()), wl_grid)

	# interpolate input parameters
	spectra_flux = interp_flux(list(params.values()))[0,:] # to return a 1D array
	spectra_wl = interp_wl(list(params.values()))[0,:] # to return a 1D array
	
	# reverse array to sort wavelength from shortest to longest
	ind_sort = np.argsort(spectra_wl)
	spectra_wl = spectra_wl[ind_sort]
	spectra_flux = spectra_flux[ind_sort]

	# store generated spectrum
	if save_spectrum:
		# name for the file
		spectrum_filename = model
		for param in params:
			spectrum_filename += f'_{param}{params[param]}'
		spectrum_filename += '.dat'

		# save spectrum
		out = open(spectrum_filename, 'w')
		out.write('# wavelength(um) flux(erg/s/cm2/A)  \n')
		for i in range(len(spectra_wl)):
			out.write('%11.7f %17.6E \n' %(spectra_wl[i], spectra_flux[i]))
		out.close()

	out = {'wavelength': spectra_wl, 'flux': spectra_flux, 'params': params}

	return out

##################################################
def read_grid(model, model_dir, params_ranges=None, convolve=False, model_wl_range=None, 
	          fit_wl_range=None, res=None, lam_res=None, wl_resample=None, disp_wl_range=None, 
	          skip_convolution=False, filename_pattern=None, path_save_spectra_conv=None):
	'''
	Description:
	------------
		Read a model grid constrained by input parameters.

	Parameters:
	-----------
	- model : str
		Atmospheric models. See available models in ``seda.Models().available_models``.  
	- model_dir : str or list
		Path to the directory (str or list) or directories (as a list) containing the model spectra (e.g., ``model_dir = ['path_1', 'path_2']``). 
	- params_ranges : dictionary, optional
		Minimum and maximum values for any model free parameters to select a model grid subset.
		E.g., ``params_ranges = {'Teff': [1000, 1200], 'logg': [4., 5.]}`` to consider spectra within those Teff and logg ranges.
		If a parameter range is not provided, the full range in ``model_dir`` is considered.
	- convolve: {``True``, ``False``}, optional (default ``False``)
		Convolve (``'True'``) or do not convolve (``'False'``) the model grid spectra to the indicated ``res`` at ``lam_res``.
	- fit_wl_range : float array, optional
		Minimum and maximum wavelengths (in micron) to resample the model spectra to the fit range. E.g., ``fit_wl_range = np.array([fit_wl_min1, fit_wl_max1])``. 
		Default values are the minimum and the maximum wavelengths in ``wl_resample``.
	- disp_wl_range : float array, optional
		Minimum and maximum wavelengths (in micron) to compute the median wavelength dispersion of model spectra to convolve them.
		Default values are the minimum and the maximum wavelengths in ``wl_resample``.
	- model_wl_range : float array (optional)
		Minimum and maximum wavelength (in microns) to cut model spectra to keep only wavelengths of interest.
		Default values are the minimum and maximum wavelengths in ``wl_resample``, if provided, with a padding to avoid issues with the resampling.
		Otherwise, ``model_wl_range=None``, so model spectra will not be trimmed.
	- res : float, optional (required if ``convolve``).
		Spectral resolution (at ``lam_res``) to smooth model spectra.
	- lam_res : float, optional.
		Wavelength of reference for ``res``.
		Default is the median wavelength of the spectrum.
	- wl_resample : float array or list, optional
		Wavelength data points to resample the grid
	- disp_wl_range : float array, optional
		Minimum and maximum wavelengths (in microns) to compute the median wavelength dispersion of model spectrum.
		Default values are the minimum and maximum wavelengths in model spectra.
	- skip_convolution : {``True``, ``False``}, optional (default ``False``)
		Convolution of model spectra (the slowest process in the code) can (``True``) or cannot (``False``) be avoided. 
		Once the code has be run and the convolved spectra were stored in ``path_save_spectra_conv``, the convolved grid can be reused for other input data with the same resolution as the convolved spectra.
	- filename_pattern : str, optional
		Pattern to select only files including it.
		Default is a common pattern in all spectra original filenames in ``model``, as indicated by ``Models(model).filename_pattern``.
	- path_save_spectra_conv: str, optional
		Directory path to store convolved model spectra. 
		If not provided (default), the convolved spectra will not be saved. 
		If the directory does not exist, it will be created. Otherwise, the spectra will be added to the existing folder.
		The convolved spectra will keep the same original names along with the ``res`` and ``lam_res`` parameters, e.g. 'original_spectrum_name_R100at1um.nc' for ``res=100`` and ``lam_res=1``.
		They will be saved as netCDF with xarray (it produces lighter files compared to normal ASCII files).

	Returns:
	--------
	Dictionary with the (convolved and resampled, if requested) model grid:
		- ``'wavelength'`` : wavelengths in microns for the model spectra in the grid.
		- ``'flux'`` : fluxes in erg/s/cm2/A for the model spectra in the grid.
		- ``'params_unique'`` : dictionary with unique (non-repetitive) values for each model free parameter

	Example:
	--------
	>>> import seda
	>>>
	>>> # models
	>>> model = 'Sonora_Elf_Owl'
	>>> model_dir = ['my_path/Sonora_Elf_Owl/spectra/output_700.0_800.0/',
	>>>              'my_path/Sonora_Elf_Owl/spectra/output_850.0_950.0/']
	>>> 
	>>> # set ranges for some (Teff and logg) free parameters to select only a grid subset
	>>> params_ranges = {'Teff': [700, 900], 'logg': [4.0, 5.0]}
	>>>
	>>> # read the grid
	>>> out_read_grid = seda.read_grid(model=model, model_dir=model_dir, 
	>>>                                params_ranges=params_ranges)

	Author: Genaro Suárez

	Date: 2023-02
	'''

	ini_time_grid = time.time() # to estimate the time elapsed reading the grid

	if wl_resample is not None:
		# handle fit_wl_range
		fit_wl_range = set_fit_wl_range(fit_wl_range=fit_wl_range, N_spectra=1, wl_spectra=[wl_resample])[0]
		# handle model_wl_range
		model_wl_range = set_model_wl_range(model_wl_range=model_wl_range, wl_spectra_min=fit_wl_range.min(), wl_spectra_max=fit_wl_range.max())
	
#	# set model_wl_range in case wl_resample is given
#	if wl_resample is not None:
#		#model_wl_range = set_model_wl_range(model_wl_range=model_wl_range, fit_wl_range=fit_wl_range)
#		model_wl_range = set_model_wl_range(model_wl_range=model_wl_range, wl_spectra_min=wl_spectra_min, wl_spectra_max=wl_spectra_max)

	# read the model spectra names and their parameters in the input folders and meeting the indicated parameters ranges 
	out_select_model_spectra = select_model_spectra(model=model, model_dir=model_dir, params_ranges=params_ranges, filename_pattern=filename_pattern)
	spectra_name_full = out_select_model_spectra['spectra_name_full']
	spectra_name = out_select_model_spectra['spectra_name']
	params = out_select_model_spectra['params']

	# unique values for each free parameter in the selected spectra
	params_unique = {} # initialize dictionary
	for param in params.keys():
		params_unique[param] = np.unique(params[param]) # save unique values for each free parameter

	# create an array with as many dimensions as the number of free parameters in the models 
	# and each dimension's size as the number of unique values for the corresponding parameter
	arr = np.array(0.)*np.nan # initialize float array of NaNs with dimension zero
	for param in params.keys(): # for each free parameter
		dim_size = len(params_unique[param]) # number of unique values in each parameter
		new_dim = np.expand_dims(arr, -1) # add a dimension (at the end)
		arr = np.repeat(new_dim, dim_size, axis=-1) # change size of new dimension

	# load model grid with the selected model spectra
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

	# save model spectra for each combination of free parameter values
	first_spec = True # reference to initialize arrays for store the grid in the first iteration with a model spectrum
	for index in np.ndindex(arr.shape): # iterate over all possible combinations of the free parameter unique values
		# update the progress bar
		grid_bar.update(1)

		# mask to select the corresponding spectrum for each combination of parameters
		mask = np.ones(len(spectra_name), bool) # initialize mask with Trues
		for i,param in enumerate(params.keys()): # for each free parameter
			mask &= params[param]==params_unique[param][index[i]] # apply criteria to the mask and update it

		# verify if there is a model spectrum for the combination of parameters
		if not np.any(mask): # if there is not a spectrum
			print('   Caveat: No spectrum in "model_dir" for the combination:')
			for i,param in enumerate(params.keys()):
				print(f'	  {param}={params_unique[param][index[i]]}')
		else: # if there is a spectrum
			# read the spectrum with the parameter combination in the iteration and cut it to model_wl_range (default value: fit_wl_range plus padding)
			spectrum_name_full = spectra_name_full[mask][0]
			if not skip_convolution: # read model spectra with original resolution
				out_read_model_spectrum = read_model_spectrum(spectrum_name_full=spectrum_name_full, model=model, 
				                                              model_wl_range=model_wl_range)
			else: # read precomputed convolved model spectra
				out_read_model_spectrum = read_model_spectrum_conv(spectrum_name_full=spectrum_name_full, model_wl_range=model_wl_range)
			wl_model = out_read_model_spectrum['wl_model'] # in um
			flux_model = out_read_model_spectrum['flux_model'] # in erg/s/cm2/A

			# convolve (if requested) the model spectrum to the indicated resolution
			if convolve and not skip_convolution: # convolve spectra only if convolve is True and skip_convolution is False
				if path_save_spectra_conv is None: # do not save the convolved spectrum
					out_convolve_spectrum = convolve_spectrum(wl=wl_model, flux=flux_model, res=res, lam_res=lam_res, disp_wl_range=disp_wl_range)
				else: # save convolved spectrum
					if not os.path.exists(path_save_spectra_conv): os.makedirs(path_save_spectra_conv) # make directory (if not existing) to store convolved spectra
					out_file = path_save_spectra_conv+spectra_name[mask][0]+f'_R{res}at{lam_res}um.nc'
					out_convolve_spectrum = convolve_spectrum(wl=wl_model, flux=flux_model, res=res, lam_res=lam_res, disp_wl_range=disp_wl_range, out_file=out_file)

#				out_convolve_spectrum = convolve_spectrum(wl=wl_model, flux=flux_model, res=res, lam_res=lam_res)
				wl_model = out_convolve_spectrum['wl_conv']
				flux_model = out_convolve_spectrum['flux_conv']

			# resample (if requested) the convolved model spectrum to the wavelength data points in the observed spectra
			if wl_resample is not None:
				# mask to select data points within the fit range or model coverage range, whichever is narrower
				mask_fit = (wl_resample >= max(fit_wl_range[0], wl_model.min())) & \
				           (wl_resample <= min(fit_wl_range[1], wl_model.max()))
				flux_model = spectres(wl_resample[mask_fit], wl_model, flux_model)
				wl_model = wl_resample[mask_fit]

			# save spectrum for each combination
			if first_spec: # for the first parameters' combination with a model spectrum
				# define arrays to save the grid
				# add a last dimension with the number of data points in the model spectrum subset
				# it is better to initialize the arrays after the first iteration to consider the model spectrum 
				# data points after resampling instead of using all data points in the original model spectrum
				wl_grid = np.repeat(np.expand_dims(arr, -1), len(wl_model), axis=-1) # to save the wavelength at each grid point
				flux_grid = np.repeat(np.expand_dims(arr, -1), len(wl_model), axis=-1) # to save the flux at each grid point

				# save first spectrum
				wl_grid[index] = wl_model
				flux_grid[index] = flux_model

				first_spec = False # to avoid initializing grid arrays in future iterations

			else: # all but first parameters' combinations	 
				# ensure the new spectrum has the same number of data points as the first one read
				if wl_model.shape!=wl_grid.shape[-1:]:
					raise ValueError(f'Spectrum {spectra_name[mask]} has a different number of data points compared to the previous ones')
				# save spectrum
				wl_grid[index] = wl_model
				flux_grid[index] = flux_model

	# close the progress bar
	grid_bar.close()

	fin_time_grid = time.time()
	print_time(fin_time_grid-ini_time_grid)

	out = {'wavelength': wl_grid, 'flux': flux_grid, 'params_unique': params_unique}

	return out

##########################
def best_bayesian_fit(output_bayes, grid=None, model_dir_ori=None, ori_res=False, save_spectrum=False):
	'''
	Description:
	------------
		Generate model spectrum with the posterior parameters.

	Parameters:
	-----------
	- 'output_bayes' : dictionary or str
		Output dictionary with the results from the nested sampling by ``bayes``.
		It can be either the name of the pickle file or simply the output dictionary.
	- grid : dictionary, optional
		Model grid (``'wavelength'`` and ``'flux'``) generated by ``seda.read_grid`` for interpolations.
		If not provided (default), then a grid subset with model spectra around the median posteriors is read.
		If provided, the code will skip reading the grid, which will save some time.
	- ori_res : {``True``, ``False``}, optional (default ``False``)
		Read (``True``) or do not read (``False``) model spectrum for the best fit with the original resolution.
	- model_dir_ori : str, list, or array
		Path to the directory (str, list, or array) or directories (as a list or array) containing the model spectra with the original resolution.
		This parameter is needed if ``ori_res=True`` and `seda.bayes` was run skipping the model spectra convolution (if `skip_convolution=True``).
	- save_spectrum : {``True``, ``False``}, optional (default ``False``)
		Save (``True``) or do not save (``False``) best model fit from the nested sampling.

	Returns:
	--------
	- '``model``\_best\_fit\_bayesian\_sampling.dat' : ascii table
		Table with best model fit (if ``save_spectrum``).
	- dictionary
		Dictionary with the best model fit from the nested sampling:
			- ``'wl_spectra_fit'`` : wavelength in um of the input spectra in ``fit_wl_range``.
			- ``'flux_spectra_fit'`` : fluxes in erg/cm2/s/A of the input spectra in ``fit_wl_range``.
			- ``'eflux_spectra_fit'`` : flux uncertainties in erg/cm2/s/A of the input spectra in ``fit_wl_range``.
			- ``'params_med'`` : median values for sampled parameters.
			- ``'params_confidence_interval'`` : confidence interval for sampled parameters.
			- ``'confidence_interval(%)'`` : central percentage considered to calculate the confidence interval.
			- ``'wl_model'`` : wavelength in um of the best scaled, convolved, and resampled model fit
			- ``'flux_model'`` : fluxes in erg/cm2/s/A of the best scaled, convolved, and resampled model fit
			- ``'wl_model_ori'`` : (if ``ori_res`` is ``True``) wavelength in um of the best scaled model fit with its original resolution
			- ``'flux_model_ori'`` : (if ``ori_res`` is ``True``) fluxes in erg/cm2/s/A of the best scaled model fit with its original resolution

	Author: Genaro Suárez

	Date: 2024-09
	'''

	# open results from the nested sampling
	try: # if given as a pickle file
		with open(output_bayes, 'rb') as file:
			output_bayes = pickle.load(file)
	except: # if given as the output of bayes_fit
		pass
	model = output_bayes['my_bayes'].model
	model_dir = output_bayes['my_bayes'].model_dir
	filename_pattern = output_bayes['my_bayes'].filename_pattern
	path_save_spectra_conv = output_bayes['my_bayes'].path_save_spectra_conv
	skip_convolution = output_bayes['my_bayes'].skip_convolution
	res = output_bayes['my_bayes'].res
	lam_res = output_bayes['my_bayes'].lam_res
	distance = output_bayes['my_bayes'].distance
	wl_spectra_min = output_bayes['my_bayes'].wl_spectra_min
	wl_spectra_max = output_bayes['my_bayes'].wl_spectra_max
	wl_spectra_fit = output_bayes['my_bayes'].wl_spectra_fit
	flux_spectra_fit = output_bayes['my_bayes'].flux_spectra_fit
	eflux_spectra_fit = output_bayes['my_bayes'].eflux_spectra_fit
	N_spectra = output_bayes['my_bayes'].N_spectra
	fit_wl_range = output_bayes['my_bayes'].fit_wl_range
	params_priors = output_bayes['my_bayes'].params_priors
	out_dynesty = output_bayes['out_dynesty']

	# compute median values and confidence intervals for all sampled parameters
	conf_interval = 68 # 68% confidence interval

	params_med = {}
	params_conf = {}
	for i,param in enumerate(params_priors): # for each parameter in the sampling
		params_med[param] = np.median(out_dynesty.samples[:,i]) # add the median of each parameter to the dictionary 

		quantile_low = 50 - conf_interval/2
		quantile_high = 50 + conf_interval/2
		params_quantile_low = np.percentile(out_dynesty.samples[:,i], quantile_low) # parameter value at quantile_low
		params_quantile_high = np.percentile(out_dynesty.samples[:,i], quantile_high) # parameter value at quantile_high
		params_conf[param] = [params_quantile_low, params_quantile_high] # add the confidence range of each parameter to the dictionary 

	# round median parameters
	params_models = Models(model).params_unique # free parameters in the models
	for i,param in enumerate(params_med): # for each sampled parameter
		if param in params_models: # for free parameters in the model grid
			params_med[param] = round(params_med[param], max_decimals(params_models[param])+1) # round to the precision (plus one decimal place) of the parameter in models
			conf_low = round(params_conf[param][0], max_decimals(params_models[param])+1)
			conf_high = round(params_conf[param][1], max_decimals(params_models[param])+1)
			params_conf[param] = [conf_low, conf_high]
		else: # parameters other than those in the grid (e.g. radius)
			params_med[param] = round(params_med[param], 2) # consider two decimals
			conf_low = round(params_conf[param][0], 2)
			conf_high = round(params_conf[param][1], 2)
			params_conf[param] = [conf_low, conf_high]

	# read grid, if needed
	if grid is None:
		# grid values around the desired parameter values
		params_ranges = {}
		for param in params_models: # for each free parameter in the grid
			params_ranges[param] = find_two_nearest(params_models[param], params_med[param])

		# read grid, convolve it (if not skip_convolution), and resample it to the input spectra
		grid = [] #  to save a grid appropriate for each input spectrum
		for i in range(N_spectra): # for each input observed spectrum
			print(f'\nFor input spectrum {i+1} of {N_spectra}')
			if not skip_convolution: # read and convolve original model spectra
				grid_each = read_grid(model=model, model_dir=model_dir, params_ranges=params_ranges, 
				                      convolve=True, res=res[i], lam_res=lam_res[i], 
				                      fit_wl_range=fit_wl_range[i], wl_resample=wl_spectra_fit[i])
			else: # read model spectra already convolved to the data resolution
				# set filename_pattern to look for model spectra with the corresponding resolution
				filename_pattern.append(Models(model).filename_pattern+f'_R{res[i]}at{lam_res[i]}um.nc')
				grid_each = read_grid(model=model, model_dir=model_dir, params_ranges=params_ranges, 
				                      res=res[i], lam_res=lam_res[i], 
				                      fit_wl_range=fit_wl_range[i], wl_resample=wl_spectra_fit[i], 
				                      skip_convolution=skip_convolution, filename_pattern=filename_pattern[i])
			# add resampled grid for each input spectrum to the same list
			grid.append(grid_each)

	# generate a synthetic spectrum with the median parameter values for only the free parameters in the models
	# (avoid radius, if included in params_med)
	params = {}
	for param in params_models: # for each free parameter in the grid
		params[param] = params_med[param]
	wl = []
	flux = []
	for i in range(N_spectra): # for each input observed spectrum
		syn_spectrum = generate_model_spectrum(params=params, model=model, grid=grid[i])
		wl.append(syn_spectrum['wavelength'])
		flux.append(syn_spectrum['flux'])

	# scale synthetic spectrum
	if distance is not None: # when radius was constrained
		wl_scaled = []
		flux_scaled = []
		for i in range(N_spectra): # for each input observed spectrum
			flux_scaled.append(scale_synthetic_spectrum(wl=wl[i], flux=flux[i], distance=distance, radius=params_med['R']))
			wl_scaled.append(wl[i])
	else: # when radius was not constrained
		wl_scaled = []
		flux_scaled = []
		for i in range(N_spectra): # for each input observed spectrum
			# DOUBLE CHECK
			scaling = np.sum(flux_obs[i]*flux/eflux_obs[i]**2) / np.sum(flux_model**2/eflux_obs[i]**2) # scaling that minimizes chi2
			flux_scaled.append(scaling*flux)
			wl_scaled.append(wl)
		
	# read best fits with the original resolution
	if ori_res:
		if skip_convolution: # when the convolution was skipped
			if model_dir_ori is None: raise Exception(f'parameter "model_dir_ori" is needed to read model spectra with the original resolution')
			else: model_dir = model_dir_ori
		# generate spectrum
		syn_spectrum = generate_model_spectrum(params=params, model=model, model_dir=model_dir)
		wl_model_ori = syn_spectrum['wavelength']
		flux_model_ori = syn_spectrum['flux']

		# scale synthetic spectrum
		flux_model_ori = scale_synthetic_spectrum(wl=wl_model_ori, flux=flux_model_ori, distance=distance, radius=params_med['R'])

	# output dictionary
	out = {'wl_spectra_fit': wl_spectra_fit, 'flux_spectra_fit': flux_spectra_fit, 'eflux_spectra_fit': eflux_spectra_fit, 
	       'wl_model': wl_scaled, 'flux_model': flux_scaled, 'params_med': params_med, 'params_confidence_interval': params_conf, 'confidence_interval(%)': conf_interval}
	if ori_res:
		out['wl_model_ori'] = wl_model_ori
		out['flux_model_ori'] = flux_model_ori

	return out

##########################
def select_model_spectra(model, model_dir, params_ranges=None, filename_pattern=None, save_results=False, out_file=None):
	'''
	Description:
	------------
		Select model spectra from the indicated models and meeting given parameters ranges.

	Parameters:
	-----------
	- model : str
		Atmospheric models. See available models in ``seda.Models().available_models``.  
	- model_dir : str or list
		Path to the directory (str or list) or directories (as a list) containing the model spectra (e.g., ``model_dir = ['path_1', 'path_2']``). 
	- params_ranges : dictionary, optional
		Minimum and maximum values for any model free parameters to select a model grid subset.
		E.g., ``params_ranges = {'Teff': [1000, 1200], 'logg': [4., 5.]}`` to consider spectra within those Teff and logg ranges.
		If a parameter range is not provided, the full range in ``model_dir`` is considered.
	- filename_pattern : str, optional
		Pattern to select only files including it.
		Default is a common pattern in all spectra original filenames in ``model``, as indicated by ``Models(model).filename_pattern``.
	- save_results : {``True``, ``False``}, optional (default ``False``)
		Save (``True``) or do not save (``False``) the output as a pickle file named '``model``\_free\_parameters.pickle'.
	- out_file : str, optional
		File name to save the results as a pickle file (it can include a path e.g. my_path/free\_params.pickle).
		Default name is '``model``\_free_parameters.pickle' and is stored at the notebook location.

	Returns:
	--------
	Dictionary with the parameters:
		- ``spectra_name``: selected model spectra names.
		- ``spectra_name_full``: selected model spectra names with full path.
		- ``params``: parameters for the selected model spectra, as given by the ``seda.separate_params`` output dictionary.

	Example:
	--------
	>>> import seda
	>>> 
	>>> model = 'Sonora_Elf_Owl'
	>>> model_dir = ['my_path/output_575.0_650.0/', 
	>>>              'my_path/output_700.0_800.0/'] # folders to seek model spectra
	>>> # set ranges for some (Teff and logg) free parameters to select only a grid subset
	>>> params_ranges = {'Teff': [700, 900], 'logg': [4.0, 5.0]}
	>>> out = seda.select_model_spectra(model=model, model_dir=model_dir,
	>>>                                 params_ranges=params_ranges)

	Author: Genaro Suárez

	Date: 2020-05
	'''

	# make sure model_dir is a list
	model_dir = var_to_list(model_dir)
	
	if isinstance(model_dir, str): model_dir = [model_dir]
	if isinstance(model_dir, np.ndarray): model_dir = model_dir.tolist()

	# set default parameters
	# if params_ranges is provided, verified that there is a minimum and a maximum values for each provided parameter
	if params_ranges is not None:
		for param in params_ranges:
			if len(params_ranges[param])!=2: raise Exception(f'{param} in "params_ranges" must have two values (minimum and maximum), but {len(params_ranges[param])} values were given')
	# if params_ranges is not provided, define params_ranges as an empty dictionary
	if params_ranges is None: params_ranges = {} # empty dictionary
	# if filename_pattern is not provided, consider the common pattern in original file names
	if filename_pattern is None: filename_pattern = Models(model).filename_pattern

	# to store files in model_dir
	files = [] # with full path
	files_short = [] # only spectra names
	for i in range(len(model_dir)):
		files_model_dir = fnmatch.filter(os.listdir(model_dir[i]), filename_pattern)
		files_model_dir.sort() # just to sort the files wrt their names
		for file in files_model_dir:
			files.append(model_dir[i]+file)
			files_short.append(file)

	# read free parameters from each model spectrum
	out_separate_params = separate_params(model=model, spectra_name=files_short)
	params_spectra = out_separate_params['params']

	# select spectra within the desired parameter ranges
	spectra_name_full = [] # full name with path
	spectra_name = [] # only spectra names

	# select spectra within the desired parameter ranges
	spectra_name_full = [] # full name with path
	spectra_name = [] # only spectra names
	for i in range(len(files)): # for each file in model_dir
		mask = True # initialize mask to apply parameter ranges criteria
		for param in params_spectra: # for each free parameter
			if param in params_ranges: # if there is an input range to constrain the free parameter
				if not params_ranges[param][0] <= params_spectra[param][i] <= params_ranges[param][1]: 
					mask = False # change mask if the file is for a spectrum out of the input ranges
		if mask: # keep only spectra within the input parameter ranges
			spectra_name_full.append(files[i]) # file with full path
			spectra_name.append(files_short[i]) # only file name

	if len(spectra_name_full)==0: 
		# show up an error when there are no models in the indicated ranges
		raise Exception(f'No model spectra in "model_dir": {model_dir} within params_ranges: {params_ranges}')
	else: 
		if not params_ranges: 
			print(f'\n      {len(spectra_name)} model spectra')
		else:
			print(f'\n      {len(spectra_name)} model spectra selected with:')
			for param in params_ranges:
				print(f'         {param} range = {params_ranges[param]}')

	# convert lists into numpy arrays
	spectra_name_full = np.array(spectra_name_full)
	spectra_name = np.array(spectra_name)

	# separate parameters from selected spectra
	out_separate_params = separate_params(model=model, spectra_name=spectra_name, save_results=save_results, out_file=out_file)

	out = {'spectra_name_full': spectra_name_full, 'spectra_name': spectra_name, 'params': out_separate_params['params']}

	return out

##########################
def read_SVO_params(filters, params):
	'''
	Description:
	------------
		Read parameters of interest from SVO for a list of filters.

	Parameters:
	-----------
	- filters : list or array
		Filter names to retrieve parameters from SVO.
	- params : list or array
		Parameter of interest following SVO names. 

	Returns:
	--------
	Dictionary with the parameters ``params`` for the input filters.

	Example:
	--------
	>>> import seda
	>>> 
	>>> filters = ['2MASS/2MASS.J', '2MASS/2MASS.H', '2MASS/2MASS.Ks']
	>>> params = ['filterID', 'WavelengthEff', 'WidthEff']
	>>> seda.read_SVO_params(filters=filters, params=params)
	    {'filterID': array(['2MASS/2MASS.J', '2MASS/2MASS.H', '2MASS/2MASS.Ks'], dtype=object),
	     'WavelengthEff': array([12350., 16620., 21590.]),
	     'WidthEff': array([1624.3190191 , 2509.40349871, 2618.86953322])}

	Author: Genaro Suárez

	Date: 2025-09-05
	'''

#	# verify input parameters are numpy arrays
#	filters = var_to_numpy(filters)
#	params = var_to_numpy(params)

	# read SVO table
	SVO_data = read_SVO_table()
	SVO_filterID = SVO_data['filterID'] # SVO ID

	# verify that all params are in SVO
	params_good = []
	params_bad = []
	for param in params:
		if param in SVO_data.colnames:
			params_good.append(param)
		else:
			params_bad.append(param)
	params = params_good
	if len(params_bad)>0: print(f'Parameters {params_bad} are not in SVO, so will be ignored')

	# indices in SVO matching input filters
	mask = np.isin(SVO_filterID, filters)

	# subset of the SVO table for the desired filters
	# for clarity add the parameter 'filterID', if not requested
	params_ori = params.copy() # copy to save input params recognized by SVO
	if 'filterID' not in params: params.insert(0, 'filterID')
	SVO_data_sel = SVO_data[mask][params]

	# dictionary with the table subset
	filters_params = {}
	for param in params:
		filters_params[param] = SVO_data_sel[param].data.data

	# sort dictionary so filters_params['filterID'] is in the same order as input filters
	# indices that sort filters_params['filterID']
	sort_ind = np.array([], dtype=int) # initialize array of integers for indices
	for i,filt in enumerate(filters):
		if filt in filters_params['filterID']:
			ind = np.where(filters_params['filterID']==filt)[0]
			sort_ind = np.append(sort_ind, ind)
	# sort dictionary
	for param in params:
		filters_params[param] = filters_params[param][sort_ind]

	# input filters not in SVO
	mask_noinSVO = ~np.isin(filters, filters_params['filterID'])
	filters = var_to_numpy(filters)
	if len(filters[mask_noinSVO])>0: print(f'Filters {filters[mask_noinSVO]} are not in SVO, so will be ingored')

	# remove 'filterID' if it wasn't requested
	if 'filterID' not in params_ori: del filters_params['filterID']

	return filters_params

##########################
# set wavelength range for the model comparison via chi-square or Bayes techniques
def set_fit_wl_range(fit_wl_range, N_spectra, wl_spectra):

	# handle fit_wl_range
	if fit_wl_range is None: # define fit_wl_range when not provided
		fit_wl_range = np.zeros((N_spectra, 2)) # Nx2 array, N:number of spectra and 2 for the minimum and maximum values for each spectrum
		for i in range(N_spectra):
			fit_wl_range[i,:] = np.array((wl_spectra[i].min(), wl_spectra[i].max()))
	elif isinstance(fit_wl_range, list): # when it is a list
		fit_wl_range = np.array(fit_wl_range).reshape(1,2)
	else: # fit_wl_range is provided
		if len(fit_wl_range.shape)==1: fit_wl_range = fit_wl_range.reshape((1, 2)) # reshape fit_wl_range array

	return fit_wl_range

##########################
# set wavelength range for the model comparison via chi-square or Bayes techniques
def set_disp_wl_range(disp_wl_range, N_spectra, wl_spectra):

	# same handling as for fit_wl_range
	disp_wl_range = set_fit_wl_range(fit_wl_range=disp_wl_range, N_spectra=N_spectra, wl_spectra=wl_spectra)

	return disp_wl_range

##########################
# set wavelength range to cut models for comparisons via chi-square or Bayes techniques
def set_model_wl_range(model_wl_range, wl_spectra_min, wl_spectra_max):

	# define model_wl_range (if not provided) in terms of input spectra coverage
	if model_wl_range is None:
		#model_wl_range = np.array([0.9*wl_spectra_min, 1.1*wl_spectra_max]) # add padding to have enough spectral coverage in models
		model_wl_range = add_pad(wl_spectra_min, wl_spectra_max) # add padding to have enough spectral coverage in models

#	# it may need an update to work with multiple spectra
#	if (model_wl_range.min()>=fit_wl_range.min()):
#		model_wl_range[0] = 0.9*fit_wl_range.min() # add padding to shorter wavelengths
#	if (model_wl_range.max()<=fit_wl_range.max()):
#		model_wl_range[1] = 1.1*fit_wl_range.max() # add padding to longer wavelengths

	return model_wl_range

##########################
# set wavelength range for photometry to cut models for comparisons via chi-square or Bayes techniques
def set_fit_phot_range(fit_phot_range, filters):
	if fit_phot_range is None: # define fit_phot_range when not provided
		# get effective wavelengths from SVO for the input filters
		SVO_data = read_SVO_table()
		SVO_filterID = SVO_data['filterID'] # SVO ID
		SVO_WavelengthEff = (SVO_data['WavelengthEff'].data*u.angstrom).to(u.micron).value # effective wavelength in um
		matching_indices = []
		for index, element in enumerate(SVO_filterID):
			if element in filters:
				matching_indices.append(SVO_WavelengthEff[index])
		lambda_eff_SVO = np.array(matching_indices)
		fit_phot_range = np.array([lambda_eff_SVO.min(), lambda_eff_SVO.max()])
	elif isinstance(fit_phot_range, list): # when it is a list
		fit_phot_range = np.array(fit_phot_range).reshape(1,2)

	return fit_phot_range

##########################
# read the SVO table with filter properties and save it locally if it doesn't already exist
def read_SVO_table():
	dir_sep = os.sep # directory separator for the current operating system
	path_synthetic_photometry = os.path.dirname(__file__)+dir_sep
	# read zero points for each filter
	svo_table = f'{path_synthetic_photometry}synthetic_photometry{dir_sep}FPS_info.xml'
	if os.path.exists(svo_table): 
		svo_data = Table.read(svo_table, format='votable') # open downloaded table with filters' info
	else:
		svo_data = Table.read('https://svo.cab.inta-csic.es/files/svo/Public/HowTo/FPS/FPS_info.xml', format='votable') # this SVO link will be updated as soon as new filters are added to FPS. 
		svo_data.write(svo_table, format='votable') # save the table to avoid reading it from the web each time the code is run, which can take a few seconds

	return svo_data

##########################
# function to generate a padding on both sizes of a range
def add_pad(min_value, max_value):

	array = np.array([0.9*min_value, 1.1*max_value]) # add padding to have enough spectral coverage in models

	return array

##########################
# set wavelength of reference associated to a given resolution as the median wavelength for a spectrum
def set_lam_res(wl_spectrum):
	lam_res = np.median(wl_spectrum)
	#lam_res = round(lam_res)
	return lam_res

##########################
def app_to_abs_flux(flux, distance, eflux=None, edistance=None, reverse=False):
	'''
	Description:
	------------
		Convert apparent fluxes into absolute fluxes considering a distance.

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
	- reverse : {``True``, ``False``}, optional (default ``False``)
		Label to indicate if apparent fluxes are converted into absolute fluxes (``False``) or vice versa (``True``).

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

	Date: 2025-05
	'''

	# set non-provided params
	if eflux is None: eflux=0
	if edistance is None: edistance=0

	# use numpy arrays
	flux = astropy_to_numpy(flux)
	eflux = astropy_to_numpy(eflux)

	if not reverse: # apparent fluxes to absolute fluxes
		# absolute fluxes
		flux_abs = flux * (distance/10.)**2
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

	else: # absolute fluxes to apparent fluxes
		# apparent fluxes
		flux_app = flux / (distance/10.)**2 
		# absolute flux errors
		eflux_app = flux_app*np.sqrt((2.*edistance/distance)**2 + (eflux/flux)**2)

		# output dictionary
		out = {'flux_app': flux_app, 'flux_abs': flux}

		if isinstance(eflux, np.ndarray): # if eflux is an array
			out['eflux_app'] = eflux_app
			out['eflux_abs'] = eflux
		elif eflux!=0: # if flux is a float non-equal to zero
			out['eflux_app'] = eflux_app
			out['eflux_abs'] = eflux
		out['distance'] = distance
		if edistance!=0:
			out['edistance'] = edistance
			out['eflux_app'] = eflux_app # if eflux_app was stored above it would replace it without issues

	return out

##########################
def app_to_abs_mag(magnitude, distance, emagnitude=None, edistance=None):
	'''
	Description:
	------------
		Convert apparent magnitudes into absolute fluxes considering a distance, assuming no extinction.

	Parameters:
	-----------
	- magnitude : float array or float
		Magnitude (in any magnitude units).
	- distance : float
		Target distance (in pc).
	- emagnitude : float array or float, optional
		Magnitude uncertainties (in any magnitude units).
	- edistance : float, optional
		Distance error (in pc).

	Returns:
	--------
	- Dictionary with absolute mages and input parameters:
		- ``'mag_abs'`` : absolute mages in the same units as the input mages.
		- ``'emag_abs'`` : (if ``emag`` or ``edistance`` is provided) absolute mag uncertainties.
		- ``'mag_app'`` : input apparent mages.
		- ``'emag_app'`` : (if provided) input apparent mag errors.
		- ``'distance'`` : input distance.
		- ``'edistance'`` : (if provided) input distance error.

	Example:
	--------
	>>> import seda
	>>> magnitude, emagnitude = 17.05, 0.03 # mag
	>>> distance, edistance = 14.43, 0.79 # pc
	>>>
	>>> seda.app_to_abs_mag(magnitude=magnitude, emagnitude=emagnitude, 
	>>>                     distance=distance, edistance=edistance)
	    {'mag_app': 17.05,
	     'mag_abs': 16.253668344532528,
	     'emag_app': 0.03,
	     'emag_abs': 0.12260857671946264,
	     'distance': 14.43,
	     'edistance': 0.79}

	Author: Genaro Suárez

	Date: 2025-04-28
	'''

	# set non-provided params
	if emagnitude is None: emagnitude=0
	if edistance is None: edistance=0

	# use numpy arrays
	magnitude = astropy_to_numpy(magnitude)
	emagnitude = astropy_to_numpy(emagnitude)
	distance = astropy_to_numpy(distance)
	edistance = astropy_to_numpy(edistance)

	# absolute magnitudes
	mag_abs = magnitude - 5.*np.log10(distance) + 5.

	# absolute magnitude errors
	emag_abs = np.sqrt(emagnitude**2 + ((5./np.log(10))*edistance/distance)**2)
	
	# output dictionary
	out = {'mag_app': magnitude, 'mag_abs': mag_abs}

	if isinstance(emagnitude, np.ndarray): # if emag is an array
		out['emag_app'] = emagnitude
		out['emag_abs'] = emag_abs
	elif emagnitude!=0: # if mag is a float non-equal to zero
		out['emag_app'] = emagnitude
		out['emag_abs'] = emag_abs

	out['distance'] = distance
	if isinstance(edistance, np.ndarray): # if edistance is an array
		out['edistance'] = edistance
	elif edistance!=0: # if distance is a float non-equal to zero
		out['edistance'] = edistance

	return out

##########################
def spt_to_teff(spt, spt_type, ref=None):
	'''
	Description:
	------------
		Convert spectral type into effective temperature using relationships in the literature.

	Parameters:
	-----------
	- spt : float, str, array, or list
		Spectral types given as conventional letters or number, as indicated in ``spt_type``.
		Convention between spectral type an float: M9=9, L0=10, ..., T0=20, ...
	- spt_type : str
		Label indicating whether the input spectral type is a string ('str') or a number ('float')
	- ref : str, optional (default 'F15')
		Reference for the spectral type-temperature relationships.
		'F15': Filippazzo et al. (2015), valid for M6-T9 (6-29)
		'K21': Kirkpatrick et al. (2021), valid for M7-Y2 (7-32)

	Returns:
	- effective temperature (in K) corresponding to the input spectral types according to the ``ref`` reference.

	Example:
	--------
	>>> import seda
	>>> 
	>>> # spectral type as a number
	>>> spt = [15, 25] # for L5 and T5 types
	>>> seda.spt_to_teff(spt, spt_type='float') # Teff in K
	    array([1581.053125, 1033.328125])
	>>> 
	>>> # spectral type as a string
	>>> spt = ['L5', 'T5']
	>>> seda.spt_to_teff(spt, spt_type='str') # Teff in K
	    array([1581.053125, 1033.328125])

	Author: Genaro Suárez

	Date: 2023-02
	'''

	# make sure the input spt is a numpy array
	spt = var_to_numpy(spt)

	# verify that spt_type is provided
	try: spt_type
	except: raise Exception(f'"spt_type" must be provided')

	# assigned default values
	if spt_type is None: spt_type = 'float' # spectral type as float number
	if ref is None: ref = 'F15' # Filippazzo et al. (2015)

	# save the original spt
	spt_ori = spt.copy()

	# if spt is provided as strings, convert it into a float
	if spt_type=='str': 
		for i in range(len(spt)):
			spt[i] = spt_str_to_float(spt[i])
		# convert array of string to an array of floats
		spt = spt.astype(float)

	# verify the spectral type is within the valid range for the indicated relationship
	# valid range for available relationships
	if ref=='F15': spt_valid = ['M6', 'T9'] #[6, 29]
	if ref=='K21': spt_valid = ['M7', 'Y2'] #[7, 32] (Table 13 in K21 says the range is L0-Y2, but the fit in Fig. 22 covers M7-Y2)

	# verification
	spt_valid_flt = [spt_str_to_float(spt_valid[0]), spt_str_to_float(spt_valid[1])]
	for i,sp in enumerate(spt):
		if sp<spt_valid_flt[0] or sp>spt_valid_flt[1]: 
			print(f'Caveat for spt={spt_ori[i]}: it is out of the "{ref}" coverage, so Teff was extrapolated.')
			print(f'   The valid spt range is {spt_valid} ({spt_valid_flt})')

	# polynomial fits
	if ref=='F15': 
		# coefficients of the polynomial
		c0 = 4.747e+03
		c1 = -7.005e+02
		c2 = 1.155e+02
		c3 = -1.191e+01
		c4 = 6.318e-01
		c5 = -1.606e-02
		c6 = 1.546e-04
	
		teff = c0 + c1*spt + c2*spt**2 + c3*spt**3 + c4*spt**4 + c5*spt**5 + c6*spt**6

	if ref=='K21': 
		teff = np.zeros(len(spt))
		for i,sp in enumerate(spt):
			sp = sp-10 # L0 in K21 is 0 instead of 10 as in F15

			if sp<=8.75: # L0-L8.75
				c0 = 2.2375e+03
				c1 = -1.4496e+02
				c2 = 4.0301e+00
			elif (sp>8.75) & (sp<=14.75): # L8.75-T4.75
				c0 = 1.4379e+03
				c1 = -1.8309e+01
				c2 = 0
			else: # T4.75-Y2
				c0 = 5.1413e+03
				c1 = -3.6865e+02
				c2 = 6.7301e+00

			teff[i] = c0 + c1*sp + c2*sp**2
	
	return teff

##########################
def parallax_to_distance(parallax, eparallax):
	'''
	Description:
	------------
		Obtain distance as the inverse of the parallax.

	Parameters:
	-----------
	- parallax : float, array
		Parallax in mas.
	- eparallax : float, array
		Parallax uncertainty in mas.

	Returns:
	--------
	- distance : float, array
		Distance in pc.
	- edistance : float, array
		Distance uncertainty in pc.

	Example:
	--------
	>>> import seda
	>>> parallax = 175.2 # mas
	>>> eparallax = 1.7 # mas
	>>> # distance
	>>> seda.parallax_to_distance(parallax=parallax, eparallax=eparallax)
	    (5.707762557077626, 0.055383540793561434)

	Author: Genaro Suárez

	Date: 2025-04-28
	'''

	distance = 1000. / parallax # in pc

	if eparallax is not None:
		edistance = distance * eparallax / parallax # pc

	return distance, edistance

##########################
def convert_photometric_table(table, save_table=False, table_name=None):
	'''
	Description:
	------------
		Convert a table with photometry in separate columns to a table with all magnitudes or fluxes in a column and all errors in another column. 

	Parameters:
	-----------
	- table : astropy table
		Table with photometric measurements and their corresponding errors listed in separate columns. 
		The magnitude or flux columns must be labeled with the corresponding SVO filter names. 
		The table must include only photometry, structured such that each magnitude or flux column is 
		immediately followed by its associated error column i.e. magnitude1, error1, magnitude2, error2, etc.
		Note: names of columns with photometric errors are irrelevant.
	- save_table : {``True``, ``False``}, optional (default ``False``)
		Locally store (``True``) or do not store (``False``) the output dictionary as an ascii table using PrettyTable.
	- table_name : str, optional
		File name to save the output table (it can include a path e.g. my_path/output_table.dat). 
		Default name is 'photometry_prettytable.dat'.

	Returns:
	--------
	Dictionary with three keys: SVO filter names, magnitudes or fluxes, and corresponding errors. 

	Example:
	--------
	>>> import seda
	>>> from astropy.io import ascii
	>>> 
	>>> # path to the seda package
	>>> path_seda = os.path.dirname(os.path.dirname(seda.__file__))
	>>> # read ascii table with photometry for 0415
	>>> phot_file = path_seda+'/docs/notebooks/data/0415-0935_photometry.dat'
	>>> photometry = ascii.read(phot_file)
	>>> 
	>>> # keep columns with magnitudes of interest
	>>> photometry = photometry['2MASS/2MASS.J', '2MASS/2MASS.eJ', 
	>>>                         '2MASS/2MASS.H', '2MASS/2MASS.eH', 
	>>>                         '2MASS/2MASS.Ks', '2MASS/2MASS.eKs']
	>>> 
	>>> # convert table
	>>> seda.convert_photometric_table(photometry)
	    {'filters': array(['2MASS/2MASS.J', '2MASS/2MASS.H', '2MASS/2MASS.Ks'], dtype='<U32'),
	     'phot': array([15.695, 15.537, 15.429]),
	     'ephot': array([0.058, 0.113, 0.201])}

	Author: Genaro Suárez

	Date: 2025-09-07
	'''

	# initialize arrays to save filter names, values, and uncertainties
	phot = np.array([])
	ephot = np.array([])
	filters = np.array([])
	# create arrays for all values, errors, and filters
	for i,col in enumerate(table.colnames): # for each column
		if i%2 == 0: # only even columns (the ones with photometric values)
			phot = np.append(phot, table[col])
			filters = np.append(filters, col)
		else: # odd columns (the ones with uncertainties)
			ephot = np.append(ephot, table[col])

	# save all as a dictionary
	out = {'filters': filters, 'phot': phot, 'ephot': ephot}

	if save_table: 
		if table_name is None: table_name = 'photometry_prettytable.dat'
		save_prettytable(my_dict=out, table_name=table_name)

	return out

##########################
def read_prettytable(filename):
	'''
	Description:
	------------
		Read ascii table created with ``prettytable``.

	Parameters:
	-----------
	- filename : str
		PrettyTable table.

	Returns:
	--------
	Astropy table with the information in the input file.

	Example:
	--------
	>>> import seda
	>>> 
	>>> out = seda.open_chi2_table('Sonora_Elf_Owl_chi2_minimization_multiple_spectra.dat')

	Author: Rocio Kiman
	'''

	# open table content
	with open(filename, 'r') as f:

		# read all lines
		lines = f.readlines()

		# keep only lines that do not start with "+"
		lines_sel = []
		for line in lines:
			if not line.startswith('+'):
				lines_sel.append(line)

		# read the cleaned data using ascii.read
		table = ascii.read(lines_sel, format='fixed_width')

		# remove last column with empty information
		last_column_name = table.colnames[-1]
		table.remove_column(last_column_name)

		# convert table into a dictionary
		my_dict = {}
		for col in table.columns:
			my_dict[col] = table[col].data

	return my_dict

##########################
# save dictionary as ascii table using prettytable
def save_prettytable(my_dict, table_name):

	# create a PrettyTable object
	table = PrettyTable()

	# add the dictionary keys as column headers
	table.field_names = my_dict.keys()

	# add the dictionary values as rows
	for row in zip(*my_dict.values()):
		table.add_row(row)
	
	# get the ASCII string representation
	ascii_table = table.get_string()

	# save file
	with open(table_name, 'w') as f:
		f.write(ascii_table)

##########################
# convert spectral type from string to float
def spt_str_to_float(spt):

	if 'M' in spt: spt = spt.replace('M', '0')
	if 'L' in spt: spt = spt.replace('L', '1')
	if 'T' in spt: spt = spt.replace('T', '2')
	if 'Y' in spt: spt = spt.replace('Y', '3')

	try:
		spt = float(spt)
	except:
		raise Exception(f'"{spt}" is not recognized. Overall, "spt" can be M, L, T, and Y with any subtypes.')

	return spt

##########################
# make sure a variable is a list
def var_to_list(x):
	if isinstance(x, str): x = [x]
	if isinstance(x, np.ndarray): x = x.tolist()
	if isinstance(x, float): x = [x]

	return x

##########################
# make sure a variable is a numpy array
def var_to_numpy(x):

	if isinstance(x, (str, int, float)): x = np.array([x])
	if isinstance(x, list): x = np.array(x)

	return x

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

#+++++++++++++++++++++++++++
# count the total number of data points in all input spectra
def input_data_stats(wl_spectra):

	N_spectra = len(wl_spectra)

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

#+++++++++++++++++++++++++++
# find the data point before and after a given value
def find_two_nearest(array, value):
	diff = array - value
	if all(diff<0) or all(diff>0): # value is out of the array coverage
		raise Exception('Parameter does not cover by the model grid')
	elif any(diff==0): # if the value is a grid node
		near_low = near_high = array[diff==0][0]
	else: # if the value is between two grid nodes
		near_low = array[diff<0].max()
		near_high = array[diff>0].min()

	return np.array([near_low, near_high])

#+++++++++++++++++++++++++++
def find_two_around_node(array, value):
	diff = array - value
	if any(diff<0): # if there are grid nodes smaller than the value
		near_low = array[diff<0].max()
	else: # if the value is the smallest grid point
		near_low = array.min()
		
	if any(diff>0): # if there are grid nodes greater than the value
		near_high = array[diff>0].min()
	else: # if the value is the greatest grid point
		near_high = array.max()

	return np.array([near_low, near_high])

#+++++++++++++++++++++++++++
# maximum number of decimals in an array elements
def max_decimals(arr):
	max_places = 0
	for num in arr:
		if isinstance(num, float): # if the element is a float
			num_str = str(num) 
			decimal_part = num_str.split('.')[1] # select decimals as a string
			max_places = max(max_places, len(decimal_part)) # compare decimals
	return max_places

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
