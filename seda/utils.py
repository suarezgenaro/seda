import numpy as np
import time
import os
import fnmatch
import xarray
import pickle
#import itertools
from spectres import spectres
from astropy import units as u
from astropy.io import ascii
from astropy.table import Column, MaskedColumn
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm
from specutils.utils.wcs_utils import vac_to_air
from sys import exit
from .models import *

##########################
def convolve_spectrum(wl, flux, res, lam_res, eflux=None, disp_wl_range=None, convolve_wl_range=None, out_file=None):
	'''
	Description:
	------------
		Convolve a spectrum to a desired resolution at a given wavelength.

	Parameters:
	-----------
	- wl : float array
		W, optionalavelength (any length units) for the spectrum.
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

###########################
#def model_points(model):
#	'''
#	Description:
#	------------
#		Maximum number of data points in the model spectra.
#
#	Parameters:
#	-----------
#	- model : str
#		Atmospheric models. See available models in ``input_parameters.ModelOptions``.  
#
#	Returns:
#	--------
#	- N_modelpoints: int
#		Maximum number of data points in the model spectra.
#
#	Author: Genaro Suárez
#	'''
#
#	if (model == 'Sonora_Diamondback'):	N_modelpoints = 385466 # number of rows in model spectra (all spectra have the same length)
#	if (model == 'Sonora_Elf_Owl'):	N_modelpoints = 193132 # number of rows in model spectra (all spectra have the same length)
#	if (model == 'LB23'): N_modelpoints = 30000 # maximum number of rows in model spectra
#	if (model == 'Sonora_Cholla'): N_modelpoints = 110979 # maximum number of rows in spectra of the grid
#	if (model == 'Sonora_Bobcat'): N_modelpoints = 362000 # maximum number of rows in spectra of the grid
#	if (model == 'ATMO2020'): N_modelpoints = 5000 # maximum number of rows of the ATMO2020 model spectra
#	if (model == 'BT-Settl'): N_modelpoints = 1291340 # maximum number of rows of the BT-Settl model spectra
#	if (model == 'SM08'): N_modelpoints = 184663 # rows of the SM08 model spectra
#
#	return N_modelpoints

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
def best_chi2_fits(chi2_pickle_file, N_best_fits=1, model_dir_ori=None, ori_res=None):
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
	- ori_res : {``True``, ``False``}, optional (default ``False``)
		Read (``True``) or do not read (``False``) model spectra with the original resolution.
	- model_dir_ori : str, list, or array
		Path to the directory (str, list, or array) or directories (as a list or array) containing the model spectra with the original resolution.
		This parameter is needed if ``ori_res`` is True and `chi2_fit.chi2` was run skipping the model spectra convolution (if `skip_convolution`` is True).

	Returns:
	--------
	Dictionary with best model fits:
		- ``'spectra_name_best'`` : model spectra names
		- ``'chi2_red_fit_best'`` : reduced chi-square
		- ``'wl_model_best'`` : (if ``ori_res``) wavelength (in um) of original model spectra.
		- ``'flux_model_best'`` : (if ``ori_res``) fluxes (in erg/s/cm2/A) of original model spectra.
		- ``'wl_array_model_conv_resam_best'`` : wavelength (in um) of resampled, convolved model spectra in the input spectra ranges.
		- ``'flux_array_model_conv_resam_best'`` : fluxes (in erg/s/cm2/A) of resampled, convolved model spectra in the input spectra ranges.
		- ``'wl_array_model_conv_resam_fit_best'`` : wavelength (in um) of resampled, convolved model spectra within the fit range.
		- ``'flux_array_model_conv_resam_fit_best'`` : fluxes (in erg/s/cm2/A) of resampled, convolved model spectra within the fit range.
		- ``'flux_residuals_best'`` : flux residuals in linear scale between model and input spectra.
		- ``'logflux_residuals_best'`` : flux residuals in log scale between model and input spectra.
		- ``'params'`` : model free parameters

	Author: Genaro Suárez
	'''

	# open results from the chi square analysis
	with open(chi2_pickle_file, 'rb') as file:
		out_chi2 = pickle.load(file)
	model = out_chi2['my_chi2'].model
	res = out_chi2['my_chi2'].res#[0] # resolution for the first input spectrum
	lam_res = out_chi2['my_chi2'].lam_res#[0] # lam_resolution for the first input spectrum
	N_modelpoints = out_chi2['my_chi2'].N_modelpoints
	N_model_spectra = out_chi2['N_model_spectra']
	spectra_name_full = out_chi2['spectra_name_full']
	spectra_name = out_chi2['spectra_name']
	chi2_red_fit = out_chi2['chi2_red_fit']
	chi2_red_wl_fit = out_chi2['chi2_red_wl_fit'] # list of reduced chi-square for each model compared
	scaling_fit = out_chi2['scaling_fit']
	wl_array_model_conv_resam = out_chi2['wl_array_model_conv_resam'] # wavelengths of resampled, convolved model spectra in the input spectra wavelength ranges
	flux_array_model_conv_resam = out_chi2['flux_array_model_conv_resam'] # fluxes of resampled, convolved model spectra in the input spectra wavelength ranges
	wl_array_model_conv_resam_fit = out_chi2['wl_array_model_conv_resam_fit'] # wavelengths of resampled, convolved model spectra within the fit ranges
	flux_array_model_conv_resam_fit = out_chi2['flux_array_model_conv_resam_fit'] # fluxes of resampled, convolved model spectra within the fit ranges
	flux_residuals = out_chi2['flux_residuals']
	logflux_residuals = out_chi2['logflux_residuals']
	skip_convolution = out_chi2['my_chi2'].skip_convolution

	# select the best fits given by N_best_fits
	if (N_best_fits > N_model_spectra): raise Exception(f"not enough model spectra (only {N_model_spectra}) in the fit to select {N_best_fits} best fits")
	sort_ind = np.argsort(chi2_red_fit)
	chi2_red_fit_best = chi2_red_fit[sort_ind][:N_best_fits]
	chi2_red_wl_fit_best = [chi2_red_wl_fit[i] for i in sort_ind][:N_best_fits]
	scaling_fit_best = scaling_fit[sort_ind][:N_best_fits]
	spectra_name_full_best = spectra_name_full[sort_ind][:N_best_fits]
	spectra_name_best = spectra_name[sort_ind][:N_best_fits]
	wl_array_model_conv_resam_best = [wl_array_model_conv_resam[i] for i in sort_ind][:N_best_fits]
	flux_array_model_conv_resam_best = [flux_array_model_conv_resam[i] for i in sort_ind][:N_best_fits]
	wl_array_model_conv_resam_fit_best = [wl_array_model_conv_resam_fit[i] for i in sort_ind][:N_best_fits]
	flux_array_model_conv_resam_fit_best = [flux_array_model_conv_resam_fit[i] for i in sort_ind][:N_best_fits]
	flux_residuals_best = [flux_residuals[i] for i in sort_ind][:N_best_fits]
	logflux_residuals_best = [logflux_residuals[i] for i in sort_ind][:N_best_fits]

	# read parameters from file name for the best fits
	out_separate_params = separate_params(model=model, spectra_name=spectra_name_best)

	# read best fits with the original resolution
	if ori_res:
		wl_model_best = np.zeros((N_best_fits, N_modelpoints))
		flux_model_best = np.zeros((N_best_fits, N_modelpoints))
		if not skip_convolution: # when the convolution was not skipped
			for i in range(N_best_fits):
				out_read_model_spectrum = read_model_spectrum(spectra_name_full=spectra_name_full_best[i], model=model)
				wl_model_best[i,:] = out_read_model_spectrum['wl_model']
				flux_model_best[i,:] = scaling_fit_best[i] * out_read_model_spectrum['flux_model'] # scaled fluxes
		else:
			if model_dir_ori is None: raise Exception(f"parameter 'model_dir_ori' is needed to read model spectra with the original resolution")
			else:
				out_select_model_spectra = select_model_spectra(model_dir=model_dir_ori, model=model) # all spectra in model_dir_ori
				for i in range(N_best_fits):
					spectrum_name = spectra_name_best[i].split('_R')[0] # convolved spectrum name without additions to match original resolution name
					if spectrum_name in out_select_model_spectra['spectra_name']: # if the i best fit is in the model_dir_ori folder 
						spectrum_name_full = out_select_model_spectra['spectra_name_full'][out_select_model_spectra['spectra_name']==spectrum_name][0]
						out_read_model_spectrum = read_model_spectrum(spectra_name_full=spectrum_name_full, model=model)
						wl_model_best[i,:] = out_read_model_spectrum['wl_model']
						flux_model_best[i,:] = scaling_fit_best[i] * out_read_model_spectrum['flux_model'] # scaled fluxes
					else: raise Exception(f"{spectrum_name} is not in {model_dir_ori}")

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

	out = {'spectra_name_best': spectra_name_best, 'chi2_red_fit_best': chi2_red_fit_best, 'chi2_red_wl_fit_best': chi2_red_wl_fit_best, 
		   #'flux_model': flux_model, 'wl_model_conv': wl_model_conv, 'flux_model_conv': flux_model_conv}
		   #'wl_model_conv': wl_array_model_conv_resam_best, 'flux_model_conv': flux_array_model_conv_resam_best,
		   'wl_array_model_conv_resam_best': wl_array_model_conv_resam_best, 'flux_array_model_conv_resam_best': flux_array_model_conv_resam_best,
		   'wl_array_model_conv_resam_fit_best': wl_array_model_conv_resam_fit_best, 'flux_array_model_conv_resam_fit_best': flux_array_model_conv_resam_fit_best,
	       'flux_residuals_best': flux_residuals_best, 'logflux_residuals_best': logflux_residuals_best}
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
		See available models in ``input_parameters.ModelOptions``.  
	- grid : dictionary, optional
		Model grid (``'wavelength'`` and ``'flux'``) generated by ``utils.read_grid`` for interpolations.
		If not provided (default), then the grid is read (``model``, ``model_dir``, ``Teff_range`` and ``logg_range`` must be provided). 
		If provided, the code will skip reading the grid, which will save some time (a few minutes).
	- model_dir : str or list, optional
		Path to the directory (str or list) or directories (as a list) containing the model spectra (e.g., ``model_dir = ['path_1', 'path_2']``). 
		Required if ``grid`` is not provided.
	- save_spectrum : {``True``, ``False``}, optional (default ``False``)
		Save (``True``) or do not save (``False``) the generated spectrum as an ascii table.
		Default name is '``model``_``params``_.dat'.

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
	'''

	# verify there is an input value for each free parameter in the grid
	params_models = Models(model).params # free parameters in the model grid
	for param in params_models:
		if param not in params: raise Exception(f'Provide "{param}" value in "params" because it is a free parameter in "{model}".')

	# verify that "params" are within the model grid coverage
	for param in params:
		if params[param]<params_models[param].min() or params[param]>params_models[param].max():
			raise Exception(f'"{param}" value in "params" is out of the "{model}" grid coverage, which is [{params_models[param].min()}, {params_models[param].max()}]')

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
	          fit_wl_range=None, res=None, lam_res=None, wl_resample=None):
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
	- params_ranges : dictionary, optional
		Minimum and maximum values for any model free parameters to select a model grid subset.
		E.g., ``params_ranges = {'Teff': [1000, 1200], 'logg': [4., 5.]}`` to consider spectra within those Teff and logg ranges.
		If a parameter range is not provided, the full range in ``model_dir`` is considered.
	- convolve: {``True``, ``False``}, optional (default ``False``)
		Convolve (``'yes'``) or do not convolve (``'no'``) the model grid spectra to the indicated ``res`` at ``lam_res``.
	- res : float, optional (required if ``convolve``).
		Spectral resolution at ``lam_res`` to smooth model spectra.
	- lam_res : float, optional (required if ``convolve``).
		Wavelength of reference for ``res``.
	- fit_wl_range : float array, optional
		Minimum and maximum wavelengths (in microns) to resample the model spectra to the fit range. E.g., ``fit_wl_range = np.array([fit_wl_min1, fit_wl_max1])``. 
		Default values are the minimum and the maximum wavelengths in ``wl_resample``.
	- model_wl_range : float array (optional)
		Minimum and maximum wavelength (in microns) to cut model spectra to keep only wavelengths of interest.
		Default values are the minimum and maximum wavelengths in ``wl_resample``, if provided, with a padding to avoid issues with the resampling.
		Otherwise, ``model_wl_range=None``, so model spectra will not be trimmed.
	- res : float, optional
		Spectral resolution at ``lam_res`` of input spectra to smooth model spectra.
	- lam_res : float, optional
		Wavelength of reference at which ``res`` is given.
	- wl_resample : float array or list, optional
		Wavelength data points to resample the grid

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
	out_select_model_spectra = select_model_spectra(model=model, model_dir=model_dir, params_ranges=params_ranges)
	spectra_name_full = out_select_model_spectra['spectra_name_full']
	spectra_name = out_select_model_spectra['spectra_name']
	params = out_select_model_spectra['params']

	# unique values for each free parameter
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
			out_read_model_spectrum = read_model_spectrum(spectra_name_full=spectrum_name_full, model=model, 
			                                              model_wl_range=model_wl_range)
			wl_model = out_read_model_spectrum['wl_model'] # in um
			flux_model = out_read_model_spectrum['flux_model'] # in erg/s/cm2/A

			# convolve (if requested) the model spectrum to the indicated resolution
			if convolve:
				out_convolve_spectrum = convolve_spectrum(wl=wl_model, flux=flux_model, res=res, lam_res=lam_res)
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
			if not np.any(np.array(index)): # for the first parameters' combination
				# define arrays to save the grid
				# add a last dimension with the number of data points in the model spectrum subset
				# it is better to initialize the arrays after the first iteration to consider the model spectrum 
				# data points after resampling instead of using all data points in the original model spectrum
				wl_grid = np.repeat(np.expand_dims(arr, -1), len(wl_model), axis=-1) # to save the wavelength at each grid point
				flux_grid = np.repeat(np.expand_dims(arr, -1), len(wl_model), axis=-1) # to save the flux at each grid point
						 
				# save first spectrum
				wl_grid[index] = wl_model
				flux_grid[index] = flux_model
			
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


	
###	# generate all combinations of parameters in the selected spectra
###	def all_combinations(dictionary):
###		values = list(dictionary.values())
###		combinations = list(itertools.product(*values))
###		return combinations
###	parameters_combinations = all_combinations(out_select_model_spectra['parameters'])
###	print(parameters_combinations)
###	exit()
###

###	# steps of all parameters in the grid
###	parameters = Models(model).parameters
###
###	# constrain parameters to the desired ranges
###	# it will help to avoid iterating below over undesired parameter values
###	if 'Teff' in parameters: 
###		if Teff_range is not None: 
###			mask_Teff = (parameters['Teff']>=Teff_range[0]) & (parameters['Teff']<=Teff_range[1])
###			parameters['Teff'] = parameters['Teff'][mask_Teff]
###	if 'logg' in parameters: 
###		if logg_range is not None: 
###			mask_logg = (parameters['logg']>=logg_range[0]) & (parameters['logg']<=logg_range[1])
###			parameters['logg'] = parameters['logg'][mask_logg]
###	if 'Z' in parameters: 
###		if Z_range is not None: 
###			mask_Z = (parameters['Z']>=Z_range[0]) & (parameters['Z']<=Z_range[1])
###			parameters['Z'] = parameters['Z'][mask_Z]
###	if 'logKzz' in parameters: 
###		if logKzz_range is not None: 
###			mask_logKzz = (parameters['logKzz']>=logKzz_range[0]) & (parameters['logKzz']<=logKzz_range[1])
###			parameters['logKzz'] = parameters['logKzz'][mask_logKzz]
###	if 'CtoO' in parameters: 
###		if CtoO_range is not None: 
###			mask_CtoO = (parameters['CtoO']>=CtoO_range[0]) & (parameters['CtoO']<=CtoO_range[1])
###			parameters['CtoO'] = parameters['CtoO'][mask_CtoO]
###	if 'fsed' in parameters: 
###		if fsed_range is not None: 
###			mask_fsed = (parameters['fsed']>=fsed_range[0]) & (parameters['fsed']<=fsed_range[1])
###			parameters['fsed'] = parameters['fsed'][mask_fsed]
###	if 'Hmix' in parameters: 
###		if Hmix_range is not None: 
###			mask_Hmix = (parameters['Hmix']>=Hmix_range[0]) & (parameters['Hmix']<=Hmix_range[1])
###			parameters['Hmix'] = parameters['Hmix'][mask_Hmix]
###
###	# generate all combinations of parameters
###	def all_combinations(dictionary):
###		values = list(dictionary.values())
###		combinations = list(itertools.product(*values))
###		return combinations
###	parameters_combinations = all_combinations(parameters)
###
###
###	# read the grid from each combination of parameters
###	free_params = np.array([Models(model).free_params]) # free parameters as numpy array
###	for combination in parameters_combinations: # for each combination of parameters
###		for i, param in enumerate(free_params): # for each free parameter
###			# name of spectrum with parameters in the iteration
###			if model=='Sonora_Elf_Owl':
###				spectrum_name = f'spectra_logzz_{logKzz_grid[i_logKzz]}_teff_{Teff_grid[i_Teff]}_grav_{g_grid}_mh_{Z_grid[i_Z]}_co_{CtoO_grid[i_CtoO]}.nc'
###			if model=='Sonora_Diamondback':
###				mask_Teff = free_params=='Teff'
###				Teff_iter = np.array(combination)[mask_Teff[0]]
###				mask_logg = free_params=='logg'
###				logg_iter = np.array(combination)[mask_logg[0]]
###				mask_Z = free_params=='Z'
###				Z_iter = np.array(combination)[mask_Z[0]]
###				mask_fsed = free_params=='fsed'
###				fsed_iter = np.array(combination)[mask_fsed[0]]
###				print(combination)
###				print(Teff_iter)
###				print(logg_iter)
###				print(Z_iter)
###				print(fsed_iter)
###				spectrum_name = f't{Teff_iter}g316f{fsed_iter}_m{Z_iter}_co1.0.spec'
###				print(spectrum_name)
###				exit()


#	# all parameter's steps in the grid
#	out_grid_ranges = grid_ranges(model)
#	Teff_grid = out_grid_ranges['Teff']
#	logg_grid = out_grid_ranges['logg']
#	logKzz_grid = out_grid_ranges['logKzz']
#	Z_grid = out_grid_ranges['Z']
#	CtoO_grid = out_grid_ranges['CtoO']
#
#	# constrain parameters to the desired ranges
#	# it will help to avoid iterating below over undesired parameter values
#	if Teff_range is not None: 
#		mask_Teff = (Teff_grid>=Teff_range[0]) & (Teff_grid<=Teff_range[1])
#		Teff_grid = Teff_grid[mask_Teff]
#	if logg_range is not None: 
#		mask_logg = (logg_grid>=logg_range[0]) & (logg_grid<=logg_range[1])
#		logg_grid = logg_grid[mask_logg]
#	if logKzz_range is not None: 
#		mask_logKzz = (logKzz_grid>=logKzz_range[0]) & (logKzz_grid<=logKzz_range[1])
#		logKzz_grid = logKzz_grid[mask_logKzz]
#	if Z_range is not None: 
#		mask_Z = (Z_grid>=Z_range[0]) & (Z_grid<=Z_range[1])
#		Z_grid = Z_grid[mask_Z]
#	if CtoO_range is not None: 
#		mask_CtoO = (CtoO_grid>=CtoO_range[0]) & (CtoO_grid<=CtoO_range[1])
#		CtoO_grid = CtoO_grid[mask_CtoO]
#
##	# define arrays to save the model grid
##	if wl_resample is not None: 
##		flux_grid = np.zeros((len(Teff_grid), len(logg_grid), len(logKzz_grid), len(Z_grid), len(CtoO_grid), len(wl_resample))) # to save the flux at each grid point
##		wl_grid = np.zeros((len(Teff_grid), len(logg_grid), len(logKzz_grid), len(Z_grid), len(CtoO_grid), len(wl_resample))) # to save the wavelength at each grid point
##		
##	else:
##		N_modelpoints = model_points(model)
##		flux_grid = np.zeros((len(Teff_grid), len(logg_grid), len(logKzz_grid), len(Z_grid), len(CtoO_grid), N_modelpoints)) # to save the flux at each grid point
##		wl_grid = np.zeros((len(Teff_grid), len(logg_grid), len(logKzz_grid), len(Z_grid), len(CtoO_grid), N_modelpoints)) # to save the wavelength at each grid point
#	
#	# create a tqdm progress bar
#	if convolve:
#		if wl_resample is not None:
#			desc = 'Reading, convolving, and resampling model grid'
#		else:
#			desc = 'Reading and convolving model grid'
#	else:
#		if wl_resample is not None:
#			desc = 'Reading and resampling model grid'
#		else:
#			desc = 'Reading model grid'
#	grid_bar = tqdm(total=len(spectra_name), desc=desc)
#
#	# read the grid with the selected model spectra
#	k = 0
#	for i_Teff in range(len(Teff_grid)): # iterate Teff
#		for i_logg in range(len(logg_grid)): # iterate logg
#			for i_logKzz in range(len(logKzz_grid)): # iterate logKzz
#				for i_Z in range(len(Z_grid)): # iterate Z
#					for i_CtoO in range(len(CtoO_grid)): # iterate C/O ration
#						# update the progress bar
#						grid_bar.update(1)
#
#						# read the spectrum for each parameter combination
#						# file name uses g instead of logg
#						if (logg_grid[i_logg]==3.25): g_grid = 17.0
#						if (logg_grid[i_logg]==3.50): g_grid = 31.0
#						if (logg_grid[i_logg]==3.75): g_grid = 56.0
#						if (logg_grid[i_logg]==4.00): g_grid = 100.0
#						if (logg_grid[i_logg]==4.25): g_grid = 178.0
#						if (logg_grid[i_logg]==4.50): g_grid = 316.0
#						if (logg_grid[i_logg]==4.75): g_grid = 562.0
#						if (logg_grid[i_logg]==5.00): g_grid = 1000.0
#						if (logg_grid[i_logg]==5.25): g_grid = 1780.0
#						if (logg_grid[i_logg]==5.50): g_grid = 3160.0
#
#						# name of spectrum with parameters in the iteration
#						spectrum_name = f'spectra_logzz_{logKzz_grid[i_logKzz]}_teff_{Teff_grid[i_Teff]}_grav_{g_grid}_mh_{Z_grid[i_Z]}_co_{CtoO_grid[i_CtoO]}.nc'
#						# look into the selected spectra to find the full path for the spectrum above
#						spectrum_name_full = [x for x in spectra_name_full if x.endswith(spectrum_name)]
#
#						if spectrum_name_full: # if there is a spectrum with the parameters in the iteration in the input directories
#							# read the spectrum with the parameter combination in the iteration and cut it to model_wl_range (default value: fit_wl_range plus padding)
#							out_read_model_spectrum = read_model_spectrum(spectra_name_full=spectrum_name_full[0], model=model, model_wl_range=model_wl_range)
#							wl_model = out_read_model_spectrum['wl_model'] # in um
#							flux_model = out_read_model_spectrum['flux_model'] # in erg/s/cm2/A
#							
#							# convolve the model spectrum to the indicated resolution
#							if convolve:
#								out_convolve_spectrum = convolve_spectrum(wl=wl_model, flux=flux_model, res=res, lam_res=lam_res)
#								wl_model = out_convolve_spectrum['wl_conv']
#								flux_model = out_convolve_spectrum['flux_conv']
#
#							# resample convolved model spectrum to the wavelength data points in the observed spectra
#							if wl_resample is not None:
#								# mask to select data points within the fit range or model coverage range, whichever is narrower
#								mask_fit = (wl_resample >= max(fit_wl_range[0], wl_model.min())) & \
#								           (wl_resample <= min(fit_wl_range[1], wl_model.max()))
#								flux_model = spectres(wl_resample[mask_fit], wl_model, flux_model)
#								wl_model = wl_resample[mask_fit]
#
#							# save flux at each combination
#							if i_Teff==0 and i_logg==0 and i_logKzz==0 and i_Z==0 and i_CtoO==0:
#								# define arrays to save the model grid
#								# it is better to initialize the arrays after the first iteration to consider the size of
#								# the array after resampling instead of using the data points in the original model spectra
#								flux_grid = np.zeros((len(Teff_grid), len(logg_grid), len(logKzz_grid), len(Z_grid), len(CtoO_grid), len(wl_model))) # to save the flux at each grid point
#								wl_grid = np.zeros((len(Teff_grid), len(logg_grid), len(logKzz_grid), len(Z_grid), len(CtoO_grid), len(wl_model))) # to save the wavelength at each grid point
#
#								# save first read model spectrum
#								flux_grid[i_Teff, i_logg, i_logKzz, i_Z, i_CtoO, :len(wl_model)] = flux_model
#								wl_grid[i_Teff, i_logg, i_logKzz, i_Z, i_CtoO, :len(wl_model)] = wl_model
#							else:
#								flux_grid[i_Teff, i_logg, i_logKzz, i_Z, i_CtoO, :len(wl_model)] = flux_model
#								wl_grid[i_Teff, i_logg, i_logKzz, i_Z, i_CtoO, :len(wl_model)] = wl_model
#						
#						else:
#						    print(f'No spectrum {spectrum_name} in "model_dir"') # no spectrum with that parameters combination in the input directories
#						k += 1
#	# close the progress bar
#	grid_bar.close()
#
#	fin_time_grid = time.time()
#	print_time(fin_time_grid-ini_time_grid)
#
#	out = {'wavelength': wl_grid, 'flux': flux_grid, 'Teff': Teff_grid, 'logg': logg_grid, 'logKzz': logKzz_grid, 'Z': Z_grid, 'CtoO': CtoO_grid}

	return out

###########################
#def grid_ranges(model):
#	'''
#	Description:
#	------------
#		Read coverage of parameters in a model grid.
#
#	Parameters:
#	-----------
#	- model : str
#		Atmospheric models. See available models in ``input_parameters.ModelOptions``.  
#
#	Returns:
#	--------
#	Dictionary with model parameter coverage:
#		- ``'Teff'`` : effective temperature.
#		- ``'logg'`` : surface gravity (logg).
#		- ``'logKzz'`` : (if provided by ``model``) diffusion parameter (logKzz).
#		- ``'Z'`` : (if provided by ``model``) metallicity at each grid point.
#		- ``'CtoO'`` : (if provided by ``model``) C/O ratio at each grid point.
#
#	Example:
#	--------
#	>>> import seda
#	>>>
#	>>> # models
#	>>> model = 'Sonora_Elf_Owl'
#	>>>
#	>>> # read model parameters
#	>>> out_grid_ranges = seda.grid_ranges(model)
#
#	Author: Genaro Suárez
#	'''
#
#	# initialize output dictionary
#	out = {}
#
#	if (model=='Sonora_Elf_Owl'):
#		# Teff
#		Teff_range1 = np.arange(275., 600.+25, 25)
#		Teff_range2 = np.arange(650., 1000.+50, 50)
#		Teff_range3 = np.arange(1100., 2400.+100, 100)
#		out['Teff'] = np.concatenate([Teff_range1, Teff_range2, Teff_range3]) # K
#		# logg
#		out['logg'] = np.arange(3.25, 5.50+0.25, 0.25) # g in cm/s2
#		#logKzz
#		out['logKzz'] = np.array([2.0, 4.0, 7.0, 8.0, 9.0]) # Kzz in cm2/s
#		# Z or [M/H]
#		out['Z'] = np.array([-1.0, -0.5, 0.0, 0.5, 0.7, 1.0]) # Z=0 means solar metallicity
#		# C/O
#		out['CtoO'] = np.array([0.5, 1.0, 1.5, 2.5]) # relative to solar C/O (equal to 1)
#
#	if (model=='Sonora_Diamondback'):
#		# Teff
#		out['Teff'] = np.arange(900., 2400.+100., 100.) # K
#		# logg
#		out['logg'] = np.array([3.5, 4., 4.5, 5., 5.5]) # g in cm/s2
#		# Z
#		out['Z'] = np.array([-0.5, 0., 0.5])
#		# fsed
#		out['fsed'] = np.array([1., 2., 3., 4., 8., 99.]) # fsed=99 means no clouds
#
#	if (model=='LB23'):
#		# Teff
#		Teff_range1 = np.arange(250., 575+25., 25.)
#		Teff_range2 = np.arange(600., 800+50., 50.)
#		out['Teff'] = np.concatenate([Teff_range1, Teff_range2]) # K
#		# logg
#		out['logg'] = np.arange(3.5, 5.0+0.25, 0.25) # g in cm/s2
#		# Z
#		out['Z'] = np.array([-0.5, 0., 0.5]) # [M/H]
#		# logKzz
#		out['logKzz'] = np.array([6]) # Kzz in cm2/s
#		# Hmix
#		out['Hmix'] = np.array([0.01, 0.1, 1.])
#
#	if (model=='Sonora_Cholla'):
#		out['Teff'] = np.arange(500., 1300.+50, 50) # K
#		out['logg'] = np.arange(3.5, 5.5+0.25, 0.25) # g in s/cm2
#		out['logKzz'] = np.array([2., 4., 7.]) # Kzz in cm2/s
#
#	if (model=='Sonora_Bobcat'):
#		# Teff
#		Teff_range1 = np.arange(200., 600.+25, 25)
#		Teff_range2 = np.arange(650., 1000.+50, 50)
#		Teff_range3 = np.arange(1100., 2400.+100, 100)
#		out['Teff'] = np.concatenate([Teff_range1, Teff_range2, Teff_range3]) # K
#		# logg
#		out['logg'] = np.arange(3., 5.5+0.25, 0.25) # g in s/cm2
#		# Z
#		out['Z'] = np.array([-0.5, 0., 0.5]) # [M/H]
#		# C/O
#		out['CtoO'] = np.array([0.5, 1.0, 1.5]) # relative to solar C/O (equal to 1)
#
#	if (model=='ATMO2020'):
#		# Teff
#		Teff_range1 = np.arange(200., 600.+50, 50)
#		Teff_range2 = np.arange(700., 3000.+100, 100)
#		out['Teff'] = np.concatenate([Teff_range1, Teff_range2]) # K
#		# logg
#		out['logg'] = np.arange(2.5, 5.5+0.5, 0.5) # g in s/cm2
#		# logKzz
#		out['logKzz'] = np.array([0, 4, 6]) # Kzz in cm2/s
#
#	if (model=='BT-Settl'):
#		# Teff
#		Teff_range1 = np.arange(260., 420.+20, 20)
#		Teff_range2 = np.arange(450., 1200.+50, 50)
#		Teff_range3 = np.arange(1300., 1500.+100, 100)
#		Teff_range4 = np.arange(1550., 2400.+50, 50)
#		Teff_range5 = np.arange(2500., 7000.+100, 100)
#		out['Teff'] = np.concatenate([Teff_range1, Teff_range2, Teff_range3, Teff_range4, Teff_range5]) # K
#		# logg
#		out['logg'] = np.arange(2., 5.5+0.5, 0.5) # g in s/cm2
#
#	if (model=='SM08'):
#		# Teff
#		out['Teff'] = np.arange(800., 2400.+100, 100) # K
#		# logg
#		out['logg'] = np.arange(3., 5.5+0.5, 0.5) # g in s/cm2
#		# fsed
#		out['fsed'] = np.array([1., 2., 3., 4.])
#
#	#out = {'Teff': Teff, 'logg': logg, 'logKzz': logKzz, 'Z': Z, 'CtoO': CtoO}
#
#	return out

###########################
#def model_filename_pattern(model):
#	
#	# common pattern depending the models
#	if (model == 'Sonora_Diamondback'):	pattern = 't*.spec*'
#	if (model == 'Sonora_Elf_Owl'):	pattern = 'spectra_logzz_*.nc*'
#	if (model == 'LB23'): pattern = 'T*21*'
#	if (model == 'Sonora_Cholla'): pattern = '*.spec*'
#	if (model == 'Sonora_Bobcat'): pattern = 'sp_t*'
#	if (model == 'ATMO2020'): pattern = 'spec_T*.txt*'
#	if (model == 'BT-Settl'): pattern = 'lte*.BT-Settl.spec.7*'
#	if (model == 'SM08'): pattern = 'sp_t*'
#
#	return pattern

##########################
def params_ranges_sampling(chi2_pickle_file, N_best_fits=3):
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
	>>> out_params_ranges_sampling = seda.params_ranges_sampling(chi2_pickle_file=chi2_pickle_file)

	Author: Genaro Suárez
	'''

	# open results from the chi square analysis
	with open(chi2_pickle_file, 'rb') as file:
		out_chi2 = pickle.load(file)

	model = out_chi2['my_chi2'].model

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
	fit_wl_range = out_bayes['my_bayes'].fit_wl_range
	params_priors = out_bayes['my_bayes'].params_priors
	out_dynesty = out_bayes['out_dynesty']

	# compute median values for all sampled parameters
	params_med = {}
	for i,param in enumerate(params_priors): # for each parameter in the sampling
		params_med[param] = np.median(out_dynesty.samples[:,i]) # add to the dictionary the median of each parameter

	# round median values for model grid parameters
	params_models = Models(model).params
	for i,param in enumerate(params_med): # for each sampled parameter
		if param in params_models: # for free parameters in the model grid
			params_med[param] = round(params_med[param], max_decimals(params_models[param])+1) # round to the precision (plus one decimal place) of the parameter in models

	# generate spectrum with the median parameter values
	if grid is None:
		# grid values around the desired parameter values
		params_ranges = {}
		for param in params_models: # for each free parameter in the grid
			params_ranges[param] = find_two_nearest(params_models[param], params_med[param])

		# read grid
		grid = read_grid(model=model, model_dir=model_dir, params_ranges=params_ranges)

	# generate synthetic spectrum
	params = {}
	for param in params_models: # for each free parameter in the grid
		params[param] = params_med[param]
	syn_spectrum = generate_model_spectrum(params=params, model=model, grid=grid)
	wl = syn_spectrum['wavelength']
	flux = syn_spectrum['flux']

	# convolve synthetic spectrum
	# (add padding on both edges to avoid issues we using spectres)
	# initialize lists for convolved model spectra for all input spectra
	wl_conv = []
	flux_conv = []
	for i in range(N_spectra): # for each input observed spectrum
		out_convolve_spectrum = convolve_spectrum(wl=wl, flux=flux, lam_res=lam_res[i], res=res[i], 
		                                          disp_wl_range=np.array([wl_spectra[i].min(), wl_spectra[i].max()]), 
		                                          convolve_wl_range=np.array([0.99*wl_spectra[i].min(), 1.01*wl_spectra[i].max()]))
		wl_conv.append(out_convolve_spectrum['wl_conv'])
		flux_conv.append(out_convolve_spectrum['flux_conv'])

	# scale synthetic spectrum
	if distance is not None:
		flux_scaled = scale_synthetic_spectrum(wl=wl, flux=flux, distance=distance, radius=params_med['R'])
		wl_scaled = wl

		wl_conv_scaled = []
		flux_conv_scaled = []
		for i in range(N_spectra): # for each input observed spectrum
			flux_conv_scaled.append(scale_synthetic_spectrum(wl=wl_conv[i], flux=flux_conv[i], distance=distance, radius=params_med['R']))
			wl_conv_scaled.append(wl_conv[i])

	# resample the convolved synthetic spectrum to the fit ranges of the input spectra
	# initialize lists for resampled, convolved model spectra for all input spectra
	flux_conv_resam = []
	wl_conv_resam = []
	flux_conv_scaled_resam = []
	wl_conv_scaled_resam = []
	for i in range(N_spectra): # for each input observed spectrum
		mask_fit = (wl_spectra[i] >= max(fit_wl_range[i][0], wl.min())) & \
		           (wl_spectra[i] <= min(fit_wl_range[i][1], wl.max()))
		flux_conv_resam.append(spectres(wl_spectra[i][mask_fit], wl_conv[i], flux_conv[i]))
		wl_conv_resam.append(wl_spectra[i][mask_fit])
		if distance is not None:
			flux_conv_scaled_resam.append(spectres(wl_spectra[i][mask_fit], wl_conv_scaled[i], flux_conv_scaled[i]))
			wl_conv_scaled_resam.append(wl_spectra[i][mask_fit])

	# save synthetic spectrum
	if save_spectrum:
		for i in range(N_spectra): # for each input observed spectrum
			out = open(f'{model}_R{res[i]}at{lam_res[i]}um_best_fit_bayesian_sampling.dat', 'w')
			out.write('# wavelength(um) flux(erg/s/cm2/A)  \n')
			for k in range(len(wl_conv)):
				out.write('%11.7f %17.6E \n' %(wl_conv[i][k], flux_conv[i][k]))
			out.close()

	# output dictionary
	out = {'wl_spectra': wl_spectra, 'flux_spectra': flux_spectra, 'eflux_spectra': eflux_spectra, 'wl_mod': wl, 'flux_mod': flux, 'wl_mod_conv': wl_conv, 'flux_mod_conv': flux_conv, 
	       'wl_mod_conv_resam': wl_conv_resam, 'flux_mod_conv_resam': flux_conv_resam, 'params_med' : params_med}
	if distance is not None:
		out['wl_mod_scaled'] = wl_scaled
		out['flux_mod_scaled'] = flux_scaled
		out['wl_mod_conv_scaled'] = wl_conv_scaled
		out['flux_mod_conv_scaled'] = flux_conv_scaled
		out['wl_mod_conv_scaled_resam'] = wl_conv_scaled_resam
		out['flux_mod_conv_scaled_resam'] = flux_conv_scaled_resam

	return out

##########################
def select_model_spectra(model, model_dir, params_ranges=None):
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
	- params_ranges : dictionary, optional
		Minimum and maximum values for any model free parameters to select a model grid subset.
		E.g., ``params_ranges = {'Teff': [1000, 1200], 'logg': [4., 5.]}`` to consider spectra within those Teff and logg ranges.
		If a parameter range is not provided, the full range in ``model_dir`` is considered.

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
	'''

	# make sure model_dir is a list
	model_dir = var_to_list(model_dir)
	
	if isinstance(model_dir, str): model_dir = [model_dir]
	if isinstance(model_dir, np.ndarray): model_dir = model_dir.tolist()

	# if params_ranges is provided, verified that there is a minimum and a maximum values for each provided parameter
	if params_ranges is not None:
		for param in params_ranges:
			if len(params_ranges[param])!=2: raise Exception(f'{param} in "params_ranges" must have two values (minimum and maximum), but {len(params_ranges[param])} values were given')

	# if params_ranges is not provided, define params_ranges as an empty dictionary
	if params_ranges is None: params_ranges = {}

	# to store files in model_dir
	files = [] # with full path
	files_short = [] # only spectra names
	for i in range(len(model_dir)):
		files_model_dir = fnmatch.filter(os.listdir(model_dir[i]), Models(model).filename_pattern)
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

	if len(spectra_name_full)==0: raise Exception('No model spectra within the indicated parameter ranges') # show up an error when there are no models in the indicated ranges
	else: 
		if not params_ranges: 
			print(f'\n      {len(spectra_name)} model spectra')
		else:
			print(f'\n      {len(spectra_name)} model spectra selected with:')
			for param in params_ranges:
				print(f'         {param} range = {params_ranges[param]}')

	# separate parameters from selected spectra
	out_separate_params = separate_params(model=model, spectra_name=spectra_name)

	out = {'spectra_name_full': np.array(spectra_name_full), 'spectra_name': np.array(spectra_name), 'params': out_separate_params['params']}

	return out

##########################
def separate_params(model, spectra_name, save_results=False):
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

	# if there is one input spectrum with its name given as a string, convert it into a list
	if isinstance(spectra_name, str): spectra_name = [spectra_name]

	out = {'spectra_name': spectra_name} # start dictionary with some parameters
	out['params'] = {}

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
		with open(f'{model}_free_parameters.pickle', 'wb') as file:
			pickle.dump(out, file)

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
#def set_model_wl_range(model_wl_range, fit_wl_range):
def set_model_wl_range(model_wl_range, wl_spectra_min, wl_spectra_max):

	# define model_wl_range (if not provided) in terms of input spectra coverage
	if model_wl_range is None:
		#model_wl_range = np.array((0.9*fit_wl_range.min(), 1.1*fit_wl_range.max())) # add padding to have enough spectral coverage in models
		model_wl_range = np.array([0.9*wl_spectra_min, 1.1*wl_spectra_max]) # add padding to have enough spectral coverage in models

#	# it may need an update to work with multiple spectra
#	if (model_wl_range.min()>=fit_wl_range.min()):
#		model_wl_range[0] = 0.9*fit_wl_range.min() # add padding to shorter wavelengths
#	if (model_wl_range.max()<=fit_wl_range.max()):
#		model_wl_range[1] = 1.1*fit_wl_range.max() # add padding to longer wavelengths

	return model_wl_range

##########################
def app_to_abs_flux(flux, distance, eflux=0, edistance=0):
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
# read a pre-stored convolved model spectrum
# it is a netCDF file with xarray produced by convolve_spectrum
def read_model_spectrum_conv(spectrum_name_full):

	# read convolved spectrum
	spectrum = xarray.open_dataset(spectrum_name_full)
	wl_model = spectrum['wl'].data # um
	flux_model = spectrum['flux'].data # erg/s/cm2/A

	# obtain fluxes in Jy
	flux_model_Jy = (flux_model*u.erg/u.s/u.cm**2/u.angstrom).to(u.Jy, equivalencies=u.spectral_density(wl_model*u.micron)).value

	out = {'wl_model': wl_model, 'flux_model': flux_model, 'flux_model_Jy': flux_model_Jy}

	return out

##########################
# make sure a variable is a list
def var_to_list(x):
	if isinstance(x, str): x = [x]
	if isinstance(x, np.ndarray): x = x.tolist()
	if isinstance(x, float): x = [x]

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

##########################
# round logg to steps of 0.25
def round_logg_point25(logg):
	logg = round(logg*4.) / 4. 
	return logg

##########################
# dictionary with basic properties from available atmospheric models
def available_models():

#	out = { 
#	'Sonora_Diamondback': {'ref': 'Morley et al (2024)',     'name': 'Sonora Diamondback',     'N_modelpoints': 385466 },
#	'Sonora_Elf_Owl':     {'ref': 'Mukherjee et al. (2024)', 'name': 'Sonora Elf Owl',         'N_modelpoints': 193132 },
#	'LB23':               {'ref': 'Lacy & Burrows (2023)',   'name': 'Lacy & Burrows (2023)',  'N_modelpoints': 30000  },
#	'Sonora_Cholla':      {'ref': 'Karalidi et al. (2021)',  'name': 'Sonora Cholla',          'N_modelpoints': 110979 },
#	'Sonora_Bobcat':      {'ref': 'Marley et al. (2021)',    'name': 'Sonora_Bobcat',          'N_modelpoints': 362000 },
#	'ATMO2020':           {'ref': 'Phillips et al. (2020)',  'name': 'ATMO 2020',              'N_modelpoints': 5000   },
#	'BT-Settl':           {'ref': 'Allard et al. (2012)',    'name': 'BT-Settl',               'N_modelpoints': 1291340},
#	'SM08':               {'ref': 'Saumon & Marley (2008)',  'name': 'Saumon & Marley (2008)', 'N_modelpoints': 184663 }
#	}

	out = {}
	out['Sonora_Diamondback'] = 'Morley et al (2024)'
	out['Sonora_Elf_Owl'] = 'Mukherjee et al. (2024)'
	out['LB23'] = 'Lacy & Burrows (2023)'
	out['Sonora_Cholla'] = 'Karalidi et al. (2021)'
	out['Sonora_Bobcat'] = 'Marley et al. (2021)'
	out['ATMO2020'] = 'Phillips et al. (2020)'
	out['BT-Settl'] = 'Allard et al. (2012)'
	out['SM08'] = 'Saumon & Marley (2008)'

	return out

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
# maximum number of decimals in an array elements
def max_decimals(arr):
	max_places = 0
	for num in arr:
		if isinstance(num, float): # if the element is a float
			num_str = str(num) 
			decimal_part = num_str.split('.')[1] # select decimals as a string
			max_places = max(max_places, len(decimal_part)) # compare decimals
	return max_places
