# Useful function for SEDA

import numpy as np
from astropy.io import ascii
from sys import exit

##########################
# convolve a spectrum to a desired resolution at a given wavelength
def convolve_spectrum(wl, flux, lam_R, R, eflux=None, disp_wl_range=None, convolve_wl_range=None):
	'''
	wl : float array
		wavelength (any length units) for the spectrum
	flux : float array
		fluxes (any flux units) for the spectrum
	flux : float array, optional
		error fluxes (any flux units) for the spectrum
	lam_R : scalar
		wavelength reference to estimate the spectral resolution of the input spectrum
	R : scalar
		resolution at lam_R to smooth the spectrum
	disp_wl_range : float array, optional
		wavelength range (minimum and maximum) to calculate the median wavelength dispersion of the input spectrum
		default values are the minimum and maximum wavelengths of wl
	convolve_wl_range : float array, optional
		wavelength range where the input spectrum will be convolved
		default values are the minimum and maximum wavelengths of wl

	Returns
	------
	out : dictionary
		dictionary with the convolved spectrum
		out['wl_conv'] : wavelengths for the convolved spectrum (equal to input wavelengths within convolve_wl_range)
		out['flux_conv'] : convolved fluxes
		out['eflux_conv'] : convolved flux errors, if input flux errors are provided

	'''

	from astropy.convolution import Gaussian1DKernel, convolve # kernel to convolve spectra

	if (disp_wl_range is None): disp_wl_range = np.array((wl.min(), wl.max())) # define disp_wl_range if not provided
	if (convolve_wl_range is None): convolve_wl_range = np.array((wl.min(), wl.max())) # define convolve_wl_range if not provided

	wl_bin = abs(wl[1:] - wl[:-1]) # wavelength dispersion of the spectrum
	wl_bin = np.append(wl_bin, wl_bin[-1]) # add an element equal to the last row to have same data points

	# define a Gaussian for convolution
	mask_fit = (wl>=disp_wl_range[0]) & (wl<=disp_wl_range[1]) # mask to obtain the median wavelength dispersion
	stddev = (lam_R/R)*(1./np.median(wl_bin[mask_fit]))/2.36 # stddev is given in pixels
	gauss = Gaussian1DKernel(stddev=stddev)

	mask_conv = (wl>=convolve_wl_range[0]) & (wl<=convolve_wl_range[1]) # range to convolve the spectrum

	flux_conv = convolve(flux[mask_conv], gauss) # convolve only the selected wavelength range 
	wl_conv = wl[mask_conv] # corresponding wavelength data points for convolved fluxes
	if (eflux is not None): eflux_conv = convolve(eflux[mask_conv], gauss)

	out = {'wl_conv': wl_conv, 'flux_conv': flux_conv}
	if (eflux is not None): out['eflux_conv'] = eflux_conv

	return out

##########################
# scale model spectrum when distance and radius are known
def scale_synthetic_spectrum(wl, flux, distance, radius):
	'''
	radius: float
		radius in Rjup
	distance: float
		distance in pc
	'''
	scaling = ((radius*6.991e4) / (distance*3.086e13))**2 # scaling = (R/d)^2
	flux_scaled = scaling*flux
	return flux_scaled

##########################
# convert time elapsed running a function into an adequate unit
def time_elapsed(time):
	'''
	time: time in seconds
	'''
	if (time<60): ftime, unit = np.round(time), 's' # s
	if ((time>=60) & (time<3600)): ftime, unit = np.round(time/60.,1), 'min' # s
	if (time>=3600): ftime, unit = np.round(time/3600.,1), 'hr' # s

	return ftime, unit

##########################
# read spectra
def read_model_spectrum(spectra_name_full, model, model_wl_range=None):

	# read model spectra files
	if (model == 'Sonora_Diamondback'):
		spec_model = ascii.read(spectra_name_full, data_start=3, format='no_header')
		wl_model = spec_model['col1'] # um (in vacuum?)
		wl_model = (wl_model*1e4) / (1.0 + 2.735182e-4 + 131.4182/(wl_model*1e4)**2 + 2.76249e8/(wl_model*1e4)**4) # wavelength (in A) in the air
		wl_model = wl_model/1e4 # in um
		flux_model = spec_model['col2'] # W/m2/m
		flux_model = flux_model * 0.1 / 1.e6 # erg/cm2/s/A
	if (model == 'Sonora_Elf_Owl'):
		import xarray
		spec_model = xarray.open_dataset(spectra_name_full) # Sonora Elf Owl model spectra have NetCDF Data Format data
		wl_model = spec_model['wavelength'].data # um
		flux_model = spec_model['flux'].data # erg/s/cm^2/cm
		flux_model = flux_model / 1.e8 # erg/s/cm2/A
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
		wl_model = spec_model['col1'] # um
		wl_model = (wl_model*1e4) / (1.0 + 2.735182e-4 + 131.4182/(wl_model*1e4)**2 + 2.76249e8/(wl_model*1e4)**4) # wavelength (in A) in the air
		wl_model = wl_model/1e4 # in um
		flux_model = spec_model['col2'] # W/m2/m
		flux_model = flux_model * 0.1 / 1.e6 # erg/s/cm/A
	if (model == 'Sonora_Bobcat'):
		spec_model = ascii.read(spectra_name_full, data_start=2, format='no_header')
		wl_model = spec_model['col1'] # um 
		wl_model = (wl_model*1e4) / (1.0 + 2.735182e-4 + 131.4182/(wl_model*1e4)**2 + 2.76249e8/(wl_model*1e4)**4) # wavelength (in A) in the air
		wl_model = wl_model/1e4 # in um
		flux_model = spec_model['col2'] # erg/cm^2/s/Hz (to an unknown distance)
		# convert flux to wavelength dependence
		flux_model = flux_model * (3e2/((wl_model*1e-4)**2)) # erg/s/cm2/A (to an unknown distance)
	if (model == 'ATMO2020'):
		spec_model = ascii.read(spectra_name_full, format='no_header')
		wl_model = spec_model['col1']*1e4 # A
		wl_model = wl_model / (1.0 + 2.735182e-4 + 131.4182/wl_model**2 + 2.76249e8/wl_model**4) # wavelength (in A) in the air
		wl_model = wl_model/1e4 # in um
		flux_model = spec_model['col2'] # W/m2/micron
		flux_model = flux_model * 0.1 # erg/s/cm2/A
	if (model == 'BT-Settl'):
		spec_model = ascii.read(spectra_name_full, format='no_header')
		wl_model = spec_model['col1'] # in A (in vacuum)
		wl_model = wl_model / (1.0 + 2.735182e-4 + 131.4182/wl_model**2 + 2.76249e8/wl_model**4) # wavelength (in A) in the air
		wl_model = wl_model/1e4
		flux_model = spec_model['col2'] # erg/cm^2/s/Hz (to an unknown distance). 10**(F_lam + DF) to convert to erg/s/cm**2/A
		# convert flux to erg/s/cm**2/A
		DF= -8.0
		flux_model = 10**(flux_model + DF) # erg/s/cm2/A
	if (model == 'SM08'):
		spec_model = ascii.read(spectra_name_full, data_start=2, format='no_header')
		wl_model = spec_model['col1'] # um (in air the alkali lines and in vacuum the rest of the spectra)
		# convert to air 
		#wl_model = (wl_model*1e4) / (1.0 + 2.735182e-4 + 131.4182/(wl_model*1e4)**2 + 2.76249e8/(wl_model*1e4)**4) # wavelength (in A) in the air
		#wl_model = wl_model/1e4
		flux_model = spec_model['col2'] # erg/cm^2/s/Hz (to an unknown distance)
		# convert flux to wavelength dependence
		flux_model = flux_model * (3e2/((wl_model*1e-4)**2)) # erg/s/cm2/A (to an unknown distance)

	# sort the array. For BT-Settl is recommended by Allard in her webpage and some models are sorted from higher to smaller wavelengths.
	sort_index = np.argsort(wl_model)
	wl_model = wl_model[sort_index]
	flux_model = flux_model[sort_index]

	# cut the model spectra to the indicated range
	if model_wl_range is not None:
		ind = np.where((wl_model>=(model_wl_range[0])) & (wl_model<=model_wl_range[1]))
		wl_model = wl_model[ind]
		flux_model = flux_model[ind]

	out = {'wl_model': wl_model, 'flux_model': flux_model}

	return out

##########################
# read best-fitting model
def best_chi2_fits(pickle_file, N_best_fits=1):
	'''
	chi2_minimization_pickle_file: pickle file from chi2.py
	N_best_fits: int with the number (default 1) of best model fits to be read
	'''

	import pickle

	# open results from the chi square analysis
	with open(pickle_file, 'rb') as file:
		out_chi2 = pickle.load(file)

	model = out_chi2['model']
	R = out_chi2['R']#[0] # resolution for the first input spectrum
	lam_R = out_chi2['lam_R']#[0] # wavelength reference for the first input spectrum
	N_rows_model = out_chi2['N_rows_model']
	chi2_wl_range = out_chi2['chi2_wl_range'][0] # for the first input spectrum
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

	# read best fits
	wl_model = np.zeros((N_rows_model, N_best_fits))
	flux_model = np.zeros((N_rows_model, N_best_fits))
	for i in range(N_best_fits):
		out_read_model_spectrum = read_model_spectrum(spectra_name_full_best[i], model)
		wl_model[:,i] = out_read_model_spectrum['wl_model']
		flux_model[:,i] = scaling_fit_best[i] * out_read_model_spectrum['flux_model'] # scaled fluxes

	# convolve spectrum
	wl_model_conv = np.zeros((N_rows_model, N_best_fits))
	flux_model_conv = np.zeros((N_rows_model, N_best_fits))
	for i in range(N_best_fits):
		out_convolve_spectrum = convolve_spectrum(wl=wl_model[:,i], flux=flux_model[:,i], lam_R=lam_R, R=R, disp_wl_range=chi2_wl_range)
		wl_model_conv[:,i] = out_convolve_spectrum['wl_conv']
		flux_model_conv[:,i] = out_convolve_spectrum['flux_conv']

	out = {'spectra_name_best': spectra_name_best, 'chi2_red_fit_best': chi2_red_fit_best, 'wl_model': wl_model, 
		   'flux_model': flux_model, 'wl_model_conv': wl_model_conv, 'flux_model_conv': flux_model_conv}

	return out
