# Useful function for SEDA

import numpy as np
from astropy.io import ascii
from scipy.interpolate import RegularGridInterpolator
from .bayes_fit import *
import time
import os
import xarray
from sys import exit

##########################
# convolve a spectrum to a desired resolution at a given wavelength
def convolve_spectrum(wl, flux, lam_res, res, eflux=None, disp_wl_range=None, convolve_wl_range=None):
	'''
	wl : float array
		wavelength (any length units) for the spectrum
	flux : float array
		fluxes (any flux units) for the spectrum
	flux : float array, optional
		error fluxes (any flux units) for the spectrum
	lam_res : scalar
		wavelength reference to estimate the spectral resolution of the input spectrum
	res : scalar
		resolution at lam_res to smooth the spectrum
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
	stddev = (lam_res/res)*(1./np.median(wl_bin[mask_fit]))/2.36 # stddev is given in pixels
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
	res = out_chi2['res']#[0] # resolution for the first input spectrum
	lam_res = out_chi2['lam_res']#[0] # wavelength reference for the first input spectrum
	N_rows_model = out_chi2['N_rows_model']
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
		out_convolve_spectrum = convolve_spectrum(wl=wl_model[:,i], flux=flux_model[:,i], lam_res=lam_res, res=res, disp_wl_range=fit_wl_range)
		wl_model_conv[:,i] = out_convolve_spectrum['wl_conv']
		flux_model_conv[:,i] = out_convolve_spectrum['flux_conv']

	out = {'spectra_name_best': spectra_name_best, 'chi2_red_fit_best': chi2_red_fit_best, 'wl_model': wl_model, 
		   'flux_model': flux_model, 'wl_model_conv': wl_model_conv, 'flux_model_conv': flux_model_conv}

	return out

##################################################
# function to obtain a model spectrum with any combination of parameters within the model grid
def interpol_Sonora_Elf_Owl(Teff_interpol, logg_interpol, logKzz_interpol, Z_interpol, CtoO_interpol, grid=None, Teff_range=None, logg_range=None, save_spectrum=False):
	'''
	Teff_interpol: (float) temperature of the interpolated synthetic spectrum (275<=Teff(K)<=2400)
	logg_interpol: (float) logg of the interpolated synthetic spectrum (3.25<=logg<=5.5)
	logKzz_interpol: (float) logKzz of the interpolated synthetic spectrum (2<=logKzz(cgs)<=9)
	Z_interpol: (float) Z of the interpolated synthetic spectrum (-1.0<=[M/H](cgs)<=1.0)
	CtoO_interpol: (float) C/O of the interpolated synthetic spectrum (0.5<=C/O<=2.5)
	grid: float array, optional
		model grid (synthetic spectra) to be used to generate the desired spectrum
		this is the output of the read_grid definition, which is a dictionary with two variables (wavelength and flux) for each parameters' combination
	Teff_range: float array, necessary when grid is not provided
		minimum and maximum Teff values to select a subsample of the model grid. Such subset will allow faster interpolation than when reading the whole grid.
	logg_range: float array, necessary when grid is not provided
		minimum and maximum Teff values to select a subsample of the model grid. Such subset will allow faster interpolation than when reading the whole grid.
	'''

	if (grid is None): # read model grid when not provided
		if (Teff_range is None): exit('missing Teff range to read a model grid subset')
		if (logg_range is None): exit('missing logg range to read a model grid subset')
		read_grid(Teff_range, logg_range)

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

	params_interpol = ([Teff_interpol, logg_interpol, logKzz_interpol, Z_interpol, CtoO_interpol])
	spectra_interpol_flux = interp_flux(params_interpol)[0,:] # to return a 1D array
	spectra_interpol_wl = interp_wl(params_interpol)[0,:] # to return a 1D array

	# reverse array to sort wavelength from shortest to longest
	ind_sort = np.argsort(spectra_interpol_wl)
	spectra_interpol_wl = spectra_interpol_wl[ind_sort]
	spectra_interpol_flux = spectra_interpol_flux[ind_sort]

	# store generated spectrum
	if save_spectrum:
		out = open(f'Elf_Owl_Teff{Teff_interpol}_logg{logg_interpol}_logKzz{logKzz_interpol}_Z{Z_interpol}_CtoO{CtoO_interpol}.dat', 'w')
		out.write('# wavelength(um) flux(erg/s/cm2/A)  \n')
		for i in range(len(spectra_interpol_wl)):
			out.write('%11.7f %17.6E \n' %(spectra_interpol_wl[i], spectra_interpol_flux[i]))
		out.close()

#	# ATTEMPT TO GENERATE MORE THAN ONE SPECTRUM
#	# (ERROR: len(Teff_interpol) IS NOT RECOGNIZE WHEN Teff_interpol IS AN INTEGER/FLOAT rather than a list/array)
#	# generate synthetic spectra with the indicated parameters
#	params_interpol = np.zeros((len(Teff_interpol), 5))
#	for i in range(len(Teff_interpol)):
#		params_interpol[i,:] = ([Teff_interpol[i], logg_interpol[i], logKzz_interpol[i], Z_interpol[i], CtoO_interpol[i]])
#
#	spectra_interpol_flux = interp_flux(params_interpol)
#	spectra_interpol_wl = interp_wl(params_interpol)
#
#	# reverse array to sort wavelength from shortest to longest
#	for k in range(spectra_interpol_wl.shape[0]):
#		ind_sort = np.argsort(spectra_interpol_wl[k,:])
#		spectra_interpol_wl[k,:] = spectra_interpol_wl[k,:][ind_sort]
#		spectra_interpol_flux[k,:] = spectra_interpol_flux[k,:][ind_sort]
#
#	# store generated spectrum
#	if (save_spectrum=='yes'):
#		for k in range(spectra_interpol_wl.shape[0]):
#			out = open(f'Elf_Owl_Teff{Teff_interpol[k]}_logg{logg_interpol[k]}_logKzz{logKzz_interpol[k]}_Z{Z_interpol[k]}_CtoO{CtoO_interpol[k]}.dat', 'w')
#			out.write('# wavelength(um) flux(erg/s/cm2/A)  \n')
#			for i in range(spectra_interpol_wl.shape[1]):
#				out.write('%11.7f %17.6E \n' %(spectra_interpol_wl[k,i], spectra_interpol_flux[k,i]))
#			out.close()
#	
#	# when only one spectrum was generated, convert wavelength and flux arrays to 1D arrays
#	if (len(Teff_interpol)==1):
#		spectra_interpol_flux = spectra_interpol_flux[0,:]
#		spectra_interpol_wl = spectra_interpol_wl[0,:]

	out = {'wavelength': spectra_interpol_wl, 'flux': spectra_interpol_flux, 'params_interpol': params_interpol}

#	print('\nsynthetic spectrum was generated')

	return out

##################################################
# to read the grid considering Teff and logg desired ranges
def read_grid(model, Teff_range, logg_range, convolve=True, wl_range=None):
#def read_grid(model, Teff_range, logg_range, convolve=True, lam_R=None, R=None):
	'''
	model : desired atmospheric model
	Teff_range : float array, necessary when grid is not provided
		minimum and maximum Teff values to select a subsample of the model grid. Such subset will allow faster interpolation than when reading the whole grid.
	logg_range : float array, necessary when grid is not provided
		minimum and maximum Teff values to select a subsample of the model grid. Such subset will allow faster interpolation than when reading the whole grid.
	convolve : bool
		if T rue (default), the synthetic spectra will be convolved to the indicated R at lam_R
	wl_range : float array (optional)
		minimum and maximum wavelength values to read from model grid
#	R : float, mandatory if convolve is True
#		input spectra resolution (default R=100) at lam_R to smooth model spectra
#	lam_R : float, mandatory if convolve is True
#		wavelength reference (default 2 um) to estimate the spectral resolution of model spectra considering R

	OUTPUT:
		dictionary with the grid ('wavelength' and 'flux') for the parameters ('Teff', 'logg', 'logKzz', 'Z', and 'CtoO') within Teff_range and logg_range and all the values for the remaining parameters.
	'''

	print('\n   reading model grid...')

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

	# read the grid in the constrained ranges
	# make N-D coordinate arrays
	Teff_mesh, logg_mesh, logKzz_mesh, Z_mesh, CtoO_mesh = np.meshgrid(Teff_grid, logg_grid, logKzz_grid, Z_grid, CtoO_grid, indexing='ij', sparse=True)

	path = '/home/gsuarez/TRABAJO/MODELS/atmosphere_models/'
	spectra = 'Sonora_Elf_Owl/spectra/'
	ini_time_interpol = time.time() # to estimate the time elapsed running interpol
	
	N_datapoints = 193132
	flux_grid = np.zeros((len(Teff_grid), len(logg_grid), len(logKzz_grid), len(Z_grid), len(CtoO_grid), N_datapoints)) # to save the flux at each grid point
	wl_grid = np.zeros((len(Teff_grid), len(logg_grid), len(logKzz_grid), len(Z_grid), len(CtoO_grid), N_datapoints)) # to save the wavelength at each grid point
	
	k = 0
	for i_Teff in range(Teff_grid.size): # iterate Teff
		for i_logg in range(len(logg_grid)): # iterate logg
			for i_logKzz in range(len(logKzz_grid)): # iterate logKzz
				for i_Z in range(len(Z_grid)): # iterate Z
					for i_CtoO in range(len(CtoO_grid)): # iterate C/O ration
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
						
						# indicate the folder containing the spectra depending on the Teff value
						if ((Teff_grid[i_Teff]>=275) & (Teff_grid[i_Teff]<=325)): folder='output_275.0_325.0/'
						if ((Teff_grid[i_Teff]>=350) & (Teff_grid[i_Teff]<=400)): folder='output_350.0_400.0/'
						if ((Teff_grid[i_Teff]>=425) & (Teff_grid[i_Teff]<=475)): folder='output_425.0_475.0/'
						if ((Teff_grid[i_Teff]>=500) & (Teff_grid[i_Teff]<=550)): folder='output_500.0_550.0/'
						if ((Teff_grid[i_Teff]>=575) & (Teff_grid[i_Teff]<=650)): folder='output_575.0_650.0/'
						if ((Teff_grid[i_Teff]>=700) & (Teff_grid[i_Teff]<=800)): folder='output_700.0_800.0/'
						if ((Teff_grid[i_Teff]>=850) & (Teff_grid[i_Teff]<=950)): folder='output_850.0_950.0/'
						if ((Teff_grid[i_Teff]>=1000) & (Teff_grid[i_Teff]<=1200)): folder='output_1000.0_1200.0/'
						if ((Teff_grid[i_Teff]>=1300) & (Teff_grid[i_Teff]<=1500)): folder='output_1300.0_1400.0/'
						if ((Teff_grid[i_Teff]>=1600) & (Teff_grid[i_Teff]<=1800)): folder='output_1600.0_1800.0/'
						if ((Teff_grid[i_Teff]>=1900) & (Teff_grid[i_Teff]<=2100)): folder='output_1900.0_2100.0/'
						if ((Teff_grid[i_Teff]>=2200) & (Teff_grid[i_Teff]<=2400)): folder='output_2200.0_2400.0/'
						
						spectrum_name = f'spectra_logzz_{logKzz_grid[i_logKzz]}_teff_{Teff_grid[i_Teff]}_grav_{g_grid}_mh_{Z_grid[i_Z]}_co_{CtoO_grid[i_CtoO]}.nc'
						
						if os.path.exists(path+spectra+folder+spectrum_name):
							# read spectrum from each combination of parameters
							spec_model = xarray.open_dataset(path+spectra+folder+spectrum_name) # Sonora Elf Owl model spectra have NetCDF Data Format data
							wl_model = spec_model['wavelength'].data # um 
							flux_model = spec_model['flux'].data # erg/s/cm^2/cm
							flux_model = flux_model / 1.e8 # erg/s/cm^2/Angstrom
				
#							# resample convolved model to the wavelength data points in the observed spectra
#							if convolve:
#								out_convolve_spectrum = seda_utils.convolve_spectrum(wl=wl_model, flux=flux_model, lam_R=lam_R, R=R)
#								wl_model = out_convolve_spectrum['wl_conv']
#								flux_model = out_convolve_spectrum['flux_conv']

							# save flux at each combination
							flux_grid[i_Teff, i_logg, i_logKzz, i_Z, i_CtoO, :] = flux_model
							wl_grid[i_Teff, i_logg, i_logKzz, i_Z, i_CtoO, :] = wl_model
                            # as wavelength is the same for all spectra, there is not need to save the wavelength of each spectrum
						
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

#	# cut the synthetic models
#	if wl_range is not None:
#		mask = (wl_model>=wl_range[0]) & (wl_model<=wl_range[1])

	fin_time_interpol = time.time()
	time_interpol = fin_time_interpol-ini_time_interpol # in seconds
	out_time_elapsed = time_elapsed(fin_time_interpol-ini_time_interpol)
	print(f'      {k} spectra in memory to define a grid for interpolations')
	print(f'         elapsed time: {out_time_elapsed[0]} {out_time_elapsed[1]}')

	out = {'wavelength': wl_grid, 'flux': flux_grid, 'Teff': Teff_grid, 'logg': logg_grid, 'logKzz': logKzz_grid, 'Z': Z_grid, 'CtoO': CtoO_grid}

	return out
