import numpy as np
import time
import os
import xarray
import pickle
from spectres import spectres
from astropy.io import ascii
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm
from astropy import units as u
from astropy.convolution import Gaussian1DKernel, convolve
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
		Read indicated model spectra.

	Parameters:
	-----------
	- model : str
		Atmospheric models. See available models in ``input_parameters.ModelOptions``.  
	- model_dir : str or list
		Path to the directory (str or list) or directories (as a list) containing the model spectra (e.g., ``model_dir = ['path_1', 'path_2']``). 
	- model_wl_range : float array, optional
		Minimum and maximum wavelength to cut model spectra.

	Returns:
	--------
	Dictionary with model spectra:
		- ``'wl_model'`` : wavelengths in microns
		- ``'flux_model'`` : fluxes in erg/s/cm2/A

	Author: Genaro Suárez
	'''

	# read model spectra files
	if (model == 'Sonora_Diamondback'):
		spec_model = ascii.read(spectra_name_full, data_start=3, format='no_header')
		wl_model = spec_model['col1'] # um (in vacuum?)
		wl_model = (wl_model*1e4) / (1.0 + 2.735182e-4 + 131.4182/(wl_model*1e4)**2 + 2.76249e8/(wl_model*1e4)**4) # wavelength (in A) in the air
		wl_model = wl_model/1e4 # in um
		flux_model = spec_model['col2'] # W/m2/m
		flux_model = flux_model * 0.1 / 1.e6 # erg/cm2/s/A
	if (model == 'Sonora_Elf_Owl'):
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
	'''
#	'''
#	chi2_minimization_pickle_file: pickle file from chi2.py
#	N_best_fits: int with the number (default 1) of best model fits to be read
#	'''

	# open results from the chi square analysis
	with open(pickle_file, 'rb') as file:
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

	return out

##################################################
# function to obtain a model spectrum with any combination of parameters within the model grid
def interpol_Sonora_Elf_Owl(Teff_interpol, logg_interpol, logKzz_interpol, Z_interpol, CtoO_interpol, grid=None, Teff_range=None, logg_range=None, save_spectrum=False):
	'''
	'''
#	'''
#	Teff_interpol: (float) temperature of the interpolated synthetic spectrum (275<=Teff(K)<=2400)
#	logg_interpol: (float) logg of the interpolated synthetic spectrum (3.25<=logg<=5.5)
#	logKzz_interpol: (float) logKzz of the interpolated synthetic spectrum (2<=logKzz(cgs)<=9)
#	Z_interpol: (float) Z of the interpolated synthetic spectrum (-1.0<=[M/H](cgs)<=1.0)
#	CtoO_interpol: (float) C/O of the interpolated synthetic spectrum (0.5<=C/O<=2.5)
#	grid: float array, optional
#		model grid (synthetic spectra) to be used to generate the desired spectrum
#		this is the output of the read_grid definition, which is a dictionary with two variables (wavelength and flux) for each parameters' combination
#	Teff_range: float array, necessary when grid is not provided
#		minimum and maximum Teff values to select a subsample of the model grid. Such subset will allow faster interpolation than when reading the whole grid.
#	logg_range: float array, necessary when grid is not provided
#		minimum and maximum Teff values to select a subsample of the model grid. Such subset will allow faster interpolation than when reading the whole grid.
#	'''

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
def read_grid_Sonora_Elf_Owl(model, model_dir, Teff_range, logg_range, convolve=True, wl_range=None):
	'''
	'''
#	'''
#	model : desired atmospheric model
#	model_dir : str or list
#		Path to the directory (str or list) or directories (as a list) containing the model spectra (e.g., ``model_dir = ['path_1', 'path_2']``). 
#		Avoid using paths with null spaces. 
#	Teff_range : float array, necessary when grid is not provided
#		minimum and maximum Teff values to select a subsample of the model grid. Such subset will allow faster interpolation than when reading the whole grid.
#	logg_range : float array, necessary when grid is not provided
#		minimum and maximum Teff values to select a subsample of the model grid. Such subset will allow faster interpolation than when reading the whole grid.
#	convolve : bool
#		if True (default), the synthetic spectra will be convolved to the indicated ``res`` at ``lam_res``
#	wl_range : float array (optional)
#		minimum and maximum wavelength values to read from model grid
#
#	OUTPUT:
#		dictionary with the grid ('wavelength' and 'flux') for the parameters ('Teff', 'logg', 'logKzz', 'Z', and 'CtoO') within Teff_range and logg_range and all the values for the remaining parameters.
#	'''
#	R : float, mandatory if convolve is True
#		input spectra resolution (default R=100) at lam_R to smooth model spectra
#	lam_R : float, mandatory if convolve is True
#		wavelength reference (default 2 um) to estimate the spectral resolution of model spectra considering R

#	print('\nreading model grid...')

	# read models in the input folders
	out_select_model_spectra = select_model_spectra(Teff_range=Teff_range, logg_range=logg_range, model=model, model_dir=model_dir)
	spectra_name_full = out_select_model_spectra['spectra_name_full']
	spectra_name = out_select_model_spectra['spectra_name']

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

#	# make N-D coordinate arrays
#	Teff_mesh, logg_mesh, logKzz_mesh, Z_mesh, CtoO_mesh = np.meshgrid(Teff_grid, logg_grid, logKzz_grid, Z_grid, CtoO_grid, indexing='ij', sparse=True)
#
	ini_time_grid = time.time() # to estimate the time elapsed reading the grid

	N_modelpoints = model_points(model)
	flux_grid = np.zeros((len(Teff_grid), len(logg_grid), len(logKzz_grid), len(Z_grid), len(CtoO_grid), N_modelpoints)) # to save the flux at each grid point
	wl_grid = np.zeros((len(Teff_grid), len(logg_grid), len(logKzz_grid), len(Z_grid), len(CtoO_grid), N_modelpoints)) # to save the wavelength at each grid point
	
	# create a tqdm progress bar
	grid_bar = tqdm(total=len(spectra_name), desc='Making a grid with the model spectra')
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
							spec_model = xarray.open_dataset(spectrum_name_full) # Sonora Elf Owl model spectra have NetCDF Data Format data
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
	# close the progress bar
	grid_bar.close()

#	# cut the synthetic models
#	if wl_range is not None:
#		mask = (wl_model>=wl_range[0]) & (wl_model<=wl_range[1])

	fin_time_grid = time.time()
	print_time(fin_time_grid-ini_time_grid)

	out = {'wavelength': wl_grid, 'flux': flux_grid, 'Teff': Teff_grid, 'logg': logg_grid, 'logKzz': logKzz_grid, 'Z': Z_grid, 'CtoO': CtoO_grid}

	return out

##########################
# get the parameter ranges and steps in each model grid
def grid_ranges(model):
	'''
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

	out = {'Teff': Teff, 'logg': logg, 'logKzz': logKzz, 'Z': Z, 'CtoO': CtoO}

	return out

##########################
# tolerance around the best-fitting spectrum parameters to define the parameter ranges for posteriors
def param_ranges_sampling(model):
	'''
	'''

	# open results from the chi square analysis
	with open(f'{model}_chi2_minimization.pickle', 'rb') as file:
		# deserialize and retrieve the variable from the file
		out_chi2 = pickle.load(file)

	if (model=='Sonora_Elf_Owl'):
		ind_best_fit = np.argsort(out_chi2['chi2_red_fit'])[:3] # index of the three best-fitting spectra
		# median parameter values from the best-fitting models
		Teff_chi2 = np.median(out_chi2['Teff'][ind_best_fit])
		logg_chi2 = np.median(out_chi2['logg'][ind_best_fit])
		logKzz_chi2 = np.median(out_chi2['logKzz'][ind_best_fit])
		Z_chi2 = np.median(out_chi2['Z'][ind_best_fit])
		CtoO_chi2 = np.median(out_chi2['CtoO'][ind_best_fit])

		# whole grid parameter ranges to avoid trying to generate a spectrum out of the grid
		out_grid_ranges = grid_ranges(model)

		Teff_search = 50 # K (half of the biggest Teff step, which is 100 K)
		logg_search = 0.25 # dex (half of the biggest logg step, which is 0.5)
		logKzz_search = 1.5 # dex (half of the biggest logKzz step, which is 3)
		Z_search = 0.25 # cgs (half of the biggest Z step, which is 0.5)
		CtoO_search = 0.50 # relative to solar C/O (half of the biggest C/O step, which is 1.0)
	
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
# generate the synthetic spectrum with the posterior parameters
def best_fit_sampling(wl_spectra, model, sampling_file, lam_res, res, distance=None, grid=None, save_spectrum=False):
	'''
	'''
#	'''
#	Output: dictionary
#		synthetic spectrum for the posterior parameters: i) with original resolution, ii) convolved to the compared observed spectra, and iii) scaled to the determined radius
#	'''

	# open results from sampling
	with open(sampling_file, 'rb') as file:
		dresults = pickle.load(file)

	if model=='Sonora_Elf_Owl':
		Teff_med = np.median(dresults.samples[:,0]) # Teff values
		logg_med = np.median(dresults.samples[:,1]) # logg values
		logKzz_med = np.median(dresults.samples[:,2]) # logKzz values
		Z_med = np.median(dresults.samples[:,3]) # Z values
		CtoO_med = np.median(dresults.samples[:,4]) # CtoO values
		if distance is not None: R_med = np.median(dresults.samples[:,5]) # R values

	# generate spectrum with the median parameter values
	if grid is None:
		# read grid around the median values	
		# first define Teff and logg ranges around the median values
		out_grid_ranges = sampling.grid_ranges(model)
		
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
		grid = interpol_model.read_grid(model=model, Teff_range=Teff_range, logg_range=logg_range)

	# generate synthetic spectrum
	syn_spectrum = interpol_model.interpol_Sonora_Elf_Owl(Teff_interpol=Teff_med, logg_interpol=logg_med, 
														  logKzz_interpol=logKzz_med, Z_interpol=Z_med, CtoO_interpol=CtoO_med, grid=grid)
	# convolve synthetic spectrum
	out_convolve_spectrum = convolve_spectrum(wl=syn_spectrum['wavelength'], flux=syn_spectrum['flux'], lam_res=lam_res, res=res, 
												   disp_wl_range=np.array([wl_spectra.min(), wl_spectra.max()]), 
												   convolve_wl_range=np.array([0.99*wl_spectra.min(), 1.01*wl_spectra.max()])) # padding on both edges to avoid issues we using spectres

	# scale synthetic spectrum
	if distance is not None:
		flux_scaled = scale_synthetic_spectrum(wl=syn_spectrum['wavelength'], flux=syn_spectrum['flux'], distance=distance, radius=R_med)
		flux_conv_scaled = scale_synthetic_spectrum(wl=out_convolve_spectrum['wl_conv'], flux=out_convolve_spectrum['flux_conv'], distance=distance, radius=R_med)

	# resample scaled synthetic spectra to the observed wavelengths
	flux_conv_resam = spectres(wl_spectra, out_convolve_spectrum['wl_conv'], out_convolve_spectrum['flux_conv'])
	if distance is not None:
		flux_conv_scaled_resam = spectres(wl_spectra, out_convolve_spectrum['wl_conv'], flux_conv_scaled)

	# save synthetic spectrum
	if save_spectrum:
		Teff_file = round(Teff_med, 1)
		logg_file = round(logg_med, 2)
		logKzz_file = round(logKzz_med,1)
		Z_file = round(Z_med, 1)
		CtoO_file = round(CtoO_med, 1)

		out = open(f'Elf_Owl_Teff{Teff_file}_logg{logg_file}_logKzz{logKzz_file}_Z{Z_file}_CtoO{CtoO_file}.dat', 'w')
		out.write('# wavelength(um) flux(erg/s/cm2/A)  \n')
		for i in range(len(out_convolve_spectrum['wl_conv'])):
			out.write('%11.7f %17.6E \n' %(out_convolve_spectrum['wl_conv'][i], out_convolve_spectrum['flux_conv'][i]))
		out.close()
		

	# output dictionary
	out = {'wl': syn_spectrum['wavelength'], 'flux': syn_spectrum['flux'], 'wl_conv': out_convolve_spectrum['wl_conv'], 
		   'flux_conv': out_convolve_spectrum['flux_conv'], 'wl_conv_resam': wl_spectra, 'flux_conv_resam': flux_conv_resam}
	if distance is not None:
		out['flux_scaled'] = flux_scaled
		out['flux_conv_scaled'] = flux_conv_scaled
		out['flux_conv_scaled_resam'] = flux_conv_scaled_resam

	return out

##########################
def select_model_spectra(Teff_range, logg_range, model, model_dir):
	'''
	Description:
	------------
		Select model spectra from the indicated models and meeting given parameters ranges.

	Parameters:
	-----------
	- Teff_range : float array
		Minimum and maximum Teff values to select a model grid subset (e.g., ``Teff_range = np.array([Teff_min, Teff_max])``)
	- logg_range : float array
		Minimum and maximum logg values to select a model grid subset
	- model : str
		Atmospheric models. See available models in ``input_parameters.ModelOptions``.  
	- model_dir : str or list
		Path to the directory (str or list) or directories (as a list) containing the model spectra (e.g., ``model_dir = ['path_1', 'path_2']``). 
		
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
	>>> out = seda.select_model_spectra(Teff_range=Teff_range, logg_range=logg_range,
	>>>                                 model=model, model_dir=model_dir)

	Author: Genaro Suárez
	'''

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
	out_separate_params = separate_params(files_short, model)
	spectra_name_Teff = out_separate_params['Teff']
	spectra_name_logg = out_separate_params['logg']

	# select spectra within the desired Teff and logg ranges
	spectra_name_full = [] # full name with path
	spectra_name = [] # only spectra names
	for i in range(len(files)):
		if ((spectra_name_Teff[i]>=Teff_range[0]) & (spectra_name_Teff[i]<=Teff_range[1]) & 
			(spectra_name_logg[i]>=logg_range[0]) & (spectra_name_logg[i]<=logg_range[1])): # spectrum with Teff and logg within the indicated ranges
			spectra_name_full.append(files[i]) # keep only spectra within the Teff and logg ranges
			spectra_name.append(files_short[i]) # keep only spectra within the Teff and logg ranges

	#--------------
	# TEST to fit only a few model spectra
	#spectra_name = spectra_name[:2]
	#spectra_name_full = spectra_name_full[:2]
	#--------------

	if len(spectra_name_full)==0: print('   ERROR: NO SYNTHETIC SPECTRA IN THE INDICATED PARAMETER RANGES'), exit() # show up an error when there are no models in the indicated ranges
	else: print(f'\n      {len(spectra_name)} model spectra selected with Teff=[{Teff_range[0]}, {Teff_range[1]}] and logg=[{logg_range[0]}, {logg_range[1]}]')

	out = {'spectra_name_full': np.array(spectra_name_full), 'spectra_name': np.array(spectra_name)}

	return out

##########################
# separate parameters from each model spectrum name
def separate_params(spectra_name, model):
	'''
	Description:
	------------
		Extract parameters from the file names for model spectra.

	Parameters:
	-----------
	- spectra_name : array or list
		Model spectra names.
	- model : str
		Atmospheric models. See available models in ``input_parameters.ModelOptions``.  

	Returns:
	--------
	Dictionary with parameters for each model spectrum.
		- ``Teff``: effective temperature (in K) for each model spectrum.
		- ``logg``: surface gravity (log g) for each model spectrum.
		- ``Z``: (if provided by ``model``) metallicity for each model spectrum.
		- ``logKzz``: (if provided by ``model``) diffusion parameter for each model spectrum. 
		- ``fsed``: (if provided by ``model``) cloudiness parameter for each model spectrum.
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

##########################
# round logg to steps of 0.25
def round_logg_point25(logg):
	logg = round(logg*4.) / 4. 
	return logg
