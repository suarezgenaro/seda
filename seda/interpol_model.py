from scipy.interpolate import RegularGridInterpolator
import numpy as np
from .seda_utils import time_elapsed
import time
import os
import xarray

from sys import exit

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

#	import seda as seda
	import sampling as sampling
	import seda_utils

	print('\n   reading model grid...')

	# all parameters's steps in the grid
	out_grid_ranges = sampling.grid_ranges(model)
	
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
