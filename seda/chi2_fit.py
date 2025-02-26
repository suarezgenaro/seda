import numpy as np
import time
import os
import astropy.units as u
import pickle
from spectres import spectres
from tqdm.auto import tqdm
from prettytable import PrettyTable
from astropy.io import ascii
from astropy.table import vstack
from lmfit import Minimizer, minimize, Parameters, report_fit # model fit for non-linear least-squares problems
from sys import exit
from .utils import *

def chi2(my_chi2):
	'''
	Description:
	------------
		Minimize the chi-square statistic to find the best model fits.

	Parameters:
	-----------
	- my_chi2 : return parameters from ``input_parameters.Chi2Options``, which also includes the parameters from ``input_parameters.InputData`` and ``input_parameters.ModelOptions``.

	Returns:
	--------
	- '``model``\_chi2\_minimization.dat' : ascii table
		Table with the names of all model spectra fits sorted by chi square and including the information: 
		spectrum name, chi square, reduced chi square, scaling, scaling error, extinction, extinction error, 
		model free parameters, radius and error (if ``distance and ``edistance`` are provided), and
		iterations to minimize chi-square.
	- '``model``\_chi2\_minimization.pickle' : dictionary
		Dictionary with the results from the chi-square minimization, namely:
			- ``my_chi2`` input dictionary
			- ``spectra_name_full``: model spectra names with full path.
			- ``spectra_name``: model spectra names.
			- ``iterations_fit``: number of iterations to minimize chi-square.
			- ``Av_fit``: visual extinction (in mag) that minimizes chi-square.
			- ``eAv_fit``: visual extinction uncertainty (in mag).
			- ``scaling_fit``: scaling factor that minimizes chi-square.
			- ``escaling_fit``: scaling factor uncertainty.
			- ``chi2_wl_fit``: chi-square as a function of wavelength.
			- ``chi2_red_wl_fit``: reduced chi-square as a function of wavelength.
			- ``chi2_fit``: total chi-square.
			- ``chi2_red_fit``: total reduced chi-square.
			- ``weight_fit``: weight given to each data point in the fit considering the equation chi2 = weight * (data-model)^2 / edata^2.
			- ``wl_array_model_conv_resam``: (if ``fit_spectra``) wavelength (in um) of resampled, convolved model spectra.
			- ``flux_array_model_conv_resam``: (if ``fit_spectra``) scaled, resampled, convolved model spectra (in erg/cm2/s/A).
			- ``flux_array_model_conv_resam_red``: (if ``fit_spectra``) scaled, resampled, convolved, reddened model spectra (in erg/cm2/s/A).
			- ``lambda_eff_mean``: (if ``fit_photometry``) mean effective wavelength (in um) of each input photometric passband.
			- ``width_eff_mean``: (if ``fit_photometry``) mean effective width (in um) of each input photometric passband.
			- ``f_phot``: (if ``fit_photometry``) fluxes (in erg/s/cm2/A) of each input photometry.
			- ``ef_phot``: (if ``fit_photometry``) flux uncertainties (in erg/s/cm2/A).
			- ``phot_synt``: (if ``fit_photometry``) synthetic fluxes (in erg/s/cm2/A) from each model spectrum considering the different filters.
			- ``phot_synt_red``: (if ``fit_photometry``) synthetic fluxes (in erg/s/cm2/A) from each reddened model spectrum considering the different filters.
			- ``radius``: (if ``distance`` is provided) radius (in Rjup) considering the ``scaling_fit`` and input ``distance``.
			- ``eradius``: (if ``edistance`` is provided) radius uncertainty (in Rjup).
			- ``wl_array_obs_fit``: input wavelengths for observed spectra within ``fit_wl_range`` (or model coverage, in case it is narrower).
			- ``flux_array_obs_fit``: input fluxes for ``wl_array_obs_fit``.
			- ``eflux_array_obs_fit``: input flux errors for ``wl_array_obs_fit``.
			- ``flux_residuals``: linear flux residual (in erg/cm2/s/A) between observed data and model spectra in ``fit_wl_range``.
			- ``logflux_residuals``: logarithm flux residual (in erg/cm2/s/A) between observed data and model spectra ``fit_wl_range``.
			- ``params``: model free parameters for each model spectrum.

	Example:
	--------
	>>> import seda
	>>> 
	>>> # load input data
	>>> wl_spectra = wl_input # in um
	>>> flux_spectra = flux_input # in erg/cm^2/s/A
	>>> eflux_spectra = eflux_input # in erg/cm^2/s/A
	>>> my_data = seda.InputData(wl_spectra=wl_spectra, flux_spectra=flux_spectra, 
	>>>                          eflux_spectra=eflux_spectra)
	>>> 
	>>> # load model options
	>>> model = 'Sonora_Elf_Owl'
	>>> model_dir = ['my_path/output_700.0_800.0/', 
	>>>              'my_path/output_850.0_950.0/'] # folders to seek model spectra
	>>> params_ranges = {'Teff': [700, 900, 'logg': [4.0, 5.0]
	>>> my_model = seda.ModelOptions(model=model, model_dir=model_dir, 
	>>>                              params_ranges=params_ranges)
	>>> 
	>>> # load chi-square options
	>>> fit_wl_range = np.array([value1, value2]) # to make the fit between value1 and value2
	>>> my_chi2 = seda.Chi2FitOptions(my_data=my_data, my_model=my_model, 
	>>>                               fit_wl_range=fit_wl_range)
	>>> 
	>>> # run chi-square fit
	>>> out_chi2 = seda.chi2(my_chi2)
	    Chi square fit ran successfully

	Author: Genaro Suárez
	'''

	ini_time_chi2 = time.time() # to estimate the time elapsed running chi2
	print('\n   Running chi-square fitting...')

	# load input parameters
	# all are stored in my_chi2 but were defined in different classes
	# from InputData
	fit_spectra = my_chi2.fit_spectra
	fit_photometry = my_chi2.fit_photometry
	wl_spectra = my_chi2.wl_spectra
	flux_spectra = my_chi2.flux_spectra
	eflux_spectra = my_chi2.eflux_spectra
	mag_phot = my_chi2.mag_phot
	emag_phot = my_chi2.emag_phot
	filter_phot = my_chi2.filter_phot
	res = my_chi2.res
	lam_res = my_chi2.lam_res
	distance = my_chi2.distance
	edistance = my_chi2.edistance
	N_spectra = my_chi2.N_spectra
	# from ModelOptions
	model = my_chi2.model
	model_dir = my_chi2.model_dir
	params_ranges = my_chi2.params_ranges
#	save_convolved_spectra = my_chi2.save_convolved_spectra
	path_save_spectra_conv = my_chi2.path_save_spectra_conv
	skip_convolution = my_chi2.skip_convolution
	# from Chi2Options
	save_results = my_chi2.save_results
	scaling_free_param = my_chi2.scaling_free_param
	scaling = my_chi2.scaling
	extinction_free_param = my_chi2.extinction_free_param
	avoid_IR_excess = my_chi2.avoid_IR_excess
	IR_excess_limit = my_chi2.IR_excess_limit
	fit_wl_range = my_chi2.fit_wl_range
	model_wl_range = my_chi2.model_wl_range
	wl_spectra_min = my_chi2.wl_spectra_min
	wl_spectra_max = my_chi2.wl_spectra_max
	N_datapoints = my_chi2.N_datapoints
	chi2_pickle_file = my_chi2.chi2_pickle_file
	chi2_table_file = my_chi2.chi2_pickle_file


#	# read the model spectra names in the input folders and meeting the indicated parameters ranges 
#	out_select_model_spectra = select_model_spectra(model=model, model_dir=model_dir, params_ranges=params_ranges)
#	spectra_name_full = out_select_model_spectra['spectra_name_full']
#	spectra_name = out_select_model_spectra['spectra_name']
#	N_model_spectra = len(spectra_name) # number of selected model spectra
#
#	#+++++++++++++++++++++++++++++++++
#	# read and convolve model spectra
#
#	# convolve model spectra to the required resolution
#	if not skip_convolution: # read and convolve original model spectra
#		# create a tqdm progress bar
#		convolving_bar = tqdm(total=N_model_spectra, desc='Convolving the spectra')
#
#		# initialize lists for convolved model spectra for all input spectra
#		wl_array_model_conv = []
#		flux_array_model_conv = []
#		for i in range(N_model_spectra): # for each model spectrum
#			# update progress bar
#			convolving_bar.update(1)
#		
#			# read model spectrum
#			out_read_model_spectrum = read_model_spectrum(spectrum_name_full=spectra_name_full[i], model=model, model_wl_range=model_wl_range)
#			wl_model = out_read_model_spectrum['wl_model'] # um
#			flux_model = out_read_model_spectrum['flux_model'] # erg/s/cm2/A
#		
#			# convolved model spectrum
#			wl_array_model_conv_each = [] # to save each convolved spectrum 
#			flux_array_model_conv_each = [] # to save each convolved spectrum 
#			for k in range(N_spectra): # for each input observed spectrum
#				# convolve models in the full wavelength range of each input spectrum plus a padding
#				convolve_wl_range = [0.9*wl_spectra[k].min(), 1.1*wl_spectra[k].max()]
#	
#				# convolve model spectrum according to the resolution and fit range of each input spectrum
#				if path_save_spectra_conv is None: # do not save the convolved spectrum
#					out_convolve_spectrum = convolve_spectrum(wl=wl_model, flux=flux_model, lam_res=lam_res[k], res=res[k], 
#					                                          disp_wl_range=fit_wl_range[k], convolve_wl_range=convolve_wl_range)
#				else: # save convolved spectrum
#					if not os.path.exists(path_save_spectra_conv): os.makedirs(path_save_spectra_conv) # make directory (if not existing) to store convolved spectra
#					out_file = path_save_spectra_conv+spectra_name[i]+f'_R{res[k]}at{lam_res[k]}um.nc'
#					out_convolve_spectrum = convolve_spectrum(wl=wl_model, flux=flux_model, lam_res=lam_res[k], res=res[k], 
#					                                          disp_wl_range=fit_wl_range[k], convolve_wl_range=convolve_wl_range, out_file=out_file)
#				# save each convolved spectrum in the list
#				wl_array_model_conv_each.append(out_convolve_spectrum['wl_conv'])
#				flux_array_model_conv_each.append(out_convolve_spectrum['flux_conv'])
#	
#			# nested list with all convolved model spectra
#			wl_array_model_conv.append(wl_array_model_conv_each)
#			flux_array_model_conv.append(flux_array_model_conv_each)
#	
#		# close progress bar
#		convolving_bar.close()
#
#	else: # read precomputed convolved spectra
#		# get original filenames by removing the extension name added when the spectra were convolved and stored
#		spectra_name_ori = []
#		for spectrum in spectra_name: # for each selected model spectrum
#			name = spectrum.split('_R')[0]
#			if name not in spectra_name_ori: # to avoid repeating the same model spectrum when inputting multiple observed spectra
#				spectra_name_ori.append(name)
#		spectra_name_full_ori = [model_dir[0]+spectrum for spectrum in spectra_name_ori] # spectra files with full path
#
#		# necessary renaming to keep only unique model spectra
#		# as for multiple input spectra the same model spectrum is
#		# read multiple times by select_model_spectra because of the 
#		# different res and lam_res input values for the input spectra
#		spectra_name = np.array(spectra_name_ori)
#		spectra_name_full = np.array(spectra_name_full_ori)
#		N_model_spectra = len(spectra_name)
#
#		# read convolved spectra and save them as a nested list
#		# initialize lists for convolved model spectra for all input spectra
#		wl_array_model_conv = []
#		flux_array_model_conv = []
#		for i in range(N_model_spectra): # for each model spectrum
#			wl_array_model_conv_each = [] # to save each convolved spectrum 
#			flux_array_model_conv_each = [] # to save each convolved spectrum 
#			for k in range(N_spectra): # for each input observed spectrum
#				# check if there is a convolved spectrum for each res and lam_res combination
#				name_ext = f'_R{res[k]}at{lam_res[k]}um.nc'
#				spectrum_name_ext = spectra_name_full[i]+name_ext
#				if os.path.exists(spectrum_name_ext): # there is a convolved spectrum 
#					# read convolved spectrum
#					out_read_model_spectrum_conv = read_model_spectrum_conv(spectrum_name_ext)
#					# save each convolved spectrum in the list
#					wl_array_model_conv_each.append(out_read_model_spectrum_conv['wl_model']) # um
#					flux_array_model_conv_each.append(out_read_model_spectrum_conv['flux_model']) # erg/s/cm2/A
#				else: raise Exception(f'Convolved model spectrum {spectrum_name_ext} does not exits.')
#
#			# nested list with all convolved model spectra
#			wl_array_model_conv.append(wl_array_model_conv_each)
#			flux_array_model_conv.append(flux_array_model_conv_each)
#
#	#+++++++++++++++++++++++++++++++++
#	# resample the convolved model spectra in input spectra wavelength ranges
#
#	# initialize lists for resampled and convolved model spectra for all input spectra
#	wl_array_model_conv_resam = []
#	flux_array_model_conv_resam = []
#
#	# create a tqdm progress bar
#	resampling_bar = tqdm(total=N_model_spectra, desc='Resampling the spectra')
#	for i in range(N_model_spectra): # for each model spectrum
#		# update progress bar
#		resampling_bar.update(1)
#
#		# list to save a resampled and convolved model spectrum for the input spectra
#		wl_array_model_conv_resam_each = []
#		flux_array_model_conv_resam_each = []
#		for k in range(N_spectra): # for each input observed spectrum
#			mask = (wl_spectra[k]>wl_array_model_conv[i][k].min()) & (wl_spectra[k]<wl_array_model_conv[i][k].max()) # input wavelengths within model wavelength coverage
#			flux_array_model_conv_resam_each.append(spectres(wl_spectra[k][mask], wl_array_model_conv[i][k], flux_array_model_conv[i][k])) # resampled fluxes
#			wl_array_model_conv_resam_each.append(wl_spectra[k][mask]) # wavelengths for resampled fluxes
#
#		# nested list with all resampled and convolved model spectra
#		wl_array_model_conv_resam.append(wl_array_model_conv_resam_each)
#		flux_array_model_conv_resam.append(flux_array_model_conv_resam_each)
#
#	# close progress bar
#	resampling_bar.close()

	# initialize lists for resampled, convolved model spectra for all input spectra
	wl_array_model_conv_resam = []
	flux_array_model_conv_resam = []
	for k in range(N_spectra): # for each input observed spectrum
		print(f'\nFor input spectrum {k+1} of {N_spectra}')

		# select files with spectra within the indicated ranges
		if not skip_convolution: # read and convolve original model spectra
			out_select_model_spectra = select_model_spectra(model=model, model_dir=model_dir, params_ranges=params_ranges)
		else: # read model spectra already convolved to the data resolution
			# set filename_pattern to look for model spectra with the corresponding resolution
			filename_pattern = Models(model).filename_pattern+f'_R{res[k]}at{lam_res[k]}um.nc'
			out_select_model_spectra = select_model_spectra(model=model, model_dir=model_dir, params_ranges=params_ranges, filename_pattern=filename_pattern)
		spectra_name_full = out_select_model_spectra['spectra_name_full']
		spectra_name = out_select_model_spectra['spectra_name']
		N_model_spectra = len(spectra_name) # number of selected model spectra

		# create a tqdm progress bar
		if not skip_convolution: # read original model spectra
			model_bar = tqdm(total=N_model_spectra, desc='Reading, convolving, and resampling model spectra')
		else:
			model_bar = tqdm(total=N_model_spectra, desc='Reading and resampling model spectra')

		# list to save a resampled and convolved model spectrum for the input spectra
		wl_array_model_conv_resam_each = []
		flux_array_model_conv_resam_each = []
		for i in range(N_model_spectra): # for each model spectrum
			# update progress bar
			model_bar.update(1)
	
			# read and convolve model spectra, if needed
			if not skip_convolution: # read original model spectra
				out_read_model_spectrum = read_model_spectrum(spectrum_name_full=spectra_name_full[i], model=model, model_wl_range=model_wl_range)
			else: # read model spectra already convolved to the data resolution
				out_read_model_spectrum = read_model_spectrum_conv(spectrum_name_full=spectra_name_full[i], model_wl_range=model_wl_range)
			wl_model = out_read_model_spectrum['wl_model'] # um
			flux_model = out_read_model_spectrum['flux_model'] # erg/s/cm2/A
	
			# convolve the model spectrum to the indicated resolution
			wl_array_model_conv_each = [] # to save each convolved spectrum 
			flux_array_model_conv_each = [] # to save each convolved spectrum 
			if not skip_convolution: # avoid convolution, even if convolve was left as True, when skip_convolution is True
				# convolve models in the full wavelength range of each input spectrum plus a padding
				convolve_wl_range = [0.9*wl_spectra[k].min(), 1.1*wl_spectra[k].max()]
		
				# convolve model spectrum according to the resolution and fit range of each input spectrum
				if path_save_spectra_conv is None: # do not save the convolved spectrum
					out_convolve_spectrum = convolve_spectrum(wl=wl_model, flux=flux_model, lam_res=lam_res[k], res=res[k], 
					                                          disp_wl_range=fit_wl_range[k], convolve_wl_range=convolve_wl_range)
				else: # save convolved spectrum
					if not os.path.exists(path_save_spectra_conv): os.makedirs(path_save_spectra_conv) # make directory (if not existing) to store convolved spectra
					out_file = path_save_spectra_conv+spectra_name[i]+f'_R{res[k]}at{lam_res[k]}um.nc'
					out_convolve_spectrum = convolve_spectrum(wl=wl_model, flux=flux_model, lam_res=lam_res[k], res=res[k], 
					                                          disp_wl_range=fit_wl_range[k], convolve_wl_range=convolve_wl_range, out_file=out_file)
				
				wl_model = out_convolve_spectrum['wl_conv']
				flux_model = out_convolve_spectrum['flux_conv']

			# resample the convolved model spectrum to the wavelength data points in the observed spectra
			mask = (wl_spectra[k]>wl_model.min()) & (wl_spectra[k]<wl_model.max()) # input wavelengths within model wavelength coverage
			flux_array_model_conv_resam_each.append(spectres(wl_spectra[k][mask], wl_model, flux_model)) # resampled fluxes
			wl_array_model_conv_resam_each.append(wl_spectra[k][mask]) # wavelengths for resampled fluxes

		# nested list with all resampled and convolved model spectra
		wl_array_model_conv_resam.append(wl_array_model_conv_resam_each)
		flux_array_model_conv_resam.append(flux_array_model_conv_resam_each)

		# close progress bar
		model_bar.close()

	# re-arrange the list with convolved, resampled spectra to have each model in the first dimension
	wl_arr = []
	flux_arr = []
	for i in range(N_model_spectra): # for each model spectrum 
		wl_arr_each = []
		flux_arr_each = []
		for k in range(N_spectra): # for each input observed spectrum
			wl_arr_each.append(wl_array_model_conv_resam[k][i])
			flux_arr_each.append(flux_array_model_conv_resam[k][i])
		wl_arr.append(wl_arr_each)
		flux_arr.append(flux_arr_each)
	wl_array_model_conv_resam = wl_arr
	flux_array_model_conv_resam = flux_arr
	
	# when reading pre-computed convolved model spectra, remove the extended name added when saving the convolved models
	# spectra_name in memory are the ones for the last input spectrum, but they have the same root name for all input spectra
#	if not skip_convolution: # avoid convolution, even if convolve was left as True, when skip_convolution is True
# FIX THAT spectra_name_full ATTACH ONLY THE FIRST FOLDER IN model_dir TO spectra_name
	if skip_convolution: # avoid convolution, even if convolve was left as True, when skip_convolution is True
		spectra_name_ori = []
		for spectrum in spectra_name: # for each selected model spectrum
			name = spectrum.split('_R')[0]
			if name not in spectra_name_ori: # to avoid repeating the same model spectrum when inputting multiple observed spectra
				spectra_name_ori.append(name)
		spectra_name_full_ori = [model_dir[0]+spectrum for spectrum in spectra_name_ori] # spectra files with full path
		# renaming
		spectra_name = np.array(spectra_name_ori)
		spectra_name_full = np.array(spectra_name_full_ori)
		N_model_spectra = len(spectra_name)

	#+++++++++++++++++++++++++++++++++
	# minimize chi-square

	# cut input spectra to the wavelength region or model coverage for the fit
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

	# initialize variables to save key parameters from the fit
	scaling_fit = np.zeros(N_model_spectra) * np.nan
	escaling_fit = np.zeros(N_model_spectra) * np.nan
	Av_fit = np.zeros(N_model_spectra) * np.nan
	eAv_fit = np.zeros(N_model_spectra) * np.nan
	iterations_fit = np.zeros(N_model_spectra) * np.nan
	chi2_fit = np.zeros(N_model_spectra) * np.nan
	chi2_red_fit = np.zeros(N_model_spectra) * np.nan
	chi2_wl_fit = []
	chi2_red_wl_fit = []

	# create a tqdm progress bar
	chi2_bar = tqdm(total=N_model_spectra, desc='Minimizing chi-square')
	for i in range(N_model_spectra): # for each model spectrum
		# update the progress bar
		chi2_bar.update(1)

		# define free parameters in the fit
		params = Parameters()
		if (extinction_free_param=='no'): params.add('extinction', value=0, vary=False) # fixed parameter
		if (extinction_free_param=='yes'): params.add('extinction', value=0) # free parameter
		if (scaling_free_param=='no'): params.add('scaling', value=scaling, vary=False) # fixed parameter
		if (scaling_free_param=='yes'): params.add('scaling', value=1e-20) # free parameter

		# minimize chi square
		data_fit = flux_spectra_fit[i] # all input fluxes
		edata_fit = eflux_spectra_fit[i] # all input flux uncertainties
		model_fit = flux_array_model_conv_resam_fit[i] # model fluxes from each resampled and convolved model spectrum

		minner = Minimizer(residuals_for_chi2, params, fcn_args=(data_fit, edata_fit, model_fit))
		out_lmfit = minner.minimize(method='leastsq') # 'leastsq': Levenberg-Marquardt (default)

		# save parameters
		iterations_fit[i] = out_lmfit.nfev # number of function evaluations in the fit
		Av_fit[i] = out_lmfit.params['extinction'].value
		eAv_fit[i] = out_lmfit.params['extinction'].stderr # half the difference between the 15.8 and 84.2 percentiles of the PDF
		scaling_fit[i] = out_lmfit.params['scaling'].value
		escaling_fit[i] = out_lmfit.params['scaling'].stderr # half the difference between the 15.8 and 84.2 percentiles of the PDF
		chi2_fit[i] = out_lmfit.chisqr # resulting chi square from lmfit
		chi2_red_fit[i] = out_lmfit.redchi # resulting reduced chi square from lmfit
		chi2_wl_fit.append(out_lmfit.residual**2) # chi square for each data point from lmfit
		chi2_red_wl_fit.append(out_lmfit.residual**2 / out_lmfit.nfree) # reduced chi square for each data point from lmfit

		# scale model fluxes by the value that minimizes chi2
		if fit_spectra:
			for k in range(N_spectra): # for each input observed spectrum
				flux_array_model_conv_resam[i][k] = flux_array_model_conv_resam[i][k] * scaling_fit[i]
				flux_array_model_conv_resam_fit[i][k] = flux_array_model_conv_resam_fit[i][k] * scaling_fit[i]

	# close progress bar
	chi2_bar.close()

	# radius from the scaling factor and cloud distance
	if distance!=None: # derive radius only if a distance is provided
		distance_km = (distance*u.parsec).to(u.km) # distance in km
		radius_km = np.sqrt(scaling_fit) * distance_km # in km
		radius = radius_km.to(u.R_jup).value # in R_jup
		if edistance!=None: # obtain radius error only if a distance error is provided
			edistance_km = (edistance*u.parsec).to(u.km) # distance error in km
			eradius_km = radius_km * np.sqrt((edistance_km/distance_km)**2 + (escaling_fit/(2*scaling_fit))**2) # in km
			eradius = eradius_km.to(u.R_jup).value # in Rjup

	#+++++++++++++++++++++++++++++++++
	# output dictionary
	out_chi2 = {'my_chi2': my_chi2}
	# add more elements to the dictionary
	out_chi2.update({'spectra_name_full': spectra_name_full, 'spectra_name': spectra_name, 'N_model_spectra': N_model_spectra, 
 				     'iterations_fit': iterations_fit, 'Av_fit': Av_fit, 'eAv_fit': eAv_fit, 'scaling_fit': scaling_fit, 
		             'escaling_fit': escaling_fit, 'chi2_wl_fit': chi2_wl_fit, 'chi2_red_wl_fit': chi2_red_wl_fit, 
		             'chi2_fit': chi2_fit, 'chi2_red_fit': chi2_red_fit})#, 'weight_fit': weight_fit})

	# add more output parameters depending on the input information
	if fit_spectra: # when only spectra are used in the fit
		# resampled and convolved model spectra in the input spectra wavelength ranges
		out_chi2['wl_array_model_conv_resam'] = wl_array_model_conv_resam
		out_chi2['flux_array_model_conv_resam'] = flux_array_model_conv_resam
		# resampled and convolved model spectra within the fit ranges
		out_chi2['wl_array_model_conv_resam_fit'] = wl_array_model_conv_resam_fit
		out_chi2['flux_array_model_conv_resam_fit'] = flux_array_model_conv_resam_fit
		# observed fluxes within the fit ranges
		out_chi2['wl_array_obs_fit'] = wl_spectra_fit
		out_chi2['flux_array_obs_fit'] = flux_spectra_fit
		out_chi2['eflux_array_obs_fit'] = eflux_spectra_fit
	#if fit_photometry: # when photometry is used in the fit
	#	out_chi2['lambda_eff_mean'] = lambda_eff_mean
	#	out_chi2['width_eff_mean'] = width_eff_mean
	#	out_chi2['f_phot'] = f_phot
	#	out_chi2['ef_phot'] = ef_phot
	#	out_chi2['phot_synt'] = phot_synt
	#	if (extinction_free_param=='yes'):
	#		out_chi2['phot_synt_red'] = phot_synt_red
	if distance!=None: # when a radius was obtained
		out_chi2['radius'] = radius
		if edistance!=None:
			out_chi2['eradius'] = eradius

	# obtain residuals in linear and logarithmic-scale for fluxes in the fit range
	flux_residuals = []
	logflux_residuals = []
	for i in range(N_model_spectra): # for each model spectrum
		flux_residuals_each = []
		logflux_residuals_each = []
		for k in range(N_spectra): # for each input observed spectrum
			# linear scale
			res_lin = flux_array_model_conv_resam_fit[i][k] - flux_spectra_fit[i][k]
			flux_residuals_each.append(res_lin)
			# log scale
			mask_pos = flux_spectra_fit[i][k]>0 # mask to avoid negative input fluxes to obtain the logarithm
			res_log = np.log10(flux_array_model_conv_resam_fit[i][k][mask_pos]) - np.log10(flux_spectra_fit[i][k][mask_pos])
			logflux_residuals_each.append(res_log)
		# nested list with all resampled and convolved model spectra in the fit range
		flux_residuals.append(flux_residuals_each)
		logflux_residuals.append(logflux_residuals_each)
	out_chi2['flux_residuals'] = flux_residuals
	out_chi2['logflux_residuals'] = logflux_residuals

	# separate physical parameters from each model spectrum name
	out_separate_params = separate_params(model=model, spectra_name=spectra_name)
	# add model spectra parameters to the output dictionary
	out_chi2.update(out_separate_params)

	# save some information
	if save_results: # save table with model spectra sorted by chi square
		# save output dictionary as pickle
		with open(chi2_pickle_file, 'wb') as file:
			# serialize and write the variable to the file
			pickle.dump(out_chi2, file)
		print('      chi-square minimization results saved successfully')

		# save table with model spectra names sorted by chi square along with the parameters from each spectrum
		out_save_params = save_params(out_chi2=out_chi2)

	print('\n   Chi-square fit ran successfully')
	fin_time_chi2 = time.time()
	print_time(fin_time_chi2-ini_time_chi2)

	return out_chi2

###########################
#def residuals_for_chi2(params, data, edata, model, extinction_curve, weight):
##	'''
##	Description:
##	------------
##		Define objective function that returns the array to be minimized.
##
##	Parameters:
##	-----------
##	- params : ''lmfit.parameter.Parameters''
##		Parameters as ``Parameters()`` for fitting models to data using ``Minimizer``.
##	- data : array
##		Input data (spectra and/or photometry) for the fit.
##	- edata : array
##		Input data uncertainties for the fit.
##	- model : array
##		Model spectrum for the fit.
##	- extinction_curve : array, (0 when ``extinction_free_param=='no'``)
##		Extinction curve for wavelengths in the fit.
##	- weight: array
##		Weight given to each data point in the fit.
##
##	Returns:
##	--------
##	Objective function to be minimized by ``Minimizer``.
##
##	Author: Genaro Suárez
##	'''
#
#	extinction = params['extinction']
#	scaling = params['scaling']
#	model_red = 10**(-extinction*extinction_curve/2.5) * scaling * model
#	return np.sqrt(weight/np.mean(weight)) * (data-model_red)/edata # the square of this equation will be minimized by minner.minimize

#########################
# Objective function to be minimized by ``Minimizer``.
def residuals_for_chi2(params, data, edata, model):

	scaling = params['scaling']

	residuals = np.array([]) # initialize numpy array to save array with residual
	for i in range(len(data)): # for each data point
		res = (data[i]-scaling*model[i])/edata[i] # the square of this equation will be minimized by minner.minimize
		residuals = np.concatenate([residuals, res])

	return  residuals

##########################
def save_params(out_chi2):
	'''
	Description:
	------------
		Create table with model spectra names sorted by chi square along with relevant parameters.

	Parameters:
	-----------
	- out_chi2 : dictionary
		Dictionary with the following parameters:
			- ``spectra_name`` : spectrum name
			- ``chi2_fit`` : chi-square
			- ``chi2_red_fit`` : reduced chi-square
			- ``scaling_fit`` : scaling factor 
			- ``e_scaling_fit`` : scaling factor uncertainty
			- ``Av_fit`` : extinction
			- ``eAv_fit`` : extinction uncertainty
			- ``params`` : fundamental parameters provided by ``model``.
			- ``R`` : radius, when a ``distance`` was provided.
			- ``eR`` : radius error, when ``distance`` and ``edistance`` were provided.
			- ``iterations_fit`` : iterations to minimize chi square.

	Returns:
	--------
	- ``chi2_table_file`` : ascii table
		ASCII table with input parameters sorted according to reduced chi-square.
		Table named ``chi2_table_file``, as specified in ``input_parameters.Chi2Options``.

	Author: Genaro Suárez
	'''

	# sort index with respect to reduced chi-square
	ind = np.argsort(out_chi2['chi2_red_fit'])

	# read and sort relevant parameters from the input dictionary
	filename = out_chi2['spectra_name'][ind] # filename
	chi2 = out_chi2['chi2_fit'][ind] # chi-square as integer
	chi2_red = out_chi2['chi2_red_fit'][ind] # reduced chi-square with three decimals
	scaling_factor = out_chi2['scaling_fit'][ind]
	escaling_factor = out_chi2['escaling_fit'][ind]
	Av = out_chi2['Av_fit'][ind] # Av
	eAv = out_chi2['eAv_fit'][ind] # Av error
	iterations = out_chi2['iterations_fit'][ind] # iterations in the minimization
	params = out_chi2['params']  # free model paramters
	for param in params:
		params[param] = params[param][ind]
	if out_chi2['my_chi2'].distance is not None:
		R = out_chi2['radius'][ind] # radius
		eR = out_chi2['eradius'][ind] # radius error

	# make dictionary with parameters of interest
	my_dict = {} # initialize dictionary
	my_dict['filename'] = filename
	my_dict['chi2'] = np.round(chi2).astype(int) # chi-square as integer
	my_dict['chi2_red'] = np.round(chi2_red,3) # reduced chi-square with three decimals
	# round scaling and its error, which have scientific notation
	scaling = np.array([])
	escaling = np.array([])
	for i,scal in enumerate(scaling_factor): # for each scaling value
	    scaling = np.append(scaling, format(scaling_factor[i], '.3e')) # keep three decimals
	    escaling = np.append(escaling, format(escaling_factor[i], '.3e')) # keep three decimals
	my_dict['scaling'] = scaling # scaling
	my_dict['escaling'] = escaling # scaling error
	my_dict['Av'] = Av # Av
	my_dict['eAv'] = eAv # Av error
	my_dict.update(params) # free model paramters
	if out_chi2['my_chi2'].distance is not None:
		my_dict['R'] = np.round(R,3) # radius
		my_dict['eR'] = np.round(eR,3) # radius error
	my_dict['iterations'] = iterations.astype(int) # iterations in the minimization

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
	with open(out_chi2['my_chi2'].chi2_table_file, 'w') as f:
		f.write(ascii_table)

	return  

##########################
def open_chi2_table(filename):
	'''
	Description:
	------------
		Open ascii table with the results from chi-square minimization by ``seda.chi2``.


	Parameters:
	-----------
	- filename : str
		Ascii file with the chi-square minimization results.

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

	return table

##########################
# number of elements in a nested list
def count_elements_nested_list(lst):
	count = 0
	for i in range(len(lst)):
		for j in range(len(lst[i])):
			count += len(lst[i][j])

	return count
