import numpy as np
import time
import os
import astropy.units as u
import pickle
import copy
from spectres import spectres
from tqdm.auto import tqdm
from astropy.io import ascii
from astropy.table import vstack
from sys import exit
from . import utils
from . import models


def chi2(my_chi2):
	'''
	Description:
	------------
		Minimize the weighted least-squares statistic to find the best-fitting models.

	Parameters:
	-----------
	- my_chi2 : object returned by :class:`~seda.Chi2Options`, which also includes
		parameters from :class:`~seda.InputData` and :class:`~seda.ModelOptions`.

	Returns:
	--------
	- '``model``\\_chi2\\_minimization.dat' : ascii table
		Table with the names of all model spectra fits sorted by the weighted least-squares statistic and including the information:
		Spectrum name, weighted least-squares statistic, reduced weighted least-squares statistic,
		scaling factor and uncertainty, model parameters, and radius and uncertainty
		(if distance and edistance are provided).
	- '``model``\\_chi2\\_minimization.pickle' : dictionary
		Dictionary with the weighted least-squares fitting results, namely:
			- ``my_chi2``: input parameters from :class:`~seda.Chi2Options`.
			- ``scaling_fit``: scaling factor that minimizes the fit statistic.
			- ``escaling_fit``: scaling factor uncertainty.
			- ``chi2_wl_fit``: weighted least-squares statistic as a function of wavelength.
			- ``chi2_red_wl_fit``: reduced weighted least-squares statistic as a function of wavelength.
			- ``chi2_fit``: total fit statistic.
			- ``chi2_red_fit``: total reduced fit statistic.
			- ``radius``: (if ``distance`` is provided) radius (in Rjup) considering 
			  the ``scaling_fit`` and input ``distance``.
			- ``eradius``: (if ``edistance`` is provided) radius uncertainty (in Rjup).
			- ``params``: model grid parameters for each model spectrum.
			- ``flux_array_model_conv_resam_scaled``: (if ``fit spectra``) input ``flux_array_model_conv_resam`` 
			  flux densities multiplied by the scaling factor ``scaling_fit``.
			- ``flux_array_model_conv_resam_scaled_fit``: (if ``fit spectra``) input ``flux_array_model_conv_resam_fit`` 
			  flux densities multiplied by the scaling factor ``scaling_fit``.
			- ``flux_residuals_spec``: (if ``fit spectra``) linear flux density residuals (in erg/cm2/s/A) between 
			  observed spectra and model spectra in ``fit_wl_range``.
			- ``logflux_residuals_spec``: (if ``fit spectra``) logarithm flux density residuals between observed 
			  spectra and model spectra in ``fit_wl_range``.
			- ``flux_syn_array_model_scaled_fit``: (if ``fit photometry``) input ``flux_syn_array_model_fit`` flux 
			  densities multiplied by the scaling factor ``scaling_fit``.
			- ``flux_residuals_phot``: (if ``fit photometry``) linear flux density residuals (in erg/cm2/s/A) between 
			  observed photometry and model spectra in ``fit_phot_range``.
			- ``logflux_residuals_phot``: (if ``fit photometry``) logarithm flux density residuals between observed 
			  photometry and model spectra in ``fit_phot_range``.

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
	>>> params_ranges = {'Teff': [700, 900], 'logg': [4.0, 5.0]}
	>>> my_model = seda.ModelOptions(model=model, model_dir=model_dir, 
	>>>                              params_ranges=params_ranges)
	>>> 
	>>> # load fitting options
	>>> fit_wl_range = np.array([value1, value2]) # to make the fit between value1 and value2
	>>> my_chi2 = seda.Chi2Options(my_data=my_data, my_model=my_model, 
	>>>                               fit_wl_range=fit_wl_range)
	>>> 
	>>> # run fit
	>>> out_chi2 = seda.chi2(my_chi2)
	    Weighted least-squares fitting ran successfully

	Author: Genaro Suárez
	'''

	ini_time_chi2 = time.time() # to estimate the time elapsed running chi2
	print('\n   Running weighted least-squares fitting...')

	# load input parameters
	# all are stored in my_chi2 but they were defined in different classes
	# from InputData
	fit_spectra = my_chi2.fit_spectra
	fit_photometry = my_chi2.fit_photometry
	wl_spectra = my_chi2.wl_spectra
	flux_spectra = my_chi2.flux_spectra
	eflux_spectra = my_chi2.eflux_spectra
	phot = my_chi2.phot
	ephot = my_chi2.ephot
	filters = my_chi2.filters
	res = my_chi2.res
	lam_res = my_chi2.lam_res
	distance = my_chi2.distance
	edistance = my_chi2.edistance
	# from ModelOptions
	model = my_chi2.model
	model_dir = my_chi2.model_dir
	wl_model = my_chi2.wl_model
	flux_model = my_chi2.flux_model
	params_ranges = my_chi2.params_ranges
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
	disp_wl_range = my_chi2.disp_wl_range
	N_model_spectra = my_chi2.N_model_spectra
	if (model is not None) & (model_dir is not None):
		spectra_name = my_chi2.spectra_name
		spectra_name_full = my_chi2.spectra_name_full
	if fit_spectra:
		N_spectra = my_chi2.N_spectra
		wl_spectra_min = my_chi2.wl_spectra_min
		wl_spectra_max = my_chi2.wl_spectra_max
		N_datapoints = my_chi2.N_datapoints
		wl_array_obs_fit = my_chi2.wl_array_obs_fit
		flux_array_obs_fit = my_chi2.flux_array_obs_fit
		eflux_array_obs_fit = my_chi2.eflux_array_obs_fit
		# make deepcopy of the following attributes to avoid modifying the original ones (stores in my_chi2) 
		# when updating the flux densities after applying the scaling factor
		wl_array_model_conv_resam = copy.deepcopy(my_chi2.wl_array_model_conv_resam)
		flux_array_model_conv_resam = copy.deepcopy(my_chi2.flux_array_model_conv_resam)
		wl_array_model_conv_resam_fit = copy.deepcopy(my_chi2.wl_array_model_conv_resam_fit)
		flux_array_model_conv_resam_fit = copy.deepcopy(my_chi2.flux_array_model_conv_resam_fit)
		weight_spec_fit = my_chi2.weight_spec_fit 
	if fit_photometry:
		flux_syn_array_model_fit = my_chi2.flux_syn_array_model_fit.copy() # make a copy because it will be scaled
		lambda_eff_array_model_fit = my_chi2.lambda_eff_array_model_fit
		width_eff_array_model_fit = my_chi2.width_eff_array_model_fit
		lambda_eff_SVO = my_chi2.lambda_eff_SVO
		width_eff_SVO = my_chi2.width_eff_SVO
		phot_fit = my_chi2.phot_fit
		ephot_fit = my_chi2.ephot_fit
		filters_fit = my_chi2.filters_fit
		lambda_eff_SVO_fit = my_chi2.lambda_eff_SVO_fit
		width_eff_SVO_fit = my_chi2.width_eff_SVO_fit
		weight_phot_fit = my_chi2.weight_phot_fit 

	# initialize variables to save key parameters from the fit
	scaling_fit = np.zeros(N_model_spectra) * np.nan
	escaling_fit = np.zeros(N_model_spectra) * np.nan
	chi2_fit = np.zeros(N_model_spectra) * np.nan
	chi2_red_fit = np.zeros(N_model_spectra) * np.nan
	chi2_wl_fit = []
	chi2_red_wl_fit = []

	if extinction_free_param:
		raise NotImplementedError('Extinction fitting is not currently supported.')

	# create a tqdm progress bar
	chi2_bar = tqdm(total=N_model_spectra, desc='Weighted least-squares fitting')
	for i in range(N_model_spectra): # for each model spectrum
		# update the progress bar
		chi2_bar.update(1)

		# prepare data for fitting
		if fit_spectra and not fit_photometry:
			data_fit = flux_array_obs_fit[i]
			edata_fit = eflux_array_obs_fit[i]
			model_fit = flux_array_model_conv_resam_fit[i]
			weight_fit = weight_spec_fit
		if not fit_spectra and fit_photometry:
			data_fit = [phot_fit]
			edata_fit = [ephot_fit]
			model_fit = [flux_syn_array_model_fit[i]]
			weight_fit = [weight_phot_fit]
		if fit_spectra and fit_photometry:
			data_fit = flux_array_obs_fit[i] + [phot_fit]
			edata_fit = eflux_array_obs_fit[i] + [ephot_fit]
			model_fit = flux_array_model_conv_resam_fit[i] + [flux_syn_array_model_fit[i]]
			weight_fit = weight_spec_fit + [weight_phot_fit]


		# calculate the optimal scaling factor analytically
		if scaling_free_param:
			# calculate the mean weight for normalization
			weight_norm = np.mean(np.concatenate(weight_fit))

			# calculate the numerator and denominator of the optimal scaling factor
			numerator = 0.0
			denominator = 0.0
			for data, edata, model_flux, weight in zip(data_fit, edata_fit, model_fit, weight_fit):
				numerator += np.sum(weight * data * model_flux / edata**2)
				denominator += np.sum(weight * model_flux**2 / edata**2)

			# check for invalid values before calculating the scaling factor
			if (not np.isfinite(numerator) or not np.isfinite(denominator)
					or denominator <= 0 or not np.isfinite(weight_norm) or weight_norm <= 0):
				raise ValueError(f'Invalid analytical scaling calculation for model spectrum {i}')

			# calculate the optimal scaling factor and its uncertainty
			scaling_best = numerator / denominator
			escaling_best = np.sqrt(weight_norm / denominator)
		else:
			# use the fixed scaling factor
			scaling_best = scaling
			escaling_best = np.nan

		# calculate the weighted least-squares statistic
		residuals = residuals_for_chi2(
			data_fit, edata_fit, model_fit, weight_fit, scaling_best
		)

		# save the best-fitting scaling factor
		scaling_fit[i] = scaling_best
		escaling_fit[i] = escaling_best

		nfree = len(residuals) - int(scaling_free_param)

		chi2_fit[i] = np.sum(residuals**2)
		chi2_red_fit[i] = chi2_fit[i] / nfree
		chi2_wl_fit.append(residuals**2)
		chi2_red_wl_fit.append(residuals**2 / nfree)

		# scale model flux densities by the value that minimizes the fit statistic
		if fit_spectra:
			for k in range(N_spectra): # for each input observed spectrum
				flux_array_model_conv_resam[i][k] = flux_array_model_conv_resam[i][k] * scaling_fit[i]
				flux_array_model_conv_resam_fit[i][k] = flux_array_model_conv_resam_fit[i][k] * scaling_fit[i]
		if fit_photometry:
			flux_syn_array_model_fit[i] = flux_syn_array_model_fit[i] * scaling_fit[i]

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
	if (model is not None) & (model_dir is not None):
		out_chi2.update({'spectra_name_full': spectra_name_full, 'spectra_name': spectra_name})
	out_chi2.update({'N_model_spectra': N_model_spectra, 'scaling_fit': scaling_fit, 
		             'escaling_fit': escaling_fit, 'chi2_wl_fit': chi2_wl_fit, 
		             'chi2_red_wl_fit': chi2_red_wl_fit, 'chi2_fit': chi2_fit, 
		             'chi2_red_fit': chi2_red_fit})#, 'weight_fit': weight_fit})

	# add more output parameters depending on the input information
	if fit_spectra: # when only spectra are used in the fit
		# resampled and convolved model spectra in the input spectra wavelength ranges
#		out_chi2['wl_array_model_conv_resam'] = wl_array_model_conv_resam
		out_chi2['flux_array_model_conv_resam_scaled'] = flux_array_model_conv_resam
		# resampled and convolved model spectra within the fit ranges
#		out_chi2['wl_array_model_conv_resam_fit'] = wl_array_model_conv_resam_fit
		out_chi2['flux_array_model_conv_resam_scaled_fit'] = flux_array_model_conv_resam_fit
#		# observed flux densities within the fit ranges
#		out_chi2['wl_array_obs_fit'] = wl_array_obs_fit
#		out_chi2['flux_array_obs_fit'] = flux_array_obs_fit
#		out_chi2['eflux_array_obs_fit'] = eflux_array_obs_fit
	if fit_photometry: # when photometry is used in the fit
#		# photometric magnitudes within the wavelength range for the fit
#		out_chi2['phot_fit'] = phot_fit
#		out_chi2['ephot_fit'] = ephot_fit
#		out_chi2['filters_fit'] = filters_fit
#		# information from SVO for the selected filters
#		out_chi2['lambda_eff_SVO_fit'] = lambda_eff_SVO_fit
#		out_chi2['width_eff_SVO_fit'] = width_eff_SVO_fit
#		# information from model spectra for the selected filters
#		out_chi2['lambda_eff_array_model_fit'] = lambda_eff_array_model_fit
#		out_chi2['width_eff_array_model_fit'] = width_eff_array_model_fit
		# synthetic photometry from each model for the selected filters
		out_chi2['flux_syn_array_model_scaled_fit'] = flux_syn_array_model_fit
	if distance!=None: # when a radius was obtained
		out_chi2['radius'] = radius
		if edistance!=None:
			out_chi2['eradius'] = eradius

	# obtain residuals in linear and logarithmic-scale for flux densities in the fit range
	flux_residuals_spec = []
	logflux_residuals_spec = []
	flux_residuals_phot = []
	logflux_residuals_phot = []
	for i in range(N_model_spectra): # for each model spectrum
		flux_residuals_each = []
		logflux_residuals_each = []
		if fit_spectra:
			for k in range(N_spectra): # for each input observed spectrum
				# linear scale
				res_lin = flux_array_model_conv_resam_fit[i][k] - flux_array_obs_fit[i][k]
				flux_residuals_each.append(res_lin)
				# log scale
				mask_pos = flux_array_obs_fit[i][k]>0 # mask to avoid negative input flux densities to obtain the logarithm
				res_log = np.log10(flux_array_model_conv_resam_fit[i][k][mask_pos]) - np.log10(flux_array_obs_fit[i][k][mask_pos])
				logflux_residuals_each.append(res_log)
			# nested list with all resampled and convolved model spectra in the fit range
			flux_residuals_spec.append(flux_residuals_each)
			logflux_residuals_spec.append(logflux_residuals_each)
		if fit_photometry: 
			# linear scale
			res_lin = flux_syn_array_model_fit[i] - phot_fit
			# log scale
			mask_pos = phot_fit>0 # mask to avoid negative input flux densities to obtain the logarithm
			res_log = np.log10(flux_syn_array_model_fit[i][mask_pos]) - np.log10(phot_fit[mask_pos])
			# nested list with residuals for filters within the fit range for each model spectrum
			flux_residuals_phot.append(res_lin)
			logflux_residuals_phot.append(res_log)

	if fit_spectra:
		out_chi2['flux_residuals_spec'] = flux_residuals_spec
		out_chi2['logflux_residuals_spec'] = logflux_residuals_spec
	if fit_photometry: 
		out_chi2['flux_residuals_phot'] = flux_residuals_phot
		out_chi2['logflux_residuals_phot'] = logflux_residuals_phot

	# separate physical parameters from each model spectrum name
	if (model is not None) & (model_dir is not None):
		out_separate_params = models.separate_params(model=model, spectra_name=spectra_name)
		# add model spectra parameters to the output dictionary
		out_chi2.update(out_separate_params)

	# save some information
	if (model is not None) & (model_dir is not None):
		if save_results: # save table with model spectra sorted by chi square
			# save output dictionary as pickle
			with open(my_chi2.chi2_pickle_file, 'wb') as file:
				# serialize and write the variable to the file
				pickle.dump(out_chi2, file)
			print('      weighted least-squares fitting results saved successfully')

			# save table with model spectra names sorted by chi square along with the parameters from each spectrum
			out_save_params = save_params(out_chi2=out_chi2)

	print('\n   Weighted least-squares fitting ran successfully')
	fin_time_chi2 = time.time()
	utils.print_time(fin_time_chi2-ini_time_chi2)

	return out_chi2

##########################
def residuals_for_chi2(data, edata, model, weight, scaling):

	# calculate the mean weight for normalization
	weight_norm = np.mean(np.concatenate(weight))

	# calculate the weighted residuals for each dataset
	residuals = np.array([])
	for i in range(len(data)):
		# calculate the residuals
		res = (data[i] - scaling*model[i]) / edata[i]
		# normalize the additional weights
		wnorm = weight[i] / weight_norm
		# apply the additional weights
		res = np.sqrt(wnorm) * res
		residuals = np.concatenate([residuals, res])

	return residuals

##########################
def save_params(out_chi2):
	'''
	Description:
	------------
		Create a table of model spectra sorted by the weighted least-squares statistic, 
		along with relevant parameters.

	Parameters:
	-----------
	- out_chi2 : dictionary
		Dictionary with the following parameters:
			- ``spectra_name`` : spectrum name.
			- ``chi2_fit`` : fit statistic.
			- ``chi2_red_fit`` : reduced fit statistic.
			- ``scaling_fit`` : scaling factor.
			- ``escaling_fit`` : scaling factor uncertainty.
			- ``params`` : model grid parameters.
			- ``R`` : radius, when a ``distance`` was provided.
			- ``eR`` : radius error, when ``distance`` and ``edistance`` were provided.

	Returns:
	--------
	- ``chi2_table_file`` : ascii table
		ASCII table with input parameters sorted according to reduced fit statistic.
		Table named ``chi2_table_file``, as specified in :class:`~seda.Chi2Options`.

	Author: Genaro Suárez
	'''

	# sort index with respect to reduced fit statistic
	ind = np.argsort(out_chi2['chi2_red_fit'])

	# read and sort relevant parameters from the input dictionary
	chi2 = out_chi2['chi2_fit'][ind] # chi-square as integer
	chi2_red = out_chi2['chi2_red_fit'][ind] # reduced chi-square with three decimals
	scaling_factor = out_chi2['scaling_fit'][ind]
	escaling_factor = out_chi2['escaling_fit'][ind]
	if 'spectra_name' in out_chi2:
		filename = out_chi2['spectra_name'][ind] # filename
		params = {}
		for param, values in out_chi2['params'].items():
			params[param] = values[ind]
	if out_chi2['my_chi2'].distance is not None:
		R = out_chi2['radius'][ind] # radius
		if out_chi2['my_chi2'].edistance is not None:
			eR = out_chi2['eradius'][ind] # radius error

	# make dictionary with parameters of interest
	my_dict = {} # initialize dictionary
	if 'spectra_name' in out_chi2:
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
	if 'spectra_name' in out_chi2:
		my_dict.update(params) # free model paramters
	if out_chi2['my_chi2'].distance is not None:
		my_dict['R'] = np.round(R,3) # radius
		if out_chi2['my_chi2'].edistance is not None:
			my_dict['eR'] = np.round(eR,3) # radius error

	# save dictionary as a PrettyTable table
	table_name = out_chi2['my_chi2'].chi2_table_file
	utils.save_prettytable(my_dict=my_dict, table_name=table_name)

	return  

##########################
# number of elements in a nested list
def count_elements_nested_list(lst):
	count = 0
	for i in range(len(lst)):
		for j in range(len(lst[i])):
			count += len(lst[i][j])

	return count
