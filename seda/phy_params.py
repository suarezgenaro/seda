import numpy as np
from astropy import units as u
from astropy.constants import L_sun
from .synthetic_photometry.synthetic_photometry import *
from .input_parameters import *
from .chi2_fit import *
from sys import exit

##########################
def bol_lum(output_fit=None, wl_spectra=None, flux_spectra=None, eflux_spectra=None, distance=None, edistance=None, 
	        flux_unit=None, wl_model=None, flux_model=None, params=None, scale_model=True, convolve_model=True, 
	        res=None, lam_res=None, complement_SED=True):
	'''
	Description:
	------------
		Calculate bolometric luminosity by integrated the input SED complemented with the best model fit.

	Parameters:
	-----------
	- output_fit : dictionary or str, optional (required if no input model spectrum is provided)
		Output dictionary with the results from ``chi2`` or ``bayes``.
		It can be either the name of the pickle file or simply the output dictionary.
	- wl_spectra : float array or list, optional (require if `output_chi2` is not provided)
		Wavelength in micron of the spectra to construct an SED.
		For multiple spectra, provide them as a list (e.g., ``wl_spectra = [wl_spectrum1, wl_spectrum2]``).
	- flux_spectra : float array or list, optional (require if `output_chi2` is not provided)
		Fluxes for the input spectra in units indicated by ``flux_unit``.
		Use a list for multiple spectra (similar to ``wl_spectra``).
	- eflux_spectra : float array or list, optional (require if `output_chi2` is not provided)
		Fluxes uncertainties in units indicated by ``flux_unit``.
		Use a list for multiple spectra (similar to ``wl_spectra``).
	- flux_unit : str, optional (default ``'erg/s/cm2/A'``)
		Units of ``flux``: ``'Jy'``, ``'erg/s/cm2/A'``, or ``erg/s/cm2/um``.
	- distance : float, optional (require if `output_chi2` is not provided)
		Target distance (in pc) used to obtain luminosity from total flux.
	- edistance : float, optional (require if `output_chi2` is not provided)
		Distance error (in pc).
	- wl_model : array, optional (required if `model_dir` is not provided)
		Wavelengths in micron of model spectrum to complement the input observed SED.
	- flux_model : array, optional (required if `model_dir` is not provided)
		Fluxes in erg/s/cm2/A of model spectrum to complement the input observed SED.
	- params : dictionary, optional (require if `output_chi2` is not provided)
		Value for each free parameter for the model spectrum used in the hybrid SED.
	- scale_model : {``True``, ``False``}, optional (default ``True``)
		Label to indicate if the input model spectrum needs to be scaled (``True``) by minimizing chi-square or it was already scaled (``False``).
	- convolve_model : {``True``, ``False``}, optional (default ``True``)
		Label to indicate if the input model spectrum needs (``True``) or does not need (``False``) to be convolved.
	- res : float, list or array, optional (required if ``convolve_model``)
		Resolving power (R=lambda/delta(lambda) at ``lam_res``) of input spectra to smooth model spectra.
		For multiple input spectra, ``res`` should be a list or array with a value for each spectrum.
	- lam_res : float, list or array, optional
		Wavelength of reference at which ``res`` is given (because resolution may change with wavelength).
		For multiple input spectra, ``lam_res`` should be a list or array with a value for each spectrum.
		Default is the integer closest to the median wavelength for each input spectrum.
		If lam_res is provided, the values are also rounded to the nearest integer.
		This will facilitate managing (saving and reading) convolved model spectra in ``seda.ModelOptions``.
	- complement_SED : {``True``, ``False``}, optional (default ``True``)
		Label indicating if the input spectra will (``True``) or will not (``False``) be complemented with a model spectrum.

	Returns:
	--------
	Dictionary with derived parameters:
		- ``'flux_tot'`` : total flux (in erg/s/cm2/A) by integrating the hybrid SED.
		- ``'eflux_tot'`` : uncertainty (in erg/s/cm2/A) associated to total flux.
		- ``'Lbol_tot'`` : bolometric luminosity (in Lsun) from the total flux.
		- ``'eLbol_tot'`` : bolometric luminosity uncertainty (in Lsun).
		- ``'logLbol_tot'`` : logarithmic bolometric luminosity.
		- ``'elogLbol_tot'`` : logarithmic bolometric luminosity uncertainty.
		- ``'flux_tot_obs'`` : total flux (in erg/s/cm2/A) by integrating the observed SED (if complement_SED).
		- ``'eflux_tot_obs'`` : uncertainty (in erg/s/cm2/A) associated to the observed flux (if complement_SED).
		- ``'Lbol_tot_obs'`` : bolometric luminosity (in Lsun) from the observed flux (if complement_SED).
		- ``'eLbol_tot_obs'`` : bolometric luminosity uncertainty (in Lsun) from the observed flux (if complement_SED).
		- ``'logLbol_tot_obs'`` : logarithmic bolometric luminosity from the observed flux (if complement_SED).
		- ``'elogLbol_tot_obs'`` : logarithmic bolometric luminosity uncertainty from the observed flux (if complement_SED).
		- ``'contribution_percentage'`` : contribution (in percentage) of each input spectrum to the total flux or luminosity (if complement_SED).
		- ``'contribution_percentage_obs'`` : contribution (in percentage) of each input spectrum to the observed flux or luminosity (if complement_SED).
		- ``'N_spectra'`` : number of input spectra (if complement_SED).
		- ``'completeness_obs'`` : completeness of the observed SED with respect to the hybrid SED (if complement_SED).
		- ``'wl_SED'`` : wavelengths in micron of the hybrid SED (if complement_SED).
		- ``'flux_SED'`` : fluxes in erg/s/cm2/A the hybrid SED (if complement_SED).
		- ``'eflux_SED'`` : fluxes uncertainties in erg/s/cm2/A the hybrid SED (if complement_SED).
		- ``'params'`` : dictionary with free parameter values for the model spectrum used in the hybrid SED (if complement_SED).
		- ``'wl_spectra'`` : input `wl_spectra` (if complement_SED).
		- ``'flux_spectra'`` : input `flux_spectra` (if complement_SED).
		- ``'eflux_spectra'`` : input `eflux_spectra` (if complement_SED).
		- ``'wl_model'`` : input `wl_model` (if complement_SED).
		- ``'flux_model'`` : input `flux_model` (if complement_SED).

	Author: Genaro Suárez

	Date: 2025-04-20
	'''

	if complement_SED: # if the SED will be complemented with a model spectrum
		# verify that necessary input parameters are provided
		if (wl_spectra is None) & (output_fit is None):
			raise Exception(f'"wl_spectra" or "output_fit" must be provided')
		if (distance is None) & (output_fit is None):
			raise Exception(f'"distance" or "output_fit" must be provided')
	
		if output_fit is not None: # output_fit is provided
			# if results from the fits are provided as a pickle file, open it
			try: # if given as a pickle file
				with open(output_fit, 'rb') as file:
					output_fit = pickle.load(file)
			except: # if given as the output of chi2_fit
				pass
	
			# open results from the chi square analysis
			try:
				output_fit['my_chi2']
			except:
				pass
			else:
				# extract relevant parameters
				wl_spectra = output_fit['my_chi2'].wl_spectra # micron
				flux_spectra = output_fit['my_chi2'].flux_spectra # erg/s/cm2/A
				eflux_spectra = output_fit['my_chi2'].eflux_spectra # erg/s/cm2/A
				distance = output_fit['my_chi2'].distance # pc
				edistance = output_fit['my_chi2'].edistance # pc
				model = output_fit['my_chi2'].model # pc
				model_dir = output_fit['my_chi2'].model_dir # pc
				N_spectra = output_fit['my_chi2'].N_spectra # pc
				res = output_fit['my_chi2'].res # pc
				lam_res = output_fit['my_chi2'].lam_res # pc
		
				# read the entire best fit model spectrum (the one stored in 
				# output_fit was trimmed to the wavelength range of the data)
				output_best_chi2_fits = best_chi2_fits(output_chi2=output_fit, N_best_fits=1, 
				                                       model_dir_ori=model_dir, ori_res=True)
				wl_model = output_best_chi2_fits['wl_model_best'][0] # um
				flux_model = output_best_chi2_fits['flux_model_best'][0] # erg/s/cm2/A scaled to match the input spectra
				spectra_name_best = output_best_chi2_fits['spectra_name_best'][0]
				params = separate_params(model=model, spectra_name=spectra_name_best)['params']
				# convert each parameter into float instead of array, as it contains only parameters for the best fit
				for param in params:
					params[param] = params[param][0]
	
			# if the fit was done using a Bayesian sampling
			try:
				output_fit['my_bayes']
			except:
				pass
			else:
				# extract relevant parameters
				wl_spectra = output_fit['my_bayes'].wl_spectra # micron
				flux_spectra = output_fit['my_bayes'].flux_spectra # erg/s/cm2/A
				eflux_spectra = output_fit['my_bayes'].eflux_spectra # erg/s/cm2/A
				distance = output_fit['my_bayes'].distance # pc
				edistance = output_fit['my_bayes'].edistance # pc
				model = output_fit['my_bayes'].model # pc
				model_dir = output_fit['my_bayes'].model_dir # pc
				N_spectra = output_fit['my_bayes'].N_spectra # pc
				res = output_fit['my_bayes'].res # pc
				lam_res = output_fit['my_bayes'].lam_res # pc
	
				# read the entire best fit model spectrum (the one stored in output_fit 
				# was trimmed to the wavelength range of the data)
				output_best_bayesian_fit = best_bayesian_fit(output_bayes=output_fit, 
				                                             model_dir_ori=model_dir, ori_res=True)
				wl_model = output_best_bayesian_fit['wl_model_ori'] # um
				flux_model = output_best_bayesian_fit['flux_model_ori'] # erg/cm2/s/A
				params = output_best_bayesian_fit['params_med']
	
		else: # no output_fit is provided
			# handle input data
			my_data = InputData(wl_spectra=wl_spectra, flux_spectra=flux_spectra, eflux_spectra=eflux_spectra, 
			                    flux_unit=flux_unit, res=res, distance=distance, edistance=edistance)
			N_spectra = my_data.N_spectra
			wl_spectra = my_data.wl_spectra # um
			flux_spectra = my_data.flux_spectra # erg/cm2/s/A
			eflux_spectra = my_data.eflux_spectra # erg/cm2/s/A
			res = my_data.res # um
			lam_res = my_data.lam_res # um

		# integrate the input SED
		flux_each = np.zeros(N_spectra)
		eflux_each = np.zeros(N_spectra)
		for i in range(N_spectra):
			# total flux
			flux_each[i] = np.trapz(flux_spectra[i], 1.e4*wl_spectra[i]) # erg/s/cm2
			eflux_each[i] = np.median(eflux_spectra[i]/flux_spectra[i]) * flux_each[i] # keep fractional errors
	
		# total flux and total luminosity
		flux_tot_obs = sum(flux_each)
		eflux_tot_obs = np.sqrt(sum(eflux_each**2))
	
		# luminosity in erg/s
		Lbol_erg_s_obs = 4.*np.pi*((distance*u.pc).to(u.cm).value)**2 * flux_tot_obs # erg/s
		eLbol_erg_s_obs = np.sqrt((2*edistance/distance)**2 + (eflux_tot_obs/flux_tot_obs)**2) * Lbol_erg_s_obs
		# luminosity in Lsun
		Lbol_tot_obs = Lbol_erg_s_obs / (L_sun.to(u.erg/u.s).value) # in Lsun
		eLbol_tot_obs = (eLbol_erg_s_obs/Lbol_erg_s_obs) * Lbol_tot_obs
		# logLbol
		logLbol_tot_obs = np.log10(Lbol_tot_obs)
		elogLbol_tot_obs = eLbol_tot_obs/(Lbol_tot_obs*np.log(10))
	
	
		# complement SED with the input model spectrum
		if (wl_model is not None) & (flux_model is not None):
			if scale_model: # scale model fluxes to minimize the chi-square statistics
				# find scaling factor by running the chi-square minimization
				my_data = InputData(wl_spectra=wl_spectra, flux_spectra=flux_spectra, 
				                    eflux_spectra=eflux_spectra, flux_unit=flux_unit, 
				                    res=res, distance=distance, edistance=edistance)
				my_model = ModelOptions(wl_model=wl_model, flux_model=flux_model)
				my_chi2 = Chi2Options(my_data=my_data, my_model=my_model)
				out_chi2 = chi2(my_chi2=my_chi2)
	
				# scale model fluxes
				flux_model = out_chi2['scaling_fit']*flux_model
	
				# convolve scaled model, if requested
				# consider the lower resolution from the input observed spectra to convolve the entire model spectrum
				mask = np.array(res)==min(res)
				res_min = np.array(res)[mask][0]
				lam_res_min = np.array(lam_res)[mask][0]
				if convolve_model:
					lam_res = out_chi2['my_chi2'].lam_res
					out_convolve_spectrum = convolve_spectrum(wl=wl_model, flux=flux_model, lam_res=lam_res_min, res=res_min) 
					flux_model = out_convolve_spectrum['flux_conv']
	
			N_spectra = my_data.N_spectra
	
			# complement observed SED with scaled model
			# sort input spectra according to their minimum values
			wl_spectra_sort, flux_spectra_sort, eflux_spectra_sort = sort_nested_list(wl_spectra, flux_spectra, eflux_spectra)
	
			# obtain median wavelength dispersion of each input spectrum to be used to decide of there is a wavelength gap between input spectra
			wl_disp = np.zeros(N_spectra)
			for i in range(N_spectra): # for each input spectrum
				wl_disp[i] = np.median(wl_spectra_sort[i][1:]-wl_spectra_sort[i][:-1])
	
			# full SED
			for i in range(N_spectra): # for each input spectrum
				# complement wavelength shorter than the minimum wavelength in the input SED plus first input spectrum
				if i==0:
					# complement wavelength shorter than the minimum wavelength in the input SED
					mask = wl_model<min(wl_spectra_sort[i])
					wl_SED = wl_model[mask]
					flux_SED = flux_model[mask]
					eflux_SED = np.repeat(np.nan, len(wl_model[mask]))
	
					# add first input spectrum
					wl_SED = np.concatenate((wl_SED, wl_spectra_sort[i]))
					flux_SED = np.concatenate((flux_SED, flux_spectra_sort[i]))
					eflux_SED = np.concatenate((eflux_SED, eflux_spectra_sort[i]))
	
				# complement gaps within the data and add intermediate and last input spectra
				else:
					# complement wavelengths in between observed spectra, if needed
					wl_disp_threshold = 2 # threshold to identify gaps based on N-times the median wavelength dispersion
					wl_max_previous = max(wl_spectra_sort[i-1]+wl_disp_threshold*wl_disp[i-1])
					wl_min_current = min(wl_spectra_sort[i]-wl_disp_threshold*wl_disp[i])
					if wl_max_previous<wl_min_current: # there is a gap between the i and i-1 spectra
						print(f'   Gap detected between input spectra #{i-1} and #{i}')
						mask = (wl_model>max(wl_spectra_sort[i-1])) & (wl_model<min(wl_spectra_sort[i]))
						wl_SED = np.concatenate((wl_SED, wl_model[mask]))
						flux_SED = np.concatenate((flux_SED, flux_model[mask]))
						eflux_SED = np.concatenate((eflux_SED, np.repeat(np.nan, len(wl_model[mask]))))
	
					# input corresponding input spectrum
					wl_SED = np.concatenate((wl_SED, wl_spectra_sort[i]))
					flux_SED = np.concatenate((flux_SED, flux_spectra_sort[i]))
					eflux_SED = np.concatenate((eflux_SED, eflux_spectra_sort[i]))
	
				# complement wavelength longer than the maximum wavelength in the input SED
				if i==N_spectra-1:
					mask = wl_model>max(wl_spectra_sort[i])
					wl_SED = np.concatenate((wl_SED, wl_model[mask]))
					flux_SED = np.concatenate((flux_SED, flux_model[mask]))
					eflux_SED = np.concatenate((eflux_SED, np.repeat(np.nan, len(wl_model[mask]))))

	else: # do not complement the SED with a model
		wl_SED = wl_spectra
		flux_SED = flux_spectra
		eflux_SED = eflux_spectra

	# Lbol from the full SED
	# total flux from
	flux_tot = np.trapz(flux_SED, (wl_SED*u.um).to((u.nm*0.1)).value) # erg/s/cm2
	mask = ~np.isnan(eflux_SED)
	eflux_tot = np.median(eflux_SED[mask]/flux_SED[mask]) * flux_tot # keep fractional errors (errors from the spectrum with the most data points will dominate, as no error is associated to the model)
	#eflux_tot = np.sqrt(sum(eflux_each**2)) # (fractional errors from each input spectrum will have its contribution)
	# luminosity in erg/s
	Lbol_erg_s = 4.*np.pi*((distance*u.pc).to(u.cm).value)**2 * flux_tot # erg/s
	eLbol_erg_s = np.sqrt((2*edistance/distance)**2 + (eflux_tot/flux_tot)**2) * Lbol_erg_s
	# luminosity in Lsun
	Lbol_tot = Lbol_erg_s / (L_sun.to(u.erg/u.s).value) # in Lsun
	eLbol_tot = (eLbol_erg_s/Lbol_erg_s) * Lbol_tot
	# logLbol
	logLbol_tot = np.log10(Lbol_tot)
	elogLbol_tot = eLbol_tot/(Lbol_tot*np.log(10))

	# print Lbol
	print('\nlog(Lbol) = {:.3f}'.format(round(logLbol_tot,3))+'\pm'+'{:.3f}'.format(round(elogLbol_tot,3)))

	if complement_SED: # if the SED will be complemented with a model spectrum
		# fraction of the hybrid SED covered by the observations
		completeness = 100*flux_tot_obs/flux_tot
		print(f'\nThe observed SED is {round(completeness,1)}% complete')

		# contribution of each input spectrum to the total observed SED (in flux or luminosity)')
		contribution_obs = np.zeros(N_spectra)
		print('\nContribution to the total observed SED (in flux or luminosity)')
		for i in range(N_spectra):
			contribution_obs[i] = 100.*flux_each[i]/flux_tot_obs
			print(f'   spectrum #{i}: {round(contribution_obs[i],1)}%')

		# contribution of each input spectrum to the total hybrid SED (in flux or luminosity)')
		contribution = np.zeros(N_spectra)
		print('Contribution to the total hybrid full SED (in flux or luminosity)')
		for i in range(N_spectra):
			contribution[i] = 100.*flux_each[i]/flux_tot
			print(f'   spectrum #{i}: {round(contribution[i],1)}%')

	# output dictionary
	out = {'flux_tot': flux_tot, 'eflux_tot': eflux_tot, 'Lbol_tot': Lbol_tot, 'eLbol_tot': eLbol_tot, 
	       'logLbol_tot': logLbol_tot, 'elogLbol_tot': elogLbol_tot}

	if complement_SED: # if the SED will be complemented with a model spectrum
		out.update({'flux_tot_obs': flux_tot_obs, 'eflux_tot_obs': eflux_tot_obs, 'Lbol_tot_obs': Lbol_tot_obs, 
		                 'eLbol_tot_obs': eLbol_tot_obs, 'logLbol_tot_obs': logLbol_tot_obs, 'elogLbol_tot_obs': elogLbol_tot_obs, 
		                 'contribution_percentage': contribution, 'contribution_percentage_obs': contribution_obs,
		                 'wl_SED': wl_SED, 'flux_SED': flux_SED, 'eflux_SED': eflux_SED, 'params': params, 
		                 'wl_spectra': wl_spectra, 'flux_spectra': flux_spectra, 'eflux_spectra': eflux_spectra,
		                 'wl_model': wl_model, 'flux_model': flux_model,
		                 'N_spectra': N_spectra, 'completeness_obs': completeness})

	return out

##################
# function to sort input spectra as nested lists according to their minimum wavelength values
def sort_nested_list(wl_spectra, flux_spectra, eflux_spectra):

	# minimum wavelength of each input spectrum
	min_vals = np.zeros(len(wl_spectra))
	for i in range(len(wl_spectra)):
	    min_vals[i] = min(wl_spectra[i])

	wl_spectra = [wl_spectra[i] for i in np.argsort(min_vals)]
	flux_spectra = [flux_spectra[i] for i in np.argsort(min_vals)]
	eflux_spectra = [eflux_spectra[i] for i in np.argsort(min_vals)]

	return wl_spectra, flux_spectra, eflux_spectra
