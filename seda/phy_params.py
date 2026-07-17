import numpy as np
import pickle
from astropy import units as u
from astropy.constants import L_sun, sigma_sb, R_jup, R_sun
from scipy.interpolate import LinearNDInterpolator
from .synthetic_photometry import synthetic_photometry
from . import input_parameters
from . import chi2_fit 
from . import models
from . import utils
from .empirical_aux._loaders import reference_color as _reference_color
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
			# open dictionary if need it
			output_fit = utils.load_output_fit(output_fit)

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
				output_best_chi2_fits = utils.best_chi2_fits(output_chi2=output_fit, N_best_fits=1, 
				                                             model_dir_ori=model_dir, ori_res=True)
				wl_model = output_best_chi2_fits['wl_model_best'][0] # um
				flux_model = output_best_chi2_fits['flux_model_best'][0] # erg/s/cm2/A scaled to match the input spectra
				spectra_name_best = output_best_chi2_fits['spectra_name_best'][0]
				params = models.separate_params(model=model, spectra_name=spectra_name_best)['params']
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
			my_data = input_parameters.InputData(wl_spectra=wl_spectra, flux_spectra=flux_spectra, eflux_spectra=eflux_spectra, 
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
			flux_each[i] = utils.np_trapz(flux_spectra[i], 1.e4*wl_spectra[i]) # erg/s/cm2
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
				my_data = input_parameters.InputData(wl_spectra=wl_spectra, flux_spectra=flux_spectra, 
				                                     eflux_spectra=eflux_spectra, flux_unit=flux_unit, 
				                                     res=res, distance=distance, edistance=edistance)
				my_model = input_parameters.ModelOptions(wl_model=wl_model, flux_model=flux_model)
				my_chi2 = input_parameters.Chi2Options(my_data=my_data, my_model=my_model)
				out_chi2 = chi2_fit.chi2(my_chi2=my_chi2)
	
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
					wl_disp_threshold = 3 # threshold to identify gaps based on N-times the median wavelength dispersion
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
	flux_tot = utils.np_trapz(flux_SED, (wl_SED*u.um).to((u.nm*0.1)).value) # erg/s/cm2
	mask = ~np.isnan(eflux_SED)
	eflux_tot = np.median(eflux_SED[mask]/flux_SED[mask]) * flux_tot # keep fractional errors (errors from the spectrum with the most data points will dominate, as no error is associated to the model)
	#eflux_tot = np.sqrt(sum(eflux_each**2)) # (fractional errors from each input spectrum will have its contribution)

#	wl_int = (wl_SED * u.um).to(u.AA).value
#	dwl = np.gradient(wl_int)
#	mask = ~np.isnan(eflux_SED)
#	eflux_tot = np.sqrt(np.sum((eflux_SED[mask] * dwl[mask])**2))

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

##########################
def teff(Lbol, eLbol, R, eR, n_mc=10000, central="median", 
	     error="percentile", percentiles=(16, 84)):
	'''
	Description:
	------------
		Calculate effective temperature using the Stefan–Boltzmann 
		law considering a known bolometric luminosity and radius.

	Parameters:
	-----------
	- Lbol : float
		Bolometric luminosity in units of L_sun.
	- eLbol : float
		Uncertainty in luminosity (L_sun).
	- R : float
		Radius in units of R_jup.
	- eR : float
		Uncertainty in radius (R_jup).
	- n_mc : int, optional (default 10000)
		Number of Monte Carlo samples for uncertainties.
	- central : str, optional (default "median")
		"mean" or "median" for central value.
	- error : str, optional (default "percentile")
		"std" or "percentile".
	- percentiles : tuple or list, optional (default [16, 84])
		Lower and upper percentiles for uncertainty.

	Returns:
	--------
	- Teff : float
		Effective temperature in K.
	- eTeff : float or tuple
		- Effective temperature uncertainty in K.
		- If error="std": error is a scalar
		- If error="percentile": error is a tuple (lower_err, upper_err)

	Example:
	--------
	>>> import seda
	>>>
	>>> # input parameters
	>>> Lbol, eLbol = 6.324e-5, 6.978e-6 # in Lsun
	>>> R, eR = 1.018, 0.059 # in Rjup
	>>>
	>>> # derive Teff (in K) from Stefan–Boltzmann law
	>>> seda.phy_params.teff(Lbol=Lbol, eLbol=eLbol, R=R, eR=eR)
	    (1592.0020910445828, (57.98628122105015, 65.18365052510921))


	Author: Genaro Suárez

	Date: 2026-05-25
	'''
   
	# ensure percentiles is a tuple
	percentiles = tuple((18, 84))
	
	# verify "central" and "error" are valid parameters
	central_valid = ["mean", "median"]
	if central not in central_valid:
		raise ValueError(
			f"central={central!r} is not recognized. "
			f"Valid options: {central_valid}."
		)
	error_value = ["std", "percentile"]
	if error not in error_value:
		raise ValueError(
			f"error={error!r} is not recognized. "
			f"Valid options: {error_value}."
		)
	
	# input values with units
	Lbol = Lbol * L_sun
	eLbol = eLbol * L_sun

	R = R * R_jup
	eR = eR * R_jup

	# deterministic Teff (not returned, but useful sanity check)
	Teff_det = (Lbol / (4 * np.pi * sigma_sb * R**2))**0.25
	Teff_det = Teff_det.to(u.K)

	# Monte Carlo simulation for Teff error
	# sample L and R values from normal distribution
	# peaking at the input values and standard deviation
	# equal to the input uncertainties.
	L_samples = np.random.normal(Lbol.value, eLbol.value, n_mc) * Lbol.unit
	R_samples = np.random.normal(R.value, eR.value, n_mc) * R.unit

	Teff_samples = (L_samples / (4 * np.pi * sigma_sb * R_samples**2))**0.25
	Teff_samples = Teff_samples.to(u.K).value

	# Teff from MC
	# remove nan values, if any
	mask_nonan = ~np.isnan(Teff_samples)
	if not all(mask_nonan):
		print(f'{len(Teff_samples[~mask_nonan])}/{n_mc} Teff values from MC are NaN')

	if central == "mean":
		Teff_val = np.mean(Teff_samples[mask_nonan])
	elif central == "median":
		Teff_val = np.median(Teff_samples[mask_nonan])
	 
	# Teff uncertainty
	if error == "std":
		Teff_err = np.std(Teff_samples[mask_nonan])

	elif error == "percentile":
		p_lo, p_hi = np.percentile(Teff_samples[mask_nonan], percentiles)
		Teff_err = (Teff_val - p_lo, p_hi - Teff_val)

	return Teff_val, Teff_err
##########################
def inclination(vsini, evsini, P, eP, R, eR, n_mc=10000, central="median",
                error="percentile", percentiles=(16, 84)):
    '''
    Description:
    ------------
        Calculate the inclination angle from an objects
        projected rotation velocity (v sin i), rotation period (P),
        and radius (R), using:
 
            sin i = (P * vsini) / (2 * pi * R)
 
        Uncertainty is propagated via Monte Carlo sampling.
 
    Parameters:
    -----------
    - vsini : float
        Projected rotation velocity in units of km/s.
    - evsini : float
        Uncertainty in vsini (km/s).
    - P : float
        Rotation period in hours.
    - eP : float
        Uncertainty in the rotation period (hours).
    - R : float
        Radius in units of R_jup.
    - eR : float
        Uncertainty in radius (R_jup).
    - n_mc : int, optional (default 10000)
        Number of Monte Carlo samples for uncertainty propagation.
    - central : str, optional (default "median")
        "mean" or "median" for the central value of the inclination.
    - error : str, optional (default "percentile")
        "std" for symmetric standard deviation, or "percentile" for
        asymmetric percentile-based errors (recommended, since the
        posterior for i is generically asymmetric near 90 deg).
    - percentiles : tuple or list, optional (default (16, 84))
        Lower and upper percentiles used when error="percentile".
 
    Returns:
    --------
    - inc : float
        Inclination angle in degrees.
    - einc : float or tuple
        Inclination uncertainty in degrees.
        - If error="std": scalar symmetric uncertainty.
        - If error="percentile": tuple (lower_err, upper_err).
 
    Notes:
    ------
    - MC samples with sin i > 1 are unphysical (vsini exceeds v_eq for those
      draws). Following Vos et al. (2017), these are set to sin i = 1 (i = 90 deg)
      rather than discarded. The number of clipped samples is reported.
    - Samples with NaN (e.g. from non-physical period draws) are excluded from
      the statistics; the number of samples estimated vs used is printed.
    - The inclination posterior is bounded at 90 deg (sin i <= 1), so
      error="percentile" is strongly preferred over "std" when vsini is
      close to 2*pi*R/P.
 
    Example:
    --------
    >>> import seda
    >>>
    >>> # 2MASS J03552337+1133437 (Suárez et al. 2023, Section 2.2)
    >>> vsini, evsini = 12.31, 0.15  # km/s (Blake et al. 2010)
    >>> P, eP       =  9.53, 0.19   # hours (Vos et al. 2022)
    >>> R, eR       =  1.22, 0.02   # R_jup (Vos et al. 2022)
    >>>
    >>> seda.phy_params.inclination(vsini=vsini, evsini=evsini,
    ...                             P=P, eP=eP, R=R, eR=eR)
        (50.4, (1.9, 2.0))
 
    Author: Theo Olsen
 
    Date: 2026-06-10
    '''
 
    # ensure percentiles is a tuple
    percentiles = tuple(percentiles)
 
    # validate "central" and "error"
    central_valid = ["mean", "median"]
    if central not in central_valid:
        raise ValueError(
            f"central={central!r} is not recognized. "
            f"Valid options: {central_valid}."
        )
    error_valid = ["std", "percentile"]
    if error not in error_valid:
        raise ValueError(
            f"error={error!r} is not recognized. "
            f"Valid options: {error_valid}."
        )
 
    # attach units
    vsini_u  = vsini  * u.km / u.s
    evsini_u = evsini * u.km / u.s
 
    P_u  = P  * u.hour
    eP_u = eP * u.hour
 
    R_u  = R  * R_jup
    eR_u = eR * R_jup
 
    # deterministic inclination (sanity check, not returned)
    v_eq_det = (2 * np.pi * R_u / P_u).to(u.km / u.s)
    sin_i_det = (vsini_u / v_eq_det).value
    sin_i_det = np.clip(sin_i_det, -1.0, 1.0)
    inc_det   = np.degrees(np.arcsin(sin_i_det))
 
    # Monte Carlo samples — drawn from Gaussian distributions
    vsini_samples = np.random.normal(vsini_u.value,  evsini_u.value, n_mc) * vsini_u.unit
    P_samples     = np.random.normal(P_u.value,      eP_u.value,     n_mc) * P_u.unit
    R_samples     = np.random.normal(R_u.value,      eR_u.value,     n_mc) * R_u.unit
 
    # equatorial velocity for each sample
    v_eq_samples = (2 * np.pi * R_samples / P_samples).to(u.km / u.s)
 
    # sin i for each sample
    sin_i_samples = (vsini_samples / v_eq_samples).value

    # Vos et al. (2017): set unphysical sin i > 1 to 1 instead of discarding
    n_clipped = int(np.sum(sin_i_samples > 1.0))
    sin_i_used = np.where(sin_i_samples > 1.0, 1.0, sin_i_samples)

    mask_valid = ~np.isnan(sin_i_used)
    n_nan = int(np.sum(~mask_valid))
    n_used = int(np.sum(mask_valid))

    if n_clipped > 0:
        print(
            f"{n_clipped}/{n_mc} MC samples had sin i > 1 and were set to sin i = 1 "
            f"(Vos et al. 2017)."
        )
    if n_nan > 0:
        print(f"{n_nan}/{n_mc} MC samples were NaN and excluded from statistics.")
    print(f"{n_used}/{n_mc} MC samples used for inclination statistics.")

    sin_i_valid = sin_i_used[mask_valid]

    # inclination in degrees
    inc_samples = np.degrees(np.arcsin(sin_i_valid))

    if inc_samples.size == 0:
        raise ValueError(
            'All Monte Carlo samples were NaN. '
            'Check that vsini, P, and R are mutually consistent.'
        )

    # central value
    if central == "mean":
        inc_val = np.mean(inc_samples)
    elif central == "median":
        inc_val = np.median(inc_samples)
 
    # uncertainty
    if error == "std":
        inc_err = np.std(inc_samples)
 
    elif error == "percentile":
        p_lo, p_hi = np.percentile(inc_samples, percentiles)
        inc_err = (inc_val - p_lo, p_hi - inc_val)
 
    return inc_val, inc_err

##########################
def evol_params(Lbol, eLbol, R, eR, model, filename=None,
                n_mc=10000, central="median", error="percentile", 
                percentiles=(16, 84), verbose=True):
	'''
	Description:
	------------
		Infer fundamental parameters by interpolating evolutionary models given a 
		bolometric luminosity and radius. Uncertainties are propagated with a 
		Monte Carlo simulation.


	Parameters:
	-----------
	- Lbol : float
		Bolometric luminosity in units of L_sun.
	- eLbol : float
		Uncertainty in bolometric luminosity (L_sun).
	- R : float
		Radius in units of R_jup.
	- eR : float
		Uncertainty in radius (R_jup).
	- model : str, required
		Evolutionary models whose tables are used. See available models in
		``seda.models.EvolutionaryModels().available_models``.
	- filename : str, optional
		Basename of an evolutionary table inside ``evolution_aux/<model>/``.
		If only one table is bundled, it is selected automatically.
		If multiple tables exist, available basenames are printed.
	- n_mc : int, optional (default 10000)
		Number of Monte Carlo samples for uncertainties.
	- central : str, optional (default "median")
		"mean" or "median" for the central value.
	- error : str, optional (default "percentile")
		"std" or "percentile".
	- percentiles : tuple or list, optional (default (16, 84))
		Lower and upper percentiles for the uncertainty when ``error="percentile"``.
	- verbose : {``True``, ``False``}, optional (default ``True``)
		Print the inferred parameters and the number and fraction of samples outside the grid.

	Returns:
	--------
	Dictionary with inferred grid columns and uncertainties:
		- One key per interpolated column (e.g. ``'mass'``, ``'age'``, ``'logg'``, ``'Teff'``)
		  in the native units defined by the model ``config.json``.
		- Matching uncertainty keys prefixed with ``e`` (e.g. ``'emass'``, ``'eage'``).
		  Each uncertainty is a scalar if ``error="std"`` or a ``(lower, upper)`` tuple if 
		  ``error="percentile"``.
		- ``'n_outside_grid'`` : number of Monte Carlo samples outside the grid coverage.
		- ``'frac_outside_grid'`` : fraction of Monte Carlo samples outside the grid.

	Example:
	--------
	>>> import seda
	>>>
	>>> # input bolometric luminosity and radius with errors
	>>> Lbol, eLbol = 6.324e-5, 6.978e-6  # in Lsun
	>>> R, eR = 1.018, 0.059              # in Rjup
	>>>
	>>> # infer parameters using the bundled solar-metallicity Bobcat table
	>>> out = seda.phy_params.evol_params(Lbol=Lbol, eLbol=eLbol, R=R, eR=eR,
	>>>                                   model='Sonora_Bobcat', filename='nc+0.0_co1.0_mass')
	>>> out['mass'], out['age']
	    (0.0133, 0.51)  

	Author: Theo Olsen

	Date: 2026-06-02
	'''

	# verify that "central" and "error" are valid strings
	central_valid = ["mean", "median"]
	if central not in central_valid:
		raise ValueError(
			f"central={central!r} is not recognized. "
			f"Valid options: {central_valid}."
		)
	error_valid = ["std", "percentile"]
	if error not in error_valid:
		raise ValueError(
			f"error={error!r} is not recognized. "
			f"Valid options: {error_valid}."
		)

	if Lbol <= 0:
		raise ValueError(f"Lbol must be positive, got {Lbol}.")

	available_models = models.EvolutionaryModels().available_models
	if model not in available_models:
		raise ValueError(
			f'Evolutionary model {model!r} is not recognized. '
			f'Available evolutionary models:\n'
			f'          {available_models}'
		)

	basename = models.resolve_evolutionary_table(model, filename) #find basename for called model (or only model if left blank)
	grid = models.read_evolutionary_model(filename=basename, model=model)

	_, plugin = models._load_evolutionary_model(model)
	if not hasattr(plugin, '_convert_inputs'):
		raise NotImplementedError(
			f'Evolutionary model "{model}" has no _convert_inputs in plugin.py. '
			f'See the tutorial on ingesting evolutionary models.'
		)
	converted = plugin._convert_inputs(Lbol, eLbol, R, eR)

	interp_axes = ('logL', 'radius')
	for axis in interp_axes:
		e_axis = f'e_{axis}'
		if axis not in converted or e_axis not in converted:
			raise ValueError(
				f'{model}/plugin.py _convert_inputs must return {axis!r} and {e_axis!r} '
				f'for Monte Carlo sampling on the grid interpolation axes.'
			)

	grid_logL = np.asarray(grid['logL'], dtype=float)
	grid_R    = np.asarray(grid['radius'], dtype=float)
	points = np.column_stack((grid_logL, grid_R))

	# Monte Carlo simulation on grid interpolation axes (units set by plugin)
	logL_samples = np.random.normal(converted['logL'], converted['e_logL'], n_mc)
	R_samples    = np.random.normal(converted['radius'], converted['e_radius'], n_mc)

	xi = np.column_stack((logL_samples, R_samples))

	# helper function to summarize a set of MC samples into a central value and an uncertainty
	def _summarize(samples):
		good = samples[~np.isnan(samples)]
		if central == "mean":
			val = np.mean(good)
		else: # median
			val = np.median(good)
		if error == "std":
			err = np.std(good)
		else: # percentile
			p_lo, p_hi = np.percentile(good, percentiles)
			err = (val - p_lo, p_hi - val)
		return val, err

	out = {}  # final inferred parameters and uncertainties
	first_samples = None  # MC samples from first column, used to count out-of-grid points
	param_samples = {}  # MC samples for every interpolated grid column
	for param, grid_values in grid.items():  # loop over each quantity stored in the table
		if param in ('logL', 'radius'):  # skip axes already used to build xi
			continue
		interp = LinearNDInterpolator(points, np.asarray(grid_values, dtype=float))  # 2D interpolator for this column
		samples = interp(xi)  # infer this parameter at each MC (logL, R) draw
		param_samples[param] = samples  # store samples for later summarization
		if first_samples is None:  # remember first column only once
			first_samples = samples  # NaN mask is the same for all columns at the same xi

	n_outside = int(np.sum(np.isnan(first_samples)))
	frac_outside = n_outside / n_mc
	print(f'{n_outside}/{n_mc} ({100*frac_outside:.1f}%) Monte Carlo samples fell '
	             'outside the evolutionary grid and were excluded from the statistics.')
	if frac_outside > 0.5:
		print(' More than half of the Monte Carlo samples are outside the '
		             'grid; the inferred values are poorly constrained.')
	if n_outside == n_mc:
		raise ValueError('All Monte Carlo samples fell outside the evolutionary grid. '
		                 'Check that Lbol and R are within the model coverage '
		                 'and that the table units/column order are correct.')

	for param, samples in param_samples.items():
		val, err = _summarize(samples)
		out[param] = val
		out[f'e{param}'] = err

	out['n_outside_grid'] = n_outside
	out['frac_outside_grid'] = frac_outside

	if verbose:
		model_info = models.EvolutionaryModels(model)
		units = getattr(model_info, 'units', {})
		def _fmt_err(err):
			if isinstance(err, tuple):
				return '(-{:.4g}, +{:.4g})'.format(err[0], err[1])
			return '{:.4g}'.format(err)
		print(f'\nInferred fundamental parameters ({model_info.name}, {basename}):')
		for param in grid:
			if param in ('logL', 'radius'):
				continue
			unit = units.get(param, '')
			unit_str = f' {unit}' if unit else ''
			print('   {} = {:.4g} {}{}'.format(
				param, out[param], _fmt_err(out[f'e{param}']), unit_str))

	return out

##########################
def color_anomaly(color, color_name, spt, table, ecolor=None,
		age_group=None, reference_stat='mean'):
	'''
	Description:
	------------
		Compute a photometric color anomaly relative to a reference sequence
		for a given spectral type. The anomaly is the magnitude difference

			anomaly = color - reference_color

		where a positive value indicates the object is redder than the
		reference sequence at that spectral type.

	Parameters:
	-----------
	- color : float
		Observed color in magnitudes.
	- color_name : str
		Photometric color identifier. Supported options match Faherty et al.
		(2016) Tables 15-16: ``J-H``, ``J-K``, ``J-W1``, ``J-W2``, ``H-K``,
		``H-W1``, ``H-W2``, ``K-W1``, ``K-W2``, and ``W1-W2``.
		Underscores and case are accepted (e.g. ``J_H``, ``j-h``).
	- spt : str or float
		Spectral type (e.g. ``'L5'``, ``'T4'``, ``'M7'``, or numeric subtype
		such as ``15.0`` for ``'L5'``). Integer SpT values use that subtype bin directly.
		Fractional SpT values (e.g. ``'L3.7'``) linearly interpolate the
		reference color between the floor and ceil integer bins.
	- table : str
		Reference table (required): ``'faherty16'`` (bundled Faherty et al. 2016
		Tables 15-16 field/normal means, or the full Table 1 sample,
		M7-L8 only) or ``'ultracool'`` (mean or median colors recomputed
		from the bundled Ultracool Sheet).
	- ecolor : float, optional
		Uncertainty in the observed color (mag). If provided, the same value
		is returned as the anomaly uncertainty.
	- age_group : str, optional (default None)
		For ``table='ultracool'``: ``None`` (all objects), ``'young'``
		(``youth_evidence`` contains YMG, lowg, SFR, or HYA), or ``'old'``
		(``youth_evidence`` is ``N`` or ``Field``). Entries with ``?`` in
		``youth_evidence`` are excluded from both subgroups.
		For ``table='faherty16'``: ``None`` or ``'old'`` use the published
		Tables 15-16 field means (identical). ``'young'`` recomputes
		a reference from all objects in Faherty et al. (2016) Table 1
		(bundled sample).
	- reference_stat : str, optional (default ``'mean'``)
		``'mean'`` or ``'median'``. For ``table='faherty16'``,
		``age_group=None``/``'old'`` always use the published Table 15-16
		means; ``age_group='young'`` also accepts ``'median'`` since that
		sequence is recomputed from Table 1 photometry.

	Returns:
	--------
	- anomaly : float
		Color anomaly in magnitudes.
	- eanomaly : float, optional
		Returned only when ``ecolor`` is provided.

	Notes:
	------
	- Faherty et al. (2016, ApJS, 225, 10) Tables 15-16 list mean infrared
	  colors for field/normal M7-L8 dwarfs using 2MASS J, H, Ks and WISE
	  W1/W2 photometry with per-band uncertainties < 0.1 mag.
	- For ``table='faherty16'``, ``age_group=None`` and
	  ``age_group='old'`` are equivalent.
	- ``table='faherty16', age_group='young'`` recomputes a reference
	  sequence from all objects in the bundled Table 1 sample using the
	  same 2MASS/WISE bands and a minimum-3-objects-per-bin rule. Integer bins with too few
	  objects but with valid bracketing bins (L6) are
	  linearly interpolated and a :class:`UserWarning` is issued. Edge bins
	  with only one or two objects (e.g. L8) use those objects directly,
	  also with a warning.
	- Ultracool Sheet references are recomputed from bundled photometry
	  (2MASS J/H/Ks and WISE W1/W2). Objects missing either band,
	  or with per-band errors > 0.1 mag, are excluded. Integer bins
	  require at least three objects.
	- For a fractional SpT input (e.g. ``L3.7``), the reference is
	  ``(1-f)*ref(floor) + f*ref(ceil)``. Both neighboring bins must exist.
	- When reference colors are built from catalog photometry, adopted SpT
	  follows each source: Faherty+16 ``spt_flt_assumed`` (optical when
	  available, otherwise infrared; Faherty et al. 2016, Sec. VII) and
	  Ultracool Sheet ``spt_adop_flt`` (optical for M/L, infrared for T).

	Example:
	--------
	>>> import seda
	>>>
	>>> # 2MASS J03552337+1133437 (L5; Suárez et al. 2023)
	>>> color = 14.05 - 11.526  # J-K from 2MASS photometry
	>>> seda.phy_params.color_anomaly(
	...     color=color, color_name='J-K', spt='L5',
	...     table='faherty16', ecolor=0.04)
	    (0.774, 0.04)

	Author: Theo Olsen

	Date: 2026-07-09
	'''
	if not table:
		raise ValueError(
			"table must be specified. Valid options: 'faherty16', 'ultracool'."
		)
	ref = _reference_color(
		color_name=color_name,
		spt=spt,
		table=table,
		age_group=age_group,
		reference_stat=reference_stat,
	)
	anomaly = float(color) - ref
	if ecolor is None:
		return anomaly
	return anomaly, float(ecolor)

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
