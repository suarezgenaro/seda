import pickle
import numpy as np
import dynesty
import time
from .utils import *
import importlib
from spectres import spectres # resample spectra
from sys import exit

def bayes(my_bayes):
	'''
	Description:
	------------
		Estimate Bayesian posteriors using dynesty dynamic nested sampling

	Parameters:
	-----------
	- my_bayes : return parameters from ``input_parameters.BayesOptions``, which also includes the parameters from ``input_parameters.InputData`` and ``input_parameters.ModelOptions``.

	Returns:
	--------
	- '``model``\_nested\_sampling.pickle' : ``dynesty`` instance
		Dynesty output, which is an initialized instance of the chosen sampler using ``dynamic_sampling`` in ``input_parameters.BayesOptions``.

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
	>>> model_dir = ['my_path/output_575.0_650.0/', 
	>>>              'my_path/output_700.0_800.0/'] # folders to seek model spectra
	>>> Teff_range = np.array((700, 900)) # Teff range in K
	>>> logg_range = np.array((4.0, 5.0)) # logg range
	>>> R_range = np.array((0.6, 1.0)) # R range in Rjup
	>>> my_model = seda.ModelOptions(model=model, model_dir=model_dir, logg_range=logg_range, 
	>>>                              Teff_range=Teff_range, R_range=R_range)
	>>> 
	>>> # load chi-square options
	>>> fit_wl_range = np.array([value1, value2]) # to make the fit between value1 and value2
	>>> my_bayes = seda.Chi2FitOptions(my_data=my_data, my_model=my_model, 
	>>>                                fit_wl_range=fit_wl_range)
	>>> 
	>>> # run chi-square fit
	>>> out_bayes = seda.bayes(my_bayes)
	    Nested sampling ran successfully

	Author: Genaro Suárez
	'''

	ini_time_bayes = time.time()
	print('\nEstimate Bayesian posteriors')

	# load input parameters
	# all are stored in my_chi2 but were defined in different classes
	# from InputData
	fit_spectra = my_bayes.fit_spectra
	fit_photometry = my_bayes.fit_photometry
	wl_spectra = my_bayes.wl_spectra
	flux_spectra = my_bayes.flux_spectra
	eflux_spectra = my_bayes.eflux_spectra
	mag_phot = my_bayes.mag_phot
	emag_phot = my_bayes.emag_phot
	filter_phot = my_bayes.filter_phot
	res = my_bayes.res
	lam_res = my_bayes.lam_res
	distance = my_bayes.distance
	edistance = my_bayes.edistance
	N_spectra = my_bayes.N_spectra
	# from ModelOptions
	model = my_bayes.model
	model_dir = my_bayes.model_dir
	Teff_range = my_bayes.Teff_range
	logg_range = my_bayes.logg_range
	R_range = my_bayes.R_range
	# from BayesOptions
	save_results = my_bayes.save_results
	fit_wl_range = my_bayes.fit_wl_range
	model_wl_range = my_bayes.model_wl_range
	logKzz_range = my_bayes.logKzz_range
	Z_range = my_bayes.Z_range
	CtoO_range = my_bayes.CtoO_range
	prior_chi2 = my_bayes.prior_chi2
	grid = my_bayes.grid
	dynamic_sampling = my_bayes.dynamic_sampling
	ndim = my_bayes.ndim
	nlive = my_bayes.nlive
	wl_spectra_min = my_bayes.wl_spectra_min
	wl_spectra_max = my_bayes.wl_spectra_max
	N_datapoints = my_bayes.N_datapoints
	pickle_file = my_bayes.pickle_filename
	Teff_range_prior = my_bayes.Teff_range_prior
	logg_range_prior = my_bayes.logg_range_prior
	logKzz_range_prior = my_bayes.logKzz_range_prior
	Z_range_prior = my_bayes.Z_range_prior
	CtoO_range_prior = my_bayes.CtoO_range_prior
	R_range_prior = my_bayes.R_range_prior

	# when multiple spectra are provided, transform the input list with the spectra to arrays having the spectra next to each other
	# rename the input list with the spectra for convenience 
	wl_spectra_list = wl_spectra
	flux_spectra_list = flux_spectra
	eflux_spectra_list = eflux_spectra
	
	wl_spectra = np.zeros(N_datapoints) # flat array with all wavelengths of input spectra
	flux_spectra = np.zeros(N_datapoints) # flat array with all fluxes of input spectra
	eflux_spectra = np.zeros(N_datapoints) # flat array with all flux errors of input spectra
	l = 0
	for k in range(N_spectra): # for each input observed spectrum
		wl_spectra[l:wl_spectra_list[k].size+l] = wl_spectra_list[k] # add wavelengths of each input spectrum
		flux_spectra[l:wl_spectra_list[k].size+l] = flux_spectra_list[k] # add fluxes of each input spectrum
		eflux_spectra[l:wl_spectra_list[k].size+l] = eflux_spectra_list[k] # add flux errors of each input spectrum
		l += wl_spectra_list[k].size # counter to indicate the minimum index

	print(f'\n   Uniform priors:')
	print(f'      Teff range = {Teff_range_prior}')
	print(f'      logg range = {logg_range_prior}')
	print(f'      logKzz range = {logKzz_range_prior}')
	print(f'      Z range = {Z_range_prior}')
	print(f'      CtoO range = {CtoO_range_prior}')
	if distance is not None: print(f'      R range = {R_range_prior}')
	print('')

	#------------------
	def prior_transform(p):
		# p (sampler's live points) takes random values between 0 and 1 for each free parameter
		v = p # just to have the same dimension as p
		v[0] = (Teff_range_prior[1]-Teff_range_prior[0])     * p[0] + Teff_range_prior[0] # Teff_range_prior.min() < Teff < Teff_range_prior.max()
		v[1] = (logg_range_prior[1]-logg_range_prior[0])     * p[1] + logg_range_prior[0] # logg_range_prior.min() < logg < logg_range_prior.max()
		v[2] = (logKzz_range_prior[1]-logKzz_range_prior[0]) * p[2] + logKzz_range_prior[0] # logKzz_range_prior.min() < logKzz < logKzz_range_prior.max()
		v[3] = (Z_range_prior[1]-Z_range_prior[0])           * p[3] + Z_range_prior[0] # Z_range_prior.min() < Z < Z_range_prior.max()
		v[4] = (CtoO_range_prior[1]-CtoO_range_prior[0])     * p[4] + CtoO_range_prior[0] # CtoO_range_prior.min() < CtoO < CtoO_range_prior.max()
		if distance: # sample radius distribution
			v[5] = (R_range_prior[1]-R_range_prior[0])       * p[5] + R_range_prior[0] # R_range.min() < R < R_range.max()

		return v

	#------------------
	def model_gen(p):
		if distance is not None: # sample radius distribution
			Teff, logg, logKzz, Z, CtoO, R = p
		else:
			Teff, logg, logKzz, Z, CtoO = p

		# generate model with the parameters
		out_interpol_model = interpol_Sonora_Elf_Owl(Teff_interpol=Teff, logg_interpol=logg, logKzz_interpol=logKzz, 
																	Z_interpol=Z, CtoO_interpol=CtoO, grid=grid)

		# convolve synthetic spectrum
		out_convolve_spectrum = convolve_spectrum(wl=out_interpol_model['wavelength'], flux=out_interpol_model['flux'], lam_res=lam_res, res=res, 
												  disp_wl_range=np.array([wl_spectra_min, wl_spectra_max]), 
												  convolve_wl_range=np.array([0.99*wl_spectra_min, 1.01*wl_spectra_max])) # padding on both edges to avoid issues we using spectres

		# resample the convolved model spectrum to the wavelength data points in the observed spectra
		flux_model = spectres(wl_spectra, out_convolve_spectrum['wl_conv'], out_convolve_spectrum['flux_conv'])
	
#		# resample model to the wavelength datapoints in the observed spectrum
#		flux_model = spectres(wl_spectra, out_interpol_model['wavelength'], out_interpol_model['flux'])#, verbose=False,fill=np.nan)

		# scaled model spectrum
		if distance is not None:
			scaling = ((R*6.991e4) / (distance*3.086e13))**2 # scaling = (R/d)^2
		else:
			scaling = np.sum(flux_spectra*flux_model/eflux_spectra**2) / np.sum(flux_model**2/eflux_spectra**2) # scaling that minimizes chi2
		flux_model = scaling*flux_model
	
		return flux_model # return scaled, convolved, resampled fluxes for the model with the above parameters

	#------------------
	def loglike(p):
		if distance is not None: # sample radius distribution
			Teff, logg, logKzz, Z, CtoO, R = p
		else:
			Teff, logg, logKzz, Z, CtoO = p

		flux_model = model_gen(p) # model spectrum for each parameters' combination

		residual2 = (flux_spectra - flux_model)**2 / eflux_spectra**2
		lnlike = -0.5 * np.sum(residual2 + np.log(2*np.pi*eflux_spectra**2))

		return lnlike

	print('\n   Starting dynesty...')
	if dynamic_sampling:
		# 'dynamic' nested sampling.
		sampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim, nlive=nlive)
		sampler.run_nested()
		results = sampler.results
#		results_file = f'{model}_dynamic_nested.pickle'

	if not dynamic_sampling:
		# 'static' nested sampling
		sampler = dynesty.NestedSampler(loglike, prior_transform, ndim, nlive=nlive)
		sampler.run_nested()
		results = sampler.results
#		results_file = f'{model}_static_nested.pickle'

	# save nested sampling result as pickle
	with open(pickle_file, 'wb') as file:
		# serialize and write the variable to the file
		pickle.dump(results, file)

	print('\nNested sampling ran successfully')
	fin_time_bayes = time.time()
	out_time_elapsed = time_elapsed(fin_time_bayes-ini_time_bayes)
	print(f'\n   elapsed time running bayes_fit: {out_time_elapsed[0]} {out_time_elapsed[1]}')

	return results
