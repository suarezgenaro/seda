import pickle
import numpy as np
import dynesty
import time
import logging
from .utils import *
import importlib
import copy
from spectres import spectres # resample spectra
from scipy.stats import truncnorm
from scipy.special import roots_legendre
from scipy.interpolate import PchipInterpolator
from scipy.integrate import cumulative_trapezoid
from dynesty.utils import resample_equal, Results
from sys import exit

def bayes(my_bayes):
	'''
	Description:
	------------
		Estimate posterior distributions of atmospheric model parameters
		and radius using Bayesian inference with dynesty nested sampling.

	Parameters:
	-----------
	- my_bayes : object returned by ``seda.BayesOptions``, which also
	  includes parameters from ``seda.InputData`` and ``seda.ModelOptions``.

	Returns:
	--------
	- out_bayes : dictionary
		Dictionary containing:
			- ``my_bayes``: input parameters from ``seda.BayesOptions``.
			- ``out_dynesty``: dynesty results, with R replacing q in the posterior
			  samples when distance uncertainty is provided.
			- ``out_dynesty_sampling``: original dynesty sampling results
			  (when distance uncertainty is provided).
			- ``samples_equal``: equal-weight posterior samples, including R
			  instead of q (when distance uncertainty is provided).
			- ``d_samples``: reconstructed distance posterior samples in pc
			  (when distance uncertainty is provided).

	Example:
	--------
	>>> import numpy as np
	>>> import seda
	>>>
	>>> # load input data
	>>> wl_spectra = wl_input # um
	>>> flux_spectra = flux_input # erg/cm^2/s/A
	>>> eflux_spectra = eflux_input # erg/cm^2/s/A
	>>> distance = 5.71 # in pc
	>>> edistance = 0.06 # in pc
	>>> my_data = seda.InputData(wl_spectra=wl_spectra,
	>>>                          flux_spectra=flux_spectra,
	>>>                          eflux_spectra=eflux_spectra,
	>>>                          distance=distance, edistance=edistance)
	>>>
	>>> # load model options
	>>> model = 'Sonora_Elf_Owl'
	>>> model_dir = ['my_path/output_575.0_650.0/',
	>>>              'my_path/output_700.0_800.0/'] # folders containing model spectra
	>>> # select a subset of the model grid
	>>> params_ranges = {'Teff': [700, 900], 'logg': [4.0, 5.0]}
	>>> my_model = seda.ModelOptions(model=model, model_dir=model_dir,
	>>>                              params_ranges=params_ranges)
	>>>
	>>> # load Bayesian options
	>>> R_range = np.array([0.6, 1.0]) # radius prior range in Rjup
	>>> fit_wl_range = [value1, value2] # wavelength range for fitting
	>>> my_bayes = seda.BayesOptions(my_data=my_data, my_model=my_model,
	>>>                             fit_wl_range=fit_wl_range, R_range=R_range)
	>>>
	>>> # run Bayesian inference
	>>> out_bayes = seda.bayes(my_bayes)
	    Bayesian sampling ran successfully

	Author: Genaro Suárez
	'''

	# suppress noisy Dynesty logging
	logging.getLogger("dynesty").setLevel(logging.WARNING)

	ini_time_bayes = time.time()
	print('\n   Estimate Bayesian posteriors')

	# load input parameters
	# all are stored in my_bayes but were defined in different classes
	# from InputData
	fit_spectra = my_bayes.fit_spectra
	fit_photometry = my_bayes.fit_photometry
	wl_spectra = my_bayes.wl_spectra
	flux_spectra = my_bayes.flux_spectra
	eflux_spectra = my_bayes.eflux_spectra
	res = my_bayes.res
	lam_res = my_bayes.lam_res
	distance = my_bayes.distance
	edistance = my_bayes.edistance
	# from ModelOptions
	model = my_bayes.model
	model_dir = my_bayes.model_dir
	params_ranges = my_bayes.params_ranges
	# from BayesOptions
	save_results = my_bayes.save_results
	fit_wl_range = my_bayes.fit_wl_range
	model_wl_range = my_bayes.model_wl_range
	R_range = my_bayes.R_range
	grid = my_bayes.grid
	params_unique = my_bayes.params_unique
	params_priors = my_bayes.params_priors
	dynamic_sampling = my_bayes.dynamic_sampling
	ndim = my_bayes.ndim
	nlive = my_bayes.nlive
	bayes_pickle_file = my_bayes.bayes_pickle_file
	if fit_spectra:
		wl_obs_spec = my_bayes.wl_spectra_fit
		flux_obs_spec = my_bayes.flux_spectra_fit
		eflux_obs_spec = my_bayes.eflux_spectra_fit
		weight_spec_fit = my_bayes.weight_spec_fit 
	if fit_photometry:
		# redefine them as lists to keep format when fit_spectra
		wl_obs_phot = [my_bayes.lambda_eff_SVO_fit]
		flux_obs_phot = [my_bayes.phot_fit]
		eflux_obs_phot = [my_bayes.ephot_fit]
		weight_phot_fit = my_bayes.weight_phot_fit 

	# set input flux densities depending whether spectra and/or photometry were provided
	if fit_spectra and not fit_photometry:
		wl_obs = wl_obs_spec
		flux_obs = flux_obs_spec
		eflux_obs = eflux_obs_spec
		weights = weight_spec_fit
	if not fit_spectra and fit_photometry:
		wl_obs = wl_obs_phot
		flux_obs = flux_obs_phot
		eflux_obs = eflux_obs_phot
		weights = [weight_phot_fit]
	if fit_spectra and fit_photometry:
		wl_obs = wl_obs_spec + wl_obs_phot
		flux_obs = flux_obs_spec + flux_obs_phot
		eflux_obs = eflux_obs_spec + eflux_obs_phot
		weights = weight_spec_fit + [weight_phot_fit]

	# conversion factor for geometric scaling
	if distance is not None:
		scaling_factor = (u.R_jup.to(u.km) / u.pc.to(u.km))**2
	
		if edistance is not None:
			q_index = list(params_priors.keys()).index('q')
		else:
			R_index = list(params_priors.keys()).index('R')

	# Gaussian normalization term (constant during sampling)
	lnlike_norm = 0.0
	for i in range(len(flux_obs)):
		lnlike_norm += -0.5 * np.sum(
			np.log(2*np.pi*eflux_obs[i]**2/weights[i]))

	if distance is not None and edistance is not None:
		Rmin, Rmax = R_range

		# Gaussian distance prior truncated to positive distances
		a = -distance / edistance
		dist = truncnorm(a, np.inf, loc=distance, scale=edistance)

		# define a numerical range of q = R/d from the radius and distance priors
		dmin = dist.ppf(1e-10)
		dmax = dist.ppf(1 - 1e-10)

		qmin = Rmin / dmax
		qmax = Rmax / dmin

		# logarithmically spaced grid to construct the CDF of q
		q_grid = np.geomspace(qmin, qmax, 20000)

		# compute the marginalized CDF of q by integrating over the radius prior
		# F(q) = 1/(Rmax-Rmin) * integral[P(d >= R/q)] dR
		nodes, weights_quad = roots_legendre(128)
		R_nodes = Rmin + (Rmax - Rmin) * (nodes + 1) / 2

		q_cdf = np.zeros_like(q_grid)
		for R, w in zip(R_nodes, weights_quad):
			q_cdf += 0.5 * w * dist.sf(R / q_grid)

		# normalize the CDF over the retained numerical range
		q_cdf = (q_cdf - q_cdf[0]) / (q_cdf[-1] - q_cdf[0])

		# remove duplicate CDF values to ensure a strictly increasing inverse
		q_cdf, idx = np.unique(q_cdf, return_index=True)
		q_grid = q_grid[idx]

		# construct the inverse CDF to transform uniform samples into q samples
		q_inverse_cdf = PchipInterpolator(q_cdf, q_grid)

	# print priors
	print('\n      Priors:')
	for param in params_priors:
		if param == 'q':
			print(f'         R: Uniform{R_range}')
			print(f'         d: Truncated Gaussian'
				  f'(mean={distance}, std={edistance}, min=0)')
		else:
			print(f'         {param}: Uniform{params_priors[param]}')

	#------------------
	def prior_transform(p):
		# transform unit-cube samples into physical parameter priors
		v = p.copy()
	
		for i, param in enumerate(params_priors):
			if param == 'q':
				# sample q from its marginalized prior using the inverse CDF
				v[i] = q_inverse_cdf(p[i])
	
			else:
				# map unit-cube samples to uniform prior ranges
				v[i] = (params_priors[param][1] - params_priors[param][0]) * p[i] + params_priors[param][0]
	
		return v

	#------------------
	def loglike(p):
		# arrange parameter values in the sampling related to model free parameters
		# to be used to generate the corresponding model spectrum

		# extract model parameters
		params = {}
		for i, param in enumerate(params_priors.keys()):
			if param in params_unique:
				params[param] = p[i]

		# generate model spectra for all datasets
		flux_models = []
		for i in range(len(flux_obs)): # for each input spectrum and/or photometric set
			# generate a model with the sampled parameters
			# (the model will have the resolution and be resampled to the fit region for each input spectrum)
			out_generate_model_spectrum = generate_model_spectrum(
				params=params, model=model, grid=grid[i])
			flux_models.append(out_generate_model_spectrum['flux'])

		# calculate a common scaling factor for all datasets
		if distance is not None:
			if edistance is not None:
				q = p[q_index] # sampled q
			else:
				R = p[R_index] # sampled R
				q = R / distance
		
			scaling = q**2 * scaling_factor

		else:
			# common scaling factor that maximizes the weighted Gaussian log-likelihood
			# (equivalently, minimizes the weighted least-squares objective)
			numerator = 0.0
			denominator = 0.0
			for i in range(len(flux_obs)):
				numerator += np.sum(weights[i] * flux_obs[i] * flux_models[i] / eflux_obs[i]**2)
				denominator += np.sum(weights[i] * flux_models[i]**2 / eflux_obs[i]**2)
			scaling = numerator / denominator

		# calculate the weighted Gaussian log-likelihood
		lnlike = lnlike_norm
		for i in range(len(flux_obs)):
			flux_model = scaling * flux_models[i]
			residual2 = weights[i] * ((flux_obs[i] - flux_model)**2 / eflux_obs[i]**2)
			lnlike += -0.5 * np.sum(residual2)

		return lnlike

	#+++++++++++++++++++++++++++++++++++++
	print('\n   Starting dynesty...')
	if dynamic_sampling:
		# 'dynamic' nested sampling.
		sampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim, nlive=nlive)
		sampler.run_nested()
		results = sampler.results

	if not dynamic_sampling:
		# 'static' nested sampling
		sampler = dynesty.NestedSampler(loglike, prior_transform, ndim, nlive=nlive)
		sampler.run_nested()
		results = sampler.results

	# output dictionary
	# remove model grid from my_bayes dictionary to have a lighter output file
	my_bayes_copy = copy.copy(my_bayes) # to avoid affecting the original object when removing "grid"
	del my_bayes_copy.grid

	# replace q with R in the output parameter definitions
	if distance is not None and edistance is not None:
		my_bayes_copy.params_priors = {}
		for param, prior in params_priors.items():
			if param == 'q':
				my_bayes_copy.params_priors['R'] = R_range
			else:
				my_bayes_copy.params_priors[param] = prior

	out_bayes = {'my_bayes': my_bayes_copy, 'out_dynesty': results}

	# reconstruct radius posterior samples when distance uncertainty is included
	if distance is not None and edistance is not None:
		# generate equal-weight posterior samples
		weights_post = np.exp(results.logwt - results.logz[-1])
		samples_equal = resample_equal(results.samples, weights_post, rstate=np.random.default_rng(42))

		# reconstruct radius and distance posterior samples
		q_samples = samples_equal[:, q_index].copy()
		R_samples, d_samples = _recover_radius(q_samples, R_range, dist)

		# replace q with R in the equal-weight samples
		samples_equal[:, q_index] = R_samples

		# create dynesty results with equal-weight posterior samples
		results_dict = results.asdict()
		results_dict['samples'] = samples_equal
		results_dict['logwt'] = np.zeros(len(samples_equal))
		results_dict['logz'] = np.zeros(len(samples_equal))

		# preserve original sampling results
		out_bayes['out_dynesty_sampling'] = results

		# replace q with R for posterior analysis
		out_bayes['out_dynesty'] = Results(results_dict)

		# save posterior samples
		out_bayes['samples_equal'] = samples_equal
		out_bayes['d_samples'] = d_samples

	# save output dictionary
	if save_results:
		with open(bayes_pickle_file, 'wb') as file:
			# serialize and write the variable to the file
			pickle.dump(out_bayes, file)
		print('      Bayesian sampling results saved successfully')

	print('\n   Bayesian sampling ran successfully')
	fin_time_bayes = time.time()
	print_time(fin_time_bayes-ini_time_bayes)

	return out_bayes


def _recover_radius(q_samples, R_range, dist, seed=42):
	"""
	Reconstruct radius posterior samples from q = R/d.

	For each posterior sample of q, draw a distance consistent with
	the original radius and distance priors, then calculate R = q*d.
	"""

	# set a random seed for reproducible sampling
	rng = np.random.default_rng(seed)

	# radius prior limits (in Jupiter radii)
	Rmin, Rmax = R_range

	R_samples = np.empty(len(q_samples))
	d_samples = np.empty(len(q_samples))

	for i, q in enumerate(q_samples):
		# the radius prior restricts the allowed distance for this q.
		dmin = Rmin / q
		dmax = Rmax / q

		# numerically integrate the conditional distance density.
		d_grid = np.linspace(dmin, dmax, 1000)
		pdf = d_grid * dist.pdf(d_grid)
		cdf = cumulative_trapezoid(pdf, d_grid, initial=0)

		if cdf[-1] <= 0:
			raise RuntimeError(f'Conditional distance distribution failed for q={q}')

		# inverse-CDF sampling preserves the conditional distribution.
		cdf /= cdf[-1]
		d = np.interp(rng.random(), cdf, d_grid)

		d_samples[i] = d
		R_samples[i] = q * d

	return R_samples, d_samples
