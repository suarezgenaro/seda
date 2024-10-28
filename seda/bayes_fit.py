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

	Returns: results by Dynesty

	'''

	ini_time_bayes = time.time()
	print('\nEstimating Bayesian posteriors...')

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
	wl_spectra_min = my_bayes.wl_spectra_min
	wl_spectra_max = my_bayes.wl_spectra_max
	N_datapoints = my_bayes.N_datapoints
	pickle_file = my_bayes.pickle_file

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

	# read model grid within the Teff and logg ranges
	if grid is None: 
		if Teff_range is None: print('Please provide the Teff_range parameter'), exit()
		if logg_range is None: print('Please provide the logg_range parameter'), exit()
		grid = read_grid(model=model, Teff_range=Teff_range, logg_range=logg_range)

	# parameter ranges for the posteriors
	if prior_chi2:
		out_param_ranges_sampling = param_ranges_sampling(model)
		Teff_range_posterior = out_param_ranges_sampling['Teff_range_posterior']
		logg_range_posterior = out_param_ranges_sampling['logg_range_posterior']
		logKzz_range_posterior = out_param_ranges_sampling['logKzz_range_posterior']
		Z_range_posterior = out_param_ranges_sampling['Z_range_posterior']
		CtoO_range_posterior = out_param_ranges_sampling['CtoO_range_posterior']
	else:
		# I AM HERE: SAMPLING DOESN'T WORK WITH ANY OF THE PARAMS DEFINITIONS BELOW. IT SAYS THERE IS SOMETHING OUT OF BOUNDS
		# define priors from input parameters or grid ranges
		if Teff_range is None: print('Please provide the Teff_range parameter'), exit()
		if logg_range is None: print('Please provide the logg_range parameter'), exit()
		Teff_range_posterior = Teff_range
		logg_range_posterior = logg_range
		out_grid_ranges = grid_ranges(model) # read grid ranges
		if logKzz_range is not None: logKzz_range_posterior = logKzz_range
		else: logKzz_range_posterior = np.array([out_grid_ranges['logKzz'].min(), out_grid_ranges['logKzz'].max()])
		if Z_range is not None: Z_range_posterior = Z_range
		else: Z_range_posterior = np.array([out_grid_ranges['Z'].min(), out_grid_ranges['Z'].max()])
		if CtoO_range is not None: CtoO_range_posterior = CtoO_range
		else: CtoO_range_posterior = np.array([out_grid_ranges['CtoO'].min(), out_grid_ranges['CtoO'].max()])

		# TEST: give a padding on both edges of each parameter
#		Teff_range_posterior = np.array([1.05*Teff_range_posterior[0], 0.95*Teff_range_posterior[1]])
#		logg_range_posterior = np.array([1.05*logg_range_posterior[0], 0.95*logg_range_posterior[1]])
#		logKzz_range_posterior = np.array([1.05*logKzz_range_posterior[0], 0.95*logKzz_range_posterior[1]])
#		Z_range_posterior = np.array([1.05*Z_range_posterior[0], 0.95*Z_range_posterior[1]])
#		CtoO_range_posterior = np.array([1.05*CtoO_range_posterior[0], 0.95*CtoO_range_posterior[1]])

#		# TEST: define ranges as lists as when prior_chi2 is True
#		Teff_range_posterior = [Teff_range[0], Teff_range[1]]
#		logg_range_posterior = [logg_range[0], logg_range[1]]
#		out_grid_ranges = grid_ranges(model) # read grid ranges
#		logKzz_range_posterior = [out_grid_ranges['logKzz'].min(), out_grid_ranges['logKzz'].max()]
#		Z_range_posterior = [out_grid_ranges['Z'].min(), out_grid_ranges['Z'].max()]
#		CtoO_range_posterior = [out_grid_ranges['CtoO'].min(), out_grid_ranges['CtoO'].max()]

#		Teff_range_posterior = [525.0, 625.0]
#		logg_range_posterior = [3.4981880270062007, 3.9981880270062007]
#		logKzz_range_posterior = [2.5, 5.5]
#		Z_range_posterior = [-1.0, -0.75]
#		CtoO_range_posterior = [1.0, 2.0]
#
##		SOLUTION: CUT THE LOGG RANGE FROM THE RANGE 3.0-5.5 to 3.3-5.5. UNDERSTAND WHY 3.0 DOESN'T WORK BUT 3.3 DOES!!!!!!!!!!!!!!!!!
#		Teff_range_posterior = [300, 700]

#		logg_range_posterior = [3.3, 5.5]


	#if distance_label=='yes':
	if distance is not None:
		if R_range is None: print('Please provide the R_range parameter'), exit()
		R_range_posterior = R_range

	print('\n')
	print(f'   Uniform priors:')
	print(f'      Teff range = {Teff_range_posterior}')
	print(f'      logg range = {logg_range_posterior}')
	print(f'      logKzz range = {logKzz_range_posterior}')
	print(f'      Z range = {Z_range_posterior}')
	print(f'      CtoO range = {CtoO_range_posterior}')
	if distance is not None: print(f'      R range = {R_range_posterior}')

	#------------------
	def prior_transform(p):
		# p (sampler's live points) takes random values between 0 and 1 for each free parameter
		v = p # just to have the same dimension as p
		v[0] = (Teff_range_posterior[1]-Teff_range_posterior[0])     * p[0] + Teff_range_posterior[0] # Teff_range_posterior.min() < Teff < Teff_range_posterior.max()
		v[1] = (logg_range_posterior[1]-logg_range_posterior[0])     * p[1] + logg_range_posterior[0] # logg_range_posterior.min() < logg < logg_range_posterior.max()
		v[2] = (logKzz_range_posterior[1]-logKzz_range_posterior[0]) * p[2] + logKzz_range_posterior[0] # logKzz_range_posterior.min() < logKzz < logKzz_range_posterior.max()
		v[3] = (Z_range_posterior[1]-Z_range_posterior[0])           * p[3] + Z_range_posterior[0] # Z_range_posterior.min() < Z < Z_range_posterior.max()
		v[4] = (CtoO_range_posterior[1]-CtoO_range_posterior[0])     * p[4] + CtoO_range_posterior[0] # CtoO_range_posterior.min() < CtoO < CtoO_range_posterior.max()
		if distance: # sample radius distribution
			v[5] = (R_range_posterior[1]-R_range_posterior[0])       * p[5] + R_range_posterior[0] # R_range.min() < R < R_range.max()

		return v

	#------------------
	def model_gen(p):
#		import interpol_model as interpol_model
#		import importlib
#		from spectres import spectres # resample spectra
	
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

	# Define the dimensionality of our problem.
	if distance is not None: # sample radius distribution
		ndim = 6
	else:
		ndim = 5

	nlive = 500 #number of nested sampling live points	

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
	print('   nested sampling results saved successfully')

	fin_time_bayes = time.time()
	out_time_elapsed = time_elapsed(fin_time_bayes-ini_time_bayes)
	print(f'   elapsed time running bayes_fit: {out_time_elapsed[0]} {out_time_elapsed[1]}')

	return results

##########################
# tolerance around the best-fitting spectrum parameters to define the parameter ranges for posteriors
def param_ranges_sampling(model):

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
	
		Teff_range_posterior = [max(Teff_chi2-Teff_search, min(out_grid_ranges['Teff'])), 
								min(Teff_chi2+Teff_search, max(out_grid_ranges['Teff']))]
		logg_range_posterior = [max(logg_chi2-logg_search, min(out_grid_ranges['logg'])), 
								min(logg_chi2+logg_search, max(out_grid_ranges['logg']))]
		logKzz_range_posterior = [max(logKzz_chi2-logKzz_search, min(out_grid_ranges['logKzz'])), 
								min(logKzz_chi2+logKzz_search, max(out_grid_ranges['logKzz']))]
		Z_range_posterior = [max(Z_chi2-Z_search, min(out_grid_ranges['Z'])), 
								min(Z_chi2+Z_search, max(out_grid_ranges['Z']))]
		CtoO_range_posterior = [max(CtoO_chi2-CtoO_search, min(out_grid_ranges['CtoO'])), 
								min(CtoO_chi2+CtoO_search, max(out_grid_ranges['CtoO']))]

		out = {'Teff_range_posterior': Teff_range_posterior, 'logg_range_posterior': logg_range_posterior, 'logKzz_range_posterior': logKzz_range_posterior, 'Z_range_posterior': Z_range_posterior, 'CtoO_range_posterior': CtoO_range_posterior}

	return out

##########################
# get the parameter ranges and steps in each model grid
def grid_ranges(model):
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
# generate the synthetic spectrum with the posterior parameters
def best_fit_sampling(wl_spectra, model, sampling_file, lam_res, res, distance=None, grid=None, save_spectrum=False):
	'''
	Output: dictionary
		synthetic spectrum for the posterior parameters: i) with original resolution, ii) convolved to the compared observed spectra, and iii) scaled to the determined radius
	'''

	import sampling
	import interpol_model
	from spectres import spectres # resample spectra

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
