import numpy as np
import time
import os
#from subprocess import check_output # unlike os.system, check_output works in Windows as well
import astropy.units as u
import pickle
from .utils import *
from astropy.io import ascii
from astropy.table import vstack
from lmfit import Minimizer, minimize, Parameters, report_fit # model fit for non-linear least-squares problems
from sys import exit

def chi2_fit(my_data, my_model, my_chi2):
	'''
	Description:
	------------
		Minimize the chi-square statistic to find the best model fits

	Parameters:
	-----------
	- my_data : parameters from ``input_parameters.InputData``
	- my_model : parameters from ``input_parameters.ModelOptions``
	- my_chi2 : parameters from ``input_parameters.Chi2Options``

	Returns:
	--------
	- ``model``\_chi2\_minimization.dat
	'''
#	- ``model``+'_chi2_minimization.dat' : file with all fitted model spectra sorted by chi square and including the information:
#		spectrum name, chi square, reduced chi square, scaling, scaling error, extinction, extinction error, effective temperature, surface gravity, and iteration.
#	'''
#	model+dynamic_sampling+'nested.pickle': results from the nested sampling provided by Dynesty
#	model+'chi2_minimization.pickle': dictionary with the results from the chi square minimization with the following parameters:
#		out_chi2['model']: model used
#		out_chi2['spectra_name']: model spectra name
#		out_chi2['out_lmfit']: output of the minner.minimize routine that minimizes chi2, which is used by seda to obtain parameters from the fit, 
#								 namely iterations, scaling factor, extinction, and (reduced) chi square
#		out_chi2['iterations_fit']: number of iterations to minimize chi square
#		out_chi2['Av_fit']: visual extinction (in mag) that minimizes chi square
#		out_chi2['eAv_fit']: visual extinction uncertainty (in mag)
#		out_chi2['scaling_fit']: scaling factor that minimizes chi square
#		out_chi2['escaling_fit']: scaling factor uncertainty
#		out_chi2['chi2_wl_fit']: chi square as a function of wavelength
#		out_chi2['chi2_red_wl_fit']: reduced chi square as a function of wavelength
#		out_chi2['chi2_fit']: total chi square
#		out_chi2['chi2_red_fit']: reduced total chi square
#		out_chi2['Teff']: effective temperature (in K)
#		out_chi2['logg']: surface gravity (log g)
#		out_chi2['radius']: radius (in Rjup) corresponding to the scaling factor and input distance (calculated only when distance and edistance are provided)
#		out_chi2['eradius']: radius uncertainty (in Rjup)
#		out_chi2['lambda_eff_mean']: mean effective wavelength (in um) of each photometric passband (when using photometry)
#		out_chi2['width_eff_mean']: mean effective width (in um) of each photometric passband (when using photometry)
#		out_chi2['f_phot']: fluxes (in erg/s/cm2/A) of input photometry (when using photometry)
#		out_chi2['ef_phot']: flux uncertainties (in erg/s/cm2/A)
#		out_chi2['phot_synt']: synthetic fluxes (in erg/s/cm2/A) from each model spectrum considering the photometric passbands (when using photometry)
#		out_chi2['phot_synt_red']: synthetic fluxes (in erg/s/cm2/A) from each reddened model spectrum considering the photometric passbands (when using photometry)
#		out_chi2['wl_array_model']: wavelengths (in um) of model spectra with their original-resolution 
#		out_chi2['flux_array_model']: scaled fluxes (in erg/cm2/s/A) of original-resolution model spectra
#		out_chi2['wl_array_model_conv']: wavelengths (in um) of convolved model spectra
#		out_chi2['flux_array_model_conv']: scaled fluxes (in erg/cm2/s/A) of convolved model spectra
#		out_chi2['flux_array_model_red']: scaled fluxes (in erg/cm2/s/A) of original-resolution reddened model spectra
#		out_chi2['flux_array_model_conv_red']: scaled fluxes (in erg/cm2/s/A) of convolved reddened model spectra
#		out_chi2['wl_array_model_conv_resam']: wavelengths (in um) of convolved model spectra re-sampled to the observed spectra (when provided)
#		out_chi2['flux_array_model_conv_resam']: scaled fluxes (in erg/cm2/s/A) of resampled, convolved model spectra
#		out_chi2['flux_array_model_conv_resam_red']: scaled fluxes (in erg/cm2/s/A) of resampled, convolved reddened model spectra
#		out_chi2['flux_residuals']: linear of flux residual (in erg/cm2/s/A) between observed data and model spectra within the fit wavelength range
#		out_chi2['logflux_residuals']: logarithm of flux residual (in erg/cm2/s/A) between observed data and model spectra within the fit wavelength range
#		out_chi2['weight_fit']: weight given to each data point in the model comparison (weight in the equation chi2 = weight * (data-model)^2 / edata^2)
#
#	EXAMPLES
#	--------
#	UPDATE THE EXAMPLES BELOW
#	Example 1: call the routine to compare atmospheric models to an observed spectrum:
#	>>> # input parameters
#	>>> model = 'ATMO2020'
#	>>> Teff_range = np.array((300, 1000)) # K
#	>>> logg_range = np.array((3.0, 5.5)) # dex
#	>>> wl_spectra = wl_SpeX # wavelength of the observed spectrum
#	>>> flux_spectra = flux_SpeX # flux of the observed spectrum
#	>>> eflux_spectra = eflux_SpeX # flux error of the observed spectrum
#	>>> R = 100 # resolution of the observed spectrum
#	>>> lam_R = 2.0 # um; wavelength reference
#
#	>>> # run code 
#	>>> seda_out = seda.seda(model=model, logg_range=logg_range, Teff_range=Teff_range, wl_spectra=wl_spectra, flux_spectra=flux_spectra, eflux_spectra=eflux_spectra, R=R, lam_R=lam_R)
#
#	>>> # output parameters
#	>>> spectra_name = seda_out['spectra_name'] # model spectra name
#	>>> scaling_fit = seda_out['scaling_fit'] # scaling factor that minimizes chi square
#	>>> escaling_fit = seda_out['escaling_fit'] # scaling factor uncertainty
#	>>> chi2_fit = seda_out['chi2_fit'] # total chi square
#	>>> Teff = seda_out['Teff'] # effective temperature (in K)
#	>>> logg = seda_out['logg'] # surface gravity (log g)
#	>>> wl_array_model = seda_out['wl_array_model'] # wavelengths (in um) of model spectra with their original-resolution 
#	>>> flux_array_model = seda_out['flux_array_model'] # scaled fluxes (in erg/cm2/s/A) of original-resolution model spectra
#	>>> wl_array_model_conv = seda_out['wl_array_model_conv'] # wavelengths (in um) of convolved model spectra
#	>>> flux_array_model_conv = seda_out['flux_array_model_conv'] # scaled fluxes (in erg/cm2/s/A) of convolved model spectra
#	>>> wl_array_model_conv_resam = seda_out['wl_array_model_conv_resam'] # wavelengths (in um) of convolved model spectra re-sampled to the observed spectra (when provided)
#	>>> flux_array_model_conv_resam = seda_out['flux_array_model_conv_resam'] # scaled fluxes (in erg/cm2/s/A) of resampled, convolved model spectra
#	>>>	logflux_residuals = seda_out['logflux_residuals'] # flux residual (in erg/cm2/s/A) between observed data and model spectra within the fit wavelength range
#
#	Example 2: call the routine to compare atmospheric models to observed photometry:
#
#	>>> # input parameters (unreal values are given as a reference)
#	>>> model = 'BT-Settl'
#	>>> logg_range = np.array((3.0, 5.5)) # dex
#	>>> Teff_range = np.array((1500, 3500)) # K
#	>>> fit_spectra = 'no'
#	>>> fit_photometry = 'yes'
#	>>> model_wl_range = np.array((0.29, 35))
#	>>> mag_phot = np.array((0.034, 15.908, 12.991, 11.864)) # input photometry
#	>>> emag_phot = np.array((0.034, 0.002, 0.026, 0.023)) # photometry errors
#	>>> filter_phot = np.array(('Gaia_G', 'PanSTARRS_i', '2MASS_J', 'WISE_W1')) # filter information following the filter_phot nomenclature above
#	>>> extinction_free_param = 'no' # do not consider extinction as a free parameter
#	>>> distance = 38.31 # pc
#	>>> edistance = 0.14 # pc
#	>>> model_wl_range = np.array((0.3, 30)) # um
#	>>> skip_convolution = 'yes' # synthetic magnitudes have been calculated for these filters
#	
#	>>> seda_out = seda.seda(model=model, logg_range=logg_range, 
#		 Teff_range=Teff_range, fit_spectra=fit_spectra, fit_photometry=fit_photometry, mag_phot=mag_phot, emag_phot=emag_phot, 
#		 filter_phot=filter_phot, extinction_free_param=extinction_free_param, distance=distance, edistance=edistance, 
#		 model_wl_range=model_wl_range, skip_convolution='yes')
#
#	>>> # output parameters
#	>>> spectra_name = seda_out['spectra_name'] 
#	>>> out_lmfit = seda_out['out_lmfit']
#	>>> iterations_fit = seda_out['iterations_fit']
#	>>> Av_fit = seda_out['Av_fit'] # mag
#	>>> eAv_fit = seda_out['eAv_fit'] # mag
#	>>> scaling_fit = seda_out['scaling_fit']
#	>>> escaling_fit = seda_out['escaling_fit']
#	>>> chi2_wl_fit = seda_out['chi2_wl_fit']
#	>>> chi2_red_wl_fit = seda_out['chi2_red_wl_fit']
#	>>> chi2_fit = seda_out['chi2_fit']
#	>>> chi2_red_fit = seda_out['chi2_red_fit']
#	>>> Teff = seda_out['Teff'] # K
#	>>> logg = seda_out['logg'] # dex
#	>>> radius = seda_out['radius'] # Rjup
#	>>> eradius = seda_out['eradius'] # Rjup
#	>>> lambda_eff_mean = seda_out['lambda_eff_mean'] # um
#	>>> width_eff_mean = seda_out['width_eff_mean'] # um
#	>>> f_phot = seda_out['f_phot'] # erg/s/cm2/A
#	>>> ef_phot = seda_out['ef_phot'] # erg/s/cm2/A
#	>>> phot_synt = seda_out['phot_synt'] # erg/s/cm2/A
#	>>> wl_array_model_convolve = seda_out['wl_array_model_conv'] # um
#	>>> flux_array_model_convolve = seda_out['flux_array_model_conv'] # erg/cm2/s/A
#	>>> weight_fit = seda_out['weight_fit']
#
#	MODIFICATION HISTORY
#				by G. Suárez
#	
#	2024/06/18	Included Lacy & Burrows (2023) extended models
#	2024/05/01	Included Nested sampling by Dynesty to create posteriors
#	2024/04/18	Added an interpolator to generate spectra from Elf-Owl models for any combination of parameters within the grid
#	2024/02/24	Sonora Diamondback atmospheric models (Morley et al. 2024) are available
#	2024/02/19	Sonora Elf Owl atmospheric models (Mukherjee et al. 2024) are available
#	2024/02/18	code split into multiple definitions
#	2023/09/21	Scaling factor can be fixed (useful when the distance and radius are known so the scaling could be derived)
#	2023/08/31	Lacy & Burrows (2023; LB23) models are available
#	2023/03/15	Sonora_Cholla models are available
#	2023/03/09	Option available to use together the three ATMO 2020 model grids 
#	2023/03/06	For multiple spectra, convolve wavelength regions between the spectra and out of the coverage
#	2020/03/10	First version of the code working to find the best fit from the grid spectra by minimizing chi square
#	-----------------------
#
#	'''

	ini_time_SEDA = time.time() # to estimate the time elapsed running SEDA
	print('\nRunning chi2 fitting...')

	# load input parameters
	# input data
	fit_spectra = my_data.fit_spectra
	fit_photometry = my_data.fit_photometry
	wl_spectra = my_data.wl_spectra
	flux_spectra = my_data.flux_spectra
	eflux_spectra = my_data.eflux_spectra
	mag_phot = my_data.mag_phot
	emag_phot = my_data.emag_phot
	filter_phot = my_data.filter_phot
	R = my_data.R
	lam_R = my_data.lam_R
	distance = my_data.distance
	edistance = my_data.edistance
	N_spectra = my_data.N_spectra
	# model grid options
	model = my_model.model
	model_dir = my_model.model_dir
	Teff_range = my_model.Teff_range
	logg_range = my_model.logg_range
	R_range = my_model.R_range
	# chi2 options
	save_results = my_chi2.save_results
	scaling_free_param = my_chi2.scaling_free_param
	scaling = my_chi2.scaling
	extinction_free_param = my_chi2.extinction_free_param
	skip_convolution = my_chi2.skip_convolution
	avoid_IR_excess = my_chi2.avoid_IR_excess
	IR_excess_limit = my_chi2.IR_excess_limit
	chi2_wl_range = my_chi2.chi2_wl_range
	model_wl_range = my_chi2.model_wl_range
	wl_spectra_min = my_chi2.wl_spectra_min
	wl_spectra_max = my_chi2.wl_spectra_max
	N_datapoints = my_chi2.N_datapoints
	pickle_file = my_chi2.pickle_file

	path_seda = os.path.dirname(__file__) # gets directory path of seda

	# read additional packages depending on the input information
	if (extinction_free_param=='yes'):
		from dust_extinction.parameter_averages import F19 # extinction curve from Fitzpatrick et al. (2019)
		from dust_extinction.averages import G21_MWAvg # Gordon et al. (2021) Milky Way Average Extinction Curve (Rv=3.17)
	if (skip_convolution=='no'): 
		from spectres import spectres # resample spectra
#		from astropy.convolution import Gaussian1DKernel, convolve # kernel to convolve spectra

	# total number of data points
	if N_spectra>1 : # when multiple spectra are provided
		# function to count the total number of elements in a list of arrays
		def get_number_of_elements(list):
			count = 0
			for element in list:
				count += element.size # add the number of points in each input spectrum
			return count
		total_N = get_number_of_elements(wl_spectra) # total number of elements in the input list

	## number of data points in the spectral ranges for the fit
	#if N_spectra>1 : # when multiple spectra are provided
	#	# number of elements for the fit
	#	total_N_fit = 0
	#	for k in range(N_spectra): # for each input observed spectrum
	#		mask = (wl_spectra[k]>=chi2_wl_range[k][0]) & (wl_spectra[k]<=chi2_wl_range[k][1])
	#		total_N_fit += wl_spectra[k][mask].size

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

	# manipulate parameters to avoid errors
	# Sonora Elf Owl spectra do not cover wavelength>15 um, 
	# error: if input spectra have longer wavelengths, and error will show up when resampling the model spectra to the input spectra
	# solution: remove >15 um data points in the input spectra
	if ((model=='Sonora_Elf_Owl') & (wl_spectra_max>15)): 
		print('\nWARNING: Input spectra have longer wavelengths than the Sonora Elf Owl coverage (up to 15 um).')
		print('         Cut input spectra to be within the Sonora Elf Owl coverage.')
		print('         I will try to cut the input spectra to fit data points only up to 15 um.')
		for i in range(N_spectra):
			mask = wl_spectra[i]<14.9 # 14.9 um rather than 15 um to give a pad for the resampling with spectres
			wl_spectra[i] = wl_spectra[i][mask]
			flux_spectra[i] = flux_spectra[i][mask]
			eflux_spectra[i] = eflux_spectra[i][mask]
	
			#model_wl_range = np.array((model_wl_range[0], 15)) # range to cut model spectra up to 15 um (not necessary, it should works if it is up to longer wavelength)
			if chi2_wl_range[i][1]>=15: chi2_wl_range[i][1] = 14.9

	# read the name of the spectra from the indicated models and meeting the parameters ranges 
	out_select_model_spectra = select_model_spectra(Teff_range=Teff_range, logg_range=logg_range, model=model, model_dir=model_dir)
	spectra_name_full = out_select_model_spectra['spectra_name_full']
	spectra_name = out_select_model_spectra['spectra_name']
	
	# separate catalog and filter name from the input filter_phot array
	if (fit_photometry=='yes'):
		sur_phot = [''] * filter_phot.size # to split catalogs' name from the input filter_phot
		fil_phot = [''] * filter_phot.size # to split filters' name from the input filter_phot
		for i in range(filter_phot.size):
			sur_phot[i] = filter_phot[i].split('_')[0]
			fil_phot[i] = filter_phot[i].split('_')[1]

	# read the maximum number of data points in model spectra
	N_rows_model = Ndata_model_spectra(model)

	#+++++++++++++++++++++++++++++++++
	if (skip_convolution=='no'): # it is possible to skip the convolution of model spectra (the slowest process in the code), 
								 # only when comparing photometry and when synthetic magnitudes are precomputed
		# array for model spectra
		wl_array_model = np.zeros((len(spectra_name), N_rows_model))
		flux_array_model = np.zeros((len(spectra_name), N_rows_model))
		# array for convolved model spectra
		wl_array_model_conv = np.zeros((len(spectra_name), N_rows_model))
		flux_array_model_conv = np.zeros((len(spectra_name), N_rows_model))

		# read model spectra to the required resolution
		ini_time_model_conv = time.time() # to estimate the time elapsed doing the convolution
		print(f'\n   {len(spectra_name)} model spectra to be convolved')
		for i in range(len(spectra_name)):
			print(f'      convolution {i+1}/{len(spectra_name)}')

			out_read_model_spectrum = read_model_spectrum(spectra_name_full=spectra_name_full[i], model=model, model_wl_range=model_wl_range)
			wl_model = out_read_model_spectrum['wl_model']
			flux_model = out_read_model_spectrum['flux_model']

			# convolved spectra
			wl_model_conv = wl_model # convolved spectrum have the same wavelength data points as the original spectrum
			if N_spectra==1 : # when only one spectrum is provided
				out_convolve_spectrum = convolve_spectrum(wl=wl_model, flux=flux_model, lam_R=lam_R, R=R, disp_wl_range=chi2_wl_range[0])
				flux_model_conv = out_convolve_spectrum['flux_conv']
			if N_spectra>1 : # when multiple spectra are provided
				flux_model_conv = np.zeros(wl_model.size) # to save convolved fluxes
				for k in range(N_spectra): # for each input spectrum
					if k==0: # for the first spectrum
						mask_conv = (wl_model<=chi2_wl_range[k+1][0]) # from the minimum of the model wavelengths to the minimum of the second spectrum
					if ((k!=0) and (k!=N_spectra-1)): # for the spectra between the second and the second last ones
						mask_conv = (wl_model>=chi2_wl_range[k][0]) & (wl_model<=chi2_wl_range[k+1][0]) # from the minimum of one spectrum to the minimum of the next spectrum
					if k==N_spectra-1: # for the last spectrum
						mask_conv = (wl_model>=chi2_wl_range[k][0]) # from the minimum of the last spectrum to the maximum of the model wavelengths
	
					out_convolve_spectrum = convolve_spectrum(wl=wl_model, flux=flux_model, lam_R=lam_R[k], R=R[k], disp_wl_range=chi2_wl_range[k,:], 
																		 convolve_wl_range=np.array((wl_model[mask_conv].min(), wl_model[mask_conv].max()))) # convolve model spectrum
					flux_model_conv[mask_conv] = out_convolve_spectrum['flux_conv']

			if i==2:
				fin_time_model_conv = time.time()
				out_time_elapsed = time_elapsed(fin_time_model_conv-ini_time_model_conv)
				print(f'      elapsed time convolving the first three spectra: {out_time_elapsed[0]} {out_time_elapsed[1]}')

			# store wavelengths and fluxes for each synthetic spectrum
			wl_array_model[i,:wl_model.size] = wl_model # original wavelengths
			flux_array_model[i,:wl_model.size] = flux_model # original fluxes
			wl_array_model_conv[i,:wl_model.size] = wl_model_conv # convolved wavelengths
			flux_array_model_conv[i,:wl_model.size] = flux_model_conv # convolved fluxes

		##################################################
		# synthetic photometry from model spectra before reddening them
		if (fit_photometry=='yes'): # only if there are observed magnitudes
			## read filters' zero points
			#filter_parms_file = '/home/gsuarez/TRABAJO/PDFs/Papers/data/Filter_Transmissions/filters_properties'
			#filter_parms = ascii.read(filter_parms_file)
			#f0_filt = filter_parms['f0(Jy)'] # in Jy
			#ef0_filt = filter_parms['ef0(Jy)'] # in Jy
			#ref_filt = filter_parms['ref']
			phot_synt = np.zeros((len(spectra_name), filter_phot.size)) # array to save synthetic photometry in erg/s/cm2/A
			lambda_eff = np.zeros((len(spectra_name), filter_phot.size)) # to save the effective wavelength of each filter from every synthetic spectrum
			lambda_mean = np.zeros((len(spectra_name), filter_phot.size)) # to save the mean wavelength of each filter from every synthetic spectrum
			width_eff = np.zeros((len(spectra_name), filter_phot.size)) # to save the effective width of each filter from every synthetic spectrum
			f0_phot = np.zeros(filter_phot.size) # to save the zero flux of each filter
			ef0_phot = np.zeros(filter_phot.size) # to save the error of the zero flux of each filter
			for i in range(len(spectra_name)):
				# synthetic photometry from the model spectra in the filters with magnitudes
				for k in range(len(fil_phot)):
					if ((sur_phot[k]=='Gaia') & (fil_phot[k]=='BP')): 
						filter_response = ascii.read(path_seda+'/aux/filter_transmissions/GAIA_GAIA3.Gbp.dat')
						f0_phot[k] = 3552.01 # Jy
						ef0_phot[k] = 0.0 # Jy
					filter_response_wl = filter_response['col1'] / 1e4 # in um
					filter_response_flux = filter_response['col2']
	
					# wavelength dispersion of model spectra in the wavelength range of the filter
					#if (wl_array_model[i,:].max()>=filter_response_wl.max()): # when the synthetic spectrum covers the whole filter passband
					ind_model_filter_response = np.where((wl_array_model[i,:]>=filter_response_wl.min()) & (wl_array_model[i,:]<=filter_response_wl.max()))[0]
					wl_model_filter_response = wl_array_model[i,ind_model_filter_response] # wavelength of the model in the filter passband
					flux_model_filter_response = flux_array_model[i,ind_model_filter_response] # fluxes of the model in the filter passband
			
					wl_dis_model_filter_response = wl_model_filter_response[1:] - wl_model_filter_response[:-1] # dispersion of model spectra (um)
					if (ind_model_filter_response.size>0): # when cropped model spectra cover the photometric passbands
						wl_dis_model_filter_response = np.insert(wl_dis_model_filter_response, wl_dis_model_filter_response.size, wl_dis_model_filter_response[-1]) # add an element equal to the last row to keep the same shape as the wl_model array
			
						# synthetic photometry for each filter
						# interpolate the model spectrum wavelength points into the filter response
						# (the result is the same if the filter response wavelength points are interpolated into the model spectra; see test below)
						filter_response_flux_int = np.interp(wl_model_filter_response, filter_response_wl, filter_response_flux) # filter response for the wavelength resolution of the model spectra (dimensionless)
						
						# normalize the response curve (it was dimensionless but now it has 1/um units)
						filter_response_flux_int_norm = filter_response_flux_int / sum(filter_response_flux_int*wl_dis_model_filter_response) # 1/um
						
						# synthetic flux density
						phot_synt[i,k] = sum(flux_model_filter_response*filter_response_flux_int_norm*wl_dis_model_filter_response) # erg/s/cm2/A (but on the surface of a dwarf, not considering the distance to the target)
			
						# compute the effective wavelength and effective width of each filter
						lambda_eff[i,k] = sum(wl_model_filter_response*filter_response_flux_int*flux_model_filter_response*wl_dis_model_filter_response) / sum(filter_response_flux_int*flux_model_filter_response*wl_dis_model_filter_response) # um
						lambda_mean[i,k] = sum(wl_model_filter_response*filter_response_flux_int*wl_dis_model_filter_response) / sum(filter_response_flux_int*wl_dis_model_filter_response) # um
						width_eff[i,k] = sum(filter_response_flux_int*wl_dis_model_filter_response) / filter_response_flux_int.max() # um
			
						##----------------------------
						## test
						## wavelength dispersion of the filter response
						#wl_dis_filter_response = filter_response_wl[1:] - filter_response_wl[:-1]
						#wl_dis_filter_response = np.insert(wl_dis_filter_response, wl_dis_filter_response.size, wl_dis_filter_response[-1]) # add an element equal to the last row to keep the same shape as the wl_model array
			
						## synthetic photometry for each filter
						## interpolate the filter response wavelength points into the model spectrum
						#flux_model_filter_response_int = np.interp(filter_response_wl, wl_model_filter_response, flux_model_filter_response) # model spectrum for the wavelength points of the filter response (dimensionless)
			
						## normalize the response curve (it was dimensionless but now it has 1/um units)
						#filter_response_flux_norm = filter_response_flux / sum(filter_response_flux*wl_dis_filter_response) # 1/um
			
						## synthetic flux density
						##print('\n',phot_synt[i,k])
						#phot_synt[i,k] = sum(flux_model_filter_response_int*filter_response_flux_norm*wl_dis_filter_response) # erg/s/cm2/A (but on the surface of a dwarf, not considering the distance to the target)
						##print(phot_synt[i,k])
			
						## compute the effective wavelength and effective width of each filter
						#lambda_eff[i,k] = sum(filter_response_wl*filter_response_flux*flux_model_filter_response_int*wl_dis_filter_response) / sum(filter_response_flux*flux_model_filter_response_int*wl_dis_filter_response) # um
						#lambda_mean[i,k] = sum(filter_response_wl*filter_response_flux*wl_dis_filter_response) / sum(filter_response_flux*wl_dis_filter_response) # um
						#width_eff[i,k] = sum(filter_response_flux*wl_dis_filter_response) / filter_response_flux.max() # um
						##----------------------------
	
	# when synthetic magnitudes are already obtained
	if (skip_convolution=='yes'): # it makes the code much faster when comparing photometry only, because synthetic magnitudes are already obtained for the filter of interest
		print('\n'+str(len(spectra_name))+' model spectra to be compared')
		# read synthetic magnitudes from all BT-Settl synthetic spectra considering select filters
		phot_synt = np.zeros((len(spectra_name), len(fil_phot)))
		lambda_eff = np.zeros((len(spectra_name), len(fil_phot)))
		width_eff = np.zeros((len(spectra_name), len(fil_phot)))
		f0_phot = np.zeros(len(fil_phot)) # to save the zero flux of each filter
		ef0_phot = np.zeros(len(fil_phot)) # to save the error of the zero flux of each filter
		N = 10000
		if (model_label=='best'):
			wl_array_model_conv = np.zeros((len(spectra_name), N))
			flux_array_model_conv = np.zeros((len(spectra_name), N))
		for i in range(len(spectra_name)):
			for j in range(mag_phot.size):
				phot_synt[i,j] = np.loadtxt(path_seda+'/aux/synthetic_photometry/'+model+'/synthetic_photometry/'+spectra_name[i]+'_synthetic_photometry_'+sur_phot[j]+'_'+fil_phot[j])
				lambda_eff[i,j] = np.loadtxt(path_seda+'/aux/synthetic_photometry/'+model+'/filter_params/'+spectra_name[i]+'_lambda_eff_'+sur_phot[j]+'_'+fil_phot[j])
				width_eff[i,j] = np.loadtxt(path_seda+'/aux/synthetic_photometry/'+model+'/filter_params/'+spectra_name[i]+'_width_eff_'+sur_phot[j]+'_'+fil_phot[j])
				if (i==0):
					f0_phot[j] = np.loadtxt(path_seda+'/aux/synthetic_photometry/'+model+'/filter_params/f0_phot_'+sur_phot[j]+'_'+fil_phot[j])
					ef0_phot[j] = np.loadtxt(path_seda+'/aux/synthetic_photometry/'+model+'/filter_params/ef0_phot_'+sur_phot[j]+'_'+fil_phot[j])
			if (model_label=='best'): # to avoid reading the convolved spectra when all the selected grid (model_label=='all') is considered
				wl_array_model_conv_all = np.loadtxt(path_seda+'/aux/synthetic_photometry/'+model+'/convolved_spectra/'+spectra_name[i]+'_wl_convolved.dat')
				flux_array_model_conv_all = np.loadtxt(path_seda+'/aux/synthetic_photometry/'+model+'/convolved_spectra/'+spectra_name[i]+'_flux_convolved.dat')

				# cut the convolved model spectra to the range of the plot
				ind = np.where((wl_array_model_conv_all>=(model_wl_range[0])) & (wl_array_model_conv_all<=model_wl_range[1]))
				wl_array_model_conv[i,ind] = wl_array_model_conv_all[ind]
				flux_array_model_conv[i,ind] = flux_array_model_conv_all[ind]
	
	# mean effective wavelength and mean effective (half) width
	if (fit_photometry=='yes'): # only if there are observed magnitudes
		lambda_eff_mean = np.zeros(len(mag_phot))
		width_eff_mean = np.zeros(len(mag_phot))
		for i in range(len(mag_phot)):
			lambda_eff_mean[i] = np.mean(lambda_eff[:,i])
			width_eff_mean[i] = np.mean(width_eff[:,i])
	#		if (lambda_eff[lambda_eff[:,i]>0,i].size > 0): # only when synthetic photometry was derived
	#			lambda_eff_mean[i] = np.mean(lambda_eff[lambda_eff[:,i]>0,i]) # exclude spectra for which no values were estimated because the spectra do not cover entirely the filter passband
	#			width_eff_mean[i] = np.mean(width_eff[width_eff[:,i]>0,i]) # exclude spectra for which no values were estimated because the spectra do not cover entirely the filter passband
	## remove filter parameters for the passband without synthetic photometry
	#lambda_eff_mean = lambda_eff_mean[lambda_eff_mean>0]
	#width_eff_mean = width_eff_mean[width_eff_mean>0]
	
	# convert photometry to erg/s/cm2/A
	if (fit_photometry=='yes'): # only if there are observed magnitudes
		# first mag to Jy
		f_phot_nu = f0_phot * 10**(-mag_phot/2.5) # in Jy
		ef_phot_nu = 10**(-mag_phot/2.5)*np.sqrt(ef0_phot**2+((-1*np.log(10)/2.5)*f0_phot*emag_phot)**2) # in Jy
		# Jy to erg/s/cm2/A
		#f_phot = (f_phot_nu / (lambda_eff_mean*1e-4)**2) * 3e-21 # in erg/s/cm2/A
		f_phot = f_phot_nu / (3.33564095e+04*(lambda_eff_mean*1e4)**2) # in erg/s/cm2/A
		ef_phot = f_phot * ef_phot_nu/f_phot_nu
	
	# print a message when model spectra don't cover the whole input spectra wavelength range
	if fit_spectra and not fit_photometry:
		for i in range(len(spectra_name)): # for each model spectrum
			mask_nozero = flux_array_model_conv[i,:]!=0 # non-zero fluxes in each convolved model spectrum
			if (wl_spectra_min<min(wl_array_model_conv[i,:][mask_nozero])): # when the shortest wavelength in the input spectra are not covered by models
				print('\n'+spectra_name[i]+' DOES NOT COVER THE SHORTEST WAVELENGTHS IN THE INPUT SPECTRA')
			if (wl_spectra_max>max(wl_array_model_conv[i,:][mask_nozero])): # when the longest wavelength in the input spectra are not covered by models
				print('\n'+spectra_name[i]+' DOES NOT COVER THE LONGEST WAVELENGTHS IN THE INPUT SPECTRA')

	##################################################
	# chi square test
	print('\n   minimizing chi2...')
	if fit_spectra:
		# resample the convolved model spectra to have the same wavelength elements as the observed spectra
		wl_array_model_conv_resam = np.zeros((len(spectra_name), N_datapoints))
		flux_array_model_conv_resam = np.zeros((len(spectra_name), N_datapoints))
		for i in range(len(spectra_name)):
			mask_flux = wl_array_model_conv[i,:]!=0 # to avoid zeros in some synthetic spectra because not all of them have the same wavelength points (they are read in an array with a fixed length) 
			mask_wl = ((wl_spectra>1.02*wl_array_model_conv[i,mask_flux].min()) & (wl_spectra<0.98*wl_array_model_conv[i,mask_flux].max())) # to avoid data points (considering a padding) out of the model coverage (could be that models cover a shorter wavelength range than the indicated one by model_wl_range)
			wl_array_model_conv_resam[i,mask_wl] = wl_spectra[mask_wl]
			flux_array_model_conv_resam[i,mask_wl] = spectres(wl_array_model_conv_resam[i,mask_wl], wl_array_model_conv[i,mask_flux], flux_array_model_conv[i,mask_flux])
		## when input spectra list is not transformed into numpy arrays with the spectra next to each other)
		#wl_array_model_conv_resam = np.zeros((len(spectra_name), N_datapoints))
		#flux_array_model_conv_resam = np.zeros((len(spectra_name), N_datapoints))
		#for i in range(len(spectra_name)): # for each convolved model spectrum
		#	mask_flux = flux_array_model_conv[i,:]!=0 # to avoid zeros in some synthetic spectra because not all of them have the same wavelength points (they are read in an array with a fixed length) 
		#	for k in range(N_spectra): # for each input spectra
		#		# select only wavelengths (considering a padding) from each input spectrum covered by each model spectrum
		#		# (could be that models cover a narrower wavelength range than the one by the input spectra)
		#		mask_wl = ((wl_spectra[k]>1.02*wl_array_model_conv[i,mask_flux].min()) & (wl_spectra[k]<0.98*wl_array_model_conv[i,mask_flux].max()))
		#		ind_ini = k * wl_spectra[k][mask_wl].size # check this with the one at 'select spectra data points to obtain chi2'
		#		ind_fin = ind_ini + mask_wl.size
		#		wl_array_model_conv_resam[i,ind_ini:ind_fin] = wl_spectra[k][mask_wl]
		#		flux_array_model_conv_resam[i,ind_ini:ind_fin] = spectres(wl_array_model_conv_resam[i,ind_ini:ind_fin], wl_array_model_conv[i,mask_flux], flux_array_model_conv[i,mask_flux])

		# OLD WAY
		#for i in range(len(spectra_name)):
		#	wl_array_model_conv_resam[i,:] = wl_spectra
		#	mask = wl_array_model_conv[i,:]!=0 # to avoid zeros in some synthetic spectra because not all of them have the same wavelength points (they are read in an array with a fixed length) 
		#	flux_array_model_conv_resam[i,:] = spectres(wl_array_model_conv_resam[i,:], wl_array_model_conv[i,mask], flux_array_model_conv[i,mask])

	# find the scaling factor and extinction value that minimize chi2 of each model spectrum
	# read extinction curve
	if (extinction_free_param=='yes'):
		ext_F19 = F19(Rv=3.1) # Fitzpatrick et al. (2019) extinction curve with Rv=3.1 for the optical and NIR (0.3-8.7 um)
		ext_F19_min = 0.3 # um
		ext_G21_MWAvg = G21_MWAvg() # Gordon et al. (2021) Milky Way Average Extinction Curve (Rv=3.17) for the NIR and MIR (1-32 um)
		G21_MWAvg_max = 32 # um
		wl_ext_cut = 3.0 # (um) wavelength point to merge both extinction curves to cover from the optical to the MIR

	# when using spectra only
	if fit_spectra and not fit_photometry:
		if (extinction_free_param=='yes'):
			ext_F19_spectra = ext_F19(wl_spectra[wl_spectra<=wl_ext_cut]*u.micron) # A_lambda/Av curve for each spectrum data point shorter than wl_ext_cut
			ext_G21_MWAvg_spectra = ext_G21_MWAvg(wl_spectra[wl_spectra>wl_ext_cut]*u.micron) # A_lambda/Av curve for each spectrum data point longer than wl_ext_cut
			ext_spectra = np.concatenate((ext_F19_spectra, ext_G21_MWAvg_spectra))
	
		# select spectra data points to obtain chi2
		if N_spectra==1 : # when only one spectrum is provided
			ind_chi2 = np.where((wl_spectra>=chi2_wl_range[0][0]) & (wl_spectra<=chi2_wl_range[0][1]))[0]
		if N_spectra>1 : # when multiple spectra are provided
			# the desired data points for the fit are not necessary all the spectra data points 
			ind_chi2 = np.full(wl_spectra.size, False, dtype=bool) # to save the indices with wavelengths within the ranges for the fit
			l = 0
			for k in range(N_spectra): # for each input observed spectrum
				index_min = l
				index_max = wl_spectra_list[k].size+l
				mask = (wl_spectra[index_min:index_max]>=chi2_wl_range[k][0]) & (wl_spectra[index_min:index_max]<=chi2_wl_range[k][1]) # input wavelengths meeting the wavelength ranges for the fit
				ind_chi2[index_min:index_max][mask] = True # select the indices in the mask
				l += wl_spectra_list[k].size

		# ATTEMPT TO SELECT A DIFFERENT ind_chi2 RANGE FOR EACH MODEL SPECTRUM
		# THIS BECAUSE MODEL SPECTRA (E.G. Sonora Cholla) has different wavelength coverage, so a different wavelength range for the fits will be convenient.
		# the lines below works well, but then it is necessary to modify the rest of code to use a ind_chi2 array that is a matrix rather than a vector
		## select spectra data points to obtain chi2
		#if N_spectra==1 : # when only one spectrum is provided
		#	ind_chi2 = np.where((wl_spectra>=chi2_wl_range[0]) & (wl_spectra<=chi2_wl_range[1]))[0]
		#if N_spectra>1 : # when multiple spectra are provided
		#	# the desired data points for the fit are not necessary all the spectra data points 
		#	ind_chi2 = np.full((len(spectra_name), wl_spectra.size), False, dtype=bool) # to save the indices with wavelengths within the ranges for the fit
		#	for i in range(len(spectra_name)): # for each model spectrum
		#		l = 0
		#		for k in range(N_spectra): # for each input observed spectrum
		#			index_min = l # first data point of the k observed spectrum
		#			index_max = wl_spectra_list[k].size+l # last data point of the k observed spectrum

		#			mask_nozero = flux_array_model_conv_resam[i,:]!=0 # non-zero fluxes in each convolved, resampled model spectrum
		#			if (min(wl_spectra[index_min:index_max])<min(wl_array_model_conv_resam[i,:][mask_nozero])): # when a input spectrum has wavelength shorter than the minimum wavelength in each model spectrum
		#				if (min(wl_array_model_conv_resam[i,:][mask_nozero]) < chi2_wl_range[k][0]): # when the minimum of the chi2_wl_range is longer than the minimum of the model spectrum
		#					mask = (wl_spectra[index_min:index_max]>=chi2_wl_range[k][0]) & (wl_spectra[index_min:index_max]<=chi2_wl_range[k][1]) # indices from the array with all input wavelengths meeting the wavelength ranges for the fit
		#				else: # when the minimum of the chi2_wl_range is shorter than the minimum of the model spectrum
		#					mask = (wl_spectra[index_min:index_max]>=min(wl_array_model_conv_resam[i,:][mask_nozero])) & (wl_spectra[index_min:index_max]<=chi2_wl_range[k][1]) # indices from the array with all input wavelengths meeting the wavelength ranges for the fit
		#			else: # when each input spectrum as full wavelength coverage by the model spectra
		#				mask = (wl_spectra[index_min:index_max]>=chi2_wl_range[k][0]) & (wl_spectra[index_min:index_max]<=chi2_wl_range[k][1]) # indices from the array with all input wavelengths meeting the wavelength ranges for the fit
		#			ind_chi2[i,index_min:index_max][mask] = True # select the indices in the mask
		#			l += wl_spectra_list[k].size # to iterate to next input spectrum

		# data and model arrays for the fit
		data_fit = flux_spectra[ind_chi2] # select fluxes
		edata_fit = eflux_spectra[ind_chi2] # select flux errors
		model_fit = flux_array_model_conv_resam[:,ind_chi2] # model data corresponding to wavelength of the selected fluxes
		if (extinction_free_param=='yes'): extinction_curve_fit = ext_spectra[ind_chi2] # extinction curve for each select flux
		weight_fit = np.repeat(1, data_fit.size) # (relevant when using spectra and photometry) same weight for all spectra points. It is to give a significant contribution to photometric data points compared to all spectra data points; it is a weight out of the summation in the chi square definition, not the flux error that divides the flux differences within the summation)
		## weights define by the wavelength dispersion of the spectra points
		#Dellam_spectrum = wl_spectra[1:] - wl_spectra[:-1] # dispersion of data points
		#Dellam_spectrum = np.insert(Dellam_spectrum, Dellam_spectrum.size, Dellam_spectrum[-1]) # add an element equal to the last row to keep the same shape as the wl_spectra array
		#weight_fit = Dellam_spectrum[ind_chi2] # weights for the spectra

	# when using photometry only
	if ((fit_spectra=='no') & (fit_photometry=='yes')):
		if (extinction_free_param=='yes'):
			ext_F19_ref_phot = ext_F19(lambda_eff_mean[lambda_eff_mean<=wl_ext_cut]*u.micron) # A_lambda/Av curve for filter passband shorter than wl_ext_cut
			ext_G21_MWAvg_ref_phot = ext_G21_MWAvg(lambda_eff_mean[lambda_eff_mean>wl_ext_cut]*u.micron) # A_lambda/Av curve for each spectrum data point longer than wl_ext_cut
			ext_ref_phot = np.concatenate((ext_F19_ref_phot, ext_G21_MWAvg_ref_phot))
	
		if (avoid_IR_excess=='yes'): # avoid IR excesses
			ind_chi2 = np.where((emag_phot!=0) & (lambda_eff_mean<IR_excess_limit))[0] # observed magnitudes for the fit (avoid magnitudes with null errors because they are usually upper limit magnitudes, and magnitudes that could be affected by IR excesses)
		else: # do not avoid regions of possible IR excesses
			ind_chi2 = np.where(emag_phot!=0)[0] # observed magnitudes for the fit (avoid magnitudes with null errors because they are usually upper limit magnitudes)
		data_fit = f_phot[ind_chi2] # select magnitudes
		edata_fit = ef_phot[ind_chi2] # select magnitude errors
		model_fit = phot_synt[:,ind_chi2] # model data corresponding to the select magnitudes
		if (extinction_free_param=='yes'): extinction_curve_fit = ext_ref_phot[ind_chi2] # extinction for each select magnitude
		#weight_fit = width_eff_mean[ind_chi2]
		weight_fit = np.repeat(1, data_fit.size) # (relevant when using spectra and photometry) same weight for all spectra points. It is to give a significant contribution to photometric data points compared to all spectra data points; it is a weight out of the summation in the chi square definition, not the flux error that divides the flux differences within the summation)
	
	# when using spectra and photometry
	if ((fit_spectra=='yes') & (fit_photometry=='yes')):
		# extinction for the spectra data points
		if (extinction_free_param=='yes'):
			ext_F19_spectra = ext_F19(wl_spectra[wl_spectra<=wl_ext_cut]*u.micron) # A_lambda/Av curve for each spectrum data point shorter than wl_ext_cut
			ext_G21_MWAvg_spectra = ext_G21_MWAvg(wl_spectra[wl_spectra>wl_ext_cut]*u.micron) # A_lambda/Av curve for each spectrum data point longer than wl_ext_cut
			ext_spectra = np.concatenate((ext_F19_spectra, ext_G21_MWAvg_spectra))
		# select spectra data points to obtain chi2
		if N_spectra==1 : # when only one spectrum is provided
			#ind_chi2_spec = np.where((wl_spectra>chi2_wl_range[0]) & (wl_spectra<chi2_wl_range[1]) & (flux_spectra>1))
			ind_chi2_spec = np.where((wl_spectra>=chi2_wl_range[0]) & (wl_spectra<=chi2_wl_range[1]))[0]
		if N_spectra>1 : # when multiple spectra are provided
			ind_chi2_spec = np.full(wl_spectra.size, False, dtype=bool) # to save the indices with wavelengths within the ranges for the fit
			l = 0
			for k in range(N_spectra): # for each input observed spectrum
				index_min = l
				index_max = wl_spectra_list[k].size+l
				mask = (wl_spectra[index_min:index_max]>=chi2_wl_range[k][0]) & (wl_spectra[index_min:index_max]<=chi2_wl_range[k][1]) # indices from the array with all input wavelengths meeting the wavelength ranges for the fit
				ind_chi2[index_min:index_max][mask] = True # select the indices in the mask
				l += wl_spectra_list[k].size
	
		# extinction for the photometry
		if (extinction_free_param=='yes'):
			ext_F19_ref_phot = ext_F19(lambda_eff_mean[lambda_eff_mean<=wl_ext_cut]*u.micron) # A_lambda/Av curve for filter passband shorter than wl_ext_cut
			ext_G21_MWAvg_ref_phot = ext_G21_MWAvg(lambda_eff_mean[lambda_eff_mean>wl_ext_cut]*u.micron) # A_lambda/Av curve for each spectrum data point longer than wl_ext_cut
			ext_ref_phot = np.concatenate((ext_F19_ref_phot, ext_G21_MWAvg_ref_phot))
		# select magnitudes to obtain chi2
		if (avoid_IR_excess=='yes'): # avoid IR excesses
			ind_chi2_phot = np.where((emag_phot!=0) & (lambda_eff_mean<IR_excess_limit))[0] # observed magnitudes for the fit (avoid magnitudes with null errors because they are usually upper limit magnitudes, and magnitudes that could be affected by IR excesses)
		else: # do not avoid regions of possible IR excesses
			ind_chi2_phot = np.where(emag_phot!=0)[0] # observed magnitudes for the fit (avoid magnitudes with null errors because they are usually upper limit magnitudes)
	
		ind_chi2 = np.concatenate((ind_chi2_spec, ind_chi2_phot)) # indices with the selected spectroscopic and photometric data points for the fit
		# data and model arrays for the fit
		data_fit = np.concatenate((flux_spectra[ind_chi2_spec], f_phot[ind_chi2_phot])) # selected data points
		edata_fit = np.concatenate((eflux_spectra[ind_chi2_spec], ef_phot[ind_chi2_phot])) # errors of the selected data points
		model_fit = np.concatenate((flux_array_model_conv_resam[:,ind_chi2_spec].T, phot_synt[:,ind_chi2_phot].T)).T # model fluxes corresponding to the selected data points 
		if (extinction_free_param=='yes'): extinction_curve_fit = np.concatenate((ext_spectra[ind_chi2_spec], ext_ref_phot[ind_chi2_phot])) # extinction for each selected data point
	
		# weight for each data point
		# (it is to give a significant contribution to photometric data points compared to all spectra data points)
		# (it is a weight out of the summation in the chi square definition, not the flux error that divides the flux differences within the summation)
		weight_phot = width_eff_mean[ind_chi2_phot] # weight for the photometry equal to each filter bandpass
		Dellam_spectrum = wl_spectra[1:] - wl_spectra[:-1] # dispersion of data points
		Dellam_spectrum = np.insert(Dellam_spectrum, Dellam_spectrum.size, Dellam_spectrum[-1]) # add an element equal to the last row to keep the same shape as the wl_spectra array
		weight_spec = Dellam_spectrum[ind_chi2_spec] # weights for the spectra
		weight_fit = np.concatenate((weight_spec, weight_phot)) # (relevant when using spectra and photometry)
		#weight_fit = np.repeat(1, data_fit.size) # same weight for all spectra and magnitudes
	
	# to save main parameters from the fit to each model spectrum
	scaling_fit = np.zeros(len(spectra_name))
	escaling_fit = np.zeros(len(spectra_name))
	Av_fit = np.zeros(len(spectra_name))
	eAv_fit = np.zeros(len(spectra_name))
	iterations_fit = np.zeros(len(spectra_name))
	chi2_fit = np.zeros(len(spectra_name))
	chi2_red_fit = np.zeros(len(spectra_name))
	chi2_wl_fit = np.zeros((len(spectra_name), wl_spectra[ind_chi2].size))
	chi2_red_wl_fit = np.zeros((len(spectra_name), wl_spectra[ind_chi2].size))	
	if (extinction_free_param=='yes'): # to save the reddened model spectra and synthetic magnitudes
		if (skip_convolution=='no'): # model spectra are read only when the convolution is done
			flux_array_model_red = np.zeros((len(spectra_name), flux_array_model.shape[1])) # for scaled fluxes of original-resolution reddened model spectra
		if (model_label=='best'): # convolved spectra are read only for the best fits
			flux_array_model_conv_red = np.zeros((len(spectra_name), flux_array_model_conv.shape[1])) # for scaled fluxes of convolved reddened model spectra
		if (fit_spectra=='yes'):
			flux_array_model_conv_resam_red = np.zeros((len(spectra_name), flux_array_model_conv_resam.shape[1])) # for scaled fluxes of resampled, convolved reddened model spectra
		if (fit_photometry=='yes'): # only if there are observed magnitudes
			phot_synt_red = np.zeros((len(spectra_name), phot_synt.shape[1])) # for synthetic fluxes from each reddened model spectrum considering the photometric passbands

	# do the fit
	for i in range(len(spectra_name)):
		# add the free parameters as params
		params = Parameters()
		if (extinction_free_param=='no'): params.add('extinction', value=0, vary=False) # fixed parameter
		if (extinction_free_param=='yes'): params.add('extinction', value=0) # free parameter
		if (scaling_free_param=='no'): params.add('scaling', value=scaling, vary=False) # fixed parameter
		if (scaling_free_param=='yes'): params.add('scaling', value=1e-20) # free parameter

		if (extinction_free_param=='no'): # when extinction is not fitted, the extinction_curve_fit parameter is not generated, so define it as zero
			extinction_curve_fit = 0

		# minimize chi square
		minner = Minimizer(chi_square, params, fcn_args=(data_fit, edata_fit, model_fit[i,:], extinction_curve_fit, weight_fit))

		out_lmfit = minner.minimize(method='leastsq') # 'leastsq': Levenberg-Marquardt (default)
		#print(out_lmfit. redchi, out_lmfit. bic)
		iterations_fit[i] = out_lmfit.nfev # number of function evaluations in the fit
		Av_fit[i] = out_lmfit.params['extinction'].value
		eAv_fit[i] = out_lmfit.params['extinction'].stderr # half the difference between the 15.8 and 84.2 percentiles of the PDF
		scaling_fit[i] = out_lmfit.params['scaling'].value
		escaling_fit[i] = out_lmfit.params['scaling'].stderr # half the difference between the 15.8 and 84.2 percentiles of the PDF
		chi2_wl_fit[i,:] = out_lmfit.residual**2 # chi square for each data point from lmfit
		chi2_red_wl_fit[i,:] = out_lmfit.residual**2 / out_lmfit.nfree # reduced chi square for each data point from lmfit
		chi2_fit[i] = out_lmfit.chisqr # resulting chi square from lmfit
		chi2_red_fit[i] = out_lmfit.redchi # resulting reduced chi square from lmfit
		
		# scale model fluxes by the values that minimizes chi2
		flux_array_model_conv[i,:] = flux_array_model_conv[i,:] * scaling_fit[i]
		if (skip_convolution=='no'): # model spectra are read only when the convolution is done
			flux_array_model[i,:] = flux_array_model[i,:] * scaling_fit[i]
		if fit_spectra:
			flux_array_model_conv_resam[i,:] = flux_array_model_conv_resam[i,:] * scaling_fit[i]
		if fit_photometry: # only if there are observed magnitudes
			phot_synt[i,:] = phot_synt[i,:] * scaling_fit[i]
	
		#-----------------------------------------------
		# redden model fluxes and synthetic magnitudes
		if (extinction_free_param=='yes'):
			# redden array_model spectra
			if (skip_convolution=='no'): # model spectra are read only when the convolution is done
				# extinction as a function of wavelength for model spectra
				ext_array_model = np.zeros(wl_array_model[0,:].size) # to keep same array size
				# extinction for the array_model spectra
				mask_ext_F19_array_model = (wl_array_model[0,:]>0) & (wl_array_model[0,:]<=wl_ext_cut)
				ext_array_model[mask_ext_F19_array_model] = ext_F19(wl_array_model[0,mask_ext_F19_array_model]*u.micron) # A_lambda/Av curve for each model spectrum data point smaller than wl_ext_cut
				mask_ext_G21_MWAvg_array_model = wl_array_model[0,:]>wl_ext_cut
				ext_array_model[mask_ext_G21_MWAvg_array_model] = ext_G21_MWAvg(wl_array_model[0,mask_ext_G21_MWAvg_array_model]*u.micron) # A_lambda/Av curve for each model spectrum data point longer than wl_ext_cut
		
				flux_array_model_red[i,:] = flux_array_model[i,:] * 10**(-Av_fit[i]*ext_array_model/2.5) # reddened spectra
		
			# redden array_model_conv spectra
			if (model_label=='best'):
				# extinction as a function of wavelength for model spectra
				ext_array_model_conv = np.zeros(wl_array_model_conv[0,:].size) # to keep same array size
				# extinction for the array_model_conv spectra
				mask_ext_F19_array_model_conv = (wl_array_model_conv[0,:]>0) & (wl_array_model_conv[0,:]<=wl_ext_cut)
				ext_array_model_conv[mask_ext_F19_array_model_conv] = ext_F19(wl_array_model_conv[0,mask_ext_F19_array_model_conv]*u.micron) # A_lambda/Av curve for each model spectrum data point smaller than wl_ext_cut
				mask_ext_G21_MWAvg_array_model_conv = (wl_array_model_conv[0,:]>wl_ext_cut) & (wl_array_model_conv[0,:]<G21_MWAvg_max)
				ext_array_model_conv[mask_ext_G21_MWAvg_array_model_conv] = ext_G21_MWAvg(wl_array_model_conv[0,mask_ext_G21_MWAvg_array_model_conv]*u.micron) # A_lambda/Av curve for each model spectrum data point longer than wl_ext_cut
	
				flux_array_model_conv_red[i,:] = flux_array_model_conv[i,:] * 10**(-Av_fit[i]*ext_array_model_conv/2.5) # reddened spectra

			# redden array_model_conv_resam spectra
			if (fit_spectra=='yes'):
				# extinction as a function of wavelength for model spectra
				ext_array_model_conv_resam = np.zeros(wl_array_model_conv_resam[0,:].size) # to keep same array size
				# extinction for the array_model_conv_resam spectra
				mask_ext_F19_array_model_conv_resam = (wl_array_model_conv_resam[0,:]>0) & (wl_array_model_conv_resam[0,:]<=wl_ext_cut)
				ext_array_model_conv_resam[mask_ext_F19_array_model_conv_resam] = ext_F19(wl_array_model_conv_resam[0,mask_ext_F19_array_model_conv_resam]*u.micron) # A_lambda/Av curve for each model spectrum data point smaller than wl_ext_cut
				mask_ext_G21_MWAvg_array_model_conv_resam = wl_array_model_conv_resam[0,:]>wl_ext_cut
				ext_array_model_conv_resam[mask_ext_G21_MWAvg_array_model_conv_resam] = ext_G21_MWAvg(wl_array_model_conv_resam[0,mask_ext_G21_MWAvg_array_model_conv_resam]*u.micron) # A_lambda/Av curve for each model spectrum data point longer than wl_ext_cut
	
				flux_array_model_conv_resam_red[i,:] = flux_array_model_conv_resam[i,:] * 10**(-Av_fit[i]*ext_array_model_conv_resam/2.5) # reddened spectra
	
			# redden synthetic magnitudes
			if (fit_photometry=='yes'): # only if there are observed magnitudes
				phot_synt_red[i,:] = phot_synt[i,:] * 10**(-Av_fit[i]*ext_ref_phot/2.5) # reddened photometric magnitudes
	
	# radius from the scaling factor and cloud distance
	if (distance!=None): # derive radius only if a distance is provided
		radius = np.sqrt(scaling_fit) * distance*3.086e13 # in km
		eradius = radius * np.sqrt((edistance/distance)**2 + (escaling_fit/(2*scaling_fit))**2)
		radius = radius / 695510. # in Rsun
		eradius = eradius / 695510. # in Rsun
		radius = radius / 0.102763 # in Rjup
		eradius = eradius / 0.102763 # in Rjup

#	# start dictionary with some output parameters from the chi square minimization
#	# sort with respect to reduced chi2
#	ind = np.argsort(chi2_red_fit)
#	spectra_name = spectra_name[ind]
#	Teff_range = Teff_range[ind]
#	logg_range = logg_range[ind]
##	I AM HERE sorting the output wrt chi2

	out_chi2 = {'model': model, 'spectra_name_full': spectra_name_full, 'spectra_name': spectra_name, 'Teff_range': Teff_range, 'logg_range': logg_range, 'R': R, 'lam_R': lam_R, 
	 'chi2_wl_range': chi2_wl_range, 'N_rows_model': N_rows_model, 'out_lmfit': out_lmfit, 
	 'iterations_fit': iterations_fit, 'Av_fit': Av_fit, 'eAv_fit': eAv_fit, 'scaling_fit': scaling_fit, 'escaling_fit': escaling_fit, 
	 'chi2_wl_fit': chi2_wl_fit, 'chi2_red_wl_fit': chi2_red_wl_fit, 'chi2_fit': chi2_fit, 'chi2_red_fit': chi2_red_fit, 'weight_fit': weight_fit}

	# add more output parameters depending on the input information
#	if (skip_convolution=='no'):
#		#out_chi2['wl_array_model']= wl_array_model
#		#out_chi2['flux_array_model'] = flux_array_model
#		#out_chi2['wl_array_model_conv'] = wl_array_model_conv
#		#out_chi2['flux_array_model_conv'] = flux_array_model_conv
#		out_chi2['wl_array_model']= wl_array_model[:,wl_array_model[0,:]>0]
#		out_chi2['flux_array_model'] = flux_array_model[:,wl_array_model[0,:]>0]
#		out_chi2['wl_array_model_conv'] = wl_array_model_conv[:,wl_array_model[0,:]>0]
#		out_chi2['flux_array_model_conv'] = flux_array_model_conv[:,wl_array_model[0,:]>0]
#	if (skip_convolution=='yes') & (model_label=='best')): # when reading the best fits to the data when predetermined synthetic photometry is used
#		out_chi2['wl_array_model_conv'] = wl_array_model_conv
#		out_chi2['flux_array_model_conv'] = flux_array_model_conv
	if fit_spectra: # when only spectra are used in the fit
		#mask_no_null = wl_array_model_conv[0,:]!=0
		#out_chi2['wl_array_model_conv'] = wl_array_model_conv[:,mask_no_null]
		#out_chi2['flux_array_model_conv'] = flux_array_model_conv[:,mask_no_null]
		out_chi2['wl_array_model_conv_resam'] = wl_array_model_conv_resam[:,ind_chi2] # only in the fitted regions
		out_chi2['flux_array_model_conv_resam'] = flux_array_model_conv_resam[:,ind_chi2] # only in the fitted regions
	if fit_photometry: # when photometry is used in the fit
		out_chi2['lambda_eff_mean'] = lambda_eff_mean
		out_chi2['width_eff_mean'] = width_eff_mean
		out_chi2['f_phot'] = f_phot
		out_chi2['ef_phot'] = ef_phot
		out_chi2['phot_synt'] = phot_synt
		if (extinction_free_param=='yes'):
			out_chi2['phot_synt_red'] = phot_synt_red
	if (distance!=None): # when a radius was obtained
		out_chi2['R_range'] = R_range
		out_chi2['distance'] = distance
		out_chi2['edistance'] = edistance
		out_chi2['radius'] = radius
		out_chi2['radius'] = radius
		out_chi2['eradius'] = eradius
	#if N_spectra>1 : # when multiple spectra are provided the function returns flat arrays with the input wavelengths, fluxes, and flux errors used in the fit
	# return observed fluxes within the fit ranges
	out_chi2['wl_array_data'] = wl_spectra[ind_chi2]
	out_chi2['flux_array_data'] = flux_spectra[ind_chi2]
	out_chi2['eflux_array_data'] = eflux_spectra[ind_chi2]

	# obtain residuals in linear and logarithmic-scale for fluxes
	flux_residuals = np.zeros((len(spectra_name), len(flux_spectra[ind_chi2])))
	logflux_residuals = np.zeros((len(spectra_name), len(flux_spectra[ind_chi2])))
	for i in range(len(spectra_name)):
		flux_residuals[i,:]	= out_chi2['flux_array_model_conv_resam'][i,:] - out_chi2['flux_array_data']
		mask_pos = out_chi2['flux_array_data']>0 # mask to avoid negative fluxes to obtain the logarithm
		logflux_residuals[i,mask_pos] = np.log10(out_chi2['flux_array_model_conv_resam'][i,:][mask_pos]) - np.log10(out_chi2['flux_array_data'][mask_pos])
	out_chi2['flux_residuals'] = flux_residuals
	out_chi2['logflux_residuals'] = logflux_residuals

#	self.wl_array_model_conv_resam = out_chi2['wl_array_model_conv_resam']
#	self.logflux_residuals = logflux_residuals

	# separate physical parameters from each model spectrum name
	out_separate_params = separate_params(spectra_name=spectra_name, model=model)
	# add model spectra parameters to the output dictionary
	out_chi2.update(out_separate_params)

	# save some information
	if (save_results=='yes'): # save table with model spectra sorted by chi square
		# save output dictionary as pickle
		with open(f'{model}_chi2_minimization.pickle', 'wb') as file:
			# serialize and write the variable to the file
			pickle.dump(out_chi2, file)
		print('      chi square minimization results saved successfully')

		# save table with model spectra names sorted by chi square along with the parameters from each spectrum
		out_save_params = save_params(dict_for_table=out_chi2)

	print('\nChi square fit ran successfully')
	fin_time_SEDA = time.time()
	out_time_elapsed = time_elapsed(fin_time_SEDA-ini_time_SEDA)
	print(f'   elapsed time running SEDA: {out_time_elapsed[0]} {out_time_elapsed[1]}')

	return out_chi2

##########################
# select model spectra from the indicated models and meeting the parameters ranges 
def select_model_spectra(Teff_range, logg_range, model, model_dir):

	'''
	Output: dictionary
		spectra_name_full : full path to each selected model spectrum
		spectra_name : selected model spectra without full path
	'''

	# to store files in model_dir
	files = [] # with full path
	files_short = [] # only spectra names
	for i in range(len(model_dir)):
		files_model_dir = os.listdir(model_dir[i])
		for file in files_model_dir:
			files.append(model_dir[i]+file)
			files_short.append(file)

	# select spectra within the desired Teff and logg ranges
	# read Teff and logg from each model spectrum
	out_separate_params = separate_params(files_short, model)
	spectra_name_Teff = out_separate_params['Teff']
	spectra_name_logg = out_separate_params['logg']

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
	else: print(f'   {len(spectra_name)} model spectra will be compared')

	out = {'spectra_name_full': np.array(spectra_name_full), 'spectra_name': np.array(spectra_name)}

	return out

##########################
# number of data points in model spectra
def Ndata_model_spectra(model):

	if (model == 'Sonora_Diamondback'):	N_rows_model = 385466 # number of rows in model spectra (all spectra have the same length)
	if (model == 'Sonora_Elf_Owl'):	N_rows_model = 193132 # number of rows in model spectra (all spectra have the same length)
	if (model == 'LB23'): N_rows_model = 30000 # maximum number of rows in model spectra
	if (model == 'Sonora_Cholla'): N_rows_model = 110979 # maximum number of rows in spectra of the grid
	if (model == 'Sonora_Bobcat'): N_rows_model = 362000 # maximum number of rows in spectra of the grid
	if (model == 'ATMO2020'): N_rows_model = 5000 # maximum number of rows of the ATMO2020 model spectra
	if (model == 'BT-Settl'): N_rows_model = 1291340 # maximum number of rows of the BT-Settl model spectra
	if (model == 'SM08'): N_rows_model = 184663 # rows of the SM08 model spectra

	return N_rows_model

##########################
#  define objective function that returns the array to be minimized
def chi_square(params, data, edata, model, extinction_curve, weight):
	extinction = params['extinction']
	scaling = params['scaling']
	model_red = 10**(-extinction*extinction_curve/2.5) * scaling * model
	return np.sqrt(weight/np.mean(weight))*(data-model_red) / edata # consider that the square of this equation will be used in the fit

##########################
# separate parameters from each model spectrum name
def separate_params(spectra_name, model):

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
			logg_fit[i] = np.log10(float(spectra_name[i].split('_')[6])) + 2 # g in cgs
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
			logg_fit[i] = round(np.log10(float(spectra_name[i].split('_')[1][:-1])),1)+2
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
# save table with model spectra names sorted by chi square along with the parameters from each spectrum
def save_params(dict_for_table):

	# separate parameters to be included in the table
	model = dict_for_table['model']
	spectra_name = dict_for_table['spectra_name']
	chi2_fit = dict_for_table['chi2_fit']
	chi2_red_fit = dict_for_table['chi2_red_fit']
	scaling_fit = dict_for_table['scaling_fit']
	escaling_fit = dict_for_table['escaling_fit']
	Av_fit = dict_for_table['Av_fit']
	eAv_fit = dict_for_table['eAv_fit']
	iterations_fit = dict_for_table['iterations_fit']
	Teff_fit = dict_for_table['Teff']
	logg_fit = dict_for_table['logg']

	# table with model spectra sorted by the resulting chi square
	ind = np.argsort(chi2_red_fit)

	out = open(f'{model}_chi2_minimization.dat', 'w')
	if (model == 'Sonora_Diamondback'):
		Z_fit = dict_for_table['Z']
		fsed_fit = dict_for_table['fsed']
		out.write('# file                               chi2     chi2_red     scaling     e_scaling   Av    eAv    Teff   logg  Z      fsed   Iterations\n')
		for i in range(len(spectra_name)):
			out.write('%-35s %8.0f %9.3f %13.3E %11.3E %6.2f %5.2f %6i %5.1f %5.1f %4i %7i \n' %(spectra_name[ind[i]], chi2_fit[ind[i]], chi2_red_fit[ind[i]], scaling_fit[ind[i]], escaling_fit[ind[i]], Av_fit[ind[i]], eAv_fit[ind[i]], Teff_fit[ind[i]], logg_fit[ind[i]], Z_fit[ind[i]], fsed_fit[ind[i]], iterations_fit[ind[i]]))
	if (model == 'Sonora_Elf_Owl'):
		logKzz_fit = dict_for_table['logKzz']
		Z_fit = dict_for_table['Z']
		CtoO_fit = dict_for_table['CtoO']
		out.write('# file                                                         chi2      chi2_red     scaling     e_scaling   Av    eAv    Teff  logg   logKzz  Z    CtoO   Iterations\n')
		for i in range(len(spectra_name)):
			out.write('%-60s %8.0f %10.3f %14.3E %11.3E %6.2f %5.2f %5i %6.2f %3.0f %9.1f %6.3f %3i \n' %(spectra_name[ind[i]], chi2_fit[ind[i]], chi2_red_fit[ind[i]], scaling_fit[ind[i]], escaling_fit[ind[i]], Av_fit[ind[i]], eAv_fit[ind[i]], Teff_fit[ind[i]], logg_fit[ind[i]], logKzz_fit[ind[i]], Z_fit[ind[i]], CtoO_fit[ind[i]], iterations_fit[ind[i]]))
	if (model == 'LB23'):
		logKzz_fit = dict_for_table['logKzz']
		Z_fit = dict_for_table['Z']
		out.write('# file                                         chi2      chi2_red      scaling     e_scaling   Av    eAv    Teff  logg   Z      logKzz  Iterations\n')
		for i in range(len(spectra_name)):
			out.write('%-45s %8.0f %10.3f %14.3E %11.3E %6.2f %5.2f %5i %6.2f %7.3f %4.1f %7i \n' %(spectra_name[ind[i]], chi2_fit[ind[i]], chi2_red_fit[ind[i]], scaling_fit[ind[i]], escaling_fit[ind[i]], Av_fit[ind[i]], eAv_fit[ind[i]], Teff_fit[ind[i]], logg_fit[ind[i]], Z_fit[ind[i]], logKzz_fit[ind[i]], iterations_fit[ind[i]]))
	if (model == 'ATMO2020'):
		logKzz_fit = dict_for_table['logKzz']
		out.write('# file                           chi2    chi2_red      scaling     e_scaling   Av    eAv    Teff  logg  logKzz  Iterations\n')
		for i in range(len(spectra_name)):
			out.write('%-30s %8.0f %9.3f %14.3E %11.3E %6.2f %5.2f %6i %4.1f %5.1f %6i \n' %(spectra_name[ind[i]], chi2_fit[ind[i]], chi2_red_fit[ind[i]], scaling_fit[ind[i]], escaling_fit[ind[i]], Av_fit[ind[i]], eAv_fit[ind[i]], Teff_fit[ind[i]], logg_fit[ind[i]], logKzz_fit[ind[i]], iterations_fit[ind[i]]))
	if (model == 'Sonora_Cholla'):
		logKzz_fit = dict_for_table['logKzz']
		out.write('# file                     chi2     chi2_red     scaling     e_scaling   Av    eAv   Teff   logg  logKzz  Iterations\n')
		for i in range(len(spectra_name)):
			out.write('%-25s %8.0f %8.3f %14.3E %11.3E %6.2f %5.2f %5i %5.1f %3i %8i \n' %(spectra_name[ind[i]], chi2_fit[ind[i]], chi2_red_fit[ind[i]], scaling_fit[ind[i]], escaling_fit[ind[i]], Av_fit[ind[i]], eAv_fit[ind[i]], Teff_fit[ind[i]], logg_fit[ind[i]], logKzz_fit[ind[i]], iterations_fit[ind[i]]))
	if (model == 'Sonora_Bobcat'):
		Z_fit = dict_for_table['Z']
		CtoO_fit = dict_for_table['CtoO']
		out.write('# file                               chi2     chi2_red     scaling     e_scaling   Av    eAv   Teff   logg  Z     CtoO  Iterations\n')
		for i in range(len(spectra_name)):
			out.write('%-35s %8.0f %8.3f %14.3E %11.3E %6.2f %5.2f %5i %6.2f %4.1f %5.1f %4i \n' %(spectra_name[ind[i]], chi2_fit[ind[i]], chi2_red_fit[ind[i]], scaling_fit[ind[i]], escaling_fit[ind[i]], Av_fit[ind[i]], eAv_fit[ind[i]], Teff_fit[ind[i]], logg_fit[ind[i]], Z_fit[ind[i]], CtoO_fit[ind[i]], iterations_fit[ind[i]]))
	if (model == 'BT-Settl'):
		out.write('# file                               chi2     chi2_red    scaling     e_scaling   Av    eAv   Teff  logg    Iterations \n')
		for i in range(len(spectra_name)):
			out.write('%-15s %8.0f %8.3f %13.3E %11.3E %6.2f %5.2f %5i %4.1f %6i \n' %(spectra_name[ind[i]], chi2_fit[ind[i]], chi2_red_fit[ind[i]], scaling_fit[ind[i]], escaling_fit[ind[i]], Av_fit[ind[i]], eAv_fit[ind[i]], Teff_fit[ind[i]], logg_fit[ind[i]], iterations_fit[ind[i]]))
	if (model == 'SM08'):
		fsed_fit = dict_for_table['fsed']
		out.write('# file            chi2     chi2_red   scaling     e_scaling   Av    eAv   Teff  logg  fsed  Iterations \n')
		for i in range(len(spectra_name)):
			out.write('%-15s %9.0f %8.3f %12.3E %11.3E %6.2f %5.2f %4i %5.1f %3i %6i \n' %(spectra_name[ind[i]], chi2_fit[ind[i]], chi2_red_fit[ind[i]], scaling_fit[ind[i]], escaling_fit[ind[i]], Av_fit[ind[i]], eAv_fit[ind[i]], Teff_fit[ind[i]], logg_fit[ind[i]], fsed_fit[ind[i]], iterations_fit[ind[i]]))
	out.close()

	return  
