import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from sys import exit

##########################
def silicate_index(wl, flux, eflux, silicate_min=None, silicate_window=None, silicate_con1=None, 
	               silicate_window_con1=None, silicate_con2=None, silicate_window_con2=None, default='SM23',
	               plot=False, plot_title=None, plot_xrange=None, plot_yrange=None, plot_save=False):
	'''
	Description:
	------------
		Measure the strength of the mid-infrared silicate absorption considering the defined silicate index in Suárez & Metchev (2022,2023).

	Parameters:
	-----------
	- wl : array
		Spectrum wavelengths in microns.
	- flux : array
		Spectrum fluxes in Jy.
	- eflux : array
		Spectrum flux errors in Jy.
	- silicate_min : float, optional (default 9.3 um)
		Wavelength reference to indicate the center of silicate absorption. 
	- silicate_window : float, optional (default 0.6 um)
		Wavelength window around ``silicate_min`` used to calculate the average flux at the absorption.
	- silicate_cont1 : float, optional (default 7.45 um)
		Wavelength reference to indicate the short-wavelength continuum of silicate absorption. 
	- silicate_window_cont1 : float, optional (default 0.5 um))
		Wavelength window around ``silicate_cont1`` used to calculate the average flux at the short-wavelength absorption continuum.
	- silicate_cont2 : float, optional (default 13.5 um)
		Wavelength reference to indicate the long-wavelength continuum of silicate absorption. 
	- silicate_window_cont2 : float, optional (default 1.0 um)
		Wavelength window around ``silicate_cont2`` used to calculate the average flux at the long-wavelength absorption continuum.
	- default : {``SM23``, ``SM22``}, optional (default ``SM23``)
		Reference to set default parameters to measure the silicate index.
		``SM23`` (default) for Suárez & Metchev (2023) and ``SM22`` for Suárez & Metchev (2022).
	- plot : {``True``, ``False``}, optional (default ``False``)
		Plot (``True``) or do not plot (``False``) the silicate index measurement.
	- plot_title : str, optional
		Plot title (default ``'Silicate Index Measurement'``.
	- plot_xrange : list or array
		Wavelength range (in microns) of the plot (default [5.2, 14] um).
	- plot_yrange : list or array
		Flux range (in Jy) of the plot (default is the flux range in ``plot_xrange``).
	- plot_save : {``True``, ``False``}, optional (default ``False``)
		Save (``'True'``) or do not save (``'False'``) the resulting plot.

	Returns:
	--------
	- Dictionary 
		Dictionary with silicate index parameters:
			- ``'silicate_index'`` : silicate index
			- ``'esilicate_index'`` : silicate index uncertainty
			- ``'flux_silicate_min'`` : flux at the absorption feature
			- ``'eflux_silicate_min'`` : flux error at the absorption feature
			- ``'cont_silicate'`` : flux at the continuum of the absorption
			- ``'econt_silicate'`` : flux uncertainty at the continuum of the absorption
			- ``'slope'`` : slope of the linear fit to the log(flux)-log(wavelength) space
			- ``'eslope'`` : slope uncertainty
			- ``'constant'`` : constant or intercept of the linear fit
			- ``'econstant'`` : constant uncertainty
			- ``'silicate_min'`` : input ``silicate_min``
			- ``'silicate_window'`` : input ``silicate_window``
			- ``'silicate_con1'`` : input ``silicate_con1``
			- ``'silicate_window_con1'`` : input ``silicate_window_con1``
			- ``'silicate_con2'`` : input ``silicate_con2``
			- ``'silicate_window_con2'`` : input ``silicate_window_con2``
			- ``'wl'`` : input ``wl``
			- ``'flux'`` : input ``flux``
			- ``'eflux'`` : input ``eflux``
	- Plot of the silicate index measurement that will be stored if ``plot_save``.

	Author: Genaro Suárez
	'''

	# handle input spectrum
	wl, flux, eflux = handle_input_spectrum(wl, flux, eflux)

	# use default values if optional values (peak and continuum regions) are not provided
	if default=='SM23': # parameters in Suárez & Metchev (2023)
		# center of the absorption and window region
		if silicate_min is None: silicate_min = 9.30 # um
		if silicate_window is None: silicate_window = 0.6 # um
		# continuum region definition
		# where CH4 and NH3 are not expected for mid-L type
		if silicate_con1 is None: silicate_con1 = 7.45 # minimum in the blue side of the peak (um)
		if silicate_window_con1 is None: silicate_window_con1 = 0.5 # window in um for the short-wavelength continuum region
		if silicate_con2 is None: silicate_con2 = 13.5 # minimum in the red side of the peak (um)
		if silicate_window_con2 is None: silicate_window_con2 = 1.0 # window in um for the long-wavelength continuum region
	elif default=='SM22': # parameters in Suárez & Metchev (2022)
		# center of the absorption and window region
		if silicate_min is None: silicate_min = 9.00 # um
		if silicate_window is None: silicate_window = 0.6 # um
		# continuum region definition
		if silicate_con1 is None: silicate_con1 = 7.5 # minimum in the blue side of the peak (um)
		if silicate_window_con1 is None: silicate_window_con1 = 0.6 # window in um for the short-wavelength continuum region
		if silicate_con2 is None: silicate_con2 = 11.5 # minimum in the red side of the peak (um)
		if silicate_window_con2 is None: silicate_window_con2 = 0.6 # window in um for the long-wavelength continuum region
	else: raise Exception(f'"{default}" is not a recognized reference to set default parameters to measure the silicate index. \nTry "SM22" or "SM23".')

	# measure silicate index
	mask_silicate_min = (wl>=(silicate_min-silicate_window/2)) & (wl<=(silicate_min+silicate_window/2))
	# using the mean
	flux_silicate_min = np.mean(flux[mask_silicate_min]) # average flux at the bottom of the feature
	eflux_silicate_min = np.std(flux[mask_silicate_min]) / np.sqrt(flux[mask_silicate_min].size) # flux error as the standard error of the mean
	## using the median
	#flux_silicate_min = np.median(flux[mask_silicate_min]) # average flux at the bottom of the feature
	#eflux_silicate_min = 1.2533 * np.std(flux[mask_silicate_min]) / np.sqrt(flux[mask_silicate_min].size) # standard error of the median = 1.2533 times the standard error of the mean
	##eflux_silicate_min = (np.percentile(flux[mask_silicate_min], 84, interpolation = 'linear') - np.percentile(flux[mask_silicate_min], 16, interpolation = 'linear')) / np.sqrt(flux[mask_silicate_min].size) # 68% confidence interval
	##eflux_silicate_min = flux_silicate_min * np.median(eflux[mask_silicate_min]/flux[mask_silicate_min]) # error to keep the fractional uncertainties

#	# data points in the first continuum region
#	mask_silicate_con1 = (wl>=(silicate_con1-silicate_window_con1/2)) & (wl<=(silicate_con1+silicate_window_con1/2))
#	# data points in the second continuum region
#	mask_silicate_con2 = (wl>=(silicate_con2-silicate_window_con2/2)) & (wl<=(silicate_con2+silicate_window_con2/2))

	# fit a line to the continuum points
	log_log_space = True # True or False to consider the log(flux)-log(lambda) space
	#--------------------------------------
	# fit a line to the continuum points in linear space
#	if (log_log_space=='no'):
#		mask_silicate_con1_con2 = np.concatenate((mask_silicate_con1, mask_silicate_con2)) # indices of the continuum data points on both regions
#		fit, cov_fit = np.polyfit(wl[mask_silicate_con1_con2,i], flux[mask_silicate_con1_con2,i], 1, w=1./eflux[mask_silicate_con1_con2,i], cov=True) # weigh by the inverse of the error
#		slope = fit[0] # slope of the histogram without any correction
#		eslope = np.sqrt(cov_fit[0,0]) # error of the slope
#		constant = fit[1]
#		econstant = np.sqrt(cov_fit[1,1]) 
#		# continuum at the point in of the absorption
#		cont_silicate = slope*silicate_min + constant
#		econt_silicate = (((slope+eslope)*silicate_min + constant) - ((slope-eslope)*silicate_min + constant)) / 2

	
	# fit a line to the continuum points in the log(flux)-log(lambda) space
	if log_log_space:
		mask_silicate_con1_con2 = ((wl>=(silicate_con1-silicate_window_con1/2)) & (wl<=(silicate_con1+silicate_window_con1/2))) | \
		                         ((wl>=(silicate_con2-silicate_window_con2/2)) & (wl<=(silicate_con2+silicate_window_con2/2))) # data points in both continuum regions
		if (flux[mask_silicate_con1_con2].min()<=0): 
			print('Warning: Logarithm of negative and/or zero fluxes in the continuum windows')
			print('   Only positive fluxes will be used')
			mask_silicate_con1_con2 = (((wl>=(silicate_con1-silicate_window_con1/2)) & (wl<=(silicate_con1+silicate_window_con1/2))) | \
			                         ((wl>=(silicate_con2-silicate_window_con2/2)) & (wl<=(silicate_con2+silicate_window_con2/2)))) & \
			                         (flux>0) # positive fluxes in both continuum regions
		logwl = np.log10(wl[mask_silicate_con1_con2])
		logflux = np.log10(flux[mask_silicate_con1_con2])
		#elogflux = (1/np.log(10)) * eflux[mask_silicate_con1_con2]/flux[mask_silicate_con1_con2]
		elogflux = eflux[mask_silicate_con1_con2] # these errors are used only for weighing the fit. Thus, assigning the log errors equal to the linear error will weight by the error bars directly plotted in the flux vs wavelength plot. Otherwise, when properly propagating the linear errors to log errors will give more weight (lower errors) to higher fluxes.
	
		fit, cov_fit = np.polyfit(logwl, logflux, 1, w=1./elogflux, cov=True) # weigh by the inverse of the error
		slope = fit[0] # slope 
		eslope = np.sqrt(cov_fit[0,0]) # slope error 
		constant = fit[1] # constant
		econstant = np.sqrt(cov_fit[1,1]) # constant error
		# the slope error has to be scaled somehow to be used in linear plots
		fit_linear, cov_fit_linear = np.polyfit(wl[mask_silicate_con1_con2], flux[mask_silicate_con1_con2], 1, w=1./eflux[mask_silicate_con1_con2], cov=True) # weigh by the inverse of the error
		slope_linear = fit_linear[0] # slope 
		eslope_linear = np.sqrt(cov_fit_linear[0,0])
		# slope error normalization
		eslope = eslope / np.log10(eslope/eslope_linear) # this is an empirical correction I deduced
	
		# continuum at the absorption peak
		cont_silicate = 10**(slope*np.log10(silicate_min) + constant)
		econt_silicate = (10**(((slope+eslope)*np.log10(silicate_min)+constant)) - 10**((slope-eslope)*np.log10(silicate_min)+constant)) / 2
		#econt_silicate = (10**(((slope+eslope)*np.log10(silicate_min)+(constant-econstant))) - 10**((slope-eslope)*np.log10(silicate_min)+(constant+econstant))) / 2
	
	## fit an exponential curve to the continuum points in the flux vs. lambda plot (this should be equivalent to a linear fit in the log(flux)-log(lambda) space)
	# ATTEMPT
	#if not log_log_space:
	#	#def exponential(x, a, b, c):
	# 	#   return a * np.exp(b * x) + c
	#	def exponential(x, a, b):
	# 	   return a*np.exp(b * x)
	#	mask_silicate_con1_con2 = np.concatenate((mask_silicate_con1, mask_silicate_con2)) # indices of the continuum data points on both regions
	#	fit_exp, cov_fit_exp = curve_fit(exponential, wl[mask_silicate_con1_con2,i], flux[mask_silicate_con1_con2,i])#, sigma=eflux[mask_silicate_con1_con2,i])

	# silicate index
	silicate_index = cont_silicate/flux_silicate_min
	esilicate_index = silicate_index * np.sqrt((econt_silicate/cont_silicate)**2 + (eflux_silicate_min/flux_silicate_min)**2)

	# output dictionary
	out = {'silicate_index': silicate_index, 'esilicate_index': esilicate_index, 'flux_silicate_min': flux_silicate_min, 
	       'eflux_silicate_min': eflux_silicate_min, 'cont_silicate': cont_silicate, 'econt_silicate': econt_silicate}
	# add parameter of the slope fit
	out['slope'] = slope
	out['eslope'] = eslope
	out['constant'] = constant
	out['econstant'] = econstant
	# add parameters used to measure the index
	out['silicate_min'] = silicate_min
	out['silicate_window'] = silicate_window
	out['silicate_con1'] = silicate_con1
	out['silicate_window_con1'] = silicate_window_con1
	out['silicate_con2'] = silicate_con2
	out['silicate_window_con2'] = silicate_window_con2
	#out['log_log_space'] = log_log_space
	# add input spectrum
	out['wl'] = wl
	out['flux'] = flux
	out['eflux'] = eflux

	# visualize how the silicate index was measured
	if plot: plot_silicate_index(out, plot_xrange, plot_yrange, plot_title, plot_save)

	return out

##########################
def water_index(wl, flux, eflux, default='SM22', water_window=None, water_max=None, water_min1=None, water_min2=None,
	            plot=False, plot_title=None, plot_xrange=None, plot_yrange=None, plot_save=False):
	'''
	Description:
	------------
		Measure the strength of the mid-infrared water absorption considering the defined water index in Cushing et al. (2006) and modified in Suárez & Metchev (2022).

	Parameters:
	-----------
	- wl : array
		Spectrum wavelengths in microns.
	- flux : array
		Spectrum fluxes in Jy.
	- eflux : array
		Spectrum flux errors in Jy.
	- default : {``SM22``, ``C06``}, optional (default ``SM22``)
		Reference to set default parameters to measure the water index.
		``SM22`` (default) for Suárez & Metchev (2022) or ``C08`` for Cushing et al (2006).
	- water_max : float, optional (default 6.25 um)
		Wavelength reference to measure the peak-like feature within the water absorption.
	- water_min1 : float, optional (default 5.80 um)
		Wavelength reference to measure the flux in the first absorption dip.
	- water_min2 : float, optional (default 6.75 um)
		Wavelength reference to measure the flux in the second absorption dip.
	- water_window : float, optional (default 0.3 um)
		Wavelength window around ``water_max``, ``water_min1``, and ``water_min2`` used to calculate the average fluxes.
	- plot : {``True``, ``False``}, optional (default ``False``)
		Plot (``True``) or do not plot (``False``) the water index measurement.
	- plot_title : str, optional
		Plot title (default ``'Water Index Measurement'``.
	- plot_xrange : list or array
		Wavelength range (in microns) of the plot (default [5.2, 14] um).
	- plot_yrange : list or array
		Flux range (in Jy) of the plot (default is the flux range in ``plot_xrange``).
	- plot_save : {``True``, ``False``}, optional (default ``False``)
		Save (``'True'``) or do not save (``'False'``) the resulting plot.

	Returns:
	--------
	- Dictionary 
		Dictionary with water index parameters:
			- ``'water_index'`` : water index
			- ``'ewater_index'`` : water index uncertainty
			- ``'flux_water_max'`` : flux at the absorption peak
			- ``'eflux_water_max'`` : flux uncertainty at the absorption peak
			- ``'flux_water_min1'`` : flux at the short-wavelength absorption dip
			- ``'eflux_water_min1'`` : flux uncertainty at the short-wavelength absorption dip
			- ``'flux_water_min2'`` : flux at the long-wavelength absorption dip
			- ``'eflux_water_min2'`` : flux uncertainty at the long-wavelength absorption dip
			- ``'water_max'`` : input ``water_max``
			- ``'water_min1'`` : input ``water_min1``
			- ``'water_min2'`` : input ``water_min2``
			- ``'water_window'`` : input ``water_window``
	- Plot of the water index measurement that will be stored if ``plot_save``.

	Author: Genaro Suárez
	'''

	# handle input spectrum
	wl, flux, eflux = handle_input_spectrum(wl, flux, eflux)

	if default=='SM22': # parameters in Suárez & Metchev (2022)
		if water_window is None: water_window = 0.30 # um
		if water_max is None: water_max = 6.25 # flux peak (um)
		if water_min1 is None: water_min1 = 5.80 # minimum in the blue side of the peak (um)
		if water_min2 is None: water_min2 = 6.75 # minimum in the red side of the peak (um)
	elif default=='C06': # parameters in Cushing et al. (2006)
		if water_window is None: water_window = 0.15 # um
		if water_max is None: water_max = 6.25 # flux peak (um)
		if water_min1 is None: water_min1 = 5.80 # minimum in the blue side of the peak (um)
		if water_min2 is None: water_min2 = 6.75 # minimum in the red side of the peak (um)
	else: raise Exception(f'"{default}" is not a recognized reference to set default parameters to measure the water index. \nTry "SM22" or "C06".')
	
	# mean flux of the peak
	mask_water_max = (wl>=(water_max-water_window/2)) & (wl<=(water_max+water_window/2))
	flux_water_max = np.mean(flux[mask_water_max])
	#eflux_water_max = np.std(flux[mask_water_max]) / np.sqrt(flux[mask_water_max].size) # in Suarez & Metchev (2022)
	eflux_water_max_1 = np.std(flux[mask_water_max]) / np.sqrt(flux[mask_water_max].size) # due to the scatter of the data in the window
	eflux_water_max_2 = flux_water_max * np.median(eflux[mask_water_max]/flux[mask_water_max]) # due to flux uncertainties
	eflux_water_max = np.sqrt(eflux_water_max_1**2 + eflux_water_max_2**2) # addition in quadrature

	# mean flux of blue minimum
	mask_water_min1 = (wl>=(water_min1-water_window/2)) & (wl<=(water_min1+water_window/2))
	flux_water_min1 = np.mean(flux[mask_water_min1])
	#eflux_water_min1 = np.std(flux[mask_water_min1]) / np.sqrt(flux[mask_water_min1].size) # in Suarez & Metchev (2022)
	eflux_water_min1_1 = np.std(flux[mask_water_min1]) / np.sqrt(flux[mask_water_min1].size) # due to the scatter of the data in the window
	eflux_water_min1_2 = flux_water_min1 * np.median(eflux[mask_water_min1]/flux[mask_water_min1]) # due to flux uncertainties
	eflux_water_min1 = np.sqrt(eflux_water_min1_1**2 + eflux_water_min1_2**2) # addition in quadrature

	# mean flux of red minimum
	mask_water_min2 = (wl>=(water_min2-water_window/2)) & (wl<=(water_min2+water_window/2))
	flux_water_min2 = np.mean(flux[mask_water_min2])
	#eflux_water_min2 = np.std(flux[mask_water_min2]) / np.sqrt(flux[mask_water_min2].size) # in Suarez & Metchev (2022)
	eflux_water_min2_1 = np.std(flux[mask_water_min2]) / np.sqrt(flux[mask_water_min2].size) # due to the scatter of the data in the window
	eflux_water_min2_2 = flux_water_min2 * np.median(eflux[mask_water_min2]/flux[mask_water_min2]) # due to flux uncertainties
	eflux_water_min2 = np.sqrt(eflux_water_min2_1**2 + eflux_water_min2_2**2) # addition in quadrature
	
	# water index
	weight1 = 0.562
	weight2 = 0.474
	water_index = flux_water_max / (weight1*flux_water_min1 + weight2*flux_water_min2)
	ewater_index = water_index * np.sqrt((eflux_water_max/flux_water_max)**2 + 
	                                 ((weight1*eflux_water_min1)/(weight1*flux_water_min1 + weight2*flux_water_min2))**2 + 
	                                 ((weight2*eflux_water_min2)/(weight1*flux_water_min1 + weight2*flux_water_min2))**2)

	# output dictionary
	out = {'water_index': water_index, 'ewater_index': ewater_index, 'flux_water_max': flux_water_max, 
	       'eflux_water_max': eflux_water_max, 'flux_water_min1': flux_water_min1, 'eflux_water_min1': eflux_water_min1, 
	       'flux_water_min2': flux_water_min2, 'eflux_water_min2': eflux_water_min2}
	# add parameters used to measure the index
	out['water_window'] = water_window
	out['water_max'] = water_max
	out['water_min1'] = water_min1
	out['water_min2'] = water_min2
	# add input spectrum
	out['wl'] = wl
	out['flux'] = flux
	out['eflux'] = eflux

	# visualize how the silicate index was measured
	if plot: plot_water_index(out, plot_xrange, plot_yrange, plot_title, plot_save)

	return out

##########################
def methane_index(wl, flux, eflux, default='SM22', methane_window=None, methane_in=None, methane_out=None, 
	              plot=False, plot_title=None, plot_xrange=None, plot_yrange=None, plot_save=False):
	'''
	Description:
	------------
		Measure the strength of the mid-infrared methane absorption considering the defined methane index in Cushing et al. (2006) and modified in Suárez & Metchev (2022).

	Parameters:
	-----------
	- wl : array
		Spectrum wavelengths in microns.
	- flux : array
		Spectrum fluxes in Jy.
	- eflux : array
		Spectrum flux errors in Jy.
	- default : {``SM22``, ``C06``}, optional (default ``SM22``)
		Reference to set default parameters to measure the methane index.
		``SM22`` (default) for Suárez & Metchev (2022) or ``C08`` for Cushing et al (2006).
	- methane_in : float, optional (default 7.65 um)
		Wavelength reference within the feature.
	- methane_out : float, optional (default 9.9 um)
		Wavelength reference out of the feature.
		Note: the default value is slightly smaller than the 10 um value in Suárez & Metchev (2022) to avoid including fluxes at the beginning of the ammonia feature.
	- methane_window : float, optional (default 0.6 um)
		Wavelength window around ``methane_in`` and ``methane_out`` used to calculate average fluxes.
	- plot : {``True``, ``False``}, optional (default ``False``)
		Plot (``True``) or do not plot (``False``) the methane index measurement.
	- plot_title : str, optional
		Plot title (default ``'Methane Index Measurement'``.
	- plot_xrange : list or array
		Wavelength range (in microns) of the plot (default [5.2, 14] um).
	- plot_yrange : list or array
		Flux range (in Jy) of the plot (default is the flux range in ``plot_xrange``).
	- plot_save : {``True``, ``False``}, optional (default ``False``)
		Save (``'True'``) or do not save (``'False'``) the resulting plot.

	Returns:
	--------
	- Dictionary 
		Dictionary with methane index parameters:
			- ``'methane_index'`` : methane index
			- ``'emethane_index'`` : methane index uncertainty
			- ``'flux_methane_in'`` : flux within the absorption
			- ``'eflux_methane_in'`` : flux uncertainty within the absorption
			- ``'flux_methane_out'`` : flux out of the absorption
			- ``'eflux_methane_out'`` : flux uncertainty out of the absorption
			- ``'methane_in'`` : input ``methane_in``
			- ``'methane_out'`` : input ``methane_out``
			- ``'methane_window'`` : input ``methane_window``
	- Plot of the methane index measurement that will be stored if ``plot_save``.

	Author: Genaro Suárez
	'''

	# handle input spectrum
	wl, flux, eflux = handle_input_spectrum(wl, flux, eflux)

	if default=='SM22': # parameters in Suárez & Metchev (2022)
		if methane_in is None: methane_in = 7.65 # (um) wavelength point in of the feature
		if methane_out is None: methane_out = 9.9 # (um) wavelength point out of the feature. Note: it is actually 10 um in Suárez & Metchev (2022), but includes a few data points within the ammonia feature.
		if methane_window is None: methane_window = 0.6 # (um) wavelength window
	elif default=='C06': # parameters in Cushing et al. (2006)
		if methane_in is None: methane_in = 8.5 # (um) wavelength point in of the feature
		if methane_out is None: methane_out = 10.0 # (um) wavelength point out of the feature 
		if methane_window is None: methane_window = 0.3 # (um) wavelength window

	# mean flux in the feature
	mask_methane_in = (wl>=(methane_in-methane_window/2)) & (wl<=(methane_in+methane_window/2))
	flux_methane_in = np.mean(flux[mask_methane_in])
	#eflux_methane_in = np.std(flux[mask_methane_in]) / np.sqrt(flux[mask_methane_in].size) # in Suarez & Metchev (2022)
	eflux_methane_in_1 = np.std(flux[mask_methane_in]) / np.sqrt(flux[mask_methane_in].size) # due to the scatter of the data in the window
	eflux_methane_in_2 = flux_methane_in * np.median(eflux[mask_methane_in]/flux[mask_methane_in]) # due to flux uncertainties
	eflux_methane_in = np.sqrt(eflux_methane_in_1**2 + eflux_methane_in_2**2) # addition in quadrature
	
	# mean flux out of the feature
	mask_methane_out = np.where((wl>=(methane_out-methane_window/2)) & (wl<=(methane_out+methane_window/2)))[0]
	flux_methane_out = np.mean(flux[mask_methane_out])
	#eflux_methane_out = np.std(flux[mask_methane_out]) / np.sqrt(flux[mask_methane_out].size) # in Suarez & Metchev (2022)
	eflux_methane_out_1 = np.std(flux[mask_methane_out]) / np.sqrt(flux[mask_methane_out].size) # due to the scatter of the data in the window
	eflux_methane_out_2 = flux_methane_out * np.median(eflux[mask_methane_out]/flux[mask_methane_out]) # due to flux uncertainties
	eflux_methane_out = np.sqrt(eflux_methane_out_1**2 + eflux_methane_out_2**2) # addition in quadrature

	# methane index
	methane_index = flux_methane_out / flux_methane_in
	emethane_index = methane_index * np.sqrt((eflux_methane_in/flux_methane_in)**2 + (eflux_methane_out/flux_methane_out)**2)

	# output dictionary
	out = {'methane_index': methane_index, 'emethane_index': emethane_index, 'flux_methane_in': flux_methane_in, 
	       'eflux_methane_in': eflux_methane_in, 'flux_methane_out': flux_methane_out, 'eflux_methane_out': eflux_methane_out}
	# add parameters used to measure the index
	out['methane_in'] = methane_in
	out['methane_out'] = methane_out
	out['methane_window'] = methane_window
	# add input spectrum
	out['wl'] = wl
	out['flux'] = flux
	out['eflux'] = eflux
	
	# visualize how the silicate index was measured
	if plot: plot_methane_index(out, plot_xrange, plot_yrange, plot_title, plot_save)

	return out

##########################
def ammonia_index(wl, flux, eflux, default='SM22', ammonia_window=None, ammonia_in=None, ammonia_out=None, 
	              plot=False, plot_title=None, plot_xrange=None, plot_yrange=None, plot_save=False):
	'''
	Description:
	------------
		Measure the strength of the mid-infrared ammonia absorption considering the defined ammonia index in Cushing et al. (2006) and modified in Suárez & Metchev (2022).

	Parameters:
	-----------
	- wl : array
		Spectrum wavelengths in microns.
	- flux : array
		Spectrum fluxes in Jy.
	- eflux : array
		Spectrum flux errors in Jy.
	- default : {``SM22``, ``C06``}, optional (default ``SM22``)
		Reference to set default parameters to measure the ammonia index.
		``SM22`` (default) for Suárez & Metchev (2022) or ``C08`` for Cushing et al (2006).
	- ammonia_in : float, optional (default 10.6 um)
		Wavelength reference within the feature.
		Note: the default value is slightly smaller than the 10.8 um value in Suárez & Metchev (2022) to be centered better within the feature.
	- ammonia_out : float, optional (default 9.9 um)
		Wavelength reference out of the feature.
		Note: the default value is slightly smaller than the 10 um value in Suárez & Metchev (2022) to avoid including fluxes at the beginning of the ammonia feature.
	- ammonia_window : float, optional (default 0.6 um)
		Wavelength window around ``ammonia_in`` and ``ammonia_out`` used to calculate average fluxes.
	- plot : {``True``, ``False``}, optional (default ``False``)
		Plot (``True``) or do not plot (``False``) the ammonia index measurement.
	- plot_title : str, optional
		Plot title (default ``'Ammonia Index Measurement'``.
	- plot_xrange : list or array
		Wavelength range (in microns) of the plot (default [5.2, 14] um).
	- plot_yrange : list or array
		Flux range (in Jy) of the plot (default is the flux range in ``plot_xrange``).
	- plot_save : {``True``, ``False``}, optional (default ``False``)
		Save (``'True'``) or do not save (``'False'``) the resulting plot.

	Returns:
	--------
	- Dictionary 
		Dictionary with ammonia index parameters:
			- ``'ammonia_index'`` : ammonia index
			- ``'eammonia_index'`` : ammonia index uncertainty
			- ``'flux_ammonia_in'`` : flux within the absorption
			- ``'eflux_ammonia_in'`` : flux uncertainty within the absorption
			- ``'flux_ammonia_out'`` : flux out of the absorption
			- ``'eflux_ammonia_out'`` : flux uncertainty out of the absorption
			- ``'ammonia_in'`` : input ``ammonia_in``
			- ``'ammonia_out'`` : input ``ammonia_out``
			- ``'ammonia_window'`` : input ``ammonia_window``
	- Plot of the ammonia index measurement that will be stored if ``plot_save``.

	Author: Genaro Suárez
	'''

	# handle input spectrum
	wl, flux, eflux = handle_input_spectrum(wl, flux, eflux)

	if default=='SM22': # parameters in Suárez & Metchev (2022)
		#if ammonia_in is None: ammonia_in = 10.8 # (um) wavelength point in of the feature
		#if ammonia_out is None: ammonia_out = 10.0 # (um) wavelength point out of the feature. Note: it is actually 10 um in Suárez & Metchev (2022), but includes a few data points within the ammonia feature.
		if ammonia_in is None: ammonia_in = 10.6 # (um) wavelength point in of the feature. Note: it is 10.8 um in Suárez & Metchev (2022), but includes it is not well center within the absorption.
		if ammonia_out is None: ammonia_out = 9.9 # (um) wavelength point out of the feature. Note: it is 10 um in Suárez & Metchev (2022), but includes a few data points within the ammonia feature.
		if ammonia_window is None: ammonia_window = 0.6 # (um) wavelength window
	elif default=='C06': # parameters in Cushing et al. (2006)
		if ammonia_in is None: ammonia_in = 8.5 # (um) wavelength point in of the feature
		if ammonia_out is None: ammonia_out = 10.0 # (um) wavelength point out of the feature 
		if ammonia_window is None: ammonia_window = 0.3 # (um) wavelength window

	# mean flux in the feature
	mask_ammonia_in = (wl>=(ammonia_in-ammonia_window/2)) & (wl<=(ammonia_in+ammonia_window/2))
	flux_ammonia_in = np.mean(flux[mask_ammonia_in])
	#eflux_ammonia_in = np.std(flux[mask_ammonia_in]) / np.sqrt(flux[mask_ammonia_in].size) # in Suarez & Metchev (2022)
	eflux_ammonia_in_1 = np.std(flux[mask_ammonia_in]) / np.sqrt(flux[mask_ammonia_in].size) # due to the scatter of the data in the window
	eflux_ammonia_in_2 = flux_ammonia_in * np.median(eflux[mask_ammonia_in]/flux[mask_ammonia_in]) # due to flux uncertainties
	eflux_ammonia_in = np.sqrt(eflux_ammonia_in_1**2 + eflux_ammonia_in_2**2) # addition in quadrature
	
	# mean flux out of the feature
	mask_ammonia_out = np.where((wl>=(ammonia_out-ammonia_window/2)) & (wl<=(ammonia_out+ammonia_window/2)))[0]
	flux_ammonia_out = np.mean(flux[mask_ammonia_out])
	#eflux_ammonia_out = np.std(flux[mask_ammonia_out]) / np.sqrt(flux[mask_ammonia_out].size) # in Suarez & Metchev (2022)
	eflux_ammonia_out_1 = np.std(flux[mask_ammonia_out]) / np.sqrt(flux[mask_ammonia_out].size) # due to the scatter of the data in the window
	eflux_ammonia_out_2 = flux_ammonia_out * np.median(eflux[mask_ammonia_out]/flux[mask_ammonia_out]) # due to flux uncertainties
	eflux_ammonia_out = np.sqrt(eflux_ammonia_out_1**2 + eflux_ammonia_out_2**2) # addition in quadrature

	# ammonia index
	ammonia_index = flux_ammonia_out / flux_ammonia_in
	eammonia_index = ammonia_index * np.sqrt((eflux_ammonia_in/flux_ammonia_in)**2 + (eflux_ammonia_out/flux_ammonia_out)**2)

	# output dictionary
	out = {'ammonia_index': ammonia_index, 'eammonia_index': eammonia_index, 'flux_ammonia_in': flux_ammonia_in, 
	       'eflux_ammonia_in': eflux_ammonia_in, 'flux_ammonia_out': flux_ammonia_out, 'eflux_ammonia_out': eflux_ammonia_out}
	# add parameters used to measure the index
	out['ammonia_in'] = ammonia_in
	out['ammonia_out'] = ammonia_out
	out['ammonia_window'] = ammonia_window
	# add input spectrum
	out['wl'] = wl
	out['flux'] = flux
	out['eflux'] = eflux
	
	# visualize how the silicate index was measured
	if plot: plot_ammonia_index(out, plot_xrange, plot_yrange, plot_title, plot_save)

	return out

##########################
# plot the silicate index measurement
def plot_silicate_index(out_silicate_index, plot_xrange=None, plot_yrange=None, plot_title=None, plot_save=True):

	# read parameters of interest
	wl = out_silicate_index['wl']
	flux = out_silicate_index['flux']
	eflux = out_silicate_index['eflux']
	si_index = out_silicate_index['silicate_index']
	esi_index = out_silicate_index['esilicate_index']
	# mean flux at the absorption
	flux_silicate_min = out_silicate_index['flux_silicate_min']
	eflux_silicate_min = out_silicate_index['eflux_silicate_min']
	# mean flux at the absorption continuum
	cont_silicate = out_silicate_index['cont_silicate']
	econt_silicate = out_silicate_index['econt_silicate']
	# continuum fit
	slope = out_silicate_index['slope']
	eslope = out_silicate_index['eslope']
	constant = out_silicate_index['constant']
	econstant = out_silicate_index['econstant']
	# silicate feature peak
	silicate_min = out_silicate_index['silicate_min'] # flux peak (um)
	silicate_window = out_silicate_index['silicate_window'] # window in um for the absorption
	# first window region for the continuum
	silicate_con1 = out_silicate_index['silicate_con1'] # center (in um) of the short-wavelength continuum region
	silicate_window_con1 = out_silicate_index['silicate_window_con1'] # window in um for the short-wavelength continuum region
	# second window region for the continuum
	silicate_con2 = out_silicate_index['silicate_con2'] # center (in um) of the long-wavelength continuum region
	silicate_window_con2 = out_silicate_index['silicate_window_con2'] # window in um for the long-wavelength continuum region

	#++++++++++++++++++++++++++++++
	# plot silicate index measurement
	fig, ax = plt.subplots()
	
	#xmin, xmax = wl[wl>0].min(), wl[wl>0].max()
	#ymin, ymax = np.percentile(flux[flux!=0], scale_percent), np.nanpercentile(flux[flux!=0], 100-scale_percent)
	#xmin, xmax = 0.90*silicate_con1-silicate_window_con1/2., 1.05*silicate_con2+silicate_window_con2/2.
	if plot_xrange is None: xmin, xmax = 5.2, 14
	else: xmin, xmax = plot_xrange
	if plot_yrange is None:
		#scale_percent = 0.1
		mask = (wl>=xmin) & (wl<=xmax)
		ymin, ymax = flux[mask].min(), flux[mask].max()
	else: ymin, ymax = plot_yrange
	
	# indicate the regions to measure the silicate index
	# silicate absorption center
	ax.fill([silicate_min-silicate_window/2, silicate_min-silicate_window/2, 
	        silicate_min+silicate_window/2, silicate_min+silicate_window/2], 
	        [ymin, ymax, ymax, ymin], facecolor='silver', linewidth=1, zorder=2)
	# short-wavelength silicate absorption continuum
	ax.fill([silicate_con1-silicate_window_con1/2, silicate_con1-silicate_window_con1/2, 
	        silicate_con1+silicate_window_con1/2, silicate_con1+silicate_window_con1/2], 
	        [ymin, ymax, ymax, ymin], facecolor='gainsboro', linewidth=1, zorder=2)
	# long-wavelength silicate absorption continuum
	ax.fill([silicate_con2-silicate_window_con2/2, silicate_con2-silicate_window_con2/2, 
	        silicate_con2+silicate_window_con2/2, silicate_con2+silicate_window_con2/2], 
	        [ymin, ymax, ymax, ymin], facecolor='gainsboro', linewidth=1, zorder=2)
	
	# plot flux uncertainty region
	default_blue = plt.rcParams['axes.prop_cycle'].by_key()['color'][0] # default blue coloe
	wl_region = np.append(wl, np.flip(wl))
	flux_region = np.append(flux-eflux, np.flip(flux+eflux))
	ax.fill(wl_region, flux_region, facecolor=default_blue, edgecolor=default_blue, linewidth=0, alpha=0.30, zorder=3)
	# plot spectrum
	plt.plot(wl, flux, color=default_blue , zorder=3)
	
	# linear fit
	xc_silicate = np.linspace(silicate_con1-silicate_window_con1/2., silicate_con2+silicate_window_con2/2., 100)
	yc_silicate = 10**(slope*np.log10(xc_silicate) + constant)
	ycu_silicate = 10**((slope+eslope)*np.log10(xc_silicate) + constant)
	ycd_silicate = 10**((slope-eslope)*np.log10(xc_silicate) + constant)
	ax.fill(np.append(xc_silicate, xc_silicate[::-1]), np.append(ycd_silicate, ycu_silicate[::-1]), 
	        facecolor='gray', edgecolor='gray', linewidth=0, alpha=0.20)
	ax.plot(xc_silicate, yc_silicate, '--', color='gray', linewidth=0.5)
	
	# fluxes in the continuum regions
	mask_silicate_con1 = (wl>=(silicate_con1-silicate_window_con1/2)) & (wl<=(silicate_con1+silicate_window_con1/2))
	mask_silicate_con2 = (wl>=(silicate_con2-silicate_window_con2/2)) & (wl<=(silicate_con2+silicate_window_con2/2))
	ax.errorbar(wl[mask_silicate_con1], flux[mask_silicate_con1], yerr=eflux[mask_silicate_con1], 
	            fmt='o', c='gray', markersize=3.0, linewidth=1.0, capsize=3, capthick=1)
	ax.errorbar(wl[mask_silicate_con2], flux[mask_silicate_con2], yerr=eflux[mask_silicate_con2], 
	            fmt='o', c='gray', markersize=3.0, linewidth=1.0, capsize=3, capthick=1)
	# interpolated continuum at the absorption
	ax.errorbar(silicate_min, cont_silicate, yerr=econt_silicate, fmt='o', c='black', 
	            markersize=3.0, linewidth=1.0, capsize=3, capthick=1)
	
	# fluxes in the absorption
	mask_silicate_min = (wl>=(silicate_min-silicate_window/2)) & (wl<=(silicate_min+silicate_window/2))
	ax.errorbar(wl[mask_silicate_min], flux[mask_silicate_min], yerr=eflux[mask_silicate_min],  
	            fmt='o', c='gray', markersize=3.0, linewidth=1.0, capsize=3, capthick=1)
	# mean flux of the absorption 
	ax.errorbar(silicate_min, flux_silicate_min, yerr=eflux_silicate_min, 
	            fmt='o', c='red', markersize=3.0, linewidth=1.0, capsize=3, capthick=1, zorder=3)
	
	plt.text(0.55, 0.8, 'Silicate index='+str(round(si_index,2))+'$\pm$'+str(round(esi_index,2)), transform=fig.transFigure, size=12)
	
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	ax.grid(True, which='both', color='gainsboro', alpha=0.5)
	
	plt.xlabel(r'$\lambda\ (\mu$m)', size=12)
	plt.ylabel(r'$F_\nu\ (Jy)$', size=12)
	if plot_title is not None: plt.title(plot_title, size=12)
	else: plt.title('Silicate Index Measurement', size=12)
	
	if plot_save: plt.savefig('silicate_index_measurement.pdf', bbox_inches='tight')
	plt.show()
	plt.close()

	return

##########################
# plot the water index measurement
def plot_water_index(out_water_index, plot_xrange=None, plot_yrange=None, plot_title=None, plot_save=True):

	# read parameters of interest
	wl = out_water_index['wl']
	flux = out_water_index['flux']
	eflux = out_water_index['eflux']
	water_index = out_water_index['water_index']
	ewater_index = out_water_index['ewater_index']
	flux_water_max = out_water_index['flux_water_max']
	eflux_water_max = out_water_index['eflux_water_max']
	flux_water_min1 = out_water_index['flux_water_min1']
	eflux_water_min1 = out_water_index['eflux_water_min1']
	flux_water_min2 = out_water_index['flux_water_min2']
	eflux_water_min2 = out_water_index['eflux_water_min2']
	water_window = out_water_index['water_window']
	water_max = out_water_index['water_max']
	water_min1 = out_water_index['water_min1']
	water_min2 = out_water_index['water_min2']

	#++++++++++++++++++++++++++++++
	# plot silicate index measurement
	fig, ax = plt.subplots()

	if plot_xrange is None: xmin, xmax = 5.2, 14
	else: xmin, xmax = plot_xrange
	if plot_yrange is None:
		mask = (wl>=xmin) & (wl<=xmax)
		ymin, ymax = flux[mask].min(), flux[mask].max()
	else: ymin, ymax = plot_yrange
	
	# indicate the regions to measure the silicate index
	# peak-like absorption feature
	ax.fill([water_max-water_window/2, water_max-water_window/2, 
	        water_max+water_window/2, water_max+water_window/2], 
	        [ymin, ymax, ymax, ymin], facecolor='silver', linewidth=1, zorder=2)
	# short-wavelength dip
	ax.fill([water_min1-water_window/2, water_min1-water_window/2, 
	        water_min1+water_window/2, water_min1+water_window/2], 
	        [ymin, ymax, ymax, ymin], facecolor='gainsboro', linewidth=1, zorder=2)
	# long-wavelength dip
	ax.fill([water_min2-water_window/2, water_min2-water_window/2, 
	        water_min2+water_window/2, water_min2+water_window/2], 
	        [ymin, ymax, ymax, ymin], facecolor='gainsboro', linewidth=1, zorder=2)

	# plot spectrum
	mask = wl>0
	plt.plot(wl[mask], flux[mask])

	# fluxes in the dip regions
	mask_water_min1 = (wl>=(water_min1-water_window/2)) & (wl<=(water_min1+water_window/2))
	mask_water_min2 = (wl>=(water_min2-water_window/2)) & (wl<=(water_min2+water_window/2))
	ax.errorbar(wl[mask_water_min1], flux[mask_water_min1], yerr=eflux[mask_water_min1], 
	            fmt='o', c='gray', markersize=3.0, linewidth=1.0, capsize=3, capthick=1)
	ax.errorbar(wl[mask_water_min2], flux[mask_water_min2], yerr=eflux[mask_water_min2], 
	            fmt='o', c='gray', markersize=3.0, linewidth=1.0, capsize=3, capthick=1)
	# fluxes in the emission-like region
	mask_water_max = (wl>=(water_max-water_window/2)) & (wl<=(water_max+water_window/2))
	ax.errorbar(wl[mask_water_max], flux[mask_water_max], yerr=eflux[mask_water_max],  
	            fmt='o', c='gray', markersize=3.0, linewidth=1.0, capsize=3, capthick=1)

	# mean flux in each dip region
	ax.errorbar(water_min1, flux_water_min1, yerr=eflux_water_min1, fmt='o', c='black', 
	            markersize=3.0, linewidth=1.0, capsize=3, capthick=1, zorder=3)
	ax.errorbar(water_min2, flux_water_min2, yerr=eflux_water_min2, fmt='o', c='black', 
	            markersize=3.0, linewidth=1.0, capsize=3, capthick=1, zorder=3)
	# mean flux in the emission-like region
	ax.errorbar(water_max, flux_water_max, yerr=eflux_water_max, fmt='o', c='red', 
	            markersize=3.0, linewidth=1.0, capsize=3, capthick=1, zorder=3)

	plt.text(0.55, 0.8, 'Water index='+str(round(water_index,2))+'$\pm$'+str(round(ewater_index,2)), transform=fig.transFigure, size=12)
	
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	ax.grid(True, which='both', color='gainsboro', alpha=0.5)
	
	plt.xlabel(r'$\lambda\ (\mu$m)', size=12)
	plt.ylabel(r'$F_\nu\ (Jy)$', size=12)
	if plot_title is not None: plt.title(plot_title, size=12)
	else: plt.title('Water Index Measurement', size=12)
	
	if plot_save: plt.savefig('water_index_measurement.pdf', bbox_inches='tight')
	plt.show()
	plt.close()

	return

##########################
# plot the methane index measurement
def plot_methane_index(out_methane_index, plot_xrange=None, plot_yrange=None, plot_title=None, plot_save=True):

	# read parameters of interest
	wl = out_methane_index['wl']
	flux = out_methane_index['flux']
	eflux = out_methane_index['eflux']
	methane_index = out_methane_index['methane_index']
	emethane_index = out_methane_index['emethane_index']
	flux_methane_in = out_methane_index['flux_methane_in']
	eflux_methane_in = out_methane_index['eflux_methane_in']
	flux_methane_out = out_methane_index['flux_methane_out']
	eflux_methane_out = out_methane_index['eflux_methane_out']
	methane_window = out_methane_index['methane_window']
	methane_in = out_methane_index['methane_in']
	methane_out = out_methane_index['methane_out']

	#++++++++++++++++++++++++++++++
	# plot silicate index measurement
	fig, ax = plt.subplots()

	if plot_xrange is None: xmin, xmax = 5.2, 14
	else: xmin, xmax = plot_xrange
	if plot_yrange is None:
		mask = (wl>=xmin) & (wl<=xmax)
		ymin, ymax = flux[mask].min(), flux[mask].max()
	else: ymin, ymax = plot_yrange
	
	# indicate the regions to measure the silicate index
	# in the absorption
	ax.fill([methane_in-methane_window/2, methane_in-methane_window/2, 
	        methane_in+methane_window/2, methane_in+methane_window/2], 
	        [ymin, ymax, ymax, ymin], facecolor='silver', linewidth=1, zorder=2)
	# out of the absorption
	ax.fill([methane_out-methane_window/2, methane_out-methane_window/2, 
	        methane_out+methane_window/2, methane_out+methane_window/2], 
	        [ymin, ymax, ymax, ymin], facecolor='gainsboro', linewidth=1, zorder=2)

	# plot spectrum
	mask = wl>0
	plt.plot(wl[mask], flux[mask])

	# fluxes out of the absorption
	mask_methane_out = (wl>=(methane_out-methane_window/2)) & (wl<=(methane_out+methane_window/2))
	ax.errorbar(wl[mask_methane_out], flux[mask_methane_out], yerr=eflux[mask_methane_out], 
	            fmt='o', c='gray', markersize=3.0, linewidth=1.0, capsize=3, capthick=1)
	# fluxes in the absorption
	mask_methane_in = (wl>=(methane_in-methane_window/2)) & (wl<=(methane_in+methane_window/2))
	ax.errorbar(wl[mask_methane_in], flux[mask_methane_in], yerr=eflux[mask_methane_in],  
	            fmt='o', c='gray', markersize=3.0, linewidth=1.0, capsize=3, capthick=1)

	# mean flux out of the absorption
	ax.errorbar(methane_out, flux_methane_out, yerr=eflux_methane_out, fmt='o', c='black', 
	            markersize=3.0, linewidth=1.0, capsize=3, capthick=1, zorder=3)
	# mean flux in the absorption
	ax.errorbar(methane_in, flux_methane_in, yerr=eflux_methane_in, fmt='o', c='red', 
	            markersize=3.0, linewidth=1.0, capsize=3, capthick=1, zorder=3)

	plt.text(0.52, 0.15, 'Methane index='+str(round(methane_index,2))+'$\pm$'+str(round(emethane_index,2)), transform=fig.transFigure, size=12)
	
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	ax.grid(True, which='both', color='gainsboro', alpha=0.5)
	
	plt.xlabel(r'$\lambda\ (\mu$m)', size=12)
	plt.ylabel(r'$F_\nu\ (Jy)$', size=12)
	if plot_title is not None: plt.title(plot_title, size=12)
	else: plt.title('Methane Index Measurement', size=12)
	
	if plot_save: plt.savefig('methane_index_measurement.pdf', bbox_inches='tight')
	plt.show()
	plt.close()

	return

##########################
# plot the ammonia index measurement
def plot_ammonia_index(out_ammonia_index, plot_xrange=None, plot_yrange=None, plot_title=None, plot_save=True):

	# read parameters of interest
	wl = out_ammonia_index['wl']
	flux = out_ammonia_index['flux']
	eflux = out_ammonia_index['eflux']
	ammonia_index = out_ammonia_index['ammonia_index']
	eammonia_index = out_ammonia_index['eammonia_index']
	flux_ammonia_in = out_ammonia_index['flux_ammonia_in']
	eflux_ammonia_in = out_ammonia_index['eflux_ammonia_in']
	flux_ammonia_out = out_ammonia_index['flux_ammonia_out']
	eflux_ammonia_out = out_ammonia_index['eflux_ammonia_out']
	ammonia_window = out_ammonia_index['ammonia_window']
	ammonia_in = out_ammonia_index['ammonia_in']
	ammonia_out = out_ammonia_index['ammonia_out']

	#++++++++++++++++++++++++++++++
	# plot silicate index measurement
	fig, ax = plt.subplots()

	if plot_xrange is None: xmin, xmax = 5.2, 14
	else: xmin, xmax = plot_xrange
	if plot_yrange is None:
		mask = (wl>=xmin) & (wl<=xmax)
		ymin, ymax = flux[mask].min(), flux[mask].max()
	else: ymin, ymax = plot_yrange
	
	# indicate the regions to measure the silicate index
	# in the absorption
	ax.fill([ammonia_in-ammonia_window/2, ammonia_in-ammonia_window/2, 
	        ammonia_in+ammonia_window/2, ammonia_in+ammonia_window/2], 
	        [ymin, ymax, ymax, ymin], facecolor='silver', linewidth=1, zorder=2)
	# out of the absorption
	ax.fill([ammonia_out-ammonia_window/2, ammonia_out-ammonia_window/2, 
	        ammonia_out+ammonia_window/2, ammonia_out+ammonia_window/2], 
	        [ymin, ymax, ymax, ymin], facecolor='gainsboro', linewidth=1, zorder=2)

	# plot spectrum
	mask = wl>0
	plt.plot(wl[mask], flux[mask])

	# fluxes out of the absorption
	mask_ammonia_out = (wl>=(ammonia_out-ammonia_window/2)) & (wl<=(ammonia_out+ammonia_window/2))
	ax.errorbar(wl[mask_ammonia_out], flux[mask_ammonia_out], yerr=eflux[mask_ammonia_out], 
	            fmt='o', c='gray', markersize=3.0, linewidth=1.0, capsize=3, capthick=1)
	# fluxes in the absorption
	mask_ammonia_in = (wl>=(ammonia_in-ammonia_window/2)) & (wl<=(ammonia_in+ammonia_window/2))
	ax.errorbar(wl[mask_ammonia_in], flux[mask_ammonia_in], yerr=eflux[mask_ammonia_in],  
	            fmt='o', c='gray', markersize=3.0, linewidth=1.0, capsize=3, capthick=1)

	# mean flux out of the absorption
	ax.errorbar(ammonia_out, flux_ammonia_out, yerr=eflux_ammonia_out, fmt='o', c='black', 
	            markersize=3.0, linewidth=1.0, capsize=3, capthick=1, zorder=3)
	# mean flux in the absorption
	ax.errorbar(ammonia_in, flux_ammonia_in, yerr=eflux_ammonia_in, fmt='o', c='red', 
	            markersize=3.0, linewidth=1.0, capsize=3, capthick=1, zorder=3)

	plt.text(0.52, 0.15, 'Ammonia index='+str(round(ammonia_index,2))+'$\pm$'+str(round(eammonia_index,2)), transform=fig.transFigure, size=12)
	
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	ax.grid(True, which='both', color='gainsboro', alpha=0.5)
	
	plt.xlabel(r'$\lambda\ (\mu$m)', size=12)
	plt.ylabel(r'$F_\nu\ (Jy)$', size=12)
	if plot_title is not None: plt.title(plot_title, size=12)
	else: plt.title('Ammonia Index Measurement', size=12)
	
	if plot_save: plt.savefig('ammonia_index_measurement.pdf', bbox_inches='tight')
	plt.show()
	plt.close()

	return


##########################
# manipulate input spectrum before measuring spectral indices
def handle_input_spectrum(wl, flux, eflux):

	# avoid nan values and zeros in wavelength
	mask_nan = (~np.isnan(flux)) & (~np.isnan(eflux)) & (wl>0)
	wl = wl[mask_nan]
	flux = flux[mask_nan]
	eflux = eflux[mask_nan]
	
	return wl, flux, eflux
