import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from sys import exit
from ..utils import *
from ..utils import normalize_flux
from numpy.typing import ArrayLike
from typing import Tuple, Literal, Optional, Union

##########################
def silicate_index(wl, flux, eflux, silicate_wl=None, silicate_window=None, 
	               continuum_wl1=None, continuum_window1=None, 
	               continuum_wl2=None, continuum_window2=None, 
	               continuum_fit='exponential', continuum_error='fit', reference='SM23', 
	               plot=False, plot_title=None, plot_xrange=None, 
	               plot_yrange=None, plot_save=False, plot_name=False):
	'''
	Description
	-----------
		Measure the strength of the mid-infrared silicate absorption considering the silicate index defined in Suárez & Metchev (2022,2023).

	Parameters
	----------
	- wl : array
		Spectrum wavelengths in microns.
	- flux : array
		Spectrum fluxes in Jy.
	- eflux : array
		Spectrum flux errors in Jy.
	- silicate_wl : float, optional (default 9.3 um)
		Wavelength reference to indicate the center of silicate absorption. 
	- silicate_window : float, optional (default 0.6 um)
		Wavelength window around ``silicate_wl`` used to calculate the average flux at the absorption.
	- continuum_wl1 : float, optional (default 7.45 um)
		Wavelength reference to indicate the short-wavelength continuum of silicate absorption. 
	- continuum_window1 : float, optional (default 0.5 um))
		Wavelength window around ``continuum_wl1`` used to select data points to fit a curve to continuum regions.
	- continuum_wl2 : float, optional (default 13.5 um)
		Wavelength reference to indicate the long-wavelength continuum of silicate absorption. 
	- continuum_window2 : float, optional (default 1.0 um)
		Wavelength window around ``continuum_wl2`` used to select data points to fit a curve to continuum regions.
	- reference : {``SM23``, ``SM22``}, optional (default ``SM23``)
		Reference to set default parameters to measure the silicate index.
		``SM23`` (default) for Suárez & Metchev (2023) and ``SM22`` for Suárez & Metchev (2022).
	- continuum_error : string, optional (default ``fit``)
		Label indicating the approach used to estimate the continuum flux uncertainty. Available options are: 
			- ``'fit'`` (default) : from the error of the curve fit.
			- ``'empirical'`` : from the scatter of the data points and the typical flux errors in the continuum regions.
	- continuum_fit : string, optional (default ``exponential``)
		Label indicating the curve fit to the continuum regions.
			- ``'line'`` : fit a line to continuum fluxes in both regions.
			- ``'exponential'`` (default) : fit an exponential (or a line in log-log space) to continuum fluxes in both regions.
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
	- plot_name : str, optional
		Filename to store the plot.
		Default is ``'Silicate_index_measurement.pdf'``.

	Returns
	-------
	- Dictionary 
		Dictionary with silicate index parameters:
			- ``'silicate_index'`` : silicate index
			- ``'esilicate_index'`` : silicate index uncertainty
			- ``'silicate_flux'`` : flux at the absorption feature
			- ``'esilicate_flux'`` : flux error at the absorption feature
			- ``'continuum_flux'`` : flux at the continuum of the absorption
			- ``'econtinuum_flux'`` : flux uncertainty at the continuum of the absorption
			- ``'slope'`` : slope of the fit to the continuum regions
			- ``'eslope'`` : slope uncertainty
			- ``'constant'`` : constant or intercept of the linear fit
			- ``'econstant'`` : constant uncertainty
			- ``'silicate_wl'`` : input ``silicate_wl``
			- ``'silicate_window'`` : input ``silicate_window``
			- ``'continuum_wl1'`` : input ``continuum_wl1``
			- ``'continuum_window1'`` : input ``continuum_window1``
			- ``'continuum_wl2'`` : input ``continuum_wl2``
			- ``'continuum_window2'`` : input ``continuum_window2``
			- ``'wl'`` : input ``wl``
			- ``'flux'`` : input ``flux``
			- ``'eflux'`` : input ``eflux``
	- Plot of the silicate index measurement that will be stored if ``plot_save``.

	Author: Genaro Suárez
	'''

	# handle input spectrum
	wl, flux, eflux = handle_input_spectrum(wl, flux, eflux)

	# use default values if optional values (peak and continuum regions) are not provided
	if reference=='SM23': # parameters in Suárez & Metchev (2023)
		# center of the absorption and window region
		if silicate_wl is None: silicate_wl = 9.30 # um
		if silicate_window is None: silicate_window = 0.6 # um
		# continuum region definition
		# where CH4 and NH3 are not expected for mid-L type
		if continuum_wl1 is None: continuum_wl1 = 7.45 # minimum in the blue side of the peak (um)
		if continuum_window1 is None: continuum_window1 = 0.5 # window in um for the short-wavelength continuum region
		if continuum_wl2 is None: continuum_wl2 = 13.5 # minimum in the red side of the peak (um)
		if continuum_window2 is None: continuum_window2 = 1.0 # window in um for the long-wavelength continuum region
	elif reference=='SM22': # parameters in Suárez & Metchev (2022)
		# center of the absorption and window region
		if silicate_wl is None: silicate_wl = 9.00 # um
		if silicate_window is None: silicate_window = 0.6 # um
		# continuum region definition
		if continuum_wl1 is None: continuum_wl1 = 7.5 # minimum in the blue side of the peak (um)
		if continuum_window1 is None: continuum_window1 = 0.6 # window in um for the short-wavelength continuum region
		if continuum_wl2 is None: continuum_wl2 = 11.5 # minimum in the red side of the peak (um)
		if continuum_window2 is None: continuum_window2 = 0.6 # window in um for the long-wavelength continuum region
	else: raise Exception(f'"{reference}" is not a recognized reference to set default parameters to measure the silicate index. \nTry "SM22" or "SM23".')

	# measure silicate index
	mask_silicate_wl = (wl>=(silicate_wl-silicate_window/2)) & (wl<=(silicate_wl+silicate_window/2))
	# using the mean
	silicate_flux = np.mean(flux[mask_silicate_wl]) # average flux at the bottom of the feature
	esilicate_flux = np.std(flux[mask_silicate_wl]) / np.sqrt(flux[mask_silicate_wl].size) # flux error as the standard error of the mean
	## using the median
	#silicate_flux = np.median(flux[mask_silicate_wl]) # average flux at the bottom of the feature
	#esilicate_flux = 1.2533 * np.std(flux[mask_silicate_wl]) / np.sqrt(flux[mask_silicate_wl].size) # standard error of the median = 1.2533 times the standard error of the mean
	##esilicate_flux = (np.percentile(flux[mask_silicate_wl], 84, interpolation = 'linear') - np.percentile(flux[mask_silicate_wl], 16, interpolation = 'linear')) / np.sqrt(flux[mask_silicate_wl].size) # 68% confidence interval
	##esilicate_flux = silicate_flux * np.median(eflux[mask_silicate_wl]/flux[mask_silicate_wl]) # error to keep the fractional uncertainties

	# continuum flux
	output = fit_continuum(wl=wl, flux=flux, eflux=eflux, wl_con1=continuum_wl1, wl_con2=continuum_wl2, 
	                       window_con1=continuum_window1, window_con2=continuum_window2, 
	                       continuum_fit=continuum_fit, continuum_error=continuum_error, ref_min=silicate_wl)
	slope = output['slope']
	eslope = output['eslope']
	constant = output['constant']
	econstant = output['econstant']
	continuum_flux = output['continuum_flux']
	econtinuum_flux = output['econtinuum_flux']

	# silicate index
	silicate_index = continuum_flux/silicate_flux
	esilicate_index = silicate_index * np.sqrt((econtinuum_flux/continuum_flux)**2 + (esilicate_flux/silicate_flux)**2)

	# output dictionary
	out = {'silicate_index': silicate_index, 'esilicate_index': esilicate_index, 'silicate_flux': silicate_flux, 
	       'esilicate_flux': esilicate_flux, 'continuum_flux': continuum_flux, 'econtinuum_flux': econtinuum_flux}
	# add parameter of the slope fit
	out['slope'] = slope
	out['eslope'] = eslope
	out['constant'] = constant
	out['econstant'] = econstant
	# add parameters used to measure the index
	out['silicate_wl'] = silicate_wl
	out['silicate_window'] = silicate_window
	out['continuum_wl1'] = continuum_wl1
	out['continuum_window1'] = continuum_window1
	out['continuum_wl2'] = continuum_wl2
	out['continuum_window2'] = continuum_window2
	out['continuum_fit'] = continuum_fit
	# add input spectrum
	out['wl'] = wl
	out['flux'] = flux
	out['eflux'] = eflux

	# visualize how the silicate index was measured
	index_name='Silicate'
	if plot_xrange is None: plot_xrange=[5.2, 14]
	if plot: plot_spectral_index_two_continuum_regions(out_feature_index=out, index_name=index_name, plot_xrange=plot_xrange, 
	                                                   plot_yrange=plot_yrange, plot_title=plot_title, plot_save=plot_save, plot_name=plot_name)

	return out

##########################
#def water_index(wl, flux, eflux, reference='SM22', water_window=None, continuum_wl=None, water_min1=None, water_min2=None,
#	            plot=False, plot_title=None, plot_xrange=None, plot_yrange=None, plot_save=False, plot_name=False):
def water_index(wl, flux, eflux, reference='SM22', 
	            continuum_wl=None, continuum_window=None,
	            water_wl1=None, water_window1=None,
	            water_wl2=None, water_window2=None,
	            plot=False, plot_title=None, plot_xrange=None, 
	            plot_yrange=None, plot_save=False, plot_name=False):
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
	- reference : {``SM22``, ``C06``}, optional (default ``SM22``)
		Reference to set default parameters to measure the water index.
		``SM22`` (default) for Suárez & Metchev (2022) or ``C08`` for Cushing et al (2006).
	- water_wl1 : float, optional (default 5.80 um)
		Wavelength reference to measure the flux in the first absorption dip.
	- water_window1 : float, optional (default 0.3 um)
		Wavelength window around ``water_wl1`` used to calculate the average flux.
	- water_wl2 : float, optional (default 6.75 um)
		Wavelength reference to measure the flux in the second absorption dip.
	- water_window2 : float, optional (default 0.3 um)
		Wavelength window around ``water_wl2`` used to calculate the average flux.
	- continuum_wl : float, optional (default 6.25 um)
		Wavelength reference to measure the flux in the feature pseudo-continuum.
	- continuum_window : float, optional (default 0.3 um)
		Wavelength window around ``continuum_wl`` used to calculate the average flux.
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
	- plot_name : str, optional
		Filename to store the plot.
		Default is ``'Water_index_measurement.pdf'``.

	Returns:
	--------
	- Dictionary 
		Dictionary with water index parameters:
			- ``'water_index'`` : water index
			- ``'ewater_index'`` : water index uncertainty
			- ``'water_flux1'`` : flux at the short-wavelength part of the absorption
			- ``'ewater_flux1'`` : flux uncertainty at the short-wavelength part of the absorption
			- ``'water_flux2'`` : flux at the long-wavelength part of the absorption
			- ``'ewater_flux2'`` : flux uncertainty at the long-wavelength part of the absorption
			- ``'continuum_flux'`` : flux at the absorption continuum
			- ``'econtinuum_flux'`` : flux uncertainty at the absorption continuum
			- ``'water_wl1'`` : input ``water_wl1``
			- ``'continuum_wl1'`` : input ``continuum_wl1``
			- ``'water_wl2'`` : input ``water_wl2``
			- ``'continuum_wl2'`` : input ``continuum_wl2``
			- ``'water_window'`` : input ``water_window``
			- ``'continuum_window'`` : input ``continuum_window``
			- ``'wl'`` : input ``wl``
			- ``'flux'`` : input ``flux``
			- ``'eflux'`` : input ``eflux``
	- Plot of the water index measurement that will be stored if ``plot_save``.

	Author: Genaro Suárez
	'''

	# handle input spectrum
	wl, flux, eflux = handle_input_spectrum(wl, flux, eflux)

	if reference=='SM22': # parameters in Suárez & Metchev (2022)
		if water_wl1 is None: water_wl1 = 5.80 # minimum in the blue side of the peak (um)
		if water_window1 is None: water_window1 = 0.30 # um
		if water_wl2 is None: water_wl2 = 6.75 # minimum in the red side of the peak (um)
		if water_window2 is None: water_window2 = 0.30 # um
		if continuum_wl is None: continuum_wl = 6.25 # flux peak (um)
		if continuum_window is None: continuum_window = 0.30 # um
	elif reference=='C06': # parameters in Cushing et al. (2006)
		if water_wl1 is None: water_wl1 = 5.80 # minimum in the blue side of the peak (um)
		if water_window1 is None: water_window1 = 0.15 # um
		if water_wl2 is None: water_wl2 = 6.75 # minimum in the red side of the peak (um)
		if water_window2 is None: water_window2 = 0.30 # um
		if continuum_wl is None: continuum_wl = 6.25 # flux peak (um)
		if continuum_window is None: continuum_window = 0.30 # um
	else: raise Exception(f'"{reference}" is not a recognized reference to set default parameters to measure the water index. \nTry "SM22" or "C06".')
	
	# mean flux at the pseudo-continuum
	mask_continuum_wl = (wl>=(continuum_wl-continuum_window/2)) & (wl<=(continuum_wl+continuum_window/2))
	continuum_flux = np.mean(flux[mask_continuum_wl])
	#econtinuum_flux = np.std(flux[mask_continuum_wl]) / np.sqrt(flux[mask_continuum_wl].size) # in Suarez & Metchev (2022)
	econtinuum_flux_1 = np.std(flux[mask_continuum_wl]) / np.sqrt(flux[mask_continuum_wl].size) # due to the scatter of the data in the window
	econtinuum_flux_2 = continuum_flux * np.median(eflux[mask_continuum_wl]/flux[mask_continuum_wl]) # due to flux uncertainties
	econtinuum_flux = np.sqrt(econtinuum_flux_1**2 + econtinuum_flux_2**2) # addition in quadrature

	# mean flux at the short-wavelength dip
	mask_water_wl1 = (wl>=(water_wl1-water_window1/2)) & (wl<=(water_wl1+water_window1/2))
	water_flux1 = np.mean(flux[mask_water_wl1])
	#ewater_flux1 = np.std(flux[mask_water_wl1]) / np.sqrt(flux[mask_water_wl1].size) # in Suarez & Metchev (2022)
	ewater_flux1_1 = np.std(flux[mask_water_wl1]) / np.sqrt(flux[mask_water_wl1].size) # due to the scatter of the data in the window
	ewater_flux1_2 = water_flux1 * np.median(eflux[mask_water_wl1]/flux[mask_water_wl1]) # due to flux uncertainties
	ewater_flux1 = np.sqrt(ewater_flux1_1**2 + ewater_flux1_2**2) # addition in quadrature

	# mean flux at the long-wavelength dip
	mask_water_wl2 = (wl>=(water_wl2-water_window2/2)) & (wl<=(water_wl2+water_window2/2))
	water_flux2 = np.mean(flux[mask_water_wl2])
	#ewater_flux2 = np.std(flux[mask_water_wl2]) / np.sqrt(flux[mask_water_wl2].size) # in Suarez & Metchev (2022)
	ewater_flux2_1 = np.std(flux[mask_water_wl2]) / np.sqrt(flux[mask_water_wl2].size) # due to the scatter of the data in the window
	ewater_flux2_2 = water_flux2 * np.median(eflux[mask_water_wl2]/flux[mask_water_wl2]) # due to flux uncertainties
	ewater_flux2 = np.sqrt(ewater_flux2_1**2 + ewater_flux2_2**2) # addition in quadrature
	
	# water index
	weight1 = 0.562
	weight2 = 0.474
	water_index = continuum_flux / (weight1*water_flux1 + weight2*water_flux2)
	ewater_index = water_index * np.sqrt((econtinuum_flux/continuum_flux)**2 + 
	                                    ((weight1*ewater_flux1)/(weight1*water_flux1 + weight2*water_flux2))**2 + 
	                                    ((weight2*ewater_flux2)/(weight1*water_flux1 + weight2*water_flux2))**2)

	# output dictionary
	out = {'water_index': water_index, 'ewater_index': ewater_index, 'continuum_flux': continuum_flux, 
	       'econtinuum_flux': econtinuum_flux, 'water_flux1': water_flux1, 'ewater_flux1': ewater_flux1, 
	       'water_flux2': water_flux2, 'ewater_flux2': ewater_flux2}
	# add parameters used to measure the index
	out['water_wl1'] = water_wl1
	out['water_window1'] = water_window1
	out['water_wl2'] = water_wl2
	out['water_window2'] = water_window2
	out['continuum_wl'] = continuum_wl
	out['continuum_window'] = continuum_window
	# add input spectrum
	out['wl'] = wl
	out['flux'] = flux
	out['eflux'] = eflux

	# visualize how the silicate index was measured
	#if plot: plot_water_index(out, plot_xrange, plot_yrange, plot_title, plot_save, plot_name)
	index_name='Water'
	if plot_xrange is None: plot_xrange=[5.2, 14]
	if plot: plot_spectral_index_one_continuum_region(out_feature_index=out, index_name=index_name, plot_xrange=plot_xrange, 
	                                                  plot_yrange=plot_yrange, plot_title=plot_title, plot_save=plot_save, plot_name=plot_name)

	return out

##########################
def methane_index(wl, flux, eflux, reference='SM22', 
	              methane_wl=None, methane_window=None, 
	              continuum_wl=None, continuum_window=None, 
	              plot=False, plot_title=None, plot_xrange=None, 
	              plot_yrange=None, plot_save=False, plot_name=False):
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
	- reference : {``SM22``, ``C06``}, optional (default ``SM22``)
		Reference to set default parameters to measure the methane index.
		``SM22`` (default) for Suárez & Metchev (2022) or ``C08`` for Cushing et al (2006).
	- methane_wl : float, optional (default 7.65 um)
		Wavelength reference of the feature.
	- methane_window : float, optional (default 0.6 um)
		Wavelength window around ``methane_wl`` used to calculate average fluxes.
	- continuum_wl : float, optional (default 9.9 um)
		Wavelength reference of the feature continuum.
		Note: the default value is slightly smaller than the 10 um value in Suárez & Metchev (2022) to avoid including fluxes at the beginning of the ammonia feature.
	- continuum_window : float, optional (default 0.6 um)
		Wavelength window around ``continuum_wl`` used to calculate the average continuum flux.
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
	- plot_name : str, optional
		Filename to store the plot.
		Default is ``'Methane_index_measurement.pdf'``.

	Returns:
	--------
	- Dictionary 
		Dictionary with methane index parameters:
			- ``'methane_index'`` : methane index
			- ``'emethane_index'`` : methane index uncertainty
			- ``'methane_flux'`` : flux at the absorption
			- ``'emethane_flux'`` : flux uncertainty at the absorption
			- ``'continuum_flux'`` : flux at the absorption continuum
			- ``'econtinuum_flux'`` : flux uncertainty at the absorption continuum
			- ``'methane_wl'`` : input ``methane_wl``
			- ``'continuum_wl'`` : input ``continuum_wl``
			- ``'methane_window'`` : input ``methane_window``
			- ``'continuum_window'`` : input ``continuum_window``
			- ``'wl'`` : input ``wl``
			- ``'flux'`` : input ``flux``
			- ``'eflux'`` : input ``eflux``
	- Plot of the methane index measurement that will be stored if ``plot_save``.

	Author: Genaro Suárez
	'''

	# handle input spectrum
	wl, flux, eflux = handle_input_spectrum(wl, flux, eflux)

	if reference=='SM22': # parameters in Suárez & Metchev (2022)
		if methane_wl is None: methane_wl = 7.65 # (um) wavelength point in of the feature
		if continuum_wl is None: continuum_wl = 9.9 # (um) wavelength point out of the feature. Note: it is actually 10 um in Suárez & Metchev (2022), 
		                                            # but includes a few data points within the ammonia feature.
		if methane_window is None: methane_window = 0.6 # (um) wavelength window
		if continuum_window is None: continuum_window = 0.6 # (um) wavelength window
	elif reference=='C06': # parameters in Cushing et al. (2006)
		if methane_wl is None: methane_wl = 8.5 # (um) wavelength point in of the feature
		if continuum_wl is None: continuum_wl = 10.0 # (um) wavelength point out of the feature 
		if methane_window is None: methane_window = 0.3 # (um) wavelength window
		if continuum_window is None: continuum_window = 0.3 # (um) wavelength window

	# mean flux in the feature
	mask_methane_wl = (wl>=(methane_wl-methane_window/2)) & (wl<=(methane_wl+methane_window/2))
	methane_flux = np.mean(flux[mask_methane_wl])
	#emethane_flux = np.std(flux[mask_methane_wl]) / np.sqrt(flux[mask_methane_wl].size) # in Suarez & Metchev (2022)
	emethane_flux_1 = np.std(flux[mask_methane_wl]) / np.sqrt(flux[mask_methane_wl].size) # due to the scatter of the data in the window
	emethane_flux_2 = methane_flux * np.median(eflux[mask_methane_wl]/flux[mask_methane_wl]) # due to flux uncertainties
	emethane_flux = np.sqrt(emethane_flux_1**2 + emethane_flux_2**2) # addition in quadrature
	
	# mean flux out of the feature
	mask_continuum_wl = (wl>=(continuum_wl-continuum_window/2)) & (wl<=(continuum_wl+continuum_window/2))
	continuum_flux = np.mean(flux[mask_continuum_wl])
	#econtinuum_flux = np.std(flux[mask_continuum_wl]) / np.sqrt(flux[mask_continuum_wl].size) # in Suarez & Metchev (2022)
	econtinuum_flux_1 = np.std(flux[mask_continuum_wl]) / np.sqrt(flux[mask_continuum_wl].size) # due to the scatter of the data in the window
	econtinuum_flux_2 = continuum_flux * np.median(eflux[mask_continuum_wl]/flux[mask_continuum_wl]) # due to flux uncertainties
	econtinuum_flux = np.sqrt(econtinuum_flux_1**2 + econtinuum_flux_2**2) # addition in quadrature

	# methane index
	methane_index = continuum_flux / methane_flux
	emethane_index = methane_index * np.sqrt((emethane_flux/methane_flux)**2 + (econtinuum_flux/continuum_flux)**2)

	# output dictionary
	out = {'methane_index': methane_index, 'emethane_index': emethane_index, 'methane_flux': methane_flux, 
	       'emethane_flux': emethane_flux, 'continuum_flux': continuum_flux, 'econtinuum_flux': econtinuum_flux}
	# add parameters used to measure the index
	out['methane_wl'] = methane_wl
	out['continuum_wl'] = continuum_wl
	out['methane_window'] = methane_window
	out['continuum_window'] = continuum_window
	# add input spectrum
	out['wl'] = wl
	out['flux'] = flux
	out['eflux'] = eflux
	
	# visualize how the silicate index was measured
	index_name='Methane'
	if plot_xrange is None: plot_xrange=[5.2, 14]
	if plot: plot_spectral_index_one_continuum_region(out_feature_index=out, index_name=index_name, plot_xrange=plot_xrange, 
	                                                  plot_yrange=plot_yrange, plot_title=plot_title, plot_save=plot_save, plot_name=plot_name)

	return out

##########################
def ammonia_index(wl, flux, eflux, reference='SM22', 
	              ammonia_wl=None, ammonia_window=None, 
	              continuum_wl=None, continuum_window=None, 
	              plot=False, plot_title=None, plot_xrange=None, 
	              plot_yrange=None, plot_save=False, plot_name=False):
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
	- reference : {``SM22``, ``C06``}, optional (default ``SM22``)
		Reference to set default parameters to measure the ammonia index.
		``SM22`` (default) for Suárez & Metchev (2022) or ``C08`` for Cushing et al (2006).
	- ammonia_wl : float, optional (default 10.6 um)
		Wavelength reference of the feature.
		Note: the default value is slightly smaller than the 10.8 um value in Suárez & Metchev (2022) to be centered better within the feature.
	- ammonia_window : float, optional (default 0.6 um)
		Wavelength window around ``ammonia_wl`` and ``continuum_wl`` used to calculate average fluxes.
	- continuum_wl : float, optional (default 9.9 um)
		Wavelength reference of the feature continuum.
		Note: the default value is slightly smaller than the 10 um value in Suárez & Metchev (2022) to avoid including fluxes at the beginning of the ammonia feature.
	- continuum_window : float, optional (default 0.6 um)
		Wavelength window around ``continuum_wl`` used to calculate the average continuum flux.
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
	- plot_name : str, optional
		Filename to store the plot.
		Default is ``'Ammonia_index_measurement.pdf'``.

	Returns:
	--------
	- Dictionary 
		Dictionary with ammonia index parameters:
			- ``'ammonia_index'`` : ammonia index
			- ``'eammonia_index'`` : ammonia index uncertainty
			- ``'ammonia_flux'`` : flux at the absorption
			- ``'eammonia_flux'`` : flux uncertainty at the absorption
			- ``'continuum_flux'`` : flux at the absorption continuum
			- ``'econtinuum_flux'`` : flux uncertainty at the absorption continuum
			- ``'ammonia_wl'`` : input ``ammonia_wl``
			- ``'continuum_wl'`` : input ``continuum_wl``
			- ``'ammonia_window'`` : input ``ammonia_window``
			- ``'continuum_window'`` : input ``continuum_window``
			- ``'wl'`` : input ``wl``
			- ``'flux'`` : input ``flux``
			- ``'eflux'`` : input ``eflux``
	- Plot of the ammonia index measurement that will be stored if ``plot_save``.

	Author: Genaro Suárez
	'''

	# handle input spectrum
	wl, flux, eflux = handle_input_spectrum(wl, flux, eflux)

	if reference=='SM22': # parameters in Suárez & Metchev (2022)
		#if ammonia_wl is None: ammonia_wl = 10.8 # (um) wavelength point in of the feature
		#if continuum_wl is None: continuum_wl = 10.0 # (um) wavelength point out of the feature. Note: it is actually 10 um in Suárez & Metchev (2022), 
		                                              # but includes a few data points within the ammonia feature.
		if ammonia_wl is None: ammonia_wl = 10.6 # (um) wavelength point in of the feature. Note: it is 10.8 um in Suárez & Metchev (2022), 
		                                         # but includes it is not well center within the absorption.
		if continuum_wl is None: continuum_wl = 9.9 # (um) wavelength point out of the feature. Note: it is 10 um in Suárez & Metchev (2022), 
		                                            # but includes a few data points within the ammonia feature.
		if ammonia_window is None: ammonia_window = 0.6 # (um) wavelength window
		if continuum_window is None: continuum_window = 0.6 # (um) wavelength window
	elif reference=='C06': # parameters in Cushing et al. (2006)
		if ammonia_wl is None: ammonia_wl = 8.5 # (um) wavelength point in of the feature
		if continuum_wl is None: continuum_wl = 10.0 # (um) wavelength point out of the feature 
		if ammonia_window is None: ammonia_window = 0.3 # (um) wavelength window
		if continuum_window is None: continuum_window = 0.3 # (um) wavelength window

	# mean flux in the feature
	mask_ammonia_wl = (wl>=(ammonia_wl-ammonia_window/2)) & (wl<=(ammonia_wl+ammonia_window/2))
	ammonia_flux = np.mean(flux[mask_ammonia_wl])
	#eammonia_flux = np.std(flux[mask_ammonia_wl]) / np.sqrt(flux[mask_ammonia_wl].size) # in Suarez & Metchev (2022)
	eammonia_flux_1 = np.std(flux[mask_ammonia_wl]) / np.sqrt(flux[mask_ammonia_wl].size) # due to the scatter of the data in the window
	eammonia_flux_2 = ammonia_flux * np.median(eflux[mask_ammonia_wl]/flux[mask_ammonia_wl]) # due to flux uncertainties
	eammonia_flux = np.sqrt(eammonia_flux_1**2 + eammonia_flux_2**2) # addition in quadrature
	
	# mean flux out of the feature
	mask_continuum_wl = (wl>=(continuum_wl-continuum_window/2)) & (wl<=(continuum_wl+continuum_window/2))
	continuum_flux = np.mean(flux[mask_continuum_wl])
	#econtinuum_flux = np.std(flux[mask_continuum_wl]) / np.sqrt(flux[mask_continuum_wl].size) # in Suarez & Metchev (2022)
	econtinuum_flux_1 = np.std(flux[mask_continuum_wl]) / np.sqrt(flux[mask_continuum_wl].size) # due to the scatter of the data in the window
	econtinuum_flux_2 = continuum_flux * np.median(eflux[mask_continuum_wl]/flux[mask_continuum_wl]) # due to flux uncertainties
	econtinuum_flux = np.sqrt(econtinuum_flux_1**2 + econtinuum_flux_2**2) # addition in quadrature

	# ammonia index
	ammonia_index = continuum_flux / ammonia_flux
	eammonia_index = ammonia_index * np.sqrt((eammonia_flux/ammonia_flux)**2 + (econtinuum_flux/continuum_flux)**2)

	# output dictionary
	out = {'ammonia_index': ammonia_index, 'eammonia_index': eammonia_index, 'ammonia_flux': ammonia_flux, 
	       'eammonia_flux': eammonia_flux, 'continuum_flux': continuum_flux, 'econtinuum_flux': econtinuum_flux}
	# add parameters used to measure the index
	out['ammonia_wl'] = ammonia_wl
	out['ammonia_window'] = ammonia_window
	out['continuum_wl'] = continuum_wl
	out['continuum_window'] = continuum_window
	# add input spectrum
	out['wl'] = wl
	out['flux'] = flux
	out['eflux'] = eflux

	# visualize how the silicate index was measured
	index_name='Ammonia'
	if plot_xrange is None: plot_xrange=[5.2, 14]
	if plot: plot_spectral_index_one_continuum_region(out_feature_index=out, index_name=index_name, plot_xrange=plot_xrange, 
	                                                  plot_yrange=plot_yrange, plot_title=plot_title, plot_save=plot_save, plot_name=plot_name)

	return out

##########################
def user_index(wl, flux, eflux, feature_wl, feature_window, 
	           continuum_wl=None, continuum_window=None,
	           continuum_wl1=None, continuum_window1=None,
	           continuum_wl2=None, continuum_window2=None,
	           index_name=None, continuum_fit=None, continuum_error='fit', 
	           plot=False, plot_title=None, plot_xrange=None, 
	           plot_yrange=None, plot_save=False, plot_name=False):
	'''
	Description:
	------------
		Spectral index defined by the user as the ratio between the feature continuum and the feature flux.

	Parameters:
	-----------
	- wl : array
		Spectrum wavelengths in microns.
	- flux : array
		Spectrum fluxes in any flux units.
	- eflux : array
		Spectrum flux errors in the same units as ``flux``.
	- feature_wl : float
		Wavelength reference to indicate the center of feature absorption. 
	- feature_window : float
		Wavelength window around ``feature_wl`` used to calculate the average flux at the absorption.
	- continuum_wl : float, optional (required if ``continuum_fit=None``)
		Wavelength reference to indicate the continuum of feature absorption. 
	- continuum_window : float, optional (required if ``continuum_fit=None``)
		Wavelength window around ``continuum_wl`` used to calculate the average continuum flux.
	- continuum_wl1 : float, optional (required if ``continuum_fit!=None``)
		Wavelength reference to indicate the short-wavelength continuum of the feature absorption. 
	- continuum_window1 : float, optional (required if ``continuum_fit!=None``)
		Wavelength window around ``continuum_wl1`` used to select data points to fit a curve to continuum regions.
	- continuum_wl2 : float, optional (required if ``continuum_fit!=None``)
		Wavelength reference to indicate the long-wavelength continuum of the feature absorption. 
	- continuum_window2 : float, optional (required if ``continuum_fit!=None``)
		Wavelength window around ``continuum_wl2`` used to select data points to fit a curve to continuum regions.
	- index_name : str (optional)
		Name the user wants to give to the index to be included in the plot label and title (if not ``plot_title``). 
		If not provided, the string ``User-defined`` will be used.
	- continuum_error : string, optional (required if ``continuum_fit!=None``; default ``fit``)
		Label indicating the approach used to estimate the continuum flux uncertainty. Available options are: 
			- ``'fit'`` (default): from the error of the curve fit.
			- ``'empirical'``: from the scatter of the data points and the typical flux errors in the continuum regions.
	- continuum_fit : string, optional (required if ``continuum_fit!=None``)
		Label indicating the curve fit to the continuum regions.
			- ``'line'``: fit a line to continuum fluxes in both regions.
			- ``'exponential'``: fit an exponential (or a line in log-log space) to continuum fluxes in both regions.
	- plot : {``True``, ``False``}, optional (default ``False``)
		Plot (``True``) or do not plot (``False``) the feature index measurement.
	- plot_title : str, optional
		Plot title (default ``'{index_name} Index Measurement'``.
	- plot_xrange : list or array
		Wavelength range (in same units as ``wl``) of the plot (default is min and max values of ``wl``).
	- plot_yrange : list or array
		Flux range (in same units as ``flux``) of the plot (default is the flux range in ``plot_xrange``).
	- plot_save : {``True``, ``False``}, optional (default ``False``)
		Save (``'True'``) or do not save (``'False'``) the resulting plot.
	- plot_name : str, optional
		Filename to store the plot.
		Default is ``'{index_name}_index_measurement.pdf'``.

	Returns:
	--------
	- Dictionary 
		Dictionary with user-defined index parameters:
			- ``'feature_index'`` : feature index
			- ``'efeature_index'`` : feature index uncertainty
			- ``'feature_flux'`` : flux at the absorption feature
			- ``'efeature_flux'`` : flux error at the absorption feature
			- ``'continuum_flux'`` : flux at the continuum of the absorption
			- ``'econtinuum_flux'`` : flux uncertainty at the continuum of the absorption
			- ``'slope'`` : (if continuum_fit!=None) slope of the fit to the continuum regions
			- ``'eslope'`` : (if continuum_fit!=None) slope uncertainty
			- ``'constant'`` : (if continuum_fit!=None) constant or intercept of the linear fit
			- ``'econstant'`` : (if continuum_fit!=None) constant uncertainty
			- ``'feature_wl'`` : input ``feature_wl``
			- ``'feature_window'`` : input ``feature_window``
			- ``'continuum_wl1'`` : input ``continuum_wl1``
			- ``'continuum_window1'`` : input ``continuum_window1``
			- ``'continuum_wl2'`` : input ``continuum_wl2``
			- ``'continuum_window2'`` : input ``continuum_window2``
			- ``'wl'`` : input ``wl``
			- ``'flux'`` : input ``flux``
			- ``'eflux'`` : input ``eflux``
	- Plot of the user-defined index measurement that will be stored if ``plot_save``.

	Author: Genaro Suárez
	'''

	# handle input spectrum
	wl, flux, eflux = handle_input_spectrum(wl, flux, eflux)

	# verification that required input parameters are provided
	if continuum_fit is None: # when continuum will be measured in one continuum wavelength region
		if continuum_wl is None: raise Exception(f'"continuum_wl" must be provided if "continuum_fit" is not provided or define "continuum_fit".')
		if continuum_window is None: raise Exception(f'"continuum_window" must be provided if "continuum_fit" is not provided or define "continuum_fit".')
	else: # when continuum will be measured from the fit to two continuum wavelength regions
		if continuum_wl1 is None: raise Exception(f'"continuum_wl1" must be provided if "continuum_fit" is provided.')
		if continuum_window1 is None: raise Exception(f'"continuum_window1" must be provided if "continuum_fit" is provided.')
		if continuum_wl2 is None: raise Exception(f'"continuum_wl2" must be provided if "continuum_fit" is provided.')
		if continuum_window2 is None: raise Exception(f'"continuum_window2" must be provided if "continuum_fit" is provided.')

	# mean flux in the feature
	mask_feature_wl = (wl>=(feature_wl-feature_window/2)) & (wl<=(feature_wl+feature_window/2))
	feature_flux = np.mean(flux[mask_feature_wl])
	efeature_flux_1 = np.std(flux[mask_feature_wl]) / np.sqrt(flux[mask_feature_wl].size) # due to the scatter of the data in the window
	efeature_flux_2 = feature_flux * np.median(eflux[mask_feature_wl]/flux[mask_feature_wl]) # due to flux uncertainties
	efeature_flux = np.sqrt(efeature_flux_1**2 + efeature_flux_2**2) # addition in quadrature

	# continuum flux
	if continuum_fit is None: # do not fit a curve to the continuum
		# mean flux in the feature continuum region
		mask_continuum_wl = (wl>=(continuum_wl-continuum_window/2)) & (wl<=(continuum_wl+continuum_window/2))
		continuum_flux = np.mean(flux[mask_continuum_wl])
		econtinuum_flux_1 = np.std(flux[mask_continuum_wl]) / np.sqrt(flux[mask_continuum_wl].size) # due to the scatter of the data in the window
		econtinuum_flux_2 = continuum_flux * np.median(eflux[mask_continuum_wl]/flux[mask_continuum_wl]) # due to flux uncertainties
		econtinuum_flux = np.sqrt(econtinuum_flux_1**2 + econtinuum_flux_2**2) # addition in quadrature
	else: # fit a curve to the continuum
		output = fit_continuum(wl=wl, flux=flux, eflux=eflux, wl_con1=continuum_wl1, wl_con2=continuum_wl2,
		                       window_con1=continuum_window1, window_con2=continuum_window1,
		                       continuum_fit=continuum_fit, continuum_error=continuum_error, ref_min=feature_wl)
		slope = output['slope']
		eslope = output['eslope']
		constant = output['constant']
		econstant = output['econstant']
		continuum_flux = output['continuum_flux']
		econtinuum_flux = output['econtinuum_flux']

	# defined index
	feature_index = continuum_flux / feature_flux
	efeature_index = feature_index * np.sqrt((efeature_flux/feature_flux)**2 + (econtinuum_flux/continuum_flux)**2)

	# output dictionary
	out = {'feature_index': feature_index, 'efeature_index': efeature_index, 'feature_flux': feature_flux, 
	       'efeature_flux': efeature_flux, 'continuum_flux': continuum_flux, 'econtinuum_flux': econtinuum_flux}
	# add parameters used to measure the index
	out['feature_wl'] = feature_wl
	out['feature_window'] = feature_window
	out['continuum_wl'] = continuum_wl
	out['continuum_window'] = continuum_window
	out['continuum_wl1'] = continuum_wl1
	out['continuum_window1'] = continuum_window1
	out['continuum_wl2'] = continuum_wl2
	out['continuum_window2'] = continuum_window2
	out['continuum_fit'] = continuum_fit
	# add parameter of the slope fit
	if continuum_fit is not None: # do not fit a curve to the continuum
		out['slope'] = slope
		out['eslope'] = eslope
		out['constant'] = constant
		out['econstant'] = econstant
	# add input spectrum
	out['wl'] = wl
	out['flux'] = flux
	out['eflux'] = eflux
	
	# visualize how the silicate index was measured
	if index_name is None: index_name = 'User-defined'
	if continuum_fit is None: # when continuum will be measured in one continuum wavelength region
		if plot: plot_spectral_index_one_continuum_region(out_feature_index=out, index_name=index_name, plot_xrange=plot_xrange, 
		                                                  plot_yrange=plot_yrange, plot_title=plot_title, plot_save=plot_save, plot_name=plot_name)
	else: # when continuum will be measured from the fit to two continuum wavelength regions
		if plot: plot_spectral_index_two_continuum_regions(out_feature_index=out, index_name=index_name, plot_xrange=plot_xrange, 
		                                                   plot_yrange=plot_yrange, plot_title=plot_title, plot_save=plot_save, plot_name=plot_name)

	return out


#########################################

def user_index_integral(
    wavelength,
    flux,
    num_range: Tuple[float, float],
    den_range: Tuple[float, float],
    *,
    mode: Literal["ratio", "difference"] = "ratio",
    normalize: bool = False,
    plot: bool = False,
    plot_save: Union[bool, str] = False,
) -> float:
    """
    Description
	-----------
    Compute a near-infrared spectral index as an integrated flux ratio or difference,
    with optional normalization and plotting of numerator/denominator regions.

    Parameters
    ----------
    wavelength : array-like
        Wavelength array (typically in microns).
    flux : array-like
        Flux array corresponding to `wavelength`. Assumed 1D, same length as `wavelength`.
    num_range : tuple of float
        Wavelength limits (λ_min, λ_max) for the numerator bandpass.
    den_range : tuple of float
        Wavelength limits (λ_min, λ_max) for the denominator bandpass.
    mode : {"ratio", "difference"}, optional
        Definition of the index.

        - ``"ratio"``: index = ∬ F_num / ∬ F_den
        - ``"difference"``: index = ∬ F_den − ∬ F_num
    normalize : bool, optional
        If True, flux is median-normalized (ignoring NaNs) before computing the index.
    plot : bool, optional
        If True, plot the spectrum and the two bandpasses.
    plot_save : bool or str, optional
        If True, saves to default filename ``"user_index.pdf"``.
        If str, saves to that specific path.

    Returns
    -------
    float
        Spectral index value.

    Notes
    -----
    For *ratio*-type indices, a global flux normalization typically cancels out and
    does not change the numerical value of the index. However, for *difference*-type
    indices (e.g., J–H defined as ∬F_den − ∬F_num), normalization directly affects
    the absolute scale of the index and therefore the boundaries of variability or
    classification regions. Users should ensure that the same normalization
    convention is applied consistently to both the target spectrum and any
    reference/template spectra.

    Author
    ------
    Natalia Oliveros-Gomez
    """

    wave = np.asarray(wavelength, dtype=float)
    flx = np.asarray(flux, dtype=float)

    if wave.shape != flx.shape:
        raise ValueError("`wavelength` and `flux` must have the same shape.")

    if normalize:
        flx = normalize_flux(flx)

    m_num = (wave >= num_range[0]) & (wave <= num_range[1])
    m_den = (wave >= den_range[0]) & (wave <= den_range[1])

    if not np.any(m_num):
        raise ValueError(f"No data points within numerator range {num_range}.")
    if not np.any(m_den):
        raise ValueError(f"No data points within denominator range {den_range}.")

    num_val = np.trapz(flx[m_num], wave[m_num])
    den_val = np.trapz(flx[m_den], wave[m_den])

    if mode == "ratio":
        if den_val == 0:
            raise ZeroDivisionError(
                f"Denominator integral is zero for range {den_range}."
            )
        index_value = num_val / den_val
    elif mode == "difference":
        index_value = den_val - num_val
    else:
        raise ValueError(f"Unknown mode={mode!r}. Use 'ratio' or 'difference'.")

    # ---- Optional plotting ----
    if plot:
        if isinstance(plot_save, str):
            savepath = plot_save
        elif plot_save is True:
            savepath = "user_index.pdf"
        else:
            savepath = None

        plot_user_index_nir(
            wave,
            flx,
            num_range,
            den_range,
            mode=mode,
            index_value=index_value,
            savepath=savepath,
        )

    return index_value

###################
# plot the spectral index as measured in the wavelength region


def plot_user_index_nir(
    wavelength,
    flux,
    num_range,
    den_range,
    *,
    mode,
    index_value,
    savepath=None,
):
    
    #Internal plotter for a single NIR index showing numerator/denominator windows.

    wave = np.asarray(wavelength)
    flx = np.asarray(flux)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Plot spectrum
    ax.plot(wave, flx, color="black", lw=1.2, label="Spectrum")

    # Highlight denominator (blue)
    ax.axvspan(
        den_range[0], den_range[1],
        color="cornflowerblue", alpha=0.25,
        label="Denominator"
    )

    # Highlight numerator (red)
    ax.axvspan(
        num_range[0], num_range[1],
        color="lightcoral", alpha=0.25,
        label="Numerator"
    )

    ax.set_xlabel("Wavelength [µm]")
    ax.set_ylabel("Flux")
    ax.set_title(f"User index ({mode}) = {index_value:.4f}")

    ax.set_xlim(wave.min(), wave.max())
    ax.legend(loc="upper right", fontsize=10)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")

    plt.show()
    return fig, ax


##########################
# plot the spectral index is measured in one continuum wavelength region
def plot_spectral_index_one_continuum_region(out_feature_index, index_name=None, plot_xrange=None, plot_yrange=None, plot_title=None, plot_save=True, plot_name=False):

	# read parameters of interest
	# spectrum
	wl = out_feature_index['wl']
	flux = out_feature_index['flux']
	eflux = out_feature_index['eflux']
	if index_name=='Water': # specific variables names for water index
		# index
		feature_index = out_feature_index['water_index']
		efeature_index = out_feature_index['ewater_index']
		feature_flux = out_feature_index['water_flux1']
		efeature_flux = out_feature_index['ewater_flux1']
		feature_flux2 = out_feature_index['water_flux2']
		efeature_flux2 = out_feature_index['ewater_flux2']
		continuum_flux = out_feature_index['continuum_flux']
		econtinuum_flux = out_feature_index['econtinuum_flux']
		# index parameters
		feature_wl = out_feature_index['water_wl1']
		feature_window = out_feature_index['water_window1']
		feature_wl2 = out_feature_index['water_wl2']
		feature_window2 = out_feature_index['water_window2']
		continuum_wl = out_feature_index['continuum_wl']
		continuum_window = out_feature_index['continuum_window']
	elif index_name=='Methane': # specific variables names for water index
		# index
		feature_index = out_feature_index['methane_index']
		efeature_index = out_feature_index['emethane_index']
		feature_flux = out_feature_index['methane_flux']
		efeature_flux = out_feature_index['emethane_flux']
		continuum_flux = out_feature_index['continuum_flux']
		econtinuum_flux = out_feature_index['econtinuum_flux']
		# index parameters
		feature_wl = out_feature_index['methane_wl']
		feature_window = out_feature_index['methane_window']
		continuum_wl = out_feature_index['continuum_wl']
		continuum_window = out_feature_index['continuum_window']
	elif index_name=='Ammonia': # specific variables names for ammonia index
		# index
		feature_index = out_feature_index['ammonia_index']
		efeature_index = out_feature_index['eammonia_index']
		feature_flux = out_feature_index['ammonia_flux']
		efeature_flux = out_feature_index['eammonia_flux']
		continuum_flux = out_feature_index['continuum_flux']
		econtinuum_flux = out_feature_index['econtinuum_flux']
		# index parameters
		feature_wl = out_feature_index['ammonia_wl']
		feature_window = out_feature_index['ammonia_window']
		continuum_wl = out_feature_index['continuum_wl']
		continuum_window = out_feature_index['continuum_window']
	else: # general names for user-defined index
		# index
		feature_index = out_feature_index['feature_index']
		efeature_index = out_feature_index['efeature_index']
		feature_flux = out_feature_index['feature_flux']
		efeature_flux = out_feature_index['efeature_flux']
		continuum_flux = out_feature_index['continuum_flux']
		econtinuum_flux = out_feature_index['econtinuum_flux']
		# index parameters
		feature_wl = out_feature_index['feature_wl']
		feature_window = out_feature_index['feature_window']
		continuum_wl = out_feature_index['continuum_wl']
		continuum_window = out_feature_index['continuum_window']

	#++++++++++++++++++++++++++++++
	# plot index measurement
	fig, ax = plt.subplots()

	if plot_xrange is None: xmin, xmax = wl.min(), wl.max()
	else: xmin, xmax = plot_xrange
	if plot_yrange is None:
		mask = (wl>=xmin) & (wl<=xmax)
		if flux[mask].min() >= 0: # when the minimum flux is positive
			ymin, ymax = 0.9*flux[mask].min(), 1.1*flux[mask].max()
		else: # when the minimum flux is negative
			ymin, ymax = 1.1*flux[mask].min(), 1.1*flux[mask].max()
	else: ymin, ymax = plot_yrange
	
	# indicate the regions to measure the index
	# absorption region
	ax.fill([feature_wl-feature_window/2, feature_wl-feature_window/2, 
	        feature_wl+feature_window/2, feature_wl+feature_window/2], 
	        [ymin, ymax, ymax, ymin], facecolor='silver', linewidth=1, zorder=2)
	if index_name=='Water': # plot second part of the feature
		ax.fill([feature_wl2-feature_window2/2, feature_wl2-feature_window2/2, 
		        feature_wl2+feature_window2/2, feature_wl2+feature_window2/2], 
		        [ymin, ymax, ymax, ymin], facecolor='silver', linewidth=1, zorder=2)
	# continuum region
	ax.fill([continuum_wl-continuum_window/2, continuum_wl-continuum_window/2, 
	        continuum_wl+continuum_window/2, continuum_wl+continuum_window/2], 
	        [ymin, ymax, ymax, ymin], facecolor='gainsboro', linewidth=1, zorder=2)

	# plot flux uncertainty region
	default_blue = plt.rcParams['axes.prop_cycle'].by_key()['color'][0] # default blue color
	wl_region = np.append(wl, np.flip(wl))
	flux_region = np.append(flux-eflux, np.flip(flux+eflux))
	ax.fill(wl_region, flux_region, facecolor=default_blue, edgecolor=default_blue, linewidth=0, alpha=0.30, zorder=3)
	# plot spectrum
	mask = wl>0
	plt.plot(wl[mask], flux[mask])

	# fluxes in the continuum region
	mask_continuum_wl = (wl>=(continuum_wl-continuum_window/2)) & (wl<=(continuum_wl+continuum_window/2))
	ax.errorbar(wl[mask_continuum_wl], flux[mask_continuum_wl], yerr=eflux[mask_continuum_wl], 
	            fmt='o', c='gray', markersize=3.0, linewidth=1.0, capsize=3, capthick=1)
	# fluxes in the absorption region
	mask_feature_wl = (wl>=(feature_wl-feature_window/2)) & (wl<=(feature_wl+feature_window/2))
	ax.errorbar(wl[mask_feature_wl], flux[mask_feature_wl], yerr=eflux[mask_feature_wl],  
	            fmt='o', c='gray', markersize=3.0, linewidth=1.0, capsize=3, capthick=1)
	if index_name=='Water': # for the second part of the feature
		mask_feature_wl2 = (wl>=(feature_wl2-feature_window2/2)) & (wl<=(feature_wl2+feature_window2/2))
		ax.errorbar(wl[mask_feature_wl2], flux[mask_feature_wl2], yerr=eflux[mask_feature_wl2],  
		            fmt='o', c='gray', markersize=3.0, linewidth=1.0, capsize=3, capthick=1)

	# mean flux out of the absorption
	ax.errorbar(continuum_wl, continuum_flux, yerr=econtinuum_flux, fmt='o', c='black', 
	            markersize=3.0, linewidth=1.0, capsize=3, capthick=1, zorder=3)
	# mean flux in the absorption
	ax.errorbar(feature_wl, feature_flux, yerr=efeature_flux, fmt='o', c='red', 
	            markersize=3.0, linewidth=1.0, capsize=3, capthick=1, zorder=3)
	if index_name=='Water': # for the second part of the feature
		ax.errorbar(feature_wl2, feature_flux2, yerr=efeature_flux2, fmt='o', c='red', 
		            markersize=3.0, linewidth=1.0, capsize=3, capthick=1, zorder=3)

	# write the index value in the plot
	feature_index_fmt = '{:.2f}'.format(round(feature_index,2))
	efeature_index_fmt = '{:.2f}'.format(round(efeature_index,2))
	label = f'{index_name} index='+feature_index_fmt+'$\pm$'+efeature_index_fmt
	plt.plot(0, 0, label=label)
	plt.legend(handlelength=0, handletextpad=0)#, frameon=False
	
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	ax.grid(True, which='both', color='gainsboro', alpha=0.5)
	
	plt.xlabel(r'$\lambda\ (\mu$m)', size=12)
	plt.ylabel(r'$F_\nu$ (Jy)', size=12)
	if plot_title is not None: plt.title(plot_title, size=12)
	else: plt.title(f'{index_name} Index Measurement', size=12)
	
	if plot_save: 
		if plot_name: plt.savefig(plot_name, bbox_inches='tight')
		else: plt.savefig(f'{index_name}_index_measurement.pdf', bbox_inches='tight')
	plt.show()
	plt.close()

	return

##########################
# plot the spectral index when continuum is measured from the fit to two continuum wavelength regions
def plot_spectral_index_two_continuum_regions(out_feature_index, index_name=None, plot_xrange=None, plot_yrange=None, plot_title=None, plot_save=True, plot_name=False):

	# read parameters of interest
	# spectrum
	wl = out_feature_index['wl']
	flux = out_feature_index['flux']
	eflux = out_feature_index['eflux']
	if index_name=='Silicate': # specific variables names for silicate index
		# index
		feature_index = out_feature_index['silicate_index']
		efeature_index = out_feature_index['esilicate_index']
		feature_flux = out_feature_index['silicate_flux']
		efeature_flux = out_feature_index['esilicate_flux']
		continuum_flux = out_feature_index['continuum_flux']
		econtinuum_flux = out_feature_index['econtinuum_flux']
		# index parameters
		feature_wl = out_feature_index['silicate_wl']
		feature_window = out_feature_index['silicate_window']
		continuum_wl1 = out_feature_index['continuum_wl1']
		continuum_window1 = out_feature_index['continuum_window1']
		continuum_wl2 = out_feature_index['continuum_wl2']
		continuum_window2 = out_feature_index['continuum_window2']
	else: # general names for user-defined index
		# index
		feature_index = out_feature_index['feature_index']
		efeature_index = out_feature_index['efeature_index']
		feature_flux = out_feature_index['feature_flux']
		efeature_flux = out_feature_index['efeature_flux']
		continuum_flux = out_feature_index['continuum_flux']
		econtinuum_flux = out_feature_index['econtinuum_flux']
		# index parameters
		feature_wl = out_feature_index['feature_wl']
		feature_window = out_feature_index['feature_window']
		continuum_wl1 = out_feature_index['continuum_wl1']
		continuum_window1 = out_feature_index['continuum_window1']
		continuum_wl2 = out_feature_index['continuum_wl2']
		continuum_window2 = out_feature_index['continuum_window2']
	# continuum fit
	slope = out_feature_index['slope']
	eslope = out_feature_index['eslope']
	constant = out_feature_index['constant']
	econstant = out_feature_index['econstant']
	continuum_fit = out_feature_index['continuum_fit']

	#++++++++++++++++++++++++++++++
	# plot silicate index measurement
	fig, ax = plt.subplots()
	
	#xmin, xmax = wl[wl>0].min(), wl[wl>0].max()
	#ymin, ymax = np.percentile(flux[flux!=0], scale_percent), np.nanpercentile(flux[flux!=0], 100-scale_percent)
	#xmin, xmax = 0.90*continuum_wl1-continuum_window1/2., 1.05*silicate_con2+continuum_window2/2.
	if plot_xrange is None: xmin, xmax = wl.min(), wl.max()
	else: xmin, xmax = plot_xrange
	if plot_yrange is None:
		#scale_percent = 0.1
		mask = (wl>=xmin) & (wl<=xmax)
		if flux[mask].min() >= 0: # when the minimum flux is positive
			ymin, ymax = 0.9*flux[mask].min(), 1.1*flux[mask].max()
		else: # when the minimum flux is negative
			ymin, ymax = 1.1*flux[mask].min(), 1.1*flux[mask].max()
	else: 
		ymin, ymax = plot_yrange
	
	# indicate the regions to measure the silicate index
	# silicate absorption center
	ax.fill([feature_wl-feature_window/2, feature_wl-feature_window/2, 
	        feature_wl+feature_window/2, feature_wl+feature_window/2], 
	        [ymin, ymax, ymax, ymin], facecolor='silver', linewidth=1, zorder=2)
	# short-wavelength silicate absorption continuum
	ax.fill([continuum_wl1-continuum_window1/2, continuum_wl1-continuum_window1/2, 
	        continuum_wl1+continuum_window1/2, continuum_wl1+continuum_window1/2], 
	        [ymin, ymax, ymax, ymin], facecolor='gainsboro', linewidth=1, zorder=2)
	# long-wavelength silicate absorption continuum
	ax.fill([continuum_wl2-continuum_window2/2, continuum_wl2-continuum_window2/2, 
	        continuum_wl2+continuum_window2/2, continuum_wl2+continuum_window2/2], 
	        [ymin, ymax, ymax, ymin], facecolor='gainsboro', linewidth=1, zorder=2)
	
	# plot flux uncertainty region
	default_blue = plt.rcParams['axes.prop_cycle'].by_key()['color'][0] # default blue color
	wl_region = np.append(wl, np.flip(wl))
	flux_region = np.append(flux-eflux, np.flip(flux+eflux))
	ax.fill(wl_region, flux_region, facecolor=default_blue, edgecolor=default_blue, linewidth=0, alpha=0.30, zorder=3)
	# plot spectrum
	plt.plot(wl, flux, color=default_blue , zorder=3)
	
	# fit to the continuum regions
	if continuum_fit=='exponential':
		xc_feature = np.linspace(continuum_wl1-continuum_window1/2., continuum_wl2+continuum_window2/2., 100)
		yc_feature = 10**(slope*np.log10(xc_feature) + constant)
		#ycu_feature = 10**((slope+eslope)*np.log10(xc_feature) + constant)
		#ycd_feature = 10**((slope-eslope)*np.log10(xc_feature) + constant)
		ycu_feature = yc_feature + econtinuum_flux
		ycd_feature = yc_feature - econtinuum_flux
	if continuum_fit=='line':
		xc_feature = np.linspace(continuum_wl1-continuum_window1/2., continuum_wl2+continuum_window2/2., 100)
		yc_feature = slope*xc_feature + constant
		#ycu_feature = (slope+eslope)*xc_feature + constant
		#ycd_feature = (slope-eslope)*xc_feature + constant
		ycu_feature = yc_feature + econtinuum_flux
		ycd_feature = yc_feature - econtinuum_flux
	ax.fill(np.append(xc_feature, xc_feature[::-1]), np.append(ycd_feature, ycu_feature[::-1]), 
	        facecolor='gray', edgecolor='gray', linewidth=0, alpha=0.20)
	ax.plot(xc_feature, yc_feature, '--', color='gray', linewidth=0.5)
	
	# fluxes in the continuum regions
	mask_continuum_wl1 = (wl>=(continuum_wl1-continuum_window1/2)) & (wl<=(continuum_wl1+continuum_window1/2))
	mask_continuum_wl2 = (wl>=(continuum_wl2-continuum_window2/2)) & (wl<=(continuum_wl2+continuum_window2/2))
	ax.errorbar(wl[mask_continuum_wl1], flux[mask_continuum_wl1], yerr=eflux[mask_continuum_wl1], 
	            fmt='o', c='gray', markersize=3.0, linewidth=1.0, capsize=3, capthick=1)
	ax.errorbar(wl[mask_continuum_wl2], flux[mask_continuum_wl2], yerr=eflux[mask_continuum_wl2], 
	            fmt='o', c='gray', markersize=3.0, linewidth=1.0, capsize=3, capthick=1)
	# interpolated continuum at the absorption
	ax.errorbar(feature_wl, continuum_flux, yerr=econtinuum_flux, fmt='o', c='black', 
	            markersize=3.0, linewidth=1.0, capsize=3, capthick=1)
	
	# fluxes in the absorption
	mask_feature_wl = (wl>=(feature_wl-feature_window/2)) & (wl<=(feature_wl+feature_window/2))
	ax.errorbar(wl[mask_feature_wl], flux[mask_feature_wl], yerr=eflux[mask_feature_wl],  
	            fmt='o', c='gray', markersize=3.0, linewidth=1.0, capsize=3, capthick=1)
	# mean flux of the absorption 
	ax.errorbar(feature_wl, feature_flux, yerr=efeature_flux, 
	            fmt='o', c='red', markersize=3.0, linewidth=1.0, capsize=3, capthick=1, zorder=3)
	
	# write the index value in the plot
	#label = f'{index_name} index='+str(round(feature_index,2))+'$\pm$'+str(round(efeature_index,2))
	feature_index_fmt = '{:.2f}'.format(round(feature_index,2))
	efeature_index_fmt = '{:.2f}'.format(round(efeature_index,2))
	label = f'{index_name} index='+feature_index_fmt+'$\pm$'+efeature_index_fmt
	plt.plot(0, 0, label=label)
	plt.legend(handlelength=0, handletextpad=0)#, frameon=False
	
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	ax.grid(True, which='both', color='gainsboro', alpha=0.5)
	
	plt.xlabel(r'$\lambda\ (\mu$m)', size=12)
	plt.ylabel(r'$F_\nu$ (Jy)', size=12)
	if plot_title is not None: plt.title(plot_title, size=12)
	else: plt.title(f'{index_name} Index Measurement', size=12)
	
	if plot_save: 
		if plot_name: plt.savefig(plot_name, bbox_inches='tight')
		else: plt.savefig(f'{index_name}_index_measurement.pdf', bbox_inches='tight')
	plt.show()
	plt.close()

	return

##########################
# manipulate input spectrum before measuring spectral indices
def handle_input_spectrum(wl, flux, eflux):

	# convert input spectrum into numpy arrays, if needed
	wl = astropy_to_numpy(wl)
	flux = astropy_to_numpy(flux)
	eflux = astropy_to_numpy(eflux)

	# avoid nan values and zeros in wavelength
	mask_nan = (~np.isnan(flux)) & (~np.isnan(eflux)) & (wl>0)
	wl = wl[mask_nan]
	flux = flux[mask_nan]
	eflux = eflux[mask_nan]

	return wl, flux, eflux

##########################
# fit curve to data in two wavelength regions
def fit_continuum(wl, flux, eflux, wl_con1, wl_con2, window_con1, window_con2, continuum_fit, continuum_error, ref_min):

	# allowed options
	continuum_fit_valid = ['line', 'exponential']
	if continuum_fit not in continuum_fit_valid:
		raise Exception(f'"{continuum_fit}" is not allowed. Valid options are {continuum_fit_valid}.')
	continuum_error_valid = ['fit', 'empirical']
	if continuum_error not in continuum_error_valid:
		raise Exception(f'"{continuum_error}" is not allowed. Valid options are {continuum_error_valid}.')

	# fit a curve to the continuum points
	#--------------------------------------
	# fit a line to the continuum points in linear space
	mask_wl_con1 = (wl>=(wl_con1-window_con1/2)) & (wl<=(wl_con1+window_con1/2)) # mask for continuum window 1
	mask_wl_con2 = (wl>=(wl_con2-window_con2/2)) & (wl<=(wl_con2+window_con2/2)) # mask for continuum window 2
	mask_wl_con1_con2 = ((wl>=(wl_con1-window_con1/2)) & (wl<=(wl_con1+window_con1/2))) | \
	                    ((wl>=(wl_con2-window_con2/2)) & (wl<=(wl_con2+window_con2/2))) # data points in both continuum regions
	if continuum_fit=='line':
		linear_wl = wl[mask_wl_con1_con2]
		linear_flux = flux[mask_wl_con1_con2]
		linear_eflux = eflux[mask_wl_con1_con2]
		fit, cov_fit = np.polyfit(linear_wl, linear_flux, 1, w=1./linear_eflux, cov=True) # weigh by the inverse of the error
		slope = fit[0] # slope of the histogram without any correction
		eslope = np.sqrt(cov_fit[0,0]) # error of the slope
		constant = fit[1]
		econstant = np.sqrt(cov_fit[1,1]) 
		# continuum at the point in of the absorption
		continuum_flux = slope*ref_min + constant
		if continuum_error=='fit':
			econtinuum_flux = (((slope+eslope)*ref_min + constant) - ((slope-eslope)*ref_min + constant)) / 2 # considering only the slope uncertainty
		                                                                                                      # (used in Suarez & Metchev 2022,2023 for silicate index)
		if continuum_error=='empirical':
			econtinuum_flux_1 = np.mean([np.std(flux[mask_wl_con1]), np.std(flux[mask_wl_con2])]) / np.sqrt(linear_flux.size) # due to the scatter of the data in both continuum windows
			econtinuum_flux_2 = continuum_flux * np.median(linear_eflux/linear_flux) # due to flux uncertainties
			econtinuum_flux = np.sqrt(econtinuum_flux_1**2 + econtinuum_flux_2**2) # addition in quadrature

	# fit an exponential to the points in linear space, which is equivalent to:
	# fit a line to the continuum points in the log(flux)-log(lambda) space
	if continuum_fit=='exponential':
		if (flux[mask_wl_con1_con2].min()<=0): 
			print('Warning: Logarithm of negative and/or zero fluxes in the continuum windows')
			print('   Only positive fluxes will be used')
			mask_wl_con1 = (wl>=(wl_con1-window_con1/2)) & (wl<=(wl_con1+window_con1/2)) & (flux>0) # positive fluxes in the continuum window 1
			mask_wl_con2 = (wl>=(wl_con2-window_con2/2)) & (wl<=(wl_con2+window_con2/2)) & (flux>0) # positive fluxes in the continuum window 2
			mask_wl_con1_con2 = (((wl>=(wl_con1-window_con1/2)) & (wl<=(wl_con1+window_con1/2))) | \
			                          ((wl>=(wl_con2-window_con2/2)) & (wl<=(wl_con2+window_con2/2)))) & \
			                          (flux>0) # positive fluxes in both continuum regions
		logwl = np.log10(wl[mask_wl_con1_con2])
		logflux = np.log10(flux[mask_wl_con1_con2])
		#elogflux = (1/np.log(10)) * eflux[mask_wl_con1_con2]/flux[mask_wl_con1_con2]
		elogflux = eflux[mask_wl_con1_con2] # these errors are used only for weighing the fit. 
		                                    # thus, assigning the log errors equal to the linear error will weight by the error bars directly plotted in the flux vs wavelength plot. 
		                                    # otherwise, when properly propagating the linear errors to log errors will give more weight (lower errors) to higher fluxes.

		fit, cov_fit = np.polyfit(logwl, logflux, 1, w=1./elogflux, cov=True) # weigh by the inverse of the error
		slope = fit[0] # slope 
		eslope = np.sqrt(cov_fit[0,0]) # slope error 
		constant = fit[1] # constant
		econstant = np.sqrt(cov_fit[1,1]) # constant error
		# the slope error has to be scaled somehow to be used in linear plots
		fit_linear, cov_fit_linear = np.polyfit(wl[mask_wl_con1_con2], flux[mask_wl_con1_con2], 1, w=1./eflux[mask_wl_con1_con2], cov=True) # weigh by the inverse of the error
		slope_linear = fit_linear[0] # slope 
		eslope_linear = np.sqrt(cov_fit_linear[0,0])
		# slope error normalization
		eslope = eslope / np.log10(eslope/eslope_linear) # this is an empirical correction I deduced
	
		# continuum at the absorption peak
		continuum_flux = 10**(slope*np.log10(ref_min) + constant)
		if continuum_error=='fit':
			econtinuum_flux = (10**(((slope+eslope)*np.log10(ref_min)+constant)) - 10**((slope-eslope)*np.log10(ref_min)+constant)) / 2 # considering only the slope uncertainty 
		                                                                                                                                # (used in Suarez & Metchev 2022,2023 for silicate index)
		if continuum_error=='empirical':
			econtinuum_flux_1 = np.mean([np.std(flux[mask_wl_con1]), np.std(flux[mask_wl_con2])]) / np.sqrt(logflux.size) # due to the scatter of the data in both continuum windows
			#econtinuum_flux_2 = continuum_flux * np.median(elogflux/logflux) # due to flux uncertainties (used in Suarez & Metchev 2022,2023 for silicate index)
			econtinuum_flux_2 = continuum_flux * np.median(eflux[mask_wl_con1_con2]/flux[mask_wl_con1_con2]) # due to flux uncertainties
			econtinuum_flux = np.sqrt(econtinuum_flux_1**2 + econtinuum_flux_2**2) # addition in quadrature
	
	## fit an exponential curve to the continuum points in the flux vs. lambda plot (this should be equivalent to a linear fit in the log(flux)-log(lambda) space)
	# ATTEMPT
	#if not log_log_space:
	#	#def exponential(x, a, b, c):
	# 	#   return a * np.exp(b * x) + c
	#	def exponential(x, a, b):
	# 	   return a*np.exp(b * x)
	#	mask_wl_con1_con2 = np.concatenate((mask_wl_con1, mask_wl_con2)) # indices of the continuum data points on both regions
	#	fit_exp, cov_fit_exp = curve_fit(exponential, wl[mask_wl_con1_con2,i], flux[mask_wl_con1_con2,i])#, sigma=eflux[mask_wl_con1_con2,i])

	out = {'slope': slope, 'eslope': eslope, 'constant': constant, 'econstant': econstant, 
	       'continuum_flux': continuum_flux, 'econtinuum_flux': econtinuum_flux}

	return out
