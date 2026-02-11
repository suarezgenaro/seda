import numpy as np
import os
from astropy.io import ascii
from astropy.table import Column, MaskedColumn, Table
from astropy import units as u
from .. import utils
from sys import exit

def synthetic_photometry(wl, flux, filters, flux_unit, eflux=None, out_file=None): 
	'''
	Description:
	------------
		Compute synthetic photometry for any SVO filters from an input spectrum.

	Parameters:
	-----------
	- wl : array
		Wavelength in um.
	- flux : array
		Fluxes in units specified by ``flux_unit``.
	- filters : list, array, or str
		Filters (following SVO filter IDs) to derive synthetic photometry.
	- flux_unit : str
		Flux and flux error units: ``'erg/s/cm2/A'`` or ``'Jy'``.
	- eflux : array (optional) 
		Flux uncertainties in units specified by ``flux_unit``.
	- out_file : str, optional
		File name to save the synthetic photometry (in erg/s/cm2/A) as prettytable.
		The file name can include a path e.g. my_path/syn_phot.dat
		If not provided, the synthetic photometry will not be saved.

	Returns:
	--------
	Python dictionary with the following parameters for each filter:
		- ``'syn_flux(erg/s/cm2/A)'`` : synthetic fluxes in erg/s/cm2/A.
		- ``'esyn_flux(erg/s/cm2/A)'`` : synthetic flux errors in erg/s/cm2/A (if input ``eflux`` is provided).
		- ``'syn_flux(Jy)'`` : synthetic fluxes in Jy.
		- ``'esyn_flux(Jy)'`` : synthetic flux errors in Jy (if input ``eflux`` is provided).
		- ``'syn_mag'`` : synthetic magnitudes.
		- ``'esyn_mag'`` : synthetic magnitude errors (if input ``eflux`` is provided).
		- ``'filters'`` : filters used to derive synthetic photometry.
		- ``'lambda_eff(um)'`` : filters' effective wavelengths in microns computed using the input spectrum.
		- ``'width_eff(um)'`` : filters' effective width in micron computed using the input spectrum.
		- ``'lambda_eff_SVO(um)'`` : filters' effective wavelengths in microns from SVO.
		- ``'width_eff(um)'`` : filters' effective width in micron from SVO.
		- ``'zero_point(Jy)'`` : filters' zero points in Jy.
		- ``'label'`` : label indicating if the filters are fully ('complete'), partially ('incomplete'), or no ('none') covered by the input spectrum or no recognized by SVO ('unrecognizable').
		- ``'coverage_perc'`` : percentage of the filter transmission covered by the spectrum.
		- ``'transmission'`` : dictionary with 2D arrays for the filter transmissions, where the first first entry is wavelength in microns and the second one is the transmission.
		- ``'wl'`` : input spectrum wavelengths.
		- ``'flux'`` : input spectrum fluxes.
		- ``'eflux'`` : input spectrum flux uncertainties (if input ``eflux`` is provided)..
		- ``'flux_unit'`` : flux units of the input spectrum.

	Example:
	--------
	>>> # obtain synthetic photometry and plot the results
	>>> import seda
	>>> 
	>>> # assume we have read a wavelengths (wl in um), fluxes (in erg/s/cm2/A), 
	>>> # and flux errors (eflux) for an input spectrum,
	>>>	# and we would like synthetic photometry for several filters
	>>> filters = (['2MASS/2MASS.J', 'Spitzer/IRAC.I1', 'WISE/WISE.W1']) # filters of interest
	>>> 
	>>>	# run the code
	>>> out = seda.synthetic_photometry(wl=wl, flux=flux, eflux=eflux, 
	>>>                                 flux_unit='erg/s/cm2/A', filters=filters)
	>>> 
	>>> # visualize the derived synthetic fluxes
	>>> seda.plot_synthetic_photometry(out)

	Author: Genaro Suárez
	'''

	dir_sep = os.sep # directory separator for the current operating system
	path_synthetic_photometry = os.path.dirname(__file__)+dir_sep

	# input spectrum to numpy
	wl = utils.astropy_to_numpy(wl)
	flux = utils.astropy_to_numpy(flux)
	if eflux is not None: eflux = utils.astropy_to_numpy(eflux)

	# remove nan and null flux values
	valid = np.isfinite(flux) & (flux!=0.0)
	wl = wl[valid]
	flux = flux[valid]
	if eflux is not None: eflux = eflux[valid]

	# ensure filters is a list
	if isinstance(filters, str): filters = [filters]

	# read filters' transmission curves and zero points
	svo_data = utils.read_SVO_table()
	filterID = svo_data['filterID'] # SVO ID
	ZeroPoint = svo_data['ZeroPoint'] # in Jy
	# fill masked values if needed
	filterID = maskedcolumn_to_numpy(filterID)
	ZeroPoint = maskedcolumn_to_numpy(ZeroPoint)

	# initialize arrays to store relevant information
	n_filters = len(filters)
	# assign NaN values to be the output for unrecognized filters by SVO
	syn_flux_Jy = np.full(n_filters, np.nan)
	esyn_flux_Jy = np.full(n_filters, np.nan)
	syn_flux_erg = np.full(n_filters, np.nan)
	esyn_flux_erg = np.full(n_filters, np.nan)
	syn_mag = np.full(n_filters, np.nan)
	esyn_mag = np.full(n_filters, np.nan)
	lambda_eff = np.full(n_filters, np.nan)
	width_eff = np.full(n_filters, np.nan)
	lambda_eff_SVO = np.full(n_filters, np.nan)
	width_eff_SVO = np.full(n_filters, np.nan)
	zero_point = np.full(n_filters, np.nan)
	coverage_perc = np.full(n_filters, np.nan)
	# initialize other variables
	label = np.full(n_filters, 'complete', dtype=object) # to assign a label to each filter based on its spectral coverage
	label[:] = 'complete'
	transmission = {} # dictionary to save the filter transmissions

	# main loop over filters
	for k, filt in enumerate(filters): # iterate on each filter
		# check first if the filter name is on SVO
		if filt not in filterID: # if filter ID is not recognized
			label[k] = 'unrecognizable'
			print(f'   Caveat: "{filt}" filter is not recognized by SVO, so will be ignored.')
			continue # to jump to the next iteration
	
		# load filter transmission
		filter_wl, filter_flux = load_filter_transmission(filt)
		transmission[filt] = np.array([filter_wl, filter_flux])

		# check coverage
		label[k], fraction_cov = filter_coverage_fraction(wl, filter_wl, filter_flux)
		coverage_perc[k] = 100.*fraction_cov
		if label[k] == 'none': # no spectral coverage of current filter
			print(f'   Caveat: No spectral coverage for {filt}, so will be ignored.')
			continue # jump to next filter

		if label[k] == 'incomplete': # filter partially covered
			print(f'   Caveat: No full spectral coverage for {filt}, so the synthetic photometry is a lower limit')
			print(f'      approx. {round(coverage_perc[k],2)}% of the filter transmission is covered by the data')
			label[k] = 'incomplete'

		# compute synthetic flux
		out_synt_flux = compute_synthetic_flux(wl, flux, filter_wl, filter_flux, eflux)
		syn_flux = out_synt_flux['syn_flux']
		esyn_flux = out_synt_flux['esyn_flux']
		lambda_eff[k] = out_synt_flux['lambda_eff']
		width_eff[k] = out_synt_flux['width_eff']
		
		# convert flux into magnitudes
		# first from erg/s/cm2/A to Jy (if needed) and then from Jy to mag
		if flux_unit == 'erg/s/cm2/A':
			syn_flux_Jy[k] = convert_flux(syn_flux, lambda_eff[k], 'erg/s/cm2/A', 'Jy')['flux_out'] # in Jy
			if eflux is not None: esyn_flux_Jy[k] = esyn_flux / syn_flux * syn_flux_Jy[k] # in Jy
			syn_flux_erg[k] = syn_flux # in erg/s/cm2/A
			if eflux is not None: esyn_flux_erg[k] = esyn_flux # in erg/s/cm2/A

		else:  # (flux_unit == 'Jy') convert Jy to erg/s/cm2/A to be an output
			syn_flux_erg[k] = convert_flux(syn_flux, lambda_eff[k], 'Jy', 'erg/s/cm2/A')['flux_out'] # in erg/s/cm2/A
			if eflux is not None: esyn_flux_erg[k] = esyn_flux / syn_flux * syn_flux_erg[k] # in erg/s/cm2/A
			syn_flux_Jy[k] = syn_flux # in Jy
			if eflux is not None: esyn_flux_Jy[k] = esyn_flux # in Jy

		# from Jy to mag
		mask_filt = filterID == filt
		if any(mask_filt) is False: raise Exception(f'   \nERROR: No zero point for filter {filt}')

		out_mag = flux_to_mag(flux=syn_flux_Jy[k], eflux=esyn_flux_Jy[k], filters=filt, flux_unit='Jy')
		syn_mag[k] = out_mag['mag'][0] # in mag
		if eflux is not None: esyn_mag[k] = out_mag['emag'][0] # in mag
		lambda_eff_SVO[k] = out_mag['lambda_eff_SVO(um)'][0] # um
		width_eff_SVO[k] = out_mag['width_eff_SVO(um)'][0] # um
		zero_point[k] = ZeroPoint[mask_filt][0] # in Jy

	# output dictionary
	out = {'syn_flux(Jy)': syn_flux_Jy, 'syn_flux(erg/s/cm2/A)': syn_flux_erg, 'syn_mag': syn_mag, 'lambda_eff(um)': lambda_eff, 
	       'width_eff(um)': width_eff, 'lambda_eff_SVO(um)': lambda_eff_SVO, 'width_eff_SVO(um)': width_eff_SVO, 
	       'zero_point(Jy)': zero_point, 'label': label, 'coverage_perc': coverage_perc, 'transmission': transmission, 
	       'wl': wl, 'flux': flux, 'flux_unit': flux_unit, 'filters': filters}
	if eflux is not None: 
		out['esyn_flux(Jy)'] = esyn_flux_Jy
		out['esyn_flux(erg/s/cm2/A)'] = esyn_flux_erg
		out['esyn_mag'] = esyn_mag
		out['eflux'] = eflux

	# save synthetic photometry
	if out_file is not None:
		# save file
		# select parameters of interest to be saved
		out_sel = {}
		out_sel['filters'] = filters
		out_sel['syn_mag'] = format_number(syn_mag)
		if eflux is not None: out_sel['esyn_mag'] = format_number(esyn_mag)
		out_sel['syn_flux(Jy)'] = format_number(syn_flux_Jy)
		if eflux is not None: out_sel['esyn_flux(Jy)'] = format_number(esyn_flux_Jy)
		out_sel['syn_flux(erg/s/cm2/A)'] = format_number(syn_flux_erg)
		if eflux is not None: out_sel['esyn_flux(erg/s/cm2/A)'] = format_number(esyn_flux_erg)
		out_sel['lambda_eff(um)'] = format_number(lambda_eff)
		out_sel['width_eff(um)'] = format_number(width_eff)
		out_sel['coverage_perc'] = format_number(coverage_perc)
		
		# save the photometry as prettytable table
		utils.save_prettytable(my_dict=out_sel, table_name=out_file)

	return out 

#+++++++++++++++++
def convert_flux(flux, wl, unit_in, unit_out, eflux=None):
	'''
	Description:
	------------
		Convert fluxes from Jy to erg/s/cm2/A or vice versa.

	Parameters:
	-----------
	- flux : array, list, or float
 		Input fluxes in units specified by ``unit_in``.
	- wl : array, float
		Wavelengths (in microns) associated to ``flux``.
	- unit_in : str
		Units of ``flux``: ``'Jy'`` or ``'erg/s/cm2/A'``.
	- unit_out : str
		Units of output fluxes: ``'Jy'`` or ``'erg/s/cm2/A'``.
	- eflux : array, float, optional
 		Flux uncertainties for ``flux`` in ``unit_in``.

	Returns:
	--------
	Dictionary with converted fluxes:
		- ``'flux_in'`` : input fluxes
		- ``'flux_out'`` : output fluxes
		- ``'wl_in'`` : input wavelengths
		- ``'eflux_in'`` : (if ``eflux``) input flux uncertainties
		- ``'eflux_out'`` : (if ``eflux``) output flux uncertainties
		- ``'unit_in'`` : ``unit_in``
		- ``'unit_out'`` : ``unit_out``

	Example:
	--------
	>>> # convert fluxes from Jy to erg/s/cm2/A
	>>> import seda
	>>> import numpy as np
	>>>
	>>> flux = np.array([0.005, 0.006]) # test fluxes in Jy
	>>> wl = np.array([1.23, 2.16]) # test wavelengths in microns
	>>> eflux = flux/10. # test flux errors in Jy
	>>> 
	>>> seda.convert_flux(flux=flux, wl=wl, eflux=eflux, unit_in='Jy', unit_out='erg/s/cm2/A')
	    {'flux_in': array([0.005, 0.006]),
	     'flux_out': array([9.90787422e-16, 3.85535568e-16]),
	     'wl_in': array([1.23, 2.16]),
	     'unit_in': 'Jy',
	     'unit_out': 'erg/s/cm2/A',
	     'eflux_in': array([0.0005, 0.0006]),
	     'eflux_out': array([9.90787422e-17, 3.85535568e-17])}

	Author: Genaro Suárez
	'''

	# convert input variables into numpy arrays if astropy
	wl = utils.astropy_to_numpy(wl)
	flux = utils.astropy_to_numpy(flux)
	if eflux is not None: eflux = utils.astropy_to_numpy(eflux)

	if unit_in=='erg/s/cm2/A':
		flux_out = (flux*u.erg/u.s/u.cm**2/(u.nm*0.1)).to(u.Jy, equivalencies=u.spectral_density(wl*u.micron)).value
		if eflux is not None:
			eflux_out = (eflux*u.erg/u.s/u.cm**2/(u.nm*0.1)).to(u.Jy, equivalencies=u.spectral_density(wl*u.micron)).value
	elif unit_in=='Jy':
		flux_out = (flux*u.Jy).to(u.erg/u.s/u.cm**2/(u.nm*0.1), equivalencies=u.spectral_density(wl*u.micron)).value
		if eflux is not None:
			eflux_out = (eflux*u.Jy).to(u.erg/u.s/u.cm**2/(u.nm*0.1), equivalencies=u.spectral_density(wl*u.micron)).value

	out = {'flux_in': flux, 'flux_out': flux_out, 'wl_in': wl, 'unit_in': unit_in, 'unit_out': unit_out}
	if eflux is not None: 
		out['eflux_in'] = eflux
		out['eflux_out'] = eflux_out

	return out

#+++++++++++++++++
def flux_to_mag(flux, filters, flux_unit='Jy', eflux=None, svo_data=None):
	'''
	Description:
	------------
		Convert fluxes into magnitudes.

	Parameters:
	-----------
	- flux : array, list, or float
 		Input fluxes in units specified by ``flux_unit``.
	- filters : array, list, or str
		Filters (following SVO filter IDs) used to obtain ``flux``.
		It must have the same size as ``flux``.
	- flux_unit : str, optional (default ``'Jy'``)
		Units of ``flux``: ``'Jy'`` or ``'erg/s/cm2/A'``.
	- eflux : array, float, optional
 		Flux uncertainties for ``flux`` in ``unit_in``.
	- svo_data : SVO table, optional
		Astropoy table from SVO.

	Returns:
	--------
	Dictionary with converted fluxes:
		- ``'mag'`` : resulting magnitudes
		- ``'emag'`` : (if ``eflux``) uncertainty of resulting magnitudes
		- ``'flux'`` : input flux
		- ``'eflux'`` : (if ``eflux``) input flux uncertainties
		- ``'filters'`` : input filter IDs
		- ``'zero_point(Jy)'`` : filters' zero points in Jy 
		- ``'lambda_eff_SVO(um)'`` : effective wavelengths in microns from SVO
		- ``'width_eff_SVO(um)'`` : effective width in microns from SVO

	Example:
	--------
	>>> # convert fluxes in Jy into magnitudes
	>>> import seda
	>>> import numpy as np
	>>> 
	>>> flux = np.array([0.005, 0.006]) # test fluxes in Jy
	>>> eflux = flux/10. # test flux errors in Jy
	>>> filters=['2MASS/2MASS.J', '2MASS/2MASS.Ks'] # test filter IDs
	>>> 
	>>> seda.synthetic_phtometry.flux_to_mag(flux=flux, eflux=eflux, filters=filters)
		{'flux': array([0.005, 0.006]),
		 'mag': array([13.75879578, 12.61461085]),
		 'filters': ['2MASS/2MASS.J', '2MASS/2MASS.Ks'],
		 'zero_point(Jy)': array([1594. ,  666.8]),
		 'flux_unit': 'Jy',
		 'lambda_eff_SVO(um)': array([1.235, 2.159]),
		 'width_eff_SVO(um)': array([0.1624319 , 0.26188695]),
		 'eflux': array([0.0005, 0.0006]),
		 'emag': array([0.10857362, 0.10857362])}

	Author: Genaro Suárez
	'''

	# convert floats (if any) into arrays
	if isinstance(flux, float): flux = np.array([flux])
	if eflux is not None: 
		if isinstance(eflux, float): eflux = np.array([eflux])
	# convert str (if any) into list
	filters = utils.var_to_list(filters)
	# convert input variables into numpy arrays if astropy
	flux = utils.astropy_to_numpy(flux)
	if eflux is not None: eflux = utils.astropy_to_numpy(eflux)

	# verify that flux and filters variables have the same size
	if len(flux)!=len(filters): raise Exception('filters does not have the size as flux')

	# if the SVO is not provided
	if svo_data is None:
		svo_data = utils.read_SVO_table()

	filterID = svo_data['filterID'] # SVO ID
	ZeroPoint = svo_data['ZeroPoint'] # in Jy
	WavelengthEff = svo_data['WavelengthEff'] # effective wavelength in A
	WidthEff = svo_data['WidthEff'] # effective width in A

	# fill masked values if needed
	filterID = maskedcolumn_to_numpy(filterID)
	ZeroPoint = maskedcolumn_to_numpy(ZeroPoint)
	WavelengthEff = maskedcolumn_to_numpy(WavelengthEff)
	WidthEff = maskedcolumn_to_numpy(WidthEff)
	
	# save the input fluxes and errors because they are replaced by Jy units when given in erg/s/cm2/A
	flux_ori = flux.copy()
	if eflux is not None: eflux_ori = eflux.copy()

	# initialize arrays to store relevant information
	# assign NaN values to be the output for unrecognized filters by SVO
	mag = np.zeros(len(filters)) * np.nan
	if eflux is not None: emag = np.zeros(len(filters)) * np.nan
	zero_point = np.zeros(len(filters)) * np.nan
	wl_eff = np.zeros(len(filters)) * np.nan
	width_eff = np.zeros(len(filters)) * np.nan
	for k, filt in enumerate(filters): # iterate on each filter
		# check first if the filter name is on SVO
		if not filt in filterID: # if filter ID is not recognized
			print(f'   Caveat: {filt} ID not recognized by SVO, so will be ignored')
		else: # if filter ID is a valid one
			mask = filterID==filt
			zero_point[k] = ZeroPoint[mask][0] # in Jy
			wl_eff[k] = (WavelengthEff[mask][0]*(u.nm*0.1)).to(u.micron).value # in um
			width_eff[k] = (WidthEff[mask][0]*(u.nm*0.1)).to(u.micron).value # in um

			# convert erg/s/cm2/A to Jy if needed
			if (flux_unit=='erg/s/cm2/A'): 
				flux[k] = convert_flux(flux=flux[k], wl=wl_eff[k], unit_in='erg/s/cm2/A', unit_out='Jy')['flux_out'] # in Jy
				if eflux is not None: eflux[k] = convert_flux(flux=flux[k], eflux=eflux[k], wl=wl_eff[k], unit_in='erg/s/cm2/A', unit_out='Jy')['eflux_out'] # in Jy

			# Jy to mag
			mag[k] = -2.5*np.log10(flux[k]/zero_point[k]) # in mag
			if eflux is not None: emag[k] = (2.5/np.log(10))*np.sqrt((eflux[k]/flux[k])**2)#+(ephot_F0/phot_F0)**2) # in mag

	# output dictionary
	out = {'flux': flux_ori, 'mag': mag, 'filters': filters, 'zero_point(Jy)': zero_point, 'flux_unit': flux_unit,
	       'lambda_eff_SVO(um)': wl_eff, 'width_eff_SVO(um)': width_eff}
	if eflux is not None: 
		out['eflux'] = eflux_ori
		out['emag'] = emag

	return out

#+++++++++++++++++
def mag_to_flux(mag, filters, flux_unit='Jy', emag=None, svo_data=None):
	'''
	Description:
	------------
		Convert magnitudes into fluxes.

	Parameters:
	-----------
	- mag : array, list, or float
 		Input magnitudes in mag.
	- filters : array, list, or str
		Filters (following SVO filter IDs) used to obtain ``mag``.
		It must have the same size as ``mag``.
	- flux_unit : str, optional (default ``'Jy'``)
		Units to return flux: ``'Jy'`` or ``'erg/s/cm2/A'``.
	- emag : array, float, optional
 		Magnitude uncertainties for ``mag`` in mag.
	- svo_data : SVO table, optional
		Astropoy table from SVO.

	Returns:
	--------
	Dictionary with converted fluxes:
		- ``'flux'`` : resulting fluxes
		- ``'eflux'`` : (if ``emag``) resulting flux uncertainties
		- ``'mag'`` : input magnitudes
		- ``'emag'`` : (if ``emag``) uncertainty of input magnitudes
		- ``'filters'`` : input filter IDs
		- ``'zero_point(Jy)'`` : filters' zero points in Jy 
		- ``'lambda_eff_SVO(um)'`` : effective wavelengths in microns from SVO
		- ``'width_eff_SVO(um)'`` : effective width in microns from SVO

	Example:
	--------
	>>> # convert magnitudes into fluxes in Jy
	>>> import seda
	>>> import numpy as np
	>>> 
	>>> mag = np.array([15.0, 15.5]) # test magnitudes in mag
	>>> emag = mag/10. # test magnitude errors in mag
	>>> filters=['2MASS/2MASS.J', '2MASS/2MASS.Ks'] # test filter IDs
	>>> 
	>>> seda.mag_to_flux(mag=mag, emag=emag, filters=filters)
	    {'mag': array([15. , 15.5]),
	     'flux': array([0.001594  , 0.00042072]),
	     'filters': ['2MASS/2MASS.J', '2MASS/2MASS.Ks'],
	     'zero_point(Jy)': array([1594. ,  666.8]),
	     'flux_unit': 'Jy',
	     'lambda_eff_SVO(um)': array([1.235, 2.159]),
	     'width_eff_SVO(um)': array([0.1624319 , 0.26188695]),
	     'emag': array([1.5 , 1.55]),
	     'eflux': array([0.00220219, 0.00060062])}

	Author: Genaro Suárez
	'''

	# convert floats (if any) into arrays
	if isinstance(mag, float): mag = np.array([mag])
	if emag is not None: 
		if isinstance(emag, float): emag = np.array([emag])
	# convert str (if any) into list
	filters = utils.var_to_list(filters)
	# convert input variables into numpy arrays if astropy
	mag = utils.astropy_to_numpy(mag)
	if emag is not None: emag = utils.astropy_to_numpy(emag)

	# verify that mag and filters variables have the same size
	if len(mag)!=len(filters): raise Exception('filters does not have the size as mag')

	# if the SVO is not provided
	if svo_data is None:
		svo_data = utils.read_SVO_table()

	filterID = svo_data['filterID'] # SVO ID
	ZeroPoint = svo_data['ZeroPoint'] # zero points in Jy
	WavelengthEff = svo_data['WavelengthEff'] # effective wavelength in A
	WidthEff = svo_data['WidthEff'] # effective width in A

	# fill masked values if needed
	filterID = maskedcolumn_to_numpy(filterID)
	ZeroPoint = maskedcolumn_to_numpy(ZeroPoint)
	WavelengthEff = maskedcolumn_to_numpy(WavelengthEff)
	WidthEff = maskedcolumn_to_numpy(WidthEff)
	
	# initialize arrays to store relevant information
	# assign NaN values to be the output for unrecognized filters by SVO
	flux = np.zeros(len(filters)) * np.nan
	if emag is not None: eflux = np.zeros(len(filters)) * np.nan
	zero_point = np.zeros(len(filters)) * np.nan
	wl_eff = np.zeros(len(filters)) * np.nan
	width_eff = np.zeros(len(filters)) * np.nan
	for k, filt in enumerate(filters): # iterate on each filter
		# check first if the filter name is on SVO
		if not filt in filterID: # if filter ID is not recognized
			print(f'   Caveat: {filt} ID is not recognized by SVO, so will be ignored')
		else: # if filter ID is a valid one
			mask = filterID==filt
			zero_point[k] = ZeroPoint[mask][0] # in Jy
			wl_eff[k] = (WavelengthEff[mask][0]*(u.nm*0.1)).to(u.micron).value # in um
			width_eff[k] = (WidthEff[mask][0]*(u.nm*0.1)).to(u.micron).value # in um

			# mag to Jy
			flux[k] = zero_point[k] * 10**(-mag[k]/2.5) # in Jy
			if emag is not None: eflux[k] = 10**(-mag[k]/2.5)*((1.*np.log(10.)/2.5)*zero_point[k]*emag[k]) # in Jy

			# convert flux from Jy to erg/s/cm2/A, if requested
			if flux_unit=='erg/s/cm2/A':
				flux[k] = convert_flux(flux=flux[k], wl=wl_eff[k], unit_in='Jy', unit_out='erg/s/cm2/A')['flux_out'] # in erg/s/cm2/A
				if emag is not None: eflux[k] = convert_flux(flux=flux[k], eflux=eflux[k], wl=wl_eff[k], unit_in='Jy', unit_out='erg/s/cm2/A')['eflux_out'] # in erg/s/cm2/A

	# output dictionary
	out = {'mag': mag, 'flux': flux, 'filters': filters, 'zero_point(Jy)': zero_point, 'flux_unit': flux_unit, 
	       'lambda_eff_SVO(um)': wl_eff, 'width_eff_SVO(um)': width_eff}
	if eflux is not None: 
		out['emag'] = emag
		out['eflux'] = eflux

	return out

#+++++++++++++++++
def load_filter_transmission(filt):
	"""Read or download filter transmission"""

	# path to folder to read and save filter transmissions
	dir_sep = os.sep # directory separator for the current operating system
	path_synphot = os.path.dirname(__file__)+dir_sep
	path_filter_trans = f'{path_synphot}filter_transmissions{dir_sep}'

	# make path_filter_trans directory (if not existing) to store filter transmissions
	if not os.path.exists(path_filter_trans): 
		os.makedirs(path_filter_trans)

	fname = filt.replace('/', '_') + '.dat' # when filter name includes '/' replace it by '_'
	fullpath = path_filter_trans + fname

	# read and download filter response
	if not os.path.exists(fullpath): # filter transmission does not exist yet
		# download from SVO
		print(f'   \nreading and storing {filt} filter directly from SVO')
		page = f'http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={filt}'
		filter_trans = Table.read(page, format='votable')
		# save filter transmission
		ascii.write(filter_trans, fullpath, format='no_header',
					formats={'Wavelength': '%.1f', 'Transmission': '%.10f'})
	else: # filter transmission already stored locally
		filter_trans = ascii.read(fullpath)

	filter_wl = (filter_trans['col1'].data*(0.1*u.nm)).to(u.micron).value # in um
	filter_flux = filter_trans['col2'] # filter transmission

	return filter_wl, filter_flux

#+++++++++++++++++
def assess_filter_coverage(wl, filter_wl):
	"""Check spectral coverage and return label and mask"""

	# no spectral coverage for the filter
	if filter_wl.max() < wl.min() or filter_wl.min() > wl.max():
		# return no coverage
		return 'none'

	# check for partial coverage
	if (filter_wl.min() < wl.min()) or (filter_wl.max() > wl.max()):
		# return incomplete coverage
		return 'incomplete'

	# full coverage
	return 'complete'

#+++++++++++++++++
def filter_coverage_fraction(wl, filter_wl, filter_flux):
	"""
	Fraction of filter transmission covered by the spectrum,
	using assess_filter_coverage output.
	"""

	label = assess_filter_coverage(wl, filter_wl)

	if label == 'none':
		return label, 0.0

	if label == 'complete':
		return label, 1.0

	# label == 'incomplete'
	# total area of the filter transmission
	area_total = utils.np_trapz(filter_flux, filter_wl)
	# area covered by the spectrum
	mask_cov = (filter_wl >= wl.min()) & (filter_wl <= wl.max())
	area_cov = utils.np_trapz(filter_flux[mask_cov], filter_wl[mask_cov])

	# fraction covered by the spectrum
	fraction = area_cov / area_total

	return label, fraction

#+++++++++++++++++
def compute_synthetic_flux(wl, flux, filter_wl, filter_flux, eflux=None):
	"""Compute synthetic flux, flux error, lambda_eff, width_eff"""

	# spectrum wavelengths within the filter wavelength range
	mask_wl = (wl >= filter_wl.min()) & (wl <= filter_wl.max())

	# resample filter transmission to the spectrum wavelength
	filter_flux_resam = np.interp(wl[mask_wl], filter_wl, filter_flux) # dimensionless

	# normalize the transmission curve (it was dimensionless but now it has 1/um units)
	filter_flux_norm = filter_flux_resam / utils.np_trapz(filter_flux_resam, wl[mask_wl]) # 1/um

	# synthetic flux density
	syn_flux = utils.np_trapz(flux[mask_wl] * filter_flux_norm, wl[mask_wl]) # in input flux units (erg/s/cm2/A or Jy)
	esyn_flux = None
	if eflux is not None:
		esyn_flux = np.median(eflux[mask_wl] / flux[mask_wl]) * syn_flux # synthetic flux error as the median fractional flux uncertainties in the filter passband

	# compute the filter's effective wavelength and effective width
	lambda_eff = utils.np_trapz(wl[mask_wl] * filter_flux_resam * flux[mask_wl], wl[mask_wl]) / \
				 utils.np_trapz(filter_flux_resam * flux[mask_wl], wl[mask_wl]) # in um
	width_eff = utils.np_trapz(filter_flux_resam, wl[mask_wl]) / filter_flux_resam.max() # in um

	# output dictionary
	out = {'syn_flux': syn_flux, 'esyn_flux': esyn_flux, 'lambda_eff': lambda_eff, 'width_eff': width_eff}

	return out

#+++++++++++++++++
def _scalar(x):
    return np.asarray(x).item()

#+++++++++++++++++
def format_number(x):
    x = np.asarray(x)

    abs_x = np.abs(x)

    small = abs_x < 1e-4
    large = abs_x >= 1e6

    mask = small | large

    out = np.empty(x.shape, dtype=object)

    scientific_formatter = np.vectorize(
        lambda v: f'{v:.6e}',
        otypes=[object]
    )
    fixed_formatter = np.vectorize(
        lambda v: f'{v:.6f}',
        otypes=[object]
    )

    out[mask] = scientific_formatter(x[mask])
    out[~mask] = fixed_formatter(x[~mask])

    return out

#+++++++++++++++++
def maskedcolumn_to_numpy(col):

	# if it is a MaskedColumn, fill masked values
	if isinstance(col, MaskedColumn):
		# check dtype: numeric -> np.nan, string -> None
		if np.issubdtype(col.dtype, np.number):
			arr = col.filled(np.nan)
		else:
			arr = col.filled(None)

	# if regular Column, take data
	elif isinstance(col, Column):
		arr = col.data
	else:
		arr = col

	# convert numeric to float, leave strings intact
	arr = np.array(arr)  # first convert to numpy array
	if np.issubdtype(arr.dtype, np.number):
		arr = arr.astype(float)

	return arr
