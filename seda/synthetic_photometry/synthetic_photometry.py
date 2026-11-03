import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import Column, MaskedColumn, Table
from astropy import units as u
from .. import utils
from sys import exit

def synthetic_photometry(wl, flux, filters=None, flux_unit=None, eflux=None, detector='auto',
	                     svo_data=None, custom_filters=None, out_file=None):
	'''
	Description:
	------------
		Compute synthetic photometry from an input spectrum using photometric
		filters compiled by SVO and/or user-defined filter transmission curves.
		
		Synthetic flux densities are computed by integrating the spectrum in F_lambda
		units using the appropriate detector weighting convention. For SVO filters,
		the weighting is automatically selected from the SVO DetectorType metadata,
		unless explicitly specified with ``detector='photon'`` o ``detector='energy'``. 
		For user-defined filters, the detector convention must be specified individually 
		for each filter.
		
		For ground-based filters, the SVO transmission curves may already include
		atmospheric transmission. Users can check the ``Comments`` and
		``ProfileReference`` fields in the SVO metadata, accessible through
		:func:`~seda.utils.read_SVO_table`, to determine whether atmospheric
		transmission is included and under which conditions. SEDA does not apply
		additional atmospheric corrections, but users can provide custom
		transmission curves incorporating their desired atmospheric conditions.

	Parameters:
	-----------
	- wl : array
		Wavelength in microns.
	- flux : array
		Flux density in units specified by ``flux_unit``.
	- flux_unit : str
		Flux density and flux density uncertainty units: ``'erg/s/cm2/A'``,
		``'Jy'``, or ``'erg/s/cm2/um'``.
	- filters : list, array, or str, optional
		Filters (following SVO filter IDs) to derive synthetic photometry.
		If not provided, only user-defined filters are used.
	- eflux : array, optional
		Flux density uncertainties in units specified by ``flux_unit``.
	- detector : str, optional
		Detector weighting used for SVO filters:

		- ``'auto'`` (default): automatically selects the weighting convention
		  from each filter's SVO DetectorType.
		- ``'photon'``: applies photon-counting weighting to all SVO filters.
		- ``'energy'``: applies energy-integrating weighting to all SVO filters.

		This argument does not affect user-defined filters, which specify their
		detector conventions individually.
	- svo_data : astropy.table.Table, optional
		Astropy table retrieved from SVO using :func:`~seda.utils.read_SVO_table`.
	- custom_filters : dict, optional
		Dictionary containing user-defined photometric filters. Each filter is
		identified by a unique name and must contain the following entries:

		- ``'wl'`` : array of filter wavelengths in microns.
		- ``'transmission'`` : array of filter transmission values.
		- ``'detector'`` : detector weighting convention, either
		  ``'photon'`` or ``'energy'``.
		- ``'zero_point'`` : optional, photometric zero point in Jy, used to
			  calculate synthetic magnitudes and their uncertainties.

		If ``'zero_point'`` is not provided, synthetic magnitudes, magnitude
		uncertainties, and zero points are returned as NaN for that filter.
		Multiple custom filters can be provided, each with its own wavelength
		array, transmission curve, detector convention, and optional zero point.
		SVO and custom filters can be used separately or simultaneously, but
		their names must be unique.
	- out_file : str, optional
		File name to save the synthetic photometry (in erg/s/cm2/A) as prettytable.
		The file name can include a path, e.g. my_path/syn_phot.dat.
		If not provided, the synthetic photometry will not be saved.

	Returns:
	--------
		Python dictionary with the following parameters for each filter:
			- ``'syn_flux(erg/s/cm2/A)'`` : synthetic flux densities in erg/s/cm2/A.
			- ``'esyn_flux(erg/s/cm2/A)'`` : synthetic flux density uncertainties in erg/s/cm2/A (if input ``eflux`` is provided).
			- ``'syn_flux(Jy)'`` : synthetic flux densities in Jy.
			- ``'esyn_flux(Jy)'`` : synthetic flux density uncertainties in Jy (if input ``eflux`` is provided).
			- ``'syn_mag'`` : synthetic magnitudes (NaN for custom filters without a zero point).
			- ``'esyn_mag'`` : synthetic magnitude errors (if input ``eflux`` and a zero point are provided).
			- ``'filters'`` : SVO and/or custom filters used to derive synthetic photometry.
			- ``'lambda_eff(um)'`` : filters' effective wavelengths in microns computed using the input spectrum.
			- ``'width_eff(um)'`` : filters' effective widths in microns computed using the input spectrum.
			- ``'lambda_eff_SVO(um)'`` : filters' effective wavelengths in microns from SVO (NaN for custom filters).
			- ``'width_eff_SVO(um)'`` : filters' effective widths in microns from SVO (NaN for custom filters).
			- ``'zero_point(Jy)'`` : filters' zero points in Jy (NaN for custom filters without a zero point).
			- ``'label'`` : label indicating if the filters are fully ('complete'), partially ('incomplete'), or not ('none') covered by the input spectrum, or not recognized by SVO ('unrecognizable').
			- ``'coverage_perc'`` : percentage of the filter transmission covered by the spectrum.
			- ``'transmission'`` : dictionary with 2D arrays for the filter transmissions, where the first entry is wavelength in microns and the second is transmission.
			- ``'detector'`` : detector weighting used for each filter when computing synthetic photometry.
			- ``'wl'`` : input spectrum wavelengths in microns.
			- ``'flux'`` : input spectrum flux densities in erg/s/cm2/A (the spectrum is internally converted to erg/s/cm2/A, if necessary, before computing the synthetic photometry).
			- ``'eflux'`` : input spectrum flux density uncertainties in erg/s/cm2/A (if input ``eflux`` is provided).
			- ``'flux_unit'`` : units of the internally stored spectrum flux densities (``'erg/s/cm2/A'``).

	Examples:
	---------
	>>> # obtain synthetic photometry using SVO filters and plot the results
	>>> import seda
	>>>
	>>> # assume we have read wavelengths (wl in um), flux densities (in erg/s/cm2/A),
	>>> # and flux density uncertainties (eflux) for an input spectrum
	>>> filters = ['2MASS/2MASS.J', 'Spitzer/IRAC.I1', 'WISE/WISE.W1']
	>>>
	>>> # calculate synthetic photometry
	>>> out = seda.synthetic_photometry.synthetic_photometry(wl=wl, flux=flux, eflux=eflux,
	>>>	                                                 flux_unit='erg/s/cm2/A',
	>>>	                                                 filters=filters)
	>>>
	>>> # visualize the derived synthetic flux densities
	>>> seda.plots.plot_synthetic_photometry(out)

	>>> # obtain synthetic photometry using two user-defined filters
	>>> import seda
	>>> import numpy as np
	>>>
	>>> # define wavelength arrays and transmission curves
	>>> filter_wl = np.linspace(1.0, 2.0, 1000)
	>>> box = np.where((filter_wl >= 1.4) & (filter_wl <= 1.6), 1.0, 0.0)
	>>> gaussian = np.exp(-0.5 * ((filter_wl - 1.5) / 0.05)**2)
	>>>
	>>> # define custom filters
	>>> custom_filters = {
	...	 'box_filter': {
	...		 'wl': filter_wl,
	...		 'transmission': box,
	...		 'detector': 'energy'
	...	 },
	...	 'gaussian_filter': {
	...		 'wl': filter_wl,
	...		 'transmission': gaussian,
	...		 'detector': 'photon'
	...	 }
	... }
	>>>
	>>> # calculate synthetic photometry using custom filters
	>>> out = seda.synthetic_photometry.synthetic_photometry(wl=wl, flux=flux, eflux=eflux,
	>>>	                                                 flux_unit='erg/s/cm2/A',
	>>>	                                                 custom_filters=custom_filters)
	>>>
	>>> # visualize the derived synthetic flux densities
	>>> seda.plots.plot_synthetic_photometry(out)

	Author: Genaro Suárez

	Initial implementation: 2020

	Last updated: 2026-10-31
	'''

	dir_sep = os.sep # directory separator for the current operating system
	path_synthetic_photometry = os.path.dirname(__file__)+dir_sep

	# validate detector
	if detector not in ('auto', 'photon', 'energy'):
		raise ValueError("detector must be 'auto', 'photon', or 'energy'")

	# input spectrum to numpy
	wl = utils.astropy_to_numpy(wl)
	flux = utils.astropy_to_numpy(flux)
	if eflux is not None: eflux = utils.astropy_to_numpy(eflux)

	# remove nan and null flux values
	valid = np.isfinite(flux) & (flux!=0.0)
	wl = wl[valid]
	flux = flux[valid]
	if eflux is not None: eflux = eflux[valid]

	# ensure that SVO filters are stored as a list, even if a single filter or no filters are provided
	if filters is None:
		filters = []
	elif isinstance(filters, str):
		filters = [filters]
	else:
		filters = list(filters)

	# initialize an empty dictionary if no custom filters are provided
	if custom_filters is None:
		custom_filters = {}

	# ensure that custom filters are provided as a dictionary
	if not isinstance(custom_filters, dict):
		raise TypeError("custom_filters must be a dictionary")

	# validate the structure of each custom filter
	for filt, custom in custom_filters.items():

		# ensure that each custom filter is defined as a dictionary
		if not isinstance(custom, dict):
			raise TypeError(f'Custom filter "{filt}" must be a dictionary')

		# ensure that wavelength and transmission arrays are provided
		if 'wl' not in custom or 'transmission' not in custom:
			raise ValueError(
				f'Custom filter "{filt}" must contain "wl" and "transmission"'
			)

		# convert wavelength and transmission arrays to numpy arrays
		filter_wl = np.asarray(custom['wl'], dtype=float)
		filter_flux = np.asarray(custom['transmission'], dtype=float)

		# ensure that both arrays are one-dimensional
		if filter_wl.ndim != 1 or filter_flux.ndim != 1:
			raise ValueError(f'Custom filter "{filt}": arrays must be one-dimensional')

		# ensure that both arrays have the same length and at least two elements
		if len(filter_wl) != len(filter_flux) or len(filter_wl) < 2:
			raise ValueError(f'Custom filter "{filt}": invalid array lengths')

		# ensure that both arrays contain only finite values
		if not np.all(np.isfinite(filter_wl)) or not np.all(np.isfinite(filter_flux)):
			raise ValueError(f'Custom filter "{filt}": arrays must contain finite values')

		# ensure that wavelengths are strictly increasing
		if np.any(np.diff(filter_wl) <= 0):
			raise ValueError(f'Custom filter "{filt}": wavelengths must be increasing')

		# ensure that transmission values are non-negative and not all zero
		if np.any(filter_flux < 0) or not np.any(filter_flux > 0):
			raise ValueError(f'Custom filter "{filt}": invalid transmission values')

		# ensure that the detector convention is provided for each custom filter
		if 'detector' not in custom:
			raise ValueError(f'Custom filter "{filt}" must contain "detector"')

		# ensure that the detector convention is valid
		if custom['detector'] not in ('photon', 'energy'):
			raise ValueError(
				f'Custom filter "{filt}": detector must be "photon" or "energy"'
			)

		# validate the zero point if provided
		if 'zero_point' in custom:
			zero_point_filt = custom['zero_point']

			if not np.isscalar(zero_point_filt) or not np.isfinite(zero_point_filt) or zero_point_filt <= 0:
				raise ValueError(f'Custom filter "{filt}": zero_point must be a positive finite number')

	# prevent duplicate names between SVO and custom filters
	duplicate_filters = set(filters) & set(custom_filters)
	if duplicate_filters:
		raise ValueError(f'Duplicate filter names between SVO and custom filters: {sorted(duplicate_filters)}')

	# ensure that at least one SVO or custom filter is provided
	if not filters and not custom_filters:
		raise ValueError("Provide at least one SVO or custom filter")

	# combine SVO and custom filter names into a single list for processing
	filters = filters + list(custom_filters)

	# identify SVO filters
	svo_filters = []

	for filt in filters:
		if filt not in custom_filters:
			svo_filters.append(filt)

	# read SVO metadata only if SVO filters are requested
	if svo_filters and svo_data is None:
		svo_data = utils.read_SVO_table()

	if svo_filters:
		filterID = svo_data['filterID'] # SVO ID
		ZeroPoint = svo_data['ZeroPoint'] # in Jy
		DetectorType = svo_data['DetectorType'] # SVO detector type

		# fill masked values if needed
		filterID = maskedcolumn_to_numpy(filterID)
		ZeroPoint = maskedcolumn_to_numpy(ZeroPoint)
		DetectorType = maskedcolumn_to_numpy(DetectorType)

	# convert input spectrum to F_lambda (erg/s/cm2/A), if needed
	if flux_unit == 'Jy':
		out = convert_flux(flux=flux, eflux=eflux, wl=wl,
						   unit_in='Jy', unit_out='erg/s/cm2/A')
		flux = out['flux_out'] # in erg/s/cm2/A
		if eflux is not None: eflux = out['eflux_out'] # in erg/s/cm2/A

	elif flux_unit == 'erg/s/cm2/um':
		flux = (flux*u.erg/u.s/u.cm**2/u.micron).to(u.erg/u.s/u.cm**2/(u.nm*0.1)).value # in erg/s/cm2/A
		if eflux is not None:
			eflux = (eflux*u.erg/u.s/u.cm**2/u.micron).to(u.erg/u.s/u.cm**2/(u.nm*0.1)).value # in erg/s/cm2/A

	flux_unit = 'erg/s/cm2/A'

	# initialize arrays to store relevant information
	n_filters = len(filters)
	# assign NaN values to be the output for unrecognized filters by SVO
	syn_flux_erg = np.full(n_filters, np.nan)
	esyn_flux_erg = np.full(n_filters, np.nan)
	syn_flux_Jy = np.full(n_filters, np.nan)
	esyn_flux_Jy = np.full(n_filters, np.nan)
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
	detector_used = np.full(n_filters, None, dtype=object)

	# main loop over filters
	for k, filt in enumerate(filters): # iterate on each filter
		# determine whether the filter is user-defined or from SVO
		is_custom = filt in custom_filters

		if is_custom:
			# retrieve the wavelength and transmission arrays of the custom filter
			custom = custom_filters[filt]
			filter_wl = np.asarray(custom['wl'], dtype=float)
			filter_flux = np.asarray(custom['transmission'], dtype=float)

		else:
			# check whether the filter is recognized by SVO
			if filt not in filterID: # if filter ID is not recognized
				label[k] = 'unrecognizable'
				print(f'   Caveat: "{filt}" filter is not recognized by SVO, so will be ignored.')
				continue # to jump to the next iteration

			# load the transmission curve from SVO
			filter_wl, filter_flux = load_filter_transmission(filt)

		# store the transmission curve
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

		# determine detector weighting for the current filter
		if is_custom:
			detector_filt = custom['detector']

		elif detector == 'auto':
			mask_filt = filterID == filt
			detector_type = DetectorType[mask_filt][0]

			try:
				detector_type = int(detector_type)
			except (ValueError, TypeError):
				raise ValueError(f'Invalid SVO DetectorType for {filt}: {detector_type}')

			if detector_type not in (0, 1):
				raise ValueError(f'Unknown SVO DetectorType for {filt}: {detector_type}')

			detector_filt = {0: 'energy', 1: 'photon'}[detector_type]

		else:
			detector_filt = detector
		detector_used[k] = detector_filt

		# compute synthetic flux density
		out_synt_flux = _compute_synthetic_flux(wl=wl, flux=flux, eflux=eflux, filter_wl=filter_wl,
		                                        filter_flux=filter_flux, detector=detector_filt)
		syn_flux_erg[k] = out_synt_flux['syn_flux'] # in erg/s/cm2/A
		esyn_flux_erg[k] = out_synt_flux['esyn_flux'] # in erg/s/cm2/A
		lambda_eff[k] = out_synt_flux['lambda_eff'] # in um
		width_eff[k] = out_synt_flux['width_eff'] # in um

		# derive synthetic magnitudes
		# convert first synthetic flux densities from erg/s/cm2/A into Jy
		out = convert_flux(flux=syn_flux_erg[k], eflux=esyn_flux_erg[k], wl=lambda_eff[k], unit_in='erg/s/cm2/A', unit_out='Jy')
		syn_flux_Jy[k] = out['flux_out'] # in Jy
		if eflux is not None: esyn_flux_Jy[k] = out['eflux_out'] # in Jy


		# calculate synthetic magnitudes
		if is_custom:
			# calculate magnitudes only if a zero point is provided
			if 'zero_point' in custom:
				zero_point[k] = custom['zero_point']

				syn_mag[k] = -2.5 * np.log10(syn_flux_Jy[k] / zero_point[k]) # mag
				if eflux is not None:
					esyn_mag[k] = (2.5 / np.log(10)) * np.sqrt((esyn_flux_Jy[k] / syn_flux_Jy[k])**2)
		else:
			# retrieve the zero point for the SVO filter
			mask_filt = filterID == filt
			if not any(mask_filt):
				raise Exception(f'\nERROR: No zero point for filter {filt}')

			# convert synthetic flux density from Jy to magnitude
			out_mag = flux_to_mag(flux=syn_flux_Jy[k], eflux=esyn_flux_Jy[k],
			                      filters=filt,	flux_unit='Jy', svo_data=svo_data)
			syn_mag[k] = out_mag['mag'][0] # mag
			if eflux is not None: esyn_mag[k] = out_mag['emag'][0] # mag
			lambda_eff_SVO[k] = out_mag['lambda_eff_SVO(um)'][0] # um
			width_eff_SVO[k] = out_mag['width_eff_SVO(um)'][0] # um
			zero_point[k] = ZeroPoint[mask_filt][0] # Jy

	# output dictionary
	out = {'syn_flux(Jy)': syn_flux_Jy, 'syn_flux(erg/s/cm2/A)': syn_flux_erg, 'syn_mag': syn_mag, 'lambda_eff(um)': lambda_eff,
	       'width_eff(um)': width_eff, 'lambda_eff_SVO(um)': lambda_eff_SVO, 'width_eff_SVO(um)': width_eff_SVO,
	       'zero_point(Jy)': zero_point, 'label': label, 'coverage_perc': coverage_perc, 'detector': detector_used, 'transmission': transmission,
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
		Convert flux densities from Jy to erg/s/cm2/A or vice versa.

	Parameters:
	-----------
	- flux : array, list, or float
 		Input flux densities in units specified by ``unit_in``.
	- wl : array, float
		Wavelengths (in microns) associated to ``flux``.
	- unit_in : str
		Units of ``flux``: ``'Jy'`` or ``'erg/s/cm2/A'``.
	- unit_out : str
		Units of output flux densities: ``'Jy'`` or ``'erg/s/cm2/A'``.
	- eflux : array, float, optional
 		Flux density uncertainties for ``flux`` in ``unit_in``.

	Returns:
	--------
	Dictionary with converted flux densities:
		- ``'flux_in'`` : input flux densities.
		- ``'flux_out'`` : output flux densities.
		- ``'wl_in'`` : input wavelengths.
		- ``'eflux_in'`` : (if ``eflux``) input flux density uncertainties.
		- ``'eflux_out'`` : (if ``eflux``) output flux density uncertainties.
		- ``'unit_in'`` : ``unit_in``.
		- ``'unit_out'`` : ``unit_out``.

	Example:
	--------
	>>> # convert flux densities from Jy to erg/s/cm2/A
	>>> import seda
	>>> import numpy as np
	>>>
	>>> flux = np.array([0.005, 0.006]) # test fluxes in Jy
	>>> wl = np.array([1.23, 2.16]) # test wavelengths in microns
	>>> eflux = flux/10. # test flux errors in Jy
	>>>
	>>> seda.synthetic_photometry.convert_flux(flux=flux, wl=wl, eflux=eflux, unit_in='Jy', unit_out='erg/s/cm2/A')
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
		Convert flux densities into magnitudes.

	Parameters:
	-----------
	- flux : array, list, or float
 		Input flux densities in units specified by ``flux_unit``.
	- filters : array, list, or str
		Filters (following SVO filter IDs) used to obtain ``flux``.
		It must have the same size as ``flux``.
	- flux_unit : str, optional (default ``'Jy'``)
		Units of ``flux``: ``'Jy'`` or ``'erg/s/cm2/A'``.
	- eflux : array, float, optional
 		Flux density uncertainties for ``flux`` in ``flux_unit``.
	- svo_data : astropy.table.Table, optional
		Astropy table retrieved from SVO using :func:`~seda.utils.read_SVO_table`.

	Returns:
	--------
	Dictionary with converted magnitudes and input flux densities:
		- ``'mag'`` : resulting magnitudes
		- ``'emag'`` : (if ``eflux``) uncertainty of resulting magnitudes
		- ``'flux'`` : input flux density
		- ``'eflux'`` : (if ``eflux``) input flux density uncertainties
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
	if svo_data is None: svo_data = utils.read_SVO_table()

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
				flux[k] = convert_flux(flux=flux[k], wl=wl_eff[k], unit_in='erg/s/cm2/A',
				                       unit_out='Jy')['flux_out'] # in Jy
				if eflux is not None:
					eflux[k] = convert_flux(flux=flux[k], eflux=eflux[k], wl=wl_eff[k],
					                         unit_in='erg/s/cm2/A', unit_out='Jy')['eflux_out'] # in Jy

			# Jy to mag
			mag[k] = -2.5 * np.log10(flux[k] / zero_point[k]) # in mag
			if eflux is not None:
				emag[k] = (2.5 / np.log(10)) * np.sqrt((eflux[k] / flux[k])**2)#+(ephot_F0/phot_F0)**2) # in mag

	# output dictionary
	out = {'flux': flux_ori, 'mag': mag, 'filters': filters, 'zero_point(Jy)': zero_point,
	       'flux_unit': flux_unit, 'lambda_eff_SVO(um)': wl_eff, 'width_eff_SVO(um)': width_eff}
	if eflux is not None:
		out['eflux'] = eflux_ori
		out['emag'] = emag

	return out

#+++++++++++++++++
def mag_to_flux(mag, filters, flux_unit='Jy', emag=None, svo_data=None):
	'''
	Description:
	------------
		Convert magnitudes into flux densities.

	Parameters:
	-----------
	- mag : array, list, or float
 		Input magnitudes in mag.
	- filters : array, list, or str
		Filters (following SVO filter IDs) used to obtain ``mag``.
		It must have the same size as ``mag``.
	- flux_unit : str, optional (default ``'Jy'``)
		Units to return flux densities: ``'Jy'`` or ``'erg/s/cm2/A'``.
	- emag : array, float, optional
 		Magnitude uncertainties for ``mag`` in mag.
	- svo_data : astropy.table.Table, optional
		Astropy table retrieved from SVO using :func:`~seda.utils.read_SVO_table`.

	Returns:
	--------
	Dictionary with converted flux densities:
		- ``'flux'`` : resulting flux densities.
		- ``'eflux'`` : (if ``emag``) resulting flux density uncertainties.
		- ``'mag'`` : input magnitudes.
		- ``'emag'`` : (if ``emag``) uncertainty of input magnitudes.
		- ``'filters'`` : input filter IDs.
		- ``'zero_point(Jy)'`` : filters' zero points in Jy.
		- ``'lambda_eff_SVO(um)'`` : effective wavelengths in microns from SVO.
		- ``'width_eff_SVO(um)'`` : effective width in microns from SVO.

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
	>>> seda.synthetic_photometry.mag_to_flux(mag=mag, emag=emag, filters=filters)
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

	# convert floats or integer (if any) into arrays
	if isinstance(mag, (float, int)): mag = np.array([mag])
	if emag is not None:
		if isinstance(emag, (float, int)): emag = np.array([emag])

	# convert str (if any) into list
	filters = utils.var_to_list(filters)
	# convert input variables into numpy arrays if astropy
	mag = utils.astropy_to_numpy(mag)
	if emag is not None: emag = utils.astropy_to_numpy(emag)

	# verify that mag and filters variables have the same size
	if len(mag)!=len(filters): raise Exception('filters does not have the size as mag')

	# if the SVO is not provided
	if svo_data is None: svo_data = utils.read_SVO_table()

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
			elif flux_unit=='erg/s/cm2/um':
				flux[k] = convert_flux(flux=flux[k], wl=wl_eff[k], unit_in='Jy', unit_out='erg/s/cm2/A')['flux_out'] # in erg/s/cm2/A
				flux[k] = (flux[k]*u.erg/u.s/u.cm**2/(u.nm*0.1)).to(u.erg/u.s/u.cm**2/u.micron).value # erg/s/cm2/um
				if emag is not None:
					eflux[k] = convert_flux(flux=flux[k], eflux=eflux[k], wl=wl_eff[k], unit_in='Jy', unit_out='erg/s/cm2/A')['eflux_out'] # in erg/s/cm2/A
					eflux[k] = (eflux[k]*u.erg/u.s/u.cm**2/(u.nm*0.1)).to(u.erg/u.s/u.cm**2/u.micron).value # erg/s/cm2/um

	# output dictionary
	out = {'mag': mag, 'flux': flux, 'filters': filters, 'zero_point(Jy)': zero_point, 'flux_unit': flux_unit,
	       'lambda_eff_SVO(um)': wl_eff, 'width_eff_SVO(um)': width_eff}
	if emag is not None:
		out['emag'] = emag
		out['eflux'] = eflux

	return out

#+++++++++++++++++
def calibrate_spectrum(wl, flux, flux_unit, mag, emag, filters, eflux=None, svo_data=None):
	'''
	Description:
	------------
		Calibrate a spectrum by scaling it to match observed photometry.

		Computes synthetic photometry from the input spectrum, compares it to
		observed magnitudes, derives an average magnitude offset, and applies
		a corresponding scaling factor to the spectrum.

	Parameters:
	-----------
	- wl : array
		Wavelength array in microns.
	- flux : array
		Flux density values in ``flux_unit`` unit corresponding to ``wl``.
	- flux_unit : str
		Flux density and flux density uncertainty units: ``'erg/s/cm2/A'``, ``'Jy'``, or ``erg/s/cm2/um``.
	- mag : array
		Observed magnitudes in mag.
	- emag : array
		Uncertainties on observed magnitudes in mag.
	- filters : list, array, or str
		List of filters (following SVO filter IDs) used for photometry.
	- eflux : array, optional
		Flux density uncertainties in ``flux_unit`` unit. If ``None``, uncertainties are ignored.
	- svo_data : astropy.table.Table, optional
		Astropy table retrieved from SVO using :func:`~seda.utils.read_SVO_table`.

	Returns:
	--------
	Python dictionary with the following parameters:
		- ``'wl``': wavelength in microns of calibrated spectrum.
		- ``'flux``': flux density in ``flux_unit`` of calibrated spectrum.
		- ``'eflux``': flux density uncertainties in ``flux_unit`` of calibrated spectrum.
		- ``'syn_mag``': synthetic magnitudes in mag for ``filters`` from the calibrated spectrum.
		- ``'esyn_mag``': synthetic magnitude errors in mag for ``filters`` from the calibrated spectrum.
		- ``'syn_flux``': synthetic flux density in ``flux_unit`` for ``filters`` from the calibrated spectrum.
		- ``'esyn_flux``': synthetic flux density uncertainties in ``flux_unit`` for ``filters`` from the calibrated spectrum.
		- ``'scaling``': scaling factor applied to the input spectrum to calibrate it.
		- ``'flux_unit``': units of input spectrum flux density and their uncertainties.

	Example:
	--------
	>>> # Flux-calibrate a near-infrared spectrum using 2MASS photometry.
	>>> # Assume the spectrum wavelength, flux density and flux density uncertainties are loaded in the variable wl, flux, and eflux,
	>>> # and the units of fluxes are 'erg/s/cm2/um'.
	>>> flux_unit='erg/s/cm2/um'
	>>>
	>>> # Example of 2MASS photometry
	>>> filters = ['2MASS/2MASS.J', '2MASS/2MASS.H', '2MASS/2MASS.Ks']
	>>> mag = [13.997, 13.284, 12.829]
	>>> emag = [0.032, 0.036, 0.029]
	>>>
	>>>	# calibrate spectrum
	>>> out = seda.synthetic_photometry.calibrate_spectrum(wl=wl, flux=flux, eflux=eflux,
	                                                       flux_unit=flux_unit,
	                                                       mag=mag, emag=emag, filters=filters)
	>>>
	>>> # visualize the results
	>>> seda.plots.plot_calibrate_spectrum(out)

	Author: Genaro Suárez

	Date: 2026-05-04
	'''

	# ensure input mag and emag are numpy arrays
	mag = utils.var_to_numpy(mag)
	emag = utils.var_to_numpy(emag)

	# synthetic photometry from the spectrum
	if svo_data is None: svo_data = utils.read_SVO_table()
	out_syn = synthetic_photometry(wl=wl, flux=flux, eflux=eflux, flux_unit=flux_unit,
	                               filters=filters, svo_data=svo_data)

	syn_mag = out_syn['syn_mag']
	if eflux is not None: esyn_mag = out_syn['esyn_mag']
	else: esyn_mag = None
	lambda_eff = out_syn['lambda_eff(um)']  # um
	width_eff = out_syn['width_eff(um)']	# um

	# residual between observed and synthetic magnitudes
	res = np.mean(mag - syn_mag)

	# scaling factor
	scaling = 10**(-res / 2.5)

	# scale spectrum
	flux_calib = flux * scaling
	if eflux is not None: eflux_calib = eflux * scaling
	else: eflux_calib = None

	# calibrate synthetic magnitudes
	syn_mag_calib = syn_mag + res
	if eflux is not None: esyn_mag_calib = esyn_mag
	else: esyn_mag_calib = None

	# calibrate synthetic flux densities
	out_syn_flux = mag_to_flux(mag=syn_mag_calib, emag=esyn_mag_calib,
	                           filters=filters, flux_unit=flux_unit,
	                           svo_data=svo_data)
	syn_flux_calib = out_syn_flux['flux']
	if eflux is not None: esyn_flux_calib = out_syn_flux['eflux']
	else: esyn_flux_calib = None

	print(f'Spectrum was scaled by a factor of {scaling}')

	# output dictionary
	out = {
		'wl': wl,
		'flux': flux_calib,
		'eflux': eflux_calib,
		'syn_mag': syn_mag_calib,
		'esyn_mag': esyn_mag_calib,
		'syn_flux': syn_flux_calib,
		'esyn_flux': esyn_flux_calib,
		'lambda_eff': lambda_eff,
		'width_eff': width_eff,
		'scaling': scaling,
		'flux_unit': flux_unit,
		'obs_mag': mag,
		'eobs_mag': emag,
		'filters': filters,
		'flux_unit': flux_unit
	}

	return out

#+++++++++++++++++
def load_filter_transmission(filt):
	#"""Read or download filter transmission"""

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
	# read locally stored filter transmission
	filter_trans = ascii.read(fullpath)
	filter_wl = (filter_trans['col1'].data*(0.1*u.nm)).to(u.micron).value # in um
	filter_flux = filter_trans['col2'] # filter transmission

	return filter_wl, filter_flux

#+++++++++++++++++
def assess_filter_coverage(wl, filter_wl):
	#"""Check spectral coverage and return label and mask"""

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
	#"""
	#Fraction of filter transmission covered by the spectrum,
	#using assess_filter_coverage output.
	#"""

	label = assess_filter_coverage(wl, filter_wl)

	if label == 'none':
		return label, 0.0

	if label == 'complete':
		return label, 1.0

	# label == 'incomplete'
	# total area of the filter transmission
	area_total = utils.np_trapz(filter_flux, filter_wl)

	# area covered by the spectrum
	wl_min = max(wl.min(), filter_wl.min())
	wl_max = min(wl.max(), filter_wl.max())

	mask_cov = (filter_wl > wl_min) & (filter_wl < wl_max)

	wl_cov = np.concatenate(([wl_min], filter_wl[mask_cov], [wl_max]))
	flux_cov = np.interp(wl_cov, filter_wl, filter_flux)

	area_cov = utils.np_trapz(flux_cov, wl_cov)

	# fraction covered by the spectrum
	fraction = area_cov / area_total

	# no transmission covered by the spectrum
	if fraction <= 0:
		return 'none', 0.0

	# consider complete coverage when at least 99.9% of the transmission is covered
	if fraction >= 0.999:
		label = 'complete'

	return label, fraction

#+++++++++++++++++
def _compute_synthetic_flux(wl, flux, filter_wl, filter_flux, eflux=None, detector='photon'):
	"""
	Compute synthetic flux density, flux density uncertainty, lambda_eff, and width_eff.

	This is an internal helper function used by :func:`~seda.synthetic_photometry.synthetic_photometry`.
	The input spectrum must be provided as F_lambda. The synthetic flux density is computed using
	photon-counting weighting by default, appropriate for most astronomical detectors. Energy-integrating
	weighting can be selected with ``detector='energy'``.
	"""

	# spectrum wavelengths within the filter wavelength range
	mask_wl = (wl >= filter_wl.min()) & (wl <= filter_wl.max())

	# resample filter transmission to the spectrum wavelength
	filter_flux_resam = np.interp(wl[mask_wl], filter_wl, filter_flux) # dimensionless

	# detector weighting
	if detector == 'photon':
		weight = wl[mask_wl] * filter_flux_resam # photon-counting detector
	elif detector == 'energy':
		weight = filter_flux_resam # energy-integrating detector
	else:
		raise ValueError("detector must be either 'photon' or 'energy'")

	# normalize the transmission curve
	weight_integral = utils.np_trapz(weight, wl[mask_wl])

	if not np.isfinite(weight_integral) or weight_integral <= 0:
		raise ValueError('Filter has no positive integrated transmission over the input spectrum')

	weight_norm = weight / weight_integral

	# synthetic flux density
	syn_flux = utils.np_trapz(flux[mask_wl] * weight_norm, wl[mask_wl]) # in input F_lambda units
	esyn_flux = None
	if eflux is not None:
		# synthetic flux uncertainty as the median fractional flux uncertainties in the filter passband
		esyn_flux = np.median(eflux[mask_wl] / flux[mask_wl]) * syn_flux

	# compute the filter's effective wavelength and effective width
	# effective wavelength
	lambda_eff = utils.np_trapz(wl[mask_wl] * flux[mask_wl] * weight, wl[mask_wl]) / \
	             utils.np_trapz(flux[mask_wl] * weight, wl[mask_wl])

	# SVO effective width (always energy-counter definition)
	width_eff = utils.np_trapz(wl[mask_wl] * filter_flux_resam, wl[mask_wl]) / \
	            (wl[mask_wl] * filter_flux_resam).max()

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
