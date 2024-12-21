from astropy.io import ascii
from astropy.table import Table
from ..utils import *
import numpy as np
import os
from sys import exit

def synthetic_photometry(wl, flux, filters, flux_unit, eflux=None): 
	'''
	Description:
	------------
		Compute synthetic magnitudes and fluxes for different filters from an input spectrum.

	Parameters:
	-----------
	- wl : float array
		Wavelength in um.
	- flux : float array
		Fluxes in units specified by ``flux_unit``.
	- flux_unit : str
		Flux and flux error units: ``erg/s/cm2/A`` or ``Jy``.
	- filters : str
		Filters (following SVO filter IDs) to derive synthetic photometry.
	- eflux : flux array (optional) 
		Flux errors in units specified by ``flux_unit``.

	Returns:
	--------
	Python dictionary with the following parameters for each filter:
		- ``syn_flux(erg/s/cm2/A)`` : synthetic fluxes in erg/s/cm2/A.
		- ``esyn_flux(erg/s/cm2/A)`` : synthetic flux errors in erg/s/cm2/A (if input ``eflux`` is provided).
		- ``syn_flux(Jy)`` : synthetic fluxes in Jy.
		- ``esyn_flux(Jy)`` : synthetic flux errors in Jy (if input ``eflux`` is provided).
		- ``syn_mag`` : synthetic magnitudes.
		- ``esyn_mag`` : synthetic magnitude errors (if input ``eflux`` is provided).
		- ``filters`` : filters used to derive synthetic photometry.
		- ``lambda_eff(um)`` : filters' effective wavelengths in microns.
		- ``width_eff(um)`` : filters' effective width in micron.
		- ``zero_point(Jy)`` : filters' zero points in Jy.
		- ``label`` : label indicating if the filters are fully ('complete'), partially ('incomplete'), or no ('none') covered by the input spectrum or no recognized by SVO ('unrecognizable').
		- ``transmission`` : dictionary with 2D arrays for the filter transmissions, where the first first entry is wavelength in microns and the second one is the transmission.
		- ``wl`` : input spectrum wavelengths.
		- ``flux`` : input spectrum fluxes.
		- ``eflux`` : input spectrum flux uncertainties (if input ``eflux`` is provided)..
		- ``flux_unit`` : flux units of the input spectrum.

	Example:
	--------
	>>> import seda
	>>> 
	>>> # assume we have read a wavelengths (wl in um), fluxes (in erg/s/cm2/A), and flux errors (eflux) for an input spectrum
	>>>	# and we want synthetic photometry for several filters
	>>> filters = (['2MASS/2MASS.J', 'Spitzer/IRAC.I1', 'WISE/WISE.W1']) # filters of interest
	>>>	# run the code
	>>> out = seda.synthetic_photometry(wl=wl, flux=flux, eflux=eflux, flux_unit='erg/s/cm2/A', filters=filters)
	>>> 
	>>> # visualize the derived synthetic fluxes
	>>> seda.plot_synthetic_photometry(out)

	Author: Genaro Suárez

	Modification history:
		- 2024/05/24: SVO VOTable table is read from a link with the most updated filter parameters
		- 2024/05/09: included VOTable table with all SVO filter zero points
		- 2024/05/07: filter transmissions are read and downloaded from SVO, if not already locally stored
		- 2024/04/20: created a function
		- 2021      : functional code but not as a function
	'''

	path_synthetic_photometry = os.path.dirname(__file__)+'/'

	# input spectrum to numpy
	wl = astropy_to_numpy(wl)
	flux = astropy_to_numpy(flux)

	mask_nonan = ~np.isnan(flux)
	wl = wl[mask_nonan]
	flux = flux[mask_nonan]
	if (eflux is not None): eflux = eflux[mask_nonan]

	# when filters parameter is given as a string, convert the variable to a list so len(filters) returns 1 rather than the string length
	if (type(filters) is str): filters = ([filters])

	# read filters transmission curves and zero points
	svo_table = f'{path_synthetic_photometry}aux/FPS_info.xml'
	if os.path.exists(svo_table): 
		svo_data = Table.read(svo_table, format='votable') # open downloaded table with filters' info
	else:
		svo_data = Table.read('https://svo.cab.inta-csic.es/files/svo/Public/HowTo/FPS/FPS_info.xml', format='votable') # this link will be updated as soon as new filters are added to FPS. 
		#if not os.path.exists(svo_table+'/aux'): os.makedirs(svo_table+'aux') 
		if not os.path.exists(svo_table): os.makedirs(path_synthetic_photometry+'aux') 
		svo_data.write(svo_table, format='votable') # save the table to avoid reading it from the web each time the code is run, which can take a few seconds
	filterID = svo_data['filterID'] # VSO ID
	ZeroPoint = svo_data['ZeroPoint'] # Jy

	# arrays to store relevant information
	syn_flux_Jy = np.zeros(len(filters))
	esyn_flux_Jy = np.zeros(len(filters))
	syn_flux_erg = np.zeros(len(filters))
	esyn_flux_erg = np.zeros(len(filters))
	syn_mag = np.zeros(len(filters))
	esyn_mag = np.zeros(len(filters))
	lambda_eff = np.zeros(len(filters))
	width_eff = np.zeros(len(filters))
	zero_point = np.zeros(len(filters))
	# assign NaN values (this will be the output for an input filter name not recognized by the SVO)
	syn_flux_Jy[:] = np.nan
	esyn_flux_Jy[:] = np.nan
	syn_flux_erg[:] = np.nan
	esyn_flux_erg[:] = np.nan
	syn_mag[:] = np.nan
	esyn_mag[:] = np.nan
	lambda_eff[:] = np.nan
	width_eff[:] = np.nan
	zero_point[:] = np.nan
	label = np.empty(len(filters), dtype=object) # to assign a label to each filter based on its spectral coverage
	label[:] = 'complete'
	transmission = {} # dictionary to save the filter transmissions
	for k in range(len(filters)): # iterate for each filter
		# check first if the filter name is on the VSO
		if not filters[k] in filterID: 
			label[k] = 'unrecognizable'
			print(f'Caveat for {filters[k]}: FILTER NOT RECOGNIZED BY THE SVO, so will be ignored')
		else:
			# read filter transmission
			# check if the filter transmission exits locally already
			path_filter_transmissions = f'{path_synthetic_photometry}/filter_transmissions/'
			if not os.path.exists(path_filter_transmissions): os.makedirs(path_filter_transmissions) # make directory (if not existing) to store filter transmissions
			filter_transmission_name = filters[k].replace('/', '_')+'.dat' # when filter name includes '/' replace it by '_'
			if not os.path.exists(path_filter_transmissions+filter_transmission_name): # filter transmission does not exits yet
				print(f'\nreading and storing filter {filters[k]} directly from the SVO')
				# read filter transmission directly from SVO
				page = f'http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={filters[k]}'
				filter_transmission = Table.read(page, format='votable')
				# save filter transmission if it doesn't exist already
				ascii.write(filter_transmission, path_filter_transmissions+filter_transmission_name, format='no_header', formats={'Wavelength': '%.1f', 'Transmission': '%.10f'})
	
			filter_transmission = ascii.read(path_filter_transmissions+filter_transmission_name) # read locally stored filter transmission
	
			filter_wl = filter_transmission['col1'] / 1e4 # in um
			filter_flux = filter_transmission['col2'] # filter transmission (named filter_flux just for ease)
			transmission[filters[k]] = np.array((filter_wl, filter_flux))
		
			# verify the spectrum fully covers the filter transmission
			if ((filter_wl.max()<wl.min()) | (filter_wl.min()>wl.max())): # filter out of the spectrum coverage
				label[k] = 'none'
				print(f'Caveat for {filters[k]}: NO WAVELENGTH COVERAGE, so synthetic photometry won\'t be obtained')
			else: # filter fully or partially covered
				if ((filter_wl.min()<wl.min()) & (filter_wl.max()>wl.min())): # blue-end of the filter partially covered
					print(f'Caveat for {filters[k]}: NO FULL SPECTRAL COVERAGE, so the synthetic photometry is a lower limit')
					label[k] = 'incomplete'
				if ((filter_wl.max()>wl.max()) & (filter_wl.min()<wl.max())): # red-end of the filter partially covered
					print(f'Caveat for {filters[k]}: NO FULL SPECTRAL COVERAGE, so the synthetic photometry is a lower limit')
					label[k] = 'incomplete'

				# wavelength dispersion of the spectrum in the filter wavelength range
				mask_wl = (wl>=filter_wl.min()) & (wl<=filter_wl.max())
		
				wl_disp = wl[mask_wl][1:] - wl[mask_wl][:-1] # dispersion of spectrum spectra (um)
				wl_disp = np.append(wl_disp, wl_disp[-1]) # add an element equal to the last row to keep the same shape as the wl array

				# synthetic photometry
				# resample filter transmission to the spectrum wavelength
				filter_flux_resam = np.interp(wl[mask_wl], filter_wl, filter_flux) # dimensionless

				# normalize the transmission curve (it was dimensionless but now it has 1/um units)
				filter_flux_resam_norm = filter_flux_resam / sum(filter_flux_resam*wl_disp) # 1/um

				# synthetic flux density
				syn_flux = sum(flux[mask_wl]*filter_flux_resam_norm*wl_disp) # in input flux units (erg/s/cm2/A or Jy)
				if (eflux is not None): esyn_flux = np.median(eflux[mask_wl]/flux[mask_wl]) * syn_flux # synthetic flux error as the median fractional flux uncertainties in the filter passband

				# compute the effective wavelength and effective width for each filter
				lambda_eff[k] = sum(wl[mask_wl]*filter_flux_resam*flux[mask_wl]*wl_disp) / sum(filter_flux_resam*flux[mask_wl]*wl_disp) # um
				width_eff[k] = sum(filter_flux_resam*wl_disp) / filter_flux_resam.max() # um
				
				# convert flux to magnitudes
				# first from erg/s/cm2/A to Jy (if needed) and then from Jy to mag
				if (flux_unit=='erg/s/cm2/A'):
					#syn_flux_Jy = syn_flux*(lambda_eff*1e-4)**2 / 3e-21 # Jy
					syn_flux_Jy[k] = 3.33564095e+04*syn_flux*(lambda_eff[k]*1e4)**2 # Jy (the above option gives the same result)
					if (eflux is not None): esyn_flux_Jy[k] = esyn_flux/syn_flux * syn_flux_Jy[k] # Jy
		
					syn_flux_erg[k] = syn_flux # erg/s/cm2/A
					if (eflux is not None): esyn_flux_erg[k] = esyn_flux # erg/s/cm2/A
		
				if (flux_unit=='Jy'): # convert Jy to erg/s/cm2/A to be an output
					syn_flux_erg[k] = syn_flux/(3.33564095e+04*(lambda_eff[k]*1e4)**2) # erg/s/cm2/A
					if (eflux is not None): esyn_flux_erg[k] = esyn_flux/syn_flux * syn_flux_erg[k] # erg/s/cm2/A
		
					syn_flux_Jy[k] = syn_flux # Jy
					if (eflux is not None): esyn_flux_Jy[k] = esyn_flux # Jy
		
				# from Jy to mag
				mask = filterID==filters[k]
				if any(mask) is False: print(f'\nERROR: NO ZERO POINT FOR FILTER {filters[k]}'), exit()
				syn_mag[k] = -2.5*np.log10(syn_flux_Jy[k]/ZeroPoint[mask]) # in mag
				if (eflux is not None): esyn_mag[k] = (2.5/np.log(10))*np.sqrt((esyn_flux_Jy[k]/syn_flux_Jy[k])**2)#+(ephot_F0/phot_F0)**2) # in mag

				zero_point[k] = ZeroPoint[mask]

				del filter_transmission # remove variable with filter transmission so it won't exit if an input filter name doesn't match an existing one

	out_synthetic_photometry = {'syn_flux(Jy)': syn_flux_Jy, 'syn_flux(erg/s/cm2/A)': syn_flux_erg, 'syn_mag': syn_mag, 'lambda_eff(um)': lambda_eff, 
	                            'width_eff(um)': width_eff, 'zero_point(Jy)': zero_point, 'label': label, 'transmission': transmission, 'wl': wl, 'flux': flux, 
	                            'flux_unit': flux_unit, 'filters': filters}
	if (eflux is not None): 
		out_synthetic_photometry['esyn_flux(Jy)'] = esyn_flux_Jy
		out_synthetic_photometry['esyn_flux(erg/s/cm2/A)'] = esyn_flux_erg
		out_synthetic_photometry['esyn_mag'] = esyn_mag
		out_synthetic_photometry['eflux'] = eflux

	return out_synthetic_photometry 
