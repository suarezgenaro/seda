import matplotlib.pyplot as plt
import importlib
import pickle
import numpy as np
import os
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, StrMethodFormatter, NullFormatter
from matplotlib.backends.backend_pdf import PdfPages # plot several pages in a single pdf
from lmfit import Minimizer, minimize, Parameters, report_fit # model fit for non-linear least-squares problems
from sys import exit
from . import utils
from . import models
from . import chi2_fit

##########################
def plot_chi2_fit(output_chi2, N_best_fits=1, xlog=False, ylog=True, xrange=None, yrange=None, 
	              ori_res=False, res=None, lam_res=None, model_dir_ori=None, out_file=None, save=True):
	'''
	Description:
	------------
		Plot spectra with the best model fits from the chi-square minimization.

	Parameters:
	-----------
	- output_chi2 : dictionary or str
		Output dictionary with the results from the chi-square minimization by ``chi2``.
		It can be either the name of the pickle file or simply the output dictionary.
	- N_best_fits : int 
		Number (default 1) of best model fits for plotting.
	- xlog : {``True``, ``False``}, optional (default ``False``)
		Use logarithmic (``True``) or linear (``False``) scale to plot the horizontal axis.
	- ylog : {``True``, ``False``}, optional (default ``True``)
		Use logarithmic (``True``) or linear (``False``) scale to plot the vertical axis.
	- xrange : list or array, optional (default is full range in the input spectra)
		Horizontal range of the plot.
	- yrange : list or array, optional (default is full range in the input spectra)
		Vertical range of the plot.
	- ori_res : {``True``, ``False``}, optional (default ``False``)
		Plot (``True``) or do not plot (``False``) model spectra with the original resolution.
	- model_dir_ori : str, list, or array
		Path to the directory (str, list, or array) or directories (as a list or array) containing the model spectra with the original resolution.
		This parameter is needed to plot the original resolution spectra (if ``ori_res`` is True) when ``seda.chi2`` was run skipping the model spectra convolution (if ``skip_convolution`` is True).
	- res : float, optional
		Spectral resolution (at ``lam_res``) desired to smooth model spectra with original resolution.
		It is needed when only photometry is fit.
	- lam_res : float, optional
		Wavelength of reference at which ``res`` is given.
		Default is the integer closest to the median wavelength of the spectrum.
	- out_file : str, optional
		File name to save the figure (it can include a path e.g. my_path/figure.pdf). 
		Note: use a supported format by savefig() such as pdf, ps, eps, png, jpg, or svg.
		Default name is 'SED_``model``_chi2.pdf', where ``model`` is read from ``output_chi2``.
	- save : {``True``, ``False``}, optional (default ``True``)
		Save (``'True'``) or do not save (``'False'``) the resulting figure.

	Returns:
	--------
	Plot of the spectra and best model fits from the chi-square minimization that will be stored if ``save`` with the name ``out_file``.

	Example:
	--------
	>>> import seda
	>>> 
	>>> # plot and save the best three model fits from the model comparison to the ATMO 2020 models
	>>> # 'ATMO2020_chi2_minimization.pickle' is the output by ``chi2_fit.chi2``.
	>>> seda.plot_chi2_fit(output_chi2='ATMO2020_chi2_minimization.pickle', N_best_fits=3)

	Author: Genaro Suárez

	Date: 2024-10, 2025-09-06
	'''

	# read best fits
	output_best_chi2_fits = utils.best_chi2_fits(output_chi2=output_chi2, N_best_fits=N_best_fits, model_dir_ori=model_dir_ori, ori_res=ori_res)
	fit_spectra = output_best_chi2_fits['fit_spectra']
	fit_photometry = output_best_chi2_fits['fit_photometry']
	spectra_name_best = output_best_chi2_fits['spectra_name_best']
	chi2_red_fit_best = output_best_chi2_fits['chi2_red_fit_best']
	if ori_res:
		wl_model_best = output_best_chi2_fits['wl_model_best']
		flux_model_best = output_best_chi2_fits['flux_model_best']
	if fit_spectra:
		wl_array_model_conv_resam_best = output_best_chi2_fits['wl_array_model_conv_resam_best']
		#flux_array_model_conv_resam_best = output_best_chi2_fits['flux_array_model_conv_resam_best']
		flux_array_model_conv_resam_scaled_best = output_best_chi2_fits['flux_array_model_conv_resam_scaled_best']
		wl_array_model_conv_resam_fit_best = output_best_chi2_fits['wl_array_model_conv_resam_fit_best']
		#flux_array_model_conv_resam_fit_best = output_best_chi2_fits['flux_array_model_conv_resam_fit_best']
		flux_array_model_conv_resam_scaled_fit_best = output_best_chi2_fits['flux_array_model_conv_resam_scaled_fit_best']
		flux_residuals_spec_best = output_best_chi2_fits['flux_residuals_spec_best']
		logflux_residuals_spec_best = output_best_chi2_fits['logflux_residuals_spec_best']
	if fit_photometry:
		#flux_syn_array_model_fit_best = output_best_chi2_fits['flux_syn_array_model_fit_best']
		flux_syn_array_model_scaled_fit_best = output_best_chi2_fits['flux_syn_array_model_scaled_fit_best']
		lambda_eff_array_model_fit_best = output_best_chi2_fits['lambda_eff_array_model_fit_best']
		width_eff_array_model_fit_best = output_best_chi2_fits['width_eff_array_model_fit_best']
		flux_residuals_phot_best = output_best_chi2_fits['flux_residuals_phot_best']
		logflux_residuals_phot_best = output_best_chi2_fits['logflux_residuals_phot_best']

	# open results from the chi square analysis
	try: # if given as a pickle file
		with open(output_chi2, 'rb') as file:
			output_chi2 = pickle.load(file)
	except: # if given as the output of chi2_fit
		pass
	model = output_chi2['my_chi2'].model
	wl_spectra = output_chi2['my_chi2'].wl_spectra
	flux_spectra = output_chi2['my_chi2'].flux_spectra
	eflux_spectra = output_chi2['my_chi2'].eflux_spectra
	phot = output_chi2['my_chi2'].phot
	ephot = output_chi2['my_chi2'].ephot
	filters = output_chi2['my_chi2'].filters
	if fit_spectra: 
		N_spectra = output_chi2['my_chi2'].N_spectra
		wl_spectra_min = output_chi2['my_chi2'].wl_spectra_min 
		wl_spectra_max = output_chi2['my_chi2'].wl_spectra_max 
	if fit_photometry: 
		phot_fit = output_chi2['my_chi2'].phot_fit
		ephot_fit = output_chi2['my_chi2'].ephot_fit
		filters_fit = output_chi2['my_chi2'].filters_fit
		fit_phot_range = output_chi2['my_chi2'].fit_phot_range
		lambda_eff_SVO = output_chi2['my_chi2'].lambda_eff_SVO
		width_eff_SVO = output_chi2['my_chi2'].width_eff_SVO
		lambda_eff_SVO_fit = output_chi2['my_chi2'].lambda_eff_SVO_fit
		width_eff_SVO_fit = output_chi2['my_chi2'].width_eff_SVO_fit

	#------------------------
	# initialize plot for best fits and residuals
	fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1, 0.3], 'hspace': 0.0})

	# set xrange equal to the input SED range, if not provided
	if xrange is None: 
		if fit_spectra: 
			xrange = [wl_spectra_min, wl_spectra_max]
		if fit_photometry: 
			mask_min = lambda_eff_SVO==lambda_eff_SVO.min()
			mask_max = lambda_eff_SVO==lambda_eff_SVO.max()
			xmin = lambda_eff_SVO[mask_min][0]-width_eff_SVO[mask_min][0]/2.
			xmax = lambda_eff_SVO[mask_max][0]+width_eff_SVO[mask_max][0]/2.
			xrange = [0.99*xmin, 1.01*xmax]

	 # short name for spectra to keep only relevant info
	label_model = models.spectra_name_short(model=model, spectra_name=spectra_name_best)

	# plot data
	if fit_spectra: 
		for k in range(N_spectra): # for each input observed spectrum
			# plot flux uncertainty region
			mask = (wl_spectra[k]>=xrange[0]) & (wl_spectra[k]<=xrange[1])
			wl_region = np.append(wl_spectra[k][mask], np.flip(wl_spectra[k][mask]))
			flux_region = np.append(flux_spectra[k][mask]-eflux_spectra[k][mask], 
			                        np.flip(flux_spectra[k][mask]+eflux_spectra[k][mask]))
			ax[0].fill(wl_region, flux_region, facecolor='black', edgecolor='white', linewidth=1, alpha=0.30, zorder=3) # in erg/s/cm2
			# plot observed spectra
			if k==0: ax[0].plot(wl_spectra[k][mask], flux_spectra[k][mask], color='black', linewidth=1.0, label='Observed spectra', zorder=3) # in erg/s/cm2
			else: ax[0].plot(wl_spectra[k][mask], flux_spectra[k][mask], color='black', linewidth=1.0, zorder=3) # in erg/s/cm2
	if fit_photometry: 
		# observed photometry
		# for photometry within the fit range, consider the effective wavelength and effective width from the best fit
		# for photometry outside the fit range, consider the effective wavelength and effective width from SVO

		# photometry out of the fit range and within plot range
		mask_nofit = ~np.isin(filters, filters_fit) & (lambda_eff_SVO>=xrange[0]) & (lambda_eff_SVO<=xrange[1])
		ax[0].errorbar(lambda_eff_SVO[mask_nofit], phot[mask_nofit], 
		               xerr=width_eff_SVO[mask_nofit]/2., yerr=ephot[mask_nofit], 
		               color='silver', fmt='.', markersize=1., capsize=2, elinewidth=1.0, 
		               markeredgewidth=0.5, zorder=3)
		# photometry within the fit range
		mask = (lambda_eff_array_model_fit_best[0]>=xrange[0]) & (lambda_eff_array_model_fit_best[0]<=xrange[1])
		ax[0].errorbar(lambda_eff_array_model_fit_best[0][mask], phot_fit[mask], 
		               xerr=width_eff_array_model_fit_best[0][mask]/2., yerr=ephot_fit[mask], 
		               color='red', fmt='.', markersize=8., capsize=2, elinewidth=1.0, 
		               markeredgewidth=0.5, label='Observed photometry', zorder=5)

	# models
	# synthetic photometry
	if fit_photometry: 
		for i in range(N_best_fits):
			if ori_res: label = '' # so no label is shown
			if not ori_res: label = label_model[i]+r' ($\chi^2_\nu=$'+str(round(chi2_red_fit_best[i],1))+')'
			mask = (lambda_eff_array_model_fit_best[0]>=xrange[0]) & (lambda_eff_array_model_fit_best[0]<=xrange[1])
			# consider the effective wavelength from the best fit
			ax[0].errorbar(lambda_eff_array_model_fit_best[0][mask], flux_syn_array_model_scaled_fit_best[i][mask], 
			               #xerr=width_eff_array_model_fit_best[0][mask]/2.,
			               fmt='.', markersize=8., capsize=2, elinewidth=1.0, 
			               markeredgewidth=0.5, zorder=6)

	# plot best fits (original resolution spectra)
	if ori_res: # plot model spectra with the original resolution
		for i in range(N_best_fits):
			mask = (wl_model_best[i,:]>xrange[0]) & (wl_model_best[i,:]<xrange[1])
			if i==0: ax[0].plot(wl_model_best[i,:][mask], flux_model_best[i,:][mask], linewidth=0.5, 
			                    color='silver', label='Models with original resolution', zorder=2, alpha=0.5) # in erg/s/cm2
			else: ax[0].plot(wl_model_best[i,:][mask], flux_model_best[i,:][mask], linewidth=0.5, 
			                 color='silver', zorder=2, alpha=0.5) # in erg/s/cm2
		if fit_photometry and not fit_spectra:
			# convolve spectra if requested (no convolved spectra are stored when fitting only photometry)
			for i in range(N_best_fits): # for the best fits
				color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i] # default color
				out = utils.convolve_spectrum(wl=wl_model_best[i], flux=flux_model_best[i], res=res, 
				                              lam_res=lam_res, disp_wl_range=xrange, convolve_wl_range=xrange)
				wl_model_best_conv = out['wl_conv']
				flux_model_best_conv = out['flux_conv']
				label = label_model[i]+r' ($\chi^2_\nu=$'+str(round(chi2_red_fit_best[i],1))+')'
				mask = (wl_model_best_conv>xrange[0]) & (wl_model_best_conv<xrange[1])
				ax[0].plot(wl_model_best_conv[mask], flux_model_best_conv[mask],
				           '--', color=color, linewidth=1.0, label=label, zorder=4) # in erg/s/cm2
	
	# plot best fits (convolved spectra)
	if fit_spectra: 
		for i in range(N_best_fits): # for the best fits
			for k in range(N_spectra): # for each input observed spectrum
				label = label_model[i]+r' ($\chi^2_\nu=$'+str(round(chi2_red_fit_best[i],1))+')'
				mask = (wl_array_model_conv_resam_fit_best[i][k]>=xrange[0]) & (wl_array_model_conv_resam_fit_best[i][k]<=xrange[1])
				color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i] # default color
				if k==0: 
					ax[0].plot(wl_array_model_conv_resam_fit_best[i][k][mask], flux_array_model_conv_resam_scaled_fit_best[i][k][mask], 
					           '--', color=color, linewidth=1.0, label=label, zorder=4) # in erg/s/cm2
				else: 
					ax[0].plot(wl_array_model_conv_resam_fit_best[i][k][mask], flux_array_model_conv_resam_scaled_fit_best[i][k][mask], 
					           '--', color=color, linewidth=1.0, zorder=4) # in erg/s/cm2

	ax[0].set_xlim(xrange[0], xrange[1])
	if yrange is not None: ax[0].set_ylim(yrange[0], yrange[1])
	ax[0].xaxis.set_minor_locator(AutoMinorLocator())
	ax[0].yaxis.set_minor_locator(AutoMinorLocator())
	if xlog:
		plt.xscale('log')
		ax[0].xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
	if ylog: ax[0].set_yscale('log')

	plt_leg = ax[0].legend(loc='best', prop={'size': 6}, handlelength=1.5, handletextpad=0.5, labelspacing=0.5) 
	# change the line width for the legend
	for line in plt_leg.get_lines():
		line.set_linewidth(1.0)

	ax[0].grid(True, which='both', color='gainsboro', linewidth=0.5, alpha=1.0)
	ax[0].set_ylabel(r'$F_\lambda\ ($erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)', size=12)
	ax[0].set_title(f'{models.Models(model).name} Atmospheric Models')

	#------------------------
	# residuals

	# reference line at zero
	ax[1].plot([xrange[0], xrange[1]], [0, 0], '--', color='black', linewidth=1.)

	for i in range(N_best_fits): # for best fits
		if fit_spectra:
			for k in range(N_spectra): # for each input observed spectrum
				color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i] # default color
				mask = (wl_array_model_conv_resam_fit_best[i][k]>=xrange[0]) & (wl_array_model_conv_resam_fit_best[i][k]<=xrange[1])
				if ylog:
					ax[1].plot(wl_array_model_conv_resam_fit_best[i][k][mask], logflux_residuals_spec_best[i][k][mask], linewidth=1.0, color=color) # in erg/s/cm2
				if not ylog:
					ax[1].plot(wl_array_model_conv_resam_fit_best[i][k][mask], flux_residuals_spec_best[i][k][mask], linewidth=1.0, color=color) # in erg/s/cm2
		if fit_photometry:
			mask = (lambda_eff_array_model_fit_best[0]>=xrange[0]) & (lambda_eff_array_model_fit_best[0]<=xrange[1])
			if ylog:
				ax[1].scatter(lambda_eff_array_model_fit_best[0][mask], logflux_residuals_phot_best[i][mask], s=15, zorder=3)
			if not ylog:
				ax[1].scatter(lambda_eff_array_model_fit_best[0][mask], flux_residuals_phot_best[i][mask], s=15, zorder=3)

	ax[1].yaxis.set_minor_locator(AutoMinorLocator())
	ax[1].grid(True, which='both', color='gainsboro', linewidth=0.5, alpha=1.0)
	ax[1].set_xlabel(r'$\lambda\ (\mu$m)', size=12)
	if ylog: ax[1].set_ylabel(r'$\Delta (\log F_\lambda$)', size=12)
	if not ylog: ax[1].set_ylabel(r'$\Delta (F_\lambda$)', size=12)

	if save:
		if out_file is None: plt.savefig(f'SED_{model}_chi2.pdf', bbox_inches='tight')
		else: plt.savefig(out_file, bbox_inches='tight')
	plt.show()
	plt.close()

	return

##########################
def plot_chi2_red(output_chi2, N_best_fits=1, xlog=False, ylog=False, out_file=None, save=True):
	'''
	Description:
	------------
		Plot reduced chi square as a function of wavelength for the best model fits form the chi-square minimization.

	Parameters:
	-----------
	- output_chi2 : dictionary or str
		Output dictionary with the results from the chi-square minimization by ``chi2``.
		It can be either the name of the pickle file or simply the output dictionary.
	- N_best_fits : int 
		Number (default 1) of best model fits for plotting.
	- xlog : {``True``, ``False``}, optional (default ``False``)
		Use logarithmic (``True``) or linear (``False``) scale to plot the horizontal axis.
	- ylog : {``True``, ``False``}, optional (default ``False``)
		Use logarithmic (``True``) or linear (``False``) scale to plot the vertical axis.
	- out_file : str, optional
		File name to save the figure (it can include a path e.g. my_path/figure.pdf). 
		Note: use file formats (pdf, eps, or ps). Image formats do not work because the figure is saved in several pages, according to ``N_best_fits``.
		Default name is 'SED_``model``_chi2.pdf', where ``model`` is read from ``output_chi2``.
	- save : {``True``, ``False``}, optional (default ``True``)
		Save (``'True'``) or do not save (``'False'``) the resulting figure.

	Returns:
	--------
	Plot of reduced chi-square against wavelength for the best model fits indicated by ``N_best_fits`` that will be stored if ``save`` with the name ``out_file``.

	Example:
	--------
	>>> import seda
	>>> 
	>>> # plot and save the best three model fits from the model comparison to the ATMO 2020 models
	>>> # 'ATMO2020_chi2_minimization.pickle' is the output by ``chi2_fit.chi2``.
	>>> seda.plot_chi2_red(output_chi2='ATMO2020_chi2_minimization.pickle', N_best_fits=3)

	Author: Genaro Suárez

	Date: 2024-10, 2025-09-07
	'''

	# read best fits
	output_best_chi2_fits = utils.best_chi2_fits(output_chi2=output_chi2, N_best_fits=N_best_fits)
	fit_spectra = output_best_chi2_fits['fit_spectra']
	fit_photometry = output_best_chi2_fits['fit_photometry']
	spectra_name_best = output_best_chi2_fits['spectra_name_best']
	chi2_red_fit_best = output_best_chi2_fits['chi2_red_fit_best']
	chi2_red_wl_fit_best = output_best_chi2_fits['chi2_red_wl_fit_best']
	if fit_spectra: wl_array_model_conv_resam_fit_best = output_best_chi2_fits['wl_array_model_conv_resam_fit_best']
	if fit_photometry: lambda_eff_array_model_fit_best = output_best_chi2_fits['lambda_eff_array_model_fit_best']

	# open results from the chi square analysis
	try: # if given as a pickle file
		with open(output_chi2, 'rb') as file:
			output_chi2 = pickle.load(file)
	except: # if given as the output of chi2_fit
		pass
	model = output_chi2['my_chi2'].model

	if save:
		if out_file is None: pdf_pages = PdfPages('chi2_'+model+'.pdf') # name of the pdf
		else: pdf_pages = PdfPages(out_file) # name of the pdf

	plot_title = models.spectra_name_short(model=model, spectra_name=spectra_name_best) # short name for spectra to keep only relevant info

	for i in range(N_best_fits): # for best fits
		fig, ax = plt.subplots()
	
		label = r' $\chi^2_r=$'+str(round(chi2_red_fit_best[i],1))

		if fit_spectra and not fit_photometry:
			wl_fit = np.array([]) # initialize numpy array to save wavelengths within the fit range of all input spectra as a single array
			for k in range(output_chi2['my_chi2'].N_spectra): # for each input observed spectrum
				wl_fit = np.concatenate([wl_fit, wl_array_model_conv_resam_fit_best[i][k]])
			ax.plot(wl_fit, chi2_red_wl_fit_best[i], marker='x', markersize=5.0, linestyle='None')
			plt.plot(wl_fit[0], chi2_red_wl_fit_best[i][0], label=label)

		if not fit_spectra and fit_photometry:
			ax.plot(lambda_eff_array_model_fit_best[0], chi2_red_wl_fit_best[i], 
			        marker='x', markersize=5.0, linestyle='None', zorder=3)
			plt.plot(lambda_eff_array_model_fit_best[0][0], chi2_red_wl_fit_best[i][0], label=label) # for the label

		if fit_spectra and fit_photometry:
			# data points from spectra
			wl_fit = np.array([]) # initialize numpy array to save wavelengths within the fit range of all input spectra as a single array
			for k in range(output_chi2['my_chi2'].N_spectra): # for each input observed spectrum
				wl_fit = np.concatenate([wl_fit, wl_array_model_conv_resam_fit_best[i][k]])
			plt_spec, = ax.plot(wl_fit, chi2_red_wl_fit_best[i][:len(wl_fit)], marker='x', 
			                   markersize=5.0, linestyle='None', label='Spectra')

			# data points from photometry
			plt_phot, = ax.plot(lambda_eff_array_model_fit_best[0], chi2_red_wl_fit_best[i][len(wl_fit):], 
			                   marker='+', color='red', markersize=5.0, linestyle='None', zorder=3, label='Photometry')

			plt_chi2, = ax.plot(wl_fit[0], chi2_red_wl_fit_best[i][0], label=label) # for the label

		ax.xaxis.set_minor_locator(AutoMinorLocator())
		ax.yaxis.set_minor_locator(AutoMinorLocator())
		if xlog:
			plt.xscale('log')
			ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
		if ylog: ax.set_yscale('log')

		# reference line at zero
		#plt.gca().margins(x=0.05, y=0.05) # to change automatic axis range padding
		xrange = plt.xlim()
		ax.plot(xrange, [0, 0], '--', color='black', linewidth=1., zorder=1)

		plt.xlim(xrange)
		plt.grid(True, which='both', color='gainsboro', alpha=0.5)
		plt.xlabel(r'$\lambda$ ($\mu$m)', size=12)
		plt.ylabel(r'$\chi^2_r$', size=12)
		plt.title(plot_title[i])

		# legend
		if fit_spectra and fit_photometry:
			# Add first legend with reduced chi-square value
			leg1 = ax.legend(handles=[plt_chi2], loc='upper right',handlelength=0, handletextpad=0)

			# Get position of the first legend
			leg1_bbox = leg1.get_window_extent()
			inv = ax.transAxes.inverted()
			leg1_bbox_axes = inv.transform(leg1_bbox)
			
			# Compute new bbox below the first legend
			# You can adjust the vertical offset (dy) as needed
			dx = 0
			dy = -0.05  # shift down
			new_bbox = (leg1_bbox_axes[0, 0] + dx,
			            leg1_bbox_axes[0, 1] + dy,
			            leg1_bbox_axes[1, 0] - leg1_bbox_axes[0, 0],
			            leg1_bbox_axes[1, 1] - leg1_bbox_axes[0, 1])

			# Add second legend with data points, anchored below the first legend
			leg2 = ax.legend(handles=[plt_spec, plt_phot], prop={'size': 10}, 
			                 #bbox_to_anchor=(1.0, -0.10, 0.0, 1.0), bbox_transform=ax.transAxes, 
			                 bbox_to_anchor=(0.92*new_bbox[0], 1.05*new_bbox[1]), loc='upper left',
			                 handlelength=1.5, handletextpad=0.0, labelspacing=0.3, frameon=False)
			# Manually add the first legend back
			ax.add_artist(leg1)
		else: plt.legend(handlelength=0, handletextpad=0)
	
		if save:
			pdf_pages.savefig(fig, bbox_inches='tight')
		plt.show()
#		plt.cla()

	if save:
		pdf_pages.close()
	plt.close('all')

##########################
def plot_bayes_fit(output_bayes, xlog=False, ylog=True, xrange=None, yrange=None, 
	               ori_res=False, res=None, lam_res=None, model_dir_ori=None, 
	               out_file=None, save=True):
	'''
	Description:
	------------
		Plot the spectra and best model fit from the Bayesian sampling.

	Parameters:
	-----------
	- output_bayes : str
		Output dictionary with the results from the nested sampling by ``bayes``.
		It can be either the name of the pickle file or simply the output dictionary.
	- xlog : {``True``, ``False``}, optional (default ``False``)
		Use logarithmic (``True``) or linear (``False``) scale to plot the horizontal axis.
	- ylog : {``True``, ``False``}, optional (default ``True``)
		Use logarithmic (``True``) or linear (``False``) scale to plot the vertical axis.
	- xrange : list or array, optional (default is full range in the input spectra)
		Horizontal range of the plot.
	- yrange : list or array, optional (default is full range in the input spectra)
		Vertical range of the plot.
	- ori_res : {``True``, ``False``}, optional (default ``False``)
		Plot (``True``) or do not plot (``False``) the best model spectrum with its original resolution.
	- model_dir_ori : str, list, or array
		Path to the directory (str, list, or array) or directories (as a list or array) containing the model spectra with the original resolution.
		This parameter is needed to plot the original resolution spectra (if ``ori_res`` is True) when ``seda.chi2`` was run skipping the model spectra convolution (if ``skip_convolution`` is True).
	- res : float, optional
		Spectral resolution (at ``lam_res``) desired to smooth model spectra with original resolution.
		It is needed when only photometry is fit.
	- lam_res : float, optional
		Wavelength of reference at which ``res`` is given.
		Default is the integer closest to the median wavelength of the spectrum.
	- out_file : str, optional
		File name to save the figure (it can include a path e.g. my_path/figure.pdf). 
		Note: use a supported format by savefig() such as pdf, ps, eps, png, jpg, or svg.
		Default name is 'SED_``model``_bayes.pdf', where ``model`` is read from ``output_bayes``.
	- save : {``True``, ``False``}, optional (default ``True``)
		Save (``'True'``) or do not save (``'False'``) the resulting figure.

	Returns:
	--------
	Plot of the spectra and best model fit from the Bayesian sampling that will be stored if ``save`` with the name ``out_file``.

	Example:
	--------
	>>> import seda
	>>> 
	>>> # plot and save the best model fit from the Bayesian sampling to Sonora Elf Owl models
	>>> # 'Sonora_Elf_Owl_bayesian_sampling.pickle' is the output by ``bayes_fit.bayes``.
	>>> seda.plot_bayes_fit(output_bayes='Sonora_Elf_Owl_bayesian_sampling.pickle')

	Author: Genaro Suárez
	'''

	# open results from sampling
	try: # if given as a pickle file
		with open(output_bayes, 'rb') as file:
			output_bayes = pickle.load(file)
	except: # if given as the output of chi2_fit
		pass
	fit_spectra = output_bayes['my_bayes'].fit_spectra
	fit_photometry = output_bayes['my_bayes'].fit_photometry
	model = output_bayes['my_bayes'].model
	wl_spectra = output_bayes['my_bayes'].wl_spectra # input spectra
	flux_spectra = output_bayes['my_bayes'].flux_spectra # input spectra
	eflux_spectra = output_bayes['my_bayes'].eflux_spectra # input spectra
	phot = output_bayes['my_bayes'].phot
	ephot = output_bayes['my_bayes'].ephot
	filters = output_bayes['my_bayes'].filters
	if fit_spectra:
		wl_spectra_fit = output_bayes['my_bayes'].wl_spectra_fit # input spectra in the fit range
		flux_spectra_fit = output_bayes['my_bayes'].flux_spectra_fit # input spectra in the fit range
		eflux_spectra_fit = output_bayes['my_bayes'].eflux_spectra_fit # input spectra in the fit range
		N_spectra = output_bayes['my_bayes'].N_spectra
		wl_spectra_min = output_bayes['my_bayes'].wl_spectra_min
		wl_spectra_max = output_bayes['my_bayes'].wl_spectra_max
		weight_spec_fit = output_bayes['my_bayes'].weight_spec_fit 
	if fit_photometry:
		phot_fit = output_bayes['my_bayes'].phot_fit
		ephot_fit = output_bayes['my_bayes'].ephot_fit
		filters_fit = output_bayes['my_bayes'].filters_fit
		lambda_eff_SVO = output_bayes['my_bayes'].lambda_eff_SVO
		width_eff_SVO = output_bayes['my_bayes'].width_eff_SVO
		lambda_eff_SVO_fit = output_bayes['my_bayes'].lambda_eff_SVO_fit
		width_eff_SVO_fit = output_bayes['my_bayes'].width_eff_SVO_fit
		weight_phot_fit = output_bayes['my_bayes'].weight_phot_fit 

	# read best fit
	output_best_bayesian_fit = utils.best_bayesian_fit(output_bayes, ori_res=ori_res, model_dir_ori=model_dir_ori)
	# best fit scaled, convolved, and resampled (one for each input spectrum)
	wl_model = output_best_bayesian_fit['wl_model']
	flux_model = output_best_bayesian_fit['flux_model']
	params_med = output_best_bayesian_fit['params_med']
	# best fit with original resolution
	if ori_res:
		wl_model_best = output_best_bayesian_fit['wl_model_best']
		flux_model_best = output_best_bayesian_fit['flux_model_best']

	# obtain residuals in linear and logarithmic-scale for fluxes in the fit range
	if fit_spectra:
		flux_residuals_spec = []
		logflux_residuals_spec = []
		for k in range(N_spectra): # for each input observed spectrum
			# linear scale
			res_lin = flux_model[k] - flux_spectra_fit[k]
			flux_residuals_spec.append(res_lin)
			# log scale
			mask_pos = flux_spectra_fit[k]>0 # mask to avoid negative input fluxes to obtain the logarithm
			res_log = np.log10(flux_model[k]) - np.log10(flux_spectra_fit[k][mask_pos])
			logflux_residuals_spec.append(res_log)
	if fit_photometry:
		flux_residuals_phot = []
		logflux_residuals_phot = []
		# linear scale
		res_lin = flux_model[-1] - phot_fit
		# log scale
		mask_pos = phot_fit>0 # mask to avoid negative input fluxes to obtain the logarithm
		res_log = np.log10(flux_model[-1][mask_pos]) - np.log10(phot_fit[mask_pos])
		# nested list with residuals for filters within the fit range for each model spectrum
		flux_residuals_phot.append(res_lin)
		logflux_residuals_phot.append(res_log)

	#------------------------
	# initialize plot for best fits and residuals
	fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1, 0.3], 'hspace': 0.0})

	# set xrange equal to the input SED range, if not provided
	if xrange is None: 
		if fit_spectra and not fit_photometry: 
			xrange = [0.99*wl_spectra_min, 1.01*wl_spectra_max]
		if not fit_spectra and fit_photometry: 
			mask_min = lambda_eff_SVO==lambda_eff_SVO.min()
			mask_max = lambda_eff_SVO==lambda_eff_SVO.max()
			xmin = lambda_eff_SVO[mask_min][0]-width_eff_SVO[mask_min][0]/2.
			xmax = lambda_eff_SVO[mask_max][0]+width_eff_SVO[mask_max][0]/2.
			xrange = [0.99*xmin, 1.01*xmax]
		if fit_spectra and fit_photometry: 
			# from spectra
			xmin_spec = wl_spectra_min
			xmax_spec = wl_spectra_max
			# from photometry
			mask_min = lambda_eff_SVO==lambda_eff_SVO.min()
			mask_max = lambda_eff_SVO==lambda_eff_SVO.max()
			xmin_phot = lambda_eff_SVO[mask_min][0]-width_eff_SVO[mask_min][0]/2.
			xmax_phot = lambda_eff_SVO[mask_max][0]+width_eff_SVO[mask_max][0]/2.

		xrange = [0.99*min(xmin_spec, xmin_phot), 1.01*max(xmax_spec, xmax_phot)]

	# plot input data
	if fit_spectra:
		for k in range(N_spectra): # for each input observed spectrum
			# plot flux uncertainty region
			mask = (wl_spectra[k]>=xrange[0]) & (wl_spectra[k]<=xrange[1])
			wl_region = np.append(wl_spectra[k][mask], np.flip(wl_spectra[k][mask]))
			flux_region = np.append(flux_spectra[k][mask]-eflux_spectra[k][mask], 
			                        np.flip(flux_spectra[k][mask]+eflux_spectra[k][mask]))
			ax[0].fill(wl_region, flux_region, facecolor='black', edgecolor='white', linewidth=1, alpha=0.30, zorder=3) # in erg/s/cm2
			# plot observed spectra
			if k==0: ax[0].plot(wl_spectra[k][mask], flux_spectra[k][mask], color='black', linewidth=1.0, label='Observed spectra', zorder=3) # in erg/s/cm2
			else: ax[0].plot(wl_spectra[k][mask], flux_spectra[k][mask], color='black', linewidth=1.0, zorder=3) # in erg/s/cm2
	if fit_photometry: 
		# observed photometry
		# for photometry within the fit range, consider the effective wavelength and effective width from the best fit
		# for photometry outside the fit range, consider the effective wavelength and effective width from SVO
		# photometry out of the fit range and within plot range
		mask_nofit = ~np.isin(filters, filters_fit) & (lambda_eff_SVO>=xrange[0]) & (lambda_eff_SVO<=xrange[1])
		ax[0].errorbar(lambda_eff_SVO[mask_nofit], phot[mask_nofit], 
		               xerr=width_eff_SVO[mask_nofit]/2., yerr=ephot[mask_nofit], 
		               color='silver', fmt='.', markersize=1., capsize=2, elinewidth=1.0, 
		               markeredgewidth=0.5, zorder=3)
		# photometry within the fit range
		mask = (lambda_eff_SVO_fit>=xrange[0]) & (lambda_eff_SVO_fit<=xrange[1])
		ax[0].errorbar(lambda_eff_SVO_fit[mask], phot_fit[mask], 
		               xerr=width_eff_SVO_fit[mask]/2., yerr=ephot_fit[mask], 
		               color='red', fmt='.', markersize=8., capsize=2, elinewidth=1.0, 
		               markeredgewidth=0.5, label='Observed photometry', zorder=5)

	# models
	# synthetic photometry
	# label for best fit
	label_model = ''
	for i,param in enumerate(params_med): # for each sampled parameter
		if i==0: label_model += f'{param}{params_med[param]}'
		else: label_model += f'_{param}{params_med[param]}'
	# find the reduced chi-square value for the best fit
	if fit_spectra and not fit_photometry:
		data_fit = flux_spectra_fit
		edata_fit = eflux_spectra_fit
		model_fit = flux_model
		weight_fit = weight_spec_fit
	if not fit_spectra and fit_photometry:
		data_fit = [phot_fit]
		edata_fit = [ephot_fit]
		model_fit = flux_model
		weight_fit = [weight_phot_fit]
	if fit_spectra and fit_photometry:
		data_fit = flux_spectra_fit + [phot_fit]
		edata_fit = eflux_spectra_fit + [ephot_fit]
		model_fit = flux_model
		weight_fit = weight_spec_fit + [weight_phot_fit]

	params = Parameters()
	params.add('extinction', value=0, vary=False) # fixed parameter
	params.add('scaling', value=1e-20) # free parameter
	minner = Minimizer(chi2_fit.residuals_for_chi2, params, fcn_args=(data_fit, edata_fit, model_fit, weight_fit))
	out_lmfit = minner.minimize(method='leastsq') # 'leastsq': Levenberg-Marquardt (default)
	chi2_red_fit = out_lmfit.redchi # resulting reduced chi square from lmfit
	
	label_model = label_model+r' ($\chi^2_\nu=$'+str(round(chi2_red_fit,1))+')'

	if fit_photometry: 
		if not ori_res: label = label_model
		if ori_res or fit_spectra: label = '' # so no label is shown
		mask = (wl_model[-1]>=xrange[0]) & (wl_model[-1]<=xrange[1])
		# consider the effective wavelength from the best fit
		ax[0].errorbar(lambda_eff_SVO_fit[mask], flux_model[-1][mask], 
		               fmt='.', markersize=8., capsize=2, elinewidth=1.0, 
		               markeredgewidth=0.5, zorder=6, label=label)

	# plot best fit with original resolution
	if ori_res:
		mask = (wl_model_best>=xrange[0]) & (wl_model_best<=xrange[1])
		ax[0].plot(wl_model_best[mask], flux_model_best[mask], color='silver', linewidth=0.5, 
		           label='Model with original resolution', zorder=2, alpha=0.5)
		if fit_photometry and not fit_spectra:
			# convolve spectra if requested (no convolved spectra are stored when fitting only photometry)
			out = utils.convolve_spectrum(wl=wl_model_best, flux=flux_model_best, res=res, 
			                              lam_res=lam_res, disp_wl_range=xrange, convolve_wl_range=xrange)
			wl_model_best_conv = out['wl_conv']
			flux_model_best_conv = out['flux_conv']
			mask = (wl_model_best_conv>xrange[0]) & (wl_model_best_conv<xrange[1])
			color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0] # default color
			ax[0].plot(wl_model_best_conv[mask], flux_model_best_conv[mask],
			           '--', color=color, linewidth=1.0, label=label_model, zorder=4) # in erg/s/cm2

	# plot best fits (convolved spectra)
	if fit_spectra: 
		# plot model spectrum
		for k in range(N_spectra): # for each input observed spectrum
			color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0] # default color
			# label
			label = ''
			for i,param in enumerate(params_med): # for each sampled parameter
				if i==0: label += f'{param}{params_med[param]}'
				else: label += f'_{param}{params_med[param]}'
			mask = (wl_model[k]>=xrange[0]) & (wl_model[k]<=xrange[1])
			if k==0: 
				ax[0].plot(wl_model[k][mask], flux_model[k][mask], 
				           '--', color=color, linewidth=1.0, label=label_model, zorder=4)
			else:
				ax[0].plot(wl_model[k][mask], flux_model[k][mask], 
				           '--', color=color, linewidth=1.0, zorder=4)

	ax[0].set_xlim(xrange[0], xrange[1])
	if yrange is not None: ax[0].set_ylim(yrange[0], yrange[1])
	ax[0].xaxis.set_minor_locator(AutoMinorLocator())
	ax[0].yaxis.set_minor_locator(AutoMinorLocator())
	if xlog:
		plt.xscale('log')
		ax[0].xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
	if ylog: ax[0].set_yscale('log')

	ax[0].legend(loc='best', prop={'size': 6}, handlelength=1.5, handletextpad=0.5, labelspacing=0.5) 
	ax[0].grid(True, which='both', color='gainsboro', linewidth=0.5, alpha=1.0)
	ax[0].set_ylabel(r'$F_\lambda\ ($erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)', size=12)
	ax[0].set_title(f'{models.Models(model).name} Atmospheric Models')

	#------------------------
	# residuals

	# reference line at zero
	ax[1].plot([xrange[0], xrange[1]], [0, 0], '--', color='black', linewidth=1.)

	if fit_spectra:
		for i in range(N_spectra): # for each input observed spectrum
			mask = (wl_spectra_fit[i]>=xrange[0]) & (wl_spectra_fit[i]<=xrange[1])
			if ylog:
				ax[1].plot(wl_spectra_fit[i][mask], logflux_residuals_spec[i][mask], linewidth=1.0, color=color) # in erg/s/cm2
			if not ylog:
				ax[1].plot(wl_spectra_fit[i][mask], flux_residuals_spec[i][mask], linewidth=1.0, color=color) # in erg/s/cm2
	if fit_photometry:
		mask = (wl_model[-1]>=xrange[0]) & (wl_model[-1]<=xrange[1])
		if ylog:
			ax[1].scatter(wl_model[-1][mask], logflux_residuals_phot[-1][mask], s=15, zorder=3)
		if not ylog:
			ax[1].scatter(wl_model[-1][mask], flux_residuals_phot[-1][mask], s=15, zorder=3)

	ax[1].yaxis.set_minor_locator(AutoMinorLocator())
	ax[1].grid(True, which='both', color='gainsboro', linewidth=0.5, alpha=1.0)
	ax[1].set_xlabel(r'$\lambda\ (\mu$m)', size=12)
	if ylog: ax[1].set_ylabel(r'$\Delta (\log F_\lambda$)', size=12)
	if not ylog: ax[1].set_ylabel(r'$\Delta (F_\lambda$)', size=12)

	if save:
		if out_file is None: plt.savefig(f'SED_{model}_bayes.pdf', bbox_inches='tight')
		else: plt.savefig(out_file, bbox_inches='tight')
	plt.show()
	plt.close()

	return

##########################
def plot_model_coverage(model, xparam, yparam, model_dir=None, params_ranges=None, 
	                    xrange=None, yrange=None, xlog=False, ylog=False, 
	                    out_file=None, save=False):
	'''
	Description:
	------------
		Plot model grid coverage for two desired parameters.

	Parameters:
	-----------
	- model : str
		Atmospheric models. See available models in ``seda.Models().available_models``.  
	- xparam : str
		Parameter in ``model`` to be plotted in the horizontal axis. ``seda.Models('model').params_unique`` provides more details about available parameters for ``model``.
	- yparam : str
		Parameter in ``model`` to be plotted in the vertical axis. ``seda.Models('model').params_unique`` provides more details about available parameters for ``model``.
	- model_dir : str or list, optional
		Path to the directory (str or list) or directories (as a list) containing model spectra (e.g., ``model_dir = ['path_1', 'path_2']``) to display their parameters coverage. 
		If not provided, the code will read pre-saved pickle files the full coverage of ``model``.
	- params_ranges : dictionary, optional
		Minimum and maximum values for any model free parameters to select a model grid subset.
		E.g., ``params_ranges = {'Teff': [1000, 1200], 'logg': [4., 5.]}`` to consider spectra within those Teff and logg ranges.
		If a parameter range is not provided, the full range in ``model_dir`` or the pre-saved pickle files is considered.
	- xrange : list or array, optional (default is full range in the input spectra)
		Horizontal range of the plot.
	- yrange : list or array, optional (default is full range in the input spectra)
		Vertical range of the plot.
	- xlog : {``True``, ``False``}, optional (default ``False``)
		Use logarithmic (``True``) or linear (``False``) scale to plot the horizontal axis.
	- ylog : {``True``, ``False``}, optional (default ``False``)
		Use logarithmic (``True``) or linear (``False``) scale to plot the vertical axis.
	- out_file : str, optional
		File name to save the figure (it can include a path e.g. my_path/figure.pdf). 
		Note: use a supported format by savefig() such as pdf, ps, eps, png, jpg, or svg.
		Default name is '``model``_``xparam``_``yparam``.pdf'.
	- save : {``True``, ``False``}, optional (default ``True``)
		Save (``'True'``) or do not save (``'False'``) the resulting figure.

	Returns:
	--------
	Plot of ``yparam`` versus ``xparam`` for ``model`` that will be stored if ``save`` with the name ``out_file``.

	Example:
	--------
	>>> import seda
	>>> 
	>>> # plot logg vs. Teff for the ATMO 2020 models
	>>> model = 'ATMO2020'
	>>> model_dir = ['my_path/CEQ_spectra/', 
	>>>              'my_path/NEQ_weak_spectra/', 
	>>>              'my_path/NEQ_strong_spectra/']
	>>> seda.plot_chi2_fit(model=model, model_dir=model_dir, xparam='Teff', yparam='logg')

	Author: Genaro Suárez
	'''

	path_plots = os.path.dirname(__file__)

	# verify model is recognized
	available_models = Models().available_models
	if model not in available_models: raise Exception(f'"{model}" models are not recognized. Available models: \n          {available_models}')

	# get the coverage of model free parameters
	if model_dir is not None: # coverage from input model spectra
		# values of free parameters in the spectra
		params = select_model_spectra(model=model, model_dir=model_dir, params_ranges=params_ranges)['params']
	else: # coverage of the full model grid
		# open results from the chi square analysis
		with open(f'{path_plots}/models_aux/model_coverage/{model}_free_parameters.pickle', 'rb') as file:
			params = pickle.load(file)['params']
	
	# verify that xparam and yparam are valid parameters
	if xparam not in params: raise Exception(f'{xparam} is not a free parameter in {model}. Valid parameters: {params.keys()}')
	if yparam not in params: raise Exception(f'{yparam} is not a free parameter in {model}. Valid parameters: {params.keys()}')

	# make plot of y_param against x_param
	#------------------------
	# initialize plot for best fits and residuals
	fig, ax = plt.subplots()

	plt.scatter(params[xparam], params[yparam], s=5, zorder=3)

	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	if xrange is not None: ax.set_xlim(xrange[0], xrange[1])
	if yrange is not None: ax.set_ylim(yrange[0], yrange[1])
	if xlog: 
		ax.set_xscale('log')
		ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
	if ylog: 
		ax.set_yscale('log')
		ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
	plt.grid(True, which='both', color='gainsboro', alpha=0.5)

	plt.xlabel(xparam)
	plt.ylabel(yparam)
	plt.title(f'{models.Models(model).name} Atmospheric Models')

	if save:
		if out_file is None: plt.savefig(f'{model}_{xparam}_{yparam}.pdf', bbox_inches='tight')
		else: plt.savefig(out_file, bbox_inches='tight')
	plt.show()
	plt.close()

##########################
def plot_model_resolution(model, spectra_name_full, xlog=True, ylog=False, xrange=None, yrange=None, 
	                      delta_wl_log=False, resolving_power=True, out_file=None, save=False):
	'''
	Description:
	------------
		Plot model grid spectral resolution or resolving power as a function of wavelength.

	Parameters:
	-----------
	- model : str
		Atmospheric models. See available models in ``seda.Models().available_models``.  
	- spectra_name_full: str, list, or array
		Spectra file names with full path.
	- xlog : {``True``, ``False``}, optional (default ``True``)
		Use logarithmic (``True``) or linear (``False``) scale for wavelength range.
	- ylog : {``True``, ``False``}, optional (default ``False``)
		Use logarithmic (``True``) or linear (``False``) scale for resolution range.
	- xrange : list or array, optional (default is full range in the input spectra)
		Horizontal range of the plot.
	- yrange : list or array, optional (default is full range in the input spectra)
		Vertical range of the plot.
	- delta_wl_log : {``True``, ``False``}, optional (default ``False``)
		Consider wavelength steps in linear (``False``) or logarithmic (``True``) scale.
	- resolving_power : {``True``, ``False``}, optional (default ``True``)
		Calculate resolving power (``True``; R=lambda/Delta(lambda)) or spectral resolution (``False``, Delta(lambda)).
	- save : {``True``, ``False``}, optional (default ``True``)
		Save (``'True'``) or do not save (``'False'``) the resulting figure.
	- out_file : str, optional
		File name to save the figure (it can include a path e.g. my_path/figure.pdf). 
		Note: use a supported format by savefig() such as pdf, ps, eps, png, jpg, or svg.
		Default name is '``model``\_resolution.pdf'.

	Returns:
	--------
	Plot of the resolution of the input model spectra that will be stored if ``save`` with the name ``out_file``.

	Example:
	--------
	>>> import seda
	>>> 
	>>> # plot and save the resolving power of a Sonora Diamondback model spectrum
	>>> model = 'Sonora_Diamondback'
	>>> spectra_name_full = 'my_path/t1000g1000f1_m0.0_co1.0.spec'
	>>> seda.plot_model_resolution(model, spectra_name_full, save=True)

	Author: Genaro Suárez
	'''

	# make sure model_dir is a list
	spectra_name_full = var_to_list(spectra_name_full)

	# read model spectra
	wl_model = []
	flux_model = []
	resolution_model = []
	for i, spectrum_name_full in enumerate(spectra_name_full):
		out_read_model_spectrum = read_model_spectrum(spectrum_name_full=spectrum_name_full, model=model)
		wl = out_read_model_spectrum['wl_model']
		flux = out_read_model_spectrum['flux_model']

		# calculate resolution
		if delta_wl_log: # obtain wavelength step in logarithm
			wl_bin = np.log10(wl[1:]) - np.log10(wl[:-1]) # wavelength dispersion of the spectrum
			wl_bin = np.append(wl_bin, wl_bin[-1]) # add an element equal to the last row to have same data points
		else:
			wl_bin = wl[1:] - wl[:-1] # wavelength dispersion of the spectrum
			wl_bin = np.append(wl_bin, wl_bin[-1]) # add an element equal to the last row to have same data points

		if resolving_power: resolution_model.append(wl / wl_bin)
		else: resolution_model.append(wl_bin)

		# nested list with all model spectra
		wl_model.append(wl)
		flux_model.append(flux)

	# plot resolution as a function of wavelength
	#------------------------
	# initialize plot for best fits and residuals
	fig, ax = plt.subplots()

	out_input_data_stats = input_data_stats(wl_spectra=wl_model)
	wl_spectra_min = out_input_data_stats['wl_spectra_min']
	wl_spectra_max = out_input_data_stats['wl_spectra_max']

	if xrange is None: xrange = [wl_spectra_min, wl_spectra_max]

	for i in range(len(spectra_name_full)):
		mask = (wl_model[i]>=xrange[0]) & (wl_model[i]<=xrange[1])
		ax.scatter(wl_model[i][mask], resolution_model[i][mask], s=5, zorder=3)

	if xrange is not None: ax.set_xlim(xrange[0], xrange[1])
	if yrange is not None: ax.set_ylim(yrange[0], yrange[1])
	ax.minorticks_on() 
	if xlog: 
		ax.set_xscale('log')
		if xrange[0]>0.1: # to avoid when spectra go down to wavelength zero
			ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}')) 
	if ylog: ax.set_yscale('log')
	plt.grid(True, which='both', color='gainsboro', alpha=0.5)

	plt.xlabel(r'$\lambda\ (\mu$m)', size=12)
	if resolving_power: 
		if delta_wl_log: plt.ylabel(r'$R=\lambda/\Delta(\log\lambda)$', size=12)
		else: plt.ylabel(r'$R=\lambda/\Delta\lambda$', size=12)
	else: 
		if delta_wl_log: plt.ylabel(r'$\Delta(\log\lambda)$', size=12)
		else: plt.ylabel(r'$\Delta\lambda$', size=12)

	plt.title(f'{models.Models(model).name} Atmospheric Models')

	if save:
		if out_file is None: plt.savefig(f'{model}_resolution.pdf', bbox_inches='tight')
		else: plt.savefig(out_file, bbox_inches='tight')
	plt.show()
	plt.close()

#########################
def plot_synthetic_photometry(out_synthetic_photometry, xlog=False, ylog=False, out_file=None, save=False):
	'''
	Description:
	------------
		Plot synthetic fluxes and input SED.

	Parameters:
	-----------
	- out_synthetic_photometry : dictionary
		Output dictionary by ``synthetic_photometry``.
	- xlog : {``True``, ``False``}, optional (default ``False``)
		Use logarithmic (``True``) or linear (``False``) scale to plot wavelengths.
	- ylog : {``True``, ``False``}, optional (default ``False``)
		Use logarithmic (``True``) or linear (``False``) scale to plot fluxes.
	- out_file : str, optional
		File name to save the figure (it can include a path e.g. my_path/figure.pdf). 
		Note: use a supported format by savefig() such as pdf, ps, eps, png, jpg, or svg.
		Default name is 'SED_synthetic_photometry.pdf', where ``model`` is read from ``out_synthetic_photometry``.
	- save : {``True``, ``False``}, optional (default ``True``)
		Save (``'True'``) or do not save (``'False'``) the resulting figure.

	Returns:
	--------
	Plot of the synthetic fluxes over the SED used to calculate the fluxes that will be stored if ``save`` with the name ``out_file``.

	Example:
	--------
	>>> import seda
	>>> 
	>>> # plot and save the data with derived synthetic fluxes
	>>> # 'out_synthetic_photometry' is the output of ``synthetic_photometry.synthetic_photometry``
	>>> seda.plot_synthetic_photometry(out_synthetic_photometry=out_synthetic_photometry, save=True)

	Author: Genaro Suárez
	'''

	# extract parameters from the output dictionary by synthetic_photometry
	wl = out_synthetic_photometry['wl'] # input spectrum wavelength
	flux = out_synthetic_photometry['flux'] # input spectrum fluxes
	flux_unit = out_synthetic_photometry['flux_unit'] # units of input spectrum fluxes
	filters = out_synthetic_photometry['filters'] # filters used to calculate synthetic photometry
	eff_wl = out_synthetic_photometry['lambda_eff(um)'] # effective wavelength (um) for all filters
	eff_width = out_synthetic_photometry['width_eff(um)'] # effective width (um) for all filters
	# read synthetic fluxes in the units corresponding to the input spectrum
	if flux_unit=='erg/s/cm2/A':
		flux_syn = out_synthetic_photometry['syn_flux(erg/s/cm2/A)']
		try: eflux_syn = out_synthetic_photometry['esyn_flux(erg/s/cm2/A)']
		except: eflux_syn = np.repeat(0, len(flux_syn))
	if flux_unit=='Jy':
		flux_syn = out_synthetic_photometry['syn_flux(Jy)']
		try: eflux_syn = out_synthetic_photometry['esyn_flux(Jy)'] # synthetic flux errors (Jy) for all filters
		except: eflux_syn = np.repeat(0, len(flux_syn))

	mag_syn = out_synthetic_photometry['syn_mag'] # synthetic magnitude for all filters
	try: emag_syn = out_synthetic_photometry['esyn_mag'] # synthetic magnitude error for all filters
	except: emag_syn = np.repeat(0, len(flux_syn))

	label_syn = out_synthetic_photometry['label'] # label for synthetic photometry
	transmission = out_synthetic_photometry['transmission'] # responses for each filter

	#------------------------
	# initialize plot for best fits and residuals
	fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1, 0.3], 'hspace': 0.0})

	ax[0].plot(wl, flux, color='black', linewidth=1, zorder=3)
	
	for i, filt in enumerate(filters):
	    if label_syn[i]=='complete':
	        ax[0].errorbar(eff_wl[i], flux_syn[i], xerr=eff_width[i]/2., yerr=eflux_syn[i], 
	                       fmt='.', markersize=1., capsize=2,  elinewidth=1.0, markeredgewidth=0.5, 
	                       label=filt, zorder=3)
	    elif label_syn[i]=='incomplete':
	        arrow = 0.05*(flux.max()-flux.min())
	        ax[0].errorbar(eff_wl[i], flux_syn[i], xerr=eff_width[i]/2., yerr=arrow, 
	                       fmt='.', markersize=1., capsize=2,  elinewidth=1.0, markeredgewidth=0.5, 
	                       lolims=1, label=filt, zorder=3)    
	
	ax[0].xaxis.set_minor_locator(AutoMinorLocator())
	ax[0].yaxis.set_minor_locator(AutoMinorLocator())
	if xlog: ax[0].set_xscale('log')
	if ylog: ax[0].set_yscale('log')
	ax[0].grid(True, which='both', color='gainsboro', linewidth=0.5, alpha=1.0)
	ax[0].legend(prop={'size': 8.0})#, handlelength=1.5, handletextpad=0.5, labelspacing=0.5,)
	
	if flux_unit=='erg/s/cm2/A': ax[0].set_ylabel(r'$F_\lambda\ ($erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)', size=12)
	if flux_unit=='Jy': ax[0].set_ylabel(r'$F_\nu$ (Jy)', size=12)
	
	#++++++++++++++++++++++++
	# filters' transmissions
	for i, filt in enumerate(filters):
		if filt in list(transmission.keys()): # only for input filters in SVO
		    ax[1].plot(transmission[filt][0,:], transmission[filt][1,:], linewidth=1.0)
	
	ax[1].xaxis.set_minor_locator(AutoMinorLocator())
	ax[1].yaxis.set_minor_locator(AutoMinorLocator())
	ax[1].grid(True, which='both', color='gainsboro', linewidth=0.5, alpha=1.0)
	
	ax[1].set_xlabel(r'$\lambda\ (\mu$m)', size=12)
	ax[1].set_ylabel('Transmission', size=12)
	
	if save:
		if out_file is None: plt.savefig('SED_synthetic_photometry.pdf', bbox_inches='tight')
		else: plt.savefig(out_file, bbox_inches='tight')
	plt.show()
	plt.close()

	return

#########################
def plot_full_SED(out_bol_lum, xlog=True, ylog=True, xrange=None, yrange=None, 
	              spectra_label=None, model_label=None, out_file=None, save=True):
	'''
	Description:
	------------
		Plot full SED considering observed data completed with a model spectrum.

	Parameters:
	-----------
	- out_bol_lum : dictionary
		Output dictionary from `bol_lum`.
	- xlog : {``True``, ``False``}, optional (default ``True``)
		Use logarithmic (``True``) or linear (``False``) scale to plot the horizontal axis.
	- ylog : {``True``, ``False``}, optional (default ``True``)
		Use logarithmic (``True``) or linear (``False``) scale to plot the vertical axis.
	- xrange : list or array, optional (default is full hybrid SED range)
		Horizontal range of the plot.
	- yrange : list or array, optional (default is full hybrid SED range)
		Vertical range of the plot.
	- spectra_label : list, optional
		List of strings to label each input spectrum in the SED.
		Default is 'Observed spectrum #i'.
	- model_label : str, optional
		String to label the model spectrum in the SED.
		Default is 'Model spectrum'.
	- out_file : str, optional
		File name to save the figure (it can include a path e.g. my_path/figure.pdf). 
		Note: use a supported format by savefig() such as pdf, ps, eps, png, jpg, or svg.
		Default name is 'Hybrid_SED.pdf'.
	- save : {``True``, ``False``}, optional (default ``True``)
		Save (``'True'``) or do not save (``'False'``) the resulting figure.

	Returns:
	--------
	Plot of full SED from observations complemented with a model that will be stored if ``save`` with the name ``out_file``.

	Example:
	--------
	>>> import seda
	>>> 
	>>> # plot and save a hybrid SED using observation and models
	>>> # 'out_bol_lum' is the output of ``bol_lum``
	>>> seda.plot_full_SED(out_bol_lum=out_bol_lum)

	Author: Genaro Suárez

	Date: 2025-06-01
	'''

	# extract relevant parameters
	wl_spectra = out_bol_lum['wl_spectra']
	flux_spectra = out_bol_lum['flux_spectra']
	eflux_spectra = out_bol_lum['eflux_spectra']
	wl_model = out_bol_lum['wl_model']
	flux_model = out_bol_lum['flux_model']
	wl_SED = out_bol_lum['wl_SED']
	flux_SED = out_bol_lum['flux_SED']
	eflux_SED = out_bol_lum['eflux_SED']
	N_spectra = out_bol_lum['N_spectra']
	params = out_bol_lum['params']

	#------------------------
	# PLOT
	# initialize plot for the SED
	fig, ax = plt.subplots()

	plt.plot(wl_SED, flux_SED, linewidth=1, color='black', label='Hybrid SED')

	if model_label is None: label_mod = f'Model spectrum'
	else: label_mod = model_label
	param_label = ''
	for param in params:
		param_label += param+str(params[param])
	label_mod = label_mod+f' ({param_label})'
	plt.plot(wl_model, flux_model, '--', linewidth=0.5, color='silver', label=label_mod)
	for i in range(N_spectra): # for each input spectrum
		if spectra_label is None: label_obs = f'Observed spectrum #{i+1}'
		else: label_obs = spectra_label[i]
		plt.plot(wl_spectra[i], flux_spectra[i], linewidth=1, label=label_obs)

	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	if xrange is not None: ax.set_xlim(xrange[0], xrange[1])
	if yrange is not None: ax.set_ylim(yrange[0], yrange[1])
	if xlog: plt.xscale('log')
	if ylog: plt.yscale('log')
	
	ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
	
	ax.grid(True, which='both', color='gainsboro', linewidth=0.5, alpha=0.5)
	ax.legend()
	
	plt.xlabel(r'$\lambda\ (\mu$m)', size=12)
	plt.ylabel(r'$F_\lambda\ ($erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)', size=12)

	if save:
		if out_file is None: plt.savefig(f'Hybrid_SED.pdf', bbox_inches='tight')
		else: plt.savefig(out_file, bbox_inches='tight')
	plt.show()
	plt.close()

	return
