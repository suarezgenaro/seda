import matplotlib.pyplot as plt
import importlib
import pickle
import numpy as np
import os
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, StrMethodFormatter, NullFormatter
from matplotlib.backends.backend_pdf import PdfPages # plot several pages in a single pdf
from sys import exit
from .utils import *
from .models import *

##########################
def plot_chi2_fit(chi2_pickle_file, N_best_fits=1, xlog=False, ylog=True, xrange=None, yrange=None, 
	              ori_res=False, model_dir_ori=None, out_file=None, save=True):
	'''
	Description:
	------------
		Plot spectra with the best model fits from the chi-square minimization.

	Parameters:
	-----------
	- chi2_pickle_file : str
		File name with the pickle file with results from the chi-square minimization.
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
	- out_file : str, optional
		File name to save the figure (it can include a path e.g. my_path/figure.pdf). 
		Note: use a supported format by savefig() such as pdf, ps, eps, png, jpg, or svg.
		Default name is 'SED_``model``_chi2.pdf', where ``model`` is read from ``chi2_pickle_file``.
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
	>>> seda.plot_chi2_fit(chi2_pickle_file='ATMO2020_chi2_minimization.pickle', N_best_fits=3)

	Author: Genaro Suárez
	'''

	# read best fits
	out_best_chi2_fits = best_chi2_fits(chi2_pickle_file=chi2_pickle_file, N_best_fits=N_best_fits, model_dir_ori=model_dir_ori, ori_res=ori_res)
	spectra_name_best = out_best_chi2_fits['spectra_name_best']
	chi2_red_fit_best = out_best_chi2_fits['chi2_red_fit_best']
	if ori_res:
		wl_model_best = out_best_chi2_fits['wl_model_best']
		flux_model_best = out_best_chi2_fits['flux_model_best']
	wl_array_model_conv_resam_best = out_best_chi2_fits['wl_array_model_conv_resam_best']
	flux_array_model_conv_resam_best = out_best_chi2_fits['flux_array_model_conv_resam_best']
	wl_array_model_conv_resam_fit_best = out_best_chi2_fits['wl_array_model_conv_resam_fit_best']
	flux_array_model_conv_resam_fit_best = out_best_chi2_fits['flux_array_model_conv_resam_fit_best']
	flux_residuals_best = out_best_chi2_fits['flux_residuals_best']
	logflux_residuals_best = out_best_chi2_fits['logflux_residuals_best']

	# open results from the chi square analysis
	with open(chi2_pickle_file, 'rb') as file:
		out_chi2 = pickle.load(file)
	model = out_chi2['my_chi2'].model
	N_spectra = out_chi2['my_chi2'].N_spectra
	wl_spectra = out_chi2['my_chi2'].wl_spectra
	flux_spectra = out_chi2['my_chi2'].flux_spectra
	eflux_spectra = out_chi2['my_chi2'].eflux_spectra
	wl_spectra_min = out_chi2['my_chi2'].wl_spectra_min 
	wl_spectra_max = out_chi2['my_chi2'].wl_spectra_max 

	#------------------------
	# initialize plot for best fits and residuals
	fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1, 0.3], 'hspace': 0.0})

	# set xrange equal to the input SED range, if not provided
	if xrange is None: xrange = [wl_spectra_min, wl_spectra_max]

	# plot flux uncertainty region
	for k in range(N_spectra): # for each input observed spectrum
		mask = (wl_spectra[k]>=xrange[0]) & (wl_spectra[k]<=xrange[1])
		wl_region = np.append(wl_spectra[k][mask], np.flip(wl_spectra[k][mask]))
		flux_region = np.append(flux_spectra[k][mask]-eflux_spectra[k][mask], 
		                        np.flip(flux_spectra[k][mask]+eflux_spectra[k][mask]))
		ax[0].fill(wl_region, flux_region, facecolor='black', edgecolor='white', linewidth=1, alpha=0.30, zorder=3) # in erg/s/cm2
		# plot observed spectra
		if k==0: ax[0].plot(wl_spectra[k][mask], flux_spectra[k][mask], color='black', linewidth=1.0, label='Observed spectra', zorder=3) # in erg/s/cm2
		else: ax[0].plot(wl_spectra[k][mask], flux_spectra[k][mask], color='black', linewidth=1.0, zorder=3) # in erg/s/cm2

	# plot best fits (original resolution spectra)
	if ori_res: # plot model spectra with the original resolution
		for i in range(N_best_fits):
			mask = (wl_model_best[i,:]>xrange[0]) & (wl_model_best[i,:]<xrange[1])
			if i==0: ax[0].plot(wl_model_best[i,:][mask], flux_model_best[i,:][mask], linewidth=0.5, 
			                    color='silver', label='Models with original resolution', zorder=2, alpha=0.5) # in erg/s/cm2
			else: ax[0].plot(wl_model_best[i,:][mask], flux_model_best[i,:][mask], linewidth=0.5, 
			                 color='silver', zorder=2, alpha=0.5) # in erg/s/cm2

	# plot best fits (convolved spectra)
	label_model = spectra_name_short(model=model, spectra_name=spectra_name_best) # short name for spectra to keep only relevant info
	for i in range(N_best_fits): # for the best fits
		for k in range(N_spectra): # for each input observed spectrum
			label = label_model[i]+r' ($\chi^2_\nu=$'+str(round(chi2_red_fit_best[i],1))+')'
			mask = (wl_array_model_conv_resam_fit_best[i][k]>=xrange[0]) & (wl_array_model_conv_resam_fit_best[i][k]<=xrange[1])
			color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i] # default color
			if k==0: 
				ax[0].plot(wl_array_model_conv_resam_fit_best[i][k][mask], flux_array_model_conv_resam_fit_best[i][k][mask], 
				           '--', color=color, linewidth=1.0, label=label, zorder=4) # in erg/s/cm2
			else: 
				ax[0].plot(wl_array_model_conv_resam_fit_best[i][k][mask], flux_array_model_conv_resam_fit_best[i][k][mask], 
				           '--', color=color, linewidth=1.0, zorder=4) # in erg/s/cm2

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
	ax[0].set_title(f'{Models(model).name} Atmospheric Models')

	#------------------------
	# residuals

	# reference line at zero
	ax[1].plot([xrange[0], xrange[1]], [0, 0], '--', color='black', linewidth=1.)

	for i in range(N_best_fits): # for best fits
		for k in range(N_spectra): # for each input observed spectrum
			color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i] # default color
			mask = (wl_array_model_conv_resam_fit_best[i][k]>=xrange[0]) & (wl_array_model_conv_resam_fit_best[i][k]<=xrange[1])
			if ylog:
				ax[1].plot(wl_array_model_conv_resam_fit_best[i][k][mask], logflux_residuals_best[i][k][mask], linewidth=1.0, color=color) # in erg/s/cm2
			if not ylog:
				ax[1].plot(wl_array_model_conv_resam_fit_best[i][k][mask], flux_residuals_best[i][k][mask], linewidth=1.0, color=color) # in erg/s/cm2

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
def plot_chi2_red(chi2_pickle_file, N_best_fits=1, xlog=False, ylog=False, out_file=None, save=True):
	'''
	Description:
	------------
		Plot reduced chi square as a function of wavelength for the best model fits form the chi-square minimization.

	Parameters:
	-----------
	- chi2_pickle_file : str
		File name with the pickle file with chi2 results.
	- N_best_fits : int 
		Number (default 1) of best model fits for plotting.
	- xlog : {``True``, ``False``}, optional (default ``False``)
		Use logarithmic (``True``) or linear (``False``) scale to plot the horizontal axis.
	- ylog : {``True``, ``False``}, optional (default ``False``)
		Use logarithmic (``True``) or linear (``False``) scale to plot the vertical axis.
	- out_file : str, optional
		File name to save the figure (it can include a path e.g. my_path/figure.pdf). 
		Note: use file formats (pdf, eps, or ps). Image formats do not work because the figure is saved in several pages, according to ``N_best_fits``.
		Default name is 'SED_``model``_chi2.pdf', where ``model`` is read from ``chi2_pickle_file``.
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
	>>> seda.plot_chi2_red(chi2_pickle_file='ATMO2020_chi2_minimization.pickle', N_best_fits=3)

	Author: Genaro Suárez
	'''

	# read best fits
	out_best_chi2_fits = best_chi2_fits(chi2_pickle_file=chi2_pickle_file, N_best_fits=N_best_fits)
	spectra_name_best = out_best_chi2_fits['spectra_name_best']
	chi2_red_fit_best = out_best_chi2_fits['chi2_red_fit_best']
	chi2_red_wl_fit_best = out_best_chi2_fits['chi2_red_wl_fit_best']
	wl_array_model_conv_resam_fit_best = out_best_chi2_fits['wl_array_model_conv_resam_fit_best']

	# open results from the chi square analysis
	with open(chi2_pickle_file, 'rb') as file:
		out_chi2 = pickle.load(file)

	model = out_chi2['my_chi2'].model
	N_spectra = out_chi2['my_chi2'].N_spectra

	if save:
		if out_file is None: pdf_pages = PdfPages('chi2_'+model+'.pdf') # name of the pdf
		else: pdf_pages = PdfPages(out_file) # name of the pdf

	plot_title = spectra_name_short(model=model, spectra_name=spectra_name_best) # short name for spectra to keep only relevant info

	for i in range(N_best_fits): # for best fits
		wl_fit = np.array([]) # initialize numpy array to save array with data points in the fit
		for k in range(N_spectra): # for each input observed spectrum
			wl_fit = np.concatenate([wl_fit, wl_array_model_conv_resam_fit_best[i][k]])

		fig, ax = plt.subplots()
	
		ax.plot(wl_fit, chi2_red_wl_fit_best[i], marker='x', markersize=5.0, linestyle='None')
		ax.annotate(r' ($\chi^2_r=$'+str(round(chi2_red_fit_best[i],1))+')', xy=(0.75, .90), xycoords='axes fraction', ha='left', va='bottom', size=12, color='black')	

		ax.xaxis.set_minor_locator(AutoMinorLocator())
		ax.yaxis.set_minor_locator(AutoMinorLocator())
		if xlog:
			plt.xscale('log')
			ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
		if ylog: ax.set_yscale('log')

		plt.grid(True, which='both', color='gainsboro', alpha=0.5)
		plt.xlabel(r'$\lambda$ ($\mu$m)', size=12)
		plt.ylabel(r'$\chi^2_r$', size=12)
		plt.title(plot_title[i])
	
		if save:
			pdf_pages.savefig(fig, bbox_inches='tight')
		plt.show()
#		plt.cla()

	if save:
		pdf_pages.close()
	plt.close('all')

##########################
def plot_bayes_fit(bayes_pickle_file, xlog=False, ylog=True, xrange=None, yrange=None, 
	               ori_res=False, model_dir_ori=None, out_file=None, save=True):
	'''
	Description:
	------------
		Plot the spectra and best model fit from the Bayesian sampling.

	Parameters:
	-----------
	- bayes_pickle_file : str
		File name with the pickle file with results from the Bayesian sampling.
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
	- out_file : str, optional
		File name to save the figure (it can include a path e.g. my_path/figure.pdf). 
		Note: use a supported format by savefig() such as pdf, ps, eps, png, jpg, or svg.
		Default name is 'SED_``model``_bayes.pdf', where ``model`` is read from ``bayes_pickle_file``.
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
	>>> seda.plot_bayes_fit(bayes_pickle_file='Sonora_Elf_Owl_bayesian_sampling.pickle')

	Author: Genaro Suárez
	'''

	# open results from sampling
	with open(bayes_pickle_file, 'rb') as file:
		out_bayes = pickle.load(file)
	wl_spectra = out_bayes['my_bayes'].wl_spectra # input spectra
	flux_spectra = out_bayes['my_bayes'].flux_spectra # input spectra
	eflux_spectra = out_bayes['my_bayes'].eflux_spectra # input spectra
	wl_spectra_fit = out_bayes['my_bayes'].wl_spectra_fit # input spectra in the fit range
	flux_spectra_fit = out_bayes['my_bayes'].flux_spectra_fit # input spectra in the fit range
	eflux_spectra_fit = out_bayes['my_bayes'].eflux_spectra_fit # input spectra in the fit range
	N_spectra = out_bayes['my_bayes'].N_spectra
	wl_spectra_min = out_bayes['my_bayes'].wl_spectra_min
	wl_spectra_max = out_bayes['my_bayes'].wl_spectra_max
	model = out_bayes['my_bayes'].model
	fit_wl_range = out_bayes['my_bayes'].fit_wl_range

	# read best fit
	out_best_bayesian_fit = best_bayesian_fit(bayes_pickle_file, ori_res=ori_res, model_dir_ori=model_dir_ori)
	# best fit scaled, convolved, and resampled (one for each input spectrum)
	wl_model = out_best_bayesian_fit['wl_model']
	flux_model = out_best_bayesian_fit['flux_model']
	params_med = out_best_bayesian_fit['params_med']
	# best fit with original resolution
	if ori_res:
		wl_model_ori = out_best_bayesian_fit['wl_model_ori']
		flux_model_ori = out_best_bayesian_fit['flux_model_ori']

	# obtain residuals in linear and logarithmic-scale for fluxes in the fit range
	flux_residuals = []
	logflux_residuals = []
	for k in range(N_spectra): # for each input observed spectrum
		# linear scale
		res_lin = flux_model[k]- flux_spectra_fit[k]
		flux_residuals.append(res_lin)
		# log scale
		mask_pos = flux_spectra_fit[k]>0 # mask to avoid negative input fluxes to obtain the logarithm
		res_log = np.log10(flux_model[k]) - np.log10(flux_spectra_fit[k][mask_pos])
		logflux_residuals.append(res_log)

	#------------------------
	# initialize plot for best fits and residuals
	fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1, 0.3], 'hspace': 0.0})

	# set xrange equal to the input SED range, if not provided
	if xrange is None: xrange = [wl_spectra_min, wl_spectra_max]

	# plot input spectra
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

	# plot best fit with original resolution
	if ori_res:
		mask = (wl_model_ori>=xrange[0]) & (wl_model_ori<=xrange[1])
		ax[0].plot(wl_model_ori[mask], flux_model_ori[mask], color='silver', linewidth=0.5, 
		           label='Model with original resolution', zorder=2, alpha=0.5)

	# plot model spectrum
	for k in range(N_spectra): # for each input observed spectrum
		color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1] # default color
		# label
		label = ''
		for i,param in enumerate(params_med): # for each sampled parameter
			if i==0: label += f'{param}{params_med[param]}'
			else: label += f'_{param}{params_med[param]}'
		mask = (wl_model[k]>=xrange[0]) & (wl_model[k]<=xrange[1])
		if k==0: 
			ax[0].plot(wl_model[k][mask], flux_model[k][mask], 
			           color=color, label=label, zorder=4)
		else:
			ax[0].plot(wl_model[k][mask], flux_model[k][mask], 
			           color=color, zorder=4)

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
	ax[0].set_title(f'{Models(model).name} Atmospheric Models')

	#------------------------
	# residuals

	# reference line at zero
	ax[1].plot([xrange[0], xrange[1]], [0, 0], '--', color='black', linewidth=1.)

	for i in range(N_spectra): # for each input observed spectrum
		mask = (wl_spectra_fit[i]>=xrange[0]) & (wl_spectra_fit[i]<=xrange[1])
		if ylog:
			ax[1].plot(wl_spectra_fit[i][mask], logflux_residuals[i][mask], linewidth=1.0, color=color) # in erg/s/cm2
		if not ylog:
			ax[1].plot(wl_spectra_fit[i][mask], flux_residuals[i][mask], linewidth=1.0, color=color) # in erg/s/cm2

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
		Default name is '``model``_``xparam``_``yparam``.pdf', where ``model`` is read from ``chi2_pickle_file``.
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
	plt.title(f'{Models(model).name} Atmospheric Models')

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

	plt.title(f'{Models(model).name} Atmospheric Models')

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
		Default name is 'SED_synthetic_photometry.pdf', where ``model`` is read from ``chi2_pickle_file``.
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

	#------------------------
	# PLOT
	# initialize plot for the SED
	fig, ax = plt.subplots()

	plt.plot(wl_SED, flux_SED, linewidth=1, color='black', label='Hybrid SED')

	if model_label is None: label_mod = f'Model spectrum'
	else: label_mod = model_label
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
