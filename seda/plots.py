import matplotlib.pyplot as plt
import importlib
import pickle
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, StrMethodFormatter, NullFormatter
from matplotlib.backends.backend_pdf import PdfPages # plot several pages in a single pdf
from sys import exit
from .utils import *

##########################
def plot_chi2_fit(chi2_pickle_file, N_best_fits=1, ylog=True, out_file=None, save=True):
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
	- ylog : {``True``, ``False``}, optional (default ``True``)
		Use logarithmic (``True``) or linear (``False``) scale to plot fluxes and residuals.
	- out_file : str, optional
		File name to save the figure (it can include a path e.g. my_path/figure.pdf). Note: use a supported format by savefig() such as pdf, ps, eps, png, jpg, or svg.
		Default name is 'SED_{``model``}_chi2.pdf', where ``model`` is read from ``chi2_pickle_file``.
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
	out_best_chi2_fits = best_chi2_fits(chi2_pickle_file=chi2_pickle_file, N_best_fits=N_best_fits)
	spectra_name_best = out_best_chi2_fits['spectra_name_best']
	chi2_red_fit_best = out_best_chi2_fits['chi2_red_fit_best']
	wl_model = out_best_chi2_fits['wl_model']
	flux_model = out_best_chi2_fits['flux_model']
	wl_model_conv = out_best_chi2_fits['wl_model_conv']
	flux_model_conv = out_best_chi2_fits['flux_model_conv']

	# open results from the chi square analysis
	with open(chi2_pickle_file, 'rb') as file:
		out_chi2 = pickle.load(file)
	model = out_chi2['model']
	wl_spectra = out_chi2['wl_array_data'] # it is an array with all spectra, which is different to the list with all spectra in self.wl_spectra
	flux_spectra = out_chi2['flux_array_data']
	eflux_spectra = out_chi2['eflux_array_data']
	wl_array_model_conv_resam = out_chi2['wl_array_model_conv_resam']
	flux_residuals = out_chi2['flux_residuals']
	logflux_residuals = out_chi2['logflux_residuals']
	# sort variables read from the pickle file
	sort_ind = np.argsort(out_chi2['chi2_red_fit'])
	flux_residuals_best = flux_residuals[sort_ind][:N_best_fits,:]
	logflux_residuals_best = logflux_residuals[sort_ind][:N_best_fits,:]

	#------------------------
	# initialize plot for best fits and residuals
	fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1, 0.3], 'hspace': 0.0})

	#------------------------
	# input spectra
	#for i in range(self.N_spectra):
	#	# plot flux uncertainty region
	#	mask = np.where(flux_spectra[i]-eflux_spectra[i]>0.1*flux_spectra[i].min()) # to avoid very small flux-eflux values
	#	wl_region = np.append(wl_spectra[i][mask], np.flip(wl_spectra[i][mask]))
	#	flux_region = np.append(flux_spectra[i][mask]-eflux_spectra[i][mask], np.flip(flux_spectra[i][mask]+eflux_spectra[i][mask]))
	#	ax[0].fill(wl_region, flux_region*(1e4*wl_region), facecolor='black', edgecolor='white', linewidth=1, alpha=0.30, zorder=2) # in erg/s/cm2
	#	# plot observed spectra
	#	ax[0].plot(wl_spectra[i], flux_spectra[i]*(1e4*wl_spectra[i]), color='black', linewidth=1.0, label='Observed spectra') # in erg/s/cm2

	# plot flux uncertainty region
	mask = flux_spectra-eflux_spectra>0.1*flux_spectra.min() # to avoid very small flux-eflux values
	wl_region = np.append(wl_spectra[mask], np.flip(wl_spectra[mask]))
	flux_region = np.append(flux_spectra[mask]-eflux_spectra[mask], np.flip(flux_spectra[mask]+eflux_spectra[mask]))
	ax[0].fill(wl_region, flux_region, facecolor='black', edgecolor='white', linewidth=1, alpha=0.30, zorder=2) # in erg/s/cm2
	# plot observed spectra
	ax[0].plot(wl_spectra, flux_spectra, color='black', linewidth=1.0, label='Observed spectra') # in erg/s/cm2

	# plot best fits
	label_model = spectra_name_short(model=model, spectra_name=spectra_name_best) # short name for spectra to keep only relevant info
	for i in range(N_best_fits):
		label = label_model[i]+r' ($\chi^2_\nu=$'+str(round(chi2_red_fit_best[i],1))+')'
		mask = (wl_model_conv[:,i]>wl_array_model_conv_resam.min()) & (wl_model_conv[:,i]<wl_array_model_conv_resam.max())
		ax[0].plot(wl_model_conv[:,i][mask], flux_model_conv[mask][:,i], '--', linewidth=1.0, label=label) # in erg/s/cm2

	if ylog: ax[0].set_yscale('log')
	ax[0].legend(loc='best', prop={'size': 6}, handlelength=1.5, handletextpad=0.5, labelspacing=0.5) 
	ax[0].xaxis.set_minor_locator(AutoMinorLocator())
	if not ylog:
		ax[0].yaxis.set_minor_locator(AutoMinorLocator())
	ax[0].grid(True, which='both', color='gainsboro', linewidth=0.5, alpha=1.0)
	if (wl_spectra.max()-wl_spectra.min()>10): # use log-scale for wavelength in broad SEDs
		plt.xscale('log')
		ax[0].xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
	ax[0].set_ylabel(r'$F_\lambda\ ($erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)', size=12)
	ax[0].set_title(f'{model_name(model)} Atmospheric Models')

	#------------------------
	# residuals
	for i in range(N_best_fits): # for best fists
		if ylog:
			ax[1].plot(wl_array_model_conv_resam[i,:], logflux_residuals_best[i,:], linewidth=1.0) # in erg/s/cm2
		if not ylog:
			ax[1].plot(wl_array_model_conv_resam[i,:], flux_residuals_best[i,:], linewidth=1.0) # in erg/s/cm2

	ax[1].yaxis.set_minor_locator(AutoMinorLocator())
	ax[1].grid(True, which='both', color='gainsboro', linewidth=0.5, alpha=1.0)
	ax[1].set_xlabel(r'$\lambda\ (\mu$m)', size=12)
	if ylog: ax[1].set_ylabel(r'$\Delta (\log F_\lambda$)', size=12)
	if not ylog: ax[1].set_ylabel(r'$\Delta (F_\lambda$)', size=12)

	if save:
		if out_file is None: plt.savefig('SED_'+out_chi2['model']+'_chi2.pdf', bbox_inches='tight')
		else: plt.savefig(out_file, bbox_inches='tight')
	plt.show()
	plt.close()

	return

##########################
def plot_chi2_red(chi2_pickle_file, N_best_fits=1, out_file=None, save=True):
	'''
	Description:
	------------
		Plot reduced chi square as a function of wavelength for the best model fits form the chi-square minimization.

	Parameters:
	-----------
	chi2_pickle_file : str
		File name with the pickle file with chi2 results.
	N_best_fits: int 
		Number (default 1) of best model fits for plotting.
	out_file : str, optional
		File name to save the figure (it can include a path e.g. my_path/figure.pdf). Note: use file formats (pdf, eps, or ps). Image formats do not work because the figure is saved in several pages, according to ``N_best_fits``.
		Default name is 'SED_{``model``}_chi2.pdf', where ``model`` is read from ``chi2_pickle_file``.
	save : {``True``, ``False``}, optional (default ``True``)
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

	# open results from the chi square analysis
	with open(chi2_pickle_file, 'rb') as file:
		out_chi2 = pickle.load(file)

	model = out_chi2['model']
	spectra_name = out_chi2['spectra_name']
	wl_spectra = out_chi2['wl_array_data']
	chi2_red_fit = out_chi2['chi2_red_fit']
	chi2_red_wl_fit = out_chi2['chi2_red_wl_fit']

	# select the best fits given by N_best_fits
	sort_ind = np.argsort(chi2_red_fit)
	chi2_red_fit_best = chi2_red_fit[sort_ind][:N_best_fits]
	chi2_red_wl_fit_best = chi2_red_wl_fit[sort_ind][:N_best_fits,:]
	spectra_name_best = spectra_name[sort_ind][:N_best_fits]

	if save:
		if out_file is None: pdf_pages = PdfPages('chi2_'+model+'.pdf') # name of the pdf
		else: pdf_pages = PdfPages(out_file) # name of the pdf

	plot_title = spectra_name_short(model=model, spectra_name=spectra_name_best) # short name for spectra to keep only relevant info
	for i in range(N_best_fits): 
		fig, ax = plt.subplots()
	
		ax.plot(wl_spectra, chi2_red_wl_fit_best[i,:], marker='x', markersize=5.0, linestyle='None')
		ax.annotate(r' ($\chi^2_r=$'+str(round(chi2_red_fit_best[i],1))+')', xy=(0.75, .90), xycoords='axes fraction', ha='left', va='bottom', size=12, color='black')	

		if (wl_spectra.max()-wl_spectra.min()>10): # use log-scale for wavelength in broad SEDs
			plt.xscale('log')
			ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))

		ax.xaxis.set_minor_locator(AutoMinorLocator())
		ax.yaxis.set_minor_locator(AutoMinorLocator())
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
def plot_bayes_fit(bayes_pickle_file, ylog=True, out_file=None, save=True):
	'''
	Description:
	------------
		Plot the spectra and best model fit from the Bayesian sampling.

	Parameters:
	-----------
	- bayes_pickle_file : str
		File name with the pickle file with results from the Bayesian sampling.
	- ylog : {``True``, ``False``}, optional (default ``True``)
		Use logarithmic (``True``) or linear (``False``) scale to plot fluxes and residuals.
	- out_file : str, optional
		File name to save the figure (it can include a path e.g. my_path/figure.pdf). Note: use a supported format by savefig() such as pdf, ps, eps, png, jpg, or svg.
		Default name is 'SED_{``model``}_bayes.pdf', where ``model`` is read from ``bayes_pickle_file``.
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
	N_spectra = out_bayes['my_bayes'].N_spectra
	wl_spectra_min = out_bayes['my_bayes'].wl_spectra_min
	wl_spectra_max = out_bayes['my_bayes'].wl_spectra_max
	model = out_bayes['my_bayes'].model

	# read best fit
	out_best_bayesian_fit = best_bayesian_fit(bayes_pickle_file)
	wl_mod_conv_scaled_resam = out_best_bayesian_fit['wl_mod_conv_scaled_resam'] # best fit
	flux_mod_conv_scaled_resam = out_best_bayesian_fit['flux_mod_conv_scaled_resam'] # best fit 
	Teff_med = out_best_bayesian_fit['Teff_med']
	logg_med = out_best_bayesian_fit['logg_med']
	logKzz_med = out_best_bayesian_fit['logKzz_med']
	Z_med = out_best_bayesian_fit['Z_med']
	CtoO_med = out_best_bayesian_fit['CtoO_med']
	R_med = out_best_bayesian_fit['R_med']
	# round parameters
	Teff_med = round(Teff_med)
	logg_med = round(logg_med, 2)
	logKzz_med = round(logKzz_med,1)
	Z_med = round(Z_med, 2)
	CtoO_med = round(CtoO_med, 2)
	R_med = round(R_med, 2)

	# obtain residuals
	logflux_residuals = []
	flux_residuals = []
	for k in range(N_spectra): # for each input observed spectrum
		flux_residuals.append(flux_mod_conv_scaled_resam[k]-flux_spectra[k])
		mask_pos = flux_spectra[k]>0 # mask to avoid negative fluxes to obtain the logarithm
		logflux_residuals.append(np.log10(flux_mod_conv_scaled_resam[k][mask_pos])-np.log10(flux_spectra[k][mask_pos]))

	#------------------------
	# initialize plot for best fits and residuals
	fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1, 0.3], 'hspace': 0.0})

	#------------------------
	# input spectra
	for i in range(N_spectra):
		# plot flux uncertainty region
		mask = flux_spectra[i]-eflux_spectra[i]>0.1*flux_spectra[i].min() # to avoid very small flux-eflux values
		wl_region = np.append(wl_spectra[i][mask], np.flip(wl_spectra[i][mask]))
		flux_region = np.append(flux_spectra[i][mask]-eflux_spectra[i][mask], np.flip(flux_spectra[i][mask]+eflux_spectra[i][mask]))
		ax[0].fill(wl_region, flux_region, facecolor='black', edgecolor='white', linewidth=1, alpha=0.30, zorder=2) # in erg/s/cm2
		# plot observed spectra
		ax[0].plot(wl_spectra[i], flux_spectra[i], label='Observed spectra')

		# plot model spectrum
		label = (f'Teff{Teff_med}_logg{logg_med}_logKzz{logKzz_med}_Z{Z_med}_CtoO{CtoO_med}_R{R_med}')
		ax[0].plot(wl_mod_conv_scaled_resam[i], flux_mod_conv_scaled_resam[i], label=label)

	if ylog: ax[0].set_yscale('log')
	ax[0].legend(loc='best', prop={'size': 6}, handlelength=1.5, handletextpad=0.5, labelspacing=0.5) 
	ax[0].xaxis.set_minor_locator(AutoMinorLocator())
	if not ylog:
		ax[0].yaxis.set_minor_locator(AutoMinorLocator())
	ax[0].grid(True, which='both', color='gainsboro', linewidth=0.5, alpha=1.0)
	if (wl_spectra_max-wl_spectra_min>10): # use log-scale for wavelength in broad SEDs
		plt.xscale('log')
		ax[0].xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
	ax[0].set_ylabel(r'$F_\lambda\ ($erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)', size=12)
	ax[0].set_title(f'{model_name(model)} Atmospheric Models')

	#------------------------
	# residuals
	for i in range(N_spectra):
		if ylog:
			ax[1].plot(wl_spectra[i], logflux_residuals[i], linewidth=1.0) # in erg/s/cm2
		if not ylog:
			ax[1].plot(wl_spectra[i], flux_residuals[i], linewidth=1.0) # in erg/s/cm2

	ax[1].yaxis.set_minor_locator(AutoMinorLocator())
	ax[1].grid(True, which='both', color='gainsboro', linewidth=0.5, alpha=1.0)
	ax[1].set_xlabel(r'$\lambda\ (\mu$m)', size=12)
	if ylog: ax[1].set_ylabel(r'$\Delta (\log F_\lambda$)', size=12)
	if not ylog: ax[1].set_ylabel(r'$\Delta (F_\lambda$)', size=12)

	if save:
		if out_file is None: plt.savefig('SED_'+model+'_bayes.pdf', bbox_inches='tight')
		else: plt.savefig(out_file, bbox_inches='tight')
	plt.show()
	plt.close()

	return

##########################
def plot_model_coverage(model, model_dir, xparam, yparam, xparam_range=None, yparam_range=None, 
	                    xlog=False, ylog=False, out_file=None, save=True):
	'''
	Description:
	------------
		Plot model grid coverage for two desired parameters.

	Parameters:
	-----------
	- model : str
		Atmospheric models. See available models in ``input_parameters.ModelOptions``.  
	- model_dir : str or list, optional
		Path to the directory (str or list) or directories (as a list) containing the model spectra (e.g., ``model_dir = ['path_1', 'path_2']``). 
	- xparam : str
		Parameter in ``model`` to be plotted in the horizontal axis. `input_parameters.ModelOptions`` provides more details about available parameters for ``model``.
	- yparam : str
		Parameter in ``model`` to be plotted in the vertical axis. `input_parameters.ModelOptions`` provides more details about available parameters for ``model``.
	- xparam_range : list or array
		Range in the plot for ``xparam``.
	- yparam_range : list or array
		Range in the plot for ``yparam``.
	- xlog : {``True``, ``False``}, optional (default ``False``)
		Use logarithmic (``True``) or linear (``False``) scale to plot the horizontal axis.
	- ylog : {``True``, ``False``}, optional (default ``False``)
		Use logarithmic (``True``) or linear (``False``) scale to plot the vertical axis.
	- out_file : str, optional
		File name to save the figure (it can include a path e.g. my_path/figure.pdf). Note: use a supported format by savefig() such as pdf, ps, eps, png, jpg, or svg.
		Default name is '{model}_{xparam}_{yparam}.pdf', where ``model`` is read from ``chi2_pickle_file``.
	- save : {``True``, ``False``}, optional (default ``True``)
		Save (``'True'``) or do not save (``'False'``) the resulting figure.

	Returns:
	--------
	Plot of ``yparam`` versus ``xparam`` for ``model``.

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

	# get spectra names from the input directories
	out_select_model_spectra = select_model_spectra(model=model, model_dir=model_dir)
	# separate parameters for the spectra
	out_separate_params = separate_params(model=model, spectra_name=out_select_model_spectra['spectra_name'])

	# make plot of y_param against x_param
	#------------------------
	# initialize plot for best fits and residuals
	fig, ax = plt.subplots()

	plt.scatter(out_separate_params[xparam], out_separate_params[yparam], s=5, zorder=3)

	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	if xparam_range is not None: ax.set_xlim(xparam_range[0], xparam_range[1])
	if yparam_range is not None: ax.set_ylim(yparam_range[0], yparam_range[1])
	if xlog: ax.set_xscale('log')
	if ylog: ax.set_yscale('log')
	plt.grid(True, which='both', color='gainsboro', alpha=0.5)

	plt.xlabel(xparam)
	plt.ylabel(yparam)
	plt.title(f'{model_name(model)} Atmospheric Models')

	if save:
		if out_file is None: plt.savefig(f'{model}_{xparam}_{yparam}.pdf', bbox_inches='tight')
		else: plt.savefig(out_file, bbox_inches='tight')
	plt.show()
	plt.close()

##########################
# get proper model name from the model parameter
def model_name(model):
	
	if model=='Sonora_Diamondback': name = 'Sonora Diamondback'
	if model=='Sonora_Elf_Owl': name = 'Sonora Elf Owl'
	if model=='LB23': name = 'Lacy & Burrows (2023)'
	if model=='Sonora_Cholla': name = 'Sonora Cholla'
	if model=='Sonora_Bobcat': name = 'Sonora Bobcat'
	if model=='ATMO2020': name = 'ATMO 2020'
	if model=='BT-Settl': name = 'BT-Settl'
	if model=='SM08': name = 'Saumon & Marley (2008)'

	return name

##########################
# short model spectrum name from file name
def spectra_name_short(model, spectra_name):
	
	short_name = []
	for spectrum_name in spectra_name:
		if model=='Sonora_Diamondback': short_name.append(spectrum_name[:-5])
		if model=='Sonora_Elf_Owl': short_name.append(spectrum_name[8:-3])
		if model=='Sonora_Cholla': short_name.append(spectrum_name[:-5])
		if model=='LB23': short_name.append(spectrum_name[:-3])
		if model=='ATMO2020': short_name.append(spectrum_name.split('spec_')[1][:-4])
		if model=='BT-Settl': short_name.append(spectrum_name[:-16])
		if model=='SM08': short_name.append(spectrum_name[3:])

	return short_name
