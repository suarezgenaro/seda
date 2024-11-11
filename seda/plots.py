import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, StrMethodFormatter, NullFormatter
from sys import exit
import importlib
import pickle
import numpy as np
from .utils import *

##########################
# plot SED and best model fits from the chi-square minimization
def plot_chi2_fit(chi2_pickle_file, N_best_fits=1, ylog=True):
	'''
	Parameters
	----------
	chi2_pickle_file : str
		file name with the pickle file with chi2 results
	N_best_fits: int 
		number (default 1) of best model fits for plotting
	ylog: boolen, optional
		use logarithmic (``True``) or linear (``False``) scale plot fluxes and residuals.
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
	#ax[0].fill(wl_region, flux_region*(1e4*wl_region), facecolor='black', edgecolor='white', linewidth=1, alpha=0.30, zorder=2) # in erg/s/cm2
	ax[0].fill(wl_region, flux_region, facecolor='black', edgecolor='white', linewidth=1, alpha=0.30, zorder=2) # in erg/s/cm2
	# plot observed spectra
	#ax[0].plot(wl_spectra, flux_spectra*(1e4*wl_spectra), color='black', linewidth=1.0, label='Observed spectra') # in erg/s/cm2
	ax[0].plot(wl_spectra, flux_spectra, color='black', linewidth=1.0, label='Observed spectra') # in erg/s/cm2

	# plot best fits
	for i in range(N_best_fits):
		if model=='Sonora_Diamondback':
			#label_model = spectra_name_best[i][:-15] # for model files on dropbox via a private link
			label_model = spectra_name_best[i][:-5] # for models on Zenodo
		if model=='Sonora_Elf_Owl':
			label_model = spectra_name_best[i][8:-3]
		if model=='Sonora_Cholla':
			label_model = spectra_name_best[i][:-5]
		if model=='LB23':
			label_model = spectra_name_best[i][:-3]
		if model=='ATMO2020':
			label_model = spectra_name_best[i].split('spec_')[1][:-4]
		if ((model=='ATMO2020_CEQ') | (model=='ATMO2020_NEQ_weak') | (model=='ATMO2020_NEQ_strong')):
			label_model = spectra_name_best[i][5:-4]
		if model=='BT-Settl':
			label_model = spectra_name_best[i][:-16]
		if model=='SM08':
			label_model = spectra_name_best[i][3:]
		label = label_model+r' ($\chi^2_\nu=$'+str(round(chi2_red_fit_best[i],1))+')'
		mask = (wl_model_conv[:,i]>wl_array_model_conv_resam.min()) & (wl_model_conv[:,i]<wl_array_model_conv_resam.max())
		#ax[0].plot(wl_model_conv[:,i][mask], flux_model_conv[mask][:,i]*(1e4*wl_model_conv[mask][:,i]), '--', linewidth=1.0, label=label) # in erg/s/cm2
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
	if model=='Sonora_Diamondback':
		ax[0].set_title('Sonora Diamondback Atmospheric Models')
	if model=='Sonora_Elf_Owl':
		ax[0].set_title('Sonora Elf Owl Atmospheric Models')
	if model=='LB23':
		ax[0].set_title('Lacy & Burrows (2023) Atmospheric Models')
	if model=='Sonora_Cholla':
		ax[0].set_title('Sonora Cholla Atmospheric Models')
	if model=='Sonora_Bobcat':
		ax[0].set_title('Sonora Bobcat Atmospheric Models')
	if model=='ATMO2020':
		ax[0].set_title('ATMO 2020 Atmospheric Models')
	if ((model=='ATMO2020_CEQ') & (model=='ATMO2020_NEQ_weak') & (model=='ATMO2020_NEQ_strong')):
		ax[0].set_title('ATMO 2020 Atmospheric Models ('+model[9:]+')')
	if model=='BT-Settl':
		ax[0].set_title('BT-Settl Atmospheric Models')
	if model=='SM08':
		ax[0].set_title('Saumon & Marley (2008) Atmospheric Models')

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

	plt.savefig('SED_'+out_chi2['model']+'_chi2.pdf', bbox_inches='tight')
	plt.show()
	plt.close()

	return

##########################
# plot for reduced chi square
def plot_chi2_red(chi2_pickle_file, N_best_fits=1):
	'''
	Parameters
	----------
	chi2_pickle_file : str
		file name with the pickle file with chi2 results
	N_best_fits: int 
		number (default 1) of best model fits for plotting
	'''

	from matplotlib.backends.backend_pdf import PdfPages # plot several pages in a single pdf

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

	pdf_pages = PdfPages('chi2_'+model+'.pdf') # name of the pdf

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
		if model=='Sonora_Diamondback':
			plot_title = spectra_name_best[i][:-15]
		if model=='Sonora_Elf_Owl':
			plot_title = spectra_name_best[i][8:-3]
		if model=='Sonora_Cholla':
			plot_title = spectra_name_best[i][:-5]
		if model=='LB23':
			plot_title = spectra_name_best[i][:-3]
		if model=='ATMO2020':
			plot_title = spectra_name_best[i].split('spec_')[1][:-4]
		if ((model=='ATMO2020_CEQ') | (model=='ATMO2020_NEQ_weak') | (model=='ATMO2020_NEQ_strong')):
			plot_title = spectra_name_best[i][5:-4]
		if model=='BT-Settl':
			plot_title = spectra_name_best[i][:-16]
		if model=='SM08':
			plot_title = spectra_name_best[i][3:]
		plt.title(plot_title)
	
		pdf_pages.savefig(fig, bbox_inches='tight')
		plt.show()
#		plt.cla()
	pdf_pages.close()
	plt.close('all')

##########################
# plot SED and best model fit from the Bayesian sampling
def plot_bayes_fit(bayes_pickle_file, ylog=True):
	'''
	Parameters
	----------
	bayes_pickle_file : str
		file name with the pickle file with bayes results
	ylog: boolen, optional
		use logarithmic (``True``) or linear (``False``) scale plot fluxes and residuals.
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
	if model=='Sonora_Diamondback':
		ax[0].set_title('Sonora Diamondback Atmospheric Models')
	if model=='Sonora_Elf_Owl':
		ax[0].set_title('Sonora Elf Owl Atmospheric Models')
	if model=='LB23':
		ax[0].set_title('Lacy & Burrows (2023) Atmospheric Models')
	if model=='Sonora_Cholla':
		ax[0].set_title('Sonora Cholla Atmospheric Models')
	if model=='Sonora_Bobcat':
		ax[0].set_title('Sonora Bobcat Atmospheric Models')
	if model=='ATMO2020':
		ax[0].set_title('ATMO 2020 Atmospheric Models')
	if ((model=='ATMO2020_CEQ') & (model=='ATMO2020_NEQ_weak') & (model=='ATMO2020_NEQ_strong')):
		ax[0].set_title('ATMO 2020 Atmospheric Models ('+model[9:]+')')
	if model=='BT-Settl':
		ax[0].set_title('BT-Settl Atmospheric Models')
	if model=='SM08':
		ax[0].set_title('Saumon & Marley (2008) Atmospheric Models')

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

	plt.savefig('SED_'+model+'_bayes.pdf', bbox_inches='tight')
	plt.show()
	plt.close()

	return
