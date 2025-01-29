import pickle
import numpy as np
import os
from sys import exit

#!!! plots.model_name(model)
#!!! utils.model_points(model)
#!!! utils.grid_ranges(model)
#!!! utilts.model_filename_pattern(model):
#!!! utilts.available_models():
# 
# plots.spectra_name_short(model, spectra_name)
# utilts.separate_params(model, spectra_name, save_results=False):

# dictionary with basic properties from available atmospheric models
class Models:

	#def __init__(self, model):
	def __init__(self, model=None):

		if model is not None:
			self.model = model
	
		# available atmospheric models
		self.available_models = ['Sonora_Diamondback', 'Sonora_Elf_Owl', 'LB23', 'Sonora_Cholla', 'Sonora_Bobcat', 'ATMO2020', 'BT-Settl', 'SM08']

		# relevant info for available models
		if model=='Sonora_Diamondback':
			self.ref = 'Morley et al (2024)'
			self.name = 'Sonora Diamondback'
			self.bibcode = '2024ApJ...975...59M'
			self.ADS = 'https://ui.adsabs.harvard.edu/abs/2024ApJ...975...59M/abstract'
			self.download = 'https://zenodo.org/records/12735103'
			self.filename_pattern = 't*.spec*'
			self.N_modelpoints = 385466
			self.free_params = ['Teff', 'logg', 'Z', 'fsed']
		if model=='Sonora_Elf_Owl': 
			self.ref = 'Mukherjee et al. (2024)'
			self.name = 'Sonora Elf Owl'
			self.bibcode = '2024ApJ...963...73M'
			self.ADS = 'https://ui.adsabs.harvard.edu/abs/2024ApJ...963...73M/abstract'
			self.download = ['https://zenodo.org/records/10385987', 'https://zenodo.org/records/10385821', 'https://zenodo.org/records/10381250']
			self.filename_pattern = 'spectra_logzz_*.nc*'
			self.N_modelpoints = 193132 
			self.free_params = ['Teff', 'logg', 'logKzz', 'Z', 'CtoO']
		if model=='LB23': 
			self.ref = 'Lacy & Burrows (2023)'
			self.name = 'Lacy & Burrows (2023)'
			self.bibcode = '2023ApJ...950....8L'
			self.ADS = 'https://ui.adsabs.harvard.edu/abs/2023ApJ...950....8L/abstract'
			self.download = 'https://zenodo.org/records/7779180'
			self.filename_pattern = 'T*21*'
			self.N_modelpoints = 30000  
			self.free_params = ['Teff', 'logg', 'Z', 'logKzz', 'Hmix']
		if model=='Sonora_Cholla': 
			self.ref = 'Karalidi et al. (2021)'
			self.name = 'Sonora Cholla'
			self.bibcode = '2021ApJ...923..269K'
			self.ADS = 'https://ui.adsabs.harvard.edu/abs/2021ApJ...923..269K/abstract'
			self.download = 'https://zenodo.org/records/4450269'
			self.filename_pattern = '*.spec*'
			self.N_modelpoints = 110979 
			self.free_params = ['Teff', 'logg', 'logKzz']
		if model=='Sonora_Bobcat': 
			self.ref = 'Marley et al. (2021)'
			self.name = 'Sonora_Bobcat'
			self.bibcode = '2021ApJ...920...85M'
			self.ADS = 'https://ui.adsabs.harvard.edu/abs/2021ApJ...920...85M/abstract'
			self.download = 'https://zenodo.org/records/5063476'
			self.filename_pattern = 'sp_t*'
			self.N_modelpoints = 362000 
			self.free_params = ['Teff', 'logg', 'Z', 'CtoO']
		if model=='ATMO2020': 
			self.ref = 'Phillips et al. (2020)'
			self.name = 'ATMO 2020'
			self.bibcode = '2020A%26A...637A..38P'
			self.ADS = 'https://ui.adsabs.harvard.edu/abs/2020A%26A...637A..38P/abstract'
			self.download = 'https://noctis.erc-atmo.eu/fsdownload/zyU96xA6o/phillips2020'
			self.filename_pattern = 'spec_T*.txt*'
			self.N_modelpoints = 5000   
			self.free_params = ['Teff', 'logg', 'logKzz']
		if model=='BT-Settl': 
			self.ref = 'Allard et al. (2012)'
			self.name = 'BT-Settl'
			self.bibcode = '2012RSPTA.370.2765A'
			self.ADS = 'https://ui.adsabs.harvard.edu/abs/2012RSPTA.370.2765A/abstract'
			self.download = 'http://phoenix.ens-lyon.fr/simulator/'
			self.filename_pattern = 'lte*.BT-Settl.spec.7*'
			self.N_modelpoints = 1291340
			self.free_params = ['Teff', 'logg']
		if model=='SM08': 
			self.ref = 'Saumon & Marley (2008)'
			self.name = 'Saumon & Marley (2008)'
			self.bibcode = '2008ApJ...689.1327S'
			self.ADS = 'https://ui.adsabs.harvard.edu/abs/2008ApJ...689.1327S/abstract'
			self.download = 'Contact authors'
			self.filename_pattern = 'sp_t*'
			self.N_modelpoints = 184663 
			self.free_params = ['Teff', 'logg', 'fsed']

		self.model_ranges()

#		self.model_reference(model)
#		self.model_name(model)
#		self.model_points(model)
#		self.model_filename_pattern(model)
#		self.model_free_params(model)
#	
#	# model references
#	def model_reference(self, model):
#		if self.model=='Sonora_Diamondback': self.ref = 'Morley et al (2024)'
#		if self.model=='Sonora_Elf_Owl': self.ref = 'Mukherjee et al. (2024)'
#		if self.model=='LB23': self.ref = 'Lacy & Burrows (2023)'
#		if self.model=='Sonora_Cholla': self.ref = 'Karalidi et al. (2021)'
#		if self.model=='Sonora_Bobcat': self.ref = 'Marley et al. (2021)'
#		if self.model=='ATMO2020': self.ref = 'Phillips et al. (2020)'
#		if self.model=='BT-Settl': self.ref = 'Allard et al. (2012)'
#		if self.model=='SM08': self.ref = 'Saumon & Marley (2008)'
#
#	# model name
#	def model_name(self, model):
#		if self.model=='Sonora_Diamondback': self.name = 'Sonora Diamondback'
#		if self.model=='Sonora_Elf_Owl': self.name = 'Sonora Elf Owl'
#		if self.model=='LB23': self.name = 'Lacy & Burrows (2023)'
#		if self.model=='Sonora_Cholla': self.name = 'Sonora Cholla'
#		if self.model=='Sonora_Bobcat': self.name = 'Sonora_Bobcat'
#		if self.model=='ATMO2020': self.name = 'ATMO 2020'
#		if self.model=='BT-Settl': self.name = 'BT-Settl'
#		if self.model=='SM08': 'Saumon & Marley (2008)'
#
#	# number of rows in model spectra
#	def model_points(self, model):
#		if self.model=='Sonora_Diamondback': self.N_modelpoints = 385466 # number of rows in model spectra (all spectra have the same length)
#		if self.model=='Sonora_Elf_Owl': self.N_modelpoints = 193132 # number of rows in model spectra (all spectra have the same length)
#		if self.model=='LB23': self.N_modelpoints = 30000 # maximum number of rows in model spectra
#		if self.model=='Sonora_Cholla': self.N_modelpoints = 110979 # maximum number of rows in spectra of the grid
#		if self.model=='Sonora_Bobcat': self.N_modelpoints = 362000 # maximum number of rows in spectra of the grid
#		if self.model=='ATMO2020': self.N_modelpoints = 5000 # maximum number of rows of the ATMO2020 model spectra
#		if self.model=='BT-Settl': self.N_modelpoints = 1291340 # maximum number of rows of the BT-Settl model spectra
#		if self.model=='SM08': self.N_modelpoints = 184663 # rows of the SM08 model spectra
#
#	# common pattern depending the models
#	def model_filename_pattern(self, model):
#		if self.model=='Sonora_Diamondback': self.filename_pattern = 't*.spec*'
#		if self.model=='Sonora_Elf_Owl': self.filename_pattern = 'spectra_logzz_*.nc*'
#		if self.model=='LB23': self.filename_pattern = 'T*21*'
#		if self.model=='Sonora_Cholla': self.filename_pattern = '*.spec*'
#		if self.model=='Sonora_Bobcat': self.filename_pattern = 'sp_t*'
#		if self.model=='ATMO2020': self.filename_pattern = 'spec_T*.txt*'
#		if self.model=='BT-Settl': self.filename_pattern = 'lte*.BT-Settl.spec.7*'
#		if self.model=='SM08': self.filename_pattern = 'sp_t*'
#
#	# free parameters
#	def model_free_params(self, model):
#		if self.model=='Sonora_Diamondback': self.free_params = ['Teff', 'logg', 'Z', 'fsed']
#		if self.model=='Sonora_Elf_Owl': self.free_params = ['Teff', 'logg', 'logKzz', 'Z', 'CtoO']
#		if self.model=='LB23': self.free_params = ['Teff', 'logg', 'Z', 'logKzz', 'Hmix']
#		if self.model=='Sonora_Cholla': self.free_params = ['Teff', 'logg', 'logKzz']
#		if self.model=='Sonora_Bobcat': self.free_params = ['Teff', 'logg', 'Z', 'CtoO']
#		if self.model=='ATMO2020': self.free_params = ['Teff', 'logg', 'logKzz']
#		if self.model=='BT-Settl': self.free_params = ['Teff', 'logg']
#		if self.model=='SM08': self.free_params = ['Teff', 'logg', 'fsed']
		

	# grid_ranges(model)
	def model_ranges(self):

		path_models = os.path.dirname(__file__)+'/'

		with open(f'{path_models}/aux/model_coverage/{self.model}_free_parameters.pickle', 'rb') as file:
			out_coverage = pickle.load(file)

		out = {}
		if 'Teff' in out_coverage: 
			#self.Teff_grid = np.unique(out_coverage['Teff'])
			out['Teff'] = np.unique(out_coverage['Teff'])
		if 'logg' in out_coverage: 
			#self.logg_grid = np.unique(out_coverage['logg'])
			out['logg'] = np.unique(out_coverage['logg'])
		if 'Z' in out_coverage: 
			#self.Z_grid = np.unique(out_coverage['Z'])
			out['Z'] = np.unique(out_coverage['Z'])
		if 'logKzz' in out_coverage: 
			#self.logKzz_grid = np.unique(out_coverage['logKzz'])
			out['logKzz'] = np.unique(out_coverage['logKzz'])
		if 'CtoO' in out_coverage: 
			#self.CtoO_grid = np.unique(out_coverage['CtoO'])
			out['CtoO'] = np.unique(out_coverage['CtoO'])
		if 'fsed' in out_coverage: 
			#self.fsed_grid = np.unique(out_coverage['fsed'])
			out['fsed'] = np.unique(out_coverage['fsed'])
		if 'Hmix' in out_coverage: 
			#self.Hmix_grid = np.unique(out_coverage['Hmix'])
			out['Hmix'] = np.unique(out_coverage['Hmix'])
		self.params = out
