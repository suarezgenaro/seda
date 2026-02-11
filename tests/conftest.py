import os
import seda
from astropy.io import ascii

def load_model_spectra_catalog():
	"""Read catalog of example spectra corresponding to models"""

	# base path to the seda package
	path_seda = os.path.dirname(os.path.dirname(seda.__file__))
	# path to example model spectra
	spectra_dir = 'seda/models_aux/model_spectra'
	# file with model names and corresponding example spectra names
	table_file = os.path.join(path_seda, spectra_dir, 'README')
	table = ascii.read(table_file)

	# make a list of 2-tuples of strings for pytest.mark.parametrize
	catalog = []
	for model in table:
		catalog.append((model[0], os.path.join(path_seda, spectra_dir, model[1])))
	
	return catalog
