from specutils.utils.wcs_utils import vac_to_air
from pathlib import Path
import re
import inspect

#+++++++++++++++++++++++++++
def _round_logg_point25(logg):
	"""Round logg to the nearest 0.25 step."""
	logg = round(logg*4.) / 4.  
	return logg

#+++++++++++++++++++++++++++
def _vac_to_air_uv_safe(wavelength):
	"""
	Convert vacuum to air wavelengths in the UV,
	supporting specutils API changes.
	"""
	try:
		# New, correct spelling (specutils ≥ newer versions)
		return vac_to_air(wavelength, method="Greisen2006")
	except ValueError:
		# Older specutils fallback
		return vac_to_air(wavelength, method="Griesen2006")

#+++++++++++++++++++++++++++
def export_plugin_function(plugin_path, funcs, header=None):
	"""
	Write functions to a plugin.py file, overwriting any existing file.
	
	Parameters
	----------
	plugin_path : str or Path
		Full path to the plugin.py file.
	funcs : function or list of functions
		One function or a list of functions to write.
	header : str, optional
		Header imports to include at the top.
	"""
	plugin_path = Path(plugin_path)
	
	# ensure funcs is a list
	if not isinstance(funcs, (list, tuple)):
		funcs = [funcs]
	
	# start with header
	content = ''
	if header:
		content += header.strip() + '\n\n'
	
	# append all functions
	for func in funcs:
		func_code = inspect.getsource(func)
		content += func_code.strip() + "\n\n"
	
	# write/overwrite plugin.py
	with open(plugin_path, 'w') as f:
		f.write(content.strip() + '\n')
