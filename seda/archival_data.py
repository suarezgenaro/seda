from importlib import resources
from astropy.io import ascii
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sys import exit

class Spectrum:
	"""
	Represents a single spectrum with wavelength, flux, flux error,
	and all metadata columns from the table as direct attributes.

	Provides both attribute-style access (e.g., `spectrum.SName`) 
	and dict-style access (e.g., `spectrum['SName']`).

	Parameters
	----------
	target : str
		Target short name.
	wavelength : array-like
		Wavelength numpy array in microns.
	flux : array-like
		Flux numpy array in Jy.
	eflux : array-like
		Flux uncertainty numpy array in Jy.
	meta_row : astropy.table.Row
		Metadata row from the table; all columns are promoted as attributes.

	Methods
	-------
	plot(show=True, **kwargs)
		Quick plot of the spectrum.

	Examples
	--------
	>>> 
	>>> import seda
	>>> 
	>>> # create an instance of the IRS class
	>>> irs = seda.archival_data.IRS()
	>>> # get spectrum and metadata
	>>> spectrum = irs.get_spectra('0355+1133')[0]
	>>> 
	>>> # extract a few attributes 
	>>> spectrum.wavelength # wavelengths in microns
	>>> spectrum.flux       # fluxes in Jy
	>>> spectrum.Name       # Object name via attribute
	>>> spectrum['Name']    # Object name via dictionary-style
	>>> spectrum.plot()     # plot spectrum

	Author: Genaro Suárez

	Date: 2026-02-18
	"""

	def __init__(self, target, wavelength, flux, eflux, meta_row):
		# core spectrum data
		self.target = target
		self.wavelength = wavelength
		self.flux = flux
		self.eflux = eflux

		# promote all metadata columns as attributes
		for col in meta_row.colnames:
			# meta_row is a astropy.table.row.Row
			setattr(self, col, meta_row[col])

	# dict-like access: spectrum['spt_adop'] works
	def __getitem__(self, key):
		return getattr(self, key)

	# list all attributes as keys
	def keys(self):
		return list(vars(self).keys())

	def plot(self, show=True, **kwargs):
		"""
		Quick plot of the spectrum.

		Parameters
		----------
		show : bool, default True
			Whether to call plt.show() immediately.
		kwargs :
			Additional keyword arguments passed to plt.plot().

		Returns
		-------
		self : Spectrum
			Returns self to allow method chaining.

		Author: Genaro Suárez

		Date: 2026-02-18
		"""

		fig, ax = plt.subplots()

		ax.plot(self.wavelength, self.flux, **kwargs)

		ax.xaxis.set_minor_locator(AutoMinorLocator())
		ax.yaxis.set_minor_locator(AutoMinorLocator())
		ax.grid(True, which='both', color='gainsboro', linewidth=0.5, alpha=0.5)

		ax.set_xlabel('Wavelength ($\mu$m)')
		ax.set_ylabel('Flux (Jy)')
		ax.set_title(f'{self.Name} ({self.SName}; {self.SpTypeopt}/{self.SpTypeir})')

		return fig, ax

class IRS:
	"""
	Interface to access Spitzer IRS archival spectra.

	Use this class to retrieve spectra for one or multiple targets
	from the Spitzer IRS archival dataset. Spectra are available with 
	their original pipeline flux calibration or rescaled to match 
	IRAC Channel 4 and WISE W3 photometry.

	Parameters
	----------
	calibration : str, default 'rescaled'
		Flux calibration of the spectra. Options are:
	
		- 'rescaled' for spectra scaled to correct observed–synthetic IRAC Ch4
		  and WISE W3 photometric offsets (Suárez & Metchev 2022).
		- 'pipeline' for spectra retaining their original pipeline flux calibration.

	Attributes
	----------
	data_path : pathlib.Path
		Path to the observations_aux folder.
	calibration : str
		Selected flux calibration ('pipeline' or 'rescaled').
	spectra_folder : pathlib.Path
		Path to the folder containing spectra with the selected flux calibration.
	table : astropy.table.Table
		Metadata table loaded from Suarez_Metchev_2022_tables.dat.

	Methods
	-------
	get_spectra(targets)
		Retrieve spectra for one or multiple targets.

	Notes
	-----
	The rescaled spectra from Suárez & Metchev (2022) include a global
	scaling factor of 1.06 and additional corrections for objects with
	observed-synthetic IRAC Ch4 and WISE W3 magnitude differences >3 sigma
	(Sections 2.4 and 2.5).
	
	These offsets are likely due to background contamination. Therefore,
	the pipeline-calibrated spectra are recommended, although the rescaled
	spectra remain the default for consistency with the published dataset.
	The pipeline flux calibration has been verified for about half of the 
	spectra using SPHEREx, with a suggested absolute flux uncertainty of 0.06 mag.

	Examples
	--------
	>>> 
	>>> import seda
	>>> 
	>>> # create an IRS instance (see Parameters and Notes for calibration options)
	>>> irs = seda.archival_data.IRS()
	>>> # example targets
	>>> target_name = ['0355+1133', '1821+1414']
	>>> # get spectra and metadata
	>>> spectra = irs.get_spectra(target_name)
	>>> 
	>>> # extract a few attributes for the first spectrum
	>>> spectra[0].wavelength # wavelengths in microns
	>>> spectra[0].flux       # fluxes in Jy
	>>> spectra[0].Name       # Object name via attribute
	>>> spectra[0]['Name']    # Object name via dictionary-style
	>>> spectra[0].plot()     # plot spectrum
	>>> for s in spectra:
	...	 print(s.Name, s.spt_adop)

	Author: Genaro Suárez

	Date: 2026-02-18
	"""

	def __init__(self, calibration='rescaled'):
		# Validate the selected flux calibration
		if calibration not in ('pipeline', 'rescaled'):
			raise ValueError(
				f"Invalid calibration '{calibration}'. "
				"Choose 'pipeline' or 'rescaled'."
			)
	
		self.calibration = calibration

		# Base path to observations_aux
		self.data_path = resources.files('seda.observations_aux')
		# Select the folder according to the flux calibration
		self.spectra_folder = self.data_path / f'spitzer_irs_spectra_{calibration}'
		self.table = self._load_table()

	def _load_table(self):
		"""
		Load the target metadata table using Astropy ASCII reader.

		Returns
		-------
		astropy.table.Table
		"""

		table_path = self.data_path / 'Suarez_Metchev_2022_tables.dat'
		table = ascii.read(table_path)

		return table

	def get_spectra(self, targets):
		"""
		Retrieve spectra for one or multiple targets.

		Parameters
		----------
		targets : str or list of str
			Target names in the format "0000+2554".

		Returns
		-------
		list of Spectrum
			List of Spectrum objects, one per target.
		"""

		# ensure targets is is a list, even for one target
		if isinstance(targets, str):
			targets = [targets]

		spectra_list = []
		for target in targets:
			# table row for each target
			row = self._get_row(target)
			# read spectrum and meta
			spectrum = self._load_spectrum(row)
			# append spectrum and meta to the list
			spectra_list.append(spectrum)

		return spectra_list

	def _get_row(self, target):
		"""
		Look up a single target in the metadata table by 'SName'.

		Returns
		-------
		astropy.table.Row
		"""

		matched = self.table[self.table['SName'] == target]
		if len(matched) == 0:
			raise ValueError(
				f"Target '{target}' not found in table. "
				f"Valid targets: {self.table['SName'].tolist()}"
			)

		return matched[0] # returns a single Row object from the table

	def _load_spectrum(self, row):
		"""
		Load a single IRS spectrum from file and wrap it in a Spectrum object.

		Parameters
		----------
		row : astropy.table.Row
			Metadata row for the target.

		Returns
		-------
		Spectrum
		"""

		target_name = row['SName']
		filename = f'{target_name}_IRS_spectrum.dat'
		file_path = self.spectra_folder / filename

		if not file_path.exists():
			raise FileNotFoundError(f'Spectrum file not found: {file_path}')

		# load spectrum using Astropy ASCII reader
		spectrum_table = ascii.read(file_path)
		wl = spectrum_table['wl(um)'].data # um
		flux = spectrum_table['flux(Jy)'].data # Jy
		eflux = spectrum_table['eflux(Jy)'].data # Jy

		# wrap in Spectrum object
		spectrum = Spectrum(target_name, wl, flux, eflux, row)

		return spectrum
