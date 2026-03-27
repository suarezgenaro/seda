from astropy.io import ascii
import astropy.units as u
import numpy as np
import xarray
from seda.models_aux._plugin_helpers import _round_logg_point25

def _read_model_spectrum(spectrum_file):
    """ 
    Read a model spectrum and return wavelength wavelength (micron) and flux (erg/s/cm2/A).
    """

    spec_model = xarray.open_dataset(spectrum_file) # Sonora Elf Owl model spectra have NetCDF Data Format data
    
    wl_model = spec_model['wavelength'].data * u.micron # um
    wl_model = wl_model.value
    flux_model = spec_model['flux'].data * u.erg/u.s/u.cm**2/u.cm # erg/s/cm2/cm
    flux_model = flux_model.to(u.erg/u.s/u.cm**2/(u.nm*0.1)).value # erg/s/cm2/A

    out = {'wl_model': wl_model, 'flux_model': flux_model}

    return out

def _separate_params(filenames):
    """ 
    Parse filenames to extract model parameters as arrays.
    """ 

    # free parameters
    Teff = np.zeros(len(filenames)) * np.nan
    logg = np.zeros(len(filenames)) * np.nan
    logKzz = np.zeros(len(filenames)) * np.nan
    Z = np.zeros(len(filenames)) * np.nan
    CtoO = np.zeros(len(filenames)) * np.nan

    # separate parameters
    for i,name in enumerate(filenames):
        # Teff 
        Teff[i] = float(name.split('_')[4]) # in K
        # logg
        logg[i] = _round_logg_point25(np.log10(float(name.split('_')[6])) + 2) # g in cgs
        # logKzz
        logKzz[i] = float(name.split('_')[2]) # Kzz in cgs
        # Z
        Z[i] = float(name.split('_')[8]) # in cgs
        # C/O
        CtoO[i] = float(name.split('_')[10][:-3])
        
    # output dictionary with parameters
    out = {'Teff': Teff, 'logg': logg, 'logKzz': logKzz, 'Z': Z, 'CtoO': CtoO}

    return out
