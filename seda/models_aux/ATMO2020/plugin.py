from astropy.io import ascii
import astropy.units as u
import numpy as np
from seda.models_aux._plugin_helpers import _vac_to_air_uv_safe

def _read_model_spectrum(spectrum_file):
    """ 
    Read a model spectrum and return wavelength wavelength (micron) and flux (erg/s/cm2/A).
    """

    spec = ascii.read(spectrum_file, format='no_header')

    wl_model = spec['col1'] * u.micron # um (in vacuum)
    wl_model = _vac_to_air_uv_safe(wl_model).value # um in the air

    flux_model = spec['col2'] * u.W/u.m**2/u.micron # W/m2/micron
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

    # separate parameters
    for i,name in enumerate(filenames):
        s = name.split('spec_')[1]

        # Teff 
        Teff[i] = float(s.split('_')[0][1:])
        # logg
        logg[i] = float(s.split('_')[1][2:])
        # logKzz
        if (s.split('lg')[1][4:-4]=='NEQ_weak'): # when the grid is NEQ_weak
            logKzz[i] = 4 
        elif (s.split('lg')[1][4:-4]=='NEQ_strong'): # when the grid is NEQ_strong
            logKzz[i] = 6 
        else: # equilibrium chemistry grid (CEQ)
            logKzz[i] = 0 

    # output dictionary with parameters
    out = {'Teff': Teff, 'logg': logg, 'logKzz': logKzz}

    return out
