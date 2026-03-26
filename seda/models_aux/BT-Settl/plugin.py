from astropy.io import ascii
import astropy.units as u
import numpy as np
from seda.models_aux._plugin_helpers import _vac_to_air_uv_safe

def _read_model_spectrum(spectrum_file):
    """ 
    Read a model spectrum and return wavelength wavelength (micron) and flux (erg/s/cm2/A).
    """

    spec_model = ascii.read(spectrum_file, format='no_header')
    
    wl_model = (spec_model['col1']*(u.nm*0.1)).to(u.micron) # um (in vacuum)
    # convert wavelength from vacuum to air
    uv_mask = wl_model < 0.2 * u.um
    nouv_mask = ~uv_mask
    if np.any(uv_mask):
        # only UV wavelengths use Greisen2006
        wl_model[uv_mask] = _vac_to_air_uv_safe(wl_model[uv_mask])
    if np.any(nouv_mask):
        # rest use default (Morton2000)
        wl_model[nouv_mask] = _vac_to_air_uv_safe(wl_model[nouv_mask])
    wl_model = wl_model.value # in um
    flux_model = spec_model['col2'] * u.erg/u.s/u.cm**2/u.Hz # erg/s/cm2/Hz (to an unknown distance). 10**(F_lam + DF) to convert to erg/s/cm2/A
    DF= -8.0
    flux_model = 10**(flux_model.value + DF) # erg/s/cm2/A

    out = {'wl_model': wl_model, 'flux_model': flux_model}

    return out

def _separate_params(filenames):
    """ 
    Parse filenames to extract model parameters as arrays.
    """ 

    # free parameters
    Teff = np.zeros(len(filenames)) * np.nan
    logg = np.zeros(len(filenames)) * np.nan

    # separate parameters
    for i,name in enumerate(filenames):
        # Teff 
        Teff[i] = float(name.split('-')[0][3:]) * 100 # K
        # logg
        logg[i] = float(name.split('-')[1]) # g in cm/s^2

    # output dictionary with parameters
    out = {'Teff': Teff, 'logg': logg}

    return out
