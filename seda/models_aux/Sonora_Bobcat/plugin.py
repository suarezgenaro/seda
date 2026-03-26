from astropy.io import ascii
import astropy.units as u
import numpy as np
from seda.models_aux._plugin_helpers import _vac_to_air_uv_safe
from seda.models_aux._plugin_helpers import _round_logg_point25

def _read_model_spectrum(spectrum_file):
    """ 
    Read a model spectrum and return wavelength wavelength (micron) and flux (erg/s/cm2/A).
    """

    spec_model = ascii.read(spectrum_file, data_start=2, format='no_header')

    wl_model = spec_model['col1'] * u.micron # um (in vacuum?)
    wl_model = _vac_to_air_uv_safe(wl_model).value # um in the air
    flux_model = spec_model['col2'] * u.erg/u.s/u.cm**2/u.Hz # erg/s/cm2/Hz
    flux_model = flux_model.to(u.erg/u.s/u.cm**2/(u.nm*0.1), equivalencies=u.spectral_density( wl_model * u.micron)).value # erg/s/cm2/A

    out = {'wl_model': wl_model, 'flux_model': flux_model}

    return out

def _separate_params(filenames):
    """ 
    Parse filenames to extract model parameters as arrays.
    """ 

    # free parameters
    Teff = np.zeros(len(filenames)) * np.nan
    logg = np.zeros(len(filenames)) * np.nan
    Z = np.zeros(len(filenames)) * np.nan
    CtoO = np.zeros(len(filenames)) * np.nan

    # separate parameters
    for i,name in enumerate(filenames):
        # Teff 
        Teff[i] = float(name.split('_')[1].split('g')[0][1:]) # K
        # logg
        logg[i] = _round_logg_point25(np.log10(float(name.split('_')[1].split('g')[1][:-2])) + 2) # g in cm/s2
        # Z
        Z[i] = float(name.split('_')[2][1:])
        # C/O
        if (len(name.split('_'))==4): # when the spectrum file name includes the C/O
            CtoO[i] = float(name.split('_')[3][2:])
        if (len(name.split('_'))==3): # when the spectrum file name does not include the C/O
            CtoO[i] = 1.0
        
    # output dictionary with parameters
    out = {'Teff': Teff, 'logg': logg, 'Z': Z, 'CtoO': CtoO}

    return out
