import astropy.units as u
import numpy as np
from seda.models_aux._plugin_helpers import _vac_to_air_uv_safe

def _read_model_spectrum(spectrum_file):
    """ 
    Read a model spectrum and return wavelength wavelength (micron) and flux (erg/s/cm2/A).
    """

    # NOTE: using np.load_txt is MUCH faster, often 50x faster, than using
    # astropy.io.ascii.read(). Like ascii.read(), np.load_txt provides a 
    # skiprows kwarg to allow you to skip over the header. This should 
    # considerably increase the speed of large workflows importing Sonora
    # Diamondback
    spec_model = np.loadtxt(spectrum_file, skiprows=3)
    wl_model = spec_model[:,0] * u.micron # um (in vacuum?)
    wl_model = _vac_to_air_uv_safe(wl_model).value # um in the air
    flux_model = spec_model[:,1] * u.W/u.m**2/u.m # W/m2/m
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
    Z = np.zeros(len(filenames)) * np.nan
    fsed = np.zeros(len(filenames)) * np.nan

    # separate parameters
    for i,name in enumerate(filenames):
        # Teff 
        Teff[i] = float(name.split('g')[0][1:]) # in K
        # logg
        logg[i] = round(np.log10(float(name.split('g')[1].split('_')[0][:-2])),1) + 2 # g in cgs
        # Z
        Z[i] = float(name.split('_')[1][1:])
        # fsed
        f = name.split('_')[0][-1:]
        if f=='c': fsed[i] = 99 # 99 to indicate nc (no clouds)
        if f!='c': fsed[i] = float(f)
        
    # output dictionary with parameters
    out = {'Teff': Teff, 'logg': logg, 'Z': Z, 'fsed': fsed}

    return out
