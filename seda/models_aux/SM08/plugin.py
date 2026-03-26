from astropy.io import ascii
import astropy.units as u
import numpy as np

def _read_model_spectrum(spectrum_file):
    """ 
    Read a model spectrum and return wavelength wavelength (micron) and flux (erg/s/cm2/A).
    """
    
    spec_model = ascii.read(spectrum_file, data_start=2, format='no_header')
    
    wl_model = spec_model['col1'] * u.micron # um (in air the alkali lines and in vacuum the rest of the spectra)
    #wl_model = _vac_to_air_uv_safe(wl_model).value # um in the air
    wl_model = wl_model.value # um
    flux_model = spec_model['col2'] * u.erg/u.s/u.cm**2/u.Hz # erg/s/cm2/Hz (to an unknown distance)
    flux_model = flux_model.to(u.erg/u.s/u.cm**2/(u.nm*0.1), equivalencies=u.spectral_density(wl_model*u.micron)).value # erg/s/cm2/A

    out = {'wl_model': wl_model, 'flux_model': flux_model}

    return out

def _separate_params(filenames):
    """ 
    Parse filenames to extract model parameters as arrays.
    """ 

    # free parameters
    Teff = np.zeros(len(filenames)) * np.nan
    logg = np.zeros(len(filenames)) * np.nan
    fsed = np.zeros(len(filenames)) * np.nan

    # separate parameters
    for i,name in enumerate(filenames):
        # Teff 
        Teff[i] = float(name.split('_')[1].split('g')[0][1:])
        # logg
        logg[i] = np.round(np.log10(float(name.split('_')[1].split('g')[1].split('f')[0])), 1) + 2 # g in cm/s^2
        # fsed
        fsed[i] = float(name.split('_')[1].split('g')[1].split('f')[1])
        
    # output dictionary with parameters
    out = {'Teff': Teff, 'logg': logg, 'fsed': fsed}

    return out
