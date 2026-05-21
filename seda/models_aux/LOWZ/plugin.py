from astropy.io import ascii
import astropy.units as u
import numpy as np
from seda.models_aux._plugin_helpers import _vac_to_air_uv_safe

def _read_model_spectrum(spectrum_file):
    """ 
    Read a model spectrum and return wavelength wavelength (micron) and flux (erg/s/cm2/A).
    """

    spec = np.loadtxt(spectrum_file)

    wl_model = spec[:,0] * u.micron # um (in vacuum)
    wl_model = _vac_to_air_uv_safe(wl_model).value # um in the air

    flux_model = spec[:,1] # erg/s/cm2/A
    out = {'wl_model': wl_model, 'flux_model': flux_model}

    return out

def _separate_params(filenames):

    Teff  = np.full(len(filenames), np.nan)
    logg  = np.full(len(filenames), np.nan)
    logZ  = np.full(len(filenames), np.nan)
    CtoO  = np.full(len(filenames), np.nan)
    logKzz = np.full(len(filenames), np.nan)

    for i, name in enumerate(filenames):
        s = name.split("LOW_Z_BD_GRID_CLEAR_")[1]
        parts = s.split("_")


        Teff[i]  = float(parts[1])
        logg[i]  = float(parts[3])
        logZ[i]  = float(parts[5])
        CtoO[i]  = float(parts[7])
        logKzz[i] = float(parts[9])


    return {'Teff': Teff, 'logg': logg, '[M/H]': logZ, 'C/O': CtoO, 'Kzz': logKzz}
