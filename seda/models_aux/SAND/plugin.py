from astropy.io import ascii
from astropy.io import fits
import astropy.units as u
import numpy as np
from seda.models_aux._plugin_helpers import _vac_to_air_uv_safe

def _read_model_spectrum(spectrum_file):
    """ 
    Read a model spectrum and return wavelength wavelength (micron) and flux (erg/s/cm2/A).
    """
    from astropy.io import fits
    spec = fits.open(spectrum_file)

    wl_model = spec[1].data['Wavelength']/10000 * u.micron
    wl_model = _vac_to_air_uv_safe(wl_model).value # um in the air

    flux_model = spec[1].data['Flux density'] # erg/s/cm2/A

    out = {'wl_model': wl_model, 'flux_model': flux_model}

    return out

def _separate_params(filenames):
    """
    Parse SAND filenames to extract model parameters.
    Ignores labels like gold/silver/bronze/inconclusive.
    """

    n = len(filenames)

    # free parameters
    z = np.full(n, np.nan)
    a = np.full(n, np.nan)
    Teff = np.full(n, np.nan)
    logg = np.full(n, np.nan)

    for i, name in enumerate(filenames):
        s = name.replace('SAND_', '').replace('.fits', '')
        parts = s.split('_')
        labels = {'gold', 'silver', 'bronze', 'inconclusive'}
        for p in parts:
            if p in labels:
                continue
            elif p.startswith('z'):
                z[i] = float(p[1:])
            elif p.startswith('a'):
                a[i] = float(p[1:])
            elif p.startswith('t'):
                Teff[i] = float(p[1:])
            elif p.startswith('g'):
                logg[i] = float(p[1:])

    return {
        '[M/H]': z,
        '[a/Fe]': a,
        'Teff': Teff,
        'logg': logg
    }
