import os
import xarray
import numpy as np
import astropy.units as u
from seda.models_aux._plugin_helpers import _vac_to_air_uv_safe
from seda.models_aux._plugin_helpers import _round_logg_point25

def _read_model_spectrum(spectrum_file):
    """ 
    Read a model spectrum and return wavelength wavelength (micron) and flux (erg/s/cm2/A).
    """

    spec = xarray.open_dataset(spectrum_file, engine = "netcdf4")
    wl_model = spec['wavelength'].data * u.micron # um (in vacuum)
    wl_model = _vac_to_air_uv_safe(wl_model).value # um in the air

    flux_model = spec['flux_emission'].data * u.erg/u.cm**2/u.s/u.cm #erg/cm2/s/cm
    flux_model = flux_model.to(u.erg/u.s/u.cm**2/(u.nm*0.1)).value # erg/s/cm2/A

    out = {'wl_model': wl_model, 'flux_model': flux_model}

    return out

def _separate_params(filenames):

    # Equilibrium models default to logKzz = 0
    logKzz = np.zeros(len(filenames))

    Teff = np.full(len(filenames), np.nan)
    logg = np.full(len(filenames), np.nan)
    logZ = np.full(len(filenames), np.nan)
    CtoO = np.full(len(filenames), np.nan)

    for i, filename in enumerate(filenames):

        try:
            name = os.path.basename(filename)

            # Remove "spectra_" and resolution/file extension
            s = name.split("spectra_")[1]
            s = s.split("_R")[0]

            parts = s.split("_")

            # Filename consists of parameter/value pairs
            params = dict(zip(parts[::2], parts[1::2]))

            Teff[i] = float(params["teff"])
            logg[i] = _round_logg_point25(np.log10(float(params["grav"]))+2) # g in cgs
            logZ[i] = float(params["mh"])
            CtoO[i] = float(params["co"])

            # Only disequilibrium models contain logkzz
            if "logkzz" in params:
                logKzz[i] = float(params["logkzz"])

        except Exception as e:
            print(f"Warning: could not parse {filename}: {e}")

    return {
        "Teff": Teff,
        "logg": logg,
        "[M/H]": logZ,
        "C/O": CtoO,
        "logKzz": logKzz
    }
