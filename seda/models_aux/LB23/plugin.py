from astropy.io import ascii
import numpy as np

def _read_model_spectrum(spectrum_file):
    """ 
    Read a model spectrum and return wavelength wavelength (micron) and flux (erg/s/cm2/A).
    """
    
    spec_model = ascii.read(spectrum_file)
    
    wl_model = spec_model['LAMBDA(mic)'] # micron
    flux_model = spec_model['FLAM'] # erg/s/cm2/A
    # convert scientific notation from 'D' to 'E'
    wl_LB23 = np.zeros(wl_model.size)
    flux_LB23 = np.zeros(wl_model.size)
    for j in range(wl_LB23.size):
        wl_LB23[j] = float(wl_model[j].replace('D', 'E'))
        flux_LB23[j] = float(flux_model[j].replace('D', 'E'))
    wl_model = wl_LB23 # um
    flux_model = flux_LB23 # erg/s/cm2/A

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
    logKzz = np.zeros(len(filenames)) * np.nan
    Hmix = np.zeros(len(filenames)) * np.nan

    # separate parameters
    for i,name in enumerate(filenames):        
        # Teff 
        Teff[i] = float(name.split('_')[0][1:]) # K
        # logg
        logg[i] = float(name.split('_')[1][1:]) # logg
        # Z (metallicity)
        Z[i] = np.round(np.log10(float(name.split('_')[2][1:])),1)
        # Kzz (radiative zone)
        logKzz[i] = np.log10(float(name.split('CDIFF')[1].split('_')[0])) # in cgs units
        # Hmix
        Hmix[i] = float(name.split('HMIX')[1][:5])
        
    # output dictionary with parameters
    out = {'Teff': Teff, 'logg': logg, 'Z': Z, 'logKzz': logKzz, 'Hmix': Hmix}

    return out
