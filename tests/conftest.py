import pytest
from specutils import Spectrum
from astropy import units as u
from astropy.io import ascii
from astropy.nddata import StdDevUncertainty

@pytest.fixture(scope="session", autouse=True)
def spex_spectrum():
    data_path = 'tests/data/0415-0935_IRTF_SpeX.dat'
    SpeX = ascii.read(data_path)
    flux_unit = u.Unit('erg/s/cm2/A')
    spex_spectrum = Spectrum(spectral_axis = SpeX['wl(um)']*u.micron, 
        flux = SpeX['flux(erg/s/cm2/A)']*flux_unit, 
        uncertainty = StdDevUncertainty(SpeX['eflux(erg/s/cm2/A)']*flux_unit))
    
    return spex_spectrum