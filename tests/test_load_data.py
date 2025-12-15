import astropy.units as u
from seda.input_parameters import InputData

import pytest

@pytest.fixture(scope="session")
def test_load_data(spex_spectrum):
    res = 100
    # distance to the target (optional and used to derive a radius)
    distance = 5.71 * u.pc # pc (parallax=175.2+-1.7; Dupuy-Liu2012)
    edistance = 0.06 * u.pc # pc

    my_data = InputData(wl_spectra=spex_spectrum.spectral_axis.value, flux_spectra=spex_spectrum.flux.value, 
                         eflux_spectra=spex_spectrum.uncertainty.array, flux_unit=spex_spectrum.flux.unit, 
                         res=res, distance=distance.value, edistance=edistance.value)
   
    assert my_data.wl_spectra is not None
    assert my_data.flux_spectra is not None
    assert my_data.eflux_spectra is not None
    assert my_data.flux_unit == spex_spectrum.flux.unit
    assert my_data.res == res
    assert my_data.distance == distance.value
    assert my_data.edistance == edistance.value 

    return my_data





