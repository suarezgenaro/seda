import astropy.units as u
from seda.input_parameters import InputData, ModelOptions, Chi2Options
from seda import models
from seda.chi2_fit import chi2
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

@pytest.fixture(scope="session")
def test_load_models():
    # models_list = models.Models().available_models
    # assert models_list == ['LB23', 'LB23_cleareq', 'SM08', 'Sonora_Bobcat', 'Sonora_Elf_Owl', 'ATMO2020', 'Sonora_Cholla', 'Sonora_Diamondback', 'BT-Settl']

    model = 'LB23_cleareq'
    assert models.Models(model).ref == 'Lacy & Burrows (2023)'
    assert models.Models(model).ADS == 'https://ui.adsabs.harvard.edu/abs/2023ApJ...950....8L/abstract'
    assert models.Models(model).download == 'https://zenodo.org/records/7779180'

    params = models.Models(model).params_unique
    assert params.keys() == {'Teff', 'logg', 'Z', 'logKzz', 'Hmix'}

    params_ranges = {
    'Teff': [500, 700], # Teff range
    'logg': [4.0, 5.0] # logg range
}

    model_dir = 'models/YDwarfModels/LacyBurrows2023/ClearEQ/'  # path to models directory
    model_conv_dir = 'models/YDwarfModels/LacyBurrows2023/ClearEQ/Convolved_res100/'  # path to convolved models directory

    # load model options
    my_model = ModelOptions(model=model, model_dir=model_dir, 
                             params_ranges=params_ranges, path_save_spectra_conv=model_conv_dir)
    assert my_model.model == model
    assert my_model.model_dir[0] == model_dir
    assert my_model.params_ranges == params_ranges  
    assert my_model.wl_model is None
    assert my_model.flux_model is None
    assert my_model.path_save_spectra_conv == model_conv_dir
    assert my_model.skip_convolution is False
    assert my_model.path_save_syn_phot is None
    assert my_model.skip_syn_phot is False

    return my_model

@pytest.fixture(scope="session")
def test_models_fit(test_load_models, test_load_data):
    chi2_pickle_file = 'tests/data/test_chi2_fit_single_spec_LB23.pkl'
    chi2_table_file = 'tests/data/test_chi2_fit_single_spec_LB23.txt'


    my_chi2 = Chi2Options(my_data=test_load_data, my_model=test_load_models,
                           chi2_pickle_file=chi2_pickle_file,
                           chi2_table_file=chi2_table_file)
                           
    assert my_chi2.chi2_pickle_file == chi2_pickle_file
    assert my_chi2.chi2_table_file == chi2_table_file
    assert my_chi2.chi2_results is None

    return my_chi2

@pytest.mark.models
def test_chi2(test_models_fit):
    out_chi2 = chi2(my_chi2=test_models_fit)

    assert out_chi2.chi2_results is not None
