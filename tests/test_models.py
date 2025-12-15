
from seda.utils import select_model_spectra
import pytest

#@pytest.mark.parametrize(("model", "model_dir"), 
    #[("LB23","models/YDwarfModels/LacyBurrows2023/ClearEQ/"),
    #("LB23","models/YDwarfModels/LacyBurrows2023/ClearNEQ/"),
    #("LB23","models/YDwarfModels/LacyBurrows2023/CloudyEQ/"),
    #("LB23","models/YDwarfModels/LacyBurrows2023/CloudyNEQ/")
    #])
#def test_seperate_parameters(model, model_dir):
#    out = select_model_spectra(model=model, model_dir=model_dir)
#
#    assert out

# we need a test like this
# def test_load_models():
#     # models_list = models.Models().available_models
#     # assert models_list == ['LB23', 'LB23_cleareq', 'SM08', 'Sonora_Bobcat', 'Sonora_Elf_Owl', 'ATMO2020', 'Sonora_Cholla', 'Sonora_Diamondback', 'BT-Settl']

#     model = 'LB23'
#     assert models.Models(model).ref == 'Lacy & Burrows (2023)'
#     assert models.Models(model).ADS == 'https://ui.adsabs.harvard.edu/abs/2023ApJ...950....8L/abstract'
#     assert models.Models(model).download == 'https://zenodo.org/records/7779180'

#     params = models.Models(model).params_unique
#     assert params.keys() == {'Teff', 'logg', 'Z', 'logKzz', 'Hmix'}

#     params_ranges = {
#     'Teff': [500, 700], # Teff range
#     'logg': [4.0, 5.0] # logg range
# }

#     model_dir = 'models/YDwarfModels/LacyBurrows2023/ClearEQ/'  # path to models directory
#     model_conv_dir = 'models/YDwarfModels/LacyBurrows2023/ClearEQ/Convolved_res100/'  # path to convolved models directory

#     # load model options
#     my_model = ModelOptions(model=model, model_dir=model_dir, 
#                              params_ranges=params_ranges, path_save_spectra_conv=model_conv_dir)
#     assert my_model.model == model
#     assert my_model.model_dir[0] == model_dir
#     assert my_model.params_ranges == params_ranges  
#     assert my_model.wl_model is None
#     assert my_model.flux_model is None
#     assert my_model.path_save_spectra_conv == model_conv_dir
#     assert my_model.skip_convolution is False
#     assert my_model.path_save_syn_phot is None
#     assert my_model.skip_syn_phot is False

#     return my_model

