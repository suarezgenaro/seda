
from seda.utils import select_model_spectra
import pytest

@pytest.mark.parametrize(("model", "model_dir"), 
    [("LB23_cleareq","models/YDwarfModels/LacyBurrows2023/ClearEQ/"),
    #("LB23","models/YDwarfModels/LacyBurrows2023/ClearNEQ/"),
    #("LB23","models/YDwarfModels/LacyBurrows2023/CloudyEQ/"),
    #("LB23","models/YDwarfModels/LacyBurrows2023/CloudyNEQ/")
    ])
def test_seperate_parameters(model, model_dir):
    out = select_model_spectra(model=model, model_dir=model_dir)

    assert out