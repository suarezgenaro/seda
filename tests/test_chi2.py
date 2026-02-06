import pytest
from seda.input_parameters import Chi2Options
from seda.chi2_fit import chi2

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
@pytest.mark.skip(reason="not working yet")
def test_chi2(test_models_fit):
    out_chi2 = chi2(my_chi2=test_models_fit)

    assert out_chi2.chi2_results is not None
