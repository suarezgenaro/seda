import importlib
from unittest.mock import patch
import numpy as np
import pytest
import seda

# Import the actual module so patch.object targets its helper function.
synphot_module = importlib.import_module(
    'seda.synthetic_photometry.synthetic_photometry'
)


# Use a flat spectrum so the expected integrated flux is known exactly.
@pytest.fixture
def spectrum():
    wl = np.linspace(1.0, 3.0, 1001)  # micron
    flux = np.full_like(wl, 1e-15)  # erg/s/cm2/A
    eflux = 0.1 * flux
    return wl, flux, eflux


# Build a simple top-hat filter with optional magnitude calibration.
def custom_filter(wl_min=1.5, wl_max=2.5, detector='energy', zero_point=None):
    wl = np.linspace(wl_min, wl_max, 101)
    filt = {'wl': wl, 'transmission': np.ones_like(wl), 'detector': detector}
    if zero_point is not None:
        filt['zero_point'] = zero_point
    return filt


@pytest.fixture
def mock_svo():
    """Provide SVO metadata and transmission without network requests."""
    svo_data = {
        'filterID': np.array(['SVO/energy', 'SVO/photon']),
        'ZeroPoint': np.array([1000.0, 1000.0]),
        'WavelengthEff': np.array([20000.0, 20000.0]),  # Angstrom
        'WidthEff': np.array([10000.0, 10000.0]),  # Angstrom
        'DetectorType': np.array([0, 1]),
    }

    def load_transmission(name):
        if name not in svo_data['filterID']:
            raise ValueError(name)
        wl = np.linspace(1.5, 2.5, 101)
        return wl, np.ones_like(wl)

    # Mock both SVO metadata and transmission downloads to keep tests offline.
    with patch('seda.utils.read_SVO_table', return_value=svo_data) as read_svo, \
         patch.object(synphot_module, 'load_filter_transmission',
                      side_effect=load_transmission) as read_curve:
        yield svo_data, read_svo, read_curve


def run_photometry(spectrum, **kwargs):
    wl, flux, eflux = spectrum
    return seda.synthetic_photometry.synthetic_photometry(
        wl=wl, flux=flux, eflux=eflux, flux_unit='erg/s/cm2/A', **kwargs
    )


# Regression check for the original SVO-only workflow and auto detector selection.
def test_svo_only(spectrum, mock_svo):
    result = run_photometry(spectrum, filters=['SVO/energy', 'SVO/photon'])
    assert result['filters'] == ['SVO/energy', 'SVO/photon']
    assert list(result['detector']) == ['energy', 'photon']
    assert list(result['label']) == ['complete', 'complete']
    np.testing.assert_allclose(result['syn_flux(erg/s/cm2/A)'], 1e-15, rtol=1e-12)
    assert np.all(np.isfinite(result['syn_mag']))
    assert np.all(np.isfinite(result['lambda_eff_SVO(um)']))
    assert mock_svo[2].call_count == 2


# Custom filters should return fluxes even without a magnitude zero point.
def test_custom_only(spectrum, mock_svo):
    result = run_photometry(spectrum, custom_filters={'custom': custom_filter()})
    assert result['filters'] == ['custom']
    assert result['label'][0] == 'complete'
    assert result['detector'][0] == 'energy'
    assert result['syn_flux(erg/s/cm2/A)'][0] == pytest.approx(1e-15)
    assert result['coverage_perc'][0] == pytest.approx(100)
    assert np.isnan(result['syn_mag'][0])
    assert np.isnan(result['zero_point(Jy)'][0])
    assert np.isnan(result['lambda_eff_SVO(um)'][0])
    assert result['transmission']['custom'].shape == (2, 101)
    mock_svo[2].assert_not_called()


def test_custom_only_does_not_request_svo(spectrum):
    """Custom-only operation should not require network access or SVO metadata."""
    with patch('seda.utils.read_SVO_table', side_effect=AssertionError('Unexpected SVO access')):
        result = run_photometry(spectrum, custom_filters={'custom': custom_filter()})
    assert result['label'][0] == 'complete'


# SVO filters should precede custom filters in the returned filter order.
def test_mixed_filters(spectrum, mock_svo):
    result = run_photometry(
        spectrum, filters=['SVO/photon'],
        custom_filters={'custom': custom_filter(detector='energy', zero_point=1000.0)},
    )
    assert result['filters'] == ['SVO/photon', 'custom']
    assert list(result['detector']) == ['photon', 'energy']
    np.testing.assert_allclose(result['syn_flux(erg/s/cm2/A)'], 1e-15, rtol=1e-12)
    assert np.all(np.isfinite(result['syn_mag']))
    assert mock_svo[2].call_count == 1


# A flat F_lambda spectrum must give the same flux for both detector conventions.
@pytest.mark.parametrize('detector', ['energy', 'photon'])
def test_constant_flux_and_uncertainty(spectrum, mock_svo, detector):
    result = run_photometry(spectrum, custom_filters={'custom': custom_filter(detector=detector)})
    assert result['syn_flux(erg/s/cm2/A)'][0] == pytest.approx(1e-15, rel=1e-12)
    assert result['esyn_flux(erg/s/cm2/A)'][0] == pytest.approx(1e-16, rel=1e-12)
    assert result['esyn_flux(Jy)'][0] / result['syn_flux(Jy)'][0] == pytest.approx(0.1)


# A sloped spectrum distinguishes photon weighting from energy weighting.
def test_detector_weighting_changes_sloped_spectrum(spectrum, mock_svo):
    wl, _, _ = spectrum
    sloped = (wl / 2.0) * 1e-15
    filters = {
        'energy': custom_filter(detector='energy'),
        'photon': custom_filter(detector='photon'),
    }
    result = seda.synthetic_photometry.synthetic_photometry(
        wl=wl, flux=sloped, flux_unit='erg/s/cm2/A', custom_filters=filters
    )
    energy, photon = result['syn_flux(erg/s/cm2/A)']
    assert photon > energy
    assert energy == pytest.approx(1e-15, rel=1e-12)
    # For a top-hat [1.5, 2.5] and F_lambda proportional to wavelength:
    # photon-weighted mean = integral(lambda^2) / integral(lambda) / 2 * 1e-15.
    a, b = 1.5, 2.5
    expected = 1e-15 * ((b**3 - a**3) / 3) / ((b**2 - a**2) / 2) / 2
    assert photon == pytest.approx(expected, rel=1e-4)


# Check magnitude conversion and the propagated 10% fractional uncertainty.
def test_custom_zero_point_and_magnitude_uncertainty(spectrum, mock_svo):
    result = run_photometry(
        spectrum, custom_filters={'custom': custom_filter(zero_point=1000.0)}
    )
    flux_jy = result['syn_flux(Jy)'][0]
    assert result['zero_point(Jy)'][0] == 1000.0
    assert result['syn_mag'][0] == pytest.approx(-2.5 * np.log10(flux_jy / 1000.0))
    assert result['esyn_mag'][0] == pytest.approx((2.5 / np.log(10)) * 0.1)


# Check full, partial, and absent overlap between spectrum and filter.
def test_coverage(spectrum, mock_svo):
    filters = {
        'complete': custom_filter(1.5, 2.5),
        'incomplete': custom_filter(0.5, 1.5),
        'none': custom_filter(3.5, 4.0),
    }
    result = run_photometry(spectrum, custom_filters=filters)
    assert list(result['label']) == ['complete', 'incomplete', 'none']
    assert result['coverage_perc'][0] == pytest.approx(100)
    assert 0 < result['coverage_perc'][1] < 100
    assert result['coverage_perc'][2] == pytest.approx(0)
    assert np.isnan(result['syn_flux(erg/s/cm2/A)'][2])


# Each malformed custom-filter definition should raise an informative error.
@pytest.mark.parametrize('invalid_filter, error', [
    ({'wl': [1.5, 2.5], 'transmission': [1, 1]}, 'detector'),
    ({'wl': [1.5, 2.5], 'transmission': [1, 1], 'detector': 'invalid'}, 'detector'),
    ({'wl': [2.5, 1.5], 'transmission': [1, 1], 'detector': 'energy'}, 'increasing'),
    ({'wl': [1.5, 2.5], 'transmission': [-1, 1], 'detector': 'energy'}, 'transmission'),
    ({'wl': [1.5, 2.5], 'transmission': [0, 0], 'detector': 'energy'}, 'transmission'),
    ({'wl': [1.5, 2.5], 'transmission': [1], 'detector': 'energy'}, 'lengths'),
    ({'wl': [1.5, 2.5], 'transmission': [1, 1], 'detector': 'energy', 'zero_point': -1}, 'zero_point'),
])
def test_invalid_custom_filters(spectrum, invalid_filter, error):
    with pytest.raises(ValueError, match=error):
        run_photometry(spectrum, custom_filters={'bad': invalid_filter})


def test_duplicate_filter_name(spectrum):
    with pytest.raises(ValueError, match='Duplicate filter names'):
        run_photometry(spectrum, filters=['same'], custom_filters={'same': custom_filter()})


def test_no_filters(spectrum):
    with pytest.raises(ValueError, match='at least one'):
        run_photometry(spectrum)


def test_invalid_detector_override(spectrum):
    with pytest.raises(ValueError, match='detector'):
        run_photometry(spectrum, filters=['SVO/energy'], detector='invalid')

