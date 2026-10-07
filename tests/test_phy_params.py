import numpy as np
import pytest
from astropy import units as u
from astropy.constants import G, M_jup, M_sun, R_jup, R_sun

import seda
from seda.evolution_aux.units import age_from_gyr, age_to_gyr, mass_to_mjup, radius_to_rjup
from tests.conftest import load_evolutionary_model_catalog, load_evolutionary_table_catalog

BOBCAT_FILENAME = 'nc+0.0_co1.0_mass'
DIAMONDBACK_FILENAME = 'nc_m0.0_mass'

# Units of the bundled tables as published. Deliberately not read from
# config.json or plugin.py: those are the pieces a bad conversion edit
# would change together. phy_params must still return M_jup, Gyr, and R_jup.
PUBLISHED_EVOLUTIONARY_FILE_UNITS = {
	'Sonora_Bobcat': {'mass': 'M_sun', 'age': 'Gyr', 'radius': 'R_sun'},
	'Sonora_Diamondback': {'mass': 'M_sun', 'age': 'Gyr', 'radius': 'R_jup'},
	'ATMO2020': {'mass': 'M_sun', 'age': 'Gyr', 'radius': 'R_sun'},
	'BHAC2015': {'mass': 'M_sun', 'age': 'log10(yr)', 'radius': 'R_sun'},
}

# ----------------------------
# Helpers
# ----------------------------
def _published_file_units(model):
	"""Return the published units of one bundled evolutionary table."""
	try:
		return PUBLISHED_EVOLUTIONARY_FILE_UNITS[model]
	except KeyError as exc:
		raise AssertionError(
			f'{model} is missing from PUBLISHED_EVOLUTIONARY_FILE_UNITS. '
			f'Add its bundled mass, age, and radius units before trusting conversions.'
		) from exc

def _convert_file_value(param, file_value, unit):
	"""Convert one raw table value with astropy. Does not call a plugin."""
	file_value = np.asarray(file_value, dtype=float)
	if param == 'mass':
		if unit == 'M_sun':
			return (file_value * M_sun).to(M_jup).value
		if unit == 'M_jup':
			return file_value
	elif param == 'radius':
		if unit == 'R_sun':
			return (file_value * R_sun).to(R_jup).value
		if unit == 'R_jup':
			return file_value
	elif param == 'age':
		if unit == 'Gyr':
			return file_value
		if unit == 'log10(yr)':
			return ((10.0 ** file_value) * u.yr).to(u.Gyr).value
	raise AssertionError(f'Cannot convert {param} from unit {unit!r}.')

def _independent_user_value(model, param, file_value):
	"""Convert one table value using the published file units, not the plugin."""
	return _convert_file_value(param, file_value, _published_file_units(model)[param])

def _predict_logg(mass, radius, mass_unit, radius_unit):
	"""log10(g [cm/s^2]) from mass and radius interpreted in the given units."""
	mass_qty = mass * (M_sun if mass_unit == 'M_sun' else M_jup)
	radius_qty = radius * (R_sun if radius_unit == 'R_sun' else R_jup)
	g = (G * mass_qty / radius_qty**2).to(u.cm / u.s**2)
	return np.log10(g.value)

def _infer_mass_radius_units(mass, radius, logg):
	"""Return the only mass/radius units that reproduce the table logg.

	Surface gravity fixes the units. Swapping R_sun with R_jup, or M_sun
	with M_jup, moves logg by about 1 dex or more.
	"""
	mass = np.asarray(mass, dtype=float)
	radius = np.asarray(radius, dtype=float)
	logg = np.asarray(logg, dtype=float)
	usable = (mass > 0) & (radius > 0) & np.isfinite(logg)
	hypotheses = (
		('M_sun', 'R_sun'),
		('M_sun', 'R_jup'),
		('M_jup', 'R_sun'),
		('M_jup', 'R_jup'),
	)
	residuals = {}
	for mass_unit, radius_unit in hypotheses:
		predicted = _predict_logg(mass[usable], radius[usable], mass_unit, radius_unit)
		residuals[(mass_unit, radius_unit)] = float(np.median(np.abs(predicted - logg[usable])))
	matches = [key for key, residual in residuals.items() if residual < 0.05]
	if len(matches) != 1:
		raise AssertionError(
			'Table mass and radius units are ambiguous. '
			f'Median |logg_predicted - logg_table| in dex: {residuals}'
		)
	for key, residual in residuals.items():
		if key not in matches and residual < 0.5:
			raise AssertionError(
				f'Wrong unit pair {key} is too close to the table logg '
				f'(median residual {residual} dex). Residuals: {residuals}'
			)
	mass_unit, radius_unit = matches[0]
	return {'mass': mass_unit, 'radius': radius_unit}, residuals

def _ages_in_gyr(age, unit):
	return np.asarray(_convert_file_value('age', age, unit), dtype=float)

def _pre_ms_contraction_is_young(age, mass, radius, mass_unit, radius_unit, age_unit):
	"""True when an inflated >=0.5 Msun track is still <50 Myr at maximum radius.

	False when no such track exists, or when maximum radius falls at an older age.
	"""
	mass_msun = np.asarray(_convert_file_value('mass', mass, mass_unit), dtype=float)
	mass_msun = (mass_msun * M_jup).to(M_sun).value
	radius_rsun = np.asarray(_convert_file_value('radius', radius, radius_unit), dtype=float)
	radius_rsun = (radius_rsun * R_jup).to(R_sun).value
	age = np.asarray(age, dtype=float)
	saw_inflated_track = False
	for mass_val in np.unique(np.round(mass_msun, decimals=6)):
		if mass_val < 0.5:
			continue
		mask = np.round(mass_msun, decimals=6) == mass_val
		track_radius = radius_rsun[mask]
		positive = track_radius > 0
		if np.count_nonzero(positive) < 2:
			continue
		track_radius = track_radius[positive]
		track_age = age[mask][positive]
		if track_radius.max() / track_radius.min() < 2.0:
			continue
		saw_inflated_track = True
		age_at_max_radius = float(_ages_in_gyr(track_age[np.argmax(track_radius)], age_unit))
		if age_at_max_radius > 0.05:
			return False
	return saw_inflated_track

def _infer_age_unit(age, mass, radius, mass_unit, radius_unit):
	"""Return Gyr or log10(yr), whichever is physically possible for this table.

	A column that includes values below 5 cannot be log10(yr): that would be
	an age under about a day, while the same column also reaches many Gyr.
	A column whose values all sit near 6–10 can be either several Gyr or
	log10(yr). In that case a contracting stellar track must be younger than
	50 Myr at its largest radius, which only the log10(yr) reading satisfies
	for BHAC15.
	"""
	plausible = {}
	for unit in ('Gyr', 'log10(yr)'):
		age_gyr = _ages_in_gyr(age, unit)
		plausible[unit] = bool(np.all((age_gyr > 1.0e-4) & (age_gyr < 20.0)))
	candidates = [unit for unit, ok in plausible.items() if ok]
	if len(candidates) == 1:
		return candidates[0]
	young = {
		unit: _pre_ms_contraction_is_young(
			age, mass, radius, mass_unit, radius_unit, unit,
		)
		for unit in ('Gyr', 'log10(yr)')
	}
	candidates = [unit for unit, ok in young.items() if ok]
	if len(candidates) != 1:
		raise AssertionError(
			'Table age unit is ambiguous. '
			f'Plausible Gyr window: {plausible}. Young inflated track: {young}.'
		)
	return candidates[0]

def _grid_radius_in_rjup(model, radius):
	"""Convert a native grid radius value to R_jup using published file units."""
	return _independent_user_value(model, 'radius', radius)

def _grid_radius_native(model, filename, idx=500):
	"""Return native-grid radius for one row of a bundled evolutionary table."""
	grid = seda.models.read_evolutionary_model(filename=filename, model=model)
	if idx < 0:
		idx = len(grid['mass']) + idx
	return float(grid['radius'][idx])

def _bundled_grid_inputs(model, filename, idx=500):
	"""Return (Lbol, R, Teff, logg, age, mass) for one row of a bundled evolutionary table."""
	grid = seda.models.read_evolutionary_model(filename=filename, model=model)
	if idx < 0:
		idx = len(grid['mass']) + idx

	Lbol = 10.0 ** grid['logL'][idx]
	R_rjup = _grid_radius_in_rjup(model, grid['radius'][idx])
	mass_msun = grid['mass'][idx]
	return Lbol, R_rjup, grid['Teff'][idx], grid['logg'][idx], grid['age'][idx], mass_msun

def _grid_sample_indices(n_rows, n_samples=5):
	"""Return evenly spaced row indices across an evolutionary table."""
	if n_rows == 1:
		return [0]
	step = max((n_rows - 1) // (n_samples - 1), 1)
	indices = sorted({min(i * step, n_rows - 1) for i in range(n_samples)})
	return indices

def _evol_sb_teff_cases():
	"""(model, filename, grid_index) cases spanning every bundled evolutionary table."""
	cases = []
	for model, filename in load_evolutionary_table_catalog():
		grid = seda.models.read_evolutionary_model(filename=filename, model=model)
		for idx in _grid_sample_indices(len(grid['mass'])):
			cases.append(
				pytest.param(
					model, filename, idx,
					id=f'{model}-{filename}-row{idx}',
				)
			)
	return cases

# ----------------------------
# Tests
# ----------------------------
@pytest.mark.parametrize('model, filename, idx', _evol_sb_teff_cases())
def test_evol_teff_matches_stefan_boltzmann(model, filename, idx):
	"""Teff from evol_params should match phy_params.teff (SB law) for the same (Lbol, R)."""
	np.random.seed(0)
	Lbol, R_rjup, _, _, _, _ = _bundled_grid_inputs(model, filename, idx=idx)
	eLbol = 1e-12 * Lbol
	eR = 1e-12 * R_rjup

	teff_evol = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=eLbol, R=R_rjup, eR=eR,
		model=model, filename=filename, n_mc=1000, verbose=False,
	)['Teff']

	teff_sb, _ = seda.phy_params.teff(
		Lbol=Lbol, eLbol=eLbol, R=R_rjup, eR=eR, n_mc=1000,
	)

	assert teff_evol == pytest.approx(teff_sb, rel=0.02), (
		f'{model}/{filename} row {idx}: evol_params Teff={teff_evol:.2f} K '
		f'differs from Stefan-Boltzmann Teff={teff_sb:.2f} K'
	)

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_evol_params_round_trip(model, filename):
	"""Feeding a grid row's (Lbol, R) should recover that row's mass/age/logg/Teff."""
	np.random.seed(0)
	Lbol, R_rjup, Teff_exp, logg_exp, age_native, mass_native = _bundled_grid_inputs(
		model, filename,
	)
	mass_exp = _independent_user_value(model, 'mass', mass_native)
	age_exp = _independent_user_value(model, 'age', age_native)

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, R=R_rjup, eR=1e-10 * R_rjup,
		model=model, filename=filename, n_mc=2000, verbose=False,
	)

	assert out['mass'] == pytest.approx(mass_exp, rel=0.05), (
		f"Expected mass ~{mass_exp} M_jup, got {out['mass']}"
	)
	assert out['age'] == pytest.approx(age_exp, rel=0.05), (
		f"Expected age ~{age_exp} Gyr, got {out['age']}"
	)
	assert out['logg'] == pytest.approx(logg_exp, rel=0.05), (
		f"Expected logg ~{logg_exp}, got {out['logg']}"
	)
	assert out['Teff'] == pytest.approx(Teff_exp, rel=0.05), (
		f"Expected Teff ~{Teff_exp} K, got {out['Teff']}"
	)
	assert 'n_outside_grid' in out and 'frac_outside_grid' in out, (
		"Output should report out-of-grid sample bookkeeping"
	)

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_evol_params_outside_grid_raises(model, filename):
	"""A luminosity far outside the grid leaves no valid samples and must raise."""
	np.random.seed(0)

	with pytest.raises(ValueError):
		seda.phy_params.evol_params(
			Lbol=1e10, eLbol=1e8, R=1.0, eR=1e-4,
			model=model, filename=filename, n_mc=500, verbose=False,
		)

def test_evol_params_invalid_filename_lists_available(capsys):
	"""An unrecognized filename should list available tables and raise."""
	with pytest.raises(ValueError, match='not recognized'):
		seda.phy_params.evol_params(
			Lbol=1e-3, eLbol=1e-4, R=1.0, eR=0.1,
			model='Sonora_Bobcat',
			filename='not_a_real_table', n_mc=100, verbose=False,
		)

	captured = capsys.readouterr()
	assert 'nc+0.0_co1.0_mass' in captured.out, (
		'invalid filename error should list available Sonora_Bobcat tables'
	)
	assert 'nc-0.5_co1.0_mass' in captured.out, (
		'invalid filename error should list available Sonora_Bobcat tables'
	)

@pytest.mark.parametrize(
	'model',
	[
		model for model in seda.models.EvolutionaryModels().available_models
		if len(seda.models.EvolutionaryModels(model).available_tables) > 1
	],
)
def test_evol_params_multiple_tables_without_filename_raises(model, capsys):
	"""Models with multiple tables must list them when filename is omitted."""
	with pytest.raises(ValueError, match='Multiple evolutionary tables'):
		seda.phy_params.evol_params(
			Lbol=1e-3, eLbol=1e-4, R=1.0, eR=0.1,
			model=model, n_mc=100, verbose=False,
		)

	captured = capsys.readouterr()
	for filename in seda.models.EvolutionaryModels(model).available_tables:
		assert filename in captured.out, (
			f'missing-table error for {model} should list {filename!r}'
		)

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_evol_params_dynamic_output_keys(model, filename):
	"""Returned keys should match grid columns with e-prefix uncertainties."""
	Lbol, R_rjup, _, _, _, _ = _bundled_grid_inputs(model, filename)

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, R=R_rjup, eR=1e-10 * R_rjup,
		model=model, filename=filename, n_mc=500, verbose=False,
	)

	for param in ('mass', 'age', 'logg', 'Teff'):
		assert param in out, (
			f'evol_params output for {model}/{filename} missing key {param!r}'
		)
		assert f'e{param}' in out, (
			f'evol_params output for {model}/{filename} missing uncertainty e{param!r}'
		)

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_evol_params_std_error_mode(model, filename):
	"""With error='std' the uncertainties should be returned as scalars."""
	Lbol, R_rjup, _, _, _, _ = _bundled_grid_inputs(model, filename)

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, R=R_rjup, eR=1e-10 * R_rjup,
		model=model, filename=filename, error="std", n_mc=1000, verbose=False,
	)

	assert np.isscalar(out["emass"]), "emass should be a scalar when error='std'"
	assert np.isscalar(out["eage"]), "eage should be a scalar when error='std'"
	assert np.isscalar(out["elogg"]), "elogg should be a scalar when error='std'"
	assert np.isscalar(out["eTeff"]), "eTeff should be a scalar when error='std'"

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_evol_params_nonpositive_lbol_raises(model, filename):
	"""A non-positive bolometric luminosity should raise an error."""
	with pytest.raises(ValueError):
		seda.phy_params.evol_params(
			Lbol=0.0, eLbol=1e-5, R=1.0, eR=0.1,
			model=model, filename=filename, n_mc=500, verbose=False,
		)

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_evol_params_reproducible_with_seed(model, filename):
	"""Fixing the random seed should make the Monte Carlo results reproducible."""
	Lbol, R_rjup, _, _, _, _ = _bundled_grid_inputs(model, filename)
	grid = seda.models.read_evolutionary_model(filename=filename, model=model)
	interp_params = [p for p in grid if p not in ('logL', 'radius')]

	np.random.seed(0)
	out1 = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-4 * Lbol, R=R_rjup, eR=1e-4 * R_rjup,
		model=model, filename=filename, n_mc=5000, verbose=False,
	)

	np.random.seed(0)
	out2 = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-4 * Lbol, R=R_rjup, eR=1e-4 * R_rjup,
		model=model, filename=filename, n_mc=5000, verbose=False,
	)

	for param in interp_params:
		assert out1[param] == pytest.approx(out2[param]), (
			f"{param} should be reproducible with a fixed random seed"
		)
		assert out1[f'e{param}'] == pytest.approx(out2[f'e{param}']), (
			f"e{param} should be reproducible with a fixed random seed"
		)

@pytest.mark.parametrize('model, filename', load_evolutionary_table_catalog())
def test_evol_params_bundled_filenames(model, filename):
	"""Each bundled evolutionary table should recover a grid row."""
	np.random.seed(0)
	Lbol, R_rjup, Teff_exp, logg_exp, age_native, mass_native = _bundled_grid_inputs(
		model, filename, idx=500,
	)
	mass_exp = _independent_user_value(model, 'mass', mass_native)
	age_exp = _independent_user_value(model, 'age', age_native)

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, R=R_rjup, eR=1e-10 * R_rjup,
		model=model, filename=filename, n_mc=500, verbose=False,
	)

	assert out['mass'] == pytest.approx(mass_exp, rel=0.05), (
		f'{model}/{filename}: expected mass ~{mass_exp} M_jup, got {out["mass"]}'
	)
	assert out['age'] == pytest.approx(age_exp, rel=0.05), (
		f'{model}/{filename}: expected age ~{age_exp} Gyr, got {out["age"]}'
	)
	assert np.isfinite(out['logg']), (
		f'{model}/{filename}: logg should be finite, got {out["logg"]}'
	)
	assert np.isfinite(out['Teff']), (
		f'{model}/{filename}: Teff should be finite, got {out["Teff"]}'
	)

def test_evol_params_bobcat_file():
	"""Run evol_params() on the bundled solar Bobcat table and print the derived parameters."""
	np.random.seed(0)

	Lbol, eLbol = 6.324e-5, 6.978e-6  # Lsun
	R, eR = 1.018, 0.059              # Rjup

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=eLbol, R=R, eR=eR,
		model='Sonora_Bobcat', filename=BOBCAT_FILENAME, verbose=True,
	)

	print("\nDerived fundamental parameters from bundled Bobcat table:")
	print(f"   mass = {out['mass']:.4g} M_jup  (err {out['emass']})")
	print(f"   age  = {out['age']:.4g} Gyr     (err {out['eage']})")
	print(f"   logg = {out['logg']:.4g} dex    (err {out['elogg']})")
	print(f"   Teff = {out['Teff']:.4g} K      (err {out['eTeff']})")
	print(f"   samples outside grid: {out['frac_outside_grid'] * 100:.1f}%")

	assert np.isfinite(out['mass']), (
		f'Bobcat example mass should be finite, got {out["mass"]}'
	)
	assert np.isfinite(out['age']), (
		f'Bobcat example age should be finite, got {out["age"]}'
	)
	assert np.isfinite(out['logg']), (
		f'Bobcat example logg should be finite, got {out["logg"]}'
	)
	assert np.isfinite(out['Teff']), (
		f'Bobcat example Teff should be finite, got {out["Teff"]}'
	)
	assert out['frac_outside_grid'] < 0.75, (
		f'Bobcat example frac_outside_grid should be < 0.75, got {out["frac_outside_grid"]}'
	)

def test_evol_params_diamondback():
	"""Sonora Diamondback evolutionary tables should round-trip through evol_params."""
	np.random.seed(0)
	model = 'Sonora_Diamondback'
	Lbol, R_rjup, Teff_exp, logg_exp, age_native, mass_native = _bundled_grid_inputs(
		model, DIAMONDBACK_FILENAME, idx=500,
	)
	mass_exp = _independent_user_value(model, 'mass', mass_native)
	age_exp = _independent_user_value(model, 'age', age_native)

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, R=R_rjup, eR=1e-10 * R_rjup,
		model=model, filename=DIAMONDBACK_FILENAME, n_mc=1000, verbose=False,
	)

	assert out['mass'] == pytest.approx(mass_exp, rel=0.01), (
		f'Diamondback round-trip mass mismatch: expected {mass_exp}, got {out["mass"]}'
	)
	assert out['age'] == pytest.approx(age_exp, rel=0.01), (
		f'Diamondback round-trip age mismatch: expected {age_exp}, got {out["age"]}'
	)
	assert out['Teff'] == pytest.approx(Teff_exp, rel=0.01), (
		f'Diamondback round-trip Teff mismatch: expected {Teff_exp}, got {out["Teff"]}'
	)
	assert out['logg'] == pytest.approx(logg_exp, rel=0.01), (
		f'Diamondback round-trip logg mismatch: expected {logg_exp}, got {out["logg"]}'
	)
	assert out['frac_outside_grid'] < 0.1, (
		f'Diamondback frac_outside_grid should be < 0.1, got {out["frac_outside_grid"]}'
	)

def test_evol_params_regular_user_output():
	"""Show what a regular user sees when calling evol_params(verbose=True)."""
	np.random.seed(0)

	seda.phy_params.evol_params(
		Lbol=6.324e-5, eLbol=6.978e-6, R=1.018, eR=0.059,
		model='Sonora_Bobcat', filename=BOBCAT_FILENAME, error="percentile", verbose=True,
	)

def test_user_unit_conversions_match_astropy():
	"""Age, mass, and radius conversions use astropy year and solar/Jupiter constants."""
	assert mass_to_mjup(1.0, 'M_sun') == pytest.approx((1.0 * M_sun).to(M_jup).value)
	assert mass_to_mjup(2.5, 'M_jup') == pytest.approx(2.5)
	assert radius_to_rjup(1.0, 'R_sun') == pytest.approx((1.0 * R_sun).to(R_jup).value)
	assert radius_to_rjup(1.25, 'R_jup') == pytest.approx(1.25)

	assert age_to_gyr(9.0, 'log10(yr)') == pytest.approx(1.0)
	assert age_to_gyr(6.0, 'log10(yr)') == pytest.approx(1e-3)
	assert float(age_from_gyr(1.0, 'log10(yr)')) == pytest.approx(9.0)
	assert age_to_gyr(4.5, 'Gyr') == pytest.approx(4.5)
	assert float(age_from_gyr(4.5, 'Gyr')) == pytest.approx(4.5)

	# 1 Gyr -> log10(yr) -> Gyr must close.
	log_age = age_from_gyr(np.array([0.001, 1.0, 10.0]), 'log10(yr)')
	assert age_to_gyr(log_age, 'log10(yr)') == pytest.approx([0.001, 1.0, 10.0])

	nonpositive = age_from_gyr(np.array([-0.1, 0.0, 1.0]), 'log10(yr)')
	assert np.isnan(nonpositive[0]) and np.isnan(nonpositive[1])
	assert nonpositive[2] == pytest.approx(9.0)

@pytest.mark.parametrize('model', sorted(seda.models.EvolutionaryModels().available_models))
def test_plugin_file_units_match_config(model):
	"""Plugin conversion units must be the bundled-table units in config.json."""
	config, plugin = seda.models._load_evolutionary_model(model)
	for key in ('mass', 'age', 'radius'):
		assert plugin._FILE_UNITS[key] == config['units'][key], (
			f'{model} plugin _FILE_UNITS[{key!r}]={plugin._FILE_UNITS[key]!r} '
			f'does not match config.json {config["units"][key]!r}'
		)

@pytest.mark.parametrize('model, filename', load_evolutionary_table_catalog())
def test_original_table_units_are_fixed_by_the_numbers(model, filename):
	"""Mass, radius, and age units must be the ones the table numbers require.

	log g is recomputed from each mass/radius unit pair. Only the original
	pair reproduces the table (the others miss by at least ~1 dex). Age is
	kept only if that reading lands on real brown-dwarf and stellar ages;
	for BHAC15 the alternate reading puts a still-contracting star at several
	Gyr. config.json and the plugin have to use those inferred units, and
	the converted values have to match astropy from that reading.
	"""
	grid = seda.models.read_evolutionary_model(filename=filename, model=model)
	inferred, residuals = _infer_mass_radius_units(
		grid['mass'], grid['radius'], grid['logg'],
	)
	inferred['age'] = _infer_age_unit(
		grid['age'], grid['mass'], grid['radius'],
		inferred['mass'], inferred['radius'],
	)

	published = _published_file_units(model)
	config, plugin = seda.models._load_evolutionary_model(model)
	for key in ('mass', 'age', 'radius'):
		assert inferred[key] == published[key], (
			f'{model}/{filename}: table numbers require {key} in {inferred[key]!r}, '
			f'not the published-unit entry {published[key]!r}. logg residuals: {residuals}'
		)
		assert config['units'][key] == inferred[key], (
			f'{model}/{filename}: config.json {key} is {config["units"][key]!r}, '
			f'but the table numbers require {inferred[key]!r}. logg residuals: {residuals}'
		)
		assert plugin._FILE_UNITS[key] == inferred[key], (
			f'{model}/{filename}: plugin reads {key} as {plugin._FILE_UNITS[key]!r}, '
			f'but the table numbers require {inferred[key]!r}. logg residuals: {residuals}'
		)

	idx = min(500, len(grid['mass']) - 1)
	for param in ('mass', 'age', 'radius'):
		expected = _convert_file_value(param, grid[param][idx], inferred[param])
		converted = plugin._to_user_units(param, grid[param][idx])
		assert float(np.asarray(converted).reshape(-1)[0]) == pytest.approx(
			float(np.asarray(expected).reshape(-1)[0]), rel=1e-10, abs=1e-12,
		), (
			f'{model}/{filename}: {param} conversion does not follow the '
			f'{inferred[param]} reading of the table'
		)

	for param in ('Teff', 'logg'):
		raw = float(grid[param][idx])
		converted = float(np.asarray(plugin._to_user_units(param, raw)).reshape(-1)[0])
		assert converted == pytest.approx(raw), (
			f'{model}/{filename}: {param} should stay in table units, got {converted}'
		)

def test_bhac_evol_params_keeps_structure_columns_in_file_units():
	"""BHAC radiative-core columns are not mass/radius and stay in file units."""
	np.random.seed(0)
	model = 'BHAC2015'
	filename = 'BHAC15_tracks+structure.txt'
	grid = seda.models.read_evolutionary_model(filename=filename, model=model)
	idx = 800
	Lbol = 10.0 ** float(grid['logL'][idx])
	R_rjup = _grid_radius_in_rjup(model, grid['radius'][idx])

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-12 * Lbol, R=R_rjup, eR=1e-12 * R_rjup,
		model=model, filename=filename, n_mc=400, verbose=False,
	)

	assert out['mass'] == pytest.approx(
		_independent_user_value(model, 'mass', grid['mass'][idx]), rel=0.05,
	)
	assert out['age'] == pytest.approx(
		_independent_user_value(model, 'age', grid['age'][idx]), rel=0.05,
	)
	assert out['age'] != pytest.approx(float(grid['age'][idx]), rel=0.01), (
		f'BHAC age should be Gyr, not the log10(yr) table value {grid["age"][idx]}'
	)
	assert out['Mrad'] == pytest.approx(float(grid['Mrad'][idx]), rel=0.05), (
		f'BHAC Mrad should stay in M_sun, got {out["Mrad"]}'
	)
	assert out['Rrad'] == pytest.approx(float(grid['Rrad'][idx]), rel=0.05), (
		f'BHAC Rrad should stay in R_sun, got {out["Rrad"]}'
	)

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_isochrone_params_user_units_round_trip(model, filename):
	"""A grid row's (Lbol, age in Gyr) should recover mass in M_jup and radius in R_jup."""
	np.random.seed(0)
	grid = seda.models.read_evolutionary_model(filename=filename, model=model)
	idx = min(500, len(grid['mass']) - 1)
	Lbol = 10.0 ** float(grid['logL'][idx])
	age_gyr = _independent_user_value(model, 'age', grid['age'][idx])
	mass_exp = _independent_user_value(model, 'mass', grid['mass'][idx])
	radius_exp = _independent_user_value(model, 'radius', grid['radius'][idx])

	out = seda.phy_params.isochrone_params(
		Lbol=Lbol, eLbol=1e-12 * Lbol, age=age_gyr, eage=0.0,
		model=model, filename=filename, n_mc=300, verbose=False,
	)

	assert out['mass'] == pytest.approx(mass_exp, rel=0.05), (
		f'{model}/{filename}: expected mass ~{mass_exp} M_jup, got {out["mass"]}'
	)
	assert out['radius'] == pytest.approx(radius_exp, rel=0.05), (
		f'{model}/{filename}: expected radius ~{radius_exp} R_jup, got {out["radius"]}'
	)
	assert out['Teff'] == pytest.approx(float(grid['Teff'][idx]), rel=0.05), (
		f'{model}/{filename}: expected Teff ~{float(grid["Teff"][idx])} K, got {out["Teff"]}'
	)

def test_isochrone_params_rejects_nonpositive_age():
	with pytest.raises(ValueError, match='Gyr'):
		seda.phy_params.isochrone_params(
			Lbol=1e-4, eLbol=1e-6, age=0.0, eage=0.0,
			model='Sonora_Bobcat', filename=BOBCAT_FILENAME, n_mc=50, verbose=False,
		)

def test_list_evolutionary_tables():
	"""EvolutionaryModels should expose bundled table basenames for each model."""
	bobcat_tables = seda.models.EvolutionaryModels('Sonora_Bobcat').available_tables
	assert 'nc+0.0_co1.0_mass' in bobcat_tables, (
		'Sonora_Bobcat bundled tables should include nc+0.0_co1.0_mass'
	)
	assert len(bobcat_tables) == 3, (
		f'expected 3 Sonora_Bobcat tables, got {len(bobcat_tables)}'
	)

	diamondback_tables = seda.models.EvolutionaryModels('Sonora_Diamondback').available_tables
	assert 'nc_m0.0_mass' in diamondback_tables, (
		'Sonora_Diamondback bundled tables should include nc_m0.0_mass'
	)
	assert len(diamondback_tables) == 9, (
		f'expected 9 Sonora_Diamondback tables, got {len(diamondback_tables)}'
	)

	atmo_tables = seda.models.EvolutionaryModels('ATMO2020').available_tables
	assert 'ATMO_CEQ_mass.txt' in atmo_tables, (
		'ATMO2020 bundled tables should include ATMO_CEQ_mass.txt'
	)
	assert len(atmo_tables) == 3, (
		f'expected 3 ATMO2020 tables, got {len(atmo_tables)}'
	)

	bhac_tables = seda.models.EvolutionaryModels('BHAC2015').available_tables
	assert 'BHAC15_tracks+structure.txt' in bhac_tables, (
		'BHAC2015 bundled tables should include BHAC15_tracks+structure.txt'
	)
	assert len(bhac_tables) == 1, (
		f'expected 1 BHAC2015 table, got {len(bhac_tables)}'
	)

def test_evolutionary_models_params_requires_model():
	"""params should require a model name, like available_tables."""
	with pytest.raises(Exception, match='Pass a model name'):
		_ = seda.models.EvolutionaryModels().params

@pytest.mark.parametrize('model', sorted(seda.models.EvolutionaryModels().available_models))
def test_evolutionary_models_params_structure(model):
	"""params should list min/max for every grid column in each bundled table."""
	model_obj = seda.models.EvolutionaryModels(model)
	params = model_obj.params

	assert set(params) == set(model_obj.available_tables), (
		f'{model} params keys should match available_tables'
	)
	for filename in model_obj.available_tables:
		grid = seda.models.read_evolutionary_model(filename=filename, model=model)
		assert set(params[filename]) == set(grid), (
			f'{model}/{filename} params columns should match grid columns'
		)
		for col, (vmin, vmax) in params[filename].items():
			assert vmin == pytest.approx(float(grid[col].min())), (
				f'{model}/{filename} {col} min mismatch: '
				f'params={vmin}, grid={float(grid[col].min())}'
			)
			assert vmax == pytest.approx(float(grid[col].max())), (
				f'{model}/{filename} {col} max mismatch: '
				f'params={vmax}, grid={float(grid[col].max())}'
			)

def test_evolutionary_models_params_bobcat_spot_check():
	"""Spot-check known coverage for the solar-metallicity Bobcat table."""
	params = seda.models.EvolutionaryModels('Sonora_Bobcat').params['nc+0.0_co1.0_mass']

	assert params['mass'] == [0.0005, 0.08], (
		f'Bobcat mass range mismatch: {params["mass"]}'
	)
	assert params['age'] == [0.001, 15.0], (
		f'Bobcat age range mismatch: {params["age"]}'
	)
	assert params['logL'] == pytest.approx([-9.213, -2.662]), (
		f'Bobcat logL range mismatch: {params["logL"]}'
	)
	assert params['Teff'] == [91.0, 2537.0], (
		f'Bobcat Teff range mismatch: {params["Teff"]}'
	)
	assert params['logg'] == pytest.approx([2.654, 5.484]), (
		f'Bobcat logg range mismatch: {params["logg"]}'
	)
	assert params['radius'] == pytest.approx([0.0769, 0.2657]), (
		f'Bobcat radius range mismatch: {params["radius"]}'
	)
	assert 'logI' not in params, (
		'Bobcat table should not expose logI in params'
	)

def test_evolutionary_models_params_atmo_spot_check():
	"""Spot-check known coverage for the ATMO 2020 CEQ table."""
	params = seda.models.EvolutionaryModels('ATMO2020').params['ATMO_CEQ_mass.txt']

	assert params['mass'] == [0.001, 0.075], (
		f'ATMO mass range mismatch: {params["mass"]}'
	)
	assert params['age'] == [0.001, 10.0], (
		f'ATMO age range mismatch: {params["age"]}'
	)
	assert params['logL'] == pytest.approx([-7.74437436, -1.27027279]), (
		f'ATMO logL range mismatch: {params["logL"]}'
	)
	assert params['Teff'] == pytest.approx([206.71029843, 3156.67625353]), (
		f'ATMO Teff range mismatch: {params["Teff"]}'
	)
	assert params['logg'] == pytest.approx([3.01108287, 5.51013179]), (
		f'ATMO logg range mismatch: {params["logg"]}'
	)
	assert params['radius'] == pytest.approx([0.07585432, 0.79547701]), (
		f'ATMO radius range mismatch: {params["radius"]}'
	)

def test_evolutionary_models_params_bhac_spot_check():
	"""Spot-check known coverage for the BHAC15 tracks+structure table."""
	params = seda.models.EvolutionaryModels('BHAC2015').params['BHAC15_tracks+structure.txt']

	assert params['mass'] == [0.01, 1.4], (
		f'BHAC mass range mismatch: {params["mass"]}'
	)
	assert params['age'] == pytest.approx([5.68945, 10.000343]), (
		f'BHAC age range mismatch: {params["age"]}'
	)
	assert params['logL'] == pytest.approx([-4.716, 0.74]), (
		f'BHAC logL range mismatch: {params["logL"]}'
	)
	assert params['Teff'] == [1206.0, 6768.0], (
		f'BHAC Teff range mismatch: {params["Teff"]}'
	)
	assert params['logg'] == pytest.approx([3.224, 5.391]), (
		f'BHAC logg range mismatch: {params["logg"]}'
	)
	assert params['radius'] == pytest.approx([0.086, 3.621]), (
		f'BHAC radius range mismatch: {params["radius"]}'
	)
	assert params['logLi'] == pytest.approx([-11.1759, 0.0]), (
		f'BHAC logLi range mismatch: {params["logLi"]}'
	)
	assert params['logTc'] == pytest.approx([5.417, 7.398]), (
		f'BHAC logTc range mismatch: {params["logTc"]}'
	)
	assert params['logRho_c'] == pytest.approx([-0.6068, 2.8806]), (
		f'BHAC logRho_c range mismatch: {params["logRho_c"]}'
	)
	assert params['Mrad'] == pytest.approx([0.0, 1.4]), (
		f'BHAC Mrad range mismatch: {params["Mrad"]}'
	)
	assert params['Rrad'] == pytest.approx([0.0, 1.745]), (
		f'BHAC Rrad range mismatch: {params["Rrad"]}'
	)
	assert params['k2conv'] == pytest.approx([0.00124, 0.4944]), (
		f'BHAC k2conv range mismatch: {params["k2conv"]}'
	)
	assert params['k2rad'] == pytest.approx([0.0, 0.3072]), (
		f'BHAC k2rad range mismatch: {params["k2rad"]}'
	)

def _expected_inclination_deg(vsini, P, R):
	"""Deterministic inclination from sin i = P*vsini / (2*pi*R)."""
	vsini_u = vsini * u.km / u.s
	P_u = P * u.hour
	R_u = R * R_jup
	v_eq = (2 * np.pi * R_u / P_u).to(u.km / u.s)
	sin_i = (vsini_u / v_eq).decompose().value
	return np.degrees(np.arcsin(np.clip(sin_i, -1.0, 1.0)))

def _vsini_for_inclination(P, R, inc_deg):
	"""Invert the inclination formula for a target inclination."""
	P_u = P * u.hour
	R_u = R * R_jup
	v_eq = (2 * np.pi * R_u / P_u).to(u.km / u.s)
	return (v_eq * np.sin(np.radians(inc_deg))).to(u.km / u.s).value

@pytest.mark.parametrize(
	'inc_deg, P, R',
	[
		(30.0, 4.0, 1.10),
		(45.0, 5.0, 1.20),
		(60.0, 3.1, 1.05),
		(85.0, 2.5, 1.30),
	],
)
def test_inclination_recovers_known_angle(inc_deg, P, R):
	"""With tiny errors, inclination should round-trip the sin i formula."""
	vsini = _vsini_for_inclination(P, R, inc_deg)
	np.random.seed(0)

	inc, einc = seda.phy_params.inclination(
		vsini=vsini, evsini=1e-10 * vsini,
		P=P, eP=1e-10 * P,
		R=R, eR=1e-10 * R,
		n_mc=5000,
	)

	assert inc == pytest.approx(inc_deg, abs=0.5), (
		f'inclination {inc:.3f} deg did not recover expected {inc_deg} deg '
		f'for P={P} hr, R={R} R_jup'
	)
	assert einc[0] >= 0 and einc[1] >= 0, (
		f'inclination asymmetric errors must be non-negative, got {einc}'
	)

def test_inclination_matches_deterministic_formula():
	"""Spot-check against the docstring example inputs."""
	vsini, evsini = 26.4, 1.2
	P, eP = 3.1, 0.1
	R, eR = 1.05, 0.06
	expected = _expected_inclination_deg(vsini, P, R)

	np.random.seed(0)
	inc, einc = seda.phy_params.inclination(
		vsini=vsini, evsini=evsini,
		P=P, eP=eP, R=R, eR=eR,
		n_mc=10000,
	)

	assert inc == pytest.approx(expected, rel=0.05), (
		f'inclination {inc:.3f} deg differs from deterministic formula '
		f'prediction {expected:.3f} deg'
	)
	assert len(einc) == 2, (
		f'expected two asymmetric inclination uncertainties, got {len(einc)}'
	)

def test_inclination_std_error_mode():
	"""With error='std', the uncertainty should be a scalar."""
	np.random.seed(0)
	inc, einc = seda.phy_params.inclination(
		vsini=20.0, evsini=1.0,
		P=4.0, eP=0.1,
		R=1.1, eR=0.05,
		error='std', n_mc=5000,
	)
	assert np.isscalar(einc), (
		"einc should be a scalar when error='std'"
	)
	assert einc > 0, (
		f'std inclination uncertainty should be positive, got {einc}'
	)
	assert np.isfinite(inc), (
		f'inclination should be finite, got {inc}'
	)

def test_inclination_invalid_central_raises():
	with pytest.raises(ValueError, match='central'):
		seda.phy_params.inclination(
			vsini=20.0, evsini=1.0,
			P=4.0, eP=0.1,
			R=1.1, eR=0.05,
			central='mode', n_mc=100,
		)

def test_inclination_reproducible_with_seed():
	"""Fixed seed should give identical MC results."""
	kwargs = dict(
		vsini=26.4, evsini=1.2,
		P=3.1, eP=0.1,
		R=1.05, eR=0.06,
		n_mc=5000,
	)
	np.random.seed(42)
	inc1, einc1 = seda.phy_params.inclination(**kwargs)
	np.random.seed(42)
	inc2, einc2 = seda.phy_params.inclination(**kwargs)

	assert inc1 == pytest.approx(inc2), (
		f'fixed seed gave inconsistent inclination: {inc1} vs {inc2}'
	)
	assert einc1 == pytest.approx(einc2), (
		f'fixed seed gave inconsistent inclination uncertainty: {einc1} vs {einc2}'
	)

def test_inclination_face_on():
	"""vsini = 0 should give i = 0 deg."""
	np.random.seed(0)
	inc, _ = seda.phy_params.inclination(
		vsini=0.0, evsini=0.1,
		P=5.0, eP=0.1,
		R=1.0, eR=0.05,
		n_mc=2000,
	)
	assert inc == pytest.approx(0.0, abs=0.5), (
		f'face-on target (vsini=0) should give i ~ 0 deg, got {inc:.3f} deg'
	)

def test_inclination_reports_clipped_samples(capsys):
	"""Unphysical draws with sin i > 1 should be clipped to 1 and reported."""
	P, R = 10.0, 0.5
	v_eq = (2 * np.pi * R * R_jup / (P * u.hour)).to(u.km / u.s).value
	vsini = 0.95 * v_eq
	evsini = 0.1 * v_eq

	np.random.seed(0)
	inc, einc = seda.phy_params.inclination(
		vsini=vsini, evsini=evsini,
		P=P, eP=0.5,
		R=R, eR=0.05,
		n_mc=1000,
	)
	captured = capsys.readouterr()
	assert 'sin i > 1' in captured.out, (
		'clipped sin i > 1 draws should be reported in stdout'
	)
	assert 'set to sin i = 1' in captured.out, (
		'clipping message should mention sin i = 1'
	)
	assert 'MC samples used for inclination statistics' in captured.out, (
		'clipping summary should report MC sample usage'
	)
	assert 0.0 < inc <= 90.0, (
		f'clipped inclination should be in (0, 90] deg, got {inc:.3f} deg'
	)
	assert len(einc) == 2, (
		f'expected two asymmetric inclination uncertainties, got {len(einc)}'
	)

def test_inclination_all_clipped_returns_edge_on(capsys):
	"""When every draw has sin i > 1, clipped samples give i = 90 deg."""
	np.random.seed(0)
	inc, einc = seda.phy_params.inclination(
		vsini=100.0, evsini=5.0,
		P=10.0, eP=0.5,
		R=0.5, eR=0.05,
		n_mc=1000,
	)
	captured = capsys.readouterr()
	assert 'sin i > 1' in captured.out, (
		'all-clipped case should still report sin i > 1 in stdout'
	)
	assert '1000/1000 MC samples used' in captured.out, (
		'all-clipped case should report that every MC sample was clipped'
	)
	assert inc == pytest.approx(90.0, abs=0.1), (
		f'all-clipped sin i > 1 draws should give i = 90 deg, got {inc:.3f} deg'
	)
