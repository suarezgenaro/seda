import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from astropy.constants import G, M_jup, M_sun, R_jup, R_sun

import seda
from tests.conftest import load_evolutionary_model_catalog, load_evolutionary_table_catalog

BOBCAT_FILENAME = 'nc+0.0_co1.0_mass'
DIAMONDBACK_FILENAME = 'nc_m0.0_mass'

# Units every evolutionary plugin must return, whatever the published table uses.
AGREED_OUTPUT_UNITS = {
	'mass': 'M_jup',
	'age': 'Gyr',
	'radius': 'R_jup',
	'Teff': 'K',
	'logL': 'log10(L/Lsun)',
	'logg': 'log(g[cm/s^2])',
}

# ----------------------------
# Helpers
# ----------------------------
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
	"""Read an age column as Gyr or as log10(yr), with astropy doing the year."""
	age = np.asarray(age, dtype=float)
	if unit == 'Gyr':
		return age
	if unit == 'log10(yr)':
		return ((10.0 ** age) * u.yr).to(u.Gyr).value
	raise AssertionError(f'Cannot read an age column in unit {unit!r}.')

def _pre_ms_contraction_is_young(age, mass_mjup, radius_rjup, age_unit):
	"""True when an inflated >=0.5 Msun track is still <50 Myr at maximum radius.

	False when no such track exists, or when maximum radius falls at an older age.
	"""
	mass_msun = (np.asarray(mass_mjup, dtype=float) * M_jup).to(M_sun).value
	radius_rsun = (np.asarray(radius_rjup, dtype=float) * R_jup).to(R_sun).value
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

def _infer_age_unit(age, mass_mjup, radius_rjup):
	"""Return Gyr or log10(yr), whichever is physically possible for this column.

	A column that includes values below 5 cannot be log10(yr): that would be
	an age under about a day, while the same column also reaches many Gyr.
	A column whose values all sit near 6–10 can be either several Gyr or
	log10(yr). In that case a contracting stellar track must be younger than
	50 Myr at its largest radius.
	"""
	plausible = {}
	for unit in ('Gyr', 'log10(yr)'):
		age_gyr = _ages_in_gyr(age, unit)
		# 1e-4 Gyr is 100 kyr, the youngest age stored in some bundled grids.
		plausible[unit] = bool(np.all((age_gyr >= 1.0e-4) & (age_gyr < 20.0)))
	candidates = [unit for unit, ok in plausible.items() if ok]
	if len(candidates) == 1:
		return candidates[0]
	young = {
		unit: _pre_ms_contraction_is_young(age, mass_mjup, radius_rjup, unit)
		for unit in ('Gyr', 'log10(yr)')
	}
	candidates = [unit for unit, ok in young.items() if ok]
	if len(candidates) != 1:
		raise AssertionError(
			'Age column unit is ambiguous. '
			f'Plausible Gyr window: {plausible}. Young inflated track: {young}.'
		)
	return candidates[0]

def _table_row(n_rows, idx):
	"""Return ``idx`` when that row exists, otherwise the middle row."""
	if idx < 0:
		idx = n_rows + idx
	if idx < 0 or idx >= n_rows:
		return n_rows // 2
	return idx

def _bundled_grid_inputs(model, filename, idx=500):
	"""Return (logL, R, Teff, logg, age, mass) for one row of a bundled evolutionary table.

	Every column is already in the agreed output units: R_jup, K, Gyr, M_jup.
	"""
	grid = seda.models.read_evolutionary_model(filename=filename, model=model)
	idx = _table_row(len(grid['mass']), idx)

	return (
		float(grid['logL'][idx]), float(grid['radius'][idx]),
		float(grid['Teff'][idx]), float(grid['logg'][idx]),
		float(grid['age'][idx]), float(grid['mass'][idx]),
	)

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
	"""Teff from evol_params should match phy_params.teff (SB law) for the same (logL, R)."""
	np.random.seed(0)
	logL, R_rjup, _, _, _, _ = _bundled_grid_inputs(model, filename, idx=idx)
	Lbol = 10.0 ** logL
	eR = 1e-12 * R_rjup

	teff_evol = seda.phy_params.evol_params(
		logL=logL, elogL=1e-12, R=R_rjup, eR=eR,
		model=model, filename=filename, n_mc=1000, verbose=False,
	)['Teff']

	teff_sb, _ = seda.phy_params.teff(
		Lbol=Lbol, eLbol=1e-12 * Lbol, R=R_rjup, eR=eR, n_mc=1000,
	)

	assert teff_evol == pytest.approx(teff_sb, rel=0.02), (
		f'{model}/{filename} row {idx}: evol_params Teff={teff_evol:.2f} K '
		f'differs from Stefan-Boltzmann Teff={teff_sb:.2f} K'
	)

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_evol_params_round_trip(model, filename):
	"""Feeding a grid row's (logL, R) should recover that row's mass/age/logg/Teff."""
	np.random.seed(0)
	logL, R_rjup, Teff_exp, logg_exp, age_exp, mass_exp = _bundled_grid_inputs(
		model, filename,
	)

	out = seda.phy_params.evol_params(
		logL=logL, elogL=1e-10, R=R_rjup, eR=1e-10 * R_rjup,
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
			logL=10.0, elogL=0.01, R=1.0, eR=1e-4,
			model=model, filename=filename, n_mc=500, verbose=False,
		)

def test_evol_params_invalid_filename_lists_available(capsys):
	"""An unrecognized filename should list available tables and raise."""
	with pytest.raises(ValueError, match='not recognized'):
		seda.phy_params.evol_params(
			logL=-3.0, elogL=0.05, R=1.0, eR=0.1,
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
			logL=-3.0, elogL=0.05, R=1.0, eR=0.1,
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
	logL, R_rjup, _, _, _, _ = _bundled_grid_inputs(model, filename)

	out = seda.phy_params.evol_params(
		logL=logL, elogL=1e-10, R=R_rjup, eR=1e-10 * R_rjup,
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
	logL, R_rjup, _, _, _, _ = _bundled_grid_inputs(model, filename)

	out = seda.phy_params.evol_params(
		logL=logL, elogL=1e-10, R=R_rjup, eR=1e-10 * R_rjup,
		model=model, filename=filename, error="std", n_mc=1000, verbose=False,
	)

	assert np.isscalar(out["emass"]), "emass should be a scalar when error='std'"
	assert np.isscalar(out["eage"]), "eage should be a scalar when error='std'"
	assert np.isscalar(out["elogg"]), "elogg should be a scalar when error='std'"
	assert np.isscalar(out["eTeff"]), "eTeff should be a scalar when error='std'"

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_evol_params_nonfinite_logl_raises(model, filename):
	"""A non-finite log bolometric luminosity should raise an error."""
	with pytest.raises(ValueError, match='logL'):
		seda.phy_params.evol_params(
			logL=np.nan, elogL=0.05, R=1.0, eR=0.1,
			model=model, filename=filename, n_mc=500, verbose=False,
		)

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_evol_params_reproducible_with_seed(model, filename):
	"""Fixing the random seed should make the Monte Carlo results reproducible."""
	logL, R_rjup, _, _, _, _ = _bundled_grid_inputs(model, filename)
	grid = seda.models.read_evolutionary_model(filename=filename, model=model)
	interp_params = [p for p in grid if p not in ('logL', 'radius')]

	np.random.seed(0)
	out1 = seda.phy_params.evol_params(
		logL=logL, elogL=1e-4, R=R_rjup, eR=1e-4 * R_rjup,
		model=model, filename=filename, n_mc=5000, verbose=False,
	)

	np.random.seed(0)
	out2 = seda.phy_params.evol_params(
		logL=logL, elogL=1e-4, R=R_rjup, eR=1e-4 * R_rjup,
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
	logL, R_rjup, Teff_exp, logg_exp, age_exp, mass_exp = _bundled_grid_inputs(
		model, filename, idx=500,
	)

	out = seda.phy_params.evol_params(
		logL=logL, elogL=1e-10, R=R_rjup, eR=1e-10 * R_rjup,
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

	logL, elogL = -4.199, 0.048  # log10(L/Lsun)
	R, eR = 1.018, 0.059         # Rjup

	out = seda.phy_params.evol_params(
		logL=logL, elogL=elogL, R=R, eR=eR,
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
	logL, R_rjup, Teff_exp, logg_exp, age_exp, mass_exp = _bundled_grid_inputs(
		model, DIAMONDBACK_FILENAME, idx=500,
	)

	out = seda.phy_params.evol_params(
		logL=logL, elogL=1e-10, R=R_rjup, eR=1e-10 * R_rjup,
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
		logL=-4.199, elogL=0.048, R=1.018, eR=0.059,
		model='Sonora_Bobcat', filename=BOBCAT_FILENAME, error="percentile", verbose=True,
	)

@pytest.mark.parametrize('model', sorted(seda.models.EvolutionaryModels().available_models))
def test_config_units_declare_the_agreed_output_units(model):
	"""Every config.json must declare the units _read_evolutionary_model returns."""
	config, _ = seda.models._load_evolutionary_model(model)
	for key, unit in AGREED_OUTPUT_UNITS.items():
		assert config['units'][key] == unit, (
			f'{model} config.json declares {key} as {config["units"][key]!r}, '
			f'but every plugin must return {key} in {unit!r}'
		)

@pytest.mark.parametrize('model, filename', load_evolutionary_table_catalog())
def test_returned_table_units_are_fixed_by_the_numbers(model, filename):
	"""The numbers themselves must prove the reader returned M_jup, R_jup, and Gyr.

	This is independent of plugin.py: nothing here reads a declared unit. log g
	is recomputed from the returned mass and radius under all four unit pairs,
	and only (M_jup, R_jup) can reproduce the returned log g column; the wrong
	pairs miss by at least ~0.4 dex. The returned age is kept only if that
	reading lands on real brown-dwarf and stellar ages, which rules out a table
	whose log10(yr) column was passed through unconverted. Stefan-Boltzmann
	then ties logL to the returned radius in R_jup.
	"""
	grid = seda.models.read_evolutionary_model(filename=filename, model=model)
	inferred, residuals = _infer_mass_radius_units(
		grid['mass'], grid['radius'], grid['logg'],
	)
	assert inferred == {'mass': 'M_jup', 'radius': 'R_jup'}, (
		f'{model}/{filename}: the returned numbers are in {inferred}, not '
		f'M_jup and R_jup. Median |logg residual| per unit pair: {residuals}'
	)

	age_unit = _infer_age_unit(grid['age'], grid['mass'], grid['radius'])
	assert age_unit == 'Gyr', (
		f'{model}/{filename}: the returned age column reads as {age_unit!r}, '
		f'not Gyr'
	)

	radius = np.asarray(grid['radius'], dtype=float) * R_jup
	teff = np.asarray(grid['Teff'], dtype=float) * u.K
	lum = (4 * np.pi * radius**2 * const.sigma_sb * teff**4).to(u.Lsun).value
	usable = lum > 0
	assert np.median(np.abs(
		np.log10(lum[usable]) - np.asarray(grid['logL'], dtype=float)[usable]
	)) < 0.01, (
		f'{model}/{filename}: logL does not follow Stefan-Boltzmann from the '
		f'returned radius in R_jup and Teff in K'
	)

def test_bhac_evol_params_converts_radiative_core_columns():
	"""BHAC radiative-core mass and radius are converted like the main columns."""
	np.random.seed(0)
	model = 'BHAC2015'
	filename = 'BHAC15_tracks+structure.txt'
	grid = seda.models.read_evolutionary_model(filename=filename, model=model)
	idx = 800
	logL = float(grid['logL'][idx])
	R_rjup = float(grid['radius'][idx])

	out = seda.phy_params.evol_params(
		logL=logL, elogL=1e-12, R=R_rjup, eR=1e-12 * R_rjup,
		model=model, filename=filename, n_mc=400, verbose=False,
	)

	for param in ('mass', 'age', 'Mrad', 'Rrad'):
		assert out[param] == pytest.approx(float(grid[param][idx]), rel=0.05), (
			f'BHAC {param} should match the grid value {grid[param][idx]}, '
			f'got {out[param]}'
		)
	assert out['Mrad'] < out['mass'], (
		f'BHAC radiative-core mass {out["Mrad"]} M_jup should be below the '
		f'total mass {out["mass"]} M_jup'
	)

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_isochrone_params_user_units_round_trip(model, filename):
	"""A grid row's (logL, age in Gyr) should recover mass in M_jup and radius in R_jup."""
	np.random.seed(0)
	grid = seda.models.read_evolutionary_model(filename=filename, model=model)
	idx = min(500, len(grid['mass']) - 1)
	logL = float(grid['logL'][idx])
	age_gyr = float(grid['age'][idx])
	mass_exp = float(grid['mass'][idx])
	radius_exp = float(grid['radius'][idx])

	out = seda.phy_params.isochrone_params(
		logL=logL, elogL=1e-12, age=age_gyr, eage=0.0,
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
			logL=-4.0, elogL=0.01, age=0.0, eage=0.0,
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

	assert params['mass'] == pytest.approx([0.5237828, 83.8052412]), (
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
	assert params['radius'] == pytest.approx([0.7483261, 2.5855689]), (
		f'Bobcat radius range mismatch: {params["radius"]}'
	)
	assert 'logI' not in params, (
		'Bobcat table should not expose logI in params'
	)

def test_evolutionary_models_params_atmo_spot_check():
	"""Spot-check known coverage for the ATMO 2020 CEQ table."""
	params = seda.models.EvolutionaryModels('ATMO2020').params['ATMO_CEQ_mass.txt']

	assert params['mass'] == pytest.approx([1.0475655, 78.5674136]), (
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
	assert params['radius'] == pytest.approx([0.7381504, 7.7409131]), (
		f'ATMO radius range mismatch: {params["radius"]}'
	)

def test_evolutionary_models_params_bhac_spot_check():
	"""Spot-check known coverage for the BHAC15 tracks+structure table."""
	params = seda.models.EvolutionaryModels('BHAC2015').params['BHAC15_tracks+structure.txt']

	assert params['mass'] == pytest.approx([10.4756551, 1466.5917205]), (
		f'BHAC mass range mismatch: {params["mass"]}'
	)
	assert params['age'] == pytest.approx([0.000489159, 10.007900987]), (
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
	assert params['radius'] == pytest.approx([0.8368797, 35.2365258]), (
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
	assert params['Mrad'] == pytest.approx([0.0, 1466.5917205]), (
		f'BHAC Mrad range mismatch: {params["Mrad"]}'
	)
	assert params['Rrad'] == pytest.approx([0.0, 16.9808720]), (
		f'BHAC Rrad range mismatch: {params["Rrad"]}'
	)
	assert params['k2conv'] == pytest.approx([0.00124, 0.4944]), (
		f'BHAC k2conv range mismatch: {params["k2conv"]}'
	)
	assert params['k2rad'] == pytest.approx([0.0, 0.3072]), (
		f'BHAC k2rad range mismatch: {params["k2rad"]}'
	)

# ----------------------------
# color_anomaly — literature regression values
# ----------------------------
# Photometry and reference colors are taken from peer-reviewed tables.
# Expected anomalies are computed from those published inputs.

# Faherty et al. 2016, ApJS, 225, 10 — Table 15 field/normal means (M7–L8)
_FAHERTY16_L1_JH = 0.81
_FAHERTY16_L3_JH = 0.96
_FAHERTY16_L4_JH = 1.07
_FAHERTY16_L4_JK = 1.74
_FAHERTY16_L5_JH = 1.09
_FAHERTY16_L5_JK = 1.75

# Faherty et al. 2016 — Table 17, 2MASS J00040288-6410358 (L1γ)
_FAHERTY16_00040288_JH = 0.96

# Suárez et al. 2023, ApJL, 954, L6 — Table 2 (2MASS J03552337+1133437)
_SUAREZ23_0355_J = 14.050
_SUAREZ23_0355_H = 12.530
_SUAREZ23_0355_KS = 11.526
_SUAREZ23_0355_JH = _SUAREZ23_0355_J - _SUAREZ23_0355_H  # 1.520 mag
_SUAREZ23_0355_JK = _SUAREZ23_0355_J - _SUAREZ23_0355_KS  # 2.524 mag
# Suárez et al. (2023) Table 2 low-gravity L5 mean J−Ks = 2.153 mag
_SUAREZ23_0355_JK_LOWG_REF = 2.153
_SUAREZ23_0355_JK_ANOM_LOWG = _SUAREZ23_0355_JK - _SUAREZ23_0355_JK_LOWG_REF  # 0.371

# Suárez et al. 2023, ApJL, 954, L6 — Table 2 (2MASS J00361617+1821104;
# inclination 51±9 deg from Vos et al. 2017, ApJ, 842, 78)
_SUAREZ23_0036_J = 12.466
_SUAREZ23_0036_KS = 11.058
_SUAREZ23_0036_JK = _SUAREZ23_0036_J - _SUAREZ23_0036_KS  # 1.408 mag
_SUAREZ23_0036_JK_ANOM_FIELD = _SUAREZ23_0036_JK - _FAHERTY16_L4_JK  # -0.332

# Suárez & Metchev 2022, MNRAS, 513, 5701 — table photometry for
# 2MASS J04151954-0935066 (T8; Burgasser et al. 2004)
_SUAREZ22_0415_J = 15.695
_SUAREZ22_0415_KS = 15.429
_SUAREZ22_0415_JK = _SUAREZ22_0415_J - _SUAREZ22_0415_KS  # 0.266 mag


def test_color_anomaly_faherty16_table17_00040288_jh():
	"""Faherty et al. (2016) Table 17 vs Table 15 field L1 mean."""
	anomaly = seda.phy_params.color_anomaly(
		color=_FAHERTY16_00040288_JH,
		color_name='J-H',
		spt='L1',
		table='faherty16',
		age_group='old',
	)
	expected = _FAHERTY16_00040288_JH - _FAHERTY16_L1_JH
	assert anomaly == pytest.approx(expected, abs=0.01), (
		'Faherty L1 J-H anomaly outside tolerable range'
	)


def test_color_anomaly_suarez23_j0355_jk_vs_faherty_field():
	"""Suárez et al. (2023) photometry vs Faherty field L5 (Tables 15–16)."""
	anomaly = seda.phy_params.color_anomaly(
		color=_SUAREZ23_0355_JK,
		color_name='J-K',
		spt='L5',
		table='faherty16',
		age_group='old',
	)
	expected = _SUAREZ23_0355_JK - _FAHERTY16_L5_JK
	assert anomaly == pytest.approx(expected, abs=0.01), (
		'J0355 J-K field anomaly outside tolerable range'
	)


def test_color_anomaly_suarez23_j0355_jh_vs_faherty_field():
	"""Suárez et al. (2023) J−H for J0355+1133 vs Faherty field L5 mean."""
	anomaly = seda.phy_params.color_anomaly(
		color=_SUAREZ23_0355_JH,
		color_name='J-H',
		spt='L5',
		table='faherty16',
		age_group='old',
	)
	expected = _SUAREZ23_0355_JH - _FAHERTY16_L5_JH
	assert anomaly == pytest.approx(expected, abs=0.01), (
		'J0355 J-H field anomaly outside tolerable range'
	)


def test_color_anomaly_suarez23_j0355_jk_young_lowg_reference():
	"""Suárez et al. (2023) low-gravity reference for J0355+1133 (Table 2)."""
	anomaly = seda.phy_params.color_anomaly(
		color=_SUAREZ23_0355_JK,
		color_name='J-K',
		spt='L5',
		table='faherty16',
		age_group='young',
	)
	# Suárez et al. (2023) Table 2: low-gravity L5 mean J-K = 2.153 mag.
	assert anomaly == pytest.approx(_SUAREZ23_0355_JK_ANOM_LOWG, abs=0.01), (
		'J0355 young J-K anomaly outside tolerable range'
	)


def test_color_anomaly_suarez23_vos17_0036_jk_vs_faherty_field():
	"""Suárez et al. (2023) photometry for Vos et al. (2017) 0036+1821."""
	anomaly = seda.phy_params.color_anomaly(
		color=_SUAREZ23_0036_JK,
		color_name='J-K',
		spt='L4',
		table='faherty16',
		age_group='old',
	)
	assert anomaly == pytest.approx(_SUAREZ23_0036_JK_ANOM_FIELD, abs=0.01), (
		'0036 J-K field anomaly outside tolerable range'
	)


def test_color_anomaly_faherty16_table15_spt_interpolation():
	"""Fractional SpT uses Faherty et al. (2016) Table 15 L3/L4 J−H means."""
	# L3.7 → 0.3*L3 + 0.7*L4 (Table 15: L3=0.96, L4=1.07 mag)
	ref_l37 = 0.3 * _FAHERTY16_L3_JH + 0.7 * _FAHERTY16_L4_JH
	observed = 1.10
	anomaly = seda.phy_params.color_anomaly(
		color=observed,
		color_name='J-H',
		spt='L3.7',
		table='faherty16',
		age_group='old',
	)
	assert anomaly == pytest.approx(observed - ref_l37, abs=1e-6), (
		'L3.7 J-H interpolated anomaly outside tolerable range'
	)


def test_color_anomaly_suarez22_0415_t8_ultracool():
	"""Suárez & Metchev (2022) T8 photometry with Ultracool Sheet backend."""
	anomaly = seda.phy_params.color_anomaly(
		color=_SUAREZ22_0415_JK,
		color_name='J-K',
		spt='T8',
		table='ultracool',
	)
	# Regression value from Suárez & Metchev (2022) photometry and the bundled
	# Ultracool Sheet reference (recomputed; not a published mean sequence).
	assert anomaly == pytest.approx(0.354, abs=0.01), (
		'T8 ultracool J-K anomaly outside tolerable range'
	)


def test_color_anomaly_with_uncertainty():
	"""Suárez et al. (2023) Table 2 J and Ks uncertainties for J0355+1133."""
	ecolor = float(np.hypot(0.024, 0.021))
	out = seda.phy_params.color_anomaly(
		color=_SUAREZ23_0355_JK,
		color_name='J-K',
		spt='L5',
		table='faherty16',
		age_group='old',
		ecolor=ecolor,
	)
	assert out[1] == pytest.approx(ecolor, abs=1e-6), (
		'returned ecolor outside tolerable range'
	)
	assert isinstance(out, tuple), (
		'color_anomaly with ecolor did not return a tuple'
	)
	with pytest.raises(ValueError, match="table='faherty16' covers"):
		seda.phy_params.color_anomaly(
			color=1.0, color_name='J-H', spt='T5',
			table='faherty16', age_group='old',
		)


def test_color_anomaly_ultracool_t_type_succeeds():
	ref = seda.empirical_aux._loaders.reference_color(
		'J-H', 'T5', table='ultracool',
	)
	anomaly = seda.phy_params.color_anomaly(
		color=ref + 0.1, color_name='J-H', spt='T5', table='ultracool',
	)
	assert anomaly == pytest.approx(0.1, abs=0.01), (
		'ultracool T5 offset outside tolerable range'
	)


def test_color_anomaly_faherty_rejects_invalid_age_group():
	with pytest.raises(ValueError, match='age_group'):
		seda.phy_params.color_anomaly(
			color=1.0, color_name='J-H', spt='L5',
			table='faherty16', age_group='not-a-group',
		)


def test_color_anomaly_faherty_requires_age_group():
	"""Faherty tables require an explicit old or young age group."""
	with pytest.raises(ValueError, match='combined young/old sample'):
		seda.empirical_aux._loaders.reference_color(
			'J-H', 'L5', table='faherty16',
		)
	with pytest.raises(ValueError, match='combined young/old sample'):
		seda.phy_params.color_anomaly(
			color=1.0, color_name='J-H', spt='L5', table='faherty16',
		)
	ref_old = seda.empirical_aux._loaders.reference_color(
		'J-H', 'L5', table='faherty16', age_group='old',
	)
	assert ref_old == pytest.approx(_FAHERTY16_L5_JH, abs=0.01), (
		'old reference color outside tolerable range'
	)


def test_color_anomaly_faherty_young_redder_than_field():
	"""Suárez et al. (2023) J0355+1133 is redder vs field than vs young sequence."""
	anomaly_field = seda.phy_params.color_anomaly(
		color=_SUAREZ23_0355_JK,
		color_name='J-K',
		spt='L5',
		table='faherty16',
		age_group='old',
	)
	anomaly_young = seda.phy_params.color_anomaly(
		color=_SUAREZ23_0355_JK,
		color_name='J-K',
		spt='L5',
		table='faherty16',
		age_group='young',
	)
	assert anomaly_field == pytest.approx(
		_SUAREZ23_0355_JK - _FAHERTY16_L5_JK, abs=0.01,
	), 'J0355 field J-K anomaly outside tolerable range'
	assert anomaly_young < anomaly_field, (
		'young anomaly not less than field anomaly'
	)


def test_color_anomaly_faherty_young_rejects_field_only_reference_stat():
	# 'old' must use the published mean only.
	with pytest.raises(ValueError, match='reference_stat'):
		seda.phy_params.color_anomaly(
			color=1.0, color_name='J-H', spt='L5',
			table='faherty16', age_group='old', reference_stat='median',
		)


def test_color_anomaly_faherty_young_accepts_median():
	"""Young sequence can use median; Suárez et al. (2023) J0355+1133 at L5."""
	ref = seda.empirical_aux._loaders.reference_color(
		'J-H', 'L5', table='faherty16', age_group='young',
		reference_stat='median',
	)
	anomaly = seda.phy_params.color_anomaly(
		color=_SUAREZ23_0355_JH,
		color_name='J-H',
		spt='L5',
		table='faherty16',
		age_group='young',
		reference_stat='median',
	)
	assert anomaly == pytest.approx(_SUAREZ23_0355_JH - ref, abs=0.01), (
		'young median J-H anomaly outside tolerable range'
	)


def test_color_anomaly_faherty_young_l6_interpolates_l5_l7():
	"""L6 young reference interpolates between L5 and L7 when the L6 bin is sparse."""
	grid = seda.empirical_aux._loaders._faherty_young_reference('J-K', 'mean')
	expected = 0.5 * grid[15] + 0.5 * grid[17]
	with pytest.warns(UserWarning, match='L6 has insufficient objects.*L5.*L7'):
		ref = seda.empirical_aux._loaders.reference_color(
			'J-K', 'L6', table='faherty16', age_group='young',
		)
	assert ref == pytest.approx(expected, abs=1e-12), (
		'L6 young J-K reference not interpolated from L5 and L7'
	)
	with pytest.warns(UserWarning, match='L6 has insufficient objects.*L5.*L7'):
		anomaly = seda.phy_params.color_anomaly(
			color=_SUAREZ23_0355_JK,
			color_name='J-K',
			spt='L6',
			table='faherty16',
			age_group='young',
		)
	assert anomaly == pytest.approx(_SUAREZ23_0355_JK - expected, abs=1e-12), (
		'L6 young J-K anomaly inconsistent with interpolated reference'
	)


def test_color_anomaly_faherty_young_l8_uses_sparse_bin():
	"""L8 young reference uses sparse Table 1 bins (one object) when present."""
	bins_jh = seda.empirical_aux._loaders._faherty_young_reference_bins('J-H')
	bins_jk = seda.empirical_aux._loaders._faherty_young_reference_bins('J-K')
	with pytest.warns(UserWarning, match='L8 has only 1 object'):
		ref_jh = seda.empirical_aux._loaders.reference_color(
			'J-H', 'L8', table='faherty16', age_group='young',
		)
	with pytest.warns(UserWarning, match='L8 has only 1 object'):
		ref_jk = seda.empirical_aux._loaders.reference_color(
			'J-K', 'L8', table='faherty16', age_group='young',
		)
	assert ref_jh == pytest.approx(float(np.mean(bins_jh[18])), abs=1e-12), (
		'L8 young J-H reference not taken from the single available object'
	)
	assert ref_jk == pytest.approx(float(np.mean(bins_jk[18])), abs=1e-12), (
		'L8 young J-K reference not taken from the single available object'
	)


def test_color_anomaly_faherty_young_insufficient_data_raises():
	# Outside the Faherty M7-L8 range; no young reference can be built.
	with pytest.raises(ValueError, match="table='faherty16' covers"):
		seda.empirical_aux._loaders.reference_color(
			'J-H', 'T5', table='faherty16', age_group='young',
		)


def test_faherty_low_gravity_label_excludes_ambiguous():
	from seda.empirical_aux._loaders import _is_low_gravity

	tokens = ['beta', 'gamma', 'delta']
	assert _is_low_gravity('gamma', tokens) is True, (
		'gamma label not recognized as low gravity'
	)
	assert _is_low_gravity('beta gamma', tokens) is True, (
		'beta gamma label not recognized as low gravity'
	)
	assert _is_low_gravity('gamma?', tokens) is False, (
		'ambiguous gamma? label incorrectly accepted'
	)
	assert _is_low_gravity('-', tokens) is False, (
		'field dash label incorrectly accepted as low gravity'
	)
	assert _is_low_gravity('', tokens) is False, (
		'empty gravity label incorrectly accepted'
	)


def test_color_anomaly_requires_table():
	with pytest.raises(ValueError, match='table must be specified'):
		seda.phy_params.color_anomaly(
			color=1.0, color_name='J-H', spt='L5', table=None,
		)
	with pytest.raises(ValueError, match='table must be specified'):
		seda.phy_params.color_anomaly(
			color=1.0, color_name='J-H', spt='L5', table='',
		)


def test_color_anomaly_invalid_color_raises():
	with pytest.raises(ValueError, match='color_name'):
		seda.phy_params.color_anomaly(
			color=1.0, color_name='Y-J', spt='L5',
			table='faherty16', age_group='old',
		)


def test_ultracool_youth_filter_excludes_ambiguous():
	from seda.empirical_aux._loaders import _matches_age_group

	assert _matches_age_group('YMG', 'young') is True, (
		'YMG not matched to young age group'
	)
	assert _matches_age_group('YMG?', 'young') is False, (
		'ambiguous YMG? incorrectly matched to young'
	)
	assert _matches_age_group('YMG?', 'old') is False, (
		'ambiguous YMG? incorrectly matched to old'
	)
	assert _matches_age_group('N', 'old') is True, (
		'field N not matched to old age group'
	)
	assert _matches_age_group('Field', 'old') is True, (
		'Field label not matched to old age group'
	)


# ----------------------------
# inclination
# ----------------------------
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
