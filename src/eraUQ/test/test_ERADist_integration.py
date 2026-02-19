"""
Integration tests for ERADist: round-trip consistency across PAR, MOM, and DATA.

Implements TEST_STRATEGY_ERADist.md §3.1:
- PAR → MOM → PAR: create via PAR, compute mean/std, recreate via MOM, verify parameters match.
- PAR → DATA → PAR: create via PAR, generate large sample, fit via DATA, verify parameters match.
- MOM → DATA → MOM: create via MOM, generate sample, fit via DATA, verify moments match.
"""
import pytest
import numpy as np

from eraUQ import ERADist


np.random.seed(42)

# Build MOM input from (mean, std, par_val) for PAR → MOM → PAR
_MOM_FROM_PAR = {
    "chisquare": lambda m, s, p: [m],
    "exponential": lambda m, s, p: [m],
    "geometric": lambda m, s, p: [m],
    "rayleigh": lambda m, s, p: [m],
    "poisson": lambda m, s, p: [m],
    "standardnormal": lambda m, s, p: [],
    "beta": lambda m, s, p: [m, s, p[2], p[3]],
    "gev": lambda m, s, p: [m, s, p[0]],
    "gevmin": lambda m, s, p: [m, s, p[0]],
    "truncatednormal": lambda m, s, p: [m, s, p[2], p[3]],
}


def _assert_par_close(par_expected, par_actual, rtol=1e-4, atol=1e-10):
    """Assert that parameter dicts match (same keys, values close)."""
    for key in par_expected:
        if key not in par_actual:
            continue
        a, b = par_expected[key], par_actual[key]
        if isinstance(a, (int, np.integer)):
            assert a == b, f"Parameter {key}: {a} != {b}"
        else:
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol, err_msg=f"Parameter {key}")


# --- PAR → MOM → PAR ---

@pytest.mark.parametrize("dist_name,par_val", [
    ("normal", [0.0, 1.0]),
    ("exponential", [1.5]),
    ("lognormal", [0.0, 1.0]),
    ("uniform", [0.0, 1.0]),
    ("gamma", [2.0, 3.0]),
    ("rayleigh", [2.0]),
    ("chisquare", [3.0]),
    ("gev", [0.2, 1.0, 0.0]),
    ("gevmin", [0.2, 1.0, 0.0]),
    ("gumbel", [1.0, 0.5]),
    ("gumbelmin", [1.0, 0.5]),
    ("beta", [2.0, 5.0, 0.0, 1.0]),
    ("geometric", [0.4]),
], ids=[
    "normal", "exponential", "lognormal", "uniform", "gamma", "rayleigh",
    "chisquare", "gev", "gevmin", "gumbel", "gumbelmin", "beta", "geometric",
])
def test_par_mom_par_round_trip(dist_name, par_val, rtol=1e-4):
    """Create via PAR → compute mean/std → recreate via MOM → verify parameters match."""
    dist_par = ERADist(dist_name, "PAR", par_val)
    mu, sigma = dist_par.mean(), dist_par.std()
    builder = _MOM_FROM_PAR.get(dist_name.lower(), lambda mu, sigma, p: [mu, sigma])
    mom_val = builder(mu, sigma, par_val)
    dist_mom = ERADist(dist_name, "MOM", mom_val)
    _assert_par_close(dist_par.Par, dist_mom.Par, rtol=rtol)


# --- PAR → DATA → PAR ---

_N_DATA = 10_000  # large sample for MLE tolerance

@pytest.mark.parametrize("dist_name,par_val,rtol,atol", [
    ("normal", [0.0, 1.0], 0.03, 0.01),  # mu can be 0; sample mean has sampling error
    ("exponential", [1.5], 0.03, 1e-10),
    ("gamma", [2.0, 3.0], 0.08, 1e-10),
    ("lognormal", [0.0, 1.0], 0.05, 0.02),  # mu_lnx can be 0; sampling error
    ("weibull", [1.0, 2.0], 0.05, 1e-10),
    ("rayleigh", [2.0], 0.05, 1e-10),
    ("chisquare", [3.0], 0.08, 1e-10),
    ("gumbel", [1.0, 0.5], 0.05, 1e-10),
    ("gumbelmin", [1.0, 0.5], 0.05, 1e-10),
    ("uniform", [0.0, 1.0], 0.05, 0.05),  # min/max of sample vary
], ids=[
    "normal", "exponential", "gamma", "lognormal", "weibull",
    "rayleigh", "chisquare", "gumbel", "gumbelmin", "uniform",
])
def test_par_data_par_round_trip(dist_name, par_val, rtol, atol):
    """Create via PAR → generate n=10000 sample → fit via DATA → verify parameters match."""
    dist_par = ERADist(dist_name, "PAR", par_val)
    X = dist_par.random(size=_N_DATA)
    dist_data = ERADist(dist_name, "DATA", X)
    _assert_par_close(dist_par.Par, dist_data.Par, rtol=rtol, atol=atol)


# --- MOM → DATA → MOM ---

_N_MOM_DATA = 5_000

@pytest.mark.parametrize("dist_name,mom_val", [
    ("normal", [0.0, 1.0]),
    ("exponential", [2.0]),
    ("lognormal", [2.0, 1.0]),
    ("gamma", [6.0, 2.0]),
    ("weibull", [2.0, 1.0]),
    ("rayleigh", [2.0]),
    ("gumbel", [2.0, 1.0]),
    ("gumbelmin", [2.0, 1.0]),
], ids=["normal", "exponential", "lognormal", "gamma", "weibull", "rayleigh", "gumbel", "gumbelmin"])
def test_mom_data_mom_round_trip(dist_name, mom_val, rtol=0.08, atol=0.05):
    """Create via MOM → generate sample → fit via DATA → verify mean/std match original."""
    dist_mom = ERADist(dist_name, "MOM", mom_val)
    mean_orig = dist_mom.mean()
    std_orig = dist_mom.std()
    X = dist_mom.random(size=_N_MOM_DATA)
    dist_data = ERADist(dist_name, "DATA", X)
    mean_fit = dist_data.mean()
    std_fit = dist_data.std()
    # atol needed when mean_orig=0 (rtol undefined) or for small magnitudes
    np.testing.assert_allclose(mean_fit, mean_orig, rtol=rtol, atol=atol, err_msg="mean mismatch")
    np.testing.assert_allclose(std_fit, std_orig, rtol=rtol, atol=atol, err_msg="std mismatch")
