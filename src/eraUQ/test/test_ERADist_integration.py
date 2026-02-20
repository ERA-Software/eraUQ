"""
Integration tests for ERADist: round-trip consistency across PAR, MOM, and DATA,
and consistency checks (mean/std vs sample, median, pdf integral).

Implements TEST_STRATEGY_ERADist.md §3.1 and §3.2:
- PAR → MOM → PAR, PAR → DATA → PAR, MOM → DATA → MOM.
- mean() vs np.mean(random(10000)), std() vs np.std(...), cdf(icdf(0.5)) ≈ 0.5,
  pdf integrates to 1 (continuous only).
"""
import pytest
import numpy as np
from scipy import integrate

from eraUQ import ERADist


np.random.seed(42)

# Distributions that are discrete (no pdf integral check)
_DISCRETE = {"binomial", "geometric", "poisson", "negativebinomial"}
# Heavy-tailed / high variance: sample std converges slowly; use looser tolerance in consistency
_HEAVY_TAILED_STD = {"pareto", "frechet", "gev", "gevmin"}
_N_CONSISTENCY = 10_000

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


# --- §3.2 Consistency tests: mean/std vs sample, cdf(icdf(0.5)) ≈ 0.5, pdf integral ---

def _check_consistency(obj, n_sample=_N_CONSISTENCY, mean_std_rtol=0.05, mean_std_atol=0.05, median_atol=0.05, check_pdf_integral=True):
    """Verify mean()/std() match sample, cdf(icdf(0.5)) ≈ 0.5, and (if continuous) pdf integrates to 1."""
    sample = obj.random(size=n_sample)
    # Heavy-tailed: sample std converges slowly; use looser tolerance
    std_rtol = 0.4 if obj.Name in _HEAVY_TAILED_STD else mean_std_rtol
    std_atol = 0.5 if obj.Name in _HEAVY_TAILED_STD else mean_std_atol
    np.testing.assert_allclose(obj.mean(), np.mean(sample), rtol=mean_std_rtol, atol=mean_std_atol, err_msg="mean vs sample mean")
    np.testing.assert_allclose(obj.std(), np.std(sample, ddof=0), rtol=std_rtol, atol=std_atol, err_msg="std vs sample std")
    median_val = obj.icdf(0.5)
    cdf_at_median = obj.cdf(median_val)
    if obj.Name in _DISCRETE:
        # Discrete: icdf(0.5) is smallest x with cdf(x) >= 0.5, so cdf(icdf(0.5)) >= 0.5
        assert cdf_at_median >= 0.5 - median_atol, f"cdf(icdf(0.5)) >= 0.5 for discrete: got {cdf_at_median}"
    else:
        np.testing.assert_allclose(cdf_at_median, 0.5, atol=median_atol, err_msg="cdf(icdf(0.5)) ≈ 0.5")
    if check_pdf_integral and obj.Name not in _DISCRETE:
        # Integrate pdf over (icdf(0.001), icdf(0.999)) for continuous
        lo, hi = obj.icdf(0.001), obj.icdf(0.999)
        integral, _ = integrate.quad(obj.pdf, lo, hi, limit=200)
        np.testing.assert_allclose(integral, 1.0, rtol=0.02, atol=0.02, err_msg="pdf integral ≈ 1")


# Consistency for PAR mode
_CONSISTENCY_PAR = [
    ("normal", [0.0, 1.0]),
    ("exponential", [1.5]),
    ("gamma", [2.0, 3.0]),
    ("lognormal", [0.0, 1.0]),
    ("weibull", [1.0, 2.0]),
    ("rayleigh", [2.0]),
    ("chisquare", [3.0]),
    ("gumbel", [1.0, 0.5]),
    ("gumbelmin", [1.0, 0.5]),
    ("beta", [2.0, 5.0, 0.0, 1.0]),
    ("uniform", [0.0, 1.0]),
    ("gev", [0.2, 1.0, 0.0]),
    ("gevmin", [0.2, 1.0, 0.0]),
    ("binomial", [10, 0.5]),
    ("geometric", [0.4]),
    ("poisson", [3.0]),
    ("negativebinomial", [5.0, 0.5]),
    ("pareto", [1.0, 2.5]),
    ("truncatednormal", [0.0, 1.0, -2.0, 2.0]),
    ("standardnormal", []),
    ("frechet", [2.0, 3.0]),
]


@pytest.mark.parametrize("dist_name,par_val", _CONSISTENCY_PAR, ids=[x[0] for x in _CONSISTENCY_PAR])
def test_consistency_PAR(dist_name, par_val):
    """§3.2: mean/std vs sample, cdf(icdf(0.5)) ≈ 0.5, pdf integral (continuous)."""
    obj = ERADist(dist_name, "PAR", par_val)
    _check_consistency(obj, check_pdf_integral=True)


# Consistency for MOM mode (same distributions with MOM inputs)
_CONSISTENCY_MOM = [
    ("normal", [0.0, 1.0]),
    ("exponential", [2.0]),
    ("gamma", [6.0, 2.0]),
    ("lognormal", [2.0, 1.0]),
    ("weibull", [2.0, 1.0]),
    ("rayleigh", [2.0]),
    ("chisquare", [6.0]),
    ("gumbel", [2.0, 1.0]),
    ("gumbelmin", [2.0, 1.0]),
    ("beta", [0.3, 0.1, 0.0, 1.0]),
    ("uniform", [0.5, 0.3]),
    ("gev", [1.0, 0.5, 0.1]),
    ("gevmin", [1.0, 0.5, 0.1]),
    ("binomial", [9.0, 1.5]),
    ("geometric", [2.5]),
    ("poisson", [3.0]),
    ("negativebinomial", [5.0, np.sqrt(20)]),
    ("pareto", [2.0, 1.0]),
    ("truncatednormal", [0.0, 1.0, -2.0, 2.0]),
    ("standardnormal", []),
]


@pytest.mark.parametrize("dist_name,mom_val", _CONSISTENCY_MOM, ids=[x[0] for x in _CONSISTENCY_MOM])
def test_consistency_MOM(dist_name, mom_val):
    """§3.2: mean/std vs sample, cdf(icdf(0.5)) ≈ 0.5, pdf integral (continuous)."""
    obj = ERADist(dist_name, "MOM", mom_val)
    _check_consistency(obj, check_pdf_integral=True)


# Consistency for DATA mode: generate from PAR, fit DATA, then run same checks
_CONSISTENCY_DATA_PAR = [
    ("normal", [0.0, 1.0]),
    ("exponential", [1.5]),
    ("gamma", [2.0, 3.0]),
    ("lognormal", [0.0, 1.0]),
    ("weibull", [1.0, 2.0]),
    ("rayleigh", [2.0]),
    ("chisquare", [3.0]),
    ("gumbel", [1.0, 0.5]),
    ("gumbelmin", [1.0, 0.5]),
    ("uniform", [0.0, 1.0]),
    ("beta", [2.0, 5.0, 0.0, 1.0]),
    ("gev", [0.2, 1.0, 0.0]),
    ("gevmin", [0.2, 1.0, 0.0]),
    ("poisson", [3.0]),
    ("negativebinomial", [5.0, 0.5]),
    ("pareto", [1.0, 2.5]),
    ("truncatednormal", [0.0, 1.0, -2.0, 2.0]),
    ("binomial", [10, 0.5]),
    ("geometric", [0.4]),
]


def _data_arg_from_sample(name, X, par_val):
    """Build DATA constructor argument from sample X and (optional) PAR for bounds/n."""
    name = name.lower()
    if name == "beta":
        return [X, par_val[2], par_val[3]]
    if name == "binomial":
        return [X, par_val[0]]
    if name == "truncatednormal":
        return [X, par_val[2], par_val[3]]
    if name == "poisson":
        return X  # lambda form
    return X


@pytest.mark.parametrize("dist_name,par_val", _CONSISTENCY_DATA_PAR, ids=[x[0] for x in _CONSISTENCY_DATA_PAR])
def test_consistency_DATA(dist_name, par_val):
    """§3.2: build via DATA from sample generated by PAR; then mean/std, median, pdf integral."""
    ref = ERADist(dist_name, "PAR", par_val)
    X = ref.random(size=_N_CONSISTENCY)
    data_arg = _data_arg_from_sample(dist_name, X, par_val)
    obj = ERADist(dist_name, "DATA", data_arg)
    _check_consistency(obj, check_pdf_integral=True)
