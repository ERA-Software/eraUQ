"""
Data fitting mode (DATA) constructor tests for ERADist.
Covers: MLE/estimation from samples, fitted params within tolerance,
invalid data, small/large sample sizes, and key cases (Beta, Binomial, Poisson
both forms, TruncatedNormal, Empirical, GEV/GEVMin, Pareto).
"""
import pytest
import numpy as np
from scipy import stats

from eraUQ import ERADist


# --- Helpers ---

def assert_has_par_and_dist(obj):
    assert hasattr(obj, "Par") and isinstance(obj.Par, dict)
    assert hasattr(obj, "Dist") and obj.Dist is not None


# Use fixed seed for reproducible samples
np.random.seed(42)


# --- Fit and verify: generate samples, fit, check params within tolerance ---

# Sample size 500 for fit tests: reduces MLE sampling variance so tolerances can be reasonable
_FIT_N = 500

@pytest.mark.parametrize("name,data_arg,true_par_check", [
    ("chisquare", lambda: stats.chi2.rvs(5, size=_FIT_N), lambda p: np.isclose(p["k"], 5, atol=1.5)),
    ("exponential", lambda: stats.expon.rvs(scale=1.5, size=_FIT_N), lambda p: np.isclose(p["lambda"], 1/1.5, rtol=0.15)),
    ("gamma", lambda: stats.gamma.rvs(2, scale=1/3, size=_FIT_N), lambda p: np.isclose(p["k"], 2, atol=0.5) and np.isclose(1/p["lambda"], 1/3, rtol=0.2)),
    ("geometric", lambda: stats.geom.rvs(0.3, size=_FIT_N).astype(float), lambda p: np.isclose(p["p"], 0.3, rtol=0.2)),
    ("gumbel", lambda: stats.gumbel_r.rvs(1, 0.5, size=_FIT_N), lambda p: np.isclose(p["b_n"], 1, atol=0.2) and np.isclose(p["a_n"], 0.5, atol=0.15)),
    ("gumbelmin", lambda: stats.gumbel_l.rvs(1, 0.5, size=_FIT_N), lambda p: np.isclose(p["b_n"], 1, atol=0.2) and np.isclose(p["a_n"], 0.5, atol=0.15)),
    ("lognormal", lambda: stats.lognorm.rvs(0.5, scale=np.exp(0.5), size=_FIT_N), lambda p: np.isclose(p["sig_lnx"], 0.5, atol=0.15) and np.isclose(p["mu_lnx"], 0.5, atol=0.2)),
    ("normal", lambda: stats.norm.rvs(1, 2, size=_FIT_N), lambda p: np.isclose(p["mu"], 1, atol=0.2) and np.isclose(p["sigma"], 2, atol=0.25)),
    ("rayleigh", lambda: stats.rayleigh.rvs(2, size=_FIT_N), lambda p: np.isclose(p["alpha"], 2, rtol=0.2)),
    ("uniform", lambda: stats.uniform.rvs(1, 4, size=_FIT_N), lambda p: np.isclose(p["lower"], 1, atol=0.1) and np.isclose(p["upper"], 5, atol=0.1)),
    ("weibull", lambda: stats.weibull_min.rvs(2, scale=1.5, size=_FIT_N), lambda p: np.isclose(p["k"], 2, atol=0.4) and np.isclose(p["a_n"], 1.5, rtol=0.2)),
], ids=["chisquare", "exponential", "gamma", "geometric", "gumbel", "gumbelmin", "lognormal", "normal", "rayleigh", "uniform", "weibull"])
def test_data_fit_and_reasonable_params(name, data_arg, true_par_check):
    X = data_arg()
    obj = ERADist(name, "DATA", X)
    assert_has_par_and_dist(obj)
    assert true_par_check(obj.Par)


# --- Beta: support bounds [[X], a, b], data in [a,b] ---

def test_data_beta_valid():
    a, b = 0.0, 1.0
    X = stats.beta.rvs(2, 5, loc=a, scale=b - a, size=200)
    obj = ERADist("beta", "DATA", [X, a, b])
    assert_has_par_and_dist(obj)
    assert obj.Par["a"] == a and obj.Par["b"] == b
    assert obj.Par["r"] > 0 and obj.Par["s"] > 0


def test_data_beta_invalid_support_a_ge_b():
    X = np.array([0.3, 0.5, 0.7])
    with pytest.raises(RuntimeError, match="Please select a different support"):
        ERADist("beta", "DATA", [X, 0.5, 0.5])
    with pytest.raises(RuntimeError, match="Please select a different support"):
        ERADist("beta", "DATA", [X, 1.0, 0.0])


def test_data_beta_invalid_data_outside_range():
    X = np.array([0.2, 0.5, 1.2])  # 1.2 > 1
    with pytest.raises(RuntimeError, match="samples must be in the support range"):
        ERADist("beta", "DATA", [X, 0.0, 1.0])
    X2 = np.array([-0.1, 0.5])
    with pytest.raises(RuntimeError, match="samples must be in the support range"):
        ERADist("beta", "DATA", [X2, 0.0, 1.0])


# --- Binomial: [[X], n], integer data in [0,n] ---

def test_data_binomial_valid():
    n = 10
    X = stats.binom.rvs(n, 0.4, size=200).astype(float)
    obj = ERADist("binomial", "DATA", [X, n])
    assert_has_par_and_dist(obj)
    assert obj.Par["n"] == n
    assert 0 <= obj.Par["p"] <= 1
    np.testing.assert_allclose(obj.Par["p"], np.mean(X) / n, atol=0.05)


def test_data_binomial_invalid_n_non_integer():
    X = np.array([1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="n must be a positive integer"):
        ERADist("binomial", "DATA", [X, 10.7])


def test_data_binomial_invalid_data_non_integer():
    X = np.array([1.5, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="samples must be integers in the range"):
        ERADist("binomial", "DATA", [X, 10])


def test_data_binomial_invalid_data_out_of_range():
    X = np.array([1.0, 11.0, 3.0])  # 11 > n=10
    with pytest.raises(RuntimeError, match="samples must be integers in the range"):
        ERADist("binomial", "DATA", [X, 10])
    X2 = np.array([-1.0, 2.0])
    with pytest.raises(RuntimeError, match="samples must be integers in the range"):
        ERADist("binomial", "DATA", [X2, 10])


# --- Poisson: [X] and [[X], t] forms ---

def test_data_poisson_lambda_form():
    X = stats.poisson.rvs(3, size=200).astype(float)
    obj = ERADist("poisson", "DATA", X)
    assert_has_par_and_dist(obj)
    assert "lambda" in obj.Par
    np.testing.assert_allclose(obj.Par["lambda"], 3, atol=0.3)


def test_data_poisson_vt_form():
    t = 1.5
    # counts per interval with rate v*t
    v_true = 2.0
    X = stats.poisson.rvs(v_true * t, size=200).astype(float)
    obj = ERADist("poisson", "DATA", [X, t])
    assert_has_par_and_dist(obj)
    assert obj.Par["t"] == t
    np.testing.assert_allclose(obj.Par["v"], v_true, rtol=0.2)


def test_data_poisson_invalid_negative():
    with pytest.raises(RuntimeError, match="non-negative integers"):
        ERADist("poisson", "DATA", np.array([1.0, -1.0, 2.0]))


def test_data_poisson_invalid_non_integer():
    with pytest.raises(RuntimeError, match="non-negative integers"):
        ERADist("poisson", "DATA", np.array([1.0, 2.5, 3.0]))


def test_data_poisson_vt_invalid_t_non_positive():
    X = np.array([1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="t must be positive"):
        ERADist("poisson", "DATA", [X, 0])
    with pytest.raises(RuntimeError, match="t must be positive"):
        ERADist("poisson", "DATA", [X, -1])


# --- TruncatedNormal: [[X], a, b], data in [a,b] ---

def test_data_truncatednormal_valid():
    a, b = -2.0, 2.0
    mu, sig = 0.0, 1.0
    # sample by rejection
    X = stats.truncnorm.rvs((a - mu) / sig, (b - mu) / sig, loc=mu, scale=sig, size=200)
    obj = ERADist("truncatednormal", "DATA", [X, a, b])
    assert_has_par_and_dist(obj)
    assert obj.Par["a"] == a and obj.Par["b"] == b
    assert obj.Par["sig_n"] > 0


def test_data_truncatednormal_invalid_a_ge_b():
    X = np.array([0.0, 0.5, 1.0])
    with pytest.raises(RuntimeError, match="upper bound a must be larger than the lower bound b"):
        ERADist("truncatednormal", "DATA", [X, 1.0, 1.0])
    with pytest.raises(RuntimeError, match="upper bound a must be larger than the lower bound b"):
        ERADist("truncatednormal", "DATA", [X, 2.0, -2.0])


def test_data_truncatednormal_invalid_data_outside_range():
    X = np.array([0.0, 0.5, 3.0])  # 3 > b=2
    with pytest.raises(RuntimeError, match="samples must be in the range"):
        ERADist("truncatednormal", "DATA", [X, -2.0, 2.0])
    X2 = np.array([-3.0, 0.0])
    with pytest.raises(RuntimeError, match="samples must be in the range"):
        ERADist("truncatednormal", "DATA", [X2, -2.0, 2.0])


# --- Empirical: with/without weights, different pdfMethods ---

def test_data_empirical_basic():
    X = np.random.randn(100)
    # val = [X, weights, pdfMethod, pdfPoints, dict]
    obj = ERADist("empirical", "DATA", [X, None, "kde", None, {}])
    assert_has_par_and_dist(obj)


def test_data_empirical_pdf_method_linear():
    X = np.random.randn(80)
    obj = ERADist("empirical", "DATA", [X, None, "linear", 50, {}])
    assert_has_par_and_dist(obj)
    assert obj.Par["pdfMethod"] == "linear"


def test_data_empirical_with_weights():
    X = np.random.randn(60)
    w = np.ones(60) / 60
    obj = ERADist("empirical", "DATA", [X, w, "kde", None, {}])
    assert_has_par_and_dist(obj)
    assert obj.Par["weights"] is not None


# --- GEV / GEVMin: gevfit_alt convergence ---

def test_data_gev_valid():
    X = stats.genextreme.rvs(-0.2, loc=1, scale=0.5, size=300)
    obj = ERADist("gev", "DATA", X)
    assert_has_par_and_dist(obj)
    assert "beta" in obj.Par and "alpha" in obj.Par and "epsilon" in obj.Par


def test_data_gevmin_valid():
    X = -stats.genextreme.rvs(-0.2, loc=1, scale=0.5, size=300)
    obj = ERADist("gevmin", "DATA", X)
    assert_has_par_and_dist(obj)
    assert "beta" in obj.Par and "alpha" in obj.Par and "epsilon" in obj.Par


# --- Pareto: x_m = min(data) ---

def test_data_pareto_valid():
    x_m = 1.0
    alpha = 2.5
    # genpareto c=1/alpha, scale=x_m/alpha, loc=x_m
    X = stats.genpareto.rvs(1 / alpha, scale=x_m / alpha, loc=x_m, size=200)
    obj = ERADist("pareto", "DATA", X)
    assert_has_par_and_dist(obj)
    # ERADist sets x_m = min(data); sample min is random >= loc
    np.testing.assert_allclose(obj.Par["x_m"], np.min(X))
    np.testing.assert_allclose(obj.Par["x_m"], x_m, atol=0.1)
    np.testing.assert_allclose(obj.Par["alpha"], alpha, rtol=0.25)


def test_data_pareto_invalid_non_positive():
    X = np.array([0.0, 1.0, 2.0])
    with pytest.raises(RuntimeError, match="The given data must be positive"):
        ERADist("pareto", "DATA", X)
    X2 = np.array([-1.0, 2.0])
    with pytest.raises(RuntimeError, match="The given data must be positive"):
        ERADist("pareto", "DATA", X2)


# --- Invalid data: negative / out-of-bounds / non-integer where required ---

@pytest.mark.parametrize("name,data,match", [
    ("chisquare", np.array([1.0, -0.1, 2.0]), "non-negative"),
    ("exponential", np.array([1.0, -0.5, 2.0]), "non-negative"),
    ("frechet", np.array([1.0, -0.1, 2.0]), "non-negative"),
    ("geometric", np.array([1.0, 2.5, 3.0]), "integers larger than 0"),
    ("geometric", np.array([0.0, 1.0, 2.0]), "integers larger than 0"),
], ids=["chisquare_neg", "exp_neg", "frechet_neg", "geom_nonint", "geom_zero"])
def test_data_invalid_raises(name, data, match):
    with pytest.raises(RuntimeError, match=match):
        ERADist(name, "DATA", data)


# --- Sample sizes: small (n<10) and large (n>10000) ---

def test_data_normal_small_sample():
    X = stats.norm.rvs(0, 1, size=5)
    obj = ERADist("normal", "DATA", X)
    assert_has_par_and_dist(obj)
    assert obj.Par["mu"] is not None and obj.Par["sigma"] > 0


def test_data_normal_large_sample():
    X = stats.norm.rvs(0, 1, size=15000)
    obj = ERADist("normal", "DATA", X)
    assert_has_par_and_dist(obj)
    np.testing.assert_allclose(obj.Par["mu"], 0, atol=0.03)
    np.testing.assert_allclose(obj.Par["sigma"], 1, atol=0.03)


def test_data_exponential_small_sample():
    X = stats.expon.rvs(scale=2, size=5)
    obj = ERADist("exponential", "DATA", X)
    assert_has_par_and_dist(obj)
    assert obj.Par["lambda"] > 0


def test_data_exponential_large_sample():
    X = stats.expon.rvs(scale=2, size=12000)
    obj = ERADist("exponential", "DATA", X)
    assert_has_par_and_dist(obj)
    np.testing.assert_allclose(1 / obj.Par["lambda"], 2, rtol=0.05)


# --- Negative binomial: can fail for unsuitable data ---

def test_data_negativebinomial_valid():
    # Use params that give reasonable variance
    k, p = 5, 0.5
    X = stats.nbinom.rvs(k, p, size=200).astype(float)
    obj = ERADist("negativebinomial", "DATA", X)
    assert_has_par_and_dist(obj)
    assert obj.Par["k"] > 0 and 0 < obj.Par["p"] <= 1


def test_data_negativebinomial_invalid_zero_mean():
    X = np.zeros(50)
    with pytest.raises(RuntimeError, match="No suitable parameters can be estimated"):
        ERADist("negativebinomial", "DATA", X)


# --- Frechet: MLE can fail to converge ---

def test_data_frechet_valid():
    # Genextreme with c=-1/k gives Frechet; use k>1
    X = stats.genextreme.rvs(-0.5, loc=2, scale=1, size=200)
    obj = ERADist("frechet", "DATA", X)
    assert_has_par_and_dist(obj)
    assert obj.Par["a_n"] > 0 and obj.Par["k"] > 0
