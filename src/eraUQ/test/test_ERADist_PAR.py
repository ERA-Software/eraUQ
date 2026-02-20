"""
Parameter mode (PAR) constructor tests for ERADist.
Covers: valid/boundary/invalid params, self.Par structure, self.Dist creation,
and key cases (Beta, Binomial, Poisson both forms, TruncatedNormal, StandardNormal, case insensitivity).
"""
import pytest

from eraUQ import ERADist


# --- Helpers ---

def assert_has_par_and_dist(obj):
    assert hasattr(obj, "Par") and isinstance(obj.Par, dict)
    assert hasattr(obj, "Dist") and obj.Dist is not None


# --- Valid parameter sets (typical values) ---

@pytest.mark.parametrize("name,val", [
    ("beta", [2.0, 5.0, 0.0, 1.0]),
    ("binomial", [10, 0.5]),
    ("chisquare", [3.0]),
    ("exponential", [1.5]),
    ("frechet", [2.0, 3.0]),
    ("gamma", [2.0, 3.0]),
    ("geometric", [0.4]),
    ("gev", [0.2, 1.0, 0.0]),
    ("gevmin", [0.2, 1.0, 0.0]),
    ("gumbel", [1.0, 0.0]),
    ("gumbelmin", [1.0, 0.0]),
    ("lognormal", [0.0, 1.0]),
    ("negativebinomial", [5.0, 0.5]),
    ("normal", [0.0, 1.0]),
    ("pareto", [1.0, 2.5]),
    ("poisson", [3.0]),
    ("rayleigh", [2.0]),
    ("standardnormal", []),
    ("truncatednormal", [0.0, 1.0, -2.0, 2.0]),
    ("uniform", [0.0, 1.0]),
    ("weibull", [1.0, 2.0]),
])
def test_par_valid_typical(name, val):
    obj = ERADist(name, "PAR", val)
    assert_has_par_and_dist(obj)
    assert obj.Name == name.lower()


def test_par_poisson_lambda_form():
    obj = ERADist("poisson", "PAR", [3.0])
    assert_has_par_and_dist(obj)
    assert "lambda" in obj.Par
    assert obj.Par["lambda"] == 3.0


def test_par_poisson_vt_form():
    obj = ERADist("poisson", "PAR", [2.0, 1.5])
    assert_has_par_and_dist(obj)
    assert obj.Par["v"] == 2.0 and obj.Par["t"] == 1.5


# --- Beta: r>0, s>0, a<b ---

def test_par_beta_valid():
    obj = ERADist("beta", "PAR", [2.0, 5.0, 0.0, 1.0])
    assert obj.Par["r"] == 2.0 and obj.Par["s"] == 5.0
    assert obj.Par["a"] == 0.0 and obj.Par["b"] == 1.0
    assert_has_par_and_dist(obj)


def test_par_beta_boundary():
    obj = ERADist("beta", "PAR", [1.0, 1.0, -1.0, 1.0])
    assert obj.Par["r"] == 1.0 and obj.Par["s"] == 1.0
    assert obj.Par["a"] < obj.Par["b"]


@pytest.mark.parametrize("par_val", [[0.0, 5.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]], ids=["r_zero", "s_zero"])
def test_par_beta_invalid_r_s_zero(par_val):
    with pytest.raises(RuntimeError, match="not defined for your parameters"):
        ERADist("beta", "PAR", par_val)


@pytest.mark.parametrize("par_val", [[2.0, 5.0, 1.0, 1.0], [2.0, 5.0, 2.0, 1.0]], ids=["a_eq_b", "a_gt_b"])
def test_par_beta_invalid_a_ge_b(par_val):
    with pytest.raises(RuntimeError, match="not defined for your parameters"):
        ERADist("beta", "PAR", par_val)


# --- Binomial: n integer, 0<=p<=1 ---

def test_par_binomial_valid():
    obj = ERADist("binomial", "PAR", [10, 0.5])
    assert obj.Par["n"] == 10 and obj.Par["p"] == 0.5
    assert_has_par_and_dist(obj)


@pytest.mark.parametrize("par_val,expected_n,expected_p", [
    ([1, 0.0], 1, 0.0),
    ([5, 1.0], 5, 1.0),
], ids=["n1_p0", "n5_p1"])
def test_par_binomial_boundary(par_val, expected_n, expected_p):
    obj = ERADist("binomial", "PAR", par_val)
    assert obj.Par["n"] == expected_n and obj.Par["p"] == expected_p


@pytest.mark.parametrize("par_val", [[10, -0.1], [10, 1.1]], ids=["p_negative", "p_gt_one"])
def test_par_binomial_invalid_p_out_of_range(par_val):
    with pytest.raises(RuntimeError, match="not defined for your parameters"):
        ERADist("binomial", "PAR", par_val)


def test_par_binomial_invalid_n_non_integer():
    with pytest.raises(RuntimeError, match="not defined for your parameters"):
        ERADist("binomial", "PAR", [10.7, 0.5])


# --- TruncatedNormal: a<b, sigma>0 ---

def test_par_truncatednormal_valid():
    obj = ERADist("truncatednormal", "PAR", [0.0, 1.0, -2.0, 2.0])
    assert obj.Par["mu_n"] == 0.0 and obj.Par["sig_n"] == 1.0
    assert obj.Par["a"] == -2.0 and obj.Par["b"] == 2.0
    assert_has_par_and_dist(obj)


@pytest.mark.parametrize("par_val", [[0.0, 1.0, 2.0, 2.0], [0.0, 1.0, 1.0, -1.0]], ids=["a_eq_b", "a_gt_b"])
def test_par_truncatednormal_invalid_a_ge_b(par_val):
    with pytest.raises(RuntimeError, match="upper bound a must be larger than the lower bound b"):
        ERADist("truncatednormal", "PAR", par_val)


def test_par_truncatednormal_invalid_sigma_negative():
    with pytest.raises(RuntimeError, match="sigma must be larger than 0"):
        ERADist("truncatednormal", "PAR", [0.0, -0.1, -1.0, 1.0])


# --- StandardNormal: empty parameter list ---

@pytest.mark.parametrize("name", ["standardnormal", "standardgaussian"], ids=["standardnormal", "standardgaussian"])
def test_par_standardnormal_empty_params(name):
    obj = ERADist(name, "PAR", [])
    assert obj.Par["mu"] == 0 and obj.Par["sigma"] == 1
    assert_has_par_and_dist(obj)


# --- Case insensitivity: Normal / normal / GAUSSIAN ---

def test_par_case_insensitivity_normal():
    o1 = ERADist("Normal", "PAR", [0.0, 1.0])
    o2 = ERADist("normal", "PAR", [0.0, 1.0])
    o3 = ERADist("GAUSSIAN", "PAR", [0.0, 1.0])
    assert o1.Par["mu"] == o2.Par["mu"] == o3.Par["mu"] == 0.0
    assert o1.Par["sigma"] == o2.Par["sigma"] == o3.Par["sigma"] == 1.0


# --- Invalid / boundary for remaining distributions ---

@pytest.mark.parametrize("name,val", [
    ("chisquare", [0.0]),
    ("chisquare", [-1.0]),
    ("exponential", [0.0]),
    ("exponential", [-1.0]),
    ("frechet", [0.0, 1.0]),
    ("frechet", [1.0, 0.0]),
    ("gamma", [0.0, 1.0]),
    ("gamma", [1.0, 0.0]),
    ("geometric", [0.0]),
    ("geometric", [1.5]),
    ("gev", [0.2, 0.0, 0.0]),
    ("gumbel", [0.0, 0.0]),
    ("lognormal", [0.0, 0.0]),
    ("lognormal", [0.0, -1.0]),
    ("negativebinomial", [0.0, 0.5]),
    ("negativebinomial", [5.0, 1.5]),
    ("normal", [0.0, 0.0]),
    ("normal", [0.0, -1.0]),
    ("pareto", [0.0, 2.0]),
    ("pareto", [1.0, 0.0]),
    ("poisson", [0.0]),
    ("poisson", [-1.0]),
    ("poisson", [1.0, 0.0]),
    ("rayleigh", [0.0]),
    ("rayleigh", [-1.0]),
    ("uniform", [1.0, 0.0]),
    ("uniform", [1.0, 1.0]),
    ("weibull", [0.0, 1.0]),
    ("weibull", [1.0, 0.0]),
], ids=[
    "chisquare-0", "chisquare-neg", "exp-0", "exp-neg", "frechet-0", "frechet-inv",
    "gamma-0", "gamma-inv", "geom-0", "geom-1.5", "gev-alpha0", "gumbel-0",
    "lognorm-0", "lognorm-negsig", "negbinom-0", "negbinom-p>1", "normal-0",
    "normal-negsig", "pareto-0", "pareto-alpha0", "poisson-0", "poisson-neg",
    "poisson-t0", "rayleigh-0", "rayleigh-neg", "uniform-inv", "uniform-eq",
    "weibull-0", "weibull-inv",
])
def test_par_invalid_raises(name, val):
    with pytest.raises(RuntimeError, match="not defined for your parameters"):
        ERADist(name, "PAR", val)


# --- Unknown distribution name ---

def test_par_unknown_distribution():
    with pytest.raises(RuntimeError, match="not available"):
        ERADist("unknown_dist", "PAR", [1.0, 2.0])
