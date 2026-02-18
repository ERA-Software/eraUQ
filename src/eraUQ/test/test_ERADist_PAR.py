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

PAR_VALID = {
    "beta": [2.0, 5.0, 0.0, 1.0],
    "binomial": [10, 0.5],
    "chisquare": [3.0],
    "exponential": [1.5],
    "frechet": [2.0, 3.0],
    "gamma": [2.0, 3.0],
    "geometric": [0.4],
    "gev": [0.2, 1.0, 0.0],
    "gevmin": [0.2, 1.0, 0.0],
    "gumbel": [1.0, 0.0],
    "gumbelmin": [1.0, 0.0],
    "lognormal": [0.0, 1.0],
    "negativebinomial": [5.0, 0.5],
    "normal": [0.0, 1.0],
    "pareto": [1.0, 2.5],
    "poisson": [3.0],
    "poisson_vt": [2.0, 1.5],
    "rayleigh": [2.0],
    "standardnormal": [],
    "truncatednormal": [0.0, 1.0, -2.0, 2.0],
    "uniform": [0.0, 1.0],
    "weibull": [1.0, 2.0],
}


@pytest.mark.parametrize("name,val", [
    ("beta", PAR_VALID["beta"]),
    ("binomial", PAR_VALID["binomial"]),
    ("chisquare", PAR_VALID["chisquare"]),
    ("exponential", PAR_VALID["exponential"]),
    ("frechet", PAR_VALID["frechet"]),
    ("gamma", PAR_VALID["gamma"]),
    ("geometric", PAR_VALID["geometric"]),
    ("gev", PAR_VALID["gev"]),
    ("gevmin", PAR_VALID["gevmin"]),
    ("gumbel", PAR_VALID["gumbel"]),
    ("gumbelmin", PAR_VALID["gumbelmin"]),
    ("lognormal", PAR_VALID["lognormal"]),
    ("negativebinomial", PAR_VALID["negativebinomial"]),
    ("normal", PAR_VALID["normal"]),
    ("pareto", PAR_VALID["pareto"]),
    ("rayleigh", PAR_VALID["rayleigh"]),
    ("standardnormal", PAR_VALID["standardnormal"]),
    ("truncatednormal", PAR_VALID["truncatednormal"]),
    ("uniform", PAR_VALID["uniform"]),
    ("weibull", PAR_VALID["weibull"]),
])
def test_par_valid_typical(name, val):
    if name == "poisson_vt":
        pytest.skip("poisson_vt tested separately")
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


def test_par_beta_invalid_r_s_zero():
    with pytest.raises(RuntimeError, match="not defined for your parameters"):
        ERADist("beta", "PAR", [0.0, 5.0, 0.0, 1.0])
    with pytest.raises(RuntimeError, match="not defined for your parameters"):
        ERADist("beta", "PAR", [2.0, 0.0, 0.0, 1.0])


def test_par_beta_invalid_a_ge_b():
    with pytest.raises(RuntimeError, match="not defined for your parameters"):
        ERADist("beta", "PAR", [2.0, 5.0, 1.0, 1.0])
    with pytest.raises(RuntimeError, match="not defined for your parameters"):
        ERADist("beta", "PAR", [2.0, 5.0, 2.0, 1.0])


# --- Binomial: n integer, 0<=p<=1 ---

def test_par_binomial_valid():
    obj = ERADist("binomial", "PAR", [10, 0.5])
    assert obj.Par["n"] == 10 and obj.Par["p"] == 0.5
    assert_has_par_and_dist(obj)


def test_par_binomial_boundary():
    obj = ERADist("binomial", "PAR", [1, 0.0])
    assert obj.Par["n"] == 1 and obj.Par["p"] == 0.0
    obj = ERADist("binomial", "PAR", [5, 1.0])
    assert obj.Par["p"] == 1.0


def test_par_binomial_invalid_p_out_of_range():
    with pytest.raises(RuntimeError, match="not defined for your parameters"):
        ERADist("binomial", "PAR", [10, -0.1])
    with pytest.raises(RuntimeError, match="not defined for your parameters"):
        ERADist("binomial", "PAR", [10, 1.1])


def test_par_binomial_invalid_n_non_integer():
    with pytest.raises(RuntimeError, match="not defined for your parameters"):
        ERADist("binomial", "PAR", [10.7, 0.5])


# --- TruncatedNormal: a<b, sigma>0 ---

def test_par_truncatednormal_valid():
    obj = ERADist("truncatednormal", "PAR", [0.0, 1.0, -2.0, 2.0])
    assert obj.Par["mu_n"] == 0.0 and obj.Par["sig_n"] == 1.0
    assert obj.Par["a"] == -2.0 and obj.Par["b"] == 2.0
    assert_has_par_and_dist(obj)


def test_par_truncatednormal_invalid_a_ge_b():
    with pytest.raises(RuntimeError, match="upper bound a must be larger than the lower bound b"):
        ERADist("truncatednormal", "PAR", [0.0, 1.0, 2.0, 2.0])
    with pytest.raises(RuntimeError, match="upper bound a must be larger than the lower bound b"):
        ERADist("truncatednormal", "PAR", [0.0, 1.0, 1.0, -1.0])


def test_par_truncatednormal_invalid_sigma_negative():
    with pytest.raises(RuntimeError, match="sigma must be larger than 0"):
        ERADist("truncatednormal", "PAR", [0.0, -0.1, -1.0, 1.0])


# --- StandardNormal: empty parameter list ---

def test_par_standardnormal_empty_params():
    obj = ERADist("standardnormal", "PAR", [])
    assert obj.Par["mu"] == 0 and obj.Par["sigma"] == 1
    assert_has_par_and_dist(obj)


def test_par_standardgaussian_empty_params():
    obj = ERADist("standardgaussian", "PAR", [])
    assert obj.Par["mu"] == 0 and obj.Par["sigma"] == 1


# --- Case insensitivity: Normal / normal / GAUSSIAN ---

def test_par_case_insensitivity_normal():
    o1 = ERADist("Normal", "PAR", [0.0, 1.0])
    o2 = ERADist("normal", "PAR", [0.0, 1.0])
    o3 = ERADist("GAUSSIAN", "PAR", [0.0, 1.0])
    assert o1.Par["mu"] == o2.Par["mu"] == o3.Par["mu"] == 0.0
    assert o1.Par["sigma"] == o2.Par["sigma"] == o3.Par["sigma"] == 1.0


# --- Invalid / boundary for remaining distributions ---

PAR_INVALID = [
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
]


def _invalid_id(p):
    name, val = p
    return f"{name}-{val}"


@pytest.mark.parametrize("name,val", PAR_INVALID, ids=[_invalid_id(x) for x in PAR_INVALID])
def test_par_invalid_raises(name, val):
    with pytest.raises(RuntimeError, match="not defined for your parameters"):
        ERADist(name, "PAR", val)


# --- Unknown distribution name ---

def test_par_unknown_distribution():
    with pytest.raises(RuntimeError, match="not available"):
        ERADist("unknown_dist", "PAR", [1.0, 2.0])
