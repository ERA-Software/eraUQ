"""
Moments mode (MOM) constructor tests for ERADist.
Covers: valid/boundary/invalid moments, parameter derivation, round-trip tests,
and key cases (Beta, Binomial, GEV/GEVMin beta=0 and beta≥0.5, TruncatedNormal mean bounds,
Frechet/Weibull fsolve convergence).
"""
import pytest
import numpy as np

from eraUQ import ERADist


# --- Helpers ---

def assert_has_par_and_dist(obj):
    assert hasattr(obj, "Par") and isinstance(obj.Par, dict)
    assert hasattr(obj, "Dist") and obj.Dist is not None


# --- Valid moment sets (typical values) ---

@pytest.mark.parametrize("name,val", [
    ("beta", [0.3, 0.1, 0.0, 1.0]),
    ("binomial", [9.0, 1.5]),
    ("chisquare", [3.0]),
    ("exponential", [2.0]),
    ("frechet", [3.0, 1.0]),
    ("gamma", [6.0, 2.0]),
    ("geometric", [2.5]),
    ("gev", [1.0, 0.5, 0.1]),
    ("gevmin", [1.0, 0.5, 0.1]),
    ("gumbel", [2.0, 1.0]),
    ("gumbelmin", [2.0, 1.0]),
    ("lognormal", [2.0, 1.0]),
    ("negativebinomial", [5.0, np.sqrt(20)]),
    ("normal", [0.0, 1.0]),
    ("poisson", [3.0]),
    ("pareto", [2.0, 1.0]),
    ("rayleigh", [2.0]),
    ("standardnormal", []),
    ("truncatednormal", [0.0, 1.0, -2.0, 2.0]),
    ("uniform", [0.5, 0.3]),
    ("weibull", [2.0, 1.0]),
])
def test_mom_valid_typical(name, val):
    if name == "poisson":
        pytest.skip("poisson tested separately")
    obj = ERADist(name, "MOM", val)
    assert_has_par_and_dist(obj)
    assert obj.Name == name.lower()


def test_mom_poisson_lambda_form():
    obj = ERADist("poisson", "MOM", [3.0])
    assert_has_par_and_dist(obj)
    assert "lambda" in obj.Par or "v" in obj.Par


def test_mom_poisson_vt_form():
    obj = ERADist("poisson", "MOM", [2.0, 1.5])
    assert_has_par_and_dist(obj)
    assert obj.Par["v"] == 2.0 / 1.5
    assert obj.Par["t"] == 1.5


# --- Beta: Requires support bounds [mean, std, a, b] ---

def test_mom_beta_valid():
    obj = ERADist("beta", "MOM", [0.3, 0.1, 0.0, 1.0])
    assert obj.Par["r"] > 0 and obj.Par["s"] > 0
    assert obj.Par["a"] == 0.0 and obj.Par["b"] == 1.0
    assert_has_par_and_dist(obj)


@pytest.mark.parametrize("mom_val,match", [
    ([0.3, 0.1, 1.0, 1.0], "Upper bound.*must be greater than lower bound"),
    ([0.3, 0.1, 1.0, 0.0], "Upper bound.*must be greater than lower bound"),
    ([0.5, 1.0, 0.0, 1.0], "Variance.*too large for a Beta distribution"),
    ([0.0, 0.1, 0.0, 1.0], "Mean.*must lie strictly between"),
    ([1.0, 0.1, 0.0, 1.0], "Mean.*must lie strictly between"),
    ([0.5, 0.0, 0.0, 1.0], "Standard deviation.*must be positive"),
], ids=["a_eq_b", "a_gt_b", "variance_too_large", "mean_at_lower", "mean_at_upper", "zero_std"])
def test_mom_beta_invalid(mom_val, match):
    with pytest.raises(RuntimeError, match=match):
        ERADist("beta", "MOM", mom_val)


# --- Binomial: Verify n becomes integer, mean > 0, 0 < p <= 1 ---

@pytest.mark.parametrize("mom_val,expected_n,expected_p", [
    ([9.0, 1.5], 12, 0.75),
    ([6.0, np.sqrt(2)], 9, 2/3),
], ids=["n12_p0.75", "n9_p2/3"])
def test_mom_binomial_valid(mom_val, expected_n, expected_p):
    obj = ERADist("binomial", "MOM", mom_val)
    assert_has_par_and_dist(obj)
    assert isinstance(obj.Par["n"], (int, np.integer))
    assert obj.Par["n"] == expected_n
    assert 0 < obj.Par["p"] <= 1
    assert obj.Par["p"] == pytest.approx(expected_p)


@pytest.mark.parametrize("mom_val,match", [
    ([0.0, 1.0], "Mean.*must be positive for binomial MOM"),
    ([-1.0, 1.0], "Mean.*must be positive for binomial MOM"),
    ([1.0, 1.0], "Probability.*must lie strictly between 0 and 1"),
    ([1.0, 1.5], "Probability.*must lie strictly between 0 and 1"),
    ([10.0, 2.1], "Please select other moments.*not a positive integer"),
], ids=["mean_zero", "mean_negative", "p_zero", "p_negative", "n_non_integer"])
def test_mom_binomial_invalid(mom_val, match):
    with pytest.raises(RuntimeError, match=match):
        ERADist("binomial", "MOM", mom_val)


# --- GEV/GEVMin: Test beta=0 (Gumbel case) and beta≥0.5 error ---

@pytest.mark.parametrize("name", ["gev", "gevmin"], ids=["gev", "gevmin"])
def test_mom_gev_beta_zero(name):
    # beta=0 corresponds to Gumbel distribution
    obj = ERADist(name, "MOM", [2.0, 1.0, 0.0])
    assert obj.Par["beta"] == 0.0
    assert_has_par_and_dist(obj)


@pytest.mark.parametrize("name,mom_val", [
    ("gev", [1.0, 0.5, 0.5]),
    ("gev", [1.0, 0.5, 0.6]),
    ("gevmin", [1.0, 0.5, 0.5]),
    ("gevmin", [1.0, 0.5, 0.6]),
], ids=["gev_beta0.5", "gev_beta0.6", "gevmin_beta0.5", "gevmin_beta0.6"])
def test_mom_gev_beta_ge_half(name, mom_val):
    with pytest.raises(RuntimeError, match="MOM can only be used for beta < 0.5"):
        ERADist(name, "MOM", mom_val)


@pytest.mark.parametrize("name", ["gev", "gevmin"], ids=["gev", "gevmin"])
def test_mom_gev_valid_beta(name):
    obj = ERADist(name, "MOM", [1.0, 0.5, 0.1])
    assert obj.Par["beta"] < 0.5
    assert_has_par_and_dist(obj)


# --- TruncatedNormal: Mean must be within [a,b] ---

def test_mom_truncatednormal_valid():
    obj = ERADist("truncatednormal", "MOM", [0.0, 1.0, -2.0, 2.0])
    assert obj.Par["mu_n"] > obj.Par["a"] and obj.Par["mu_n"] < obj.Par["b"]
    assert_has_par_and_dist(obj)


@pytest.mark.parametrize("mom_val", [
    [-2.0, 1.0, -2.0, 2.0],  # mean at lower bound
    [2.0, 1.0, -2.0, 2.0],   # mean at upper bound
    [-3.0, 1.0, -2.0, 2.0],  # mean below bounds
    [3.0, 1.0, -2.0, 2.0],   # mean above bounds
], ids=["mean_at_lower", "mean_at_upper", "mean_below", "mean_above"])
def test_mom_truncatednormal_mean_outside_bounds(mom_val):
    with pytest.raises(RuntimeError, match="The mean of the distribution must be within"):
        ERADist("truncatednormal", "MOM", mom_val)


@pytest.mark.parametrize("mom_val", [[0.0, 1.0, 2.0, 2.0], [0.0, 1.0, 1.0, -1.0]], ids=["a_eq_b", "a_gt_b"])
def test_mom_truncatednormal_invalid_a_ge_b(mom_val):
    with pytest.raises(RuntimeError, match="The upper bound a must be larger than the lower bound b"):
        ERADist("truncatednormal", "MOM", mom_val)


def test_mom_truncatednormal_fsolve_failure():
    # Test case that might cause fsolve to fail
    # This is hard to predict, but we can test with extreme values
    # Note: This might pass or fail depending on fsolve convergence
    try:
        obj = ERADist("truncatednormal", "MOM", [0.0, 0.01, -0.1, 0.1])
        assert_has_par_and_dist(obj)
    except RuntimeError as e:
        assert "fsolve did not converge" in str(e)


# --- Frechet/Weibull: Test fsolve convergence success/failure ---

@pytest.mark.parametrize("name,val", [("frechet", [3.0, 1.0]), ("weibull", [2.0, 1.0])], ids=["frechet", "weibull"])
def test_mom_frechet_weibull_valid(name, val):
    obj = ERADist(name, "MOM", val)
    assert obj.Par["a_n"] > 0 and obj.Par["k"] > 0
    assert_has_par_and_dist(obj)


# --- Invalid moments (negative std, incompatible combinations) ---

@pytest.mark.parametrize("name,val,match", [
    ("normal", [0.0, -1.0], "The standard deviation must be non-negative"),
    ("exponential", [0.0], "The first moment cannot be zero"),
    ("lognormal", [0.0, 1.0], "the first moment must be greater than zero"),
    ("lognormal", [-1.0, 1.0], "the first moment must be greater than zero"),
    ("chisquare", [0.0], "Please select other moments"),
    ("chisquare", [-1.0], "Please select other moments"),
    ("geometric", [0.0], "Please select other moments"),
    ("geometric", [-1.0], "Please select other moments"),
    ("poisson", [0.0], "Please select other moments"),
    ("poisson", [-1.0], "Please select other moments"),
    ("poisson", [2.0, 0.0], "Please select other moments"),
    ("poisson", [2.0, -1.0], "The standard deviation must be non-negative"),
    ("rayleigh", [0.0], "Please select other moments"),
    ("rayleigh", [-1.0], "Please select other moments"),
    ("gamma", [0.0, 1.0], "Mean.*must be positive for gamma MOM"),
    ("gamma", [1.0, 0.0], "Standard deviation.*must be positive for gamma MOM"),
    ("negativebinomial", [0.0, 1.0], "Mean.*must be positive for negative binomial MOM"),
    ("pareto", [0.0, 1.0], "Please select other moments"),
    ("gumbel", [2.0, 0.0], "Please select other moments"),
    ("gumbelmin", [2.0, 0.0], "Please select other moments"),
], ids=[
    "normal_neg_std",
    "exp_zero_mean",
    "lognormal_zero_mean",
    "lognormal_neg_mean",
    "chisquare_zero",
    "chisquare_neg",
    "geometric_zero",
    "geometric_neg",
    "poisson_zero",
    "poisson_neg",
    "poisson_t_zero",
    "poisson_neg_std",
    "rayleigh_zero",
    "rayleigh_neg",
    "gamma_zero_mean",
    "gamma_zero_std",
    "negbinom_zero_mean",
    "pareto_zero_mean",
    "gumbel_zero_std",
    "gumbelmin_zero_std",
])
def test_mom_invalid_moments(name, val, match):
    with pytest.raises(RuntimeError, match=match):
        ERADist(name, "MOM", val)


# --- Round-trip tests: PAR → MOM → verify equivalence ---

# (mean, std, par_val) -> MOM input list; default is [mean, std]
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


@pytest.mark.parametrize("dist_name,par_val", [
    ("normal", [0.0, 1.0]),
    ("beta", [2.0, 5.0, 0.0, 1.0]),
    ("binomial", [10, 0.5]),
    ("exponential", [1.5]),
    ("gamma", [2.0, 3.0]),
    ("lognormal", [0.0, 1.0]),
    ("uniform", [0.0, 1.0]),
    ("gev", [0.2, 1.0, 0.0]),
    ("gevmin", [0.2, 1.0, 0.0]),
    ("gumbel", [1.0, 0.5]),
    ("gumbelmin", [1.0, 0.5]),
    ("pareto", [1.0, 2.5]),
    ("rayleigh", [2.0]),
    ("chisquare", [3.0]),
    ("geometric", [0.4]),
], ids=[
    "normal", "beta", "binomial", "exponential", "gamma", "lognormal",
    "uniform", "gev", "gevmin", "gumbel", "gumbelmin",
    "pareto", "rayleigh", "chisquare", "geometric",
])
def test_mom_round_trip(dist_name, par_val, rtol=1e-4):
    dist_par = ERADist(dist_name, "PAR", par_val)
    m, s = dist_par.mean(), dist_par.std()
    builder = _MOM_FROM_PAR.get(dist_name.lower(), lambda m, s, p: [m, s])
    mom_val = builder(m, s, par_val)
    dist_mom = ERADist(dist_name, "MOM", mom_val)
    for key in dist_par.Par:
        if key in dist_mom.Par:
            v_par, v_mom = dist_par.Par[key], dist_mom.Par[key]
            if isinstance(v_par, (int, np.integer)):
                assert v_par == v_mom, f"Parameter {key} mismatch: {v_par} != {v_mom}"
            else:
                np.testing.assert_allclose(v_par, v_mom, rtol=rtol, atol=1e-10, err_msg=f"Parameter {key} mismatch")


# --- Edge cases: very small/large moments ---

@pytest.mark.parametrize("val,mean_tol,std_tol", [
    ([0.001, 0.001], 1e-6, 1e-6),
    ([1000.0, 100.0], 1.0, 1.0),
], ids=["small", "large"])
def test_mom_normal_edge_moments(val, mean_tol, std_tol):
    obj = ERADist("normal", "MOM", val)
    assert_has_par_and_dist(obj)
    assert abs(obj.mean() - val[0]) < mean_tol
    assert abs(obj.std() - val[1]) < std_tol


@pytest.mark.parametrize("mean", [0.01, 1000.0], ids=["small", "large"])
def test_mom_exponential_edge_mean(mean):
    obj = ERADist("exponential", "MOM", [mean])
    assert_has_par_and_dist(obj)
    assert obj.Par["lambda"] > 0


# --- Case insensitivity ---

@pytest.mark.parametrize("names,val,expected_mu,expected_sigma", [
    (["Normal", "normal", "GAUSSIAN"], [0.0, 1.0], 0.0, 1.0),
    (["StandardNormal", "standardnormal", "STANDARDGAUSSIAN"], [], 0, 1),
], ids=["normal", "standardnormal"])
def test_mom_case_insensitivity(names, val, expected_mu, expected_sigma):
    objs = [ERADist(n, "MOM", val) for n in names]
    assert all(o.Par["mu"] == expected_mu for o in objs)
    assert all(o.Par["sigma"] == expected_sigma for o in objs)


# --- Unknown distribution name ---

def test_mom_unknown_distribution():
    with pytest.raises(RuntimeError, match="not available"):
        ERADist("unknown_dist", "MOM", [1.0, 2.0])
