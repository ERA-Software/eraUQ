"""
Special distribution tests (TEST_STRATEGY_ERADist.md §4): GEVMin sign flips,
negative binomial shift by k, empirical options, truncated normal convergence.
"""
import pytest
import numpy as np
from scipy import stats

from eraUQ import ERADist


def assert_has_par_and_dist(obj):
    assert hasattr(obj, "Par") and isinstance(obj.Par, dict)
    assert hasattr(obj, "Dist") and obj.Dist is not None


# --- 4.1 GEVMin: sign flips in mean(), pdf(), cdf(), random(), icdf() ---

@pytest.fixture
def gev_gevmin_pair():
    """Same underlying params: GEV (maxima) and GEVMin (minima = -GEV(-x))."""
    par = [0.2, 1.0, 0.0]  # beta, alpha, epsilon
    gev = ERADist("gev", "PAR", par)
    gevmin = ERADist("gevmin", "PAR", par)
    return gev, gevmin


def test_gevmin_sign_flip_mean(gev_gevmin_pair):
    """GEVMin mean = -GEV mean (minima vs maxima)."""
    gev, gevmin = gev_gevmin_pair
    np.testing.assert_allclose(gevmin.mean(), -gev.mean())


def test_gevmin_sign_flip_pdf(gev_gevmin_pair):
    """GEVMin pdf(x) = GEV.pdf(-x)."""
    gev, gevmin = gev_gevmin_pair
    x = np.linspace(-2, 2, 5)
    for xi in x:
        np.testing.assert_allclose(gevmin.pdf(xi), gev.pdf(-xi))


def test_gevmin_sign_flip_cdf(gev_gevmin_pair):
    """GEVMin cdf(x) = 1 - GEV.cdf(-x)."""
    gev, gevmin = gev_gevmin_pair
    x = np.linspace(-2, 2, 5)
    for xi in x:
        np.testing.assert_allclose(gevmin.cdf(xi), 1 - gev.cdf(-xi))


def test_gevmin_sign_flip_random(gev_gevmin_pair):
    """GEVMin samples = -GEV samples (same seed)."""
    gev, gevmin = gev_gevmin_pair
    np.random.seed(123)
    r_gev = gev.random(size=100)
    np.random.seed(123)
    r_gevmin = gevmin.random(size=100)
    np.testing.assert_allclose(r_gevmin, -r_gev)


def test_gevmin_sign_flip_icdf(gev_gevmin_pair):
    """GEVMin icdf(y) = -GEV.icdf(1-y)."""
    gev, gevmin = gev_gevmin_pair
    y = np.linspace(0.01, 0.99, 7)
    for yi in y:
        np.testing.assert_allclose(gevmin.icdf(yi), -gev.icdf(1 - yi))


# --- 4.2 Negative binomial: shift by k in all methods ---

@pytest.mark.parametrize("k,p", [(5, 0.5), (3, 0.4)], ids=["k5_p0.5", "k3_p0.4"])
def test_negativebinomial_shift_mean(k, p):
    """ERADist mean = scipy nbinom.mean() + k (support shifted by k)."""
    obj = ERADist("negativebinomial", "PAR", [k, p])
    internal_mean = k * (1 - p) / p
    np.testing.assert_allclose(obj.mean(), internal_mean + k)


@pytest.mark.parametrize("k,p", [(5, 0.5), (3, 0.4)], ids=["k5_p0.5", "k3_p0.4"])
def test_negativebinomial_shift_pdf_cdf(k, p):
    """pdf/cdf use shift: support starts at k."""
    obj = ERADist("negativebinomial", "PAR", [k, p])
    assert obj.cdf(k - 0.1) == 0
    assert obj.cdf(k) >= 0 and obj.pdf(k) > 0
    np.testing.assert_allclose(obj.cdf(k), stats.nbinom.pmf(0, n=k, p=p))


@pytest.mark.parametrize("k,p", [(5, 0.5), (3, 0.4)], ids=["k5_p0.5", "k3_p0.4"])
def test_negativebinomial_shift_random_icdf(k, p):
    """random() and icdf() in shifted support [k, ...]; icdf(0) = k."""
    obj = ERADist("negativebinomial", "PAR", [k, p])
    assert np.all(obj.random(size=200) >= k)
    np.testing.assert_allclose(obj.icdf(0), k)
    assert obj.icdf(0.5) >= k and obj.icdf(0.99) >= k


@pytest.mark.parametrize("k_invalid", [5.3, -1, 0], ids=["k_float", "k_negative", "k_zero"])
def test_negativebinomial_par_k_invalid(k_invalid):
    """PAR: k must be a positive integer."""
    with pytest.raises(RuntimeError, match="not defined for your parameters"):
        ERADist("negativebinomial", "PAR", [k_invalid, 0.5])


# --- 4.3 Empirical: pdfMethods, pdfPoints, kdeKwargsDict ---

@pytest.mark.parametrize("pdf_method,pdf_points,kwargs,par_key,par_val", [
    ("cubic", 40, {}, "pdfMethod", "cubic"),
    ("linear", 50, {}, "pdfPoints", 50),
    ("kde", None, {"bw_method": "scott"}, "bw_method", "scott"),
], ids=["pdfMethod_cubic", "pdfPoints_50", "kde_kwargs"])
def test_empirical_options(pdf_method, pdf_points, kwargs, par_key, par_val):
    """Empirical: pdfMethod, pdfPoints, and kdeKwargsDict are passed and stored."""
    X = np.random.randn(80)
    obj = ERADist("empirical", "DATA", [X, None, pdf_method, pdf_points, kwargs])
    assert_has_par_and_dist(obj)
    assert obj.Par[par_key] == par_val


# --- 4.4 Truncated normal: DATA MLE convergence (explicit success) ---

def test_truncatednormal_data_mle_success():
    """DATA mode: MLE fit succeeds for valid data in [a,b]."""
    a, b = -2.0, 2.0
    X = stats.truncnorm.rvs(-2, 2, loc=0, scale=1, size=300)
    obj = ERADist("truncatednormal", "DATA", [X, a, b])
    assert_has_par_and_dist(obj)
    assert obj.Par["a"] == a and obj.Par["b"] == b
    assert obj.Par["sig_n"] > 0
