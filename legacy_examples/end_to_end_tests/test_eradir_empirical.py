"""
End-to-end tests matching legacy_examples/Example_ERADist_empirical.py.

Tests ERADist('empirical','DATA', ...) with bimodal data: mean, std, random(n),
pdf(x), cdf(x), icdf(cdf) round-trip. Reference outputs were generated with seed 2025.
"""
import numpy as np
import pytest

from eraUQ import ERADist


def sample_bimodal_gaussian(
    n_samples=1000, mix_weights=(0.4, 0.6), means=(-2, 3), stds=(0.7, 1.2)
):
    comps = np.random.choice([0, 1], size=n_samples, p=mix_weights)
    data = np.where(
        comps == 0,
        np.random.normal(loc=means[0], scale=stds[0], size=n_samples),
        np.random.normal(loc=means[1], scale=stds[1], size=n_samples),
    )
    return data


def _ref_path():
    from pathlib import Path
    return Path(__file__).parent / "ref" / "eradir_empirical_ref.json"


def test_eradir_empirical_moments_and_random_match_legacy():
    """ERADist empirical mean, std, random(n) match legacy Example_ERADist_empirical."""
    from conftest import load_ref_json
    ref = load_ref_json(_ref_path())
    np.random.seed(2025)
    n = 2000
    data = sample_bimodal_gaussian(
        n_samples=n, mix_weights=(0.2, 0.8), means=(-2, 3), stds=(0.5, 1.0)
    )
    weights = None
    dist = ERADist("empirical", "DATA", [data, weights, "kde", None, {"bw_method": None}])
    mean_dist = dist.mean()
    std_dist = dist.std()
    samples = dist.random(n)
    np.testing.assert_array_almost_equal(data, ref["data"])
    assert mean_dist == pytest.approx(ref["mean_dist"])
    assert std_dist == pytest.approx(ref["std_dist"])
    np.testing.assert_array_almost_equal(samples, ref["samples"])


def test_eradir_empirical_pdf_cdf_icdf_match_legacy():
    """ERADist empirical pdf(x), cdf(x), icdf(cdf) match legacy Example_ERADist_empirical."""
    from conftest import load_ref_json
    ref = load_ref_json(_ref_path())
    np.random.seed(2025)
    n = 2000
    data = sample_bimodal_gaussian(
        n_samples=n, mix_weights=(0.2, 0.8), means=(-2, 3), stds=(0.5, 1.0)
    )
    weights = None
    dist = ERADist("empirical", "DATA", [data, weights, "kde", None, {"bw_method": None}])
    # Match exact RNG sequence from legacy: mean, std, samples = random(n), then x
    _ = dist.mean()
    _ = dist.std()
    _ = dist.random(n)
    x = dist.random(n)
    pdf = dist.pdf(x)
    cdf = dist.cdf(x)
    icdf = dist.icdf(cdf)
    np.testing.assert_array_almost_equal(x, ref["x"])
    np.testing.assert_array_almost_equal(pdf, ref["pdf"])
    np.testing.assert_array_almost_equal(cdf, ref["cdf"])
    np.testing.assert_array_almost_equal(icdf, ref["icdf"])
