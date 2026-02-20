"""
End-to-end tests matching legacy_examples/Example_ERADist.py.

Tests ERADist with lognormal (PAR), moments (MOM), and data-fit (DATA):
mean, std, random samples, pdf, cdf, icdf round-trip.
Reference outputs were generated with the same seed (2021) and workflow.
"""
import numpy as np
import pytest

from eraUQ import ERADist


def _ref_path():
    from pathlib import Path
    return Path(__file__).parent / "ref" / "eradir_lognormal_ref.json"


def test_eradir_lognormal_moments_and_samples_match_legacy():
    """Lognormal ERADist(PAR) mean, std, random samples match legacy Example_ERADist."""
    from conftest import load_ref_json
    ref = load_ref_json(_ref_path())
    np.random.seed(2021)
    dist = ERADist("lognormal", "PAR", [2, 0.5])
    mean_dist = dist.mean()
    std_dist = dist.std()
    n = 10000
    samples = dist.random(n)
    assert mean_dist == pytest.approx(ref["mean_dist"])
    assert std_dist == pytest.approx(ref["std_dist"])
    np.testing.assert_array_almost_equal(samples, ref["samples"])


def test_eradir_lognormal_pdf_cdf_icdf_match_legacy():
    """Lognormal ERADist pdf, cdf, icdf on random x match legacy Example_ERADist."""
    from conftest import load_ref_json
    ref = load_ref_json(_ref_path())
    np.random.seed(2021)
    dist = ERADist("lognormal", "PAR", [2, 0.5])
    n = 10000
    # Match exact RNG sequence from legacy: first random(n) used for samples, then x
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
