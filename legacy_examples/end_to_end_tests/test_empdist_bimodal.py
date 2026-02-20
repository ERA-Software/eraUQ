"""
End-to-end tests matching legacy_examples/Example_EmpDist.py.

Tests EmpDist on bimodal Gaussian mixture: pdf on grid, cdf on grid, icdf on [0,1],
and random(size=M). Reference outputs were generated with seed 2025, N=100, M=2000.
"""
import numpy as np
import pytest

from eraUQ import EmpDist


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
    return Path(__file__).parent / "ref" / "empdist_ref.json"


def test_empdist_pdf_cdf_icdf_on_grid_match_legacy():
    """EmpDist pdf(x_grid), cdf(x_grid), icdf(y_grid) match legacy Example_EmpDist."""
    from conftest import load_ref_json
    ref = load_ref_json(_ref_path())
    np.random.seed(2025)
    N = 100
    data = sample_bimodal_gaussian(
        n_samples=N, mix_weights=(0.2, 0.8), means=(-2, 3), stds=(0.5, 1.0)
    )
    weights = np.ones_like(data)
    dist_emp = EmpDist(
        data, weights=weights, pdfMethod="kde", pdfPoints=None, bw_method=0.1
    )
    x_grid = np.linspace(data.min() - 1, data.max() + 1, 1000)
    pdf_vals = dist_emp.pdf(x_grid)
    cdf_vals = dist_emp.cdf(x_grid)
    y_grid = np.linspace(0, 1, 1000)
    icdf_vals = dist_emp.icdf(y_grid)
    np.testing.assert_array_almost_equal(data, ref["data"])
    np.testing.assert_array_almost_equal(pdf_vals, ref["pdf_vals"])
    np.testing.assert_array_almost_equal(cdf_vals, ref["cdf_vals"])
    np.testing.assert_array_almost_equal(icdf_vals, ref["icdf_vals"])


def test_empdist_random_match_legacy():
    """EmpDist random(size=M) matches legacy Example_EmpDist."""
    from conftest import load_ref_json
    ref = load_ref_json(_ref_path())
    np.random.seed(2025)
    N = 100
    data = sample_bimodal_gaussian(
        n_samples=N, mix_weights=(0.2, 0.8), means=(-2, 3), stds=(0.5, 1.0)
    )
    weights = np.ones_like(data)
    dist_emp = EmpDist(
        data, weights=weights, pdfMethod="kde", pdfPoints=None, bw_method=0.1
    )
    M = 2000
    sampled = dist_emp.random(size=M)
    np.testing.assert_array_almost_equal(sampled, ref["sampled"])
