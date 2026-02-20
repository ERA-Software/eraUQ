"""
End-to-end tests matching legacy_examples/Example_ERANataf_empirical.py.

Tests ERANataf with one empirical marginal (bimodal data): random(5), pdf, cdf,
X2U with Jacobian, U2X with Jacobian. Reference outputs were generated with seed 2025.
"""
import numpy as np
import pytest

from eraUQ import ERADist, ERANataf


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
    return Path(__file__).parent / "ref" / "nataf_empirical_ref.json"


def test_eranataf_empirical_random_pdf_cdf_match_legacy():
    """ERANataf with empirical marginal: random(5), pdf(X), cdf(X) match legacy."""
    from conftest import load_ref_json
    ref = load_ref_json(_ref_path())
    np.random.seed(2025)
    n = 2000
    data = sample_bimodal_gaussian(
        n_samples=n, mix_weights=(0.2, 0.8), means=(-2, 3), stds=(0.5, 1.0)
    )
    M = list()
    M.append(ERADist("normal", "PAR", [4, 2]))
    M.append(ERADist("gumbel", "MOM", [1, 2]))
    M.append(ERADist("empirical", "DATA", [data, None, "linear", None, {}]))
    Rho = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])
    T_Nataf = ERANataf(M, Rho)
    X = T_Nataf.random(5)
    PDF_X = T_Nataf.pdf(X)
    CDF_X = T_Nataf.cdf(X)
    np.testing.assert_array_almost_equal(X, ref["X"])
    np.testing.assert_array_almost_equal(PDF_X, ref["PDF_X"])
    np.testing.assert_array_almost_equal(CDF_X, ref["CDF_X"], decimal=5)


def test_eranataf_empirical_X2U_U2X_jacobian_match_legacy():
    """ERANataf with empirical: X2U(.,'Jac') and U2X(.,'Jac') match legacy."""
    from conftest import load_ref_json
    ref = load_ref_json(_ref_path())
    np.random.seed(2025)
    n = 2000
    data = sample_bimodal_gaussian(
        n_samples=n, mix_weights=(0.2, 0.8), means=(-2, 3), stds=(0.5, 1.0)
    )
    M = list()
    M.append(ERADist("normal", "PAR", [4, 2]))
    M.append(ERADist("gumbel", "MOM", [1, 2]))
    M.append(ERADist("empirical", "DATA", [data, None, "linear", None, {}]))
    Rho = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])
    T_Nataf = ERANataf(M, Rho)
    X = T_Nataf.random(5)
    U, Jac_X2U = T_Nataf.X2U(X, "Jac")
    X_backtransform, Jac_U2X = T_Nataf.U2X(U, "Jac")
    np.testing.assert_array_almost_equal(U, ref["U"])
    np.testing.assert_array_almost_equal(Jac_X2U, ref["Jac_X2U"])
    # Empirical marginal U2X can have small numerical differences (interpolation / trapz)
    np.testing.assert_array_almost_equal(
        X_backtransform, ref["X_backtransform"], decimal=2
    )
    np.testing.assert_array_almost_equal(Jac_U2X, ref["Jac_U2X"], decimal=2)
