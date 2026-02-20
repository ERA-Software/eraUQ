"""
End-to-end tests matching legacy_examples/Example_ERANataf.py.

Tests ERANataf with normal, Gumbel, exponential marginals: random(5), pdf, cdf,
X2U with Jacobian, U2X with Jacobian. Reference outputs were generated with seed 2021.
"""
import numpy as np
import pytest

from eraUQ import ERADist, ERANataf


def _ref_path():
    from pathlib import Path
    return Path(__file__).parent / "ref" / "nataf_ref.json"


def test_eranataf_random_pdf_cdf_match_legacy():
    """ERANataf random(5), pdf(X), cdf(X) match legacy Example_ERANataf."""
    from conftest import load_ref_json
    ref = load_ref_json(_ref_path())
    np.random.seed(2021)
    M = list()
    M.append(ERADist("normal", "PAR", [4, 2]))
    M.append(ERADist("gumbel", "MOM", [1, 2]))
    M.append(ERADist("exponential", "PAR", 4))
    Rho = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])
    T_Nataf = ERANataf(M, Rho)
    X = T_Nataf.random(5)
    PDF_X = T_Nataf.pdf(X)
    CDF_X = T_Nataf.cdf(X)
    np.testing.assert_array_almost_equal(X, ref["X"])
    np.testing.assert_array_almost_equal(PDF_X, ref["PDF_X"])
    np.testing.assert_array_almost_equal(CDF_X, ref["CDF_X"], decimal=5)


def test_eranataf_X2U_U2X_jacobian_match_legacy():
    """ERANataf X2U(.,'Jac') and U2X(.,'Jac') match legacy Example_ERANataf."""
    from conftest import load_ref_json
    ref = load_ref_json(_ref_path())
    np.random.seed(2021)
    M = list()
    M.append(ERADist("normal", "PAR", [4, 2]))
    M.append(ERADist("gumbel", "MOM", [1, 2]))
    M.append(ERADist("exponential", "PAR", 4))
    Rho = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])
    T_Nataf = ERANataf(M, Rho)
    X = T_Nataf.random(5)
    U, Jac_X2U = T_Nataf.X2U(X, "Jac")
    X_backtransform, Jac_U2X = T_Nataf.U2X(U, "Jac")
    np.testing.assert_array_almost_equal(U, ref["U"])
    np.testing.assert_array_almost_equal(Jac_X2U, ref["Jac_X2U"])
    np.testing.assert_array_almost_equal(X_backtransform, ref["X_backtransform"])
    np.testing.assert_array_almost_equal(Jac_U2X, ref["Jac_U2X"])
