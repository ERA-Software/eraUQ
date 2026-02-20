"""
End-to-end tests matching legacy_examples/Example_ERARosen_and_ERACond.py.

Tests ERARosen built from ERADist and ERACond marginals: random(n), X2U, U2X, pdf.
Reference outputs were generated with seed 2021 and n=5.
"""
import numpy as np
import pytest

from eraUQ import ERADist, ERACond, ERARosen


def _ref_path():
    from pathlib import Path
    return Path(__file__).parent / "ref" / "rosen_ref.json"


def test_erarosen_random_X2U_U2X_match_legacy():
    """ERARosen random samples, X2U, U2X round-trip match legacy Example_ERARosen_and_ERACond."""
    from conftest import load_ref_json
    ref = load_ref_json(_ref_path())
    np.random.seed(2021)
    n = 5
    x1_dist = ERADist("normal", "PAR", [3, 2], "A")
    x2_dist = ERADist("normal", "PAR", [5, 4], "B")
    a = 3
    x3_dist = ERACond("normal", "PAR", lambda X: [X[0] * X[1] + a, 2], "C")

    def subtraction(a, b):
        return a - b

    x4_dist = ERACond("normal", "PAR", lambda X: [subtraction(X[0], X[1]), abs(X[0])], "D")
    x5_dist = ERACond("exponential", "PAR", lambda X: abs(X[0] ** 2 - X[1]), "E")
    x6_dist = ERACond("normal", "PAR", lambda X: [3 * X, 4], "F")
    x7_dist = ERACond("normal", "PAR", lambda X: [X[0] + X[1] - X[2], 1], "G")
    dist = [x1_dist, x2_dist, x3_dist, x4_dist, x5_dist, x6_dist, x7_dist]
    depend = [[], [], [0, 1], [0, 2], [2, 1], 3, [2, 3, 4]]
    X_dist = ERARosen(dist, depend)
    X = X_dist.random(n)
    U = X_dist.X2U(X)
    X_backtransform = X_dist.U2X(U)
    np.testing.assert_array_almost_equal(X, ref["X"])
    np.testing.assert_array_almost_equal(U, ref["U"])
    np.testing.assert_array_almost_equal(X_backtransform, ref["X_backtransform"])


def test_erarosen_pdf_match_legacy():
    """ERARosen joint PDF at sampled X matches legacy Example_ERARosen_and_ERACond."""
    from conftest import load_ref_json
    ref = load_ref_json(_ref_path())
    np.random.seed(2021)
    n = 5
    x1_dist = ERADist("normal", "PAR", [3, 2], "A")
    x2_dist = ERADist("normal", "PAR", [5, 4], "B")
    a = 3
    x3_dist = ERACond("normal", "PAR", lambda X: [X[0] * X[1] + a, 2], "C")

    def subtraction(a, b):
        return a - b

    x4_dist = ERACond("normal", "PAR", lambda X: [subtraction(X[0], X[1]), abs(X[0])], "D")
    x5_dist = ERACond("exponential", "PAR", lambda X: abs(X[0] ** 2 - X[1]), "E")
    x6_dist = ERACond("normal", "PAR", lambda X: [3 * X, 4], "F")
    x7_dist = ERACond("normal", "PAR", lambda X: [X[0] + X[1] - X[2], 1], "G")
    dist = [x1_dist, x2_dist, x3_dist, x4_dist, x5_dist, x6_dist, x7_dist]
    depend = [[], [], [0, 1], [0, 2], [2, 1], 3, [2, 3, 4]]
    X_dist = ERARosen(dist, depend)
    X = X_dist.random(n)
    pdf = X_dist.pdf(X)
    np.testing.assert_array_almost_equal(pdf, ref["pdf"])
