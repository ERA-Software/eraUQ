"""
Generate reference .json files for legacy end-to-end tests.
Run from project root: PYTHONPATH=src python3 legacy_examples/end_to_end_tests/generate_reference_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np

# Ensure we can import eraUQ (run from project root with PYTHONPATH=src)
_root = Path(__file__).resolve().parents[2]
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from eraUQ import ERADist, ERANataf, ERARosen, ERACond, EmpDist

REF_DIR = Path(__file__).parent / "ref"
REF_DIR.mkdir(exist_ok=True)


def to_json_serializable(obj):
    """Convert numpy types to native Python for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return obj


def save_ref(name, **kwargs):
    """Save reference dict as JSON (arrays and scalars)."""
    data = {k: to_json_serializable(v) for k, v in kwargs.items()}
    path = REF_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {path.name}")


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


def main():
    # 1. ERADist lognormal (Example_ERADist)
    np.random.seed(2021)
    dist = ERADist("lognormal", "PAR", [2, 0.5])
    mean_dist = dist.mean()
    std_dist = dist.std()
    n = 10000
    samples = dist.random(n)
    x = dist.random(n)
    pdf = dist.pdf(x)
    cdf = dist.cdf(x)
    icdf = dist.icdf(cdf)
    save_ref(
        "eradir_lognormal_ref",
        mean_dist=mean_dist,
        std_dist=std_dist,
        samples=samples,
        x=x,
        pdf=pdf,
        cdf=cdf,
        icdf=icdf,
    )

    # 2. ERARosen / ERACond (Example_ERARosen_and_ERACond)
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
    dist_list = [x1_dist, x2_dist, x3_dist, x4_dist, x5_dist, x6_dist, x7_dist]
    depend = [[], [], [0, 1], [0, 2], [2, 1], 3, [2, 3, 4]]
    X_dist = ERARosen(dist_list, depend)
    X = X_dist.random(n)
    U = X_dist.X2U(X)
    X_backtransform = X_dist.U2X(U)
    pdf = X_dist.pdf(X)
    save_ref(
        "rosen_ref",
        X=X,
        U=U,
        X_backtransform=X_backtransform,
        pdf=pdf,
    )

    # 3. ERANataf parametric (Example_ERANataf)
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
    U, Jac_X2U = T_Nataf.X2U(X, "Jac")
    X_backtransform, Jac_U2X = T_Nataf.U2X(U, "Jac")
    save_ref(
        "nataf_ref",
        X=X,
        PDF_X=PDF_X,
        CDF_X=CDF_X,
        U=U,
        Jac_X2U=Jac_X2U,
        X_backtransform=X_backtransform,
        Jac_U2X=Jac_U2X,
    )

    # 4. EmpDist (Example_EmpDist)
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
    M = 2000
    sampled = dist_emp.random(size=M)
    save_ref(
        "empdist_ref",
        data=data,
        pdf_vals=pdf_vals,
        cdf_vals=cdf_vals,
        icdf_vals=icdf_vals,
        sampled=sampled,
        x_grid=x_grid,
        y_grid=y_grid,
    )

    # 5. ERADist empirical (Example_ERADist_empirical)
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
    x = dist.random(n)
    pdf = dist.pdf(x)
    cdf = dist.cdf(x)
    icdf = dist.icdf(cdf)
    save_ref(
        "eradir_empirical_ref",
        mean_dist=mean_dist,
        std_dist=std_dist,
        samples=samples,
        x=x,
        pdf=pdf,
        cdf=cdf,
        icdf=icdf,
        data=data,
    )

    # 6. ERANataf empirical (Example_ERANataf_empirical)
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
    U, Jac_X2U = T_Nataf.X2U(X, "Jac")
    X_backtransform, Jac_U2X = T_Nataf.U2X(U, "Jac")
    save_ref(
        "nataf_empirical_ref",
        X=X,
        PDF_X=PDF_X,
        CDF_X=CDF_X,
        U=U,
        Jac_X2U=Jac_X2U,
        X_backtransform=X_backtransform,
        Jac_U2X=Jac_U2X,
    )

    print("Done. Reference data in", REF_DIR)


if __name__ == "__main__":
    main()
