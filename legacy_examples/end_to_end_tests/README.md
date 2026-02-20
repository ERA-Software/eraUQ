# Legacy end-to-end tests

These tests ensure that refactoring does not change the numerical results of the legacy examples in `legacy_examples/`. Each test runs the same workflow as the corresponding example (with fixed seeds) and compares outputs to stored reference data. Reference data is stored as **JSON** in `ref/` (human-readable, diff-friendly in Git).

## Running the tests

From the **project root**:

```bash
pytest legacy_examples/end_to_end_tests/ -v
```

Or with `PYTHONPATH` if the package is not installed:

```bash
PYTHONPATH=src pytest legacy_examples/end_to_end_tests/ -v
```

## Regenerating reference data

If you intentionally change the behaviour and need to update the baseline:

```bash
PYTHONPATH=src python3 legacy_examples/end_to_end_tests/generate_reference_data.py
```

Then re-run the tests. Commit the updated `ref/*.json` files only if the behaviour change is desired.

## Mapping: legacy example → test module

| Legacy example | Test module | Testable outputs |
|----------------|-------------|------------------|
| `Example_EmpDist.py` | `test_empdist_bimodal.py` | `data`, `pdf_vals`, `cdf_vals`, `icdf_vals`, `sampled` |
| `Example_ERADist.py` | `test_eradir_lognormal.py` | `mean_dist`, `std_dist`, `samples`, `x`, `pdf`, `cdf`, `icdf` |
| `Example_ERADist_empirical.py` | `test_eradir_empirical.py` | `mean_dist`, `std_dist`, `samples`, `x`, `pdf`, `cdf`, `icdf` |
| `Example_ERARosen_and_ERACond.py` | `test_erarosen_eracond.py` | `X`, `U`, `X_backtransform`, `pdf` |
| `Example_ERANataf.py` | `test_eranataf_parametric.py` | `X`, `PDF_X`, `CDF_X`, `U`, Jacobians, `X_backtransform` |
| `Example_ERANataf_empirical.py` | `test_eranataf_empirical.py` | same as above with one empirical marginal |

Plots and graph figures from the legacy examples are not tested; only arrays and scalars are compared. Reference data is stored as JSON (human-readable, diff-friendly).
