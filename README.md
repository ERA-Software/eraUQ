# eraUQ

Base software of the **Engineering Risk Analysis** group at TUM for uncertainty quantification and reliability analysis. It provides probability distributions (ERADist, EmpDist), dependence models (ERANataf, ERARosen, ERACond), and related utilities for risk and reliability applications.

## Install and use

Install from PyPI:

```bash
pip install eraUQ
```

Then use in Python:

```python
from eraUQ import ERADist, EmpDist, ERACond, ERANataf, ERARosen
dist = ERADist("Normal", "MOM", [0, 4])
xs = dist.random(10_000)
```

## Example notebook

**[Usage overview](notebooks/demonstrate_usage_overview.ipynb)** — install, distributions, sampling, and basic usage.

### [**<img src="https://colab.research.google.com/img/colab_favicon_256px.png" alt="Colab Logo" width="28" style="vertical-align: middle;"/> Open the demonstration notebook in Google Colab**](https://colab.research.google.com/github/ERA-Software/eraUQ/blob/main/notebooks/demonstrate_usage_overview.ipynb) 

---

# DEV Instructions
## Installation
First, ensure the repository is cloned and you are on the right branch

The following commands assume that you are **inside the root directory of the package**
(i.e. the directory containing `pyproject.toml`).

```bash
git checkout <branch-name>
pip install -e .
```
Alternatively, a command with the absolute path can be used:

```bash
py -m pip install -e <absolute/path/to/repo/root/directory>
```

or

```bash
pip install -e <absolute/path/to/repo/root/directory>
```
