# STAMPS — Spatial and Temporal Analysis and Mapping Python Suite

**Version:** 1.0.3  
**License:** [MIT](LICENSE)  
**Maintainer:** [STEMLab](https://stemlab.bse.ntu.edu.tw/wordpress/), National Taiwan University

> **Upgrading from 1.0.0?** The repository layout was flattened: `setup.py` now sits at
> the repository root and the package is imported as `stamps.<module>`. If you have
> notebooks or scripts written for 1.0.0, delete any `sys.path.insert(...)` lines and
> replace `stamps.stamps.` with `stamps.`. See [Migration from 1.0.0](#migration-from-100).

---

## Overview

STAMPS is a comprehensive Python library for **spatiotemporal geostatistics**, with a focus on **Bayesian Maximum Entropy (BME)** — a powerful framework for incorporating heterogeneous prior knowledge and soft (uncertain) observations into spatial estimation and mapping under uncertainty.

### Key capabilities

| Category | Features |
|---|---|
| **Dependence modelling** | Empirical covariance/variogram estimation, model fitting (WLS, MLE), anisotropy analysis, Linear Model of Coregionalisation (LMC) |
| **Variogram models** | Nugget, Exponential, Spherical, Gaussian, Hole-cosine, Hole-sine, Linear, Power, Matérn, Mexican Hat |
| **Kriging** | Ordinary/Simple kriging, multivariate cokriging, kriging with Normal Score Transformation (NST) |
| **BME estimation** | Posterior moments, full posterior PDF, credible intervals, mode — with hard data, interval soft data, and full probabilistic soft data |
| **Soft data encoding** | Uniform (interval), Gaussian, Student-T, piecewise-linear PDFs; neighbourhood selection and combination utilities |
| **Space-time** | Separable/non-separable space-time covariance models; space-time BME estimation |
| **Simulation** | Cholesky (unconditional/conditional), Sequential Gaussian (unconditional/conditional/interval-constrained), Circular Embedding |
| **Analysis** | EOF/PCA, MCA, CCA, Empirical Kalman Filter, Entropy, MEP-PDF, GAM, DLNM, STL decomposition |
| **Categorical BME** | Probability-table estimation for categorical spatial data; categorical BME prior and estimation |
| **NST transform** | Normal Score Transformation, Gaussian CDF inversion, back-transformation utilities |

---

## Repository layout

```
stamps/                          # repository root  (git clone …/stamps)
├── setup.py, setup.cfg, pyproject.toml
├── stamps/                      # the Python package  → import stamps.<module>
├── tutorials/                   # Jupyter notebooks (run in order)
├── tests/                       # pytest suite
├── docs/                        # mkdocs documentation site and docs/quickstart.py
├── README.md, LICENSE, BMELIB_COMPARISON.md
```

## Package structure

```
stamps/
├── bme/                          # Bayesian Maximum Entropy core
│   ├── BMEprobaEstimations.py    # BMEPosteriorMoments, BMEPosteriorPDF, credible intervals, mode
│   ├── BMEoptions.py             # BME configuration options
│   ├── softconverter.py          # Soft data encoding (probaUniform, probaGaussian, probaStudentT, …)
│   ├── bme_transform.py          # Normal Score Transformation
│   ├── BMEcatPdf.py              # Categorical BME posterior
│   └── pystks_variable.py        # Soft-data type constants
├── models/
│   └── covmodel.py               # Covariance and variogram model functions
├── estimation/                   # Spatial estimation
│   ├── kriging.py                # kriging, cokriging, cokrigingT
│   ├── idw.py, kernelsmoothing.py, localmeanBME.py, regression.py, stmean.py
│   └── designmatrix.py
├── categorical/                  # Categorical BME (prior, first-order models, ME solvers)
├── stats/
│   ├── dependence/
│   │   ├── covariance/           # stcov, stcovfit (covmodelfit, coregfit), mlecovfit, anisotropy
│   │   ├── ptable/               # Probability-table estimation and fitting
│   │   ├── gw_pmodel.py          # Geographically weighted P-model
│   │   └── regions.py
│   ├── analysis/                 # EOF/PCA, MCA, CCA, EKF, entropy, MEP-PDF, GAM/DLNM/STL (R, optional)
│   ├── evaluate/                 # Validation metrics
│   └── simulation/
│       └── simulation.py         # simuchol, simucholcond, simuseqcond, simuseqcondInt, …
├── general/                      # coord2dist, coord2K, neighbours, isspacetime, findpairs, valstvgx
├── graph/                        # dataplot, modelplot, categorical plots
├── mvn/                          # Multivariate normal integration (QMC)
├── aot/                          # Optional ahead-of-time compiled kernels
└── stest/                        # Deprecated alias of estimation/ (emits DeprecationWarning)
```

`stamps.stest`, `stamps.stats.dependence.stcov`, `stamps.stats.dependence.stcovfit` and
`stamps.stats.dependence.mlecovfit` are kept as backward-compatibility shims and will be
removed in a future release; new code should import from `stamps.estimation` and
`stamps.stats.dependence.covariance`.

---

## Relationship to BMElib (MATLAB)

STAMPS inherits its core spatiotemporal geostatistics engine from
**BMElib** ([Serre & Christakos, UNC](https://mserre.sph.unc.edu/BMElib_web/)),
the original MATLAB reference implementation of Bayesian Maximum Entropy theory.
The fundamental BME mathematics — posterior moment computation, soft data encoding,
space-time covariance models, and random field simulation — are all rooted in BMElib.

Notable improvements include: an **all-Gaussian analytical fast-path** that bypasses
numerical integration when soft data are Gaussian; **Quasi-Monte Carlo integration**
for higher accuracy; robust **SVD-based solvers** with PSD jitter for numerical
stability; **Iterated Alternating WLS (IALS)** for LMC fitting; generalised
**multivariate cokrigingT**; a full **variogram model family** (`*V`); and an
extended **spatiotemporal analysis toolbox** (EOF, EKF, MCA, entropy, GAM, …)
that goes well beyond BMElib's original scope.

For the full comparison — including a 15-point feature table and a complete
MATLAB-to-Python function mapping — see
**[BMELIB_COMPARISON.md](./BMELIB_COMPARISON.md)**.

---

## Installation

### Requirements

Python ≥ 3.9. Core dependencies are installed automatically:

```
numpy >= 1.21
scipy >= 1.7
pandas >= 1.3
six >= 1.15
```

**Optional dependencies (install separately as needed):**

| Package | Purpose |
|---|---|
| `matplotlib` | All visualisation functions |
| `nlopt` | Nonlinear optimisation for `covmodelfit`, `coregfit` (needed by the quick start and tutorial 03) |
| `xgboost` + `xgboostlss` | Categorical BME prior (`BMEcatPrior`) |
| `rpy2` + R | `gam`, `dlnm`, `stl` time-series decomposition |
| `jupyter`, `nbformat` | Running the tutorial notebooks |

### Install from source

```bash
git clone https://github.com/NTU-STEMLab/stamps.git
cd stamps
pip install -e .
```

### Install into a specific conda environment

```bash
conda create -n bme python=3.12   # once
conda activate bme
pip install -e /path/to/stamps/
```

### Install optional dependencies

```bash
pip install matplotlib nlopt jupyter
# Categorical BME prior:
pip install xgboost xgboostlss
# R-based functions:
pip install rpy2
# and in R: install.packages(c("zoo", "dlnm", "nlme", "mgcv"))
```

### Verify installation

```bash
python -c "import stamps; from stamps.bme.softconverter import probaUniform; print('stamps OK')"
pip show stamps   # should show Version: 1.0.3
```

---

## Quick start

The example below is self-contained (synthetic data). It requires `nlopt` for `covmodelfit`.

```python
import numpy as np
from stamps.estimation.kriging import kriging
from stamps.stats.dependence.covariance.stcov import stcov
from stamps.stats.dependence.covariance.stcovfit import covmodelfit
from stamps.bme.BMEprobaEstimations import BMEPosteriorMoments
from stamps.bme.softconverter import probaUniform

# --- 0. Synthetic data ---
rng = np.random.default_rng(0)
ch = rng.uniform(0, 10_000, size=(150, 2))                     # hard data locations (m)
zh = 20 + 3*np.sin(ch[:, 0]/2_000) + rng.normal(0, 1.5, 150)   # hard data values
cs = rng.uniform(0, 10_000, size=(30, 2))                      # soft data locations
z_mid = 20 + 3*np.sin(cs[:, 0]/2_000)
z_low, z_high = z_mid - 2.0, z_mid + 2.0                       # interval soft data
gx, gy = np.meshgrid(np.linspace(0, 10_000, 20), np.linspace(0, 10_000, 20))
ck = np.column_stack([gx.ravel(), gy.ravel()])                 # estimation grid

# --- 1. Empirical covariance and model fit ---
lagS       = np.arange(0, 8_000, 500.0)
lagS_range = np.full(len(lagS), 250.0)
cov_h, n_h, lag_h, _ = stcov(ch, None, zh, lagS, lagS_range)
ok = (n_h > 0) & ~np.isnan(cov_h)
lag_h, cov_h, n_h = lag_h[ok], cov_h[ok], n_h[ok]

covmodel  = ['nuggetC', 'sphericalC']
covparam0 = [(1.0,), (4.0, 4_000.0)]                           # initial guesses
covparam, _ = covmodelfit(lag_h.reshape(-1, 1), np.array([[0.0]]),
                          cov_h.reshape(-1, 1), n_h.reshape(-1, 1),
                          covmodel, covparam0)

# --- 2. Ordinary kriging ---
zk_mean, zk_var = kriging(ck, ch, zh, covmodel, covparam,
                          nhmax=20, dmax=8_000.0, order=0)

# --- 3. BME with interval soft data ---
softpdftype, nl, limi, probdens = probaUniform(z_low, z_high)
zs = [(softpdftype, nl[i:i+1], limi[i:i+1], probdens[i:i+1]) for i in range(len(cs))]
result = BMEPosteriorMoments(ck, ch=ch, cs=cs, zh=zh.reshape(-1, 1), zs=zs,
                             covmodel=covmodel, covparam=covparam,
                             order=0,                          # constant prior mean
                             nhmax=15, nsmax=8, dmax=np.array([[8_000.0]]))
bme_mean, bme_var = result[:, 0], result[:, 1]
```

Notes on the conventions used above:

- `stcov` returns `(cov, n_pairs, lag, _)`; empty lag bins are removed before fitting.
- `covmodelfit` takes the lag, covariance and pair-count arrays as 2-D `(n_lags, 1)`
  arrays plus a `(1, 1)` dummy temporal lag for purely spatial data.
- Soft data are passed to `BMEPosteriorMoments` as a list with one
  `(softpdftype, nl, limi, probdens)` tuple per location.
- `order=0` uses a constant prior mean estimated from the data; `order=np.nan` means a
  zero prior mean and is rarely appropriate for real variables.
- `BMEPosteriorMoments` returns an `(nk, 3)` array: posterior mean, posterior variance,
  and a diagnostic column.

---

## Tutorials

Interactive Jupyter notebooks are in `tutorials/`. They import the installed package
directly (`from stamps.<module> import …`), so install STAMPS first, then start
Jupyter from the same environment.

| Notebook | Topic |
|---|---|
| `01_data_loading_and_exploration.ipynb` | Data I/O, histograms, IDW, kernel smoothing |
| `02_covariance_and_variogram_models.ipynb` | Covariance/variogram model gallery, categorical dependence |
| `03_empirical_statistics_and_covariance_fitting.ipynb` | Empirical covariance, model fitting, NST, LMC (coregfit) |
| `04_kriging_and_cokriging.ipynb` | Ordinary kriging, cokriging, kriging with NST |
| `05_bme_interval_soft_data.ipynb` | BME with uniform interval soft data |
| `06_bme_probabilistic_soft_data.ipynb` | BME with Gaussian soft PDFs, posterior PDF, credible intervals |
| `07_bme_spacetime.ipynb` | Space-time BME with CST covariance models |
| `08_simulation.ipynb` | Unconditional/conditional random field simulation |
| `09_bme_categorical.ipynb` | BME for categorical spatial data |
| `10_anisotropy_analysis.ipynb` | Anisotropy estimation and analysis |

**Run the tutorials in order** — each saves intermediate results in `tutorials/data/`
that the next one reads.

```bash
conda activate bme
pip install jupyter matplotlib nlopt
cd tutorials/
jupyter lab
```

If you use several Python environments, make sure the notebook kernel is the one where
STAMPS is installed (`python -m ipykernel install --user --name bme` registers it).

---

## Running the tests

```bash
pip install pytest
pytest tests -q
```

---

## Migration from 1.0.0

Version 1.0.0 shipped with the package nested one level deeper (`stamps/stamps/`), and
the tutorials inserted the repository root into `sys.path` so that modules were imported
as `stamps.stamps.<module>`. Since 1.0.1 the package is installed normally and imported
as `stamps.<module>`. To update existing code:

1. Remove any `REPO_ROOT = …` / `sys.path.insert(0, REPO_ROOT)` lines.
2. Replace `from stamps.stamps.` with `from stamps.` (and `import stamps.stamps.` with
   `import stamps.`).
3. Prefer the current module paths: `stamps.estimation` instead of `stamps.stest`, and
   `stamps.stats.dependence.covariance.stcov` / `.stcovfit` instead of
   `stamps.stats.dependence.stcov` / `.stcovfit`. The old paths still work but emit a
   `DeprecationWarning`.
4. Reinstall: `pip uninstall -y stamps && pip install -e /path/to/stamps/`.

---

## License

STAMPS is released under the [MIT License](LICENSE). Versions 1.0.0–1.0.2 were released under GPL-3.0.

---

## Citing STAMPS

If you use STAMPS in your research, please cite:

```bibtex
@misc{yu2018stamps,
  author       = {Hwa-Lung Yu and Shang-Chen Ku and Chieh-Han Lee
                  and Hua-Ting Tseng and Shih-Yao Lee},
  title        = {STAMPS: Spatial and Temporal Analysis and Mapping Python Suite},
  howpublished = {\url{https://github.com/NTU-STEMLab/stamps}},
  year         = {2018}
}
```
