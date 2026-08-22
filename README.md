# STAMPS — Spatial and Temporal Analysis and Mapping Python Suite

**Version:** 1.0.0  
**License:** [GNU General Public License v3.0](LICENSE)  
**Maintainer:** [STEMLab](https://stemlab.bse.ntu.edu.tw/wordpress/), National Taiwan University

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
| **Categorical dependence** | Probability table estimation (`probatablecalc`) for categorical spatial data |
| **NST transform** | Normal Score Transformation, Gaussian CDF inversion, back-transformation utilities |

---

## Package structure

```
stamps/
├── bme/                        # Bayesian Maximum Entropy core
│   ├── BMEprobaEstimations.py  # BMEPosteriorMoments, PDF, CI, Mode
│   ├── BMEoptions.py           # BME configuration options
│   ├── softconverter.py        # Soft data encoding utilities
│   ├── bme_transform.py        # Normal Score Transformation
│   └── pystks_variable.py      # Variable class
├── models/                     # Covariance and variogram model functions
│   └── covmodel.py
├── stest/                      # Spatial estimation
│   ├── kriging.py              # kriging, cokriging, cokrigingT
│   ├── BMEcatPrior.py          # Categorical BME prior
│   └── ...
├── stats/
│   ├── dependence/             # Spatial dependence modelling
│   │   ├── stcov.py            # Empirical covariance (stcov)
│   │   ├── stcovfit.py         # Model fitting (covmodelfit, coregfit)
│   │   ├── mlecovfit.py        # Maximum likelihood fitting
│   │   ├── anisotropy.py       # Anisotropy estimation
│   │   ├── probatablecalc.py   # Categorical dependence
│   │   └── probatablefit.py    # Categorical model fitting
│   ├── analysis/               # Spatiotemporal analysis
│   │   ├── eof.py              # EOF/PCA/HOSVD
│   │   ├── pca.py, mca.py, cca.py
│   │   ├── ent.py, mepdf.py    # Entropy / MEP-PDF
│   │   ├── ekf.py              # Empirical Kalman Filter
│   │   ├── gam.py, dlnm.py, stl.py  # R-based (optional)
│   │   └── ...
│   └── simulation/             # Random field generation
│       └── simulation.py       # simuchol, simucholcond, simuseqcond,
│                               # simuseqcondInt, simuseqcondME,
│                               # simuprobabilistic, simuinterval,
│                               # stationary_gaussian_process
├── general/                    # Utility functions
│   ├── coord2dist.py           # Distance matrices
│   ├── coord2K.py              # Covariance matrix assembly
│   ├── neighbours.py           # Neighbourhood selection
│   ├── findpairs.py            # Duplicate coordinate detection
│   └── valstvgx.py             # Space-time coordinate conversion
├── graph/                      # Visualisation helpers
│   ├── dataplot.py             # colorplot, histscatterplot
│   └── modelplot.py            # modelplot
└── mvn/                        # Multivariate normal integration
    └── mvn.py                  # QMC integration (mvPro, mvMomVec)
```

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
**[BMELIB_COMPARISON.md](./stamps/BMELIB_COMPARISON.md)**.

---

## Installation

### Requirements

**Core dependencies (installed automatically):**
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
| `nlopt` | Nonlinear optimisation for `covmodelfit`, `coregfit` |
| `xgboost` + `xgboostlss` | Categorical BME prior (`BMEcatPrior`) |
| `rpy2` + R | `gam`, `dlnm`, `stl` time-series decomposition |
| `nbformat` | Running/generating the tutorial notebooks |

### Install from source

```bash
# Clone the repository
git clone https://github.com/NTU-STEMLab/stamps.git

# Navigate to the package root and install in editable mode
cd stamps/stamps
pip install -e .
```

### Install into a specific conda environment

```bash
conda activate bme
pip install -e /path/to/stamps/stamps/
```

### Install optional dependencies

```bash
pip install matplotlib nlopt xgboost
# For R-based functions:
pip install rpy2
# Install R packages: zoo, dlnm, nlme, mgcv
```

### Verify installation

```bash
python -c "import stamps; print('stamps OK')"
pip show stamps   # should show Version: 1.0.0
```

---

## Quick start

```python
import numpy as np
from stamps.stamps.stest.kriging import kriging
from stamps.stamps.stats.dependence.stcov import stcov
from stamps.stamps.stats.dependence.stcovfit import covmodelfit
from stamps.stamps.bme.BMEprobaEstimations import BMEPosteriorMoments
from stamps.stamps.bme.softconverter import probaUniform

# --- 1. Fit a covariance model ---
lagS       = np.array([500, 1000, 2000, 4000, 6000])
lagS_range = 500 * np.ones(5)
lag_h, cov_h, n_h, _ = stcov(ch, np.array([[0]]), zh, lagS, lagS_range)

covmodel, covparam = covmodelfit(lag_h, cov_h, n_h,
                                  ['nuggetC', 'sphericalC'],
                                  [(5.0,), (80.0, 4000.0)])

# --- 2. Ordinary Kriging ---
zk_mean, zk_var = kriging(ck, ch, zh, covmodel, covparam,
                           nhmax=20, dmax=8000.0, order=0)

# --- 3. BME with interval soft data ---
zs = probaUniform(z_low, z_high)
result = BMEPosteriorMoments(
    ck,
    ch=ch, cs=cs,
    zh=zh.reshape(-1, 1),
    zs=tuple(zs),
    covmodel=covmodel, covparam=covparam,
    order=np.nan,
    nhmax=15, nsmax=8,
    dmax=np.array([[8000.0]]),
)
bme_mean = result[:, 0]
bme_var  = result[:, 1]
```

---

## Tutorials

Interactive Jupyter Notebook tutorials are in the `tutorials/` directory:

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

**Run tutorials in order** (each saves intermediate results used by the next).

```bash
conda activate bme
pip install nbformat matplotlib
cd tutorials/
jupyter lab
```

---

## License

STAMPS is released under the [GNU General Public License v3.0](LICENSE).

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
