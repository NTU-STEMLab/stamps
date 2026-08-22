# STAMPS vs BMElib — Detailed Comparison

STAMPS inherits its core spatiotemporal geostatistics engine from
**BMElib** ([Serre & Christakos, UNC](https://mserre.sph.unc.edu/BMElib_web/)),
the original MATLAB reference implementation of Bayesian Maximum Entropy (BME) theory.
The fundamental BME mathematics — posterior moment computation, soft data encoding,
space-time covariance models, and random field simulation — are all rooted in BMElib.

This page provides a detailed account of the design differences, algorithmic
improvements, and new capabilities introduced in STAMPS relative to the original
MATLAB library.

---

## Key improvements over BMElib

| # | Area | BMElib (MATLAB) | STAMPS (Python) |
|---|---|---|---|
| 1 | **Language & cost** | MATLAB (proprietary licence required) | Python (free, open-source; runs on any platform) |
| 2 | **All-Gaussian analytical fast-path** | Always triggers numerical integration even when soft data are Gaussian | Detects the pure-Gaussian case at runtime and returns **exact closed-form** posterior moments, PDF, mode, and credible intervals — no integration, no optimisation |
| 3 | **Numerical integration** | Standard adaptive quadrature | **Quasi-Monte Carlo (QMC)** integration via `scipy.stats.qmc` (Sobol / Halton sequences), delivering higher accuracy per sample, especially in high dimensions |
| 4 | **Numerical stability** | Standard `inv()` / `\` operator; fails on near-singular matrices | `_robust_SVD` / `_robust_pinv` with condition-number thresholding; automatic PSD jitter (`+ ε I`) applied when a covariance matrix is near-singular |
| 5 | **Adaptive neighbourhood** | Fixed neighbourhood; estimation silently degrades with sparse data | Automatically expands the search radius when fewer than `nhmin` neighbours are found within `dmax`, with configurable fallback strategy |
| 6 | **Non-negativity truncation** | No enforcement; negative variances or PDF ordinates possible under numerical noise | All posterior variances and PDF values are clipped to `≥ 0`, preventing physically impossible outputs |
| 7 | **Normal Score Transformation (NST)** | Separate helper scripts with no unified interface | Fully integrated `bme_transform.py` module exposing `other2gauss`, `gaussinv`, `gausspdf`, `transformderiv`, `pdfgauss2other` with consistent call signatures |
| 8 | **Multivariate LMC fitting** | `coregfit.m` — single-pass Weighted Least Squares (WLS) | `coregfit` — **Iterated Alternating WLS (IALS)**: alternates between fitting sill coefficients and projecting sill matrices onto the positive-semidefinite cone, converging to a valid LMC |
| 9 | **Cokriging with transformation** | `krigingT.m` — univariate only (one target variable) | `cokrigingT` — generalised to the **multivariate** setting; accepts per-variable reference CDFs and back-transforms each co-variable independently |
| 10 | **Variogram models** | Covariance models only (`*C` family) | Adds a full **variogram (`*V`) model family**: `nuggetV`, `exponentialV`, `sphericalV`, `gaussianV`, `holecosV`, `holesinV`, `linearV`, `powerV` — enabling direct variogram-based fitting workflows |
| 11 | **Categorical dependence** | Not available | `probatablecalc` estimates the empirical **conditional probability table** of categorical spatial data across distance classes, analogous to a variogram for discrete variables |
| 12 | **Random number reproducibility** | `rand` / `randn` modify a global MATLAB state | All simulation functions accept an explicit `seed` argument passed to `numpy.random.default_rng(seed)` — every run is **exactly reproducible** |
| 13 | **Parallel processing** | Serial `for` loops over estimation grid nodes | `BMEPosteriorMoments` (and related) accept an `n_workers` argument; grid nodes are distributed across threads via `concurrent.futures.ThreadPoolExecutor` |
| 14 | **Expanded analysis toolbox** | Limited to BME/kriging and simulation | EOF/PCA (`eof`), MCA, CCA, Empirical Kalman Filter (`ekf`), MEP-PDF (`mepdf`), Entropy (`ent`), GAM, DLNM, STL decomposition — far beyond BMElib's scope |
| 15 | **Modern Python conventions** | MATLAB 1-based indexing throughout | Strict **0-based indexing** in all internal logic; `numpy`/`scipy` vectorisation replaces MATLAB loops; Python type hints on all public functions; NumPy-style docstrings |

---

## BMElib function → STAMPS mapping

The tables below map every translated MATLAB function to its Python equivalent in STAMPS.

### Posterior estimation (`bmeprobalib`)

| BMElib function | STAMPS equivalent | Module |
|---|---|---|
| `BMEprobaMoments.m` | `BMEPosteriorMoments` | `bme/BMEprobaEstimations.py` |
| `BMEprobaPdf.m` | `BMEPosteriorPDF_grid` | `bme/BMEprobaEstimations.py` |
| `BMEprobaMode.m` | `BMEPosteriorMode` | `bme/BMEprobaEstimations.py` |
| `BMEprobaCI.m` | `BMEPosteriorCI` | `bme/BMEprobaEstimations.py` |
| `BMEprobaTMode.m` | `BMEprobaTMode` | `bme/BMEprobaEstimations.py` |
| `BMEprobaTPdf.m` | `BMEprobaTPdf` | `bme/BMEprobaEstimations.py` |

### Soft data encoding (`bmeprobalib`)

| BMElib function | STAMPS equivalent | Module |
|---|---|---|
| `probaUniform.m` | `probaUniform` | `bme/softconverter.py` |
| `probaGaussian.m` | `probaGaussian` | `bme/softconverter.py` |
| `probaStudentT.m` | `probaStudentT` | `bme/softconverter.py` |
| `probasplit.m` | `probasplit` | `bme/softconverter.py` |
| `probaoffset.m` | `probaoffset` | `bme/softconverter.py` |
| `probaneighbours.m` | `probaneighbours` | `bme/softconverter.py` |
| `probacombinedupli.m` | `probacombineddupli` | `bme/softconverter.py` |
| `probacat.m` | `probacat` | `bme/softconverter.py` |
| `proba2val.m` | `proba2val` | `bme/softconverter.py` |
| `proba2stat.m` | `proba2stat` | `bme/softconverter.py` |
| `proba2interval.m` | `proba2interval` | `bme/softconverter.py` |
| `proba2probdens.m` | `proba2probdens` | `bme/softconverter.py` |

### Normal Score Transformation (`bmeprobalib`)

| BMElib function | STAMPS equivalent | Module |
|---|---|---|
| `other2gauss.m` | `other2gauss` | `bme/bme_transform.py` |
| `gaussinv.m` | `gaussinv` | `bme/bme_transform.py` |
| `gausspdf.m` | `gausspdf` | `bme/bme_transform.py` |
| `transformderiv.m` | `transformderiv` | `bme/bme_transform.py` |
| `pdfgauss2other.m` | `pdfgauss2other` | `bme/bme_transform.py` |

### Kriging & cokriging (`kriginglib`)

| BMElib function | STAMPS equivalent | Module |
|---|---|---|
| `kriging.m` | `kriging` | `stest/kriging.py` |
| `krigingMP.m` | `cokriging` | `stest/kriging.py` |
| `krigingT.m` | `cokrigingT` | `stest/kriging.py` |

### Covariance / variogram fitting (`modelslib` + `statlibm`)

| BMElib function | STAMPS equivalent | Module |
|---|---|---|
| `coregfit.m` | `coregfit` | `stats/dependence/stcovfit.py` |
| `covmodelfit` (WLS) | `covmodelfit` | `stats/dependence/stcovfit.py` |
| `stcov.m` | `stcov` | `stats/dependence/stcov.py` |
| `mlecovfit.m` | `mlecovfit` | `stats/dependence/mlecovfit.py` |

### Simulation (`simulib`)

| BMElib function | STAMPS equivalent | Module |
|---|---|---|
| `simuchol.m` | `simuchol` | `stats/simulation/simulation.py` |
| `anisosimuchol.m` | `anisosimuchol` | `stats/simulation/simulation.py` |
| `simucholcond.m` | `simucholcond` | `stats/simulation/simulation.py` |
| `simucholcondME.m` | `simucholcondME` | `stats/simulation/simulation.py` |
| `simuseq.m` | `simuseq` | `stats/simulation/simulation.py` |
| `simuseqcond.m` | `simuseqcond` | `stats/simulation/simulation.py` |
| `simuseqcondME.m` | `simuseqcondME` | `stats/simulation/simulation.py` |
| `simuseqcondInt.m` | `simuseqcondInt` | `stats/simulation/simulation.py` |
| `simuprobabilistic.m` | `simuprobabilistic` | `stats/simulation/simulation.py` |
| `simuinterval.m` | `simuinterval` | `stats/simulation/simulation.py` |
| `stationary_gaussian_process` (CE) | `stationary_gaussian_process` | `stats/simulation/simulation.py` |

---

## Capabilities present in STAMPS but absent from BMElib

The following capabilities are **new** to STAMPS and have no direct counterpart in BMElib:

| Capability | STAMPS module | Description |
|---|---|---|
| **EOF / HOSVD** | `stats/analysis/eof.py` | Empirical Orthogonal Functions, multiway HOSVD |
| **PCA / MCA / CCA** | `stats/analysis/pca.py`, `mca.py`, `cca.py` | Classical multivariate analysis |
| **Empirical Kalman Filter** | `stats/analysis/ekf.py` | Sequential data assimilation |
| **MEP-PDF** | `stats/analysis/mepdf.py` | Maximum Entropy Principle PDF estimation |
| **Entropy** | `stats/analysis/ent.py` | Shannon entropy of spatial fields |
| **GAM / DLNM / STL** | `stats/analysis/gam.py`, `dlnm.py`, `stl.py` | R-based time-series analysis (via `rpy2`) |
| **Categorical dependence** | `stats/dependence/probatablecalc.py` | Probability table estimation for categorical data |
| **Categorical BME prior** | `stest/BMEcatPrior.py` | XGBoost-LSS-based categorical prior construction |
| **Variogram models** | `models/covmodel.py` (`*V` family) | `nuggetV`, `exponentialV`, `sphericalV`, … |
| **Anisotropy estimation** | `stats/dependence/anisotropy.py` | Directional variogram fitting and anisotropy ratio |
| **QMC multivariate integration** | `mvn/mvn.py` | `mvPro`, `mvMomVec` via Quasi-Monte Carlo |

---

*For installation instructions, quick-start examples, and tutorial notebooks, return to the
[main README](README.md).*
