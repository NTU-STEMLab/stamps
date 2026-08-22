# -*- coding: utf-8 -*-
"""Generate 02_covariance_and_variogram_models.ipynb"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb.metadata['language_info'] = {'name': 'python', 'version': '3.9.0'}
cells = []

cells.append(nbf.v4.new_markdown_cell(
"# Tutorial 02 - Covariance & Variogram Model Gallery + Categorical Spatial Dependence\n\n"
"**Package:** stamps_v3  \n"
"**MATLAB equivalent:** `MODELSLIBtutorial.m`  \n"
"**Dataset:** Falmagne soil dataset (soil-type codes for categorical analysis)\n\n"
"---\n\n"
"## Learning objectives\n\n"
"1. Visualise the full gallery of **covariance models** (`*C` functions).\n"
"2. Visualise the full gallery of **variogram models** (`*V` functions).\n"
"3. Demonstrate the mathematical duality: $C(h) = C(0) - \\gamma(h)$.\n"
"4. Build and plot a **nested covariance model** (nugget + two spherical structures).\n"
"5. Compute **empirical bivariate probability tables** for the categorical soil-type "
"variable using `probatablecalc`, revealing the spatial dependence between soil categories.\n"
))

cells.append(nbf.v4.new_code_cell(
"import sys, os, pickle\n"
"import numpy as np\n"
"import pandas as pd\n"
"import matplotlib.pyplot as plt\n"
"import matplotlib.gridspec as gridspec\n\n"
"REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))\n"
"if REPO_ROOT not in sys.path:\n"
"    sys.path.insert(0, REPO_ROOT)\n\n"
"from stamps.stamps.models.covmodel import (\n"
"    nuggetC, exponentialC, sphericalC, gaussianC, holecosC, holesinC, mexicanhatC, maternC,\n"
"    nuggetV, exponentialV, sphericalV, gaussianV, holecosV, holesinV, linearV, powerV,\n"
"    get_model,\n"
")\n"
"from stamps.stamps.general.coord2K import coord2K\n\n"
"plt.rcParams.update({'figure.dpi': 110, 'font.size': 10})\n"
"DATA_DIR = os.path.join(os.getcwd(), 'data')\n\n"
"# Load data from Tutorial 01\n"
"with open(os.path.join(DATA_DIR, 'tutorial01_data.pkl'), 'rb') as f:\n"
"    d = pickle.load(f)\n"
"ch   = d['ch']; code = d['code']\n"
"print('Loaded Falmagne data:', ch.shape)\n"
))

cells.append(nbf.v4.new_markdown_cell(
"## 1. Distance vector for plotting\n\n"
"All covariance / variogram models in stamps_v3 accept a **distance matrix** as "
"their first argument.  For 1-D plotting we pass a row vector of lags.\n"
))

cells.append(nbf.v4.new_code_cell(
"h = np.linspace(0, 2.0, 300)  # dimensionless lag (units of the range)\n"
"# practical range ar=1 throughout this section for visual comparison\n"
"ar = 1.0\n"
"sill = 1.0\n"
))

cells.append(nbf.v4.new_markdown_cell(
"## 2. Covariance model gallery\n\n"
"All `*C` functions follow: `C = model(dist, sill, ar)`, where\n"
"- `sill` is the variance (plateau value at $h = 0$)\n"
"- `ar` is the **practical range** (distance where $C \\approx 0.05 \\cdot \\text{sill}$)\n\n"
"> **Exception:** `holecosC(dist, a_half, p_half)` uses amplitude/periodicity parameterisation.  \n"
"> **Exception:** `maternC(dist, sill, [ar, shape])` takes a list for the last parameter.  \n"
"> **Exception:** `mexicanhatC(dist, sill, b, c)` takes three shape parameters.\n"
))

cells.append(nbf.v4.new_code_cell(
"fig, axes = plt.subplots(2, 4, figsize=(18, 8))\n"
"axes = axes.ravel()\n\n"
"cov_defs = [\n"
"    ('nuggetC',      lambda h: nuggetC(h, sill),                    'Nugget'),\n"
"    ('exponentialC', lambda h: exponentialC(h, sill, ar),           'Exponential'),\n"
"    ('sphericalC',   lambda h: sphericalC(h, sill, ar),             'Spherical'),\n"
"    ('gaussianC',    lambda h: gaussianC(h, sill, ar),              'Gaussian'),\n"
"    ('holecosC',     lambda h: holecosC(h, 0.5, 0.5),              'Hole cosine\\n(a=0.5, p=0.5)'),\n"
"    ('holesinC',     lambda h: holesinC(h, sill, ar),               'Hole sine'),\n"
"    ('mexicanhatC',  lambda h: mexicanhatC(h, sill, 2.0, 1.5),     'Mexican hat\\n(b=2, c=1.5)'),\n"
"    ('maternC',      lambda h: maternC(h, sill, [ar, 1.5]),         'Matern (nu=1.5)'),\n"
"]\n\n"
"for ax, (name, fn, title) in zip(axes, cov_defs):\n"
"    ax.plot(h, fn(h), 'steelblue', lw=2.5)\n"
"    ax.axhline(0, color='k', lw=0.6, ls='--')\n"
"    ax.set_title(title, fontsize=10)\n"
"    ax.set_xlabel('Lag h'); ax.set_ylabel('C(h)')\n"
"    ax.set_xlim([0, 2]); ax.set_ylim(bottom=min(-0.05, fn(h).min() - 0.1))\n\n"
"plt.suptitle('Covariance model gallery  (sill=1, ar=1)', fontsize=13, y=1.01)\n"
"plt.tight_layout(); plt.show()\n"
))

cells.append(nbf.v4.new_markdown_cell(
"## 3. Variogram model gallery\n\n"
"Variogram functions $\\gamma(h)$ satisfy $\\gamma(0) = 0$ and increase with lag.\n"
"All `*V` functions follow: `V = model(dist, sill, ar)` (same convention as `*C`).  \n\n"
"Two **intrinsic** models (`linearV`, `powerV`) have no finite sill:\n"
"- `linearV(dist, slope)` — slope replaces sill\n"
"- `powerV(dist, slope, exponent)` — slope + exponent\n"
))

cells.append(nbf.v4.new_code_cell(
"fig, axes = plt.subplots(2, 4, figsize=(18, 8))\n"
"axes = axes.ravel()\n\n"
"var_defs = [\n"
"    ('nuggetV',      lambda h: nuggetV(h, sill),                   'Nugget'),\n"
"    ('exponentialV', lambda h: exponentialV(h, sill, ar),          'Exponential'),\n"
"    ('sphericalV',   lambda h: sphericalV(h, sill, ar),            'Spherical'),\n"
"    ('gaussianV',    lambda h: gaussianV(h, sill, ar),             'Gaussian'),\n"
"    ('holecosV',     lambda h: holecosV(h, 0.5, 0.5),             'Hole cosine\\n(a=0.5, p=0.5)'),\n"
"    ('holesinV',     lambda h: holesinV(h, sill, ar),              'Hole sine'),\n"
"    ('linearV',      lambda h: linearV(h, 0.5),                   'Linear (slope=0.5)'),\n"
"    ('powerV',       lambda h: powerV(h, 0.5, 1.5),               'Power (slope=0.5, exp=1.5)'),\n"
"]\n\n"
"for ax, (name, fn, title) in zip(axes, var_defs):\n"
"    ax.plot(h, fn(h), 'tomato', lw=2.5)\n"
"    ax.axhline(0, color='k', lw=0.6, ls='--')\n"
"    ax.set_title(title, fontsize=10)\n"
"    ax.set_xlabel('Lag h'); ax.set_ylabel('gamma(h)')\n"
"    ax.set_xlim([0, 2]); ax.set_ylim(bottom=-0.05)\n\n"
"plt.suptitle('Variogram model gallery  (sill=1, ar=1 where applicable)', fontsize=13, y=1.01)\n"
"plt.tight_layout(); plt.show()\n"
))

cells.append(nbf.v4.new_markdown_cell(
"## 4. Mathematical duality: $C(h)$ vs $\\gamma(h)$\n\n"
r"For a stationary random field with variance $\sigma^2 = C(0)$:"
"\n\n"
r"$$\gamma(h) = C(0) - C(h)$$"
"\n\n"
"This means a covariance model and its variogram counterpart carry exactly the same information.  "
"stamps_v3 provides both for user convenience — you can work in whichever representation you prefer.\n"
))

cells.append(nbf.v4.new_code_cell(
"fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
"models = [('Exponential', exponentialC, exponentialV),\n"
"          ('Spherical',   sphericalC,   sphericalV),\n"
"          ('Gaussian',    gaussianC,    gaussianV)]\n\n"
"for ax, (name, Cfn, Vfn) in zip(axes, models):\n"
"    C = Cfn(h, sill, ar)\n"
"    V = Vfn(h, sill, ar)\n"
"    ax.plot(h, C,            'steelblue', lw=2.5, label='C(h)   covariance')\n"
"    ax.plot(h, V,            'tomato',    lw=2.5, label='gamma(h) variogram')\n"
"    ax.plot(h, C + V,        'k--',       lw=1,   label='C(h)+gamma(h)=C(0)')\n"
"    ax.axhline(sill, color='gray', ls=':', lw=1)\n"
"    ax.set_xlabel('Lag h'); ax.set_title(f'{name}: C(h) vs gamma(h)')\n"
"    ax.legend(fontsize=8)\n\n"
"plt.tight_layout(); plt.show()\n"
"print('C(0) =', sill, '  [dashed line == C(0)]')\n"
))

cells.append(nbf.v4.new_markdown_cell(
"## 5. Nested covariance model\n\n"
"A **nested model** sums several basic structures to capture variability at "
"multiple spatial scales.  In stamps_v3, a nested model is specified as:\n\n"
"```python\n"
"covmodel = ['nuggetC', 'sphericalC', 'sphericalC']\n"
"covparam = [(c0,), (c1, a1), (c2, a2)]\n"
"```\n\n"
"and evaluated via `coord2K(c1, c2, covmodel, covparam)`.\n"
))

cells.append(nbf.v4.new_code_cell(
"# Nested model: nugget + short-range spherical + long-range spherical\n"
"c0, c1, c2 = 0.15, 0.45, 0.40     # sills (sum = 1.0)\n"
"a1, a2     = 2_000, 8_000          # ranges in metres\n\n"
"# Evaluate manually over a lag vector for plotting\n"
"h_m = np.linspace(0, 15_000, 500)  # lags in metres\n"
"C_nested = (nuggetC(h_m, c0)\n"
"           + sphericalC(h_m, c1, a1)\n"
"           + sphericalC(h_m, c2, a2))\n\n"
"# Plot\n"
"fig, ax = plt.subplots(figsize=(8, 4))\n"
"ax.plot(h_m/1000, nuggetC(h_m, c0),          'gray',       ls='--', lw=1.5, label=f'Nugget  c0={c0}')\n"
"ax.plot(h_m/1000, sphericalC(h_m, c1, a1),   'steelblue',  ls='--', lw=1.5, label=f'Sph1    c1={c1}, a={a1} m')\n"
"ax.plot(h_m/1000, sphericalC(h_m, c2, a2),   'darkorange', ls='--', lw=1.5, label=f'Sph2    c2={c2}, a={a2} m')\n"
"ax.plot(h_m/1000, C_nested,                   'k-',         lw=2.5,  label='Nested sum')\n"
"ax.axhline(c0 + c1 + c2, color='gray', ls=':', lw=1)\n"
"ax.set_xlabel('Lag (km)'); ax.set_ylabel('C(h)')\n"
"ax.set_title('Nested covariance: nugget + Sph1 + Sph2'); ax.legend()\n"
"plt.tight_layout(); plt.show()\n"
))

cells.append(nbf.v4.new_markdown_cell(
"## 6. Categorical spatial dependence — `probatablecalc`\n\n"
"The Falmagne dataset contains a **soil type code** (1–4).  "
"Unlike a continuous variable, its spatial dependence cannot be captured by a "
"single covariance function.  Instead, we use **bivariate probability tables** "
"$P_{k\\ell}(h)$, defined as:\n\n"
r"$$P_{k\ell}(h) = \Pr\!\left[Z(\mathbf{s}) = k \;\wedge\; Z(\mathbf{s}+\mathbf{h}) = \ell\right]$$"
"\n\nAt lag $h = 0$, the diagonal entries $P_{kk}(0)$ equal the **proportion** of category $k$.  "
"As lag increases, $P_{kk}(h)$ decays toward $p_k^2$ (independence), "
"revealing the spatial persistence of each soil type.\n\n"
"stamps_v3 function: `stamps.stamps.stats.dependence.probatablecalc.probatablecalc`\n"
))

cells.append(nbf.v4.new_code_cell(
"from stamps.stamps.stats.dependence import probatablecalc\n\n"
"# Build indicator (one-hot) matrix for soil types 1-4\n"
"categories = np.unique(code)\n"
"nc = len(categories)   # 4 soil types\n"
"indicators = np.zeros((len(code), nc))\n"
"for j, cat in enumerate(categories):\n"
"    indicators[:, j] = (code == cat).astype(float)\n\n"
"print(f'Categories: {categories}')\n"
"print('Indicator matrix shape:', indicators.shape)\n"
"print('Column sums (proportions):', indicators.sum(axis=0) / len(code))\n\n"
"# Distance bins: 0..12000 m in 1 km steps\n"
"coord_limit = np.arange(0, 13_000, 1_000).astype(float)\n\n"
"D, P, O = probatablecalc(ch, indicators, coord_limit)\n"
"print(f'Distance lags: {D.shape[0]} bins')\n"
"print(f'Probability table shape: {P.shape}  (nc x nc x nlags)')\n"
"print(f'Pairs per lag bin: {O}')\n"
))

cells.append(nbf.v4.new_code_cell(
"# Plot diagonal P_kk(h): spatial persistence of each soil type\n"
"fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
"cat_colors = ['tomato', 'steelblue', 'goldenrod', 'mediumseagreen']\n"
"cat_names  = [f'Type {k}' for k in categories]\n\n"
"# Proportions p_k (independence baseline)\n"
"p_k = indicators.mean(axis=0)\n\n"
"# Left: diagonal P_kk(h)\n"
"ax = axes[0]\n"
"for j, (col, lbl, pk) in enumerate(zip(cat_colors, cat_names, p_k)):\n"
"    pjj = P[j, j, :]           # P_kk for all lags\n"
"    valid = ~np.isnan(pjj)\n"
"    ax.plot(D[valid] / 1000, pjj[valid], 'o-', color=col, lw=2, ms=5, label=lbl)\n"
"    ax.axhline(pk**2, color=col, ls='--', lw=0.8, alpha=0.6)\n"
"ax.set_xlabel('Lag (km)'); ax.set_ylabel(r'$P_{kk}(h)$')\n"
"ax.set_title('Diagonal: spatial persistence of each soil type\\n(dashed = independence level $p_k^2$)')\n"
"ax.legend()\n\n"
"# Right: proportion at lag 0\n"
"ax = axes[1]\n"
"proportions = [P[j, j, 0] for j in range(nc)]\n"
"bars = ax.bar(cat_names, proportions, color=cat_colors, edgecolor='k', alpha=0.8)\n"
"for bar, v in zip(bars, proportions):\n"
"    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,\n"
"            f'{v:.3f}', ha='center', va='bottom', fontsize=9)\n"
"ax.set_ylabel('Estimated proportion'); ax.set_title('Category proportions (lag h=0)')\n\n"
"plt.tight_layout(); plt.show()\n"
))

cells.append(nbf.v4.new_code_cell(
"# Plot full 4x4 P matrix as a grid of subplots\n"
"fig, axes = plt.subplots(nc, nc, figsize=(14, 12), sharex=True)\n\n"
"for k in range(nc):\n"
"    for l in range(nc):\n"
"        ax   = axes[k, l]\n"
"        pkll = P[k, l, :]\n"
"        valid = ~np.isnan(pkll)\n"
"        if valid.any():\n"
"            ax.plot(D[valid] / 1000, pkll[valid], 'o-',\n"
"                    color=cat_colors[k], lw=1.5, ms=3)\n"
"        # independence baseline\n"
"        ax.axhline(p_k[k] * p_k[l], color='gray', ls='--', lw=0.8)\n"
"        if l == 0:\n"
"            ax.set_ylabel(cat_names[k], fontsize=8)\n"
"        if k == nc - 1:\n"
"            ax.set_xlabel('Lag (km)', fontsize=8)\n"
"        if k == 0:\n"
"            ax.set_title(cat_names[l], fontsize=8)\n\n"
"fig.suptitle('Bivariate probability tables for all soil-type pairs\\n(dashed = independence)', fontsize=12, y=1.01)\n"
"plt.tight_layout(); plt.show()\n"
))

cells.append(nbf.v4.new_markdown_cell(
"## 7. Summary\n\n"
"| What we covered | Key takeaway |\n"
"|---|---|\n"
"| `*C` model gallery | 8 covariance models; all take `(dist, sill, ar)` |\n"
"| `*V` model gallery | 8 variogram models; same convention + 2 intrinsic (no sill) |\n"
"| C/gamma duality | $C(h) + \\gamma(h) = C(0)$ always |\n"
"| Nested model | Sum basic structures to capture multi-scale variability |\n"
"| `probatablecalc` | Categorical analogue of the variogram; "
"reveals how soil-type persistence decays with distance |\n\n"
"**Next:** Tutorial 03 — Empirical covariance estimation and WLS model fitting.\n"
))

nb.cells = cells
import nbformat
with open("tutorials/02_covariance_and_variogram_models.ipynb", "w") as f:
    nbformat.write(nb, f)
print("Written: tutorials/02_covariance_and_variogram_models.ipynb")
