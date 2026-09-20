# -*- coding: utf-8 -*-
"""Generate 09_bme_categorical.ipynb"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb.metadata['language_info'] = {'name': 'python', 'version': '3.9.0'}
cells = []

# ---------------------------------------------------------------------------
# Cell 0 – Title & learning objectives
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"# Tutorial 09 – BME for Categorical Data\n\n"
"**Package:** stamps_v3  \n"
"**MATLAB equivalent:** `BMEPROBALIBtutorial.m` (categorical section) + `STATLIBtutorial.m`  \n"
"**Dataset:** Falmagne soil dataset – 118 georeferenced soil samples, Dinant area, Belgium\n\n"
"---\n\n"
"## Learning objectives\n\n"
"By the end of this notebook you will be able to:\n\n"
"1. **Represent** categorical observations as one-hot soft-data PDFs.\n"
"2. **Compute** empirical bivariate probability tables with `probatablecalc`.\n"
"3. **Fit** a smooth probability model (kernel regression) with `probatablefit`.\n"
"4. **Map** soil types over a regular grid using three estimators:\n"
"   - **BMEcatPdf** – full Maximum Entropy BME (most rigorous, slowest).\n"
"   - **MCPcatPdf** – Maximum Conditional Probability fast approximation.\n"
"   - **HBMEcatPdf** – Hybrid BME: MCP regional prior + local BME update.\n"
"5. **Validate** each method with Leave-One-Out cross-validation (LOOCV).\n"
"6. **Interpret** posterior category probability maps.\n\n"
"---\n\n"
"## Background: categorical BME in a nutshell\n\n"
"Categorical BME (Christakos & Li, 1998) extends classical BME to discrete\n"
"variables. Instead of a Gaussian covariance model, spatial dependence is\n"
"encoded by a **bivariate probability table** $P(X_i = c_1, X_j = c_2 \\mid h)$\n"
"that gives the joint probability of observing two categories at two locations\n"
"separated by lag $h$.\n\n"
"The **Maximum Entropy** principle integrates this prior dependence structure\n"
"with the observed data (hard or soft) to produce a **posterior marginal PDF**\n"
"$P(X_k = c \\mid \\text{data})$ at each estimation point $k$.  The MAP\n"
"(most probable category) is then taken as the point estimate.\n\n"
"### Workflow at a glance\n\n"
"```\n"
"Observations (soil codes 1–4)\n"
"      │\n"
"      ▼\n"
"One-hot PDFs  ──► probatablecalc  ──► empirical P(c1,c2|h)\n"
"                                           │\n"
"                                           ▼\n"
"                                   probatablefit  ──► smooth Pmodel(h)\n"
"                                                           │\n"
"                                                           ▼\n"
"                                    BMEcatPdf / MCPcatPdf / HBMEcatPdf\n"
"                                                           │\n"
"                                                           ▼\n"
"                                  pk[nk, nc]  ──► MAP class & probability maps\n"
"```\n"
))

# ---------------------------------------------------------------------------
# Cell 1 – Imports & setup
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
"import sys, os\n"
"import numpy as np\n"
"import pandas as pd\n"
"import matplotlib.pyplot as plt\n"
"import matplotlib.colors as mcolors\n"
"import matplotlib.patches as mpatches\n"
"from scipy import stats\n"
"\n"
"# ── make stamps importable ─────────────────────────────────────────────────\n"
"REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))\n"
"if REPO_ROOT not in sys.path:\n"
"    sys.path.insert(0, REPO_ROOT)\n"
"\n"
"# ── stamps categorical BME functions ───────────────────────────────────────\n"
"from stamps.categorical import (\n"
"    BMEcatPdf, MCPcatPdf, HBMEcatPdf, tune_regularization_loocv\n"
")\n"
"from stamps.stats.dependence import probatablecalc, probatablefit\n"
"\n"
"plt.rcParams.update({'figure.dpi': 110, 'font.size': 10})\n"
"DATA_DIR = os.path.join(os.getcwd(), 'data')\n"
"print('Data directory:', DATA_DIR)\n"
))

# ---------------------------------------------------------------------------
# Cell 2 – Load data
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 1. Load and explore the Falmagne dataset\n\n"
"The file follows the **GeoEAS** format (title, column-count, column names, data).\n"
"Column 6 (`soil type`) records one of four soil classes: **1=silt-loam,\n"
"2=sandy-loam, 3=loam, 4=loamy-sand** (Dinant, Belgium classification).\n"
))

cells.append(nbf.v4.new_code_cell(
"def read_geoeas(filepath):\n"
"    \"\"\"Parse a GeoEAS text file. Returns (header, col_names, data_array).\"\"\"\n"
"    with open(filepath) as fh:\n"
"        lines = fh.readlines()\n"
"    title   = lines[0].strip()\n"
"    ncols   = int(lines[1].strip())\n"
"    col_names = [lines[2 + k].strip() for k in range(ncols)]\n"
"    data = np.array(\n"
"        [list(map(float, ln.split())) for ln in lines[2 + ncols :] if ln.strip()]\n"
"    )\n"
"    return title, col_names, data\n"
"\n"
"title, col_names, data = read_geoeas(os.path.join(DATA_DIR, 'Falmagne.txt'))\n"
"print('Title   :', title)\n"
"print('Columns :', col_names)\n"
"print('Shape   :', data.shape)\n"
"\n"
"east  = data[:, 0]          # easting  (m)\n"
"north = data[:, 1]          # northing (m)\n"
"sand  = data[:, 2]          # sand content (%)\n"
"silt  = data[:, 3]          # silt content (%)\n"
"clay  = data[:, 4]          # clay content (%)\n"
"code  = data[:, 5].astype(int)   # soil type 1–4\n"
"\n"
"categories = np.array([1, 2, 3, 4])   # the four soil types\n"
"nc = len(categories)\n"
"n  = len(code)\n"
"print(f'\\n{n} observations  |  {nc} categories: {categories}')\n"
"print('Category counts:')\n"
"for c in categories:\n"
"    print(f'  Type {c}: {np.sum(code == c):3d} ({100*np.mean(code == c):.1f}%)')\n"
))

# ---------------------------------------------------------------------------
# Cell 3 – Spatial plot of categories
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"### 1.1 Spatial distribution of soil types\n\n"
"Each sampling location is coloured by its observed soil type.\n"
))

cells.append(nbf.v4.new_code_cell(
"CMAP4 = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']   # red / blue / green / orange\n"
"LABELS = ['Type 1 (silt-loam)', 'Type 2 (sandy-loam)',\n"
"          'Type 3 (loam)',       'Type 4 (loamy-sand)']\n"
"\n"
"fig, ax = plt.subplots(figsize=(6, 5))\n"
"for i, (cat, col, lab) in enumerate(zip(categories, CMAP4, LABELS)):\n"
"    mask = code == cat\n"
"    ax.scatter(east[mask]/1e3, north[mask]/1e3,\n"
"               c=col, s=40, label=lab, edgecolors='k', linewidths=0.4, zorder=3)\n"
"ax.set_xlabel('Easting (km)');  ax.set_ylabel('Northing (km)')\n"
"ax.set_title('Falmagne – observed soil types (n=118)')\n"
"ax.legend(fontsize=8, loc='upper right')\n"
"plt.tight_layout();  plt.show()\n"
))

# ---------------------------------------------------------------------------
# Cell 4 – One-hot encoding → soft PDFs
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 2. Encoding hard data as soft PDFs\n\n"
"BMEcatPdf expects **soft data** – a probability mass function (PMF) over all\n"
"categories at each data location.  For **hard data** (known category),\n"
"the PMF is a one-hot vector:\n\n"
"$$p_s^{(i)} = [\\mathbf{1}(c_i=1),\\; \\mathbf{1}(c_i=2),\\;\n"
"               \\mathbf{1}(c_i=3),\\; \\mathbf{1}(c_i=4)]$$\n\n"
"This is the degenerate (Dirac) distribution: certainty at the observed class.\n"
))

cells.append(nbf.v4.new_code_cell(
"# coordinates array\n"
"cs = np.column_stack([east, north])   # (118, 2)\n"
"\n"
"# One-hot encoding: ps[i, k] = 1 if code[i] == categories[k], else 0\n"
"ps = (code[:, None] == categories[None, :]).astype(float)  # (118, 4)\n"
"\n"
"print('cs shape:', cs.shape)\n"
"print('ps shape:', ps.shape)\n"
"print('First 5 rows of ps:')\n"
"print(ps[:5])\n"
"print('Corresponding codes:', code[:5])\n"
))

# ---------------------------------------------------------------------------
# Cell 5 – probatablecalc: empirical bivariate probability tables
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 3. Empirical bivariate probability tables\n\n"
"`probatablecalc` estimates the joint probability\n"
"$P(X_i = c_1, X_j = c_2)$ for all category pairs as a function of the\n"
"inter-point distance $h$.  The result encodes the **spatial dependence\n"
"structure** of the categorical field – the analogue of the variogram for\n"
"continuous variables.\n\n"
"### Parameters\n"
"- **`coord_limit`** – edges of distance bins (metres). The first bin captures\n"
"  collocated points ($h=0$); subsequent bins cover user-defined lag classes.\n"
"- **`chunk_size`** – memory-saving chunking (no effect on results).\n"
))

cells.append(nbf.v4.new_code_cell(
"# ── Distance bins ──────────────────────────────────────────────────────────\n"
"# Rule of thumb: cover up to ~1/2 of the max data extent, with ~15-20 bins.\n"
"d_extent = np.sqrt((east.max()-east.min())**2 + (north.max()-north.min())**2)\n"
"print(f'Data extent: {d_extent/1e3:.1f} km')\n"
"\n"
"d_max_emp  = 25_000   # 25 km  (roughly 60 % of the full extent)\n"
"n_bins     = 20\n"
"coord_limit = np.concatenate([[0], np.linspace(500, d_max_emp, n_bins)])\n"
"print(f'Distance limits (km): {np.round(coord_limit/1e3, 1)}')\n"
"\n"
"# ── Compute empirical tables ────────────────────────────────────────────────\n"
"print('\\nComputing empirical bivariate probability tables ...')\n"
"D_emp, P_emp, O_emp = probatablecalc(cs, ps, coord_limit)\n"
"print(f'Done.  D shape={D_emp.shape},  P shape={P_emp.shape},  O shape={O_emp.shape}')\n"
"print(f'Pair counts per bin (first 6): {O_emp[:6].astype(int)}')\n"
))

# ---------------------------------------------------------------------------
# Cell 6 – visualise empirical tables
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"### 3.1 Visualise empirical probability tables\n\n"
"Each panel below shows how $P(X_i = c_1, X_j = c_2 \\mid h)$ varies with lag\n"
"$h$ for one category pair.  **Diagonal panels** (same category) show\n"
"auto-indicator correlograms (high near $h=0$, decaying to the marginal\n"
"frequency squared).  **Off-diagonal panels** show cross-category transition\n"
"probabilities.\n"
))

cells.append(nbf.v4.new_code_cell(
"valid = O_emp > 0   # only bins with data\n"
"D_v   = D_emp[valid]\n"
"\n"
"fig, axes = plt.subplots(nc, nc, figsize=(10, 9), sharex=True, sharey=False)\n"
"fig.suptitle('Empirical bivariate probability tables – Falmagne soil types', y=1.01)\n"
"\n"
"for r in range(nc):\n"
"    for c_ in range(nc):\n"
"        ax = axes[r, c_]\n"
"        vals = P_emp[r, c_, :][valid]\n"
"        ax.plot(D_v/1e3, vals, 'o-', ms=3, color=CMAP4[r] if r == c_ else 'steelblue')\n"
"        ax.axhline(vals[-1] if len(vals) else 0, ls='--', lw=0.8, color='gray')\n"
"        if r == 0:\n"
"            ax.set_title(f'→ Type {c_+1}', fontsize=8)\n"
"        if c_ == 0:\n"
"            ax.set_ylabel(f'Type {r+1} →', fontsize=8)\n"
"        ax.tick_params(labelsize=7)\n"
"\n"
"fig.text(0.5, -0.01, 'Lag distance h (km)', ha='center', fontsize=9)\n"
"fig.text(-0.01, 0.5, 'P(X_i=row, X_j=col | h)', va='center',\n"
"         rotation='vertical', fontsize=9)\n"
"plt.tight_layout();  plt.show()\n"
))

# ---------------------------------------------------------------------------
# Cell 7 – probatablefit: smooth model
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 4. Fitting a smooth probability model\n\n"
"`probatablefit` applies **kernel regression smoothing** (Gaussian kernel) to\n"
"the empirical tables and returns fitted values at a finer distance grid\n"
"`dfit`.  This continuous `Pmodel` is what BMEcatPdf uses at prediction time.\n\n"
"### Key parameters\n"
"- **`kstd`** – kernel standard deviation (bandwidth). Larger → smoother.\n"
"  A good starting value is ≈ `0.5 * d_max / n_bins`.\n"
"- **`options[1]`** – polynomial order inside the kernel window (0 = constant).\n"
"- **`options[2]`** – weight by pair count (`1` = recommended).\n"
))

cells.append(nbf.v4.new_code_cell(
"# ── Fitting grid ───────────────────────────────────────────────────────────\n"
"n_fit  = 80\n"
"d_fit  = np.linspace(0, d_max_emp, n_fit)   # must not exceed max(D_emp)\n"
"\n"
"kstd   = 0.5 * d_max_emp / n_bins   # ≈ 625 m\n"
"opts   = [0, 0, 1]                  # no display / poly-0 / weighted\n"
"\n"
"print(f'kstd = {kstd/1e3:.2f} km')\n"
"print('Fitting smooth probability model ...')\n"
"Pmodel = probatablefit(d_fit, D_emp, P_emp, O_emp.reshape(-1, 1), kstd, opts)\n"
"print(f'Pmodel shape: {Pmodel.shape}   (nc × nc × n_fit)')\n"
"\n"
"# dfit / Pmodel are the inputs to BMEcatPdf\n"
"dmodel = d_fit.copy()\n"
))

# ---------------------------------------------------------------------------
# Cell 8 – plot fitted model
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
"fig, axes = plt.subplots(nc, nc, figsize=(10, 9), sharex=True)\n"
"fig.suptitle('Fitted smooth probability model (kernel regression)', y=1.01)\n"
"\n"
"for r in range(nc):\n"
"    for c_ in range(nc):\n"
"        ax = axes[r, c_]\n"
"        # empirical (dots)\n"
"        ax.plot(D_v/1e3, P_emp[r, c_, :][valid], 'o', ms=3,\n"
"                color='gray', alpha=0.6, label='empirical')\n"
"        # fitted (line)\n"
"        ax.plot(d_fit/1e3, Pmodel[r, c_, :], '-',\n"
"                color=CMAP4[r] if r == c_ else 'steelblue',\n"
"                lw=1.5, label='fitted')\n"
"        if r == 0:\n"
"            ax.set_title(f'→ Type {c_+1}', fontsize=8)\n"
"        if c_ == 0:\n"
"            ax.set_ylabel(f'Type {r+1} →', fontsize=8)\n"
"        ax.tick_params(labelsize=7)\n"
"\n"
"handles = [plt.Line2D([0],[0], ls='none', marker='o', color='gray', label='Empirical'),\n"
"           plt.Line2D([0],[0], lw=1.5, color='steelblue', label='Fitted model')]\n"
"fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=8)\n"
"fig.text(0.5, 0.01, 'Lag distance h (km)', ha='center', fontsize=9)\n"
"fig.text(-0.01, 0.5, 'P(X_i=row, X_j=col | h)', va='center',\n"
"         rotation='vertical', fontsize=9)\n"
"plt.tight_layout();  plt.show()\n"
))

# ---------------------------------------------------------------------------
# Cell 9 – build estimation grid
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 5. Build the estimation grid\n\n"
"We create a regular `nkx × nky` grid covering the data extent.\n"
))

cells.append(nbf.v4.new_code_cell(
"nkx, nky = 50, 50\n"
"x_grid = np.linspace(east.min(),  east.max(),  nkx)\n"
"y_grid = np.linspace(north.min(), north.max(), nky)\n"
"XX, YY = np.meshgrid(x_grid, y_grid)\n"
"ck = np.column_stack([XX.ravel(), YY.ravel()])   # (nk, 2)\n"
"nk = ck.shape[0]\n"
"print(f'Estimation grid: {nkx} × {nky} = {nk} nodes')\n"
))

# ---------------------------------------------------------------------------
# Cell 10 – neighbourhood parameters
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 6. Neighbourhood parameters\n\n"
"| Parameter | Description |\n"
"|-----------|-------------|\n"
"| `nsmax`   | Max soft-data neighbours to include in the ME table |\n"
"| `dmax`    | Maximum search radius (metres) |\n\n"
"**Guideline:** `dmax` should cover the correlation range of the fitted\n"
"probability model (~5–15 km for this dataset), and `nsmax ≤ 10` keeps\n"
"the ME optimisation tractable.\n"
))

cells.append(nbf.v4.new_code_cell(
"nsmax    = 8        # max neighbours in the ME table\n"
"dmax_bme = 15_000.  # 15 km search radius\n"
"\n"
"# Estimator options (shared by all three methods)\n"
"base_options = dict(\n"
"    show_progress  = 0,\n"
"    tol            = 1e-4,\n"
"    estimator      = 'MPL',    # Maximum Pseudo-Likelihood (fast & stable)\n"
"    model_structure= 'full',   # use all (nsmax choose 2) bivariate constraints\n"
")\n"
"print('nsmax:', nsmax, '  dmax:', dmax_bme/1e3, 'km')\n"
"print('Estimator options:', base_options)\n"
))

# ---------------------------------------------------------------------------
# Cell 11 – BMEcatPdf estimation
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 7. BME categorical estimation (BMEcatPdf)\n\n"
"**BMEcatPdf** is the rigorous Bayesian Maximum Entropy estimator.  For each\n"
"estimation node it:\n\n"
"1. Finds the `nsmax` nearest data points within `dmax`.\n"
"2. Builds the bivariate probability constraints from `Pmodel`.\n"
"3. Solves a Maximum Entropy optimisation to get the prior multivariate PMF\n"
"   $f(X_k, X_{s_1}, \\dots, X_{s_m})$.\n"
"4. Multiplies by the observed soft data PDFs to get the unnormalised\n"
"   posterior, then marginalises over the neighbours to obtain\n"
"   $P(X_k = c \\mid \\text{data})$.\n"
))

cells.append(nbf.v4.new_code_cell(
"import time\n"
"\n"
"bme_options = {**base_options, 'show_progress': 1}\n"
"\n"
"print(f'Running BMEcatPdf on {nk} nodes ...')\n"
"t0 = time.perf_counter()\n"
"pk_bme = BMEcatPdf(ck, cs, ps, dmodel, Pmodel,\n"
"                   nsmax=nsmax, dmax=dmax_bme, options=bme_options)\n"
"t_bme = time.perf_counter() - t0\n"
"\n"
"print(f'\\nBMEcatPdf finished in {t_bme:.1f}s  ({t_bme/nk*1000:.1f} ms/node)')\n"
"print('pk_bme shape:', pk_bme.shape)\n"
"\n"
"# MAP estimate: category with highest posterior probability\n"
"map_bme = categories[np.argmax(pk_bme, axis=1)]   # (nk,)\n"
"# Maximum posterior probability (confidence)\n"
"maxp_bme = pk_bme.max(axis=1)\n"
"print('BME MAP class distribution:', {c: int(np.sum(map_bme == c)) for c in categories})\n"
))

# ---------------------------------------------------------------------------
# Cell 12 – MCPcatPdf (fast)
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 8. MCP fast approximation (MCPcatPdf)\n\n"
"**MCPcatPdf** (Maximum Conditional Probability) approximates the BME posterior\n"
"by assuming that the data points are **conditionally independent given\n"
"the estimation point**.  This avoids the ME optimisation and is therefore\n"
"orders of magnitude faster, at the cost of some accuracy when data are\n"
"spatially clustered.\n"
))

cells.append(nbf.v4.new_code_cell(
"mcp_options = {**base_options, 'show_progress': 0}\n"
"\n"
"print(f'Running MCPcatPdf on {nk} nodes ...')\n"
"t0 = time.perf_counter()\n"
"pk_mcp = MCPcatPdf(ck, cs, ps, dmodel, Pmodel,\n"
"                   nsmax=nsmax, dmax=dmax_bme, options=mcp_options)\n"
"t_mcp = time.perf_counter() - t0\n"
"\n"
"print(f'MCPcatPdf finished in {t_mcp:.1f}s  ({t_mcp/nk*1000:.1f} ms/node)')\n"
"map_mcp  = categories[np.argmax(pk_mcp, axis=1)]\n"
"maxp_mcp = pk_mcp.max(axis=1)\n"
"print('MCP speed-up vs BME: ×{:.0f}'.format(t_bme / t_mcp if t_mcp > 0 else np.inf))\n"
))

# ---------------------------------------------------------------------------
# Cell 13 – HBMEcatPdf
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 9. Hybrid BME (HBMEcatPdf)\n\n"
"**HBMEcatPdf** implements a two-step hybrid approach:\n\n"
"1. **Step 1 — regional prior (MCP):** `MCPcatPdf` is called with `nsmax_total`\n"
"   neighbours, restricted to `dmax_mcp`, to build a spatially varying soft prior\n"
"   that captures the regional category trend.\n"
"2. **Step 2 — local update (BME):** `BMEcatPdf` is called with only the `nsmax`\n"
"   closest neighbours, using the MCP prior. The full ME table (all `nsmax` pairs)\n"
"   is solved exactly, accounting for local spatial redundancy.\n\n"
"### ⚠️  Critical parameter: `nsmax_total` and `dmax_mcp`\n\n"
"At distances **beyond the covariance correlation range** the bivariate probability\n"
"table factorises to the independence product $P(c')\\cdot P(c)$.  Each such\n"
"out-of-range neighbour adds $\\log P(c)$ to the MCP log-posterior, causing:\n\n"
"$$\\log p(X_0{=}c) \\approx \\log P(c) \\quad\\text{(global marginal, same everywhere)}$$\n\n"
"**If `nsmax_total` is too large (e.g. 40 out of 118 data points), the MCP prior\n"
"degenerates to the global category frequencies at every estimation location, making\n"
"the final HBME map nearly flat.**\n\n"
"**Rule of thumb:**\n"
"- Set `dmax_mcp` ≈ the covariance correlation range (here ~8 km from the fitted model).\n"
"- Set `nsmax_total` ≈ 2–3 × `nsmax` (i.e. the expected number of data points\n"
"  within `dmax_mcp`).\n"
))

cells.append(nbf.v4.new_code_cell(
"# dmax_mcp limits the MCP prior search to the correlation range (~8 km).\n"
"# Only neighbours within that radius have meaningful bivariate correlation\n"
"# and can contribute spatial information to the prior.\n"
"dmax_mcp   = 8_000.    # ≈ covariance correlation range (metres)\n"
"nsmax_total = 15       # max neighbours within dmax_mcp for the MCP prior\n"
"\n"
"hbme_options = {**base_options,\n"
"                'nsmax_total': nsmax_total,\n"
"                'dmax_mcp'   : dmax_mcp,\n"
"                'show_progress': 0}\n"
"\n"
"print(f'Running HBMEcatPdf on {nk} nodes ...')\n"
"print(f'  nsmax={nsmax} (BME step)  nsmax_total={nsmax_total} (MCP prior step)')\n"
"print(f'  dmax_bme={dmax_bme/1e3:.0f} km   dmax_mcp={dmax_mcp/1e3:.0f} km')\n"
"t0 = time.perf_counter()\n"
"pk_hbme = HBMEcatPdf(ck, cs, ps, dmodel, Pmodel,\n"
"                     nsmax=nsmax, dmax=dmax_bme, options=hbme_options)\n"
"t_hbme = time.perf_counter() - t0\n"
"\n"
"print(f'HBMEcatPdf finished in {t_hbme:.1f}s  ({t_hbme/nk*1000:.1f} ms/node)')\n"
"map_hbme  = categories[np.argmax(pk_hbme, axis=1)]\n"
"maxp_hbme = pk_hbme.max(axis=1)\n"
))

# ---------------------------------------------------------------------------
# Cell 14 – visualise MAP maps
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 10. MAP class maps\n\n"
"Each cell is coloured by its **most probable category** under each method.\n"
"The data locations are overlaid as black-bordered circles.\n"
))

cells.append(nbf.v4.new_code_cell(
"# Build a ListedColormap for the 4 soil types\n"
"cmap_cat  = mcolors.ListedColormap(CMAP4)\n"
"bounds_cat = [0.5, 1.5, 2.5, 3.5, 4.5]\n"
"norm_cat   = mcolors.BoundaryNorm(bounds_cat, cmap_cat.N)\n"
"\n"
"def plot_map_cat(ax, values_grid, title, cs_data, code_data):\n"
"    \"\"\"Plot a categorical map with data overlay.\"\"\"\n"
"    im = ax.pcolormesh(\n"
"        XX/1e3, YY/1e3,\n"
"        values_grid.reshape(nky, nkx),\n"
"        cmap=cmap_cat, norm=norm_cat, shading='auto'\n"
"    )\n"
"    # data overlay – colour by true category\n"
"    for i, (cat, col) in enumerate(zip(categories, CMAP4)):\n"
"        mask = code_data == cat\n"
"        ax.scatter(cs_data[mask, 0]/1e3, cs_data[mask, 1]/1e3,\n"
"                   c=col, s=25, edgecolors='k', linewidths=0.5, zorder=4)\n"
"    ax.set_title(title, fontsize=9)\n"
"    ax.set_xlabel('Easting (km)', fontsize=8)\n"
"    ax.set_ylabel('Northing (km)', fontsize=8)\n"
"    return im\n"
"\n"
"patches = [mpatches.Patch(color=CMAP4[i], label=LABELS[i]) for i in range(nc)]\n"
"\n"
"fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n"
"plot_map_cat(axes[0], map_bme,  'BMEcatPdf – MAP class',  cs, code)\n"
"plot_map_cat(axes[1], map_mcp,  'MCPcatPdf – MAP class',  cs, code)\n"
"plot_map_cat(axes[2], map_hbme, 'HBMEcatPdf – MAP class', cs, code)\n"
"\n"
"fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=8,\n"
"           bbox_to_anchor=(0.5, -0.04))\n"
"plt.tight_layout()\n"
"plt.show()\n"
))

# ---------------------------------------------------------------------------
# Cell 15 – posterior uncertainty maps
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 11. Posterior uncertainty maps\n\n"
"The **maximum posterior probability** (i.e. the probability assigned to the\n"
"MAP class) measures classification confidence.  Values near 0.25 (uniform\n"
"over 4 classes) indicate high uncertainty; values near 1 indicate near-certain\n"
"classification.\n"
))

cells.append(nbf.v4.new_code_cell(
"def plot_map_prob(ax, prob_grid, title):\n"
"    im = ax.pcolormesh(\n"
"        XX/1e3, YY/1e3,\n"
"        prob_grid.reshape(nky, nkx),\n"
"        cmap='YlOrRd', vmin=0.25, vmax=1.0, shading='auto'\n"
"    )\n"
"    plt.colorbar(im, ax=ax, label='P(MAP class)')\n"
"    ax.set_title(title, fontsize=9)\n"
"    ax.set_xlabel('Easting (km)', fontsize=8)\n"
"    ax.set_ylabel('Northing (km)', fontsize=8)\n"
"\n"
"fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
"plot_map_prob(axes[0], maxp_bme,  'BMEcatPdf – confidence')\n"
"plot_map_prob(axes[1], maxp_mcp,  'MCPcatPdf – confidence')\n"
"plot_map_prob(axes[2], maxp_hbme, 'HBMEcatPdf – confidence')\n"
"plt.tight_layout();  plt.show()\n"
))

# ---------------------------------------------------------------------------
# Cell 16 – individual category probability maps
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 12. Individual category probability maps (BMEcatPdf)\n\n"
"The full posterior PMF gives the probability of **each** category at every\n"
"grid node — not just the MAP class.  These \"probability of occurrence\" maps\n"
"are useful for uncertainty propagation and risk assessment.\n"
))

cells.append(nbf.v4.new_code_cell(
"fig, axes = plt.subplots(2, 2, figsize=(11, 9))\n"
"\n"
"for idx, (cat, ax, col) in enumerate(zip(categories, axes.ravel(), CMAP4)):\n"
"    prob_cat = pk_bme[:, idx].reshape(nky, nkx)\n"
"    # custom colormap: white → category colour\n"
"    cmap_i = mcolors.LinearSegmentedColormap.from_list(\n"
"        f'type{cat}', ['white', col]\n"
"    )\n"
"    im = ax.pcolormesh(XX/1e3, YY/1e3, prob_cat,\n"
"                       cmap=cmap_i, vmin=0, vmax=1, shading='auto')\n"
"    plt.colorbar(im, ax=ax, label=f'P(Type {cat})')\n"
"    # overlay data\n"
"    for c2 in categories:\n"
"        m = code == c2\n"
"        ax.scatter(cs[m, 0]/1e3, cs[m, 1]/1e3,\n"
"                   c=CMAP4[c2-1], s=20, edgecolors='k',\n"
"                   linewidths=0.4, zorder=4)\n"
"    ax.set_title(f'BME P(soil = {cat}: {LABELS[idx]})', fontsize=9)\n"
"    ax.set_xlabel('Easting (km)', fontsize=8)\n"
"    ax.set_ylabel('Northing (km)', fontsize=8)\n"
"\n"
"plt.tight_layout();  plt.show()\n"
))

# ---------------------------------------------------------------------------
# Cell 17 – LOOCV validation
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 13. Leave-One-Out Cross-Validation (LOOCV)\n\n"
"We evaluate predictive accuracy by predicting each data point using all\n"
"remaining points, then comparing the predicted MAP class with the true class.\n\n"
"**Metrics used:**\n"
"- **Accuracy** – fraction of correctly predicted MAP classes.\n"
"- **Mean log-likelihood** – $\\frac{1}{n}\\sum_i \\log P(\\hat{X}_i = c_i)$;\n"
"  higher is better; measures probabilistic calibration.\n"
))

cells.append(nbf.v4.new_code_cell(
"def loocv_accuracy(cs_data, ps_data, code_data, dmodel, Pmodel,\n"
"                   nsmax, dmax, options, categories, method_fn):\n"
"    \"\"\"Leave-One-Out cross-validation for a categorical estimator.\"\"\"\n"
"    n = cs_data.shape[0]\n"
"    correct = 0\n"
"    log_liks = []\n"
"\n"
"    opts = {**options, 'show_progress': 0}\n"
"\n"
"    for i in range(n):\n"
"        ck_i   = cs_data[i:i+1, :]          # hold-out location\n"
"        cs_loo = np.delete(cs_data, i, 0)   # training coordinates\n"
"        ps_loo = np.delete(ps_data, i, 0)   # training PDFs\n"
"\n"
"        pk_i = method_fn(ck_i, cs_loo, ps_loo,\n"
"                         dmodel, Pmodel, nsmax, dmax, options=opts)\n"
"        pred_cat = categories[np.argmax(pk_i[0])]\n"
"        true_cat = code_data[i]\n"
"        correct += (pred_cat == true_cat)\n"
"\n"
"        # log-likelihood of the true category\n"
"        true_idx = np.where(categories == true_cat)[0][0]\n"
"        prob_true = np.clip(pk_i[0, true_idx], 1e-15, 1.0)\n"
"        log_liks.append(np.log(prob_true))\n"
"\n"
"    accuracy   = correct / n\n"
"    mean_ll    = np.mean(log_liks)\n"
"    return accuracy, mean_ll\n"
"\n"
"print('Running LOOCV for BMEcatPdf  ...')\n"
"acc_bme,  ll_bme  = loocv_accuracy(cs, ps, code, dmodel, Pmodel,\n"
"                                    nsmax, dmax_bme, base_options,\n"
"                                    categories, BMEcatPdf)\n"
"print(f'  BME  – Accuracy: {acc_bme:.3f}  |  Mean log-likelihood: {ll_bme:.3f}')\n"
"\n"
"print('Running LOOCV for MCPcatPdf  ...')\n"
"acc_mcp,  ll_mcp  = loocv_accuracy(cs, ps, code, dmodel, Pmodel,\n"
"                                    nsmax, dmax_bme, base_options,\n"
"                                    categories, MCPcatPdf)\n"
"print(f'  MCP  – Accuracy: {acc_mcp:.3f}  |  Mean log-likelihood: {ll_mcp:.3f}')\n"
"\n"
"print('Running LOOCV for HBMEcatPdf ...')\n"
"hbme_loocv_opts = {**base_options,\n"
"                   'nsmax_total': nsmax_total,   # same as estimation cell\n"
"                   'dmax_mcp'   : dmax_mcp}\n"
"acc_hbme, ll_hbme = loocv_accuracy(cs, ps, code, dmodel, Pmodel,\n"
"                                    nsmax, dmax_bme, hbme_loocv_opts,\n"
"                                    categories, HBMEcatPdf)\n"
"print(f'  HBME – Accuracy: {acc_hbme:.3f}  |  Mean log-likelihood: {ll_hbme:.3f}')\n"
))

# ---------------------------------------------------------------------------
# Cell 18 – LOOCV bar chart
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
"methods   = ['BMEcatPdf', 'MCPcatPdf', 'HBMEcatPdf']\n"
"accuracies = [acc_bme,  acc_mcp,  acc_hbme]\n"
"log_liks_  = [ll_bme,   ll_mcp,   ll_hbme]\n"
"\n"
"fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
"\n"
"bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']\n"
"axes[0].bar(methods, accuracies, color=bar_colors, edgecolor='k')\n"
"axes[0].set_ylim(0, 1);  axes[0].set_ylabel('LOOCV Accuracy')\n"
"axes[0].set_title('Classification Accuracy (LOOCV)')\n"
"for i, v in enumerate(accuracies):\n"
"    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)\n"
"\n"
"axes[1].bar(methods, log_liks_, color=bar_colors, edgecolor='k')\n"
"axes[1].set_ylabel('Mean Log-Likelihood')\n"
"axes[1].set_title('Probabilistic Score – Mean Log-Likelihood (LOOCV)')\n"
"for i, v in enumerate(log_liks_):\n"
"    axes[1].text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=10)\n"
"\n"
"plt.tight_layout();  plt.show()\n"
"\n"
"print('\\nSummary table:')\n"
"df_res = pd.DataFrame({'Method': methods,\n"
"                        'LOOCV Accuracy': accuracies,\n"
"                        'Mean Log-Lik' : log_liks_})\n"
"print(df_res.to_string(index=False))\n"
))

# ---------------------------------------------------------------------------
# Cell 19 – Estimator comparison side-by-side per category
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 14. Per-category precision recall\n\n"
"We compute a simple **per-class precision and recall** from the LOOCV\n"
"predictions (MAP class vs. true class) to see which soil types are most\n"
"or least well predicted.\n"
))

cells.append(nbf.v4.new_code_cell(
"def loocv_predictions(cs_data, ps_data, code_data, dmodel, Pmodel,\n"
"                       nsmax, dmax, options, categories, method_fn):\n"
"    \"\"\"Return array of predicted MAP classes for LOOCV.\"\"\"\n"
"    n    = cs_data.shape[0]\n"
"    preds = np.zeros(n, dtype=int)\n"
"    opts  = {**options, 'show_progress': 0}\n"
"    for i in range(n):\n"
"        ck_i   = cs_data[i:i+1, :]\n"
"        cs_loo = np.delete(cs_data, i, 0)\n"
"        ps_loo = np.delete(ps_data, i, 0)\n"
"        pk_i   = method_fn(ck_i, cs_loo, ps_loo,\n"
"                           dmodel, Pmodel, nsmax, dmax, options=opts)\n"
"        preds[i] = categories[np.argmax(pk_i[0])]\n"
"    return preds\n"
"\n"
"print('Computing per-class metrics for BMEcatPdf ...')\n"
"preds_bme = loocv_predictions(cs, ps, code, dmodel, Pmodel,\n"
"                               nsmax, dmax_bme, base_options, categories, BMEcatPdf)\n"
"\n"
"# Confusion matrix\n"
"conf = np.zeros((nc, nc), dtype=int)\n"
"for true_c, pred_c in zip(code, preds_bme):\n"
"    r_idx = np.where(categories == true_c)[0][0]\n"
"    c_idx = np.where(categories == pred_c)[0][0]\n"
"    conf[r_idx, c_idx] += 1\n"
"\n"
"print('\\nConfusion matrix (rows=true, cols=predicted):')\n"
"df_conf = pd.DataFrame(conf,\n"
"    index   = [f'True {c}' for c in categories],\n"
"    columns = [f'Pred {c}' for c in categories])\n"
"print(df_conf)\n"
"\n"
"# Per-class precision and recall\n"
"precision = np.diag(conf) / conf.sum(axis=0).clip(min=1)\n"
"recall    = np.diag(conf) / conf.sum(axis=1).clip(min=1)\n"
"f1_score  = 2 * precision * recall / (precision + recall).clip(min=1e-9)\n"
"\n"
"df_pr = pd.DataFrame({'Category': [f'Type {c}' for c in categories],\n"
"                       'Precision': precision,\n"
"                       'Recall'   : recall,\n"
"                       'F1-score' : f1_score})\n"
"print('\\nPer-class precision / recall (BMEcatPdf LOOCV):')\n"
"print(df_pr.to_string(index=False))\n"
))

# ---------------------------------------------------------------------------
# Cell 20 – Confusion matrix heatmap
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
"fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
"\n"
"# Confusion matrix\n"
"im = axes[0].imshow(conf, cmap='Blues', vmin=0)\n"
"for r in range(nc):\n"
"    for c_ in range(nc):\n"
"        axes[0].text(c_, r, str(conf[r, c_]),\n"
"                     ha='center', va='center', fontsize=12)\n"
"axes[0].set_xticks(range(nc));  axes[0].set_xticklabels([f'Pred {c}' for c in categories])\n"
"axes[0].set_yticks(range(nc));  axes[0].set_yticklabels([f'True {c}' for c in categories])\n"
"axes[0].set_title('Confusion Matrix – BMEcatPdf (LOOCV)', fontsize=10)\n"
"plt.colorbar(im, ax=axes[0])\n"
"\n"
"# Bar chart: precision / recall\n"
"x = np.arange(nc)\n"
"w = 0.3\n"
"axes[1].bar(x - w/2, precision, width=w, label='Precision', color='steelblue')\n"
"axes[1].bar(x + w/2, recall,    width=w, label='Recall',    color='darkorange')\n"
"axes[1].set_xticks(x)\n"
"axes[1].set_xticklabels([f'Type {c}' for c in categories])\n"
"axes[1].set_ylim(0, 1.05);  axes[1].set_ylabel('Score')\n"
"axes[1].set_title('Per-Class Precision & Recall – BMEcatPdf (LOOCV)', fontsize=10)\n"
"axes[1].legend()\n"
"\n"
"plt.tight_layout();  plt.show()\n"
))

# ---------------------------------------------------------------------------
# Cell 21 – Spatial error map (LOOCV)
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"### 14.1 Spatial distribution of LOOCV errors\n\n"
"Plotting the misclassified data locations reveals whether prediction errors\n"
"cluster spatially (e.g. near class boundaries or in data-sparse regions).\n"
))

cells.append(nbf.v4.new_code_cell(
"correct_mask   = preds_bme == code\n"
"incorrect_mask = ~correct_mask\n"
"\n"
"fig, ax = plt.subplots(figsize=(6, 5))\n"
"for cat, col in zip(categories, CMAP4):\n"
"    m = code == cat\n"
"    # Correctly classified\n"
"    ax.scatter(cs[m & correct_mask,   0]/1e3,\n"
"               cs[m & correct_mask,   1]/1e3,\n"
"               c=col, s=40, edgecolors='k', linewidths=0.4,\n"
"               label=f'Type {cat} (correct)', zorder=3)\n"
"    # Misclassified (marked with 'x')\n"
"    ax.scatter(cs[m & incorrect_mask, 0]/1e3,\n"
"               cs[m & incorrect_mask, 1]/1e3,\n"
"               marker='X', c=col, s=80, edgecolors='k', linewidths=0.7,\n"
"               label=f'Type {cat} (wrong)',   zorder=4)\n"
"\n"
"ax.set_xlabel('Easting (km)');  ax.set_ylabel('Northing (km)')\n"
"ax.set_title(f'LOOCV errors – BMEcatPdf  '\n"
"             f'(correct: {correct_mask.sum()}/{n}, '\n"
"             f'wrong: {incorrect_mask.sum()}/{n})')\n"
"ax.legend(fontsize=7, ncol=2)\n"
"plt.tight_layout();  plt.show()\n"
))

# ---------------------------------------------------------------------------
# Cell 22 – Sensitivity to nsmax
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 15. Sensitivity analysis – effect of `nsmax`\n\n"
"We use the fast **MCPcatPdf** estimator to quickly assess how the number of\n"
"neighbours influences LOOCV accuracy.\n"
))

cells.append(nbf.v4.new_code_cell(
"nsmax_grid  = [2, 4, 6, 8, 12, 16]\n"
"acc_vs_ns   = []\n"
"ll_vs_ns    = []\n"
"\n"
"for ns in nsmax_grid:\n"
"    a, ll = loocv_accuracy(cs, ps, code, dmodel, Pmodel,\n"
"                           ns, dmax_bme, base_options,\n"
"                           categories, MCPcatPdf)\n"
"    acc_vs_ns.append(a)\n"
"    ll_vs_ns.append(ll)\n"
"    print(f'  nsmax={ns:3d}  accuracy={a:.3f}  log-lik={ll:.3f}')\n"
"\n"
"fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
"axes[0].plot(nsmax_grid, acc_vs_ns, 'o-', color='steelblue')\n"
"axes[0].set_xlabel('nsmax'); axes[0].set_ylabel('LOOCV Accuracy')\n"
"axes[0].set_title('Accuracy vs. nsmax (MCPcatPdf)')\n"
"axes[0].grid(True, alpha=0.3)\n"
"\n"
"axes[1].plot(nsmax_grid, ll_vs_ns, 's-', color='darkorange')\n"
"axes[1].set_xlabel('nsmax'); axes[1].set_ylabel('Mean Log-Likelihood')\n"
"axes[1].set_title('Log-Likelihood vs. nsmax (MCPcatPdf)')\n"
"axes[1].grid(True, alpha=0.3)\n"
"\n"
"plt.tight_layout();  plt.show()\n"
))

# ---------------------------------------------------------------------------
# Cell 23 – Summary & next steps
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 16. Summary & guidelines\n\n"
"### What we did\n\n"
"| Step | Function | Purpose |\n"
"|------|----------|---------|\n"
"| 1 | `probatablecalc` | Estimate empirical bivariate probability tables $P(c_1,c_2 \\mid h)$ |\n"
"| 2 | `probatablefit`  | Smooth the tables with kernel regression → `Pmodel` |\n"
"| 3 | `BMEcatPdf`      | Rigorous ME posterior, handles clustered data via ME |\n"
"| 4 | `MCPcatPdf`      | Fast approximation, good when data spacing >> correlation range |\n"
"| 5 | `HBMEcatPdf`     | Hybrid: spatially varying prior + local BME update |\n"
"| 6 | LOOCV            | Assess accuracy and probabilistic calibration |\n\n"
"### Practical guidelines\n\n"
"- **Probability model:** choose `kstd ≈ 0.5 × d_max_emp / n_bins`.  If the\n"
"  smoothed curves oscillate, increase `kstd` or reduce `n_bins`.\n"
"- **Neighbourhood:** `dmax` should cover 1–2 correlation ranges.  Start with\n"
"  `nsmax = 6–10` for BMEcatPdf.\n"
"- **Estimator choice:**\n"
"  - Use **BMEcatPdf** when accuracy is paramount and runtime is acceptable.\n"
"  - Use **MCPcatPdf** for rapid exploration or large grids.\n"
"  - Use **HBMEcatPdf** when the prior matters (sparse data, strong trend).\n"
"- **Soft data:** one-hot vectors represent hard data.  If you have genuinely\n"
"  uncertain category observations (e.g. photo-interpretation), supply a\n"
"  non-trivial PMF vector in `ps`.\n\n"
"### Next steps\n\n"
"- **Prior with `BMEcatPrior`:** use XGBoost or XGBoostLSS to learn a\n"
"  spatially varying prior from ancillary covariates (DEM, satellite bands …).\n"
"- **Regularisation tuning:** use `tune_regularization_loocv` to select the\n"
"  optimal `reg` parameter for `MLE_reg` or `ME_dual` estimators.\n"
"- **Space-time extension:** add a time column to `cs` / `ck`, and extend\n"
"  `probatablecalc` to a spatio-temporal lag grid.\n"
))

# ---------------------------------------------------------------------------
# Assemble & write
# ---------------------------------------------------------------------------
nb.cells = cells

out_path = 'tutorials/09_bme_categorical.ipynb'
import os
os.makedirs('tutorials', exist_ok=True)

with open(out_path, 'w', encoding='utf-8') as fh:
    nbf.write(nb, fh)

print(f'Written → {out_path}')
