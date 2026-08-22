# -*- coding: utf-8 -*-
"""Generate 01_data_loading_and_exploration.ipynb"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb.metadata['language_info'] = {'name': 'python', 'version': '3.9.0'}
cells = []

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"# Tutorial 01 - Data Loading, Exploration & Basic Spatial Estimation\n\n"
"**Package:** stamps_v3  \n"
"**MATLAB equivalent:** `IOLIBtutorial.m` + `GENLIBtutorial.m` + `STATLIBtutorial.m` §1-3  \n"
"**Dataset:** Falmagne soil dataset - 118 georeferenced soil samples from the Dinant area, Belgium\n\n"
"---\n\n"
"## Learning objectives\n\n"
"By the end of this notebook you will be able to:\n\n"
"1. Load a **GeoEAS-format** text file using `numpy` / `pandas`.\n"
"2. Explore and visualise a spatial dataset (locations, histograms, scatter plots).\n"
"3. Build a **regular estimation grid** with `numpy.meshgrid`.\n"
"4. Apply two simple **non-parametric spatial estimators**:\n"
"   - Inverse-distance weighting (IDW)\n"
"   - Gaussian kernel smoothing\n"
"5. Display **spatial estimation maps** with `matplotlib`.\n\n"
"These foundational skills are the building blocks for all subsequent tutorials.\n"
))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
"import sys, os\n"
"import numpy as np\n"
"import pandas as pd\n"
"import matplotlib.pyplot as plt\n"
"from scipy import stats\n"
"from scipy.spatial import KDTree\n\n"
"# make the stamps package importable\n"
"REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))\n"
"if REPO_ROOT not in sys.path:\n"
"    sys.path.insert(0, REPO_ROOT)\n\n"
"plt.rcParams.update({'figure.dpi': 110, 'font.size': 10})\n"
"DATA_DIR = os.path.join(os.getcwd(), 'data')\n"
"print('Data directory:', DATA_DIR)\n"
))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 1. Load the Falmagne dataset\n\n"
"`Falmagne.txt` follows the **GeoEAS** format:\n\n"
"```\n"
"Line 1      : file title\n"
"Line 2      : number of columns p\n"
"Lines 3...2+p : column names\n"
"Remaining   : data (whitespace-delimited)\n"
"```\n\n"
"We parse it with a compact helper that replicates MATLAB's `readGeoEAS`.\n"
))

cells.append(nbf.v4.new_code_cell(
"def read_geoeas(filepath):\n"
"    '''Read a GeoEAS-format text file.\n\n"
"    Returns\n"
"    -------\n"
"    df    : pd.DataFrame  - data with column names as headers\n"
"    title : str           - file title (first line)\n"
"    '''\n"
"    with open(filepath) as f:\n"
"        lines = f.readlines()\n"
"    title    = lines[0].strip()\n"
"    ncols    = int(lines[1].strip())\n"
"    colnames = [lines[2 + i].strip() for i in range(ncols)]\n"
"    data     = np.loadtxt(filepath, skiprows=2 + ncols)\n"
"    return pd.DataFrame(data, columns=colnames), title\n\n"
"df, title = read_geoeas(os.path.join(DATA_DIR, 'Falmagne.txt'))\n"
"print('Title :', title)\n"
"print(f'Shape : {df.shape[0]} samples x {df.shape[1]} variables\\n')\n"
"print(df.head())\n"
))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
"# Extract columns\n"
"ch   = df.iloc[:, 0:2].values\n"
"sand = df.iloc[:, 2].values\n"
"silt = df.iloc[:, 3].values\n"
"clay = df.iloc[:, 4].values\n"
"code = df.iloc[:, 5].values.astype(int)\n\n"
"print(f'Samples : {len(sand)}')\n"
"print(f'Sand    : mean={sand.mean():.1f}%  range=[{sand.min():.1f}, {sand.max():.1f}]')\n"
"print(f'Silt    : mean={silt.mean():.1f}%  range=[{silt.min():.1f}, {silt.max():.1f}]')\n"
"print(f'Clay    : mean={clay.mean():.1f}%  range=[{clay.min():.1f}, {clay.max():.1f}]')\n"
"print(f'Codes   : {np.unique(code)}')\n"
))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("## 2. Visualise sampling locations"))

cells.append(nbf.v4.new_code_cell(
"SOIL_COLORS = {1: 'tomato', 2: 'steelblue', 3: 'goldenrod', 4: 'mediumseagreen'}\n"
"SOIL_LABELS = {1: 'Type 1 (loam)', 2: 'Type 2 (clay-loam)',\n"
"               3: 'Type 3 (silt-loam)', 4: 'Type 4 (silt)'}\n\n"
"fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
"ax = axes[0]\n"
"for k in np.unique(code):\n"
"    mask = code == k\n"
"    ax.scatter(ch[mask, 0], ch[mask, 1], c=SOIL_COLORS[k],\n"
"               label=SOIL_LABELS[k], edgecolors='k', lw=0.4, s=55, zorder=3)\n"
"ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')\n"
"ax.set_title('Sampling locations - soil type'); ax.set_aspect('equal')\n"
"ax.legend(loc='upper right', fontsize=8)\n\n"
"ax = axes[1]\n"
"sc = ax.scatter(ch[:, 0], ch[:, 1], c=sand, cmap='hot_r',\n"
"                edgecolors='k', lw=0.4, s=60, zorder=3)\n"
"plt.colorbar(sc, ax=ax, label='Sand (%)')\n"
"ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')\n"
"ax.set_title('Sand content (%)'); ax.set_aspect('equal')\n\n"
"plt.tight_layout(); plt.show()\n"
))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("## 3. Descriptive statistics"))

cells.append(nbf.v4.new_code_cell(
"variables = {'Sand': sand, 'Silt': silt, 'Clay': clay}\n\n"
"rows = []\n"
"for name, v in variables.items():\n"
"    rows.append({\n"
"        'Variable': name,\n"
"        'Mean':     np.mean(v),\n"
"        'Variance': np.var(v, ddof=1),\n"
"        'Std dev':  np.std(v, ddof=1),\n"
"        'Skewness': stats.skew(v),\n"
"        'Kurtosis': stats.kurtosis(v),\n"
"        'Min':      np.min(v),\n"
"        'Max':      np.max(v),\n"
"    })\n\n"
"print(pd.DataFrame(rows).set_index('Variable').round(2).to_string())\n"
))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 4. Histograms and kernel density estimates\n\n"
"A **scaled histogram** (density, not counts) allows visual comparison across variables.  "
"The Gaussian KDE provides a smooth estimate of the underlying probability density.\n"
))

cells.append(nbf.v4.new_code_cell(
"fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n"
"bins   = np.linspace(0, 100, 21)\n"
"bw     = bins[1] - bins[0]\n"
"colors = ['sandybrown', 'slategray', 'peru']\n\n"
"for ax, (name, v), col in zip(axes, variables.items(), colors):\n"
"    counts, edges = np.histogram(v, bins=bins)\n"
"    density = counts / (len(v) * bw)\n"
"    ax.bar(edges[:-1], density, width=bw, align='edge',\n"
"           color=col, alpha=0.6, edgecolor='white', label='Histogram')\n"
"    kde_x = np.linspace(max(-5, v.min() - 10), min(105, v.max() + 10), 300)\n"
"    kde   = stats.gaussian_kde(v, bw_method=0.3)\n"
"    ax.plot(kde_x, kde(kde_x), 'r-', lw=2, label='KDE')\n"
"    ax.set_xlabel(f'{name} (%)'); ax.set_ylabel('Density')\n"
"    ax.set_title(f'{name} - histogram & KDE'); ax.legend()\n\n"
"plt.tight_layout(); plt.show()\n"
))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 5. Empirical CDF\n\n"
r"The empirical CDF $\hat{F}(z) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}[z_i \le z]$ "
"is a non-parametric estimate of the true CDF.  "
"We overlay the CDF obtained by numerically integrating the KDE.\n"
))

cells.append(nbf.v4.new_code_cell(
"def ecdf(x):\n"
"    xs = np.sort(x)\n"
"    ps = np.arange(1, len(xs) + 1) / len(xs)\n"
"    return xs, ps\n\n"
"fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n\n"
"for ax, (name, v), col in zip(axes, variables.items(), colors):\n"
"    xs, ps = ecdf(v)\n"
"    ax.step(xs, ps, color='steelblue', lw=2, label='Empirical CDF', where='post')\n"
"    kde_x = np.linspace(-5, 105, 600)\n"
"    kde_p = np.cumsum(stats.gaussian_kde(v, bw_method=0.3)(kde_x)) * (kde_x[1] - kde_x[0])\n"
"    ax.plot(kde_x, np.clip(kde_p, 0, 1), 'r-', lw=2, label='KDE CDF')\n"
"    ax.set_xlabel(f'{name} (%)'); ax.set_ylabel('F(z)')\n"
"    ax.set_title(f'{name} - CDF'); ax.set_xlim([0, 100]); ax.legend()\n\n"
"plt.tight_layout(); plt.show()\n"
))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 6. Build a regular estimation grid\n\n"
"We create a **17 x 19 = 323-node** regular grid with 1 000 m spacing -\n"
"matching the MATLAB tutorial.\n\n"
"```matlab\n"
"% MATLAB equivalent:\n"
"minc=[178000 90000]; dc=[1000 1000]; nc=[17 19];\n"
"[ck]=creategrid(minc, dc, nc);\n"
"```\n\n"
"```python\n"
"# Python equivalent:\n"
"EE, NN = np.meshgrid(east_vals, north_vals)\n"
"ck = np.column_stack([EE.ravel(), NN.ravel()])\n"
"```\n"
))

cells.append(nbf.v4.new_code_cell(
"min_e, min_n = 178_000, 90_000\n"
"d_e,   d_n   = 1_000,   1_000\n"
"n_e,   n_n   = 17,      19\n\n"
"east_vals  = min_e + np.arange(n_e) * d_e\n"
"north_vals = min_n + np.arange(n_n) * d_n\n"
"EE, NN     = np.meshgrid(east_vals, north_vals)\n"
"ck         = np.column_stack([EE.ravel(), NN.ravel()])\n\n"
"print(f'Grid : {n_e} x {n_n} = {len(ck)} nodes')\n"
"print(f'Easting  : {east_vals[0]:.0f} to {east_vals[-1]:.0f} m')\n"
"print(f'Northing : {north_vals[0]:.0f} to {north_vals[-1]:.0f} m')\n\n"
"fig, ax = plt.subplots(figsize=(7, 7))\n"
"ax.plot(ch[:, 0], ch[:, 1], 'ko', ms=5, zorder=4, label='Data (118)')\n"
"ax.plot(ck[:, 0], ck[:, 1], '+r', ms=4, zorder=3, label=f'Grid ({len(ck)})')\n"
"ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')\n"
"ax.set_title('Data locations and estimation grid'); ax.set_aspect('equal'); ax.legend()\n"
"plt.tight_layout(); plt.show()\n"
))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 7. Inverse-distance weighting (IDW)\n\n"
r"$$\hat{z}(\mathbf{s}_0) = \frac{\displaystyle\sum_{i=1}^n w_i\,z_i}"
r"{\displaystyle\sum_{i=1}^n w_i}, \qquad w_i = \frac{1}{d_i^p}$$"
"\n\n"
"- $p = \\infty$ -> **nearest-neighbour** (step-function map)  \n"
"- $p = 2$ -> classic **IDW** (smooth transitions)\n\n"
"Corresponds to MATLAB's `invdist(ck, ch, z, p, nhmax, dmax)`.\n"
))

cells.append(nbf.v4.new_code_cell(
"def idw(ck, ch, z, power=2, nhmax=20, dmax=50_000):\n"
"    '''Inverse-distance weighting interpolation.'''\n"
"    tree = KDTree(ch)\n"
"    dists, idx = tree.query(ck, k=min(nhmax, len(z)),\n"
"                            distance_upper_bound=dmax + 1e-9)\n"
"    zk = np.full(len(ck), np.nan)\n"
"    for i, (d_row, id_row) in enumerate(zip(dists, idx)):\n"
"        valid = (d_row < dmax) & np.isfinite(d_row)\n"
"        d_v, id_v = d_row[valid], id_row[valid]\n"
"        if not len(d_v):\n"
"            continue\n"
"        if np.isinf(power):\n"
"            zk[i] = z[id_v[0]]\n"
"        elif d_v[0] == 0.0:\n"
"            zk[i] = z[id_v[0]]\n"
"        else:\n"
"            w = 1.0 / d_v**power\n"
"            zk[i] = np.dot(w, z[id_v]) / w.sum()\n"
"    return zk\n\n"
"zk_nn  = idw(ck, ch, sand, power=np.inf, nhmax=20, dmax=50_000)\n"
"zk_idw = idw(ck, ch, sand, power=2,      nhmax=20, dmax=50_000)\n"
"print(f'Nearest-neighbour: {np.sum(~np.isnan(zk_nn))} / {len(ck)} nodes')\n"
"print(f'IDW (p=2)        : {np.sum(~np.isnan(zk_idw))} / {len(ck)} nodes')\n"
))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 8. Gaussian kernel smoothing\n\n"
r"$$\hat{z}(\mathbf{s}_0) = \frac{\displaystyle\sum_i K_h(\|\mathbf{s}_0-\mathbf{s}_i\|)\,z_i}"
r"{\displaystyle\sum_i K_h(\|\mathbf{s}_0-\mathbf{s}_i\|)}, \qquad K_h(d) = e^{-d^2/(2h^2)}$$"
"\n\nCorresponds to MATLAB's `kernelsmoothing(ck, ch, z, h^2, nhmax, dmax)`.\n"
))

cells.append(nbf.v4.new_code_cell(
"def kernel_smoothing(ck, ch, z, bandwidth2=1e6, nhmax=20, dmax=50_000):\n"
"    '''Gaussian kernel smoothing.'''\n"
"    tree = KDTree(ch)\n"
"    dists, idx = tree.query(ck, k=min(nhmax, len(z)),\n"
"                            distance_upper_bound=dmax + 1e-9)\n"
"    zk = np.full(len(ck), np.nan)\n"
"    for i, (d_row, id_row) in enumerate(zip(dists, idx)):\n"
"        valid = (d_row < dmax) & np.isfinite(d_row)\n"
"        d_v, id_v = d_row[valid], id_row[valid]\n"
"        if not len(d_v):\n"
"            continue\n"
"        w = np.exp(-d_v**2 / (2.0 * bandwidth2))\n"
"        zk[i] = np.dot(w, z[id_v]) / w.sum()\n"
"    return zk\n\n"
"zk_ks = kernel_smoothing(ck, ch, sand, bandwidth2=1e6, nhmax=20, dmax=50_000)\n"
"print(f'Kernel smoothing: {np.sum(~np.isnan(zk_ks))} / {len(ck)} nodes')\n"
))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("## 9. Visualise the estimation maps"))

cells.append(nbf.v4.new_code_cell(
"def plot_grid_map(ax, ck, zk, title, n_e, n_n, cmap='hot_r', vmin=None, vmax=None):\n"
"    east_u  = np.unique(ck[:, 0])\n"
"    north_u = np.unique(ck[:, 1])\n"
"    Z    = zk.reshape(n_n, n_e)\n"
"    vmin = vmin if vmin is not None else np.nanpercentile(zk, 2)\n"
"    vmax = vmax if vmax is not None else np.nanpercentile(zk, 98)\n"
"    im   = ax.pcolormesh(east_u, north_u, Z, cmap=cmap,\n"
"                         vmin=vmin, vmax=vmax, shading='nearest')\n"
"    plt.colorbar(im, ax=ax, label='Sand (%)')\n"
"    ax.scatter(ch[:, 0], ch[:, 1], c=sand, cmap=cmap, vmin=vmin, vmax=vmax,\n"
"               edgecolors='k', lw=0.5, s=45, zorder=3)\n"
"    ax.set_title(title); ax.set_aspect('equal')\n"
"    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')\n\n"
"fig, axes = plt.subplots(1, 3, figsize=(18, 6))\n"
"vlo, vhi = 0, sand.max() * 1.05\n"
"plot_grid_map(axes[0], ck, zk_nn,  'Nearest-neighbour', n_e, n_n, vmin=vlo, vmax=vhi)\n"
"plot_grid_map(axes[1], ck, zk_idw, 'IDW  (p = 2)',      n_e, n_n, vmin=vlo, vmax=vhi)\n"
"plot_grid_map(axes[2], ck, zk_ks,  'Kernel smoothing',  n_e, n_n, vmin=vlo, vmax=vhi)\n"
"plt.suptitle('Sand content estimates - non-parametric spatial methods', fontsize=13, y=1.01)\n"
"plt.tight_layout(); plt.show()\n"
))

# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
"## 10. Summary & next steps\n\n"
"| Method | Requires covariance model? | Provides error variance? |\n"
"|---|:---:|:---:|\n"
"| Nearest-neighbour | No | No |\n"
"| IDW | No | No |\n"
"| Kernel smoothing | No | No |\n"
"| **Kriging** (Tutorial 04) | **Yes** | **Yes** |\n"
"| **BME** (Tutorials 05-07) | **Yes** | **Yes (full posterior PDF)** |\n\n"
"**Key limitation of non-parametric methods:** they ignore spatial autocorrelation and "
"cannot quantify estimation uncertainty. The subsequent tutorials address this by first "
"characterising the covariance structure (Tutorials 02-03) and then using it for "
"inference (Tutorials 04-07).\n"
))

cells.append(nbf.v4.new_code_cell(
"import pickle\n\n"
"tutorial_data = {\n"
"    'ch': ch, 'sand': sand, 'silt': silt, 'clay': clay, 'code': code,\n"
"    'ck': ck, 'EE': EE, 'NN': NN,\n"
"    'east_vals': east_vals, 'north_vals': north_vals,\n"
"    'n_e': n_e, 'n_n': n_n,\n"
"}\n\n"
"save_path = os.path.join(DATA_DIR, 'tutorial01_data.pkl')\n"
"with open(save_path, 'wb') as f:\n"
"    pickle.dump(tutorial_data, f)\n\n"
"print('Saved:', save_path)\n"
"print('Keys :', list(tutorial_data.keys()))\n"
))

nb.cells = cells
import nbformat
with open("tutorials/01_data_loading_and_exploration.ipynb", "w") as f:
    nbformat.write(nb, f)
print("Written: tutorials/01_data_loading_and_exploration.ipynb")
