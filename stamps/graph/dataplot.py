# -*- coding: utf-8 -*-
"""
Created on Fri Jul 3 09:46:39 2015
@author: hdragon689

refine on Tue Oct 1 12:16:25 2019
@refiner: HuaTing
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib.dates import DateFormatter,rrulewrapper, RRuleLocator
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def histplot(x, bins=None, ax=None, density=False, kde=False,
             bw_method='scott', color='gray', alpha=0.8, label=None,
             xlabel='content', ylabel=None, show=True):
  """Plot a histogram with optional density normalisation and KDE overlay.

  Parameters
  ----------
  x : array-like
      1-D array of observations.
  bins : int or array-like or None
      Number of bins or bin edges.  ``None`` lets matplotlib choose.
  ax : matplotlib.axes.Axes or None
      Axes to draw on.  ``None`` creates a new figure.
  density : bool
      If ``True`` the histogram is normalised to a probability density
      (area = 1).  Default ``False`` (counts).
  kde : bool
      If ``True`` overlay a Gaussian kernel density estimate.
  bw_method : str or float
      Bandwidth method passed to ``scipy.stats.gaussian_kde``
      (e.g. ``'scott'``, ``'silverman'``, or a scalar).
  color : str
      Bar fill colour.
  alpha : float
      Bar transparency.
  label : str or None
      Histogram legend label.
  xlabel : str
      X-axis label (default ``'content'``).
  ylabel : str or None
      Y-axis label.  Auto-set to ``'Density'`` or ``'Count'``
      when ``None``.
  show : bool
      Call ``plt.show()`` when ``True``.

  Returns
  -------
  ax : matplotlib.axes.Axes
  """
  x = np.asarray(x).ravel()
  if ax is None:
      fig, ax = plt.subplots()

  if ylabel is None:
      ylabel = 'Density' if density else 'Count'

  if density and bins is not None:
      bins_arr = np.asarray(bins) if not np.isscalar(bins) else None
      if bins_arr is not None:
          bw = bins_arr[1] - bins_arr[0]
          counts, edges = np.histogram(x, bins=bins_arr)
          dens = counts / (len(x) * bw)
          ax.bar(edges[:-1], dens, width=bw, align='edge',
                 color=color, alpha=alpha, edgecolor='white', label=label)
      else:
          ax.hist(x, bins=bins, density=True, edgecolor='white', linewidth=1,
                  color=color, alpha=alpha, label=label)
  else:
      ax.hist(x, bins=bins, density=density, edgecolor='black', linewidth=1,
              color=color, alpha=alpha, label=label)

  if kde:
      from scipy.stats import gaussian_kde
      kde_obj = gaussian_kde(x, bw_method=bw_method)
      lo = x.min() - 0.1 * np.ptp(x)
      hi = x.max() + 0.1 * np.ptp(x)
      kde_x = np.linspace(lo, hi, 300)
      ax.plot(kde_x, kde_obj(kde_x), 'r-', lw=2, label='KDE')

  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.grid(True, alpha=0.3)
  if show:
      plt.show()
  return ax

def scattergram(c1,c2,s=8,show=True):
  '''
  Display one data pair against each other.

  Syntax:
      scattergram(c1,c2,s=2,show=True)

  Input:
  c1        n by 1    1D numpy array of first class observations
  c2        n by 1    1D numpy array of second class observations
  s         int       size of point
  '''
  plt.scatter(c1,c2,s=s)
  plt.xlabel('class1 content')
  plt.ylabel('class2 content')
  if show:
    plt.show()

def colorplot(c, z=None, ax=None, s=100, cmap='hot_r', zrange=None,
              colorbar=True, show=True,
              codes=None, categories=None, labels=None, colors=None,
              ch_grid=None, grid_kw=None,
              xlabel='x-axis', ylabel='y-axis', title=None,
              aspect='equal'):
  """Plot coloured symbols at 2-D coordinates.

  Supports two modes controlled by the *codes* parameter:

  * **Continuous** (default, ``codes=None``): markers are coloured by a
    continuous scalar *z* using *cmap*.
  * **Categorical** (``codes`` provided): markers are coloured by integer
    class codes using a discrete colour legend constructed from
    *categories*, *labels*, and *colors*.

  Parameters
  ----------
  c : (n, 2) array
      Spatial coordinates.
  z : (n,) array or None
      Continuous values (required when ``codes`` is ``None``).
  ax : Axes or None
      Axes to draw on.  ``None`` creates a new figure.
  s : float
      Marker size.
  cmap : str
      Continuous-mode colour map.
  zrange : list of two floats or None
      ``[vmin, vmax]`` for the continuous colour scale.
  colorbar : bool
      Show colour bar in continuous mode.
  show : bool
      Call ``plt.show()``.
  codes : (n,) int array or None
      Integer class codes for categorical mode.  When provided, *z* is
      ignored and *categories*/*labels*/*colors* are used.
  categories : list of int or None
      Ordered class codes (categorical mode).
  labels : list of str or None
      Human-readable class names aligned with *categories*.
  colors : list of str or None
      Colours aligned with *categories*.
  ch_grid : (m, 2) array or None
      Estimation grid coordinates to overlay as faint ``'+'`` markers.
  grid_kw : dict or None
      Extra keyword arguments for the grid overlay scatter.
  xlabel, ylabel : str
      Axis labels.
  title : str or None
      Axes title.
  aspect : str or None
      Axes aspect ratio (default ``'equal'``).  ``None`` to skip.

  Returns
  -------
  ax : matplotlib.axes.Axes
  """
  if ax is None:
      fig, ax = plt.subplots()

  if ch_grid is not None:
      _gkw = dict(marker='+', c='red', s=max(s * 0.3, 4), alpha=0.4, zorder=1)
      if grid_kw:
          _gkw.update(grid_kw)
      ax.scatter(ch_grid[:, 0], ch_grid[:, 1], **_gkw)

  if codes is not None:
      codes = np.asarray(codes).ravel()
      if categories is None:
          categories = sorted(np.unique(codes).tolist())
      if labels is None:
          labels = [str(c_) for c_ in categories]
      if colors is None:
          _cm = plt.colormaps['tab10']
          colors = [_cm(i % 10) for i in range(len(categories))]
      for cat, lbl, col in zip(categories, labels, colors):
          m = codes == cat
          ax.scatter(c[m, 0], c[m, 1], c=col, label=lbl,
                     edgecolors='k', lw=0.4, s=s, zorder=3)
      ax.legend(fontsize=8)
  else:
      if z is None:
          raise ValueError("Either z or codes must be provided")
      if zrange is None:
          zrange = [None, None]
      sc = ax.scatter(c[:, 0], c[:, 1], c=np.asarray(z).reshape(-1),
                      s=s, cmap=cmap, vmin=zrange[0], vmax=zrange[1],
                      linewidths=1, edgecolors='black', zorder=3)
      if colorbar:
          plt.colorbar(sc, ax=ax)

  xmax, xmin = np.max(c[:, 0]), np.min(c[:, 0])
  ymax, ymin = np.max(c[:, 1]), np.min(c[:, 1])
  xpadding = 0.05 * (xmax - xmin)
  ypadding = 0.05 * (ymax - ymin)
  ax.set_xlim(xmin - xpadding, xmax + xpadding)
  ax.set_ylim(ymin - ypadding, ymax + ypadding)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  if title:
      ax.set_title(title)
  if aspect:
      ax.set_aspect(aspect)
  if show:
      plt.show()
  return ax

def markerplot(c,z,ax=None,symsize=1,zrange=None,show=True): 
  '''
  Plot the values of a vector at a set of two dimensional coordinates
  using symbols of varying sizes such that the size of the displayed
  symbols at these coordinates is a function of the corresponding values. 
  
  Syntax:
      ax = markerplot(c,z,ax=None,simsize=1,zrange=None,show=True)
  
  Input:
  c         n by 2    2D numpy array of spatial coordinates.
  z         n by 1    2D numpy array of observations.
  ax        ax        Optional. Axes object for markerplot. Default is None 
                      that creates new plot.
  symsize   scalar    The size scheme. The size scale for marker display. 
                      Default is 1  
  zrange    list      Two scalar contains min and max of marker size range, 
                      [zmin, zmax].
  
  '''  
  
  if ax is None:
    ax=plt.figure().add_subplot(111)
  else:
    ax=ax
  z=z.reshape(z.size,1)  
  zmax,zmin = np.max(z),np.min(z)
  z=(z-zmin)/(zmax-zmin)*20**2
  if zrange is None:
    zrange = [None,None]

  plt.scatter(x=c[:,0],y=c[:,1],s=z*symsize,vmin=zrange[0], vmax=zrange[1],alpha = 0.5)

  xmax,xmin=np.max(c[:,0]),np.min(c[:,0])
  ymax,ymin=np.max(c[:,1]),np.min(c[:,1])
  xpadding , ypadding=0.05*(xmax-xmin),0.05*(ymax-ymin)
  plt.xlim(xmin-xpadding,xmax+xpadding)  
  plt.ylim(ymin-ypadding,ymax+ypadding)
  plt.xlabel('x-axis')
  plt.ylabel('y-axis')    
  if show:    
    plt.show()
  return ax
  
def tsplot(t,z,start,ax=None,fmt='b-',plotinterval=None,Dateshowfmt = None,show=True):
  ''' 
  Plot time series data.

  Syntax:
      tsplot(t,z,start,ax=None,fmt='b-',plotinterval=None,Dateshowfmt = None,show=True)

  Input:
  t             1 by n    1D or 2D array of time in datetime or 
                          numpy.datetime64 formats
  z             1 by n    1D or 2D array of observations
  start         string    the begin date of data.
                          Input format depend on your temporal resolution.
                          temporal resolution is monthly, Input format: year/month   ex:2019-01
                          temporal resolution is daily,   Input format: year/month/day ex:2019-01-02
  ax            axes      the axes object to be plotted
  fmt           string    line format for time series plot. 
                          Details can refer to plt.plot?    
  plotinterval  int       The interval of tick you want to show on your xlabel.Unit depend on your 
                          START input format.
                          ex: If your START input format is year/month,
                          then 'plotinterval=3' means plot interval is 3 months;
                          If your START input format is year/month/day
                          then 'plotinterval=3' means plot interval is 3 days;
                          if None, there will be totally ten ticks show on xlabel
  Dateshowfmt   str       The date format you want to show the tick.
                          ex: '%Y/%m/%d' or '%Y/%m' or '%Y'

  Remark: Details of Dateshowfmt can refer to 
          https://matplotlib.org/3.1.1/api/dates_api.html#matplotlib.dates.DateFormatter    
  '''
  tzdf = pd.DataFrame(np.nan, index=np.arange(np.max(t)-np.min(t)+1), columns=['z'])
  tzdf.loc[t - np.min(t),:] = z.reshape(-1,1)
  z = tzdf.values
  t = tzdf.index

  if ax is None:  
    fig, ax = plt.subplots()
  else:
    ax=ax
  if plotinterval is None:
    plotinterval = int(len(z)/10)

  dlen = len(start.split('-'))
  if dlen==3:
    freq = 'D'
    DateFmt = '%Y/%m/%d'
    from matplotlib.dates import DAILY
    rule = rrulewrapper(DAILY, interval=plotinterval)
  elif dlen==2:
    freq = 'M'
    DateFmt = '%Y/%m'
    from matplotlib.dates import MONTHLY
    rule = rrulewrapper(MONTHLY, interval=plotinterval)
  elif dlen==1:
    freq = 'Y'
    DateFmt = '%Y'
    from matplotlib.dates import YEARLY
    rule = rrulewrapper(YEARLY, interval=plotinterval)

  if Dateshowfmt is not None:
    DateFmt = Dateshowfmt
  
  loc = RRuleLocator(rule)

  formatter = DateFormatter(DateFmt)
  dates = pd.date_range(start=start, periods=len(z), freq=freq)
  plt.plot(dates, z)

  ax.xaxis.set_major_locator(loc)
  ax.xaxis.set_major_formatter(formatter)
  ax.xaxis.set_tick_params(rotation=30, labelsize=10)
  if show:
    plt.show()

  return ax

def histscatterplot(data,columns=None,font_scale = 0.9,half=False,show=True):

  '''
  Plot pairwise relationships in a dataset.

  Syntax:
      histscatterplot(data,columns=None,font_scale = 0.9)

  Input:
  data        m by n    a 2D array with m observations and n variables. 
                        This inputcan also be a pandas.DataFrame.
  columns     list      list of the titles of variables.
  font_scale  float     the size of the font elements.
  half        bool      set True to prevent ploting repeated scatter plot 
  
  '''  
  
  try:
      import seaborn as sns
  except ModuleNotFoundError as exc:
      raise ModuleNotFoundError(
          "histscatterplot requires seaborn.  Install it with:  pip install seaborn"
      ) from exc
  if type(data) is not pd.core.frame.DataFrame:
    data = pd.DataFrame(data,columns=columns)      
  sns.set(font_scale=font_scale)
  g = sns.pairplot(data)
  if half:
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
      g.axes[i, j].set_visible(False)
  if show:
    plt.show()


  
def plot_ecdf(ax, x, kde=True, bw_method=0.3, color='steelblue',
              label='Empirical CDF', kde_label='KDE CDF', kde_color='r',
              xlabel=None, ylabel='F(z)', title=None, xlim=None, show=False):
    """Plot the empirical CDF with an optional KDE-based CDF overlay.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x : array-like
        1-D array of observations.
    kde : bool
        Overlay a CDF derived from ``scipy.stats.gaussian_kde``.
    bw_method : str or float
        Bandwidth method for the KDE.
    color : str
        Step-function colour.
    label : str
        ECDF legend label.
    kde_label : str
        KDE-CDF legend label.
    kde_color : str
        KDE-CDF line colour.
    xlabel, ylabel : str or None
        Axis labels.
    title : str or None
        Axes title.
    xlim : tuple or None
        ``(xmin, xmax)`` for the x-axis.
    show : bool
        Call ``plt.show()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    x = np.asarray(x).ravel()
    xs = np.sort(x)
    ps = np.arange(1, len(xs) + 1) / len(xs)
    ax.step(xs, ps, color=color, lw=2, label=label, where='post')

    if kde:
        from scipy.stats import gaussian_kde
        lo = xs[0] - 0.05 * np.ptp(xs)
        hi = xs[-1] + 0.05 * np.ptp(xs)
        kde_x = np.linspace(lo, hi, 600)
        kde_p = np.cumsum(gaussian_kde(x, bw_method=bw_method)(kde_x))
        kde_p *= (kde_x[1] - kde_x[0])
        ax.plot(kde_x, np.clip(kde_p, 0, 1), '-', color=kde_color,
                lw=2, label=kde_label)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend()
    if show:
        plt.show()
    return ax


def plot_grid_map(ax, ck, zk, title='', n_e=None, n_n=None,
                  ch=None, zh=None,
                  cmap='hot_r', vmin=None, vmax=None,
                  colorbar_label='', xlabel='Easting (m)', ylabel='Northing (m)',
                  scatter_kw=None):
    """Plot a continuous gridded field as a colour mesh with an optional data overlay.

    Visualises a scalar field *zk* estimated on a regular 2-D grid *ck* using
    ``pcolormesh``.  Optionally overlays the original data locations *ch* as a
    scatter plot whose marker colours are matched to the same colour scale.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw.
    ck : np.ndarray, shape (n_e * n_n, 2)
        Grid node coordinates ``[easting, northing]`` in row-major (C) order:
        northing varies first (outer loop), easting varies second (inner loop).
        If *n_e* and *n_n* are ``None`` the grid dimensions are inferred from
        ``np.unique`` of each column.
    zk : np.ndarray, shape (n_e * n_n,)
        Scalar field values at each grid node.  ``NaN`` entries are rendered
        transparently.
    title : str, optional
        Axes title (default: empty string).
    n_e : int or None, optional
        Number of grid nodes in the easting direction.  If ``None``, inferred
        automatically from the unique easting values in *ck* (works only for
        fully regular grids).
    n_n : int or None, optional
        Number of grid nodes in the northing direction.  If ``None``, inferred
        automatically (see *n_e*).
    ch : np.ndarray, shape (m, 2) or None, optional
        Coordinates of the original data sites to overlay as scatter points.
        If ``None``, no scatter overlay is drawn.
    zh : np.ndarray, shape (m,) or None, optional
        Values at the data sites, used to colour the scatter markers so they
        use the same colour scale as the grid.  If ``None`` (but *ch* is given),
        black markers are drawn instead.
    cmap : str or Colormap, optional
        Matplotlib colormap name or object (default: ``'hot_r'``).
    vmin : float or None, optional
        Lower colour-scale limit.  Defaults to the 2nd percentile of *zk*.
    vmax : float or None, optional
        Upper colour-scale limit.  Defaults to the 98th percentile of *zk*.
    colorbar_label : str, optional
        Label for the colorbar (default: empty string).
    xlabel : str, optional
        X-axis label (default: ``'Easting (m)'``).
    ylabel : str, optional
        Y-axis label (default: ``'Northing (m)'``).
    scatter_kw : dict or None, optional
        Extra keyword arguments forwarded to ``ax.scatter`` for the data
        overlay (e.g. ``{'s': 60, 'lw': 0.8}``).  The keys ``c``, ``cmap``,
        ``vmin``, ``vmax``, ``zorder`` are set automatically and should not
        be overridden here.

    Returns
    -------
    im : matplotlib.collections.QuadMesh
        The ``pcolormesh`` artist, allowing the caller to attach a shared
        colorbar or adjust properties after the call.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from stamps.graph.dataplot import plot_grid_map
    >>> rng = np.random.default_rng(0)
    >>> n_e, n_n = 10, 12
    >>> E = np.linspace(0, 9, n_e)
    >>> N = np.linspace(0, 11, n_n)
    >>> EE, NN = np.meshgrid(E, N)
    >>> ck = np.column_stack([EE.ravel(), NN.ravel()])
    >>> zk = rng.standard_normal(n_e * n_n)
    >>> ch = rng.uniform(0, 9, (20, 2))
    >>> zh = rng.standard_normal(20)
    >>> fig, ax = plt.subplots()
    >>> im = plot_grid_map(ax, ck, zk, title='Demo', ch=ch, zh=zh,
    ...                    colorbar_label='Value')
    """
    zk = np.asarray(zk, dtype=float)

    east_u  = np.unique(ck[:, 0])
    north_u = np.unique(ck[:, 1])

    if n_e is None:
        n_e = len(east_u)
    if n_n is None:
        n_n = len(north_u)

    Z    = zk.reshape(n_n, n_e)
    vmin = vmin if vmin is not None else float(np.nanpercentile(zk, 2))
    vmax = vmax if vmax is not None else float(np.nanpercentile(zk, 98))

    im = ax.pcolormesh(east_u, north_u, Z, cmap=cmap,
                       vmin=vmin, vmax=vmax, shading='nearest')
    plt.colorbar(im, ax=ax, label=colorbar_label)

    if ch is not None:
        _skw = dict(edgecolors='k', lw=0.5, s=45, zorder=3)
        if scatter_kw:
            _skw.update(scatter_kw)
        _c = zh if zh is not None else 'k'
        _ckw = dict(cmap=cmap, vmin=vmin, vmax=vmax) if zh is not None else {}
        ax.scatter(ch[:, 0], ch[:, 1], c=_c, **_ckw, **_skw)

    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im


# =============================================================================
# Spatial PDF / quantile map helpers
# =============================================================================

def _render_map(ax, ck, values, title, cmap, vmin, vmax, colorbar_label,
                xlabel, ylabel, n_e, n_n, ch, zh, scatter_kw):
    """Internal: render *values* at *ck* as pcolormesh when coords form a
    complete regular grid, or as a scatter plot when they do not (e.g. a
    subset of a full grid with some nodes missing).
    """
    ck = np.asarray(ck)
    nk = len(values)
    n_e_inf = len(np.unique(ck[:, 0])) if n_e is None else n_e
    n_n_inf = len(np.unique(ck[:, 1])) if n_n is None else n_n

    if nk == n_e_inf * n_n_inf:
        # Complete regular grid → colour mesh
        return plot_grid_map(ax, ck, values, title=title, n_e=n_e, n_n=n_n,
                             ch=ch, zh=zh,
                             cmap=cmap, vmin=vmin, vmax=vmax,
                             colorbar_label=colorbar_label,
                             xlabel=xlabel, ylabel=ylabel,
                             scatter_kw=scatter_kw)

    # Sparse / irregular set → scatter plot
    vm = vmin if vmin is not None else float(np.nanpercentile(values, 2))
    vx = vmax if vmax is not None else float(np.nanpercentile(values, 98))
    sc = ax.scatter(ck[:, 0], ck[:, 1], c=values, cmap=cmap,
                    vmin=vm, vmax=vx, s=60, edgecolors='none')
    plt.colorbar(sc, ax=ax, label=colorbar_label)
    if ch is not None:
        _skw = dict(edgecolors='k', lw=0.5, s=45, zorder=3)
        if scatter_kw:
            _skw.update(scatter_kw)
        _c = zh if zh is not None else 'k'
        _ckw = dict(cmap=cmap, vmin=vm, vmax=vx) if zh is not None else {}
        ax.scatter(ch[:, 0], ch[:, 1], c=_c, **_ckw, **_skw)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return sc

def plot_exceedance_map(ax, ck, z_out_list, pdf_out_list, threshold,
                        title='', cmap='YlOrRd', vmin=0.0, vmax=1.0,
                        colorbar_label='P(Z > threshold)',
                        xlabel='Easting (m)', ylabel='Northing (m)',
                        n_e=None, n_n=None,
                        ch=None, zh=None, scatter_kw=None):
    """Plot a map of exceedance probability P(Z > *threshold*) at each grid node.

    The posterior PDF at each location is provided as parallel lists of z-grids
    and PDF values — exactly the format returned by
    ``BMEPosteriorPDF(ck, ..., z_grid=None)``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw.
    ck : ndarray of shape (nk, 2)
        Grid node coordinates (flat).
    z_out_list : list of nk ndarrays
        Per-point z-grids from :func:`BMEPosteriorPDF`.
    pdf_out_list : list of nk ndarrays
        Per-point PDF values from :func:`BMEPosteriorPDF`.
    threshold : float
        The z value above which to compute exceedance probability.
    title, cmap, vmin, vmax, colorbar_label, xlabel, ylabel :
        Passed to :func:`plot_grid_map`.
    n_e, n_n : int or None
        Grid dimensions; inferred if ``None``.
    ch, zh, scatter_kw :
        Optional data overlay passed to :func:`plot_grid_map`.

    Returns
    -------
    im : QuadMesh
        The pcolormesh artist from :func:`plot_grid_map`.
    """
    from scipy.integrate import cumulative_trapezoid

    nk = len(z_out_list)
    prob_exceed = np.full(nk, np.nan)

    for k in range(nk):
        zg = np.asarray(z_out_list[k], dtype=float)
        pg = np.asarray(pdf_out_list[k], dtype=float)
        area = np.trapezoid(pg, zg)
        if area <= 0 or np.isnan(area):
            continue
        pg_norm = pg / area
        cdf = cumulative_trapezoid(pg_norm, zg, initial=0.0)
        cdf = np.clip(cdf, 0.0, 1.0)
        cdf_at_thresh = float(np.interp(threshold, zg, cdf))
        prob_exceed[k] = 1.0 - cdf_at_thresh

    return _render_map(ax, ck, prob_exceed, title=title,
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       colorbar_label=colorbar_label,
                       xlabel=xlabel, ylabel=ylabel,
                       n_e=n_e, n_n=n_n, ch=ch, zh=zh, scatter_kw=scatter_kw)


def plot_iqr_map(ax, ck, zq_lo, zq_hi,
                 title='', cmap='BuPu', vmin=None, vmax=None,
                 colorbar_label='Inter-quantile range',
                 xlabel='Easting (m)', ylabel='Northing (m)',
                 n_e=None, n_n=None,
                 ch=None, zh=None, scatter_kw=None):
    """Plot a map of the inter-quantile range (IQR) at each grid node.

    Simply computes ``zq_hi − zq_lo`` and renders the result via
    :func:`plot_grid_map`.  A narrow IQR means low posterior uncertainty;
    a wide IQR means high uncertainty.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw.
    ck : ndarray of shape (nk, 2)
        Grid node coordinates (flat).
    zq_lo : ndarray of shape (nk,)
        Lower quantile values (e.g. Q10).
    zq_hi : ndarray of shape (nk,)
        Upper quantile values (e.g. Q90).
    title, cmap, vmin, vmax, colorbar_label, xlabel, ylabel :
        Passed to :func:`plot_grid_map`.
    n_e, n_n, ch, zh, scatter_kw :
        Passed to :func:`plot_grid_map`.

    Returns
    -------
    im : QuadMesh
        The pcolormesh artist.
    """
    iqr = np.asarray(zq_hi, dtype=float) - np.asarray(zq_lo, dtype=float)
    return _render_map(ax, ck, iqr, title=title,
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       colorbar_label=colorbar_label,
                       xlabel=xlabel, ylabel=ylabel,
                       n_e=n_e, n_n=n_n, ch=ch, zh=zh, scatter_kw=scatter_kw)


def plot_pdf_ribbon_transect(ax, transect_x, z_out_list, pdf_out_list,
                             quantile_levels=None,
                             ribbon_color='steelblue', ribbon_alpha=0.25,
                             median_color='steelblue', median_lw=2.0,
                             true_values=None, true_color='tomato',
                             xlabel='Transect position', ylabel='Z'):
    """Plot a ribbon transect showing quantile bands of the posterior PDF.

    For each point along a 1-D transect, the posterior PDF is converted to a
    CDF and inverted at the requested quantile levels.  The outermost pair
    of quantiles is drawn as a filled ribbon, inner pairs as successively
    darker ribbons, and the median as a solid line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw.
    transect_x : array_like of shape (n,)
        Positions along the transect (e.g. easting values).
    z_out_list : list of n ndarrays
        Per-point z-grids from :func:`BMEPosteriorPDF`.
    pdf_out_list : list of n ndarrays
        Per-point PDF values from :func:`BMEPosteriorPDF`.
    quantile_levels : list of float or None
        Quantile levels to compute.  Must be symmetric around the median
        (e.g. ``[0.1, 0.3, 0.5, 0.7, 0.9]``).  Default
        ``[0.1, 0.25, 0.5, 0.75, 0.9]``.
    ribbon_color : str
        Base colour for the filled ribbons.
    ribbon_alpha : float
        Alpha of the outermost ribbon; inner ribbons are progressively more
        opaque.
    median_color : str
        Colour of the median line.
    median_lw : float
        Line width of the median line.
    true_values : array_like of shape (n,) or None
        Known true values along the transect, drawn as a dashed line.
    true_color : str
        Colour for the true-value line.
    xlabel, ylabel : str
        Axis labels.

    Returns
    -------
    None
    """
    from scipy.integrate import cumulative_trapezoid

    if quantile_levels is None:
        quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_levels = sorted(quantile_levels)
    transect_x = np.asarray(transect_x, dtype=float)
    n = len(transect_x)
    nq = len(quantile_levels)

    qtl_values = np.full((n, nq), np.nan)
    for k in range(n):
        zg = np.asarray(z_out_list[k], dtype=float)
        pg = np.asarray(pdf_out_list[k], dtype=float)
        area = np.trapezoid(pg, zg)
        if area <= 0 or np.isnan(area):
            continue
        pg_norm = pg / area
        cdf = cumulative_trapezoid(pg_norm, zg, initial=0.0)
        cdf = np.clip(cdf, 0.0, 1.0)
        for j, q in enumerate(quantile_levels):
            qtl_values[k, j] = float(np.interp(float(q), cdf, zg))

    # Draw ribbons from outermost pair inward
    n_pairs = nq // 2
    for p in range(n_pairs):
        lo_idx = p
        hi_idx = nq - 1 - p
        lo_q = quantile_levels[lo_idx]
        hi_q = quantile_levels[hi_idx]
        alpha = ribbon_alpha + p * (0.5 - ribbon_alpha) / max(n_pairs - 1, 1)
        label = f'Q{int(lo_q*100)}–Q{int(hi_q*100)}'
        ax.fill_between(transect_x,
                        qtl_values[:, lo_idx], qtl_values[:, hi_idx],
                        color=ribbon_color, alpha=alpha, label=label)

    # Draw median
    if 0.5 in quantile_levels:
        med_idx = quantile_levels.index(0.5)
        ax.plot(transect_x, qtl_values[:, med_idx],
                color=median_color, lw=median_lw, label='Median (Q50)')

    if true_values is not None:
        ax.plot(transect_x, np.asarray(true_values),
                color=true_color, lw=2, ls='-.', label='True value')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


if __name__ == "__main__":
  c=np.random.rand(30,2)
  z=np.random.rand(30,1)
  z1 = np.random.rand(30,1)
  z2 = np.random.randn(300)
  zr=[0,0.5]
  zt = 7 + 8 * z2
  df = pd.DataFrame(np.random.randn(1000, 4), columns=['A','B','C','D'])
  
  histplot(z2,bins = 30)
  scattergram(z,z1,s = 10)
  colorplot(c,z,ax=None,cmap='jet_r',zrange=zr)
  markerplot(c,z,ax=None,symsize=1,zrange=zr)
  tsplot(np.arange(len(zt)),zt,start='2019-01-01',ax=None,fmt='b-',
    plotinterval=30,Dateshowfmt='%Y/%m')
  histscatterplot(df,half=True)
  
