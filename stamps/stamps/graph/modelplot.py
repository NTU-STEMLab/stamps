# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 22:16:03 2015

@author: hdragon689
"""
from six.moves import range
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..stats.dependence.stcovfit import covmodelest,anisocovmodelest
from ..general.coord2K import coord2dist
from ..stats.dependence.stcovfit import cal_cov_mod


def modelplot(C, rLag, tLag=None, covmodel=None, covparam=None,
              show_variogram=False, components=False,
              lag_scale=1.0, lag_unit=None, ax=None):
    """Plot empirical covariance and, optionally, an overlay covariance model.

    Creates a publication-quality figure comparing the empirical covariance
    (computed e.g. by :func:`~stamps.stamps.stats.dependence.stcov.stcov`)
    against a parametric covariance model.

    * **Pure spatial** (``tLag=None``) — a single 2-D line plot of
      ``C`` vs ``rLag``.  When *show_variogram* is ``True`` a second panel
      shows the equivalent variogram ``gamma(h) = C(0) - C(h)``.
    * **Space-time** (``tLag`` provided) — a three-panel layout: spatial
      profile (top-left), temporal profile (bottom-left), and a 3-D scatter /
      wireframe of the full space-time covariance surface (right).

    Parameters
    ----------
    C : np.ndarray, shape (ns, nt) or (ns,)
        Empirical covariance values.
    rLag : np.ndarray, shape (ns,)
        Spatial lag values.
    tLag : np.ndarray, shape (nt,) or None
        Temporal lag values.  ``None`` for pure-spatial.
    covmodel : list of str or None
        Nested covariance model name list.
    covparam : list of list or None
        Nested covariance parameter list.
    show_variogram : bool
        If ``True`` (pure-spatial only), add a second panel with the
        equivalent variogram ``gamma(h) = C(0) - C(h)``.
    components : bool
        If ``True`` and a nested model (len(covmodel) > 1) is provided,
        each model structure is plotted individually as a dashed line
        alongside the solid total.
    lag_scale : float
        Divide lag values by this factor for display (e.g. ``1000`` for km).
    lag_unit : str or None
        Lag axis unit label (e.g. ``'km'``).  ``None`` uses ``'Distance'``.
    ax : matplotlib.axes.Axes or None
        Pre-existing axes (pure-spatial single-panel only).  ``None``
        creates a new figure.

    Returns
    -------
    list
        ``[ax, line]`` (spatial) or ``[ax1, ax2, ax3]`` (space-time).
    """
    xlabel = f'Lag ({lag_unit})' if lag_unit else 'Distance'

    if tLag is not None:
        tLagM, rLagM = np.meshgrid(tLag, rLag)
        fig = plt.figure(figsize=(9, 4))
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (1, 0))
        ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2, projection='3d')
        ax1.plot(rLag / lag_scale, C[:, 0], 'bo', label='Empirical covariance')
        ax1.set(xlabel=xlabel, ylabel='Covariance', title='Spatial Covariance')
        ax1.grid()
        ax2.plot(tLag, C[0, :], 'bo', label='Empirical covariance')
        ax2.set(xlabel='Temporal lag', ylabel='Covariance', title='Temporal Covariance')
        ax2.grid()
        ax3.scatter(rLagM / lag_scale, tLagM, C, color='b')
        nonaC = C[~np.isnan(C)]
        Cint = (nonaC.max() - nonaC.min()) * 0.02
        ax3.set(xlabel='r', ylabel=r'$\tau$', zlabel='C', title='S/T Covariance',
                xlim=[rLag[0] / lag_scale, rLag[-1] / lag_scale],
                ylim=[tLag[0], tLag[-1]],
                zlim=[nonaC.min() - Cint, nonaC.max() + Cint])
        if covmodel is not None and covparam is not None:
            rLagI = np.linspace(rLag[0], rLag[-1], 50)
            tLagI = np.linspace(tLag[0], tLag[-1], 50)
            tLagMI, rLagMI = np.meshgrid(tLagI, rLagI)
            modelcov, _ = covmodelest(rLagMI, tLagMI, covmodel, covparam)
            ax1.plot(rLagI / lag_scale, modelcov[:, 0], 'r-', label='Covariance model')
            ax2.plot(tLagI, modelcov[0, :], 'r-', label='Covariance model')
            ax3.plot_wireframe(rLagMI / lag_scale, tLagMI, modelcov,
                               color='r', rstride=3, cstride=3)
            ax3.view_init(30, -15)
        plt.tight_layout()
        return [ax1, ax2, ax3]

    # --- Pure spatial ---
    n_panels = 2 if show_variogram else 1
    if ax is not None and n_panels == 1:
        axes_arr = [ax]
    else:
        fig, axes_arr = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4))
        if n_panels == 1:
            axes_arr = [axes_arr]

    ax_c = axes_arr[0]
    C_flat = np.asarray(C).flat[:]
    rLag_s = np.asarray(rLag).ravel() / lag_scale

    has_model = covmodel is not None and covparam is not None
    if has_model:
        rLagI = np.linspace(rLag.ravel()[0], rLag.ravel()[-1], 50)
        tLagI = np.array([0])
        tLagMI, rLagMI = np.meshgrid(tLagI, rLagI)
        modelcov, _ = covmodelest(rLagMI, tLagMI, covmodel, covparam)
        modelcov_flat = modelcov.flat[:]

        line1, = ax_c.plot(rLag_s, C_flat, 'o', color='steelblue', ms=5,
                           label='Empirical')
        line2, = ax_c.plot(rLagI / lag_scale, modelcov_flat, '-', color='tomato',
                           lw=2, label='Fitted model')
        line = [line1, line2]

        if components and len(covmodel) > 1:
            _comp_colors = ['gray', 'steelblue', 'darkorange', 'green',
                            'purple', 'brown']
            for ci, (cm_name, cm_par) in enumerate(zip(covmodel, covparam)):
                mc_i, _ = covmodelest(rLagMI, tLagMI, [cm_name], [cm_par])
                col = _comp_colors[ci % len(_comp_colors)]
                ax_c.plot(rLagI / lag_scale, mc_i.flat[:], '--', color=col,
                          lw=1.5, label=cm_name)
        ax_c.legend(loc='best', fontsize=9)
    else:
        line, = ax_c.plot(rLag_s, C_flat, 'o-', color='steelblue',
                          label='Empirical covariance')
        ax_c.legend(loc='best')

    ax_c.axhline(0, color='gray', ls='--', lw=0.8)
    ax_c.set_xlabel(xlabel)
    ax_c.set_ylabel('Covariance')
    ax_c.grid(True, alpha=0.3)

    if show_variogram:
        ax_v = axes_arr[1]
        C0 = C_flat[0]
        gamma_emp = C0 - C_flat
        ax_v.plot(rLag_s, gamma_emp, 's', color='tomato', ms=5,
                  label='Empirical')
        if has_model:
            gamma_fit = modelcov_flat[0] - modelcov_flat
            ax_v.plot(rLagI / lag_scale, gamma_fit, '-', color='steelblue',
                      lw=2, label='Fitted model')
            ax_v.legend(fontsize=9)
        ax_v.set_xlabel(xlabel)
        ax_v.set_ylabel(r'$\gamma(h)$')
        ax_v.set_title('Equivalent variogram')
        ax_v.grid(True, alpha=0.3)
        plt.tight_layout()

    return [axes_arr[0], line]

def anisomodelplot(C, r, covmodel=None, covparam=None, theta=None, ratio=None):
    """Plot directional (anisotropic) empirical covariances and an optional fitted model.

    Displays one subplot per direction angle.  Each subplot shows the
    empirical covariance as a function of spatial (and optionally temporal)
    lag for that direction.  If *covmodel* and *covparam* are provided, the
    fitted anisotropic model is overlaid using
    :func:`~stamps.stamps.stats.dependence.stcovfit.anisocovmodelest`.

    Parameters
    ----------
    C : list of np.ndarray
        Length ``nang`` list.  Each element is a 2-D array of shape
        ``(ns, nt)`` containing the empirical covariance for a particular
        direction angle.  For the pure-spatial case ``nt = 1``.
    r : list
        Lag descriptor.  Two formats are accepted:

        * ``[rLags, tLags, angles]`` — space-time case.  ``rLags`` and
          ``tLags`` are 1-D arrays of spatial and temporal lags; ``angles``
          is a 1-D array of direction angles (in radians).
        * ``[rLags, angles]`` — pure-spatial case.
    covmodel : list of str or None, optional
        Covariance model name list in stamps format.  ``None`` plots only the
        empirical values.
    covparam : list of list or None, optional
        Corresponding covariance parameters.
    theta : float or None, optional
        Principal axis angle of the anisotropic ellipse (radians).
    ratio : float or None, optional
        Anisotropy ratio: secondary/principal axis range, in ``(0, 1]``.

    Returns
    -------
    None
        The function creates a :func:`matplotlib.pyplot.figure` directly and
        does not return any axes objects.

    Notes
    -----
    For the pure-spatial case (``tLag is None`` or has a single value) the
    figure uses a 2-column grid with ``ceil(nang / 2)`` rows.  Each subplot
    is annotated with the direction angle in degrees.

    For the space-time case a 2-column grid with ``nang`` rows is used;
    the left column shows the spatial profile and the right column the
    temporal profile for each angle.

    The model is evaluated via
    :func:`~stamps.stamps.stats.dependence.stcovfit.anisocovmodelest` using
    the same spatial and temporal lag grids as the empirical data.

    ### Original Reference ###
    Plot the empirical and modeled covariances.
    Syntax: anisomodelplot(C, rLag, tLag, covmodel, covparam)

    Input:
        C      ns by nt  2D array of empirical covariance
        r      list      [rLags, tLags, angles] or [rLags, angles]
        theta  scalar    principal axis angle (radians)
        ratio  scalar    anisotropy ratio, range (0, 1]
    Remark:
        details of covmodel/covparam refer to stamps.general.coord2K
        details of anisotropic params refer to stamps.stats.stcovfit.anisocovmodelest
    """


    nang=len(C)
    if len(r)==3:
        rLag=r[0]
        tLag=r[1]
        aLag=r[2]
    elif len(r)==2:
        rLag=r[0]
        aLag=r[1]
        tLag=None

    modelcov=[None]*nang

    if covmodel is not None:
        for i in range(nang):
            ang=np.array([aLag[i]])
            if tLag is None:
                tLag=np.array([0])
            tLagI,rLagI=np.meshgrid(tLag,rLag)
            covang,_= anisocovmodelest(covmodel,covparam,theta,ratio,ang,rLagI,tLagI)
            modelcov[i]=covang[0]

    if tLag is None or tLag.size == 1:
        ax=[None]*nang
        cols=2
        rows=int(np.ceil(nang/cols))
        plt.figure()
        for i in range(nang):
            if i<rows:
                ax[i]=plt.subplot2grid((rows,cols),(i,0))
            else:
                ax[i]=plt.subplot2grid((rows,cols),(i-rows,1))
            ax[i].plot(rLag,C[i][:,0],'bo')
            ax[i].set_xlabel('Spatial distance')
            ax[i].set_ylabel('Covariance')
            ax[i].set_xlim([rLag[0],rLag[-1]])
            ax[i].text(0.95, 0.95, '$%5.2f^o$' % (aLag[i]/np.pi*180.),
                verticalalignment='top', horizontalalignment='right',
                transform=ax[i].transAxes,color='black', fontsize=10)
            if modelcov[0] is not None:
                ax[i].plot(rLag,modelcov[i][:,0],'r-')
            if i==0 or i-rows==0:
                ax[i].set_title('Spatial Covariance')
    else:
        ax=[None]*nang*2
        cols=2
        rows=nang
        plt.figure()
        for i in range(nang):
            ax[i]=plt.subplot2grid((rows,cols),(i,0))
            ax[i].plot(rLag,C[i][:,0],'bo')
            ax[i+rows]=plt.subplot2grid((rows,cols),(i,1))
            ax[i+rows].plot(tLag,C[i][0,:],'bo')
            ax[i].text(0.95, 0.95, '$%5.2f^o$' % (aLag[i]/np.pi*180.),
                verticalalignment='top', horizontalalignment='right',
                transform=ax[i].transAxes,color='black', fontsize=10)
            ax[i+rows].text(0.95, 0.95, '$%5.2f^o$' % (aLag[i]/np.pi*180.),
                verticalalignment='top', horizontalalignment='right',
                transform=ax[i+rows].transAxes,color='black', fontsize=10)
            if i==0:
                ax[i].set_title('Spatial Covariance')
                ax[i+rows].set_title('Temporal Covariance')
            if i==nang-1:
                ax[i].set_xlabel('Spatial distance')
                ax[i].set_ylabel('Covariance')
                ax[i].set_xlim([rLag[0],rLag[-1]])
                ax[i+rows].set_xlabel('Temporal lag')
                ax[i+rows].set_ylabel('Covariance')
                ax[i+rows].set_xlim([tLag[0],tLag[-1]])
            else:
                ax[i].axes.get_xaxis().set_ticks([])
                ax[i+rows].axes.get_xaxis().set_ticks([])
            if modelcov[0] is not None:
                ax[i].plot(rLag,modelcov[i][:,0],'r-')
                ax[i+rows].plot(tLag,modelcov[i][0,:],'r-')


def mlemodelplot(ch, zh, covmodel=None, covparam=None):
    """Scatter-plot raw cross-products vs distance and overlay a covariance model.

    For each pair of observations the product ``z_i * z_j`` is plotted
    against the pairwise Euclidean distance.  If a covariance model is
    provided it is overlaid as a red line, giving a visual check of the
    MLE-fitted model against the raw data cloud.

    Parameters
    ----------
    ch : np.ndarray, shape (nh, nd)
        Observation coordinate matrix.  The last column may contain
        ``numpy.datetime64`` values, which are converted to ``float64``
        relative to the first observation time.
    zh : np.ndarray, shape (nh,) or (nh, 1)
        Observed values; reshaped to ``(nh, 1)`` internally.
    covmodel : list of str or None, optional
        Nested covariance model name list in stamps format.  ``None`` plots
        only the data scatter.
    covparam : list of list or None, optional
        Corresponding covariance parameters.

    Returns
    -------
    None
        The function calls :func:`matplotlib.pyplot.figure` and
        :func:`matplotlib.pyplot.show` directly.

    Notes
    -----
    Only the upper triangle of the ``(nh × nh)`` cross-product matrix is
    plotted (using ``numpy.triu_indices``) to avoid duplicating symmetric
    pairs.

    The model overlay is evaluated on 50 equally spaced lag values covering
    ``[0, 2/3 × max_distance]`` for both spatial and temporal axes.

    For space-time data (``ch.shape[1] > 2``) the temporal distance is also
    computed, but the current implementation prints an empty string and the
    plot is not yet generated for this case (marked as *to be written*).

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.stamps.graph.modelplot import mlemodelplot
    >>> rng = np.random.default_rng(0)
    >>> ch = rng.uniform(0, 100, (20, 1))
    >>> zh = rng.standard_normal(20)
    >>> mlemodelplot(ch, zh)
    """
    zh = zh.reshape(zh.size, 1)
    cov=zh.dot(zh.T)
    covu=cov[np.triu_indices(zh.size)]
    dist_s=coord2dist(ch[:,:2],ch[:,:2])
    distu_s=dist_s[np.triu_indices(zh.size)]
    if ch.shape[1]>2:
        if type(ch[0,-1])==np.datetime64:
            origin=ch[0,-1]
            ch[:,-1]=np.double(np.asarray(ch[:,-1],dtype='datetime64')-origin)
            ch=ch.astype(np.double)
        dist_t=coord2dist(ch[:,2],ch[:,2])
        distu_t=dist_t[np.triu_indices(zh.size)]
    if (covmodel is not None) and (covparam is not None):
        rLagI=np.linspace(0,dist_s.max()*2./3,50)
        if ch.shape[1]>2:
            tLagI=np.linspace(0,dist_t.max()*2./3,50)
        else:
            tLagI=np.array([0])
        tLagMI,rLagMI=np.meshgrid(tLagI,rLagI)
        modelcov,covi=covmodelest(rLagMI,tLagMI,covmodel,covparam)
    if ch.shape[1]>2:
        print('')
    else:
        plt.figure()
        plt.scatter(distu_s.flat[:],covu.flat[:])
        plt.plot(rLagI,modelcov,'r-')
        plt.xlim(0,rLagI.max())
        plt.ylim(covu.min(),covu.max())
        plt.xlabel('Distance')
        plt.ylabel('Covariance')


def covariance_model_plot(cov_model, s_range=None, t_range=None, show=True):
    """Plot a space-time covariance model over a regular spatial/temporal lag grid.

    Evaluates the specified covariance model on a ``100 × 100`` grid of
    spatial and temporal lags and produces:

    * A 2-D line plot of the **spatial** profile (temporal lag = 0).
    * A 2-D line plot of the **temporal** profile (spatial lag = 0).
    * A 3-D wireframe of the full space-time covariance surface.

    Parameters
    ----------
    cov_model : list of list
        Covariance model specification.  Each inner list has the form
        ``[model_type, sill, s_range, ..., t_range, ...]``.  The function
        reads the maximum spatial range from ``max(i[2] for i in cov_model)``
        and the maximum temporal range from ``max(i[4] for i in cov_model)``
        when the corresponding *s_range* / *t_range* arguments are ``None``.
    s_range : float or None, optional
        Maximum spatial lag to display.  If ``None``, inferred from the model
        parameters.
    t_range : float or None, optional
        Maximum temporal lag to display.  If ``None``, inferred from the model
        parameters.
    show : bool, optional
        If ``True`` (default), calls :func:`matplotlib.pyplot.show`.

    Returns
    -------
    cov_t_lags_grid : np.ndarray, shape (100, 100)
        Meshgrid of temporal lag values.
    cov_s_lags_grid : np.ndarray, shape (100, 100)
        Meshgrid of spatial lag values.
    cov_z_grid : np.ndarray, shape (100, 100)
        Evaluated covariance values on the grid.

    Notes
    -----
    The covariance model is evaluated via
    :func:`~stamps.stamps.stats.dependence.stcovfit.cal_cov_mod`.
    The model format is the legacy list-of-tuples format used by
    :func:`cal_cov_mod`, not the standard stamps ``[sill, [range]]`` format.

    Examples
    --------
    >>> from stamps.stamps.graph.modelplot import covariance_model_plot
    >>> cov_model = [['exponentialCST', 1.0, 50.0, None, 5.0]]
    >>> tg, sg, zg = covariance_model_plot(cov_model, show=False)
    """
    s_range = max([i[2] for i in cov_model]) if not s_range else s_range
    t_range = max([i[4] for i in cov_model]) if not t_range else t_range

    cov_s_lags = np.linspace(0, s_range, 100)
    cov_t_lags = np.linspace(0, t_range, 100)
    cov_t_lags_grid, cov_s_lags_grid = np.meshgrid(cov_t_lags,cov_s_lags)


    cov_st_lags = np.array(
        list(zip(*list(map(
            lambda x:x.flatten(),
            np.meshgrid(cov_s_lags,cov_t_lags)
            ))))
        )
    cov_z = cal_cov_mod(cov_st_lags, cov_model)
    cov_z_grid = cov_z.reshape((cov_s_lags.shape[0],cov_t_lags.shape[0]))

    if show:
        plt.figure(1)
        plt.subplot(211)
        plt.plot(cov_s_lags_grid[:,0], cov_z_grid[:,0], 'b--')
        plt.subplot(212)
        plt.plot(cov_t_lags_grid[0], cov_z_grid[0], 'r--')

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(2)
        ax3d = fig.add_subplot(111, projection='3d')
        ax3d.plot_wireframe(cov_t_lags_grid, cov_s_lags_grid, cov_z_grid)

    
        plt.show()

    return cov_t_lags_grid, cov_s_lags_grid, cov_z_grid


def _eval_single_covmodel(modname, covparam, end_dis=1.3):
    """Evaluate a single analytic covariance model on a lag vector.

    Returns ``(x, y, c0)`` where *x* is the lag array, *y* the covariance
    values, and *c0* the sill.
    """
    if modname == 'Gaussian':
        c0, ar = covparam
        x = np.linspace(0, ar * end_dis, 500)
        y = c0 * np.exp(-3 * x ** 2 / ar ** 2)
    elif modname == 'Exponential':
        c0, ar = covparam
        x = np.linspace(0, ar * end_dis, 500)
        y = c0 * np.exp(-3 * x / ar)
    elif modname == 'Spherical':
        c0, ar = covparam
        x = np.linspace(0, ar, 500)
        y = c0 - c0 * (3 / 2 * x / ar - 1 / 2 * (x / ar) ** 3)
        x_ = np.linspace(ar, ar * end_dis, 5)
        y_ = np.zeros(len(x_))
        x = np.hstack((x, x_))
        y = np.hstack((y, y_))
    elif modname == 'Nugget':
        c0, ar = covparam[0], 2000
        x = np.linspace(0, ar * end_dis, 500)
        y = np.zeros(len(x))
        x = np.hstack((np.array([0]), x))
        y = np.hstack((np.array([c0]), y))
    else:
        raise ValueError(f"Unknown model: {modname}")
    return x, y, c0


def covmodel_plot(modname, covparam, end_dis=1.3, plot=True, ch2semi=False,
                  ax=None):
    """Plot parametric covariance (or semivariogram) model curve(s).

    **Single model** — when *modname* is a string, plots one curve (legacy
    behaviour).

    **Gallery mode** — when *modname* is a list of ``(name, param, title)``
    tuples, creates a subplot grid displaying every model in the list.
    In gallery mode *covparam* is ignored (parameters come from the tuples)
    and the *ax* parameter is ignored.

    Parameters
    ----------
    modname : str or list of tuples
        Model name string **or** a list of ``(name, [sill, range], title)``
        tuples for gallery mode.
    covparam : list or tuple
        ``[sill, effective_range]`` — used only in single-model mode.
    end_dis : float
        Distance axis extent as a multiple of the effective range.
    plot : bool
        ``True`` → show figure.  ``False`` → draw on *ax* (or current axes)
        without showing.
    ch2semi : bool
        Convert covariance to semivariogram ``γ(h) = C(0) − C(h)``.
    ax : Axes or None
        Pre-existing axes (single-model mode only).  ``None`` creates a
        new figure when *plot* is ``True``.

    Returns
    -------
    ax or fig, axes
        Single-model mode returns ``None`` (legacy) or the axes when *ax*
        is provided.  Gallery mode returns ``(fig, axes)``.
    """
    # --- Gallery mode ---
    if isinstance(modname, list):
        models = modname
        n = len(models)
        ncols = min(n, 4)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
        axes_flat = np.asarray(axes).ravel()
        _default_colors = ['steelblue', 'tomato', 'darkorange', 'mediumseagreen',
                           'mediumpurple', 'goldenrod', 'teal', 'brown']
        ylabel = r'$\gamma(h)$' if ch2semi else 'C(h)'
        for i, entry in enumerate(models):
            a = axes_flat[i]
            if len(entry) == 4:
                # (callable/name, param_or_h, y_vals_or_None, title)
                mname, mparam, y_or_none, mtitle = entry
            else:
                mname, mparam, mtitle = entry
                y_or_none = None

            if callable(mname):
                x = mparam
                y = mname(x) if y_or_none is None else y_or_none
                c0 = y[0] if len(y) > 0 else 0.0
            else:
                x, y, c0 = _eval_single_covmodel(mname, mparam, end_dis)

            if ch2semi:
                y = c0 - y
            a.plot(x, y, color=_default_colors[i % len(_default_colors)], lw=2.5)
            a.axhline(0, color='k', lw=0.6, ls='--')
            a.set_title(mtitle, fontsize=10)
            a.set_xlabel('Lag h')
            a.set_ylabel(ylabel)
        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)
        kind = 'Variogram' if ch2semi else 'Covariance'
        fig.suptitle(f'{kind} model gallery', fontsize=13, y=1.01)
        plt.tight_layout()
        if plot:
            plt.show()
        return fig, axes

    # --- Single model mode ---
    x, y, c0 = _eval_single_covmodel(modname, covparam, end_dis)
    if ch2semi:
        y = c0 - y

    if ax is not None:
        ax.plot(x, y, lw=2, label=modname)
        ax.legend()
        return ax

    if not plot:
        plt.plot(x, y, label=modname)
        plt.legend()
        return
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label=modname)
    plt.title(modname)
    plt.xlabel('distance')
    plt.ylabel('covariance' if not ch2semi else 'semivariogram')
    plt.legend()
    plt.show()

def semivario_plot(modname, covparam, end_dis=1.3, plot=True):
    """Plot a semivariogram model curve.

    A convenience wrapper around :func:`covmodel_plot` that sets
    ``ch2semi=True``, converting the covariance ``C(h)`` to the
    semivariogram ``γ(h) = C(0) − C(h)`` before plotting.

    Parameters
    ----------
    modname : {'Gaussian', 'Exponential', 'Spherical', 'Nugget'}
        Name of the covariance model; see :func:`covmodel_plot`.
    covparam : list or tuple of length 2
        ``[sill, effective_range]``; see :func:`covmodel_plot`.
    end_dis : float, optional
        Distance axis extent as a multiple of the effective range (default 1.3).
    plot : bool, optional
        If ``True`` (default), creates a new figure and calls
        :func:`matplotlib.pyplot.show`.  Pass ``False`` to draw on the
        current axes.

    Returns
    -------
    None

    Notes
    -----
    This is a **wrapper** for :func:`covmodel_plot` with ``ch2semi=True``.
    All logic is delegated to that function.

    Examples
    --------
    >>> from stamps.stamps.graph.modelplot import semivario_plot
    >>> semivario_plot('Spherical', [1.0, 100.0], end_dis=1.5, plot=False)
    """
    covmodel_plot(modname, covparam, end_dis=end_dis, plot=plot, ch2semi=True)
    return


def lmcplot(lmc_result, V_tensor, D_arr, varnames=None, colors=None,
            lag_scale=1.0, lag_unit=None, figsize=None):
    """Plot diagonal auto-covariances from a fitted Linear Model of Coregionalization.

    For each variable, compares the empirical auto-covariance (diagonal of
    *V_tensor*) against the fitted LMC model (sum of weighted model structures
    from *lmc_result*).  Produces one subplot per variable arranged in a
    single row.

    Parameters
    ----------
    lmc_result : list of dict
        Output of :func:`~stamps.stamps.stats.dependence.covariance.stcovfit.coregfit`.
        Each dict contains:

        * ``'model'`` — model name string (e.g. ``'nuggetC'``, ``'sphericalC'``).
        * ``'sill_matrix'`` — ``(nv, nv)`` fitted sill matrix for this structure.
        * ``'range_param'`` — scalar range parameter, or ``None`` for a nugget.

    V_tensor : np.ndarray, shape (nv, nv, n_lags)
        Empirical cross-covariance tensor.  The diagonal slice
        ``V_tensor[vi, vi, :]`` is the auto-covariance for variable *vi*.
    D_arr : np.ndarray, shape (n_lags,)
        Shared spatial lag axis (same units as the original data coordinates).
    varnames : list of str, optional
        Variable names for subplot titles.  Defaults to
        ``['Var 0', 'Var 1', ...]``.
    colors : list, optional
        Marker colours for each variable's empirical points.  Defaults to a
        built-in palette.
    lag_scale : float, optional
        Divide lag values by this factor for axis labels
        (e.g. ``1000`` to display kilometres when coordinates are in metres).
        Default ``1.0``.
    lag_unit : str or None, optional
        Unit label appended to the x-axis (e.g. ``'km'``, ``'m'``).
        If ``None``, no unit is shown.
    figsize : tuple or None, optional
        Figure size ``(width, height)`` in inches.  Defaults to
        ``(5.5 * nv, 4)``.

    Returns
    -------
    axes : list of matplotlib.axes.Axes
        One ``Axes`` object per variable.

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.stamps.graph.modelplot import lmcplot
    >>> # Synthetic LMC result and V_tensor
    >>> lmc = [{'model': 'sphericalC', 'sill_matrix': np.eye(2), 'range_param': 5000.0}]
    >>> V = np.zeros((2, 2, 10))
    >>> d = np.arange(0, 10000, 1000, dtype=float)
    >>> axes = lmcplot(lmc, V, d, varnames=['Sand', 'Silt'], lag_scale=1000, lag_unit='km')
    """
    from ..models.covmodel import get_model

    nv = V_tensor.shape[0]
    n_lags = V_tensor.shape[2]

    if varnames is None:
        varnames = [f'Var {i}' for i in range(nv)]
    _default_colors = ['steelblue', 'goldenrod', 'tomato', 'mediumpurple',
                       'seagreen', 'sienna']
    if colors is None:
        colors = [_default_colors[i % len(_default_colors)] for i in range(nv)]
    if figsize is None:
        figsize = (5.5 * nv, 4)

    xlabel = f'Lag ({lag_unit})' if lag_unit else 'Lag'

    # Fine lag grid for smooth model curves
    h_plot = np.linspace(0, D_arr.max(), 300)
    h_plot_scaled = h_plot / lag_scale
    D_scaled = D_arr / lag_scale

    fig, axes = plt.subplots(1, nv, figsize=figsize)
    if nv == 1:
        axes = [axes]

    for vi, (ax, vname, col) in enumerate(zip(axes, varnames, colors)):
        # Empirical diagonal auto-covariance (drop NaN / empty bins)
        cov_emp = V_tensor[vi, vi, :]
        valid = ~np.isnan(cov_emp)
        ax.plot(D_scaled[valid], cov_emp[valid],
                'o', color=col, ms=5, label='Empirical')

        # Fitted LMC diagonal: sum over all structures
        C_fit = np.zeros(len(h_plot))
        for struct in lmc_result:
            model_fn = get_model(struct['model'])
            b_ii = struct['sill_matrix'][vi, vi]
            rp = struct.get('range_param', None)
            if rp is None:
                C_fit += model_fn(h_plot, b_ii)
            else:
                C_fit += model_fn(h_plot, b_ii, rp)

        ax.plot(h_plot_scaled, C_fit, '-', color='k', lw=2, label='LMC fit')
        ax.axhline(0, color='gray', ls='--', lw=0.7, alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Covariance')
        ax.set_title(f'{vname}: empirical vs LMC')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return axes