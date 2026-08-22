# -*- coding: utf-8 -*-
import copy
import warnings as _warnings_mod

from six.moves import range
import numpy as np
import scipy.stats
import scipy.linalg
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree

from .softconverter import proba2stat, pdf2cdf
from ..general.coord2K import coord2K, coord2Ksplit
from .pystks_variable import get_standard_order, get_standard_soft_pdf_type
from .BMEoptions import BMEoptions
# maxentpdf_gc / maxentcondpdf_gc are imported lazily inside the functions
# that need them to avoid a circular import with analysis/__init__.py.
from ..general.valstvgx import valstv2stg, valstg2stv
from ..general.neighbours import neighbours, neighbours_index_kd
from ..mvn.qmc import qmc

# pyAllMomentsNG is only used in the legacy BMEPosteriorPDF_backup function.
# It is not available in stamps_v3; provide a stub so the module loads.
try:
    from ..mvn.pyAllMoments import pyAllMomentsNG
except ImportError:
    def pyAllMomentsNG(*args, **kwargs):  # type: ignore[misc]
        raise NotImplementedError(
            "pyAllMomentsNG is not available in this installation. "
            "Use BMEPosteriorPDF instead."
        )


# ===================================================================
# Robust SVD wrapper – Fix A: dgesdd → gesvd fallback
# ===================================================================

def _robust_svd(a, full_matrices=True):
    """Drop-in replacement for ``np.linalg.svd`` that automatically
    falls back to the more robust ``gesvd`` LAPACK driver when the
    default ``dgesdd`` driver fails to converge.

    NumPy's ``np.linalg.svd`` uses the divide-and-conquer algorithm
    ``dgesdd``, which is fast but can raise ``LinAlgError: SVD did not
    converge`` on ill-conditioned or near-singular matrices common in
    spatiotemporal BME covariance matrices.  SciPy exposes
    ``scipy.linalg.svd(lapack_driver='gesvd')`` which uses the older
    QR-iteration algorithm that is slower but substantially more robust.

    Parameters
    ----------
    a : array_like
        Matrix to decompose.
    full_matrices : bool, optional
        If True (default), U and Vh have shapes (M, M) and (N, N).
        If False, the shapes are (M, K) and (K, N) where K = min(M, N).

    Returns
    -------
    U, s, Vh : ndarray
        Same as ``np.linalg.svd``.
    """
    try:
        return np.linalg.svd(a, full_matrices=full_matrices)
    except np.linalg.LinAlgError:
        _warnings_mod.warn(
            "[STARBME] np.linalg.svd (dgesdd) failed to converge; "
            "falling back to scipy.linalg.svd (gesvd).",
            RuntimeWarning, stacklevel=2)
        return scipy.linalg.svd(
            a, full_matrices=full_matrices, lapack_driver='gesvd')


def _robust_pinv(a, rcond=1e-15):
    """Pseudo-inverse using ``_robust_svd`` instead of ``np.linalg.svd``.

    ``np.linalg.pinv`` delegates to the default dgesdd SVD driver which
    can raise ``LinAlgError`` or produce overflow/invalid-value warnings
    on ill-conditioned covariance matrices.  This function follows the
    same algorithm (truncated SVD) but uses ``_robust_svd`` so that it
    automatically falls back to the more stable gesvd driver.

    Parameters
    ----------
    a : array_like, shape (M, N)
        Matrix to pseudo-invert.
    rcond : float, optional
        Singular values smaller than ``rcond * max(s)`` are set to zero.

    Returns
    -------
    a_pinv : ndarray, shape (N, M)
        The pseudo-inverse of *a*.
    """
    a = np.asarray(a)
    if a.size == 0:
        return a.T  # empty matrix edge case
    U, s, Vh = _robust_svd(a, full_matrices=False)
    # Threshold small singular values (same logic as np.linalg.pinv)
    cutoff = rcond * s.max()
    large = s > cutoff
    s_inv = np.zeros_like(s)
    s_inv[large] = 1.0 / s[large]
    return (Vh.T * s_inv[np.newaxis, :]) @ U.T


KHS_DICT = {'k': 0, 'h': 1, 's': 2}


def _bme_posterior_pdf(
    ck, ch=None, cs=None, zh=None, zs=None,
    covmodel=None, covparam=None, covmat=None,
    order=np.nan, options=None,
    general_knowledge='gaussian',
    #  specific_knowledge='unknown',  
    pdfk=None, pdfh=None, pdfs=None, hk_k=None, hk_h=None, hk_s=None,
    gui_args=None):

    def __get_zs_integration_limits(zs):
        '''get soft data integration limit''' 
        ranges = []
        for zsi in zs:
            pdftype = get_standard_soft_pdf_type(zsi[0])
            if pdftype in  [1, 2]: # nl, limi, probdens
                nl = int(np.asarray(zsi[1]).flat[0])
                limi = np.asarray(zsi[2]).ravel()
                ranges.append((float(limi[0]), float(limi[nl-1])))
            elif pdftype in [10]:
                zm = float(np.asarray(zsi[1]).flat[0])
                zstd = float(np.sqrt(np.asarray(zsi[2]).flat[0]))
                ranges.append((zm-3*zstd, zm+3*zstd))
        ranges = np.array(ranges)
        return ranges.copy()
    order = get_standard_order(order)
    nk = ck.shape[0]
    nh = ch.shape[0] if ch is not None else 0
    ns = cs.shape[0] if cs is not None else 0

    # --- Auto-Gaussian fallback for _bme_posterior_pdf ---
    # Same logic as in _bme_posterior_moments: when the number of
    # non-Gaussian soft data exceeds a threshold, convert them to
    # Gaussian (mean, variance) to avoid intractable high-dimensional
    # QMC integration.
    if zs and general_knowledge == 'gaussian':
        _NS_AUTO_GAUSS_PDF = 20
        _use_gauss_approx_pdf = (
            options is not None and options.get('soft_approx_gaussian', False))
        if not _use_gauss_approx_pdf:
            _ns_nongauss_pdf = sum(
                1 for zsi in zs
                if get_standard_soft_pdf_type(zsi[0]) != 10)
            if _ns_nongauss_pdf > _NS_AUTO_GAUSS_PDF:
                import warnings
                warnings.warn(
                    "[_bme_posterior_pdf] ns_nongaussian={} exceeds "
                    "auto-Gaussian threshold ({}). Approximating "
                    "non-Gaussian soft PDFs as Gaussian.".format(
                        _ns_nongauss_pdf, _NS_AUTO_GAUSS_PDF))
                _use_gauss_approx_pdf = True
        if _use_gauss_approx_pdf:
            zs_converted = []
            for zsi in zs:
                pdftype = get_standard_soft_pdf_type(zsi[0])
                if pdftype in [1, 2]:
                    zs_gau_m, zs_gau_v = proba2stat(
                        zsi[0],
                        np.atleast_2d(zsi[1]),
                        np.atleast_2d(zsi[2]),
                        np.atleast_2d(zsi[3]))
                    zs_converted.append(
                        (10, float(np.asarray(zs_gau_m).flat[0]),
                         float(np.asarray(zs_gau_v).flat[0])))
                else:
                    zs_converted.append(zsi)
            zs = zs_converted

    x_all_split = _get_x_all_split(nk, zh, zs)
    Xh = _get_x(x_all_split, 'h')
    mean_all_split = _get_mean_all_split(x_all_split, order)
    #get cov_all_split
    if covmat is None:
        cov_all_split = _get_cov_all_split(ck, ch, cs, covmodel, covparam)
        _cov_rows = []
        for _row in cov_all_split:
            _non_none = [i for i in _row if i is not None]
            if _non_none:
                _cov_rows.append(np.hstack(_non_none))
        covmat = np.vstack(_cov_rows)
    else:
        cov_all_split = np.vsplit(
            covmat, [nk, nk+nh, nk+nh+ns]
            )[:-1] #exclude final empty array
        cov_all_split = \
            [np.hsplit(c, [nk, nk+nh, nk+nh+ns][:-1])\
            for c in cov_all_split]

    #find duplicated point (covariance-based + exact coordinate match)
    if ns:
        # Covariance-based detection (unit-free)
        _ck_dup_pdf, _cs_dup_pdf, _ = _find_ck_cs_near_duplicates(
            covmat, nk, nh, ns)
        # Also check exact coordinate equality
        _exact_pairs = np.array(
            np.all((ck[:, None, :] == cs[None, :, :]), axis=-1).nonzero()
            ).T
        if _ck_dup_pdf.size > 0:
            _cov_pairs = np.column_stack([_ck_dup_pdf, _cs_dup_pdf])
            if _exact_pairs.size > 0:
                dup_ck_cs_idx = np.unique(
                    np.vstack([_cov_pairs, _exact_pairs]), axis=0)
            else:
                dup_ck_cs_idx = _cov_pairs
        else:
            dup_ck_cs_idx = _exact_pairs if _exact_pairs.size > 0 \
                else np.array([]).reshape(0, 2).astype(int)
    else:
        dup_ck_cs_idx = np.array([[]])

    if general_knowledge == 'gaussian':
        
        def __get_fG_Xkh_each_ck(fG_Xkh_):
            if Xh is None:
                def fG_Xkh_each_ck(xk):
                    xk_origin_shape = xk.shape
                    xk = xk.flatten()
                    input_xk = xk.T
                    return fG_Xkh_(input_xk).reshape(xk_origin_shape)
            else:
                def fG_Xkh_each_ck(xk):
                    xk_origin_shape = xk.shape
                    xk = xk.flatten()
                    input_xk = np.vstack(
                        (xk, np.tile(Xh, xk.size))
                        ).T
                    return fG_Xkh_(input_xk).reshape(xk_origin_shape)
            return fG_Xkh_each_ck     
        
        #fG_Xh: const
        if nh == 0:
            fG_Xh = 1.
        else:
            fG_Xh = _get_multivariate_normal_pdf(
                x_all_split, mean_all_split, cov_all_split, 'h')(
                    _get_x(x_all_split, 'h').T)

        def __get_fG_Xs_gvn_Xh_all_ck():
            fG_Xs_gvn_Xh = _get_multivariate_normal_pdf(
                x_all_split, mean_all_split, cov_all_split, 's_h')
            def fG_Xs_gvn_Xh_all_ck(xs):
                xs_origin_shape = xs.shape
                xs = xs.reshape(-1, xs_origin_shape[-1])
                output = fG_Xs_gvn_Xh(xs)
                return output.reshape(xs_origin_shape[:-1]+(1,))
            return fG_Xs_gvn_Xh_all_ck
        fG_Xs_gvn_Xh = __get_fG_Xs_gvn_Xh_all_ck()

        zs_limits = __get_zs_integration_limits(zs)

        # --- Importance-sampling transform for ALL qmc variants ---
        # Compute conditional Gaussian f(xs|xh) parameters for IS.
        m_s_given_h_is = _get_mean_a_given_b(
            x_all_split, mean_all_split,
            cov_all_split, 's', 'h')
        cov_s_given_h_is = _get_sigma_a_given_b(
            cov_all_split, 's', 'h')
        # Ensure PSD
        _eigvals_is = np.linalg.eigvalsh(cov_s_given_h_is)
        if np.any(_eigvals_is < 0):
            _eps_is = max(abs(_eigvals_is.min()) * 2, 1e-12)
            cov_s_given_h_is += np.eye(cov_s_given_h_is.shape[0]) * _eps_is

        from scipy.special import erfinv as _erfinv_1
        _u1, _s1, _vh1 = _robust_svd(cov_s_given_h_is)
        _s1 = np.maximum(_s1, 1e-14)
        _A_is1_full = _vh1.T * np.sqrt(_s1)
        _mu_is1 = m_s_given_h_is.ravel()

        # --- PCA dimension reduction for _bme_posterior_pdf ---
        _pca_thr1 = options.get('pca_integration_threshold', 0.999) \
            if options is not None else 0.999
        _total_var1 = _s1.sum()
        if _total_var1 > 0 and _pca_thr1 < 1.0:
            _cumvar1 = np.cumsum(_s1) / _total_var1
            _ns_eff1 = int(np.searchsorted(_cumvar1, _pca_thr1) + 1)
            _ns_eff1 = max(1, min(_ns_eff1, ns))
        else:
            _ns_eff1 = ns

        if _ns_eff1 < ns:
            import warnings
            warnings.warn(
                "[PCA-pdf] Reducing QMC integration dimension from "
                "{ns} to {ne} (threshold={thr}, variance retained="
                "{vr:.4f})".format(
                    ns=ns, ne=_ns_eff1, thr=_pca_thr1,
                    vr=_s1[:_ns_eff1].sum() / _total_var1))
            _A_is1 = _A_is1_full[:, :_ns_eff1]
        else:
            _A_is1 = _A_is1_full

        _ns_qmc1 = _ns_eff1  # QMC integration dimension

        fS_Xs = _get_fs(zs)

        if options['integration method'] in ('qmc', 'qmc_T'):
            # Importance sampling: absorb Gaussian into sampling measure
            def __qmc_int_fG_Xs_gvn_Xh__fS_Xs(u_array):
                u_c = np.clip(u_array, 1e-6, 1.0 - 1e-6)
                z = np.sqrt(2.0) * _erfinv_1(2.0 * u_c - 1.0)
                xs_array = (_A_is1.dot(z.T) + _mu_is1.reshape(-1, 1)).T
                return fS_Xs(xs_array)
            xmin = np.zeros(_ns_qmc1)
            xmax = np.ones(_ns_qmc1)

        elif options['integration method'] == 'qmc_F':
            Fsinv = _get_Fsinv(zs)
            def __qmc_int_fG_Xs_gvn_Xh__fS_Xs(Fx_array):
                return fG_Xs_gvn_Xh(Fsinv(Fx_array))
            xmin = np.zeros(zs_limits[:,0].shape)
            xmax = np.ones(zs_limits[:,1].shape)

        else:
            # Fallback: unknown method → use IS as default
            def __qmc_int_fG_Xs_gvn_Xh__fS_Xs(u_array):
                u_c = np.clip(u_array, 1e-6, 1.0 - 1e-6)
                z = np.sqrt(2.0) * _erfinv_1(2.0 * u_c - 1.0)
                xs_array = (_A_is1.dot(z.T) + _mu_is1.reshape(-1, 1)).T
                return fS_Xs(xs_array)
            xmin = np.zeros(_ns_qmc1)
            xmax = np.ones(_ns_qmc1)

        int_fG_Xs_gvn_Xh__fS_Xs, e, info = qmc(
            __qmc_int_fG_Xs_gvn_Xh__fS_Xs,
            xmin, xmax,
            abserr=options[1,0],
            relerr=options[3,0],
            maxeval=int(options[2,0]),
            pow2min=10,
            showinfo=options['qmc_showinfo']
            )

        if int_fG_Xs_gvn_Xh__fS_Xs == 0 or np.isnan(int_fG_Xs_gvn_Xh__fS_Xs) or np.isinf(int_fG_Xs_gvn_Xh__fS_Xs):
            import warnings
            warnings.warn(
                "[STARBME] PDF normalization constant NC={:.4e} (ns={:d}). "
                "IS integration may have failed.  Posterior PDF values "
                "at these estimation locations may be unreliable.".format(
                    float(int_fG_Xs_gvn_Xh__fS_Xs), ns))
            # Set to a tiny positive value to avoid division-by-zero
            # downstream.  The posterior PDF will be nearly zero, but at
            # least it won't produce NaN.
            int_fG_Xs_gvn_Xh__fS_Xs = 1e-300


        def __get_fSk_Xk_each_ck(zsk):
            def fSk_Xk_each_ck(xk):
                if zsk is None:
                    return np.ones(xk.shape)
                else:
                    xk_origin_shape = xk.shape
                    xk = xk.flatten()
                    pdf_type = get_standard_soft_pdf_type(zsk[0])
                    if pdf_type == 2:
                        nl = zsk[1][0]
                        limi = zsk[2]
                        probdens = zsk[3]
                        y_i = np.interp(
                            xk, limi[:nl], probdens[:nl],
                            left = 0., right = 0.)
                    elif pdf_type == 1:
                        # BUG FIX: was using zs[1],zs[2],zs[3] (global list)
                        # instead of zsk[1],zsk[2],zsk[3] (individual soft datum)
                        nl = int(zsk[1][0])
                        limi = zsk[2]
                        probdens = zsk[3]
                        # Histogram PDF: probdens[j] is constant for limi[j] <= x < limi[j+1]
                        bin_idx = np.searchsorted(limi[:nl], xk, side='right') - 1
                        y_i = np.where(
                            (bin_idx >= 0) & (bin_idx < nl - 1),
                            probdens[np.clip(bin_idx, 0, nl - 2)],
                            0.0)
                    elif pdf_type == 10:
                        zm = zsk[1]
                        zstd = np.sqrt(zsk[2])
                        try:
                            y_i = scipy.stats.norm.pdf(
                                xk, loc=zm, scale=zstd)
                        except FloatingPointError:
                            y_i = np.zeros(xk.shape)
                      
                    return y_i.reshape(xk_origin_shape)
            return fSk_Xk_each_ck

        def __get_fG_Xs_gvn_Xkh_each_ck(ck_i, cs, zs,
            x_all_split_each_ck, mean_all_split_each_ck, cov_all_split_each_ck):
            idx_result = np.where(np.all(ck_i == cs, axis=1))[0]
            if idx_result.size == 0:
                x_all_split_each_ck_dup = x_all_split_each_ck
                mean_all_split_each_ck_dup = mean_all_split_each_ck
                cov_all_split_each_ck_dup = cov_all_split_each_ck
            elif idx_result.size == 1:
                x_all_split_each_ck_dup =\
                    x_all_split_each_ck[:2] +\
                    [np.delete(x_all_split_each_ck[2], idx_result, axis=0)]
                mean_all_split_each_ck_dup =\
                    mean_all_split_each_ck[:2] +\
                    [np.delete(mean_all_split_each_ck[2], idx_result, axis=0)]
                cov_all_split_each_ck_dup = copy.deepcopy(cov_all_split_each_ck)
                cov_all_split_each_ck_dup[0][2] =\
                    np.delete(cov_all_split_each_ck[0][2], idx_result, axis=1)
                cov_all_split_each_ck_dup[1][2] =\
                    np.delete(cov_all_split_each_ck[1][2], idx_result, axis=1)
                cov_all_split_each_ck_dup[2][0] =\
                    np.delete(cov_all_split_each_ck[2][0], idx_result, axis=0)
                cov_all_split_each_ck_dup[2][1] =\
                    np.delete(cov_all_split_each_ck[2][1], idx_result, axis=0)
                cov_all_split_each_ck_dup[2][2] =\
                    np.delete(cov_all_split_each_ck[2][2], idx_result, axis=0)
                #be careful below
                cov_all_split_each_ck_dup[2][2] =\
                    np.delete(cov_all_split_each_ck_dup[2][2], idx_result, axis=1)
            elif idx_result.size > 1: #strange
                raise ValueError('ck match cs twice. (strange)')
            def __get_fG_Xs_gvn_Xkh_each_ck_eack_xk(xk):
                fG_Xs_gvn_Xkh_container = []
                xk_origin_shape = xk.shape
                xk = xk.flatten()
                if options['integration method'] == 'qmc_T':
                    for xk_i in xk:
                        x_all_split_each_ck_dup[0] = np.array([[xk_i]])
                        m = _get_mean_a_given_b(
                            x_all_split_each_ck_dup,
                            mean_all_split_each_ck_dup,
                            cov_all_split_each_ck_dup, 's', 'kh')
                        v = _get_sigma_a_given_b(cov_all_split_each_ck_dup, 's', 'kh')
                        u, s, vh = _robust_svd(v)
                        A = vh.T*np.sqrt(s)

                        fG_Xs_gvn_Xkh_ = lambda xs, A=A, m=m: (A, m)

                        def fG_Xs_gvn_Xkh_each_ck_eack_xk(xs, ff):
                            xs_origin_shape = xs.shape
                            xs = xs.reshape(-1, xs_origin_shape[-1])
                            A, m = ff(xs)
                            xs[xs==0.] = 10**-5
                            xs[xs==1.] = 1 - 10**-5
                            xs2 = np.sqrt(2) * erfinv(2*xs - 1)
                            xs_array = (A.dot(xs2.T) + m).T
                            return xs_array

                        fG_Xs_gvn_Xkh_container.append(
                            lambda xs, ff=fG_Xs_gvn_Xkh_: fG_Xs_gvn_Xkh_each_ck_eack_xk(xs, ff))
                else:
                    for xk_i in xk:
                        x_all_split_each_ck_dup[0] = np.array([[xk_i]])
                        fG_Xs_gvn_Xkh_ = _get_multivariate_normal_pdf(
                        x_all_split_each_ck_dup,
                        mean_all_split_each_ck_dup,
                        cov_all_split_each_ck_dup, 's_kh')
                        def fG_Xs_gvn_Xkh_each_ck_eack_xk(xs, ff):
                            xs_origin_shape = xs.shape
                            xs = xs.reshape(-1, xs_origin_shape[-1])
                            output = ff(xs)
                            return output.reshape(xs_origin_shape[:-1]+(1,))
                        fG_Xs_gvn_Xkh_container.append(
                            lambda xs, ff=fG_Xs_gvn_Xkh_: fG_Xs_gvn_Xkh_each_ck_eack_xk(xs, ff))
                return np.array(
                    fG_Xs_gvn_Xkh_container).reshape(xk_origin_shape)
            return __get_fG_Xs_gvn_Xkh_each_ck_eack_xk

        def __get_fS_Xs_dup(ck_i, cs, zs):
            idx_result = np.where(np.all(ck_i == cs, axis=1))[0]
            if idx_result.size == 0:
                if options['integration method'] == 'qmc_F':
                    return Fsinv
                else:
                    return fS_Xs
            elif idx_result.size == 1:
                if options['integration method'] == 'qmc_F':
                    return _get_Fsinv(
                        [zs_i for i, zs_i in enumerate(zs) if i != idx_result])
                else:
                    return _get_fs(
                        [zs_i for i, zs_i in enumerate(zs) if i != idx_result])
            elif idx_result.size > 1: #strange
                raise ValueError('ck match cs twice. (strange)')

        def __get_pdf_each_ck(i):
            ck_i = ck[i]
            idx_result = np.where(np.all(ck_i == cs, axis=1))[0]
            if idx_result.size == 0:
                zs_dup = zs
            elif idx_result.size == 1:
                zs_dup = [zs_i for ii, zs_i in enumerate(zs) if ii != idx_result]
            elif idx_result.size > 1: #strange
                raise ValueError('ck match cs twice. (strange)')
            # Precompute IS parameters for the dup version if needed
            _is_dup_available = (idx_result.size == 0)
            # When no duplicate, zs_dup == zs and the IS parameters
            # from the first site (_A_is1, _mu_is1) apply directly.
            # When duplicate exists (idx_result.size == 1), dimensions
            # differ so we fall back to the original integration domain.

            def _fK_Xk(xk):
                xk_origin_shape = xk.shape
                xk = xk.flatten()
                zs_limits = __get_zs_integration_limits(zs_dup)
                if options['integration method'] == 'qmc':
                    if _is_dup_available:
                        # --- IS: absorb f(xs|xh) into sampling measure ---
                        # Integrand becomes f(xs|xk,xh)/f(xs|xh) * fS(xs)
                        # over [0,1]^ns_eff (IS with PCA-reduced f(xs|xh)).
                        def __qmc_int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup(u_array):
                            u_c = np.clip(u_array, 1e-6, 1.0 - 1e-6)
                            z = np.sqrt(2.0) * _erfinv_1(2.0 * u_c - 1.0)
                            x_array = (_A_is1.dot(z.T) + _mu_is1.reshape(-1, 1)).T
                            G_Xs_gvn_Xkh_dup_i = np.hstack(
                                [fi(x_array) for fi in fG_Xs_gvn_Xkh_dup[i](xk)])
                            fS_Xs_dup_i = fS_Xs_dup[i](x_array)
                            # Evaluate f(xs|xh) for IS correction
                            fG_Xs_gvn_Xh_val = fG_Xs_gvn_Xh(x_array)
                            # Ratio = f(xs|xk,xh) * fS(xs) / f(xs|xh)
                            return G_Xs_gvn_Xkh_dup_i * fS_Xs_dup_i / (fG_Xs_gvn_Xh_val + 1e-300)
                        xmin = np.zeros(_ns_qmc1)
                        xmax = np.ones(_ns_qmc1)
                    else:
                        # Dup case: dimensions differ, use original integration domain
                        def __qmc_int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup(x_array):
                            G_Xs_gvn_Xkh_dup_i =\
                                np.hstack(
                                    [fi(x_array) for fi in fG_Xs_gvn_Xkh_dup[i](xk)])
                            fS_Xs_dup_i = fS_Xs_dup[i](x_array)
                            return G_Xs_gvn_Xkh_dup_i * fS_Xs_dup_i
                        xmin = zs_limits[:,0].copy()
                        xmax = zs_limits[:,1].copy()

                elif options['integration method'] == 'qmc_F':
                    def __qmc_int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup(Fx_array):
                        G_Xs_gvn_Xkh_dup_i =\
                            np.hstack(
                                [fi(fS_Xs_dup[i](Fx_array)) for fi in fG_Xs_gvn_Xkh_dup[i](xk)])
                        return G_Xs_gvn_Xkh_dup_i
                    xmin = np.zeros(zs_limits[:,0].shape)
                    xmax = np.ones(zs_limits[:,1].shape)

                elif options['integration method'] == 'qmc_T':
                    def __qmc_int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup(Fx_array):
                        G_Xs_gvn_Xkh_dup_i =\
                            [fi(Fx_array) for fi in fG_Xs_gvn_Xkh_dup[i](xk)]
                        fS_Xs_dup_i =\
                            np.hstack(
                                [fS_Xs_dup[i](xs_array_i) for xs_array_i in G_Xs_gvn_Xkh_dup_i] )
                        return fS_Xs_dup_i
                    xmin = np.zeros(zs_limits[:,0].shape)
                    xmax = np.ones(zs_limits[:,1].shape)

                int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup , e, info = qmc(
                    __qmc_int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup,
                    xmin, xmax,
                    abserr=options[1,0],
                    relerr=options[3,0],
                    maxeval=int(options[2,0]),
                    pow2min=10,
                    showinfo=options['qmc_showinfo']
                    )
                int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup =\
                    int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup.reshape(xk_origin_shape)
                xk = xk.reshape(xk_origin_shape)
                if options['ck pdf debug']:
                    print('fG_Xkh[i](xk):', fG_Xkh[i](xk))
                    print('fSk_Xk[i](xk):', fSk_Xk[i](xk))
                    print('int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup:', int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup)
                    print('fG_Xh:', fG_Xh)
                    print('int_fG_Xs_gvn_Xh__fS_Xs:', int_fG_Xs_gvn_Xh__fS_Xs)
                    
                return (fG_Xkh[i](xk) * fSk_Xk[i](xk)
                    * int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup
                    / fG_Xh / int_fG_Xs_gvn_Xh__fS_Xs)
            return _fK_Xk

        cov_hs_range = range(nk, nk+nh+ns)
        
        fG_Xkh = [] # a list contains each ck's fG_Xkh
        fSk_Xk = [] # a list contains each ck's fSk_Xk
        fS_Xs_dup = []
        fG_Xs_gvn_Xkh_dup = [] # a list contains each ck's fG_Xs_gvn_Xkh_dup
        fK_Xk = []
        for i in range(nk):
            #get x/mean/cov_all_split at each ck point
            x_all_split_each_ck =\
                [x_all_split[0][i:i+1,:]] + x_all_split[1:]
            mean_all_split_each_ck =\
                [mean_all_split[0][i:i+1,:]] + mean_all_split[1:]
            covmat_each_ck =\
                covmat[np.ix_(
                    [i]+list(cov_hs_range),
                    [i]+list(cov_hs_range)
                    )]
            cov_all_split_each_ck = np.vsplit(
                covmat_each_ck, [1, 1+nh, 1+nh+ns]
                )[:-1] #exclude final empty array
            cov_all_split_each_ck = \
                [np.hsplit(c, [1, 1+nh, 1+nh+ns][:-1])\
                for c in cov_all_split_each_ck]

            # get fG_Xkh each ck part
            fG_Xkh_ = _get_multivariate_normal_pdf(
                x_all_split_each_ck,
                mean_all_split_each_ck,
                cov_all_split_each_ck, 'kh')
            fG_Xkh.append(
                __get_fG_Xkh_each_ck(fG_Xkh_))

            # get fSk_Xk
            idx_result = np.where(np.all(ck[i] == cs, axis=1))[0]
            if idx_result.size == 0:
                fSk_Xk.append(__get_fSk_Xk_each_ck(None))
            elif idx_result.size == 1:
                fSk_Xk.append(__get_fSk_Xk_each_ck(zs[idx_result[0]]))
            elif idx_result.size > 1: #strange
                raise ValueError('ck match cs twice. (strange)')

            # get fS_Xs_dup
            fS_Xs_dup.append(__get_fS_Xs_dup(ck[i], cs, zs))

            # get fG_Xs_gvn_Xkh
            fG_Xs_gvn_Xkh_dup.append(
                __get_fG_Xs_gvn_Xkh_each_ck(ck[i], cs, zs,
                    x_all_split_each_ck,
                    mean_all_split_each_ck,
                    cov_all_split_each_ck))

            fK_Xk.append(__get_pdf_each_ck(i))

        return np.array([fK_Xk]).reshape((-1, 1))
    else: #general knowledge is not gaussian
        pass

def _bme_posterior_moments(
    ck, ch=None, cs=None, zh=None, zs=None,
    covmodel=None, covparam=None, covmat=None,
    order=np.nan, options=None,
    general_knowledge='gaussian',
    #  specific_knowledge='unknown',
    pdfk=None, pdfh=None, pdfs=None, hk_k=None, hk_h=None, hk_s=None,
    gui_args=None, ck_cov_output=False):

    '''
        no neighbour considered, so spatial-temporal range
        should be transform first (no dmax support).

        covmat:
            covariance matrix, a np 2d array
            with shape (nk+nh+ns) by (nk+nh+ns)
        if covmat provieded, covmodel and covparam are simply skipped.
    '''

    if general_knowledge == 'gaussian':
        if zs:
            # --- Fast Gaussian Approximation ---
            # When enabled, convert non-Gaussian soft PDFs (histogram/linear)
            # to Gaussian (mean, variance) so the fast analytical path is used
            # instead of the expensive QMC integration.
            _use_gauss_approx = (
                options is not None and options['soft_approx_gaussian'])

            # --- Auto-Gaussian fallback for very high dimensions ---
            # When the number of non-Gaussian soft data in the
            # neighbourhood exceeds a threshold, even importance-sampling
            # QMC can struggle because the integrand fs(xs) in
            # high-dimensional space still requires many samples.
            # In that case, automatically approximate non-Gaussian
            # soft PDFs as Gaussian and use the fast analytical path.
            _NS_AUTO_GAUSS = 20  # threshold for auto-approximation
            if not _use_gauss_approx:
                _ns_nongauss = sum(
                    1 for zsi in zs
                    if get_standard_soft_pdf_type(zsi[0]) != 10)
                if _ns_nongauss > _NS_AUTO_GAUSS:
                    import warnings
                    warnings.warn(
                        "ns_nongaussian={} exceeds auto-Gaussian threshold "
                        "({}). Approximating non-Gaussian soft PDFs as "
                        "Gaussian for numerical stability.".format(
                            _ns_nongauss, _NS_AUTO_GAUSS))
                    _use_gauss_approx = True

            if _use_gauss_approx:
                zs_converted = []
                for zsi in zs:
                    pdftype = get_standard_soft_pdf_type(zsi[0])
                    if pdftype in [1, 2]:  # histogram or linear -> convert
                        zs_gau_m, zs_gau_v = proba2stat(
                            zsi[0],
                            np.atleast_2d(zsi[1]),
                            np.atleast_2d(zsi[2]),
                            np.atleast_2d(zsi[3])
                        )
                        zs_converted.append(
                            (10, float(np.asarray(zs_gau_m).flat[0]),
                             float(np.asarray(zs_gau_v).flat[0])))
                    else:
                        zs_converted.append(zsi)
                zs = zs_converted

            all_zs_type = np.array(
                [get_standard_soft_pdf_type(zsi[0]) for zsi in zs]
                )
            if (all_zs_type==10).all(): # all soft type are gaussian
                # --- Covariance-based near-duplicate detection ---
                # Use row-correlation of the covariance matrix (unit-free)
                # instead of exact coordinate equality.
                nk_g = ck.shape[0]
                nh_g = ch.shape[0] if ch is not None else 0
                ns_g = cs.shape[0] if cs is not None else 0
                if covmat is not None:
                    _covmat_g = covmat
                else:
                    _cov_tmp_g = _get_cov_all_split(
                        ck, ch, cs, covmodel, covparam)
                    _rows_g = [np.hstack([b for b in row if b is not None])
                               for row in _cov_tmp_g
                               if any(b is not None for b in row)]
                    _covmat_g = np.vstack(_rows_g)
                _ck_dup_g, _cs_dup_g, _rank_def_g = \
                    _find_ck_cs_near_duplicates(
                        _covmat_g, nk_g, nh_g, ns_g)

                # Also check exact equality as before (catches the exact case
                # even if corrcoef has numerical noise)
                _exact_dup = np.where((ck==cs[:,None]).all(-1))[1]
                dup_index = np.unique(np.concatenate(
                    [_ck_dup_g, _exact_dup]))

                if dup_index.size > 0: #has duplicated point
                    if not ck_cov_output:
                        mask = np.ones(ck.shape[0], dtype=bool)
                        mask[dup_index] = False
                        mvs = np.empty((ck.shape[0],3))
                        mvs[:] = np.nan
                        ck_dup = ck[dup_index, :]
                        ck_no_dup = ck[mask]
                        mvs[dup_index, :] =\
                            _bme_proba_gaussian_dup(
                                ck_dup, ch, cs, zh, zs,
                                covmodel, covparam, covmat, order, options)
                        if ck_no_dup.size > 0:
                            if covmat is not None:
                                _keep_g = np.concatenate([
                                    np.where(mask)[0],
                                    np.arange(nk_g, nk_g + nh_g),
                                    np.arange(nk_g + nh_g,
                                              nk_g + nh_g + ns_g)])
                                covmat_no_dup = covmat[
                                    np.ix_(_keep_g, _keep_g)]
                            else:
                                covmat_no_dup = None
                            mvs[mask, :] =\
                                _bme_proba_gaussian(
                                    ck_no_dup, ch, cs, zh, zs,
                                    covmodel, covparam, covmat_no_dup,
                                    order, options)
                        return mvs
                    else:
                        raise ValueError('We can not do ck_cov_output with duplicated point.')
                else:
                    if not ck_cov_output:
                        mvs = _bme_proba_gaussian(
                            ck, ch, cs, zh, zs,
                            covmodel, covparam, covmat, order, options)
                        return mvs
                    else:
                        mvs, ckcov = _bme_proba_gaussian(
                            ck, ch, cs, zh, zs,
                            covmodel, covparam, covmat, order, options, ck_cov_output)
                        return mvs, ckcov
            else: # has non-gaussian
                nk = ck.shape[0]
                nh = ch.shape[0] if ch is not None else 0
                ns = cs.shape[0] if cs is not None else 0

                # ============================================================
                # COVARIANCE-BASED NEAR-DUPLICATE DETECTION (unit-free)
                # ============================================================
                # When ck ≈ cs_i, the covariance matrix loses rank because
                # their rows become nearly identical. We detect this via
                # np.corrcoef (Pearson row-correlation), which is dimensionless.
                # Near-duplicate ck's are routed to the analytical Gaussian
                # dup path; the remaining ck's proceed with QMC IS.
                # ============================================================
                if covmat is not None:
                    _covmat_for_check = covmat
                else:
                    _cov_tmp = _get_cov_all_split(
                        ck, ch, cs, covmodel, covparam)
                    _rows = [np.hstack([b for b in row if b is not None])
                             for row in _cov_tmp
                             if any(b is not None for b in row)]
                    _covmat_for_check = np.vstack(_rows)

                _ck_dup_idx, _cs_dup_partner, _is_rank_def = \
                    _find_ck_cs_near_duplicates(
                        _covmat_for_check, nk, nh, ns)

                if _ck_dup_idx.size > 0:
                    import warnings
                    warnings.warn(
                        "[STARBME] Covariance-based near-duplicate detection: "
                        "{} ck point(s) nearly co-located with cs. "
                        "Routing to analytical dup path.".format(
                            _ck_dup_idx.size))

                    # Convert non-Gaussian soft data to Gaussian for the
                    # analytical dup path
                    zs_gauss_dup = []
                    for zsi in zs:
                        pdftype = get_standard_soft_pdf_type(zsi[0])
                        if pdftype == 10:
                            zs_gauss_dup.append(zsi)
                        else:
                            zs_gau_m, zs_gau_v = proba2stat(
                                zsi[0],
                                np.atleast_2d(zsi[1]),
                                np.atleast_2d(zsi[2]),
                                np.atleast_2d(zsi[3]))
                            zs_gauss_dup.append(
                                (10, float(np.asarray(zs_gau_m).flat[0]),
                                 float(np.asarray(zs_gau_v).flat[0])))

                    _nondup_mask = np.ones(nk, dtype=bool)
                    _nondup_mask[_ck_dup_idx] = False

                    mvs = np.empty((nk, 4))
                    mvs[:] = np.nan
                    mvs[:, 3] = 0.0  # default delta_mu=0

                    # --- Dup ck's: analytical Gaussian dup path (delta_mu=0) ---
                    mvs[_ck_dup_idx, :3] = _bme_proba_gaussian_dup(
                        ck[_ck_dup_idx], ch, cs, zh, zs_gauss_dup,
                        covmodel, covparam, covmat, order, options)

                    # --- Non-dup ck's: QMC IS (recursive call) ---
                    if _nondup_mask.any():
                        ck_nondup = ck[_nondup_mask]
                        if covmat is not None:
                            # Rebuild covmat without the dup ck rows/cols
                            _keep = np.concatenate([
                                np.where(_nondup_mask)[0],           # ck
                                np.arange(nk, nk + nh),             # ch
                                np.arange(nk + nh, nk + nh + ns)])  # cs
                            covmat_nondup = covmat[np.ix_(_keep, _keep)]
                        else:
                            covmat_nondup = None
                        mvs_nondup = _bme_posterior_moments(
                            ck_nondup, ch, cs, zh, zs,
                            covmodel, covparam, covmat_nondup,
                            order, options, general_knowledge,
                            pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
                            gui_args, ck_cov_output=False)
                        # mvs_nondup may be 3-col (Gaussian) or 4-col (QMC)
                        if mvs_nondup.shape[1] == 4:
                            mvs[_nondup_mask] = mvs_nondup
                        else:
                            mvs[_nondup_mask, :3] = mvs_nondup

                    if not ck_cov_output:
                        return mvs
                    else:
                        # For ck_cov_output, fall back to Gaussian approx
                        zs_gau_all = []
                        for zsi in zs:
                            zs_gau_m, zs_gau_v = proba2stat(
                                zsi[0],
                                np.atleast_2d(zsi[1]),
                                np.atleast_2d(zsi[2]),
                                np.atleast_2d(zsi[3]))
                            zs_gau_all.append((10, zs_gau_m, zs_gau_v))
                        _, ckcov = _bme_proba_gaussian(
                            ck, ch, cs, zh, zs_gau_all,
                            covmodel, covparam, covmat,
                            order, options, ck_cov_output=True)
                        return mvs, ckcov

                # If rank-deficient but no ck-cs dups, regularise covmat
                if _is_rank_def and covmat is not None:
                    _eigvals_cm = np.linalg.eigvalsh(covmat)
                    _eps_cm = max(abs(min(_eigvals_cm.min(), 0)) * 2, 1e-12)
                    covmat = covmat + np.eye(covmat.shape[0]) * _eps_cm

                x_all_split = _get_x_all_split(nk, zh, zs)
                mean_all_split = _get_mean_all_split(x_all_split, order)
                if covmat is not None:
                    cov_all_split = np.vsplit(
                        covmat, [nk, nk+nh, nk+nh+ns]
                        )[:-1] #exclude final empty array
                    cov_all_split = \
                        [np.hsplit(c, [nk, nk+nh, nk+nh+ns][:-1])\
                        for c in cov_all_split]
                else:
                    cov_all_split = _get_cov_all_split(
                        ck, ch, cs, covmodel, covparam)

                # --- Compute conditional Gaussian f(xs|xh) ---
                # Used for (a) importance-sampling transform, and
                # (b) fallback clipped-box integration when ns == 1.
                m_s_given_h = _get_mean_a_given_b(
                    x_all_split, mean_all_split,
                    cov_all_split, 's', 'h')
                cov_s_given_h = _get_sigma_a_given_b(
                    cov_all_split, 's', 'h')
                # --- FIX B: Eigenvalue floor for cov_s_given_h ---
                # Not just fixing negatives — near-zero eigenvalues make
                # pinv(cov_s_given_h) huge, so the Fused-Gaussian proposal
                # is dominated by the prior and ignores the soft data.
                # A proportional floor keeps the proposal balanced.
                _eigvals = np.linalg.eigvalsh(cov_s_given_h)
                _min_eigval_floor = max(_eigvals.max() * 1e-8, 1e-12)
                if np.any(_eigvals < _min_eigval_floor):
                    _eps = _min_eigval_floor - min(_eigvals.min(), 0)
                    cov_s_given_h += np.eye(cov_s_given_h.shape[0]) * _eps

                # -------------------------------------------------------
                # Pre-QMC Marginal-Overlap Screening
                # -------------------------------------------------------
                # For each type-2 (uniform, 2-point) soft datum, compute
                # the 1-D marginal overlap probability:
                #   P_i = P(xs_i in [lo_i, hi_i] | xh)
                #       = Phi((hi-m_cond)/sigma_cond)
                #         - Phi((lo-m_cond)/sigma_cond)
                # When P_i < min_overlap_prob AND m_cond is OUTSIDE
                # [lo_i, hi_i], the interval is expanded so that m_cond
                # lies at least 1*sigma_cond inside the new boundary.
                # This is applied before QMC so the Fused-Gaussian IS
                # proposal has good support inside the (inflated) interval,
                # dramatically improving Mon_NC SNR.
                #
                # This pre-screening is independent of (and complementary
                # to) the outlier check that runs in _bme_proba_gaussian
                # when QMC falls back to Gaussian.
                # -------------------------------------------------------
                _outlier_mode_qmc = options.get('outlier_handling', 'inflate')
                _min_olap_prob    = options.get('min_overlap_prob', 0.10)
                _margin_qmc       = options.get('outlier_inflate_margin', 0.10)

                # Pre-check whether Prior IS will be used (all type-2 uniform
                # intervals with nl=2).  Needed by the screening below to
                # choose a stronger inflation criterion that guarantees ≥95%
                # per-dimension acceptance from the conditional prior.
                _use_prior_is_check = (
                    ns > 0
                    and all(
                        get_standard_soft_pdf_type(zsi[0]) == 2
                        and int(np.asarray(zsi[1]).flat[0]) == 2
                        for zsi in zs
                    )
                )

                if _outlier_mode_qmc == 'inflate' and ns > 0:
                    from scipy.stats import norm as _snorm_qmc
                    _zs_modified  = list(zs)       # shallow copy
                    _diag_csg     = np.maximum(np.diag(cov_s_given_h), 0.0)
                    _std_csg      = np.sqrt(_diag_csg)
                    _m_sg_ravel   = np.asarray(m_s_given_h).ravel()

                    for _si_q in range(ns):
                        _zsi_q  = _zs_modified[_si_q]
                        _ptyp_q = get_standard_soft_pdf_type(_zsi_q[0])
                        if _ptyp_q != 2:          # only uniform intervals
                            continue
                        _nl_q = int(np.asarray(_zsi_q[1]).flat[0])
                        if _nl_q != 2:            # only 2-point (simple) uniform
                            continue

                        _lo_q = float(np.asarray(_zsi_q[2]).ravel()[0])
                        _hi_q = float(np.asarray(_zsi_q[2]).ravel()[1])
                        _m_q  = float(_m_sg_ravel[_si_q])
                        _s_q  = max(float(_std_csg[_si_q]), 1e-10)

                        # Marginal acceptance probability under the
                        # conditional prior N(m_cond, sigma_cond^2)
                        _p_olap = (
                            _snorm_qmc.cdf((_hi_q - _m_q) / _s_q)
                            - _snorm_qmc.cdf((_lo_q - _m_q) / _s_q)
                        )

                        if _use_prior_is_check:
                            # PRIOR IS mode: inflate ALL intervals to guarantee
                            # ≥95% per-dimension acceptance from the conditional
                            # prior.  For ns=8 this gives joint P≈66%; for ns=12
                            # joint P≈54%.  Use expand-only so the original
                            # interval is always a subset of the inflated one.
                            # N_sigma=2.0 → 95.4% per-dimension coverage.
                            _target_acc = 0.954  # 2-sigma coverage
                            if _p_olap >= _target_acc:
                                continue   # already satisfies target
                            _n_sig = 2.0
                            _new_lo = min(_lo_q, _m_q - _n_sig * _s_q)
                            _new_hi = max(_hi_q, _m_q + _n_sig * _s_q)
                        else:
                            # Standard mode: inflate only outliers (m_cond
                            # outside [lo, hi] with low overlap probability).
                            if _lo_q <= _m_q <= _hi_q:
                                continue
                            if _p_olap >= _min_olap_prob:
                                continue  # sufficient overlap, keep as-is

                            # Expand: extend the interval toward m_cond so
                            # that m_cond lies 1*sigma_cond inside new edge.
                            if _m_q < _lo_q:
                                _new_lo = _m_q - _s_q
                                _new_hi = _hi_q
                            else:  # m_q > hi_q
                                _new_lo = _lo_q
                                _new_hi = _m_q + _s_q

                            # Apply fractional safety margin
                            _new_w   = (_new_hi - _new_lo) * (1.0 + _margin_qmc)
                            _new_ctr = (_new_lo + _new_hi) / 2.0
                            _new_lo  = _new_ctr - _new_w / 2.0
                            _new_hi  = _new_ctr + _new_w / 2.0

                        _new_limi = np.atleast_2d(
                            np.array([_new_lo, _new_hi], dtype=float))
                        _new_pd   = np.atleast_2d(
                            np.array([1.0 / (_new_hi - _new_lo),
                                      1.0 / (_new_hi - _new_lo)],
                                     dtype=float))
                        _zs_modified[_si_q] = (
                            _zsi_q[0], _zsi_q[1], _new_limi, _new_pd)

                    zs = _zs_modified   # use inflated zs for QMC

                fs = _get_fs(zs)

                #split hard and soft of m_k_gvn_hs
                m_k = _get_mean(mean_all_split, 'k')
                # --- FIX A: Regularise Σ_{hs,hs} before inversion ---
                # When data locations are close together the joint
                # covariance matrix Σ_{hs,hs} becomes ill-conditioned.
                # An ill-conditioned pinv amplifies noise in xs,
                # making m_k|hs swing wildly (key cause of low/NaN
                # estimates even for small ns).
                _sigma_hs_hs = _get_sigma(cov_all_split, 'hs', 'hs')
                _cond_hs = np.linalg.cond(_sigma_hs_hs)
                if _cond_hs > 1e10:
                    _eigvals_hs = np.linalg.eigvalsh(_sigma_hs_hs)
                    _floor_hs = max(_eigvals_hs.max() * 1e-10, 1e-12)
                    _sigma_hs_hs = (
                        _sigma_hs_hs
                        + np.eye(_sigma_hs_hs.shape[0]) * _floor_hs
                    )
                _inv_sigma_hs_hs = _robust_pinv(_sigma_hs_hs)
                Bm_sigma_inv_multi = (
                    _get_sigma(cov_all_split, 'k', 'hs').dot(
                        _inv_sigma_hs_hs)
                    )
                m_hs = _get_mean(mean_all_split, 'hs')
                m_k_gvn_hs_part = m_k - Bm_sigma_inv_multi.dot(m_hs)
                x_hs = _get_x(x_all_split, 'hs') #put true Xs later

                sigma_k_given_hs = _get_sigma_a_given_b(
                    cov_all_split, 'k', 'hs')
                diag_sigma_k_given_hs =\
                    np.diag(sigma_k_given_hs)

                # ============================================================
                # IMPORTANCE-SAMPLING QMC INTEGRATION (Fused Gaussian Proposal)
                # ============================================================
                # The integral is I = ∫ g(xs) · f(xs|xh) · fs(xs) dxs
                #
                # IMPROVED PROPOSAL ("Fused Gaussian"):
                # We approximate fs(xs) as N(μ_soft, Σ_soft) and construct
                #   q(xs) ∝ f(xs|xh) · N(μ_soft, Σ_soft)  =  N(μ_IS, Σ_IS)
                # where:
                #   Σ_IS^{-1} = Σ_{s|h}^{-1} + Σ_soft^{-1}
                #   μ_IS      = Σ_IS (Σ_{s|h}^{-1} μ_{s|h} + Σ_soft^{-1} μ_soft)
                #
                # KEY IDENTITY:  f_prior(x)/q(x) = Z_fused / N_soft_approx(x)
                # where Z_fused is a constant that cancels in Mon/Mon_NC.
                # So the IS weight simplifies to:
                #   w(x) = fs(x) / N_soft_approx(x)
                # This avoids ALL multivariate logpdf computations.
                # ============================================================
                from scipy.special import erfinv as _erfinv

                # 1. Approximate each soft datum as Gaussian (mean, variance)
                #    for the Fused-Gaussian IS proposal.
                #
                #    For simple 2-point uniform PDFs [lo, hi] we use a
                #    *support-matched* (tight) variance  σ² = ((hi-lo)/6)²
                #    instead of the moment-matched  σ² = (hi-lo)²/12.
                #    The tight variance ensures [µ ± 3σ] = [lo, hi], so
                #    ~99.7% of IS samples per dimension land inside the
                #    interval, dramatically improving the SNR of Mon_NC.
                #    The IS weight  f_s(x) / N_tight(x)  is well-behaved
                #    and nearly constant inside the interval; the 3-fold
                #    reduction in σ_soft propagates automatically through
                #    _cov_IS and the weight denominator _sigma_soft below.
                _mu_soft_list = []
                _var_soft_list = []
                for zsi in zs:
                    _pdftype_is = get_standard_soft_pdf_type(zsi[0])
                    if _pdftype_is == 10:  # Gaussian
                        _mu_soft_list.append(float(np.asarray(zsi[1]).flat[0]))
                        _var_soft_list.append(float(np.asarray(zsi[2]).flat[0]))
                    elif _pdftype_is == 2:  # piecewise-linear PDF
                        _nl_is = int(np.asarray(zsi[1]).flat[0])
                        _limi_is = np.asarray(zsi[2]).ravel()
                        if _nl_is == 2:
                            # Simple uniform [lo, hi]: use tight IS variance
                            # so that ±3σ spans the entire interval.
                            _lo_is = float(_limi_is[0])
                            _hi_is = float(_limi_is[1])
                            _mu_soft_list.append((_lo_is + _hi_is) / 2.0)
                            _var_soft_list.append(
                                ((_hi_is - _lo_is) / 6.0) ** 2)
                        else:
                            # General piecewise-linear: fall back to moments
                            _m, _v = proba2stat(
                                zsi[0], np.atleast_2d(zsi[1]),
                                np.atleast_2d(zsi[2]), np.atleast_2d(zsi[3]))
                            _mu_soft_list.append(float(np.asarray(_m).flat[0]))
                            _var_soft_list.append(float(np.asarray(_v).flat[0]))
                    else:
                        _m, _v = proba2stat(
                            zsi[0], np.atleast_2d(zsi[1]),
                            np.atleast_2d(zsi[2]), np.atleast_2d(zsi[3]))
                        _mu_soft_list.append(float(np.asarray(_m).flat[0]))
                        _var_soft_list.append(float(np.asarray(_v).flat[0]))

                _mu_soft = np.array(_mu_soft_list)          # (ns,)
                _var_soft = np.maximum(np.array(_var_soft_list), 1e-14)  # (ns,)
                _sigma_soft = np.sqrt(_var_soft)             # (ns,)

                # 2. Calculate Fused Gaussian Parameters (μ_IS, Σ_IS)
                _prec_prior = _robust_pinv(cov_s_given_h)
                _prec_soft = np.diag(1.0 / _var_soft)

                _cov_IS = _robust_pinv(_prec_prior + _prec_soft)

                # Ensure _cov_IS is PSD
                _eigvals_is = np.linalg.eigvalsh(_cov_IS)
                if np.any(_eigvals_is < 0):
                    _eps_is = max(abs(_eigvals_is.min()) * 2, 1e-12)
                    _cov_IS += np.eye(_cov_IS.shape[0]) * _eps_is

                _term_prior = _prec_prior.dot(m_s_given_h)
                _term_soft = (_mu_soft / _var_soft).reshape(-1, 1)
                _mu_IS = _cov_IS.dot(_term_prior + _term_soft)

                # 3. SVD factorisation of Σ_IS (for sampling)
                _u_svd, _s_svd, _vh_svd = _robust_svd(_cov_IS)
                _s_svd = np.maximum(_s_svd, 1e-14)
                _A_is_full = _vh_svd.T * np.sqrt(_s_svd)
                _mu_is_ravel = _mu_IS.ravel()

                # --------------------------------------------------------
                # PCA DIMENSION REDUCTION (on Σ_IS)
                # --------------------------------------------------------
                _pca_thr = options.get('pca_integration_threshold', 0.999)
                _total_var = _s_svd.sum()
                if _total_var > 0 and _pca_thr < 1.0:
                    _cumvar = np.cumsum(_s_svd) / _total_var
                    _ns_eff = int(np.searchsorted(_cumvar, _pca_thr) + 1)
                    _ns_eff = max(1, min(_ns_eff, ns))
                else:
                    _ns_eff = ns

                if _ns_eff < ns:
                    _A_is = _A_is_full[:, :_ns_eff]
                else:
                    _A_is = _A_is_full

                _ns_qmc = _ns_eff

                # --------------------------------------------------------
                # PRIOR IS OVERRIDE (all-uniform-interval soft data)
                # --------------------------------------------------------
                # The Fused Gaussian IS proposal creates per-dimension IS
                # weights of the form  w_i ∝ exp(+0.5·z_i²).  These have
                # *infinite* variance under N(0,1) (the χ² MGF diverges at
                # t=0.5), and the joint weight variance grows as ~5.65^ns.
                #   ns=8  → joint Var ≈ 10^6  → ESS ≈ 0.03 (barely works)
                #   ns=12 → joint Var ≈ 10^9  → ESS ≈ 3e-5 (always fails)
                # This explains the dramatic increase in QMC fallbacks when
                # nsmax is raised from 8 to 12.
                #
                # FIX: for all-uniform-interval (type-2 nl=2) soft data,
                # switch to "Prior IS" — sample from the conditional prior
                # N(μ_{s|h}, Σ_{s|h}) and use *binary* IS weights:
                #   w = 1  if all xs_i ∈ [lo_i, hi_i]   (accepted)
                #   w = 0  otherwise                      (rejected)
                # The constant factor 1/∏(hi-lo) cancels in Mon/Mon_NC.
                # With the inflate strategy, the acceptance rate per
                # dimension is ~95%, giving joint acceptance:
                #   ns=8  → ~66%   ESS ≈ 21 600
                #   ns=12 → ~54%   ESS ≈ 17 700
                # — far superior and robust regardless of nsmax.
                _use_prior_is = (
                    len(zs) > 0
                    and all(
                        get_standard_soft_pdf_type(zsi[0]) == 2
                        and int(np.asarray(zsi[1]).flat[0]) == 2
                        for zsi in zs
                    )
                )
                if _use_prior_is:
                    # Re-factorize the conditional prior for sampling.
                    # No PCA truncation needed: the binomial acceptance
                    # probability has negligible Monte Carlo error even
                    # for the full ns-dimensional sampling.
                    _u_pr, _s_pr, _vh_pr = _robust_svd(cov_s_given_h)
                    _s_pr = np.maximum(_s_pr, 1e-14)
                    _A_is = _vh_pr.T * np.sqrt(_s_pr)   # (ns, ns)
                    _mu_is_ravel = np.asarray(m_s_given_h).ravel()
                    _ns_qmc = ns   # full ns dims; ESS is sufficient without PCA

                # Non-negativity inside the QMC integrand is DISABLED.
                # The full truncated-normal mean  μ_T = μ + σλ  inflates
                # near-zero estimates to ~0.8σ ("yellow floor" artifact)
                # in data-sparse regions where μ ≈ 0 but σ is large.
                # Instead, _apply_nonneg_truncation handles nonneg
                # post-hoc with zero-clamping for mean + truncated-normal
                # variance, avoiding the constant-background problem.
                _nonneg_in_integrand = False

                # Pre-compute σ_{k|hs} per estimation point (for truncation)
                if _nonneg_in_integrand:
                    _sigma_khs_vec = np.sqrt(
                        np.maximum(diag_sigma_k_given_hs, 1e-30)
                    ).reshape(1, -1)  # (1, nk)

                def func_moments_is(u_array):
                    """Importance-sampling integrand over [0,1]^ns_qmc.

                    Two IS modes (selected by _use_prior_is):

                    FUSED GAUSSIAN IS (_use_prior_is=False, default for mixed
                    soft data):
                      Samples xs from N(μ_IS, Σ_IS), the fused Gaussian that
                      combines the conditional prior with the Gaussian approx
                      of each soft datum.  IS weight w = ∏_i f_s_i / N_soft_i
                      is computed per-component in log-space (Fix C).

                    PRIOR IS (_use_prior_is=True, for all-uniform-interval data):
                      Samples xs from the conditional prior N(μ_{s|h}, Σ_{s|h}).
                      IS weight = 1 if all xs_i ∈ [lo_i, hi_i], 0 otherwise.
                      This gives bounded, non-heavy-tailed weights and an ESS
                      of ~54% of N for ns=12 (vs ~3e-5 for Fused Gaussian).

                    When _nonneg_in_integrand is True, replaces the raw
                    conditional moments with truncated-normal moments
                    (Z_k ≥ 0) at each sample point, so the final QMC
                    integral directly gives E[Z|data, Z≥0] and
                    Var[Z|data, Z≥0].
                    """
                    nMon = 3
                    npts = u_array.shape[0]

                    # --- Transform [0,1]^ns_eff → N(μ_IS, Σ_IS) ---
                    u_clamped = np.clip(u_array, 1e-6, 1.0 - 1e-6)
                    z_std = np.sqrt(2.0) * _erfinv(2.0 * u_clamped - 1.0)
                    x_array = (_A_is.dot(z_std.T) + _mu_is_ravel.reshape(-1, 1)).T

                    if _nonneg_in_integrand:
                        # Per-point NC: 3 moment columns × nk + nk NC columns
                        res = np.empty((npts, nMon * nk + nk))
                    else:
                        # Shared NC: 3 moment columns × nk + 1 shared NC column
                        res = np.empty((npts, nMon * nk + 1))

                    x_hs_npts = np.tile(x_hs, (1, npts))
                    x_hs_npts[nh:, :] = x_array.T

                    m_k_gvn_hs_npts = (
                        m_k_gvn_hs_part + Bm_sigma_inv_multi.dot(x_hs_npts)
                    ).T

                    # --- IS weight computation (two modes) ---
                    # FUSED GAUSSIAN IS (default for mixed soft data):
                    #   log w(x) = Σ_i [ log f_{s,i}(x_i) - log N_soft_i(x_i) ]
                    #   Each ratio f_{s,i}/N_soft_i is O(1) inside the support.
                    #   BUT: for uniform intervals, log N_soft ∝ +z²/2 produces
                    #   heavy-tailed weights with variance growing as ~5.65^ns.
                    # PRIOR IS (active when _use_prior_is=True):
                    #   Samples drawn from conditional prior N(μ_{s|h}, Σ_{s|h}).
                    #   IS weight = 1 if all xs_i inside their intervals, 0 if not.
                    #   log_weight stays 0; the valid flag tracks acceptance.
                    #   The constant 1/∏(hi-lo) cancels in Mon/Mon_NC.
                    log_weight = np.zeros(npts)
                    valid = np.ones(npts, dtype=bool)
                    _log_2pi = np.log(2.0 * np.pi)

                    for _i_s, _zsi in enumerate(zs):
                        _xi = x_array[:, _i_s]  # (npts,)

                        # Evaluate 1-D soft PDF f_{s,i}(x_i)
                        _ptype = get_standard_soft_pdf_type(_zsi[0])
                        if _ptype == 2:   # linear
                            # .flat[0] handles scalar, 1-D or 2-D nl
                            _nl = int(np.asarray(_zsi[1]).flat[0])
                            _limi_1d = np.asarray(_zsi[2]).ravel()[:_nl]
                            _pd_1d   = np.asarray(_zsi[3]).ravel()[:_nl]
                            _y = np.interp(
                                _xi, _limi_1d, _pd_1d,
                                left=0.0, right=0.0)
                        elif _ptype == 1:  # histogram
                            _nl = int(np.asarray(_zsi[1]).flat[0])
                            _limi = np.asarray(_zsi[2]).ravel()
                            _pd = np.asarray(_zsi[3]).ravel()
                            _bidx = np.searchsorted(
                                _limi[:_nl], _xi, side='right') - 1
                            _y = np.where(
                                (_bidx >= 0) & (_bidx < _nl - 1),
                                _pd[np.clip(_bidx, 0, _nl - 2)], 0.0)
                        elif _ptype == 10:  # Gaussian soft
                            _zm = _zsi[1]
                            _zstd = np.sqrt(_zsi[2])
                            _y = scipy.stats.norm.pdf(
                                _xi, loc=_zm, scale=_zstd)
                        else:
                            _y = np.ones(npts)

                        # Points where f_{s,i} = 0 → total weight = 0
                        _ok = _y > 0
                        valid &= _ok

                        # For Fused Gaussian IS: accumulate log(f_s) - log(N_soft)
                        # For Prior IS (uniform intervals): weight = 1 inside
                        # all intervals (constant cancels in Mon/Mon_NC ratio),
                        # so log_weight stays 0 and only valid tracks acceptance.
                        if not _use_prior_is:
                            _d = (_xi - _mu_soft[_i_s]) / _sigma_soft[_i_s]
                            _log_n = (-0.5 * _d**2
                                      - np.log(_sigma_soft[_i_s])
                                      - 0.5 * _log_2pi)
                            log_weight[_ok] += np.log(_y[_ok]) - _log_n[_ok]
                        # valid[~_ok] is already False (set by valid &= _ok above)

                    # Clip and exponentiate
                    log_weight = np.clip(log_weight, -500, 500)
                    total_weight = np.where(
                        valid, np.exp(log_weight), 0.0
                    ).reshape(npts, 1)

                    if _nonneg_in_integrand:
                        # --------------------------------------------------
                        # TRUNCATED-NORMAL moments at each sample (Z_k ≥ 0)
                        # --------------------------------------------------
                        # For Z_k|xs,xh ~ N(μ, σ²), truncated to [0,∞):
                        #   α   = −μ/σ
                        #   Φ₊  = Φ(μ/σ) = P(Z ≥ 0)
                        #   λ   = φ(α)/Φ₊           (inverse Mills ratio)
                        #   δ   = λ(λ−α)            (variance reduction)
                        #   μ_T = μ + σλ
                        #   σ²_T = σ²(1−δ)
                        #   E[Z²|Z≥0] = σ²_T + μ_T²
                        # --------------------------------------------------
                        _alpha = -m_k_gvn_hs_npts / _sigma_khs_vec
                        _Phi_pos = scipy.stats.norm.cdf(
                            m_k_gvn_hs_npts / _sigma_khs_vec)
                        _phi_alpha = scipy.stats.norm.pdf(_alpha)

                        # Safe inverse Mills ratio (avoid div-by-zero)
                        _safe_Phi = np.maximum(_Phi_pos, 1e-12)
                        _lam = _phi_alpha / _safe_Phi
                        _delta = _lam * (_lam - _alpha)

                        _mu_trunc = (
                            m_k_gvn_hs_npts + _sigma_khs_vec * _lam)
                        _var_trunc = (
                            diag_sigma_k_given_hs.reshape(1, -1)
                            * np.maximum(1.0 - _delta, 0.0))

                        # Where almost no mass above 0, clamp to 0
                        _dead = _Phi_pos < 1e-12
                        _mu_trunc = np.where(_dead, 0.0, _mu_trunc)
                        _var_trunc = np.where(_dead, 0.0, _var_trunc)

                        # Φ₊-weighted IS weight per estimation point.
                        # The correct truncated integrals are:
                        #   E[Z|data,Z≥0] = E_q[μ_T · Φ₊ · w] / E_q[Φ₊ · w]
                        #   E[Z²|data,Z≥0] = E_q[(σ²_T+μ_T²) · Φ₊ · w] / E_q[Φ₊ · w]
                        # where Φ₊ = P(Z_k≥0 | xs, xh) depends on k.
                        _tw_nonneg = total_weight * _Phi_pos  # (npts, nk)

                        # E[Z|Z≥0] · Φ₊ · w
                        res[:, 0*nk:1*nk] = _mu_trunc * _tw_nonneg
                        # E[Z²|Z≥0] · Φ₊ · w
                        res[:, 1*nk:2*nk] = (
                            (_var_trunc + _mu_trunc**2) * _tw_nonneg)
                        # Skewness (set to 0)
                        res[:, 2*nk:3*nk] = 0.0
                        # Per-point NC: Φ₊ · w  (nk columns)
                        res[:, 3*nk:4*nk] = _tw_nonneg
                    else:
                        res[:, 0*nk:1*nk] = (
                            m_k_gvn_hs_npts * total_weight)
                        res[:, 1*nk:2*nk] = (
                            res[:, :nk] * m_k_gvn_hs_npts)
                        res[:, 2*nk:3*nk] = (
                            3 * diag_sigma_k_given_hs * res[:, :nk]
                            - 2 * res[:, nk:2*nk] * m_k_gvn_hs_npts)
                        # Shared NC (one column)
                        res[:, -1:] = total_weight
                    return res

                # QMC integration over the unit cube [0,1]^ns_eff
                xmin = np.zeros(_ns_qmc)
                xmax = np.ones(_ns_qmc)

                Mon, e, info = qmc(
                    func_moments_is, xmin, xmax,
                    abserr=options[1,0],
                    relerr=options[3,0],
                    maxeval=int(options[2,0]),
                    showinfo=options['qmc_showinfo'])

                # =============================================================
                # Helper: Gaussian fallback (used by both NC paths)
                # =============================================================
                def _gauss_fallback(reason_msg):
                    import warnings
                    warnings.warn(reason_msg)
                    zs_gauss_fb = []
                    for zsi in zs:
                        pdftype = get_standard_soft_pdf_type(zsi[0])
                        if pdftype == 10:
                            zs_gauss_fb.append(zsi)
                        else:
                            zs_gau_m, zs_gau_v = proba2stat(
                                zsi[0],
                                np.atleast_2d(zsi[1]),
                                np.atleast_2d(zsi[2]),
                                np.atleast_2d(zsi[3]))
                            zs_gauss_fb.append(
                                (10, float(np.asarray(zs_gau_m).flat[0]),
                                 float(np.asarray(zs_gau_v).flat[0])))
                    mvs_fb = _bme_proba_gaussian(
                        ck, ch, cs, zh, zs_gauss_fb,
                        covmodel, covparam, covmat,
                        order, options, ck_cov_output)
                    return mvs_fb

                # =============================================================
                # Helper: covariance output via Gaussian
                # =============================================================
                def _cov_output_from_gaussian():
                    zs_gau = []
                    for zsi in zs:
                        zs_gau_m, zs_gau_v = proba2stat(
                            zsi[0],
                            np.atleast_2d(zsi[1]),
                            np.atleast_2d(zsi[2]),
                            np.atleast_2d(zsi[3]))
                        zs_gau.append((10, zs_gau_m, zs_gau_v))
                    _mvs22, _ckcov22 = _bme_proba_gaussian(
                        ck, ch, cs, zh, zs_gau,
                        covmodel, covparam, covmat,
                        order, options, ck_cov_output)
                    return _ckcov22

                if _nonneg_in_integrand:
                    # =========================================================
                    # Nonneg path: per-point NC from Φ₊-weighted integrand
                    # Mon has 4*nk elements: 3*nk moments + nk per-point NCs
                    # =========================================================
                    Mon_NC_per_k = Mon[3*nk:4*nk]        # (nk,)
                    e_NC_per_k   = e[3*nk:4*nk]          # (nk,)

                    # Guard: fall back if worst per-point NC is bad
                    _nc_min = np.nanmin(Mon_NC_per_k)
                    _nc_max_err = np.nanmax(e_NC_per_k)
                    if (_nc_min == 0
                            or np.any(np.isnan(Mon_NC_per_k))
                            or np.any(np.isinf(Mon_NC_per_k))
                            or _nc_min < 2.0 * _nc_max_err
                            or _nc_min < 1e-100):
                        return _gauss_fallback(
                            "[STARBME] QMC nonneg per-point NC min={:.4e} "
                            "+/- {:.4e} (ns={:d}). Falling back to Gaussian "
                            "approximation.".format(
                                float(_nc_min), float(_nc_max_err), ns))

                    # Normalise moments per point
                    Mon_mom = Mon[:3*nk].reshape((3, nk)).T  # (nk, 3)
                    e_mom   = e[:3*nk].reshape((3, nk)).T    # (nk, 3)
                    for j in range(nk):
                        Mon_mom[j, :] /= Mon_NC_per_k[j]

                    M1 = Mon_mom[:, 0:1]
                    M2 = Mon_mom[:, 1:2]

                    # Per-point error propagation
                    _delta_mu = np.zeros((nk, 1))
                    for j in range(nk):
                        _nc_j = Mon_NC_per_k[j]
                        _delta_mu[j, 0] = np.sqrt(
                            (e_mom[j, 0] / _nc_j) ** 2
                            + (M1[j, 0] * e_NC_per_k[j] / _nc_j) ** 2)

                    mvs = np.empty((nk, 4))
                    mvs[:, 0:1] = M1
                    mvs[:, 1:2] = np.maximum(
                        M2 - M1**2 - _delta_mu**2, 0.0)
                    mvs[:, 2:3] = 0.0
                    mvs[:, 3:4] = -1.0  # flag: nonneg already applied

                    if not ck_cov_output:
                        return mvs
                    else:
                        return mvs, _cov_output_from_gaussian()

                else:
                    # =========================================================
                    # Standard path: shared NC (single column)
                    # =========================================================
                    Mon_NC = Mon[-1]
                    Mon_NC_err = e[-1]
                    # Guard against zero / NaN / Inf normalisation constant
                    if (Mon_NC == 0 or np.isnan(Mon_NC) or np.isinf(Mon_NC) or
                            Mon_NC < 2.0 * Mon_NC_err or Mon_NC < 1e-100):
                        return _gauss_fallback(
                            "[STARBME] QMC normalization constant Mon_NC="
                            "{:.4e} +/- {:.4e} (ns={:d}). IS integration "
                            "unstable. Falling back to Gaussian "
                            "approximation.".format(
                                float(Mon_NC), float(Mon_NC_err), ns))

                    Mon_raw = Mon.copy()
                    e_raw = e.copy()

                    Mon = Mon_raw[:-1].reshape((-1, nk)).T  # (nk, nMon)
                    Mon /= Mon_NC
                    # Add unconditional σ²_{k|hs} (law of total variance)
                    Mon[:, 1:2] += diag_sigma_k_given_hs.reshape((-1, 1))

                    M1 = Mon[:, 0:1]
                    M2 = Mon[:, 1:2]
                    M3 = Mon[:, 2:3]

                    # Per-point QMC error via error propagation
                    e_mom = e_raw[:-1].reshape((-1, nk)).T  # (nk, 3)
                    _e_num = e_mom[:, 0:1]
                    _e_den = Mon_NC_err
                    _delta_mu = np.sqrt(
                        (_e_num / Mon_NC)**2
                        + (M1 * _e_den / Mon_NC)**2)

                    mvs = np.empty((nk, 4))
                    mvs[:, 0:1] = M1
                    mvs[:, 1:2] = np.maximum(
                        M2 - M1**2 - _delta_mu**2, 0.0)
                    mvs[:, 2:3] = M3 - 3*M2*M1 + 2*M1**3
                    mvs[:, 3:4] = _delta_mu

                    if not ck_cov_output:
                        return mvs
                    else:
                        return mvs, _cov_output_from_gaussian()
        else:
            if not ck_cov_output:
                mvs = _bme_proba_gaussian(
                    ck, ch, cs, zh, zs, covmodel, covparam, covmat, order, options)
                return mvs
            else:
                mvs, ckcov = _bme_proba_gaussian(
                    ck, ch, cs, zh, zs,
                    covmodel, covparam, covmat, order, options, ck_cov_output)
                return mvs, ckcov
    else:
        raise ValueError("Now we can not consider non-gaussian GK." )

def _bme_proba_gaussian(
    ck, ch=None, cs=None, zh=None, zs=None,
    covmodel=None, covparam=None, covmat=None,
    order=np.nan, options=None, ck_cov_output=False):
    '''
    no neighbour consider, no data format transform.

    ch, cs, zh: np.2darray or None
    zs: new zs data or None, see softconverter.py for detail.
    ck_cov_output: if True, result will additionally return
        covariance between ck
    NOTE zs there should gaussian type e.g.
        zs = ((10, mean1, var1),...,(10, mean2, var2))
    '''

    nk, nh, ns = __get_khs_size(ck, ch, cs)
    x_all_split, mean_all_split, cov_all_split, covmat =\
        __get_all_split(
            ck, ch, cs, zh, zs, covmodel, covparam, covmat, order)
    
    has_remove_index, remove_index =\
        __get_covmat_remove_index(covmat)

    if has_remove_index: #need change data
        s_remove_index = remove_index-(nk+nh)
        if (s_remove_index >= 0).all():
            cs = np.delete(cs, s_remove_index, axis=0)
            zs = [zs_i for idx, zs_i in enumerate(zs) if idx not in s_remove_index]
            covmat = np.delete(covmat, nk+nh+s_remove_index, axis=0)
            covmat = np.delete(covmat, nk+nh+s_remove_index, axis=1)

            x_all_split, mean_all_split, cov_all_split, covmat =\
                __get_all_split(
                ck, ch, cs, zh, zs, covmodel, covparam, covmat, order)
        else: # need remove k or h — use covariance-based dup detection
            # The heuristic flagged a ck (or ch) index for removal, which
            # means ck is nearly co-located with cs. Detect via row
            # correlation and route those ck's through the dup path.
            _ck_dup_idx, _cs_dup_partner, _ = \
                _find_ck_cs_near_duplicates(covmat, nk, nh, ns)

            if _ck_dup_idx.size > 0:
                import warnings
                warnings.warn(
                    "[STARBME] _bme_proba_gaussian: {} ck point(s) "
                    "nearly co-located with cs (covariance-based). "
                    "Routing to dup path.".format(_ck_dup_idx.size))

                _nondup_mask = np.ones(nk, dtype=bool)
                _nondup_mask[_ck_dup_idx] = False
                mvs = np.empty((nk, 3))
                mvs[:] = np.nan

                # Near-duplicate ck's → analytical dup path
                mvs[_ck_dup_idx, :] = _bme_proba_gaussian_dup(
                    ck[_ck_dup_idx], ch, cs, zh, zs,
                    covmodel, covparam, covmat, order, options)

                # Non-duplicate ck's → standard Gaussian path
                if _nondup_mask.any():
                    ck_nd = ck[_nondup_mask]
                    if covmat is not None:
                        _keep = np.concatenate([
                            np.where(_nondup_mask)[0],
                            np.arange(nk, nk + nh),
                            np.arange(nk + nh, nk + nh + ns)])
                        covmat_nd = covmat[np.ix_(_keep, _keep)]
                    else:
                        covmat_nd = None
                    mvs[_nondup_mask, :] = _bme_proba_gaussian(
                        ck_nd, ch, cs, zh, zs,
                        covmodel, covparam, covmat_nd,
                        order, options)
                return mvs
            else:
                # Rank-deficient but no ck-cs dups (cs-cs or ch-ch dups).
                # Regularise the matrix and continue.
                _eigvals_r = np.linalg.eigvalsh(covmat)
                _eps_r = max(abs(min(_eigvals_r.min(), 0)) * 2, 1e-12)
                covmat = covmat + np.eye(covmat.shape[0]) * _eps_r
                x_all_split, mean_all_split, cov_all_split, covmat = \
                    __get_all_split(
                        ck, ch, cs, zh, zs,
                        covmodel, covparam, covmat, order)
    nk, nh, ns = __get_khs_size(ck, ch, cs)

    if ns == 0 and nh == 0:
        mvs = np.empty((ck.shape[0],3))
        mvs[:] = np.nan
        return mvs
        #raise ValueError('hard and soft data can not both without input')

    if ns == 0: # only hard data
        mean_k_given_h = _get_mean_a_given_b(
            x_all_split, mean_all_split,
            cov_all_split, sub_a='k', sub_b='h')
        sigma_k_given_h = _get_sigma_a_given_b(
            cov_all_split, sub_a='k', sub_b='h')

        skewness = np.zeros(mean_k_given_h.shape)
        mvs = np.hstack(
            (mean_k_given_h, sigma_k_given_h.diagonal().reshape((-1,1)),
            skewness)
            )
        if ck_cov_output:
            return mvs, sigma_k_given_h
        else:
            return mvs
    else: # both hard and soft data (hard data can be empty)
        mean_k = _get_mean(mean_all_split, 'k')
        
        #check outlier data
        mean_s_given_h = _get_mean_a_given_b(
            x_all_split, mean_all_split,
            cov_all_split, sub_a='s', sub_b='h')
        sigma_s_given_h = _get_sigma_a_given_b(cov_all_split, 's', 'h')
        general_mean = mean_s_given_h.ravel()
        general_var = np.diag(sigma_s_given_h)
        del sigma_s_given_h
        # Build soft_mean / soft_var correctly for ALL soft PDF types.
        # Non-Gaussian entries store (pdftype, nl, limi, probdens), so
        # zs_i[1] is nl (number of limits), NOT the mean.  We must call
        # proba2stat to obtain mean and variance for every type.
        _soft_mean_list, _soft_var_list = [], []
        for _zsi in zs:
            _pdftype = get_standard_soft_pdf_type(_zsi[0])
            if _pdftype == 10:  # Gaussian: (10, mean, var)
                _soft_mean_list.append(
                    float(np.asarray(_zsi[1]).flat[0]))
                _soft_var_list.append(
                    float(np.asarray(_zsi[2]).flat[0]))
            else:
                _sm, _sv = proba2stat(
                    _zsi[0], np.atleast_2d(_zsi[1]),
                    np.atleast_2d(_zsi[2]), np.atleast_2d(_zsi[3]))
                _soft_mean_list.append(
                    float(np.asarray(_sm).flat[0]))
                _soft_var_list.append(
                    float(np.asarray(_sv).flat[0]))
        soft_mean = np.array(_soft_mean_list)
        soft_var  = np.array(_soft_var_list)

        problem_bool = (np.abs(general_mean - soft_mean)\
            > 3*(np.sqrt(general_var) + np.sqrt(soft_var))
            )
        if problem_bool.any():
            problem_index = np.where(problem_bool)[0]
            import warnings as _w

            # Determine handling strategy from options
            _outlier_mode = 'exclude'
            _inflate_margin = 0.10
            if options is not None:
                _outlier_mode   = options.get('outlier_handling',     'exclude')
                _inflate_margin = options.get('outlier_inflate_margin', 0.10)

            if _outlier_mode == 'inflate':
                # -------------------------------------------------------
                # Strategy A – uncertainty inflation
                #
                # Each flagged soft datum is kept in the neighbourhood but
                # its uncertainty is increased until it becomes compatible
                # with the conditional prior E[Zs|Zh], plus an extra
                # fractional margin.  Two sub-cases:
                #
                #   Type-10 (Gaussian): inflate the variance directly.
                #       The datum arrives here as N(m, v) – either because
                #       it was originally Gaussian, or because it was
                #       converted from a non-Gaussian type (e.g. uniform)
                #       by _gauss_fallback.
                #       The inflated variance v_new satisfies:
                #           gap ≤ 3*(prior_std + sqrt(v_new)) * (1+margin)
                #       ⟹  sqrt(v_new) = gap/(3*(1+margin)) − prior_std
                #
                #   Type-2 with nl=2 (piecewise-linear 2-point, i.e. uniform):
                #       Widen the interval symmetrically about its centre.
                #       (This path is taken when the function is called
                #       directly with non-Gaussian zs, before _gauss_fallback.)
                #
                #   All other types: fall through to exclusion.
                # -------------------------------------------------------
                _sqrt3 = np.sqrt(3.0)
                _inflated_any = False
                _fallback_excl = []   # indices that cannot be inflated

                for _idx in problem_index:
                    _zsi = zs[_idx]
                    _pdftype_flag = get_standard_soft_pdf_type(_zsi[0])
                    _prior_mean_i = float(general_mean[_idx])
                    _prior_std_i  = float(np.sqrt(np.maximum(
                        general_var[_idx], 0.0)))
                    _gap = abs(_prior_mean_i - soft_mean[_idx])

                    if _pdftype_flag == 10:
                        # --- Gaussian soft datum: inflate variance ---
                        _sm  = float(np.asarray(_zsi[1]).flat[0])
                        _sv  = float(np.asarray(_zsi[2]).flat[0])
                        # soft_std_new must satisfy:
                        #   gap <= 3*(prior_std + soft_std_new)*(1+margin)
                        # => soft_std_new = gap/(3*(1+margin)) - prior_std
                        _soft_std_min = max(
                            _gap / (3.0 * (1.0 + _inflate_margin))
                            - _prior_std_i, 0.0)
                        _new_var = max(_soft_std_min ** 2, _sv)
                        _w.warn(
                            'Soft datum at local index {i} (within this '
                            'neighbourhood batch of ns={ns}) is '
                            'inconsistent with the conditional prior '
                            'E[Zs|Zh] (gap={gap:.2f}, prior_std={ps:.2f}). '
                            'Inflating Gaussian variance {v:.4f} -> '
                            '{nv:.4f} '
                            '(outlier_handling="inflate").'.format(
                                i=_idx, ns=len(zs),
                                gap=_gap, ps=_prior_std_i,
                                v=_sv, nv=_new_var))
                        zs[_idx] = (10, _sm, _new_var)
                        _inflated_any = True

                    elif _pdftype_flag == 2:
                        # --- Uniform-interval datum: widen interval ---
                        _nl_flag = int(np.asarray(_zsi[1]).flat[0])
                        if _nl_flag != 2:
                            _fallback_excl.append(_idx)
                            continue
                        _limi_flag = np.asarray(_zsi[2]).ravel()
                        _lo  = float(_limi_flag[0])
                        _hi  = float(_limi_flag[1])
                        _ctr = (_lo + _hi) / 2.0
                        _soft_std_min = max(
                            _gap / (3.0 * (1.0 + _inflate_margin))
                            - _prior_std_i, 0.0)
                        # For uniform: std = width/(2√3)
                        # => new_width = 2√3 * soft_std_min
                        _min_width = 2.0 * _sqrt3 * _soft_std_min
                        _new_width = max(_min_width, _hi - _lo)
                        _new_lo = _ctr - _new_width / 2.0
                        _new_hi = _ctr + _new_width / 2.0
                        _w.warn(
                            'Soft datum at local index {i} (within this '
                            'neighbourhood batch of ns={ns}) is '
                            'inconsistent with the conditional prior '
                            'E[Zs|Zh] (gap={gap:.2f}, prior_std={ps:.2f}). '
                            'Inflating interval [{lo:.1f}, {hi:.1f}] -> '
                            '[{nlo:.1f}, {nhi:.1f}] '
                            '(outlier_handling="inflate").'.format(
                                i=_idx, ns=len(zs),
                                gap=_gap, ps=_prior_std_i,
                                lo=_lo, hi=_hi,
                                nlo=_new_lo, nhi=_new_hi))
                        _new_limi = np.atleast_2d(
                            np.array([_new_lo, _new_hi], dtype=float))
                        _new_pd = np.atleast_2d(
                            np.array([1.0/_new_width, 1.0/_new_width],
                                     dtype=float))
                        zs[_idx] = (_zsi[0], _zsi[1], _new_limi, _new_pd)
                        _inflated_any = True

                    else:
                        # Unknown type: cannot inflate, fall back to exclusion
                        _fallback_excl.append(_idx)

                # Exclude anything that could not be inflated
                if _fallback_excl:
                    _fb_idx = np.array(_fallback_excl)
                    _w.warn(
                        'warning: the soft data at local index(es) {i} '
                        '(within this neighbourhood batch of ns={ns}) '
                        'cannot be inflated (unsupported PDF type) and will '
                        'be excluded from this estimation.'.format(
                            i=str(_fb_idx), ns=len(zs)))
                    cs = np.delete(cs, _fb_idx, axis=0)
                    zs = [_zs_i for _ei, _zs_i in enumerate(zs)
                          if _ei not in _fallback_excl]
                    covmat = np.delete(covmat, nk+nh+_fb_idx, axis=0)
                    covmat = np.delete(covmat, nk+nh+_fb_idx, axis=1)

                # Always rebuild covariance partition after any modification
                x_all_split, mean_all_split, cov_all_split, covmat = \
                    __get_all_split(
                        ck, ch, cs, zh, zs,
                        covmodel, covparam, covmat, order)
                nk, nh, ns = __get_khs_size(ck, ch, cs)
                mean_s_given_h = _get_mean_a_given_b(
                    x_all_split, mean_all_split,
                    cov_all_split, sub_a='s', sub_b='h')

            elif _outlier_mode == 'exclude':
                # -------------------------------------------------------
                # Original behaviour: delete flagged data entirely
                # -------------------------------------------------------
                _w.warn(
                    'warning: the soft data at local index(es) {i} '
                    '(within this neighbourhood batch of ns={ns}) '
                    'are far from the conditional prior pdf E[Zs|Zh]. '
                    'These observations will be excluded from this '
                    'estimation.'.format(
                        i=str(problem_index), ns=len(zs)))
                cs = np.delete(cs, problem_index, axis=0)
                zs = [zs_i for idx, zs_i in enumerate(zs)
                      if idx not in problem_index]
                covmat = np.delete(covmat, nk+nh+problem_index, axis=0)
                covmat = np.delete(covmat, nk+nh+problem_index, axis=1)

                x_all_split, mean_all_split, cov_all_split, covmat = \
                    __get_all_split(
                        ck, ch, cs, zh, zs,
                        covmodel, covparam, covmat, order)
                nk, nh, ns = __get_khs_size(ck, ch, cs)
                mean_s_given_h = _get_mean_a_given_b(
                    x_all_split, mean_all_split,
                    cov_all_split, sub_a='s', sub_b='h')
            # else: 'ignore' – keep all soft data as-is; no further action

        # Early-return: all soft data were excluded → pure hard-data kriging
        if ns == 0:
            mean_k_given_h_fb = _get_mean_a_given_b(
                x_all_split, mean_all_split,
                cov_all_split, sub_a='k', sub_b='h')
            sigma_k_given_h_fb = _get_sigma_a_given_b(
                cov_all_split, sub_a='k', sub_b='h')
            skew_fb = np.zeros(mean_k_given_h_fb.shape)
            mvs_ns0 = np.hstack(
                (mean_k_given_h_fb,
                 sigma_k_given_h_fb.diagonal().reshape((-1, 1)),
                 skew_fb))
            if ck_cov_output:
                return mvs_ns0, sigma_k_given_h_fb
            else:
                return mvs_ns0

        mean_hs = _get_mean(mean_all_split, 'hs')

        NC, useful_args = _get_int_fg_a_given_b_fs_s(
            x_all_split, mean_all_split,
            cov_all_split, zs,
            sub_multi = 's_h', sub_s='s'
            )

        NC = __check_normalized_constant(NC, options)
        if NC == 0 or np.isnan(NC) or np.isinf(NC):
            # --- Fallback: Gaussian approximation when analytical NC fails ---
            # This typically happens when the soft PDFs have very narrow
            # support that doesn't overlap well with the conditional Gaussian.
            # Instead of returning NaN, approximate the soft data as Gaussian
            # and use the hard+soft-Gaussian analytical path.
            import warnings
            warnings.warn(
                "[STARBME] Analytical normalization constant NC={:.4e} "
                "(nk={}, nh={}, ns={}). Falling back to Gaussian "
                "approximation of soft data for this batch.".format(
                    float(NC) if not np.isnan(NC) else float('nan'),
                    nk, nh, ns))
            zs_gauss_fb = []
            for zsi in zs:
                pdftype = get_standard_soft_pdf_type(zsi[0])
                if pdftype == 10:
                    zs_gauss_fb.append(zsi)
                else:
                    zs_gau_m, zs_gau_v = proba2stat(
                        zsi[0],
                        np.atleast_2d(zsi[1]),
                        np.atleast_2d(zsi[2]),
                        np.atleast_2d(zsi[3]))
                    zs_gauss_fb.append(
                        (10, float(np.asarray(zs_gau_m).flat[0]),
                         float(np.asarray(zs_gau_v).flat[0])))
            # Re-run with all-Gaussian soft data (analytical solution)
            x_all_split_fb, mean_all_split_fb, cov_all_split_fb, covmat_fb = \
                __get_all_split(
                    ck, ch, cs, zh, zs_gauss_fb,
                    covmodel, covparam, covmat, order)
            nk_fb, nh_fb, ns_fb = __get_khs_size(ck, ch, cs)

            # --- Part 3 fix: Add soft noise to covariance diagonal ---
            # Without this, the conditional variance Σ_k|hs treats the
            # Gaussian-approximated soft data as exact observations,
            # producing a Kriging variance that ignores soft uncertainty.
            # Adding the soft variance σ²_i to the (i,i) diagonal of
            # the soft data block makes the inversion account for noise.
            for i_s, zsi_fb in enumerate(zs_gauss_fb):
                soft_var_i = zsi_fb[2]  # variance from Gaussian approx
                if soft_var_i > 0:
                    idx = nk_fb + nh_fb + i_s
                    covmat_fb[idx, idx] += soft_var_i
            # Re-split the modified covariance matrix
            cov_all_split_fb = np.vsplit(
                covmat_fb,
                [nk_fb, nk_fb + nh_fb, nk_fb + nh_fb + ns_fb]
                )[:-1]
            cov_all_split_fb = [
                np.hsplit(c, [nk_fb, nk_fb + nh_fb, nk_fb + nh_fb + ns_fb][:-1])
                for c in cov_all_split_fb]

            mean_k_fb = _get_mean(mean_all_split_fb, 'k')
            mean_k_given_hs_fb = _get_mean_a_given_b(
                x_all_split_fb, mean_all_split_fb,
                cov_all_split_fb, sub_a='k', sub_b='hs')
            sigma_k_given_hs_fb = _get_sigma_a_given_b(
                cov_all_split_fb, sub_a='k', sub_b='hs')
            skewness_fb = np.zeros(mean_k_given_hs_fb.shape)
            mvs = np.hstack((
                mean_k_given_hs_fb,
                sigma_k_given_hs_fb.diagonal().reshape((-1, 1)),
                skewness_fb))
            if ck_cov_output:
                return mvs, sigma_k_given_hs_fb
            else:
                return mvs

        (sigma_t_prime, inv_sigma_s_given_h,
            inv_sigma_tilde_s, mean_tilde_s,
            alias_c, alias_fgfs1234, alias_mean_d, alias_sigma_d
            ) = useful_args

        # ============================================================
        # NC-FREE FORMULATION  (NC cancels algebraically in all terms)
        # ============================================================
        # Previously the code multiplied by NC and then divided by NC
        # in every formula.  For large ns the NC can underflow to 0,
        # but every ratio  (hat_x_hs / NC),  (tt / NC)  is finite.
        # We now compute the NC-free quantities directly.
        #
        #   mean_t_s  = sigma_t' · (Σ_s|h⁻¹ · m_s|h  +  R⁻¹ · μ̃_s)
        #   mean_t_hs = [ x_h ; mean_t_s ]          (NC-free)
        #   BME_mean  = cond_k_hs · mean_t_hs + (m_k - cond_k_hs · m_hs)
        #   aa_nc[ss] = sigma_t'       (the posterior soft cov, NC-free)
        #   bb_nc     = (mean_t_hs - m_hs)(mean_t_hs - m_hs)^T
        #   tt_nc     = cond_k_hs · (aa_nc + bb_nc) · cond_k_hs^T
        #   BME_var   = diag(Σ_k|hs) - BME_mean² + m_k²
        #             - 2 m_k · cond_k_hs · m_hs
        #             + 2 m_k · cond_k_hs · mean_t_hs
        #             + diag(tt_nc)
        # ============================================================
        mean_t_s = sigma_t_prime.dot(
            inv_sigma_s_given_h.dot(mean_s_given_h)
            + inv_sigma_tilde_s.dot(mean_tilde_s)
            )
        if not nh:
            mean_t_hs = mean_t_s
        else:
            x_h = _get_x(x_all_split, 'h')
            mean_t_hs = np.vstack((x_h, mean_t_s))

        sigma_k_hs = _get_sigma(
            cov_all_split, sub_a='k', sub_b='hs')
        inv_sigma_hs_hs = _get_sigma(
            cov_all_split, sub_a='hs', sub_b='hs', inv=True)
        cond_k_hs = sigma_k_hs.dot(inv_sigma_hs_hs)

        BME_mean_k_given_hs_b = mean_k - cond_k_hs.dot(mean_hs)
        BME_mean_k_given_hs = (
            cond_k_hs.dot(mean_t_hs) + BME_mean_k_given_hs_b)

        sigma_k_given_hs = _get_sigma_a_given_b(
            cov_all_split, sub_a='k', sub_b='hs')

        # NC-free variance correction terms
        n_hs = nh + ns
        aa_nc = np.zeros((n_hs, n_hs))
        aa_nc[nh:nh+ns, nh:nh+ns] = sigma_t_prime
        bb_nc = (mean_t_hs - mean_hs).dot((mean_t_hs - mean_hs).T)
        tt_nc = cond_k_hs.dot(aa_nc + bb_nc).dot(cond_k_hs.T)

        if not ck_cov_output:
            sigma_k_given_hs_diag = \
                sigma_k_given_hs.diagonal().reshape((-1, 1))
            tt_nc_diag = tt_nc.diagonal().reshape((-1, 1))
            BME_var_k_given_hs = (
                sigma_k_given_hs_diag - BME_mean_k_given_hs**2
                + mean_k**2
                - 2 * mean_k * cond_k_hs.dot(mean_hs)
                + 2 * mean_k * cond_k_hs.dot(mean_t_hs)
                + tt_nc_diag
                )
        else:
            exm = cond_k_hs.dot(mean_t_hs).dot(mean_k.T)
            emm = cond_k_hs.dot(mean_hs).dot(mean_k.T)
            BME_var_k_given_hs_cov = (
                sigma_k_given_hs
                - BME_mean_k_given_hs.dot(BME_mean_k_given_hs.T)
                + mean_k.dot(mean_k.T)
                + 2 * exm - 2 * emm
                + tt_nc
                )
            BME_var_k_given_hs = \
                BME_var_k_given_hs_cov.diagonal().reshape((-1, 1))
            
        skewness = np.zeros(BME_mean_k_given_hs.shape)
        mvs = np.hstack(
            (BME_mean_k_given_hs, BME_var_k_given_hs, skewness)
            )
        if ck_cov_output:
            return mvs, BME_var_k_given_hs_cov
        else:
            return mvs

def _bme_proba_gaussian_dup(
    ck, ch=None, cs=None, zh=None, zs=None,
    covmodel=None, covparam=None, covmat=None,
    order=np.nan, options=None, ck_cov_output=False):

    nk, nh, ns = __get_khs_size(ck, ch, cs)
    x_all_split, mean_all_split, cov_all_split, covmat =\
        __get_all_split(
            ck, ch, cs, zh, zs, covmodel, covparam, covmat, order)

    if ns == 0: #strange, should have ck duplicates with cs
        raise ValueError('no cs, strange.')
    else:
        NC, (sigma_d, inv_sigma_s_given_h,
            inv_sigma_tilde_s, mean_tilde_s,
            alias_c, alias_fgfs1234, alias_mean_d, alias_sigma_d
            ) =\
            _get_int_fg_a_given_b_fs_s(
                x_all_split, mean_all_split, cov_all_split, zs, 
                sub_multi = 's_h', sub_s='s'
                )

        # no used but give a warning   
        #NC = __check_normalized_constant(NC)
        if NC == 0:
            print('warning: NC found to be 0.')
            if options['debug']:
                # import pdb
                # pdb.set_trace()
                pass
        # --- Find ck→cs partner mapping (covariance-based + exact) ---
        # Use row-correlation of covmat to find the closest cs for each ck.
        _ck_dup_d, _cs_dup_d, _ = _find_ck_cs_near_duplicates(
            covmat, nk, nh, ns)
        # Also try exact coordinate match
        _exact_ck, _exact_cs = np.where((cs == ck[:, None]).all(-1))
        # Merge: prefer exact match, fill in from covariance-based
        _cs_for_ck = np.full(nk, -1, dtype=int)
        for ci, si in zip(_ck_dup_d, _cs_dup_d):
            _cs_for_ck[ci] = si
        for ci, si in zip(_exact_ck, _exact_cs):
            _cs_for_ck[ci] = si  # exact takes precedence

        # For any ck without a partner, find the closest cs
        # via highest row-correlation
        _unmatched = np.where(_cs_for_ck < 0)[0]
        if _unmatched.size > 0:
            _corrmat_d = np.corrcoef(covmat)
            for ui in _unmatched:
                # ck index ui, cs block starts at nk+nh
                _corr_row = np.abs(
                    _corrmat_d[ui, nk + nh: nk + nh + ns])
                _cs_for_ck[ui] = int(np.argmax(_corr_row))

        cs_dup_index = _cs_for_ck
        return np.hstack((
            alias_mean_d[cs_dup_index,:],
            np.diagonal(alias_sigma_d).reshape((-1,1))[cs_dup_index,:],
            np.zeros((nk, 1))
            ))

        # NC, (sigma_d, inv_sigma_s_given_h,
        #     inv_sigma_tilde_s, mean_tilde_s,
        #     alias_c, alias_fgfs1234, alias_mean_d, alias_sigma_d
        #     ) =\
        #     _get_int_fg_a_given_b_fs_s(
        #         x_all_split, mean_all_split, cov_all_split, zs, 
        #         sub_multi = 's_h', sub_s='s'
        #         )
        # NC = __check_normalized_constant(NC)
        
        # #get top part
        # #ns x 1
        # top_part2 = NC * alias_mean_d
        # top_part = alias_c * alias_fgfs1234 * alias_mean_d
        # import pdb
        # pdb.set_trace()
        # #get each nc_dup
        # dup_NC = np.ones(top_part.shape)
        # # find ck in cs index
        # ck_dup_index, cs_dup_index = np.where((cs==ck[:,None]).all(-1))
        # for ck_i, cs_i in zip(ck_dup_index, cs_dup_index):
        #     #get x/mean/cov_all_split at each ck by remove dup cs point
        #     x_all_split_each_ck =\
        #         x_all_split[:2]\
        #         + [np.delete(x_all_split[2], cs_i, axis=0)]
        #     mean_all_split_each_ck =\
        #         mean_all_split[:2]\
        #         + [np.delete(mean_all_split[2], cs_i, axis=0)]

        #     covmat_each_ck = np.delete(covmat, nk+nh+cs_i, axis=0)
        #     covmat_each_ck = np.delete(covmat_each_ck, nk+nh+cs_i, axis=1)
           
        #     cov_all_split_each_ck = np.vsplit(
        #         covmat_each_ck, [nk, nk+nh, nk+nh+ns-1]
        #         )[:-1] #exclude final empty array
        #     cov_all_split_each_ck = \
        #         [np.hsplit(c, [nk, nk+nh, nk+nh+ns-1][:-1])\
        #         for c in cov_all_split_each_ck]
        #     zs_each_ck = [z for idx_z, z in enumerate(zs) if idx_z != cs_i]

        #     dup_NC[cs_i, 0] = __check_normalized_constant(
        #         __get_normalized_constant(
        #             x_all_split_each_ck,
        #             mean_all_split_each_ck,
        #             cov_all_split_each_ck, zs_each_ck, 
        #             sub_multi = 's_h', sub_s='s'
        #             )
        #         )
        # return (top_part/dup_NC)[cs_dup_index,:]

def _apply_nonneg_truncation(mvs):
    """Non-negativity correction: zero-clamp mean + truncated-normal variance.

    For a posterior N(μ, σ²) truncated to [0, ∞), the truncated
    distribution has:

        α  = −μ/σ
        Φ₊ = Φ(μ/σ) = P(Z ≥ 0)   (survival probability)
        λ  = φ(α) / Φ₊            (inverse Mills ratio)
        δ  = λ(λ − α)             (variance reduction factor, ∈ [0, 1])

        Mean_trunc = μ + σλ        (always ≥ 0)
        Var_trunc  = σ²(1 − δ)    (always ≤ σ²)

    **Mean**: We use zero-clamping  max(μ, 0)  rather than the full
    truncated-normal mean  μ + σλ.  The latter inflates near-zero
    estimates to ~0.8σ ("yellow floor" artifact) in data-sparse
    regions where μ ≈ 0 but σ is large.  Zero-clamping is
    conservative but honest.

    **Variance**: We DO apply the truncated-normal formula  σ²(1−δ)
    for every point, even those with μ > 0.  Knowing Z ≥ 0 always
    reduces uncertainty, and the reduction is smooth and physically
    meaningful:

        μ ≫  0  →  δ ≈ 0  →  Var ≈ σ²   (constraint irrelevant)
        μ  =  0  →  δ ≈ 0.64 →  Var ≈ 0.36σ²
        μ ≪  0  →  δ → 1   →  Var → 0   (almost all mass < 0)

    Parameters
    ----------
    mvs : ndarray, shape (nk, 3) or (nk, 4)
        Columns: [mean, variance, skewness, (delta_mu)].
        If delta_mu == -1 for a point, that point's nonneg correction
        was already applied inside the QMC integrand; skip it here.

    Returns
    -------
    mvs_out : ndarray, shape (nk, 3)
        Corrected array with truncated variance; delta_mu column removed.
    """
    nk = mvs.shape[0]
    mu  = mvs[:, 0].copy()
    var = mvs[:, 1].copy()

    # Check for the "already handled" flag in column 3
    _has_flag = mvs.shape[1] >= 4
    if _has_flag:
        _delta_col = mvs[:, 3].copy()
    else:
        _delta_col = np.zeros(nk)

    for i in range(nk):
        # If the QMC integrand already computed truncated moments,
        # the delta_mu column is set to -1 as a flag.  Skip these
        # points — their mean and variance are already correct.
        if _has_flag and _delta_col[i] < 0:
            continue

        # Skip NaN / invalid points
        if np.isnan(mu[i]) or np.isnan(var[i]) or var[i] <= 0:
            if not np.isnan(mu[i]) and mu[i] < 0:
                mvs[i, 0] = 0.0
            continue

        sigma_i = np.sqrt(var[i])
        # α = −μ/σ  (positive when μ < 0)
        alpha = -mu[i] / sigma_i

        # Φ₊ = Φ(μ/σ) = P(Z ≥ 0)
        Phi_pos = scipy.stats.norm.cdf(mu[i] / sigma_i)

        if Phi_pos < 1e-12:
            # Almost no probability above 0 → clamp both mean and var
            mvs[i, 0] = 0.0
            mvs[i, 1] = 0.0
            continue

        # Inverse Mills ratio  λ = φ(α) / Φ₊
        phi_alpha = scipy.stats.norm.pdf(alpha)
        lam = phi_alpha / Phi_pos

        # Variance reduction factor  δ = λ(λ − α) = λ(λ + μ/σ)
        delta = lam * (lam - alpha)

        # Truncated-normal variance:  σ²(1 − δ)
        mvs[i, 1] = var[i] * max(1.0 - delta, 0.0)

        # Zero-clamp the mean (avoid yellow-floor inflation)
        if mu[i] < 0:
            mvs[i, 0] = 0.0

    # Remove delta_mu column (return 3-col array)
    return mvs[:, :3]


def _get_x_all_split(nk, zh, zs):
    '''
        Create the "estimated" observed values 
    for the estimation and observations
        For now, zero is used for estimation points
    (which should be specified as NaN)
        mean values are used for soft data
    '''
    x_all_split = []
    xk = np.empty((nk, 1))  # will be replaced later
    x_all_split.append(xk)
    x_all_split.append(zh) # e.g. xh
    if zs:
        xs = np.empty((len(zs), 1))
        for i, zsi in enumerate(zs):
            if get_standard_soft_pdf_type(zsi[0]) == 10:  # gaussian/normal
                xs[i] = zsi[1] #z_mean
            else:
                # Normalise to 2-D for proba2stat (handles scalar, 1-D, 2-D nl)
                xs[i], dummy_v = proba2stat(
                    zsi[0],
                    np.atleast_2d(zsi[1]),
                    np.atleast_2d(zsi[2]),
                    np.atleast_2d(zsi[3])
                    )
        x_all_split.append(xs)
    else:
        x_all_split.append(None)
    return x_all_split

def _get_mean_all_split(x_all_split, order):
    '''
    Obtain the trend estimations at the estimation and data locations based 
    upon the specified trend order
    '''
    if isinstance(order, np.ndarray):  # user defined general knowledge, (row x 1 2d array)
        start_i = 0
        mean_all_split = []
        for i in x_all_split:
            if i is not None:
                end_i = start_i + i.size
                mean_all_split.append(order[start_i:end_i, :])
                start_i = end_i
            else:
                mean_all_split.append(None)   
    elif order == 0:  # constant mean, exclude zk, average h and s
        xx=[x for x in x_all_split[1:] if x is not None]
        constant_mean_ = np.vstack(xx).mean()
        mean_all_split = []
        for i in x_all_split:
            if i is not None:
                mean_all_split.append(np.ones(i.shape)*constant_mean_)
            else:
                mean_all_split.append(None)
        #for means in mean_all_split:
        #  means[:] = constant_mean_
    elif np.isnan(order):  # zero mean: 
        mean_all_split = []
        for i in x_all_split:
            if i is not None:
                mean_all_split.append(np.zeros(i.shape))
            else:
                mean_all_split.append(None)
    return mean_all_split

def _get_cov_all_split(ck, ch, cs, covmodel, covparam):
  '''
  Obtain the covariance in the split ways.
  See coord2K
  '''
  return coord2Ksplit((ck, ch, cs), (ck, ch, cs),
                      covmodel, covparam)[0]

def _get_x(x_all_split, sub):
  '''
  Retrieve the estimated observed values given specified class, i.e., sub
  sub can be k, h, and s for estimation, hard, and soft data
  '''
  idx = [KHS_DICT[i] for i in sub]
  output=[x_all_split[i] for i in idx if x_all_split[i] is not None]
  if len(output)>0:
    return np.vstack(output)
  else:
    return None

def _get_mean(mean_all_split, sub):
  '''
  Retrieve the expected values given specified class, i.e., sub
  sub can be k, h, and s for estimation, hard, and soft data
  '''
  idx = [KHS_DICT[i] for i in sub]     
  output=[mean_all_split[i] for i in idx if mean_all_split[i] is not None]
  if len(output)>0:
    return np.vstack(output)
  else:
    return None

def _get_sigma(cov_all_split, sub_a, sub_b, inv=False):
  '''
  Retrieve the cross-covaiance between specified class, i.e., sub_a and sub_b
  sub_a and sub_b can be k, h, and s for estimation, hard, and soft data
  '''
  idx_a = [KHS_DICT[i] for i in sub_a]
  idx_b = [KHS_DICT[i] for i in sub_b]

  cov_a_b = []
  for i in idx_a:
    output=[cov_all_split[i][j] for j in idx_b if cov_all_split[i][j] is not None]
    if len(output)>0:
      cov_a_b.append(np.hstack(output))
  # Filter out any None entries (rows where all covariance blocks were None)
  cov_a_b = [x for x in cov_a_b if x is not None]
  if len(cov_a_b) == 0:
    # All covariance blocks are None — return an empty 2D array
    return np.array([]).reshape((0, 0))
  cov_a_b = np.vstack(cov_a_b)
  if not inv:
    return cov_a_b
  else:
    # Check for singularity before computing pseudo-inverse
    if cov_a_b.size == 0:
      return cov_a_b
    cond = np.linalg.cond(cov_a_b)
    if cond > 1e12:
      import warnings
      warnings.warn(
        "Covariance matrix for sub ({a},{b}) is nearly singular "
        "(condition number = {c:.2e}). Estimation results at this "
        "location may be unreliable. Consider checking whether the "
        "estimation point overlaps with data locations or whether "
        "the covariance model parameters are appropriate.".format(
          a=sub_a, b=sub_b, c=cond))
    return _robust_pinv(cov_a_b)

def _get_mean_a_given_b(x_all_split, mean_all_split,
    cov_all_split, sub_a, sub_b):
    '''
    Obtain the conditonal mean a given b by using conditonal Gaussian formula
    '''      

    if 'k' not in sub_b:                    
        x_b = _get_x(x_all_split, sub_b)
        mean_a = _get_mean(mean_all_split, sub_a)
        mean_b = _get_mean(mean_all_split, sub_b)
        if mean_b is not None: # consider the case that data in sub_b does not exist
            sigma_a_b = _get_sigma(cov_all_split, sub_a, sub_b)
            inv_sigma_b_b = _get_sigma(cov_all_split, sub_b, sub_b, inv=True)
            output=mean_a + sigma_a_b.dot(inv_sigma_b_b).dot(x_b - mean_b)
        else:
           output=mean_a
        return output
    else:
        nlim=np.asarray(x_all_split[0]).size
        smtx=np.ones((1,nlim))
        if nlim>1:
            idx = [KHS_DICT[i] for i in sub_b if i != 'k']
            xhs=np.vstack([x_all_split[i] for i in idx])
            x_b=[x_all_split[0].reshape((1,nlim)),xhs.dot(smtx)]
            x_b=np.vstack(x_b)
        else:            
           x_b = _get_x(x_all_split, sub_b)
        
        mean_a = _get_mean(mean_all_split, sub_a)
        mean_b = _get_mean(mean_all_split, sub_b)
        if mean_b is not None: # consider the case that data in sub_b does not exist
            sigma_a_b = _get_sigma(cov_all_split, sub_a, sub_b)
            inv_sigma_b_b = _get_sigma(cov_all_split, sub_b, sub_b, inv=True)
            output=mean_a + sigma_a_b.dot(inv_sigma_b_b).dot(x_b - mean_b)
        else:
            output=mean_a
        return output

def _get_sigma_a_given_b(cov_all_split, sub_a, sub_b):
  '''
  Obtain the conditional covariance a given b by using conditonal Gaussian 
  formula
  '''
  sigma_a_a = _get_sigma(cov_all_split, sub_a, sub_a)
  sigma_a_b = _get_sigma(cov_all_split, sub_a, sub_b)
  if sigma_a_b.size > 0:
    sigma_b_b = _get_sigma(cov_all_split, sub_b, sub_b)
    if sigma_b_b.size == 0:
      # sub_b data does not exist; return unconditional covariance
      return sigma_a_a
    inv_sigma_b_b = _get_sigma(cov_all_split, sub_b, sub_b, inv=True)
    sigma_b_a = _get_sigma(cov_all_split, sub_b, sub_a)
    return sigma_a_a - sigma_a_b.dot(inv_sigma_b_b).dot(sigma_b_a)
  else:
    return sigma_a_a

def _get_multivariate_normal_pdf(x_all_split, mean_all_split,
    cov_all_split, sub_multi):
    '''
    Obtain multivariate Gaussian pdf or conditional multivariate Gaussian
    based upon the specified notations, i.e., sub_multi
    
    Note:
    sub_multi     string    h, s, and k for hard, soft and estimation locations
                            a_b represents a given b, e.g., k_h
    '''                                   
    if "_" in sub_multi:  # "given" type
        sub_a, sub_b = sub_multi.split('_')#x_sub.split('_') (temporarily change by HL)
        m = _get_mean_a_given_b(x_all_split, mean_all_split,
                                cov_all_split, sub_a, sub_b)
        v = _get_sigma_a_given_b(cov_all_split, sub_a, sub_b)
    else:  # single_sub
        sub_a = sub_multi
        m = _get_mean(mean_all_split, sub_a)
        v = _get_sigma(cov_all_split, sub_a, sub_a)
    return scipy.stats.multivariate_normal(m.T[0], v).pdf

def _get_fs(zs):
    '''
    the product of fs distributions
    x   ndim(e.g. npts) by ns
            x is a 2-D np array with the dimension of 
            ndim(number of samples at each integral) 
            by ns (the number of integrals, i.e., number 
            of soft data)      
    '''
    def fs(x):
        res = np.ones((x.shape[0],1))
        for idx_k, zsi in enumerate(zs):
            pdf_type = get_standard_soft_pdf_type(zsi[0])
            if pdf_type == 2:
                nl = int(np.asarray(zsi[1]).flat[0])
                limi = np.asarray(zsi[2]).ravel()
                probdens = np.asarray(zsi[3]).ravel()
                y_i = np.interp(
                    x[:,idx_k:idx_k+1], limi[:nl], probdens[:nl],
                    left = 0., right = 0.)
            elif pdf_type == 1:
                # BUG FIX: was using zs[1],zs[2],zs[3] (global list)
                # instead of zsi[1],zsi[2],zsi[3] (individual soft datum)
                nl = int(np.asarray(zsi[1]).flat[0])
                limi = np.asarray(zsi[2]).ravel()
                probdens = np.asarray(zsi[3]).ravel()
                # Histogram PDF: probdens[j] is constant for limi[j] <= x < limi[j+1]
                # There are nl limit values defining nl-1 bins
                xi = x[:, idx_k].ravel()
                bin_idx = np.searchsorted(limi[:nl], xi, side='right') - 1
                # Clip to valid bin range [0, nl-2]; values outside get 0
                y_i = np.where(
                    (bin_idx >= 0) & (bin_idx < nl - 1),
                    probdens[np.clip(bin_idx, 0, nl - 2)],
                    0.0
                ).reshape(-1, 1)
            elif pdf_type == 10:
                zm = zsi[1]
                zstd = np.sqrt(zsi[2])
                try:
                    y_i = scipy.stats.norm.pdf(
                        x[:,idx_k:idx_k+1], loc=zm, scale=zstd)
                except FloatingPointError:
                    y_i = np.zeros((x.shape[0], 1))
              
            if not (y_i.all() or y_i.any()):
                return np.zeros((x.shape[0], 1))
            else:
                res *= y_i
        return res
    return fs

def _get_Fsinv(zs):

    Fs = pdf2cdf(zs)
    def Fsinv(x):  
        res = np.zeros(x.shape)
        for idx_k, Fsi in enumerate(Fs):
            pdf_type = get_standard_soft_pdf_type(Fsi[0])
            if pdf_type == 2:
                nl = Fsi[1][0]
                limi = Fsi[2]
                probdens = Fsi[3]
                probCDFs = Fsi[4]

                alpha = np.diff(probdens) / np.diff(limi)
                i = np.searchsorted(probCDFs[1:],x[:,idx_k:idx_k+1])
                D = np.abs(probdens[i]**2 + 2*alpha[i] * (x[:,idx_k:idx_k+1] - probCDFs[i]))
                y_i = limi[i] + (-probdens[i] + np.sqrt(D)) / alpha[i]
                
            res[:,idx_k:idx_k+1] = y_i
        return res
    return Fsinv

def _get_int_fg_a_given_b_fs_s(x_all_split, mean_all_split,
  cov_all_split, zs, sub_multi, sub_s='s'):
  '''
  The upper right part and lower right part of the last row of formula (1)
  The evaluation is based upon Eqns. (8) or (9) in the cases of s_h and s_kh 
  respectively
  '''

  #fg='s_kh', fs='s'
  sub_a, sub_b = sub_multi.split('_')
  sigma_a_given_b = _get_sigma_a_given_b(cov_all_split, sub_a, sub_b)
  try:
      inv_sigma_a_given_b = _robust_pinv(sigma_a_given_b)
  except np.linalg.LinAlgError as e:
      # import pdb
      # pdb.set_trace()
      raise e
  # mean_tilde_s = zs[1]  # mean
  # sigma_tilde_s = np.diag(zs[2].T[0])  # cov matrix
  mean_tilde_s = []
  sigma_tilde_s = []
  for zsi in zs:
      mean_tilde_s.append([zsi[1]])
      sigma_tilde_s.append(zsi[2])
  mean_tilde_s = np.array(mean_tilde_s) # mean
  sigma_tilde_s = np.diag(sigma_tilde_s)  # cov matrix
  try:
      inv_sigma_tilde_s = _robust_pinv(sigma_tilde_s)
  except np.linalg.LinAlgError as e:
      # import pdb
      # pdb.set_trace()
      raise e

  sigma_t = _robust_pinv(inv_sigma_a_given_b + inv_sigma_tilde_s)
  ns = mean_tilde_s.shape[0]

  # --- Use log-space determinants (slogdet) to avoid underflow/overflow ---
  # For large ns (e.g., 19), np.linalg.det() of ns×ns covariance matrices
  # can underflow to 0 or overflow to Inf, making NC = 0/NaN/Inf even
  # though the final BME formulas are well-conditioned (NC cancels).
  _sign_t, _logdet_t = np.linalg.slogdet(sigma_t)
  _sign_ab, _logdet_ab = np.linalg.slogdet(sigma_a_given_b)
  _sign_ts, _logdet_ts = np.linalg.slogdet(sigma_tilde_s)

  # log(alias_c) = 0.5*logdet_t - 0.5*(logdet_ab + logdet_ts) - (ns/2)*log(2π)
  _log_fgfs_front = (0.5 * _logdet_t
                     - 0.5 * (_logdet_ab + _logdet_ts)
                     - (ns / 2.0) * np.log(2 * np.pi))
  # Check sign: all covariance matrices should be PSD, so signs should be +1
  _det_sign = _sign_t * _sign_ab * _sign_ts
  if _det_sign <= 0:
      # Singular or negative determinant — fall back to old-style det
      import warnings as _w
      _w.warn("[STARBME] slogdet sign issue in NC computation "
              "(signs: t={}, ab={}, ts={}). NC may be unreliable.".format(
                  _sign_t, _sign_ab, _sign_ts))
  alias_c = _log_fgfs_front  # store log-space value for now
  fgfs_front = _log_fgfs_front  # log-space
  # fgfs_front = np.sqrt(det_sigma_t) /\
  #       ((2*np.pi)**(ns/2.) *
  #        np.sqrt(det_sigma_a_given_b * det_sigma_tilde_s)
  #        )

  mean_a_given_b = _get_mean_a_given_b(
        x_all_split, mean_all_split, cov_all_split, sub_a, sub_b)
  fgfs_1 = np.diag((mean_a_given_b.T).dot(
        inv_sigma_a_given_b).dot(mean_a_given_b))
  fgfs_2 = (mean_tilde_s.T).dot(inv_sigma_tilde_s).dot(mean_tilde_s)
  fgfs_3 = (mean_a_given_b.T).dot(inv_sigma_a_given_b) +\
        (mean_tilde_s.T).dot(inv_sigma_tilde_s)
  fgfs_4 = inv_sigma_tilde_s.dot(mean_tilde_s) +\
        inv_sigma_a_given_b.dot(mean_a_given_b)
  alias_sigma_d = sigma_t
  alias_mean_d = alias_sigma_d.dot(fgfs_4)
  # 1 x 1 array (scalar)
  _quadratic_form = (
      (fgfs_1 + fgfs_2 - np.diag(fgfs_3.dot(sigma_t).dot(fgfs_4)))
      ).item()

  # --- Compute NC in log-space, then exponentiate ---
  # log(NC) = log(fgfs_front) + (-1/2) * quadratic_form
  _log_NC = _log_fgfs_front + (-0.5) * _quadratic_form
  # Clamp to prevent overflow (exp(709) is near double max)
  _log_NC = np.clip(_log_NC, -700, 700)
  NC = _det_sign * np.exp(_log_NC)

  # Store the exponential part for backward compatibility
  fgfs_end = np.exp(np.clip(-0.5 * _quadratic_form, -700, 700))
  alias_c = NC / fgfs_end if abs(fgfs_end) > 1e-300 else 0.0
  alias_fgfs1234 = fgfs_end

  return NC, \
        (sigma_t, inv_sigma_a_given_b, inv_sigma_tilde_s, mean_tilde_s,
            alias_c, alias_fgfs1234, alias_mean_d, alias_sigma_d)

def _changetimeform(ck,ch=None,cs=None):
  '''
  Change the time format into float while it is in datetime format
  '''  
  
  if type(ck[0,-1])==np.datetime64:
    origin=ck[0,-1]
    ck[:,-1]=np.double(np.asarray(ck[:,-1],dtype='datetime64')-origin)
    ck=ck.astype(np.double)
    if ch is not None and ch.size>0:
      if (not type(ch[0,-1]==np.datetime64)):
        print ('Time format of ch is not consistent with ck (np.datetime64)')
        raise
      ch[:,-1]=np.double(np.asarray(ch[:,-1],dtype='datetime64')-origin)
      ch=ch.astype(np.double)
    if cs is not None and cs.size>0:
      if (not type(cs[0,-1]==np.datetime64)):
        print ('Time format of cs is not consistent with ck (np.datetime64)')
        raise
      cs[:,-1]=np.double(np.asarray(cs[:,-1],dtype='datetime64')-origin)
      cs=cs.astype(np.double)
      
  return ck,ch,cs

def _set_nh_ns(ck,ch,cs,nhmax,nsmax,dmax):
  '''
  Set the size of nhmax and nsmax that limits the size of matrix to be allocated
  it can be important for an efficient S/T estimation
  '''
  if dmax is not None and np.all(dmax):
    nhmax = int(nhmax)
    nsmax = int(nsmax)
    dmax = np.array(dmax,ndmin=2)
    return nhmax,nsmax,dmax
  
  if ck[0,:].size<3:
    if dmax is None:
      dmax_=0
      if ch is not None:
        ch=np.array(ch,ndmin=2)
        maxd_h=pdist(ch).max()
        dmax_=np.max([dmax_,maxd_h])
      if cs is not None:
        cs=np.array(cs,ndmin=2)
        maxd_s=pdist(cs).max()
        dmax_=np.max([dmax_,maxd_s])
      dmax=np.array(dmax_).reshape(1,1)

    if nhmax is None:
      if ch is not None:
        nhmax=ch.shape[0]
      else:
        nhmax=0

    if nsmax is None:
      if cs is not None:
        nsmax=cs.shape[0]
      else:
        nsmax=0

    
  else:
    maxd=0
    maxt=0
    if dmax is None:
      if ch is not None:
        dummy=np.random.rand(ch.shape[0],1)
        _,cMS_h,tME_h,_=valstv2stg(ch,dummy)
        if nhmax is None:
          nhmax=cMS_h.shape[0]*3
        maxd_h=pdist(cMS_h).max()
        maxt_h=pdist(tME_h.reshape((tME_h.size,1))).max()
      else:
        maxd_h=0
        maxt_h=0
        nhmax=0
      maxd=np.max([maxd_h,maxd]) 
      maxt=np.max([maxt_h,maxt])
      if cs is not None: 
        dummy=np.random.rand(cs.shape[0],1)
        _,cMS_s,tME_s,_=valstv2stg(cs,dummy)
        if nsmax is None:
          if zs[0]==10 or zs[0] == 'gaussian':
            nsmax=cMS_s.shape[0]*3
          else:
            nsmax=3
        maxd_s=pdist(cMS_s).max()
        maxt_s=pdist(tME_s.reshape((tME_s.size,1))).max()
        maxd=np.max([maxd_s,maxd])
        maxt=np.max([maxt_s,maxt])
      else:
        nsmax=0
        maxd_s=0
        maxt_s=0
      maxd=np.max([maxd_s,maxd])
      maxt=np.max([maxt_s,maxt])
    dmax=np.array([maxd,maxt,np.nan]).reshape(1,3)

  return nhmax,nsmax,dmax

def _stratio(covparam):
  '''
  Estimate the S/T ratio for dmax
  '''
  nm=len(covparam)
  sills= np.array([covparam[k][0] for k in range(nm)])  
  hrange = np.array([covparam[k][1][0] for k in range(nm)]) 
  idx0 = np.where([hrange[k] is not None for k in range(nm)])[0]
  idx=np.where(sills[idx0]==sills.max())[0]
  ratio=covparam[idx0[idx]][1][0]/covparam[idx0[idx]][2][0]
  return ratio


def _normalize_zs(zs, ns):
    """Convert *batch* soft-data format to the *per-point* list expected internally.

    ``probaUniform`` / ``probaGaussian`` return a 4-element batch list::

        [softpdftype, nl_arr(ns,1), limi_arr(ns,*), probdens_arr(ns,*)]

    The internal BME engine (``_get_x_all_split``, neighbourhood picking, etc.)
    requires a per-point list::

        [(type_0, nl_0, limi_0, pd_0),
         (type_1, nl_1, limi_1, pd_1), ...]

    This helper detects the format from the type of ``zs[0]``:

    * ``int / np.integer`` → batch format → convert
    * anything else (tuple/list) → already per-point → pass through

    Parameters
    ----------
    zs : list or None
        Raw soft-data argument as passed by the caller.
    ns : int
        Number of soft-data locations (from ``cs.shape[0]``).

    Returns
    -------
    list or None
        Per-point list of length *ns*, or ``None`` if *zs* is ``None``.
    """
    if zs is None or ns == 0:
        return zs
    # Already per-point: first element is a tuple/list, not an integer type code
    if not isinstance(zs[0], (int, np.integer)):
        return zs
    # Batch format: unpack and split row-by-row
    _stype, _nl, _limi, _pd = zs
    _nl   = np.atleast_2d(np.asarray(_nl))
    _limi = np.atleast_2d(np.asarray(_limi))
    _pd   = np.atleast_2d(np.asarray(_pd))
    return [
        (_stype, _nl[i:i+1], _limi[i:i+1], _pd[i:i+1])
        for i in range(ns)
    ]


def _bme_posterior_prepare(
    ck, ch=None, cs=None, zh=None, zs=None,
    covmodel=None, covparam=None, covmat=None,
    order=np.nan, options=None,
    nhmax=None, nsmax=None, dmax=None,
    general_knowledge='gaussian',
    #  specific_knowledge='unknown',  
    pdfk=None,pdfh=None,pdfs=None,hk_k=None,hk_h=None,hk_s=None,
    gui_args=None):

    '''
    check and configure arguments and 
        find neighbor ckhs index for bme posterior calculation

    ckhs_idx_list:  [ck_idx, ch_idx, cs_idx] represents
        these ck have the same neighbors ch and cs

    return (output_arguments, configured_arguments):
        a tuple contain arguments
    '''
    print('preparing...', end='')
    if covmat is None:
        if (covmodel is None) or (covparam is None):
            raise ValueError(
                'Covariance model and their associated parameters '\
                'should be specified if no covarinace matrix provided.')


    dk = ck.shape[1]
    nk = ck.shape[0]
    nh = ch.shape[0] if ch is not None else 0
    ns = cs.shape[0] if cs is not None else 0

    ck, ch, cs = _changetimeform(ck, ch, cs)
    nhmax, nsmax, dmax = _set_nh_ns(ck, ch, cs, nhmax, nsmax, dmax)
    if dmax.size == 3 and np.isnan(dmax[0][2]):
        dmax[0][2] = _stratio(covparam)
    stratio = dmax[0][2] if dk == 3 else 1.
      
    if options is None:
        options = BMEoptions()

    if gui_args:
        qpgd = gui_args[0]
    
    if general_knowledge == 'gaussian':
        order = get_standard_order(order)

        if covmat is not None: #consider covmat if exists, not distance, no need to calculate distance
            ckhs_idx_list = []
            if nh != 0:
                covmat_k_h = covmat[:nk, nk:nk+nh] # slice cov(k x h)
                # sort from big to small, clip with nhmax
                k_by_h_idx = (-covmat_k_h).argsort(axis=1)[:, :nhmax]
                # sort h index (inplace)
                k_by_h_idx.sort(axis=1)
                # make ch_ck_dict
                ch_ck_dict = {}
                for k_idx, h_idx in enumerate(k_by_h_idx):
                    tuple_h_idx = tuple(h_idx)
                    if tuple_h_idx not in ch_ck_dict.keys():
                        k_idx_mul = np.where(
                            np.all(h_idx == k_by_h_idx, axis=1)
                            )[0]
                        ch_ck_dict[tuple_h_idx] = list(k_idx_mul)
                    else:
                        continue #skip duplicated row
                if ns != 0:
                    # slice cov(k x s)
                    covmat_k_s = covmat[:nk, nk+nh:nk+nh+ns]
                    for ch_idx, ck_idx in ch_ck_dict.items():
                        picked_covmat_k_s = covmat_k_s[ck_idx, :]
                        # sort from big to small, clip with nsmax
                        picked_k_by_s_idx =\
                            (-picked_covmat_k_s).argsort(axis=1)[:, :nsmax]
                        # sort s index (inplace)
                        picked_k_by_s_idx.sort(axis=1)
                        # make ch_ck_dict
                        cs_ck_dict = {}
                        for picked_k_idx, s_idx in enumerate(picked_k_by_s_idx):
                            tuple_s_idx = tuple(s_idx)
                            if tuple_s_idx not in cs_ck_dict.keys():
                                picked_k_idx_mul = np.where(
                                    np.all(s_idx == picked_k_by_s_idx, axis=1)
                                    )[0]
                                cs_ck_dict[tuple_s_idx] = list(picked_k_idx_mul)
                            else:
                                continue #skip duplicated row
                        for cs_idx, ck2_idx in cs_ck_dict.items():
                            ck_idx = np.array(ck_idx)
                            ckhs_idx_list.append(
                                [ck_idx[ck2_idx,], ch_idx, cs_idx]
                                )
                else: # ns = 0
                    for ch_idx, ck_idx in ch_ck_dict.items():
                        ckhs_idx_list.append([ck_idx, ch_idx, ()])
            elif nh == 0 and ns != 0: # nh = 0, ns != 0
                covmat_k_s = covmat[:nk, nk+nh:nk+nh+ns] # slice cov(k x s)
                # sort from big to small, clip with nsmax
                k_by_s_idx = (-covmat_k_s).argsort(axis=1)[:, :nsmax]
                # sort s index (inplace)
                k_by_s_idx.sort(axis=1)
                # make cs_ck_dict
                cs_ck_dict = {}
                for k_idx, s_idx in enumerate(k_by_s_idx):
                    tuple_s_idx = tuple(s_idx)
                    if tuple_s_idx not in cs_ck_dict.keys():
                        k_idx_mul = np.where(
                            np.all(s_idx == k_by_s_idx, axis=1)
                            )[0]
                        cs_ck_dict[tuple_s_idx] = list(k_idx_mul)
                    else:
                        continue #skip duplicated row
                for cs_idx, ck_idx in cs_ck_dict.items():
                    ckhs_idx_list.append([ck_idx, (), cs_idx])
            else: # nh = 0, ns = 0
                raise ValueError("nh and ns shouldn't be both 0.")
        else:
            #aggregate ck for same hard data and soft data
            # chs = np.vstack(ch, cs)
            ck_norm = np.copy(ck)
            ck_norm[:, -1] = ck_norm[:, -1] * stratio
            if dk == 3:
                dmax_norm = (dmax[0][0]**2 + (dmax[0][1] * stratio)**2)**0.5
            else:
                dmax_norm = dmax[0][0]

            if isinstance(ch, np.ndarray) and nhmax != 0:
                ch_norm = np.copy(ch)
                ch_norm[:, -1] = ch_norm[:, -1] * stratio
                ch_tree = cKDTree(ch_norm)
            if isinstance(cs, np.ndarray) and nsmax != 0:
                cs_norm = np.copy(cs)
                cs_norm[:, -1] = cs_norm[:, -1] * stratio
                cs_tree = cKDTree(cs_norm)

            ckhs_idx_list = []
            if isinstance(ch, np.ndarray) and nhmax != 0: #has harddata
                ch_ck_dict =\
                    neighbours_index_kd(ck_norm, ch_tree, nhmax, dmax_norm)
                for ch_idx, ck_idx in ch_ck_dict.items():
                    if isinstance(cs, np.ndarray) and nsmax != 0: #both hard and soft
                        picked_ck_norm = ck_norm[ck_idx, :]
                        cs_ck_dict =\
                            neighbours_index_kd(
                                picked_ck_norm, cs_tree, nsmax, dmax_norm
                                )
                        for cs_idx, ck2_idx in cs_ck_dict.items():
                            ck_idx = np.array(ck_idx)
                            ckhs_idx_list.append(
                                [ck_idx[ck2_idx,], ch_idx, cs_idx]
                                )
                    else: #only harddata
                        ckhs_idx_list.append([ck_idx, ch_idx, ()])
            elif isinstance(cs, np.ndarray) and nsmax != 0: #only softdata
                cs_ck_dict =\
                    neighbours_index_kd(ck_norm, cs_tree, nsmax, dmax_norm)
                for cs_idx, ck_idx in cs_ck_dict.items():
                    ckhs_idx_list.append([ck_idx, (), cs_idx])
    else:
        raise ValueError("Now we can not consider non-gaussian GK." )

    configured_arguments =\
        (ck, ch, cs, zh, zs,
        covmodel, covparam, covmat,
        order, options,
        nhmax, nsmax, dmax,
        general_knowledge,
        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
        gui_args)
    output_arguments = \
        (ckhs_idx_list,)
    print('done')
    return (output_arguments, configured_arguments)

def __get_khs_size(k,h,s):
    if k.shape[0] == 0:
        raise ValueError('ck can not be empty.')
    else:
        return list(map(__get_coord_size,[k, h, s]))

def __get_coord_size(c):
    n = c.shape[0] if c is not None else 0
    return n

def __get_all_split(ck, ch, cs, zh, zs, covmodel, covparam, covmat, order):
    nk, nh, ns = __get_khs_size(ck, ch, cs)
    x_all_split = _get_x_all_split(nk, zh, zs)
    mean_all_split = _get_mean_all_split(x_all_split, order)
    if covmat is not None:
        cov_all_split = np.vsplit(
            covmat, [nk, nk+nh, nk+nh+ns]
            )[:-1] #exclude final empty array
        cov_all_split = \
            [np.hsplit(c, [nk, nk+nh, nk+nh+ns][:-1])\
            for c in cov_all_split]
    else:
        cov_all_split =\
            _get_cov_all_split(ck, ch, cs, covmodel, covparam)
        cov_k_khs = np.hstack([i for i in cov_all_split[0] if i is not None])
        if len([i for i in cov_all_split[1] if i is not None]) != 0:
            cov_h_khs = np.hstack([i for i in cov_all_split[1] if i is not None])
        else:
            cov_h_khs = np.array([]).reshape((-1, cov_k_khs.shape[1]))
        if len([i for i in cov_all_split[2] if i is not None]) != 0:
            cov_s_khs = np.hstack([i for i in cov_all_split[2] if i is not None])
        else:
            cov_s_khs = np.array([]).reshape((-1, cov_k_khs.shape[1]))
        covmat = np.vstack([cov_k_khs, cov_h_khs, cov_s_khs])
    return x_all_split, mean_all_split, cov_all_split, covmat

def __get_normalized_constant(
    x_all_split, mean_all_split, cov_all_split, zs, 
    sub_multi = 's_h', sub_s='s'):

    NC, useful_args = _get_int_fg_a_given_b_fs_s(
        x_all_split, mean_all_split,
        cov_all_split, zs, 
        sub_multi = 's_h', sub_s='s'
        )
    return NC

def __get_covmat_remove_index(covmat):
    if np.linalg.matrix_rank(covmat) < covmat.shape[0]:
        # print('warning: matrix rank less than covmat row')
        has_remove_index = True
        diff_n = covmat.shape[0] - np.linalg.matrix_rank(covmat)
        corrmat = np.corrcoef(covmat)
        remove_index = np.vstack(
            np.unravel_index(
                (np.abs(corrmat).ravel()).argsort(),
                covmat.shape
                )
            ).T
        remove_index = remove_index[
            remove_index[:, 0] < remove_index[:, 1]
            ]
        remove_index_res = np.unique(remove_index[:, 1][-diff_n:])
        remove_index_len = remove_index_res.size
        i = 1
        while remove_index_len < diff_n:
            remove_index_res = np.unique(remove_index[:, 1][-(diff_n+i):])
            remove_index_len = remove_index_res.size
            i += 1
        # print('warning: remove_index_res:', remove_index_res)
    else:
        has_remove_index = False
        remove_index_res = np.array([])
    return has_remove_index, remove_index_res


def _find_ck_cs_near_duplicates(covmat, nk, nh, ns):
    """Detect near-duplicate ck-cs pairs using covariance matrix row
    correlation.

    This is a unit-free, scale-free criterion: when two locations are
    nearly co-located, their covariance rows become nearly identical,
    so ``np.corrcoef`` (Pearson correlation between rows) approaches 1.
    No distance threshold or coordinate-unit knowledge is needed.

    Parameters
    ----------
    covmat : (n, n) ndarray
        Joint covariance matrix ordered as [ck, ch, cs].
    nk, nh, ns : int
        Number of estimation, hard, and soft data points.

    Returns
    -------
    ck_dup_idx : 1-D int array
        Indices into ck that are near-duplicates of some cs.
    cs_dup_partner : 1-D int array
        For each entry in *ck_dup_idx*, the index into cs of its partner.
    is_rank_deficient : bool
        True if covmat is rank-deficient (regardless of whether
        ck-cs duplicates were found).
    """
    _CORR_THRESH = 1.0 - 1e-6   # dimensionless threshold

    rank = np.linalg.matrix_rank(covmat)
    n_total = covmat.shape[0]
    if rank >= n_total:
        return (np.array([], dtype=int),
                np.array([], dtype=int),
                False)

    # Row-correlation matrix (unit-free)
    corrmat = np.corrcoef(covmat)

    # Extract the ck-vs-cs block:
    #   ck indices  = [0, nk)
    #   ch indices  = [nk, nk+nh)
    #   cs indices  = [nk+nh, nk+nh+ns)
    ck_cs_corr = np.abs(corrmat[:nk, nk + nh: nk + nh + ns])

    # Find (ck_i, cs_j) pairs with near-perfect correlation
    dup_pairs = np.argwhere(ck_cs_corr > _CORR_THRESH)

    if dup_pairs.size == 0:
        return (np.array([], dtype=int),
                np.array([], dtype=int),
                True)  # rank-deficient but no ck-cs dups

    # Each ck should map to at most one cs partner (take the first)
    ck_dup_idx = []
    cs_dup_partner = []
    _seen_ck = set()
    for ck_i, cs_j in dup_pairs:
        if ck_i not in _seen_ck:
            ck_dup_idx.append(ck_i)
            cs_dup_partner.append(cs_j)
            _seen_ck.add(ck_i)

    return (np.array(ck_dup_idx, dtype=int),
            np.array(cs_dup_partner, dtype=int),
            True)


# ===================================================================
# Adaptive Neighborhood Selection & Assimilation
# ===================================================================

def _pivoted_cholesky_select(cov_dd, relevance, n_select,
                             min_pivot_ratio=1e-8):
    """Relevance-weighted pivoted Cholesky decomposition for selecting
    a linearly independent, informative subset of data points.

    The algorithm greedily picks the data point whose *residual variance*
    (diagonal of the Schur complement) times *relevance weight* is
    largest, then deflates.  This naturally avoids selecting near-
    duplicate points because deflation drives their residual variance to
    zero.

    Parameters
    ----------
    cov_dd : (n, n) ndarray
        Covariance matrix of the candidate data neighbourhood
        (hard *and* soft combined, ordered [ch; cs]).
    relevance : (n,) ndarray
        Per-point relevance score (e.g. |cov(ck, d_i)| averaged over
        ck batch).  Must be >= 0.
    n_select : int
        Maximum number of points to retain.
    min_pivot_ratio : float, optional
        Stop early when the best weighted pivot drops below
        ``min_pivot_ratio * first_pivot``.

    Returns
    -------
    selected : 1-D int array of length <= n_select
        Indices (into ``cov_dd``) of the selected data points, in
        selection order.
    """
    n = cov_dd.shape[0]
    n_select = min(n_select, n)
    if n_select <= 0:
        return np.array([], dtype=int)

    # Work on a copy – we modify the diagonal during deflation
    diag = np.diag(cov_dd).copy()
    diag = np.maximum(diag, 0.0)          # safety: clip any -ε artefact
    relevance = np.asarray(relevance, dtype=float).copy()
    relevance = np.maximum(relevance, 0.0)

    L = np.zeros((n, n_select), dtype=float)  # Cholesky factor (col-major)
    selected = np.empty(n_select, dtype=int)
    remaining = np.ones(n, dtype=bool)

    first_pivot = None

    for j in range(n_select):
        # Weighted residual variance
        score = diag * relevance
        score[~remaining] = -1.0

        best = int(np.argmax(score))
        best_val = score[best]

        if first_pivot is None:
            first_pivot = best_val if best_val > 0 else 1.0
        if best_val < min_pivot_ratio * first_pivot:
            selected = selected[:j]
            break

        selected[j] = best
        remaining[best] = False

        pivot_std = np.sqrt(max(diag[best], 1e-300))
        L[best, j] = pivot_std

        # Deflate: compute column j of the Cholesky factor
        col = cov_dd[best, :].copy()
        if j > 0:
            col -= L[:, :j] @ L[best, :j]
        col /= pivot_std
        L[:, j] = col
        # Update residual diagonal
        diag -= col ** 2
        diag = np.maximum(diag, 0.0)

    return selected


def _build_merge_map(cov_dd, selected, n_data):
    """Map each discarded point to its most-correlated retained partner.

    Parameters
    ----------
    cov_dd : (n, n) ndarray
        Covariance matrix of the full neighbourhood (before selection).
    selected : 1-D int array
        Indices of retained points.
    n_data : int
        Total number of candidate data points (``n``).

    Returns
    -------
    merge_map : dict  {int -> int}
        ``merge_map[discarded_idx] = retained_idx``.
        Only contains entries for *discarded* points.
    """
    all_idx = np.arange(n_data)
    sel_set = set(selected)
    discarded = np.array([i for i in all_idx if i not in sel_set], dtype=int)

    merge_map = {}
    if discarded.size == 0 or selected.size == 0:
        return merge_map

    # Use absolute correlation to handle negative cross-covariance
    sel_arr = np.asarray(selected)
    for d_i in discarded:
        # Covariance between d_i and each retained point
        cov_row = np.abs(cov_dd[d_i, sel_arr])
        best_partner = sel_arr[int(np.argmax(cov_row))]
        merge_map[int(d_i)] = int(best_partner)

    return merge_map


def _assimilate_neighborhood(picked_ck, picked_ch, picked_cs,
                             picked_zh, picked_zs, picked_covmat,
                             nhmax_nsmax_total=None,
                             min_pivot_ratio=1e-8, min_merge_var=1e-6):
    """Adaptive neighbourhood selection + information-preserving assimilation.

    1. **Select** the most informative, linearly independent subset of
       data via relevance-weighted pivoted Cholesky.
    2. **Assimilate** each discarded point into its most-correlated
       retained partner so that observational information is *not* lost.

    Assimilation rules (type-aware)
    -------------------------------
    * **Hard + Hard** → Soft Gaussian: mean = average of two values,
      var = (difference / 2)² + ``min_merge_var``.
    * **Hard + Soft** → The hard datum dominates (informational certainty);
      the soft datum is dropped to avoid double-counting.
    * **Soft + Soft** → Equal-weight Gaussian mixture approximated as a
      single Gaussian via moment matching.  Works for all soft types
      (histogram, linear, Gaussian) because ``proba2stat`` extracts
      (mean, var) from any of them.

    Parameters
    ----------
    picked_ck : (nk_batch, d) ndarray  – estimation points for this batch.
    picked_ch : (nh_local, d) ndarray or None – local hard-data coords.
    picked_cs : (ns_local, d) ndarray or None – local soft-data coords.
    picked_zh : (nh_local, 1) ndarray or None – hard observations.
    picked_zs : list of soft-data tuples or None
        Each element is ``(pdf_type, nl, limi, probdens)`` for histogram/
        linear, or ``(pdf_type, mean, var)`` for Gaussian (type 10).
    picked_covmat : (nk+nh+ns, nk+nh+ns) ndarray or None
        Joint covariance matrix ordered [ck; ch; cs].
    nhmax_nsmax_total : int or None
        Budget for total retained data points (hard + soft).
        If None, defaults to nh + ns (no budget limit, only linear-
        independence pruning).
    min_pivot_ratio : float
        Passed to ``_pivoted_cholesky_select``.
    min_merge_var : float
        Minimum variance injected when merging two hard data points
        into a soft Gaussian.

    Returns
    -------
    new_ck, new_ch, new_cs, new_zh, new_zs, new_covmat
        Possibly smaller arrays after assimilation.  ``new_ch`` may
        shrink and ``new_cs`` / ``new_zs`` may grow when Hard+Hard
        merges create new soft data.
    """
    nk = picked_ck.shape[0]
    nh = picked_ch.shape[0] if picked_ch is not None else 0
    ns = picked_cs.shape[0] if picked_cs is not None else 0
    n_data = nh + ns

    # Nothing to do if already tiny
    if n_data <= 1:
        return (picked_ck, picked_ch, picked_cs,
                picked_zh, picked_zs, picked_covmat)

    # ---- 1. Build relevance vector (|mean cov(ck, d_i)|) ----
    if picked_covmat is not None:
        cov_k_d = picked_covmat[:nk, nk:]          # (nk, n_data)
        relevance = np.abs(cov_k_d).mean(axis=0)   # (n_data,)
        cov_dd = picked_covmat[nk:, nk:]            # (n_data, n_data)
    else:
        # If no precomputed covmat we cannot do Cholesky; just pass through
        return (picked_ck, picked_ch, picked_cs,
                picked_zh, picked_zs, picked_covmat)

    # ---- 2. Budget ----
    if nhmax_nsmax_total is None:
        budget = n_data
    else:
        budget = min(int(nhmax_nsmax_total), n_data)

    # ---- 3. Pivoted Cholesky selection ----
    selected = _pivoted_cholesky_select(
        cov_dd, relevance, budget, min_pivot_ratio=min_pivot_ratio)

    if selected.size >= n_data:
        # Everything kept – nothing to do
        return (picked_ck, picked_ch, picked_cs,
                picked_zh, picked_zs, picked_covmat)

    # ---- 4. Build merge map ----
    merge_map = _build_merge_map(cov_dd, selected, n_data)

    # ---- 5. Assimilate ----
    # Represent every datum in a unified indexing: 0..nh-1 = hard,
    # nh..nh+ns-1 = soft.
    # After assimilation we collect final_hard and final_soft lists.

    # Pre-extract (mean, var) for every soft datum
    soft_stats = []  # list of (mean_float, var_float) for each soft point
    for si in range(ns):
        zsi = picked_zs[si] if picked_zs is not None else None
        if zsi is None:
            soft_stats.append((np.nan, np.nan))
            continue
        ptype = get_standard_soft_pdf_type(zsi[0])
        if ptype in (1, 2):
            # histogram / linear → moment-match via proba2stat
            _nl = np.array(zsi[1]).reshape(1, 1)
            _limi = np.array(zsi[2]).reshape(1, -1)
            _pd = np.array(zsi[3]).reshape(1, -1)
            _m, _v = proba2stat(ptype, _nl, _limi, _pd)
            soft_stats.append((float(np.asarray(_m).flat[0]),
                               float(np.asarray(_v).flat[0])))
        elif ptype == 10:
            soft_stats.append((float(np.asarray(zsi[1]).flat[0]),
                               float(np.asarray(zsi[2]).flat[0])))
        else:
            soft_stats.append((np.nan, np.nan))

    # Buckets: for each retained index, collect what gets merged into it
    retained_set = set(int(s) for s in selected)
    merge_buckets = {int(s): [] for s in selected}
    for d_from, d_to in merge_map.items():
        merge_buckets[d_to].append(d_from)

    # Final data lists
    final_ch_list = []    # coordinates
    final_zh_list = []    # observations
    final_cs_list = []    # coordinates
    final_zs_list = []    # soft data tuples
    retained_data_indices = []   # original data indices for covmat slicing

    def _is_hard(idx):
        return idx < nh

    def _get_hard_val(idx):
        return float(picked_zh[idx].flat[0])

    def _get_hard_coord(idx):
        return picked_ch[idx]

    def _get_soft_coord(idx):
        return picked_cs[idx - nh]

    def _get_soft_mean_var(idx):
        return soft_stats[idx - nh]

    def _get_soft_tuple(idx):
        return picked_zs[idx - nh]

    def _make_gaussian_zs(mean, var):
        """Create a Gaussian soft datum tuple (type=10, mean, var)."""
        return (10, float(mean), max(float(var), min_merge_var))

    for ret_idx in sorted(retained_set):
        to_merge = merge_buckets[ret_idx]

        if not to_merge:
            # --- No merges: keep datum as-is ---
            if _is_hard(ret_idx):
                final_ch_list.append(_get_hard_coord(ret_idx))
                final_zh_list.append(_get_hard_val(ret_idx))
            else:
                final_cs_list.append(_get_soft_coord(ret_idx))
                final_zs_list.append(_get_soft_tuple(ret_idx))
            retained_data_indices.append(ret_idx)
            continue

        # --- Merge required ---
        # Collect all participants: retained + discarded
        participants = [ret_idx] + to_merge

        n_hard_parts = sum(1 for p in participants if _is_hard(p))
        n_soft_parts = len(participants) - n_hard_parts

        if n_hard_parts == len(participants):
            # ---- All Hard → convert to Soft Gaussian ----
            vals = np.array([_get_hard_val(p) for p in participants])
            merge_mean = float(np.mean(vals))
            merge_var = float(np.var(vals)) + min_merge_var
            # Use coordinate of the retained point
            final_cs_list.append(_get_hard_coord(ret_idx))
            final_zs_list.append(_make_gaussian_zs(merge_mean, merge_var))
            retained_data_indices.append(ret_idx)

        elif n_hard_parts > 0 and n_soft_parts > 0:
            # ---- Hard + Soft mixture → Hard dominates ----
            # Keep the hard datum as-is; soft information is approximately
            # captured by the hard observation.  (Hard is delta-function,
            # which overwhelms any soft PDF in the posterior.)
            hard_parts = [p for p in participants if _is_hard(p)]
            # If multiple hard data merge here, average them into soft
            if len(hard_parts) == 1:
                h_idx = hard_parts[0]
                final_ch_list.append(_get_hard_coord(h_idx))
                final_zh_list.append(_get_hard_val(h_idx))
            else:
                vals = np.array([_get_hard_val(p) for p in hard_parts])
                merge_mean = float(np.mean(vals))
                merge_var = float(np.var(vals)) + min_merge_var
                final_cs_list.append(_get_hard_coord(hard_parts[0]))
                final_zs_list.append(_make_gaussian_zs(merge_mean, merge_var))
            retained_data_indices.append(ret_idx)

        else:
            # ---- All Soft → Gaussian mixture moment matching ----
            means_vars = [_get_soft_mean_var(p) for p in participants]
            # Filter out NaN entries
            valid = [(m, v) for m, v in means_vars
                     if np.isfinite(m) and np.isfinite(v) and v > 0]
            if not valid:
                # Fallback: keep retained datum as-is
                final_cs_list.append(_get_soft_coord(ret_idx))
                final_zs_list.append(_get_soft_tuple(ret_idx))
                retained_data_indices.append(ret_idx)
                continue
            # Equal-weight mixture → moment matching
            mix_means = np.array([m for m, _ in valid])
            mix_vars = np.array([v for _, v in valid])
            w = 1.0 / len(valid)
            merged_mean = float(np.sum(w * mix_means))
            merged_var = float(
                np.sum(w * (mix_vars + mix_means ** 2)) - merged_mean ** 2)
            merged_var = max(merged_var, min_merge_var)
            final_cs_list.append(_get_soft_coord(ret_idx))
            final_zs_list.append(_make_gaussian_zs(merged_mean, merged_var))
            retained_data_indices.append(ret_idx)

    # ---- 6. Rebuild arrays ----
    new_nh = len(final_ch_list)
    new_ns = len(final_cs_list)
    new_ch = (np.array(final_ch_list) if new_nh > 0 else None)
    new_cs = (np.array(final_cs_list) if new_ns > 0 else None)
    new_zh = (np.array(final_zh_list).reshape(-1, 1) if new_nh > 0
              else None)
    new_zs = final_zs_list if new_ns > 0 else None

    # ---- 7. Rebuild covmat ----
    # We keep the rows/cols of the retained data points.
    # New soft data created from merges re-use the retained point's row
    # (since they occupy the same spatial location → same covariance).
    retained_data_indices = np.array(retained_data_indices, dtype=int)
    # Map back to original covmat indices: ck part stays, data part maps
    keep_ck = np.arange(nk)
    keep_data = nk + retained_data_indices  # offset into full covmat
    keep_all = np.concatenate([keep_ck, keep_data])
    new_covmat = picked_covmat[np.ix_(keep_all, keep_all)]

    # The new covmat ordering is [ck; mixed_h_s].
    # We need to reorder it to [ck; new_ch; new_cs].
    # Since we appended hard then soft in order, this is already correct
    # *if* the output ch/cs arrays match the retained_data_indices order.
    # But merges can change hard→soft.  We need a proper reorder.

    # Build the mapping: which of the retained_data_indices are hard
    # and which are soft in the *output*.
    # After the loop above, data in new_covmat[nk:] corresponds to
    # retained_data_indices order (which is sorted).
    # We just need to know: in the output, which positions are hard and
    # which are soft (they may have changed due to Hard+Hard→Soft merges).
    # The caller (_bme_posterior_moments) expects covmat ordered as
    # [ck, ch, cs].  We've built ch_list first, then cs_list, so let's
    # reorder the data block of the covmat accordingly.

    # The retained_data_indices entries that contributed to final_ch_list
    # come first, then those that contributed to final_cs_list.
    # We tracked this via the order we appended:
    # We rebuild a permutation.
    perm_data = []  # indices into retained_data_indices
    _cur = 0
    # We go through retained_data_indices in order, same order as the
    # final_*_list construction (sorted retained_set order).
    # So perm_data is just 0..len(retained_data_indices)-1 split into
    # hard-slots first, soft-slots second.
    # Actually, we need to track this more carefully.

    # Simpler approach: since we're iterating retained_set in sorted order
    # and appending to ch or cs lists, let's just record which output
    # slot each retained_data_index maps to.
    # Re-derive: go through sorted retained, flag H or S.
    _h_perm = []
    _s_perm = []
    for i, ret_idx in enumerate(sorted(retained_set)):
        to_merge = merge_buckets[ret_idx]
        participants = [ret_idx] + to_merge
        n_hard_parts = sum(1 for p in participants if _is_hard(p))
        n_soft_parts = len(participants) - n_hard_parts
        if not to_merge:
            if _is_hard(ret_idx):
                _h_perm.append(i)
            else:
                _s_perm.append(i)
        elif n_hard_parts == len(participants):
            _s_perm.append(i)   # became soft
        elif n_hard_parts > 0 and n_soft_parts > 0:
            hard_parts = [p for p in participants if _is_hard(p)]
            if len(hard_parts) == 1:
                _h_perm.append(i)
            else:
                _s_perm.append(i)   # became soft
        else:
            _s_perm.append(i)

    data_perm = np.array(_h_perm + _s_perm, dtype=int)
    # Reorder the data block of new_covmat
    n_kept = retained_data_indices.size
    data_block_idx = nk + data_perm
    full_perm = np.concatenate([np.arange(nk), data_block_idx])
    new_covmat = new_covmat[np.ix_(full_perm, full_perm)]

    _msg_parts = []
    if nh != new_nh or ns != new_ns:
        _msg_parts.append(
            "nh: {} -> {}, ns: {} -> {}".format(nh, new_nh, ns, new_ns))
    if n_data - int(selected.size) > 0:
        _msg_parts.append("{} points discarded, {} merged".format(
            n_data - int(selected.size), len(merge_map)))
    if _msg_parts:
        print("[STARBME] Assimilation: " + "; ".join(_msg_parts))

    return (picked_ck, new_ch, new_cs, new_zh, new_zs, new_covmat)


def __check_normalized_constant(NC, options):
    if NC == 0:
        print('Warning NC is equals to zero.')
        if options['debug']:
            pass
    elif np.isnan(NC):
        print('NC is equals to NaN.')
        if options['debug']:
            pass
    elif np.isinf(NC):
        print('NC is equals to Inf.')
        if options['debug']:
            pass
    return NC

def _process_estimation_chunk(
    ck_idx_piece, picked_ch, picked_cs, picked_zh, picked_zs,
    ck, ch, cs, zh, zs,
    covmat, covmodel, covparam, order, options,
    general_knowledge, has_user_defined_general_knowledge,
    pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
    ck_cov_output, ch_idx, cs_idx, nk, nh):
    """Process a single chunk of estimation points for parallel dispatch.

    This is a standalone function that can be called from a thread pool.
    It performs assimilation, order picking, and BME estimation for one
    chunk of estimation locations sharing the same neighborhood.

    Returns
    -------
    (ck_idx_piece, picked_mvs) : tuple
        The original index array and the estimated moments (4-column).
    """
    picked_ck = ck[ck_idx_piece, :]
    if covmat is not None:
        covidx = np.hstack(
            (ck_idx_piece, nk + ch_idx, nk + nh + cs_idx))
        picked_covmat = covmat[np.ix_(covidx, covidx)]
    else:
        picked_covmat = covmat

    # --- Adaptive neighbourhood assimilation ---
    if picked_covmat is not None:
        try:
            (picked_ck, picked_ch, picked_cs,
             picked_zh, picked_zs,
             picked_covmat) = _assimilate_neighborhood(
                picked_ck, picked_ch, picked_cs,
                picked_zh, picked_zs, picked_covmat)
        except Exception as _assim_err:
            import warnings as _w
            _w.warn(
                "[STARBME] Assimilation skipped: {}"
                .format(_assim_err))

    # picked order for specified general knowledge
    if has_user_defined_general_knowledge:
        _nh_a = picked_ch.shape[0] if picked_ch is not None else 0
        _ns_a = picked_cs.shape[0] if picked_cs is not None else 0
        _nk_a = picked_ck.shape[0]
        order_idx = np.hstack(
            (ck_idx_piece, nk + ch_idx, nk + nh + cs_idx))
        if order_idx.shape[0] == _nk_a + _nh_a + _ns_a:
            picked_order = order[order_idx, :]
        else:
            picked_order = order[ck_idx_piece, :]
            _remaining = _nh_a + _ns_a
            if _remaining > 0 and order.shape[0] > nk:
                _data_order = order[nk:, :]
                _mean_row = np.mean(_data_order, axis=0,
                                    keepdims=True)
                picked_order = np.vstack([
                    picked_order,
                    np.tile(_mean_row, (_remaining, 1))])
    else:
        picked_order = order

    try:
        picked_mvs = _bme_posterior_moments(
            picked_ck, picked_ch, picked_cs,
            picked_zh, picked_zs,
            covmodel, covparam, picked_covmat,
            picked_order, options, general_knowledge,
            pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
            None, ck_cov_output)  # gui_args=None for thread safety
    except (np.linalg.LinAlgError, TypeError, ValueError) as e:
        import warnings
        _nh = picked_ch.shape[0] if picked_ch is not None else 0
        _ns = picked_cs.shape[0] if picked_cs is not None else 0
        warn_msg = (
            "BME estimation failed for estimation location(s):\n"
            "  ck = {ck}\n"
            "  Number of hard data in neighborhood: {nh}\n"
            "  Number of soft data in neighborhood: {ns}\n"
            "  Error: {err}\n"
            "This may indicate a singular or ill-conditioned "
            "covariance matrix."
        ).format(ck=picked_ck, nh=_nh, ns=_ns, err=str(e))
        warnings.warn(warn_msg)
        picked_mvs = np.empty((picked_ck.shape[0], 4))
        picked_mvs[:] = np.nan

    # Pad Gaussian-path results (3-col) to 4-col with delta_mu=0
    if (isinstance(picked_mvs, np.ndarray)
            and picked_mvs.ndim == 2
            and picked_mvs.shape[1] == 3):
        picked_mvs = np.hstack([
            picked_mvs,
            np.zeros((picked_mvs.shape[0], 1))])

    return (ck_idx_piece, picked_mvs)


# ======================================================================
# Prior Mean Helpers  (BMEpriorMean + backward-compat aliases)
# ======================================================================

def _kernel_smooth(c_query, c_data, z_data, bandwidth, kernel='gaussian',
                   truncate=False, fallback_mean=None):
    """Internal helper: kernel-smoothed mean at query locations.

    Parameters
    ----------
    c_query : (nq, nd)
    c_data  : (nd_pts, nd)
    z_data  : (nd_pts, 1)
    bandwidth : float  (must be > 0)
    kernel  : 'gaussian' | 'epanechnikov'
    truncate : bool  – zero out weights beyond bandwidth
    fallback_mean : float | None  – value used where all weights are zero;
                    defaults to the mean of z_data

    Returns
    -------
    mu : (nq, 1)
    """
    from scipy.spatial.distance import cdist

    if fallback_mean is None:
        fallback_mean = float(z_data.mean())

    D = cdist(c_query, c_data)                    # (nq, n_data)

    if kernel == 'gaussian':
        # Clip exponent to prevent catastrophic underflow to 0.0:
        # exp(-500) ~ 7e-218, still a valid float64 and the ratio
        # (W @ z) / W_sum gives correct result.  True underflow only
        # occurs beyond exp(-709) ≈ 5e-308 per term, impossible here.
        u2 = np.clip((D / bandwidth) ** 2, 0.0, 1400.0)
        W  = np.exp(-0.5 * u2)
    elif kernel == 'epanechnikov':
        u = D / bandwidth
        W = np.where(u < 1.0, 0.75 * (1.0 - u ** 2), 0.0)
    else:
        raise ValueError("kernel must be 'gaussian' or 'epanechnikov'")

    if truncate:
        W[D > bandwidth] = 0.0

    W_sum = W.sum(axis=1, keepdims=True)          # (nq, 1)

    # Rows where ALL weights are truly zero (can happen with Epanechnikov
    # truncation or extremely far-field grid points).
    zero_mask = (W_sum == 0.0).ravel()

    # Safe division: set zero denominators to 1.0 temporarily
    W_sum_safe = np.where(zero_mask[:, None], 1.0, W_sum)
    mu = (W @ z_data) / W_sum_safe                # (nq, 1)

    # Replace sparse locations with the fallback mean
    if zero_mask.any():
        mu[zero_mask, 0] = fallback_mean

    return mu


def _loocv_bandwidth(c_data, z_data, bandwidth_range=None, n_bw=30,
                     kernel='gaussian', verbose=False):
    """Leave-one-out cross-validation to select optimal kernel bandwidth.

    Uses the LOO identity: for each candidate bandwidth h, exclude each data
    point from its own prediction by setting the self-distance to infinity.
    Selects the bandwidth minimising the mean squared LOO prediction error.

    Parameters
    ----------
    c_data : (n, nd)  Data locations.
    z_data : (n, 1)   Data values.
    bandwidth_range : array-like of float, optional
        Candidate bandwidths to evaluate.  If ``None``, 30 log-spaced values
        spanning from ``0.5 × 10th-pct NN distance`` to
        ``0.5 × max pairwise distance`` are used.
    n_bw : int, default 30
        Number of candidates when *bandwidth_range* is auto-generated.
    kernel : {'gaussian', 'epanechnikov'}
    verbose : bool  – print CV error table

    Returns
    -------
    best_bw : float
    cv_errors : list of (bandwidth, MSE) tuples
    """
    from scipy.spatial.distance import cdist

    n = len(z_data)
    D = cdist(c_data, c_data)
    np.fill_diagonal(D, np.inf)    # exclude self

    if bandwidth_range is None:
        nn_dists = D.min(axis=1)
        bw_min = max(float(np.percentile(nn_dists, 10)) * 0.5, 1e-6)
        # max pairwise (exclude inf diagonal)
        finite_D = D[D != np.inf]
        bw_max = float(np.max(finite_D)) * 0.5 if len(finite_D) > 0 else bw_min * 100
        bandwidth_range = np.exp(np.linspace(np.log(bw_min), np.log(bw_max), n_bw))

    cv_errors = []
    for bw in bandwidth_range:
        if kernel == 'gaussian':
            u2 = np.clip((D / bw) ** 2, 0.0, 1400.0)
            W  = np.exp(-0.5 * u2)
        elif kernel == 'epanechnikov':
            u = D / bw
            W = np.where(u < 1.0, 0.75 * (1.0 - u ** 2), 0.0)
        else:
            raise ValueError("kernel must be 'gaussian' or 'epanechnikov'")

        W_sum = W.sum(axis=1, keepdims=True)
        zero_mask = (W_sum == 0.0).ravel()
        W_sum_safe = np.where(zero_mask[:, None], 1.0, W_sum)
        mu_loo = (W @ z_data) / W_sum_safe
        mu_loo[zero_mask, 0] = float(z_data.mean())   # fallback for isolated pts

        mse = float(np.mean((mu_loo.ravel() - z_data.ravel()) ** 2))
        cv_errors.append((float(bw), mse))

    best_bw, best_mse = min(cv_errors, key=lambda x: x[1])
    if verbose:
        print("LOOCV bandwidth search:")
        for bw, mse in cv_errors:
            marker = " <-- optimal" if bw == best_bw else ""
            print(f"  bw={bw:10.1f}  MSE={mse:.4f}{marker}")
    return best_bw, cv_errors


def BMEpriorMean(
        ck, ch, zh, cs=None, zs=None,
        method='kernel',
        bandwidth=None,
        kernel='gaussian',
        truncate=False,
        use_soft_data=False,
        cv_bandwidth=False,
        n_cv_bandwidths=30,
        cv_bandwidth_range=None,
        cv_verbose=False,
        covmodel=None, covparam=None,
        order_trend=0):
    """Compute a prior mean surface for BME at all estimation and data locations.

    Returns a stacked ``(nk + nh + ns, 1)`` array ready to pass directly as
    the ``order`` argument to :func:`BMEPosteriorMoments`.  This replaces the
    scalar ``order=np.nan`` (zero prior) or ``order=0`` (global mean prior)
    with a spatially-varying, data-driven estimate — giving BME behaviour
    closer to Ordinary Kriging in regions without nearby data.

    Parameters
    ----------
    ck : ndarray, shape (nk, nd)
        Estimation locations.
    ch : ndarray, shape (nh, nd)
        Hard-data locations.
    zh : ndarray, shape (nh,) or (nh, 1)
        Hard-data observed values.
    cs : ndarray, shape (ns, nd), optional
        Soft-data locations.
    zs : list of per-location soft-data tuples, optional
        Standard stamps format ``(softpdftype, nl, limi, probdens)``.
        Required (with *cs*) for methods that need soft data expected values.
    method : str, default ``'kernel'``
        Prior mean estimation method.  Choices:

        ``'kernel'``
            Gaussian (or Epanechnikov) kernel smoother.  Data beyond
            *bandwidth* still contribute with exponentially decaying weight,
            so the smoother never produces NaN even at map edges.  By default
            only hard data are used (``use_soft_data=False``), avoiding the
            upward bias that arises when interval soft data midpoints are
            higher than the true hard-data mean.

        ``'kernel_truncate'``
            Same as ``'kernel'`` but the kernel is **truncated to zero beyond
            bandwidth** (wraps the existing
            :mod:`stamps.stest.kernelsmoothing_truncate` module).  Estimates
            are NaN at locations with no data within *bandwidth*; fill those
            with the global hard-data mean.

        ``'idw'``
            Inverse-distance weighting (power = 2).  Fast; exhibits sharper
            transitions near data clusters than Gaussian kernel.

        ``'global_hard'``
            Constant equal to the arithmetic mean of the hard data *zh* only.
            This is the recommended replacement for ``order=0`` when the soft
            data midpoints are biased relative to the hard data.

        ``'global'``
            Constant equal to the arithmetic mean of hard *and* soft data
            expected values (original ``order=0`` behaviour).  Can be biased
            high when interval soft data span regions with higher true values.

        ``'localmeanBME'``
            GLS regression using the full spatial covariance matrix as the
            weight (wraps :func:`stamps.stest.localmeanBME.localmeanBME`).
            Requires *covmodel* and *covparam*.  Returns the best linear
            unbiased estimate of the constant (``order_trend=0``) or linear
            (``order_trend=1``) trend at every location, accounting for
            spatial autocorrelation.  Effectively the same as computing the
            Ordinary-Kriging mean but without the neighbourhood truncation.

    bandwidth : float, optional
        Characteristic smoothing length in the same distance units as *ck/ch*.
        Rules of thumb:

        * Set to ≈1–2× the covariance **range** for smooth trends.
        * Set to ≈0.5× the range for tighter locality.
        * Leave as ``None`` (auto) to use a robust default:
          ``max(1.5 × median-NN-distance, 0.1 × study-area-extent)``.
          This prevents the bandwidth from being excessively narrow when
          data contain closely-spaced survey transects.

        Ignored by ``'global_hard'``, ``'global'``, and ``'localmeanBME'``.
        Overridden when *cv_bandwidth=True*.
    kernel : {'gaussian', 'epanechnikov'}, default ``'gaussian'``
        Kernel shape; only used when *method* is ``'kernel'`` or
        ``'kernel_truncate'``.
    truncate : bool, default ``False``
        If ``True``, forces the truncated kernel regardless of *method*
        (equivalent to ``method='kernel_truncate'``).
    use_soft_data : bool, default ``False``
        If ``True``, the **expected values** of soft data PDFs are blended into
        the kernel smoother alongside the hard data.  This can slightly improve
        coverage in data-sparse areas but risks biasing the prior upward if
        soft data intervals are systematically offset from the hard data.
        Has no effect on ``'global_hard'``, ``'localmeanBME'``.
    cv_bandwidth : bool, default ``False``
        If ``True``, the bandwidth is selected automatically by
        **leave-one-out cross-validation (LOOCV)** on the hard data.  The
        bandwidth minimising the mean squared LOO prediction error is used.
        This is slower than the analytic auto-bandwidth but produces a more
        statistically optimal smoothing length and is recommended when:

        * The data contain irregularly-spaced clusters (e.g. survey transects
          mixed with scattered points), which can fool the NN-distance heuristic.
        * A principled, objective bandwidth is needed.

        Overrides *bandwidth* when active.
    n_cv_bandwidths : int, default 30
        Number of candidate bandwidths evaluated during LOOCV.
    cv_bandwidth_range : array-like of float, optional
        Explicit list/array of bandwidth candidates for LOOCV.  If ``None``,
        30 log-spaced values from ``0.5 × 10th-pct NN distance`` to
        ``0.5 × max pairwise distance`` are auto-generated.
    cv_verbose : bool, default ``False``
        Print the LOOCV error table to stdout.
    covmodel : list of str, optional
        Covariance model names; required for ``method='localmeanBME'``.
    covparam : list of tuples, optional
        Covariance parameters; required for ``method='localmeanBME'``.
    order_trend : {0, 1, np.nan}, default ``0``
        Regression trend order for ``method='localmeanBME'``:
        0 = constant mean, 1 = linear trend, np.nan = zero mean.

    Returns
    -------
    order : ndarray, shape (nk + nh + ns, 1)
        Prior mean at each location, stacked as ``[ck, ch, cs]``.
        Pass directly as ``order=BMEpriorMean(...)`` to
        :func:`BMEPosteriorMoments`.

    Notes
    -----
    **Why not just use order=0?**
    ``order=0`` computes the prior mean as the *arithmetic mean of all hard
    and soft data*.  When interval soft data midpoints are higher than the
    true hard-data mean (e.g., intervals spanning 20–40% while precise
    measurements average 15%), the global prior is biased high.  In data-void
    areas the BME estimate then reverts to this elevated prior — producing
    maps that appear *higher at the edges* than in well-sampled interiors.
    Using ``method='global_hard'`` or a kernel smoother over hard data only
    eliminates this bias.

    **Auto-bandwidth robustness**
    The default auto-bandwidth uses ``max(1.5 × median-NN, 0.1 × extent)``,
    where *extent* is the largest dimension of the hard-data bounding box.
    The ``0.1 × extent`` floor prevents the bandwidth from becoming
    excessively narrow (< 5% of domain size) when data contain closely-spaced
    survey points that pull the median NN distance down.  The LOOCV option
    provides a fully data-driven, objective bandwidth.

    **Relationship to Ordinary Kriging**
    OK implicitly estimates a local mean via the unbiasedness constraint.
    ``method='kernel'`` approximates this by kernel-smoothing the hard data.
    ``method='localmeanBME'`` goes further: it uses GLS to account for
    spatial autocorrelation in the regression, giving the best linear
    unbiased estimate of the mean trend.

    Examples
    --------
    Basic usage — kernel-smoothed prior from hard data only:

    >>> order_arr = BMEpriorMean(ck, ch, zh, cs=cs, zs=zs_list,
    ...                               method='kernel', bandwidth=5000.0)
    >>> result = BMEPosteriorMoments(
    ...     ck, ch=ch, cs=cs, zh=zh, zs=zs_list,
    ...     covmodel=['exponentialC'], covparam=[(30.0, 8000.0)],
    ...     order=order_arr, nhmax=12, nsmax=4,
    ...     dmax=np.array([[15000.0]]))

    LOOCV bandwidth selection (data-driven optimal):

    >>> order_cv = BMEpriorMean(ck, ch, zh, cs=cs, zs=zs_list,
    ...                              method='kernel', cv_bandwidth=True,
    ...                              cv_verbose=True)

    GLS-based prior (most rigorous, needs covmodel/covparam):

    >>> order_gls = BMEpriorMean(ck, ch, zh, cs=cs, zs=zs_list,
    ...     method='localmeanBME',
    ...     covmodel=['exponentialC'], covparam=[(30.0, 8000.0)])
    """
    from scipy.spatial.distance import cdist

    # ------------------------------------------------------------------ #
    # Normalise inputs
    # ------------------------------------------------------------------ #
    ck  = np.atleast_2d(np.asarray(ck,  dtype=float))
    ch  = np.atleast_2d(np.asarray(ch,  dtype=float))
    zh  = np.asarray(zh, dtype=float).reshape(-1, 1)
    nk, nh = ck.shape[0], ch.shape[0]

    # Optional: extract soft data expected values
    cs_present = cs is not None and zs is not None and len(zs) > 0
    if cs_present:
        cs = np.atleast_2d(np.asarray(cs, dtype=float))
        ns = len(zs)
        zs_mean = np.empty((ns, 1))
        for i, zsi in enumerate(zs):
            if get_standard_soft_pdf_type(zsi[0]) == 10:   # Gaussian
                zs_mean[i] = float(np.asarray(zsi[1]).flat[0])
            else:
                mu_s, _ = proba2stat(
                    zsi[0],
                    np.atleast_2d(zsi[1]),
                    np.atleast_2d(zsi[2]),
                    np.atleast_2d(zsi[3]))
                zs_mean[i] = float(np.asarray(mu_s).flat[0])
    else:
        cs = np.zeros((0, ch.shape[1]))
        ns = 0
        zs_mean = np.zeros((0, 1))

    # Build query array: ck ∪ ch ∪ cs
    parts = [ck, ch]
    if ns > 0:
        parts.append(cs)
    c_query = np.vstack(parts)     # (nk+nh+ns, nd)
    n_query = c_query.shape[0]

    # ------------------------------------------------------------------ #
    # method='global_hard' or 'global'
    # ------------------------------------------------------------------ #
    if method in ('global_hard', 'global'):
        if method == 'global_hard':
            mu_val = float(zh.mean())
        else:   # 'global': hard + soft
            all_vals = [zh]
            if use_soft_data and ns > 0:
                all_vals.append(zs_mean)
            elif ns > 0:
                all_vals.append(zs_mean)  # keep original order=0 behaviour
            mu_val = float(np.vstack(all_vals).mean())
        return np.full((n_query, 1), mu_val)

    # ------------------------------------------------------------------ #
    # method='localmeanBME'
    # ------------------------------------------------------------------ #
    if method == 'localmeanBME':
        if covmodel is None or covparam is None:
            raise ValueError(
                "BMEpriorMean with method='localmeanBME' requires "
                "covmodel and covparam.")
        from ..general.coord2K import coord2K
        from ..stest.regression import regression
        from ..stest.designmatrix import designmatrix

        # Build full data set: hard (+ optionally soft)
        if cs_present and ns > 0:
            ms = zs_mean
            vs, _ = zip(*[
                proba2stat(
                    zsi[0],
                    np.atleast_2d(zsi[1]),
                    np.atleast_2d(zsi[2]),
                    np.atleast_2d(zsi[3]))
                for zsi in zs
            ])
            vs = np.array([float(np.asarray(v).flat[0]) for v in vs]).reshape(-1, 1)

            Khh = np.asarray(coord2K(ch, ch, covmodel, covparam)[0])
            Ksh = np.asarray(coord2K(cs, ch, covmodel, covparam)[0])
            Kss = np.asarray(coord2K(cs, cs, covmodel, covparam)[0])
            # Add soft data variances to diagonal of Kss
            Kss_aug = Kss + np.diag(vs.ravel())

            c_data = np.vstack([ch, cs])
            z_data = np.vstack([zh, ms])
            K_full = np.block([[Khh, Ksh.T],
                                [Ksh, Kss_aug]])
        else:
            Khh = np.asarray(coord2K(ch, ch, covmodel, covparam)[0])
            c_data = ch
            z_data = zh
            K_full = Khh

        # Single GLS regression on all data
        best, Vbest, mm, _ = regression(c_data, z_data, order_trend, K_full)

        if best.size == 0:
            # order_trend=nan → zero mean everywhere
            return np.zeros((n_query, 1))

        # Fitted values at data locations
        mu_data = mm  # (nh [+ ns], 1)
        mu_h = mu_data[:nh]
        mu_s = mu_data[nh:] if ns > 0 else np.zeros((0, 1))

        # Predict at ck
        X_ck, _ = designmatrix(ck, order_trend)
        mu_k = X_ck.dot(best)  # (nk, 1)

        parts = [mu_k, mu_h]
        if ns > 0:
            parts.append(mu_s)
        return np.vstack(parts).reshape(-1, 1)

    # ------------------------------------------------------------------ #
    # Kernel / IDW smoothers
    # ------------------------------------------------------------------ #
    from scipy.spatial.distance import cdist as _cdist

    # Build the data set for the smoother
    if use_soft_data and ns > 0:
        c_data = np.vstack([ch, cs])
        z_data = np.vstack([zh, zs_mean])
    else:
        c_data = ch
        z_data = zh

    z_hard_mean = float(zh.mean())   # fallback for data-void extrapolation

    # ---------------------------------------------------------------- #
    # Bandwidth selection
    # ---------------------------------------------------------------- #
    if cv_bandwidth and method not in ('idw',):
        # LOOCV: use only hard data for cross-validation
        bandwidth, _ = _loocv_bandwidth(
            ch, zh,
            bandwidth_range=cv_bandwidth_range,
            n_bw=n_cv_bandwidths,
            kernel=kernel,
            verbose=cv_verbose,
        )
    elif bandwidth is None:
        # Robust auto-bandwidth:
        #   max(1.5 × median NN distance,  0.1 × study-area extent)
        # The extent floor prevents an excessively narrow bandwidth when
        # closely-spaced survey transects pull the median NN distance down.
        D_nn = _cdist(ch, ch)
        np.fill_diagonal(D_nn, np.inf)
        nn_median = float(np.median(D_nn.min(axis=1)))
        extent = float(np.max(ch.max(axis=0) - ch.min(axis=0)))
        bw_nn  = nn_median * 1.5
        bw_ext = extent * 0.10          # 10 % of study area max dimension
        bandwidth = max(bw_nn, bw_ext, 1e-10)

    # ---------------------------------------------------------------- #
    # IDW: separate path (bandwidth not used)
    # ---------------------------------------------------------------- #
    if method == 'idw':
        from scipy.spatial.distance import cdist as _cdist2
        D = _cdist2(c_query, c_data)
        eps = 1e-10
        W   = 1.0 / (D ** 2 + eps)
        W_sum = W.sum(axis=1, keepdims=True)
        # IDW weights are never truly zero (eps guard), so simple division
        mu_all = (W @ z_data) / W_sum
        return mu_all.reshape(-1, 1)

    # ---------------------------------------------------------------- #
    # Gaussian / Epanechnikov kernel smoother
    # ---------------------------------------------------------------- #
    if method not in ('kernel', 'kernel_truncate'):
        raise ValueError(
            f"method must be one of: 'kernel', 'kernel_truncate', 'idw', "
            f"'global_hard', 'global', 'localmeanBME'.  Got: '{method}'")

    use_truncate = truncate or method == 'kernel_truncate'

    mu_all = _kernel_smooth(
        c_query, c_data, z_data,
        bandwidth=bandwidth,
        kernel=kernel,
        truncate=use_truncate,
        fallback_mean=z_hard_mean,
    )

    return mu_all.reshape(-1, 1)


def compute_bme_local_prior(
        ck, ch, cs=None, zh=None, zs=None,
        bandwidth=None, method='kernel', kernel='gaussian',
        cv_bandwidth=False):
    """Backward-compatible alias for :func:`BMEpriorMean`.

    Calls ``BMEpriorMean(method='kernel', use_soft_data=True, ...)``
    to preserve the original behaviour (soft data expected values blended
    into the kernel smoother).

    .. deprecated::
        Prefer :func:`BMEpriorMean` with explicit ``use_soft_data``
        and ``method`` arguments.
    """
    return BMEpriorMean(
        ck=ck, ch=ch, zh=zh if zh is not None else np.zeros((ch.shape[0], 1)),
        cs=cs, zs=zs,
        method=method if method in ('kernel', 'kernel_truncate', 'idw')
               else 'kernel',
        bandwidth=bandwidth,
        kernel=kernel,
        use_soft_data=True,   # old default: blend soft data
        cv_bandwidth=cv_bandwidth,
    )


# Backward-compatible name kept for any code written against the old API.
compute_bme_prior = BMEpriorMean


def BMEPosteriorMoments(
    ck, ch=None, cs=None, zh=None, zs=None,
    covmodel=None, covparam=None, covmat=None,
    order=np.nan, options=None,
    nhmax=None, nsmax=None, dmax=None,
    general_knowledge='gaussian',
    #  specific_knowledge='unknown',  
    pdfk=None,pdfh=None,pdfs=None,hk_k=None,hk_h=None,hk_s=None,
    gui_args=None, ck_cov_output=False, n_workers=1):
    """Compute the BME posterior mean, variance, and third central moment at all estimation locations.

    This is the primary estimation function of the stamps BME engine.  For
    each estimation location in *ck* it evaluates the Bayesian Maximum Entropy
    (BME) posterior distribution conditioned on all available hard and soft
    data, and returns three summary statistics.

    The function selects one of two internal computation paths:

    * **Analytical Gaussian path** (fast) — used whenever every soft datum is
      of type 10 (Gaussian PDF), or when ``options['soft_approx_gaussian']``
      is ``True``, or when more than 20 non-Gaussian soft data are present in
      the neighbourhood (automatic threshold).  Reduces to Simple Kriging or
      Ordinary Kriging depending on *order*.
    * **QMC Importance-Sampling path** (general) — used for non-Gaussian soft
      data (interval / histogram / linear PDFs).  Draws Quasi-Monte Carlo
      samples from the conditional Gaussian prior, weights them by the soft
      likelihood, and estimates the normalised posterior moments.  If the QMC
      normalisation constant is numerically unstable, a per-location Gaussian
      fallback is applied automatically.

    Parameters
    ----------
    ck : ndarray, shape (nk, nd)
        Estimation locations.  ``nd`` is the number of coordinate dimensions:
        ``nd=2`` for purely spatial problems (easting, northing) or ``nd=3``
        for space-time problems (easting, northing, time).  The last column is
        treated as the time axis when ``nd=3``.
    ch : ndarray, shape (nh, nd), optional
        Hard-data (exact observation) locations.  Same coordinate convention
        as *ck*.  Pass ``None`` when no hard data are available.
    cs : ndarray, shape (ns, nd), optional
        Soft-data locations.  Pass ``None`` when no soft data are available.
    zh : ndarray, shape (nh,) or (nh, 1), optional
        Hard-data observed values, one scalar per row of *ch*.  Reshaped
        internally to ``(nh, 1)`` if necessary.
    zs : list of tuples, length ns, optional
        Soft-data probabilistic information.  Each element ``zs[i]``
        describes the PDF at location ``cs[i]`` using the stamps format:

        * ``(1,  nl, limi, probdens)`` — histogram PDF (type 1)
        * ``(2,  nl, limi, probdens)`` — linear PDF (type 2)
        * ``(10, mean, variance)``     — Gaussian PDF (type 10)
        * ``None``                     — missing observation (skipped)

        where ``nl`` is the number of PDF classes (shape ``(1,)`` or
        scalar), ``limi`` are the class boundaries (shape ``(1, nl+1)`` for
        type 1/2), and ``probdens`` are the probability densities
        (shape ``(1, nl)``).
        Use :func:`stamps.bme.softconverter.probaUniform` or
        :func:`stamps.bme.softconverter.probaGaussian` to construct these
        tuples from raw interval or Gaussian parameters.
    covmodel : callable or str, optional
        Covariance model function (e.g. ``exponentialC``, ``sphericalC``).
        Required unless *covmat* is provided.  Accepts the signature
        ``covmodel(dist, sill, range_param)``.
    covparam : list or ndarray, optional
        Parameters for *covmodel* in stamps nested-model format.  Required
        unless *covmat* is provided.  See :func:`stamps.models.covmodel` for
        the expected structure.
    covmat : ndarray, shape (nk+nh+ns, nk+nh+ns), optional
        Pre-computed full covariance matrix covering all estimation and data
        locations in the stacked order ``[ck; ch; cs]``.  When provided,
        *covmodel* and *covparam* are ignored and no distances need to be
        computed.  Useful when the same matrix is reused across multiple calls
        (e.g. in a GUI loop).
    order : {np.nan, 0, ndarray of shape (nk+nh+ns, 1)}, default ``np.nan``
        Prior-mean specification controlling the large-scale trend:

        * ``np.nan``  — **zero-mean prior**.  Equivalent to Simple Kriging.
          Estimates revert to 0 wherever data are absent.  Appropriate only
          when the field has been pre-de-meaned.
        * ``0``       — **global-mean prior**.  A single constant equal to
          the weighted average of all data (equivalent to Ordinary Kriging).
          Estimates revert to the data mean in data-sparse areas.
        * ``ndarray`` — **spatially-varying prior mean**, one value per row
          of the stacked ``[ck; ch; cs]`` coordinate array.  Use the output
          of :func:`BMEpriorMean` (method ``'kernel'`` or ``'kriging'``) to
          build this array for a data-adaptive locally-varying trend.
    options : BMEoptions or None, optional
        Dictionary-like object controlling numerical options.  When ``None``
        (default), :class:`stamps.bme.BMEoptions` defaults are used.
        Relevant keys:

        * ``'soft_approx_gaussian'`` *(bool, default False)* — force the
          analytical Gaussian approximation for all soft data, bypassing QMC.
          Speeds up computation at the cost of accuracy when intervals are wide
          or asymmetric.
        * ``'outlier_handling'`` *(str, default ``'inflate'``)* — how to treat
          soft data whose interval is statistically incompatible with the
          conditional prior ``E[Zs|Zh]``:
          ``'inflate'`` expands the interval to restore compatibility;
          ``'exclude'`` removes the datum from the neighbourhood;
          ``'ignore'`` keeps all data unchanged.
        * ``'outlier_inflate_margin'`` *(float, default 0.10)* — fractional
          extra margin added beyond the minimum inflation required.
        * ``'min_overlap_prob'`` *(float, default 0.10)* — minimum overlap
          probability (``P(interval | prior)``) below which a soft datum is
          flagged for handling.
        * ``'pca_integration_threshold'`` *(float, default 0.999)* — fraction
          of conditional variance retained when PCA dimension-reduction is
          applied before QMC integration.  Reducing this (e.g. to 0.95)
          dramatically lowers QMC dimension at a small accuracy cost.
        * ``'nonneg_estimate'`` *(bool, default False)* — clamp negative
          posterior means to zero using truncated-Normal theory.  Appropriate
          for physically non-negative variables (concentrations, proportions).
    nhmax : int or None, optional
        Maximum number of hard-data neighbours included in each local
        neighbourhood.  ``None`` (default) includes all hard data within
        *dmax*.  Increasing this improves accuracy but raises the cost of
        the covariance system solve.
    nsmax : int or None, optional
        Maximum number of soft-data neighbours.  ``None`` includes all soft
        data within *dmax*.  Raising *nsmax* beyond ~16 can render QMC
        integration prohibitively expensive; consider enabling
        ``'pca_integration_threshold'`` or ``'soft_approx_gaussian'``.
    dmax : float or array-like, optional
        Maximum search radius.  For purely spatial problems supply a single
        positive float (metres or matching coordinate units).  For space-time
        problems supply a three-element array ``[s_max, t_max, s_t_ratio]``
        where ``s_t_ratio`` converts time units to space units.  ``None``
        (default) includes all data regardless of distance.
    general_knowledge : {'gaussian'}, default ``'gaussian'``
        Prior structural model.  Currently only ``'gaussian'`` (Gaussian
        random field) is supported.  Reserved for future extension to
        non-Gaussian structural models.
    pdfk, pdfh, pdfs : list or None, optional
        Legacy MATLAB-compatibility arguments carrying pre-computed prior PDFs
        at estimation, hard-data, and soft-data locations respectively.
        **Not needed for standard usage** — pass ``None``.
    hk_k, hk_h, hk_s : list or None, optional
        Legacy MATLAB-compatibility arguments for hard-knowledge (mean, variance)
        tuples at each coordinate set.  **Not needed for standard usage** —
        pass ``None``.
    gui_args : tuple or None, optional
        Hook for GUI progress dialogs.  Pass a tuple ``(QProgressDialog,)``
        when running inside a Qt application; pass ``None`` for headless use.
    ck_cov_output : bool, default ``False``
        If ``True``, also return the posterior covariance matrix among the
        estimation locations (useful for uncertainty propagation).
    n_workers : int, default 1
        Number of parallel threads for neighbourhood processing via
        :class:`concurrent.futures.ThreadPoolExecutor`.  Values ``> 1``
        are effective when many independent neighbourhood groups exist.
        Thread-safety relies on the GIL; set ``n_workers=1`` for deterministic
        reproducibility.

    Returns
    -------
    zk : ndarray, shape (nk, 3)
        BME posterior statistics at each estimation location:

        * ``zk[:, 0]`` — posterior mean  ``E[Zk | Zh, Zs]``
        * ``zk[:, 1]`` — posterior variance  ``Var[Zk | Zh, Zs]``
          (guaranteed non-negative; ``delta_mu`` QMC-noise contribution is
          subtracted before clamping)
        * ``zk[:, 2]`` — third central moment
          ``E[(Zk - E[Zk])^3 | Zh, Zs]``
          (zero for the Gaussian fast-path; non-zero for QMC path)

        Rows corresponding to estimation locations with no neighbours within
        *dmax* will contain ``np.nan`` in all three columns.

        If *ck_cov_output* is ``True``, returns a tuple
        ``(zk, ck_cov)`` where ``ck_cov`` is the ``(nk, nk)`` posterior
        covariance matrix computed via the Gaussian approximation.

    Notes
    -----
    **BME integration theory**

    The BME posterior PDF at estimation location :math:`\\mathbf{c}_k` is:

    .. math::

        f_K(z_k) \\propto f_G(z_k | \\mathbf{z}_h)
            \\cdot
            \\underbrace{\\int f_G(\\mathbf{z}_s | z_k, \\mathbf{z}_h)
                          \\cdot f_S(\\mathbf{z}_s)\\,d\\mathbf{z}_s}_{\\text{SI}}

    where :math:`f_G` is the Gaussian prior and :math:`f_S` is the product
    of soft-data likelihoods.  The normalisation constant is:

    .. math::

        \\text{NC} = \\int f_G(\\mathbf{z}_s | \\mathbf{z}_h)
                          \\cdot f_S(\\mathbf{z}_s)\\,d\\mathbf{z}_s

    The posterior moments are:

    .. math::

        E[Z_k | \\mathbf{z}_h, \\mathbf{z}_s]
            = \\mu_{k|h} + \\frac{\\text{SI}_1}{\\text{NC}}

    **QMC Importance-Sampling (non-Gaussian path)**

    The integrals SI and NC are estimated via Quasi-Monte Carlo Importance
    Sampling (QMC-IS).  Sobol' sequences are drawn in ``[0,1]^{ns_eff}``
    (where ``ns_eff ≤ ns`` after PCA dimension reduction), transformed to
    samples from the conditional Gaussian prior via the inverse CDF, and
    weighted by the product soft-data likelihood.  The normalisation constant
    is checked for stability:  if ``Mon_NC < 2 × Mon_NC_err`` or
    ``Mon_NC ≤ 0``, a per-location **Gaussian fallback** is triggered —
    replacing all non-Gaussian soft data by matched Gaussian PDFs (same mean
    and variance) and recomputing analytically.

    **Gaussian fast-path (all-Gaussian routing)**

    When every soft datum has type 10 (Gaussian), the posterior is also
    Gaussian and the closed-form Kriging update equations are used:

    .. math::

        E[Z_k | \\mathbf{Z}_{hs}] =
            \\boldsymbol{\\Sigma}_{k,hs}
            \\boldsymbol{\\Sigma}_{hs,hs}^{-1}
            (\\mathbf{z}_{hs} - \\boldsymbol{\\mu}_{hs}) + \\mu_k

    This path is also forced automatically when ``ns_nongauss > 20`` or when
    ``options['soft_approx_gaussian']`` is ``True``.

    **Neighbourhood grouping**

    Before looping over estimation points, :func:`_bme_posterior_prepare`
    groups all *ck* that share the same *nhmax* hard-data and *nsmax*
    soft-data neighbours.  Each group is processed as a single chunk (up to
    250 nodes), allowing vectorised covariance computation and, with
    ``n_workers > 1``, parallel thread dispatch.

    **QMC noise diagnostic**

    After all chunks complete, a diagnostic line is printed reporting the
    median and maximum absolute QMC error ``delta_mu`` and the relative error
    ``delta_mu / |mean|`` across all estimation points that used QMC.  Use
    this to judge whether *nsmax* is too large or ``pca_integration_threshold``
    should be lowered.

    Examples
    --------
    **Hard data only (equivalent to Ordinary Kriging)**

    >>> import numpy as np
    >>> from stamps.bme import BMEPosteriorMoments
    >>> from stamps.models.covmodel import exponentialC
    >>>
    >>> rng = np.random.default_rng(42)
    >>> ch = rng.uniform(0, 100, (30, 2))   # 30 hard-data locations, 2-D
    >>> zh = rng.normal(50, 10, (30, 1))    # observed values
    >>> ck = np.mgrid[0:100:20j, 0:100:20j].reshape(2, -1).T  # 400-node grid
    >>>
    >>> covmodel = exponentialC
    >>> covparam = [[0., 1.], [1., [1., 30.]]]   # nugget + exponential
    >>>
    >>> zk = BMEPosteriorMoments(
    ...     ck, ch=ch, zh=zh,
    ...     covmodel=covmodel, covparam=covparam,
    ...     nhmax=20, dmax=60.0, order=0,
    ... )
    >>> print(f"Mean range: {zk[:, 0].min():.1f} – {zk[:, 0].max():.1f}")
    >>> print(f"Max variance: {zk[:, 1].max():.2f}")

    **With interval soft data (full BME)**

    >>> from stamps.bme.softconverter import probaUniform
    >>> from stamps.bme import BMEPosteriorMoments, BMEpriorMean
    >>>
    >>> cs = rng.uniform(0, 100, (50, 2))   # 50 soft-data locations
    >>> s_lo = rng.uniform(30, 50, 50)
    >>> s_hi = s_lo + rng.uniform(5, 20, 50)
    >>> zs = [probaUniform(lo, hi) for lo, hi in zip(s_lo, s_hi)]
    >>>
    >>> # Build a spatially-varying prior mean
    >>> order = BMEpriorMean(ck, ch, zh, cs=cs, zs=zs, method='kernel')
    >>>
    >>> from stamps.bme.BMEoptions import BMEoptions
    >>> opts = BMEoptions()
    >>> opts['outlier_handling'] = 'inflate'
    >>> opts['pca_integration_threshold'] = 0.99
    >>>
    >>> zk = BMEPosteriorMoments(
    ...     ck, ch=ch, cs=cs, zh=zh, zs=zs,
    ...     covmodel=covmodel, covparam=covparam,
    ...     nhmax=20, nsmax=8, dmax=60.0,
    ...     order=order, options=opts,
    ... )
    >>> bme_mean = zk[:, 0]
    >>> bme_var  = zk[:, 1]

    ### Original Reference ###
    ck: n by d np 2d array
        the estimated data coordinate, usually, d = 3 for spatial-temporal
        coordinate, first two column for spatial, e.g. x, y, and last column
        for temporal, e.g. t.

    ch: n by d np 2d array
        the hard data coordinate.

    cs: n by d np 2d array
        the soft data coordinate.

    zh: n by 1 np 2d array
        the hard data measurement.

    zs: a sequence of soft data record, e.g. (zs1, zs2, zs3, ..., zsn)
        each zsi(i=1~n) is a sequence of data arguments,
        first item should be softpdftype to determind
        the other rest arguments format, e.g.
        syntax: (softdata_type, *softdata_args)
            zs1 = (1, nl, limi, probdens)
            zs2 = (10, zm, zstd)
        if element is emtpy, put None
            e.g. (zs1, zs2, None, ..., zsn)

    covmodel: 

    covmat:
        covariance matrix, a np 2d array
        with shape (nk+nh+ns) by (nk+nh+ns)
    if covmat provieded, covmodel and covparam are simply skipped.
    
    #SI = integrate (fg_s_given_kh * fs_s) dx_s
    #NC = integrate (fg_s_given_h * fs_s) dx_s
    #pdf_k = (fg_kh * SI) / (fg_h * NC)  # eq.1
    #exp_k = ... # eq.2
    #exp_kp = ... # eq.3
    #if general_knowledge == gaussian and specific_knowledge == unknown
    #exp_k = ... # eq.4
    #var_k = ... # eq.5

    gui_args: a tuple with gui arguments

    
    return 
    """
    (output_arguments, configured_arguments) = _bme_posterior_prepare(
        ck, ch, cs, zh, zs,
        covmodel, covparam, covmat,
        order, options,
        nhmax, nsmax, dmax,
        general_knowledge,
        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
        gui_args)
    (ckhs_idx_list,) = output_arguments

    (ck, ch, cs, zh, zs,
        covmodel, covparam, covmat,
        order, options,
        nhmax, nsmax, dmax,
        general_knowledge,
        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
        gui_args) = configured_arguments

    if isinstance(order, np.ndarray):
        has_user_defined_general_knowledge = True
    elif order == 0 or np.isnan(order):
        has_user_defined_general_knowledge = False
    else:
        raise ValueError('order type error')

    nk = ck.shape[0]
    nh = ch.shape[0] if ch is not None else 0
    ns = cs.shape[0] if cs is not None else 0

    # Normalise zs: probaUniform/probaGaussian return a 4-element batch list
    # [type, nl_arr, limi_arr, pd_arr].  The rest of the engine expects a
    # per-point list [(type_0, nl_0, limi_0, pd_0), ...].
    # _normalize_zs detects the format and converts if needed.
    zs = _normalize_zs(zs, ns)

    zk = np.empty((nk, 4))  # [mean, variance, skewness, delta_mu (QMC error)]
    zk[:, 3] = 0.0  # default delta_mu = 0 (Gaussian path has no QMC noise)
    cur_cnt = 0
    cum_cnt = 0

    # --- Performance logging ---
    import time as _time
    _t_start = _time.time()
    _approx_mode = (options is not None
                    and options['soft_approx_gaussian'])
    if ns > 0:
        _all_types = set(
            get_standard_soft_pdf_type(zsi[0]) for zsi in zs if zsi is not None)
        _has_nongaussian = bool(_all_types - {10})
        print("[STARBME] BME estimation: nk={}, nh={}, ns={}, "
              "soft types={}, approx_gaussian={}".format(
                  nk, nh, ns, _all_types, _approx_mode))
        if _has_nongaussian and not _approx_mode:
            print("[STARBME] WARNING: Non-Gaussian soft data detected. "
                  "Using QMC integration (slow). Enable 'Fast Gaussian "
                  "Approximation' in Prediction > Configures for speed-up.")
    else:
        print("[STARBME] BME estimation: nk={}, nh={}, ns=0".format(nk, nh))

    # --- Ensure n_workers is sane ---
    n_workers = max(1, int(n_workers))

    if general_knowledge == 'gaussian':
        # ---- Build list of independent work items ----
        _work_items = []
        for ck_idx, ch_idx, cs_idx in ckhs_idx_list:
            ck_idx = np.array(ck_idx, dtype=int)
            ch_idx = np.array(ch_idx, dtype=int)
            cs_idx = np.array(cs_idx, dtype=int)
            # Fresh copies per neighborhood group so that assimilation
            # in one chunk does not corrupt data for sibling chunks.
            picked_ch = (
                ch[ch_idx, :].copy() if isinstance(ch, np.ndarray) else None)
            picked_cs = (
                cs[cs_idx, :].copy() if isinstance(cs, np.ndarray) else None)
            picked_zh = (
                zh[ch_idx, :].copy() if isinstance(zh, np.ndarray) else None)
            picked_zs = (
                [zs[cs_idx_i] for cs_idx_i in cs_idx]
                if zs is not None else None)
            ck_count = ck_idx.shape[0]
            split_count = np.ceil(ck_count / 250.)
            for ck_idx_piece in np.array_split(ck_idx, split_count):
                # Each chunk gets its own copy of neighbourhood data
                # to prevent cross-chunk corruption from assimilation.
                _work_items.append(dict(
                    ck_idx_piece=ck_idx_piece,
                    picked_ch=picked_ch.copy() if picked_ch is not None else None,
                    picked_cs=picked_cs.copy() if picked_cs is not None else None,
                    picked_zh=picked_zh.copy() if picked_zh is not None else None,
                    picked_zs=list(picked_zs) if picked_zs is not None else None,
                    ch_idx=ch_idx,
                    cs_idx=cs_idx,
                ))

        _n_chunks = len(_work_items)

        # ---- Parallel or sequential dispatch ----
        if n_workers > 1 and _n_chunks > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading

            _actual_workers = min(n_workers, _n_chunks)
            print("[STARBME] Parallel estimation: {} chunks across {} "
                  "threads".format(_n_chunks, _actual_workers))

            _cancel_event = threading.Event()

            def _submit_chunk(item):
                """Wrapper to check cancellation before heavy work."""
                if _cancel_event.is_set():
                    return (item['ck_idx_piece'], None)
                return _process_estimation_chunk(
                    ck_idx_piece=item['ck_idx_piece'],
                    picked_ch=item['picked_ch'],
                    picked_cs=item['picked_cs'],
                    picked_zh=item['picked_zh'],
                    picked_zs=item['picked_zs'],
                    ck=ck, ch=ch, cs=cs, zh=zh, zs=zs,
                    covmat=covmat, covmodel=covmodel,
                    covparam=covparam, order=order, options=options,
                    general_knowledge=general_knowledge,
                    has_user_defined_general_knowledge=has_user_defined_general_knowledge,
                    pdfk=pdfk, pdfh=pdfh, pdfs=pdfs,
                    hk_k=hk_k, hk_h=hk_h, hk_s=hk_s,
                    ck_cov_output=ck_cov_output,
                    ch_idx=item['ch_idx'], cs_idx=item['cs_idx'],
                    nk=nk, nh=nh)

            with ThreadPoolExecutor(max_workers=_actual_workers) as executor:
                futures = {
                    executor.submit(_submit_chunk, item): item
                    for item in _work_items
                }
                for future in as_completed(futures):
                    ck_idx_piece, picked_mvs = future.result()
                    if picked_mvs is None:
                        # Cancelled
                        _cancel_event.set()
                        return False
                    zk[ck_idx_piece, :] = picked_mvs
                    cur_cnt += ck_idx_piece.size
                    # Update progress on the main thread
                    if gui_args:
                        qpgd = gui_args[0]
                        if qpgd.wasCanceled():
                            _cancel_event.set()
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            return False
                        else:
                            qpgd.setValue(qpgd.value() + ck_idx_piece.size)
                    else:
                        if cur_cnt - cum_cnt >= 2000:
                            print(cur_cnt, '/', nk)
                            cum_cnt = cur_cnt

        else:
            # ---- Sequential execution (original path) ----
            if n_workers > 1 and _n_chunks <= 1:
                print("[STARBME] Only {} chunk(s) — running "
                      "sequentially".format(_n_chunks))
            for item in _work_items:
                ck_idx_piece, picked_mvs = _process_estimation_chunk(
                    ck_idx_piece=item['ck_idx_piece'],
                    picked_ch=item['picked_ch'],
                    picked_cs=item['picked_cs'],
                    picked_zh=item['picked_zh'],
                    picked_zs=item['picked_zs'],
                    ck=ck, ch=ch, cs=cs, zh=zh, zs=zs,
                    covmat=covmat, covmodel=covmodel,
                    covparam=covparam, order=order, options=options,
                    general_knowledge=general_knowledge,
                    has_user_defined_general_knowledge=has_user_defined_general_knowledge,
                    pdfk=pdfk, pdfh=pdfh, pdfs=pdfs,
                    hk_k=hk_k, hk_h=hk_h, hk_s=hk_s,
                    ck_cov_output=ck_cov_output,
                    ch_idx=item['ch_idx'], cs_idx=item['cs_idx'],
                    nk=nk, nh=nh)
                zk[ck_idx_piece, :] = picked_mvs
                if gui_args:
                    qpgd = gui_args[0]
                    if qpgd.wasCanceled():
                        return False
                    else:
                        qpgd.setValue(qpgd.value() + ck_idx_piece.size)
                else:
                    cur_cnt += ck_idx_piece.size
                    if cur_cnt - cum_cnt >= 2000:
                        print(cur_cnt, '/', nk)
                        cum_cnt = cur_cnt

        print(cur_cnt, '/', nk)
        _elapsed = _time.time() - _t_start
        _n_nan = int(np.sum(np.isnan(zk[:, 0])))
        print("[STARBME] BME estimation completed in {:.1f}s "
              "({:.3f}s per point)".format(_elapsed, _elapsed / max(nk, 1)))
        if _n_nan > 0:
            import warnings
            warnings.warn(
                "[STARBME] {:d} of {:d} estimation locations produced NaN "
                "results. This typically occurs when estimation points are "
                "very close to (but not exactly at) soft data locations, "
                "causing the QMC integration to fail. Consider:\n"
                "  1. Enabling 'Fast Gaussian Approximation' to bypass QMC.\n"
                "  2. Reducing the PCA threshold to lower integration dimension.\n"
                "  3. Checking that estimation grid does not nearly overlap "
                "with soft data locations.".format(_n_nan, nk))

        # --- QMC noise diagnostic ---
        _delta_mu_col = zk[:, 3]
        _valid_delta = ~np.isnan(_delta_mu_col) & (_delta_mu_col > 0)
        if np.any(_valid_delta):
            _dmu = _delta_mu_col[_valid_delta]
            _mu_abs = np.abs(zk[_valid_delta, 0])
            _rel_err = np.where(_mu_abs > 0, _dmu / _mu_abs, np.inf)
            _rel_err_finite = _rel_err[np.isfinite(_rel_err)]
            print("[STARBME] QMC noise diagnostic: "
                  "delta_mu  median={:.4g}, max={:.4g}; "
                  "rel_err  median={:.2%}, max={:.2%} "
                  "({} of {} points had QMC error info)".format(
                      np.median(_dmu), np.max(_dmu),
                      np.median(_rel_err_finite) if len(_rel_err_finite) > 0 else 0,
                      np.max(_rel_err_finite) if len(_rel_err_finite) > 0 else 0,
                      int(np.sum(_valid_delta)), nk))

        # --- Non-negativity constraint ---
        # When QMC IS was used with nonneg_estimate, the integrand
        # already computed truncated-normal moments (flagged by
        # delta_mu == -1).  For Gaussian-fallback and analytical
        # points, post-hoc truncation is still applied by
        # _apply_nonneg_truncation (which skips flagged points).
        if options is not None and options.get('nonneg_estimate', False):
            _n_neg_before = int(np.nansum(zk[:, 0] < 0))
            _n_qmc_nonneg = int(np.sum(zk[:, 3] < 0)) if zk.shape[1] >= 4 else 0
            zk = _apply_nonneg_truncation(zk)  # returns 3-col
            print("[STARBME] Non-negativity correction: {} points total, "
                  "{} via QMC integrand, {} via post-hoc truncation. "
                  "{} had negative means (clamped to 0).".format(
                      nk, _n_qmc_nonneg, nk - _n_qmc_nonneg,
                      _n_neg_before))
        else:
            zk = zk[:, :3]  # strip delta_mu column

        return zk
    else:
      nk=len(pdfk)
      moments=np.empty((nk,3))

      for k in range(nk):
        print ('BME MOMENTS:' + str(k+1) + '/' + str(nk))
        
        cklocal=ck[k:k+1,:]
        pdfk_local=[pdfk[k]]
        hk_k_local=[hk_k[k]]
        
        pdf_k=BMEPosteriorPDF(cklocal, ch, cs, zh, zs, covmodel, covparam,
              order, options, nhmax, nsmax, dmax, general_knowledge,
              pdfk=pdfk_local,pdfh=pdfh,pdfs=pdfs,
              hk_k=hk_k_local,hk_h=hk_h,hk_s=hk_s)[0]
          
        zmin=hk_k[k][0]-6*np.sqrt(hk_k[k][1])
        zmax=hk_k[k][0]+6*np.sqrt(hk_k[k][1])
        # Non-negativity: restrict integration domain to [0, ∞)
        if options is not None and options.get('nonneg_estimate', False):
            zmin = max(zmin, 0.0)
        
        xxx=np.linspace(zmin,zmax,100)
        aaa=pdf_k(xxx,0)

        maxpts = options[2][0]
        aEps = 0
        rEps = options[3][0]

        from cubature import cubature

        mon1_for_cubature = lambda x_array: x_array[:,0] * pdf_k(x_array[:,0],0)[:,0]
        mon1,mon1_err = cubature(
            func=mon1_for_cubature, ndim=1, fdim=1, xmin=np.array([zmin]),
            xmax=np.array([zmax]), adaptive='h', maxEval = maxpts,
            abserr = 0, relerr = rEps, vectorized = True)
        mon2_for_cubature = lambda x_array: x_array[:,0]**2 * pdf_k(x_array[:,0],0)[:,0]
        mon2,mon2_err = cubature(
            func=mon2_for_cubature,ndim=1, fdim=1, xmin=np.array([zmin]),
            xmax=np.array([zmax]), adaptive='h', maxEval = maxpts,
            abserr = 0, relerr = rEps, vectorized = True)  
        mon3_for_cubature = lambda x_array: x_array[:,0]**3 * pdf_k(x_array[:,0],0)[:,0]
        mon3,mon3_err = cubature(
            func=mon3_for_cubature,ndim=1, fdim=1, xmin=np.array([zmin]),
            xmax=np.array([zmax]), adaptive='h', maxEval = maxpts,
            abserr = 0, relerr = rEps, vectorized = True)

        moments[k,0]=mon1
        moments[k,1]=mon2-mon1**2
        moments[k,2]=mon3-3*mon1*mon2-mon1**3

      return moments[:,0], moments[:,1], moments[:,2]

def _bme_posterior_pdf_closures(
    ck, ch=None, cs=None, zh=None, zs=None,
    covmodel=None, covparam=None, covmat=None,
    order=np.nan, options=None,
    nhmax=None, nsmax=None, dmax=None,
    general_knowledge='gaussian',
    #  specific_knowledge='unknown',  
    pdfk=None,pdfh=None,pdfs=None,hk_k=None,hk_h=None,hk_s=None,
    gui_args=None):
    """Return the BME posterior PDF as a callable function object at each estimation location.

    **Internal helper** — most users should call :func:`BMEPosteriorPDF`
    directly; it evaluates the posterior on an automatic z-grid and returns
    clean (nk, nz) arrays.

    Where :func:`BMEPosteriorMoments` summarises the posterior with three
    scalar statistics, ``_bme_posterior_pdf_closures`` returns the **raw
    posterior density integrand** at each estimation location as a Python
    callable.  The caller must evaluate each returned function over a z-grid,
    normalise the result, and assemble arrays manually.

    For most practical workflows prefer :func:`BMEPosteriorPDF`, which
    handles z-grid construction, normalisation, and array assembly
    automatically.

    Parameters
    ----------
    ck : ndarray, shape (nk, nd)
        Estimation locations.  ``nd=2`` for purely spatial problems, ``nd=3``
        for space-time problems (last column treated as time).
    ch : ndarray, shape (nh, nd), optional
        Hard-data (exact observation) locations.  Pass ``None`` when no hard
        data are available.
    cs : ndarray, shape (ns, nd), optional
        Soft-data locations.  Pass ``None`` when no soft data are available.
    zh : ndarray, shape (nh,) or (nh, 1), optional
        Hard-data observed values, one scalar per row of *ch*.
    zs : list of tuples, length ns, optional
        Soft-data probabilistic information.  Each element ``zs[i]``
        describes the PDF at location ``cs[i]`` using the stamps format:

        * ``(1,  nl, limi, probdens)`` — histogram PDF (type 1)
        * ``(2,  nl, limi, probdens)`` — linear PDF (type 2)
        * ``(10, mean, variance)``     — Gaussian PDF (type 10)
        * ``None``                     — missing / skipped observation

        where ``nl`` is the number of PDF classes, ``limi`` are class
        boundaries (shape ``(1, nl+1)``), and ``probdens`` are probability
        densities (shape ``(1, nl)``).  See
        :func:`stamps.bme.softconverter.probaUniform` for construction.
    covmodel : callable or str, optional
        Covariance model function (e.g. ``exponentialC``).  Required unless
        *covmat* is provided.
    covparam : list or ndarray, optional
        Nested covariance model parameters.  Required unless *covmat* is
        provided.
    covmat : ndarray, shape (nk+nh+ns, nk+nh+ns), optional
        Pre-computed full covariance matrix over all locations in stacked
        order ``[ck; ch; cs]``.  When provided, *covmodel* and *covparam*
        are ignored.
    order : {np.nan, 0, ndarray of shape (nk+nh+ns, 1)}, default ``np.nan``
        Prior-mean specification.  Same semantics as in
        :func:`BMEPosteriorMoments`:

        * ``np.nan``  — zero-mean prior (Simple Kriging-like).
        * ``0``       — global-mean prior (Ordinary Kriging-like).
        * ``ndarray`` — spatially-varying prior from :func:`BMEpriorMean`.
    options : BMEoptions or None, optional
        Numerical options object from :class:`stamps.bme.BMEoptions`.
        Relevant keys: ``'soft_approx_gaussian'``, ``'outlier_handling'``,
        ``'pca_integration_threshold'``, ``'nonneg_estimate'``.
        ``None`` uses all defaults.  See :func:`BMEPosteriorMoments` for a
        full description of each key.
    nhmax : int or None, optional
        Maximum number of hard-data neighbours per estimation location.
    nsmax : int or None, optional
        Maximum number of soft-data neighbours per estimation location.
    dmax : float or array-like, optional
        Maximum search radius (scalar for spatial, 3-element for space-time).
    general_knowledge : {'gaussian'}, default ``'gaussian'``
        Prior structural model.  Only ``'gaussian'`` is currently supported.
    pdfk, pdfh, pdfs : list or None, optional
        Legacy MATLAB-compatibility prior PDF containers.  Pass ``None`` for
        standard use.
    hk_k, hk_h, hk_s : list or None, optional
        Legacy MATLAB-compatibility hard-knowledge (mean, variance) tuples.
        Pass ``None`` for standard use.
    gui_args : tuple or None, optional
        Qt progress-dialog hook ``(QProgressDialog,)``.  Pass ``None`` for
        headless / scripted use.

    Returns
    -------
    zk : ndarray, shape (nk, 1), dtype=object
        Object array where ``zk[i, 0]`` is a callable posterior PDF at
        estimation location ``ck[i]``.  Each callable has the signature::

            pdf_value = zk[i, 0](z_array, flag)

        where ``z_array`` is a 1-D array of z-values to evaluate and ``flag``
        is an integer mode selector (pass ``0`` for the standard normalised
        PDF).  Evaluation returns a 2-D array of shape ``(len(z_array), 1)``.

        Estimation locations outside *dmax* from all data will have
        ``zk[i, 0] = None``.

    Notes
    -----
    **Relationship to BMEPosteriorMoments**

    ``BMEPosteriorPDF`` and :func:`BMEPosteriorMoments` share the same
    neighbourhood-preparation and internal ``_bme_posterior_pdf`` / Gaussian
    routing logic.  The key difference is in what they return:

    * :func:`BMEPosteriorMoments` integrates the posterior analytically or via
      QMC and returns scalar summary statistics.
    * ``BMEPosteriorPDF`` wraps the posterior integrand as a Python closure so
      the caller can evaluate it at arbitrary z-values.  No integration is
      performed inside this function.

    **Evaluating the returned PDF**

    The recommended pattern for evaluating and normalising the posterior PDF
    at a single location is::

        z_grid = np.linspace(z_min, z_max, 500)
        pdf_raw = zk[i, 0](z_grid, 0)[:, 0]        # shape (500,)
        dz = z_grid[1] - z_grid[0]
        pdf_norm = pdf_raw / (pdf_raw.sum() * dz)   # normalise to unit area

    For automated grid evaluation across all estimation locations with
    normalisation and CI extraction, use :func:`BMEPosteriorPDF` instead.

    **All-Gaussian fast path**

    When all soft data are type 10 or ``'soft_approx_gaussian'`` is ``True``,
    the returned callable is backed by the analytical Gaussian conditional
    distribution — it evaluates ``scipy.stats.norm.pdf`` internally and is
    therefore extremely fast.

    **Performance considerations**

    Unlike :func:`BMEPosteriorMoments`, ``_bme_posterior_pdf_closures`` does
    not support ``n_workers > 1`` parallel dispatch or the QMC noise
    diagnostic.  For large grids, call :func:`BMEPosteriorMoments` for the
    summary statistics and reserve this function for selected points (e.g. to
    verify the shape of the distribution at a few key locations).

    Examples
    --------
    **Retrieve and plot the posterior PDF at a single location**

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from stamps.bme import BMEPosteriorPDF
    >>> from stamps.models.covmodel import exponentialC
    >>>
    >>> rng = np.random.default_rng(0)
    >>> ch = rng.uniform(0, 100, (30, 2))
    >>> zh = rng.normal(50, 10, (30, 1))
    >>> ck = np.array([[50.0, 50.0]])   # single estimation point
    >>>
    >>> covmodel = exponentialC
    >>> covparam = [[0., 1.], [1., [1., 30.]]]
    >>>
    >>> zk_pdf = BMEPosteriorPDF(
    ...     ck, ch=ch, zh=zh,
    ...     covmodel=covmodel, covparam=covparam,
    ...     nhmax=15, dmax=60.0, order=0,
    ... )
    >>>
    >>> z_grid = np.linspace(20, 80, 300)
    >>> pdf_raw = zk_pdf[0, 0](z_grid, 0)[:, 0]
    >>> dz = z_grid[1] - z_grid[0]
    >>> pdf_norm = pdf_raw / (pdf_raw.sum() * dz)
    >>>
    >>> plt.plot(z_grid, pdf_norm)
    >>> plt.xlabel('z'); plt.ylabel('Posterior PDF'); plt.show()

    **Compare with BMEPosteriorMoments at the same point**

    >>> from stamps.bme import BMEPosteriorMoments
    >>> zk_mom = BMEPosteriorMoments(
    ...     ck, ch=ch, zh=zh,
    ...     covmodel=covmodel, covparam=covparam,
    ...     nhmax=15, dmax=60.0, order=0,
    ... )
    >>> print(f"Moments → mean={zk_mom[0,0]:.2f}, var={zk_mom[0,1]:.2f}")
    >>> pdf_mean = np.trapz(z_grid * pdf_norm, z_grid)
    >>> print(f"PDF integral → mean={pdf_mean:.2f}")
    """
    (output_arguments, configured_arguments) = _bme_posterior_prepare(
        ck, ch, cs, zh, zs,
        covmodel, covparam, covmat,
        order, options,
        nhmax, nsmax, dmax,
        general_knowledge,
        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
        gui_args)
    (ckhs_idx_list,) = output_arguments

    (ck, ch, cs, zh, zs,
        covmodel, covparam, covmat,
        order, options,
        nhmax, nsmax, dmax,
        general_knowledge,
        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
        gui_args) = configured_arguments
   
    if isinstance(order, np.ndarray):
        has_user_defined_general_knowledge = True
    elif order == 0 or np.isnan(order):
        has_user_defined_general_knowledge = False
    else:
        raise ValueError('order type error')
    nk = ck.shape[0]
    nh = ch.shape[0] if ch is not None else 0
    ns = cs.shape[0] if cs is not None else 0

    # Normalise batch → per-point format (same as BMEPosteriorMoments)
    zs = _normalize_zs(zs, ns)

    zk = np.empty((ck.shape[0],1), dtype=object) # to 1 pdf function
    if general_knowledge == 'gaussian':
        for ck_idx, ch_idx, cs_idx in ckhs_idx_list:
            ck_idx = np.array(ck_idx, dtype=int)
            ch_idx = np.array(ch_idx, dtype=int)
            cs_idx = np.array(cs_idx, dtype=int)
            picked_ch =\
                ch[ch_idx, :] if isinstance(ch, np.ndarray) else None
            picked_cs =\
                cs[cs_idx, :] if isinstance(cs, np.ndarray) else None
            picked_zh =\
                zh[ch_idx, :] if isinstance(zh, np.ndarray) else None
            picked_zs =\
                [zs[cs_idx_i] for cs_idx_i in cs_idx] if zs is not None else None
            ck_count = ck_idx.shape[0]
            split_count = np.ceil(ck_count/250.)
            for ck_idx_piece in np.array_split(ck_idx, split_count):
                picked_ck = ck[ck_idx_piece, :]
                if covmat is not None:
                    covidx = np.hstack(
                        (ck_idx_piece, nk+ch_idx, nk+nh+cs_idx)
                        )
                    picked_covmat = covmat[np.ix_(covidx, covidx)]
                else:
                    picked_covmat = covmat

                # --- Adaptive neighbourhood assimilation (PDF path) ---
                if picked_covmat is not None:
                    try:
                        (picked_ck, picked_ch, picked_cs,
                         picked_zh, picked_zs,
                         picked_covmat) = _assimilate_neighborhood(
                            picked_ck, picked_ch, picked_cs,
                            picked_zh, picked_zs, picked_covmat)
                    except Exception:
                        pass  # fall through with original data

                #picked order for specified general knowledge
                if has_user_defined_general_knowledge:
                    order_idx = np.hstack(
                        (ck_idx_piece, nk+ch_idx, nk+nh+cs_idx)
                        )
                    picked_order = order[order_idx, :]
                else:
                    picked_order = order

                try:
                    picked_mvs = _bme_posterior_pdf(
                        picked_ck, picked_ch, picked_cs,
                        picked_zh, picked_zs,
                        covmodel, covparam, picked_covmat,
                        picked_order, options, general_knowledge,
                        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
                        gui_args)
                except Exception as e:
                    # import pdb
                    # pdb.set_trace()
                    raise e
                zk[ck_idx_piece, :] = picked_mvs
                if gui_args:
                    if gui_args[0].wasCanceled(): #cancel by user
                        return False
                    else:
                        gui_args[0].setValue(gui_args[0].value()+ck_idx_piece.size)
        return zk

def BMEPosteriorPDF_backup(
  ck, ch, cs, zh, zs=None,
  covmodel=None, covparam=None,
  order=np.nan, options=None,
  nhmax=None, nsmax=None, dmax=None,
  general_knowledge='gaussian',
  pdfk=None,pdfh=None,pdfs=None,hk_k=None,hk_h=None,hk_s=None):
  '''
  To obtain the BME posterior PDF with specified general and specific 
  knowledge PDFs
    
  Input:
  ck    N by 3    2D array of the S/T coordinates of estimation points
  ch    N by 3    2D array of the S/T coordinates of hard data
  cs    N by 3    2D array of the S/T coordinates of soft data
  zh    N by 1    2D array to specify the observed values at ch
  zs    N by k    2D array to specify the uncertain observed values at cs 
                  with the format as follows. 
                  zs = (softpdftype, mean, variance)
                  zs = (softpdftype, nl, limi, probadens)
                  The more details can go to see the function 
                  get_standard_soft_pdf_type function in starpy.bme.pystks_variable.py
  order integer   to specify the trend forms. NaN and 0 for zero and contant     
                  or string means respectively        
  options         BME options (look into BMEprobaMoments?)



  Note: the Gaussian part should be moved back into the proper places in order
  to generalize this function         
                    
  '''

    #SI = integrate (fg_s_given_kh * fs_s) dx_s
    #NC = integrate (fg_s_given_h * fs_s) dx_s
    #pdf_k = (fg_kh * SI) / (fg_h * NC)  # eq.1
    #exp_k = ... # eq.2
    #exp_kp = ... # eq.3
    #if general_knowledge == gaussian and specific_knowledge == unknown
    #exp_k = ... # eq.4
    #var_k = ... # eq.5

  # fg is gaussian, i.e., the Matlab version case

  ck,ch,cs=_changetimeform(ck,ch,cs)
  
  nhmax,nsmax,dmax=_set_nh_ns(ck,ch,cs,nhmax,nsmax,dmax)
  if dmax.size == 3 and dmax[0][2] is np.nan:
    dmax[0][2]=_stratio(covparam)

  if options is None:
    options=BMEoptions()
  
  if general_knowledge == 'gaussian':
    
    if (covmodel is None) or (covparam is None):
      print ('covariance model and their associated parameters should be specified')

    else:
      order = get_standard_order(order)
      nk = ck.shape[0]
      x_all_split = _get_x_all_split(nk, zh, zs)
      mean_all_split = _get_mean_all_split(x_all_split, order)
      cov_all_split = _get_cov_all_split(ck, ch, cs, covmodel, covparam)
      
      def fg_kh(xk):
        xk=np.array(xk)
        nlim=xk.size
        if nlim==1:
          output=_get_multivariate_normal_pdf(
              x_all_split, mean_all_split, cov_all_split, 'kh')(
                  np.vstack((xk, _get_x(x_all_split, 'h'))).T)
        else:
          xlim=[xk.reshape((nlim,1)),\
                np.ones((nlim,1)).dot(_get_x(x_all_split,'h').T)]
          output=_get_multivariate_normal_pdf(
              x_all_split, mean_all_split, cov_all_split, 'kh')(
              np.hstack(xlim))
        return output
              
      # fg_kh = lambda xk: _get_multivariate_normal_pdf(
      #         x_all_split, mean_all_split, cov_all_split, 'kh')(
      #            np.vstack((xk, _get_x(x_all_split, 'h'))).T)
      fg_h = _get_multivariate_normal_pdf(
              x_all_split, mean_all_split, cov_all_split, 'h')(
                  _get_x(x_all_split, 'h').T)
      if zs is None:
        return lambda xk: (fg_kh(xk)/fg_h)#[0][0]
      else:
        softpdftype = get_standard_soft_pdf_type(zs[0])
        if softpdftype == 10: #specific_knowledge == 'gaussian':
          SI = lambda xk: _get_int_fg_a_given_b_fs_s(
                  [xk] + x_all_split[1:], mean_all_split,
                  cov_all_split, zs, 's_kh', 's')[0]
          NC = _get_int_fg_a_given_b_fs_s(
                  x_all_split, mean_all_split,
                  cov_all_split, zs, 's_h', 's')[0]
          return lambda xk: (fg_kh(xk) * SI(xk) / fg_h / NC)#[0][0]
        elif softpdftype == 1:
          pass # to do
        elif softpdftype == 2:
          pass # to do
  else:
    nk = ck.shape[0]
    if len(pdfk) != nk:
      print ('The number of pdfk functions is not equal'+\
            'to the number of estimation locations')
      raise
    if len(hk_k) != nk:
      print ('Then number of hk_k is not equal to the number of estimation locations')
      raise
    

    mpdf=[None]*nk
    pdf_k=[None]*nk
    pdfs_kh=[None]*nk
    pdfs_h=[None]*nk
    zhlocal=[None]*nk
    zslocal=[None]*nk
    nklocal=1
    
    
    maxpts = options[2][0]
    aEps = 0
    rEps = options[3][0]
        
    for k in range(nk):
      
      if nk>1:
        print ('BMEPDF:' + str(k+1) + '/' + str(nk))
      
      cklocal=ck[k:k+1,:]
      pdfklocal=[pdfk[k]]
      hk_k_local=[hk_k[k]]
      
      chlocal, zhlocal[k], dhlocal, sumnhlocal, idxhlocal = \
            neighbours( cklocal, ch, zh, nhmax, dmax )
      pdfhlocal=[pdfh[m] for m in idxhlocal]
      hk_h_local=[hk_h[m] for m in idxhlocal]
      
      if cs is not None:
        zsdummy=np.empty((cs.shape[0],1))
        cslocal, zslocals, dslocal, sumnslocal, idxslocal = \
              neighbours( cklocal, cs, zsdummy, nsmax, dmax )
        pdfslocal=[pdfs[m] for m in idxslocal]
        hk_s_local=[hk_s[m] for m in idxslocal]
        idxslocal=idxslocal.flat
        if len(idxslocal)>0:
          zslocal[k]=[zs[0],zs[1][idxslocal,:],zs[2][idxslocal,:],zs[3][idxslocal,:]]#[zs[1][m],zs[2][m],zs[3][m]] for m in idxslocal]#[zs[m] for m in idxslocal]#              
      else:
        cslocal=cs
        zslocal=zs
        hk_s_local=hk_s
        pdfslocal=None
      
      x_all_split = _get_x_all_split(nklocal, zhlocal[k], zslocal[k])
      mean_all_split = _get_mean_all_split(x_all_split, order)
      
      # Make the covariance function into correlation function
      
      var=np.sum([covparam[m][0] for m in range(len(covparam))]) 
      for m in range(len(covparam)):
        covparam[m][0]=covparam[m][0]/var   
      
      cov_all_split = _get_cov_all_split(cklocal, chlocal, cslocal, covmodel, covparam)  
      
      if ch is not None:
        cov_kh=_get_sigma(cov_all_split, 'kh', 'kh', inv=False)
       # cov_hh=_get_sigma(cov_all_split, 'h', 'h', inv=False)
        cov_skh = _get_sigma(cov_all_split, 'skh', 'skh', inv=False)
        cov_sh = _get_sigma(cov_all_split, 'sh', 'sh', inv=False)
        
      if not isinstance(pdfk,list):
        pdfk=[pdfk]
      if not isinstance(hk_k,list):
        hk_k=[hk_k]

      from ..stats.analysis.mepdf import maxentpdf_gc, maxentcondpdf_gc  # lazy import
      pdf_k[k],_=maxentcondpdf_gc(ppdf=pdfklocal+pdfhlocal,R=cov_kh,
                             hk=hk_k_local+hk_h_local,k_num=len(pdfklocal))  
      if zslocal[k] is not None:                       
        pdfs_kh[k],_=maxentcondpdf_gc(ppdf=pdfslocal+pdfklocal+pdfhlocal,R=cov_skh,
                             hk=hk_s_local+hk_k_local+hk_h_local,
                             k_num=len(pdfslocal))
        pdfs_h[k],_=maxentcondpdf_gc(ppdf=pdfslocal+pdfhlocal,R=cov_sh,
                             hk=hk_s_local+hk_h_local,
                             k_num=len(pdfslocal))


                            
      # write up a pyallmoments here 
      # to integrate the softdata into the equation (1) calculation 
      # in the BME_OP_chapter


      #pdfs_h[k]                       

      # mpdf_kh,_=maxentpdf_gc(ppdf=pdfklocal+pdfhlocal,R=cov_kh,
      #                       hk=hk_k_local+hk_h_local)
      # mpdf_hh,_=maxentpdf_gc(pdfhlocal,cov_hh,hk_h)

      def mpdfk(xk,k):
        xk=np.array(xk)
        xk=xk.reshape((xk.size,1))
        nlim=xk.size
        up=np.empty((nlim,1))
        
        zh_n=np.ones((nlim,1)).dot(zhlocal[k].T)
        xkzh=np.hstack([xk,zh_n])
        pdfk=pdf_k[k](xkzh)
        
        if zslocal[k] is not None:
          up,_,_=pyAllMomentsNG(zslocal[k], xkzh, pdfs_kh[k],aEps,rEps,maxpts)
          bottom,_,_=pyAllMomentsNG(zslocal[k], zhlocal[k].T, pdfs_h[k],aEps,rEps,maxpts)                
          pdfk=pdfk*up/bottom

        return pdfk

      mpdf[k] = mpdfk
    
    return mpdf

def BMEprobaGaussian(ck, ch, cs, zh, zs=None,
  covmodel=None, covparam=None, order=np.nan,
  nhmax=None, nsmax=None, dmax=None, gui_args=None):
    
  '''
  The BME function considers both general and specific knowledges are Gaussian
  This function can consider the hard-only or soft-only data cases

  zs maybe = [] (empty list for no soft data)
  'if zs is not None' should' be 'if zs'
  '''
  
  ck, ch, cs = _changetimeform(ck, ch, cs)
  if nhmax is None:
      nhmax, nsmax, dmax = _set_nh_ns(ck, ch, cs, nhmax, nsmax, dmax)
    
  if zs:
      # should add GUI from here
      order = get_standard_order(order)
      #softpdftype = get_standard_soft_pdf_type(zs[0])
      nk = ck.shape[0]
      if zh is not None:
          nh = zh.shape[0]
      else:
          nh = 0
      # ns = zs[1].shape[0]
      ns = cs.shape[0] if cs is not None else 0
      x_all_split = _get_x_all_split(nk, zh, zs)
      mean_all_split = _get_mean_all_split(x_all_split, order)
      cov_all_split = _get_cov_all_split(ck, ch, cs, covmodel, covparam)
      
      mean_k = _get_mean(mean_all_split, 'k')
      mean_hs = _get_mean(mean_all_split, 'hs')
      mean_s_given_h = _get_mean_a_given_b(
                x_all_split, mean_all_split,
                cov_all_split,sub_a='s', sub_b='h')

      NC, (sigma_t_prime, inv_sigma_s_given_h,inv_sigma_tilde_s, mean_tilde_s) =\
          _get_int_fg_a_given_b_fs_s(x_all_split, mean_all_split,
                                     cov_all_split, zs, 
                                     sub_multi = 's_h', sub_s='s')

      hat_x_s = NC * sigma_t_prime.dot(
          inv_sigma_s_given_h.dot(mean_s_given_h)
          + inv_sigma_tilde_s.dot(mean_tilde_s)
          )
      if nh>0:
        hat_x_h = _get_x(x_all_split, 'h') * NC
        hat_x_hs = np.vstack((hat_x_h, hat_x_s))
      else:
        hat_x_hs = hat_x_s

      sigma_k_hs = _get_sigma(
          cov_all_split, sub_a='k', sub_b='hs')

      inv_sigma_hs_hs = _get_sigma(
          cov_all_split, sub_a='hs', sub_b='hs', inv=True)

      cond_k_hs = sigma_k_hs.dot(inv_sigma_hs_hs)

      BME_mean_k_given_hs_a = cond_k_hs.dot(hat_x_hs)

      BME_mean_k_given_hs_b = mean_k - cond_k_hs.dot(mean_hs)

      BME_mean_k_given_hs =\
          BME_mean_k_given_hs_a / NC + BME_mean_k_given_hs_b

      sigma_k_given_hs = _get_sigma_a_given_b(
          cov_all_split, sub_a='k', sub_b='hs')

      sigma_k_given_hs_diag =\
            sigma_k_given_hs.diagonal().reshape((-1,1))

      mean_t = hat_x_hs/NC
      aa = np.zeros(inv_sigma_hs_hs.shape)
      aa[nh:nh+ns, nh:nh+ns] = sigma_t_prime*NC
      bb = (mean_t - mean_hs).dot((mean_t - mean_hs).T) * NC
      tt = cond_k_hs.dot(aa + bb).dot(cond_k_hs.T)
      tt_diag = tt.diagonal().reshape((-1,1))

      BME_var_k_given_hs = (
        sigma_k_given_hs_diag - BME_mean_k_given_hs**2 + mean_k**2
        - 2*mean_k * cond_k_hs.dot(mean_hs)
        + 2*mean_k * cond_k_hs.dot(hat_x_hs) / NC
        + tt_diag / NC
        )
      
      skewness = np.zeros(BME_mean_k_given_hs.shape)
      mvs = np.hstack(
          (BME_mean_k_given_hs, BME_var_k_given_hs, skewness)
          )
      return mvs
  else: # only hard data
    order = get_standard_order(order)
    #softpdftype = get_standard_soft_pdf_type(zs[0])
    nk = ck.shape[0]
    dm = ck[0].size
    
    if dm<3 or nk<100:
        nh = zh.shape[0]
        x_all_split = _get_x_all_split(nk, zh, zs)
        mean_all_split = _get_mean_all_split(x_all_split, order)
        cov_all_split = _get_cov_all_split(ck, ch, cs, covmodel, covparam)
          
        # mean_k=_get_mean(mean_all_split,'k')   
        mean_k_given_h = _get_mean_a_given_b(
            x_all_split, mean_all_split,
            cov_all_split,sub_a='k', sub_b='h')
        sigma_k_given_h = _get_sigma_a_given_b(
            cov_all_split, sub_a='k', sub_b='h')
        return mean_k_given_h, sigma_k_given_h
    else:
        dummy = np.random.rand(ck.shape[0],1)
        _, cMS_k, tME_k, _ = valstv2stg(ck, dummy)
        nklocal = cMS_k.shape[0]
        mean_k_given_h = np.empty((nklocal, 0))
        sigma_k_given_h = np.empty((nklocal, 0))

        # Here should add spatial split for large spatial data at a time
        # or GUI will become freezed
        for tt in range(tME_k.size):
            cklocal =\
                np.hstack([
                    np.mean(cMS_k,0), tME_k[tt]
                    ]).reshape(1, 3)
            cklocals =\
                np.hstack([
                    cMS_k, np.ones((nklocal, 1))*tME_k[tt]
                    ])
            chlocal, zhlocal, dhlocal, sumnhlocal, idxhlocal = \
                neighbours(cklocal, ch, zh, nhmax, dmax)
            cslocal = cs
            zslocal = zs

            x_all_split = _get_x_all_split(nklocal, zhlocal, zslocal)
            mean_all_split = _get_mean_all_split(x_all_split, order)
            cov_all_split = _get_cov_all_split(
                cklocals, chlocal, cslocal, covmodel, covparam)
          
            mean_k_given_h_ = _get_mean_a_given_b(
                      x_all_split, mean_all_split,
                      cov_all_split,sub_a='k', sub_b='h')
            sigma_k_given_h_ = np.diag(
                _get_sigma_a_given_b(
                    cov_all_split, sub_a='k', sub_b='h'
                    )
                ).reshape(nklocal, 1)
                      
            mean_k_given_h=np.hstack([mean_k_given_h,mean_k_given_h_])
            sigma_k_given_h=np.hstack([sigma_k_given_h,sigma_k_given_h_])
            print (str(tt+1) + '/' + str(tME_k.size))
            if gui_args:
                gui_args[0].setValue(cMS_k.shape[0]*(tt+1))
        ck2,mean_k_given_h_v=valstg2stv(mean_k_given_h, cMS_k, tME_k)
        ck2,sigma_k_given_h_v=valstg2stv(sigma_k_given_h, cMS_k, tME_k)
        
        # ck != ck2 will occur when ck is not get from grid input
        # need to be fixed ASAP.
        if not np.all(ck2==ck):
            print ('warning: ck and ck2 are not the same')
            raise ValueError('Now ck only can input with grid.')

        return mean_k_given_h_v, sigma_k_given_h_v


# =============================================================================
# All-Gaussian fast-path routing helper
# =============================================================================

def _check_all_gaussian_zs(zs, options=None):
    """Mirror the Gaussian routing logic from ``_bme_posterior_moments``.

    Determines whether every soft datum in *zs* is (or can be treated as)
    Gaussian, respecting the ``'soft_approx_gaussian'`` and automatic
    high-dimensionality fallback rules already present in the Moments engine.

    Parameters
    ----------
    zs : list or None
        Sequence of soft-data tuples in the internal
        ``(softpdftype, ...)`` format accepted by ``BMEPosteriorMoments``.
        Each element is one of:

        * ``(10, mean, variance)``             — Gaussian soft datum
        * ``(1, nl_arr, limi_1d, pd_1d)``      — Histogram soft datum
        * ``(2, nl_arr, limi_1d, pd_1d)``      — Linear soft datum
    options : BMEOptions or None
        BME options object.  The ``'soft_approx_gaussian'`` key is read;
        ``None`` is treated as all defaults (no approximation).

    Returns
    -------
    all_gaussian : bool
        ``True`` iff every element in *zs_out* has soft-PDF type 10.
    zs_out : list
        The (possibly converted) soft-data list with non-Gaussian types
        replaced by Gaussian ``(10, mean, var)`` tuples when the
        approximation flag is set or the auto-threshold is exceeded.

    Notes
    -----
    The auto-Gaussian threshold mirrors ``_bme_posterior_moments``:
    when more than 20 soft data points are non-Gaussian, numerical
    integration becomes intractable and the Gaussian approximation is
    applied automatically.
    """
    if not zs:
        return True, []

    _use_gauss_approx = (
        options is not None
        and options.get('soft_approx_gaussian', False))

    _NS_AUTO_GAUSS = 20  # mirrors _bme_posterior_moments threshold
    if not _use_gauss_approx:
        _ns_nongauss = sum(
            1 for zsi in zs
            if get_standard_soft_pdf_type(zsi[0]) != 10)
        if _ns_nongauss > _NS_AUTO_GAUSS:
            _use_gauss_approx = True

    if _use_gauss_approx:
        zs_out = []
        for zsi in zs:
            pdftype = get_standard_soft_pdf_type(zsi[0])
            if pdftype in [1, 2]:
                m, v = proba2stat(
                    zsi[0],
                    np.atleast_2d(zsi[1]),
                    np.atleast_2d(zsi[2]),
                    np.atleast_2d(zsi[3]))
                zs_out.append(
                    (10,
                     float(np.asarray(m).flat[0]),
                     float(np.asarray(v).flat[0])))
            else:
                zs_out.append(zsi)
        zs = zs_out

    all_gaussian = all(
        get_standard_soft_pdf_type(zsi[0]) == 10 for zsi in zs)
    return all_gaussian, zs


def _get_gauss_approx_moments(
        ck, ch, cs, zh, zs,
        covmodel, covparam, covmat, order, options,
        nhmax, nsmax, dmax, general_knowledge,
        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s, gui_args):
    """Return fast Gaussian-approximated moments for use as optimization bounds.

    Forces ``soft_approx_gaussian=True`` on a *copy* of *options* so that
    non-Gaussian soft PDFs are matched to Gaussian (mean, variance) before
    calling ``BMEPosteriorMoments``.  The result is inexpensive (analytical,
    no QMC) and is used only as a bracket for mode/CI optimization.

    Parameters
    ----------
    (same as ``BMEPosteriorMoments``)

    Returns
    -------
    moments : ndarray of shape (nk, 3)
        Approximate ``[mean, variance, skewness]`` for each estimation point.
    """
    import copy as _copy
    if options is None:
        from .BMEoptions import BMEoptions as _BMEopts
        opts_approx = _BMEopts()
    else:
        opts_approx = _copy.deepcopy(options)
    opts_approx['soft_approx_gaussian'] = True

    return BMEPosteriorMoments(
        ck, ch, cs, zh, zs,
        covmodel, covparam, covmat,
        order, opts_approx,
        nhmax, nsmax, dmax,
        general_knowledge,
        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
        gui_args, ck_cov_output=False)


# =============================================================================
# BMEPosteriorMode
# =============================================================================

def BMEPosteriorMode(
        ck,
        ch=None,
        cs=None,
        zh=None,
        zs=None,
        covmodel=None,
        covparam=None,
        covmat=None,
        order=np.nan,
        options=None,
        nhmax=None,
        nsmax=None,
        dmax=None,
        general_knowledge='gaussian',
        pdfk=None,
        pdfh=None,
        pdfs=None,
        hk_k=None,
        hk_h=None,
        hk_s=None,
        gui_args=None):
    """BME mode prediction with probabilistic soft data.

    Adapted from ``BMEprobaMode.m`` (BMElib bmeprobalib, Jan 1, 2001).

    BME computation of the **mode** of the posterior PDF at a set of
    estimation points, using both hard data and soft probabilistic data.
    The function covers the most general BME scenario (multi-field,
    nested covariance models, space-time, non-stationary mean) and
    reduces to the kriging estimate when only hard data is available.

    **All-Gaussian analytical fast-path** — When every soft datum has
    type 10 (Gaussian), or when the ``'soft_approx_gaussian'`` option is
    ``True``, the posterior is exactly Gaussian and the mode equals the
    analytical posterior mean.  In this case ``scipy.optimize`` is never
    invoked and no QMC integration is performed.

    Parameters
    ----------
    ck : ndarray of shape (nk, d)
        Coordinates of the nk estimation locations.  Each row is a
        coordinate vector; the number of columns d is the space dimension
        (typically 3 for space-time: x, y, t).
    ch : ndarray of shape (nh, d) or None
        Coordinates of the nh hard-data locations.
    cs : ndarray of shape (ns, d) or None
        Coordinates of the ns soft-data locations.
    zh : ndarray of shape (nh, 1) or None
        Observed values at the hard-data locations *ch*.
    zs : list of tuples or None
        Soft-data records.  Each element is a tuple
        ``(softpdftype, ...)`` whose remaining fields depend on the type:

        * ``(10, mean, variance)``         — Gaussian
        * ``(1,  nl_arr, limi, probdens)`` — Histogram (type 1)
        * ``(2,  nl_arr, limi, probdens)`` — Linear piecewise (type 2)

        See ``probasyntax`` / ``get_standard_soft_pdf_type`` for details.
        Pass ``None`` or an empty list when no soft data are available.
    covmodel : str or callable
        Name of the covariance model (see ``modelslib``).  Variogram
        models are not supported.
    covparam : array_like of shape (1, k)
        Parameters for *covmodel*.
    covmat : ndarray of shape (nk+nh+ns, nk+nh+ns) or None
        Pre-computed covariance matrix.  When provided, *covmodel* and
        *covparam* are ignored.
    order : float or ndarray
        Polynomial drift order along the spatial axes.
        Use ``np.nan`` (default) for a zero-mean field, ``0`` for
        constant mean, ``1`` for linear, etc.
    options : BMEOptions or None
        BME options object returned by ``BMEoptions()``.  Key entries:

        * ``'soft_approx_gaussian'`` (bool) — force Gaussian approximation
          of all soft PDFs; enables the analytical fast-path.
        * ``[1,0]``, ``[3,0]``, ``[2,0]`` — QMC error tolerances and
          maximum function evaluations (used in the non-Gaussian path).
    nhmax : int or None
        Maximum number of hard-data points in the local neighbourhood.
    nsmax : int or None
        Maximum number of soft-data points.  Values above 20 risk
        numerical issues with QMC integration.
    dmax : float or None
        Maximum search radius for local neighbourhood selection.
    general_knowledge : str
        ``'gaussian'`` (default) for Gaussian general knowledge
        (prior covariance model).
    pdfk, pdfh, pdfs, hk_k, hk_h, hk_s : optional
        Parameters for non-Gaussian general-knowledge specifications
        (advanced use; rarely needed).
    gui_args : tuple or None
        GUI progress-bar arguments ``(QProgressDialog, ...)`` for
        cancellation support.

    Returns
    -------
    zk_mode : ndarray of shape (nk, 1)
        Mode of the BME posterior PDF at each estimation location.
        A value coded as ``NaN`` means no estimation was performed
        (insufficient data in the local neighbourhood).
    info : ndarray of shape (nk, 1), dtype int
        Status codes for each estimation location:

        * ``NaN`` — no computation (no hard or soft data nearby)
        * ``0``   — BME with at least 1 soft datum; converged
        * ``2``   — mode optimization did not converge within the
                    maximum number of iterations (dubious result)
        * ``3``   — kriging only, no soft data in neighbourhood
        * ``4``   — hard datum present at the estimation location
        * ``10``  — dubious results from the PDF evaluation step

    Notes
    -----
    All conventions for nested covariance models, multivariate or
    space-time cases are identical to those of ``BMEPosteriorMoments``.

    The non-Gaussian mode is found by:

    1. Computing fast Gaussian-approximated moments to derive an
       initial search bracket ``[mean − 4σ, mean + 4σ]``.
    2. Evaluating the un-normalized posterior PDF (from
       ``BMEPosteriorPDF``) on a coarse grid of 50 points.
    3. Refining the grid maximum with
       ``scipy.optimize.minimize_scalar(method='bounded')``.

    For the all-Gaussian case the mode is returned analytically as the
    posterior mean (no optimization step at all).

    See Also
    --------
    BMEPosteriorMoments : Posterior mean and variance.
    BMEPosteriorCI      : Posterior credible interval.
    BMEPosteriorPDF     : Posterior PDF closure objects.
    """
    from scipy.optimize import minimize_scalar as _minimize_scalar

    nk = np.atleast_2d(ck).shape[0]
    ns = np.atleast_2d(cs).shape[0] if cs is not None else 0
    zs = _normalize_zs(zs, ns)
    zs_list = list(zs) if zs is not None else []

    # ------------------------------------------------------------------
    # Fast-path: all soft data Gaussian (or none) → Mode = Mean exactly
    # ------------------------------------------------------------------
    _all_gauss, _ = _check_all_gaussian_zs(zs_list, options)

    if _all_gauss:
        moments = BMEPosteriorMoments(
            ck, ch, cs, zh, zs,
            covmodel, covparam, covmat,
            order, options,
            nhmax, nsmax, dmax,
            general_knowledge,
            pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
            gui_args, ck_cov_output=False)
        zk_mode = moments[:, 0:1].copy()   # (nk, 1)
        info = np.full((nk, 1), 0.0)
        if not zs_list:
            info[:] = 3   # kriging only, no soft data
        return zk_mode, info

    # ------------------------------------------------------------------
    # Non-Gaussian path
    # ------------------------------------------------------------------
    # Step 1: fast Gaussian-approximated moments → optimization brackets
    approx_moments = _get_gauss_approx_moments(
        ck, ch, cs, zh, zs,
        covmodel, covparam, covmat, order, options,
        nhmax, nsmax, dmax, general_knowledge,
        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s, gui_args)

    # Step 2: get posterior PDF closures (one per estimation point)
    pdf_closures = BMEPosteriorPDF(
        ck, ch, cs, zh, zs,
        covmodel, covparam, covmat,
        order, options,
        nhmax, nsmax, dmax,
        general_knowledge,
        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s, gui_args)

    zk_mode = np.full((nk, 1), np.nan)
    info = np.full((nk, 1), 0.0)

    _N_COARSE = 50   # coarse grid resolution
    _NONNEG = (options is not None and options.get('nonneg_estimate', False))

    for k in range(nk):
        m_k = float(approx_moments[k, 0])
        v_k = float(approx_moments[k, 1]) if not np.isnan(approx_moments[k, 1]) else 1.0
        sig_k = np.sqrt(max(v_k, 1e-14))

        zmin_k = m_k - 4.0 * sig_k
        zmax_k = m_k + 4.0 * sig_k
        if _NONNEG:
            zmin_k = max(zmin_k, 0.0)
        if zmin_k >= zmax_k:
            zmax_k = zmin_k + 8.0 * max(sig_k, 1e-6)

        pdf_fn = pdf_closures[k, 0]   # callable: z_1d → pdf_1d

        try:
            # Coarse grid search
            z_coarse = np.linspace(zmin_k, zmax_k, _N_COARSE)
            pdf_coarse = pdf_fn(z_coarse)
            if pdf_coarse is None or np.all(np.isnan(pdf_coarse)):
                info[k, 0] = 10
                continue
            i_max = int(np.nanargmax(pdf_coarse))

            # Bounded refinement around coarse maximum
            z_lo = z_coarse[max(0, i_max - 1)]
            z_hi = z_coarse[min(_N_COARSE - 1, i_max + 1)]
            if z_lo >= z_hi:
                zk_mode[k, 0] = z_coarse[i_max]
                info[k, 0] = 0
                continue

            res = _minimize_scalar(
                lambda z_val: -float(
                    np.nanmax(pdf_fn(np.array([z_val])))),
                bounds=(z_lo, z_hi),
                method='bounded',
                options={'xatol': 1e-6, 'maxiter': 500})

            zk_mode[k, 0] = float(res.x)
            info[k, 0] = 0 if res.success else 2

        except Exception:
            # Fallback: coarse-grid maximum
            try:
                zk_mode[k, 0] = float(z_coarse[i_max])
            except Exception:
                pass
            info[k, 0] = 10

    return zk_mode, info


# =============================================================================
# BMEPosteriorPDF  (evaluates posterior PDF on automatic or explicit z-grid)
# =============================================================================

def BMEPosteriorPDF(
        ck,
        ch=None,
        cs=None,
        zh=None,
        zs=None,
        covmodel=None,
        covparam=None,
        covmat=None,
        order=np.nan,
        options=None,
        nhmax=None,
        nsmax=None,
        dmax=None,
        general_knowledge='gaussian',
        pdfk=None,
        pdfh=None,
        pdfs=None,
        hk_k=None,
        hk_h=None,
        hk_s=None,
        gui_args=None,
        n_grid=200,
        z_grid=None):
    """Evaluate the BME posterior PDF at each estimation location.

    Adapted from ``BMEprobaPdf.m`` (BMElib bmeprobalib, Jan 1, 2001).

    BME computation of the posterior probability density function at a set
    of estimation points, using both hard data and soft probabilistic data.
    The function handles the complete BME scenario — multiple field variables,
    nested covariance models, space-time fields, non-stationary mean — and
    reduces to the kriging posterior PDF when only hard data is available.

    The z evaluation range is determined automatically from the data and
    model: Gaussian-approximated posterior moments are computed at each
    estimation location, and the z-grid spans ``[mean − 4σ, mean + 4σ]``
    per point.  An explicit *z_grid* may be supplied to override this.

    **All-Gaussian analytical fast-path** — When every soft datum is
    Gaussian (type 10), or when ``'soft_approx_gaussian'`` is ``True``,
    the posterior is exactly Gaussian and the PDF is evaluated analytically
    using ``scipy.stats.norm.pdf``.  No QMC integration is performed.

    Parameters
    ----------
    ck : ndarray of shape (nk, d)
        Coordinates of the nk estimation locations.
    ch : ndarray of shape (nh, d) or None
        Coordinates of the hard-data locations.
    cs : ndarray of shape (ns, d) or None
        Coordinates of the soft-data locations.
    zh : ndarray of shape (nh, 1) or None
        Observed values at *ch*.
    zs : list of tuples or None
        Soft-data records.  See ``BMEPosteriorMoments`` for format.
    covmodel : str or callable
        Covariance model name.
    covparam : array_like
        Covariance model parameters.
    covmat : ndarray or None
        Pre-computed full covariance matrix.
    order : float or ndarray
        Polynomial drift order (``np.nan`` for zero mean).
    options : BMEOptions or None
        BME options object.
    nhmax, nsmax : int or None
        Neighbourhood size limits.
    dmax : float or None
        Maximum neighbourhood search radius.
    general_knowledge : str
        ``'gaussian'`` (default).
    pdfk, pdfh, pdfs, hk_k, hk_h, hk_s : optional
        Advanced parameters for non-Gaussian general knowledge.
    gui_args : tuple or None
        GUI cancellation hook.
    n_grid : int
        Number of z-grid points used when *z_grid* is ``None`` (auto-range).
        Default 200.
    z_grid : array_like of shape (nz,) or None
        Optional explicit z values at which to evaluate the posterior PDF.
        When ``None`` (default), an automatic per-point grid of *n_grid*
        equally spaced points spanning ``[mean − 4σ, mean + 4σ]`` is
        constructed from Gaussian-approximated moments.  When supplied, all
        estimation points share this grid and the returned *pdf_values* is a
        regular 2-D ``(nk, nz)`` array.

    Returns
    -------
    z_out : ndarray of shape (nz,) or list of nk ndarrays
        Z values at which the PDF is evaluated.  A single shared array
        when *z_grid* is supplied; a list of per-point arrays when
        *z_grid* is ``None``.
    pdf_values : ndarray of shape (nk, nz) or list of nk ndarrays
        Normalised PDF values.  Each row (or list element) is the posterior
        PDF for the corresponding estimation point.
    info : ndarray of shape (nk,)
        Status codes:

        * ``NaN`` — no computation (no hard or soft data)
        * ``0``   — BME with at least 1 soft datum
        * ``3``   — kriging only (no soft data)
        * ``10``  — dubious results from PDF evaluation

    Notes
    -----
    When *z_grid* is supplied, all estimation points share the same grid,
    and the returned *pdf_values* is a regular 2-D array.  When *z_grid*
    is ``None``, each point gets its own grid adapted to its posterior
    range; the output is a list of 1-D arrays.

    See Also
    --------
    BMEPosteriorMoments : Posterior mean and variance.
    BMEPosteriorMode    : Posterior mode.
    BMEPosteriorCI      : Posterior credible interval.
    """
    nk = np.atleast_2d(ck).shape[0]
    ns = np.atleast_2d(cs).shape[0] if cs is not None else 0
    zs = _normalize_zs(zs, ns)
    zs_list = list(zs) if zs is not None else []

    # Use a shared grid?
    _auto_grid = (z_grid is None or (
        hasattr(z_grid, '__len__') and len(z_grid) == 0))

    # ------------------------------------------------------------------
    # All-Gaussian fast-path
    # ------------------------------------------------------------------
    _all_gauss, _ = _check_all_gaussian_zs(zs_list, options)

    if _all_gauss:
        moments = BMEPosteriorMoments(
            ck, ch, cs, zh, zs,
            covmodel, covparam, covmat,
            order, options,
            nhmax, nsmax, dmax,
            general_knowledge,
            pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
            gui_args, ck_cov_output=False)

        info = np.zeros(nk)
        if not zs_list:
            info[:] = 3

        if _auto_grid:
            z_out_list = []
            pdf_out_list = []
            for k in range(nk):
                m_k = float(moments[k, 0])
                v_k = float(moments[k, 1]) if not np.isnan(moments[k, 1]) else 1.0
                sig_k = np.sqrt(max(v_k, 1e-14))
                zg = np.linspace(m_k - 4.0 * sig_k, m_k + 4.0 * sig_k, n_grid)
                pg = scipy.stats.norm.pdf(zg, loc=m_k, scale=sig_k)
                z_out_list.append(zg)
                pdf_out_list.append(pg)
            return z_out_list, pdf_out_list, info
        else:
            z_arr = np.asarray(z_grid, dtype=float).ravel()
            pdf_values = np.empty((nk, len(z_arr)))
            for k in range(nk):
                m_k = float(moments[k, 0])
                v_k = float(moments[k, 1]) if not np.isnan(moments[k, 1]) else 1.0
                sig_k = np.sqrt(max(v_k, 1e-14))
                pdf_values[k, :] = scipy.stats.norm.pdf(
                    z_arr, loc=m_k, scale=sig_k)
            return z_arr, pdf_values, info

    # ------------------------------------------------------------------
    # Non-Gaussian path: evaluate PDF closures on z_grid
    # ------------------------------------------------------------------
    # Step 1: fast approximate moments for default grid construction
    approx_moments = _get_gauss_approx_moments(
        ck, ch, cs, zh, zs,
        covmodel, covparam, covmat, order, options,
        nhmax, nsmax, dmax, general_knowledge,
        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s, gui_args)

    # Step 2: PDF closures
    pdf_closures = _bme_posterior_pdf_closures(
        ck, ch, cs, zh, zs,
        covmodel, covparam, covmat,
        order, options,
        nhmax, nsmax, dmax,
        general_knowledge,
        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s, gui_args)

    info = np.zeros(nk)
    _NONNEG = (options is not None and options.get('nonneg_estimate', False))

    if _auto_grid:
        z_out_list = []
        pdf_out_list = []
        for k in range(nk):
            m_k = float(approx_moments[k, 0])
            v_k = float(approx_moments[k, 1]) if not np.isnan(approx_moments[k, 1]) else 1.0
            sig_k = np.sqrt(max(v_k, 1e-14))
            zmin_k = m_k - 4.0 * sig_k
            zmax_k = m_k + 4.0 * sig_k
            if _NONNEG:
                zmin_k = max(zmin_k, 0.0)
            zg = np.linspace(zmin_k, zmax_k, n_grid)
            try:
                pg = pdf_closures[k, 0](zg)
                if pg is None or np.all(np.isnan(pg)):
                    pg = np.zeros(n_grid)
                    info[k] = 10
            except Exception:
                pg = np.zeros(n_grid)
                info[k] = 10
            z_out_list.append(zg)
            pdf_out_list.append(np.asarray(pg, dtype=float))
        return z_out_list, pdf_out_list, info
    else:
        z_arr = np.asarray(z_grid, dtype=float).ravel()
        nz = len(z_arr)
        pdf_values = np.zeros((nk, nz))
        for k in range(nk):
            try:
                pv = pdf_closures[k, 0](z_arr)
                if pv is not None and not np.all(np.isnan(pv)):
                    pdf_values[k, :] = np.asarray(pv, dtype=float).ravel()
                else:
                    info[k] = 10
            except Exception:
                info[k] = 10
        return z_arr, pdf_values, info


def BMEPosteriorPDF_grid(
        z_grid,
        ck,
        ch=None,
        cs=None,
        zh=None,
        zs=None,
        covmodel=None,
        covparam=None,
        covmat=None,
        order=np.nan,
        options=None,
        nhmax=None,
        nsmax=None,
        dmax=None,
        general_knowledge='gaussian',
        pdfk=None,
        pdfh=None,
        pdfs=None,
        hk_k=None,
        hk_h=None,
        hk_s=None,
        gui_args=None,
        n_grid=200):
    """Deprecated alias for :func:`BMEPosteriorPDF`.

    .. deprecated::
        Use ``BMEPosteriorPDF(ck, ..., z_grid=z_grid)`` instead.
        The parameter order has changed: *ck* is now the first argument and
        *z_grid* is an optional keyword argument at the end.
    """
    import warnings
    warnings.warn(
        "BMEPosteriorPDF_grid is deprecated. "
        "Use BMEPosteriorPDF(ck, ..., z_grid=z_grid) instead. "
        "Note: the parameter order has changed — ck is now first.",
        DeprecationWarning,
        stacklevel=2,
    )
    return BMEPosteriorPDF(
        ck,
        ch=ch, cs=cs, zh=zh, zs=zs,
        covmodel=covmodel, covparam=covparam, covmat=covmat,
        order=order, options=options,
        nhmax=nhmax, nsmax=nsmax, dmax=dmax,
        general_knowledge=general_knowledge,
        pdfk=pdfk, pdfh=pdfh, pdfs=pdfs,
        hk_k=hk_k, hk_h=hk_h, hk_s=hk_s,
        gui_args=gui_args,
        n_grid=n_grid,
        z_grid=z_grid,
    )


# =============================================================================
# BMEPosteriorCI
# =============================================================================

def BMEPosteriorCI(
        ck,
        ch=None,
        cs=None,
        zh=None,
        zs=None,
        covmodel=None,
        covparam=None,
        covmat=None,
        order=np.nan,
        options=None,
        nhmax=None,
        nsmax=None,
        dmax=None,
        general_knowledge='gaussian',
        ci_probs=None,
        pdfk=None,
        pdfh=None,
        pdfs=None,
        hk_k=None,
        hk_h=None,
        hk_s=None,
        gui_args=None,
        n_grid=400):
    """BME posterior credible interval with probabilistic soft data.

    Adapted from ``BMEprobaCI.m`` (BMElib bmeprobalib, Jan 1, 2001).

    BME computation of the **credible interval** (CI) and the BME posterior
    PDF at a set of estimation points, using both hard data and soft
    probabilistic data.  The function is intended to be as general as
    possible, covering multi-field variables, nested covariance models,
    space-time fields, and non-stationary mean.

    In the most common case the CI is a single contiguous interval
    ``[zlCI, zuCI]``.  In principle the posterior may have multiple
    disconnected modes; in such cases the returned bounds enclose the
    union of all sub-intervals that collectively attain the required
    probability.

    **All-Gaussian analytical fast-path** — When every soft datum is
    Gaussian (type 10), or when ``'soft_approx_gaussian'`` is ``True``,
    the posterior is exactly Gaussian and the CI is evaluated analytically
    using ``scipy.stats.norm.interval``.  No QMC integration is performed.

    Parameters
    ----------
    ck : ndarray of shape (nk, d)
        Coordinates of the nk estimation locations.
    ch : ndarray of shape (nh, d) or None
        Coordinates of the hard-data locations.
    cs : ndarray of shape (ns, d) or None
        Coordinates of the soft-data locations.
    zh : ndarray of shape (nh, 1) or None
        Observed values at *ch*.
    zs : list of tuples or None
        Soft-data records.  See ``BMEPosteriorMoments`` for format.
    covmodel : str or callable
        Covariance model name.
    covparam : array_like
        Covariance model parameters.
    covmat : ndarray or None
        Pre-computed full covariance matrix.
    order : float or ndarray
        Polynomial drift order (``np.nan`` for zero mean).
    options : BMEOptions or None
        BME options object.  When this object contains an entry
        ``options[19:29]`` (0-based indices 19–28) with confidence
        probabilities, those values are used as *ci_probs* (matching the
        MATLAB ``options(20:29)`` convention with 1-based indexing).
        The explicit *ci_probs* keyword argument always takes precedence.
    nhmax, nsmax : int or None
        Neighbourhood size limits.
    dmax : float or None
        Maximum search radius.
    general_knowledge : str
        ``'gaussian'`` (default).
    ci_probs : array_like of float or None
        Confidence probability levels, e.g. ``[0.68, 0.95]``.  Values
        must lie in ``[0.01, 0.99]``.  Defaults to ``[0.68]`` (≈ 1σ
        interval).  Mirrors ``options(20:29)`` in the MATLAB version.
    pdfk, pdfh, pdfs, hk_k, hk_h, hk_s : optional
        Advanced parameters for non-Gaussian general knowledge.
    gui_args : tuple or None
        GUI cancellation hook.
    n_grid : int
        Number of grid points used to evaluate the posterior PDF in the
        non-Gaussian path.  Default 400.

    Returns
    -------
    zlCI : ndarray of shape (nk, nCI)
        Lower bounds of the credible intervals.  ``zlCI[i, j]`` is the
        lower bound for estimation point i at confidence level
        ``PCI[j]``.
    zuCI : ndarray of shape (nk, nCI)
        Upper bounds of the credible intervals.
    pdfCI : ndarray of shape (nk, nCI)
        PDF values at the lower CI bounds (``pdfCI[i, j]``).  For a
        symmetric unimodal posterior the PDF at the lower and upper
        bounds is approximately equal.
    PCI : ndarray of shape (nCI,)
        Confidence probability levels (sorted, same as *ci_probs* after
        filtering to ``[0.01, 0.99]``).
    z_grids : list of nk ndarrays
        Z-grid on which the posterior PDF was evaluated for each
        estimation point.
    pdf_grids : list of nk ndarrays
        Normalised posterior PDF values on *z_grids*.

    Notes
    -----
    All conventions for nested covariance models, multivariate or
    space-time cases are identical to those of ``BMEPosteriorMoments``.

    The non-Gaussian CI is computed numerically:

    1. Fast Gaussian-approximated moments set the evaluation range.
    2. The posterior PDF is evaluated on a fine grid via
       ``BMEPosteriorPDF``.
    3. The PDF is normalised by trapezoidal integration.
    4. The CDF is built by cumulative trapezoidal integration.
    5. The CI bounds are found by linear interpolation on the CDF.

    For the all-Gaussian case the interval is returned analytically via
    ``scipy.stats.norm.interval(ci_prob, loc=mean, scale=sqrt(var))``.

    See Also
    --------
    BMEPosteriorMoments : Posterior mean and variance.
    BMEPosteriorMode    : Posterior mode.
    BMEPosteriorPDF     : Posterior PDF on an automatic or explicit z-grid.
    """
    nk = np.atleast_2d(ck).shape[0]
    _ns_ci = np.atleast_2d(cs).shape[0] if cs is not None else 0
    # Normalise batch → per-point format (same as BMEPosteriorMoments)
    zs = _normalize_zs(zs, _ns_ci)
    zs_list = list(zs) if zs is not None else []

    # ------------------------------------------------------------------
    # Parse and validate ci_probs
    # ------------------------------------------------------------------
    if ci_probs is None:
        # Try to read from options (mirrors options(20:29) in MATLAB,
        # i.e. 0-based index 19:29)
        _ci_from_opts = None
        if options is not None:
            try:
                _ci_from_opts = np.asarray(options[19:29], dtype=float)
                _ci_from_opts = _ci_from_opts[
                    (~np.isnan(_ci_from_opts))
                    & (_ci_from_opts >= 0.01)
                    & (_ci_from_opts <= 0.99)]
            except Exception:
                pass
        if _ci_from_opts is not None and len(_ci_from_opts) > 0:
            ci_probs = list(_ci_from_opts)
        else:
            ci_probs = [0.68]   # default: ~1σ interval

    PCI = np.asarray(
        [p for p in np.atleast_1d(ci_probs)
         if 0.01 <= float(p) <= 0.99],
        dtype=float)
    if len(PCI) == 0:
        _warnings_mod.warn(
            "[BMEPosteriorCI] No valid ci_probs in [0.01, 0.99]. "
            "Using default [0.68].")
        PCI = np.array([0.68])
    PCI = np.sort(PCI)
    nCI = len(PCI)

    # ------------------------------------------------------------------
    # All-Gaussian analytical fast-path
    # ------------------------------------------------------------------
    _all_gauss, _ = _check_all_gaussian_zs(zs_list, options)

    if _all_gauss:
        moments = BMEPosteriorMoments(
            ck, ch, cs, zh, zs,
            covmodel, covparam, covmat,
            order, options,
            nhmax, nsmax, dmax,
            general_knowledge,
            pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
            gui_args, ck_cov_output=False)

        zlCI = np.empty((nk, nCI))
        zuCI = np.empty((nk, nCI))
        pdfCI = np.empty((nk, nCI))
        z_grids = []
        pdf_grids = []

        for k in range(nk):
            m_k = float(moments[k, 0])
            v_k = float(moments[k, 1]) if not np.isnan(moments[k, 1]) else 1.0
            sig_k = np.sqrt(max(v_k, 1e-14))

            # Shared z-grid for this point
            zg = np.linspace(m_k - 4.0 * sig_k, m_k + 4.0 * sig_k, n_grid)
            pg = scipy.stats.norm.pdf(zg, loc=m_k, scale=sig_k)
            z_grids.append(zg)
            pdf_grids.append(pg)

            for j, p_j in enumerate(PCI):
                lo, hi = scipy.stats.norm.interval(
                    float(p_j), loc=m_k, scale=sig_k)
                zlCI[k, j] = lo
                zuCI[k, j] = hi
                pdfCI[k, j] = scipy.stats.norm.pdf(lo, loc=m_k, scale=sig_k)

        return zlCI, zuCI, pdfCI, PCI, z_grids, pdf_grids

    # ------------------------------------------------------------------
    # Non-Gaussian path: numerical CDF inversion
    # ------------------------------------------------------------------
    zlCI = np.full((nk, nCI), np.nan)
    zuCI = np.full((nk, nCI), np.nan)
    pdfCI = np.full((nk, nCI), np.nan)
    z_grids = []
    pdf_grids = []

    # Evaluate the full posterior PDF on per-point grids
    z_out_list, pdf_out_list, _info = BMEPosteriorPDF(
        ck, ch, cs, zh, zs,
        covmodel, covparam, covmat,
        order, options,
        nhmax, nsmax, dmax,
        general_knowledge,
        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s, gui_args,
        n_grid=n_grid,
        z_grid=None,   # auto grid
    )

    for k in range(nk):
        zg = z_out_list[k]
        pg = pdf_out_list[k].copy()
        z_grids.append(zg)

        # Normalise PDF  (np.trapz removed in NumPy 2.0 → use np.trapezoid)
        area = np.trapezoid(pg, zg)
        if area <= 0 or np.isnan(area):
            pdf_grids.append(pg)
            continue
        pg_norm = pg / area
        pdf_grids.append(pg_norm)

        # Build CDF by cumulative trapezoidal integration
        from scipy.integrate import cumulative_trapezoid
        cdf = cumulative_trapezoid(pg_norm, zg, initial=0.0)
        cdf = np.clip(cdf, 0.0, 1.0)

        for j, p_j in enumerate(PCI):
            # Equal-tails CI: (1 - p_j)/2 and (1 + p_j)/2 quantiles
            alpha_lo = (1.0 - float(p_j)) / 2.0
            alpha_hi = 1.0 - alpha_lo

            try:
                zlCI[k, j] = float(np.interp(alpha_lo, cdf, zg))
                zuCI[k, j] = float(np.interp(alpha_hi, cdf, zg))
                pdfCI[k, j] = float(
                    np.interp(zlCI[k, j], zg, pg_norm))
            except Exception:
                pass

    return zlCI, zuCI, pdfCI, PCI, z_grids, pdf_grids


# =============================================================================
# BMEPosteriorQtl
# =============================================================================

def BMEPosteriorQtl(
        ck,
        ch=None,
        cs=None,
        zh=None,
        zs=None,
        covmodel=None,
        covparam=None,
        covmat=None,
        order=np.nan,
        options=None,
        nhmax=None,
        nsmax=None,
        dmax=None,
        general_knowledge='gaussian',
        quantiles=None,
        pdfk=None,
        pdfh=None,
        pdfs=None,
        hk_k=None,
        hk_h=None,
        hk_s=None,
        gui_args=None,
        n_grid=400):
    """BME posterior quantiles at arbitrary probability levels.

    Returns the z-values corresponding to requested quantile levels of the
    posterior distribution at each estimation location.  This is a direct
    quantile-extraction alternative to :func:`BMEPosteriorCI`: where
    ``BMEPosteriorCI`` computes symmetric equal-tail credible intervals and
    returns six outputs, ``BMEPosteriorQtl`` accepts arbitrary quantile
    levels and returns a single clean ``(nk, nq)`` array.

    **All-Gaussian analytical fast-path** — When every soft datum is
    Gaussian (type 10), or when ``'soft_approx_gaussian'`` is ``True``,
    the posterior is exactly Gaussian and quantiles are evaluated analytically
    using ``scipy.stats.norm.ppf``.  No QMC integration is performed.

    Parameters
    ----------
    ck : ndarray of shape (nk, d)
        Coordinates of the nk estimation locations.
    ch : ndarray of shape (nh, d) or None
        Coordinates of the hard-data locations.
    cs : ndarray of shape (ns, d) or None
        Coordinates of the soft-data locations.
    zh : ndarray of shape (nh, 1) or None
        Observed values at *ch*.
    zs : list of tuples or None
        Soft-data records.  See ``BMEPosteriorMoments`` for format.
    covmodel : str or callable
        Covariance model name.
    covparam : array_like
        Covariance model parameters.
    covmat : ndarray or None
        Pre-computed full covariance matrix.
    order : float or ndarray
        Polynomial drift order (``np.nan`` for zero mean).
    options : BMEOptions or None
        BME options object.
    nhmax, nsmax : int or None
        Neighbourhood size limits.
    dmax : float or None
        Maximum neighbourhood search radius.
    general_knowledge : str
        ``'gaussian'`` (default).
    quantiles : array_like of shape (nq,) or None
        Quantile levels to evaluate, each in the open interval ``(0, 1)``.
        Default ``[0.1, 0.3, 0.5, 0.7, 0.9]``.
    pdfk, pdfh, pdfs, hk_k, hk_h, hk_s : optional
        Advanced parameters for non-Gaussian general knowledge.
    gui_args : tuple or None
        GUI cancellation hook.
    n_grid : int
        Number of z-grid points used when numerically approximating the
        posterior PDF for CDF inversion.  Ignored for the all-Gaussian
        fast path.  Default 400.

    Returns
    -------
    zq : ndarray of shape (nk, nq)
        Quantile values.  ``zq[k, j]`` is the ``quantiles[j]`` quantile of
        the posterior at estimation location ``k``.  NaN where computation
        failed (insufficient neighbourhood data).

    Notes
    -----
    **Numerical path** (non-Gaussian soft data):

    1. :func:`BMEPosteriorPDF` is called with ``z_grid=None`` to obtain an
       automatic per-point z-grid spanning ``[mean − 4σ, mean + 4σ]``.
    2. The PDF is normalised by trapezoidal integration.
    3. The CDF is built by ``scipy.integrate.cumulative_trapezoid``.
    4. Each quantile is extracted by linear interpolation on the CDF.

    See Also
    --------
    BMEPosteriorMoments : Posterior mean, variance and skewness.
    BMEPosteriorCI      : Symmetric equal-tail credible intervals.
    BMEPosteriorPDF     : Full posterior PDF on an automatic or explicit z-grid.
    """
    import scipy.stats

    # ------------------------------------------------------------------
    # Validate / default quantile levels
    # ------------------------------------------------------------------
    if quantiles is None:
        quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    quantiles = np.asarray(quantiles, dtype=float).ravel()
    if np.any(quantiles <= 0) or np.any(quantiles >= 1):
        raise ValueError(
            "All quantile levels must be strictly in (0, 1); "
            f"got {quantiles[~((quantiles > 0) & (quantiles < 1))]}."
        )
    nq = len(quantiles)

    ck = np.atleast_2d(np.asarray(ck, dtype=float))
    nk = ck.shape[0]
    ns = np.atleast_2d(cs).shape[0] if cs is not None else 0
    zs = _normalize_zs(zs, ns)
    zs_list = list(zs) if zs is not None else []

    # ------------------------------------------------------------------
    # All-Gaussian fast path
    # ------------------------------------------------------------------
    _all_gauss, _ = _check_all_gaussian_zs(zs_list, options)

    if _all_gauss:
        moments = BMEPosteriorMoments(
            ck, ch, cs, zh, zs,
            covmodel, covparam, covmat,
            order, options,
            nhmax, nsmax, dmax,
            general_knowledge,
            pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
            gui_args, ck_cov_output=False)

        zq = np.full((nk, nq), np.nan)
        for k in range(nk):
            m_k = float(moments[k, 0])
            v_k = float(moments[k, 1]) if not np.isnan(moments[k, 1]) else 1.0
            sig_k = np.sqrt(max(v_k, 1e-14))
            zq[k, :] = scipy.stats.norm.ppf(quantiles, loc=m_k, scale=sig_k)
        return zq

    # ------------------------------------------------------------------
    # Non-Gaussian numerical path: PDF → CDF → quantile inversion
    # ------------------------------------------------------------------
    zq = np.full((nk, nq), np.nan)

    z_out_list, pdf_out_list, _info = BMEPosteriorPDF(
        ck, ch, cs, zh, zs,
        covmodel, covparam, covmat,
        order, options,
        nhmax, nsmax, dmax,
        general_knowledge,
        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s, gui_args,
        n_grid=n_grid,
        z_grid=None,
    )

    from scipy.integrate import cumulative_trapezoid

    for k in range(nk):
        zg = z_out_list[k]
        pg = pdf_out_list[k].copy()

        area = np.trapezoid(pg, zg)
        if area <= 0 or np.isnan(area):
            continue
        pg_norm = pg / area

        cdf = cumulative_trapezoid(pg_norm, zg, initial=0.0)
        cdf = np.clip(cdf, 0.0, 1.0)

        for j, q_j in enumerate(quantiles):
            try:
                zq[k, j] = float(np.interp(float(q_j), cdf, zg))
            except Exception:
                pass

    return zq