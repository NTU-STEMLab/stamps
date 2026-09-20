"""
Spatial random field simulation for the stamps package.

Adapted from the MATLAB bmelib 2.0b1 ``simulib`` directory and the
pre-existing ``stats/simuchol.py`` / ``stats/simuce.py`` stubs in stamps.

Four algorithm families are provided:

Section A  Circular Embedding
    stationary_gaussian_process

Section B  Cholesky Methods
    simuchol, simucholcond, simucholcondME, anisosimuchol

Section C  Sequential Methods  (iterative kriging-based)
    simuseq, simuseqcond, simuseqcondME, simuseqcondInt

Section D  Soft-Data Simulation Utilities
    simuprobabilistic, simuinterval

Notes
-----
*  All array indexing is 0-based throughout.
*  Randomness is handled exclusively through ``numpy.random.default_rng``.
   Pass a ``seed`` keyword argument to any function for reproducibility.
*  ``scipy.linalg.cholesky`` is used with ``lower=True``; i.e. the
   decomposition is ``K = L @ L.T``, so simulated draws are ``Zh = L @ w``
   where ``w ~ N(0, I)``.  (MATLAB returns the upper triangular factor.)
*  ``scipy.linalg.solve`` is used wherever ``inv(K22) @ v`` would appear in
   the MATLAB source, for better numerical stability.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.linalg as sl
import scipy.interpolate as si

from ...general.coord2K import coord2K
from ...general.findpairs import findpairs as _findpairs_raw


def _has_pairs(c1: np.ndarray, c2: np.ndarray) -> bool:
    """Return True if c1 and c2 share any identical coordinate rows."""
    if c2 is None or len(c2) == 0:
        return False
    result = _findpairs_raw(c1, c2)
    arr = np.asarray(result)
    return arr.ndim >= 2 and arr.shape[0] > 0


__all__ = [
    # Section A
    "stationary_gaussian_process",
    # Section B
    "simuchol",
    "simucholcond",
    "simucholcondME",
    "anisosimuchol",
    # Section C
    "simuseq",
    "simuseqcond",
    "simuseqcondME",
    "simuseqcondInt",
    # Section D
    "simuprobabilistic",
    "simuinterval",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _psd_cholesky(K: np.ndarray, lower: bool = True) -> np.ndarray:
    """Cholesky decomposition with automatic nugget regularisation.

    Attempts a clean Cholesky first; if the matrix is not positive
    definite (due to numerical cancellation), progressively adds a small
    diagonal jitter until decomposition succeeds.

    Parameters
    ----------
    K : ndarray of shape (n, n)
        Symmetric matrix to decompose.
    lower : bool
        If True return lower triangular factor ``L`` s.t. ``K ≈ L @ L.T``.

    Returns
    -------
    L : ndarray of shape (n, n)
    """
    try:
        return sl.cholesky(K, lower=lower)
    except sl.LinAlgError:
        pass
    # Add progressively larger diagonal jitter
    diag_mean = np.mean(np.abs(np.diag(K)))
    for exp in range(-12, 0):
        jitter = diag_mean * (10.0 ** exp)
        try:
            return sl.cholesky(K + jitter * np.eye(K.shape[0]), lower=lower)
        except sl.LinAlgError:
            continue
    raise sl.LinAlgError(
        "Matrix is not positive definite even after nugget regularisation."
    )


def _make_rng(seed):
    """Return a ``numpy.random.Generator`` from *seed*.

    Parameters
    ----------
    seed : int, numpy.random.Generator, or None

    Returns
    -------
    numpy.random.Generator
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _kriging_me(
    ck: np.ndarray,
    ch: np.ndarray,
    cs: np.ndarray,
    zh: np.ndarray,
    zs: np.ndarray,
    vs: np.ndarray,
    covmodel: list,
    covparam: list,
    nhmax: int,
    nsmax: int,
    dmax: float,
    order=None,
) -> tuple[float, float]:
    """Simple kriging with Gaussian (Measurement-Error) soft data.

    This private helper mirrors the logic of ``krigingME.m`` from bmelib.
    Soft data are treated as hard measurements corrupted by independent
    Gaussian noise with variance ``vs``.

    Parameters
    ----------
    ck : ndarray of shape (1, d)
        Estimation point coordinates.
    ch : ndarray of shape (nh, d)
        Hard conditioning coordinates.
    cs : ndarray of shape (ns0, d)
        Soft conditioning coordinates.
    zh : ndarray of shape (nh,) or (nh, 1)
        Hard data values.
    zs : ndarray of shape (ns0,) or (ns0, 1)
        Soft data means.
    vs : ndarray of shape (ns0,) or (ns0, 1)
        Soft data variances.
    covmodel : list
        Covariance model specification (same convention as ``coord2K``).
    covparam : list
        Covariance parameters.
    nhmax : int
        Maximum number of hard neighbours used.
    nsmax : int
        Maximum number of soft neighbours used.
    dmax : float
        Search distance.
    order : int or None
        Polynomial trend order; ``None`` means zero-mean.

    Returns
    -------
    zk : float
        Kriging estimate at ``ck``.
    vk : float
        Kriging variance at ``ck``.
    """
    from ...general.neighbours import neighbours

    zh = np.asarray(zh).ravel()
    zs = np.asarray(zs).ravel()
    vs = np.asarray(vs).ravel()
    dmax_arr = np.array([[dmax]])   # neighbours() expects shape (1,1)

    # --- select hard neighbourhood ---
    if ch is not None and len(ch) > 0:
        ch_nbr, zh_nbr, _, _, _ = neighbours(ck, ch, zh.reshape(-1, 1),
                                              nhmax, dmax_arr)
        zh_nbr = zh_nbr.ravel()
    else:
        ch_nbr = np.empty((0, ck.shape[1]))
        zh_nbr = np.array([])

    # --- select soft neighbourhood ---
    if cs is not None and len(cs) > 0:
        cs_nbr, zs_nbr_2d, _, _, _ = neighbours(ck, cs, zs.reshape(-1, 1),
                                                  nsmax, dmax_arr)
        zs_nbr = zs_nbr_2d.ravel()

        # get vs for selected neighbours by distance match
        from scipy.spatial.distance import cdist
        if len(cs_nbr) > 0:
            dists = cdist(cs_nbr, cs)
            idx = np.argmin(dists, axis=1)
            vs_nbr = vs[idx]
        else:
            vs_nbr = np.array([])
    else:
        cs_nbr = np.empty((0, ck.shape[1]))
        zs_nbr = np.array([])
        vs_nbr = np.array([])

    nh_loc = len(zh_nbr)
    ns_loc = len(zs_nbr)

    if nh_loc + ns_loc == 0:
        return np.nan, np.nan

    # stack all conditioning data
    if nh_loc > 0 and ns_loc > 0:
        c_all = np.vstack([ch_nbr, cs_nbr])
        z_all = np.concatenate([zh_nbr, zs_nbr])
    elif nh_loc > 0:
        c_all = ch_nbr
        z_all = zh_nbr
    else:
        c_all = cs_nbr
        z_all = zs_nbr

    # build covariance matrices
    K11, _ = coord2K(ck, ck, covmodel, covparam)
    K22, _ = coord2K(c_all, c_all, covmodel, covparam)
    K12, _ = coord2K(ck, c_all, covmodel, covparam)

    # augment K22 diagonal for soft data measurement error
    noise = np.zeros(nh_loc + ns_loc)
    noise[nh_loc:] = vs_nbr
    K22 = K22 + np.diag(noise)

    # solve K22 * w = K12.T  (more stable than explicit inverse)
    try:
        w = sl.solve(K22, K12.T, assume_a='pos')
    except sl.LinAlgError:
        w = np.linalg.lstsq(K22, K12.T, rcond=None)[0]

    # Kriging estimate:  zk = w^T z_all
    #   w  has shape (n_all, 1)  — the solution of  K22 · w = K12.T
    #   K12 @ w  has shape (1, 1)  — that is the variance-reduction term,
    #            NOT the estimate.  Using it here was the original bug.
    zk = float(w.ravel() @ z_all)           # (n_all,) · (n_all,)  → scalar ✓
    vk = float(K11[0, 0] - (K12 @ w)[0, 0])  # C(0) − k^T K⁻¹ k  → variance ✓
    vk = max(vk, 0.0)
    return zk, vk


# ===========================================================================
# SECTION A  –  Circular Embedding
# ===========================================================================

def stationary_gaussian_process(
    m: int,
    n: int,
    rho_func: Callable[[np.ndarray], float],
    *,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a stationary Gaussian random field by circular embedding.

    Generates two statistically independent realisations of a stationary
    2-D Gaussian random field on an ``m × n`` grid using the block-circulant
    FFT embedding method.

    Adapted from the MATLAB stub in ``stamps/stats/simuce.py``.

    Reference
    ---------
    Kroese, D. P., & Botev, Z. I. (2015). Spatial Process Simulation.
    In *Stochastic Geometry, Spatial Statistics and Random Fields*
    (pp. 369–404). Springer International Publishing.
    DOI: 10.1007/978-3-319-10064-7_12

    Parameters
    ----------
    m : int
        Number of rows in the output grid (y-axis size).
    n : int
        Number of columns in the output grid (x-axis size).
    rho_func : callable
        Scalar covariance function ``rho(h)`` where ``h`` is a 1-D
        ``ndarray`` of length 2, representing the spatial lag vector
        ``[dx, dy]``.  The function must satisfy
        ``Cov(X_t, X_s) = rho(t - s)``.

        Example::

            rho = lambda h: (
                (1 - h[0]**2/50**2 - h[0]*h[1]/(15*50) - h[1]**2/15**2)
                * np.exp(-(h[0]**2/50**2 + h[1]**2/15**2))
            )

    seed : int, numpy.random.Generator, or None, optional
        Seed for reproducibility.  Passed to
        ``numpy.random.default_rng(seed)``.

    Returns
    -------
    field1 : ndarray of shape (m, n)
        First independent Gaussian random field realisation.
    field2 : ndarray of shape (m, n)
        Second independent Gaussian random field realisation with the
        same covariance structure.
    tx : ndarray of shape (n,)
        Column (x) coordinate vector ``[0, 1, ..., n-1]``.
    ty : ndarray of shape (m,)
        Row (y) coordinate vector ``[0, 1, ..., m-1]``.

    Raises
    ------
    ValueError
        If the circulant embedding does not yield a positive semi-definite
        matrix (i.e. the covariance function is not embeddable).

    Notes
    -----
    The algorithm embeds the ``m × n`` covariance matrix into a
    ``(2m-1) × (2n-1)`` block-circulant matrix whose eigenvalues are
    computed via FFT.  Two independent fields are extracted as the real
    and imaginary parts of a single complex FFT transform.
    """
    rng = _make_rng(seed)
    tx = np.arange(n, dtype=float)
    ty = np.arange(m, dtype=float)

    # Sample covariance function at all grid lags
    Rows = np.zeros((m, n))
    Cols = np.zeros((m, n))
    for i in range(n):
        for j in range(m):
            Rows[j, i] = rho_func(np.array([tx[i] - tx[0], ty[j] - ty[0]]))
            Cols[j, i] = rho_func(np.array([tx[0] - tx[i], ty[j] - ty[0]]))

    # Build block-circulant embedding matrix
    BlkCirc_row = np.block([
        [Rows,            Cols[:, -1:0:-1]],
        [Cols[-1:0:-1, :], Rows[-1:0:-1, -1:0:-1]],
    ])

    # Eigenvalues via FFT
    lam = np.real(np.fft.fft2(BlkCirc_row)) / (2 * m - 1) / (2 * n - 1)

    neg_min = np.min(lam[lam < 0]) if np.any(lam < 0) else 0.0
    if abs(neg_min) > 1e-15:
        raise ValueError(
            f"Could not find a positive definite circulant embedding "
            f"(min negative eigenvalue = {neg_min:.3e}). "
            "Try a covariance function with a valid embedding."
        )
    lam = np.sqrt(np.maximum(lam, 0.0))

    # Generate complex Gaussian noise and apply spectral factor
    noise_r = rng.standard_normal((2 * m - 1, 2 * n - 1))
    noise_i = rng.standard_normal((2 * m - 1, 2 * n - 1))
    F = np.fft.fft2(lam * (noise_r + 1j * noise_i))

    # Extract upper-left sub-block
    F = F[:m, :n]
    field1 = np.real(F)
    field2 = np.imag(F)
    return field1, field2, tx, ty


# ===========================================================================
# SECTION B  –  Cholesky Methods
# ===========================================================================

def simuchol(
    ch: np.ndarray,
    covmodel: list,
    covparam: list,
    ns: int = 1,
    *,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Unconditional simulation by the Cholesky method.

    Implements the traditional non-conditional simulation method based on
    a Cholesky decomposition of the covariance matrix.  This simulation
    method is especially recommended for simulating independently several
    sets of a limited number of hard values (fewer than a few hundred).
    Simulated values are zero-mean Gaussian distributed.

    Adapted from ``simuchol.m`` (bmelib simulib, Jan 1, 2001) and the
    pre-existing ``stamps/stats/simuchol.py``.

    Parameters
    ----------
    ch : ndarray of shape (nh, d)
        Coordinates for the locations where values are to be simulated.
        Each row is a coordinate vector; the number of columns equals the
        spatial dimension.  There is no restriction on dimensionality.
    covmodel : list of str
        Covariance model specification (same convention as ``coord2K``).
        Variogram models are **not** supported.
    covparam : list
        Parameter values for ``covmodel`` following the ``coord2K``
        convention.
    ns : int, optional
        Number of independent simulation realisations.  Default is 1.
    seed : int, numpy.random.Generator, or None, optional
        Random seed for reproducibility.

    Returns
    -------
    Zh : ndarray of shape (nh, ns)
        Zero-mean simulated Gaussian values at the coordinates ``ch``.
        Each column is an independent realisation.  When ``ns=1`` the
        result is a single-column matrix.
    L : ndarray of shape (nh, nh)
        Lower-triangular Cholesky factor of the global covariance matrix
        ``K`` such that ``K = L @ L.T``.

    Notes
    -----
    All conventions for specifying nested models, multivariate or
    space-time cases are the same as for ``kriging``.

    The MATLAB source uses the upper-triangular factor ``L`` such that
    ``K = L.T @ L``; this Python implementation uses the lower-triangular
    convention ``K = L @ L.T`` (``scipy.linalg.cholesky`` with
    ``lower=True``).
    """
    rng = _make_rng(seed)
    K, _ = coord2K(ch, ch, covmodel, covparam)
    L = _psd_cholesky(K)
    n = K.shape[0]
    W = rng.standard_normal((n, ns))
    Zh = L @ W
    return Zh, L


def simucholcond(
    ch: np.ndarray,
    ch0: np.ndarray,
    zh0: np.ndarray,
    covmodel: list,
    covparam: list,
    ns: int = 1,
    *,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional simulation by the Cholesky method (hard data).

    Generates conditional simulations using the Cholesky decomposition
    method.  The conditioning values are zero-mean Gaussian distributed
    hard data.  Simulated values are Gaussian distributed.

    Adapted from ``simucholcond.m`` (bmelib simulib, Jan 1, 2001).

    Parameters
    ----------
    ch : ndarray of shape (nh, d)
        Coordinates for the simulation locations.  Each row is a
        coordinate vector; the number of columns equals the spatial
        dimension.
    ch0 : ndarray of shape (nh0, d)
        Coordinates for the hard conditioning values (same convention
        as ``ch``).
    zh0 : ndarray of shape (nh0,) or (nh0, 1)
        Hard conditioning values at the locations specified in ``ch0``.
    covmodel : list of str
        Covariance model specification.  Variogram models are **not**
        available for this function.
    covparam : list
        Parameter values for ``covmodel``.
    ns : int, optional
        Number of independent simulation realisations.  Default is 1.
    seed : int, numpy.random.Generator, or None, optional
        Random seed for reproducibility.

    Returns
    -------
    Zh : ndarray of shape (nh, ns)
        Simulated Gaussian values at the coordinates ``ch``.  Each
        column is an independent realisation.
    L : ndarray of shape (nh, nh)
        Lower-triangular Cholesky factor of the conditional covariance
        matrix ``K_cond = K11 − K12 K22⁻¹ K12.T``, such that
        ``K_cond = L @ L.T``.

    Raises
    ------
    ValueError
        If ``ch`` and ``ch0`` share identical coordinates.

    Notes
    -----
    1. When using ``simucholcond``, the hard data values should be
       zero-mean Gaussian distributed.  If this is not the case, subtract
       the mean from ``zh0`` prior to simulation and apply appropriate
       transformations afterwards (see ``bme_transform``).

    2. All conventions for nested models, multivariate or space-time
       cases are the same as for ``kriging``.

    The conditional distribution uses the Schur complement formula::

        K_cond  = K11 - K12 @ inv(K22) @ K12.T
        mu_cond = K12 @ inv(K22) @ zh0

    where ``K11`` is ``cov(ch, ch)``, ``K22 = cov(ch0, ch0)`` and
    ``K12 = cov(ch, ch0)``.
    """
    zh0 = np.asarray(zh0).reshape(-1, 1)

    # Guard: ch and ch0 must not share coordinates
    if _has_pairs(ch, ch0):
        raise ValueError("ch and ch0 cannot contain identical coordinates.")

    rng = _make_rng(seed)

    K11, _ = coord2K(ch, ch, covmodel, covparam)
    K22, _ = coord2K(ch0, ch0, covmodel, covparam)
    K12, _ = coord2K(ch, ch0, covmodel, covparam)

    # K22 \ K12.T  →  shape (nh0, nh) : used for K_cond
    K22_inv_K12T = sl.solve(K22, K12.T, assume_a='pos')
    # K22 \ zh0    →  shape (nh0, 1) : used for mu_cond
    K22_inv_zh0 = sl.solve(K22, zh0, assume_a='pos')

    K_cond = K11 - K12 @ K22_inv_K12T      # (nh, nh)
    mu_cond = K12 @ K22_inv_zh0             # (nh, 1)

    L = _psd_cholesky(K_cond)
    n = L.shape[0]
    W = rng.standard_normal((n, ns))
    Zh = L @ W + mu_cond          # broadcast over ns columns
    return Zh, L


def simucholcondME(
    ch: np.ndarray,
    ch0: np.ndarray,
    cs0: np.ndarray,
    zh0: np.ndarray,
    zs0: np.ndarray,
    vs0: np.ndarray,
    covmodel: list,
    covparam: list,
    ns: int = 1,
    *,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional simulation by Cholesky with Gaussian soft data (ME).

    Simulates values conditionally to both hard and soft data.  The
    probabilistic soft data are assumed to be completely characterised by
    their mean and their variance (Measurement Error concept, see
    ``krigingME``).

    Adapted from ``simucholcondME.m`` (bmelib simulib, Jan 1, 2001).

    Parameters
    ----------
    ch : ndarray of shape (nh, d)
        Coordinates for the simulation locations.
    ch0 : ndarray of shape (nh0, d)
        Coordinates for the hard conditioning values.
    cs0 : ndarray of shape (ns0, d)
        Coordinates for the soft (Gaussian ME) conditioning values.
    zh0 : ndarray of shape (nh0,) or (nh0, 1)
        Hard conditioning values at the locations ``ch0``.
    zs0 : ndarray of shape (ns0,) or (ns0, 1)
        Means of the Gaussian soft data at the locations ``cs0``.
    vs0 : ndarray of shape (ns0,) or (ns0, 1)
        Variances of the Gaussian soft data at the locations ``cs0``.
    covmodel : list of str
        Covariance model specification.  Variogram models are **not**
        available for this function.
    covparam : list
        Parameter values for ``covmodel``.
    ns : int, optional
        Number of independent simulation realisations.  Default is 1.
    seed : int, numpy.random.Generator, or None, optional
        Random seed for reproducibility.

    Returns
    -------
    Zh : ndarray of shape (nh, ns)
        Simulated Gaussian values at the coordinates ``ch``.
    L : ndarray of shape (nh, nh)
        Lower-triangular Cholesky factor of the conditional covariance
        matrix ``K_cond = L @ L.T``.

    Raises
    ------
    ValueError
        If ``ch`` and ``ch0`` share identical coordinates.

    Notes
    -----
    1. Both hard and soft data values should be zero-mean Gaussian
       distributed.  Subtract the mean from ``zh0`` and ``zs0`` prior to
       simulation and apply appropriate transformations afterwards (see
       ``bme_transform``).

    2. ``ch0`` and ``zh0`` may be passed as empty arrays (shape
       ``(0, d)`` and ``(0,)`` respectively) when no hard data are
       available.

    3. All conventions for nested models, multivariate or space-time
       cases are the same as for ``kriging``.

    The soft data are treated as noisy measurements.  The joint
    conditioning covariance matrix is augmented::

        K22 = cov([ch0; cs0], [ch0; cs0]) + diag([0, ..., 0, vs0])

    where zeros occupy the ``nh0 × nh0`` hard-data block.
    """
    zh0 = np.asarray(zh0).reshape(-1, 1)
    zs0 = np.asarray(zs0).reshape(-1, 1)
    vs0 = np.asarray(vs0).ravel()

    # Guard: ch and ch0 must not share coordinates
    if _has_pairs(ch, ch0):
        raise ValueError("ch and ch0 cannot contain identical coordinates.")

    rng = _make_rng(seed)

    nh0 = len(zh0)
    c_all = np.vstack([ch0, cs0]) if nh0 > 0 else cs0
    z_all = np.vstack([zh0, zs0]) if nh0 > 0 else zs0  # (nh0+ns0, 1)

    K11, _ = coord2K(ch, ch, covmodel, covparam)
    K22, _ = coord2K(c_all, c_all, covmodel, covparam)
    K12, _ = coord2K(ch, c_all, covmodel, covparam)

    # Augment K22 diagonal for soft data measurement error
    noise = np.zeros(K22.shape[0])
    noise[nh0:] = vs0
    K22 = K22 + np.diag(noise)

    K22_inv_K12T = sl.solve(K22, K12.T, assume_a='pos')
    K22_inv_z    = sl.solve(K22, z_all,  assume_a='pos')

    K_cond  = K11 - K12 @ K22_inv_K12T   # (nh, nh)
    mu_cond = K12 @ K22_inv_z             # (nh, 1)

    L = _psd_cholesky(K_cond)
    n = L.shape[0]
    W = rng.standard_normal((n, ns))
    Zh = L @ W + mu_cond
    return Zh, L


def anisosimuchol(
    ch: np.ndarray,
    covmodel: list,
    covparam: list,
    theta: float,
    ratio: float,
    ns: int = 1,
    *,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Unconditional anisotropic simulation via Cholesky decomposition.

    A convenience wrapper around :func:`simuchol` that applies a
    geometric anisotropy transformation to the coordinates before
    simulation.  This allows simulation of anisotropic random fields
    using any isotropic covariance model by mapping coordinates to an
    isotropic space.

    Adapted from ``anisosimuchol`` in the pre-existing
    ``stamps/stats/simuchol.py``.

    Parameters
    ----------
    ch : ndarray of shape (nh, d)
        Original (anisotropic) coordinates for the simulation locations.
    covmodel : list of str
        Isotropic covariance model specification (same convention as
        ``coord2K``).
    covparam : list
        Parameter values for ``covmodel``.
    theta : float
        Rotation angle of the principal anisotropy axis (degrees from
        north, clockwise).
    ratio : float
        Anisotropy ratio (minor axis / major axis), in ``(0, 1]``.
    ns : int, optional
        Number of independent simulation realisations.  Default is 1.
    seed : int, numpy.random.Generator, or None, optional
        Random seed for reproducibility.

    Returns
    -------
    Zh : ndarray of shape (nh, ns)
        Zero-mean simulated Gaussian values in the original coordinate
        space.  Each column is an independent realisation.
    L : ndarray of shape (nh, nh)
        Lower-triangular Cholesky factor of the (isotropic-space)
        covariance matrix.  Identical to the ``L`` returned by
        :func:`simuchol` on the transformed coordinates.
    """
    from ..dependence.anisotropy import aniso2iso  # lazy: avoids optional rpy2 at import
    ch_iso = aniso2iso(ch, theta, ratio)
    Zh, L = simuchol(ch_iso, covmodel, covparam, ns, seed=seed)
    return Zh, L


# ===========================================================================
# SECTION C  –  Sequential Methods
# ===========================================================================

def _simuseq_single(
    ch: np.ndarray,
    covmodel: list,
    covparam: list,
    nhmax: int,
    dmax: float,
    options: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run one sequential simulation realisation (internal helper)."""
    from ...stest.kriging import kriging

    nh = ch.shape[0]
    ch_sim = ch

    # Optionally randomise simulation path
    if options["random_path"]:
        path = rng.permutation(nh)
        ch_sim = ch_sim[path, :]
    else:
        path = np.arange(nh)

    # --- Simulate the first location unconditionally ---
    vk_first, _ = coord2K(ch_sim[[0], :], ch_sim[[0], :], covmodel, covparam)
    zhtemp = np.full(nh, np.nan)
    zhtemp[0] = rng.standard_normal() * np.sqrt(float(vk_first[0, 0]))

    chtemp = ch_sim[[0], :]

    for i in range(1, nh):
        ck = ch_sim[[i], :]
        zk = np.nan
        dmax_iter = 0.0

        # Expand search radius until at least one neighbour is found
        while np.isnan(zk):
            dmax_iter += dmax
            zk, vk = kriging(
                ck, chtemp, zhtemp[:i].reshape(-1, 1),
                covmodel, covparam,
                nhmax, np.array([[dmax_iter]]), order=None
            )

        # Extract scalar from potentially array-valued kriging variance before
        # calling Python's built-in max() – passing an ndarray is deprecated
        # since NumPy 1.25 and will raise in a future version.
        vk_scalar = float(np.asarray(vk).flat[0])
        zk_scalar = float(np.asarray(zk).flat[0])
        zhtemp[i] = rng.standard_normal() * np.sqrt(max(vk_scalar, 0.0)) + zk_scalar
        chtemp = np.vstack([chtemp, ck])

        if options["verbose"]:
            print(f"{i + 1}/{nh}")

    # Restore original coordinate order
    zh = np.empty(nh)
    zh[path] = zhtemp
    return zh


def simuseq(
    ch: np.ndarray,
    covmodel: list,
    covparam: list,
    nhmax: int,
    dmax: float,
    options: dict | None = None,
    ns: int = 1,
    *,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Unconditional simulation by the sequential method.

    Implements the sequential non-conditional simulation method based on
    the iterated use of conditional distributions, so that values are
    simulated one at a time.  This method is especially recommended for
    simulating a few sets of a large number of values (above a few
    hundred), as it does not require storing a global covariance matrix
    in memory.

    Adapted from ``simuseq.m`` (bmelib simulib, Jan 1, 2001).

    Parameters
    ----------
    ch : ndarray of shape (nh, d)
        Coordinates for the simulation locations.  Each row is a
        coordinate vector; the number of columns equals the spatial
        dimension.
    covmodel : list of str
        Covariance model specification.  Variogram models are **not**
        available for this function.
    covparam : list
        Parameter values for ``covmodel``.
    nhmax : int
        Maximum number of previously simulated values that are considered
        when simulating at each new location in the sequence.
    dmax : float
        Maximum distance between a simulation location and previously
        visited simulation locations.  All locations closer than
        ``dmax`` are included in the local conditioning set.  When no
        previously simulated value lies within ``dmax``, the search
        radius is doubled repeatedly until at least one neighbour is
        found.
    options : dict or None, optional
        Optional control parameters.  Recognised keys:

        ``"verbose"`` : bool, default ``False``
            Print progress (current index / total).
        ``"random_path"`` : bool, default ``True``
            Randomly permute the simulation path before starting.
            Recommended to reduce path-order artefacts.
    ns : int, optional
        Number of independent simulation realisations.  Default is 1.
        When ``ns > 1`` the output has shape ``(nh, ns)`` instead of
        ``(nh,)`` (which is returned for ``ns == 1`` for backward
        compatibility).
    seed : int, numpy.random.Generator, or None, optional
        Random seed for reproducibility.

    Returns
    -------
    zh : ndarray of shape (nh,) when ns == 1, or (nh, ns) when ns > 1
        Zero-mean simulated Gaussian values at the coordinates ``ch``.

    Notes
    -----
    All conventions for nested models, multivariate or space-time cases
    are the same as for ``kriging``.

    When ``options["random_path"]`` is ``True`` (default), the output
    ``zh`` is reordered so that its indices correspond to the original
    row order of ``ch``.
    """
    opts = {"verbose": False, "random_path": True}
    if options is not None:
        opts.update(options)

    rng = _make_rng(seed)
    nh = ch.shape[0]

    if ns == 1:
        return _simuseq_single(ch, covmodel, covparam, nhmax, dmax, opts, rng)

    # Multiple realisations — stack horizontally
    Zh = np.empty((nh, ns))
    for j in range(ns):
        Zh[:, j] = _simuseq_single(ch, covmodel, covparam, nhmax, dmax, opts, rng)
    return Zh


def simuseqcond(
    ch: np.ndarray,
    ch0: np.ndarray,
    zh0: np.ndarray,
    model: list,
    param: list,
    nhmax: int,
    dmax: float,
    order=None,
    options: dict | None = None,
    *,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Conditional simulation by the sequential method (hard data).

    Generates conditional simulations using the sequential simulation
    method, where conditioning values are hard data.  Unlike
    :func:`simuseq`, this function supports both variogram and covariance
    models.  A polynomial mean of specified order may be used, so hard
    data do not need to be zero-mean distributed.  Simulated values are
    Gaussian distributed.

    Adapted from ``simuseqcond.m`` (bmelib simulib, Jan 1, 2001).

    Parameters
    ----------
    ch : ndarray of shape (nh, d)
        Coordinates for the simulation locations.
    ch0 : ndarray of shape (nh0, d)
        Coordinates for the hard conditioning values.
    zh0 : ndarray of shape (nh0,) or (nh0, 1)
        Hard conditioning values at the locations ``ch0``.
    model : list of str
        Variogram or covariance model specification (see ``kriging``).
    param : list
        Parameter values for ``model``.
    nhmax : int
        Maximum number of conditioning or previously simulated values
        considered per simulation step.
    dmax : float
        Maximum search distance.  Expanded repeatedly if no neighbours
        are found within the current radius.
    order : int or None, optional
        Order of the polynomial trend.  Use ``None`` for a zero-mean
        field (equivalent to MATLAB's ``NaN``).  Note that ``order=None``
        can only be used with covariance models, not variogram models.
    options : dict or None, optional
        Optional control parameters with keys:

        ``"verbose"`` : bool, default ``False``
            Print progress.
        ``"random_path"`` : bool, default ``True``
            Randomly permute the simulation path.

    seed : int, numpy.random.Generator, or None, optional
        Random seed for reproducibility.

    Returns
    -------
    zh : ndarray of shape (nh,)
        Simulated Gaussian values at ``ch``.  Depending on ``order``,
        values are not necessarily zero-mean.

    Raises
    ------
    ValueError
        If ``ch`` and ``ch0`` share identical coordinates.

    Notes
    -----
    All conventions for nested models, multivariate or space-time cases
    are the same as for ``kriging``.
    """
    from ...stest.kriging import kriging

    zh0 = np.asarray(zh0).ravel()

    if _has_pairs(ch, ch0):
        raise ValueError("ch and ch0 cannot contain identical coordinates.")

    opts = {"verbose": False, "random_path": True}
    if options is not None:
        opts.update(options)

    rng = _make_rng(seed)
    nh = ch.shape[0]
    nh0 = ch0.shape[0]

    if opts["random_path"]:
        path = rng.permutation(nh)
        ch = ch[path, :]
    else:
        path = np.arange(nh)

    # Conditioning set starts as all hard data
    chtemp = ch0.copy()
    zhtemp = zh0.copy()

    for i in range(nh):
        ck = ch[[i], :]
        zk = np.nan
        dmax_iter = 0.0

        while np.isnan(zk):
            dmax_iter += dmax
            zk, vk = kriging(
                ck, chtemp, zhtemp.reshape(-1, 1),
                model, param,
                nhmax, np.array([[dmax_iter]]), order=order
            )

        new_z = rng.standard_normal() * np.sqrt(max(vk, 0.0)) + zk
        chtemp = np.vstack([chtemp, ck])
        zhtemp = np.append(zhtemp, new_z)

        if opts["verbose"]:
            print(f"{i + 1}/{nh}")

    # Simulated values are those appended after the initial nh0 entries
    zh_path = zhtemp[nh0:]

    # Restore original order
    zh = np.empty(nh)
    zh[path] = zh_path
    return zh


def simuseqcondME(
    ch: np.ndarray,
    ch0: np.ndarray,
    cs0: np.ndarray,
    zh0: np.ndarray,
    zs0: np.ndarray,
    vs0: np.ndarray,
    covmodel: list,
    covparam: list,
    nhmax: int,
    nsmax: int,
    dmax: float,
    order=None,
    options: dict | None = None,
    *,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Conditional simulation by sequential method with Gaussian soft data.

    Simulates hard values conditionally to both hard and soft data, where
    the probabilistic soft data are completely characterised by their mean
    and variance (Measurement Error concept).

    Adapted from ``simuseqcondME.m`` (bmelib simulib, Jan 1, 2001).

    Parameters
    ----------
    ch : ndarray of shape (nh, d)
        Coordinates for the simulation locations.
    ch0 : ndarray of shape (nh0, d)
        Coordinates for the hard conditioning values.
    cs0 : ndarray of shape (ns0, d)
        Coordinates for the soft (Gaussian ME) conditioning values.
    zh0 : ndarray of shape (nh0,) or (nh0, 1)
        Hard conditioning values.
    zs0 : ndarray of shape (ns0,) or (ns0, 1)
        Means of the Gaussian soft data at ``cs0``.
    vs0 : ndarray of shape (ns0,) or (ns0, 1)
        Variances of the Gaussian soft data at ``cs0``.
    covmodel : list of str
        Covariance model specification.  Variogram models are **not**
        available for this function.
    covparam : list
        Parameter values for ``covmodel``.
    nhmax : int
        Maximum number of hard conditioning or previously simulated
        hard-data neighbours per step.
    nsmax : int
        Maximum number of soft conditioning neighbours per step.
    dmax : float
        Maximum search distance.  Expanded if no neighbours are found.
    order : int or None, optional
        Polynomial trend order.  ``None`` means zero-mean.
    options : dict or None, optional
        ``"verbose"`` (bool, default ``False``) and
        ``"random_path"`` (bool, default ``True``).
    seed : int, numpy.random.Generator, or None, optional
        Random seed.

    Returns
    -------
    zh : ndarray of shape (nh,)
        Simulated Gaussian values at ``ch``.

    Raises
    ------
    ValueError
        If ``ch`` and ``ch0`` share identical coordinates.

    Notes
    -----
    1. Hard and soft data values should be zero-mean Gaussian distributed.
       Subtract the mean first and apply appropriate transformations
       afterwards if needed (see ``bme_transform``).

    2. ``ch0`` and ``zh0`` may be empty arrays when no hard data are
       available.

    3. All conventions for nested models, multivariate or space-time
       cases are the same as for ``kriging``.
    """
    zh0 = np.asarray(zh0).ravel()
    zs0 = np.asarray(zs0).ravel()
    vs0 = np.asarray(vs0).ravel()

    if _has_pairs(ch, ch0):
        raise ValueError("ch and ch0 cannot contain identical coordinates.")

    opts = {"verbose": False, "random_path": True}
    if options is not None:
        opts.update(options)

    rng = _make_rng(seed)
    nh = ch.shape[0]
    nh0 = len(zh0)

    if opts["random_path"]:
        path = rng.permutation(nh)
        ch = ch[path, :]
    else:
        path = np.arange(nh)

    chtemp = ch0.copy() if nh0 > 0 else np.empty((0, ch.shape[1]))
    zhtemp = zh0.copy()

    for i in range(nh):
        ck = ch[[i], :]
        zk = np.nan
        dmax_iter = 0.0

        while np.isnan(zk):
            dmax_iter += dmax
            zk, vk = _kriging_me(
                ck, chtemp, cs0, zhtemp, zs0, vs0,
                covmodel, covparam, nhmax, nsmax, dmax_iter, order
            )

        new_z = rng.standard_normal() * np.sqrt(max(vk, 0.0)) + zk
        chtemp = np.vstack([chtemp, ck]) if len(chtemp) > 0 else ck
        zhtemp = np.append(zhtemp, new_z)

        if opts["verbose"]:
            print(f"{i + 1}/{nh}")

    zh_path = zhtemp[nh0:]

    zh = np.empty(nh)
    zh[path] = zh_path
    return zh


def simuseqcondInt(
    ch: np.ndarray,
    ch0: np.ndarray,
    cs0: np.ndarray,
    zh0: np.ndarray,
    a0: np.ndarray,
    b0: np.ndarray,
    covmodel: list,
    covparam: list,
    nhmax: int,
    nsmax: int,
    dmax: float,
    order=None,
    options: dict | None = None,
    *,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Conditional simulation by sequential method with interval soft data.

    Simulates hard values conditionally to both hard and interval soft
    data.  At each simulation step the full posterior PDF is computed
    from the interval constraints using :func:`BMEPosteriorPDF`, and
    a value is drawn by inverse-CDF sampling.

    Adapted from ``simuseqcondInt.m`` (bmelib simulib, Jan 1, 2001).

    Parameters
    ----------
    ch : ndarray of shape (nh, d)
        Coordinates for the simulation locations.
    ch0 : ndarray of shape (nh0, d)
        Coordinates for the hard conditioning values.
    cs0 : ndarray of shape (ns0, d)
        Coordinates for the soft interval conditioning values.
    zh0 : ndarray of shape (nh0,) or (nh0, 1)
        Hard conditioning values.
    a0 : ndarray of shape (ns0,)
        Lower bounds of the conditioning intervals at ``cs0``.
    b0 : ndarray of shape (ns0,)
        Upper bounds of the conditioning intervals at ``cs0``.
    covmodel : list of str
        Covariance model specification.
    covparam : list
        Parameter values for ``covmodel``.
    nhmax : int
        Maximum number of hard conditioning / previously simulated
        neighbours per step.
    nsmax : int
        Maximum number of soft interval conditioning neighbours per step.
    dmax : float
        Maximum search distance.  Expanded if no neighbours are found.
    order : int or None, optional
        Polynomial trend order.  ``None`` means zero-mean.
    options : dict or None, optional
        ``"verbose"`` (bool, default ``False``) and
        ``"random_path"`` (bool, default ``True``).

        Additional keys forwarded to ``BMEPosteriorPDF``:

        ``"n_grid"`` : int, default ``80``
            Number of PDF evaluation points on the local z-grid.
        ``"grid_half_width"`` : float, default ``4.0``
            Half-width of the z-grid in units of the local kriging
            standard deviation.

    seed : int, numpy.random.Generator, or None, optional
        Random seed.

    Returns
    -------
    zh : ndarray of shape (nh,)
        Simulated values at ``ch``.  Depending on ``order``, values are
        not necessarily zero-mean.

    Raises
    ------
    ValueError
        If ``ch`` and ``ch0`` share identical coordinates.

    Notes
    -----
    1. When no hard data are available, ``ch0`` and ``zh0`` may be passed
       as empty arrays.

    2. The interval soft data are first approximated by a uniform
       distribution ``(mean = (a+b)/2, var = (b-a)²/12)`` via the ME
       kriging step to build the local z-grid centre.  The full posterior
       is then computed non-parametrically via ``BMEPosteriorPDF``.

    3. All conventions for nested models, multivariate or space-time
       cases are the same as for ``kriging``.
    """
    from ...bme.BMEprobaEstimations import BMEPosteriorPDF

    zh0 = np.asarray(zh0).ravel()
    a0 = np.asarray(a0).ravel()
    b0 = np.asarray(b0).ravel()

    if _has_pairs(ch, ch0):
        raise ValueError("ch and ch0 cannot contain identical coordinates.")

    opts = {
        "verbose": False,
        "random_path": True,
        "n_grid": 80,
        "grid_half_width": 4.0,
    }
    if options is not None:
        opts.update(options)

    rng = _make_rng(seed)
    nh = ch.shape[0]
    nh0 = len(zh0)

    # Uniform approximation of interval data for the ME kriging step
    zs0_me = (b0 + a0) / 2.0
    vs0_me = (b0 - a0) ** 2 / 12.0

    if opts["random_path"]:
        path = rng.permutation(nh)
        ch = ch[path, :]
    else:
        path = np.arange(nh)

    chtemp = ch0.copy() if nh0 > 0 else np.empty((0, ch.shape[1]))
    zhtemp = zh0.copy()

    for i in range(nh):
        ck = ch[[i], :]
        zk = np.nan
        dmax_iter = 0.0

        # Expand search radius until we get a valid kriging estimate
        while np.isnan(zk):
            dmax_iter += dmax
            zk, vk = _kriging_me(
                ck, chtemp, cs0, zhtemp, zs0_me, vs0_me,
                covmodel, covparam, nhmax, nsmax, dmax_iter, order
            )

        # Build a local z-grid centred on the kriging estimate
        hw = opts["grid_half_width"]
        ng = opts["n_grid"]
        std_k = np.sqrt(max(vk, 1e-12))
        z_file = np.linspace(zk - hw * std_k, zk + hw * std_k, ng)

        # Build uniform soft-data representations for BMEPosteriorPDF
        # Each element of zs is (z_vals, pdf_vals) for one soft datum.
        zs_local: list = []
        from ...general.neighbours import neighbours
        if len(cs0) > 0:
            cs0_nbr, a_nbr_2d, _, _, idx_arr = neighbours(
                ck, cs0, a0.reshape(-1, 1), nsmax, np.array([[dmax_iter]])
            )
            if len(cs0_nbr) > 0:
                from scipy.spatial.distance import cdist as _cdist
                dist_all = _cdist(cs0_nbr, cs0)
                nbr_idx = np.argmin(dist_all, axis=1)
                a_nbr = a0[nbr_idx]
                b_nbr = b0[nbr_idx]
                for a_j, b_j in zip(a_nbr, b_nbr):
                    width = b_j - a_j
                    if width < 1e-14:
                        width = 1e-14
                    pdf_val = 1.0 / width
                    zs_local.append((np.array([a_j, b_j]),
                                     np.array([pdf_val, pdf_val])))

        # Evaluate posterior PDF on z_file
        try:
            z_grid_out, pdf_out, _ = BMEPosteriorPDF(
                ck,
                ch=chtemp if len(chtemp) > 0 else None,
                zh=zhtemp.reshape(-1, 1) if len(zhtemp) > 0 else None,
                zs=zs_local if zs_local else None,
                cs=cs0_nbr if (len(cs0) > 0 and len(cs0_nbr) > 0) else None,
                covmodel=covmodel,
                covparam=covparam,
                nhmax=nhmax,
                nsmax=nsmax,
                dmax=dmax_iter,
                order=order if order is not None else np.nan,
                z_grid=z_file.reshape(-1, 1),
            )
            z_grid_out = np.asarray(z_grid_out).ravel()
            pdf_out = np.asarray(pdf_out).ravel()
        except Exception:
            # Fallback: draw from kriging Gaussian if PDF fails
            z_grid_out = z_file
            from scipy.stats import norm as _norm
            pdf_out = _norm.pdf(z_file, loc=zk, scale=std_k)

        # Inverse-CDF sampling (handle non-strictly-monotone CDF)
        dz = np.diff(z_grid_out)
        pdf_mid = 0.5 * (pdf_out[:-1] + pdf_out[1:])
        cdf = np.concatenate([[0.0], np.cumsum(pdf_mid * dz)])
        cdf_range = cdf[-1] - cdf[0]
        if cdf_range < 1e-14:
            new_z = zk
        else:
            u = rng.uniform(cdf[0], cdf[-1])
            new_z = float(np.interp(u, cdf, z_grid_out))

        chtemp = np.vstack([chtemp, ck]) if len(chtemp) > 0 else ck
        zhtemp = np.append(zhtemp, new_z)

        if opts["verbose"]:
            print(f"{i + 1}/{nh}")

    zh_path = zhtemp[nh0:]
    zh = np.empty(nh)
    zh[path] = zh_path
    return zh


# ===========================================================================
# SECTION D  –  Soft-Data Simulation Utilities
# ===========================================================================

def simuprobabilistic(
    Zs: np.ndarray,
    softpdftype: int,
    NV: np.ndarray,
    WidthV: np.ndarray,
    probdensV: np.ndarray,
    *,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate probabilistic soft data from hard data values.

    Given a set of hard data ``Zs`` and a template describing the
    *shape* of the soft PDF (relative widths and probability densities),
    produces the actual ns-dimensional soft PDF at the soft data points
    by centering the template on each simulated hard value.

    Adapted from ``simuprobabilistic.m`` (bmelib simulib, Jan 1, 2000).

    Parameters
    ----------
    Zs : ndarray of shape (ns, nSim)
        Simulated hard data values at the ``ns`` soft data points for
        ``nSim`` realisations.
    softpdftype : int
        Type of the soft PDF template:

        1 – Histogram (piecewise constant, non-uniform widths)
        2 – Linear (piecewise linear, non-uniform widths)
        3 – Grid Histogram (piecewise constant, uniform widths)
        4 – Grid Linear (piecewise linear, uniform widths)

    NV : ndarray of shape (ns,)
        Number of intervals for each soft datum dimension.
    WidthV : ndarray
        Interval widths for each soft datum.

        If ``softpdftype`` is 1 or 2: shape ``(ns, max(NV))`` matrix of
        interval widths.

        If ``softpdftype`` is 3 or 4: shape ``(ns, 1)`` vector of the
        constant interval width for each dimension.

    probdensV : ndarray
        Probability density values along each dimension.

        If ``softpdftype`` is 1 or 3: shape ``(ns, max(NV))`` matrix of
        density values for each interval.

        If ``softpdftype`` is 2 or 4: shape ``(ns, max(NV)+1)`` matrix of
        density values at the interval *limits*.

    seed : int, numpy.random.Generator, or None, optional
        Random seed for reproducibility.

    Returns
    -------
    nl : ndarray of shape (ns,)
        Number of interval limits per soft datum, ``nl = NV + 1``.
    limi : ndarray
        Interval limits for the simulated soft PDFs.

        If ``softpdftype`` is 1 or 2: shape ``(ns, max(nl), nSim)``.

        If ``softpdftype`` is 3 or 4: shape ``(ns, 3, nSim)`` where
        the three columns are lower limit, increment, and upper limit.

    probdens : ndarray
        Probability density values at the interval limits.

        If ``softpdftype`` is 1 or 3: shape ``(ns, max(nl)-1, nSim)``.

        If ``softpdftype`` is 2 or 4: shape ``(ns, max(nl), nSim)``.

    Notes
    -----
    The algorithm draws a sample ``V`` from the probability distribution
    described by ``WidthV`` and ``probdensV`` (which is centred at zero),
    then shifts the template so that its centre falls on the simulated
    hard value: ``limi = Zs - V + template_limits``.

    The ``WidthV`` and ``probdensV`` arrays must be normalised so that
    the total probability integrates to 1.  An exception is raised
    otherwise.
    """
    rng = _make_rng(seed)

    Zs = np.asarray(Zs, dtype=float)
    NV = np.asarray(NV, dtype=int).ravel()
    ns, nSim = Zs.shape
    NVmax = int(np.max(NV))
    nl = NV + 1
    eps = 1e-10

    # --- validate shapes ---
    if softpdftype in (1, 2):
        if WidthV.shape != (ns, NVmax):
            raise ValueError(
                f"WidthV must have shape ({ns}, {NVmax}) for "
                f"softpdftype={softpdftype}."
            )
    elif softpdftype in (3, 4):
        if WidthV.shape != (ns, 1):
            raise ValueError(
                f"WidthV must have shape ({ns}, 1) for "
                f"softpdftype={softpdftype}."
            )
        # Expand to (ns, NVmax) for uniform processing below
        WidthV = np.tile(WidthV, (1, NVmax))
    else:
        raise ValueError(f"softpdftype={softpdftype} is not valid (1–4).")

    if softpdftype in (1, 3):
        if probdensV.shape != (ns, NVmax):
            raise ValueError(
                f"probdensV must have shape ({ns}, {NVmax}) for "
                f"softpdftype={softpdftype}."
            )
    else:  # 2 or 4
        if probdensV.shape != (ns, NVmax + 1):
            raise ValueError(
                f"probdensV must have shape ({ns}, {NVmax + 1}) for "
                f"softpdftype={softpdftype}."
            )

    # --- build cumulative distribution of V (the template shift) ---
    # limiV: (ns, max(nl))  cumulative interval limits starting at 0
    # UV   : (ns, max(nl))  cumulative probability up to each limit
    limiV = np.zeros((ns, int(np.max(nl))))
    UV = np.zeros_like(limiV)

    is_linear = softpdftype in (2, 4)

    for s in range(ns):
        for iv in range(1, nl[s]):
            limiV[s, iv] = limiV[s, iv - 1] + WidthV[s, iv - 1]
            if is_linear:
                UV[s, iv] = (UV[s, iv - 1]
                             + WidthV[s, iv - 1]
                             * 0.5 * (probdensV[s, iv - 1]
                                      + probdensV[s, iv]))
            else:
                UV[s, iv] = (UV[s, iv - 1]
                             + WidthV[s, iv - 1] * probdensV[s, iv - 1])

        # Normalisation check
        total = UV[s, nl[s] - 1]
        if not (1.0 - eps < total < 1.0 + eps):
            raise ValueError(
                f"WidthV and probdensV are not normalised for soft datum "
                f"index {s} (integral = {total:.6f}, expected 1.0)."
            )

    # --- draw V from the template distribution ---
    U = rng.uniform(0.0, 1.0, (ns, nSim))
    V = np.zeros((ns, nSim))

    for s in range(ns):
        UV_grid = UV[s, : nl[s]]
        for j in range(nSim):
            u = U[s, j]
            iv = int(np.sum(UV_grid < u))         # 0-based interval index
            iv = min(iv, nl[s] - 2)               # clamp to last valid interval

            if is_linear:
                f0 = probdensV[s, iv]
                f1 = probdensV[s, iv + 1]
                w = WidthV[s, iv]
                fsp = (f1 - f0) / w if w > 0 else 0.0
                if abs(fsp) > 1e-14:
                    fso = f0 - limiV[s, iv] * fsp
                    disc = f0 ** 2 + 2.0 * fsp * (u - UV_grid[iv])
                    V[s, j] = (1.0 / fsp) * (-fso + np.sqrt(max(disc, 0.0)))
                elif f0 > 1e-14:
                    V[s, j] = limiV[s, iv] + (u - UV_grid[iv]) / f0
                else:
                    V[s, j] = limiV[s, iv]
            else:
                if probdensV[s, iv] > 1e-14:
                    V[s, j] = limiV[s, iv] + (u - UV_grid[iv]) / probdensV[s, iv]
                else:
                    V[s, j] = limiV[s, iv]

    # --- centre template on each Zs realisation ---
    Y = Zs - V   # (ns, nSim)

    # --- build output arrays ---
    if softpdftype in (1, 2):
        max_nl = int(np.max(nl))
        limi = np.full((ns, max_nl, nSim), np.nan)
        for s in range(ns):
            for iv in range(nl[s]):
                limi[s, iv, :] = Y[s, :] + limiV[s, iv]

        if softpdftype == 1:
            probdens = np.full((ns, max_nl - 1, nSim), np.nan)
            for s in range(ns):
                for iv in range(NV[s]):
                    probdens[s, iv, :] = probdensV[s, iv]
        else:  # type 2
            probdens = np.full((ns, max_nl, nSim), np.nan)
            for s in range(ns):
                for iv in range(nl[s]):
                    probdens[s, iv, :] = probdensV[s, iv]

    else:  # Grid types (3 or 4)
        limi = np.full((ns, 3, nSim), np.nan)
        for s in range(ns):
            limi[s, 0, :] = Y[s, :]
            limi[s, 1, :] = WidthV[s, 0]
            limi[s, 2, :] = Y[s, :] + NV[s] * WidthV[s, 0] + eps

        if softpdftype == 3:
            probdens = np.full((ns, int(np.max(nl)) - 1, nSim), np.nan)
            for s in range(ns):
                for iv in range(NV[s]):
                    probdens[s, iv, :] = probdensV[s, iv]
        else:  # type 4
            probdens = np.full((ns, int(np.max(nl)), nSim), np.nan)
            for s in range(ns):
                for iv in range(nl[s]):
                    probdens[s, iv, :] = probdensV[s, iv]

    return nl, limi, probdens


def simuinterval(
    Zs: np.ndarray,
    I: float | np.ndarray,
    Vmethod: int,
    Vvalue: float = 0.0,
    *,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate soft interval data from hard data values.

    Generates the lower and upper bounds of soft interval data centred
    on known Z values at the soft data points.  Four methods control
    how the interval is placed relative to each Z value.

    Adapted from ``simuinterval.m`` (bmelib simulib, Jan 1, 2000).

    Parameters
    ----------
    Zs : ndarray of shape (ns, nSim)
        Known Z values at the ``ns`` soft data points for ``nSim``
        realisations.
    I : float or ndarray
        Interval length parameter.  Interpretation depends on
        ``Vmethod``:

        - ``Vmethod`` 0, 1, 2: scalar interval length.
        - ``Vmethod`` 3: sorted vector of breakpoints ``[I₀, I₁, …]``
          defining categorical intervals.  A value ``Zs(i,j)``
          falling in the k-th interval ``(Iₖ, Iₖ₊₁]`` produces
          ``a = Iₖ``, ``b = Iₖ₊₁``.

    Vmethod : int
        Method for placing the interval:

        0 – Symmetric: ``a = Zs − I/2``, ``b = Zs + I/2``  (V = 0).
        1 – Random:    ``a = Zs + V − I/2``, ``b = Zs + V + I/2``
            where V ~ Uniform(−I/2, I/2).
        2 – Fixed:     ``a = Zs + V − I/2``, ``b = Zs + V + I/2``
            where V = ``Vvalue`` (constant).
        3 – Categorical: ``I`` is a vector of breakpoints; the interval
            containing ``Zs`` is returned.

    Vvalue : float, optional
        Fixed offset used only when ``Vmethod=2``.  Must satisfy
        ``−I/2 ≤ Vvalue ≤ I/2``.  Default is 0.
    seed : int, numpy.random.Generator, or None, optional
        Random seed for reproducibility.  Only used when
        ``Vmethod=1``.

    Returns
    -------
    a : ndarray of shape (ns, nSim)
        Lower bounds of the simulated intervals.
    b : ndarray of shape (ns, nSim)
        Upper bounds of the simulated intervals.  By construction,
        ``a ≤ Zs ≤ b`` for ``Vmethod`` 0, 1, 2.

    Raises
    ------
    ValueError
        If ``Vmethod=2`` and ``Vvalue`` lies outside ``[−I/2, I/2]``.
    ValueError
        If ``Vmethod`` is not in {0, 1, 2, 3}.

    Notes
    -----
    For ``Vmethod=3``, the infinite lower sentinel is set to ``−5`` and
    the infinite upper sentinel to ``5``, matching the MATLAB source
    convention.  Adjust the breakpoint vector ``I`` accordingly for
    data that lie outside this range.
    """
    rng = _make_rng(seed)

    Zs = np.asarray(Zs, dtype=float)
    ns, nSim = Zs.shape

    if Vmethod == 0:
        V = 0.0
    elif Vmethod == 1:
        V = (rng.uniform(0.0, 1.0, (ns, nSim)) - 0.5) * I
    elif Vmethod == 2:
        half = I / 2.0
        if not (-half - 1e-12 <= Vvalue <= half + 1e-12):
            raise ValueError(
                f"Vvalue={Vvalue} must be in the interval [−I/2, I/2] = "
                f"[{-half:.4g}, {half:.4g}]."
            )
        V = Vvalue
    elif Vmethod == 3:
        ZkGrid = np.asarray(I, dtype=float)
        mInf, pInf = -5.0, 5.0
        ZkGridInf = np.concatenate([[mInf], ZkGrid, [pInf]])
        a = np.empty((ns, nSim))
        b = np.empty((ns, nSim))
        for j in range(nSim):
            for s_idx in range(ns):
                z_val = Zs[s_idx, j]
                ia = int(np.sum(ZkGrid < z_val))   # 0-based index
                a[s_idx, j] = ZkGridInf[ia]
                b[s_idx, j] = ZkGridInf[ia + 1]
        return a, b
    else:
        raise ValueError(
            f"Vmethod={Vmethod} is not a valid option (must be 0, 1, 2, or 3)."
        )

    a = Zs + V - I / 2.0
    b = Zs + V + I / 2.0
    return a, b
