import os
import numpy as np
from .latticeseq_b2 import latticeseq_b2


def qmc(
    func, xmin, xmax, args=(), kwargs={},
    nshifts=8, pow2min=3, pow2max=20,
    chebyshevk=2, abserr=0, relerr=1e-06, maxeval=None,
    showinfo=True):
    """Quasi-Monte Carlo (QMC) numerical integration over a rectangular domain.

    Approximates the multivariate integral

    .. math::

        Q = \\int_{\\mathbf{x}_{\\min}}^{\\mathbf{x}_{\\max}} f(\\mathbf{x})\\, d\\mathbf{x}

    using a randomised lattice sequence (base-2 Sobol/lattice points generated
    by ``latticeseq_b2``).  Multiple independent random shifts of the lattice
    are used to obtain an unbiased error estimate via the Chebyshev inequality.

    The integration domain is an axis-aligned *n*-dimensional rectangle
    ``[xmin, xmax]``.

    Parameters
    ----------
    func : callable
        The integrand function.  Two calling conventions are supported:

        * **Vectorised** — ``func(x, *args, **kwargs)`` where ``x`` has shape
          ``(npts, ndim)`` and the return has shape ``(npts, fdim)``.
        * **Scalar** — ``func(x, *args, **kwargs)`` where ``x`` is a 1-D
          array of shape ``(ndim,)`` and the return is a scalar or 1-D array
          of shape ``(fdim,)``.  Vectorised mode is detected automatically.
    xmin : array-like, shape (1, ndim) or (ndim,)
        Lower integration bounds.  Converted internally to shape ``(1, ndim)``.
    xmax : array-like, shape (1, ndim) or (ndim,)
        Upper integration bounds.  Converted internally to shape ``(1, ndim)``.

        .. note::
            The maximum number of supported dimensions for the current
            generating vector is **250**.
    args : tuple, optional
        Positional arguments forwarded to *func* after *x*.
    kwargs : dict, optional
        Keyword arguments forwarded to *func*.
    nshifts : int, optional
        Number of independent random shifts of the lattice sequence used to
        estimate the standard error (default 8).  More shifts give a more
        reliable error estimate.
    pow2min : int, optional
        Minimum number of function evaluations is ``2**pow2min * nshifts``
        (default 3, i.e. at least 8 × nshifts evaluations).
    pow2max : int, optional
        Maximum number of function evaluations is ``2**pow2max * nshifts``
        (default 20).  Overridden by *maxeval* if set.
    chebyshevk : float, optional
        Multiplier for the standard error to form the error estimate.
        Based on the Chebyshev inequality, the confidence that the true
        error is within ``chebyshevk * stderr`` is at least
        ``1 − 1/chebyshevk²``:

        +---+-----------+
        | k | Confidence|
        +===+===========+
        | 2 | 75%       |
        | 3 | ~88%      |
        | 4 | ~93%      |
        | 5 | ~96%      |
        | 8 | ~98%      |
        +---+-----------+

        Default is 2.
    abserr : float, optional
        Absolute error tolerance (default 0, i.e. ignored unless > 0).
    relerr : float, optional
        Relative error tolerance (default 1e-6).  Convergence is declared
        when, for every output component *i*:
        ``chebyshevk * stderr[i] <= max(abserr, relerr * |Q[i]|)``.
    maxeval : int or None, optional
        If set, ``pow2max`` is adjusted so that the total evaluations
        ``2**pow2max * nshifts`` does not exceed *maxeval*.
    showinfo : bool, optional
        If ``True`` (default), print the QMC result, std, evaluation count,
        and requested tolerances after convergence.

    Returns
    -------
    Q : np.ndarray, shape (fdim,)
        Approximation to the integral.
    estAbsErr : np.ndarray, shape (fdim,)
        Estimated absolute error: ``chebyshevk * stderr(nshifts independent
        approximations)``.
    infos : dict
        Diagnostic information with keys:

        * ``'stdQ'`` — np.ndarray of shape (fdim,): standard deviation of
          the *nshifts* independent approximations.
        * ``'nbFunEvals'`` — int: total number of function evaluations.

    Notes
    -----
    **Convergence criterion** uses a combined absolute/relative tolerance:
    a component is converged if its error is within *either* the absolute
    tolerance *or* the relative tolerance.  This prevents near-zero
    components from blocking convergence forever.

    **Lattice sequence** — the generating vector is loaded from
    ``latticeseq_b2`` (a base-2 lattice rule optimised for variance
    reduction in high dimensions).

    **Vectorisation detection** — ``func`` is called with two points
    ``(xmin, xmax)`` to probe whether it is vectorised.  If it raises an
    exception, non-vectorised mode is assumed.

    In the BME context, ``qmc`` is used in ``BMEPosteriorMoments`` and
    ``BMEPosteriorPDF`` to compute the normalisation constant and moment
    integrals for non-Gaussian soft-data PDFs.

    Examples
    --------
    Integrate the unit sphere volume in 3-D spherical coordinates:

    >>> import numpy as np
    >>> from stamps.stamps.mvn.qmc import qmc
    >>> def sphere(x):
    ...     r, theta, phi = x[:, 0], x[:, 1], x[:, 2]
    ...     return (r ** 2 * np.sin(phi)).reshape(-1, 1)
    >>> Q, err, info = qmc(sphere, [0, 0, 0], [1, 2*np.pi, np.pi],
    ...                    relerr=1e-4, showinfo=False)
    >>> float(Q)   # should be ≈ 4π/3 ≈ 4.189
    4.18...

    ### Original Reference ###
    Lattice sequence approximation of multivariate integral (using latticeseq_b2).

    Inputs:
        func   function to integrate over the n-dimensional rectangular area
        xmin   shape (1, n) lower bounds
        xmax   shape (1, n) upper bounds
        args / kwargs  passed to func
        nshifts  number of randomizations for stderr estimation
        chebyshevk  multiplier of stderr for error estimate
        maxeval  max function evaluations

    Outputs:
        Q            approximation to the integral
        estAbsErr    estimated absolute error
        infos        dict with stdQ and nbFunEvals
    """

    if maxeval != None:
        pow2max = min(int(np.rint(np.log2(maxeval/nshifts))), 20)

    xmin = np.array(xmin, ndmin=2)
    xmax = np.array(xmax, ndmin=2)
    scale = xmax - xmin
    jacobian = np.prod(scale)

    ndim = xmin.shape[1] # get ndim
    latgen = latticeseq_b2(s=ndim)

    np.random.seed(1) # Mersenne Twister reset
    shifts = np.random.rand(nshifts, ndim)

    # test vectorized and get fdim
    try:
        fdim = func(np.vstack((xmin,xmax)), *args, **kwargs).shape[1]
        vectorized = True
    except Exception as e: #has a risk if func is for no vectorized but
                         #can accept 2d numpy array
        print ('qmc warning: integration function is not vectorized.', e)
        val = func(xmin.ravel(), *args, **kwargs)
        if hasattr(val,'__len__'):
            fdim = len(val)
        else:
            fdim = 1
        vectorized = False
    acc = np.zeros((nshifts,fdim))
    m = 0

    for m in range(0, pow2max + 1):
        # generate an array of 2**(m-1) x s unless m=0, 1 x s
        x = latgen.calc_block(m)
        if vectorized == True:
            npts = x.shape[0]
            x_vec = np.tile(x, (nshifts, 1))
            sh_vec = np.repeat(shifts, npts, axis=0)
            x_sh_vec = (scale * ((x_vec + sh_vec) % 1) + xmin)
            v_vec = func(x_sh_vec, *args, **kwargs)
            acc += jacobian* v_vec.reshape((nshifts, npts, fdim)).sum(axis=1)
        else:
            for xk in x:
                for j in range(nshifts):
                    xshifted = (scale * ((xk + shifts[j,:]) % 1) + xmin).ravel()
                    #force convert to 1d array
                    fval = np.array(func(xshifted, *args, **kwargs), ndmin=1)
                    acc[j] += jacobian * fval
        Q = np.mean(acc/2**m, axis=0)
        stdQ = np.std(acc/2**m, axis=0) / np.sqrt(nshifts)
        if m < pow2min: continue
        # Use combined absolute/relative tolerance for each component:
        # A component is converged if its error is within EITHER the
        # absolute tolerance OR the relative tolerance.  This prevents
        # near-zero Q components from blocking convergence forever.
        err_est = stdQ * chebyshevk
        tol = np.maximum(abserr, relerr * np.abs(Q))
        if (err_est <= tol).all(): break
    if showinfo:
        print ("QMC Q=", Q, " std=", stdQ, " N=", 2**m * nshifts, \
              " s=", ndim, " relErrReq=", relerr, " absErrReq=", abserr)
    
    #Q = Q
    estAbsErr = chebyshevk*stdQ
    #stdQ = stdQ
    nbFunEvals = 2**m*nshifts
    infos = {'stdQ': stdQ, 'nbFunEvals': nbFunEvals}
    
    return Q, estAbsErr, infos


if __name__ == '__main__':
    import time

    radius = 1.
    xmin = np.array([0, 0, 0], np.float64)
    xmax = np.array([radius, 2*np.pi, np.pi], np.float64)
    def integrand_sphere(x_array):
        r, theta, phi = x_array
        return r**2*np.sin(phi)

    def integrand_sphere_vectorized(x_array):
        r = x_array[:,0:1]
        theta = x_array[:,1:2]
        phi = x_array[:,2:3]
        # r, theta, phi = x_array
        return r**2*np.sin(phi)
 
    # # TOO SLOW without vectorized...
    # v,e = qmc(integrand_sphere, xmin, xmax)
    # print v,e
    method = 'qmc'
    abserr = 0
    relerr = 10**-7

    print ('Using                 :', method)
    sss = time.time()
    v, e, infos = \
        qmc(
            integrand_sphere_vectorized, xmin, xmax,
            abserr=abserr, relerr=relerr, pow2max=24, showinfo=True
            )
    finfo = {
        'thinksAbsErrReqReached': e < abserr,
        'thinksRelErrReqReached': e < np.abs(v) * relerr,
        'stdQ': infos['stdQ'],
        'nbFunEvals': infos['nbFunEvals'] }
    print ('Time                  : %.3g sec' % (time.time() - sss))
    orgprintopts = np.get_printoptions()
    np.set_printoptions(formatter={'float': '{: 10.03e}'.format})
    print ('Value                 :', v)
    print ('EstAbsErr             :', e)
    for key, val in finfo.items():
        print( "%-22s: %s" % (key, val))
    np.set_printoptions(orgprintopts)
    
    # print 'sphere volume:', v
    # print 'sphere volume error:', e
    print ('exact sphere volume:', 4/3. * np.pi * radius**3)
