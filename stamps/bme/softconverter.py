# -*- coding: utf-8 -*-
import csv
import warnings

import numpy as np
from scipy.stats import norm as _norm_dist, t as _t_dist
from six.moves import range

from .pystks_variable import get_standard_soft_pdf_type


def ud2ud(file_name, usecols, skiprows, delimiter):
    '''使用者自定型
    mean, var: n by 1 numpy array
    
    '''
    limi_idx = usecols[0] # idx of limi

    # First pass: read nl values
    with open(file_name, "r", encoding='utf-8') as f:
        c = csv.reader(f, delimiter=delimiter)
        #skip rows
        for skiprow in range(skiprows):
            next(c)
        nol = []
        for line in c:
            d = line
            a = int(d[limi_idx]) + 1
            nol.append(a)

    nolmax = max(nol)

    # Second pass: read limi and probdens values
    with open(file_name, "r", encoding='utf-8') as f2:
        readcsv = csv.reader(f2, delimiter=delimiter)
        #skip rows
        for skiprow in range(skiprows):
            next(readcsv)
        limi = []
        probdens = []
        for i in readcsv:
            d2 = i
            a2 = int(d2[limi_idx])
            px1 = tuple(d2[limi_idx + 1 : limi_idx + 1 + a2 + 1])
            px1 = px1 + tuple([0] * (nolmax - a2 - 1))
            px = list(map(float, px1))
            limi.append(px)

            py1 = tuple(d2[limi_idx + 1 + a2 + 1 : limi_idx + 1 + a2 + 1 + a2 + 1])
            py1 = py1 + tuple([0] * (nolmax - a2 - 1))
            py = list(map(float, py1))
            probdens.append(py)

    nl = np.array(nol, ndmin=2).T
    limi = np.array(limi, ndmin=2)
    probdens = np.array(probdens, ndmin=2)
    return nl, limi, probdens

def gs2ud(mean, var):
    '''轉換高斯型資料至使用者自定型
    mean, var: n by 1 numpy array
    
    '''
    softpdftype = 2
    limi_norm = np.array([-3.719,-3.090,-2.326,
                          -1.645,-0.524,-0.253,
                           0.000,
                           0.253, 0.524, 1.645,
                           2.326, 3.090, 3.719,])
    limi_n = len(limi_norm)

    nl = np.ones((mean.shape[0],1), dtype=int) * limi_n
    limi = np.kron(np.sqrt(var),limi_norm).reshape((-1,limi_n))+mean
    probdens = limi-mean
    probdens = -(probdens**2)/2/var
    probdens = 1/np.sqrt(2*np.pi*var)*np.exp(probdens)

    nl, limi, probdens, _ = proba2probdens(softpdftype, nl, limi, probdens)

    return nl, limi, probdens

def uf2ud(low, up):

    softpdftype = 2
    limi_norm = np.linspace(-1,1,5)
    limi_n = len(limi_norm)
    nl = np.ones((low.shape[0],1), dtype=int) * limi_n
    limi = np.kron((up-low)/2.,limi_norm)+(up+low)/2.
    probdens = np.ones((low.shape[0],limi_n))*(1./(up-low))

    nl, limi, probdens, _ = proba2probdens(softpdftype, nl, limi, probdens)

    return nl, limi, probdens

def ud2zs(softpdftype, nl, limi, probdens):
    '''
    user defined to new zs data
    new zs data: a sequence of zsdata,
        zsdata is a sequence of pdftype, *pdf_args
        e.g. zsdata1 = (2, nl, limi, probdens)
        e.g. zsdata2 = (10, mean, var)
        e.g. new zs data = (zsdata1, zsdata2)
    '''
    zsdata = []
    if nl.size != 0:
        for n_i, l_i, p_i in zip(nl, limi, probdens):
            zsdata.append([softpdftype, n_i, l_i, p_i])

    return zsdata

def ud2zs_temp(softpdftype, nl, limi, probdens):
    '''
    user defined to new zs data
    new zs data: a sequence of zsdata,
        zsdata is a sequence of pdftype, *pdf_args
        e.g. zsdata1 = (2, nl, limi, probdens)
        e.g. zsdata2 = (10, mean, var)
        e.g. new zs data = (zsdata1, zsdata2)
    '''
    zsdata = []
    if nl.size != 0:
        for n_i, l_i, p_i in zip(nl, limi, probdens):
            zsdata.append([softpdftype, n_i, l_i, p_i])
    return zsdata
    
def uf2zs(softpdftype, low, up):
    '''
    new zs data
    '''
    limi_norm = np.linspace(-1,1,5)
    limi_n = len(limi_norm)
    nl = np.ones((low.shape[0],1), dtype=int) * limi_n
    limi = np.kron((up-low)/2.,limi_norm)+(up+low)/2.
    probdens = np.ones((low.shape[0],limi_n))*(1./(up-low))
    zsdata = []
    for n_i, l_i, p_i in zip(nl, limi, probdens):
        zsdata.append([softpdftype, n_i, l_i, p_i])

    return zsdata

def zs2ud(zs):
    '''new zs data'''
    softpdftype = 2
    mean = np.array([zsi[1] for zsi in zs]).reshape((-1,1))
    var = np.array([zsi[2] for zsi in zs]).reshape((-1,1))
    nl, limi, probdens = gs2ud(mean, var)

    return softpdftype, nl, limi, probdens

def proba2probdens(softpdftype, nl, limi, probdens):
    '''
    proba2probdens            - Normalizes the probability density function (Jan 1, 2001)
  
    Calculates the norm (area under the curve) of a function template, and returns
    a normalized pdf. This function uses the syntax of probabilistic data (see 
    probasyntax) to define the pdf.
   
    SYNTAX :
   
    [probdens,norm]=proba2probdens(softpdftype,nl,limi,probdenstemplate);
   
    INPUT :
   
    softpdftype scalar      indicates the type of soft pdf representing the  
                            probabilitic soft data.  
                            softpdftype may take value 1, 2, 3 or 4, as follow:
                            1 for Histogram, 2 for Linear, 3 for Grid histogram, 
                            and 4 for Grid Linear.
                            In current status, only softpdftype 2 is avaiable for use 
    nl          ns by 1     2D array of the number of interval limits. nl(i) is the number  
                            of interval limits used to define the soft pdf for soft data 
                            point i. (see probasyntax for more explanations)
    limi        ns by l     2D array of interval limits, where l is equal to
                            either max(nl) or 3 depending of the softpdftype.
                            limi(i,:) are the limits of intervals for the i-th 
                            soft data. (see probasyntax for more explanations)
    probdenstemplate        ns by p matrix of non-normalized probability density values,  
                            where p is equal to either max(nl)-1 or max(nl), depending on 
                            the softpdftype. probdenstemplate(i,:) are the values of the  
                            non-normalized probability density corresponding to the intervals  
                            for the i-th soft data defined in limi(i,:). (see probasyntax for 
                            more explanations)
   
    OUTPUT :
    nl          ns by 1     2D array of the number of interval limits. nl(i) is the number  
                            of interval limits used to define the soft pdf for soft data 
                            point i. (see probasyntax for more explanations)
    limi        ns by l     2D array of interval limits, where l is equal to
                            either max(nl) or 3 depending of the softpdftype.
                            limi(i,:) are the limits of intervals for the i-th 
                            soft data. (see probasyntax for more explanations)
    probdens    ns by p     2D array of normalized probability density values,
                            each row of probadens is equal to the corresponding row
                            of probadenstemplate divided by its normalization constant
    norm        ns by 1     2D array of the normalization constants (area under the curve)
                            of each row of probadenstemplat, e.g. norm(is) is the
                            normalization constant for probdenstemplate(is,:)  
    ''' 
    softpdftype = 2 # always 2 for now
    
    if type(nl) is int: # only one soft data
      limi = limi.reshape(1, limi.size)
      probdens = probdens.reshape(1, probdens.size)
      nl = np.array(nl).reshape(1, 1)
    elif limi.shape != probdens.shape:
      limi = limi.reshape((len(nl), limi.size//len(nl)))
      probdens = probdens.reshape((len(nl), limi.size//len(nl)))
      nl = nl.reshape((len(nl), 1))
    norm_probdens = np.zeros(probdens.shape)
    area = np.zeros(nl.shape)
    m = 0
    for nl_i, limi_i, probdens_i in zip(nl, limi, probdens):
        nl_i = int(nl_i[0])
        limi_i = limi_i[: nl_i]
        probdens_i_original = np.copy(probdens_i) #copy
        probdens_i = probdens_i[:nl_i]

        height = limi_i[1:] - limi_i[:-1]
        sum_up_low = probdens_i[:-1] + probdens_i[1:]
        area[m] = (sum_up_low * height / 2.).sum()
        norm_probdens_i = probdens_i / area[m]
        probdens_i_original[ :nl_i] = norm_probdens_i
        norm_probdens[m]=probdens_i_original#.append( probdens_i_original )
        m = m+1
    
    # NOTE: Removed legacy len(nl)==1 flattening that broke downstream
    # consumers expecting 2D arrays (e.g. softpdftypecheckargs).
    # Arrays are always returned as 2D (ns x nlMax).
      
    return nl, limi, norm_probdens, area

def proba2stat(softpdftype, nl, limi, probdens):
    def range_include_end(start, end, step):
        include_end_indexs = np.where( (end - start)%step == 0 )[0]
        revised_end = end.copy()
        revised_end[ include_end_indexs ] += step[ include_end_indexs ]
        result = list(map( lambda x:np.arange(*x), zip(start, revised_end, step) ))
        return result

    softpdftype = get_standard_soft_pdf_type(softpdftype)
    if nl.shape[0] == 0:
        return np.array([]).reshape( (0,1) ), np.array([]).reshape( (0,1) )

    if softpdftype == 1 or softpdftype == 2:
        L1 = limi[:,:-1]
        L2 = limi[:,1:]
    elif softpdftype == 3 or softpdftype == 4: #grid
        nlMax = nl.max()
        limi_expand = np.zeros( (nl.shape[0], nlMax) )
        for idx, i in enumerate( range_include_end(limi[:,0], limi[:,2], limi[:,1]) ):
            limi_expand[idx][:i.size] = i
        L1 = limi_expand[:,:-1]
        L2 = limi_expand[:,1:]

    if softpdftype == 1 or softpdftype == 3:
        P1 = probdens
        XsMean_mat = (1/2.) * P1 * (L2**2 - L1**2)
        Xs2Mean_mat = (1/3.) * P1 * (L2**3 - L1**3)
    elif softpdftype == 2 or softpdftype == 4:
        P1 = probdens[:,:-1]
        P2 = probdens[:,1:]
        dL = L2 - L1
        # Guard against divide-by-zero when consecutive limi values are equal
        safe_dL = np.where(np.abs(dL) < 1e-30, 1.0, dL)
        fsp = np.where(np.abs(dL) < 1e-30, 0.0, (P2 - P1) / safe_dL)
        fso = P1 - L1 * fsp

        L1p2 = L1 * L1
        L1p3 = L1p2 * L1
        L1p4 = L1p3 * L1

        L2p2 = L2 * L2
        L2p3 = L2p2 * L2
        L2p4 = L2p3 * L2
        
        XsMean_mat = ( 1 / 2. ) * ( fso * (L2p2 - L1p2) ) + ( 1 / 3. ) * (fsp * (L2p3 - L1p3) )
        Xs2Mean_mat = ( 1 / 3. ) * ( fso * (L2p3 - L1p3) ) + ( 1 / 4. ) * ( fsp * (L2p4 - L1p4) )
        # Replace any residual NaN/Inf from degenerate intervals with zero
        np.nan_to_num(XsMean_mat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(Xs2Mean_mat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    XsMean = []
    Xs2Mean = []
    for nl_i, XsMean_i, Xs2Mean_i in zip( nl, XsMean_mat, Xs2Mean_mat ):
        XsMean.append( [ XsMean_i[ : nl_i[0] - 1].sum()] )
        Xs2Mean.append( [ Xs2Mean_i[ : nl_i[0] - 1].sum()] )

    XsMean, Xs2Mean = np.array( XsMean ), np.array( Xs2Mean )
    softmean = XsMean
    softvar = Xs2Mean - XsMean**2

    return softmean, softvar

def proba2quantile(softdpftype, nl, limi, probdens, quantiles = []):
    '''give a discrete pdf, return the quantiles user gave'''

    #default quantiles
    if quantiles:
        pass
    else:
        quantiles = [ .05, .25, .50, .75, .95, ]

    
    #get cdf
    probdens_quantile = []
    # probdens_cdf = []

    for nl_i, limi_i, probdens_i in zip( nl, limi, probdens ):

        #clip
        nl_i = int( nl_i[ 0 ] )
        limi_i = limi_i[ : nl_i ]
        probdens_i = probdens_i[ :nl_i ]

        #get area_i
        height = limi_i[ 1: ] - limi_i[ :-1 ]
        sum_up_low = probdens_i[ :-1 ] + probdens_i[ 1: ]
        area_i = (sum_up_low * height / 2.)
        
        #set cdf
        probdens_cdf_i = [ 0. ]
        for p in area_i:
            probdens_cdf_i.append( probdens_cdf_i[ -1 ] + p )

        #get interp
        probdens_quantile.append( np.interp( quantiles, probdens_cdf_i, limi_i ) )

    return np.array( probdens_quantile )

def pdf2cdf(zs):
    Fs = []
    for idx_k, zsi in enumerate(zs):
        softpdftype = zsi[0]
        if softpdftype == 2:
            nl = zsi[1]
            limi = zsi[2]
            probdens = zsi[3]
            height = np.diff(limi)
            sum_up_low = probdens[:-1] + probdens[1:]    
            cumsum = np.hstack([0,np.cumsum(height*sum_up_low/2)])
            probCDFs = cumsum/float(cumsum[nl-1])
            Fs.append([softpdftype,nl,limi,probdens,probCDFs])     
    return Fs

def probaUniform(zlow, zup):
    '''
    function [softpdftype,nl,limi,probdens]=probaUniform(zlow,zup)
    % probaUniform      - creates Uniform soft probabilistic data  at a set of data points
    %
    % This program generates uniform probabilistic soft pdf at a set of data points.
    % At each data point the lower and upper bound of the z values are known to be zlow and zup.
    % The soft probrabilistic distribution is given by the uniform pdf as follow
    % pdf(z)=1/(zup-low) for zlow<z<zup, 0 otherwise.
    %
    % SYNTAX :
    %
    % [softpdftype,nl,limi,probdens]=probaUniform(zlow,zup);
    %
    % INPUT :
    %
    % zlow     ns by 1       vector of the lower bound value at ns soft data point
    % zup      ns by 1       vector of the upper bound value at ns soft data point
    %
    % OUTPUT :
    %
    % softpdftype scalar=2   indicates the type of soft pdf representing the probabilitic soft data.  
    % nl          ns by 1    vector of number of limits  nl (see probasyntax)
    % limi        ns by 2    matrix representing the limits  values (see probasyntax)
    % probdens    ns by 2    matrix representing the probability densities (see probasyntax)
    %
    % SEE ALSO probaGaussian, probaStundentT and probasyntax.m 
    '''

    ns=len(zlow)
    softpdftype=2#*np.ones((ns,1))
    nl=2*np.ones((ns,1)).astype(int)
    zlow=np.array(zlow).reshape((ns,1))
    zup=np.array(zup).reshape((ns,1))
    limi=np.hstack([zlow, zup])
    probdens=np.kron(1./np.diff(limi),[1,1])
    zs=[softpdftype,nl,limi,probdens]
    #zs=[[softpdftype[m],nl[m],limi[m],probdens[m]] for m in range(ns)]

    return zs


# =============================================================================
# Private helpers  (ported from BMElib genlib)
# =============================================================================

def _isdupli(p: np.ndarray) -> bool:
    """Return True if *p* contains at least one pair of duplicate rows.

    Adapted from ``isdupli.m`` (BMElib genlib).

    Parameters
    ----------
    p : array_like of shape (n, d)
        Matrix of n point coordinates in a d-dimensional space.

    Returns
    -------
    bool
        ``True`` when ``p`` has fewer unique rows than total rows.
    """
    p = np.atleast_2d(p)
    return p.shape[0] > np.unique(p, axis=0).shape[0]


def _finddupli(p: np.ndarray):
    """Find duplicate coordinate rows in *p*.

    Adapted from ``finddupli.m`` (BMElib genlib).

    Parameters
    ----------
    p : array_like of shape (n, d)
        Matrix of n point coordinates in a d-dimensional space.

    Returns
    -------
    iu : ndarray of shape (nu,)
        0-based indices of the rows that are **not** part of any duplicate
        cluster (i.e. the unique representatives).  When there are no
        duplicates, ``iu`` contains all indices 0 … n-1.
    ir : list of ndarray
        Each element ``ir[k]`` is a 1-D array of 0-based indices that form
        the k-th cluster of duplicate points.  ``p[ir[k]]`` are all equal.
        When there are no duplicates, ``ir`` is an empty list.

    Notes
    -----
    The original MATLAB function uses 1-based indexing; all indices here
    are 0-based.
    """
    p = np.atleast_2d(p).astype(float)
    n = p.shape[0]
    all_idx = np.arange(n)

    if not _isdupli(p):
        return all_idx, []

    # Sort rows lexicographically and track original positions
    sort_order = np.lexsort(p[:, ::-1].T)
    ps = p[sort_order]

    # Find rows equal to their predecessor
    same_as_prev = np.all(ps[1:] == ps[:-1], axis=1)  # length n-1

    # Detect block boundaries: a block starts where same_as_prev[i-1]==False
    # and ends where same_as_prev[i]==False (for blocks of length >= 2)
    padded = np.concatenate([[False], same_as_prev, [False]])
    starts = np.where(~padded[:-1] & padded[1:])[0]   # first of each dup group
    ends   = np.where( padded[:-1] & ~padded[1:])[0]  # last  of each dup group

    ir = []
    dup_set = set()
    for s, e in zip(starts, ends):
        cluster_sorted = sort_order[s: e + 1]
        ir.append(cluster_sorted)
        dup_set.update(cluster_sorted)

    iu = np.array(sorted(set(all_idx) - dup_set), dtype=int)
    return iu, ir


# =============================================================================
# Argument checking
# =============================================================================

def softpdftypeCheckArgs(
    softpdftype: int,
    nl: np.ndarray,
    limi: np.ndarray,
    probdens: np.ndarray,
) -> None:
    """Check the validity of soft PDF arguments.

    Adapted from ``softpdftypeCheckArgs.m`` (BMElib bmeprobalib, Jan 1, 2001).

    Validates all four arguments defining a probabilistic soft PDF.  Raises a
    ``ValueError`` with a descriptive message when an inconsistency is found.

    Parameters
    ----------
    softpdftype : int
        Integer code for the type of soft PDF:

        * ``1`` – Histogram
        * ``2`` – Linear (piecewise-linear PDF)
        * ``3`` – Grid histogram
        * ``4`` – Grid linear

        See ``probasyntax`` in the original BMElib documentation for details.
    nl : ndarray of shape (ns, 1)
        Number of interval limits for each of the ns soft data points.
        ``nl[i, 0]`` is the number of limits used for soft datum i.
    limi : ndarray of shape (ns, l)
        Matrix of interval limits.  ``l`` equals ``max(nl)`` for
        softpdftype 1 or 2, and 3 for softpdftype 3 or 4.
        ``limi[i, :]`` are the limits for soft datum i (0-based row index).
    probdens : ndarray of shape (ns, p)
        Matrix of probability-density values.  ``p`` equals ``max(nl) - 1``
        for softpdftype 1 or 3, and ``max(nl)`` for softpdftype 2 or 4.

    Raises
    ------
    ValueError
        When any argument fails a consistency check.
    """
    if not np.isscalar(softpdftype) and np.size(softpdftype) != 1:
        raise ValueError("softpdftype must be a scalar integer")
    spt = int(softpdftype)
    if spt not in (1, 2, 3, 4):
        raise ValueError("softpdftype must be 1, 2, 3, or 4")

    nl = np.atleast_2d(np.asarray(nl))
    limi = np.atleast_2d(np.asarray(limi))
    probdens = np.atleast_2d(np.asarray(probdens))

    ns = nl.shape[0]
    if ns > 0 and nl.shape[1] != 1:
        raise ValueError("nl must be an (ns, 1) column vector")
    if limi.shape[0] != ns or probdens.shape[0] != ns:
        raise ValueError("nl, limi, and probdens must have the same number of rows")

    if ns > 0:
        nl_max = int(nl.max())
        if spt in (1, 2):
            if limi.shape[1] != nl_max:
                raise ValueError(
                    f"limi must be (ns, max(nl)) = (ns, {nl_max}) for softpdftype {spt}"
                )
        else:  # 3 or 4
            if limi.shape[1] != 3:
                raise ValueError(
                    f"limi must be (ns, 3) for softpdftype {spt} (Grid types)"
                )
        if spt in (1, 3):
            if probdens.shape[1] != nl_max - 1:
                raise ValueError(
                    f"probdens must be (ns, max(nl)-1) = (ns, {nl_max - 1}) "
                    f"for softpdftype {spt}"
                )
        else:  # 2 or 4
            if probdens.shape[1] != nl_max:
                raise ValueError(
                    f"probdens must be (ns, max(nl)) = (ns, {nl_max}) "
                    f"for softpdftype {spt}"
                )
        # For grid types: verify nl is consistent with limi(:,0):limi(:,1):limi(:,2)
        if spt > 2:
            for i in range(ns):
                a, step, b = limi[i, 0], limi[i, 1], limi[i, 2]
                expected_nl = int(round((b - a) / step)) + 1
                if int(nl[i, 0]) != expected_nl:
                    raise ValueError(
                        f"Row {i} of limi requires nl[{i}] = {expected_nl}, "
                        f"got {int(nl[i, 0])}"
                    )


# =============================================================================
# Interval extraction
# =============================================================================

def proba2interval(
    softpdftype: int,
    nl: np.ndarray,
    limi: np.ndarray,
) -> tuple:
    """Extract the lower and upper domain bounds from soft probabilistic data.

    Adapted from ``proba2interval.m`` (BMElib bmeprobalib, Jan 1, 2001).

    Transforms the compact ``(softpdftype, nl, limi, probdens)`` representation
    into a pair of vectors that simply state where the soft PDF is defined.

    Parameters
    ----------
    softpdftype : int
        Type of soft PDF (1 = Histogram, 2 = Linear, 3 = Grid histogram,
        4 = Grid linear).  See ``probasyntax`` in BMElib documentation.
    nl : ndarray of shape (ns, 1)
        Number of interval limits for each soft data point.
    limi : ndarray of shape (ns, l)
        Matrix of interval limits.  ``l == max(nl)`` for types 1/2,
        ``l == 3`` for types 3/4.

    Returns
    -------
    a : ndarray of shape (ns,)
        Lower bounds of the domain where each soft PDF is defined.
    b : ndarray of shape (ns,)
        Upper bounds of the domain where each soft PDF is defined.

    Notes
    -----
    Ported from MATLAB (1-based) to Python (0-based):

    * Types 1/2: ``b[i] = limi[i, nl[i] - 1]`` (last valid column, 0-based).
    * Types 3/4: ``b[i] = limi[i, 2]``.
    """
    nl = np.atleast_2d(np.asarray(nl, dtype=int))
    limi = np.atleast_2d(np.asarray(limi, dtype=float))
    ns = nl.shape[0]
    if ns == 0:
        return np.array([np.nan]), np.array([np.nan])

    a = limi[:, 0].copy()

    if softpdftype in (1, 2):
        # Last limit for row i is at column nl[i,0]-1 (0-based)
        col_idx = nl.ravel() - 1          # 0-based last-limit column per row
        row_idx = np.arange(ns)
        b = limi[row_idx, col_idx]
    else:  # 3 or 4
        b = limi[:, 2].copy()

    return a, b


# =============================================================================
# PDF evaluation
# =============================================================================

def proba2val(
    z: np.ndarray,
    softpdftype: int,
    nl: np.ndarray,
    limi: np.ndarray,
    probdens: np.ndarray,
) -> np.ndarray:
    """Evaluate a probabilistic soft PDF at specified z values.

    Adapted from ``proba2val.m`` (BMElib bmeprobalib, Jan 1, 2001).

    Computes the PDF value ``f_s(z[j])`` for each element of ``z``, where the
    PDF is defined by a single soft datum (``nl`` must have exactly one row).

    Parameters
    ----------
    z : array_like of shape (n,)
        Values at which to evaluate the soft PDF.
    softpdftype : int
        Type of soft PDF:

        * ``1`` – Histogram (piecewise constant)
        * ``2`` – Linear (piecewise linear, i.e. linear interpolation)
        * ``3`` – Grid histogram (uniform-grid piecewise constant)
        * ``4`` – Grid linear (uniform-grid piecewise linear)
    nl : array_like of shape (1, 1) or scalar
        Number of interval limits for the **single** soft datum.
    limi : array_like of shape (1, l)
        Interval limits for the single soft datum (0-based row 0).
    probdens : array_like of shape (1, p)
        Probability-density values for the single soft datum (0-based row 0).

    Returns
    -------
    pdf : ndarray of shape (n,)
        PDF values corresponding to each element of ``z``.

    Notes
    -----
    This function only supports a **single** soft datum (``nl`` with one row),
    matching the original BMElib implementation.  For batch evaluation across
    multiple soft data points, call this function in a loop.

    Piecewise-linear evaluation (type 2) uses ``numpy.interp``, which is
    equivalent to MATLAB's ``interp1`` with zero extrapolation outside the
    domain (``left=0``, ``right=0``).

    Index convention: all indices are 0-based (Python), adapted from the
    original 1-based MATLAB code.
    """
    z_arr = np.asarray(z, dtype=float).ravel()
    n = len(z_arr)
    nl_val = int(np.asarray(nl).ravel()[0])
    limi_1d = np.asarray(limi, dtype=float).ravel()[:nl_val]
    probdens_1d = np.asarray(probdens, dtype=float).ravel()

    # Sort z for efficient interval lookup; track order for later unsort
    sort_idx = np.argsort(z_arr)
    z_sorted = z_arr[sort_idx]
    pdf_sorted = np.zeros(n, dtype=float)

    spt = int(softpdftype)
    if spt == 1:
        # Histogram: piecewise constant – probdens has nl-1 bins
        for il in range(nl_val - 1):
            mask = (limi_1d[il] <= z_sorted) & (z_sorted < limi_1d[il + 1])
            pdf_sorted[mask] = probdens_1d[il]

    elif spt == 2:
        # Linear: piecewise linear – probdens has nl values (one per node)
        pdf_sorted = np.interp(z_sorted, limi_1d, probdens_1d[:nl_val],
                               left=0.0, right=0.0)

    elif spt == 3:
        # Grid histogram: uniform grid, piecewise constant
        step = limi_1d[1]   # limi encodes [start, step, end]
        start = limi_1d[0]
        end   = limi_1d[2]
        mask = (start < z_sorted) & (z_sorted < end)
        if mask.any():
            bin_idx = np.floor((z_sorted[mask] - start) / step).astype(int)
            n_bins = nl_val - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            pdf_sorted[mask] = probdens_1d[bin_idx]

    elif spt == 4:
        # Grid linear: uniform grid, piecewise linear interpolation
        step  = limi_1d[1]
        start = limi_1d[0]
        end   = limi_1d[2]
        limivec = np.arange(start, end + step / 2.0, step)
        mask = (start < z_sorted) & (z_sorted < end)
        if mask.any():
            raw_idx = np.floor((z_sorted[mask] - start) / step).astype(int)
            raw_idx = np.clip(raw_idx, 0, len(probdens_1d) - 2)
            x0 = limivec[raw_idx]
            x1 = limivec[raw_idx + 1]
            denom = x1 - x0
            safe_denom = np.where(np.abs(denom) < 1e-30, 1.0, denom)
            slope = np.where(
                np.abs(denom) < 1e-30,
                0.0,
                (probdens_1d[raw_idx + 1] - probdens_1d[raw_idx]) / safe_denom,
            )
            intercept = probdens_1d[raw_idx] - slope * x0
            pdf_sorted[mask] = intercept + slope * z_sorted[mask]

    # Unsort: place sorted PDF values back in original order
    pdf = np.empty(n, dtype=float)
    pdf[sort_idx] = pdf_sorted
    return pdf.reshape(np.asarray(z).shape)


# =============================================================================
# Soft PDF constructors
# =============================================================================

def probaGaussian(
    zm: np.ndarray,
    zv: np.ndarray,
) -> tuple:
    """Create Gaussian soft probabilistic data at a set of data points.

    Adapted from ``probaGaussian.m`` (BMElib bmeprobalib, Jan 1, 2001).

    Generates a piecewise-linear (softpdftype = 2) representation of a
    Gaussian PDF for each of the ns soft data points.  The PDF at point i
    has mean ``zm[i]`` and variance ``zv[i]``.  Thirteen quantile nodes
    spanning the range [0.0001, 0.9999] are used to describe the shape,
    matching the original BMElib convention.

    Parameters
    ----------
    zm : array_like of shape (ns,) or (ns, 1)
        Mean value at each of the ns soft data points.
    zv : array_like of shape (ns,) or (ns, 1)
        Variance at each of the ns soft data points.

    Returns
    -------
    softpdftype : int
        Always ``2`` (Linear piecewise-linear PDF).
    nl : ndarray of shape (ns, 1), dtype int
        Number of interval limits; ``nl[i, 0] = 13`` for all i.
    limi : ndarray of shape (ns, 13)
        Interval limit nodes for each soft datum.
        ``limi[i, j] = zm[i] + sig[i] * xnorm[j]``, where ``xnorm`` are
        fixed standard-normal quantile points.
    probdens : ndarray of shape (ns, 13)
        Normalized PDF values at the nodes.
        ``probdens[i, j] = N(limi[i,j]; zm[i], zv[i])`` normalised so
        that the trapezoidal area integrates to 1.

    See Also
    --------
    probaStudentT, probaUniform, proba2probdens

    Notes
    -----
    The 13 probability nodes used are::

        [0.0001, 0.001, 0.01, 0.05, 0.3, 0.4, 0.5,
         0.6, 0.7, 0.95, 0.99, 0.999, 0.9999]

    These correspond to the ``gaussinv`` call in the original MATLAB code with
    a standard normal distribution (mean=0, variance=1).
    """
    zm = np.asarray(zm, dtype=float).ravel()
    zv = np.asarray(zv, dtype=float).ravel()
    sig = np.sqrt(zv)         # (ns,)
    ns = len(zm)

    softpdftype = 2
    nl = 13 * np.ones((ns, 1), dtype=int)

    # 13 standard-normal quantile nodes (matching BMElib probaGaussian.m)
    _probs = np.array([0.0001, 0.001, 0.01, 0.05, 0.3,
                       0.4, 0.5, 0.6, 0.7, 0.95,
                       0.99, 0.999, 0.9999])
    xnorm = _norm_dist.ppf(_probs)          # shape (13,)

    # limi[i, j] = zm[i] + sig[i] * xnorm[j]   — (ns, 13)
    limi = sig[:, np.newaxis] * xnorm[np.newaxis, :] + zm[:, np.newaxis]

    # probdens[i, j] = N(xnorm[j]; 0, 1) / sig[i]   — (ns, 13)
    fnorm = _norm_dist.pdf(xnorm)           # standard-normal PDF at nodes
    probdens = (1.0 / sig[:, np.newaxis]) * fnorm[np.newaxis, :]

    # Normalise so that trapezoidal integral = 1 for each row
    diffs = np.diff(limi, axis=1)                                  # (ns, 12)
    areas = 0.5 * (probdens[:, :-1] + probdens[:, 1:]) * diffs    # (ns, 12)
    norm_const = areas.sum(axis=1, keepdims=True)                  # (ns, 1)
    # Guard against zero norm (degenerate distributions)
    norm_const = np.where(norm_const == 0.0, 1.0, norm_const)
    probdens = probdens / norm_const

    return softpdftype, nl, limi, probdens


def probaStudentT(
    zave: np.ndarray,
    zvar: np.ndarray,
    nobvs: np.ndarray,
) -> tuple:
    """Create Student-t soft probabilistic data at a set of data points.

    Adapted from ``probaStudentT.m`` (BMElib bmeprobalib, Jan 1, 2001).

    Generates a piecewise-linear (softpdftype = 2) representation of the
    sampling distribution of the *expected value* (long-run arithmetic mean)
    given a sample of ``nobvs[i]`` observations with sample mean ``zave[i]``
    and sample variance ``zvar[i]``.  The PDF is a rescaled Student-t:

    .. math::

        f_S(m) = \\frac{1}{s_n} t_{n-1}\\!\\left(\\frac{m - \\bar{z}}{s_n}\\right),
        \\quad s_n = \\sqrt{\\hat{\\sigma}^2 / n}

    where :math:`n` = ``nobvs[i]``, :math:`\\hat{\\sigma}^2` = ``zvar[i]``,
    and :math:`t_{n-1}` is the Student-t PDF with :math:`n-1` degrees of
    freedom.  When ``nobvs[i] == 1``, the function falls back to a Gaussian
    (since Student-t with 0 d.o.f. is undefined).

    Parameters
    ----------
    zave : array_like of shape (ns,) or (ns, 1)
        Sample mean at each of the ns soft data points.
    zvar : array_like of shape (ns,) or (ns, 1)
        Sample variance at each of the ns soft data points.
    nobvs : array_like of shape (ns,) or (ns, 1)
        Number of observations at each of the ns soft data points.

    Returns
    -------
    softpdftype : int
        Always ``2`` (Linear piecewise-linear PDF).
    nl : ndarray of shape (ns, 1), dtype int
        Number of interval limits; ``nl[i, 0] = 13`` for all i.
    limi : ndarray of shape (ns, 13)
        Interval limit nodes, scaled to each point's distribution.
    probdens : ndarray of shape (ns, 13)
        Normalised PDF values at each node.

    Notes
    -----
    This function requires ``scipy.stats.t`` (equivalent to MATLAB's
    Statistics Toolbox ``tinv`` / ``tpdf``).

    The 13 probability nodes used (matching the original MATLAB ``probaStudentT.m``)
    for the Student-t case are::

        [0.001, 0.01, 0.02, 0.05, 0.3, 0.4, 0.5, 0.6, 0.7, 0.95, 0.98, 0.99, 0.999]

    and for the standard-normal fallback (``nobvs==1``)::

        [0.00001, 0.001, 0.01, 0.05, 0.3, 0.4, 0.5, 0.6, 0.7, 0.95, 0.99, 0.999, 0.99999]

    Index convention: 0-based (adapted from 1-based MATLAB code).

    See Also
    --------
    probaGaussian, probaUniform, proba2probdens
    """
    zave  = np.asarray(zave,  dtype=float).ravel()
    zvar  = np.asarray(zvar,  dtype=float).ravel()
    nobvs = np.asarray(nobvs, dtype=float).ravel()
    sig   = np.sqrt(zvar)
    ns    = len(zave)

    softpdftype = 2
    nl       = 13 * np.ones((ns, 1), dtype=int)
    limi     = np.full((ns, 13), np.nan)
    probdens = np.full((ns, 13), np.nan)

    _probs_norm = np.array([0.00001, 0.001, 0.01, 0.05, 0.3,
                            0.4, 0.5, 0.6, 0.7, 0.95,
                            0.99, 0.999, 0.99999])
    _probs_t    = np.array([0.001, 0.01, 0.02, 0.05, 0.3,
                            0.4, 0.5, 0.6, 0.7, 0.95,
                            0.98, 0.99, 0.999])

    for i in range(ns):
        n_i = nobvs[i]
        sn  = sig[i] / np.sqrt(max(n_i, 1.0))

        if n_i > 1:
            # Hybrid quantile nodes: geometric mean of t and normal quantiles
            # (matching the sqrt(xnormT * xnorm) * sign(xnorm) formula in MATLAB)
            xnorm_std = _norm_dist.ppf(_probs_norm)
            xnorm_t   = _t_dist.ppf(_probs_t, df=n_i - 1)
            xnorm = np.sqrt(np.abs(xnorm_t * xnorm_std)) * np.sign(xnorm_std)
            x = sn * xnorm + zave[i]
            f = _t_dist.pdf((x - zave[i]) / sn, df=n_i - 1) / sn
        else:
            # Fallback: Gaussian (Student-t with 0 d.o.f. undefined)
            xnorm = _norm_dist.ppf(_probs_norm)
            x = sn * xnorm + zave[i]
            f = _norm_dist.pdf(x, loc=zave[i], scale=sn)

        limi[i, :13]     = x
        probdens[i, :13] = f

        if (i + 1) % 500 == 0:
            print(f"probaStudentT: {i + 1} / {ns} = {100.0 * (i + 1) / ns:.1f}% done")

    # Normalise each row so that trapezoidal integral = 1
    diffs = np.diff(limi, axis=1)
    areas = 0.5 * (probdens[:, :-1] + probdens[:, 1:]) * diffs
    norm_const = areas.sum(axis=1, keepdims=True)
    norm_const = np.where(norm_const == 0.0, 1.0, norm_const)
    probdens = probdens / norm_const

    return softpdftype, nl, limi, probdens


# =============================================================================
# Soft data manipulation utilities
# =============================================================================

def probaoffset(
    softpdftype: int,
    nl: np.ndarray,
    limi: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Shift the interval limits of probabilistic soft data by a per-row offset.

    Adapted from ``probaoffset.m`` (BMElib bmeprobalib, Jan 1, 2001).

    Returns a new ``limi`` matrix where each row i has been shifted by
    ``offset[i]``.  For Grid types (3, 4) only the start and end columns
    (columns 0 and 2) are shifted; the step column (column 1) is unchanged.

    Parameters
    ----------
    softpdftype : int
        Type of soft PDF (1 = Histogram, 2 = Linear, 3 = Grid histogram,
        4 = Grid linear).
    nl : array_like of shape (ns, 1)
        Number of interval limits per soft datum (unused here but kept for
        API consistency with the rest of the ``proba*`` family).
    limi : ndarray of shape (ns, l)
        Original interval limits matrix.
    offset : array_like of shape (ns,) or scalar
        Per-row shift values.  A scalar is broadcast to all rows.

    Returns
    -------
    limioffset : ndarray of shape (ns, l)
        New interval limits matrix with offsets applied.

    Raises
    ------
    ValueError
        If ``softpdftype`` is not 1, 2, 3, or 4.

    Notes
    -----
    For types 1 and 2, the full ``limi`` row is shifted by ``offset``::

        limioffset[i, :] = limi[i, :] + offset[i]

    For types 3 and 4, only columns 0 and 2 (start and end of the grid)
    are shifted; column 1 (grid step) remains unchanged::

        limioffset[i, 0] = limi[i, 0] + offset[i]
        limioffset[i, 2] = limi[i, 2] + offset[i]

    Index convention: 0-based (adapted from 1-based MATLAB code).
    """
    nl     = np.atleast_2d(np.asarray(nl, dtype=int))
    limi   = np.atleast_2d(np.asarray(limi, dtype=float)).copy()
    offset = np.asarray(offset, dtype=float).ravel()
    ns     = nl.shape[0]

    if offset.size == 1:
        offset = np.full(ns, offset[0])

    if softpdftype in (1, 2):
        limioffset = limi + offset[:, np.newaxis]
    elif softpdftype in (3, 4):
        limioffset = limi.copy()
        limioffset[:, 0] += offset
        limioffset[:, 2] += offset
    else:
        raise ValueError("softpdftype must be 1, 2, 3, or 4")

    return limioffset


def probasplit(
    c: np.ndarray,
    softpdftype: int,
    nl: np.ndarray,
    limi: np.ndarray,
    probdens: np.ndarray,
    index: np.ndarray,
) -> tuple:
    """Split probabilistic soft data into two sets by index.

    Adapted from ``probasplit.m`` (BMElib bmeprobalib, Jan 1, 2001).

    Partitions the coordinate matrix ``c`` and the associated soft data into
    two complementary subsets: the rows selected by ``index`` (set 1) and the
    remaining rows (set 2).  Trailing unused columns in ``limi`` and
    ``probdens`` are trimmed to the actual ``max(nl)`` for each subset.

    Parameters
    ----------
    c : ndarray of shape (n, d)
        Coordinates of n points with soft probabilistic data.
    softpdftype : int
        Type of soft PDF (1 = Histogram, 2 = Linear, 3 = Grid histogram,
        4 = Grid linear).
    nl : ndarray of shape (n, 1)
        Number of interval limits per point.
    limi : ndarray of shape (n, l)
        Interval limits matrix.
    probdens : ndarray of shape (n, p)
        Probability-density matrix.
    index : array_like of int
        0-based row indices selecting set 1 from ``c``, ``nl``, ``limi``,
        and ``probdens``.

    Returns
    -------
    c1 : ndarray of shape (ni, d)
        Coordinates of the selected points (subset 1).
    c2 : ndarray of shape (n-ni, d)
        Coordinates of the remaining points (subset 2).
    nl1 : ndarray of shape (ni, 1)
        nl values for subset 1.
    limi1 : ndarray of shape (ni, l1)
        limi values for subset 1, trimmed to ``max(nl1)`` columns
        (for types 1 and 2) or kept as 3 columns (types 3 and 4).
    probdens1 : ndarray of shape (ni, p1)
        probdens values for subset 1, trimmed to ``max(nl1) - 1`` (types
        1, 3) or ``max(nl1)`` (types 2, 4) columns.
    nl2 : ndarray of shape (n-ni, 1)
        nl values for subset 2.
    limi2 : ndarray of shape (n-ni, l2)
        limi values for subset 2.
    probdens2 : ndarray of shape (n-ni, p2)
        probdens values for subset 2.

    Notes
    -----
    The ``index`` argument uses **0-based** row indices (adapted from the
    1-based MATLAB code).
    """
    c = np.atleast_2d(np.asarray(c, dtype=float))
    nl = np.atleast_2d(np.asarray(nl, dtype=int))
    limi = np.atleast_2d(np.asarray(limi, dtype=float))
    probdens = np.atleast_2d(np.asarray(probdens, dtype=float))

    index = np.asarray(index, dtype=int).ravel()
    n = c.shape[0]
    all_idx = np.arange(n)
    complement = np.setdiff1d(all_idx, index, assume_unique=True)

    c1 = c[index]
    c2 = c[complement]
    nl1 = nl[index]
    nl2 = nl[complement]
    limi1_full = limi[index]
    limi2_full = limi[complement]
    probdens1_full = probdens[index]
    probdens2_full = probdens[complement]

    # Trim to the minimum necessary columns for each subset
    def _trim(limi_sub, probdens_sub, nl_sub, spt):
        if nl_sub.size == 0:
            return limi_sub, probdens_sub
        nl_max_sub = int(nl_sub.max())
        if spt in (1, 2):
            limi_sub = limi_sub[:, :nl_max_sub]
        # else keep all 3 columns for grid types
        if spt in (1, 3):
            probdens_sub = probdens_sub[:, :nl_max_sub - 1]
        else:
            probdens_sub = probdens_sub[:, :nl_max_sub]
        return limi_sub, probdens_sub

    limi1, probdens1 = _trim(limi1_full, probdens1_full, nl1, softpdftype)
    limi2, probdens2 = _trim(limi2_full, probdens2_full, nl2, softpdftype)

    return c1, c2, nl1, limi1, probdens1, nl2, limi2, probdens2


def probaneighbours(
    c0: np.ndarray,
    c: np.ndarray,
    softpdftype: int,
    nl: np.ndarray,
    limi: np.ndarray,
    probdens: np.ndarray,
    nmax: int,
    dmax,
) -> tuple:
    """Select a neighbourhood subset of soft probabilistic data around a centre.

    Adapted from ``probaneighbours.m`` (BMElib bmeprobalib, Jan 1, 2001).

    Returns the soft data points in ``c`` that are within Euclidean distance
    ``dmax`` from ``c0``, keeping at most ``nmax`` of the closest ones.

    Parameters
    ----------
    c0 : array_like of shape (1, d)
        Coordinate of the origin (estimation) point.
    c : array_like of shape (n, d)
        Coordinates of the n candidate soft data points.
    softpdftype : int
        Type of soft PDF (1 = Histogram, 2 = Linear, 3 = Grid histogram,
        4 = Grid linear).
    nl : ndarray of shape (n, 1)
        Number of interval limits per candidate point.
    limi : ndarray of shape (n, l)
        Interval limits matrix.
    probdens : ndarray of shape (n, p)
        Probability-density matrix.
    nmax : int
        Maximum number of neighbours to return.
    dmax : float or array_like
        Maximum Euclidean distance.  Pass a scalar for purely spatial or
        temporal data, or a length-3 array ``[ds, dt, dst]`` for
        space-time data (see ``neighbours`` in ``general.neighbours``).

    Returns
    -------
    csub : ndarray of shape (m, d)
        Coordinates of selected neighbours (m ≤ n).
    nlsub : ndarray of shape (m, 1)
        nl values for selected neighbours.
    limisub : ndarray of shape (m, l)
        limi values for selected neighbours.
    probdenssub : ndarray of shape (m, p)
        probdens values for selected neighbours.
    dsub : ndarray of shape (m, 1)
        Distances from ``c0`` to each selected neighbour.
    nsub : int
        Number of selected neighbours (= m).
    index : ndarray of shape (m,)
        0-based indices of selected points within the original ``c`` array.

    Notes
    -----
    Internally this function packs ``[nl, limi, probdens]`` into a single
    matrix, delegates to :func:`stamps.general.neighbours.neighbours` for
    distance filtering, then unpacks the result.

    Index convention: 0-based (adapted from 1-based MATLAB code).
    """
    # Lazy import to avoid circular dependencies
    from ..general.neighbours import neighbours as _neighbours

    c0_arr = np.atleast_2d(np.asarray(c0, dtype=float))
    c_arr  = np.atleast_2d(np.asarray(c, dtype=float))
    nl_arr = np.atleast_2d(np.asarray(nl, dtype=float))
    limi_arr = np.atleast_2d(np.asarray(limi, dtype=float))
    probdens_arr = np.atleast_2d(np.asarray(probdens, dtype=float))
    dmax_arr = np.atleast_1d(np.asarray(dmax, dtype=float))

    # Pack soft data into a single matrix for joint distance filtering
    z_packed = np.hstack([nl_arr, limi_arr, probdens_arr])

    csub, z_sub, dsub, nsub, index = _neighbours(
        c0_arr, c_arr, z_packed, nmax, dmax_arr
    )

    if nsub == 0 or z_sub.size == 0:
        empty_nl  = np.empty((0, 1),                dtype=int)
        empty_l   = np.empty((0, limi_arr.shape[1]),  dtype=float)
        empty_p   = np.empty((0, probdens_arr.shape[1]), dtype=float)
        return (
            np.empty((0, c_arr.shape[1])),
            empty_nl, empty_l, empty_p,
            np.empty((0, 1)),
            0,
            np.empty(0, dtype=int),
        )

    z_sub = np.atleast_2d(z_sub)
    nl_ncols    = nl_arr.shape[1]
    limi_ncols  = limi_arr.shape[1]

    nlsub_raw  = z_sub[:, :nl_ncols]
    limisub_raw = z_sub[:, nl_ncols: nl_ncols + limi_ncols]
    probdenssub_raw = z_sub[:, nl_ncols + limi_ncols:]

    # Re-derive integer nl values and trim to correct column count
    nlsub = np.round(nlsub_raw).astype(int)
    spt = int(softpdftype)

    if spt in (1, 2):
        lsub = int(nlsub.max())
    else:
        lsub = 3

    if spt in (1, 3):
        psub = int(nlsub.max()) - 1
    else:
        psub = int(nlsub.max())

    lsub = min(lsub, limisub_raw.shape[1])
    psub = min(psub, probdenssub_raw.shape[1])

    limisub     = limisub_raw[:, :lsub]
    probdenssub = probdenssub_raw[:, :psub]
    index_arr   = np.asarray(index, dtype=int).ravel()

    return csub, nlsub, limisub, probdenssub, dsub, int(nsub), index_arr


def probacat(
    softpdftype1: int,
    nl1: np.ndarray,
    limi1: np.ndarray,
    probdens1: np.ndarray,
    softpdftype2: int,
    nl2: np.ndarray,
    limi2: np.ndarray,
    probdens2: np.ndarray,
) -> tuple:
    """Concatenate two sets of probabilistic soft data.

    Adapted from ``probacat.m`` (BMElib bmeprobalib, Jan 1, 2001).

    Vertically concatenates two soft-data sets that share the same
    ``softpdftype``.  The resulting ``limi`` and ``probdens`` matrices are
    wide enough to accommodate both sets; any "short" rows are zero-padded on
    the right.

    Parameters
    ----------
    softpdftype1 : int
        Soft PDF type of the first set (1 = Histogram, 2 = Linear,
        3 = Grid histogram, 4 = Grid linear).
    nl1 : ndarray of shape (ns1, 1)
        nl values of the first set.
    limi1 : ndarray of shape (ns1, l1)
        limi values of the first set.
    probdens1 : ndarray of shape (ns1, p1)
        probdens values of the first set.
    softpdftype2 : int
        Soft PDF type of the second set.  **Must equal** ``softpdftype1``.
    nl2 : ndarray of shape (ns2, 1)
        nl values of the second set.
    limi2 : ndarray of shape (ns2, l2)
        limi values of the second set.
    probdens2 : ndarray of shape (ns2, p2)
        probdens values of the second set.

    Returns
    -------
    softpdftype : int
        Common soft PDF type (equals ``softpdftype1``).
    nl : ndarray of shape (ns1 + ns2, 1)
        Concatenated nl values.
    limi : ndarray of shape (ns1 + ns2, max(l1, l2))
        Concatenated limi matrix, zero-padded to uniform width.
    probdens : ndarray of shape (ns1 + ns2, max(p1, p2))
        Concatenated probdens matrix, zero-padded to uniform width.

    Raises
    ------
    ValueError
        If ``softpdftype1 != softpdftype2``.

    Notes
    -----
    Zero-padding the shorter rows does not affect interpretation because
    only the first ``nl[i]`` entries of each row are meaningful, and
    ``nl[i]`` is preserved exactly.

    Index convention: 0-based (adapted from 1-based MATLAB code).
    """
    if int(softpdftype1) != int(softpdftype2):
        raise ValueError(
            f"softpdftype1 ({softpdftype1}) must equal softpdftype2 ({softpdftype2})"
        )
    softpdftype = int(softpdftype1)

    nl1 = np.atleast_2d(np.asarray(nl1, dtype=int))
    nl2 = np.atleast_2d(np.asarray(nl2, dtype=int))
    limi1 = np.atleast_2d(np.asarray(limi1, dtype=float))
    limi2 = np.atleast_2d(np.asarray(limi2, dtype=float))
    probdens1 = np.atleast_2d(np.asarray(probdens1, dtype=float))
    probdens2 = np.atleast_2d(np.asarray(probdens2, dtype=float))

    ns1, ml1 = limi1.shape[0], limi1.shape[1]
    ns2, ml2 = limi2.shape[0], limi2.shape[1]
    mp1 = probdens1.shape[1]
    mp2 = probdens2.shape[1]

    nl = np.vstack([nl1, nl2])

    max_l = max(ml1, ml2)
    limi = np.zeros((ns1 + ns2, max_l), dtype=float)
    limi[:ns1, :ml1] = limi1
    limi[ns1:, :ml2] = limi2

    max_p = max(mp1, mp2)
    probdens = np.zeros((ns1 + ns2, max_p), dtype=float)
    probdens[:ns1, :mp1] = probdens1
    probdens[ns1:, :mp2] = probdens2

    return softpdftype, nl, limi, probdens


def probacombinedupli(
    cs: np.ndarray,
    softpdftype: int,
    nl: np.ndarray,
    limi: np.ndarray,
    probdens: np.ndarray,
    method: str = "mult",
) -> tuple:
    """Combine duplicate soft probabilistic data at coincident locations.

    Adapted from ``probacombinedupli.m`` (BMElib bmeprobalib, Jan 1, 2001).

    Each cluster of soft data points sharing identical space-time coordinates
    is merged into a single soft datum using either multiplicative or additive
    combination of their PDFs:

    * **Multiplicative** (``'mult'``):
      :math:`f_S(x) = A \\cdot f_{S1}(x) \\cdots f_{Sn}(x)`, where *A* is a
      normalisation constant.  Preferred when duplicates represent independent
      measurements of the same quantity.

    * **Additive** (``'add'``):
      :math:`f_S(x) = A \\cdot (f_{S1}(x) + \\cdots + f_{Sn}(x))`.  Used as a
      fallback when multiplicative combination yields a zero PDF.

    * **Auto** (``'auto'``):
      Attempts multiplicative combination; if the result is identically zero
      (no overlap between the individual PDFs), falls back to additive for
      that cluster.

    Parameters
    ----------
    cs : ndarray of shape (ns, d)
        Coordinates of the ns soft data points (may contain duplicates).
    softpdftype : int
        Type of soft PDF.  **Currently only type 2 (Linear) is supported.**
    nl : ndarray of shape (ns, 1)
        Number of interval limits per point.
    limi : ndarray of shape (ns, l)
        Interval limits matrix.
    probdens : ndarray of shape (ns, p)
        Probability-density matrix.
    method : {'mult', 'add', 'auto'}, optional
        Combination method (default ``'mult'``).

    Returns
    -------
    csC : ndarray of shape (nsC, d)
        Coordinates of the nsC unique points after merging (nsC ≤ ns).
    nlC : ndarray of shape (nsC, 1)
        nl values for the merged soft data.
    limiC : ndarray of shape (nsC, l)
        limi values for the merged soft data.
    probdensC : ndarray of shape (nsC, p)
        Normalised probdens values for the merged soft data.
    idxFailed : list of ndarray
        List of 0-based index clusters for which multiplicative combination
        failed (zero combined PDF) and additive was used instead.
        Empty when ``method='add'``.

    Raises
    ------
    ValueError
        If ``softpdftype != 2`` (only Linear type is implemented).
        If ``method`` is not ``'mult'``, ``'add'``, or ``'auto'``.
        If ``cs`` and ``nl`` have a different number of rows.

    Notes
    -----
    * This function is only implemented for **softpdftype = 2** (Linear PDF).
      Contact the BMElib developers if support for other types is required.
    * The maximum number of interval limits per combined datum (``nlmax``) is
      capped at **100** to remain compatible with the FORTRAN routines in
      ``mvnlib``.
    * Index convention: all indices are **0-based** (adapted from the
      1-based MATLAB code).

    See Also
    --------
    probasplit, probacat, proba2val, proba2probdens
    """
    nlmax = 100  # upper bound for nl (FORTRAN mvnlib compatibility)

    cs = np.atleast_2d(np.asarray(cs, dtype=float))
    nl = np.atleast_2d(np.asarray(nl, dtype=int))
    limi = np.atleast_2d(np.asarray(limi, dtype=float))
    probdens = np.atleast_2d(np.asarray(probdens, dtype=float))

    if cs.shape[0] != nl.shape[0]:
        raise ValueError("cs and nl must have the same number of rows")
    if int(softpdftype) != 2:
        raise ValueError(
            "probacombinedupli only supports softpdftype = 2.  "
            "Contact the developers to extend to other types."
        )
    if method not in ("mult", "add", "auto"):
        raise ValueError("method must be 'mult', 'add', or 'auto'")

    ns, d = cs.shape
    idxFailed = []

    if not _isdupli(cs):
        return cs, nl, limi, probdens, idxFailed

    iu, ir = _finddupli(cs)

    # Initialise the output with the non-duplicate rows
    if iu.size > 0:
        csC, _, nl_u, limi_u, probdens_u, _, _, _ = probasplit(
            cs, softpdftype, nl, limi, probdens, iu
        )
        softpdftype_out, nlC, limiC, probdensC = (
            softpdftype, nl_u.copy(), limi_u.copy(), probdens_u.copy()
        )
    else:
        csC         = np.empty((0, d), dtype=float)
        nlC         = np.empty((0, 1), dtype=int)
        limiC       = np.empty((0, limi.shape[1]), dtype=float)
        probdensC   = np.empty((0, probdens.shape[1]), dtype=float)
        softpdftype_out = softpdftype

    # Process each duplicate cluster
    for cluster_idx in ir:
        # Build a merged grid from all points in this cluster
        limiK_list = []
        for j in cluster_idx:
            limiJ = limi[j, :int(nl[j, 0])]
            probdensJ = probdens[j, :int(nl[j, 0])]

            # Extend limi slightly past end-points that have nonzero density
            delta = min(np.sqrt(np.finfo(float).eps), np.diff(limiJ).min() / 100.0)
            if probdensJ[0] > 0:
                limiJ = np.concatenate([[limiJ[0] - delta], limiJ])
            if probdensJ[-1] > 0:
                limiJ = np.concatenate([limiJ, [limiJ[-1] + delta]])
            limiK_list.append(limiJ)

        limiK = np.unique(np.concatenate(limiK_list))
        nlK   = len(limiK)

        # Evaluate each cluster member's PDF on the merged grid
        probdensKJ = np.ones((len(cluster_idx), nlK), dtype=float)
        for row, j in enumerate(cluster_idx):
            nl_j_arr      = np.array([[int(nl[j, 0])]])
            limi_j_arr    = limi[j, :int(nl[j, 0])][np.newaxis, :]
            probdens_j_arr = probdens[j, :int(nl[j, 0])][np.newaxis, :]
            probdensKJ[row, :] = proba2val(
                limiK, softpdftype, nl_j_arr, limi_j_arr, probdens_j_arr
            )

        # Combine
        if method == "mult":
            probdensK = np.prod(probdensKJ, axis=0)
            if probdensK.sum() == 0.0:
                warnings.warn(
                    "One combined soft PDF is zero; switching method to 'auto'.",
                    RuntimeWarning, stacklevel=2,
                )
                method = "auto"
                probdensK = np.sum(probdensKJ, axis=0)
                idxFailed.append(cluster_idx)
        elif method == "add":
            probdensK = np.sum(probdensKJ, axis=0)
        else:  # "auto"
            probdensK = np.prod(probdensKJ, axis=0)
            if probdensK.sum() == 0.0:
                probdensK = np.sum(probdensKJ, axis=0)
                idxFailed.append(cluster_idx)

        # Keep only nodes where this or adjacent nodes have nonzero density
        keep = (probdensK > 0) | (
            np.concatenate([[0], probdensK[:-1]]) > 0
        ) | (
            np.concatenate([probdensK[1:], [0]]) > 0
        )
        limiK    = limiK[keep]
        probdensK = probdensK[keep]
        nlK       = len(limiK)

        # Normalise: proba2probdens returns (nl, limi, norm_probdens, area)
        # limi is unchanged; we only need the normalised probdens (3rd return value)
        _, _, probdensK_norm, _ = proba2probdens(
            softpdftype,
            np.array([[nlK]]),
            limiK[np.newaxis, :],
            probdensK[np.newaxis, :],
        )
        probdensK = probdensK_norm.ravel()[:nlK]
        # limiK and nlK are unchanged by normalisation

        # Downsample if exceeds nlmax
        if nlK > nlmax:
            cdf_vals = np.zeros(nlK)
            heights  = np.diff(limiK)
            area_seg = 0.5 * (probdensK[:-1] + probdensK[1:]) * heights
            cdf_vals[1:] = np.cumsum(area_seg)
            quantile_probs = np.linspace(0.001, 0.999, nlmax)
            new_limi  = np.interp(quantile_probs, cdf_vals, limiK)
            new_pdens = np.array([
                proba2val(
                    np.array([z]), softpdftype,
                    np.array([[nlK]]),
                    limiK[np.newaxis, :],
                    probdensK[np.newaxis, :],
                )[0]
                for z in new_limi
            ])
            limiK    = new_limi
            probdensK = new_pdens
            nlK       = nlmax

        # Append this cluster's combined datum
        new_coord  = cs[cluster_idx[0], :][np.newaxis, :]
        csC = np.vstack([csC, new_coord])

        softpdftype_out, nlC, limiC, probdensC = probacat(
            softpdftype_out, nlC, limiC, probdensC,
            softpdftype, np.array([[nlK]]),
            limiK[np.newaxis, :], probdensK[np.newaxis, :],
        )

    return csC, nlC, limiC, probdensC, idxFailed
