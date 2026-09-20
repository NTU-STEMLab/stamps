# -*- coding: utf-8 -*-
# import math
from six.moves import range
import numpy as np
# import scipy.ndimage.filters.gaussian_filter as gaussian_filter
#import logging
#
#log_manager = logging.getLogger('debug_log')
#log_manager.setLevel(logging.DEBUG)
#fh = logging.FileHandler('debug.txt')
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#fh.setFormatter(formatter)
#log_manager.addHandler(fh)
            
def kernelsmoothing(grid_s,grid_t,grid_z,bs,bt,ktype = "gaussian", DataObj = None):
    """Kernel smoothing of a space-time field on a regular grid (in-place).

    Replaces each non-NaN value in *grid_z* with its local kernel-weighted
    average computed over **all** other grid nodes.  The kernel weight for a
    pair of nodes separated by spatial distance *ds* and temporal lag *dt* is:

    .. math::

        w = K\\!\\left(\\frac{d_s^2}{b_s^2} + \\frac{d_t^2}{b_t^2}\\right)

    where *K* is a Gaussian (``ktype='gaussian'``) or quadratic (Epanechnikov)
    kernel function.

    Parameters
    ----------
    grid_s : np.ndarray, shape (ns, 2)
        Spatial coordinates (easting, northing) of the *ns* grid nodes.
    grid_t : np.ndarray, shape (1, nt)
        Temporal coordinate array of *nt* time steps (2-D row-vector
        convention used internally; the first row is extracted automatically).
    grid_z : np.ndarray, shape (ns, nt)
        Observed or estimated values at each space-time grid node.  NaN
        entries are treated as missing and are excluded from the weighted
        average.
    bs : float
        Spatial bandwidth (same units as *grid_s*).  Controls the spatial
        extent of smoothing.
    bt : float
        Temporal bandwidth (same units as *grid_t*).  Controls the temporal
        extent of smoothing.
    ktype : {'gaussian', 'gau', 'quadratic', 'qua'}, optional
        Kernel type.  ``'gaussian'`` (default) uses an infinite-support
        Gaussian kernel; ``'quadratic'`` uses a compact quadratic (Epanechnikov)
        kernel that is zero beyond the bandwidth.
    DataObj : object or None, optional
        GUI progress-reporter object.  Must expose ``getProgressText()``,
        ``setProgressRange()``, ``setCurrentProgress()``, and
        ``wasProgressCanceled()``.  If ``None`` (default), a no-op
        ``NoUseDataObj`` is created automatically.

    Returns
    -------
    grid_trend : np.ndarray, shape (ns, nt)
        Smoothed field with the same shape as *grid_z*.  NaN positions in
        the original grid are preserved.
    False
        Returned if the user cancels the computation via *DataObj*.

    Notes
    -----
    This function performs kernel smoothing **in place on the entire grid**
    (i.e., each node uses **all** other nodes, including itself).  This is
    distinct from ``kernelsmoothing_est``, which evaluates the smooth at a
    separate set of estimation coordinates.

    The quadratic kernel truncates at unit normalised distance:
    ``w = 1 − (ds²/bs² + dt²/bt²)`` for ``ds²/bs² + dt²/bt² ≤ 1``,
    otherwise ``w = 0``.

    ### Original Reference ###
    grid_s: row by 2 np array
    grid_t: 1 by col np array
    grid_z: row by col np array
    bs: float
    bt: flaot
    ktype: string
    DataObj: for GUI use

    return grid_trend, raw by col np array, or return False if fail
    """
    if not DataObj:
        from ..bme.nousedataobj import NoUseDataObj
        DataObj = NoUseDataObj()
        
    title = DataObj.getProgressText()
        
    #use old code ( grid_t was 1d np array )
    grid_t = grid_t[0]
    
    #determ func
    func = TypeToFunc(ktype)
    
    #create grid_trend
    grid_trend = grid_z.copy()
    
    DataObj.setProgressRange(0,len(grid_s))
    DataObj.setCurrentProgress(0, title + "\n- By Kernel Smoothing...")
    for index_s in range(grid_s.shape[0]):
        if not DataObj.wasProgressCanceled():  
            for index_t in range(grid_t.shape[0]):
                if np.isnan(grid_z[index_s][index_t]):
                    pass
                else:
                    ds = np.sqrt(((grid_s - grid_s[index_s])**2).sum(axis=1))
                    dt = np.abs(grid_t - grid_t[index_t])
                    
#                    index_ss = np.where(ds<=bs)
#                    index_tt = np.where(dt<=bt)
#                    selected_ds = ds[index_ss]
#                    selected_dt = dt[index_tt]
#                    
#                    selected_grid_z = grid_z[np.ix_(index_ss[0],index_tt[0])]
                    
                    ds_nomal = (np.array([ds]).T / bs)**2
                    dt_nomal = (dt / bt)**2
#                    ds_nomal = (np.array([selected_ds]).T / bs)**2
#                    dt_nomal = (selected_dt / bt)**2
                    
                    dr_matrix = ds_nomal + dt_nomal
                    kernel_w = func(dr_matrix)
                    
                    up = kernel_w * grid_z
                    kernel_w[np.isnan(grid_z)] = np.nan
                    down = kernel_w
                    grid_trend[index_s][index_t] = up[~np.isnan(up)].sum() / down[~np.isnan(down)].sum()
            
#                    up = kernel_w * selected_grid_z
#                    kernel_w[np.isnan(selected_grid_z)] = np.nan
#                    down = kernel_w
#                    grid_trend[index_s][index_t] = up[~np.isnan(up)].sum() / down[~np.isnan(down)].sum()
                    
            DataObj.setCurrentProgress(index_s + 1)
        else:
            return False
        
    DataObj.setCurrentProgress(text = title)
    return grid_trend
 
def kernelsmoothing_est(grid_s, grid_t, grid_z, 
                        est_grid_s, est_grid_t,
                        bs, bt, ktype = "gau", DataObj = None):
    """Kernel-smoothed trend estimation at arbitrary space-time locations.

    Evaluates the kernel-weighted average of the training field *grid_z*
    at each estimation point defined by (*est_grid_s*, *est_grid_t*).
    This is the **prediction** counterpart to ``kernelsmoothing``: it
    uses a fixed data grid (grid_s, grid_t, grid_z) and computes smoothed
    estimates at a separate set of target locations.

    The kernel weight for an estimation point ``(es, et)`` and data node
    ``(s, t)`` is:

    .. math::

        w = K\\!\\left(\\frac{\\|\\mathbf{es} - \\mathbf{s}\\|^2}{b_s^2}
                      + \\frac{(et-t)^2}{b_t^2}\\right)

    Parameters
    ----------
    grid_s : np.ndarray, shape (ns, 2)
        Spatial coordinates of the *ns* **data** nodes.
    grid_t : np.ndarray, shape (1, nt)
        Temporal coordinates of *nt* **data** time steps (2-D row-vector).
    grid_z : np.ndarray, shape (ns, nt)
        Observed values at each data space-time node.  NaN marks missing.
    est_grid_s : np.ndarray, shape (nq, 2)
        Spatial coordinates of the *nq* **estimation** nodes.
    est_grid_t : np.ndarray, shape (1, mt)
        Temporal coordinates of *mt* **estimation** time steps.
    bs : float
        Spatial bandwidth (same units as *grid_s*).
    bt : float
        Temporal bandwidth (same units as *grid_t*).
    ktype : {'gau', 'gaussian', 'qua', 'quadratic'}, optional
        Kernel type (default ``'gau'``).
    DataObj : object or None, optional
        GUI progress-reporter (same interface as ``kernelsmoothing``).

    Returns
    -------
    est_grid_trend : np.ndarray, shape (nq, mt)
        Smoothed estimates at each estimation node.  Entries remain NaN if
        all contributing data weights are zero.
    False
        Returned if the user cancels via *DataObj*.

    Notes
    -----
    Unlike ``kernelsmoothing``, the estimation grid does *not* need to
    coincide with the data grid, making this function suitable for
    prediction at unsampled locations (e.g., filling a fine-resolution map
    from sparse observations).
    """
    if not DataObj:
        from ..bme.nousedataobj import NoUseDataObj
        DataObj = NoUseDataObj()

    title = DataObj.getProgressText()
    
    #determ func
    func = TypeToFunc(ktype)
    
    grid_t = grid_t[0]
    est_grid_t = est_grid_t[0]
    
    #create grid_trend
    est_grid_trend = np.empty((est_grid_s.shape[0],est_grid_t.shape[0]))
    est_grid_trend[:] = np.nan
    
    DataObj.setProgressRange(0,len(est_grid_s))
    DataObj.setCurrentProgress(0, title + "\n- By Kernel Smoothing...")
    for index_s in range(len(est_grid_s)):
        if not DataObj.wasProgressCanceled():  
            for index_t in range(len(est_grid_t)):
                ds = np.sqrt(((grid_s - est_grid_s[index_s])**2).sum(axis=1))
                dt = np.abs(grid_t - est_grid_t[index_t])
                
    #            index_ss = np.where(ds<=bs)
    #            index_tt = np.where(dt<=bt)
    #            selected_ds = ds[index_ss]
    #            selected_dt = dt[index_tt]
    #            selected_grid_z = grid_z[np.ix_(index_ss[0],index_tt[0])]
                
                ds_nomal = (np.array([ds]).T / bs)**2
                dt_nomal = (dt / bt)**2
    #            ds_nomal = (np.array([selected_ds]).T / bs)**2
    #            dt_nomal = (selected_dt / bt)**2
                
                dr_matrix = ds_nomal + dt_nomal
                kernel_w = func(dr_matrix)
                
                up = kernel_w * grid_z
                kernel_w[np.isnan(grid_z)] = np.nan
                down = kernel_w
                est_grid_trend[index_s][index_t] = up[~np.isnan(up)].sum() / down[~np.isnan(down)].sum()
            DataObj.setCurrentProgress(index_s + 1)
        else:
            return False
#            up = kernel_w * selected_grid_z
#            kernel_w[np.isnan(selected_grid_z)] = np.nan
#            down = kernel_w
#            est_grid_trend[index_s][index_t] = up[~np.isnan(up)].sum() / down[~np.isnan(down)].sum()
            
#    print 'grid_s=', grid_s
#    print 'grid_t=', grid_t
#    print 'grid_z=', grid_z
#    print 'grid_s_est=', est_grid_s
#    print 'grid_t_est=', est_grid_t
#    print 'grid_tr_est=', est_grid_trend
    return est_grid_trend

#def kernelsmoothing_cv(grid_s,grid_t,grid_z,bs,bt,ktype = "gaussian",sample = None):
#    #determ func
#    func = TypeToFunc(ktype)
#    
#    #create grid_trend
#    grid_trend = grid_z.copy()
#    
##    #sample list 
##    try:
##        sample_index_list = random.sample([[i,j] for i in range(len(grid_s)) for j in range(len(grid_t)) if ~np.isnan(grid_z[i][j])],sample)
##    except ValueError:
##        sample_index_list = [[i,j] for i in range(len(grid_s)) for j in range(len(grid_t)) if ~np.isnan(grid_z[i][j])]
#
#    
##    #get true sample number
##    samplenumber = len(sample_index_list)
##    
##    for (index_s,index_t) in sample_index_list:
#    for index_s in range(len(grid_s)):
#        for index_t in range(len(grid_t)):
#            if np.isnan(grid_z[index_s][index_t]):
#                pass
#            else:
#                ds = np.sqrt(((grid_s - grid_s[index_s])**2).sum(axis=1))
#                dt = np.abs(grid_t - grid_t[index_t])
#                
#                index_ss = np.where(ds<=bs)
#                index_tt = np.where(dt<=bt)
#                selected_ds = ds[index_ss]
#                selected_dt = dt[index_tt]
#                selected_grid_z = grid_z[np.ix_(index_ss[0],index_tt[0])]
#                          
#                ds_nomal = (np.array([selected_ds]).T / bs)**2
#                dt_nomal = (selected_dt / bt)**2
#                
#                dr_matrix = ds_nomal + dt_nomal
#                kernel_w = func(dr_matrix)
#                
#                up = kernel_w * selected_grid_z
#                kernel_w[np.isnan(selected_grid_z)] = np.nan
#                down = kernel_w
#                
#                #abstract self
#                
#                grid_trend[index_s][index_t] = (up[~np.isnan(up)].sum() - grid_z[index_s][index_t]) / (down[~np.isnan(down)].sum() - 1.)
#    error = (grid_z - grid_trend )**2
#    error = error[~np.isnan(error)]
#    samplenumber = error.size
#    square_error = error.sum()
#    mean_square_error = square_error/samplenumber
#    return mean_square_error
                 
def TypeToFunc(kerneltype):
    """Map a kernel-type string to its corresponding weight function.

    Parameters
    ----------
    kerneltype : {'gau', 'gaussian', 'qua', 'quadratic'}
        Short or long name of the kernel.

    Returns
    -------
    func : callable
        One of ``gaussian`` or ``quadratic`` defined in this module.

    Raises
    ------
    KeyError
        If *kerneltype* is not a recognised key.
    """
    dictionary={"gau":gaussian,
                "gaussian":gaussian,
                "qua":quadratic,
                "quadratic":quadratic}
    return dictionary[kerneltype]

def gaussian(dr_matrix):
    """Gaussian kernel weight function.

    Computes element-wise Gaussian kernel weights given a matrix of
    squared normalised distances.

    .. math::

        w = \\exp(-3 \\, r^2)

    where ``r² = ds²/bs² + dt²/bt²`` is the pre-computed squared
    normalised distance passed as *dr_matrix*.

    Parameters
    ----------
    dr_matrix : np.ndarray
        Array of squared normalised distances, all values ≥ 0.

    Returns
    -------
    answer : np.ndarray
        Kernel weights with the same shape as *dr_matrix*.  Values range
        in (0, 1]; they approach 0 but never become exactly zero.

    Notes
    -----
    The factor of 3 in the exponent mirrors the convention used in
    ``exponentialC`` and ``gaussianC`` covariance models, where ``ar`` is
    the **practical range** at which the weight is ≈ 5 %.
    """
    answer = np.exp(-3 * dr_matrix)
    #answer[(dr_matrix > 1)] = 0.0
    return answer

def quadratic(dr_matrix):
    """Quadratic (Epanechnikov) kernel weight function.

    Computes element-wise quadratic kernel weights:

    .. math::

        w = \\max(1 - r^2,\\; 0)

    where ``r² = ds²/bs² + dt²/bt²``.  This kernel has compact support
    (zero weight beyond the bandwidth) and is optimal in the
    mean-integrated-squared-error sense for 1-D density estimation.

    Parameters
    ----------
    dr_matrix : np.ndarray
        Array of squared normalised distances, all values ≥ 0.

    Returns
    -------
    answer : np.ndarray
        Kernel weights with the same shape as *dr_matrix*.  Entries with
        ``dr_matrix > 1`` are set to 0.
    """
    #log_manager.info(str(np.sqrt(dr_matrix)))
    answer =  1 - dr_matrix
    answer[dr_matrix > 1] = 0.0
    #log_manager.info(answer)
    #answer[dr_matrix < 0] = 0.0
    return answer

if __name__ == "__main__":
    
    import time
    func=gaussian

    grid_z = np.array([[1,np.nan,3.,4,5],
                          [5,6,1,7,8],
                          [1,np.nan,4,2,5],
                          [5,2,6,3,1.]])
    #grid_trend = grid_z.copy()
    grid_s=np.array([[1,3.],[1,8],[3,2],[4,1]])
    grid_t=np.array([[1,3,5,7,9.]])
    bs=8
    bt=2
    
    print (kernelsmoothing(grid_s,grid_t,grid_z,bs,bt,ktype = "gaussian"))
    
    print (kernelsmoothing_est(grid_s, grid_t, grid_z, 
                        est_grid_s = grid_s, est_grid_t = grid_t,
                        bs = bs, bt = bt, ktype = "gau"))