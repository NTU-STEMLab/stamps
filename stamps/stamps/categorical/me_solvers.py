# -*- coding: utf-8 -*-
"""
stamps.categorical.me_solvers
==============================
Maximum Entropy solver backends for categorical BME.

Contains the five ME solver variants (GIS, IIS, MPL, MLE_reg, ME_dual)
and the orchestrator ``maxentropytable`` that dispatches to them.
"""
import numpy as np
from scipy.optimize import minimize

from ..general.coord2dist import coord2dist
from ..stats.dependence.ptable.fit import probamodel2bitable_separable
from ..stats.dependence.ptable.recenter import ipf_recenter
from ._core import probamodel2bitable


def probamodel2bitable_2d(d1, d2, dmodel1, dmodel2, Pmodel):
    """Bilinear interpolation of a 2-D bivariate probability model.

    Uses successive 1-D ``np.interp`` calls (with clamping) to evaluate
    the 4-D table ``Pmodel[a, b, :, :]`` at a single query point
    ``(d1, d2)``.

    Parameters
    ----------
    d1, d2 : float
        Query distances in each covariate-difference axis.
    dmodel1 : (n1,) array
        Grid for covariate-1 differences.
    dmodel2 : (n2,) array
        Grid for covariate-2 differences.
    Pmodel : (nc, nc, n1, n2) ndarray
        Bivariate probability tables on the 2-D grid.

    Returns
    -------
    Pd : (nc, nc) ndarray
        Interpolated bivariate probability table.
    """
    nc = Pmodel.shape[0]
    dmodel1 = np.asarray(dmodel1, dtype=float)
    dmodel2 = np.asarray(dmodel2, dtype=float)
    Pd = np.zeros((nc, nc))
    for a in range(nc):
        for b in range(nc):
            tmp = np.array([np.interp(d1, dmodel1, Pmodel[a, b, :, j])
                            for j in range(len(dmodel2))])
            Pd[a, b] = np.interp(d2, dmodel2, tmp)
    return Pd


def sumoverallexcepttwo(P,dim1,dim2):
    """
    Sums a multidimensional array along all axes except two.
    
    This is used to calculate a bivariate marginal PDF from a full
    multivariate PDF.

    Parameters
    ----------
    P : np.ndarray
        The full multidimensional probability table, shape ``(nc, nc, ..., nc)``
        with `ndim` dimensions.
    dim1 : int
        The first dimension (axis) *not* to sum over.
    dim2 : int
        The second dimension (axis) *not* to sum over.

    Returns
    -------
    S : np.ndarray
        The resulting 2D bivariate marginal table, shape ``(nc, nc)``.
    """
    ndim=P.ndim

    sumaxis = tuple(np.delete(np.arange(ndim),[dim1,dim2]))
    # S = P.sum(axis = sumaxis, keepdims=True)
    S = P.sum(axis = sumaxis)
    return S

def sumoverallexceptone(P, dim):
    """
    Sums a multidimensional array along all axes except one.

    This is used to calculate a univariate (axis) marginal PDF from a full
    multivariate PDF — the unary-constraint sibling of
    :func:`sumoverallexcepttwo`.

    Parameters
    ----------
    P : np.ndarray
        The full multidimensional probability table, shape ``(nc, ..., nc)``.
    dim : int
        The dimension (axis) *not* to sum over.

    Returns
    -------
    S : np.ndarray
        The resulting 1D marginal, shape ``(nc,)``.
    """
    ndim = P.ndim
    sumaxis = tuple(np.delete(np.arange(ndim), [dim]))
    return P.sum(axis=sumaxis)

def multiplysubtable(P,dim,index,ptheor,pest):
    """
    (Legacy GIS function) Multiplies a subtable by a ratio of scalars.
    
    This is a helper function for the `iterativerescaling_GIS` algorithm.
    It multiplies elements in the full PDF `P` corresponding to a
    specific bivariate marginal by a correction ratio.

    Parameters
    ----------
    P : np.ndarray
        The full multidimensional PDF, shape ``(nc, ..., nc)``.
    dim : tuple or list
        A 2-element list `[dim1, dim2]` specifying the axes of the marginal.
    index : tuple or list
        A 2-element list `[idx1, idx2]` specifying the indices within
        the marginal (which category pair).
    ptheor : float
        The theoretical (target) probability for this category pair.
    pest : float
        The estimated (model) probability for this category pair.

    Returns
    -------
    Pfit : np.ndarray
        A *copy* of `P` with the relevant elements rescaled.
    """

    # Create a list of slices
    slices = [slice(None)] * P.ndim  # Start with all slices
    slices[dim[0]] = index[0]  # Set the first dimension index
    slices[dim[1]] = index[1]  # Set the second dimension index

    # Convert to a tuple
    slices = tuple(slices)
    Pfit = P.copy()
    if ptheor == 0:
        Pfit[slices] = 0
    else:
        Pfit[slices] *= ptheor / pest

    return Pfit

def iterativerescaling_GIS(Pbiv,tol):
    """
    Performs Maximum Entropy fitting using Generalized Iterative Scaling (GIS).
    
    Note: This is a legacy estimator. 'ME_dual' or 'MPL' are generally
    preferred due to speed and stability.

    Parameters
    ----------
    Pbiv : list of lists
        A sparse, `ndim x ndim` nested list. `Pbiv[i][j]` is the
        target bivariate probability table (shape ``(nc, nc)``) for the
        pair (i, j), or `None`/`[]` if the constraint is not active.
    tol : float
        Stopping criterion (maximum absolute difference in `Pfit`
        between iterations).

    Returns
    -------
    Pfit : np.ndarray
        The fitted multivariate probability table, shape ``(nc, ..., nc)``
        with `ndim` dimensions.
    niter : int
        The number of iterations performed.
    """
    ndim, nc = len(Pbiv),len(Pbiv[0][1])
    Pfit=np.ones([nc]*ndim)/(nc**ndim)
    Pfitold=Pfit.copy()

    niter=0
    while True:
        niter=niter+1
        for i in range(ndim):
            for j in range(i+1,ndim):
                Ptheor = Pbiv[i][j] # theoretical value for the bivariate probability
                
                # --- kNN FIX: Skip if this constraint is not active ---
                if Ptheor is None or len(Ptheor) == 0:
                    continue
                # ---
                
                Pest = sumoverallexcepttwo(Pfit, i, j) # integral over the other dims
                
                ## if nan, mutiply ratio equal 1
                with np.errstate(divide='ignore', invalid='ignore'): # Suppress 0/0 warnings
                    rescale_ratio_gis = np.where(~np.isnan(Ptheor), Ptheor / Pest, 1)
                    rescale_ratio_gis[Pest == 0] = 1 # Handle 0/0 case explicitly
                
                # Reshape the 2D array for broadcasting
                expand_shape = [1] * Pfit.ndim
                expand_shape[i] = expand_shape[j] =  nc
                
                # rescale by theoritical value to get maximum entropy
                Pfit*=rescale_ratio_gis.reshape(expand_shape)
                #Pfit*=rescale_ratio_iis.reshape(expand_shape)

        if np.max(abs(Pfit-Pfitold))<tol:
            break
        else:
            Pfitold=Pfit.copy()

    return Pfit,niter

def iterativerescaling_IIS(Pbiv,tol):
    """
    Performs Maximum Entropy fitting using Improved Iterative Scaling (IIS).
    
    Note: This is a legacy estimator. 'ME_dual' or 'MPL' are generally
    preferred due to speed and stability.

    Parameters
    ----------
    Pbiv : list of lists
        A sparse, `ndim x ndim` nested list. `Pbiv[i][j]` is the
        target bivariate probability table (shape ``(nc, nc)``) for the
        pair (i, j), or `None`/`[]` if the constraint is not active.
    tol : float
        Stopping criterion (maximum absolute difference in `Pfit`
        between iterations).

    Returns
    -------
    Pfit : np.ndarray
        The fitted multivariate probability table, shape ``(nc, ..., nc)``
        with `ndim` dimensions.
    niter : int
        The number of iterations performed.
    """
    ndim, nc = len(Pbiv),len(Pbiv[0][1])
    Pfit=np.ones([nc]*ndim)/(nc**ndim)
    Pfitold=Pfit.copy()

    niter=0
    while True:
        niter=niter+1
        for i in range(ndim):
            for j in range(i+1,ndim):
                Ptheor = Pbiv[i][j] # theoretical value for the bivariate probability
                
                # --- kNN FIX: Skip if this constraint is not active ---
                if Ptheor is None or len(Ptheor) == 0:
                    continue
                # ---
                
                Pest = sumoverallexcepttwo(Pfit, i, j) # integral over the other dims
                
                ## if nan, mutiply ratio equal 1
                #rescale_ratio_gis = np.where(~np.isnan(Ptheor), Ptheor / Pest, 1) # GIS
                with np.errstate(divide='ignore', invalid='ignore'): # Suppress 0/0 warnings
                    rescale_ratio_iis = np.where(~np.isnan(Ptheor), Ptheor*(1-Pest)/(1-Ptheor)/Pest, 1) #IIS
                    rescale_ratio_iis[np.isnan(rescale_ratio_iis)] = 1.0 # Handle 0/0 cases
                
                # Reshape the 2D array for broadcasting
                expand_shape = [1] * Pfit.ndim
                expand_shape[i] = expand_shape[j] =  nc
                
                # rescale by theoritical value to get maximum entropy
                #Pfit*=rescale_ratio_gis.reshape(expand_shape)
                Pfit*=rescale_ratio_iis.reshape(expand_shape)

        if np.max(abs(Pfit-Pfitold))<tol:
            break
        else:
            Pfitold=Pfit.copy()

    return Pfit,niter


def likeli_MLE(mu,Pbiv,reg,nbiv,nc): 
    """
    Objective function for regularized Maximum Pseudo-Likelihood (MPL).
    
    Calculates the negative regularized pseudo-log-likelihood. This is the
    objective function minimized by `MLEestimator`. It treats all bivariate
    pairs as independent.

    Parameters
    ----------
    mu : np.ndarray
        1D array of Lagrange multipliers, shape ``(nbiv * nc * nc,)``.
    Pbiv : np.ndarray
        1D array of *all* target bivariate probabilities, flattened.
        Shape ``(nbiv * nc * nc,)``. May contain NaNs for inactive constraints.
    reg : float
        The L2 regularization strength.
    nbiv : int
        The number of active bivariate constraints.
    nc : int
        The number of categories.

    Returns
    -------
    obj : float
        The scalar value of the objective function.
    """
    
    mu_vec = mu.reshape(nbiv,nc*nc)
    
    # Log-partition function for each *independent* bivariate distribution
    # This is log(sum(exp(mu_ij))) for each pair ij
    log_z = np.log(np.sum(np.exp(mu_vec), axis=1))
    z = np.sum(log_z)
    
    #z = np.sum(np.exp(mu))
    # --- FIX: Use nansum to ignore missing constraints ---
    mp = np.nansum(mu*Pbiv)
    # ---
    m2 = np.sum(mu**2)/reg/2
    obj = -mp+z+m2
      
    return obj
      
def likeli_diff_MLE(mu,Pbiv,reg,nbiv,nc):
    """
    Gradient of the regularized Maximum Pseudo-Likelihood objective function.

    Parameters
    ----------
    mu : np.ndarray
        1D array of Lagrange multipliers, shape ``(nbiv * nc * nc,)``.
    Pbiv : np.ndarray
        1D array of *all* target bivariate probabilities, flattened.
        Shape ``(nbiv * nc * nc,)``. May contain NaNs.
    reg : float
        The L2 regularization strength.
    nbiv : int
        The number of active bivariate constraints.
    nc : int
        The number of categories.

    Returns
    -------
    dLdm_nonan : np.ndarray
        The flattened gradient of the objective function, shape ``(nbiv * nc * nc,)``.
    """
    
    mu_vec = mu.reshape(nbiv,nc*nc)
    expmu_vec = np.exp(mu_vec)
    z=np.sum(expmu_vec,1).reshape(nbiv,1)
    z2=np.tile(z,(1,nc*nc))
    
    # P_model_decoupled - P_target + regularization_gradient
    dLdm=(expmu_vec/z2).ravel()-Pbiv+mu/reg
    
    # --- FIX: Set gradient to 0 for missing (nan) constraints ---
    dLdm_nonan = np.nan_to_num(dLdm, nan=0.0)
    # ---
          
    return dLdm_nonan


def MLEestimator(active_constraints, ndim, nc, tol, reg=0.1):
    """
    Fits a MaxEnt PDF using regularized Maximum Pseudo-Likelihood (MPL).
    
    This estimator (mislabeled as MLE) is a composite likelihood method.
    It finds the parameters `mu` for each bivariate marginal *independently*
    and then combines them to form the full PDF, assuming a log-linear model.
    This is fast but ignores higher-order interactions.

    Parameters
    ----------
    active_constraints : list
        A list of tuples, ``[((i, j), Pbiv_table), ...]``.
        - ``(i, j)``: A tuple of the two dimension indices.
        - ``Pbiv_table``: The ``(nc, nc)`` target probability table.
    ndim : int
        The total number of dimensions (e.g., estimation point + neighbors).
    nc : int
        The number of categories.
    tol : float
        The convergence tolerance for the L-BFGS-B optimizer.
    reg : float, optional
        L2 regularization strength (default 0.1).

    Returns
    -------
    Pfit : np.ndarray
        The fitted multivariate probability table, shape ``(nc, ..., nc)``
        with `ndim` dimensions.
    niter : int
        The number of function evaluations from the optimizer.
    """
    
    # --- START LOGIC FIX --- (This part is OK)
    nbiv = len(active_constraints) # O(n*k) or O(n^2)
    mu_0 = np.zeros(nbiv*nc*nc)
    Pbiv_vec = np.zeros([nbiv,nc*nc])
    idx = np.zeros([nbiv, 2], dtype=np.int16)
    
    for k, (pair, Pbiv_table) in enumerate(active_constraints):
        i, j = pair
        if Pbiv_table is None or len(Pbiv_table) == 0:
            Pbiv_ravel = np.full((nc*nc), np.nan) 
        else:
            Pbiv_ravel = Pbiv_table.ravel()
        
        # --- START CONVERGENCE FIX --- (This is OK)
        if not np.all(np.isnan(Pbiv_ravel)):
            eps = 1e-10
            Pbiv_ravel = np.clip(Pbiv_ravel, eps, 1.0 - (eps * nc * nc))
            if np.nansum(Pbiv_ravel) > 0:
                Pbiv_ravel /= np.nansum(Pbiv_ravel)
        # --- END CONVERGENCE FIX ---
            
        Pbiv_vec[k] = Pbiv_ravel
        idx[k,:]=[i,j]
    # --- END LOGIC FIX ---
    
    Pbiv_all=Pbiv_vec.reshape(nbiv*nc*nc)
    
    res = minimize(likeli_MLE, mu_0, method='L-BFGS-B', jac=likeli_diff_MLE
                   ,args=(Pbiv_all, reg, nbiv, nc) 
                   ,options={'disp': False,'ftol':tol}) 
    
    niter = res.nfev
    
    mu_mat = (res.x).reshape(nbiv, nc, nc) # Reshape to (nbiv, nc, nc)
    
    # --- START BROADCASTING FIX ---
    # Create a dictionary to map (i, j) pairs to their mu_k table
    mu_dict = {tuple(idx[k]): mu_mat[k] for k in range(nbiv)}
    
    Pfit = np.zeros([nc]*ndim) # Calculate maximum entropy pdf
    
    # Loop over all possible (i, j) pairs, just like GIS/IIS
    for i in range(ndim):
        for j in range(i + 1, ndim):
            # Check if this pair (i, j) has an active constraint
            if (i, j) in mu_dict:
                mu_k = mu_dict[(i, j)]
                
                # Create a (1, 1, ..., nc, ..., nc, ..., 1) shape
                broadcast_shape = [1] * ndim
                broadcast_shape[i] = nc
                broadcast_shape[j] = nc
                
                # Reshape mu[k] and add to Pfit
                Pfit += mu_k.reshape(broadcast_shape)
    # --- END BROADCASTING FIX ---
    
    # --- FIX: Use stable softmax for final Pfit --- (This is OK)
    max_logit = np.max(Pfit)
    P_exp = np.exp(Pfit - max_logit)
    Pfit = P_exp / np.sum(P_exp)
    # ---

    return Pfit,niter


def likeli_MPL(mu, Pbiv_all, nbiv, nc):
    """
    Objective function for non-regularized Maximum Pseudo-Likelihood (MPL).
    
    Calculates the negative composite log-likelihood. This is the
    objective function minimized by `MPLestimator`. It treats all bivariate
    pairs as independent.

    Parameters
    ----------
    mu : np.ndarray
        1D array of Lagrange multipliers, shape ``(nbiv * nc * nc,)``.
    Pbiv_all : np.ndarray
        1D array of *all* target bivariate probabilities, flattened.
        Shape ``(nbiv * nc * nc,)``. May contain NaNs.
    nbiv : int
        The number of active bivariate constraints.
    nc : int
        The number of categories.

    Returns
    -------
    obj : float
        The scalar value of the objective function.
    """
    mu_mat = mu.reshape(nbiv, nc*nc)
    Pbiv_mat = Pbiv_all.reshape(nbiv, nc*nc)
    
    # Log-partition function for each bivariate distribution
    # This is log(sum(exp(mu_ij))) for each pair ij
    log_z = np.log(np.sum(np.exp(mu_mat), axis=1))
    
    # Sum of dot products between parameters and target probabilities
    # --- FIX: Use nansum to ignore missing constraints ---
    mu_dot_pbiv = np.nansum(mu_mat * Pbiv_mat, axis=1)
    # ---
    
    # The total objective is the sum over all bivariate pairs
    # Note: Scipy.minimize finds a minimum. The ME dual is a min problem
    # likeli_MLE was -mp+z+m2. So MPL should be -mp+z.
    obj = np.sum(log_z - mu_dot_pbiv) 
    
    return obj
      
def likeli_diff_MPL(mu, Pbiv_all, nbiv, nc):
    """
    Gradient of the non-regularized MPL objective function.

    Parameters
    ----------
    mu : np.ndarray
        1D array of Lagrange multipliers, shape ``(nbiv * nc * nc,)``.
    Pbiv_all : np.ndarray
        1D array of *all* target bivariate probabilities, flattened.
        Shape ``(nbiv * nc * nc,)``. May contain NaNs.
    nbiv : int
        The number of active bivariate constraints.
    nc : int
        The number of categories.

    Returns
    -------
    dLdm_nonan : np.ndarray
        The flattened gradient of the objective function, shape ``(nbiv * nc * nc,)``.
    """
    mu_mat = mu.reshape(nbiv, nc*nc)
    Pbiv_mat = Pbiv_all.reshape(nbiv, nc*nc)
    
    # Calculate the model probabilities P_model = exp(mu) / Z
    exp_mu = np.exp(mu_mat)
    z = np.sum(exp_mu, axis=1, keepdims=True)
    P_model = exp_mu / z
    
    # Gradient is the difference between model and target probabilities
    dLdm = (P_model - Pbiv_mat).ravel()
    
    # --- FIX: Set gradient to 0 for missing (nan) constraints ---
    dLdm_nonan = np.nan_to_num(dLdm, nan=0.0)
    # ---
    
    return dLdm_nonan

def MPLestimator(active_constraints, ndim, nc, tol):
    """
    Fits a MaxEnt PDF using non-regularized Maximum Pseudo-Likelihood (MPL).
    
    This estimator is a composite likelihood method. It finds the parameters
    `mu` for each bivariate marginal *independently* and then combines them.
    It is fast but ignores higher-order interactions and can be unstable
    if target probabilities are 0 or 1.

    Parameters
    ----------
    active_constraints : list
        A list of tuples, ``[((i, j), Pbiv_table), ...]``.
        - ``(i, j)``: A tuple of the two dimension indices.
        - ``Pbiv_table``: The ``(nc, nc)`` target probability table.
    ndim : int
        The total number of dimensions (e.g., estimation point + neighbors).
    nc : int
        The number of categories.
    tol : float
        The convergence tolerance for the L-BFGS-B optimizer.

    Returns
    -------
    Pfit : np.ndarray
        The fitted multivariate probability table, shape ``(nc, ..., nc)``
        with `ndim` dimensions.
    niter : int
        The number of function evaluations from the optimizer.
    """

    # --- START LOGIC FIX --- (This part is OK)
    nbiv = len(active_constraints) # O(n*k) or O(n^2)
    mu_0 = np.zeros(nbiv*nc*nc)
    Pbiv_vec = np.zeros([nbiv,nc*nc])
    idx = np.zeros([nbiv, 2], dtype=np.int16)
    
    for k, (pair, Pbiv_table) in enumerate(active_constraints):
        i, j = pair
        if Pbiv_table is None or len(Pbiv_table) == 0:
            Pbiv_ravel = np.full((nc*nc), np.nan)
        else:
            Pbiv_ravel = Pbiv_table.ravel()

        # --- START CONVERGENCE FIX --- (This is OK)
        if not np.all(np.isnan(Pbiv_ravel)):
            eps = 1e-10
            Pbiv_ravel = np.clip(Pbiv_ravel, eps, 1.0 - (eps * nc * nc))
            if np.nansum(Pbiv_ravel) > 0:
                Pbiv_ravel /= np.nansum(Pbiv_ravel)
        # --- END CONVERGENCE FIX ---
            
        Pbiv_vec[k] = Pbiv_ravel
        idx[k,:]=[i,j]
    # --- END LOGIC FIX ---
    
    Pbiv_all=Pbiv_vec.reshape(nbiv*nc*nc)
    
    res = minimize(likeli_MPL, mu_0, method='L-BFGS-B', jac=likeli_diff_MPL
                   ,args=(Pbiv_all, nbiv, nc)
                   ,options={'disp': False,'ftol':tol}) # Use tol
    
    niter = res.nfev
    
    from collections import defaultdict

    mu_mat = (res.x).reshape(nbiv, nc, nc)

    mu_dict = defaultdict(lambda: np.zeros((nc, nc)))
    for k in range(nbiv):
        mu_dict[tuple(idx[k])] += mu_mat[k]

    Pfit = np.zeros([nc] * ndim)
    for (i, j), mu_k in mu_dict.items():
        broadcast_shape = [1] * ndim
        broadcast_shape[i] = nc
        broadcast_shape[j] = nc
        Pfit += mu_k.reshape(broadcast_shape)

    max_logit = np.max(Pfit)
    P_exp = np.exp(Pfit - max_logit)
    Pfit = P_exp / np.sum(P_exp)

    return Pfit, niter

# --- START: New ME_dual functions ---

def _calculate_true_me_pdf(mu, nbiv, nc, ndim, idx, alpha=None):
    """
    Helper function to calculate the full ME PDF from Lagrange multipliers.
    
    This function builds the full, log-linear multivariate PDF table
    `P(X_0, ..., X_n)` from the set of bivariate multipliers `mu` and,
    optionally, unary (first-order) multipliers `alpha`:

        log P(x_0, ..., x_n) = sum_i alpha_i(x_i)
                             + sum_(i,j) mu_ij(x_i, x_j) - log Z

    It uses the log-sum-exp trick for numerical stability.

    When the same ``(i, j)`` pair appears multiple times in *idx* (e.g. from
    multiple constraint layers such as spatial + covariate), their Lagrange
    multiplier matrices are **accumulated** (summed) so that every constraint
    contributes to the log-linear model.

    Parameters
    ----------
    mu : np.ndarray
        1D array of all pairwise Lagrange multipliers, shape ``(nbiv * nc * nc,)``.
    nbiv : int
        Number of bivariate constraints.
    nc : int
        Number of categories.
    ndim : int
        Total number of dimensions in the full PDF.
    idx : np.ndarray
        Array mapping `mu` indices to dimension pairs, shape ``(nbiv, 2)``.
    alpha : np.ndarray or None
        Unary Lagrange multipliers, shape ``(ndim, nc)`` (or flat
        ``(ndim * nc,)``).  ``None`` disables unary potentials
        (backward-compatible pairwise-only model).

    Returns
    -------
    Pfit : np.ndarray
        The calculated full multivariate PDF, shape ``(nc, ..., nc)``.
    """
    from collections import defaultdict

    mu_mat = mu.reshape(nbiv, nc, nc)
    mu_dict = defaultdict(lambda: np.zeros((nc, nc)))
    for k in range(nbiv):
        mu_dict[tuple(idx[k])] += mu_mat[k]

    P_logits = np.zeros([nc] * ndim)

    for (i, j), mu_k in mu_dict.items():
        broadcast_shape = [1] * ndim
        broadcast_shape[i] = nc
        broadcast_shape[j] = nc
        P_logits += mu_k.reshape(broadcast_shape)

    if alpha is not None:
        alpha_mat = np.asarray(alpha, dtype=float).reshape(ndim, nc)
        for i in range(ndim):
            broadcast_shape = [1] * ndim
            broadcast_shape[i] = nc
            P_logits += alpha_mat[i].reshape(broadcast_shape)

    max_logit = np.max(P_logits)
    P_exp = np.exp(P_logits - max_logit)
    Pfit = P_exp / np.sum(P_exp)

    return Pfit

def likeli_ME(theta, Pbiv_all, nbiv, nc, ndim, idx, reg, q_unary=None):
    """
    Objective function for the regularized *true* Maximum Entropy dual problem.
    
    Without unary constraints (``q_unary is None``):

        L(mu) = log(Z) - mu . P_target + ||mu||^2 / (2 reg)

    With unary (first-order) constraints:

        L(mu, alpha) = log(Z) - mu . P_target - alpha . q
                     + (||mu||^2 + ||alpha||^2) / (2 reg)

    This is the function to be minimized. It is numerically stable due
    to the L2 regularization term.

    Parameters
    ----------
    theta : np.ndarray
        1D array of Lagrange multipliers.  Shape ``(nbiv * nc * nc,)`` when
        ``q_unary is None``; otherwise ``(nbiv * nc * nc + ndim * nc,)``
        laid out as ``[mu, alpha]``.
    Pbiv_all : np.ndarray
        1D array of all target bivariate probabilities, flattened.
        Shape ``(nbiv * nc * nc,)``. May contain NaNs.
    nbiv : int
        Number of bivariate constraints.
    nc : int
        Number of categories.
    ndim : int
        Total number of dimensions in the full PDF.
    idx : np.ndarray
        Array mapping `mu` indices to dimension pairs, shape ``(nbiv, 2)``.
    reg : float
        L2 regularization strength.
    q_unary : np.ndarray or None
        Flattened unary marginal targets, shape ``(ndim * nc,)``.
        May contain NaNs for unconstrained axes.  ``None`` disables the
        unary block (backward compatible).

    Returns
    -------
    obj : float
        The scalar value of the dual objective function.
    """
    from collections import defaultdict

    n_mu = nbiv * nc * nc
    if q_unary is None:
        mu = theta
        alpha = None
    else:
        mu = theta[:n_mu]
        alpha = theta[n_mu:]

    mu_mat = mu.reshape(nbiv, nc, nc)

    mu_dict = defaultdict(lambda: np.zeros((nc, nc)))
    for k in range(nbiv):
        mu_dict[tuple(idx[k])] += mu_mat[k]

    P_logits = np.zeros([nc] * ndim)
    for (i, j), mu_k in mu_dict.items():
        broadcast_shape = [1] * ndim
        broadcast_shape[i] = nc
        broadcast_shape[j] = nc
        P_logits += mu_k.reshape(broadcast_shape)

    if alpha is not None:
        alpha_mat = alpha.reshape(ndim, nc)
        for i in range(ndim):
            broadcast_shape = [1] * ndim
            broadcast_shape[i] = nc
            P_logits += alpha_mat[i].reshape(broadcast_shape)

    max_logit = np.max(P_logits)
    log_Z = max_logit + np.log(np.sum(np.exp(P_logits - max_logit)))

    mu_dot_P = np.nansum(mu * Pbiv_all)

    l2_penalty = np.sum(mu**2) / (2.0 * reg)

    obj = log_Z - mu_dot_P + l2_penalty

    if alpha is not None:
        obj -= np.nansum(alpha * q_unary)
        obj += np.sum(alpha**2) / (2.0 * reg)

    return obj

def likeli_diff_ME(theta, Pbiv_all, nbiv, nc, ndim, idx, reg, q_unary=None):
    """
    Gradient of the regularized *true* Maximum Entropy dual problem.
    
    Pairwise block:  grad(L)_mu    = (P_model_ij - P_target_ij) + mu / reg
    Unary block:     grad(L)_alpha = (P_model_i  - q_i)         + alpha / reg
    
    `P_model` marginals are calculated from the *full* ME PDF; the targets
    are the bivariate tables and (optionally) the unary marginals.

    Parameters
    ----------
    theta : np.ndarray
        1D array of Lagrange multipliers (``[mu]`` or ``[mu, alpha]``;
        see :func:`likeli_ME`).
    Pbiv_all : np.ndarray
        1D array of all target bivariate probabilities, flattened.
        Shape ``(nbiv * nc * nc,)``. May contain NaNs.
    nbiv : int
        Number of bivariate constraints.
    nc : int
        Number of categories.
    ndim : int
        Total number of dimensions in the full PDF.
    idx : np.ndarray
        Array mapping `mu` indices to dimension pairs, shape ``(nbiv, 2)``.
    reg : float
        L2 regularization strength.
    q_unary : np.ndarray or None
        Flattened unary marginal targets, shape ``(ndim * nc,)``.
        May contain NaNs for unconstrained axes.

    Returns
    -------
    grad_all : np.ndarray
        The flattened gradient of the objective function, same shape as
        *theta*.
    """
    n_mu = nbiv * nc * nc
    if q_unary is None:
        mu = theta
        alpha = None
    else:
        mu = theta[:n_mu]
        alpha = theta[n_mu:]

    Pbiv_target = Pbiv_all.reshape(nbiv, nc*nc)
    
    # 1. Calculate the full P_model (Pfit) from current parameters
    P_model_full = _calculate_true_me_pdf(mu, nbiv, nc, ndim, idx, alpha=alpha)
    
    # 2. Calculate the model's bivariate marginals (P_model)
    grad_mu = np.zeros(n_mu)
    
    for k in range(nbiv):
        i = idx[k, 0]
        j = idx[k, 1]
        
        # Get the bivariate marginal from the full Pfit
        P_model_k = sumoverallexcepttwo(P_model_full, i, j)
        
        # Get the target probabilities
        P_target_k = Pbiv_target[k, :]
        
        # Gradient for this block is (P_model - P_target)
        grad_k = P_model_k.ravel() - P_target_k
        
        # --- FIX: Set gradient to 0 for missing (nan) constraints ---
        grad_k_nonan = np.nan_to_num(grad_k, nan=0.0)
        grad_mu[k*nc*nc : (k+1)*nc*nc] = grad_k_nonan
        # ---
        
    # --- START CONVERGENCE FIX ---
    # 3. Add the gradient of the L2 regularization term
    grad_mu += mu / reg
    # --- END CONVERGENCE FIX ---

    if alpha is None:
        return grad_mu

    # 4. Unary block gradient: (axis marginal of P_model) - q_i
    q_mat = q_unary.reshape(ndim, nc)
    grad_alpha = np.zeros((ndim, nc))
    for i in range(ndim):
        P_model_i = sumoverallexceptone(P_model_full, i)
        g_i = P_model_i - q_mat[i]
        grad_alpha[i] = np.nan_to_num(g_i, nan=0.0)
    grad_alpha = grad_alpha.ravel() + alpha / reg

    return np.concatenate([grad_mu, grad_alpha])

def ME_dual_estimator(active_constraints, ndim, nc, tol, reg=0.1,
                      unary_constraints=None):
    """
    Fits a MaxEnt PDF by solving the *true* regularized dual problem.
    
    This estimator solves the full, constrained Maximum Entropy problem
    using L-BFGS-B on the regularized dual objective function. It correctly
    handles all dependencies specified in `active_constraints` and,
    optionally, per-axis first-order (unary) marginal constraints.

    Parameters
    ----------
    active_constraints : list
        A list of tuples, ``[((i, j), Pbiv_table), ...]``.
        - ``(i, j)``: A tuple of the two dimension indices.
        - ``Pbiv_table``: The ``(nc, nc)`` target probability table.
    ndim : int
        The total number of dimensions (e.g., estimation point + neighbors).
    nc : int
        The number of categories.
    tol : float
        The convergence tolerance for the L-BFGS-B optimizer.
    reg : float, optional
        L2 regularization strength (default 0.1).
    unary_constraints : np.ndarray or None, optional
        Per-axis first-order marginal targets, shape ``(ndim, nc)``.
        Row ``i`` is the target marginal ``q_i`` for axis ``i``; rows of
        NaN leave that axis unconstrained.  For a *consistent* constraint
        set, the bivariate tables should be IPF-recentered to these
        margins first (see
        :func:`stamps.stamps.stats.dependence.ptable.recenter.ipf_recenter`);
        :func:`maxentropytable` does this automatically.

    Returns
    -------
    Pfit : np.ndarray
        The fitted multivariate probability table, shape ``(nc, ..., nc)``
        with `ndim` dimensions.
    niter : int
        The number of function evaluations from the optimizer.
    """
    
    # --- START LOGIC FIX (This part is OK) ---
    nbiv = len(active_constraints) # O(n*k) or O(n^2)
    Pbiv_vec = np.zeros([nbiv,nc*nc])
    idx = np.zeros([nbiv, 2], dtype=np.int16)
    
    for k, (pair, Pbiv_table) in enumerate(active_constraints):
        i, j = pair
        if Pbiv_table is None or len(Pbiv_table) == 0:
            Pbiv_ravel = np.full((nc*nc), np.nan)
        else:
            Pbiv_ravel = Pbiv_table.ravel()

        # --- START CONVERGENCE FIX (This part is OK) ---
        if not np.all(np.isnan(Pbiv_ravel)):
            eps = 1e-10
            Pbiv_ravel = np.clip(Pbiv_ravel, eps, 1.0 - (eps * nc * nc))
            if np.nansum(Pbiv_ravel) > 0:
                Pbiv_ravel /= np.nansum(Pbiv_ravel)
        # --- END CONVERGENCE FIX ---

        Pbiv_vec[k] = Pbiv_ravel
        idx[k,:] = [i, j]
    # --- END LOGIC FIX ---
    
    Pbiv_all = Pbiv_vec.ravel()

    # --- Unary (first-order) constraint block ---
    if unary_constraints is not None:
        q_unary = np.asarray(unary_constraints, dtype=float)
        if q_unary.shape != (ndim, nc):
            raise ValueError(
                f'unary_constraints must have shape ({ndim}, {nc}), '
                f'got {q_unary.shape}')
        q_unary = q_unary.copy()
        # Normalise each constrained row (clip away degenerate zeros)
        for i in range(ndim):
            if not np.any(np.isnan(q_unary[i])):
                q_unary[i] = np.clip(q_unary[i], 1e-6, None)
                q_unary[i] /= q_unary[i].sum()
        q_unary_flat = q_unary.ravel()
        theta_0 = np.zeros(nbiv*nc*nc + ndim*nc)
    else:
        q_unary_flat = None
        theta_0 = np.zeros(nbiv*nc*nc)

    res = minimize(likeli_ME, theta_0, method='L-BFGS-B', jac=likeli_diff_ME,
                   args=(Pbiv_all, nbiv, nc, ndim, idx, reg, q_unary_flat),
                   options={'disp': False, 'ftol': tol}) 
    
    niter = res.nfev
    
    # --- Final Pfit Calculation ---
    theta_final = res.x
    if unary_constraints is not None:
        n_mu = nbiv * nc * nc
        Pfit = _calculate_true_me_pdf(
            theta_final[:n_mu], nbiv, nc, ndim, idx,
            alpha=theta_final[n_mu:])
    else:
        Pfit = _calculate_true_me_pdf(theta_final, nbiv, nc, ndim, idx)

    return Pfit, niter


def ME_sequential_estimator(spatial_constraints, covariate_layer_constraints,
                            ndim, nc, tol, reg=0.1):
    """Alternating-layer MaxEnt solver: spatial first, then each covariate layer.

    Solves the spatial constraints with :func:`ME_dual_estimator`, then
    sequentially adds each covariate layer's constraints.  The final
    Lagrange multiplier vector is the concatenation of all layers, and one
    last joint fine-tuning pass is run.

    Parameters
    ----------
    spatial_constraints : list
        ``[((i,j), Pbiv_table), ...]`` from the spatial Pmodel.
    covariate_layer_constraints : list of list
        One inner list per covariate layer, each with the same format as
        *spatial_constraints*.
    ndim, nc, tol, reg : same as :func:`ME_dual_estimator`.

    Returns
    -------
    Pfit : ndarray  — fitted ME table.
    niter : int     — total function evaluations.
    """
    total_nfev = 0

    Pfit_sp, nfev_sp = ME_dual_estimator(
        spatial_constraints, ndim, nc, tol, reg=reg)
    total_nfev += nfev_sp

    all_constraints = list(spatial_constraints)
    for layer_constraints in covariate_layer_constraints:
        all_constraints.extend(layer_constraints)
        Pfit_seq, nfev_seq = ME_dual_estimator(
            all_constraints, ndim, nc, tol, reg=reg)
        total_nfev += nfev_seq

    Pfit_final, nfev_final = ME_dual_estimator(
        all_constraints, ndim, nc, tol * 0.1, reg=reg)
    total_nfev += nfev_final

    return Pfit_final, total_nfev

# --- END: New ME_dual functions ---


def maxentropytable(c, dmodel, Pmodel, options=None):
    """
    Estimates the Maximum Entropy multivariate probability table.
    
    This is the core ME solver. It takes a set of estimation and data
    locations, builds a set of bivariate constraints based on a
    probability model, and then solves for the full multivariate PDF
    using a specified estimator.

    Parameters
    ----------
    c : np.ndarray
        Coordinates of *all* locations (estimation + data), shape ``(ndim, d)``.
        Typically, `c[0]` is the estimation point.
    dmodel : np.ndarray
        1D array of distances for `Pmodel`, shape ``(nd,)``.
    Pmodel : np.ndarray
        3D array of bivariate probability tables, shape ``(nc, nc, nd)``.
    options : dict, optional
        Dictionary of options:
        - 'tol' (float): Stopping criterion for estimators. Default 1e-5.
        - 'estimator' (str): 'MPL', 'MLE_reg', 'GIS', 'IIS', 'ME_dual'.
          Default 'MPL'.
        - 'reg' (float): Regularization for 'MLE_reg' and 'ME_dual'.
          Default 1e-7.
        - 'model_structure' (str): 'full' (all pairs, O(n^2)) or 'knn'
          (k-nearest pairs, O(n*k)). Default 'full'.
        - 'k_neighbors' (int): `k` for 'knn' model. Default 5.
        - 'separable_pmodel' (dict): Optional separable combination of 1-D
          Pmodels; same keys as in :func:`MCPcatPdf`.  Ignores the scalar
          ``dmodel`` / ``Pmodel`` arguments for pairwise lookup when set.
        - 'covariate_constraints' (list of dict): Approach B covariate
          constraint layers.  Each dict contains 'name', 'dmodel', 'Pmodel',
          'weight'.  Requires '_cov_point_values' with per-point covariate
          values (set by :func:`BMEcatPdf` automatically).
        - '_cov_point_values' (dict): Mapping from covariate name to
          ``(ndim,)`` array of covariate values at ``[ck0] + neighbours``.
        - 'unary_constraints' (ndarray, (ndim, nc)): Per-axis first-order
          marginal targets ``q_i``.  When present, every bivariate target
          table is IPF-recentered to margins ``(q_i, q_j)`` (preserving the
          global odds-ratio/dependence structure), and matching unary
          constraints are enforced in the ME solve.  Rows of NaN leave the
          corresponding axis unconstrained.  Requires
          ``estimator='ME_dual'``.

    Returns
    -------
    Pfit : np.ndarray
        The fitted multivariate probability table, shape ``(nc, ..., nc)``
        with `ndim` dimensions.
    niter : int
        The number of iterations or function evaluations.
    """
    
    # --- START: Options handling ---
    if options is None:
        options = {}
    
    tol = options.get('tol', 1e-5)
    estimator = options.get('estimator', 'MPL')
    reg = options.get('reg', 1e-7)
    model_structure = options.get('model_structure', 'full')
    k_neighbors = options.get('k_neighbors', 5)
    # --- END: Options handling ---

    sep_me = options.get('separable_pmodel')

    if sep_me is not None:
        comps = sep_me['components']
        ncat = int(comps[0]['Pmodel'].shape[0])
        mode_sep = sep_me.get('mode', 'product')
        pm_sep = np.asarray(sep_me['p_marginal'], dtype=float).ravel()
        pm_sep = np.clip(pm_sep, 1e-15, None)
        pm_sep /= pm_sep.sum()
        dmodels_sep = [co['dmodel'] for co in comps]
        Pmodels_sep = [co['Pmodel'] for co in comps]
        coord_cols_sep = [
            np.asarray(co.get('coord_cols', co.get('dims', [])), dtype=int)
            for co in comps
        ]

        def _pair_dist_vec(i, j):
            dv = np.empty(len(comps), dtype=float)
            for k, cols in enumerate(coord_cols_sep):
                dv[k] = float(coord2dist(c[i:i + 1, cols], c[j:j + 1, cols])[0, 0])
            return dv

        def _norm_biv(Pd):
            if not np.all(np.isnan(Pd)):
                Pd = np.clip(Pd, 0, None)
                Pd_sum = np.sum(Pd)
                if Pd_sum > 0:
                    Pd /= Pd_sum
                else:
                    Pd = np.ones((ncat, ncat)) / (ncat * ncat)
            else:
                Pd = np.ones((ncat, ncat)) / (ncat * ncat)
            return Pd

        def get_prob_pair(i, j):
            dv = _pair_dist_vec(i, j)
            Pd = probamodel2bitable_separable(
                dv, dmodels_sep, Pmodels_sep, pm_sep, mode=mode_sep)
            return _norm_biv(Pd)
    else:
        ncat = Pmodel.shape[0]  # number of catogorical class

    ndim = len(c)           # number of data for probability table

    # --- START: Refactored Constraint Building ---
    
    # This list will hold the active constraints: [ ((i, j), Pbiv_table), ... ]
    active_constraints = []
    
    dist_matrix = coord2dist(c, c)

    if sep_me is None:
        def get_prob(d):
            Pd = probamodel2bitable(d, dmodel, Pmodel)
            # --- FIX: Ensure Pbiv sums to 1 after interpolation ---
            if not np.all(np.isnan(Pd)):
                Pd = np.clip(Pd, 0, None) # Remove any negative props
                Pd_sum = np.sum(Pd)
                if Pd_sum > 0:
                    Pd /= Pd_sum
                else:
                    # If sum is 0, return a valid (uniform) table to avoid nans
                    Pd = np.ones((ncat, ncat)) / (ncat * ncat)
            else:
                Pd = np.ones((ncat, ncat)) / (ncat * ncat) # Handle NaN case
            return Pd
        
    added_edges = set()

    if model_structure == 'full':
        for i in range(ndim):
            for j in range(i + 1, ndim):
                if sep_me is None:
                    d = dist_matrix[i, j]
                    Pbiv_table = get_prob(d)
                else:
                    Pbiv_table = get_prob_pair(i, j)
                active_constraints.append( ((i, j), Pbiv_table) )
                added_edges.add((i, j))

    elif model_structure == 'knn': 
        for i in range(ndim):
            k_actual = min(k_neighbors, ndim - 1)
            if k_actual <= 0: continue

            # We need the k_actual + 1 smallest elements (self + k neighbors)
            k_to_find = k_actual + 1
            
            if k_to_find >= ndim:
                # This happens if k_neighbors >= ndim-1. We just take all points.
                neighbor_indices = np.arange(ndim)
            else:
                # Partition the array. The indices of the k_to_find smallest
                # elements will be in the slice [0 : k_to_find].
                idx_partitioned = np.argpartition(dist_matrix[i], k_to_find)
                neighbor_indices = idx_partitioned[0 : k_to_find]

            # Now, loop through the k_actual+1 potential neighbors
            for j in neighbor_indices:
                if i == j:
                    continue # Explicitly skip self
                
                # Add the edge, ensuring i < j
                edge = tuple(sorted((i, j))) 
                added_edges.add(edge)
                
        # --- END: ROBUST kNN BUG FIX ---
            
        # Now create the constraint list from the unique edges
        for (i, j) in added_edges:
            if sep_me is None:
                d = dist_matrix[i, j]
                Pbiv_table = get_prob(d)
            else:
                Pbiv_table = get_prob_pair(i, j)
            active_constraints.append( ((i, j), Pbiv_table) )
            
    else:
        raise ValueError("model_structure must be 'full' or 'knn'")

    # --- END: Refactored Constraint Building ---

    # --- Covariate constraint layers (Approach B) ---
    spatial_constraints = list(active_constraints)
    covariate_layer_list = []

    cov_layers = options.get('covariate_constraints')
    _cov_point_values = options.get('_cov_point_values')
    if cov_layers is not None and _cov_point_values is not None:
        for layer in cov_layers:
            if layer.get('type') == 'interaction':
                # --- 2-D interaction constraint ---
                cov1_vals = np.asarray(
                    _cov_point_values[layer['cov1_name']], dtype=float)
                cov2_vals = np.asarray(
                    _cov_point_values[layer['cov2_name']], dtype=float)
                cov_dmodel1 = np.asarray(layer['dmodel1'], dtype=float)
                cov_dmodel2 = np.asarray(layer['dmodel2'], dtype=float)
                cov_Pmodel_2d = np.asarray(layer['Pmodel'], dtype=float)
                cov_nc = cov_Pmodel_2d.shape[0]
                alpha = float(layer.get('weight', 1.0))

                this_layer = []
                edges_iter = (
                    ((i, j) for i in range(ndim) for j in range(i + 1, ndim))
                    if model_structure == 'full' else added_edges
                )
                for (i, j) in edges_iter:
                    d1 = abs(float(cov1_vals[i]) - float(cov1_vals[j]))
                    d2 = abs(float(cov2_vals[i]) - float(cov2_vals[j]))
                    Pbiv_cov = probamodel2bitable_2d(
                        d1, d2, cov_dmodel1, cov_dmodel2, cov_Pmodel_2d)
                    Pbiv_cov = np.clip(Pbiv_cov, 0, None)
                    s = Pbiv_cov.sum()
                    if s > 0:
                        Pbiv_cov /= s
                    else:
                        Pbiv_cov = np.ones((cov_nc, cov_nc)) / (cov_nc * cov_nc)
                    if alpha != 1.0:
                        Pbiv_cov = alpha * Pbiv_cov + (1.0 - alpha) * (
                            np.ones_like(Pbiv_cov) / (cov_nc * cov_nc))
                        Pbiv_cov /= Pbiv_cov.sum()
                    this_layer.append(((i, j), Pbiv_cov))

                covariate_layer_list.append(this_layer)
                active_constraints.extend(this_layer)
            else:
                # --- 1-D covariate constraint (existing logic) ---
                cov_dmodel = np.asarray(layer['dmodel'], dtype=float)
                cov_Pmodel = np.asarray(layer['Pmodel'], dtype=float)
                cov_nc = cov_Pmodel.shape[0]
                alpha = float(layer.get('weight', 1.0))
                cov_vals = np.asarray(
                    _cov_point_values[layer['name']], dtype=float)

                def _cov_biv(d_cov, cov_dm=cov_dmodel,
                             cov_Pm=cov_Pmodel, _nc=cov_nc):
                    Pd = probamodel2bitable(d_cov, cov_dm, cov_Pm)
                    if not np.all(np.isnan(Pd)):
                        Pd = np.clip(Pd, 0, None)
                        s = Pd.sum()
                        if s > 0:
                            Pd /= s
                        else:
                            Pd = np.ones((_nc, _nc)) / (_nc * _nc)
                    else:
                        Pd = np.ones((_nc, _nc)) / (_nc * _nc)
                    return Pd

                this_layer = []
                edges_iter = (
                    ((i, j) for i in range(ndim) for j in range(i + 1, ndim))
                    if model_structure == 'full' else added_edges
                )
                for (i, j) in edges_iter:
                    d_cov = abs(cov_vals[i] - cov_vals[j])
                    Pbiv_cov = _cov_biv(d_cov)
                    if alpha != 1.0:
                        Pbiv_cov = alpha * Pbiv_cov + (1.0 - alpha) * (
                            np.ones_like(Pbiv_cov) / (cov_nc * cov_nc))
                        Pbiv_cov /= Pbiv_cov.sum()
                    this_layer.append(((i, j), Pbiv_cov))

                covariate_layer_list.append(this_layer)
                active_constraints.extend(this_layer)
    # --- END: Covariate constraint layers ---

    has_cov_layers = len(covariate_layer_list) > 0
    cov_solver = options.get('covariate_solver', 'joint')

    # --- First-order (unary) constraints: IPF-recenter bivariate targets ---
    # For a consistent constraint set, every pairwise target P(a, b | d_ij)
    # must have margins (q_i, q_j); otherwise the pairwise constraints
    # (margins = global p) contradict the unary constraints and the
    # regularized dual returns an uncontrolled compromise.  IPF-recentering
    # transports the global odds ratios to the local first-order level.
    q_unary = options.get('unary_constraints')
    if q_unary is not None:
        q_unary = np.asarray(q_unary, dtype=float)
        if q_unary.shape != (ndim, ncat):
            raise ValueError(
                f"options['unary_constraints'] must have shape "
                f"({ndim}, {ncat}), got {q_unary.shape}")
        if estimator != 'ME_dual':
            raise ValueError(
                "options['unary_constraints'] requires estimator='ME_dual' "
                f"(got '{estimator}').")

        def _recenter(pair, Ptab):
            i, j = pair
            if Ptab is None or np.all(np.isnan(Ptab)):
                return Ptab
            # NaN rows -> keep that axis at the table's own margin
            q_i = q_unary[i] if not np.any(np.isnan(q_unary[i])) \
                else Ptab.sum(axis=1)
            q_j = q_unary[j] if not np.any(np.isnan(q_unary[j])) \
                else Ptab.sum(axis=0)
            return ipf_recenter(Ptab, q_i, q_j)

        active_constraints = [
            (pair, _recenter(pair, Ptab))
            for (pair, Ptab) in active_constraints
        ]
        spatial_constraints = active_constraints[:len(spatial_constraints)]
    # --- END first-order constraints ---

    #### Estimate the maximum entropy table
    if estimator == 'GIS':
        Pbiv = [[[] for _ in range(ndim)] for _ in range(ndim)]
        for (i, j), Pbiv_table in active_constraints:
            Pbiv[i][j] = Pbiv_table
        Pfit, niter = iterativerescaling_GIS(Pbiv, tol)

    elif estimator == 'IIS':
        Pbiv = [[[] for _ in range(ndim)] for _ in range(ndim)]
        for (i, j), Pbiv_table in active_constraints:
            Pbiv[i][j] = Pbiv_table
        Pfit, niter = iterativerescaling_IIS(Pbiv, tol)

    elif estimator == 'ME_dual':
        if has_cov_layers and cov_solver == 'sequential' and q_unary is None:
            Pfit, niter = ME_sequential_estimator(
                spatial_constraints, covariate_layer_list,
                ndim, ncat, tol, reg=reg)
        else:
            try:
                Pfit, niter = ME_dual_estimator(
                    active_constraints, ndim, ncat, tol, reg=reg,
                    unary_constraints=q_unary)
            except Exception:
                if has_cov_layers and q_unary is None:
                    Pfit, niter = ME_sequential_estimator(
                        spatial_constraints, covariate_layer_list,
                        ndim, ncat, tol, reg=reg)
                else:
                    raise

    elif estimator == 'MLE_reg':
        Pfit, niter = MLEestimator(active_constraints, ndim, ncat, tol, reg=reg)

    elif estimator == 'MPL':
        Pfit, niter = MPLestimator(active_constraints, ndim, ncat, tol)

    else:
        raise ValueError(
            f"Unknown estimator: {estimator}. "
            "Choose from 'GIS', 'IIS', 'MPL', 'MLE_reg', or 'ME_dual'."
        )

    return Pfit,niter
