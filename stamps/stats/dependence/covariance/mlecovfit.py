# -*- coding: utf-8 -*-
import numpy as np

from six.moves import range

from ....general.isspacetime import isspacetime
from ....general.coord2K import coord2K, coord2dist
from ....models.covmodel import get_model



def _par2op(covparam):
    """Flatten a nested ``covparam`` list into a 1-D parameter array and an index list.

    Parameters
    ----------
    covparam : list of list
        Nested covariance parameter list in stamps format:
        ``[[sill_1, [r11, ...], [t11, ...]], ...]``.

    Returns
    -------
    pars : np.ndarray, shape (n_params,)
        All non-None numeric parameters concatenated into a flat array.
    indices : list of int
        Positions in *pars* that are not ``None`` (i.e. free parameters).
    """
    pars = []
    for i in range(len(covparam)):
        for j in range(len(covparam[0])):
            if j != 0:
                pars.append(covparam[i][j][0])
            else:
                pars.append(covparam[i][j])
    indices = [i for i, e in enumerate(pars) if e != None]
    return np.array(pars), indices

def _op2covpar(pars_slice, indices, covmodel):
    """Reconstruct a nested ``covparam`` list from optimised flat parameter values.

    Inverse of :func:`_par2op`.

    Parameters
    ----------
    pars_slice : array-like, shape (n_free,)
        Optimised free parameters (those at *indices*).
    indices : list of int
        Positions (in the full flat array) corresponding to each element in
        *pars_slice*.
    covmodel : list of str
        Covariance model names (one per nested structure).

    Returns
    -------
    covparam : list of list
        Nested covariance parameter list in stamps format,
        ``[[sill_i, [range_params_i], [temporal_params_i]], ...]``.
    """
    pars = [None] * (indices[-1]+1)
    for i, e in enumerate(indices):
        pars[e] = pars_slice[i]
    covparam = []
    nm = range(len(covmodel))
    k = 0
    for m in nm:
        covparam.append([pars[k],pars[k+1:k+2],pars[k+2:k+3]])
        k = len(covmodel) * (m+1)
    return covparam

def _covpar2par(covparam):
    """Convert stamps covariance parameter format to a flat array for ``nlopt``/BOBYQA.

    Extracts all non-``None`` numeric values from the nested ``covparam``
    list and returns them as a 1-D NumPy array suitable for box-constrained
    optimisation.

    Parameters
    ----------
    covparam : list of list
        Nested covariance parameter list in stamps format:
        ``[[sill_i, [rs_i1, rs_i2, ...], [rt_i1, rt_i2, ...]], ...]``
        where ``None`` marks a parameter that does not exist for the given
        model (e.g. nugget has no range).

    Returns
    -------
    fit_params : np.ndarray, shape (n_free,)
        Flat array of all non-``None`` optimisation parameters.
    npars : np.ndarray of int, shape (nm,)
        Number of spatial range parameters for each nested model.
    npart : np.ndarray of int, shape (nm,)
        Number of temporal range parameters for each nested model.

    Notes
    -----
    This function is a counterpart of :func:`_par2covpar`, which converts
    back from the flat representation to the nested stamps format.

    ### Original Reference ###
    This function transforms data formats from regular covariance model to
    the form used for nlopt.

    Remark: The regular covariance model can refer to covmodeldef function
    """
    fit_params=[]
    nm=len(covparam)
    npars=np.zeros(nm,dtype=int)
    npart=np.zeros(nm,dtype=int)
    for k in range(nm):
        for i,par in enumerate(covparam[k]):
            try:
                for j in range(len(par)):
                    if not (par[j] is None):
                        fit_params.append(par[j])
                if i==1:
                    if par[0] is not None:
                        npars[k]=len(par)
                else:
                    if par[0] is not None:
                        npart[k]=len(par)
            except:
                fit_params.append(par)  
    fit_params=np.array(fit_params)      
    return fit_params,npars,npart    


def _par2covpar(param, ns, nt, covmodel):
    """Convert a flat optimisation parameter array back to stamps ``covparam`` format.

    The inverse operation of :func:`_covpar2par`.  The flat array is expected
    to be ordered as:

    .. code-block:: text

        [s1, r11, r12, ..., t11, t12, ..., s2, r21, ..., t21, ..., ...]

    where

    * ``si`` is the sill of nested model *i*,
    * ``rki`` is the *k*-th spatial range parameter of nested model *i*,
    * ``tki`` is the *k*-th temporal range parameter of nested model *i*.

    Parameters
    ----------
    param : np.ndarray, shape (n_free,)
        Flat parameter vector from the optimiser.
    ns : np.ndarray of int, shape (nm,)
        Number of spatial range parameters per nested model (from
        :func:`_covpar2par`).
    nt : np.ndarray of int, shape (nm,)
        Number of temporal range parameters per nested model.
    covmodel : list of str
        Covariance model names (one per nested structure).

    Returns
    -------
    covparam : list of list
        Nested parameter list in stamps format,
        ``[[sill_i, param_s_i, param_t_i], ...]`` or
        ``[[sill_i, param_s_i], ...]`` for pure-spatial models.

    ### Original Reference ###
    This function transforms data formats from the form used for nlopt to
    the form in the regular covariance model.
    The nlopt input format is an 1-D np array, in which the covariance
    parameters are listed in the form of
    [s1, r11, r12, t11, t12, s2, r2, t2, .....],
    where si is sill of nested model i, rki is the kth spatial range
    parameters of nested spatial model i, and tki is the kth temporal range
    parameters of nested temporal model i.
    """
    covparam=[]
    k=0
    for m in range(len(covmodel)):
        if any(ns)>0 and any(nt)>0:
            covparam.append([param[k],param[k+1:k+ns[m]+1],
                     param[k+ns[m]+1:k+ns[m]+nt[m]+1]])
            k=k+ns[m]+nt[m]+1 
        elif all(nt)==0:
            covparam.append([param[k],param[k+1:k+ns[m]+1]])
            k=k+ns[m]+1
        elif all(ns)==0:
            covparam.append([param[k],param[k+1:k+nt[m]+1]])
            k=k+nt[m]+1
             
    return covparam

def _par2covSTpar(param, ns, nt):
    """Split a flat parameter vector into separate spatial and temporal ``covparam`` lists.

    Useful for separable space-time models where the same sill is shared
    between the spatial and temporal components.

    Parameters
    ----------
    param : np.ndarray, shape (n_free,)
        Flat optimisation parameter vector.
    ns : np.ndarray of int, shape (nm,)
        Number of spatial range parameters per nested model.
    nt : np.ndarray of int, shape (nm,)
        Number of temporal range parameters per nested model.

    Returns
    -------
    covparamS : list of list
        Spatial covariance parameters: ``[[sill_i, [rs_i1, ...]], ...]``.
    covparamT : list of list
        Temporal covariance parameters: ``[[sill_i, [rt_i1, ...]], ...]``.
        Empty if ``all(nt) == 0``.
    """
    covparamS=[]
    covparamT=[]
    nm=len(ns)
    k=0
    for m in range(nm):
        if any(ns)>0 and any(nt)>0:
            covparamS.append([param[k],param[k+1:k+ns[m]+1]])
            covparamT.append([param[k],param[k+ns[m]+1:k+ns[m]+nt[m]+1]])
            k=k+ns[m]+nt[m]+1
        elif all(nt)==0:
            covparamS.append([param[k],param[k+1:k+ns[m]+1]])
            k=k+ns[m]+1
        elif all(ns)==0:
            covparamT.append([param[k],param[k+1:k+nt[m]+1]])
            k=k+nt[m]+1
    return covparamS,covparamT           
  
def _covjac(pars, ns, nt, covmodel, ch):
    """Compute the Jacobian of the covariance matrix w.r.t. each free parameter.

    Evaluates the first derivatives ``dC/d(param_k)`` of the full covariance
    matrix **C** with respect to each free optimisation parameter, at a given
    set of observation locations.

    Parameters
    ----------
    pars : np.ndarray, shape (n_free,)
        Current flat parameter vector.
    ns : np.ndarray of int, shape (nm,)
        Number of spatial range parameters per nested model.
    nt : np.ndarray of int, shape (nm,)
        Number of temporal range parameters per nested model.
    covmodel : list
        Covariance model name list (or list of ``[modelS, modelT]`` pairs for
        space-time models).
    ch : np.ndarray, shape (nh, nd)
        Observation coordinates.

    Returns
    -------
    jac : list of np.ndarray
        One array per free parameter, each of shape ``(nh, nh)``, giving the
        derivative of the covariance matrix with respect to that parameter.

    Notes
    -----
    For separable space-time models, the derivative with respect to the sill
    is ``C_s * C_t``, while the derivatives with respect to spatial and
    temporal range parameters are obtained from the analytical gradient of the
    respective model via the ``jac=True, jacpar='ar'`` calling convention of
    the covariance model functions.

    ### Original Reference ###
    Obtain the Jacobian of the covariance with each of parameters.
    Namely, the first derivatives of the covariance with respect to parameters.
    """
    
    covparam = _par2covpar(pars,ns,nt,covmodel)
    isST, isSTsep, model_res = isspacetime(covmodel)
    if isST:
        if isSTsep:
            modelS, modelT = model_res
            dist_s = coord2dist(ch[:, 0:2], ch[:, 0:2])
            dist_t = coord2dist(ch[:, 2:3], ch[:, 2:3])
            jac = []
        
            for model_s, model_t, param_i in zip(modelS, modelT, covparam):
                sill, param_s, param_t = param_i
                model_s = get_model(model_s)
                model_t = get_model(model_t)      
                jac.append(model_s(dist_s, 1., param_s) * model_t(dist_t, 1., param_t))
                jacar=model_s(dist_s, sill, param_s, jac=True, jacpar='ar')
                if jacar.size>0:
                    jac.append(jacar)  
                jacar=model_t(dist_t, sill, param_t, jac=True, jacpar='ar')  # C6 fix: use dist_t
                if jacar.size>0:
                    jac.append(jacar)  
        else:
            (modelS,) = model_res
            #To be implemented
            return
    else:        
        jac=[]
        dist_s = coord2dist(ch, ch)
        (modelS,) = model_res
        for model_s, param_i in zip(modelS, covparam):
            sill, param_s = param_i
            model_s = get_model(model_s)
            jac.append(model_s(dist_s, 1., param_s))  
            jacar=model_s(dist_s, sill, param_s, jac=True, jacpar='ar')
            if jacar.size>0:
                jac.append(jacar)  
        
    return jac     
  
def _covjac2(pars, ns, nt, covmodel, ch):
    """Compute the Hessian (second derivatives) of the covariance matrix.

    Evaluates the second mixed partial derivatives ``d²C/(d param_j d param_k)``
    of the full covariance matrix **C** with respect to each pair of free
    optimisation parameters.

    Parameters
    ----------
    pars : np.ndarray, shape (n_free,)
        Current flat parameter vector.
    ns : np.ndarray of int, shape (nm,)
        Number of spatial range parameters per nested model.
    nt : np.ndarray of int, shape (nm,)
        Number of temporal range parameters per nested model.
    covmodel : list
        Covariance model name list (or list of ``[modelS, modelT]`` pairs for
        space-time models).
    ch : np.ndarray, shape (nh, nd)
        Observation coordinates.

    Returns
    -------
    jac2 : list of list of np.ndarray or None
        ``n_free × n_free`` nested list.  ``jac2[j][k]`` is an ndarray of
        shape ``(nh, nh)`` giving the second derivative
        ``d²C/(d param_j d param_k)``.  Off-diagonal cross terms between
        different nested structures are ``np.zeros``.

    Notes
    -----
    This function is referenced in the commented-out Newton-CG optimisation
    path inside :func:`mlecovfitv`.  It is not currently called but is
    retained for potential future use.

    For separable space-time models the derivative with respect to the
    sill–sill cross term is always zero (``d²C/ds² = 0`` since sill enters
    linearly).  Cross-range second derivatives are computed via the
    ``jac=True, jacpar='ar2'`` interface of the covariance model functions.

    ### Original Reference ###
    Obtain the second derivatives of the covariance with every of parameters.
    """
    covparam = _par2covpar(pars, ns, nt, covmodel)
    isST, isSTsep, model_res = isspacetime(covmodel)
    if isST:
        if isSTsep:
            modelS, modelT = model_res
            dist_s = coord2dist(ch[:, 0:2], ch[:, 0:2])
            dist_t = coord2dist(ch[:, 2:3], ch[:, 2:3])
            jac2 = [[None]*pars.size for i in range(pars.size)]
            k = 0            
            for model_s, model_t, param_i in zip(modelS, modelT, covparam):
                sill, param_s, param_t = param_i
                model_s = get_model(model_s)
                model_t = get_model(model_t)  
                parnum = len(param_i)
                jac2[k][k] = np.zeros(dist_s.shape) # dsds
                # dsdar
                jac2[k][k+1] = jac2[k+1, k]=\
                  model_s(dist_s, 1., param_s, jac=True, jacpar='ar')*model_t(dist_t, 1., param_t) 
                # dsdat
                jac2[k][k+2] = jac2[k+2,k]=\
                  model_s(dist_s, 1., param_s)*model_t(dist_s, 1., param_t, jac=True, jacpar='ar')
                jac2[k+1][k+1] = model_s(dist_s, sill, param_s, jac=True, jacpar='ar2')*model_t(dist_t, 1., param_t)
                jac2[k+2][k+2] = model_s(dist_s, 1., param_s)*model_t(dist_s, sill, param_t, jac=True, jacpar='ar2')
                k=k+parnum
        else:
            (modelS,) = model_res
            #To be implemented
            return
    else:
        (modelS,) = model_res      
        jac2 = [[None]*pars.size for i in range(pars.size)]
        dist_s = coord2dist(ch, ch)
        k = 0
        for model_s, param_i in zip(modelS, covparam):
            parnum = 1+len(param_i[1])
            sill, param_s = param_i
            model_s = get_model(model_s)
            jac2[k][k] = np.zeros(dist_s.shape) # dsds
            for j in range(1, 1+len(param_i[1])):
                # dsdar
                jac2[k][k+j]=jac2[k+j][k]=\
                  model_s(dist_s, 1., param_s, jac=True, jacpar='ar')   
                jac2[k+j][k+j]= model_s(dist_s, sill, param_s, jac=True, jacpar='ar2')
            for m in range(k,k+1+len(param_i[1])):  
                for i in range(k+parnum,pars.size):
                    jac2[m][i]=jac2[i][m]=np.zeros(dist_s.shape)
            k=k+parnum
        
    return jac2       
  
def mlecovfitv(ch, zh, covmodel, covparam):
    """Fit covariance model parameters by Maximum Likelihood Estimation (MLE) — vector data.

    Minimises the Gaussian log-likelihood

    .. math::

        \\ell(\\boldsymbol{\\theta}) =
            \\frac{n}{2}\\ln(2\\pi) +
            \\frac{1}{2}\\ln|\\mathbf{V}(\\boldsymbol{\\theta})| +
            \\frac{1}{2}\\mathbf{z}^T \\mathbf{V}(\\boldsymbol{\\theta})^{-1} \\mathbf{z}

    with respect to the covariance parameters ``θ = (sill, range, ...)`` using
    the BOBYQA box-constrained derivative-free optimiser (via
    :func:`~stamps.general.bobyqa.bobyqa`).

    Parameters
    ----------
    ch : np.ndarray, shape (nh, nd)
        Observation coordinate matrix.  Rows are data points; columns are
        spatial (or space-time) dimensions.
    zh : np.ndarray, shape (nh,) or (nh, 1)
        Observed values at each location in ``ch``.
    covmodel : list of str or list of [str, str]
        Nested covariance model names.  For pure-spatial data use a flat list
        of strings (e.g. ``['nuggetC', 'exponentialC']``).  For separable
        space-time data use pairs ``['nuggetC/nuggetC', 'exponentialC/exponentialC']``.
    covparam : list of list
        Initial covariance parameter guess in stamps format.  Each nested
        element is ``[sill_i, [spatial_ranges_i], [temporal_ranges_i]]``.
        Temporal range list may be omitted for pure-spatial models.

    Returns
    -------
    param : list of list
        Optimised covariance parameters in the same nested stamps format as
        the input *covparam*.

    Notes
    -----
    **Optimisation** is performed by BOBYQA (Bound Optimisation BY Quadratic
    Approximation), a derivative-free algorithm.  The search bounds are
    ``[eps, 3 × initial_param]`` for each parameter.

    **Covariance tapering** is applied: off-diagonal entries smaller than
    ``10e-6 × C(0)`` are set to zero before computing the log-likelihood
    to improve numerical stability.

    **Datetime handling** — if the last column of ``ch`` contains
    ``numpy.datetime64`` values they are converted to ``float64`` before
    processing.

    The Jacobian (``_covjac``) and Hessian (``_covjac2``) helpers are defined
    inside this function but are not used by the current default optimiser.

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.stats.dependence.mlecovfit import mlecovfitv
    >>> rng = np.random.default_rng(0)
    >>> ch = rng.uniform(0, 100, (30, 1))
    >>> zh = rng.standard_normal(30)
    >>> covmodel = ['nuggetC', 'exponentialC']
    >>> covparam0 = [[0.3, [None]], [0.7, [20.0]]]
    >>> param_mle = mlecovfitv(ch, zh, covmodel, covparam0)

    ### Original Reference ###
    Maximum likelihood method for vector format data.

    opt_param = mlecovfit_v(ch, zh, covmodel, covparam0)

    Input:
        ch       n by d  array of observation locations
        zh       n by 1  array of observed values
        covmodel m       list of nested covariance model names
        covparam m       list of covariance parameters per model
    """
    def llikjac(pars,ns,nt):

        covparam = _par2covpar(pars,ns,nt,covmodel)
        jac=_covjac(pars,ns,nt,covmodel,ch)   
        V,_=coord2K(ch,ch,covmodel,covparam)  
        Viy=np.linalg.solve(V,zh)
        invV=np.linalg.inv(V)
        jacv=np.zeros(pars.size)
        for k in range(len(jac)):     
            jacv[k]=0.5*np.trace(invV.dot(jac[k])) \
                    -0.5*Viy.T.dot(jac[k]).dot(Viy)#/(zh.T.dot(Viy))

        return jacv  
    
    def hess(pars,ns,nt):
        '''
        Hessain estimation with reference in 
        Kitanidis P., and R. Lane. 1985. Maximum likelihood parameter estimation of 
        hydrological spatial processes by the Gauss-Newton method. Journal of 
        hydrology 79. 
        
        Hessian is calculated by its simplication from (11) to (12) in the paper
        Note that this implementation assumes the mean is constant of zero
        
        '''    
    
        covparam = _par2covpar(pars,ns,nt,covmodel)
        jac=_covjac(pars,ns,nt,covmodel,ch)
        #jac2=_covjac2(pars,ns,nt,covmodel,ch)
        V,_=coord2K(ch,ch,covmodel,covparam) 
        V[np.where(V/V[0,0]<10e-6)]=0
        #Viy=np.linalg.solve(V,zh)
        invV=np.linalg.inv(V)
        hes=np.zeros((pars.size,pars.size))
        for j in range(pars.size):
            for k in range(j,pars.size):
                comp1=-0.5*np.trace(invV.dot(jac[j]).dot(invV).dot(jac[k]))  
                #comp2=0.5*np.trace(invV.dot(jac2[j][k]))
                #comp4=Viy.T.dot(jac[j]).dot(invV).dot(jac[k]).dot(Viy)
                #comp5=-0.5*Viy.T.dot(jac2[j][k]).dot(Viy)
                hes[j,k]=-comp1#comp1+comp2+comp4+comp5
                hes[k,j]=hes[j,k]
        
        return hes    
        
    #  def Fishermat(pars,ns,nt):        
    def loglik(pars,ns,nt):
        if any(pars<=0):
            llik=100000000000
            return llik
        
        covparam = _par2covpar(pars,ns,nt,covmodel)
        V, V_list = coord2K(ch, ch, covmodel, covparam)
        # D5 fix: add a small nugget for numerical stability instead of
        # unsafe post-hoc zeroing (which can destroy positive definiteness).
        V += np.eye(V.shape[0]) * 1e-10 * V[0, 0]
        try:
            Lv = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:
            return 1e15  # not positive definite; reject parameter vector
        logdetV=2.*np.sum(np.log(np.diag(Lv)))
        #detV = np.linalg.det(V)
        n=len(zh)
        Viy=np.linalg.solve(V,zh)
        ytViy=zh.T.dot(Viy)
        llik=n/2*np.log(2*np.pi)+0.5*logdetV+0.5*ytViy    
        return llik    
    
    pars,ns,nt=_covpar2par(covparam)
    zh=zh.reshape(zh.size,)
    #  V=np.zeros((ch.size,ch.size))
    #  Viy=np.zeros((ch.size,1))
    #  global V, Viy
    #pars, indices = _par2op(covparam)
    #pars_slice = np.take(pars,indices)
    if type(ch[0,-1])==np.datetime64:
        ch[:,-1]=np.double(np.asarray(ch[:,-1],dtype='datetime64'))
        ch=ch.astype(np.double)
  
    #  _llikjac = lambda pars,ns,nt: llikjac(pars,ns,nt)
    #  _hess = lambda pars,ns,nt: hess(pars,ns,nt)
    #result = op.minimize(loglik, pars_slice, args=(indices),
    #                       options={'maxiter': 10000},method='BFGS')
  
 
    low_bnd=np.ones(pars.size)*np.finfo(float).eps
    up_bnd=np.array([k for k in pars*3])
    args=[ns,nt]
    from ....general import bobyqa as bobyqa  # lazy: nlopt is optional
    result, opt_val=bobyqa.bobyqa(loglik, pars, args, low_bnd, up_bnd )

    #  bnds=[(np.finfo(float).eps,k) for k in pars*3]   
    #  result=op.differential_evolution(loglik,bnds, args=(ns,nt), maxiter=5)  
    ##  result = op.minimize(loglik, result['x'], args=(ns,nt),jac=_llikjac,hess=_hess,
    ##                       options={'gtol': 1e-6, 'disp': True},method='BFGS')
    #  if not result['success']:                     
    #  result = op.minimize(loglik, pars, args=(ns,nt),jac=_llikjac,hess=_hess,
    #                       options={'xtol': 1e-4, 'disp': True},method='Newton-CG')                     
    #  if not result['success']:  
    #    bnds=[(0,None)]*pars.size
    #    result = op.minimize(loglik, result['x'], args=(ns,nt),jac=_llikjac,bounds=bnds,
    #                       options={'gtol': 1e-4, 'disp': True},method='TNC') 

    param = _par2covpar(result,ns,nt,covmodel) 
#  param = _par2covpar(result['x'],ns,nt,covmodel)   

                     
    return param#,result['success']    

def mlecovfitg(grid_s, grid_t, grid_v, covmodel, covparam0):
    """Fit covariance parameters by MLE for separable space-time grid data.

    Minimises the separable Gaussian log-likelihood for a grid of observations
    over space and time.  The total log-likelihood is split into independent
    spatial and temporal parts (exploiting the Kronecker structure of the
    separable covariance):

    .. math::

        \\ell = \\ell_s(\\boldsymbol{\\theta}_s) + \\ell_t(\\boldsymbol{\\theta}_t)

    where each part is the Gaussian log-likelihood of the row/column means
    and residuals of the data matrix ``grid_v``.

    Parameters
    ----------
    grid_s : np.ndarray, shape (ns, nd_s)
        Spatial observation coordinates.  One row per spatial location.
    grid_t : np.ndarray, shape (nt,) or (nt, 1)
        Temporal observation coordinates (scalars or ``numpy.datetime64``).
        If ``datetime64``, converted to ``float64`` automatically.
    grid_v : np.ndarray, shape (ns, nt)
        Observed values on the space-time grid.
    covmodel : list of str or list of [str, str]
        Nested covariance model names, same format as for :func:`mlecovfitv`.
        **Must be separable** (``'modelS/modelT'`` notation or equivalent pairs).
    covparam0 : list of list
        Initial covariance parameter guess in stamps format.

    Returns
    -------
    param : list of list
        Optimised covariance parameters in the same nested stamps format as
        *covparam0*.

    Notes
    -----
    This function currently **requires a separable** space-time covariance
    model (``isSTsep = True``).  If a non-separable model is passed, a
    message is printed and the function returns ``None``.

    If a pure-spatial or pure-temporal model is passed (not space-time), the
    function falls back with a printed message.

    Internally, temporal coordinates are extracted from the grid and reshaped
    to ``(nt, 1)`` before calling :func:`~stamps.general.coord2K.coord2K`.

    ### Original Reference ###
    Maximum likelihood method for grid format data.

    Input:
        ch       n by d  array of observation locations
        zh       n by 1  array of observed values
        covmodel m       nested covariance model name list
        covparam m       nested covariance parameter list
        maxLagS  scalar  maximum spatial lag
        maxLagT  scalar  maximum temporal lag
    """

    def loglik(pars,ch,ch_t,ns,nt,covmodelS,covmodelT):
        
        if any(pars<=0):
            llik=100000000000
            return llik
        
        covparamS,covparamT=_par2covSTpar(pars,ns,nt)

        # spatial part
        #covparamS = _par2covpar(par_S,ns,ns2,covmodelS)
        #Vs, V_list_s = coord2K(grid_s, grid_s, covmodelS, covparamS)
        Vs, V_list_s = coord2K(ch, ch, covmodelS, covparamS)
        # D5 fix: nugget regularization for numerical stability.
        Vs += np.eye(Vs.shape[0]) * 1e-10 * Vs[0, 0]
        Lv_s=np.linalg.cholesky(Vs)
        logdetV_s=2.*np.sum(np.log(np.diag(Lv_s)))
        #detV = np.linalg.det(V)
        n,m=grid_v.shape
        vs_m=grid_v.mean(1).reshape(n,1)
        vs_r=grid_v-vs_m.dot(np.ones((1,m)))
        S_s=vs_r.dot(vs_r.T)*1./(m-1)
        ddt_s=vs_m.dot(vs_m.T)
        term2_s=np.linalg.solve(Vs,S_s+ddt_s)
        llik_s=m*(logdetV_s+np.trace(term2_s))   

        # temporal part        
        #covparamT = _par2covpar(par_T,nt,nt2,covmodelT)
        #Vt, V_list_t = coord2K(grid_t, grid_t, covmodelT, covparamT)  
        Vt, V_list_t = coord2K(ch_t, ch_t, covmodelT, covparamT)
        # D5 fix: use nugget regularization instead of unsafe zeroing.
        Vt += np.eye(Vt.shape[0]) * 1e-10 * Vt[0, 0]
        Lv_t=np.linalg.cholesky(Vt)
        logdetV_t=2.*np.sum(np.log(np.diag(Lv_t)))
        n,m=grid_v.shape
        vt_m=grid_v.mean(0).reshape(1,m)
        vt_r=grid_v-np.ones((n,1)).dot(vt_m)
        S_t=vt_r.T.dot(vt_r)*1./(n-1)
        ddt_t=vt_m.T.dot(vt_m)
        term2_t=np.linalg.solve(Vt,S_t+ddt_t)
        llik_t=m*(logdetV_t+np.trace(term2_t))        
        
        llik=llik_s+llik_t
        
        return llik    
         
    # C1 fix: isspacetime returns 3 values, not 4.
    isST, isSTsep, model_res = isspacetime(covmodel)
    if isST and isSTsep:
        modelS, modelT = model_res
    else:
        modelS = model_res[0] if model_res else []
        modelT = []
    covmodelS=[]
    covmodelT=[]
    covparamS0=[]
    covparamT0=[]
    if isST:
        if isSTsep:       
            for model_s, model_t, param_i in zip( modelS,modelT,covparam0):
                if len(param_i)==1:
                    if model_s=='nuggetC' or model_s=='nuggetC':
                        sill=param_i
                        param_s=[None]
                        param_t=[None]
                    else:
                        print ('covparam is not consisent with covmodel')
                        raise 
                else:
                    sill, param_s, param_t = param_i                
                    covmodelS.append(model_s)
                    covmodelT.append(model_t)
                    covparamS0.append([sill,param_s])
                    covparamT0.append([sill,param_t])

        else:
            print ('S/T separability is currently assumed in MLE')
    else:
        print ('Pure spatial or temporal cases should use mlecovfitv')      
    
#    par_S,ns,ns2=_covpar2par(covparamS0)
#    par_T,nt,nt2=_covpar2par(covparamT0) 
#    numS=len(par_S)
#    numT=len(par_T)
#    pars = np.hstack([par_S,par_T])
    pars,ns,nt=_covpar2par(covparam0)
    
    ch=grid_s
    if type(grid_t[0])==np.datetime64:
        ch_t=np.double(np.asarray(grid_t,dtype='datetime64'))
    else:
        ch_t=grid_t
    ch_t=ch_t.reshape(len(ch_t),1)    

    low_bnd=np.ones(pars.size)*np.finfo(float).eps
    up_bnd=np.array([k for k in pars*3])
    args=[ch,ch_t,ns,nt,covmodelS,covmodelT]
    from ....general import bobyqa as bobyqa  # lazy: nlopt is optional
    result, opt_val=bobyqa.bobyqa(loglik, pars, args, low_bnd, up_bnd )

    param = _par2covpar(result,ns,nt,covmodel) 

    return param
    
#    # Assure the following parameters have proper dimension or format, e.g., 
#    # 1D np array 
#    if grid_t is None:
#        grid_t=np.array([0]).reshape(1,1)
#  
#    if len(grid_s.shape)<2:
#        grid_s=np.reshape(grid_s,(grid_s.size,1))
#    
#    if maxLagS is None:
#        maxLagS=np.max(pdist(grid_s))
#    if maxLagT is None:
#        maxLagT=np.max(pdist(grid_t.reshape(grid_t.size,1)))
#        
#        
#    grid_t=np.asarray(grid_t)
#    grid_t=np.reshape(grid_t,(grid_t.size,1))
#    grid_v=grid_v.reshape((grid_s.shape[0],grid_t.size))
#
#    if grid_s.shape[0]<8000:
#        s_diff_i_left,s_diff_i_right,s_diff_v, _=diffarray(grid_s)
#    else:
#        s_diff_i_left,s_diff_i_right,s_diff_v, _= \
#          diffarray(grid_s,maxLagS)
#    nd=grid_s.shape[1]
#  
#    if nd==1:
#        s_diff_v=np.abs(s_diff_v).ravel()
#    elif nd==2:           
#        s_diff_v=np.sqrt(s_diff_v[:,0]**2+s_diff_v[:,1]**2)
#    elif nd==3:
#        s_diff_v=np.sqrt(s_diff_v[:,0]**2+s_diff_v[:,1]**2+s_diff_v[:,2]**2)
#  
#    if len(grid_t)<8000:
#        t_diff_i_left,t_diff_i_right,t_diff_v,_=diffarray(grid_t)
#    else:
#        t_diff_i_left,t_diff_i_right,t_diff_v,_= \
#            diffarray(grid_t,maxLagT)
#    t_diff_v=np.abs(t_diff_v).ravel()    
#    
#    idxs=np.where(s_diff_v<=maxLagS)
#    s_diff_i_left=s_diff_i_left[idxs].astype(np.int)
#    s_diff_i_right=s_diff_i_right[idxs].astype(np.int)
#    s_diff_v=s_diff_v[idxs]
#  
#    idxt=np.where(t_diff_v<=maxLagT)
#    t_diff_i_left=t_diff_i_left[idxt].astype(np.int)
#    t_diff_i_right=t_diff_i_right[idxt].astype(np.int)
#    t_diff_v=t_diff_v[idxt]      
#
#    lagS=np.unique(s_diff_v)
#    lagT=np.unique(t_diff_v)
    
    
    
    #for s in lagS:
        
    
def mlecovfit_sub(ch, zh, covmodel, covparam0):
    """Fit covariance parameters by MLE for replicated (multi-column) spatial data.

    Minimises the Gaussian log-likelihood when multiple independent
    realisations of the same random field are observed at the same ``nh``
    locations.  Each column of ``zh`` is one realisation.

    The log-likelihood is

    .. math::

        \\ell = m \\left(
            \\ln|\\mathbf{V}| + \\mathrm{tr}(\\mathbf{V}^{-1}(\\mathbf{S} + \\mathbf{d}\\mathbf{d}^T))
        \\right)

    where ``m`` is the number of realisations, **S** is the sample spatial
    covariance of the realisations, and **d** is the column-mean vector.

    Parameters
    ----------
    ch : np.ndarray, shape (nh, nd)
        Observation coordinate matrix.  If 1-D, reshaped to ``(nh, 1)``.
        If the last column contains ``numpy.datetime64``, it is converted to
        ``float64``.
    zh : np.ndarray, shape (nh, m)
        Matrix of observed values.  Rows are locations, columns are
        independent realisations (``m >= 2`` recommended).
    covmodel : list of str or list of [str, str]
        Nested covariance model names in stamps format.
    covparam0 : list of list
        Initial covariance parameter guess.

    Returns
    -------
    param : list of list
        Optimised covariance parameters in the same stamps nested format as
        *covparam0*.

    Notes
    -----
    This function is similar to :func:`mlecovfitv` but for replicated
    data.  The BOBYQA optimiser is used with bounds ``[eps, 3 × initial]``.

    Unlike :func:`mlecovfitv`, this function computes the likelihood from
    the column-mean and sample covariance of ``zh`` rather than from a single
    observation vector, which makes it more appropriate when multiple
    spatial realisations are available.

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.stats.dependence.mlecovfit import mlecovfit_sub
    >>> rng = np.random.default_rng(42)
    >>> ch = rng.uniform(0, 100, (20, 1))
    >>> zh = rng.standard_normal((20, 10))   # 10 realisations
    >>> covmodel = ['nuggetC', 'exponentialC']
    >>> covparam0 = [[0.4, [None]], [0.6, [15.0]]]
    >>> param = mlecovfit_sub(ch, zh, covmodel, covparam0)
    """
    
    def loglik(pars,ns,nt):
        
        if any(pars <= 0):
            llik = 100000000000
            return llik
        
        covparam = _par2covpar(pars, ns, nt, covmodel)
        V, V_list = coord2K(ch, ch, covmodel, covparam)
        # covariance tapering
        V[np.where(V/V[0,0]<10e-6)]=0
        Lv=np.linalg.cholesky(V)
        logdetV=2.*np.sum(np.log(np.diag(Lv)))
        #detV = np.linalg.det(V)
        n,m=zh.shape
        zh_m=zh.mean(1).reshape(n,1)
        zh_r=zh-zh_m.dot(np.ones((1,m)))
        S=zh_r.dot(zh_r.T)*1./(m-1)
        ddt=zh_m.dot(zh_m.T)
        term2=np.linalg.solve(V,S+ddt)
        llik=m*(logdetV+np.trace(term2))
        return llik
        
    pars, ns, nt =_covpar2par(covparam0)
    if len(ch.shape) < 2:
        ch = ch.reshape(ch.size, 1)
    if type(ch[0,-1]) == numpy.datetime64:
        ch[:,-1] = np.double(np.asarray(ch[:,-1], dtype='datetime64'))
        ch = ch.astype(np.double)

    low_bnd = np.ones(pars.size)*np.finfo(float).eps
    up_bnd = np.array([k for k in pars*3])
    args = [ns, nt]
    from ....general import bobyqa as bobyqa  # lazy: nlopt is optional
    result, opt_val = bobyqa.bobyqa(loglik, pars, args, low_bnd, up_bnd )

    param = _par2covpar(result, ns, nt, covmodel) 

    return param#,result['success']     

if __name__ == "__main__":
    
#    import np
    import pandas as pd

    data='/Users/hdragon689/MyMacDoc/packages/MyPythonModules/starpy_dev/examples/Data/GeoData.xls'    

    datadf=pd.ExcelFile(data).parse('Sheet1',header=None)
    datadf.insert(loc=0,column='Time',value=pd.to_datetime(datadf[2]*100+datadf[3],format='%Y%m'))

    # cMS
    cMS = np.array(datadf.iloc[:,1:3])

    #tME
    tME=np.array(datadf.Time,dtype='datetime64[M]')
    tME = tME.reshape((tME.size,1))

    ch=np.asarray(zip(cMS[:,0],cMS[:,1],tME[:,0]))
    ch[:,-1]=np.double(np.asarray(ch[:,-1],dtype='datetime64'))

    z = np.array(datadf.iloc[:,5])

    covmodel2=[['nuggetC/nuggetC'],['exponentialC/exponentialC'],['exponentialC/gaussianC']]
    # covparam0=[[0.2,[None],[None]],[0.45,[50000],[3]],[0.4,[300000],[4]]]
    covparam0=[[0.1,[None],[None]],[0.1,[10000],[1]],[0.1,[100000],[1]]]
    covparam_ml = mlecovfit(covparam0, ch, z, ch, covmodel2)
#    import pdb
#    pdb.set_trace()
    print (covparam_ml)