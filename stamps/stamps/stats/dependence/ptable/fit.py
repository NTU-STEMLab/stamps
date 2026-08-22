import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def probamodel2bitable(d, dmodel, Pmodel):
    """Build a bivariate probability table at distance *d* by linear interpolation.

    Distances outside ``[dmodel[0], dmodel[-1]]`` are **silently clamped** to
    the nearest tabulated value.  For the upper bound this is physically correct:
    the joint table converges to ``p(c) × p(c')`` (independence) as d → ∞, and
    ``dmodel[-1]`` is chosen to be at or beyond that asymptote by ``auto_dmax``.
    Clamping prevents ``ValueError`` when pairwise distances among selected
    neighbours slightly exceed the model range.

    Parameters
    ----------
    d : float
        Query distance.
    dmodel : (nd,) array_like
        Tabulated distances.
    Pmodel : (nc, nc, nd) ndarray
        Bivariate probability tables at each tabulated distance.

    Returns
    -------
    Pd : (nc, nc) ndarray
        Interpolated bivariate probability table at distance *d*.
    """
    nc = Pmodel.shape[0]
    Pd = np.zeros((nc, nc))
    dmodel = np.asarray(dmodel, dtype=float)
    for k in range(nc):
        for l in range(nc):
            # np.interp clamps d to [dmodel[0], dmodel[-1]] without error
            Pd[k, l] = np.interp(d, dmodel, Pmodel[k, l])
    return Pd


def probamodel2bitable_separable(d_components, dmodels, Pmodels, p_marginal, mode='product'):
    """Combine 1-D Pmodels along independent axes (3-D = 2 components, 4-D = 3, …).

    **Product mode (default)** — Markov-chain composition
    :math:`P = P_1 D^{-1} P_2 D^{-1} \\cdots` with :math:`D=\\mathrm{diag}(p)`.

    **Additive mode** — :math:`P = \\sum_k P_k - (K-1)\\,p p^{\\top}`,
    then clip and row-renormalise to :math:`p_a`.

    Parameters
    ----------
    d_components : (K,) array_like
        Distance per axis (e.g. horizontal, vertical, time).
    dmodels, Pmodels : list of length K
        ``(nd,)`` grids and ``(nc, nc, nd)`` fitted tables per axis.
    p_marginal : (nc,) array_like
        Category proportions (Pmodel row order).
    mode : {'product', 'additive'}

    Returns
    -------
    Pd : (nc, nc) ndarray
        Joint table :math:`P(a\\text{ at origin}, b\\text{ at neighbour})`.
    """
    d_components = np.asarray(d_components, dtype=float).ravel()
    K = int(d_components.size)
    if len(dmodels) != K or len(Pmodels) != K:
        raise ValueError(
            'd_components, dmodels, and Pmodels must have the same length '
            f'(got {K}, {len(dmodels)}, {len(Pmodels)})'
        )
    p = np.asarray(p_marginal, dtype=float).ravel()
    if p.size == 0:
        raise ValueError('p_marginal is empty')
    p = np.clip(p, 1e-15, None)
    p = p / p.sum()
    nc = p.size

    for j, Pm in enumerate(Pmodels):
        if Pm.shape[0] != nc or Pm.shape[1] != nc:
            raise ValueError(
                f"Pmodels[{j}] must be ({nc}, {nc}, nd); got {Pm.shape}"
            )

    mode = (mode or 'product').lower()
    if mode not in ('product', 'additive'):
        raise ValueError("mode must be 'product' or 'additive'")

    if K == 1:
        return probamodel2bitable(float(d_components[0]), dmodels[0], Pmodels[0])

    if mode == 'product':
        Pacc = probamodel2bitable(float(d_components[0]), dmodels[0], Pmodels[0])
        Dinv = np.diag(1.0 / p)
        for j in range(1, K):
            Pj = probamodel2bitable(float(d_components[j]), dmodels[j], Pmodels[j])
            Pacc = Pacc @ Dinv @ Pj
        return Pacc

    Psum = np.zeros((nc, nc), dtype=float)
    for j in range(K):
        Psum += probamodel2bitable(float(d_components[j]), dmodels[j], Pmodels[j])
    P_indep = np.outer(p, p)
    Pd = Psum - float(K - 1) * P_indep
    Pd = np.maximum(Pd, 0.0)
    rs = Pd.sum(axis=1, keepdims=True)
    rs = np.where(rs > 0, rs, 1.0)
    Pd = Pd / rs * p[:, np.newaxis]
    return Pd


def neighbours(c0,c,Z,nmax,dmax):
    '''
    % neighbours                - radial neighbourhood selection (Jan 1,2001)
    %
    % Select a subset of coordinates and variables based on
    % their distances from the coordinate c0.
    %
    % SYNTAX :
    %
    % [csub,Zsub,dsub,nsub,index]=neighbours(c0,c,Z,nmax,dmax);
    %
    % INPUT :
    %
    % c0      1 by d   vector of coordinates, where d is the dimension
    %                  of the space
    % c       n by d   matrix of coordinates
    % Z       n by k   matrix of values, where each column corresponds to
    %                  the values of a same variable and each line corresponds
    %                  to the values of the diferent variables at the corresponding
    %                  c coordinates
    % nmax    scalar     maximum number of lines of Z that must be kept.
    % dmax    scalar     maximum Euclidian distance between c0 and c.
    %
    % OUTPUT :
    %
    % csub    m by d   matrix which is a subset of lines of the c matrix (m<=n)
    % Zsub    m by k   matrix which is a subset of lines of the Z matrix.
    % dsub    m by 1   vector of distances between csub and c0.
    % nsub    scalar   length of the dsub vector.
    % index   m by 1   vector giving the ordering of the lines in csub with
    %                  respect to the initial matrix c.
    %
    % NOTE :
    %
    % 1-In the case of space/time coordinates, dmax is a vector of length 3,
    % and the last column of c0 and c is the temporal coordinate. In this case,
    % dmax(1) is the maximum spatial distance between the coordinate in c and c0,
    % dmax(2) is the maximum temporal distance between coordinate in c and c0,
    % and dmax(3) refers to the space/time metric, such that :
    %
    % space/time distance=spatial distance+dmax(3)*temporal distance.
    % 
    % The space/time distance is used to select the nmax closest coordinates c
    % from the estimation coordinates c0.
    %
    % 2- It is possible to specify additional 1 by 1 and n by 1 index vectors,
    % taking integer values from 1 to nv. The values in the index vectors specify
    % which of the nv variable is known at each one of the corresponding coordinates.
    % The c0 and c matrices of coordinates and the index vectors are then grouped
    % together using the MATLAB cell array notation, so that c0={c0,index0} and
    % c={c,index}.
    '''
    if isinstance(dmax,np.ndarray):
        dmax = dmax.ravel().tolist()
    if isinstance(dmax,int)|isinstance(dmax,float):
        dmax = [dmax]
    if np.isinf(nmax):
        nmax = None
    
    
    c0 = c0.reshape(1,-1)
    isST = len(dmax)==3

    if isST:

        ## spatio and temporal case 
        dt = cdist(c0[:,-1].reshape(-1,1),cs[:,-1].reshape(-1,1)).ravel() ## calculate the temporal distance between c0 and c
        ds = cdist(c0,cs).ravel() ## calculate the spatial distance between c0 and c
        d = ds + dt*dmax[2] ## combine distance in space and time
        index=np.where((ds<=dmax[2])&(dt<=dmax[1]))[0] ## find distances in space & time<=dmax
        dstdf= pd.DataFrame(d.reshape(-1,1)).loc[index] ## add the idx on each distance by create the dataframe and remove the points that the distance more than dmax
        sltidx = dstdf.sort_values(0).iloc[:nmax].index## get the index of at most the closest nsmax amount of point.
    else:
        ## pure spatail case
        d = cdist(c0,c).ravel() ## calculate the distance between c0 and c
        ddf= pd.DataFrame(d.reshape(-1,1)) ## add the idx on each distance by create the dataframe
        ddf = ddf.loc[d<dmax[0]]## remove the points that the distance more than dmax
        sltidx = ddf.sort_values(0).iloc[:nmax].index## get the index of at most the closest nsmax amount of point.
    csub,Zsub,dsub,nsub,index = c[sltidx,:],Z[sltidx,:],d.ravel()[sltidx],len(sltidx),sltidx

    return csub,Zsub,dsub,nsub,index

def designmatrix(c,order):
    '''
    % designmatrix              - design matrix in a linear regression model (Jan 1,2001)
    %
    % Build the design matrix associated with a polynomial
    % mean of a given order in a linear regression model
    % of the form z=X*b+e.
    %
    % SYNTAX :
    %
    % [X,index]=designmatrix(c,order);
    %
    % INPUT :
    %
    % c       n by d       matrix of coordinates for the locations. A line
    %                      corresponds to the vector of coordinates at a
    %                      location, so the number of columns in c corresponds
    %                      to the dimension of the space. There is no restriction
    %                      on the dimension of the space.
    % order   scalar       order of the polynomial mean along the spatial axes
    %                      specified in c, where order>=0. When order=NaN, an empty
    %                      X matrix is returned.
    %
    % OUTPUT :
    %
    % X       n by k       design matrix, where each column corresponds to one
    %                      of the polynomial term, sorted in the first place 
    %                      with respect to the degree of the polynomial term,
    %                      and sorted in the second place with respect to the 
    %                      axis number. 
    %
    % index   1 or 2 by k  matrix associated with the columns of X. The first line
    %                      specifies the degree of the estimated polynomial term for
    %                      the corresponding column of X, and the second line specifies
    %                      the axis number to which this polynomial term belongs. The
    %                      axis are numbered according to the columns of c. E.g., the axis
    %                      2 corresponds to the second column of c. Note that the value 0
    %                      in the second line of index is associated with the polynomial
    %                      term of degree equal to 0 (i.e., the constant term) that is 
    %                      defined jointly for all the axes. In the singular case where c
    %                      is a column vector (i.e., the dimension of the space is equal
    %                      to 1), there is only one line for the index variable.
    %
    % NOTE :
    %
    % 1- It is also possible to process several variables at the same time
    % (multivariate case). It is needed to specify additionally tags in the
    % c matrix. These tags are provided as a vector of values that refers to
    % the variable, the values ranging from 1 to nv, where nv is the number
    % of variables. E.g., if there are 3 variables, the input index column vector
    % must be defined, and the elements in index are equal to 1, 2 or 3. The
    % c and index variables are grouped using the MATLAB cell array notation,
    % so that c={c, index}, is now the correct input variable. Using the same
    % logic, order is now a column vector specifying the order of the polynomial
    % mean for each variable. For the output variable index, there is an additional
    % first column that refers to the variable number associated with the
    % corresponding column of X.
    %
    % 2- For space/time data, the convention is that the last column of the c
    % matrix of coordinates corresponds to the time axis. Is is then possible to
    % specify a different order for the polynomial along the spatial axes and the
    % temporal axis. For the univariate case, order is a 1 by 2 vector, where
    % order(1) is the order of the spatial polynomial and order(2) is the order of
    % the temporal polynomial. For the multivariate case where nv different variables
    % are considered, order is a nv by 2 matrix, where the first and second columns
    % of order contain the order of the spatial and the temporal polynomial for
    % each of the nv variables, respectively. If in that case order is entered as
    % a 1 by 2 matrix, the same spatial order corresponding to order(1) will be used
    % for all the variables.
    '''
    n,nd=c.shape
    X =np.array([]).reshape(n,0)
    index = np.array([]).reshape(2,0)
    
    order=[order,order]
    if ~(np.isnan(order[0]) & np.isnan(order[1])):
        X=np.hstack((X,np.ones((n,1))))
        index=np.hstack((index,np.array([0,0]).reshape(-1,1)))
    if ~np.isnan(order[0]):
        for j in range(1,order[1]+1):
            for k in range(1,nd):
                X=np.hstack((X,c[:,[k]]**j))
                index=np.hstack((index,np.array([j,k]).reshape(-1,1)))
    if ~np.isnan(order[1]):
        for j in range(1,order[1]+1):
            X=np.hstack((X,c[:,[nd-1]]**j))
            index=np.hstack((index,np.array([j,nd]).reshape(-1,1)))
    if ((index.size!=0)&(nd==1)):
        index=index[[0],:]

    return X,index

def gausspdf(z,param):
    '''
    % gausspdf                  - Gaussian probability distribution function (Jan 1,2001)
    %
    % Compute the values of the probability distribution
    % function for a Gaussian distribution with
    % specified mean and variance parameters.
    %
    % SYNTAX :
    %
    % [pdf]=gausspdf(z,param);
    %
    % INPUT :
    %
    % z        n by k   matrix of values for which the probability
    %                   distribution function must be computed.
    % param    1 by 2   parameters of the Gaussian distribution, where :
    %                   param(1) is the mean of the distribution,
    %                   param(2) is the variance of the distribution.
    %
    % OUTPUT :
    %
    % pdf      n by k   matrix of values for the probability distribution
    %                   function computed at the corresponding z values.
    '''
    
    m,v=param

    if v<0:
        raise SyntaxError('a variance cannot be negative')

    if v!=0:
        A=1/np.sqrt(2*np.pi*v)
        pdf=A*np.exp(-0.5*((z-m)/np.sqrt(v))**2)
    else:
        pdf=np.zeros(z.shape)
        pdf=np.where(z==m,np.inf,pdf)
    return pdf

def regression(c,z,order,K= None):
    '''
    % regression                - parameters estimation in a linear regression model (Jan 1,2001)
    %
    % Standard implementation of the least squares estimation
    % procedure in a linear regression model, where the
    % deterministic part of the linear model is a polynomial
    % of arbitrary order. Though it is presented here in a
    % spatial context, it can be used for a wide variety of
    % other non spatial cases. 
    %
    % SYNTAX :
    %
    % [best,Vbest,zest,index]=regression(c,z,order,K); 
    %
    % INPUT :
    %
    % c        n by d       matrix of coordinates for the locations. A line
    %                       corresponds to the vector of coordinates at a
    %                       location, so the number of columns in c corresponds
    %                       to the dimension of the space. There is no restriction
    %                       on the dimension of the space.
    % z        n by 1       vector of values at the c coordinates.
    % order    scalar       order of the polynomial mean along the spatial axes
    %                       specified in c, where order>=0.
    % K        n by n       optional square symmetric matrix of covariance for the
    %                       values specified in z. When K is specified, the generalized
    %                       least squares parameter estimates are computed, whereas the
    %                       ordinary least squares parameter estimates are computed
    %                       otherwise.
    %
    % OUTPUT :
    %
    % best     k by 1       vector of parameter estimates.
    % Vbest    k by k       square symmetric matrix  of covariance for the best parameter
    %                       estimates.
    % zest     n by 1       vector of estimated regression values at the c coordinates.
    % index    k by 1 or 2  matrix associated with best. The first column specifies the
    %                       degree of the estimated polynomial term for the corresponding
    %                       best element, and the second column specifies the axis number
    %                       to which this polynomial term belongs. The axis are numbered
    %                       according to the columns of c. E.g., the axis 2 corresponds to
    %                       the second column of c. Note that the value 0 in the second
    %                       column of index is associated with the polynomial term of degree
    %                       equal to 0 (i.e., the constant term) that is defined jointly for
    %                       all the axes. In the singular case where c is a column vector
    %                       (i.e., the dimension of the space is equal to 1), there is only
    %                       one column for the index variable.
    %
    % NOTE :
    %
    % 1- It is also possible to process several variables at the same time
    % (multivariate case). It is needed to specify additionally tags in the
    % c matrix. These tags are provided as a vector of values that refers to
    % the variable, the values ranging from 1 to nv, where nv is the number
    % of variables. E.g., if there are 3 variables, the input index column vector
    % must be defined, and the elements in index are equal to 1, 2 or 3. The
    % c and index variables are grouped using the MATLAB cell array notation,
    % so that c={c,index}, is now the correct input variable. Using the same
    % logic, order is now a column vector specifying the order of the polynomial
    % mean for each variable. For the output variable index, there is an additional
    % first column that refers to the variable number associated with the
    % corresponding best elements. If regression.m is used for processing several
    % variables at the same time and if K is not specified, the results are the
    % same than those obtained when using the function separately for each variable.
    % This is because there are no interactions taken into account between the
    % variables in the ordinary least squares case.
    %
    % 2- For space/time data, the convention is that the last column of the c
    % matrix of coordinates corresponds to the time axis. Is is then possible to
    % specify a different order for the polynomial along the spatial axes and the
    % temporal axis. For the univariate case, order is a 1 by 2 vector, where
    % order(1) is the order of the spatial polynomial and order(2) is the order of
    % the temporal polynomial. For the multivariate case where nv different variables
    % are considered, order is a nv by 2 matrix, where the first and second columns
    % of order contain the order of the spatial and the temporal polynomial for
    % each of the nv variables, respectively. If in that case order is entered as
    % a 1 by 2 matrix, the same spatial order corresponding to order(1) and the same
    % temporal order corresponding to order(2) will be used for all the variables.
    '''
    from numpy.linalg import inv

    X,index=designmatrix(c.reshape(-1,1),order)
    index=index.T
    n,p=X.shape
    Xt=X.T
    z = z.reshape(-1,1)
    if X.size !=0:
        if K is None:
            invXtX=inv(np.dot(Xt,X))
            best=np.dot(np.dot(invXtX,Xt),z)
            zest=np.dot(X,best)
            resi=z-zest
            s2=(np.dot(resi.T,resi))/(n-p)
            Vbest=invXtX*s2
        else:
            XtinvK=np.dot(X.T,inv(K))
            invXtinvKX=inv(np.dot(XtinvK,X))
            best=np.dot(np.dot(invXtinvKX,XtinvK),z)
            zest=np.dot(X,best)
            Vbest=invXtinvKX
    
    return best,Vbest,zest,index

def smooth(dk,d,p,o,kstd,order,weighting):
    '''
    % smooth                 - smoothing using a Gaussian kernel regression method
    %                          (December 1,2003)
    %
    % Implementation of the regression.m function in a moving
    % neighbourhood context. Regression is conducted locally
    % at a set of coordinates using a least squares estimation
    % procedure for a linear regression model, where the
    % deterministic part of the linear model is a polynomial of
    % a given order. Instead of using ordinary least squares,
    % the function uses a diagonal covariance matrix, where the
    % variances are inversely proportional to the weights provided
    % by a Gaussian kernel, as well as inversely proportional to
    % the number of observations. 
    %
    % SYNTAX :
    %
    % [pk]=smooth(dk,d,p,o,kstd,order,weighting); 
    %
    % INPUT :
    %
    % dk          nk by 1   vector of distances for the smoothed estimates.
    % d           n by 1    vector of distances for which the bivariate 
    %                       probabilities have been estimated.
    % p           n by 1    vector of probabilities at distances specified in d.
    % o           n by 1    vector of number of pairs separated by distances
    %                       specified in d. 
    % kstd        scalar    standard deviation of the Gaussian kernel function
    %                       which is used in the kernel regression smoothing
    % order       scalar    order of the polynomial used in the kernel regression
    %                       smoothing
    % weighting   scalar    weigthing is equal to 1 or 0, depending if the user wants
    %                       or does not want to use the number of observations
    %                       as weights.
    %
    % OUTPUT :
    %
    % pk          nk by 1   vector of estimated probabilities at distances dk.
    '''

    nk = len(dk)
    pk=np.zeros((nk,1))*np.nan
    nmax=np.inf 

    for i in range(nk):
        dmaxi=4*kstd
        dsub,posub,trash,nsub,_ = neighbours(dk[i],d,np.hstack((p,o)),nmax,dmaxi)
        if nsub>0:
            dsub=dsub-dk[i]
            w=gausspdf(dsub,[0,kstd**2])
            w = 1/w
            if weighting==1:
                w=w*posub[:,[1]]
            K=np.diag(w.ravel())
            best,Vbest,zest,index=regression(dsub,posub[:,[0]],order,K)
            pk[i]=best[0]
        else:
            print(f'Warning: No estimation at distance {dk[i]} - increase smoothing parameter')
    return pk

def _probatablefit_kernel(dfit,d,P,o,kstd,options = [0,0,1]):
    
    '''
    % _probatablefit_kernel   - kernel-regression fitting of bivariate probability tables
    %                          (December 1,2003)
    %                          (internal; use probatablefit(fit_method='kernel') instead)
    %
    % SYNTAX : [Pfit]=probatablefit(dfit,d,P,o,kstd,options);
    %
    % INPUT :
    %
    % dfit      n by 1       vector of values that specify the distances for which fitted
    %                        bivariate probability values are sought. Maximum distance in
    %                        dfit cannot exceed maximum distance in d.
    % d         ncl by 1     vector giving the sorted values of the mean distance separating
    %                        the pairs of points that belong to the same distance class. The
    %                        first value corresponds to a null distance. 
    % P         nc by nc     symmetric array of cells that contains the bivariate probability
    %                        tables estimates between the nc categories for the distance classes
    %                        specified in d. Diagonal cells contain the ncl by 1 vector of
    %                        probability estimates for the same category, whereas off-diagonal
    %                        cells contain the ncl by 1 vector of cross-category probability
    %                        estimates.For diagonal cells, the first value of each vector is
    %                        the estimated proportion of the corresponding category, whereas
    %                        this first value is always equal to 0 for off-diagonal cells.
    % o         ncl by 1     vector giving the number of pairs of points that belong to the 
    %                        corresponding distance classes. 
    % kstd      scalar       standard deviation of the Gaussian kernel function which is used
    %                        in the kernel regression smoothing (the higher the value is, the
    %                        stronger the smoothing will be). Suggested starting value is
    %                        about 0.5*max(d)/length(d).
    % options   1 by 3       optional vector of parameters that can be used if default values are
    %                        not satisfactory (otherwise this vector can simply be omitted from the
    %                        input list of variables), where :
    %                        options(1) displays the fitted probability tables if the value is set
    %                        to one (default value is 0).
    %                        options(2) is the order of the polynomial used in the kernel regression
    %                        smoothing (default value is 0).
    %                        options(3) uses the number of obervations as weights for the fitting
    %                        if the value is set to one (default value is 1).
    %
    % OUTPUT :
    %
    % Pfit      nc by nc     cell array that contains the fitted bivariate probability tables
    %                        between the nc categories for the distance classes specified in dfit.
    %                        Diagonal cells contain the n by 1 vector of probability estimates
    %                        for the same category, whereas off-diagonal cells contain the n by
    %                        1 vector of cross-category probability estimates.
    '''
    
    
    if np.max(dfit)>np.max(d):
        raise SyntaxError('maximum distance in dfit cannot exceed maximum distance in d')

    # Normalise d and o to column vectors so vstack works regardless of
    # whether the caller passes 1-D arrays (from probatablecalc) or
    # explicit (n,1) column vectors.
    d = np.asarray(d, dtype=float).reshape(-1, 1)
    o = np.asarray(o, dtype=float).reshape(-1, 1)

    ### Compute table of bivariate probabilities at null distance

    P0=probamodel2bitable(0,d.ravel(),P)

    ### Create symmetric vectors for distance and number of 
    ### observations around null distance

    nv= P.shape[0]
    nd= d.shape[0]
    dsym=np.vstack((np.flipud(-d),d[1:nd]))
    osym=np.vstack((np.flipud(o),o[1:nd]))


    ### Interpolate bivariate probabilities at dfit using a kernel 
    ### regression method. Values are mirrored first around null distance ;
    ### this makes possible to avoid the border effect when smoothing for
    ### dfit values close to zero
    Pfit = []
    for i in range(nv):
        Pfit.append([])
        for j in range(nv):
            Pfit[i].append([]) # corresponding to matlab code Pfit=cell(nv,nv); create nv,nv list 
    for i in range(nv):
        for j in range(nv):
            Pij = P[i, j, :]                        # shape (nd,)
            valid = np.isfinite(Pij)                 # True where category pair (i,j)
                                                     # had observed pairs at that lag

            # Replace NaN with 0 so arithmetic is clean; set weight to 0 for
            # those bins so they don't influence the kernel regression.
            # Using the global O as weight for bins where P[i,j] is NaN would
            # inject NaN-contaminated, spuriously-weighted points into the fit
            # and cause wild spikes (most visible for rare diagonal pairs).
            Pij_safe = np.where(valid, Pij, 0.0)
            o_ij     = np.where(valid, o.ravel(), 0.0).reshape(-1, 1)

            # Build symmetric (mirrored) vectors
            psym    = np.hstack((np.flipud(-Pij_safe) + 2*P0[i, j],
                                 Pij_safe[1:nd])).reshape(-1, 1)
            osym_ij = np.vstack((np.flipud(o_ij), o_ij[1:nd]))

            Pfit[i][j] = smooth(dfit, dsym, psym, osym_ij, kstd,
                                options[1], options[2])
            Pfit[j][i] = Pfit[i][j]
            minPfitij=np.min(Pfit[i][j])
            if (minPfitij<0)&(minPfitij>-np.spacing(1)):
                index=(Pfit[i][j]<0)
                Pfit[i][j][index]=0
            if max(Pfit[i][j])<0:
                print(f'Warning: Non valid model - probabilities below 0 for categories {i}-{j}')
                print('Try to change polynomial order or smoothing parameter')
            if max(Pfit[i][j])>1:
                print(f'Warning: Non valid model - probabilities above 1 for categories {i}-{j}')
                print('Try to change polynomial order or smoothing parameter');
            if np.isnan(sum(Pfit[i][j])):
                print(f'Warning : some fitted values are NaN\'s for categories {i}-{j}')
                print('Increase smoothing parameter or use other distance classes in probatablecalc.m')
    Pfit = np.array(Pfit)[:,:,:,0]
    return Pfit


# ---------------------------------------------------------------------------
# Spline-based Pmodel fitting
# ---------------------------------------------------------------------------

def _loocv_smooth_score(log_s, d, p_iso, w):
    """LOOCV score for a given smoothing parameter (log-scale)."""
    from scipy.interpolate import UnivariateSpline

    s = np.exp(log_s)
    n = len(d)
    if n < 5:
        try:
            spl = UnivariateSpline(d, p_iso, w=w, s=s, k=min(3, n - 1))
            p_fit = spl(d)
            return float(np.sum(w ** 2 * (p_iso - p_fit) ** 2))
        except Exception:
            return 1e10

    sse = 0.0
    for k in range(n):
        mask = np.ones(n, dtype=bool)
        mask[k] = False
        d_tr, p_tr, w_tr = d[mask], p_iso[mask], w[mask]
        if len(d_tr) < 4:
            continue
        try:
            spl = UnivariateSpline(d_tr, p_tr, w=w_tr, s=s, k=3)
            pred = float(np.clip(spl(d[k]), 0, 1))
            sse += w[k] ** 2 * (p_iso[k] - pred) ** 2
        except Exception:
            sse += w[k] ** 2 * p_iso[k] ** 2
    return sse


def _fit_one_cell_spline(d, p_cell, o, d_fit, is_diagonal,
                         enforce_monotone, spline_smooth, spline_k):
    """Fit one (i,j) cell of the bivariate table using a smoothing spline.

    Parameters
    ----------
    d : (n_emp,) array  — empirical lag distances.
    p_cell : (n_emp,) array — empirical P[i,j,:].
    o : (n_emp,) array — pair counts.
    d_fit : (n_fit,) array — evaluation grid.
    is_diagonal : bool — True for self-transition (i==j).
    enforce_monotone : bool — apply isotonic regression before spline.
    spline_smooth : 'auto', None, or float.
    spline_k : int — spline degree (default 3).

    Returns
    -------
    p_fitted : (n_fit,) array.
    """
    from scipy.interpolate import UnivariateSpline, PchipInterpolator
    from scipy.optimize import minimize_scalar

    valid = np.isfinite(p_cell) & (o > 0)
    d_v = d[valid]
    p_v = p_cell[valid].copy()
    o_v = o[valid]

    if len(d_v) < 2:
        return np.full(len(d_fit), np.nan)

    # ── Isotonic regression ───────────────────────────────────────────────
    if enforce_monotone and len(d_v) >= 2:
        try:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(
                increasing=(not is_diagonal), out_of_bounds='clip')
            p_v = iso.fit_transform(d_v, p_v, sample_weight=o_v)
        except ImportError:
            order = np.argsort(d_v)
            d_v, p_v, o_v = d_v[order], p_v[order], o_v[order]
            if is_diagonal:
                for idx in range(1, len(p_v)):
                    if p_v[idx] > p_v[idx - 1]:
                        p_v[idx] = p_v[idx - 1]
            else:
                for idx in range(1, len(p_v)):
                    if p_v[idx] < p_v[idx - 1]:
                        p_v[idx] = p_v[idx - 1]

    order = np.argsort(d_v)
    d_v, p_v, o_v = d_v[order], p_v[order], o_v[order]

    # Weights: sqrt(pair_count) — matches UnivariateSpline convention
    w_v = np.sqrt(np.clip(o_v, 1, None))

    k_actual = min(spline_k, len(d_v) - 1)
    if k_actual < 1:
        return np.full(len(d_fit), p_v[0] if len(p_v) else np.nan)

    # ── Choose smoothing parameter ────────────────────────────────────────
    if spline_smooth == 'auto':
        res = minimize_scalar(
            _loocv_smooth_score, bounds=(-5, 12), method='bounded',
            args=(d_v, p_v, w_v))
        s_opt = np.exp(res.x)
    elif spline_smooth is None:
        s_opt = len(d_v)
    else:
        s_opt = float(spline_smooth)

    # ── Fit spline ────────────────────────────────────────────────────────
    try:
        spl = UnivariateSpline(d_v, p_v, w=w_v, s=s_opt, k=k_actual)
        p_fitted = spl(d_fit)
    except Exception:
        if len(d_v) >= 2:
            p_fitted = np.interp(d_fit, d_v, p_v)
        else:
            p_fitted = np.full(len(d_fit), p_v[0])

    p_fitted = np.clip(p_fitted, 0.0, 1.0)

    # ── Post-fit monotonicity enforcement on the evaluation grid ──────────
    if enforce_monotone:
        if is_diagonal:
            for idx in range(1, len(p_fitted)):
                if p_fitted[idx] > p_fitted[idx - 1]:
                    p_fitted[idx] = p_fitted[idx - 1]
        else:
            for idx in range(1, len(p_fitted)):
                if p_fitted[idx] < p_fitted[idx - 1]:
                    p_fitted[idx] = p_fitted[idx - 1]

    return p_fitted


def spline_fit_pmodel(d_fit, D_emp, P_emp, O_emp,
                      enforce_monotone=True,
                      spline_smooth='auto',
                      spline_k=3,
                      verbose=False):
    """Fit bivariate probability tables using isotonic + smoothing spline.

    Drop-in replacement for :func:`probatablefit` (kernel smoother), with
    better behaviour on sparse data and no NaN gaps.

    Parameters
    ----------
    d_fit : (n_fit,) array
        Evaluation distances.
    D_emp : (n_emp,) array
        Mean lag distances from ``probatablecalc``.
    P_emp : (nc, nc, n_emp) array
        Empirical bivariate tables.
    O_emp : (n_emp,) or (n_emp, 1) array
        Pair counts per lag bin.
    enforce_monotone : bool
        If True (default), apply isotonic regression before the spline fit
        and post-fit monotonicity clipping.  Self-transitions (diagonal)
        are forced non-increasing; cross-transitions non-decreasing.
    spline_smooth : ``'auto'``, None, or float
        Smoothing parameter for ``UnivariateSpline``:

        * ``'auto'`` — LOOCV-optimised (recommended).
        * ``None``   — scipy default (``s = n``).
        * float      — fixed smoothing factor.
    spline_k : int
        Spline degree (default 3 = cubic).
    verbose : bool
        Print diagnostics.

    Returns
    -------
    Pfit : (nc, nc, n_fit) ndarray
        Fitted bivariate probability model, same shape as ``probatablefit``
        output.
    """
    D_emp = np.asarray(D_emp, dtype=float).ravel()
    O_emp = np.asarray(O_emp, dtype=float).ravel()
    P_emp = np.asarray(P_emp, dtype=float)
    d_fit = np.asarray(d_fit, dtype=float).ravel()

    nc = P_emp.shape[0]
    n_fit = len(d_fit)
    Pfit = np.full((nc, nc, n_fit), np.nan)

    for i in range(nc):
        for j in range(i, nc):
            is_diag = (i == j)
            p_cell = P_emp[i, j, :]

            fitted = _fit_one_cell_spline(
                D_emp, p_cell, O_emp, d_fit,
                is_diagonal=is_diag,
                enforce_monotone=enforce_monotone,
                spline_smooth=spline_smooth,
                spline_k=spline_k,
            )
            Pfit[i, j, :] = fitted
            Pfit[j, i, :] = fitted

            if verbose and np.any(np.isnan(fitted)):
                print(f'[spline_fit] Warning: NaN in cell ({i},{j})')

    return Pfit


# ---------------------------------------------------------------------------
# Pseudocount regularisation helpers
# ---------------------------------------------------------------------------

def regularize_pmodel(Pmodel, dmodel, O_emp, alpha, d_emp=None):
    """
    Apply Dirichlet pseudocount regularisation to a smooth bivariate model.

    Shrinks each cell ``Pmodel[c1, c2, k]`` toward the independence product
    ``p(c1) * p(c2)``.  The shrinkage fraction at lag *k* is
    ``alpha / (O_k + alpha)``, so sparse lags (small ``O_k``) are
    regularised more heavily than well-sampled lags.

    The operation implements the Dirichlet-Multinomial posterior mean:

    .. math::

        \\tilde{P}_{c_1 c_2}^{(k)} =
        \\frac{O^{(k)} \\, P_{c_1 c_2}^{(k)} + \\alpha \\, p(c_1)p(c_2)}
             {O^{(k)} + \\alpha}

    Parameters
    ----------
    Pmodel : np.ndarray, shape (nc, nc, nd)
        Smooth bivariate probability model from ``probatablefit``.
    dmodel : np.ndarray, shape (nd,)
        Distances corresponding to the columns of ``Pmodel``.
    O_emp : array_like, shape (n_emp,)
        Pair counts from ``probatablecalc`` at the empirical lag distances.
    alpha : float
        Pseudocount strength.  ``alpha = 0`` leaves ``Pmodel`` unchanged.
        Jeffreys default: ``nc**2 / 2``  (e.g. 8 for nc = 4).
    d_emp : array_like, shape (n_emp,), optional
        Distances at which ``O_emp`` was tabulated.  If ``None``, ``O_emp``
        is assumed to be already tabulated at ``dmodel`` distances (lengths
        must match).

    Returns
    -------
    Pmodel_reg : np.ndarray, shape (nc, nc, nd)
        Regularised bivariate probability model, with the same shape and
        normalisation conventions as the input ``Pmodel``.

    Notes
    -----
    * The marginal ``p(c)`` used to build the independence product is
      estimated from the last **non-NaN** slice of ``Pmodel`` (the largest
      lag at which ``probatablefit`` had sufficient pairs).  Using the raw
      last slice would propagate NaN through every regularised cell when
      ``probatablefit`` could not fit the extreme lags.
    * Slices that are ``NaN`` in the input (typically the few largest lags
      that lack data) are **replaced** by the independence product
      ``p(c1)*p(c2)`` — the correct theoretical limit at large separation.
    * Pass the returned ``Pmodel_reg`` directly to ``BMEcatPdf``,
      ``MCPcatPdf``, or ``HBMEcatPdf`` in place of the raw ``Pmodel``.

    Examples
    --------
    >>> alpha_opt, _ = alpha_empirical_bayes(P_emp, O_emp)
    >>> Pmodel_raw = _probatablefit_kernel(d_fit, D_emp, P_emp, O_emp, kstd)
    >>> Pmodel = regularize_pmodel(Pmodel_raw, d_fit, O_emp, alpha=alpha_opt)
    """
    if alpha == 0:
        return Pmodel.copy()

    Pmodel = np.asarray(Pmodel, dtype=float)
    dmodel = np.asarray(dmodel, dtype=float).ravel()
    O_emp  = np.asarray(O_emp,  dtype=float).ravel()
    nc, _, nd = Pmodel.shape

    # Interpolate O_emp onto the dmodel grid
    if d_emp is None:
        if len(O_emp) != nd:
            raise ValueError(
                f"O_emp length ({len(O_emp)}) != dmodel length ({nd}). "
                "Pass d_emp to specify the distances for O_emp."
            )
        O_interp = O_emp.copy()
    else:
        d_emp = np.asarray(d_emp, dtype=float).ravel()
        O_interp = np.interp(dmodel, d_emp, O_emp,
                             left=float(O_emp[0]), right=float(O_emp[-1]))

    # -----------------------------------------------------------------------
    # Estimate marginal p(c) from the last NON-NaN slice of Pmodel.
    # probatablefit produces NaN at large lags where no pairs exist; using
    # Pmodel[:, :, -1] directly would poison P_indep and all regularised
    # cells, collapsing every prediction to the dominant category.
    # -----------------------------------------------------------------------
    valid_mask = np.all(~np.isnan(Pmodel), axis=(0, 1))   # shape (nd,)
    if valid_mask.any():
        last_valid_k = int(np.where(valid_mask)[0][-1])
        P_far = Pmodel[:, :, last_valid_k]
    else:
        # Extreme fallback: assume uniform marginal
        P_far = np.full((nc, nc), 1.0 / nc ** 2)

    p_marg = P_far.sum(axis=1)
    p_marg = np.clip(p_marg, 1e-15, None)
    p_marg /= p_marg.sum()
    P_indep = np.outer(p_marg, p_marg)   # shape (nc, nc)

    # -----------------------------------------------------------------------
    # Apply shrinkage per lag class.
    # NaN slices (beyond the data range) are replaced by P_indep, which is
    # the correct theoretical limit (independence at infinite separation).
    # -----------------------------------------------------------------------
    Pmodel_reg = Pmodel.copy()
    for k in range(nd):
        P_k = Pmodel[:, :, k]
        if np.any(np.isnan(P_k)):
            # Beyond data range: replace with the independence product
            Pmodel_reg[:, :, k] = P_indep
        else:
            O_k   = max(float(O_interp[k]), 0.0)
            denom = O_k + alpha
            Pmodel_reg[:, :, k] = (O_k * P_k + alpha * P_indep) / denom

    return Pmodel_reg


def alpha_empirical_bayes(P_emp, O_emp):
    """
    Find the optimal pseudocount alpha via Empirical Bayes marginal likelihood.

    Maximises the Dirichlet-Multinomial marginal log-likelihood of the
    observed bivariate pair counts over the scalar hyperparameter *alpha*,
    using 1-D bounded optimisation (typically < 1 ms for standard datasets).

    The marginal log-likelihood for distance class *k* is:

    .. math::

        \\log p(\\mathbf{n}^{(k)} \\mid \\alpha) =
        \\log\\Gamma(\\alpha) - \\log\\Gamma(O^{(k)} + \\alpha)
        + \\sum_{c_1, c_2}
          \\Bigl[
            \\log\\Gamma\\!\\left(n_{c_1 c_2}^{(k)} + \\tfrac{\\alpha}{n_c^2}\\right)
            - \\log\\Gamma\\!\\left(\\tfrac{\\alpha}{n_c^2}\\right)
          \\Bigr]

    where :math:`n_{c_1 c_2}^{(k)} = O^{(k)} \\cdot P_{c_1 c_2}^{(k)}`.

    Parameters
    ----------
    P_emp : np.ndarray, shape (nc, nc, n_emp)
        Empirical bivariate table from ``probatablecalc``.
    O_emp : array_like, shape (n_emp,)
        Pair counts from ``probatablecalc``.

    Returns
    -------
    alpha_opt : float
        Optimal pseudocount.  Typical range for nc = 4 is 5–50.
    log_ml_opt : float
        Log marginal likelihood at the optimum (useful for sanity checks
        or model comparison across different datasets / bandwidths).

    Notes
    -----
    Lag classes with fewer than 2 pairs are skipped (they carry no
    information for calibrating alpha).

    A rough heuristic for a quick default is ``alpha = nc**2 / 2``
    (Jeffreys prior, half-count per cell), which corresponds to
    log_alpha ≈ log(nc**2 / 2).

    Examples
    --------
    >>> D_emp, P_emp, O_emp = probatablecalc(cs, ps, coord_limit)
    >>> alpha_opt, log_ml = alpha_empirical_bayes(P_emp, O_emp)
    >>> print(f"Optimal alpha = {alpha_opt:.1f}")
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import gammaln

    P_emp = np.asarray(P_emp, dtype=float)
    O_emp = np.asarray(O_emp, dtype=float).ravel()
    nc    = P_emp.shape[0]
    nc2   = float(nc ** 2)

    def neg_log_ml(log_alpha):
        alpha      = np.exp(log_alpha)
        alpha_cell = alpha / nc2
        L = 0.0
        for k in range(len(O_emp)):
            O_k = float(O_emp[k])
            if O_k < 2:
                continue
            # Skip lag classes with any NaN cell (no pairs → no information)
            if np.any(np.isnan(P_emp[:, :, k])):
                continue
            L += gammaln(alpha) - gammaln(O_k + alpha)
            for c1 in range(nc):
                for c2 in range(nc):
                    n_cc = O_k * float(P_emp[c1, c2, k])
                    L += gammaln(n_cc + alpha_cell) - gammaln(alpha_cell)
        return -L

    res       = minimize_scalar(neg_log_ml, bounds=(-2.0, 10.0), method='bounded')
    alpha_opt = float(np.exp(res.x))
    return alpha_opt, float(-res.fun)


# ======================================================================
#  Build Pmodel pipeline  (convenience wrapper)
# ======================================================================

def _pmodel_pipeline(D_emp, P_emp, O_emp, n_fit=80, kstd=None,
                     n_bins=None, d_max_emp=None, opts=None,
                     regularize=True, alpha=None,
                     fit_method='spline',
                     enforce_monotone=True,
                     spline_smooth='auto',
                     spline_k=3,
                     verbose=False):
    """
    Internal Pmodel fitting pipeline: raw fit → regularisation → (dmodel, Pmodel).

    Called by :func:`probatablefit` (the public API) and aliased as
    :func:`build_pmodel` for backward compatibility.  See
    :func:`probatablefit` for the full parameter documentation.

    Parameters
    ----------
    D_emp : np.ndarray, shape (n_emp,)
        Mean distances from ``probatablecalc``.
    P_emp : np.ndarray, shape (nc, nc, n_emp)
        Empirical bivariate tables from ``probatablecalc``.
    O_emp : np.ndarray, shape (n_emp,)
        Pair counts from ``probatablecalc``.
    n_fit : int, optional
        Number of points on the fitted distance grid (default 80).
    kstd : float or None, optional
        Kernel bandwidth for ``probatablefit``.  If ``None``, a default
        of ``0.5 * d_max_emp / n_bins`` is used; *n_bins* and
        *d_max_emp* must be given in that case.  **Only used when**
        ``fit_method='kernel'``.
    n_bins : int or None, optional
        Number of empirical bins (used only for the default *kstd*).
    d_max_emp : float or None, optional
        Max empirical distance.  If ``None``, uses ``max(D_emp[finite])``.
    opts : list or None, optional
        Options triplet for ``probatablefit`` (default ``[0, 0, 1]``).
        **Only used when** ``fit_method='kernel'``.
    regularize : bool, optional
        Whether to apply pseudocount regularisation (default ``True``).
    alpha : float or None, optional
        Pseudocount for regularisation.  If ``None`` and
        ``regularize=True``, the Empirical-Bayes optimal is used.
    fit_method : ``{'spline', 'kernel'}``, optional
        Fitting back-end (default ``'spline'``).

        * ``'spline'`` — isotonic regression + smoothing spline with
          LOOCV-optimised smoothing parameter.  Robust to sparse data
          and produces no NaN gaps.
        * ``'kernel'`` — legacy kernel regression via
          :func:`probatablefit`.  May produce NaN at distances with
          insufficient data.
    enforce_monotone : bool, optional
        When ``fit_method='spline'``, apply isotonic regression before
        the spline fit and post-fit monotonicity clipping (default
        ``True``).  Self-transitions are forced non-increasing;
        cross-transitions non-decreasing.
    spline_smooth : ``'auto'``, None, or float, optional
        Smoothing parameter for ``UnivariateSpline``
        (default ``'auto'`` = LOOCV-optimised):

        * ``'auto'`` — optimise via Leave-One-Out Cross-Validation.
        * ``None``   — scipy default (``s = len(data)``).
        * float      — fixed smoothing factor.
    spline_k : int, optional
        Spline degree (default 3 = cubic).
    verbose : bool, optional
        Print diagnostic messages.

    Returns
    -------
    dmodel : np.ndarray, shape (n_fit,)
        Fitted distance grid.
    Pmodel : np.ndarray, shape (nc, nc, n_fit)
        Smoothed (and optionally regularised) bivariate probability
        model.
    info : dict
        Extra diagnostics: ``'Pmodel_raw'``, ``'alpha'``,
        ``'alpha_log_ml'``, ``'kstd'``, ``'d_fit'``, ``'fit_method'``.
    """
    D_emp = np.asarray(D_emp, dtype=float).ravel()
    O_emp = np.asarray(O_emp, dtype=float).ravel()
    P_emp = np.asarray(P_emp, dtype=float)

    # defaults
    finite_D = D_emp[np.isfinite(D_emp)]
    if d_max_emp is None:
        d_max_emp = float(finite_D.max()) if len(finite_D) > 0 else 1.0

    # Limit d_fit to the range where empirical data exist
    d_fit_max = (min(d_max_emp, float(finite_D.max()) * 0.98)
                 if len(finite_D) > 0 else d_max_emp)
    d_fit = np.linspace(0, d_fit_max, n_fit)

    # ── raw fit ───────────────────────────────────────────────────────────
    if fit_method == 'spline':
        if verbose:
            print(f'[build_pmodel] fit_method=spline  monotone={enforce_monotone}  '
                  f'smooth={spline_smooth}  n_fit={n_fit}  '
                  f'd_fit=[0, {d_fit_max:.0f}]')

        Pmodel_raw = spline_fit_pmodel(
            d_fit, D_emp, P_emp, O_emp,
            enforce_monotone=enforce_monotone,
            spline_smooth=spline_smooth,
            spline_k=spline_k,
            verbose=verbose,
        )
        kstd_used = None

    elif fit_method == 'kernel':
        if kstd is None:
            nb = n_bins if n_bins is not None else max(len(D_emp) - 1, 1)
            kstd = 0.5 * d_max_emp / nb
        if opts is None:
            opts = [0, 0, 1]

        if verbose:
            print(f'[build_pmodel] fit_method=kernel  kstd={kstd:.0f}  '
                  f'n_fit={n_fit}  d_fit=[0, {d_fit_max:.0f}]')

        Pmodel_raw = _probatablefit_kernel(d_fit, D_emp, P_emp,
                                           O_emp.reshape(-1, 1), kstd, opts)
        kstd_used = kstd

    else:
        raise ValueError(
            f"Unknown fit_method '{fit_method}'. "
            "Choose from 'spline' or 'kernel'."
        )

    # ── regularisation ────────────────────────────────────────────────────
    alpha_used = None
    alpha_ll   = None
    if regularize:
        if alpha is None:
            alpha_used, alpha_ll = alpha_empirical_bayes(P_emp, O_emp)
        else:
            alpha_used = float(alpha)
        Pmodel = regularize_pmodel(Pmodel_raw, d_fit, O_emp,
                                   alpha=alpha_used, d_emp=D_emp)
        if verbose:
            print(f'[build_pmodel] alpha={alpha_used:.2f}  '
                  f'log-ML={alpha_ll}')
    else:
        Pmodel = Pmodel_raw.copy()

    info = dict(Pmodel_raw=Pmodel_raw, alpha=alpha_used,
                alpha_log_ml=alpha_ll, kstd=kstd_used, d_fit=d_fit,
                fit_method=fit_method)
    return d_fit, Pmodel, info


# ======================================================================
#  Public unified API
# ======================================================================

def probatablefit(D_emp, P_emp, O_emp, n_fit=80, kstd=None,
                  n_bins=None, d_max_emp=None, opts=None,
                  regularize=True, alpha=None,
                  fit_method='spline',
                  enforce_monotone=True,
                  spline_smooth='auto',
                  spline_k=3,
                  verbose=False):
    """Fit a bivariate probability model from empirical tables.

    This is the main entry point for Pmodel fitting in stamps.  It runs the
    full pipeline — raw fit, optional regularisation, distance-grid
    construction — and returns the ``(dmodel, Pmodel)`` pair expected by
    :func:`~stamps.bme.BMEcatPdf.BMEcatPdf` and
    :func:`~stamps.bme.BMEcatPdf.MCPcatPdf`.

    Two fitting back-ends are available via ``fit_method``:

    * **``'spline'``** *(default)* — isotonic regression + smoothing spline
      with LOOCV-optimised smoothing.  Robust to sparse data and produces
      no NaN gaps.  Recommended for all new work.
    * **``'kernel'``** — legacy Gaussian kernel regression
      (:func:`_probatablefit_kernel`).  Reproduces the original MATLAB
      behaviour; may produce NaN at lag distances with insufficient data.

    Parameters
    ----------
    D_emp : (n_emp,) array
        Mean lag distances from :func:`~stamps.stats.dependence.probatablecalc`.
    P_emp : (nc, nc, n_emp) array
        Empirical bivariate tables.
    O_emp : (n_emp,) array
        Pair counts per lag bin.
    n_fit : int, optional
        Points on the fitted distance grid (default 80).
    kstd : float or None, optional
        Kernel bandwidth.  Auto-computed when ``None``; only used when
        ``fit_method='kernel'``.
    n_bins : int or None, optional
        Number of empirical bins (for auto *kstd*; kernel mode only).
    d_max_emp : float or None, optional
        Maximum empirical distance.  Auto-detected when ``None``.
    opts : list or None, optional
        Options triplet for the kernel fitter (default ``[0, 0, 1]``; kernel
        mode only).
    regularize : bool, optional
        Apply pseudo-count regularisation (default ``True``).
    alpha : float or None, optional
        Pseudo-count.  When ``None`` and ``regularize=True`` the Empirical
        Bayes optimum is used.
    fit_method : ``{'spline', 'kernel'}``, optional
        Fitting back-end (default ``'spline'``).
    enforce_monotone : bool, optional
        Isotonic constraint in spline fit (default ``True``; spline only).
    spline_smooth : ``'auto'``, None, or float, optional
        Smoothing parameter (default ``'auto'`` = LOOCV; spline only).
    spline_k : int, optional
        Spline degree (default 3; spline only).
    verbose : bool, optional
        Print diagnostic messages.

    Returns
    -------
    dmodel : (n_fit,) ndarray
        Fitted distance grid.
    Pmodel : (nc, nc, n_fit) ndarray
        Smoothed (and optionally regularised) bivariate probability model.
    info : dict
        Diagnostics: ``'Pmodel_raw'``, ``'alpha'``, ``'alpha_log_ml'``,
        ``'kstd'``, ``'d_fit'``, ``'fit_method'``.

    Examples
    --------
    >>> from stamps.stats.dependence import probatablecalc, adaptive_bins
    >>> from stamps.stats.dependence import probatablefit
    >>> coord_limit = adaptive_bins(cs_1d, n_bins=12)
    >>> D_emp, P_emp, O_emp = probatablecalc(cs_1d, ps, coord_limit)
    >>> dmodel, Pmodel, info = probatablefit(D_emp, P_emp, O_emp)
    >>> # Using the kernel back-end (legacy):
    >>> dmodel, Pmodel, info = probatablefit(D_emp, P_emp, O_emp, fit_method='kernel')
    """
    return _pmodel_pipeline(
        D_emp, P_emp, O_emp,
        n_fit=n_fit, kstd=kstd, n_bins=n_bins, d_max_emp=d_max_emp,
        opts=opts, regularize=regularize, alpha=alpha,
        fit_method=fit_method, enforce_monotone=enforce_monotone,
        spline_smooth=spline_smooth, spline_k=spline_k, verbose=verbose,
    )


# Backward-compatible alias — existing code calling build_pmodel still works.
build_pmodel = probatablefit


# ======================================================================
#  LOOCV-based Pmodel tuning
# ======================================================================

def loocv_tune_pmodel(cs, ps, code, categories,
                      coord_limit_candidates, kstd_candidates,
                      nsmax, dmax, base_options,
                      method_fn=None,
                      n_fit=80, opts_fit=None,
                      regularize=True, verbose=True):
    """
    Grid-search over binning and smoothing to maximise LOOCV log-likelihood.

    For each ``(coord_limit, kstd)`` combination in the candidate grids
    the function:

    1. Computes empirical tables via ``probatablecalc``.
    2. Fits and regularises ``Pmodel`` via ``build_pmodel``.
    3. Runs Leave-One-Out Cross-Validation using *method_fn* (default
       ``MCPcatPdf``, which is fast).
    4. Records LOOCV accuracy and mean log-likelihood.

    Parameters
    ----------
    cs : np.ndarray, shape (n, d)
        Data coordinates.
    ps : np.ndarray, shape (n, nc)
        Soft data (one-hot encoded categories).
    code : np.ndarray, shape (n,)
        Integer class labels for each data point.
    categories : np.ndarray
        Sorted unique category labels.
    coord_limit_candidates : list of np.ndarray
        List of candidate ``coord_limit`` arrays.  Each array is a
        valid input to ``probatablecalc``.
    kstd_candidates : list of float
        List of candidate kernel bandwidths.
    nsmax : int
        Maximum neighbours for the estimator.
    dmax : float
        Maximum search radius (metres).
    base_options : dict
        Options dict passed to *method_fn*.
    method_fn : callable or None, optional
        Categorical estimator function (default ``MCPcatPdf``).
        Signature: ``method_fn(ck, cs, ps, dmodel, Pmodel, nsmax,
        dmax, options=...)``.
    n_fit : int, optional
        Number of fitted distance grid points (default 80).
    opts_fit : list or None, optional
        ``probatablefit`` options triplet (default ``[0, 0, 1]``).
    regularize : bool, optional
        Whether to regularise Pmodel (default ``True``).
    verbose : bool, optional
        Print progress (default ``True``).

    Returns
    -------
    results : list of dict
        One entry per candidate combination, each containing:
        ``'coord_limit'``, ``'kstd'``, ``'n_bins'``, ``'accuracy'``,
        ``'mean_ll'``, ``'dmodel'``, ``'Pmodel'``, ``'label'``.
    best : dict
        The entry from *results* with the highest ``'mean_ll'``.

    Examples
    --------
    >>> from stamps.stamps.stats.dependence.probatablecalc import adaptive_bins
    >>> cl_adapt = adaptive_bins(cs, n_bins=12, method='equal_pairs')
    >>> cl_uniform = np.concatenate([[0], np.linspace(500, 25000, 20)])
    >>> results, best = loocv_tune_pmodel(
    ...     cs, ps, code, categories,
    ...     coord_limit_candidates=[cl_adapt, cl_uniform],
    ...     kstd_candidates=[500, 750, 1000],
    ...     nsmax=8, dmax=8000, base_options=base_options)
    """
    from .empirical import probatablecalc  # local import to avoid circular

    if method_fn is None:
        # Lazy import to avoid circular dependency
        from ...bme.BMEcatPdf import MCPcatPdf
        method_fn = MCPcatPdf

    if opts_fit is None:
        opts_fit = [0, 0, 1]

    n = cs.shape[0]
    results = []

    total = len(coord_limit_candidates) * len(kstd_candidates)
    counter = 0

    for cl_idx, cl in enumerate(coord_limit_candidates):
        cl = np.asarray(cl, dtype=float)
        n_bins_label = len(cl) - 1   # number of bins (excl zero bin)

        # Compute empirical tables once per coord_limit
        D_emp, P_emp, O_emp = probatablecalc(cs, ps, cl)

        for kstd in kstd_candidates:
            counter += 1
            label = f'bins={n_bins_label} kstd={kstd:.0f}'
            if verbose:
                print(f'  [{counter}/{total}] {label} ...', end='', flush=True)

            # Build Pmodel
            try:
                dmodel, Pmodel, _info = _pmodel_pipeline(
                    D_emp, P_emp, O_emp,
                    n_fit=n_fit, kstd=kstd, n_bins=n_bins_label,
                    opts=opts_fit, regularize=regularize, verbose=False)
            except Exception as e:
                if verbose:
                    print(f' FAILED ({e})')
                continue

            # LOOCV
            opts_loo = {**base_options, 'show_progress': 0}
            correct = 0
            log_liks = []

            for i in range(n):
                ck_i   = cs[i:i+1, :]
                cs_loo = np.delete(cs, i, 0)
                ps_loo = np.delete(ps, i, 0)

                try:
                    pk_i = method_fn(ck_i, cs_loo, ps_loo,
                                     dmodel, Pmodel, nsmax, dmax,
                                     options=opts_loo)
                except Exception:
                    continue

                pred_cat = categories[np.argmax(pk_i[0])]
                true_cat = code[i]
                correct += (pred_cat == true_cat)

                true_idx = np.where(categories == true_cat)[0][0]
                prob_true = np.clip(pk_i[0, true_idx], 1e-15, 1.0)
                log_liks.append(np.log(prob_true))

            acc = correct / n if n > 0 else 0.0
            mll = float(np.mean(log_liks)) if log_liks else -np.inf

            entry = dict(coord_limit=cl, kstd=kstd, n_bins=n_bins_label,
                         accuracy=acc, mean_ll=mll,
                         dmodel=dmodel, Pmodel=Pmodel, label=label)
            results.append(entry)

            if verbose:
                print(f'  acc={acc:.3f}  ll={mll:.3f}')

    # Find best by mean log-likelihood
    if results:
        best = max(results, key=lambda x: x['mean_ll'])
    else:
        best = None

    if verbose and best is not None:
        print(f'\n  ★ Best: {best["label"]}  '
              f'acc={best["accuracy"]:.3f}  ll={best["mean_ll"]:.3f}')

    return results, best