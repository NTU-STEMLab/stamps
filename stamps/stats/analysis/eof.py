# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:57:18 2015

@author: hdragon689
"""
import numpy as np
import warnings
import scipy.sparse.linalg as ssl  
import scipy.linalg

from scipy.linalg import svd, diagsvd


def _robust_svd(a, full_matrices=True):
    """SVD with automatic gesvd fallback when dgesdd fails."""
    try:
        return np.linalg.svd(a, full_matrices=full_matrices)
    except np.linalg.LinAlgError:
        warnings.warn(
            "[STARBME] np.linalg.svd (dgesdd) failed; "
            "falling back to scipy.linalg.svd (gesvd).",
            RuntimeWarning, stacklevel=2)
        return scipy.linalg.svd(
            a, full_matrices=full_matrices, lapack_driver='gesvd')
try:
    from sklearn.decomposition import FastICA 
    from sklearn import mixture
except Exception as e:
    print('warning: some functions cannot be used because:', e)


def eof(X, n='all', norm=0, norms_direc=0, method='svd'):

    '''
    EOF - computes EOF of a matrix.
    
    Usage: [L, lambda, PC, EOFs, EC, error, norms] = EOF( M, num, norm, ... )
    
    Input: 
    M          m by n          the matrix with m obs and n items on which to  
                                                         perform the EOF.  
    num        scalar/string   the number of EOFs to return.  If num='all'(default), 
                                                         then all EOFs are returned.  
    norm       bool            normlaization flag. If it is true, then all time series 
                                                         are normalized by their standard deviation before EOFs 
                                                         are computed.  Default is false.  In this case,the 7-th 
                                                         output argument will be the standard deviations of each column.
    norm_direc integer         to designate the column==0 or row==1 to be normalized 
    method     string          method to be used for EOF calculation. svd or eig.
                                                         svd refers to singular value decomposition (default)
                                                         and eig refers to eigenvalue decomposition. In the
                                                         case of eig method, M should be a square matrix of 
                                                         the covariance functions to be assessed
    
    Output: 
    
    L       1 by k          1D array for the eigenvalues of the covariance matrix 
                                                    ( i.e. they are normalized by 1/(m-1), where m is the 
                                                    number of rows ).  
    lambdaU k by k          2D array for singular values (k by n for full matrix)
    PC      m by k          unitary matrix having left singular vectors as columns
    EOFs    n by k          principal components. unitary matrix haveing right 
                                                    sigular vectors as rows
    EC      m by k          expansion coefficients
    error   1 by k          1D array of the reconstruction error (L2-norm) for 
                                                    each item
    norms   1 by n          1D array for standard deviation of each item or one                    
    coefficients (PCs in other terminology) and error is the reconstruction
    error (L2-norm).
    
    Remark:
    Data is not detrended before handling.  If needed, perform the detrending before 
    using this function to fix that.
    This code is modified from the Matlab code by David M. Kaplan in 2003 by 
    Hwa-Lung Yu 2015/5/31
    
    '''  

    def rSVD(A, r, t=0, direction='default', Omega=None, n_rand=1, multi=False, queue=None):
        '''
        %-----------------------------------------------
        % Randomized truncated SVD
        % Input:
        %   A: m*n matrix to decompose (Need to be a numpy array)
        %   r: the number of singular values to keep (rank of S)
        %   t (optional): the number used in power (default = 0)
        %   direction (optional): to sample columns (=0) or sample rows (=1)
        %                         (default = 0, if m <= n
        %                                  = 1, if m > n )
        %   Omega (optional): the projection matrix to use in sampling
        %                     (program will compute A * Omega for direction = 0
        %                                           Omega * A for direction = 1)
        %
        % Output: classical output as the builtin svd matlab function
        %-----------------------------------------------
        '''

        m, n = A.shape
        U = np.zeros((m, r, n_rand))
        V = np.zeros((n, r, n_rand))
        S = np.zeros((r, r, n_rand))
        
        for k in range(n_rand):

            if direction == 'default':
                if m <= n:
                    direction = 0
                else:
                    direction = 1

            if direction == 0:
                Omega = np.random.randn(n, r)/np.sqrt(n)
            else:
                Omega = np.random.randn(m, r)/np.sqrt(m)

            if direction == 1:
                A = A.T
                Omega = Omega.T

            # Compute SVD
            if t == 0:
                Y = np.dot(A, Omega)
            else:
                Y = np.dot(np.dot(A, A.T)**t, np.dot(A, Omega))
            Q, _ = linalg.qr(Y, mode='economic')
            B = np.dot(Q.T, A)
            U_tild, S_temp, Vh_tild = linalg.svd(B, full_matrices=0)

            if direction == 0:
                U_temp = np.dot(Q, U_tild)
                Vh_temp = Vh_tild
            else:
                Vh_temp = np.dot(Q, U_tild)
                U_temp = Vh_tild

            V_temp = Vh_temp.T
            S_temp = linalg.diagsvd(S_temp, r, r)
            U[:, :, k] = U_temp[:, :r]
            S[:, :, k] = S_temp[:r, :r]
            V[:, :, k] = V_temp[:, :r]

        if multi:
            queue.put([U, S, V])
        else:
            return U, S, V

    def combine_orth(W_0, U_tild, tol_t=1e-2, tol_F=1e-4, iter_max=100):
        '''
        %-----------------------------------------------
        % Combine U_i to W
        % Input:
        %   W_0: the initial matrix used in updating. (Maybe U_0 is good.)
        %   U_tild: U_tild(:, :, i) = U_i
        %   tol_t(optional): stepsize < tol_t will stop. (default = 1e-2)
        %   tol_F(optional): 0 < (F_new -F_old)/F_old < tol_F will stop (default = 1e-3)
        %   iter_max(optional): max number of iteration.(default = 100)
        %
        % Output:
        %   W: the combined result
        %   n_iter: number of iterations
        %   W_record: W_record(:, :, i) = i_iter-th combined result
        %-----------------------------------------------
        '''
        
        m, r, n = U_tild.shape
        U_tild = np.reshape(U_tild, (m, n*r))
        W_record = np.zeros((m, r, iter_max))

        W = W_0
        GF = np.dot(U_tild, np.dot(U_tild.T, W))/n
        n_iter = 0
        F_old = np.trace(np.dot(GF.T, W))
        t = 1
        
        while n_iter < iter_max:
            # Calculate the gradient
            U = np.hstack([GF, W])
            V = np.hstack([W, -GF])
            Y = W + t*(GF - np.dot(W, np.dot(W.T, GF)))/2
            X = np.dot(np.linalg.inv(-np.eye(2*r) + t*np.dot(V.T, U))/2, np.dot(V.T, Y))
            #  Updating
            W_new = Y - t*np.dot(U, X)/2
            
            F_new = np.trace(np.dot(np.dot(W_new.T, U_tild), np.dot(U_tild.T, W_new))/n)

            while F_new < F_old:
                t = t/2
                if t < tol_t:
                    print ('The step size is too small.')
                    W_record[:, :, n_iter+1:-1] = []
                    break
                Y = W + t/2*(GF - np.dot(W, np.dot(W.T, GF)))
                X = np.dot(np.linalg.inv(np.eye(2*r) + t/2*np.dot(V.T, U)), np.dot(V.T, Y))
                W_new = Y - t/2*np.dot(U, X)

                F_new = np.trace(np.dot(np.dot(W_new.T, U_tild), np.dot(U_tild.T, W_new))/n)

            if (F_new-F_old)/F_old < tol_F:
                print ("The change rate(%f) is too small." % ((F_new-F_old)/F_old))
                break
            else:
                W = W_new
                GF = np.dot(U_tild, np.dot(U_tild.T, W))/n
                F_old = F_new
                W_record[:, : , n_iter] = W
                n_iter += 1
                
        W_record = np.delete(W_record, np.s_[n_iter:iter_max], 2)

        return W, n_iter, W_record

    # EOF start

    s = X.shape
    ss = np.min(s)    
    
    # Normalized by standard deviation if desired

    if norm:
        norms = np.std(X,axis=norms_direc)  
    else:
        norms = np.ones(s[1])
                
    X = np.dot(X, np.diag(1/norms))

    # Using SVD to solve the eigen problem of covariance matrix
    
    if method=='svd':
        # To solve with full matrices
        if n=='all' or n>=ss:
            U, lambdaU, Vh = svd(X, full_matrices=True) 
            PCs = U
            EOFs = Vh.T
        else:
        # To solve with the first n singular vectors/values
            U, lambdaU, Vh = ssl.svds(X, n)
            lambdaU = lambdaU[::-1]
            PCs = U[:,::-1]
            EOFs = Vh[::-1,:].T
        
        L = lambdaU**2/(s[0]-1)

        # Check computation errors
        if n=='all':
            lambdaU2 = diagsvd(lambdaU, s[0], s[1])
            ECs = PCs.dot(lambdaU2)
            diff = X-np.dot(ECs, EOFs.T)
            error = np.sqrt(sum(sum(diff*np.conj(diff))))  
        else:
            lambdaU2 = np.diag(lambdaU)
            ECs = np.dot(PCs, lambdaU2)
            diff = X-np.dot(ECs,EOFs.T)
            error = np.sqrt(sum(sum(diff*np.conj(diff))))

    else:
        CU = 1./(s[0]-1)*X.T.dot(X)
        L, EOFs = np.linalg.eig(CU)
        if np.any(np.iscomplex(EOFs)):
            EOFs = EOFs.real
            L = L.real
        ECs = (EOFs.T.dot(X.T)).T
        lambdaU2 = np.sqrt(L)    
     # L= lambdaU2**2/(s[0]-1)
        lambdaU2[np.where(np.isnan(lambdaU2))] = 0.
        PC = ECs*1/lambdaU2 # the dimension of L should be checked
        lambdaU2 = np.diag(lambdaU2)
        diff = X-np.dot(ECs,EOFs.T)
        error = np.sqrt(sum(sum(diff*np.conj(diff)))) 

    return L, lambdaU2, PCs, EOFs, ECs, error, norms

''' Extended EOF '''

def eeof(M, tl, n='all'):
    '''
    Extended EOF - computes EOF of a matrix.
    
    Usage: [L, lambda, PC, EOFs, EC, error, norms] = EEOF( M, num, norm, ... )
    
    Input: 
    M       n by p        the matrix with n obs and p items on which to perform 
                                                the EOF.  
    tl      integer       Window size to assess the time lags. In this case, the 
                                                space-time dynamic pattern of size of tl by p is assessed                 

    Output:
    EEOF    list          the n-tlag principal extended EOFs. Each has size of 
                                                tlag by p. For the space-time case, p is the number of 
                                                stations
    EEC     n-tl by n-tl  Each column contains the extended EC for the corresponding
                                                extended EOF 
    lam     1 by n-tl     1D array of the eigenvalues of the corresponding n-tl EEOFs                  
 
    '''
    n,p=M.shape
    # XX=np.zeros((w,p*(n-w)))
    XX=np.zeros((n-tl,p*tl))
    for i in range(n-tl):
        # M2=M[i:n-w+i,:]
        M2=M[i:tl+i,:]
        Mv=M2.reshape(1,M2.size,order='F')
        XX[i,:]=Mv
 
 # lam, sigv, PC, EOFs, EC, diff, error=sreof(XX)
    EOFs,EC,sigv,lam=sreof(XX)
    
    # to create the high-dimensional matrix with consistant format of M
    EEOF=[]
    for i in range(EOFs.shape[1]):
        EEOF.append(EOFs[:,i].reshape(tl,p))

    EEC=EC
    
    return EEOF, EEC, lam

'''
def eeofplot(EEOF,EEC): 
    '''

        
    

'''
def bveof(X,Y)
to estimate the bivariate EOFs
'''
     
''' varimax function'''   
def varimax_(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    '''
    This code is obtained from
    https://en.wikipedia.org/wiki/Talk%3aVarimax_rotation
    '''  
#  from numpy import eye, asarray, dot, sum
#  from numpy.linalg import svd
    p, k = Phi.shape
    R = np.eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = svd(np.dot(Phi.T, Lambda**3 - (gamma/p) * Lambda.dot(np.diag(np.diag(Lambda.T.dot(Lambda))))))
        R = np.dot(u,vh)
        d = np.sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return np.dot(Phi, R), R


def sreof( U, m='all', norm=0, *args ):
    '''
    Scaled and rotaed EOF - computes scaled and rotated EOF of a matrix. 
    
    Usage: [EOF,EC,lam,eigval] = SREOF( M, num, norm, ... )
    
    Input: 
    M     m by n          the 2D array with m obs and n items on which to perform 
                                                the EOF.  
    num   scalar/string   the number of EOFs to return.  If num='all'(default), 
                                                then all EOFs are returned.  
    norm  bool            normlaization flag. If it is true, then all time series 
                                                are normalized by their standard deviation before EOFs 
                                                are computed.  Default is false.  In this case,the 7-th 
                                                output argument will be the standard deviations of each column.
    Others 
    ... are extra arguments to be given to the svds function.  These will
    be ignored in the case that all EOFs are to be returned, in which case
    the svd function is used instead. Use these with care. 
    Try scipy.sparse.linalg.svds?
    
    Output: 
    
    EOF     m by num        the EOF results are listed in the columns  
    EC      n by num        the correponding EC results are listed in the columns
    lam     1 by num        1D array of the singular values the EOFs
    eigval  1 by num        1D array of the eigenvalues of the EOFs

    
    Remark:
    The result is re-scaled from the original EOF results by considering the 
    eigenvalues of each EOFs. The rotation is performed on the scaled EOFs.
    By doing this, the EOF rotation can have relatively low impacts from the number
    of EOFs to be rotated and therefore the results can be stabilized. The 
    rotation is based upon the 

    '''  

    L,lambdaU,PC,EOFs,EC,error,norms=eof(U,m,norm,*args)
    
    SEOF = EOFs.dot(lambdaU.T[:,:EOFs.shape[1]]) # scaled EOFs by singular values
    
    # rotate the EOFs by using varimax method  
#  rotm,REOF,__= varimax(SEOF) # this is a slower varimax
#  REOF=np.dot(EOFs,rotm)
    REOF, rotm = varimax_(SEOF, gamma = 1.0, q = 50, tol = 1e-7)
    REOF = np.dot(EOFs[:,:rotm.shape[0]],rotm)
    # rotate the corresponding ECs
    NewEOFs=REOF
    NewEC=np.dot(EC[:,:rotm.shape[0]],np.linalg.inv(rotm.T))
#  NewEC=np.dot(EC,np.linalg.inv(rotm.T))

    
    if (type(m) is str) or (m == 'all'):
        m = rotm.shape[0]

    for i in range(m):
        maxi = np.where((np.abs(NewEOFs[:,i]).max()==np.abs(NewEOFs[:,i])))
        signi = float(NewEOFs[:,i][maxi]/np.abs(NewEOFs[:,i][maxi]))
        NewEOFs[:,i] = signi*NewEOFs[:,i]
        NewEC[:,i] = signi*NewEC[:,i]
        
    return NewEOFs, NewEC, lambdaU, L


    
def srpca(U, m='all', norm=0, *args):
    '''
    Scaled and rotaed EC - computes scaled and rotated EC of a matrix. 
    
    Usage: [L, lambda, PC, EOFs, EC, error, norms] = SRPCA( M, num, norm, ... )
    
    Input: 
    M     m by n          the 2D array with m obs and n items on which to perform 
                                                the EOF.  
    num   scalar/string   the number of EOFs to return.  If num='all'(default), 
                                                then all EOFs are returned.  
    norm  bool            normlaization flag. If it is true, then all time series 
                                                are normalized by their standard deviation before EOFs 
                                                are computed.  Default is false.  In this case,the 7-th 
                                                output argument will be the standard deviations of each column.
    Others 
    ... are extra arguments to be given to the svds function.  These will
    be ignored in the case that all EOFs are to be returned, in which case
    the svd function is used instead. Use these with care. 
    Try scipy.sparse.linalg.svds?
    
    Output: 
    
    EOF   m by num        the EOF results are listed in the columns  
    EC    n by num        the correponding EC results are listed in the columns 

    
    Remark:
    The result is re-scaled from the original EOF results by considering the 
    eigenvalues of each EOFs. The rotation is performed on the scaled EOFs.
    By doing this, the EOF rotation can have relatively low impacts from the number
    of EOFs to be rotated and therefore the results can be stabilized. The 
    rotation is based upon the 

    '''  
        
    L,lambdaU,PC,EOFs,EC,error,norms=eof(U,m,norm,*args)

    SEC=EC*lambdaU # scaled EOFs by eigenvalues

    # rotate the EOFs by using varimax method  
    rotm,REC, eps= varimax(SEC)
    REC=np.dot(EC,rotm)

    # rotate the corresponding ECs
    NewEC=REC
    NewEOFs=np.dot(EOFs,np.linalg.inv(rotm.T))
    Newlambda=lambdaU
    
    for i in range(m):
        maxi=np.where((np.abs(NewEC[:,i]).max()==np.abs(NewEC[:,i])))
        signi=float(NewEC[:,i][maxi]/np.abs(NewEC[:,i][maxi]))
        NewEOFs[:,i]=signi*NewEOFs[:,i]
        NewEC[:,i]=signi*NewEC[:,i]
        
    return NewEOFs, NewEC, Newlambda


def varimax(amat,target_basis=None):
    '''
    [ROTM,OPT_AMAT] = varimax(AMAT,TARGET_BASIS)
    
    Gives a (unnormalized) VARIMAX-optimal rotation of matrix AMAT:
    The matrix  AMAT*ROTM  is optimal according to the VARIMAX
    criterion, among all matricies of the form  AMAT*R  for  
    R  an orthogonal matrix. Also (optionally) returns the rotated AMAT matrix
    OPT_AMAT = AMAT*ROTM.
    
    Uses the standard algorithm of Kaiser(1958), Psychometrika.
    
    Inputs:
    
    AMAT          N by K     matrix of "K component loadings"
    TARGET_BASIS  N by N     (optional) an N by N matrix whose columns 
                                                     represent a basis toward which the rotation will 
                                                     be oriented; the default is the identity matrix 
                                                     (the natural coordinate system); this basis need 
                                                     not be orthonormal, but if it isn't, it should be
                                                     used with great care!
    Outputs: 
    
    ROTM         K by K      Optimizing rotation matrix
    OPT_AMAT     N by K      Optimally rotated matrix  (AMAT*ROTM)
    
    Modified by Trevor Park in April 2002 from an original file by J.O. Ramsay  
    Modified by H-L Yu into python code in June 2015 
    '''
    MAX_ITER=50
    EPSILON=1e-7
    
    amatd=amat.shape
    
    if np.size(amatd) != 2:
        raise RuntimeError('AMAT must be two-dimensional')
        
    n=amatd[0]
    k=amatd[1]
    rotm=np.eye(k)
    
    if k==1:
        return
        
    if target_basis==None:
        target_basis_flag=0
        target_basis=np.eye(n)    
    else:
        target_basis_flag=1
        if np.size(target_basis.shape) !=2:
            raise RuntimeError('TARGET_BASIS must be two-dimensional')
        if target_basis.shape==(n,n):
            amat=np.dot(np.linalg.inv(target_basis),amat)
        else:
            raise RuntimeError('TARGET_BASIS must be a basis for the column space')
    
    varnow=np.sum(np.var(amat**2,0))
    not_converged=1
    iterx=0
    while not_converged and iterx < MAX_ITER:
        for j in range(k-1):
            for l in range(j+1,k):
                # Calculate optimal 2-D planar rotation angle for columns j,l
                phi_max=np.angle(n*np.sum(np.vectorize(complex)(amat[:,j],amat[:,l])**4) \
                                 - np.sum(np.vectorize(complex)(amat[:,j],amat[:,l])**2)**2)/4
                sub_rot = np.array([[np.cos(phi_max),-np.sin(phi_max)],\
                                            [np.sin(phi_max),np.cos(phi_max)]])
                amat[:,[j,l]]=np.dot(amat[:,[j,l]],sub_rot)
                rotm[:,[j,l]]=np.dot(rotm[:,[j,l]],sub_rot)   
                
        varold = varnow
        varnow = np.sum(np.var(amat**2,0))      
    
        if varnow==0:
            return 
        
        not_converged = ((varnow-varold)/varnow > EPSILON)
        iterx= iterx +1
        
    if iterx >= MAX_ITER:
        warnings.warn('Maximum number of iterations reached in function')
    
    if target_basis_flag:  
        opt_amat=target_basis*amat
    else:
        opt_amat=np.dot(amat,rotm)
        
    eps=(varnow-varold)/varnow  

    return rotm, opt_amat, eps  

def princomp(A):
        """ 
        coeff,score,latent=princomp(A)   
        
        This function performs principal components analysis (PCA) 
        on the n-by-p data matrix A. Rows of A correspond to observations, 
        columns to variables. 
        
        Input:
        
        A       n by p      matrix of n observations of p variables
         
        Output:  
        
        coeff   p by p      a p-by-p matrix, each column containing coefficients 
                                                for one principal component.
        score   n by p      the principal component scores; that is, the 
                                                representation of A in the principal component space. 
                                                Rows of SCORE correspond to observations, columns to 
                                                components.
        latent : 
                a vector containing the eigenvalues 
                of the covariance matrix of A.
                
         Ref: this function is downloaded from the link
         http://glowingpython.blogspot.tw/2011/07/principal-component-analysis-with-numpy.html
         """
        # computing eigenvalues and eigenvectors of covariance matrix
        M = (A-np.mean(A.T,axis=1)).T # subtract the mean (along columns)
        [latent,coeff] = np.linalg.eig(np.cov(M)) # attention:not always sorted
        score = np.dot(coeff.T,M) # projection of the data in the new space
        return coeff,score,latent

'''Independent component analysis'''

def stica(M, num='all', norm=0, norms_direc=0, ortho='s'):
    ''' ICA for S/T dataset that decomposes independent S/T signals from S/T 
            dataset  
    stica - computes ICAs of a matrix.
    
    Usage: [L, lambda, PC, EOFs, EC, error, norms] = stica( M, num, norm, ... )
    
    Input: 
    M          m by n          the matrix with m obs and n items on which to  
                                                         perform the ICA.  
    num        scalar/string   the number of ICAs to return.  If num='all'(default), 
                                                         then all ICAs are returned.  
    norm       bool            normlaization flag. If it is true, then all time series 
                                                         are normalized by their standard deviation before EOFs 
                                                         are computed.  Default is false.  In this case,the 7-th 
                                                         output argument will be the standard deviations of each column.
    norm_direc integer         to designate the column==0 or row==1 to be normalized 
    ortho      string          method to identify orthogonal vectors, 's' denotes
                                                         the spatial independent functions (default) 
                                                         and 't' denotes the temporal independent functions

    Output: 
    EOFs    n by k          Spatial functions which are independent while ortho is 
                                                    's'
    EC      m by k          Temporal functions which are independent while ortho is 
                                                    't'
    lambda  k by 1          1D array for singular values (k by n for full matrix) 
    
    Note:  Singular values are approximated by the norm of the mixing matrix of 
    the ICA. This should be further checked. 
                                                    
            
    ''' 

    m,n=M.shape
    # ss=np.min([m,n])      
    # Normalized by standard deviation if desired

    if norm:
        norms=np.std(M,axis=norms_direc)  
    else:
        norms=np.ones(n)
                
    M=np.dot(M, np.diag(1/norms))  
    
    if num == 'all':
        n_com=n
    else:
        n_com=num
    
    ica = FastICA(n_components=n_com)
    if ortho == 't':
        ICAs=ica.fit_transform(M)
        eofICA=ica.mixing_
        lambdaM=np.linalg.norm(eofICA,axis=0)
    else:
        eofICA=ica.fit_transform(M.T)
        ICAs=ica.mixing_
        lambdaM=np.linalg.norm(ICAs,axis=0)
    
    return eofICA,ICAs,lambdaM 
    
def srica(U, m='all', norm=0, ortho='s', *args):
    '''
    Scaled and rotaed ICA - computes scaled and rotated ICA of a matrix. 
    
    Usage: [EOF,EC,lam,eigval] = SREOF( M, num, norm, ... )
    
    Input: 
    M     m by n          the 2D array with m obs and n items on which to perform 
                                                the EOF.  
    num   scalar/string   the number of EOFs to return.  If num='all'(default), 
                                                then all EOFs are returned.  
    norm  bool            normlaization flag. If it is true, then all time series 
                                                are normalized by their standard deviation before EOFs 
                                                are computed.  Default is false.  In this case,the 7-th 
                                                output argument will be the standard deviations of each column.
    ortho      string     method to identify orthogonal vectors, 's' denotes
                                                the spatial independent functions (default) 
                                                and 't' denotes the temporal independent functions

    Output: 
    
    EOF     m by num        the EOF (spatial function) results are listed in the 
                                                    columns  
    EC      n by num        the correponding EC (temporal function) results are 
                                                    listed in the columns
    lam     1 by num        1D array of the singular values the EOFs/ECs which are 
                                                    with respect to 's' or 't'
    
    Remark:
    The result is re-scaled from the original EOF results by considering the 
    eigenvalues of each EOFs. The rotation is performed on the scaled EOFs.
    By doing this, the EOF rotation can have relatively low impacts from the number
    of EOFs to be rotated and therefore the results can be stabilized. The 
    rotation is based upon the 

    '''  
    
    EOFs,EC,lambdaU=stica(U, num=m,norm=norm,ortho=ortho)
    
    if ortho == 's': 
        SEOF=EOFs*lambdaU#EOFs.dot(np.diag(lambdaU)) # scaled EOFs by eigenvalues

        # rotate the EOFs by using varimax method  
        # rotm1,REOF1,__= varimax(SEOF) # this is a slower varimax
        REOF,rotm=varimax_(SEOF, gamma = 1.0, q = 50, tol = 1e-7)
        REOF=np.dot(EOFs[:,:rotm.shape[0]],rotm)
    
        # rotate the corresponding ECs
        NewEOFs=REOF
        NewEC=np.dot(EC[:,:rotm.shape[0]],np.linalg.inv(rotm.T))
    
        if (type(m) is str) or (m == 'all'):
            m=rotm.shape[0]

        for i in range(m):
            maxi=np.where((np.abs(NewEOFs[:,i]).max()==np.abs(NewEOFs[:,i])))
            signi=float(NewEOFs[:,i][maxi]/np.abs(NewEOFs[:,i][maxi]))
            NewEOFs[:,i]=signi*NewEOFs[:,i]
            NewEC[:,i]=signi*NewEC[:,i]
    else:
        SEC=EC*lambdaU # scaled EOFs by eigenvalues

        # rotate the EOFs by using varimax method  
        REC,rotm = varimax_(SEC, gamma = 1.0, q = 50, tol = 1e-7)
        REC=np.dot(EC,rotm) #np.dot(EC[:,:rotm.shape[0]],rotm)

        # rotate the corresponding ECs
        NewEC=REC
        NewEOFs=np.dot(EOFs,np.linalg.inv(rotm.T))
    
        for i in range(m):
            maxi=np.where((np.abs(NewEC[:,i]).max()==np.abs(NewEC[:,i])))
            signi=float(NewEC[:,i][maxi]/np.abs(NewEC[:,i][maxi]))
            NewEOFs[:,i]=signi*NewEOFs[:,i]
            NewEC[:,i]=signi*NewEC[:,i]    
        
    return NewEOFs,NewEC,lambdaU 

def eofclass(EOFs,num='all'):
    '''
    To classify the identified EOF locations.  
    
    EOFclass,means,covs,weights,logprob,prob_comps,clf=eofclass(EOFs,num='all')   
    
    Input:
    EOFs      m by n        the EOF (spatial function) results are listed in the 
                                                    columns
    num       scalar/string number of EOFs functions to be classified. Default is 
                                                    'all' that all  functions will be used                           
    
    Output:
    EOFclass  m by num      indicators for the identified regions. 1 denotes the 
                                                    identified locations; otherwise, 0 is shown 
    means     num by 2      the mean of EOF values of the two classified regions.
    covs      num by 2      the variances of EOF values of the two classified 
                                                    regions.
    bound     num by 1      a 1-D array denotes the boundary values to classify 
                                                    EOF values in the num of EOF functions.
    
    Note: The classification is performed by using the two-component Gaussian 
    mixture model to reveal the mean and variance of the close-to-zero EOF region.
    t-test is used to identify the locations which are significantly deviate from 
    zeros.
    '''  
    
    m,n=EOFs.shape
    
    if num == 'all':
        num=n
    
    EOFclass=np.zeros([m,num])
    means=np.zeros([num,2])
    covs=np.zeros([num,2])
    weights=np.zeros([num,2])
#  logprob=[None]*num
#  prob_comps=[None]*num
    clf=[None]*num
    bound=np.zeros(num)
    for i in range(num):
        clf[i]=mixture.GMM(n_components=2)
        clf[i].fit(EOFs[:,i:i+1])
        weights[i,0:2]=clf[i].weights_
        means[i,0:2]=clf[i].means_.reshape(2,)
        covs[i,0:2]=clf[i].covars_.reshape(2,)
        signs=means[i,0:2]/np.abs(means[i,0:2])
        popmean=np.min(np.abs(means[i,0:2]))
        idx=np.where(np.abs(means[i,0:2])==popmean)
        popmean=popmean*signs[idx]
        bound[i]=np.sqrt(covs[i,idx])*1.95996
        EOFclass[np.where(EOFs[:,i]>=bound[i]),i:i+1]=1
                
        
#    EOFclass[:,i:i+1]=clf[i].predict(EOFs[:,i:i+1]).reshape(EOFs[:,i].size,1)
#    logprob[i],prob_comps[i]=clf[i].score_samples(EOFs[:,i:i+1])
#    idxmax=np.where(EOFs[:,i]==EOFs[:,i].max())
#    if EOFclass[idxmax,i]==0:
#      EOFclass[np.where(EOFclass[:,i]==0),i]=2
#      EOFclass[np.where(EOFclass[:,i]==1),i]=0
#      EOFclass[np.where(EOFclass[:,i]==2),i]=1
#      weights[i,0],weights[i,1]=weights[i,1],weights[i,0]
#      means[i,0],means[i,1]=means[i,1],means[i,0]
#      covs[i,0],covs[i,1]=covs[i,1],covs[i,0]
#      prob_comps[i][:,[0,1]]=prob_comps[i][:,[1,0]]      
                
#  return EOFclass,means,covs,weights,logprob,prob_comps,clf  
    return EOFclass,means,covs,bound
    
## EOF classification
#EOF1=EOFdf.values[:,0:1]
#clf = mixture.GMM(n_components=2)
#clf.fit(EOF1)
#wei=clf.weights_
#mea=clf.means_
#cov=clf.covars_
#EOF1_classlabel=clf.predict(EOF1).reshape(EOF1.size,1) # preform classification on EOF1
#xxx=np.linspace(EOF1.min(),EOF1.max())
#xxx=xxx.reshape(xxx.size,1)
#plt.figure()
#plt.hist(EOF1,bins=20,normed=True)
#plt.plot(xxx,np.exp(clf.score_samples(xxx)[0]))
#plt.show()


'''
#!/usr/bin/env python
# -*- coding: ascii -*-

"""Higher order singular value decomposition routines

as introduced in:
        Lieven de Lathauwer, Bart de Moor, Joos Vandewalle,
        'A multilinear singular value decomposition',
        SIAM J. Matrix Anal. Appl. 21 (4), 2000, 1253-1278

implemented by Jiahao Chen <jiahao@mit.edu>, 2010-06-11

Disclaimer: this code may or may not work.
"""

__author__ = 'Jiahao Chen <jiahao@mit.edu>'
__copyright__ = 'Copyright (c) 2010 Jiahao Chen'
__license__ = 'Public domain'

try:
        import numpy as np
except ImportError:
        print("Error: HOSVD requires numpy")
        raise ImportError



def unfold(A,n):
        """Computes the unfolded matrix representation of a tensor

        Parameters
        ----------

        A : ndarray, shape (M_1, M_2, ..., M_N)

        n : (integer) axis along which to perform unfolding,
                                    starting from 1 for the first dimension

        Returns
        -------

        Au : ndarray, shape (M_n, M_(n+1)*M_(n+2)*...*M_N*M_1*M_2*...*M_(n-1))
                 The unfolded tensor as a matrix

        Raises
        ------
        ValueError
                if A is not an ndarray

        LinAlgError
                if axis n is not in the range 1:N

        Notes
        -----
        As defined in Definition 1 of:

                Lieven de Lathauwer, Bart de Moor, Joos Vandewalle,
                "A multilinear singular value decomposition",
                SIAM J. Matrix Anal. Appl. 21 (4), 2000, 1253-1278
        """

        if type(A) != type(np.zeros((1))):
                print("Error: Function designed to work with numpy ndarrays")
                raise ValueError
        
        if not (1 <= n <= A.ndim):
                print("Error: axis %d not in range 1:%d" % (n, A.ndim))
                raise np.linalg.LinAlgError

        s = A.shape

        m = 1
        for i in range(len(s)):
                m *= s[i]
        m //= s[n-1]

        #The unfolded matrix has shape (s[n-1],m)
        Au = np.zeros((s[n-1],m))

        index = [0]*len(s)

        for i in range(s[n-1]):
                index[n-1] = i
                for j in range(m):
                        Au[i,j] = A[tuple(index)]

                        #increment (n-1)th index first
                        index[n-2] += 1

                        #carry over: exploit python's automatic looparound of addressing!
                        for k in range(n-2,n-1-len(s),-1):
                                if index[k] == s[k]:
                                        index[k-1] += 1
                                        index[k] = 0

        return Au



def fold(Au, n, s):
        """Reconstructs a tensor given its unfolded matrix representation

        Parameters
        ----------

        Au : ndarray, shape (M_n, M_(n+1)*M_(n+2)*...*M_N*M_1*M_2*...*M_(n-1))
                 The unfolded matrix representation of a tensor

        n : (integer) axis along which to perform unfolding,
                                    starting from 1 for the first dimension

        s : (tuple of integers of length N) desired shape of resulting tensor

        Returns
        -------
        A : ndarray, shape (M_1, M_2, ..., M_N)

        Raises
        ------
        ValueError
                if A is not an ndarray

        LinAlgError
                if axis n is not in the range 1:N

        Notes
        -----
        Defined as the natural inverse of the unfolding operation as defined in Definition 1 of:

                Lieven de Lathauwer, Bart de Moor, Joos Vandewalle,
                "A multilinear singular value decomposition",
                SIAM J. Matrix Anal. Appl. 21 (4), 2000, 1253-1278
        """

        m = 1
        for i in range(len(s)):
                m *= s[i]
        m //= s[n-1]

        #check for shape compatibility
        if Au.shape != (s[n-1], m):
                print("Wrong shape: need", (s[n-1], m), "but have instead", Au.shape)
                raise np.linalg.LinAlgError

        A = np.zeros(s)

        index = [0]*len(s)

        for i in range(s[n-1]):
                index[n-1] = i
                for j in range(m):
                        A[tuple(index)] = Au[i,j]

                        #increment (n-1)th index first
                        index[n-2] += 1

                        #carry over: exploit python's automatic looparound of addressing!
                        for k in range(n-2,n-1-len(s),-1):
                                if index[k] == s[k]:
                                        index[k-1] += 1
                                        index[k] = 0

        return A



def HOSVD(A):
        """Computes the higher order singular value decomposition of a tensor

        Parameters
        ----------

        A : ndarray, shape (M_1, M_2, ..., M_N)

        Returns
        -------
        U : list of N matrices, with the nth matrix having shape (M_n, M_n)
                The n-mode left singular matrices U^(n), n=1:N

        S : ndarray, shape (M_1, M_2, ..., M_N)
                The core tensor

        D : list of N lists, with the nth list having length M_n
                The n-mode singular values D^(n), n=1:N

        Raises
        ------
        ValueError
                if A is not an ndarray

        LinAlgError
                if axis n is not in the range 1:N

        Notes
        -----
        Returns the quantities in Equation 22 of:

                Lieven de Lathauwer, Bart de Moor, Joos Vandewalle,
                "A multilinear singular value decomposition",
                SIAM J. Matrix Anal. Appl. 21 (4), 2000, 1253-1278
        """

        Transforms = []
        NModeSingularValues = []

        #--- Compute the SVD of each possible unfolding
        for i in range(len(A.shape)):
                U,D,V = _robust_svd(unfold(A,i+1))
                Transforms.append(np.asmatrix(U))
                NModeSingularValues.append(D)

        #--- Compute the unfolded core tensor
        axis = 1 #An arbitrary choice, really
        Aun = unfold(A,axis)

        #--- Computes right hand side transformation matrix
        B = np.ones((1,))
        for i in range(axis-A.ndim,axis-1):
                B = np.kron(B, Transforms[i])

        #--- Compute the unfolded core tensor along the chosen axis
        Sun = Transforms[axis-1].transpose().conj() * Aun * B

        S = fold(Sun, axis, A.shape)

        return Transforms, S, NModeSingularValues



function [U,S,V] = fsvd(A, k, i, usePowerMethod)
% FSVD Fast Singular Value Decomposition 
% 
%   [U,S,V] = FSVD(A,k,i,usePowerMethod) computes the truncated singular
%   value decomposition of the input matrix A upto rank k using i levels of
%   Krylov method as given in [1], p. 3.
% 
%   If usePowerMethod is given as true, then only exponent i is used (i.e.
%   as power method). See [2] p.9, Randomized PCA algorithm for details.
% 
%   [1] Halko, N., Martinsson, P. G., Shkolnisky, Y., & Tygert, M. (2010).
%   An algorithm for the principal component analysis of large data sets.
%   Arxiv preprint arXiv:1007.5510, 0526. Retrieved April 1, 2011, from
%   http://arxiv.org/abs/1007.5510. 
%   
%   [2] Halko, N., Martinsson, P. G., & Tropp, J. A. (2009). Finding
%   structure with randomness: Probabilistic algorithms for constructing
%   approximate matrix decompositions. Arxiv preprint arXiv:0909.4061.
%   Retrieved April 1, 2011, from http://arxiv.org/abs/0909.4061.
% 
%   See also SVD.
% 
%   Copyright 2011 Ismail Ari, http://ismailari.com.

        if nargin < 3
                i = 1;
        end

        % Take (conjugate) transpose if necessary. It makes H smaller thus
        % leading the computations to be faster
        if size(A,1) < size(A,2)
                A = A';
                isTransposed = true;
        else
                isTransposed = false;
        end

        n = size(A,2);
        l = k + 2;

        % Form a real n×l matrix G whose entries are iid Gaussian r.v.s of zero
        % mean and unit variance
        G = randn(n,l);


        if nargin >= 4 && usePowerMethod
                % Use only the given exponent
                H = A*G;
                for j = 2:i+1
                        H = A * (A'*H);
                end
        else
                % Compute the m×l matrices H^{(0)}, ..., H^{(i)}
                % Note that this is done implicitly in each iteration below.
                H = cell(1,i+1);
                H{1} = A*G;
                for j = 2:i+1
                        H{j} = A * (A'*H{j-1});
                end

                % Form the m×((i+1)l) matrix H
                H = cell2mat(H);
        end

        % Using the pivoted QR-decomposiion, form a real m×((i+1)l) matrix Q
        % whose columns are orthonormal, s.t. there exists a real
        % ((i+1)l)×((i+1)l) matrix R for which H = QR.  
        % XXX: Buradaki column pivoting ile yapılmayan hali.
        [Q,~] = qr(H,0);

        % Compute the n×((i+1)l) product matrix T = A^T Q
        T = A'*Q;

        % Form an SVD of T
        [Vt, St, W] = svd(T,'econ');

        % Compute the m×((i+1)l) product matrix
        Ut = Q*W;

        % Retrieve the leftmost m×k block U of Ut, the leftmost n×k block V of
        % Vt, and the leftmost uppermost k×k block S of St. The product U S V^T
        % then approxiamtes A. 

        if isTransposed
                V = Ut(:,1:k);
                U = Vt(:,1:k);     
        else
                U = Ut(:,1:k);
                V = Vt(:,1:k);
        end
        S = St(1:k,1:k);
end



if __name__ == '__main__':
        print()
        print("Higher order singular value decomposition routines")
        print()
        print("as introduced in:")
        print("    Lieven de Lathauwer, Bart de Moor, Joos Vandewalle,")
        print("    'A multilinear singular value decomposition',")
        print("    SIAM J. Matrix Anal. Appl. 21 (4), 2000, 1253-1278")

        print()
        print("Here are some worked examples from the paper.")

        print()
        print()
        print("Example 1 from the paper (p. 1256)")

        A = np.zeros((3,2,3))

        A[0,0,0]=A[0,0,1]=A[1,0,0]=1
        A[1,0,1]=-1
        A[1,0,2]=A[2,0,0]=A[2,0,2]=A[0,1,0]=A[0,1,1]=A[1,1,0]=2
        A[1,1,1]=-2
        A[1,1,2]=A[2,1,0]=A[2,1,2]=4
        #other elements implied zero

        #test: compute unfold(A,1)
        print()
        print("The input tensor is:")
        print(A)
        print()
        print("Its unfolding along the first axis is:")
        print(unfold(A,1))

        """
        print()
        print()
        print("Example 2 from the paper (p. 1257)""

        A = np.zeros((2,2,2))
        A[0,0,0] = A[1,1,0] = A[0,0,1] = 1
        #other elements implied zero
        """

        """
        print()
        print()
        print("Example 3 from the paper (p. 1257)""

        A = np.zeros((2,2,2))
        A[1,0,0] = A[0,1,0] = A[0,0,1] = 1
        #other elements implied zero
        """
        print()
        print()
        print("Example 4 from the paper (pp. 1264-5)")
        A = np.zeros((3,3,3))

        A[:,0,:] = np.asmatrix([[0.9073, 0.8924, 2.1488],
        [0.7158, -0.4898, 0.3054],
        [-0.3698, 2.4288, 2.3753]]).transpose()

        A[:,1,:] = np.asmatrix([[1.7842, 1.7753, 4.2495],
        [1.6970, -1.5077, 0.3207],
        [0.0151, 4.0337, 4.7146]]).transpose()

        A[:,2,:] = np.asmatrix([[2.1236, -0.6631, 1.8260],
        [-0.0740, 1.9103, 2.1335],
        [1.4429, -1.7495,-0.2716]]).transpose()

        print("The input tensor has matrix unfolding along axis 1:")
        print(unfold(A,1))
        print()

        U, S, D = HOSVD(A)

        print("The left n-mode singular matrices are:")
        print(U[0])
        print()
        print(U[1])
        print()
        print(U[2])
        print()

        print("The core tensor has matrix unfolding along axis 1:")
        print(unfold(S, 1))
        print()

        print("The n-mode singular values are:")
        print(D[0])
        print(D[1])
        print(D[2])
'''


    
# ---------------------------------------------------------------------------
# Backward-compatibility wrappers preserved from stamps_v3
# ---------------------------------------------------------------------------

def meof( X, m='all', centered=True, stack_direc=0 ):
    '''Multivariate Empirical Orthogonal Function - Computing multivariate EOF.

    Preserved for backward compatibility with stamps_v3.
    Internally calls the updated eof() with the new API.
    '''
    n_v = len(X)
    mvX = np.concatenate(X, axis=stack_direc)
    if centered:
        mvX = mvX - mvX.mean(axis=0)
    L, lambdaU, PC, mvEOFs, mvECs, error, norms = eof(mvX, n=m, norm=0)
    return L, lambdaU, mvEOFs, mvECs, n_v


def srmeof( X, m='all', centered=True, stack_direc=0 ):
    '''Scaled and Rotated Multivariate EOF.

    Preserved for backward compatibility with stamps_v3.
    Internally calls the updated sreof() with the new API.
    '''
    n_v = len(X)
    mvX = np.concatenate(X, axis=stack_direc)
    if centered:
        mvX = mvX - mvX.mean(axis=0)
    L, s, NewEOFs, NewECs, explained_variance = sreof(mvX, m=m, norm=0)
    return L, s, NewEOFs, NewECs, n_v


# ---------------------------------------------------------------------------
# Tensor operations: unfold, fold, HOSVD
# (Extracted from the archived HOSVD string block in the original eof.py.
#  Based on Lieven de Lathauwer, Bart de Moor, Joos Vandewalle,
#  "A multilinear singular value decomposition",
#  SIAM J. Matrix Anal. Appl. 21(4), 2000, 1253-1278)
# ---------------------------------------------------------------------------

def unfold(A, n):
    """Compute the mode-n unfolded (matricised) representation of a tensor.

    Parameters
    ----------
    A : numpy.ndarray, shape (M_1, M_2, ..., M_N)
        Input tensor.
    n : int
        Mode axis along which to unfold, **1-based** (first dimension = 1).

    Returns
    -------
    Au : numpy.ndarray, shape (M_n, M_{n+1} * ... * M_N * M_1 * ... * M_{n-1})
        The mode-n unfolding of *A*.

    Raises
    ------
    ValueError
        If *A* is not a numpy ndarray.
    numpy.linalg.LinAlgError
        If *n* is not in the range [1, N].

    Notes
    -----
    Definition 1 of de Lathauwer *et al.* (2000).
    """
    if not isinstance(A, np.ndarray):
        raise ValueError("unfold requires a numpy ndarray.")
    if not (1 <= n <= A.ndim):
        raise np.linalg.LinAlgError(
            "axis %d not in range 1:%d" % (n, A.ndim))

    s = A.shape
    m = 1
    for i in range(len(s)):
        m *= s[i]
    m //= s[n - 1]

    Au = np.zeros((s[n - 1], m))
    index = [0] * len(s)

    for i in range(s[n - 1]):
        index[n - 1] = i
        for j in range(m):
            Au[i, j] = A[tuple(index)]
            index[n - 2] += 1
            for k in range(n - 2, n - 1 - len(s), -1):
                if index[k] == s[k]:
                    index[k - 1] += 1
                    index[k] = 0

    return Au


def fold(Au, n, s):
    """Reconstruct a tensor from its mode-n unfolded matrix.

    Parameters
    ----------
    Au : numpy.ndarray
        Mode-n unfolded representation, shape
        (M_n, M_{n+1} * ... * M_N * M_1 * ... * M_{n-1}).
    n : int
        Mode axis, **1-based**.
    s : tuple of int
        Desired shape of the reconstructed tensor.

    Returns
    -------
    A : numpy.ndarray, shape *s*

    Raises
    ------
    numpy.linalg.LinAlgError
        If the shape of *Au* is incompatible with *s* and *n*.

    Notes
    -----
    Natural inverse of ``unfold``; Definition 1 of de Lathauwer *et al.* (2000).
    """
    m = 1
    for i in range(len(s)):
        m *= s[i]
    m //= s[n - 1]

    if Au.shape != (s[n - 1], m):
        raise np.linalg.LinAlgError(
            "Shape mismatch: need %s but got %s" % ((s[n - 1], m), Au.shape))

    A = np.zeros(s)
    index = [0] * len(s)

    for i in range(s[n - 1]):
        index[n - 1] = i
        for j in range(m):
            A[tuple(index)] = Au[i, j]
            index[n - 2] += 1
            for k in range(n - 2, n - 1 - len(s), -1):
                if index[k] == s[k]:
                    index[k - 1] += 1
                    index[k] = 0

    return A


def HOSVD(A):
    """Compute the Higher-Order Singular Value Decomposition (HOSVD) of a tensor.

    Parameters
    ----------
    A : numpy.ndarray, shape (M_1, M_2, ..., M_N)
        Input tensor.

    Returns
    -------
    U : list of numpy.matrix, length N
        n-mode left singular matrices U^(n), n = 0, 1, ..., N-1.
        The *n*-th matrix has shape (M_n, M_n).
    S : numpy.ndarray, shape (M_1, M_2, ..., M_N)
        Core tensor.
    D : list of numpy.ndarray, length N
        n-mode singular values D^(n).

    Raises
    ------
    ValueError
        If *A* is not a numpy ndarray.

    Notes
    -----
    Equation 22 of:

        Lieven de Lathauwer, Bart de Moor, Joos Vandewalle,
        "A multilinear singular value decomposition",
        SIAM J. Matrix Anal. Appl. 21(4), 2000, 1253-1278.
    """
    if not isinstance(A, np.ndarray):
        raise ValueError("HOSVD requires a numpy ndarray.")

    Transforms = []
    NModeSingularValues = []

    for i in range(len(A.shape)):
        U, D, V = _robust_svd(unfold(A, i + 1))
        Transforms.append(np.asmatrix(U))
        NModeSingularValues.append(D)

    # Compute core tensor via the Kronecker-product formulation (axis 1)
    axis = 1
    Aun = unfold(A, axis)

    B = np.ones((1,))
    for i in range(axis - A.ndim, axis - 1):
        B = np.kron(B, Transforms[i])

    Sun = Transforms[axis - 1].transpose().conj() * Aun * B
    S = fold(Sun, axis, A.shape)

    return Transforms, S, NModeSingularValues


if __name__ == "__main__":
    pass
