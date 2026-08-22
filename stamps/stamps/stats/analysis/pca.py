# -*- coding: utf-8 -*-
# @Author: Chieh-Han Lee
# @Date:   2017-02-28 18:56:26
# @Last Modified by:   Chieh-Han Lee
# @Last Modified time: 2017-02-28 18:58:59
import numpy

def pca(A):
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
    
    for i in xrange(m):
        maxi=np.where((np.abs(NewEC[:,i]).max()==np.abs(NewEC[:,i])))
        signi=np.float(NewEC[:,i][maxi]/np.abs(NewEC[:,i][maxi]))
        NewEOFs[:,i]=signi*NewEOFs[:,i]
        NewEC[:,i]=signi*NewEC[:,i]
        
    return NewEOFs, NewEC, Newlambda

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

        for i in xrange(m):
            maxi=np.where((np.abs(NewEOFs[:,i]).max()==np.abs(NewEOFs[:,i])))
            signi=np.float(NewEOFs[:,i][maxi]/np.abs(NewEOFs[:,i][maxi]))
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
    
        for i in xrange(m):
            maxi=np.where((np.abs(NewEC[:,i]).max()==np.abs(NewEC[:,i])))
            signi=np.float(NewEC[:,i][maxi]/np.abs(NewEC[:,i][maxi]))
            NewEOFs[:,i]=signi*NewEOFs[:,i]
            NewEC[:,i]=signi*NewEC[:,i]    
        
    return NewEOFs,NewEC,lambdaU


if __name__ == '__main__':

    pass