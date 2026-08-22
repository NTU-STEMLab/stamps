# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 01:23:34 2015

@author: hdragon689
"""

from six.moves import range
import numpy


def gs(X, row_vecs=True, norm = True):
  '''
  Gram-Schmidt orthogonalization

  Syntax: Y = gs(X, row_vecs=True, norm = True)
  
  Input:
  X           m by n      a 2D array with data to be orthogonalized
  row_vecs    bool        True if X is row-wise; otherwise, it is column-wise
                          default is True    
  norm        bool        
    
  Remark:
  Row-wise data implies X=[[g1],[g2],[g3],...] where gi vector has shape of 1 by n 
  Column-wise data implies X=[[g1,g2,g3,]] where gi vector has shape of n by 1    
  
  The original code is downloaded from https://gist.github.com/iizukak/1287876    
  '''  
  if not row_vecs:
    X = X.T
  Y = X[0:1,:].copy()
  for i in range(1, X.shape[0]):
    proj = numpy.diag((X[i,:].dot(Y.T)/numpy.linalg.norm(Y,axis=1)**2).flat).dot(Y)
  #  print proj
    Y = numpy.vstack((Y, X[i,:] - proj.sum(0)))
  if norm:
    Y = numpy.diag(1/numpy.linalg.norm(Y,axis=1)).dot(Y)
  if row_vecs:
    return Y
  else:
    return Y.T    

def wqr(X, col_vecs=True, norm = True, wt=None):
  '''
  Weighted QR decomposition by Gram-Schmidt process
  
  Syntax: Q,R = wqr(X, col_vecs=True, norm = True, wt=None)
  
  Input:
  X           m by n      a 2D array with data to be orthogonalized
  col_vecs    bool        True if X is column-wise; otherwise, it is row-wise
                          default is True (column-wise)    
  norm        bool        True to provide the normalized orhtogonal vectors
  wt          m by 1      a 2D array with weighting function for orthogonalization process. 
                          Default considers the uniform weights for the vectors  
  
  Return:
  Q           m by n      a 2D array of orthogalized vectors 
  R           n by n      the upper triangular matrix such that X=QR
    
  Remark:
  1) Row-wise data implies X=[[g1],[g2],[g3],...] where gi vector has shape of 1 by n 
     Column-wise data implies X=[[g1,g2,g3,]] where gi vector has shape of n by 1   
  2) If row-wise data is provided, the inversion relationship will be A=R*Q
  

  '''
  if not col_vecs:
    X = X.T

  m,n=X.shape   
  
  if wt is None:
    wt=numpy.ones(X[:,0:1].shape)    
  else:
    wt=wt.reshape(m,1)
  wtdlg=numpy.diag(wt.flat)
    
  R=numpy.zeros((n,n))
  Rt=numpy.zeros((n,n)) 
  Y = X[:,0:1].copy()
  for i in range(1, n):
    Rt[i,0:i]=numpy.dot(Y.T.dot(wtdlg),X[:,i:i+1]).T
    proj= (numpy.diag((Rt[i,0:i]/numpy.dot(wt.T,Y*Y)).flat).dot(Y.T)).T
    Rt[i,0:i]=Rt[i,0:i]/numpy.sqrt(numpy.dot(wt.T,Y*Y))# /numpy.sqrt(numpy.dot(wt.T,Y[:,i-1:i]*Y[:,i-1:i]))
    Y = numpy.hstack((Y, X[:,i:i+1] - proj.sum(1).reshape(m,1)))
  R=Rt.T
  for i in range(n):
    R[i,:]=R[i,:]/numpy.sqrt(numpy.dot(wt.T,Y[:,i:i+1]*Y[:,i:i+1]))
  # numpy.fill_diagonal(R,numpy.sqrt(numpy.dot(wt.T,Y*Y)).flat)
    numpy.fill_diagonal(R,numpy.ones(n))
  if norm:
    R = numpy.dot(numpy.diag(numpy.sqrt(numpy.dot(wt.T,Y*Y)).flat),R)
    Y = numpy.dot(Y,numpy.diag((numpy.sqrt(1/numpy.dot(wt.T,Y*Y)).flat)))    
  if col_vecs:
    return Y,R
  else:
    return Y.T,R.T 
    
if __name__ == "__main__":
  X=numpy.random.rand(5,10)
  Q1=gs(X)
  Q2,R2=wqr(X.T)
  print (numpy.allclose(Q1.T,Q2))
  print (numpy.allclose(X.T,Q2.dot(R2)))
  I=numpy.diag(numpy.ones(X.shape[0]))
  print (numpy.allclose(I,(Q2.T).dot(Q2)))
    