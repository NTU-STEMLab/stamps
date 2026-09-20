# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:11:51 2016

@author: hdragon689
"""
import numpy as np
import itertools

def entropyD(*X):
  ''' Empirical entropy estimator for multivariate data
  
  Input 
  
  X   N by d  discrete data with dimension d and observations N
  
  Output
  
  H   scalar  entropy

  http://blog.biolab.si/2012/06/15/computing-joint-entropy-in-python/
  '''  
  
  n_instances = len(X[0])
  H = 0
  for classes in itertools.product(*[set(x) for x in X]):
    v = np.array([True] * n_instances)
    for predictions, c in zip(X, classes):
      v = np.logical_and(v, predictions == c)
    p = np.mean(v)
    H += -p * np.log(p) if p > 0 else 0
  return H
  
def entropyC(data,bins=None):
  '''
  Histogram-based entropy estimation for continuous data
  
  Input:
  data  N by d    data with d dimension and N observations
  bins  sequence or int, optional
      The bin specification:

    * A sequence of arrays describing the bin edges along each dimension.
    * The number of bins for each dimension (nx, ny, ... =bins)
    * The number of bins for all dimensions (nx=ny=...=bins).
    Default is 10 for all dimensions
    
  Output:
  H     scalar    entropy
  
  
  '''
  if bins is None:
    h,edges=np.histogramdd(data,normed=True)
  else:
    h,edges=np.histogramdd(data,normed=True,bins=bins)
  nd=len(edges)
  p=h
  for k in range(nd):
    p=p*(np.diff(edges)[k])
  
  idx=np.where(p>0)  
  hh=-p[idx]*np.log(p[idx])
  H=hh.sum()
  return H,p,edges
  
def condentropy(Y,X,bins=None):
  '''
  Estimate conditional entropy and transformation 
  
  condtional entropy
  H(Y|X)=H(X,Y)-H(X)  

  transinformation (mutual information)
  T(X,Y)=H(Y)-H(Y|X)
  
  Input:
  Y     N by 1     2-D array of the conditional entropy to be assessed
  X     N by d     2-D array of the data to be conditioned on
  
  Output:
  cH    scalar     The conditional entropy of Y given X
  tH    scalar     The transinformation T(X,Y), i.e., mutual information I(X,Y)
  H     scalar     The joint entropy of Y and X
  '''
  
  Y=np.array(Y,ndmin=2)
  m,n=Y.shape  
  if m<n:
    Y=Y.T
    m,n=Y.shape

  X=np.array(X,ndmin=2)  
  if X.shape[0] != m:
    X=X.T
    
  A=np.hstack(tuple([Y,X]))
  if bins is None:
    bins=20
  Hx=entropyC(X,bins=bins)[0]
  Hy=entropyC(Y,bins=bins)[0]
  H=entropyC(A,bins=bins)[0]
  cH=H-Hx
  tH=Hy-cH
  return cH,tH,H   

#def entropy(*X):
#    return np.sum(-p * np.log2(p) if p > 0 else 0 
#      for p in (np.mean(reduce(np.logical_and, (predictions == c for predictions, c in zip(X, classes))))
#      for classes in itertools.product(*[set(x) for x in X])))

if __name__ == "__main__":  
   
  '''Test under Multiple Gaussian distribution'''  
  m=[0,0,0]
  cov=[[1,0.3,0.6],[0.3,1,0.5],[0.6,0.5,1]]
  data2=np.random.multivariate_normal(m,cov,10000)
  
  Hn,pn,edges=entropyC(data2,bins=(1000,1000,1000))
  dx=np.diff(edges)[:,0]
  
  HnT=3./2*np.log(2*np.pi)+0.5*np.log(np.linalg.det(cov))+3./2-np.log(np.prod(dx))  
  
  print (Hn)
  print (HnT)

 


  