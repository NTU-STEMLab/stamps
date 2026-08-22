# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:16:16 2015

@author: hdragon689
"""
import numpy
from scipy.stats import norm as normal
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss

from ...bme.softconverter import proba2probdens
from ...general.ortho import wqr


   
def _gkxf(limi,data_m,data_std,nk,gk,hk,log=False,method='GH'): 
  limi=numpy.array(limi)
  limi=limi.reshape(limi.size)
  if method == 'GH':
    limi2=(limi-data_m)/data_std 
    gkx=numpy.zeros((limi2.size,nk))
    for i in range(nk):  
      if i<=0:
        gkx[:,i]=gk[i](limi2)-hk[i]
      else:
        gkx[:,i]=gk[i](limi2,0)-hk[i]  
  elif method == 'GL':
    limi2=limi
    nkm=nk
    if log is True:
      limi2=limi2[limi2>0]
      nkm=nk-1
    gkx=numpy.zeros((limi.size,nk))
    for i in range(nkm):  
      if i<=0:
        gkx[limi>0,i]=gk[i](limi2)
      else:
        gkx[limi>0,i]=gk[i](limi2,0)
    if log is True:
      gkx[limi>0,nkm]=gk[nkm](limi2)
  return gkx          

def _cons(limi,lam,data_m,data_std,nk,gk,hk,A=None,log=False,method='GH'):
  if numpy.unique(limi).size>1:
    if log is True:
      limi=limi[limi>0]
    limi2=numpy.unique(limi)  
    gkx=_gkxf(limi2,data_m,data_std,nk,gk,hk,log=log,method=method)
    pmf=numpy.exp(-gkx.dot(lam)).reshape(limi2.size,)
    softpdftype=2
    nl,limi,pdf,area=proba2probdens(softpdftype,limi2.size,limi2,pmf)
  else:
    if A is not None:
      area=A
    else:
      area=1.
  return area

def update_mu(
    H_old, max_step,
    alpha_multiplier,
    mu, dx, R, Q, nx, wk, hk2, nx_multiplier):

    alpha = 1.
    step = 1
    flag=0

    while step <= max_step:
        mu_temp = mu + alpha * dx
        if numpy.array_equal(mu, mu_temp): #dx is too small, just return
            return mu, flag
        else:
            lam_temp = numpy.dot(numpy.linalg.inv(R), mu_temp)
            logZ = numpy.log(
                numpy.sum(
                    (
                        numpy.exp(-Q.dot(mu_temp)).reshape(nx* nx_multiplier, ))*wk
                    )
                )
            H = logZ + numpy.sum(lam_temp * hk2)
            if H < H_old: # new mu is better, done.
                return mu_temp, flag
            else: # need line search, update variable
                alpha = alpha * alpha_multiplier
                step += 1

    if H >= H_old and step > max_step: # script rurn here if copnverge failed
        print ("The optimal lambda search can not converge, try to increase max_step.")
        flag=1
        return mu_temp, flag
        #raise ValueError("The optimal lambda search can not converge, try to increase max_step.")
    else:
        print ('Strange error...H or H_old maybe Nan or Inf, try to increase nx.')
        flag=1
        return mu_temp, flag

    


def maxentpdf_gh(data,limi=None,nx=20,order=4,plot=False):
  ''' Estimate the maximum entropy pdf with moment constraints up to the order
  of n by Gauss-Hermite quadrature. 
  
  Input
  data      n by 1    n observations of the variable of interest
  limi      nx by 1   nx limis for pdf output
  nx        scalar    the order for Gauss-Hermite Quadrature
  order     scalar    the largest order of moments to be considered
  
  Output
  pdf       nx by 1   estimated maximum entropy PDF at limi's     
  limi      nx by 1   
  '''
  
  data=data.reshape(data.size,1)
  # standardized the dataset
  data_m=numpy.mean(data,0)
  data_std=numpy.std(data,0)
  
  Z=numpy.zeros(data.shape)
  
  for k in range(data.shape[1]):  
    Z[:,k]=(data[:,k]-numpy.mean(data[:,k]))/numpy.std(data[:,k])
  
  gk,hk=gkhk(Z,order=order,corr=False)
  
  nk=len(hk)
  x,w=hermgauss(nx)
  gkx4gh=numpy.zeros((nx,nk))
  for i in range(nk):  
    if i<=0:
      gkx4gh[:,i:i+1]=gk[i]((hk[0]+numpy.sqrt(2.*hk[1])*x).T)
    else:
      gkx4gh[:,i:i+1]=gk[i]((hk[0]+numpy.sqrt(2.*hk[1])*x).T,hk[0])

  tol=1
  lam=numpy.zeros((nk,1))

  Egk=numpy.zeros((nk,1))
  b=numpy.zeros((nk,1))
  Egij=numpy.zeros((nk,nk))
  A=numpy.zeros((nk,nk))
  wk=numpy.sqrt(2.*hk[1])*w*numpy.exp(x**2.)

  # orthogonalizing
  gkx4gh2=numpy.zeros(gkx4gh.shape) 
  for i in range(nk):
    gkx4gh2[:,i]=gkx4gh[:,i]-hk[i]

  hk2=numpy.zeros(len(hk))  

  Q,R=numpy.linalg.qr(gkx4gh2) # I should perform a weighted QR decomposition
  mu=R.dot(lam)
  tolmin=1000
  opti_mu=mu
  opti_R=R
  flag=0

  while tol>0.000001 and flag==0:  
  
    lam=numpy.dot(numpy.linalg.inv(R),mu)
    pmf=numpy.exp(-gkx4gh2.dot(lam)).reshape(nx,)
    const=numpy.sum(pmf*wk)
    pdf=pmf/const
  
    H_old=numpy.log(const)+numpy.sum(lam*hk2) 
  
    Q,R=wqr(gkx4gh2,wt=pdf*wk) 
    mu=R.dot(lam)      
      
    for k in range(nk):
      Egk[k]=numpy.sum(wk*pdf*Q[:,k])
      b[k,0]=Egk[k]-hk2[k]

    for i in range(nk):
      for j in range(i,nk):
        Egij[i,j]=numpy.sum(wk*pdf*Q[:,i]*Q[:,j])
        A[i,j]=Egij[i,j]-Egk[i]*Egk[j]
        if j>i:
          A[j,i]=A[i,j]

    Q1=numpy.sum(wk*pdf)#numpy.mean(numpy.diag(Egij2))  
    Ainv=numpy.eye(nk)
    for i in range(nk):
      for j in range(nk):
        Ainv[i,j]=Ainv[i,j]+Egk[i]*Egk[j]/(Q1**2-numpy.sum(Egk**2))     

    dx=Ainv.dot(b)
    tol = numpy.max(numpy.abs(b))
    if tol<tolmin:
      tolmin=tol
      opti_mu=mu
      opti_R=R
    mu,flag = update_mu( H_old, 10, 0.5, mu, dx, R, Q, nx, wk, hk2, 1)
    if flag == 1:
      mu=opti_mu
      R=opti_R

  lam=numpy.dot(numpy.linalg.inv(R),mu)
  
  if plot is True:
    if limi is not None:
      nxx=len(limi)
    else:
      nxx=nx
      limi=numpy.linspace(data_m-4.*data_std,data_m+4.*data_std,nxx)   
    
    gkx=_gkxf(limi,data_m,data_std,nk,gk,hk)
    pmf=numpy.exp(-gkx.dot(lam)).reshape(nxx,)
    softpdftype=2
    nl,limi,pdf,A=proba2probdens(softpdftype,nxx,limi,pmf)
  
  gkxf = lambda xk: _gkxf(xk,data_m,data_std,nk,gk,hk)
  cons = lambda xk: _cons(xk,lam,data_m,data_std,nk,gk,hk)#,A)

  ppdf = lambda xk: (numpy.exp(-gkxf(xk).dot(lam))/cons(xk)).flat[:]
  
  if plot is True:
    return ppdf,pdf,limi,lam,A,gkx
  else:
    return ppdf
  
def maxentpdf_gh2(data,limi=None,nx=20,order=4):
  ''' Estimate the bivariate maximum entropy pdf with moment constraints up to 
  the order of n by Gauss-Hermite quadrature. 
  
  Input
  data      n by 2    n observations of the variable of interest
  limi      list      list contains the limis along x and y directions respectively
  nx        scalar    the order for Gauss-Hermite Quadrature
  order     scalar    the largest order of moments to be considered
  
  Output
  pdf       nx by 1   estimated maximum entropy PDF at limi's     
  limi      nx by 1   
  '''  

  # standardized the dataset
  data_m=numpy.mean(data,0)
  data_std=numpy.std(data,0)
  Z=numpy.zeros(data.shape)
  
  for k in range(data.shape[1]):  
    Z[:,k]=(data[:,k]-numpy.mean(data[:,k]))/numpy.std(data[:,k])
  
  gk,hk_list=gkhk(Z,order=4,corr=True)
  hk=numpy.hstack(hk_list)

  # Gauss-Hermite integration

  nk=len(hk)
  nv=hk_list[0].size
  x,w=hermgauss(nx)
  xi1,xi2=numpy.meshgrid(x,x)
  xx=numpy.hstack([xi1.reshape(nx*nx,1),xi2.reshape(nx*nx,1)])
  gkx4gh=numpy.zeros((nx*nx,nk))
  
  scaled_x=numpy.tile(hk_list[0],(nx*nx,1))+\
      numpy.tile(numpy.sqrt(2.*hk_list[1]),(nx*nx,1))*xx
    
  m=0
  for i in range(len(hk_list)-1):  
    if i<=0:
      gkx4gh[:,m:m+nv]=gk[i](scaled_x)
    else:
      gkx4gh[:,m:m+nv]=gk[i](scaled_x,hk_list[0])
    m=m+nv  

  gkx4gh[:,nk-1]=gk[-1](scaled_x,scaled_x,hk_list[0],hk_list[0],xidx=2)


  tol=1
  lam=numpy.zeros((nk,1))
  lam[2]=lam[3]=0.5

  Egk=numpy.zeros((nk,1))
  b=numpy.zeros((nk,1))
  Egij=numpy.zeros((nk,nk))
  A=numpy.zeros((nk,nk))
  w1,w2=numpy.meshgrid(w,w)
  ww=numpy.hstack([w1.reshape(nx*nx,1),w2.reshape(nx*nx,1)])
  wk=numpy.tile(numpy.sqrt(2.*hk_list[1]),(nx*nx,1))*ww*numpy.exp(xx**2.)
  wk=wk[:,0]*wk[:,1]

  # orthogonalizing
  gkx4gh2=numpy.zeros(gkx4gh.shape) 
  for i in range(nk):
    gkx4gh2[:,i]=gkx4gh[:,i]-hk[i]

  hk2=numpy.zeros(hk.shape)  

  Q,R=numpy.linalg.qr(gkx4gh2) 
  mu=R.dot(lam)
  hk2=numpy.transpose(numpy.linalg.pinv(R)).dot(hk2)
  tolmin=1000
  opti_mu=mu
  opti_R=R
  flag=0
  
  while tol>0.000001 and flag==0:  
  
    lam=numpy.dot(numpy.linalg.pinv(R),mu)
    pmf=numpy.exp(-gkx4gh2.dot(lam)).reshape(nx*nx,)
    const=numpy.sum(pmf*wk)
    pdf=pmf/const
  
    H_old=numpy.log(const)+numpy.sum(lam*hk2) 
  
    Q,R=wqr(gkx4gh2,wt=pdf*wk) 
    mu=R.dot(lam)      
    
    for k in range(nk):
      Egk[k]=numpy.sum(wk*pdf*Q[:,k])
      b[k,0]=Egk[k]-hk2[k]
    for i in range(nk):
      for j in range(i,nk):
        Egij[i,j]=numpy.sum(wk*pdf*Q[:,i]*Q[:,j])
        A[i,j]=Egij[i,j]-Egk[i]*Egk[j]
        if j>i:
          A[j,i]=A[i,j]

    Q1=numpy.sum(wk*pdf)
    Ainv=numpy.eye(nk)
    for i in range(nk):
      for j in range(nk):
        Ainv[i,j]=Ainv[i,j]+Egk[i]*Egk[j]/(Q1**2-numpy.sum(Egk**2))   

    dx=Ainv.dot(b)
    tol = numpy.max(numpy.abs(b))
    if tol<tolmin:
      tolmin=tol
      opti_mu=mu
      opti_R=R
    mu,flag = update_mu( H_old, 100, 0.75, mu, dx, R, Q, nx, wk, hk2, nx )
    if flag == 1:
      mu=opti_mu
      R=opti_R
      
  lam=numpy.dot(numpy.linalg.inv(R),mu)
   
  if limi is None:   
    limi=numpy.zeros(scaled_x.shape)
    nxy=numpy.array([nx,nx])
    for k in range(data.shape[1]): # back-transformed
      limi[:,k]=scaled_x[:,k]*data_std[k]+data_m[k]
  else:
    limig=numpy.meshgrid(limi[0],limi[1])
    nxy=numpy.array([limi[0].size,limi[1].size])
    limi=limig
        
  limi2=numpy.zeros(limi.shape)
  for k in range(limi.shape[1]):
    limi2[:,k]=(limi[:,k]-data_m[k])/data_std[k]
  
  m=0  
  gkx=numpy.zeros((limi.shape[0],nk))
  for i in range(len(hk_list)-1):  
    if i<=0:
      gkx[:,m:m+nv]=gk[i](limi2)
    else:
      gkx[:,m:m+nv]=gk[i](limi2,hk_list[0])
    m=m+nv  

  gkx[:,nk-1]=gk[-1](limi2,limi2,hk_list[0],hk_list[0],xidx=2)
    
  gkx2=numpy.zeros(gkx.shape)
  for i in range(nk):
    gkx2[:,i]=gkx[:,i]-hk[i]
   
  pdf=numpy.exp(-gkx2.dot(lam)).reshape(limi.shape[0])

  return pdf,limi,lam,gkx2,nxy


def gkhk(Z=None,order=4,corr=False,log=False):
  '''Obtain the moment-based general knowledge and the expected value from the 
  data
  
  Input
  Z         m by n    the data of n variables and m observations  
  order     scalar    the highest order for the moment constraints
  corr      bool      If the correlation of the data to be considered. Default 
                      is False
  log       bool      If the log of each variable is considerd. Default is False
  
  Output
  gk        list      the functionals for moment-based general knowledge with the
                      size of the number of general knowledge bases
  hk        list      the expected values associated with gk's.  
  '''
  g1=lambda x: x
  g2=lambda x,m: (x-numpy.repeat(m,x.shape[0],axis=0).reshape(x.shape))**2
  g3=lambda x,m: (x-numpy.repeat(m,x.shape[0],axis=0).reshape(x.shape))**3
  g4=lambda x,m: (x-numpy.repeat(m,x.shape[0],axis=0).reshape(x.shape))**4
  g5=lambda x,m: (x-numpy.repeat(m,x.shape[0],axis=0).reshape(x.shape))**5
  g6=lambda x,m: (x-numpy.repeat(m,x.shape[0],axis=0).reshape(x.shape))**6
  g7=lambda x,m: (x-numpy.repeat(m,x.shape[0],axis=0).reshape(x.shape))**7
  g8=lambda x,m: (x-numpy.repeat(m,x.shape[0],axis=0).reshape(x.shape))**8
  g9=lambda x,m: (x-numpy.repeat(m,x.shape[0],axis=0).reshape(x.shape))**9
  g10=lambda x,m: (x-numpy.repeat(m,x.shape[0],axis=0).reshape(x.shape))**10
  glog=lambda x: numpy.log(x)  
  
  def g11(x,y,mx,my,xidx=None):
    '''The general knowledge base to estimate the correlation between the 
    observations x and y with their associated mean mx and my
    
    Input:
    x       m by nx     2D array of nx variables with m observations
    y       m by ny     2D array of ny variables with m observations
    mx      nx by 1     1D array with mean of nx variables
    my      ny by 1     1D array with mean of ny variables
    xidx    idxn by 1   1D array with the index of the flatten correlation to be
                        returned (optional)
    
    Output  
    h11     idxn by 1   1D array with the correlations of the specified index. 
                        If xidx is None, the entire flatten correlation will be
                        returned                                
    '''
    xz=x.shape[1]
    yz=y.shape[1]
    a=x-numpy.tile(mx,(x.shape[0],1))
    b=y-numpy.tile(my,(x.shape[0],1))
    c=numpy.zeros((x.shape[0],xz*yz))
    k=0
    for i in range(xz):
      for j in range(yz):
        c[:,k]=a[:,i]*b[:,j]
        k=k+1
    if xidx is not None:
      c=c[:,xidx]
    return c
  
  gf=[g1,g2,g3,g4,g5,g6,g7,g8,g9,g10]  
  gk=gf[:order]
  
  if Z is not None:    
    h1=numpy.mean(g1(Z),0).reshape(1,Z.shape[1])
    h2=numpy.mean(g2(Z,h1),0).reshape(1,Z.shape[1])
    h3=numpy.mean(g3(Z,h1),0).reshape(1,Z.shape[1])
    h4=numpy.mean(g4(Z,h1),0).reshape(1,Z.shape[1])
    h5=numpy.mean(g5(Z,h1),0).reshape(1,Z.shape[1])
    h6=numpy.mean(g6(Z,h1),0).reshape(1,Z.shape[1])
    h7=numpy.mean(g7(Z,h1),0).reshape(1,Z.shape[1])
    h8=numpy.mean(g8(Z,h1),0).reshape(1,Z.shape[1])
    h9=numpy.mean(g9(Z,h1),0).reshape(1,Z.shape[1])
    h10=numpy.mean(g10(Z,h1),0).reshape(1,Z.shape[1])
    hlog=numpy.mean(glog(Z),0).reshape(1,Z.shape[1])
    if corr is True:
      if Z.shape[1]==2:
        h11=numpy.mean(g11(Z,Z,h1,h1,xidx=2),0)
      elif Z.shape[1]==1:
        h11=numpy.mean(g11(Z,Z,h1,h1,xidx=0),0)
      else:
        print ("Z should be n by 1 or n by 2")
        return
  
    hf=[h1,h2,h3,h4,h5,h6,h7,h8,h9,h10]  
    hk=hf[:order]

  if corr is True:
    gk.append(g11)
    if Z is not None:
      hk.append(h11)
  
  if log is True:
    gk.append(glog)
    if Z is not None:
      hk.append(hlog)
  
  if Z is None:
    return gk
  else:    
    return gk,hk

def maxentpdf_gcp(data,nx=20,order=4):
  ''' Multidimesional maximum entropy pdf with Gaussian copula
  
   Input
   
   data   m by n      a 2D numpy array of n variables and m observations.   
   nx     integer     the number of limits for Gaussian-Hermite integration for 
                      maximum entropy pdf evaluation. Default is 20
   order  integer     the maximum order of moment constrants for each of the 
                      marginal PDFs                  
                      
   Output
   mpdf   function    a function of multidimensional PDF in which the input is 
                      a list of which the limits of each variable is either 
                      a scalar or a multidimensional numpy array
   ppdf   list        a list of marginal PDF for every variable
   pdf    list        a list of every marginal PDF values at the default limits 
                      corresponding to the nx
   limi   list        a list of default limits for every variable with respect 
                      to the nx
   lam    list        a list of maximum entropy parameters for each marginal PDFs              
  '''

  m,n=numpy.shape(data)
  data_m=numpy.mean(data,0)
  data_std=numpy.std(data,0)
  R=numpy.corrcoef(data.T)
    
  ppdf=[None]*n    
  pdf=[None]*n
  limi=[None]*n
  lam=[None]*n
  gkx=[None]*n
  A=[None]*n
  for k in range(data.shape[1]):  
    ppdf[k],pdf[k],limi[k],lam[k],A[k],gkx[k]=maxentpdf_gh(data[:,k],nx=nx,order=order)
  
  def mppdf(xk):
    if isinstance(xk[0],numpy.ndarray):
      flag_one=0
      xkshp=numpy.shape(xk[0])
      if numpy.any(numpy.equal(xk[0].shape,1)):  
        if xk[0].shape[1] != 1:
          for i in range(len(xk)):
            xk[i]=xk[i].T
      else:
        for i in range(len(xk)):
          xk[i]=xk[i].reshape(xk[i].size,1)
    else:
      flag_one=1
      for i in range(len(xk)):
        xk[i]=numpy.array(xk[i])
    Z=numpy.zeros((xk[0].size,len(xk)))
    for k in range(data.shape[1]):  
      Z[:,k:k+1]=(xk[k]-data_m[k])/data_std[k]

    gcpZ=1./numpy.sqrt(numpy.linalg.det(R))
    temp=Z.dot(numpy.linalg.inv(R)-numpy.eye(n))
    ZRZdiag=numpy.sum(temp*Z,1)
    gcp=gcpZ*numpy.exp(-0.5*ZRZdiag)  

    mppdf=numpy.ones((xk[0].size,1)).flat[:]   
    for k in range(n):
      mppdf=mppdf*ppdf[k](xk[k])
    if flag_one:
      mppdf=mppdf*gcp
    else:
      mppdf=(mppdf*gcp).reshape(xkshp)  
    
    return mppdf
    
  mpdf = lambda xk: mppdf(xk)
  
  return mpdf,ppdf,pdf,limi,lam  

def maxentpdf_gkhk(hki,nx=20,order=None,log=False,method='GH'):
  ''' Estimate the maximum entropy pdf with moment constraints up to a specified 
  order by Gauss-Hermite quadrature. 
  
  Input
  hki       list      a list of the hk's corresponding the moments of contraints
                      up to the specified order, log, and corr. The sequences is
                      [orders,log]. If no log is included, the seequence of hki 
                      are in the increasing order where if some of the orders 
                      are not available, the hki should be None 
                      specified
  nx        integer   the order for Gauss-Hermite Quadrature
  order     integer   the largest order of moments to be considered. If None, 
                      the order is the nh-nc-nl, where nh is the number of hki's, 
                      and nl is [0,1] for [False, True] of log
  log       bool      to determine if hki includes the log function in the general
                      knowledge base
  method    string    method for numerical integration. Default is Gauss-Hermit 
                      for moments only constraints. If log is True, Gauss-
                      Lengendre (GL) is used default. 
                       
  Output
  ppdf      nx by 1   estimated maximum entropy PDF function 
  limi      nx by 1   estimated parameters for the PDF
  '''
  
  hk=hki[:].copy()
  if len(hk) >= 2 and log is False:
    # moment constranits standardization
    hk[0]=0.
    for i in range(len(hk)-1):
      hk[i+1]=hki[i+1]/numpy.sqrt(hki[1])**(i+2)    
      
  nk=len(hk)
  if log is False:
    gk=gkhk(order=nk)
    nkm=nk
  else:
    gk=gkhk(order=nk-1,log=True)
    nkm=nk-1

  idx=[]  
  for i,e in enumerate(hk):
    if e is not None:
      idx.append(i)
  
  gk=[gk[i] for i in idx]
  hk=[hk[i] for i in idx]     
     
  if log is True:
    method='GL'     
     
  nk=len(gk)  
  if method == 'GH': 
    x,w=hermgauss(nx)
    gkx4gh=numpy.zeros((nx,nk))
    for i in range(nkm):  
      if i<=0:
        gkx4gh[:,i]=gk[i](hk[0]+numpy.sqrt(2.*hk[1])*x)
      else:
        gkx4gh[:,i]=gk[i](hk[0]+numpy.sqrt(2.*hk[1])*x,hk[0])    
  elif method == 'GL':      
    x,w=leggauss(nx)
    gkx4gh=numpy.zeros((nx,nk))
    a=hk[0]-5.*numpy.sqrt(hk[1])
    b=hk[0]+5.*numpy.sqrt(hk[1])
    if log is True and a<=0:
      a=numpy.finfo(float).eps
    for i in range(nkm):  
      if i<=0:
        gkx4gh[:,i]=gk[i]((a+b)/2+(b-a)/2*x)
      else:
        gkx4gh[:,i]=gk[i]((a+b)/2+(b-a)/2*x,hk[0])
    if log is True:
      gkx4gh[:,nkm]=gk[nkm]((a+b)/2+(b-a)/2*x)      

  tol=1
  lam=numpy.zeros((nk,1))

  Egk=numpy.zeros((nk,1))
  b=numpy.zeros((nk,1))
  Egij=numpy.zeros((nk,nk))
  A=numpy.zeros((nk,nk))
  if method == 'GH':
    wk=numpy.sqrt(2.*hk[1])*w*numpy.exp(x**2.)
  elif method == 'GL':
    wk=numpy.ones(w.shape)

  # orthogonalizing
  gkx4gh2=numpy.zeros(gkx4gh.shape) 
  for i in range(nk):
    gkx4gh2[:,i]=gkx4gh[:,i]-hk[i]

  hk2=numpy.zeros(len(hk))  

  Q,R=numpy.linalg.qr(gkx4gh2) # I should perform a weighted QR decomposition
  mu=R.dot(lam)
  tolmin=1000
  opti_mu=mu
  opti_R=R
  flag=0
  
  while tol>0.000001 and flag==0:    
  
    lam=numpy.dot(numpy.linalg.inv(R),mu)
    pmf=numpy.exp(-gkx4gh2.dot(lam)).reshape(nx,)
    const=numpy.sum(pmf*wk)
    pdf=pmf/const
  
    H_old=numpy.log(const)+numpy.sum(lam*hk2) 
  
    Q,R=wqr(gkx4gh2,wt=pdf*wk) 
    mu=R.dot(lam)      
      
    for k in range(nk):
      Egk[k]=numpy.sum(wk*pdf*Q[:,k])
      b[k,0]=Egk[k]-hk2[k]
#      
#    import pdb 
#    pdb.set_trace()

    for i in range(nk):
      for j in range(i,nk):
        Egij[i,j]=numpy.sum(wk*pdf*Q[:,i]*Q[:,j])
        A[i,j]=Egij[i,j]-Egk[i]*Egk[j]
        if j>i:
          A[j,i]=A[i,j]

    Q1=numpy.sum(wk*pdf)#numpy.mean(numpy.diag(Egij2))  
    Ainv=numpy.eye(nk)
    for i in range(nk):
      for j in range(nk):
        Ainv[i,j]=Ainv[i,j]+Egk[i]*Egk[j]/(Q1**2-numpy.sum(Egk**2))     

    dx=Ainv.dot(b)
    tol = numpy.max(numpy.abs(b))
    if tol<tolmin:
      tolmin=tol
      opti_mu=mu
      opti_R=R 
    mu,flag = update_mu( H_old, 100, 0.75, mu, dx, R, Q, nx, wk, hk2, 1 )
    if flag == 1:
      mu=opti_mu
      R=opti_R

  lam=numpy.dot(numpy.linalg.inv(R),mu) 

#  import pdb 
#  pdb.set_trace()  
  
  nxx=50
  limi_test=numpy.linspace(hki[0]-4.*numpy.sqrt(hki[1]),hki[0]+4.*numpy.sqrt(hki[1]),nxx) 
  if log is True:
    limi_test=limi_test[limi_test>0]
    nxx=limi_test.size

  data_m=hki[0]               
  data_std=numpy.sqrt(hki[1])  
  gkx=_gkxf(limi_test,data_m,data_std,nk,gk,hk,log=log,method=method)
  pmf=numpy.exp(-gkx.dot(lam)).reshape(nxx,)
  softpdftype=2
  nl,limi,pdf,A=proba2probdens(softpdftype,nxx,limi_test,pmf) 
#  data_m=hki[0]               
#  data_std=numpy.sqrt(hki[1])    
  
  gkxf = lambda xk: _gkxf(xk,data_m,data_std,nk,gk,hk,log=log,method=method)
  cons = lambda xk: _cons(xk,lam,data_m,data_std,nk,gk,hk,A,log=log,method=method)    
  def mypdf(xk):
    pdf=(numpy.exp(-gkxf(xk).dot(lam))/cons(xk))
    if log is True:
      pdf[numpy.where(xk<0)]=0.
    return pdf.flat[:]  
       
  ppdf = lambda xk: mypdf(xk)
  
  return ppdf,lam


def maxentpdf_gc(ppdf,R,hk):
  ''' Multidimesional maximum entropy pdf with Gaussian copula
  
   Input
   
   ppdf   list        list of n functions of marginal pdfs
   corr   n by n      a correlation matrix for n random variables   
   hk     list        list of hk values for each variable in ppdfs
            
                      
   Output
   mpdf   function    a function of multidimensional PDF in which the input is 
                      a list of which the limits of each variable is either 
                      a scalar or a multidimensional numpy array
   ppdf   list        a list of marginal PDF for every variable             
  '''
  
  n=len(ppdf)
  hk2=numpy.vstack(hk)
  data_m=hk2[:,0]
  data_var=hk2[:,1]
  data_std=numpy.sqrt(hk2[:,1])
#  data_m=numpy.zeros(n)
#  data_std=numpy.zeros(n)
  ccdf=[]
  for i in range(n):
#    data_m[i]=hk[i][0]
#    data_std[i]=numpy.sqrt(hk[i][1])
    ccdf.append(pdf2cdf(ppdf[i],[data_m[i],data_var[i]]))
    
  def mppdf(xk):
    if isinstance(xk[0],numpy.ndarray) and not isinstance(xk,numpy.ndarray):
      flag_one=0
      xkshp=numpy.shape(xk[0])
      if numpy.any(numpy.equal(xk[0].shape,1)):  
        if xk[0].shape[1] != 1:
          for i in range(len(xk)):
            xk[i]=xk[i].T
      else:
        for i in range(len(xk)):
          xk[i]=xk[i].reshape(xk[i].size,1)
    elif isinstance(xk,numpy.ndarray):
      flag_one=2
      mxk,nxk=xk.shape
      xk=xk.T             
    else:
      flag_one=1
      for i in range(len(xk)):
        xk[i]=numpy.array(xk[i])
        if isinstance(xk[i],numpy.ndarray):
          xk[i]=xk[i].reshape(xk[i].size)
    if isinstance(xk[0],numpy.ndarray):      
      Z=numpy.zeros((xk[0].size,len(xk)))
    else:
      Z=numpy.zeros((1,len(xk)))
    for k in range(n):
      ppf=normal.ppf(ccdf[k](xk[k]))
      if (numpy.isinf(ppf)).any():
        ppf[numpy.logical_and(numpy.isinf(ppf),ppf>0)]=12.
        ppf[numpy.logical_and(numpy.isinf(ppf),ppf<0)]=-12.
      Z[:,k:k+1]=ppf.reshape((xk[k].size,1))
     # Z[:,k:k+1]=(xk[k]-data_m[k])/data_std[k]

    gcpZ=1./numpy.sqrt(numpy.linalg.det(R))
    temp=Z.dot(numpy.linalg.inv(R)-numpy.eye(n))
    ZRZdiag=numpy.sum(temp*Z,1)
    gcp=gcpZ*numpy.exp(-0.5*ZRZdiag)
#    gcp=gcpZ*numpy.exp(-0.5*Z.dot(numpy.linalg.inv(R)-numpy.eye(n)).dot(Z.T))
#    gcp=numpy.diag(gcp)      

    mppdf=numpy.ones((xk[0].size,1)).flat[:]   
    for k in range(n):
      mppdf=mppdf*ppdf[k](xk[k])
    if flag_one==1 or flag_one==2:
      mppdf=(mppdf*gcp).reshape((xk[k].size,1))
    else:
      mppdf=(mppdf*gcp).reshape(xkshp)
    
    return mppdf
    
  mpdf = lambda xk: mppdf(xk)
  
  return mpdf,ppdf  

def maxentcondpdf_gc(ppdf,R,hk,k_num=1):
  ''' Multidimesional maximum entropy pdf with Gaussian copula
  
   Input
   
   ppdf   list        list of n functions of marginal pdfs
   corr   n by n      a correlation matrix for n random variables   
   hk     list        list of hk values for each variable in ppdfs
            
                      
   Output
   mpdf   function    a function of multidimensional PDF in which the input is 
                      a list of which the limits of each variable is either 
                      a scalar or a multidimensional numpy array
   ppdf   list        a list of marginal PDF for every variable             
  '''
  
  n=len(ppdf)
  hk2=numpy.vstack(hk)
  data_m=hk2[:,0]
  data_var=hk2[:,1]
  # data_std=numpy.sqrt(hk2[:,1])
#  data_m=numpy.zeros(n)
#  data_std=numpy.zeros(n)
  ccdf=[]
  for i in range(n):
#    data_m[i]=hk[i][0]
#    data_std[i]=numpy.sqrt(hk[i][1])
    ccdf.append(pdf2cdf(ppdf[i],[data_m[i],data_var[i]]))
    
  def mcondpdf(xk):
    if isinstance(xk[0],numpy.ndarray) and not isinstance(xk,numpy.ndarray):
      flag_one=0
      xkshp=numpy.shape(xk[0])
      if numpy.any(numpy.equal(xk[0].shape,1)):  
        if xk[0].shape[1] != 1:
          for i in range(len(xk)):
            xk[i]=xk[i].T
      else:
        for i in range(len(xk)):
          xk[i]=xk[i].reshape(xk[i].size,1)
    elif isinstance(xk,numpy.ndarray):
      flag_one=2
      mxk,nxk=xk.shape
      xk=xk.T             
    else:
      flag_one=1
      for i in range(len(xk)):
        xk[i]=numpy.array(xk[i])
        xk[i]=xk[i].reshape(xk[i].size)
    Z=numpy.zeros((xk[0].size,len(xk)))
    for k in range(n):  
      ppf=normal.ppf(ccdf[k](xk[k]))
      if (numpy.isinf(ppf)).any():
        ppf[numpy.logical_and(numpy.isinf(ppf),ppf>0)]=12.
        ppf[numpy.logical_and(numpy.isinf(ppf),ppf<0)]=-12.
      Z[:,k:k+1]=ppf.reshape((xk[k].size,1))
     # Z[:,k:k+1]=(normal.ppf(ccdf[k](xk[k]))).reshape((xk[k].size,1))
     # Z[:,k:k+1]=(xk[k]-data_m[k])/data_std[k]

    gcpZ_kh=1./numpy.sqrt(numpy.linalg.det(R))
    temp_kh=Z.dot(numpy.linalg.inv(R)-numpy.eye(n))
    ZRZdiag_kh=numpy.sum(temp_kh*Z,1)
    gcp_kh=gcpZ_kh*numpy.exp(-0.5*ZRZdiag_kh)
    
    gcpZ_h=1./numpy.sqrt(numpy.linalg.det(R[k_num:,k_num:]))
    temp_h=Z[:,k_num:].dot(numpy.linalg.inv(R[k_num:,k_num:])-numpy.eye(n-k_num))
    ZRZdiag_h=numpy.sum(temp_h*Z[:,k_num:],1)
    gcp_h=gcpZ_h*numpy.exp(-0.5*ZRZdiag_h)
#    gcp=gcpZ*numpy.exp(-0.5*Z.dot(numpy.linalg.inv(R)-numpy.eye(n)).dot(Z.T))
#    gcp=numpy.diag(gcp)      

    mppdf=numpy.ones((xk[0].size,1))#k_num))
    for k in range(k_num):
      ppdfk=ppdf[k](xk[k]).reshape((xk[k].size,1))
      mppdf=mppdf*ppdfk#numpy.kron(numpy.ones((1,k_num)),ppdfk)
    if flag_one==1 or flag_one==2:
      gcp=(gcp_kh/gcp_h).reshape((xk[0].size,1))
      #gcpp=numpy.kron(numpy.ones((1,k_num)),gcp)
      mppdf=(mppdf*gcp).reshape((xk[k].size,1))#k_num))
      mppdf[numpy.isnan(mppdf)]=0.
    else:
      mppdf=(mppdf*gcp_kh/gcp_h).reshape(xkshp)
      mppdf[numpy.isnan(mppdf)]=0.
    
    return mppdf
    
  mpdf = lambda xk: mcondpdf(xk)
  
  return mpdf,ppdf  



def pdf2cdf(pdf,hk):
  '''
  Create a 1-D cdf function based upon a 1-D pdf function approximated by the 
  trapezoidal integration
  
  Input:
  pdf     function
  hk      1 by 2      a list or numpy array with the first two moments of 
                      the pdf functions                 
  '''
  nlim=50
  nstd=5
  pdf0=numpy.empty((0))
  
  while pdf0.size<2 or not (pdf0[0]==0 and pdf0[-1]==nlim-1):
    z=numpy.linspace(hk[0]-nstd*numpy.sqrt(hk[1]),hk[0]+nstd*numpy.sqrt(hk[1]),nlim)
    pdf0=numpy.where(pdf(z)==0)[0]
    if (pdf0.size>=2 and (pdf0[0]==0 and pdf0[-1]==nlim-1)):
      idd=numpy.where(numpy.diff(pdf0)>1)[0]
      zmin=z[pdf0[idd.min()]]
      zmax=z[pdf0[idd.max()+1]]
      break
    elif nstd>7:
      zmin=hk[0]-5*numpy.sqrt(hk[1])
      zmax=hk[0]+5*numpy.sqrt(hk[1])
      break
    else:
      nstd=nstd+1
      
  z=numpy.linspace(zmin,zmax,100)
  dz=numpy.diff(z)
  pdfz=(pdf(z[1:])+pdf(z[0:-1]))/2.*dz
  cdfz=numpy.hstack((0.,numpy.cumsum(pdfz)))
  if cdfz[-1]!=1:
    cdfz=cdfz/cdfz[-1]
      
  def cdf(xk):
    if type(xk) is numpy.float64:
      xk=numpy.array([xk])
    out=numpy.empty(xk.shape)
    id0=numpy.where(xk<=zmin)[0]
    id1=numpy.where(xk>=zmax)[0]
    idx=numpy.where(numpy.logical_and(xk>zmin,xk<zmax))[0]
    out[id0]=0
    out[id1]=1
    out[idx]=numpy.interp(xk[idx],z,cdfz)
    
    return out
    
  return cdf    

  
if __name__ == "__main__":  
  
  import pandas as pd
  from starpy.general.valstvgx import valstv2stg
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  from starpy.graph.dataplot import tsplot
  import matplotlib.cm as cm
  
  data='./examples/Data/GeoData.xls'

  datadf=pd.ExcelFile(data).parse('Sheet1',header=None)
  datadf.insert(loc=0,column='Time',value=pd.to_datetime(datadf[2]*100+datadf[3],format='%Y%m'))

  # performs valstv2stg
  # ch
  xy=(datadf.iloc[:,[1,2]]).values
  tt=numpy.asarray(datadf['Time'].values,dtype='datetime64[M]')
  tt=tt.reshape((tt.size,1))
  #ch=np.asarray(zip(xy[:,0],xy[:,1],tt[:,0]), \
  #                  dtype=[('x','f4'),('y','f4'),('t','datetime64[M]')])  
  ch=numpy.asarray(zip(xy[:,0],xy[:,1],tt[:,0]))                    
                  # we use the list to create ch because it is not allowed to 
                  # have float and datetime in the same numpy array

  # zh
  zh=(datadf[4].values).reshape((datadf.shape[0],1))

  Z,cMS,tME,_=valstv2stg(ch,zh)

  b=numpy.empty((0,0))
  b=numpy.append(b,-26.3724653739771)
  b=numpy.append(b,0.968464993459824)
  Z[18,0]=b[0]+b[1]*Z[23,0]
  
  print ('''Data time series''')
  fig=plt.figure()
  ax1=tsplot(tME,Z[0],fmt='b-')
  tsplot(tME,Z[20],fmt='r--',ax=ax1)
  plt.xlabel('Time')
  plt.ylabel('Level(m)')
  plt.show()
   
  
  print ('''Test 1-D pdf''')  
  data=Z[0].T
  ppdf,pdf,limi,lam,A,gkx=maxentpdf_gh(data,nx=15,order=6)

  
  print (lam)
  
  limi2=numpy.linspace(min(data)-5,max(data)+5,200)
  pdf2=ppdf(limi2)

  plt.figure()
  plt.hold(True)
  plt.hist(data,normed=True)
  plt.plot(limi,pdf,'o')
  plt.plot(limi2,pdf2)
  plt.xlabel('level(m)')
  plt.ylabel('Proba')
  plt.show()
  
  print ('''Test 2-D pdf''')  
  data=Z[[0,20]].T
  pdf,limi,lam,gkx,nxy=maxentpdf_gh2(data,nx=20,order=5)
  
  print (lam)
  
  fig=plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  plt.hold(True)
  pdfg=pdf.reshape(nxy[0],nxy[1])
  scaled_x1g=limi[:,0].reshape(nxy[0],nxy[1])
  scaled_x2g=limi[:,1].reshape(nxy[0],nxy[1])
 # ax.plot_wireframe(scaled_x1g, scaled_x2g, pdfg,color='r', rstride=2, cstride=2)
  ax.plot_surface(scaled_x1g, scaled_x2g, pdfg, rstride=2, cstride=2, cmap=cm.coolwarm, shade='interp')  
  plt.xlabel('Z1')
  plt.ylabel('Z2')
  ax.set_zlabel('Proba')
  plt.show()  
  
  print ('''Test 3-D pdf''')
  data=Z[[0,20,30]].T
  mpdf,ppdf,pdf,limi,lam=maxentpdf_gcp(data,nx=80,order=6)  
  aaa=mpdf([1,2,3])
  x1=numpy.linspace(min(data[:,0]),max(data[:,0]),50)
  x2=numpy.linspace(min(data[:,1]),max(data[:,1]),50)
  x3=numpy.linspace(min(data[:,2]),max(data[:,2]),50)
  x1v,x2v,x3v=numpy.meshgrid(x1,x2,x3)
#  x2v,x3v=numpy.meshgrid(x2,x3)
#  x1v,x3v=numpy.meshgrid(x1,x3)
  mpdff=mpdf([x1v,x2v,x3v])
  plt.figure(); plt.plot(x1,ppdf[0](x1)); plt.show()
  plt.figure(); plt.plot(x2,ppdf[1](x2)); plt.show()
  plt.figure(); plt.plot(x3,ppdf[2](x3)); plt.show()

  
  x1v,x2v=numpy.meshgrid(x1,x2)
  mpdff2=numpy.zeros(x1v.shape)
  # Integrate across x3
  for k,x3k in enumerate(x3):
    dx3=x3[1]-x3[0]
    x3v=x3k*numpy.ones(x1v.shape)
    mpdff2_=mpdf([x1v,x2v,x3v])
    mpdff2=mpdff2+mpdff2_*dx3
    
  fig=plt.figure()
  ax = fig.add_subplot(111, projection='3d')    
  ax.plot_surface(x1v, x2v, mpdff2, rstride=2, cstride=2, 
                  cmap=cm.coolwarm, shade='interp')  
  plt.xlabel('Z1')
  plt.ylabel('Z2')
  ax.set_zlabel('Proba')  
  plt.show()  
  
  x2v,x3v=numpy.meshgrid(x2,x3)
  mpdff2=numpy.zeros(x1v.shape)
  # Integrate across x1  
  for k,x1k in enumerate(x1):
    dx3=x1[1]-x1[0]
    x1v=x1k*numpy.ones(x1v.shape)
    mpdff2_=mpdf([x1v,x2v,x3v])
    mpdff2=mpdff2+mpdff2_*dx3  
  
  fig=plt.figure()
  ax = fig.add_subplot(111, projection='3d')  
  ax.plot_surface(x2v, x3v, mpdff2, rstride=2, cstride=2, cmap=cm.coolwarm, shade='interp')  
  plt.xlabel('Z2')
  plt.ylabel('Z3')
  ax.set_zlabel('Proba')  
  plt.show()    
  
  print ('''Test User-specified hk's''')
  from scipy.stats import norm
  
  hk=[10,10,30]
  ppdf,lam=maxentpdf_gkhk(hk,nx=80,method='GH')
  print (lam)
  xx=numpy.linspace(hk[0]-5*numpy.sqrt(hk[1]),hk[0]+5*numpy.sqrt(hk[1]),1000)
  plt.figure()
  plt.hold(True) 
  plt.plot(xx,ppdf(xx),'o')
  plt.plot(xx, norm.pdf(xx,loc=hk[0],scale=numpy.sqrt(hk[1])),'r--')
  plt.xlabel('Concentration at $X_1$')
  plt.ylabel('Probability')  
  plt.show()
  mm=(xx*ppdf(xx)).sum()*(xx[1]-xx[0])
  dx=xx[1]-xx[0]
  print ('Mean: %f' % mm)
  print ('Variance: %f' % (((xx-mm)**2.)*ppdf(xx)*dx).sum())
  print ('Third Central Moment: %f' % (((xx-mm)**3.)*ppdf(xx)*dx).sum())
  
  print ('''Test of GK with log function''')

  from scipy.stats import gamma  
  a = 1.99323054838
  r = gamma.rvs(a, size=1000)  
  h1=numpy.mean(r)
  h2=numpy.var(r)
  h3=numpy.mean((r-h1)**3)
  h4=numpy.mean(numpy.log(r))
#  hk=[h1,h2,h3,h4]
  hk=[h1,h4]
  print (hk)
  ppdf,lam=maxentpdf_gkhk(hk,nx=100,log=True)
  
  print (lam)
  xx=numpy.linspace(hk[0]-5*numpy.sqrt(hk[1]),hk[0]+5*numpy.sqrt(hk[1]),100)
  plt.figure()
  plt.hold(True) 
  rv = gamma(a)
  plt.plot(xx[xx>=0], rv.pdf(xx[xx>=0]), 'k-', lw=2, label='frozen pdf')
  plt.plot(xx[xx>=0],ppdf(xx[xx>=0]),'o')
 # plt.plot(xx, norm.pdf(xx,loc=hk[0],scale=numpy.sqrt(hk[1])),'r--')
  plt.xlabel('Concentration at $X_1$')
  plt.ylabel('Probability')  
  plt.show()
  
  h1=numpy.mean(r)
  h2=numpy.var(r)
  h3=numpy.mean((r-h1)**3)
  h4=numpy.mean((r-h1)**4)
  h5=numpy.mean((r-h1)**5)
  h6=numpy.mean((r-h1)**6)
  h7=numpy.mean((r-h1)**7)  
  hk=[h1,h2,h3,h4,h5,h6,h7]
  ppdf2,lam=maxentpdf_gkhk(hk,nx=70)    
  print (hk)
  print (lam)
  xx=numpy.linspace(hk[0]-5*numpy.sqrt(hk[1]),hk[0]+5*numpy.sqrt(hk[1]),100)
  plt.figure()
  plt.hold(True) 
  rv = gamma(a)
  plt.plot(xx[xx>=0], rv.pdf(xx[xx>=0]), 'k-', lw=2, label='frozen pdf')
  plt.plot(xx[xx>=0],ppdf2(xx[xx>=0]),'o')    
  
  print ('''Test of multiple gk-hk maxentpdf''')

  ppdfs=[None]*2  
  m1=numpy.mean(data[:,0])
  m2=numpy.var(data[:,0])
  m3=numpy.mean((data[:,0]-m1)**3)
  m4=numpy.mean((data[:,0]-m1)**4)
  m5=numpy.mean((data[:,0]-m1)**5)
  m6=numpy.mean((data[:,0]-m1)**6)
  hk1=[m1,m2,m3,m4,m5,m6]
  ppdfs[0],lam=maxentpdf_gkhk(hk1,nx=70)
  
  a = 1.99323054838
  r = gamma.rvs(a, size=1000)  
  h1=numpy.mean(r)
  h2=numpy.var(r)
  h3=numpy.mean(numpy.log(r))
  hk2=[h1,h2,h3]
  ppdfs[1],lam=maxentpdf_gkhk(hk2,nx=70,log=True,method='GL')  
  
  hk=[hk1,hk2]  
  cov=numpy.array([[1,0.3],[0.3,1]])
  mmpdf2,_=maxentpdf_gc(ppdfs,cov,hk)
  
  aaa=mmpdf2([1,2])
  x1=numpy.linspace(hk[0][0]-2*numpy.sqrt(hk[0][1]),hk[0][0]+2*numpy.sqrt(hk[0][1]),50)
  x2=numpy.linspace(hk[1][0]-2*numpy.sqrt(hk[1][1]),hk[1][0]+2*numpy.sqrt(hk[1][1]),50)
  x1v,x2v=numpy.meshgrid(x1,x2)
  zmpdf=mmpdf2([x1v,x2v])
  fig=plt.figure()
  ax = fig.add_subplot(111, projection='3d')  
  ax.plot_surface(x1v, x2v, zmpdf, rstride=2, cstride=2, cmap=cm.coolwarm, shade='interp')  
  plt.xlabel('Z1')
  plt.ylabel('Z2')
  ax.set_zlabel('Proba')  
  plt.show()    
    
  
  
  
  
  