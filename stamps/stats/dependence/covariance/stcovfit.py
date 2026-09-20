# -*- coding: utf-8 -*-
from six.moves import range
import numpy as np
from copy import deepcopy

from ....models.covmodel import get_model
from ....general.isspacetime import isspacetime

def covmodelwls(covmodel,covparam,lag_cov_s,lag_cov_t,
  lag_cov_v,lag_cov_n,lag_th=None,theta=None,ratio=None):
  '''

  This function calculate the value of objective function of 
  weighted least square for covaraince function fitting 
  proposed by Cressie in 1985

  Input:    
    
  covmodel    m             list of m nested covariance models in each of 
                            which the spatial and temporal components are put 
                            in a list as [covmodelS,covmodelT] 
  covparam    m             list of covariance parameters for m covmodels in 
                            each component the parameters are listed as 
                            [sill,[covparamS1,covparamS2,..],
                            [covparamT1,covparamT2,..]]
  lag_cov_s   nls by nlt    2D np array of the meshgrid of spatial lags. nls 
                            and nlt denote number of spatial and temporal lags
                            respectively                                        
  lag_cov_t   nls by nlt    2D np array of the meshgrid of temporal lags
  
  lag_cov_v   nls by nlt    2D np array of the empirical covariance values 
                            in the S/T lag meshgrid, ref. stcov function
  lag_cov_n   nls by nlt    2D np array of the number of data pair counts 
                            in the S/T lag meshgrid
  lag_th      na by 1       Optional. For anisotropic case, the directions to 
                            be evaluted   
  theta       1 by nd-1     vector of angle values that characterize the anisotropy. 
                            In a two dimensional space, angle is the trigonometric angle
                            between the horizontal axis and the principal axis of the
                            ellipse. In a three dimensional space, spherical coordinates
                            are used, such that angle(1) is the horizontal trigonometric
                            angle and angle(2) is the vertical trigonometric angle for the
                            principal axis of the ellipsoid. All the angles are measured
                            counterclockwise in degrees and are between -pi/2 and pi/2.                           
  ratio       1 by nd-1     1D array of vector that characterize the ratio for the length of the axes
                            for the ellipse (in 2D) or ellipsoid (in 3D). In a two dimensional
                            space, ratio is the length of the principal axis of the ellipse
                            divided by the length of the secondary axis, so that ratio<1. 
                            i.e., a_max=a, a_min=ratio*a. 
                            In a three dimensional space, ratio(1) is the length of the principal
                            axis of the ellipsoid divided by the length of the second axis, 
                            whereas ratio(2) is length of the principal axis of the ellipsoid
                            divided by the length of the third axis, so that ratio(1)<1 and
                            ratio(2)>1                                 

  Output: 
        
  objfv       scalar        the objective function from Cressie's 1985 paper

  '''
  if lag_th is None:
    lag_cov_est, dummy = covmodelest(lag_cov_s,lag_cov_t,covmodel,covparam)              
    ersqr = (lag_cov_v - lag_cov_est) ** 2
    wt = lag_cov_n / (lag_cov_est[0][0] ** 2 + lag_cov_est ** 2)
    wt /= wt.sum()
    objfv = wt * ersqr
    objfv = objfv[np.where(~np.isnan(objfv))].sum()
  else:
    lag_cov_est,_=anisocovmodelest(covmodel,covparam,theta,ratio,
                       lag_th,lag_cov_s,lag_cov_t)
    objfv=0.
    for i in range(lag_th.size):
      ersqr = (lag_cov_v[i] - lag_cov_est[i]) ** 2
      wt = lag_cov_n[i] / (lag_cov_est[i][0][0] ** 2 + lag_cov_est[i] ** 2)
      wt /= wt.sum()
      objfvi = wt * ersqr
      objfvii = objfvi[np.where(~np.isnan(objfvi))].sum()                 
      objfv=objfv+objfvii 
    
  return objfv

def covmodelfit(lag_cov_s,lag_cov_t,lag_cov_v,lag_cov_n,covmodel,covparam0,
  theta0=None,ratio0=None,lag_th=None,lower_bnd=None, upper_bnd=None): 
  '''

  This function automatically fits the S/T covariance models to S/T empiricl 
  covariances 

  Input:    
    
  lag_cov_s   nls by nlt    2D np array of the meshgrid of spatial lags. nls 
                            and nlt denote number of spatial and temporal lags
                            respectively                                        
  lag_cov_t   nls by nlt    2D np array of the meshgrid of temporal lags
  
  lag_cov_v   nls by nlt    2D np array of the empirical covariance values 
                            in the S/T lag meshgrid, ref. stcov function. For 
                            the anisotropic case, lag_cov_v should be a list 
                            containing na elements with nls by nlt covariance 
                            estimates at different theta speicified in lag_th. 
  lag_cov_n   nls by nlt    2D np array of the number of data pair counts 
                            in the S/T lag meshgrid
  covmodel    m             list of m nested covariance models in each of 
                            which the spatial and temporal components are put 
                            in a list as [covmodelS,covmodelT] 
  covparam0   m             list of intial values covariance parameters for 
                            m covmodels in each component the parameters are 
                            listed as [sill,[covparamS1,covparamS2,..],
                            [covparamT1,covparamT2,..]]
  theta0      scalar        Direction of principle axis of geometric anisotropy 
                            (in radian)
  ratio0      scalar        The ratio between the maximum and minimum ranges of 
                            the anisotropic ellipse. ratio=maximum/minimum and
                            therefore it has range of [0,1]
  lag_th      1 by na       the spatial or S/T covariance model at angle theta 
                            to be evaluated                       
  lower_bnd   m             Optional. The lower bound for covaraince parameters 
                            with the same format of covparam0. Default is None.
  lower_bnd   m             Optional. The lower bound for covaraince parameters 
                            with the same format of covparam0. Default is None.                           
  Output: 
        
  covparam    m             Optimal covariance estimation
  theta       scalar        Optimal direction of principle axis in anisotropic case.
                            Only availabe if anisotropy parameters are specified.
                            E.g., theta0. 
  ratio       scalar        Optimal ratio between principle and secondary axis in 
                            anistropic case
  opt_val     scalar        Optimal value of the objective function

  Remark: 
  1) the fitting uses the weighted least square criteria for covaraince 
  function fitting proposed by Cressie in 1985. The optimization algorithm is 
  bobyqa obtained from nlopt package at http://ab-initio.mit.edu/nlopt/
  2) details of theta and ratio can refer to starpy.stats.stcovfit.anisocovmodelest

  '''
  def objwls(param,ns,nt):
    ''' This function is used by nlopt to assess the objective function
    At this stage, the objective is based upon the covmodelwls function
    '''  
    if theta0 is None:
      covparam=par2covpar(param,ns,nt)
      return covmodelwls(covmodel,covparam,lag_cov_s,lag_cov_t,
                    lag_cov_v,lag_cov_n)  
    else:
      # I should add some check about the inputs for anisotropy 
      covparam=par2covpar(param[:-2],ns,nt)
      return covmodelwls(covmodel,covparam,lag_cov_s,lag_cov_t,
                    lag_cov_v,lag_cov_n,lag_th,param[-2],param[-1]) 
      
  
  def par2covpar(param,ns,nt):   
    '''This function transforms data formats from the form used for nlopt to 
       the form in the regular covariance model
       The nlopt input format is an 1-D np array, in which the covariance 
       parameters are listed in the form of
       [s1, r11, r12, t11, t12, s2, r2, t2, .....] 
       where 
       si is sill of nested model i, 
       rki is the kth spatial range parameters of nested spatial model i
       tki is the kth temporal range parameters of nested temporal model i

       Return format:
         - nugget  (ns=0, nt=0): [sill]
         - spatial (ns>0, nt=0): [sill, param_s_array]
         - s-t sep (ns>0, nt>0): [sill, param_s_array, param_t_array]
    ''' 
    covparam=[]
    k=0
    for m in range(len(covmodel)):
      if ns[m]>0 and nt[m]>0:
        covparam.append([param[k],param[k+1:k+ns[m]+1],
                       param[k+ns[m]+1:k+ns[m]+nt[m]+1]])
      elif ns[m]>0 and nt[m]==0:
        covparam.append([param[k],param[k+1:k+ns[m]+1]])
      elif ns[m]==0 and nt[m]>0:
        covparam.append([param[k],param[k+1:k+nt[m]+1]])
      else:
        # Nugget: no spatial or temporal range parameters — return [sill] only
        covparam.append([param[k]])
      
      k=k+ns[m]+nt[m]+1        
    return covparam
    
  def covpar2par(covparam):    
    '''This function transforms data formats from regular covariance model to 
       the form used for nlopt. 
       
       Remark: The regular covariance model can refer to covmodeldef function.
       Scalar range parameters (e.g. covparam = [(sill, range)]) are supported
       in addition to list-wrapped parameters (e.g. [(sill, [range])]).
       '''                 
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
        except TypeError:
          # par is a non-iterable scalar (e.g. a bare sill or range value)
          fit_params.append(par)
          if i == 1:
            npars[k] += 1   # one scalar spatial range parameter
          elif i >= 2:
            npart[k] += 1   # one scalar temporal range parameter
    fit_params=np.array(fit_params, dtype=float)      
    return fit_params,npars,npart          
  
  pars,ns,nt=covpar2par(covparam0)
  if theta0 is not None:
    pars=np.append(pars,[theta0,ratio0])
    
  args=[ns,nt]
  nm=len(covparam0)
  idsill=[0]
  idnrg=[]
  if nm>1:
    k=0
    for i in range(nm):
      if ns[i]>1:        
        idnrg.append(np.arange(k+2,k+ns[i]+1))
      if nt[i]>1:
        idnrg.append(np.arange(k+ns[i]+2,k+ns[i]+nt[i]+1))
      if i < nm-1:  
        idsill.append(k+ns[i]+nt[i]+1)
        k=k+ns[i]+nt[i]+1
  
  idnrg=[item for sublist in idnrg for item in sublist]
  idnrg=np.array(idnrg)
      
  npars=len(pars)
  if theta0 is None:    
    idrg=np.setdiff1d(np.arange(npars),idsill)
    idrg=np.setdiff1d(idrg,idnrg)
  else:
    idrg=np.setdiff1d(np.arange(npars-2),idsill)
    idaniso=np.arange(npars-2,npars)

  if lower_bnd:
    low_bnd=covpar2par(lower_bnd)
  else:  
    low_bnd=np.empty(npars)
    low_bnd[idsill]=pars[idsill]*0.3    
    low_bnd[idrg]=pars[idrg]*0.3
    if idnrg.size>0:
      low_bnd[idnrg]=pars[idnrg]*0.8
      low_bnd[idnrg[np.where(low_bnd[idnrg]<0.5)]]=0.5
    if theta0 is not None:
      low_bnd[idaniso[0]]=pars[idaniso[0]]-np.pi/4
      low_bnd[idaniso[1]]=pars[idaniso[1]]*0.5
  if upper_bnd: 
    up_bnd=covpar2par(upper_bnd)
  else:
    up_bnd=np.empty(npars)
    up_bnd[idsill]=pars[idsill]*2.
    up_bnd[idrg]=pars[idrg]*5.
    if idnrg.size>0:
      up_bnd[idnrg]=pars[idnrg]*1.2
    if theta0 is not None:
      up_bnd[idaniso[0]]=pars[idaniso[0]]+np.pi/4
      up_bnd[idaniso[1]]=np.min([pars[idaniso[1]]*2,1.])
          
  from ....general import bobyqa as bobyqa  # lazy: nlopt is optional
  result, opt_val=bobyqa.bobyqa( objwls, pars, args, low_bnd, up_bnd )
  
  if theta0 is None:
    covparam=par2covpar(result,ns,nt)
    return covparam, opt_val
  else:
    covparam=par2covpar(result[:-2],ns,nt)
    theta=result[-2]
    ratio=result[-1]
    return covparam, theta, ratio, opt_val
      
  
    
    
    #    
    ##===============================================================================
    # def autofitcov(CSTguess,COVparams,Smodels,Tmodels):
    #    CSTguess=CSTguess.reshape((-1,3)).tolist()
    #    fittingmodels=[]
    #    for cst,smodel,tmodel in zip(CSTguess,Smodels,Tmodels):
    #        cst.insert(1,smodel)
    #        cst.insert(3,tmodel)
    #        fittingmodels.append(cst)
    # 
    #    fittingmodels = tuple(fittingmodels)
    #    return fitcovariance(fittingmodels,COVparams)
    #===============================================================================

def covmodeldef(covmodel, covparam):
  ''' covariance model definition
  
  SYNTAX:
    fit_models=covmodeldef(covmodel,covparam)
  
  INPUTS:  
  covmodel    m             list of m nested covariance models in each of 
                            which the spatial and temporal components are put 
                            in a list as [covmodelS,covmodelT] 
  covparam    m             list of covariance parameters for m covmodels in 
                            each component the parameters are listed as 
                            [sill,[covparamS1,covparamS2,..],
                            [covparamT1,covparamT2,..]]
  OUTPUTS:                          
  fit_models  n_model by 5  list in which row for nest model number and 
                            col for cell, spatial model, spatial range,
                            temporal model, temporal range
  
  Remark: the use of fit_models is originally designed for widely-used 
  covariance models that has only two parameters for both spatial and 
  temporal models                           
  '''
  fit_models=[]
  for i,(modelS,modelT) in enumerate(covmodel):
    if modelS=='nugget' or modelS=='nug':
      rangeS=np.NaN
      rangeT=np.NaN
    else:
      rangeS=covparam[i][1][0]
      rangeT=covparam[i][2][0]
    fit_models.append([covparam[i][0],modelS,rangeS,
                                      modelT,rangeT])
  return fit_models

def covmodelest(lag_cov_s, lag_cov_t, covmodel, covparam):

  '''
  This function calculate the model predicted value at spatial-temporal coordinate that user input

  Syntax:
  COV_est =covmodelest(lag_cov_s,lag_cov_t,covmodel,covparam)
  
  Input:    
    
  lag_cov_s   nls by nlt    2D np array of the meshgrid of spatial lags. nls 
                            and nlt denote number of spatial and temporal lags
                            respectively
                                        
  lag_cov_t   nls by nlt    2D np array of the meshgrid of temporal lags
  covmodel    m             list of m nested covariance models in each of 
                            which the spatial and temporal components are put 
                            in a list as ['covmodelS/covmodelT']. The details of
                            available covariance model see below
  covparam    m             list of covariance parameters for m covmodels in 
                            each component the parameters are listed as 
                            [sill,[covparamS1,covparamS2,..],
                            [covparamT1,covparamT2,..]]                           
    
  Output:
  COV_est     nls by nlt    2D np array of the model predicted value of 2D meshgrid
  Cov_i       list          list includes m nls by nlt 2D np array for
                            all nested models      
  
  Remark: Five covariance models are available for use now, including
  gaussian(gau), exponential, spherical, holecos, and nugget
    
  
  
  '''

  isST, isSTsep, model_res = isspacetime(covmodel)
  if isST:
    if isSTsep:
      modelS, modelT = model_res
      Ki = []
      for model_s, model_t, param_i in zip(modelS, modelT, covparam):
        if len(param_i) == 1:
          if model_s=='nuggetC' or model_s=='nuggetC':
            sill=param_i
            param_s=[None]
            param_t=[None]
          else:
            print ('covparam is not consisent with covmodel')
            raise 
        else:
          sill, param_s, param_t = param_i
        model_s = get_model(model_s)
        model_t = get_model(model_t)                       
        Ki.append( sill * model_s( lag_cov_s, 1., param_s ) * model_t( lag_cov_t, 1., param_t ) )
      return sum(Ki), Ki # K, KK in matlab
    else:
      (modelS,) = model_res
      Ki=[]
      for model_s, param_i in zip(modelS, covparam):
        sill, param_s, s_t_ratio = param_i
        model_s = get_model(model_s)
        Ki.append(
          sill * model_s(lag_cov_s + s_t_ratio * lag_cov_t, 1., param_s))
      return sum(Ki), Ki # K, KK in matlab
  else:
    (modelS,) = model_res
    Ki = []
    for model_s, param_i in zip(modelS, covparam):
      if len(param_i) == 1:
        # Nugget only: [sill]
        if model_s == 'nuggetC':
          sill = param_i[0]
          param_s = [None]
        else:
          raise ValueError('covparam is not consistent with covmodel')
      elif len(param_i) == 2:
        # Normal spatial case: [sill, param_s]
        sill, param_s = param_i
      else:
        # 3-element case: [sill, param_s, param_t] produced by par2covpar's else
        # branch when ns==0 and nt==0 (nugget).  Use sill + spatial params only.
        sill = param_i[0]
        param_s = param_i[1] if (hasattr(param_i[1], '__len__') and len(param_i[1]) > 0) else [None]
      model_s = get_model(model_s)
      Ki.append( sill * model_s(lag_cov_s, 1., param_s))
    return sum(Ki), Ki # K, KK in matlab

def anisocovmodelest(covmodel,covparam,theta,ratio,lag_th,lag_cov_s,lag_cov_t=None):
  '''
  Evaluate the spatially-aniostrpy covariance with given directional angle of 
  the prinipal axis with maximum range and its ratios to the other axes, and 
  isotropic model
  
  INPUT:
  covmodel    m             list of m nested isotropic covariance models in each of 
                            which the spatial and temporal components are put 
                            in a list as ['covmodelS/covmodelT']. The details of
                            available covariance model see below
  covparam    m             list of isotropic covariance parameters for m covmodels in 
                            each component the parameters are listed as 
                            [sill,[covparamS1,covparamS2,..],
                            [covparamT1,covparamT2,..]]      
  theta       1 by nd-1     vector of angle values that characterize the anisotropy. 
                            In a two dimensional space, angle is the trigonometric angle
                            between the horizontal axis and the principal axis of the
                            ellipse. In a three dimensional space, spherical coordinates
                            are used, such that angle(1) is the horizontal trigonometric
                            angle and angle(2) is the vertical trigonometric angle for the
                            principal axis of the ellipsoid. All the angles are measured
                            counterclockwise in degrees and are between -pi/2 and pi/2.                           
  ratio       1 by nd-1     1D array of vector that characterize the ratio for the length of the axes
                            for the ellipse (in 2D) or ellipsoid (in 3D). In a two dimensional
                            space, ratio is the length of the principal axis of the ellipse
                            divided by the length of the secondary axis, so that ratio<1. 
                            i.e., a_max=a, a_min=ratio*a. 
                            In a three dimensional space, ratio(1) is the length of the principal
                            axis of the ellipsoid divided by the length of the second axis, 
                            whereas ratio(2) is length of the principal axis of the ellipsoid
                            divided by the length of the third axis, so that ratio(1)<1 and
                            ratio(2)>1
  lag_th     1 by na        the spatial or S/T covariance model at angle theta 
                            to be evaluated
  lag_cov_s  nls by nlt     the spatial or S/T covariance model with spatial 
                            distances to be evaluated
  lag_cov_t  nls by nlt     the temporal lags of S/T covariance model to be evaluated

  OUTPUT:

  Csta       (ns by nt) by na  a list with na length contains S/T covariance model  
                               at every angle      
  Cstai      list          a list with na length. Each component has a list 
                           containing the details of each component of nested
                           model
  
  '''  
    
  isST, isSTsep, model_res = isspacetime(covmodel)
  Ka = [None]*lag_th.size
  Kai = [None]*lag_th.size
  for i, phi in enumerate(lag_th):
    if isST:
      if isSTsep:
        modelS, modelT = model_res
        Ki = []
        for model_s, model_t, param_i in zip(modelS, modelT, covparam):
          if len(param_i)==1:
            if model_s == 'nuggetC' or model_t == 'nuggetC':
              sill=param_i
              param_s = [None]
              param_t = [None]
            else:
              print ('covparam is not consisent with covmodel')
              raise 
          else:
            sill, param_s, param_t = param_i
            if param_s.size==0: 
              param_s=[None]
            if param_t.size==0:
              param_t=[None]
          model_s = get_model(model_s)
          model_t = get_model(model_t)
          if param_s[0] is not None:  #param_s[0] is the correlation length
            param_s = (
              param_s * ratio
              / np.sqrt(
                ratio**2 * np.cos(phi-theta)**2
                + np.sin(phi-theta)**2)
              )
          Ki.append(
            sill * model_s(lag_cov_s, 1., param_s)
            * model_t(lag_cov_t, 1., param_t)
            )
          Ka[i] = sum(Ki)
          Kai[i] = Ki  # K, KK in matlab
      else:
        (modelS,) = model_res
        Ki = []
        for model_s, param_i in zip(modelS, covparam):
          sill, param_s, s_t_ratio = param_i
          model_s = get_model(model_s)
          # Assume param_s[0] is the correlation length
          # This may not be true in non-seperable S/T covariance
          if param_s[0] is not None:  
            param_s = param_s*ratio/np.sqrt(ratio**2*np.cos(phi-theta)**2+np.sin(phi-theta)**2)          
          Ki.append( sill * model_s( lag_cov_s + s_t_ratio * lag_cov_t, 1., param_s ) )
        Ka[i] = sum(Ki)
        Kai[i] = Ki # K, KK in matlab
    else:
      (modelS,) = model_res
      Ki = []
      for model_s, param_i in zip(modelS, covparam):
        if len(param_i) == 1:
          if model_s == 'nuggetC' :
            sill = param_i[0]
            param_s = [None]
          else:
            print ('covparam is not consisent with covmodel')
            raise ValueError('covparam is not consistent with covmodel')
        else:  # C3 fix: only unpack when len(param_i) > 1
          sill, param_s = param_i
        model_s = get_model(model_s)
        if param_s[0] is not None:
          param_s = (
            param_s * ratio
            / np.sqrt(
              ratio**2 * np.cos(phi-theta)**2
              + np.sin(phi-theta)**2)
            )
        Ki.append(sill * model_s(lag_cov_s, 1., param_s))
      Ka[i]=sum(Ki)
      Kai[i]=Ki # K, KK in matlab  
  return Ka, Kai

def covdownscale(bigcovmod, scale, method=None):
    '''
        covmodel:
            [[c,model_s, bs, model_t, bt],
              ...,
             [c,model_s, bs, model_t, bt]]

        return estimated downscale covariance model
    '''

    model_count = len(bigcovmod)
    #parse covmod to get lower and upper bound
    low_bnd = []
    up_bnd = []
    for cov in bigcovmod:
        low_bnd += [0.5*cov[0], 0.5*cov[2], 0.5*cov[4]]
        up_bnd += [1.5*cov[0], 1.5*cov[2], 1.5*cov[4]]
    low_bnd = np.array(low_bnd)
    up_bnd = np.array(up_bnd)
    big_bt = int(up_bnd[2::3].max()) # bt max, e.g. opt function range

    low_bnd[1::3] = up_bnd[1::3] #spatial not change
    up_bnd[2::3] *= scale #temparal change scale
    init_gss = low_bnd + (up_bnd - low_bnd)*0.5

    smallcovmod = deepcopy(bigcovmod)
    big_cov_mod_f = get_cov_mod_func(bigcovmod)
    
    def opt_func(x, big_cov_mod_f, smallcovmod, big_bt):
        for i, cst in enumerate(np.split(x, len(x)/3)):
            smallcovmod[i][0] = cst[0]
            smallcovmod[i][2] = cst[1]
            smallcovmod[i][4] = cst[2]
        small_cov_mod_f = get_cov_mod_func(smallcovmod)

        range_big_bt = range(big_bt)
        big_lags = np.zeros((len(range_big_bt),2))
        big_lags[:,1] = range_big_bt
        big_cov_z = big_cov_mod_f(big_lags)

        range_small_bt = range(scale)
        small_lags = np.zeros((len(range_small_bt), 2))
        small_lags[:,1] = range_small_bt
        multiplier = np.arange(
            scale,0,-1, dtype=np.float64
            ).reshape((scale,1))
        multiplier[1:,:] *=2
        opt_val = 0.0

        for i in range_big_bt:
            small_lags_copy = deepcopy(small_lags)
            small_lags_copy[:,1] += i*scale
            small_cov_z = small_cov_mod_f(small_lags_copy)
            opt_val +=\
                abs(big_cov_z[i][0]\
                    - (small_cov_z * multiplier / scale**2).sum()
                    )
        print (x, opt_val)
        return opt_val
            
    args = (big_cov_mod_f, smallcovmod, big_bt)

    from ....general import bobyqa as bobyqa  # lazy: nlopt is optional
    result, opt_val = bobyqa.bobyqa(
        opt_func, init_gss, args, low_bnd, up_bnd, stop_val=10**-8)

    #scale small covariance
    for cov in smallcovmod:
        cov[4] /= scale


    return smallcovmod, opt_val
  
def get_cov_mod_func(covmodel):
    '''
        get covariance model as a python function with input lag
    '''
    def cov_func(lag):
        '''
            lag: n by 2 2D np array, column0 is for spatial, 1 for time
        '''
        s_lag = lag[:, 0]
        t_lag = lag[:, 1]
        Ki = []
        for cov in covmodel:
            c, fs, bs, ft, bt = cov
            fs = get_model(fs)
            ft = get_model(ft)
            Ki.append(c*fs(s_lag, 1., bs)*ft(t_lag, 1., bt))
        return sum(Ki).reshape((-1, 1))
    return cov_func

def cal_cov_mod(lag, cov_mod):
    ''' calculate covariance value at each lag
    lag: n by 2 np 2d array, [[s_lag, t_lag],...,[s_lag, t_lag]]
    cov_mod: [[c,model_s, bs, model_t, bt], ..., [c,model_s, bs, model_t, bt]]
    '''
    cov_f = get_cov_mod_func(cov_mod)
    cov_z = cov_f(lag)
    return cov_z

def _index_to_fit_func(index):
    dictionary={"gaussian":_Gau,
                "exponential":_Exp,
                "spherical":_Sph,
                "holecos":_HoC,
                "nugget":_Nug,
                "gau":_Gau,
                "exp":_Exp,
                "sph":_Sph,
                "hoc":_HoC,
                "nug":_Nug,}
    return dictionary[index]
  

def _Gau(bandwidth,lag):
    value = np.exp(-3*(lag/bandwidth)**2)
    return value

def _Exp(bandwidth,lag):
    value = np.exp(-3*(lag/bandwidth))
    return value

def _Sph(bandwidth,lag):
    value=lag.copy()
    boollag = lag <= bandwidth
    value[boollag] = 1.0-1.5*(lag[boollag]/bandwidth)\
                    +0.5*(lag[boollag]/bandwidth)**3;
    boollag = lag > bandwidth 
    value[boollag] = 0.0
    return value;

def _HoC(bandwidth,lag):
    value = np.cos(3.1415926*lag/bandwidth)
    return value

def _Nug(bandwidth,lag):
    value=lag.copy()
    boollag = lag == 0
    value[boollag] = 1.
    boollag = lag != 0
    value[boollag] = 0.
    return value;

def coregfit(
    d: np.ndarray,
    V: list,
    o: np.ndarray,
    models,
    param0,
    options: dict = None,
) -> list:
    """Fit a Linear Model of Coregionalization (LMC) by Iterated Alternating WLS.

    Use an iterated least-squares algorithm to fit a *multivariate* covariance
    model — the Linear Model of Coregionalization (LMC) — to a set of
    empirical (cross-)covariance or (cross-)variogram estimates.

    Under the LMC, the joint cross-covariance between variable ``i`` and
    variable ``j`` at spatial lag ``h`` is:

    .. math::

        C_{ij}(h) = \\sum_{k=1}^{K} B_k[i,j] \\cdot g_k(h)

    where ``g_k`` is the common normalised shape function (evaluated with
    unit sill) for structure ``k`` and :math:`B_k` is a positive-semi-definite
    (PSD) ``nv × nv`` *sill matrix* to be estimated.  All cross-covariances
    share the same set of shape functions; only the sill coefficients differ.

    The estimation is performed by Iterated Alternating Weighted Least Squares
    (IALS): for each structure ``k`` in turn, the optimal ``B_k`` is obtained
    analytically given the current estimates of all other ``B_{l≠k}``, then
    projected onto the cone of PSD matrices.  This is analogous to
    ``coregfit.m`` in the original BMElib package.

    Parameters
    ----------
    d : array_like, shape (nc,)
        Sorted vector of representative distances for each lag class (the
        class centres returned by :func:`stcov`).
    V : array_like or nested list
        Empirical (cross-)covariance estimates:

        * **Univariate** — a 1-D array of length ``nc``.
        * **Multivariate** — an ``nv × nv`` nested Python list (or list of
          lists), where ``V[i][j]`` is a 1-D array of length ``nc`` giving
          the empirical covariance between variable ``i`` and variable ``j``
          at the lags in ``d``.  Diagonal entries ``V[i][i]`` are the
          auto-covariances; off-diagonal entries are cross-covariances.

    o : array_like, shape (nc,)
        Number of pairs of points contributing to each lag class.  Used as
        weights when ``options['weighted'] = True``.
    models : str or list of str
        Name(s) of the covariance model(s) used to build the nested
        structure.  Each name must correspond to a function in
        ``stamps.models.covmodel`` (e.g. ``'exponentialC'``,
        ``'gaussianC'``, ``'nuggetC'``).  A *single* string means a
        single-structure LMC; a list of ``K`` strings defines a
        ``K``-structure nested LMC.
    param0 : tuple or list of tuples
        Initial parameter guess for each model structure.  Each element is a
        tuple ``(sill_0, range_param)`` where:

        * ``sill_0`` — arbitrary scalar initial sill (will be overwritten
          by the fitted ``nv × nv`` sill matrix).
        * ``range_param`` — the range parameter(s) kept **fixed** during
          fitting (scalar, list, or ``None`` for a nugget effect).

        For a single model, pass one tuple; for ``K`` structures, pass a
        list of ``K`` tuples.
    options : dict, optional
        Override any of the following defaults:

        ``display`` : bool (default ``False``)
            If ``True``, print iteration info and fitted values.
        ``rtol`` : float (default ``1e-4``)
            Relative tolerance for IALS convergence.
        ``max_iter`` : int (default ``1000``)
            Maximum number of IALS iterations.
        ``weighted`` : bool (default ``True``)
            If ``True``, weight each lag class by ``o / sum(o)`` (WLS);
            otherwise use equal weights (OLS).
        ``cross_terms`` : bool (default ``True``)
            If ``True``, include cross-covariance terms in the WSS
            criterion; otherwise fit only the diagonal terms.

    Returns
    -------
    lmc_params : list of dict
        Fitted LMC parameters — one dictionary per structure ``k``:

        ``'model'`` : str
            Model name (same as the corresponding entry in ``models``).
        ``'sill_matrix'`` : ndarray, shape ``(nv, nv)``
            Fitted positive-semi-definite sill matrix for structure ``k``.
        ``'range_param'`` : scalar or list or None
            Fixed range parameter(s) for structure ``k``, unchanged from
            ``param0``.

    Notes
    -----
    * The output list is directly consumable by :func:`cokriging` and
      :func:`cokrigingT` as the ``lmc_models`` argument.
    * The algorithm is based on the Iterated Alternating WLS described in
      Goovaerts (1997) *Geostatistics for Natural Resources Evaluation*.
    * If a single model string is passed, the function directly solves the
      single-structure WLS problem without iteration (since there are no
      competing structures).
    * Negative eigenvalues in the unconstrained WLS estimate of ``B_k`` are
      clipped to a small positive value (``0.001 * min_positive_eigenvalue``)
      to maintain positive semi-definiteness.

    Examples
    --------
    Fit a two-variable LMC with one exponential structure:

    >>> import numpy as np
    >>> from stamps.stats.dependence.stcovfit import coregfit
    >>> d = np.array([10., 20., 30., 40., 50.])
    >>> o = np.array([100, 150, 130, 120, 110], dtype=float)
    >>> # Synthetic: V[i][j][lag] = sill_ij * exp(-3*d/range)
    >>> rng = np.random.default_rng(0)
    >>> V = [[None, None], [None, None]]
    >>> C_true = np.array([[1.0, 0.6], [0.6, 0.8]])
    >>> for i in range(2):
    ...     for j in range(2):
    ...         V[i][j] = C_true[i,j] * np.exp(-3*d/25) + rng.normal(0, 0.02, 5)
    >>> lmc = coregfit(d, V, o, 'exponentialC', (1.0, 25.0))
    >>> print(lmc[0]['sill_matrix'].round(2))
    """
    from ....models.covmodel import get_model  # local to avoid circular import

    d = np.asarray(d, dtype=float).ravel()
    o = np.asarray(o, dtype=float).ravel()
    nc = len(d)

    # ------------------------------------------------------------------ #
    # Default options
    # ------------------------------------------------------------------ #
    defaults = {
        "display": False,
        "rtol": 1e-4,
        "max_iter": 1000,
        "weighted": True,
        "cross_terms": True,
    }
    opts = {**defaults, **(options or {})}

    # ------------------------------------------------------------------ #
    # Determine nv (number of variables) and normalise V to nested list
    # ------------------------------------------------------------------ #
    if isinstance(V, np.ndarray) and V.ndim == 1:
        # univariate
        nv = 1
        V_cell = [[V]]
    elif isinstance(V, np.ndarray) and V.ndim == 2:
        # rows = variables, columns = lags  (square if nv==nc accidentally)
        # Treat as nv × nc where nv rows each holding a 1-D series
        # This is ambiguous; prefer the nested-list path.
        nv = 1
        V_cell = [[V]]
    else:
        # nested list V[i][j] each of length nc
        nv = len(V)
        V_cell = V

    # ------------------------------------------------------------------ #
    # Normalise models / param0 to lists
    # ------------------------------------------------------------------ #
    if isinstance(models, str):
        models_list = [models]
        param0_list = [param0 if not isinstance(param0[0], (list, tuple, np.ndarray))
                       else param0[0]]
        nm = 1
    else:
        models_list = list(models)
        param0_list = list(param0)
        nm = len(models_list)

    # ------------------------------------------------------------------ #
    # Evaluate unit-sill shape functions  g[:, k] for each structure
    # ------------------------------------------------------------------ #
    g = np.zeros((nc, nm))
    range_params = []
    for k, (model_k, param_k) in enumerate(zip(models_list, param0_list)):
        model_fn = get_model(model_k)
        # param_k = (sill, range_param) or just (sill,) for nugget
        if len(param_k) >= 2:
            rp = param_k[1]
        else:
            rp = None
        g[:, k] = np.asarray(model_fn(d, 1.0, rp)).ravel()
        range_params.append(rp)

    # ------------------------------------------------------------------ #
    # Weights
    # ------------------------------------------------------------------ #
    o_sum = o.sum()
    if opts["weighted"] and o_sum > 0:
        w = o / o_sum
    else:
        w = np.ones(nc) / nc

    # ------------------------------------------------------------------ #
    # Single structure: closed-form WLS, no iteration needed
    # ------------------------------------------------------------------ #
    if nm == 1:
        gk = g[:, 0]
        denom = w.dot(gk ** 2)
        C_k = np.zeros((nv, nv))
        for row in range(nv):
            for col in range(row, nv):
                C_k[row, col] = (w * gk).dot(V_cell[row][col]) / (denom + 1e-300)
                C_k[col, row] = C_k[row, col]
        # Project to PSD
        C_k = _project_psd(C_k)
        return [{"model": models_list[0], "sill_matrix": C_k, "range_param": range_params[0]}]

    # ------------------------------------------------------------------ #
    # Multiple structures: IALS iterations
    # ------------------------------------------------------------------ #
    # Initialise each C_k as a diagonal matrix with mean(V[i][i]) / nm
    C = []
    for k in range(nm):
        diag_vals = np.array([np.mean(V_cell[i][i]) for i in range(nv)])
        C.append(np.diag(np.maximum(diag_vals / nm, 0.0)))

    WSS_old = np.inf
    for _iter in range(opts["max_iter"]):
        for ki in range(nm):
            other_idx = [j for j in range(nm) if j != ki]
            gki = g[:, ki]

            C_temp = np.zeros((nv, nv))
            for row in range(nv):
                for col in range(row, nv):
                    # Residual after removing all other structures
                    dV = np.array(V_cell[row][col], dtype=float)
                    for l in other_idx:
                        dV = dV - g[:, l] * C[l][row, col]
                    # WLS estimate (unnormalized)
                    C_temp[row, col] = (w * gki).dot(dV)
                    C_temp[col, row] = C_temp[row, col]

            # Project to PSD
            C_temp = _project_psd(C_temp)

            # Normalise
            denom = w.dot(gki ** 2)
            C[ki] = C_temp / (denom + 1e-300)

        # ---- convergence criterion (weighted sum of squares) ---- #
        WSS = 0.0
        for ki in range(nm):
            col_range = range(nv) if opts["cross_terms"] else range(0)
            for row in range(nv):
                cols = range(row, nv) if opts["cross_terms"] else [row]
                for col in cols:
                    residual = V_cell[row][col] - g[:, ki] * C[ki][row, col]
                    WSS += float(np.sum(w * residual ** 2))

        if WSS_old != np.inf and abs((WSS_old - WSS) / (abs(WSS_old) + 1e-300)) < opts["rtol"]:
            break
        WSS_old = WSS

    if opts["display"]:
        print(f"coregfit converged after {_iter + 1} iterations, WSS={WSS:.6g}")

    return [
        {"model": models_list[k], "sill_matrix": C[k], "range_param": range_params[k]}
        for k in range(nm)
    ]


def _project_psd(C: np.ndarray, min_eigval_frac: float = 1e-3) -> np.ndarray:
    """Project a symmetric matrix to the positive-semi-definite cone.

    Eigenvalues that are ≤ 0 are replaced by
    ``min_eigval_frac * min_positive_eigenvalue`` (or ``1e-9`` if all
    eigenvalues are non-positive).

    Parameters
    ----------
    C : ndarray, shape (n, n)
        Symmetric matrix to project.
    min_eigval_frac : float
        Fraction of the smallest positive eigenvalue to use for clipping.

    Returns
    -------
    ndarray, shape (n, n)
        Nearest PSD matrix (symmetric).
    """
    C_sym = (C + C.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(C_sym)
    pos_mask = eigvals > 0
    if pos_mask.any():
        floor = min_eigval_frac * eigvals[pos_mask].min()
    else:
        floor = 1e-9
    eigvals_clipped = np.where(eigvals <= 0, floor, eigvals)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


if __name__ == "__main__":
    covmodel=[['nuggetC/nuggetC'],['exponentialC/exponentialC'],['exponentialC/exponentialC']]
    # covmodel=[['nugget','nugget'],['exponential','exponential'],['exponential','exponential']]
    covparam=[[0.2,[None],[None]],[0.7,[100],[10]],[0.3,[50],[5]]]
    # fit_models = [[0.7,'exponential',100,'exponential',10],
    #               [0.3,'exponential',50,'exponential',5]]
    lag_s = np.linspace(0,100,11)
    lag_t = np.linspace(0,10,6)
    lag_th = np.linspace(-np.pi/2,np.pi/2,10)
    lag_cov_t, lag_cov_s = np.meshgrid( lag_t, lag_s )
    lag_cov_est,cov_est_i = covmodelest( lag_cov_s,lag_cov_t,covmodel, covparam)
    Ka,Kai=anisocovmodelest(covmodel,covparam,np.pi/6,0.8,lag_th,lag_cov_s,lag_cov_t)
    lag_cov_n = np.ones( lag_cov_est.shape )
    lag_cov_v = lag_cov_est + 0.2*(np.random.rand( *lag_cov_est.shape ) - 0.5)
    objv = covmodelwls(covmodel,covparam,lag_cov_s,lag_cov_t,
                  lag_cov_v,lag_cov_n)
    result, opt_val=covmodelfit(lag_cov_s,lag_cov_t,lag_cov_v,lag_cov_n,covmodel,covparam)
    print (objv)

    try:
        from matplotlib import pyplot as plt
        lag_s_line = np.linspace(0,100,51)
        lag_t_line = np.linspace(0,10,51)
        lag_cov_t_line, lag_cov_s_line = np.meshgrid( lag_t_line, lag_s_line )
        lag_cov_v_line, dummy = covmodelest( lag_cov_s_line,lag_cov_t_line,
                                     covmodel,covparam )

        plt.figure(1)
        plt.subplot(211)
        plt.plot(lag_s, lag_cov_v[:,0], 'bo', lag_s_line, lag_cov_v_line[:,0], 'b--')
        plt.subplot(212)
        plt.plot(lag_t, lag_cov_v[0], 'ro', lag_t_line, lag_cov_v_line[0], 'r--')

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(2)
        ax3d = fig.add_subplot(111, projection='3d')
        ax3d.plot_wireframe(lag_cov_s_line, lag_cov_t_line, lag_cov_v_line)
        ax3d.scatter(lag_cov_s, lag_cov_t,lag_cov_v,c='r')
        plt.show()
    except ImportError as e:
        print ('Warning: Import matplotlib fault, cannot draw.')
        print ('Error Message:', str(e))
