#!/usr/bin/python
#-*- coding: utf-8 -*-

'''
於2011年5月10號建立
Created on 2011/5/10

參考matlab版的stcov而做，想法與實作方法上有些不同

這是用來算時空分析的變異數的函數，輸入與輸出說明如下：
lagCOVv,lagCOVn,lagCOVs,lagCOVt, = stcov(grid_s,grid_t,grid_v,
                       lagS,lagS_range,
                       lagT,lagT_range):
    輸入變數：
    grid_s：np 2維陣列(因為包含x與y)，為所有測站的空間座標值
    grid_t：np 1維陣列，為所有測量的時間座標值
    grid_v：np 2維陣列，以grid_s為列，grid_t為行所對應的測量值
    lagS：np 1維陣列，變異數空間軸的點
    lagS_range：np 1維陣列，對應該點計算時的空間範圍
    lagT：np 1維陣列，變異數時間軸的點
    lagT_range：np 1維陣列，對應該點計算時的時間範圍
    
    輸出變數：
    lagCOVs：lagCOVv相對應點之空間差值 
            [[0,0,0,0,0]
             [1,1,1,1,1]
             [2,2,2,2,2]
                [...]   ]
    lagCOVt：lagCOVv相對應點之時間差值
            [[1,2,3,4,5]
             [1,2,3,4,5]
             [1,2,3,4,5]
                [...]   ]
    lagCOVv：以lagS為列，lagT為行的2維變異數陣列
    lagCOVn：變異數陣列計算時包含的總資料筆數 
    
    舉例：
           以下都必須轉成np array
    grid_s = [ [1.,3.]
               [2.,4.]
               [5.,7.] ]
    grid_t = [ 1. , 4., 6. ]
    grid_v = [ [1.,2.,3.]
               [4.,np.nan,6.]
               [7.,8.,9.]
    
@作者：顧尚真
@author: KSJ
'''
import itertools
import warnings
from six.moves import range
import numpy as np
import re
try:
    from ....aot.general.diffarray import diffarray as pdist_numba
    from ....aot.general.diffarray import cdist as cdist_numba
except ImportError:
    # numba AOT module not available; falling back to pure-Python diffarray.
    # Enable ImportWarning (-W default::ImportWarning) to see this message.
    warnings.warn(
        'stamps: numba AOT diffarray not found; using pure-Python fallback.',
        ImportWarning, stacklevel=2,
    )
    from ....general.diffarray import diffarray as pdist_numba
    from ....general.diffarray import cdist as cdist_numba
# Alias so that the function body can use either name -------------------------
diffarray = pdist_numba
cdist_numba_alias = cdist_numba
# ----------------------------------------------------------------------------
from ....general.diffarray import diffarray_split

from scipy.spatial.distance import cdist as cdist_scipy
from scipy.spatial import cKDTree as KDTree
from ....general.valstvgx import valstv2stg
import pandas as pd

try:
    import dask
except ImportError:
    # dask is optional; stcov_dask / stcov_dask_half will be unavailable.
    warnings.warn(
        'stamps: dask not installed; stcov_dask_half is unavailable.',
        ImportWarning, stacklevel=2,
    )


### For anisotropy development
def stcov(grid_s, grid_t=None, grid_v=None, lagS=None, lagS_range=None,
          lagT=None, lagT_range=None, ang=None, angtol=None, DataObj=None):
  '''
  Estimates the space/time cross covariance between variable Zst 
  with measurements at fixed monitoring stations. The monitoring 
  stations for Zst are located at coordinates cMS, and 
  measuring events are at times tME
  
  SYNTAX:
  COVv,COVn,lagSS,lagTT=  
  stcov(grid_s,grid_t,grid_v,lagS,lagS_range,lagT,lagT_range, ang=None, 
        angtol=None, DataObj = None)
  
  Input:
  	
  grid_s      ns by 2     2D array    cMS
  grid_t      nt by 1     2D array or None
                                      tME. Pass None (default) for pure-spatial
                                      observations; a dummy time step is used
                                      internally.
  grid_v      ns by nt    2D array, or 1D (ns,) array for the pure-spatial case.
                                      When 1D, the array is automatically reshaped
                                      to (ns, 1).
  lagS        nls by 1    1D array    vector with the r lags    
  lagS_range  nls by 1    1D array    vector with the tolerance for the r lags
  lagT        nlt by 1    1D array    vector with the t lags. default is None
                                      for pure spatial case
  lagT_range  nlt by 1    1D array    vector with the tolerance for the t lags
  ang         scalar                  specified direction in radian for directional covariance. 
                                      Default is None for omnidirectional. 
                                      Direction is in radian and zero denotes 
                                      the horizontal axis
  angtol      scalar                  angle tolerance in radian for directional
                                      covariance evaluation. Default is 20 degree
  	
  Output (pure-spatial case: grid_t=None and lagT=None)
  	
  COVv        nls          1D array   Empirical covariance at each spatial lag
  COVn        nls          1D array   Number of pairs per spatial lag
  lagSS       nls          1D array   Spatial lag centres
  lagTT       nls          1D array   Temporal lag (all zeros for pure-spatial)

  Output (space-time case)

  COVv        nls by nlt  2D array    Empirical covariance at LagS by LagT
  COVn        nls by nlt  2D array    Number of pairs for covariance evaluation
  lagSS       nls by nlt  2D array    Meshed lagS        
  lagTT       nls by nlt  2D array    Meshed lagT 
  
  Remark: this function calculates the pdist-based covariance function
  '''

  # Detect pure-spatial call before any parameter modification
  _pure_spatial = (grid_t is None) and (lagT is None)

  #if no GUI, give a no use obj
  if DataObj:    
    title = DataObj.getProgressText()   
    DataObj.setProgressRange(0,len(lagS))
    DataObj.setCurrentProgress(0, title + "\n- Calculating Covariance ...")

  # Assure the following parameters have proper dimension or format, e.g., 
  # 1D np array

  # Auto-coerce 1D grid_v to 2D column (ns, 1) for the pure-spatial case
  grid_v = np.asarray(grid_v)
  if grid_v.ndim == 1:
    grid_v = grid_v[:, np.newaxis]

  if grid_t is None:
    grid_t=np.array([0]).reshape(1,1)
  
  lagS=lagS.ravel()
  lagS_range=lagS_range.ravel()
  if lagT is None:  
    lagT=np.array([0],ndmin=1)
    lagT_range=np.array([0],ndmin=1)
  else:
    lagT=lagT.ravel()
    lagT_range=lagT_range.ravel()  
  
  if len(grid_s.shape)<2:
    grid_s=np.reshape(grid_s,(grid_s.size,1))
  grid_t=np.asarray(grid_t)
  grid_t=np.reshape(grid_t,(grid_t.size,1))
  grid_v=grid_v.reshape((grid_s.shape[0],grid_t.size))
  
  if angtol is None:
    angtol = np.pi*20/180
  
  #find spatial difference between x and y, then remember each index
  #將空間軸與時間軸各轉為每點相差的值，並記住其index (用"_i_"表示)
  if grid_s.shape[0]<8000:
    s_diff_i_left,s_diff_i_right,s_diff_v, angdiff=diffarray(grid_s,aniso=True)
  else:
    s_diff_i_left,s_diff_i_right,s_diff_v, angdiff= \
          diffarray(grid_s,lagS.max()+lagS_range[-1],aniso=True)
  nd=grid_s.shape[1]
  if nd==1:
    s_diff_v=np.abs(s_diff_v).ravel()
  elif nd==2:           
    s_diff_v=np.sqrt(s_diff_v[:,0]**2+s_diff_v[:,1]**2)
  elif nd==3:
    s_diff_v=np.sqrt(s_diff_v[:,0]**2+s_diff_v[:,1]**2+s_diff_v[:,2]**2)
  
  if len(grid_t)<8000:
    t_diff_i_left,t_diff_i_right,t_diff_v,_=diffarray(grid_t)
  else:
    search_rng = lagT.max()+lagT_range[-1]
    if np.issubdtype(grid_t.dtype, np.datetime64):
      search_rng = np.timedelta64(int(round(search_rng)), re.search(r'\[(.*?)\]',grid_t.dtype.str).group(1))
    t_diff_i_left,t_diff_i_right,t_diff_v,_= diffarray(grid_t, search_rng)
  t_diff_v=np.abs(t_diff_v).ravel()
  
  # extract a portion of s_diff_x and t_diff_x here
  #  to reduce the memory use
  if ang is None:
    idxs=np.where(s_diff_v<=lagS.max()+lagS_range[-1])
  else:
    if ang+angtol>np.pi/2:
      idxs1=np.where(np.logical_or(np.logical_and(np.logical_and(angdiff<=np.pi/2,
                angdiff>ang-angtol),s_diff_v<=lagS.max()+lagS_range[-1]),np.isnan(angdiff)))
      idxs2=np.where(np.logical_and(np.logical_and(angdiff>-np.pi/2,
                angdiff<ang+angtol-np.pi),s_diff_v<=lagS.max()+lagS_range[-1]))  
      idxs=np.union1d(idxs1[0],idxs2[0]) 
    elif ang-angtol<-np.pi/2:
      idxs1=np.where(np.logical_or(np.logical_and(np.logical_and(angdiff<=ang+angtol,
                angdiff>-np.pi/2),s_diff_v<=lagS.max()+lagS_range[-1]),np.isnan(angdiff)))      
      idxs2=np.where(np.logical_and(np.logical_and(angdiff<=np.pi/2,
                angdiff>ang-angtol+np.pi),s_diff_v<=lagS.max()+lagS_range[-1])) 
      idxs=np.union1d(idxs1[0],idxs2[0])           
    else:
      idxs=np.where(np.logical_or(np.logical_and(np.logical_and(angdiff<=ang+angtol,
                angdiff>ang-angtol),s_diff_v<=lagS.max()+lagS_range[-1]),np.isnan(angdiff)))
    #idxs=np.where(np.logical_and(s_diff_v<=lagS.max()+lagS_range[-1],np.logical_and(angdiff<=ang+angtol,angdiff>ang-angtol))  
  s_diff_i_left=s_diff_i_left[idxs].astype(int)
  s_diff_i_right=s_diff_i_right[idxs].astype(int)
  s_diff_v=s_diff_v[idxs]
  
  idxt=np.where(t_diff_v<=lagT.max()+lagT_range[-1])
  t_diff_i_left=t_diff_i_left[idxt].astype(int)
  t_diff_i_right=t_diff_i_right[idxt].astype(int)
  t_diff_v=t_diff_v[idxt]  


  #meshgrid s,t    
  lagTT,lagSS=np.meshgrid(lagT,lagS)

  #create result by lagSS's shape
  lagCOVv = np.empty(lagSS.shape)
  lagCOVv[:]=np.nan

  lagCOVm1 = np.empty(lagSS.shape)
  lagCOVm1[:]=np.nan
  lagCOVm2 = np.empty(lagSS.shape)
  lagCOVm2[:]=np.nan  
  #lagCOVm2 = np.zeros(lagSS.shape)
  lagCOVn = np.zeros(lagSS.shape)
   
  #loop lagS first
  for indexS,m in enumerate(lagS):
    
    #find all index that s_diff_v in range lagS_range
    index_diff_s = np.where(
        (s_diff_v >= m-lagS_range[indexS])
        & (s_diff_v <= m+lagS_range[indexS]) )[0]
    s_diff_i_left_select=s_diff_i_left[index_diff_s]
    if s_diff_i_left_select.size == 0: #no match
      continue
    s_diff_i_right_select=s_diff_i_right[index_diff_s]
    
    # For STAR-GUI
    if DataObj:
      if not DataObj.wasProgressCanceled():
        DataObj.setCurrentProgress(indexS + 1) #rest
      else:
        return False

    #loop lagT then ( if lagS matched )
    for indexT,n in enumerate(lagT):
      index_diff_t=np.where( (t_diff_v >= n-lagT_range[indexT]) & (t_diff_v <= n+lagT_range[indexT]) )[0]
      if DataObj:          
        DataObj.setCurrentProgress(indexS + 1) #rest
                
      t_diff_i_left_select=t_diff_i_left[index_diff_t]
      if t_diff_i_left_select.size == 0:
        continue
      t_diff_i_right_select=t_diff_i_right[index_diff_t]
      #get grid_v that matched range
      grid_v_left_select=grid_v[np.ix_(s_diff_i_left_select,t_diff_i_left_select)]
      grid_v_right_select=grid_v[np.ix_(s_diff_i_right_select,t_diff_i_right_select)]
    	
      #maybe has np.nan, exclude it
      grid_v_not_nan_index= ~(np.isnan(grid_v_left_select) | np.isnan(grid_v_right_select))    
      grid_v_left_select=grid_v_left_select[grid_v_not_nan_index]
      grid_v_right_select=grid_v_right_select[grid_v_not_nan_index]
    	
    	
      lagCOVv[indexS][indexT] = (grid_v_left_select * grid_v_right_select).sum()
      lagCOVm1[indexS][indexT] = grid_v_left_select.sum()
      lagCOVm2[indexS][indexT] = grid_v_right_select.sum()
      lagCOVn[indexS][indexT] = grid_v_left_select.size
        
      if DataObj:          
        DataObj.setCurrentProgress(indexS + 1) #rest
    	
    if DataObj:          
      DataObj.setCurrentProgress(indexS + 1) #rest

                
  lagCOVv/=lagCOVn
  lagCOVm1/=lagCOVn
  lagCOVm2/=lagCOVn
  lagCOVv-=lagCOVm1*lagCOVm2
  
  if DataObj:  
    DataObj.setCurrentProgress(text = title)

  # For pure-spatial calls squeeze the single time dimension so outputs are 1D
  if _pure_spatial:
    return lagCOVv[:, 0], lagCOVn[:, 0], lagSS[:, 0], lagTT[:, 0]

  return lagCOVv,lagCOVn,lagSS,lagTT

def stcov_kdtree(
    grid_s, grid_t, grid_v,
    lagS, lagS_range, lagT, lagT_range,
    ang = None, angtol = None, DataObj = None):

    def _assure_dimension(
        grid_s, grid_t, grid_v, lagS, lagS_range, lagT, lagT_range, angtol ): 
        if grid_t is None:
            grid_t=np.array([0]).reshape(1,1)

        lagS = lagS.ravel()
        lagS_range = lagS_range.ravel()
        if lagT is None:  
            lagT=np.array([0],ndmin=1)
            lagT_range=np.array([0],ndmin=1)
        else:
            lagT=lagT.ravel()
            lagT_range=lagT_range.ravel()  

        if len(grid_s.shape)<2:
            grid_s=np.reshape(grid_s,(grid_s.size,1))
        grid_t=np.asarray(grid_t)
        grid_t=np.reshape(grid_t,(grid_t.size,1))
        grid_v=grid_v.reshape((grid_s.shape[0],grid_t.size))
      
        if angtol is None:
            angtol = np.pi*20/180.

        return grid_s, grid_t, grid_v, lagS, lagS_range, lagT, lagT_range, angtol


    # D3 fix: ang is not implemented in stcov_kdtree; raise instead of silently
    # ignoring and returning omnidirectional covariance.
    if ang is not None:
        raise NotImplementedError(
            "stcov_kdtree does not support directional filtering (ang). "
            "Use stcov() instead for directional covariance estimation."
        )

    grid_s, grid_t, grid_v, lagS, lagS_range, lagT, lagT_range, angtol = \
        _assure_dimension( grid_s, grid_t, grid_v, lagS, lagS_range, lagT, lagT_range, angtol )

    # meshgrid s,t    
    lagTT, lagSS = np.meshgrid( lagT, lagS )

    # create result by lagSS's shape
    index_s_count, index_t_count = lagSS.shape
    lagCOVv = np.zeros((index_s_count, index_t_count))
    lagCOVm1 = np.zeros((index_s_count, index_t_count))
    lagCOVm2 = np.zeros((index_s_count, index_t_count))
    lagCOVn = np.zeros((index_s_count, index_t_count))

    if DataObj:
        title = DataObj.getProgressText()
        total_step_count = index_s_count * index_t_count
        current_step_count = 0
        DataObj.setProgressRange(0, total_step_count)
        sub_title = title + "- Calculating Covariance ...\n"
        DataObj.setCurrentProgress(
            current_step_count, sub_title)

    s_tree = KDTree(grid_s, leafsize=40)
    t_tree = KDTree(grid_t, leafsize=40)

    for indexS, (s_dis, s_tor) in enumerate(zip(lagS, lagS_range)):
        dis_upper = s_dis + s_tor
        set_upper = s_tree.query_pairs(dis_upper)
        dis_lower = s_dis - s_tor - 10**-15
        if dis_lower < 0: # include lag = 0
            dis_lower = 0
            set_lower = set()
            set_upper.update({(i,i) for i in range(grid_s.shape[0])})
        else:
            set_lower = s_tree.query_pairs(dis_lower)
        s_index = np.array(list(set_upper - set_lower))
        if s_index.size == 0: #no match
            continue

        for indexT, (t_dis, t_tor) in enumerate(zip(lagT, lagT_range)):
            # print indexS, indexT
            dis_upper = t_dis + t_tor
            set_upper = t_tree.query_pairs(dis_upper)
            dis_lower = t_dis - t_tor - 10**-15
            if dis_lower < 0: # include lag = 0
                dis_lower = 0
                set_lower = set()
                set_upper.update({(i,i) for i in range(grid_t.shape[0])})
            else:
                set_lower = t_tree.query_pairs(dis_lower)
            t_index = np.array(list(set_upper - set_lower))

            if t_index.size == 0: #no match
                continue

            #get grid_v that matched range
            grid_v_left = grid_v[np.ix_(s_index[:,0], t_index[:,0])]
            grid_v_right = grid_v[np.ix_(s_index[:,1], t_index[:,1])]

            #maybe has np.nan, exclude it
            grid_v_not_nan_index = ~(np.isnan(grid_v_left) | np.isnan(grid_v_right))    
            grid_v_left = grid_v_left[grid_v_not_nan_index]
            grid_v_right = grid_v_right[grid_v_not_nan_index]

            lagCOVv[indexS][indexT] += (grid_v_left * grid_v_right).sum()
            lagCOVm1[indexS][indexT] += grid_v_left.sum()
            lagCOVm2[indexS][indexT] += grid_v_right.sum()
            lagCOVn[indexS][indexT] += grid_v_left.size

            # For STAR-GUI
            if DataObj:
                if not DataObj.wasProgressCanceled():
                    current_step_count += 1
                    DataObj.setCurrentProgress(
                        current_step_count ,
                        sub_title + '({c}/{n})'.format(
                            c=current_step_count,
                            n=total_step_count
                            )
                        ) #rest
                    DataObj.drawGUI()
                else:
                    return False

    lagCOVv/=lagCOVn
    lagCOVm1/=lagCOVn
    lagCOVm2/=lagCOVn
    lagCOVv-=lagCOVm1*lagCOVm2

    return lagCOVv,lagCOVn,lagSS,lagTT

def cov_avg_nd( coord, value, lagC, lagC_range ):
    '''
    get covariance of each range(block) and their pair number
     
    input
    coord: a list of delta matrixs
    value: row x 1, 2d np array
    lagC: a list of lag_limit(1d np array)
    lagC_range: a list of range_limit(1d np array)
    
    return 
    lagCOVv: dim of each lagC's len
    lagCOVn: dim of each lagC's len
    '''
    
    # delta_coord = [] #a list store each delta coordinate
    # for i in range( coord_dim ):
    #     coord_i = coord[ :, i:i+1 ]
    #     delta_coord.append( coord_i - coord_i.T )
        
    delta_value_multi = value * value.T
    delta_value_add_left = value * np.ones( value.T.shape )
    delta_value_add_right = value.T * np.ones( value.shape )
    
    matched_bool_matrix = [] # a coord_dim list store every matched bool matrix ( coord_dim x lag_num )
    for coord_i, lags, ranges in zip( coord, lagC, lagC_range ):
        matched_bool_matrix_i = [] #store 1 of coord_dim matched matrix
        for lag_i, range_i in zip( lags, ranges ):
            matched_bool_matrix_i.append( ( coord_i > lag_i - range_i ) & ( coord_i <= lag_i + range_i ) )
        matched_bool_matrix.append( matched_bool_matrix_i )
    
    #check a range has data included
    # has_data_bool_matrix = []
    # for matched_bool_matrix_i in matched_bool_matrix:
    #     has_data_bool_matrix_i = [] 
    #     for each_matchhed_bool_matrix in matched_bool_matrix_i:
    #         has_data_bool_matrix_i.append( each_matchhed_bool_matrix.any() )
    #     has_data_bool_matrix.append( has_data_bool_matrix_i )
        
    #get index set
    #use itertools to get the index of each possible case(each block)
    lag_dim_len = [ len(i) for i in lagC ]
    index_set = list( itertools.product( *list(map( range, lag_dim_len )) ) )
    
    #get each block value ( covariance )
    ####set result lagCOV
    lagCOVv = np.empty( lag_dim_len ) #index for each dim
    lagCOVm1 = np.empty( lag_dim_len )
    lagCOVm2 = np.empty( lag_dim_len )
    lagCOVv[:] = np.nan
    lagCOVm1[:] = np.nan
    lagCOVm2[:] = np.nan
    
    lagCOVn = np.zeros( lag_dim_len )
    
    
    for idx in index_set:
        #set the result bool matrix
        res_idx_matrix = np.ones( matched_bool_matrix[0][0].shape, dtype = bool )
        #logical-and for each dim 
        for idx_i, m_b_mat_i in zip( idx, matched_bool_matrix ):
            res_idx_matrix = np.logical_and( res_idx_matrix, m_b_mat_i[ idx_i ] )
        if res_idx_matrix.any() and np.sum( res_idx_matrix ) > 1: #has data and data num at least 2 or cannot calculate covariance
            select_dv = delta_value_multi[ res_idx_matrix ]
            select_dm1 = delta_value_add_left[ res_idx_matrix ]
            select_dm2 = delta_value_add_right[ res_idx_matrix ]
            
            lagCOVv[idx] = select_dv.sum()
            lagCOVm1[idx] = select_dm1.sum()
            lagCOVm2[idx] = select_dm2.sum()
            lagCOVn[idx] =  select_dv.size
        else:
            print ("can not calculate")
    # for j in xrange( lag_num ):    
    #     for i in xrange( coord_dim ): #row of matched_bool_matrix
    #         has_data_bool_matrix[ i ][ j ] =  matched_bool_matrix[ i ][ j ].any()
           
        
    lagCOVv /= lagCOVn
    lagCOVm1 /= lagCOVn
    lagCOVm2 /= lagCOVn
    lagCOVv -= lagCOVm1*lagCOVm2
    
    return lagCOVv, lagCOVn
        
def stcov_split(grid_s, grid_t, grid_v,
    lagS, lagS_range,
    lagT = None, lagT_range = None, 
    ang = None, angtol = None, DataObj = None):
    '''
        Estimates the space/time cross covariance between variable Zst 
        with measurements at fixed monitoring stations. The monitoring 
        stations for Zst are located at coordinates cMS, and 
        measuring events are at times tME

        split means calculate covariance a piece at a time to prevent memory error

        SYNTAX:
        COVv,COVn,lagSS,lagTT =
        stcov_split(
            grid_s, grid_t, grid_v,
            lagS, lagS_range, lagT, lagT_range,
            ang=None, angtol = None, DataObj = None 
            )

        Input:
          
        grid_s      ns by 2     2D array    cMS
        grid_t      nt by 1     2D array    tME. For spatial case, one should specify 
                                            a specified time or None for the spatial 
                                            observations
        grid_v      ns by nt    2D array    Data Zst
        lagS        nls by 1    1D array    vector with the r lags    
        lagS_range  nls by 1    1D array    vector with the tolerance for the r lags
        lagT        nlt by 1    1D array    vector with the t lags. default is None
                                            for pure spatial case
        lagT_range  nlt by 1    1D array    vector with the tolerance for the t lags
        ang         scalar                  specified direction in radian for directional covariance. 
                                            Default is None for omnidirectional. 
                                            Direction is in radian and zero denotes 
                                            the horizontal axis
        angtol      scalar                  angle tolerance in radian for directional
                                            covariance evaluation. Default is 20 degree
          
        Output:
          
        COVv        nls by nlt  2D array    Empirical covariance at LagS by LagT
        COVn        nls by nlt  2D array    Number of pairs for covariance evaluation
        lagSS       nls by nlt  2D array    Meshed lagS        
        lagTT       nls by nlt  2D array    Meshed lagT 

        Remark: this function calculates the pdist-based covariance function
    '''

    def _assure_dimension(
        grid_s, grid_t, grid_v, lagS, lagS_range, lagT, lagT_range, angtol ): 
        if grid_t is None:
            grid_t=np.array([0]).reshape(1,1)
      
        lagS = lagS.ravel()
        lagS_range = lagS_range.ravel()
        if lagT is None:  
            lagT = np.array([0],ndmin=1)
            lagT_range = np.array([0],ndmin=1)
        else:
            lagT = lagT.ravel()
            lagT_range = lagT_range.ravel()  
      
        if len(grid_s.shape)<2:
            grid_s = np.reshape(grid_s, (grid_s.size,1))
        grid_t = np.asarray(grid_t)
        grid_t = np.reshape(grid_t, (grid_t.size,1))
        grid_v = grid_v.reshape((grid_s.shape[0], grid_t.size))
      
        if angtol is None:
            angtol = np.pi*20/180.

        return grid_s, grid_t, grid_v, lagS, lagS_range, lagT, lagT_range, angtol

    def _diff_space_time(
        grid_s, grid_t, lagS, lagS_range, lagT, lagT_range, ang, angtol ):
        # find spatial difference between x and y, then remember each index
        # 將空間軸與時間軸各轉為每點相差的值，並記住其index (用"_i_"表示)
        # 只取出計算會用到的部份

        has_gui = True if DataObj else False
        has_aniso = False if ang is None else True

        if not has_gui:
            s_diff_i_left, s_diff_i_right, s_diff_v, angdiff= \
                pdist_numba(grid_s, lagS.max() + lagS_range[-1],
                    True, has_aniso)
            t_diff_i_left, t_diff_i_right, t_diff_v,_ = \
            pdist_numba(grid_t, lagT.max() + lagT_range[-1],
                        True, has_aniso)
        else:
            res_s =\
                diffarray_split(grid_s, lagS.max() + lagS_range[-1],
                    True, has_aniso,
                    gui_args=(DataObj,'Covariance\n- Spatial Diffarray...')
                    )
            if res_s:
                s_diff_i_left, s_diff_i_right, s_diff_v, angdiff = res_s
            else:
                return False

            res_t =\
                diffarray_split(grid_t, lagT.max() + lagT_range[-1],
                    True, has_aniso,
                    gui_args=(DataObj,'Covariance\n- Temporal Diffarray...')
                    )
            if res_t:
                t_diff_i_left, t_diff_i_right, t_diff_v, _ = res_t
            else:
                return False

        #各維度平方相加開根號
        s_diff_v **= 2
        s_diff_v = s_diff_v.sum( axis=1 ) # 順便打成一維, keepdims = True )
        s_diff_v **= 0.5
        np.abs( t_diff_v, out = t_diff_v ) # inplace change, faster
        t_diff_v = t_diff_v.ravel()


        # has_gui = True if DataObj else False
        # has_aniso = False if ang is None else True
 
        # if not has_gui:
        #     s_diff_i_left, s_diff_i_right, s_diff_v, angdiff= \
        #         pdist_numba(grid_s, lagS.max() + lagS_range[-1],
        #             True, has_aniso)
        # else:
        #     s_diff_i_left, s_diff_i_right, s_diff_v, angdiff= \
        #         pdist_numba(grid_s, lagS.max() + lagS_range[-1],
        #             True, has_aniso, DataObj)
        #     # for large data with gui,
        #     # here should add gui response for better user experience
        #     s_count = len(grid_s)
        #     s_diff_i_left = np.empty(0, dtype=np.int64)
        #     s_diff_i_right = np.empty(0, dtype=np.int64)
        #     s_diff_v = np.empty((0,grid_s.shape[1]))
        #     angdiff = np.empty(0)
        #     step_length = 4000
        #     step_count = (s_count / step_length) + 1

        #     if DataObj:
        #         title = DataObj.getProgressText()
        #         DataObj.setProgressRange(0, (step_count+1) * step_count / 2)
        #         sub_title = title + "\n- Calculating Covariance ...\nDiffarray..."
        #         DataObj.setCurrentProgress(0, sub_title)
        #         total_step_count = (step_count+1) * step_count / 2
        #         current_step_count = 0

        #     for step_1 in range(0, s_count, step_length):
        #         for step_2 in range(step_1, s_count, step_length):
        #             if step_1 == step_2:
        #                 grid_s_i = grid_s[step_1:step_1+step_length, :]
        #                 d_i_h, d_i_t, d_v, d_a =\
        #                     pdist_numba(
        #                         grid_s_i,
        #                         lagS.max() + lagS_range[-1],
        #                         True, has_aniso)
        #             else:
        #                 grid_s_i = grid_s[step_1:step_1 + step_length, :]
        #                 grid_s_j = grid_s[step_2:step_2 + step_length, :]
        #                 d_i_h, d_i_t, d_v, d_a =\
        #                     cdist_numba(
        #                         grid_s_i, grid_s_j,
        #                         lagS.max() + lagS_range[-1],
        #                         has_aniso)
            
        #             s_diff_i_left = np.append(s_diff_i_left, d_i_h + step_1)
        #             s_diff_i_right = np.append(s_diff_i_right, d_i_t + step_2)
        #             s_diff_v = np.append(s_diff_v, d_v, axis=0)
        #             angdiff = np.append(angdiff, d_a)

        #             if DataObj:
        #                 if not DataObj.wasProgressCanceled():
        #                     current_step_count += 1
        #                     DataObj.setCurrentProgress(
        #                         current_step_count ,
        #                         sub_title + '({c}/{n})'.format(
        #                             c=current_step_count,
        #                             n=total_step_count
        #                             )
        #                         ) #rest
        #                     DataObj.drawGUI()
        #                 else:
        #                     return False

        # #各維度平方相加開根號
        # s_diff_v **= 2
        # s_diff_v = s_diff_v.sum( axis=1 ) # 順便打成一維, keepdims = True )
        # s_diff_v **= 0.5
      
        # t_diff_i_left, t_diff_i_right, t_diff_v,_ = \
        #     pdist_numba(grid_t, lagT.max() + lagT_range[-1],
        #                 True, has_aniso)
        # np.abs( t_diff_v, out = t_diff_v ) # inplace change, faster
        # t_diff_v = t_diff_v.ravel()
      
        # extract a portion of s_diff_x and t_diff_x here
        #  to reduce the memory use
        # 距離跟時間維度已在diffarray裡考慮  這裡只考慮角度
        if ang:
            if ang+angtol>np.pi/2:
                idxs1=np.where(np.logical_or(np.logical_and(np.logical_and(angdiff<=np.pi/2,
                    angdiff>ang-angtol),s_diff_v<=lagS.max()+lagS_range[-1]),np.isnan(angdiff)))
                idxs2=np.where(np.logical_and(np.logical_and(angdiff>-np.pi/2,
                    angdiff<ang+angtol-np.pi),s_diff_v<=lagS.max()+lagS_range[-1]))  
                idxs=np.union1d(idxs1[0],idxs2[0]) 
            elif ang-angtol<-np.pi/2:
                idxs1=np.where(np.logical_or(np.logical_and(np.logical_and(angdiff<=ang+angtol,
                    angdiff>-np.pi/2),s_diff_v<=lagS.max()+lagS_range[-1]),np.isnan(angdiff)))      
                idxs2=np.where(np.logical_and(np.logical_and(angdiff<=np.pi/2,
                    angdiff>ang-angtol+np.pi),s_diff_v<=lagS.max()+lagS_range[-1])) 
                idxs=np.union1d(idxs1[0],idxs2[0])           
            else:
                idxs=np.where(np.logical_or(np.logical_and(np.logical_and(angdiff<=ang+angtol,
                    angdiff>ang-angtol),s_diff_v<=lagS.max()+lagS_range[-1]),np.isnan(angdiff)))

            s_diff_i_left = s_diff_i_left[idxs]
            s_diff_i_right = s_diff_i_right[idxs]
            s_diff_v = s_diff_v[idxs]

            idxt = np.where(t_diff_v<=lagT.max()+lagT_range[-1])
            t_diff_i_left = t_diff_i_left[idxt]
            t_diff_i_right = t_diff_i_right[idxt]
            t_diff_v = t_diff_v[idxt]

        return (s_diff_i_left, s_diff_i_right, s_diff_v,
            t_diff_i_left, t_diff_i_right, t_diff_v)

    def _has_overlay_lag(lag_, range_):
        bd_min = lag_ - range_
        bd_max = lag_ + range_
        for i in range(bd_min.size-1):
            bd_i_min = bd_min[i]
            bd_i_max = bd_max[i]
            bd_j_min = bd_min[i+1:]
            bd_j_max = bd_max[i+1:]
            if ((bd_i_min >= bd_j_max) & (bd_i_max < bd_j_min)).all():
                pass
            else:
                return True
        return False
        

    if DataObj:
        title = DataObj.getProgressText()
    # Assure the following parameters have proper dimension or format, e.g., 
    # 1D np array
    grid_s, grid_t, grid_v, lagS, lagS_range, lagT, lagT_range, angtol = \
        _assure_dimension( grid_s, grid_t, grid_v, lagS, lagS_range, lagT, lagT_range, angtol )

    has_overlay_s = True if _has_overlay_lag(lagS, lagS_range) else False
    has_overlay_t = True if _has_overlay_lag(lagT, lagT_range) else False

    #find spatial difference between x and y, then remember each index
    #將空間軸與時間軸各轉為每點相差的值，並記住其index (用"_i_"表示)
    res =\
        _diff_space_time( grid_s, grid_t, lagS, lagS_range, lagT, lagT_range, ang, angtol )
    if res:
        s_diff_i_left, s_diff_i_right, s_diff_v, t_diff_i_left, t_diff_i_right, t_diff_v = res
    else:
        return False

    # meshgrid s,t    
    lagTT, lagSS = np.meshgrid(lagT, lagS)

    # create result by lagSS's shape
    lagCOVv = np.zeros(lagSS.shape)
    lagCOVm1 = np.zeros(lagSS.shape)
    lagCOVm2 = np.zeros(lagSS.shape)
    lagCOVn = np.zeros(lagSS.shape)

    if DataObj:
        total_step = lagTT.size
        DataObj.setProgressRange(0, total_step)
        sub_title = DataObj.getProgressText() +\
            'Covariance\n- Averaging covariance pairs...'
    #loop lagS first
    for indexS, m in enumerate(lagS):
        #find all index that s_diff_v in range lagS_range
        index_diff_s = np.where(
            (s_diff_v >= m-lagS_range[indexS])
            & (s_diff_v < m+lagS_range[indexS])
            )[0]
        if index_diff_s.size == 0: #no match
            continue
        #loop lagT then ( if lagS matched )
        t_diff_i_left_overlay = np.copy(t_diff_i_left)
        t_diff_i_right_overlay = np.copy(t_diff_i_right)
        t_diff_v_overlay = np.copy(t_diff_v)
        for indexT, n in enumerate(lagT):
            index_diff_t = np.where(
                (t_diff_v_overlay >= n-lagT_range[indexT])
                & (t_diff_v_overlay < n+lagT_range[indexT])
                )[0]
            if index_diff_t.size == 0: #no match
                continue
            #calculate a piece at one time
            index_diff_s_len = index_diff_s.shape[0]
            index_diff_t_len = index_diff_t.shape[0]

            S_STEP = 10000
            T_STEP = 10000

            for s_idx in range(0, index_diff_s_len, S_STEP):
                idx_dif_s_slct = index_diff_s[ s_idx:s_idx + S_STEP ]
                s_diff_i_left_select = s_diff_i_left[ idx_dif_s_slct ]
                s_diff_i_right_select = s_diff_i_right[ idx_dif_s_slct ]
                for t_idx in range(0, index_diff_t_len, T_STEP):
                    idx_dif_t_slct = index_diff_t[ t_idx:t_idx + T_STEP ]
                    t_diff_i_left_select = t_diff_i_left_overlay[ idx_dif_t_slct ]
                    t_diff_i_right_select = t_diff_i_right_overlay[ idx_dif_t_slct ]
                    #get grid_v that matched range
                    grid_v_left_select =\
                        grid_v[np.ix_(s_diff_i_left_select,t_diff_i_left_select)]
                    grid_v_right_select =\
                        grid_v[np.ix_(s_diff_i_right_select,t_diff_i_right_select)]
                    #maybe has np.nan, exclude it
                    grid_v_not_nan_index =\
                        ~(np.isnan(grid_v_left_select) | np.isnan(grid_v_right_select))    
                    grid_v_left_select =\
                        grid_v_left_select[grid_v_not_nan_index]
                    grid_v_right_select =\
                        grid_v_right_select[grid_v_not_nan_index]
                    lagCOVv[indexS][indexT] +=\
                        (grid_v_left_select * grid_v_right_select).sum()
                    lagCOVm1[indexS][indexT] += grid_v_left_select.sum()
                    lagCOVm2[indexS][indexT] += grid_v_right_select.sum()
                    lagCOVn[indexS][indexT] += grid_v_left_select.size

                    # For STAR-GUI
                    if DataObj:
                        if not DataObj.wasProgressCanceled():
                            current_count = indexS * len(lagT) + (indexT + 1)
                            DataObj.setCurrentProgress(
                                current_count,
                                sub_title + '({c}/{n})'.format(
                                    c=current_count,
                                    n=total_step
                                    )
                                ) #rest
                            DataObj.drawGUI()
                        else:
                            return False

            #if not overlay, pop used row (temporal)
            if not has_overlay_t:
                t_diff_i_left_overlay = t_diff_i_left_overlay[~index_diff_t]
                t_diff_i_right_overlay = t_diff_i_right_overlay[~index_diff_t]
                t_diff_v_overlay = t_diff_v_overlay[~index_diff_t]
        #if not overlay, pop used row (spatial)
        if not has_overlay_s:
            s_diff_i_left = s_diff_i_left[~index_diff_s]
            s_diff_i_right = s_diff_i_right[~index_diff_s]
            s_diff_v = s_diff_v[~index_diff_s]
    lagCOVv/=lagCOVn
    lagCOVm1/=lagCOVn
    lagCOVm2/=lagCOVn
    lagCOVv-=lagCOVm1*lagCOVm2
  
    if DataObj: 
        DataObj.setCurrentProgress(text = title)
    return lagCOVv, lagCOVn, lagSS, lagTT

def _stcov(
    grid_s1, grid_t1, grid_v1,
    grid_s2, grid_t2, grid_v2):

    '''
    Calculate spatial temporal covariance
    grid_s nstation by ndim   np 2d array
    grid_t 1 by ntime         np 2d array
    grid_v ns by nt           denote their observations

    return np 2d array
        [[ds, dt, v1, v2],
         [ds, dt, v1, v2],
         [...]]
    '''

    ds = cdist_scipy(grid_s1, grid_s2)
    dt = cdist_scipy(grid_t1.T, grid_t2.T)
    ds_out = np.kron(ds, np.ones(dt.shape))
    dt_out = np.tile(dt, ds.shape)
    v1_out = np.tile(
        grid_v1.T.reshape((-1, 1)),
        (1, grid_t2.shape[1]*grid_s2.shape[0])
        )
    v2_out = np.tile(
        grid_v2.reshape((1, -1)),
        (grid_t1.shape[1]*grid_s1.shape[0], 1)
        )

    stvv = np.dstack((ds_out, dt_out, v1_out, v2_out)).reshape((-1, 4))
    nan_idx = (np.isnan(stvv[:,2]) | np.isnan(stvv[:,3]))
    return stvv[~nan_idx]

def stcov_split2(
    grid_s, grid_t, grid_v,
    lagS, lagS_range,
    lagT = None, lagT_range = None, 
    ang = None, angtol = None, DataObj = None):

    '''
    try to solve memory error problem (not working)
    * grid_t is nt by 1 2d array
    '''

    grid_t = grid_t.T

    lagTT, lagSS = np.meshgrid(lagT, lagS)
    lagCOVv = np.zeros(lagSS.shape)
    lagCOVm1 = np.zeros(lagSS.shape)
    lagCOVm2 = np.zeros(lagSS.shape) 
    lagCOVn = np.zeros(lagSS.shape)

    s_count = grid_s.shape[0]
    t_count = grid_t.size
    s_step_length = 60
    t_step_length = 60
    s_step_count = int(np.ceil(float(s_count) / s_step_length))
    t_step_count = int(np.ceil(float(t_count) / t_step_length))
    total_step_count = s_step_count * t_step_count
    total_step_count = (total_step_count + 1) * total_step_count // 2
    current_step_count = 0
    if DataObj:
        title = DataObj.getProgressText()
        DataObj.setProgressRange(0, total_step_count)
        sub_title = title + "\n- Calculating Covariance..."
        DataObj.setCurrentProgress(0, sub_title)

    st_step = list(
        itertools.product(
            range(0, s_count, s_step_length),
            range(0, t_count, t_step_length)
            )
        )
    for idx, (s_step, t_step) in enumerate(st_step):
        piece_grid_s = grid_s[s_step:s_step+s_step_length, :]
        piece_grid_t = grid_t[:, t_step:t_step+t_step_length]
        piece_grid_v = grid_v[
            s_step:s_step+s_step_length,
            t_step:t_step+t_step_length
            ]
        for s_step2, t_step2 in st_step[idx:]:
            current_step_count += 1
            piece_grid_s2 = grid_s[s_step2:s_step2+s_step_length, :]
            piece_grid_t2 = grid_t[:, t_step2:t_step2+t_step_length]
            piece_grid_v2 = grid_v[
                s_step2:s_step2+s_step_length,
                t_step2:t_step2+t_step_length
                ]
            stvv = _stcov(
                piece_grid_s, piece_grid_t, piece_grid_v,
                piece_grid_s2, piece_grid_t2, piece_grid_v2)
            if stvv.size == 0:
                continue

            for lag_s_idx, (lag_s, lag_s_range) in\
                enumerate(zip(lagS, lagS_range)):

                #ds in range
                select_s_idx = np.where( 
                    (stvv[:, 0] >= lag_s-lag_s_range)
                    & (stvv[:, 0] <= lag_s+lag_s_range)
                    )[0]
                if select_s_idx.size == 0: #no data valid
                    continue
                stvv_select = stvv[select_s_idx]
                for lag_t_idx, (lag_t, lag_t_range) in\
                    enumerate(zip(lagT, lagT_range)):

                    #dt in range (stvv_select)
                    select_t_idx = np.where( 
                        (stvv_select[:, 1] >= lag_t-lag_t_range)
                        & (stvv_select[:, 1] <= lag_t+lag_t_range)
                        )[0]
                    if select_t_idx.size == 0: #no match
                        continue
                    stvv_select_final = stvv_select[select_t_idx]
                    v1_select_final = stvv_select_final[:, 2]
                    v2_select_final = stvv_select_final[:, 3]
                    lagCOVv[lag_s_idx][lag_t_idx] +=\
                        (v1_select_final * v2_select_final).sum()
                    lagCOVm1[lag_s_idx][lag_t_idx] +=\
                        v1_select_final.sum()
                    lagCOVm2[lag_s_idx][lag_t_idx] +=\
                        v2_select_final.sum()
                    lagCOVn[lag_s_idx][lag_t_idx] +=\
                        v1_select_final.size
    lagCOVv /= lagCOVn
    lagCOVm1 /= lagCOVn
    lagCOVm2 /= lagCOVn
    lagCOVv -= lagCOVm1*lagCOVm2

    return lagCOVv, lagCOVn, lagSS, lagTT


if __name__ == "__main__":
  import pandas as pd
  import time
  from ....general.coord2dist import coord2dist
  fname='./examples/Data/dataSEKSGUI_Hard'
  data=pd.read_csv(fname,sep='  ',header=None).values
  Z,cMS,tME,nanloc=valstv2stg(data[:,0:3],data[:,3])
  tME=tME.reshape(tME.size,1)
  stime=time.time()
  s_diff_i_left,s_diff_i_right,s_diff_v,_=diffarray(cMS)
  print ('s-time is ' + str(time.time()-stime))
  stime=time.time()
  t_diff_i_left,t_diff_i_right,t_diff_v,_=diffarray(tME)
  print ('t-time is ' + str(time.time()-stime))
  
  dist=coord2dist(cMS,cMS)  
  lagS=np.linspace(0,dist.max(),10)
  lagStol=np.ones(lagS.size)*(lagS[1]-lagS[0])
  lagT=np.arange(24)
  lagTtol=np.ones(lagT.size)*0.5
  
  stime=time.time()
  COV,COVn,s,t=stcov_bme(cMS,tME,Z,lagS,lagStol,lagT,lagTtol)
  print ('stcov_bme time is ' + str(time.time()-stime))
  
  stime=time.time()
  COV2,COVn2,s2,t2=stcov(cMS,tME,Z,lagS,lagStol,lagT,lagTtol)
  print ('stcov time is ' + str(time.time()-stime))
  
  stime=time.time()
  COV5,COVn5,s5,t5=stcov_split(cMS,tME,Z,lagS,lagStol,lagT,lagTtol)
  print ('stcov_split time is ' + str(time.time()-stime))
  
  ang=0.
  stime=time.time()
  COV3,COVn3,s3,t3=stcov(cMS,tME,Z,lagS,lagStol,lagT,lagTtol, ang)
  print ('stcov with aniso time is ' + str(time.time()-stime))
  
  stime=time.time()
  COV4,COVn4,s4,t4=stcov(cMS,tME,Z,lagS,lagStol)
  print ('stcov with only spatial time is ' + str(time.time()-stime))
  
  print ('Is variable "COV" of stcov and stcov_split the same?', np.array_equal( COV2, COV5 ))
  print ('Is variable "COVn" of stcov and stcov_split the same?', np.array_equal( COVn2, COVn5 ))


# ---------------------------------------------------------------------------
# Dask-parallel covariance functions preserved from stamps_v3
# ---------------------------------------------------------------------------

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def stcov_dask(grid_s, grid_t, grid_v,
    s_lag, s_lag_range,
    t_lag = None, t_lag_range = None, 
    ang = None, angtol = None, DataObj = None):
    '''Compute space/time covariance using a chunked (non-Dask) approach.

    Preserved from stamps_v3 for backward compatibility.
    '''

    # meshgrid s,t    
    lagTT, lagSS = np.meshgrid(t_lag, s_lag)

    # create result by lagSS's shape
    lagCOVv = np.zeros(lagSS.shape)
    lagCOVm1 = np.zeros(lagSS.shape)
    lagCOVm2 = np.zeros(lagSS.shape)
    lagCOVn = np.zeros(lagSS.shape)

    s_chunksize = 1000
    t_chunksize = 30
    chunk_grid_s = np.split(grid_s, range(s_chunksize, grid_s.shape[0], s_chunksize))
    chunk_grid_t = np.split(grid_t.reshape(-1,1), range(t_chunksize, grid_t.shape[0], t_chunksize))

    for idx_s_i in range(len(chunk_grid_s)):
        for idx_s_j in range(len(chunk_grid_s)):

            s_i = chunk_grid_s[idx_s_i]
            s_j = chunk_grid_s[idx_s_j]

            ds = cdist_scipy(s_i, s_j)

            for idx_s_lag, m in enumerate(s_lag):
                s_head, s_tail = np.where(
                    (ds >= m - s_lag_range[idx_s_lag])
                    & (ds <= m + s_lag_range[idx_s_lag])
                    )
                if s_head.size == 0: #no match
                    continue

                grid_v_head_s = grid_v[idx_s_i * s_chunksize + s_head]
                grid_v_tail_s = grid_v[idx_s_j * s_chunksize + s_tail]

                for idx_t_i in range(len(chunk_grid_t)):
                    for idx_t_j in range(len(chunk_grid_t)):
                    
                        t_i = chunk_grid_t[idx_t_i]
                        t_j = chunk_grid_t[idx_t_j]

                        dt = cdist_scipy(t_i, t_j)

                        for idx_t_lag, n in enumerate(t_lag):
                            t_head, t_tail = np.where(
                                (dt >= n - t_lag_range[idx_t_lag])
                                & (dt <= n + t_lag_range[idx_t_lag])
                                )
                            if t_head.size == 0: #no match
                                continue

                            grid_v_head = grid_v_head_s[:, idx_t_i * t_chunksize + t_head]
                            grid_v_tail = grid_v_tail_s[:, idx_t_j * t_chunksize + t_tail]

                            grid_v_ht = grid_v_head * grid_v_tail
                            bool_notnan = ~np.isnan(grid_v_ht)
                            grid_v_ht = grid_v_ht[bool_notnan]
                            lagCOVv[idx_s_lag][idx_t_lag] += np.sum(grid_v_ht)
                            lagCOVm1[idx_s_lag][idx_t_lag] += np.sum(grid_v_head[bool_notnan])
                            lagCOVm2[idx_s_lag][idx_t_lag] += np.sum(grid_v_tail[bool_notnan])
                            lagCOVn[idx_s_lag][idx_t_lag] += grid_v_ht.size

    lagCOVv /= lagCOVn
    lagCOVm1 /= lagCOVn
    lagCOVm2 /= lagCOVn
    lagCOVv -= lagCOVm1*lagCOVm2
    
    return lagCOVv, lagCOVn, lagSS, lagTT


def stcov_dask_half(grid_s, grid_t, grid_v,
    s_lag, s_lag_range,
    t_lag = None, t_lag_range = None, 
    ang = None, angtol = None, DataObj = None, dask_client=None):
    '''Compute space/time covariance using Dask distributed computation.

    Requires a running Dask client. Preserved from stamps_v3.
    '''
    def print_current_status():
        print(
            'dask_tasks:', len(dask_tasks), ','
            's_chunk', current_s_chunk, '/', total_s_chunk, ', ',
            's_lag', idx_s_lag + 1, '/', len(s_lag), ', ',
            't_chunk', idx_t_i + 1, '/', len(chunk_grid_t)
            ) 

    dask_cov_index = dask.delayed(_cov_index)

    lagTT, lagSS = np.meshgrid(t_lag, s_lag)
    lagCOVv = np.zeros(lagSS.shape)
    lagCOVm1 = np.zeros(lagSS.shape)
    lagCOVm2 = np.zeros(lagSS.shape)
    lagCOVn = np.zeros(lagSS.shape)

    s_chunksize = 33
    t_chunksize = 200
    print('s_chunksize', s_chunksize, 't_chunksize', t_chunksize)
    chunk_grid_s = np.split(grid_s, range(s_chunksize, grid_s.shape[0], s_chunksize))
    chunk_grid_t = np.split(grid_t.reshape(-1,1), range(t_chunksize, grid_t.shape[0], t_chunksize))

    total_s_chunk = len(chunk_grid_s)
    total_s_chunk = total_s_chunk * (total_s_chunk+1) / 2
    current_s_chunk = 0
    dask_tasks = []

    for idx_s_i in range(len(chunk_grid_s)):
        for idx_s_j in range(idx_s_i, len(chunk_grid_s)):
            current_s_chunk += 1
            s_i = chunk_grid_s[idx_s_i]
            s_j = chunk_grid_s[idx_s_j]
            ds = cdist_scipy(s_i, s_j)

            for idx_s_lag, m in enumerate(s_lag):
                s_head, s_tail = np.where(
                    (ds >= m - s_lag_range[idx_s_lag])
                    & (ds <= m + s_lag_range[idx_s_lag])
                    )
                if s_head.size == 0:
                    continue

                grid_v_d = dask.delayed(grid_v)
                s_head_d = dask.delayed(s_head)
                s_tail_d = dask.delayed(s_tail)

                dask_tasks.append(
                    dask_cov_index(
                        grid_v_d, idx_s_i, idx_s_j, s_chunksize, s_head_d, s_tail_d,
                        chunk_grid_t, lagSS.shape, idx_s_lag, t_chunksize, t_lag, t_lag_range
                        )
                    )

    dask_worker_count = len(dask_client.scheduler_info()['workers'])

    for cindx, task in enumerate(chunks(dask_tasks, dask_worker_count)):
        print((cindx+1)*dask_worker_count,'/', len(dask_tasks))
        dask_res = list(map(lambda x: np.sum(x,axis=0), zip(*dask.compute(*task))))
        lagCOVv += dask_res[0]
        lagCOVm1 += dask_res[1]
        lagCOVm2 += dask_res[2]
        lagCOVn += dask_res[3]

    lagCOVv /= lagCOVn
    lagCOVm1 /= lagCOVn
    lagCOVm2 /= lagCOVn
    lagCOVv -= lagCOVm1*lagCOVm2
    
    return lagCOVv, lagCOVn, lagSS, lagTT


def _cov_index(grid_v, idx_s_i, idx_s_j, s_chunksize, s_head, s_tail, chunk_grid_t, cov_shape, idx_s_lag, t_chunksize, t_lag, t_lag_range):
    lagCOVv = np.zeros(cov_shape)
    lagCOVm1 = np.zeros(cov_shape)
    lagCOVm2 = np.zeros(cov_shape)
    lagCOVn = np.zeros(cov_shape)

    grid_v_head_s = grid_v[idx_s_i * s_chunksize + s_head]
    grid_v_tail_s = grid_v[idx_s_j * s_chunksize + s_tail]

    if idx_s_i == idx_s_j:
        for idx_t_i in range(len(chunk_grid_t)):
            for idx_t_j in range(idx_t_i, len(chunk_grid_t)):
                t_i = chunk_grid_t[idx_t_i]
                t_j = chunk_grid_t[idx_t_j]
                dt = cdist_scipy(t_i, t_j)
                for idx_t_lag, n in enumerate(t_lag):
                    t_head, t_tail = np.where(
                        (dt >= n - t_lag_range[idx_t_lag])
                        & (dt <= n + t_lag_range[idx_t_lag])
                        )
                    if t_head.size == 0:
                        continue
                    grid_v_head = grid_v_head_s[:, idx_t_i * t_chunksize + t_head]
                    grid_v_tail = grid_v_tail_s[:, idx_t_j * t_chunksize + t_tail]
                    grid_v_ht = grid_v_head * grid_v_tail
                    bool_notnan = ~np.isnan(grid_v_ht)
                    grid_v_ht = grid_v_ht[bool_notnan]
                    if idx_t_i == idx_t_j:
                        lagCOVv[idx_s_lag][idx_t_lag] += np.sum(grid_v_ht)
                        lagCOVm1[idx_s_lag][idx_t_lag] += np.sum(grid_v_head[bool_notnan])
                        lagCOVm2[idx_s_lag][idx_t_lag] += np.sum(grid_v_tail[bool_notnan])
                        lagCOVn[idx_s_lag][idx_t_lag] += grid_v_ht.size
                    else:
                        lagCOVv[idx_s_lag][idx_t_lag] += np.sum(grid_v_ht) * 2
                        lagCOVm1[idx_s_lag][idx_t_lag] += np.sum(grid_v_head[bool_notnan] + grid_v_tail[bool_notnan])
                        lagCOVm2[idx_s_lag][idx_t_lag] += np.sum(grid_v_head[bool_notnan] + grid_v_tail[bool_notnan])
                        lagCOVn[idx_s_lag][idx_t_lag] += grid_v_ht.size * 2
    else:
        for idx_t_i in range(len(chunk_grid_t)):
            for idx_t_j in range(len(chunk_grid_t)):
                t_i = chunk_grid_t[idx_t_i]
                t_j = chunk_grid_t[idx_t_j]
                dt = cdist_scipy(t_i, t_j)
                for idx_t_lag, n in enumerate(t_lag):
                    t_head, t_tail = np.where(
                        (dt >= n - t_lag_range[idx_t_lag])
                        & (dt <= n + t_lag_range[idx_t_lag])
                        )
                    if t_head.size == 0:
                        continue
                    grid_v_head = grid_v_head_s[:, idx_t_i * t_chunksize + t_head]
                    grid_v_tail = grid_v_tail_s[:, idx_t_j * t_chunksize + t_tail]
                    grid_v_ht = grid_v_head * grid_v_tail
                    bool_notnan = ~np.isnan(grid_v_ht)
                    grid_v_ht = grid_v_ht[bool_notnan]
                    lagCOVv[idx_s_lag][idx_t_lag] += np.sum(grid_v_ht) * 2
                    lagCOVm1[idx_s_lag][idx_t_lag] += np.sum(grid_v_head[bool_notnan] + grid_v_tail[bool_notnan])
                    lagCOVm2[idx_s_lag][idx_t_lag] += np.sum(grid_v_head[bool_notnan] + grid_v_tail[bool_notnan])
                    lagCOVn[idx_s_lag][idx_t_lag] += grid_v_ht.size * 2

    return [lagCOVv, lagCOVm1, lagCOVm2, lagCOVn]


def _cov_equal_index(cov_shape, idx_s_lag, idx_t_i, t_i, idx_t_j, t_j,
    t_chunksize, t_lag, t_lag_range, grid_v_head_s, grid_v_tail_s):
    lagCOVv = np.zeros(cov_shape)
    lagCOVm1 = np.zeros(cov_shape)
    lagCOVm2 = np.zeros(cov_shape)
    lagCOVn = np.zeros(cov_shape)

    dt = cdist_scipy(t_i, t_j)
    for idx_t_lag, n in enumerate(t_lag):
        t_head, t_tail = np.where(
            (dt >= n - t_lag_range[idx_t_lag])
            & (dt <= n + t_lag_range[idx_t_lag])
            )
        if t_head.size == 0:
            continue
        grid_v_head = grid_v_head_s[:, idx_t_i * t_chunksize + t_head]
        grid_v_tail = grid_v_tail_s[:, idx_t_j * t_chunksize + t_tail]
        grid_v_ht = grid_v_head * grid_v_tail
        bool_notnan = ~np.isnan(grid_v_ht)
        grid_v_ht = grid_v_ht[bool_notnan]
        if idx_t_i == idx_t_j:
            lagCOVv[idx_s_lag][idx_t_lag] += np.sum(grid_v_ht)
            lagCOVm1[idx_s_lag][idx_t_lag] += np.sum(grid_v_head[bool_notnan])
            lagCOVm2[idx_s_lag][idx_t_lag] += np.sum(grid_v_tail[bool_notnan])
            lagCOVn[idx_s_lag][idx_t_lag] += grid_v_ht.size
        else:
            lagCOVv[idx_s_lag][idx_t_lag] += np.sum(grid_v_ht) * 2
            lagCOVm1[idx_s_lag][idx_t_lag] += np.sum(grid_v_head[bool_notnan] + grid_v_tail[bool_notnan])
            lagCOVm2[idx_s_lag][idx_t_lag] += np.sum(grid_v_head[bool_notnan] + grid_v_tail[bool_notnan])
            lagCOVn[idx_s_lag][idx_t_lag] += grid_v_ht.size * 2
    return [lagCOVv, lagCOVm1, lagCOVm2, lagCOVn]


def _cov_unequal_index(cov_shape, idx_s_lag, idx_t_i, t_i, idx_t_j, t_j,
    t_chunksize, t_lag, t_lag_range, grid_v_head_s, grid_v_tail_s):
    lagCOVv = np.zeros(cov_shape)
    lagCOVm1 = np.zeros(cov_shape)
    lagCOVm2 = np.zeros(cov_shape)
    lagCOVn = np.zeros(cov_shape)

    dt = cdist_scipy(t_i, t_j)
    for idx_t_lag, n in enumerate(t_lag):
        t_head, t_tail = np.where(
            (dt >= n - t_lag_range[idx_t_lag])
            & (dt <= n + t_lag_range[idx_t_lag])
            )
        if t_head.size == 0:
            continue
        grid_v_head = grid_v_head_s[:, idx_t_i * t_chunksize + t_head]
        grid_v_tail = grid_v_tail_s[:, idx_t_j * t_chunksize + t_tail]
        grid_v_ht = grid_v_head * grid_v_tail
        bool_notnan = ~np.isnan(grid_v_ht)
        grid_v_ht = grid_v_ht[bool_notnan]
        lagCOVv[idx_s_lag][idx_t_lag] += np.sum(grid_v_ht) * 2
        lagCOVm1[idx_s_lag][idx_t_lag] += np.sum(grid_v_head[bool_notnan] + grid_v_tail[bool_notnan])
        lagCOVm2[idx_s_lag][idx_t_lag] += np.sum(grid_v_head[bool_notnan] + grid_v_tail[bool_notnan])
        lagCOVn[idx_s_lag][idx_t_lag] += grid_v_ht.size * 2
    return [lagCOVv, lagCOVm1, lagCOVm2, lagCOVn]


def cov_cal(Zi, Zj, Tmin, Tmax):
    '''Compute covariance between two time series within a time window.

    Zi, Zj : pd.Series with datetime index
    Tmin, Tmax : str or pd.Timedelta

    Returns cov_v, cov_n, cov_m1, cov_m2
    '''
    Zi = Zi[~Zi.isnull()]
    Zj = Zj[~Zj.isnull()]

    df_v = pd.DataFrame(
        data=Zi.to_frame().values * Zj.to_frame().T.values,
        index=Zi.index, columns=Zj.index)
    df_dt = pd.DataFrame(
        data=df_v.index.to_frame().values - df_v.columns.to_frame().T.values,
        index=df_v.index, columns=df_v.columns)

    bool_picker = (Tmax >= df_dt) & (df_dt >= Tmin)
    cov_v = df_v[bool_picker].sum().sum()
    df_dt = df_dt[bool_picker]
    df_dt_count_col = df_dt.count(axis='index')
    df_dt_count_row = df_dt.count(axis='columns')
    cov_n = df_dt_count_col.sum()
    cov_m1 = (df_dt_count_row * Zi).sum()
    cov_m2 = (df_dt_count_col * Zj).sum()

    return cov_v, cov_n, cov_m1, cov_m2
