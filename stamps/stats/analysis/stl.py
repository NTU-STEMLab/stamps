# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 23:03:23 2014

@author: hdragon689
"""

import datetime
import pandas
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
except ImportError as e:
    raise ImportError(
        'stamps: rpy2 is required for STL functions. '
        'Install it with:  pip install rpy2'
    ) from e
import matplotlib.pyplot as plt
from numpy import asarray, ceil, isnan


# import r functions
zoo_ = importr('zoo')
ts_ = robjects.r['ts']
stl_ = robjects.r['stl']
naaction_ = robjects.r['na.approx'] # Generic functions for replacing each 
                                      # NA with interpolated values.


def stl(data, ns, np=None, nt=None, nl=None, isdeg=0, itdeg=1, ildeg=1,
        nsjump=None, ntjump=None, nljump=None, robust=False, ni=2, no=0, fulloutput=False):
  """
  Seasonal-Trend decomposition procedure based on LOESS
  
  data : pandas.Series or 1D numpy array, in which np is required 
         for numpy array input
  
  ns : int
  	Length of the seasonal smoother.
  	The value of  ns should be an odd integer greater than or equal to 3.
  	A value ns>6 is recommended. As ns  increases  the  values  of  the
  	seasonal component at a given point in the seasonal cycle (e.g., January
  	values of a monthly series with  a  yearly cycle) become smoother.
  
  np : int
  	Period of the seasonal component.
  	For example, if  the  time series is monthly with a yearly cycle, then
  	np=12.
  	If no value is given, then the period will be determined from the
  	``data`` timeseries.
  
  nt : int
  	Length of the trend smoother.
  	The  value  of  nt should be an odd integer greater than or equal to 3.
  	A value of nt between 1.5*np and 2*np is  recommended. As nt increases,
  	the values of the trend component become  smoother.
  	If nt is None, it is estimated as the smallest odd integer greater
  	or equal to ``(1.5*np)/[1-(1.5/ns)]``
  
  nl : int
  	Length of the low-pass filter.
  	The value of nl should  be an odd integer greater than or equal to 3.
  	The smallest odd integer greater than or equal to np is used by default.
  
  isdeg : int
  	Degree of locally-fitted polynomial in seasonal smoothing.
  	The value is 0 or 1.
  
  itdeg : int
  	Degree of locally-fitted polynomial in trend smoothing.
  	The value is 0 or 1.
  
  ildeg : int
  	Degree of locally-fitted polynomial in low-pass smoothing.
  	The value is 0 or 1.
  
  nsjump : int
  	Skipping value for seasonal smoothing.
  	The seasonal smoother skips ahead nsjump points and then linearly
  	interpolates in between.  The value  of nsjump should be a positive
  	integer; if nsjump=1, a seasonal smooth is calculated at all n points.
  	To make the procedure run faster, a reasonable choice for nsjump is
  	10%-20% of ns. By default, nsjump= 0.1*ns.

  ntjump : int
  	Skipping value for trend smoothing. If None, ntjump= 0.1*nt
  
  nljump : int
  	Skipping value for low-pass smoothing. If None, nljump= 0.1*nl
  
  robust : bool
  	logical indicating if robust fitting be used in the loess procedure      
  
  ni :int
  	Number of loops for updating the seasonal and trend  components.
  	The value of ni should be a positive integer.
  	See the next argument for advice on the  choice of ni.
  	If ni is None, ni is set to 2 for robust fitting, to 5 otherwise.

  no : int
  	Number of iterations of robust fitting. The value of no should
  	be a nonnegative integer. If the data are well behaved without
  	outliers, then robustness iterations are not needed. In this case
  	set no=0, and set ni=2 to 5 depending on how much security
  	you want that  the seasonal-trend looping converges.
  	If outliers are present then no=3 is a very secure value unless
  	the outliers are radical, in which case no=5 or even 10 might
  	be better.  If no>0 then set ni to 1 or 2.
  	If None, then no is set to 15 for robust fitting, to 0 otherwise.
  
  fulloutput : bool
  	If True, a dictionary holding the full output of the original R routine
  	will be returned.
  
  returns

  data : pandas.DataFrame or 2D numpy array 
  	A panda.dataframe has the seasonal, trend, and remainder components. 
    The output format is consistent with the one of the input data. 
    If numpt array is used, the three columns are in order 
    for seasonal, trend, and remainder
  
  Note: this code is downloaded from https://gist.github.com/andreas-h/7808564 
  and modified by H-L Yu 2015/06/09
  """         

  # make sure that data doesn't start or end with nan
  #    _data = data.copy()
  #    _data = _data.dropna()
  # TODO: account for non-monthly series
  # Mark by H-L  
  #  _idx = pandas.date_range(start=_data.index[0], end=_data.index[-1],
  #                          offset=pandas.datetools.MonthBegin())
  #    _idx = pandas.date_range(start=_data.index[0], end=_data.index[-1])                        
  #    data = pandas.Series(index=_idx)
  #    data[_data.index] = _data
 
  # zoo package contains na.approx


 
    
  # find out the period of the time series
  if np is None:
    try: 
      freq=data.index.freqstr[0]
      if freq=='M':
        np = 12
      if freq=='D':
        np = 365
      if freq=='W':
        np = 52
    except:
        np=12   # if no np is provided and also can not be recognized by data
                # itself. The data is asssumed to be np==12
        
        # The following comment-out block is from the original code
        # TODO: find out the offset of the Series, and set np accordingly
        #if isinstance(data.index.offset, pandas.core.datetools.MonthEnd):
        #    np = 12
        #else:
        #    raise NotImplementedError()
    # fill default values
      
#  if np is 12:
#    start = robjects.IntVector([data.index[0].year, data.index[0].month])
#    ts = ts_(robjects.FloatVector(asarray(data)), start=start, frequency=np)
  
  # convert data to R object    
  ts = ts_(robjects.FloatVector(asarray(data)), start=1, frequency=np, \
          deltat=1)

  if robust:
    ni=1; no=15;
  else:
    ni=2; no=0;

  if nt is None:
    nt = robjects.rinterface.R_NilValue
  else:
    nt = nt + 1 if nt % 2 == 0 else nt # to assure nt is an odd number

  if nl is None:
    nl = np if np % 2 == 1 else np + 1 
    
  if type(ns)==str and (ns=='per' or ns=='periodic'):
    if isnan(data).any():        
      if ntjump is None:
        ntjump = robjects.rinterface.R_NilValue
      if nljump is None:
        nljump = robjects.rinterface.R_NilValue
      if nsjump is None:
        nsjump = robjects.rinterface.R_NilValue
      result = stl_(ts, ns, isdeg, nt, itdeg, nl, ildeg, nsjump, ntjump, nljump,
                    False, ni, no, naaction_)    
    else:
      result = stl_(ts, ns)
  else:
    if type(nt)==robjects.rinterface.Sexp:
      nt = ceil((1.5 * np) / (1 - (1.5 / ns)))      
    if ntjump is None:
      ntjump = ceil(nt / 10.)
    if nljump is None:
      nljump = ceil(nl / 10.) 
    if nsjump is None:
      nsjump = ceil(ns / 10.)
    
    result = stl_(ts, ns, isdeg, nt, itdeg, nl, ildeg, nsjump, ntjump, nljump,
                  False, ni, no, naaction_)    
 
  res_ts = asarray(result[0])
    
  try:
    res_ts = pandas.DataFrame({"seasonal" : pandas.Series(res_ts[:,0],
                                                    index=data.index),
                               "trend" : pandas.Series(res_ts[:,1],
                                                    index=data.index),
                               "remainder" : pandas.Series(res_ts[:,2],
                                                    index=data.index)})
  except:
    return res_ts #, data
    raise
#        res_ts = pandas.DataFrame({"seasonal" : pandas.Series(index=data.index),
#                                   "trend" : pandas.Series(index=data.index),
#                                   "remainder" : pandas.Series(index=data.index)})
 
  if fulloutput:
    return {"time.series" : res_ts,
            "weights" : result[1],
            "call" : result[2],
            "win" : result[3],
            "deg" : result[4],
            "jump" : result[5],
            "ni" : result[6],
            "no" : result[7]}
  else:
    return res_ts


def stl_plot(data, res, tME=None,title=None):
  '''
  Plot the data and the extracted trend, seasonal, and residual components from
  stl analysis
  
  SYNTAX: stl_plot(data, res, tME=None, title=None)  
  
  INPUT: 
  data    pandas.Series       time series data of the original data 
                              or 1D numpy array
  res     pandas.DataFrame    time series of trend, seasonal, and residual 
                              components. This input is supposed to be the 
                              output of stl function 
  tME     1D numpy array      time index for data and res        
  title   string              title of this figure                            
  '''  

  if tME==None:   
    try:
      tME=data.index
    except:
      tME=np.arange(1,data.size)  
    
  f, axarr = plt.subplots(4, sharex=True)
  if title:
    axarr[0].set_title(title)
  axarr[0].plot(tME, data)
  axarr[0].set_ylabel('Original Data')
  try: # try dataframe
    axarr[1].plot(tME, res['trend'])
    axarr[1].set_ylabel('trend')
    axarr[2].plot(tME, res['seasonal'])
    axarr[2].set_ylabel('seasonal')
    axarr[3].plot(tME, res['remainder'])
    axarr[3].set_ylabel('remainder')
  except: # do numpy array
    axarr[1].plot(tME, res[:,1])
    axarr[1].set_ylabel('trend')
    axarr[2].plot(tME, res[:,0])
    axarr[2].set_ylabel('seasonal')
    axarr[3].plot(tME, res[:,2])
    axarr[3].set_ylabel('remainder')  
  plt.show()  


if __name__ == "__main__":
    
    import numpy as np
    
    data = np.arange(85.) / 12.
    data = np.sin(data * (2*np.pi))
    data += np.arange(85.) / 12. * .5
    data += .1 * np.random.randn(85)
    idx = pandas.date_range(start=datetime.datetime(1999,1,1), \
        end=datetime.datetime(2006,2,1),freq='M')
    data = pandas.Series(data, index=idx)
 
    res = stl(data, np=12, ns=24) #, nt=11)
    
    stl_plot(data,res)
