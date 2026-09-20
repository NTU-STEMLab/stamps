# -*- coding: utf-8 -*-
"""
Data conversion between space/time vector and space/time grid formats

Created on Sat Jun 27 11:40:16 2015

@author: hdragon689
"""

from six.moves import range
import numpy as np
import pandas as pd


def valstv2stg(ch, z, cMS=None, tME=None):
  '''
  Converts the values of a space/time variable from a s/t vector 
  format (i.e. the variable z is listed as a vector of n values)
  to a grid format (i.e. the variable Z is given as a nMS by nME matrix 
  corresponding to nMS Monitoring Sites and nME Measuring Events).
  Use help stgridsyntax for information on the s/t grid format.
 
  SYNTAX :
 
  [Z,cMS,tME,nanratio]=valstv2stg(ch,z,cMS,tME);
 
  INPUT :
 
  ch         n by d+1     matrix of space/time coordinates for spatial domain of dimension d
  z          n by 1       vector of field value at coordinate ch
  cMS        nMS by 2     optional matrix of spatial coordinates for the nMS Measuring Sites
  tME        1 by nME     optional vector of times of the tME Measuring Events
 
  OUTPUT :
 
  Z          nMS by nME   matrix of values for the variable Z corresponding to 
                          nMS Monitoring Sites and nME Measuring Event
  cMS        nMS by d     matrix of spatial coordinates for the nMS Measuring Sites
  tME        1 by nME     vector of times of the tME Measuring Events
  nanratio   scalar       ratio of the NaNs in Z (0<=nanratio<=1) 
 
  NOTE : 
 
  cMS and tME can be provided as input if they are both known.  In that case ch 
  must be a nMS*nME by 3 matrix of the points corresponding to nMS Monitoring Sites
  and nME Measuring Events, listed with space cycling quicker then time.
  '''
  
  if 'pandas' in str(type(z)):
    z=z.values
    
  cols=['x','y','t','z']

  data=np.hstack((ch,z.reshape(z.size,1)))
  datadf=pd.DataFrame(data,columns=cols)  
  datadf['x']=datadf['x'].astype(np.double)
  datadf['y']=datadf['y'].astype(np.double)
  datadf['z']=datadf['z'].astype(np.double)
  dtable=pd.pivot_table(datadf, values=datadf.columns[3], index=['y', 'x'], columns=['t']) 
  # cMS
  cMS_=list(zip(*np.array(dtable.index)))
  cMS_=np.array([np.asarray(cMS_[0]),np.asarray(cMS_[1])])
  cMS_=cMS_.T.astype(float)
  cMS_=cMS_[:,[1,0]]

  #tME
  try:
    tME_=np.array(dtable.columns).astype(ch[:,2].dtype)
  except AttributeError:
    tME_=np.array(dtable.columns).astype(type(ch[:,2]))

  # Z
  Z_=dtable.values.astype(np.double)  
  
  nt=Z_.shape[1]
  
  if cMS is not None:
    Z=[]
    for i in range(cMS.shape[0]):
      ii=np.where(np.logical_and(cMS_[:,0]==cMS[i,0],cMS_[:,1]==cMS[i,1]))
      Z.append(Z_[ii,:].reshape(nt))
    Z=np.asarray(Z,dtype=np.double) 
  else:
    cMS=cMS_
    Z=Z_
  if tME is not None:
    tidx=[]
    for j,tMEi in enumerate(tME):
      tidx.append(np.where(tME==tMEi))
    tidx=np.asarray(tidx).reshape(tME.size)  
    Z=Z[:,tidx]
  else:
    tME=tME_    
        
  # nonlocations
  nanloc=list(
    zip(
      np.where(np.isnan(Z))[0],
      np.where(np.isnan(Z))[1]
      )
    )
  
  return Z, cMS, tME, nanloc

def valstg2stv(Z, cMS, tME):
  '''
  Converts the coordinates and values of a space/time variable 
  from a grid format (i.e. the variable Z is given as a nMS by nME matrix  
  corresponding to nMS Measuring Sites and nME Measuring Events),
  to a s/t vector format (i.e. the variable z is listed as a vector of nMS*nME values,
  corresponding to points with space/time coordinates, where the spatial coordinate
  cycle quicker than the time coordinates).
 
  SYNTAX :
 
  [ch,z]=valstg2stv(Z,cMS,tME);
 
  INPUT :
 
  Z        nMS by nME       matrix of values for the variable Z corresponding to 
                            nMS Monitoring Sites and nME Measuring Event
  cMS      nMS by 2         matrix of 2D spatial coordinates for the nMS Measuring Sites
  tME      1 by nME         vector of times of the tME Measuring Events
 
  OUTPUT :
 
  ch       nMS*nME by 3     matrix of space time coordinates, listing the space/time
                            locations of the points corresponding to nMS Monitoring Sites
                            and nME Measuring Event (space cycles quicker then time) 
  z        nMS*nME by 1     vector of values for the variable Z corresponding to the
                            s/t points ch
  '''
  
  nc=cMS.shape[0]
  nt=tME.size
  
  zh=(Z.T).reshape(nc*nt,1)
  ch=np.asarray(list(zip(np.tile(cMS[:,0],nt),np.tile(cMS[:,1],nt), \
                  tME.repeat(nc))))
                  
  return ch, zh

def valstv2stg2(ch, zh, cMS=None, tME=None):
    if 'pandas' in str(type(ch)):
        ch = ch.values
    if 'pandas' in str(type(zh)):
        zh = zh.values
    
    cols = ['x','y','t','z']
    data = np.hstack((ch, zh.reshape(zh.size, 1)))
    datadf = pd.DataFrame(data, columns=cols)
    datadf['x'] = datadf['x'].astype(float)
    datadf['y'] = datadf['y'].astype(float)
    datadf['z'] = datadf['z'].astype(np.double)
    dtable = pd.pivot_table(
        datadf, values=datadf.columns[3], index=['y', 'x'], columns=['t'],
        dropna=True
    )
    
    if cMS is not None:
        c_index = pd.MultiIndex.from_arrays(cMS.T, names=('x', 'y'))
        dtable = dtable.swaplevel().reindex(c_index)
    else:
        dtable = dtable.swaplevel()
    cMS = np.array(dtable.index.to_list())
    if tME is not None:
        dtable = dtable.reindex(columns = tME)
    tME = dtable.columns.to_numpy()
    
    Z = dtable.values
   
    # nonlocations
    nanloc=list(
        zip(
          np.where(np.isnan(Z))[0],
          np.where(np.isnan(Z))[1]
          )
        )
    
    return Z, cMS, tME, nanloc

if __name__ == "__main__":
  import time
  
  data='../examples/Data/GeoData.xls'      
  datadf=pd.ExcelFile(data).parse('Sheet1',header=None)
  ch=datadf.iloc[:,0:3].values
  z=datadf.iloc[:,4].values.reshape(ch.shape[0],1)
  
  stime=time.time()
  Z1,cMS,tME,nanloc=valstv2stg(ch,z)
  print(time.time()-stime)
  
  ch2,z2=valstg2stv(Z1,cMS,tME)
  
  Z2,cMS2,tME2,nanloc=valstv2stg(ch2,z2,cMS[3:,:],tME[3:])
  Z3,cMS3,tME3,nanloc=valstv2stg(ch2,z2)
  
  id=np.where((Z1!=Z2))
  print(id)
