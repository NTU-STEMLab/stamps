# -*- coding: utf-8 -*-
from six.moves import range

import numpy as np
from scipy.interpolate import griddata

from ..general.coord2dist import coord2dist
from ..stats.analysis.eof import eof
try:
    from ..stats.analysis.stl import stl  # optional: requires rpy2
except (ImportError, ModuleNotFoundError):
    stl = None  # stmean_stl will raise at call-time if rpy2 is absent
from . import idw


#try:
#    from pylab import griddata as pylab_griddata #qgis
#except ImportError:
#    from matplotlib.pylab import griddata as pylab_griddata #OSGEO4W qgis

def _tricube(x):
    w=np.zeros(x.size)
    xx=np.abs(x)
    idx=np.where(np.bitwise_and(xx>=0,xx<1))
    w[idx]=(1-xx[idx]**3)**3
    return w

def _bicube(x):
    w=np.zeros(x.size)
    idx=np.where(np.abs(x)<1)
    w[idx]=(1-np.abs(x[idx])**2)**2
    return w 

def _exp(x):
    w=np.zeros(x.size)
    w=np.exp(-3*x)
    return w   

def stmean( grid_s, grid_t, grid_z, pars=None,kernel='tricube', smooth=True, DataObj = None ):
    """Estimates space/time mean values from measurements.

    Assuming a separable additive space/time mean trend model, this 
    function calculates the spatial mean component ms and temporal mean 
    component mt of space/time random field grid_z, using measurements at 
    fixed measuring  sites n and fixed measuring events m. 
    The spatial mean component ms is obtained by averaging the measurements
    at each measuring sites. Then a smoothed spatial mean component mss
    is obtained by applying a specified spatial filter to ms.
    Similarly mt is obtained by averaging the measurement for each
    measuring event, and a smoothed temporal mean component mts is obtained
    by applying the same function of temporal filter to mt.

    Args:
        grid_s (ndarray): shape of (n, 2).
            Data point coordinates.
        grid_t (ndarray): shape of (m, 1).
            Data point timestamps.
        grid_z (ndarray): shape of (n, m).
            Matrix of measurements at the n monitoring sites and m measuring events. 
            grid_z may have NaN values.
        pars (array_like): Length is 4. Parameters to smooth the spatial and temporal average.

            p[0]=dNeib, distance (radius) of spatial neighborhood

            p[1]=tNeib, time (radius) ofs temporal neighborhood

            p[2]=ar, spatial range of kernel smoothing function 

            p[3]=at, temporal range of kernel smoothing function
        kernel (str): {'exp', 'tricube', 'bicube'}. Specified kernel function for smoothing.
        smooth (bool): Whether to smooth the spa-temporal mean by given kernel function.
        DataObj (bool): GUI for STAR-BME.

    Returns:
        mean_s (ndarray): shape of (n, 1). vector of spatial average
        mss (ndarray): shape of  (n, 1). The vector of smoothed spatial average
        mean_t (ndarray): shape of (1, m). vector of temporal average
        mts (ndarray): shape of (1, m). vector of smoothed temporal average 
        STmean (ndarray):shape of (n, m). Estimated spa-temporal mean.

    Examples:
        Randomly sampling 30 points in 20000x20000 fields and suppose there are 24 measurement events.
        >>> grid_s = np.random.random_sample((30,2))*20000
        >>> grid_t = np.arange(1,25)

        Randomly sampling 36 measuring events of each point.
        >>> grid_z = np.random.random_sample((30,24))+np.random.random_sample((30,1))*10
        
        Define the smoothing parameters of kernel.
        >>> dNeib,tNeib,ar,at = 5000,5,5000,5
        >>> pars = [dNeib,tNeib,ar,at]
        
        Calculate the spa-temporal mean.
        >>> mean_s,mss,mean_t,mts,STmean = stmean(grid_s = grid_s, grid_t = grid_t,grid_z = grid_z, pars = pars,kernel='tricube',smooth = True)
    """

#    if not DataObj:
#        from nousedataobj import NoUseDataObj
#        DataObj = NoUseDataObj()
    if DataObj:    
        title = DataObj.getProgressText()    
        DataObj.setProgressRange(0,len(grid_s))
        DataObj.setCurrentProgress(0, title + "\n- By STMean...")

    grid_t=np.reshape(grid_t,(1,grid_t.size))  
    grid_t=grid_t.astype(np.float64)

    mask_grid_z = np.ma.masked_array(grid_z,np.isnan(grid_z))
    mean_s = np.array( mask_grid_z.mean( axis = 1 ) , ndmin = 2).T
    mean_t = np.array( mask_grid_z.mean( axis = 0 ) , ndmin = 2)
    mean_st = mask_grid_z.mean()


    mss=np.zeros(mean_s.shape)
    mts=np.zeros(mean_t.shape)

    s_dist=coord2dist(grid_s,grid_s)
    t_dist=coord2dist(grid_t.T,grid_t.T,1) 
    if pars is None:
        dNeib=np.max(s_dist)*0.4
        tNeib=np.max(t_dist)*0.4
        ar=dNeib
        at=tNeib    
    else:
        dNeib=pars[0]
        tNeib=pars[1]
        ar=pars[2]
        at=pars[3]
    
    if kernel == 'exp':
        fun='_exp'
    elif kernel == 'tricube':
        fun='_tricube'
    elif kernel == 'bicube':
        fun='_bicube'
        
    for i in range(grid_s.shape[0]):
        ids=np.where(s_dist[i]<=dNeib)
        w=eval(fun+'(s_dist[i][ids]/ar)')#_tricube(s_dist[i]/dNeib)
        w=w/np.sum(w)  # normalize
        mss[i]=(w*mean_s.flat[ids].flat[:]).sum()
        
    for j in range(grid_t.size):
        idt=np.where(t_dist[j]<=tNeib)
        w=eval(fun+'(t_dist[j][idt]/at)')#w=_tricube(t_dist[j]/tNeib)
        w=w/np.sum(w)  # normalize
        mts[0,j]=(w*mean_t.flat[idt].flat[:]).sum()
    if smooth:
        STmean = mss + mts - mean_st 
    else:
        STmean = mean_s+mean_t-mean_st
    STmean[np.where(np.isnan(grid_z))] = np.nan
    
    return mean_s,mss,mean_t,mts,STmean

def stmean_stl( grid_s, grid_t, grid_z, pars=None, kernel='tricube', DataObj = None ):
    '''
% stmean_stl                  - Estimates space/time mean values from measurements
%
% Assuming a separable additive space/time mean trend model, this 
% function calculates the spatial mean component ms and temporal mean 
% component mt of space/time random field Z, using measurements at 
% fixed measuring  sites cMS and fixed measuring events tME. 
% The spatial mean component ms is obtained by averaging the measurements
% at each measuring sites. Then a smoothed spatial mean component mss
% is obtained by applying an exponential spatial filter to ms.
% In this function, a smoothed temporal mean component mts is obtained
% by applying the stl function to extract the periodic trend of mt. 
% Then the space/time mean trend is simply given by 
% mst(s,t)=mss(s)+mts(t)-mean(mts)
%
% SYNTAX :
%
% [ms,mss,mt,mts,stmean]=stmean(cMS,tME,Z,pars,DataObj=None);
%
% INPUTS :  
%  cMS    nMS by 2   matrix of spatial x-y coordinates for the nMS monitoring
%                    sites
%  idMS   nMS by 1   vector with a unique id number for each monitoring site
%  tME    1 by nME   vector with the time of the measuring events
%  Z      nMS by nME matrix of measurements for Z at the nMS monitoring
%                    sites and nME measuring events. Z may have NaN values.
%  pars   1 by 2     parameters to smooth the spatial and temporal average
%                    p[0]=dNeib  distance (radius) of spatial neighborhood
%                    p[1]=np     the time span for seasonal periodic variation 
                                                                 (default=12)
                                         p[2]=ar  spatial range of kernel smoothing function             
%
% OUTPUT :
%
%  ms     nMS by 1   vector of spatial average
%  mss    nMS by 1   vector of smoothed spatial average
%  mt     1 by nME   vector of temporal average
%  mts    nMS by 1   vector of smoothed temporal average   

    Note: See stats.stl 
    '''

#    if not DataObj:
#        from nousedataobj import NoUseDataObj
#        DataObj = NoUseDataObj()
    if DataObj:    
        title = DataObj.getProgressText()    
        DataObj.setProgressRange(0,len(grid_s))
        DataObj.setCurrentProgress(0, title + "\n- By STMean...")

    grid_t=np.reshape(grid_t,(1,grid_t.size))  
    grid_t=grid_t.astype(np.float64)

    mask_grid_z = np.ma.masked_array(grid_z,np.isnan(grid_z))
    mean_s = np.array( mask_grid_z.mean( axis = 1 ) , ndmin = 2).T
    mean_t = np.array( mask_grid_z.mean( axis = 0 ) , ndmin = 2)
#  mean_st = mask_grid_z.mean()

    if kernel == 'exp':
        fun='_exp'
    elif kernel == 'tricube':
        fun='_tricube'
    elif kernel == 'bicube':
        fun='_bicube'
        
    mss=np.zeros(mean_s.shape)
    mts=np.zeros(mean_t.shape)

    s_dist=coord2dist(grid_s,grid_s)
    if pars is None:
        dNeib=np.max(s_dist)*0.4
        nper=12
    else:
        dNeib=pars[0]
        nper=pars[1]
        ar=pars[2]
        
    for i in range(grid_s.shape[0]):
        ids=np.where(s_dist[i]<=dNeib)
        w=eval(fun+'(s_dist[i][ids]/ar)')#_tricube(s_dist[i]/dNeib)
        w=w/np.sum(w)  # normalize
        mss[i]=(w*mean_s.flat[ids].flat[:]).sum()
        
    mts=stl(mean_t.ravel(),ns='per',np=nper)
            
    return mean_s,mss,mean_t,mts  

def stmean_eof(grid_z, explained_variance_ratio_=90., centered=True, normalized=True, method='svd'):

    '''
    Remove mean trend with Empirical Orthogonal Function.

    Args:
        grid_z (ndarray): space-time data matrix (n_sites x n_times).
        explained_variance_ratio_ (float): cumulative explained variance threshold (%).
            EOFs are kept until this percentage of variance is explained.
        centered (bool): center data before EOF decomposition (norm=1 maps to centered).
        normalized (bool): normalize EOFs (norms_direc=1).
        method (str): decomposition method ('svd' or 'eig').

    Returns:
        meantrend (ndarray): reconstructed mean trend matrix.
    '''
    # Map old centered/normalized args to new eof API (norm, norms_direc)
    norm = 1 if centered else 0
    norms_direc = 1 if normalized else 0

    L, s, PCs, EOFs, ECs, error, norms = eof(grid_z, n='all', norm=norm,
                                              norms_direc=norms_direc, method=method)
    
    sum_var = sum(L)
    i = len(L) - 1  # default: keep all
    for i in range(len(L)):
        explained = (sum(L[:i+1])/ sum_var) * 100
        if explained > explained_variance_ratio_:
            break

    meantrend = np.dot(ECs[:,:i+1],EOFs.T[:i+1,:])

    print("Meantrend explained %f variance of data" % explained)
    
    return meantrend

def stmeaninterp(grid_s, grid_t, grid_z, est_grid_s, est_grid_t,
                             method='idw', pars=None, smooth=True, DataObj = None):
    """Interpolating space/time trend through space/time mean values estimating from measurements.

    Assuming a separable additive space/time mean trend model, this 
    function calculates the spatial mean component ms and temporal mean 
    component mt of space/time random field grid_z, using measurements at 
    fixed measuring  sites n and fixed measuring events m. 
    The spatial mean component ms is obtained by averaging the measurements
    at each measuring sites. Then a smoothed spatial mean component mss
    is obtained by applying a specified spatial filter to ms.
    Similarly mt is obtained by averaging the measurement for each
    measuring event, and a smoothed temporal mean component mts is obtained
    by applying the same function of temporal filter to mt.

    Args:
        grid_s (ndarray): shape of (n, 2).
            Data point coordinates.
        grid_t (ndarray): shape of (m, 1).
            Data point timestamps.
        grid_z (ndarray): shape of (n, m).
            Matrix of measurements at the n monitoring sites and m measuring events. 
            grid_z may have NaN values.
        est_grid_s (ndarray): shape of (k, 2). 
            Point coordinates at which to interpolate data.
        est_grid_t (ndarray): shape of (j, 1). 
            Time at which to interpolate data.
        method (str):{'linear', 'nearest', 'cubic', 'idw'}. 
            method used to estimate grid trend
        pars (array_like): Length is 4. Parameters to smooth the spatial and temporal average.

            p[0]=dNeib, distance (radius) of spatial neighborhood
            
            p[1]=tNeib, time (radius) ofs temporal neighborhood
            
            p[2]=ar, spatial range of kernel smoothing function 
            
            p[3]=at, temporal range of kernel smoothing function
        kernel (str): {'exp', 'tricube', 'bicube'}. Specified kernel function for smoothing.
        smooth (bool): Whether to smooth the spa-temporal mean by given kernel function.
        DataObj (bool): GUI for STAR-BME.

    Returns:
        grid_trend_est (ndarray): shape of (k, j). 
            space-time interpolation on given coordinates and time.

    Examples:
        Randomly sampling 30 points in 20000x20000 fields and suppose there are 24 measurement events.
        >>> grid_s = np.random.random_sample((30,2))*20000
        >>> grid_t = np.arange(1,25)

        Randomly sampling 36 measuring events of each point.
        >>> grid_z = np.random.random_sample((30,24))+np.random.random_sample((30,1))*10
        
        Define the smoothing parameters of kernel.
        >>> dNeib,tNeib,ar,at = 5000,5,5000,5
        >>> pars = [dNeib,tNeib,ar,at]
        
        Creating the coordinate and time that used to interpolate.
        >>> xx,yy = np.mgrid[0:20000:50j, 0:20000:50j]
        >>> est_grid_s = np.vstack((xx.ravel(),yy.ravel())).T
        >>> est_grid_t = grid_t

        Interpolating spa-temporal trend on given coordinates and time.
        >>> grid_trend_est = stmeaninterp(grid_s, grid_t, grid_z, est_grid_s, est_grid_t,
                             method='idw', pars=pars, smooth=True, DataObj = None)
    """

#    if not DataObj:
#        from nousedataobj import NoUseDataObj
#        DataObj = NoUseDataObj()
    if DataObj:        
        title = DataObj.getProgressText()
        DataObj.setProgressRange(0,len(grid_s))
        DataObj.setCurrentProgress(0, title + "\n- By STMean...")    
        
    # grid_t can be np.datetime or other datetime formats
    # Create the ordinal values for grid_t(tME) for the following operations 
    # by H-L Yu   
        
    # TO-DO add try except for error capture from the wrong input format  
    
    grid_t=np.reshape(grid_t,(1,grid_t.size))  
    grid_t=grid_t.astype(np.float64)
    est_grid_t=np.reshape(est_grid_t,(1,est_grid_t.size))
    est_grid_t=est_grid_t.astype(np.float64) 
    
    ms,mss,mt,mts,STmean = stmean(grid_s,grid_t,grid_z,pars)
    if smooth:
        mean_s=mss
        mean_t=mts
    else:
        mean_s=ms
        mean_t=mt
    
    mask_grid_z = np.ma.masked_array(grid_z,np.isnan(grid_z))
#  mean_s = np.array( mask_grid_z.mean( axis = 1 ) , ndmin = 2).T
#  mean_t = np.array( mask_grid_z.mean( axis = 0 ) , ndmin = 2)
    mean_st = mask_grid_z.mean()
        
    if method=='linear':
        mean_s_est = griddata(grid_s, mean_s,est_grid_s)
        mean_s_est[np.isnan(mean_s_est)]=np.mean(mean_s) 
        #Set the nan values to average of mean_s
    elif method=='nearest':
        mean_s_est = griddata(grid_s, mean_s,est_grid_s,method='nearest')
    elif method=='cubic':
        mean_s_est = griddata(grid_s, mean_s,est_grid_s,method='cubic')
    elif method=='idw':
        temp_x,temp_y = list(map(np.array,zip(*grid_s)))
        temp_x_est, temp_y_est = list(map(np.array,zip(*est_grid_s)))
        mean_s_est = idw.idw_est( temp_x, temp_y, mean_s.T[0], temp_x_est, temp_y_est, power = 2 )
    
    mean_s_est=mean_s_est.ravel()  
#    mean_s_est = pylab_griddata(temp_x,temp_y,mean_s.T[0],
#                                temp_x_est,temp_y_est,interp = 'nn')
        #temp_x_est and temp_y_est is not mono increasing
        #i don't know whether there is a bug in it
#    
#    mean_s_est = mean_s_est.diagonal()
    mean_s_est = np.array( mean_s_est,ndmin=2 ).T
    # enforce the temporally linear interpolation
    est_tmin=np.min(np.hstack([est_grid_t[0],grid_t[0]]))
    est_tmax=np.max(np.hstack([est_grid_t[0],grid_t[0]]))
    data_tgrid=np.hstack([est_tmin,grid_t[0],est_tmax])
    data_tmean=np.hstack([mean_t[0][0],mean_t[0],mean_t[0][-1]])
    mean_t_est = np.array(np.interp(est_grid_t[0],data_tgrid,data_tmean),ndmin=2)
        
        #est_z_2d = pylab_griddata(pylab_x, pylab_y, point_value,est_x_2d, est_y_2d,interp = 'nn')
                
    grid_trend_est = mean_s_est + mean_t_est - mean_st
        
    return grid_trend_est
        



if __name__ == "__main__":
        grid_s=np.array([[1,3.],[1,8],[4,1],[3,2]])
        grid_t=np.array([[1,3,5,7,9.]])
        grid_z = np.array([[1,np.nan,3.,4,5],
                                                    [5,6,1,7,8],
                                                    [1,np.nan,4,2,5],
                                                    [5,2,6,3,1.]])
        
        grid_s_est=grid_s#np.array([[1,6.],[1,6]])[::-1]
        grid_t_est=grid_t#np.array([[1]])

        # ms,mss,mt,mts = stmean(grid_s,grid_t,grid_z)
        # print ms
        # print mss
        # print mt
        # print mts
        # grid_trend = stmeaninterp(grid_s,grid_t,grid_z,grid_s_est,grid_t_est)
        # print grid_trend
