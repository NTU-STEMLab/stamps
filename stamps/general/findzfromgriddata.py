# -*- coding: utf-8 -*-
'''
Created on 2011/11/12

@author: ksj
'''
import numpy


def findzfromgriddata(x, y, t, s_grid, t_grid, z_grid, DataObj = None):
    '''
    x: raw by 1 numpy array
    y: ditto
    t: ditto
    s_grid: raw by 2 numpy array
    t_grid: 1 by col numpy array
    z_grid: raw by col numpy array
    DataObj: for GUI use
    
    return z, raw by 1 numpy array,or return False if fail
    '''
    
    if not DataObj:
        from nousedataobj import NoUseDataObj
        DataObj = NoUseDataObj()
        
    title = DataObj.getProgressText()
    
    z = numpy.empty(x.shape)
    z[:] = numpy.nan
    
    DataObj.setProgressRange(0,len(s_grid))
    DataObj.setCurrentProgress(0, title + "\n- Find GridZ to RawZ...")
    
    for index_s, xy in enumerate(s_grid):
        if not DataObj.wasProgressCanceled():
            index_x = numpy.where(x == xy[0])[0]
            yyyy = y[index_x]
            index_y = numpy.where(yyyy == xy[1])[0]
            select_index = index_x[index_y]
            
            for index_tt, tt in enumerate(t_grid[0]):
                if numpy.isnan(z_grid[index_s, index_tt]): # no value
                    pass
                else:
                    tttt=t[select_index]
                    index_t = numpy.where(tttt == tt)[0]
                    last_index = index_t
                    
                    if len(last_index) == 0:
                        pass # not find
                    elif len(last_index) == 1:
                        z[select_index[last_index]] = z_grid[index_s, index_tt]
            DataObj.setCurrentProgress(index_s + 1)
        else:
            return False
    DataObj.setCurrentProgress(text = title)
    return z
    
if __name__ == "__main__":
    x = numpy.array([1,3,1,2,3.,6],ndmin=2).T
    y = numpy.array([2,2,1,2,4.,6],ndmin=2).T
    t = numpy.array([1,1,2,2,2.,1],ndmin=2).T
    s_grid = numpy.array([[1.,1],
                          [1,2],
                          [2,2],
                          [3,2],
                          [3,4]])
    
    t_grid = numpy.array([[1,2.]])
    z_grid = numpy.array([[numpy.nan,72.],
                          [2,numpy.nan],
                          [numpy.nan,2],
                          [55,numpy.nan],
                          [9999,998.]])
    
    z = findzfromgriddata(x, y, t, s_grid, t_grid, z_grid)
    print (numpy.hstack((x,y,t,z)))