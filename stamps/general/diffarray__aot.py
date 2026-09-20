# -*- coding: utf-8 -*-
import os

from six.moves import range
import numpy as np
from numba.pycc import CC



MODULE_NAME = os.path.split(__file__)[1][:-8] # file name without '__aot.py'

cc = CC(MODULE_NAME)

@cc.export(
    'diffarray', 'Tuple((i8[:],i8[:],f8[:,:],f8[:]))(f8[:,:],f8,b1,b1)'
    )
def diffarray(x, r, include, aniso):
    '''
    Accelerate function "diffarray" with numba

    r == -1.0 means all range to be considered
    '''

    row_count, col_count = x.shape
    d = 0 if include == True else 1

    k=0
    row_count_output = (row_count - d) * (row_count - d + 1) / 2
    index_head = np.empty(row_count_output, dtype = np.int64)
    index_tail = np.empty(row_count_output, dtype = np.int64)
    diff = np.empty((row_count_output, col_count))
    angdiff = np.empty(row_count_output)

    for i in range(row_count - d):
        index_head[k : k+row_count-i-d] = i
        index_tail[k : k+row_count-i-d] = np.arange(row_count)[i+d:]
        diff[k : k+row_count-i-d] = x[i]-x[np.arange(row_count)[i+d:]]
        k = k + row_count - i - d

    if aniso == True:
        angdiff[:] = np.arctan2(diff[:,1], diff[:,0])

    if r != -1.0:
        idxx = np.where(np.sqrt(diff[:,0]**2. + diff[:,1]**2.) <= r)
        index_head = index_head[idxx]
        index_tail = index_tail[idxx]
        diff = diff[idxx]
        if aniso == True:
            angdiff = angdiff[idxx]

    return index_head, index_tail, diff, angdiff

@cc.export(
    'cdist',
    'Tuple((i8[:],i8[:],f8[:,:],f8[:]))(f8[:,:],f8[:,:],f8,b1)'
    )
def cdist(x, y, r, aniso):
    x_row, x_col = x.shape
    y_row, y_col = y.shape
    if x_col != y_col:
        raise ValueError(
            'two array must have the same column length')
    if x_col != 2 and aniso == True:
        raise ValueError(
            'anisotropy is only suit for column length equal to 2')

    index_head = np.empty(x_row * y_row, dtype=np.int64)
    index_tail = np.empty(x_row * y_row, dtype=np.int64)
    for i in range(x_row):
        index_head[i*y_row:(i+1)*y_row] = i
        index_tail[i*y_row:(i+1)*y_row] = np.arange(y_row)

    diff = np.empty((x_row*y_row, x_col))
    for i in range(x_col):
        diff[:,i] = (x[:,i:i+1] - y[:,i]).ravel()
    if aniso == True:
        angdiff = np.arctan2(diff[:,1], diff[:,0]).ravel()
    else:
        angdiff = np.empty(x_row*y_row)

    if r != -1.0:
        dis_arr = np.zeros(x_row*y_row)
        for i in range(x_col):
            dis_arr = dis_arr+diff[:,i]**2

        idxx = np.where(np.sqrt(dis_arr) <= r)
        index_head = index_head[idxx]
        index_tail = index_tail[idxx]
        diff = diff[idxx]
        if aniso == True:
            angdiff = angdiff[idxx]

    return index_head.ravel(), index_tail.ravel(), diff, angdiff.ravel()


if __name__ == "__main__":
    cc.compile()