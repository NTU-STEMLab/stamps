# -*- coding: utf-8 -*-
# @Author: Chieh-Han Lee
# @Date:   2017-02-21 15:22:10
# @Last Modified by:   Chieh-Han Lee
# @Last Modified time: 2017-02-26 22:13:39

import numpy


def itranspose(b, dim_order):

    '''
    Inverse numpy transpose - Rearrange the dimsions of b so that 
                              numpy.transpose(a, dim_order) will produce
                              b.

    Input:
    b             ndarray   
    dim_order     list

    Return:
    a             ndarray
    '''
    
    inverseorder = numpy.zeros(len(dim_order)).astype(int)
    inverseorder[dim_order] = range(len(dim_order))
    a = numpy.transpose(b, inverseorder)

    return a

def unfolding(A, ndim):

    '''
    Matirx unfoldings -  Unfolding a given tensor

    Input:
    A           ndarray         nth-order tensor with shape (I_1, I_2,...,I_n)
    ndim        integer         aixs along which to perform unfolding

    Output:
    M           ndarray         unfoled tensor as a matrix with shape
                                ( I_ndim, I_(ndim+1)*I_(ndim+2)*...*I_n*I_1*I_2*...*I_(n-1) )
    
    '''

    s = A.shape
    n = A.ndim

    A_dim_order = range(n)
    
    M_dim_order = list(A_dim_order)
    M_dim_order.insert(0, M_dim_order.pop(ndim-1))
    rdims = [M_dim_order[0]]
    cdims = M_dim_order[1:]
    
    # reshape in NumPy defaults to the ‘C’ order, whereas Matlab uses the Fortran order
    M = numpy.reshape(numpy.transpose(A, M_dim_order),
                     (s[rdims[0]], numpy.prod([s[i] for i in cdims])), 'F')

    return M


def folding(M, ndim, s):

    '''
    Matrix foldings - Folding a given matrix
    
    Inputs:
    M
    ndim
    s

    Return:

    '''
    n = len(s)

    A_dim_order = range(n)
    M_dim_order = list(A_dim_order)
    M_dim_order.insert(0, M_dim_order.pop(ndim-1))
    rdims = [M_dim_order[0]]
    cdims = M_dim_order[1:]

    A = numpy.reshape(M, 
        numpy.concatenate([[s[rdims[0]]], [s[i] for i in cdims]]).ravel(), 'F')

    if len(M_dim_order) >= 2:
        A = itranspose(A, M_dim_order)

    return A


def ttm(A, U, transpose=False, ndim=None):

    '''
    Tensor times matrix - Computes the n-mode of a tensor T by a matrix U

    Input:
    A
    U
    transpose
    ndim
    
    Return:

    '''
    
    N = A.ndim
    s = A.shape
    if isinstance(U, list):
        nu = list(enumerate(U))
    else:
        nu = [(ndim-1, U)]
    
    for n, u in nu:
        
        dim_order = numpy.concatenate([[n], range(n), 
                    range(n+1, N)]).ravel().astype(int)
        
        if n == 0:
            new_A = numpy.transpose(A, dim_order)
            new_A = numpy.reshape( new_A, (s[n], A.size/s[n]), 'F' )
        else:
            new_A = numpy.transpose(Y, dim_order)
            new_A = numpy.reshape( new_A, (s[n], Y.size/s[n]), 'F')

        if transpose:
            new_A = numpy.dot(u.T, new_A)
            r = u.shape[1]
        else:
            new_A = numpy.dot(u, new_A)
            r = u.shape[0]

        new_s = numpy.concatenate([(r,), s[:n], s[n+1:N]]).ravel().astype(int)
        
        Y = numpy.reshape( new_A, new_s, 'F' )
        Y = itranspose(Y, dim_order)
        
    return Y

if __name__ == '__main__':

    A = numpy.zeros((2,2,2))
    A[:,:,0] = numpy.asmatrix([[1,2],
                               [3,4]]).transpose()
    A[:,:,1] = numpy.asmatrix([[5,6],
                               [7,8]]).transpose()

    mode = 1
    a = unfolding(A, mode)

    A = folding(a, mode, A.shape)

    X = ttm(A, a, True, mode)
    
        



