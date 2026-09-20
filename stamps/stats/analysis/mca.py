# -*- coding: utf-8 -*-
# @Author: Chieh-Han Lee
# @Date:   2017-02-27 00:02:51
# @Last Modified by:   Chieh-Han Lee
# @Last Modified time: 2017-03-22 13:50:40

# import matplotlib.pyplot as plt
import numpy
import scipy.sparse.linalg as ssl
# import seaborn as sns

from scipy import linalg


def mca(X, Y, m='all',centered=True, normalized=True, norm_direc=0):

    '''
    Maximum Covariance Analysis - 

    Input:
    X       M by N
    Y       L by N

    '''

    if X.shape[1] == Y.shape[1]:
        N = X.shape[1]

    if centered:
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

    if normalized:
        Xnorms = numpy.std(X, axis=norm_direc)
        X = numpy.dot(X, numpy.diag(1/Xnorms))
        Ynorms = numpy.std(Y, axis=norm_direc)
        Y = numpy.dot(Y, numpy.diag(1/Ynorms))

    Cxy = (1./N)*numpy.dot(X, Y.T)

    ss = min(Cxy.shape)

    if m == 'all' or m >=ss:
        U, S, Vh = linalg.svd(Cxy)
        V = Vh.T
    else:
        U, S, Vh = ssl.svds(Cxy, m)
        U = U[:,::-1]
        S = S[::-1]
        Vh = Vh[::-1,:]
        V = Vh.T

    ECU = numpy.dot(U.T, X)
    ECV = numpy.dot(Vh, Y)

    return U, V, ECU, ECV, S


def smca(X, Y, m='all', centered=True, normalized=True, norm_direc=0, regress='heterogeneous'):

    '''
    Scaled MCA - 
    '''

    if X.shape[1] == Y.shape[1]:
        N = X.shape[1]

    U, V, ECU, ECV, S = mca(X, Y, m, centered, normalized, norm_direc)

    if m == 'all':
        m = len(S)

    U = []
    V = []
    if regress == 'heterogeneous':
    # Heterogeneous regression maps
        for k in range(m):
            uk = (S[k]*1./N)*numpy.dot(X, ECV[k,:].T)
            U.append(uk)
            vk = (S[k]*1./N)*numpy.dot(Y, ECU[k,:].T)
            V.append(vk)
    elif regress == 'homogeneous':
    # Homogeneous regression maps
        for k in range(m):
            uk = (S[k]*1./N)*numpy.dot(X, ECU[k,:].T)
            U.append(uk)
            vk = (S[k]*1./N)*numpy.dot(Y, ECV[k,:].T)
            V.append(vk)

    U = numpy.asarray(U)
    ECU = numpy.dot(U.T, X)
    V = numpy.asarray(V)
    ECV = numpy.dot(V.T, Y)

    return U, V, ECU, ECV, S


# def mcaplot(U, V, ECU, ECV, S):

#     ax = sns.heatmap()



def RMSC():

    '''
    Normalized Root Mean Squared Covariance
    '''
    pass


if __name__ == '__main__':
    
    X = numpy.random.random((10,5))
    Y = numpy.random.random((30,5))

    U, V = smca(X, Y, regress='homogeneous')