import numpy as np

def getK(Pp, H, R=None):
    '''get Kalman gain'''
    return Pp.dot(H.T).dot(np.linalg.pinv(H.dot(Pp).dot(H.T)+R))

def getXu(Xp, K, D, H):
    '''get updated state estimation'''
    return Xp + K.dot(D - H.dot(Xp))

def getPu(K, H, Pp):
    '''get updated covariance estimation'''
    KH = K.dot(H)
    return (np.eye(KH.shape[0]) - KH).dot(Pp)
