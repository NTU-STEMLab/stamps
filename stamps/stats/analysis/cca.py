# -*- coding: utf-8 -*-
# @Author: Chieh-Han Lee
# @Date:   2017-02-27 00:02:58
# @Last Modified by:   Chieh-Han Lee
# @Last Modified time: 2017-03-01 13:29:02

import numpy

from .eof import eof

def cca(X, Y):

    '''
    Conanical Correlation Analysis - 
    '''

    X_L, X_lambdaU2, X_PCs, X_EOFs, X_ECs, X_error, X_norms = eof(X, normalized=False)
    Y_L, Y_lambdaU2, Y_PCs, Y_EOFs, Y_ECs, Y_error, Y_norms = eof(Y, normalized=False)


if __name__ == '__main__':
    
    pass

