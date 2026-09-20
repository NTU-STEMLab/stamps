# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 23:53:36 2015

@author: hdragon689
"""
import matplotlib.pyplot as plt

from ..bme.softconverter import proba2probdens

def softpdfplot(limi, proba, softpdftype=2, nl=None):
    '''
    argument format can refer to the function of
    starpy.bme.softconverter.proba2probdens
    '''
    if nl is None:
      nl = len(limi)    
    
    probdens = proba2probdens(softpdftype, nl, limi, proba)  
    plt.plot(limi, probdens)
    plt.show()
  
def mepdfplot(mepdf, npar, dim):
    '''
    Input:
    mepdf    func
    '''
