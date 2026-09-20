# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 08:48:06 2015

@author: hdragon689
"""
import numpy as np
import shapefile as shp
import matplotlib.pyplot as plt
#  import matplotlib.patches as patches
from matplotlib.patches import Polygon
#  from matplotlib.collections import PatchCollection


def polygonbkplot(shpdata,ax=None,maxx=None,minx=None,maxy=None,miny=None):
  '''
  Plot polygon shapefile without filling and can be used as a background of 
  other maps
  
  Syntax: 
    polygonbkplot(shpdata,ax=None,maxx=None,minx=None,maxy=None,miny=None)
    
  Input: 
    shpfile   string      the path of shapefile
    ax        ax          optional. the axes object of the map to be plotted. 
                          Details of the axes can refer to matplotlib package
    maxx      scalar      optional. maxx, minx, maxy, miny are coordinates of 
    minx                  longitude and latitude, respectively, that define the  
    maxy                  rectangular boundary of the map in the ax
    miny                      
                          
  '''  
  
  sf = shp.Reader(shpdata)
  recs    = sf.records()
  shapes  = sf.shapes()
  Nshp    = len(shapes)
  # cns     = []
  # for nshp in xrange(Nshp):
  #   cns.append(recs[nshp][1])

  # cns = np.array(cns)
#  cm    = plt.get_cmap('Dark2')
#  cccol = cm(1.*np.arange(Nshp)/Nshp)
  if ax is None:
    ax=plt.figure().add_subplot(111)

  for nshp in range(Nshp):
#    ptchs   = []
    pts     = np.array(shapes[nshp].points)
    prt     = shapes[nshp].parts
    par     = list(prt) + [pts.shape[0]]
    for pij in range(len(prt)):
      ax.add_patch(Polygon(pts[par[pij]:par[pij+1]],fill=False,closed=True))     
  
  if [maxx,minx,maxy,miny]!=[None]*4:    
    ax.set_xlim(minx,maxx)
    ax.set_ylim(miny,maxy) 