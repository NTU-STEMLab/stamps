# -*- coding: utf-8 -*-
import numpy
from scipy.spatial.distance import cdist


def findpairs(c1, c2):
  '''
  Find pairs of coordinates that are identical for two diferent
  matrices of coordinates.

  SYNTAX :

  [index]=findpairs(c1,c2);

  INPUT :

  c1        n1 by d  numpy matrix of coordinates, where d is the dimension
                     of the space.
  c2        n2 by d  numpy matrix of coordinates.

  OUTPUT :

  index     n by 2   numpy matrix of indices for identical coordinates
                     in c1 and c2. The first column of index 
                     refers to the c1 matrix, whereas the second
                     column of index refers to the c2 matrix.

  '''
  
  try:
    idx=findpairs_1(c1,c2)
  except:
    idx=findpairs_2(c1,c2)
  return idx  

def findpairs_1( c1, c2):
  idx=numpy.where(cdist(c1,c2)==0.0)
  # zip() returns a lazy iterator in Python 3; convert to ndarray immediately.
  pairs = list(zip(idx[0], idx[1]))
  if pairs:
      pair_idx_list = numpy.array(pairs)
  else:
      pair_idx_list = numpy.empty((0, 2), dtype=int)
  return pair_idx_list

def findpairs_2( c1, c2 ):
    '''return index pair'''

    if c1.shape[0] <= c2.shape[0] :
        smaller_c = c1
        bigger_c = c2
        smaller_is_c1 = True
    else:
        smaller_c = c2
        bigger_c = c1
        smaller_is_c1 = False

    pair_idx_list = []
    for small_idx, row_i in enumerate( smaller_c ):
        big_idx = numpy.where( numpy.all( row_i == bigger_c, axis = 1 ) )[0]
        if len(big_idx):
            pair_idx_list.append( [ small_idx, big_idx[0] ] )

    if pair_idx_list:
        pair_idx_list = numpy.array( pair_idx_list )
        if smaller_is_c1:
            return pair_idx_list
        else:
            pair_idx_list[ :, [ 0, 1 ] ] =  pair_idx_list[ :, [ 1, 0 ] ]
            return pair_idx_list
    else:
        return pair_idx_list

if __name__ == "__main__":
    import time
    c1 = numpy.array([[1,1,1],[1,2,1],[1,1,2.],[2,2,1]])
    c2 = numpy.array([[1,2,5],[2,2,2.],[0,0,1.],[1,0,0],[2,1,2],[2,2,2]])
    res=findpairs( c1, c2 )
    print (res)
    c1=numpy.random.rand(1000,3)
    c2=numpy.random.rand(1000,3)
    aaa = time.time()
    result1 = findpairs(c1,c2)
    print ('neighbor_kd Time: ', time.time() - aaa)
    aaa = time.time()
    result2 = findpairs_1(c1,c2)
    print ('neighbor_kd Time: ', time.time() - aaa)