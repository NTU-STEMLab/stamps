# -*- coding: utf-8 -*-
import os
import sys
import numpy
try:
    import nlopt
except ImportError as e:
    print ('Cannot import modeul "nlopt", try to install "nlopt" first.')
    raise e



def bobyqa(obj_func, init_guess, args, low_bnd, up_bnd,
    stop_val=None, maxeval=3000,
    ftol_rel=None, ftol_abs=None,
    xtol_rel=None, xtol_abs=None):
    '''
    Optimization by bobyqa method
    
    Function:
        bobyqa(obj_func, init_guess, args, low_bnd, up_bnd, maxeval)
    
    Input: 
    obj_func      function
        objective function to be minimized
    init_guess    array
        a 1D numpy array for the parameters to be estimated
    args          array
        a 1D numpy array for the function's arguments
    low_bnd       array
        optional. a 1D numpy array for parameter's lower bound
    up_bnd        array
        optional. a 1D numpy array for parameter's upper bound
    maxeval       int
        optional. a integer for maximum evaluation
    '''

    opt = nlopt.opt(nlopt.LN_BOBYQA, len(init_guess))
    opt.set_min_objective(
        lambda x, grad = numpy.array([]): obj_func(x, *args)
        )
    if not (low_bnd is None):
        opt.set_lower_bounds(low_bnd)
    if not (up_bnd is None):  
        opt.set_upper_bounds(up_bnd)
    if stop_val:
        opt.set_stopval(stop_val)
    opt.set_maxeval(maxeval)
    if ftol_rel:
        opt.set_ftol_rel(ftol_rel)
    if ftol_abs:
        opt.set_ftol_abs(ftol_abs)
    if xtol_rel:
        opt.set_xtol_rel(xtol_rel)
    if xtol_abs:
        opt.set_xtol_abs(xtol_abs)
    try:
        result = opt.optimize(init_guess)
    except nlopt.RoundoffLimited:
        print ('starpy Warning: roundoff limited found. Set relative value to 10**-8')
        opt.set_ftol_rel(10**-8)
        result = opt.optimize(init_guess)
    opt_val = opt.last_optimum_value()
    return result, opt_val
