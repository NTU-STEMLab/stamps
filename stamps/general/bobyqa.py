# -*- coding: utf-8 -*-
import warnings
import numpy
try:
    import nlopt
except ImportError as e:
    print ('Cannot import module "nlopt", try to install "nlopt" first.')
    raise e


class StampsOptimizationWarning(UserWarning):
    """The optimiser stopped because floating-point precision limited progress.

    The returned parameters are normally converged; silence it with
    ``warnings.filterwarnings('ignore', category=StampsOptimizationWarning)``.
    """


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

    Output:
    result        array   the best parameters found
    opt_val       float   the objective value at result

    If nlopt stops with RoundoffLimited (no further progress possible at
    floating-point precision) the search is restarted once from the best
    point found so far with ftol_rel=1e-8; if that also stops on roundoff,
    the best point found is returned.  A StampsOptimizationWarning is issued.
    '''
    x0 = numpy.array(init_guess, dtype=float)
    if low_bnd is not None:
        x0 = numpy.maximum(x0, numpy.asarray(low_bnd, dtype=float))   # nlopt rejects a start
    if up_bnd is not None:                                            # outside the bounds
        x0 = numpy.minimum(x0, numpy.asarray(up_bnd, dtype=float))
    best = {'x': x0.copy(), 'f': numpy.inf}

    def _objective(x, grad=numpy.array([])):
        val = obj_func(x, *args)
        fval = float(numpy.asarray(val, dtype=float).ravel()[0])
        if fval < best['f']:
            best['f'] = fval
            best['x'] = numpy.array(x, dtype=float)
        return fval

    opt = nlopt.opt(nlopt.LN_BOBYQA, len(x0))
    opt.set_min_objective(_objective)
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
        result = opt.optimize(x0)
        opt_val = opt.last_optimum_value()
    except nlopt.RoundoffLimited:
        warnings.warn(
            'stamps bobyqa: optimisation stopped at floating-point precision '
            '(nlopt RoundoffLimited); restarted from the best point with '
            'ftol_rel=1e-8. The result is normally converged.',
            StampsOptimizationWarning, stacklevel=2)
        opt.set_ftol_rel(1e-8)
        try:
            result = opt.optimize(best['x'])
            opt_val = opt.last_optimum_value()
        except nlopt.RoundoffLimited:
            result, opt_val = best['x'], best['f']
    return result, opt_val
