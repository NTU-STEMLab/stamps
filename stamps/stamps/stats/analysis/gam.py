import pandas as pd
try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
except ImportError as e:
    raise ImportError(
        'stamps: rpy2 is required for GAM functions. '
        'Install it with:  pip install rpy2'
    ) from e

rmgcv= importr('mgcv')

from .dlnm import gam as dlnm_gam


def construct_gam_formula(y=None, x=None, s_x=None, f_x=None, by_str=None):
    '''
    construct r's gam formula

    y: str, response column name
    x: a list of linear function column name
    s_x: a list of smoothing function column name
    f_x: a list of factirial function column name
    by_str: if exists, ignore all variable, create formula by by_str

    return r's gam formula
    '''
    if by_str:
        fmla = ro.Formula(by_str)
    else:
        x_list = []

        if x:
            x_list += x
        if s_x:
            x_list += ['s({i})'.format(i=i) for i in s_x]
        if f_x:
            x_list += ['factor({i})'.format(i=i) for i in f_x]

        x_valid = [i for i in x_list if i]
        if len(x_valid) > 0:
            fmla_str = '{y} ~ '.format(y=y) + ' + '.join(x_valid)
        else:
            raise ValueError('No valid x found.')

        # if not x:
        #     pass
        # else:
        #     pass
        # if not s_x:
        #     s_str = ''
        # else:
        #     s_str = ' + '.join(['s('+i+', k=6)' for i in s_x])
        # if not f_x:
        #     f_str = ''
        # else:
        #     f_str = ' + '.join(['factor('+i+')' for i in f_x])

        # if s_str and f_str:
        #     fmla_str = '{y} ~ {s} + {f}'.format(y=y, s=s_str, f=f_str)
        # elif s_str:
        #     fmla_str = '{y} ~ {s}'.format(y=y, s=s_str)
        # else:
        #     fmla_str = '{y} ~ {f}'.format(y=y, f=f_str)
        fmla = ro.Formula(fmla_str)
    return fmla

def construct_gam_model(gam_formula, data, drop_intercept=None, control=None):
    '''
    gam_formula: r's gam formula (from construct_gam_formula)
    data: pandas dataframe, columns name should fit gam_formula
    control: None or a python dict where it's key is string, (for r list

    return r's gam model
    '''

    if isinstance(control, dict):
        ctl_str = "list("
        kvpair = []
        for k in control:
            if isinstance(control[k], bool):
                kvpair.append('='.join([k, str(control[k]).upper()]))
            else:
                kvpair.append('='.join([k, str(control[k])]))
        ctl_str += ','.join(kvpair)
        ctl_str += ')'
        control = ro.r(ctl_str)


    if drop_intercept:
        model = dlnm_gam(gam_formula, data=data,
            drop_intercept=drop_intercept, control=control)
    else:
        model = dlnm_gam(gam_formula, data=data, control=control)
    return model

def predict_gam(gam_model, data):
    '''
    gam_model: r's gam model (from construct_gam_model)
    data: pandas dataframe, columns name should fit gam_formula

    return gam predict result
    '''
    def _predict_gam(
        obj, newdata, ptype = "link", se_fit = False, terms = ro.NULL,
        block_size = ro.NULL, newdata_gurrantedd = ro.NULL,
        na_action = ro.r('na.pass'), cluster = ro.NULL):
    
        result = rmgcv.predict_gam(obj, newdata, type=ptype, se_fit=se_fit,
                                   terms=terms, block_size=block_size,
                                   newdata_gurrantedd=newdata_gurrantedd,
                                   na_action=na_action, cluster=cluster)
        return result
    value = _predict_gam(gam_model, data)
    return pandas2ri.ri2py(value)

def gam(df, s_x, f_x, y, est_df=None, est_s_x=None, est_f_x=None, est_y=None):
    '''use gam pythonically

    input:
    df: pandas dataframe
    s_x: sequences of non-linear relation column name
    f_x: sequences of factirial relation column name
    y: str, response column name
    est_*: like above, for prediction

    note:
        if est_df is None, gam will fit df data

    return:
    est_df*: est_df with addtional column y(named df's y's column name)
    '''

    r_gam_formula = construct_gam_formula(y, s_x=s_x, f_x=f_x)
    r_gam_model = construct_gam_model(r_gam_formula, data=df)

    if not est_y:
        est_y = y

    if est_df is None:
        est_df = df.copy()
        while est_y in est_df.columns:
            est_y += '_'
        est_df[est_y] = pandas2ri.ri2py(r_gam_model.rx2('fitted.values'))
    else:
        if est_s_x:
            est_df = est_df.rename(
                columns=dict(i for i in zip(est_s_x, s_x)))
        else:
            est_s_x = s_x
        if est_f_x:
            est_df = est_df.rename(
                columns=dict(i for i in zip(est_f_x, f_x)))
        else:
            est_f_x = f_x
        while est_y in est_df.columns:
            est_y += '_' 
        est_df[est_y] = predict_gam(r_gam_model, data=est_df)
        if est_s_x != s_x:
            est_df = est_df.rename(
                columns=dict(i for i in zip(s_x, est_s_x)))
        if est_f_x != f_x:
            est_df = est_df.rename(
                columns=dict(i for i in zip(f_x, est_f_x)))
    return est_df

