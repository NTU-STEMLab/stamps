# -*- coding: utf-8 -*-
import numpy as np


def isspacetime(models):
    '''
    analyze nest covariance model expression

    syntax:
        isST, isSTsep, model_res = isspacetime(model)

    input:
        model: a list of covariance model expression

    return:
        isST:    bool, whether model is a space-time model
        isSTsep: bool, whether model is a space-time model 
                       which separate space and time part
        model_res: a list of analyzed covariance model result.
            if isSTsep is True,
                model_res[0] is a list of space covariance model
                model_res[1] is a list of time covariance model
            if isSTsep is False,
                model_res[0] is a list of space|time|space-time
                covarinace model

    note: there is no spelling check with model name
    '''

    isST, isSTsep, model_res =\
        zip(*[parse_model(m) for m in models])

    if not (all(isSTsep) or (not any(isSTsep))):
        raise ValueError(
            'All space-time models must be jointly '
            'either separable or non-separable.')
    if not (all(isST) or (not any(isST))):
        raise ValueError(
            'All models must be jointly either space, time or space-time.')

    return isST[0], isSTsep[0], list(zip(*model_res))

def parse_model(model):
    '''
    analyze single covariance model expression

    syntax:
        isST, isSTsep, model_res = isspacetime(model)

    input:
        model: a covariance model expression

    return:
        isST:    bool, whether model is a space-time model
        isSTsep: bool, whether model is a space-time model 
                       which separate space and time part
        model_res: a list of analyzed covariance model result.
            if isSTsep is True,
                model_res[0] is a space covariance model
                model_res[1] is a time covariance model
            if isSTsep is False,
                model_res[0] is a space|time|space-time
                covarinace model

    note: there is no spelling check with model name
    '''
    if u'/' in model:
        isSTsep = True
        isST = True
        if any(map(lambda x:x.endswith(u'ST'), model.split(u'/'))):
            raise ValueError(
                'Separable model cannot contain another non-separable model.')
    else:
        isSTsep = False
        if model.endswith(u'ST'):
            isST = True
        else:
            isST = False
    if isSTsep:
        return isST, isSTsep, model.split(u'/') # a list
    else:
        # just return not separate model anyway
        return isST, isSTsep, [model]
