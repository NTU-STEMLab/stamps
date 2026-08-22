# -*- coding: utf-8 -*-
import numpy as np
import six


def get_standard_soft_pdf_type(pdf_type):
    '''A function to define the softdata type in BME functions.

    softpdf types can be specified by string or integer respectively:

    |softpdftype   |   PDF types           |  associated parameters   |
    |--------------| ----------------------|--------------------------|
    |1             |   histogram           |  (nl, limiB, probadens)  |
    |2             |   linear              |  (nl, limi, probadens)   |
    |10            |   Gaussian or normal  |  (mean, var)             |
    
    More details: the way to define the flexible soft information follows 
    the definition of BMElib in Matlab as follows
    For a case of uniform distribution. For the histogram and linear-based 
    softdata can be expressed as below:
    
    - Histogram: softpdftype=1;nl=2;limi=[0,1];  probdens=[1]
    - Linear: softpdftype=2;nl=2;limi=[0,1];  probdens=[1,1]

    Args:
        pdf_type (Union[str, int]): {histogram, linear, gaussian, normal, 1, 2, 10}. The softdata type description.

    Returns:
        (int): {1, 2, 10}. The softdata type represent in integer.

    Examples:
        >>> get_standard_soft_pdf_type('histogram')
        1
        >>> get_standard_soft_pdf_type('gaussian')
        10
        >>> get_standard_soft_pdf_type(10)
        10
    '''
    if isinstance(pdf_type, six.string_types):
        if 'histogram'.startswith(pdf_type.lower()):
            return 1
        elif 'linear'.startswith(pdf_type.lower()):
            return 2
        elif 'gaussian'.startswith(pdf_type.lower()) or\
            'normal'.startswith(pdf_type.lower()):
            return 10
        else:
            raise ValueError('No supported pdftype found')
    else:
        if int(pdf_type) in [1]:
            return 1
        elif int(pdf_type) in [2]:
            return 2
        elif int(pdf_type) in [10]:
            return 10

def get_standard_order(order):
    '''A function to obtain the standard format of order parameter.

    The order parameter determines the trend form in bme estimation.
    
    Args:
        order (Union[str, int]): 0 for constant mean and np.nan for the zero mean,
            'zero mean' or 'constant mean' that explicitly describe the trend form.

    Returns:
        (Union[int, np.nan]): {0, np.nan}. 0 for constant mean and np.nan for the zero mean.

    Examples:
        >>> get_standard_order('zero mean')
        nan
        >>> get_standard_order('constant mean')
        0
        >>> get_standard_order(0)
        0
    
    Note: Consider Overparameterization...
        For S/T estimation, only zero and constant mean are supported for now 
        to avoid the overparameterization of the trend modeling                                        
    
    '''
    ## For the string part, the string should be transformed captions into letters    
    ## before the comparison
    if isinstance(order, six.string_types): #base string
        if order.lower() == 'zero mean':
            return np.nan
        elif order.lower() == 'constant mean':
            return 0
        else:
            raise ValueError('No supported order (string) found')
    elif isinstance(order, np.ndarray):
        return order
    elif isinstance(order, int) or np.isnan(order): #int or np.nan
        if np.isnan(order):
            return order
        elif order in  [0, 0.0]:
            return 0
        else:
            raise ValueError('No supported order (number) found')
    else:
        print ('warning: "order" is not str or int, just return without modifing.')
        return order

if __name__ == "__main__":
    print (get_standard_soft_pdf_type(1))
    print (get_standard_soft_pdf_type('hist'))
    print (get_standard_soft_pdf_type('h'))
    print (get_standard_soft_pdf_type('H'))
    print (get_standard_soft_pdf_type(1))
    print (get_standard_soft_pdf_type('li'))
    print (get_standard_soft_pdf_type('l'))
    print (get_standard_soft_pdf_type('LineAr'))

    print (get_standard_order('Zero Mean'))
