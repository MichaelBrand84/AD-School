################################################################################
#                                                                              #
#	Michael Brand                                                              #    
#	Math module for arguments of type FloatAad                                 #
#	PYTHON 3.8.2                                                               #
#                                                                              #
################################################################################

import numpy as np
from floataad import FloatAad

@np.vectorize
def sqrt(x):
    newValue = np.sqrt(x.value)
    newDerivative = (
        (x, 1. / (2 * np.sqrt(x.value))),
    )
    return FloatAad(newValue, newDerivative)

@np.vectorize
def exp(x):
    newValue = np.exp(x.value)
    newDerivative = (
        (x, newValue),
    )
    return FloatAad(newValue, newDerivative)

@np.vectorize
def log(x):
    newValue = np.log(x.value)
    newDerivative = (
        (x, 1. / x.value),
    )
    return FloatAad(newValue, newDerivative)

@np.vectorize
def sin(x):
    newValue = np.sin(x.value)
    newDerivative = (
        (x, np.cos(x.value)),
    )
    return FloatAad(newValue, newDerivative)

@np.vectorize
def cos(x):
    newValue = np.cos(x.value)
    newDerivative = (
        (x, -np.sin(x.value)),
    )
    return FloatAad(newValue, newDerivative)

@np.vectorize
def tan(x):
    return sin(x) / cos(x)