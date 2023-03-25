################################################################################
#                                                                              #
#	Michael Brand                                                              #    
#	Augmented Float values for Adjoint Algorithmic Differentiation (AAD)       #
#	PYTHON 3.8.2                                                               #
#                                                                              #
#   Based on code written by Sidney Radcliffe                                  #
#   https://sidsite.com/posts/autodiff/  [last access 25.03.2023]              #
#                                                                              #
################################################################################

from collections import defaultdict
import numpy as np
import math

class FloatAad:

    def __init__(self, value, derivatives = ()):
        self.value = value
        self.derivatives = derivatives

    def __pos__(self):
        return self

    def __neg__(self):
        return neg(self)
        
    """def __add__(self, other):
        if type(other) in [int, float]:
            return add(self, FloatAad(other))
        else:
            return add(self, other)
            
    def __radd__(self, other):
        if type(other) in [int, float]:
            return add(FloatAad(other), self)
        else:
            return add(other, self)
    """
    
    def __add__(self, other):
        if type(other) in [int, float]:
            newValue = self.value + other
            newDerivative = (
                (self, 1),
            )
            return FloatAad(newValue, newDerivative)
        else:
            return add(self, other)
        
    def __radd__(self, other):
        if type(other) in [int, float]:
            newValue = other + self.value
            newDerivative = (
                (self, 1),
            )
            return FloatAad(newValue, newDerivative)
        else:
            return add(other, self)
    
    """def __sub__(self, other):
        if type(other) in [int, float]:
            return add(self, FloatAad(-other))
        else:
            return add(self, neg(other))
    
        
    def __rsub__(self, other):
        if type(other) in [int, float]:
            return add(FloatAad(other), neg(self))
        else:
            return add(other, neg(self))
    """
    
    def __sub__(self, other):
        if type(other) in [int, float]:
            newValue = self.value - other
            newDerivative = (
                (self, 1),
            )
            return FloatAad(newValue, newDerivative)
        else:
            return add(self, neg(other))
        
    def __rsub__(self, other):
        if type(other) in [int, float]:
            newValue = other - self.value
            newDerivative = (
                (self, -1),
            )
            return FloatAad(newValue, newDerivative)
        else:
            return add(other, neg(self)) 

    """def __mul__(self, other):
        if type(other) in [int, float]:
            return mul(self, FloatAad(other))
        else:
            return mul(self, other)
        
    def __rmul__(self, other):
        if type(other) in [int, float]:
            return mul(FloatAad(other), self)
        else:
            return mul(other, self)
    """

    def __mul__(self, other):
        if type(other) in [int, float]:
            newValue = self.value * other
            newDerivative = (
                (self, other), 
            )
            return FloatAad(newValue, newDerivative)
        else:
            return mul(self, other)
        
    def __rmul__(self, other):
        if type(other) in [int, float]:
            newValue = other * self.value
            newDerivative = (
                (self, other), 
            )
            return FloatAad(newValue, newDerivative)
        else:
            return mul(other, self)

    """def __truediv__(self, other):
        if type(other) in [int, float]:
            return mul(self, inv(FloatAad(other)))
        else:
            return mul(self, inv(other))
        
    def __rtruediv__(self, other):
        if type(other) in [int, float]:
            return mul(FloatAad(other), inv(self))
        else:
            return mul(other, inv(self))
    """
    
    def __truediv__(self, other):
        if type(other) in [int, float]:
            newValue = self.value / other
            newDerivative = (
                (self, 1 / other),
            )
            return FloatAad(newValue, newDerivative)
        else:
            return mul(self, inv(other))
        
    def __rtruediv__(self, other):
        if type(other) in [int, float]:
            newValue = other / self.value
            newDerivative = (
                (self, - other / math.pow(self.value,2)),
            )
            return FloatAad(newValue, newDerivative)
        else:
            return mul(other, inv(self))

    """def __pow__(self, other):
        if type(other) in [int, float]:
            return pow(self, FloatAad(other))
        else:
            return pow(self, other)
        
    def __rpow__(self, other):
        if type(other) in [int, float]:
            return pow(FloatAad(other), self)
        else:
            return pow(other, self)
    """

    def __pow__(self, other):
        if type(other) in [int, float]:
            newValue = math.pow(self.value, other)
            newDerivative = (
                (self, other * math.pow(self.value, other - 1)),
            ) 
            return FloatAad(newValue, newDerivative)           
        else:
            return pow(self, other)
    
    def __rpow__(self, other):
        if type(other) in [int, float]:
            newValue = math.pow(other, self.value)
            newDerivative = (
                (self, math.pow(other, self.value) * math.log(other)),
            )
            return FloatAad(newValue, newDerivative)
        else:
            return pow(other, self)


float2FloatAad = np.vectorize(lambda x: FloatAad(x))

def neg(a):
    newValue = -1 * a.value
    newDerivative = (
        (a, -1),
    )
    return FloatAad(newValue, newDerivative)
    
def add(a, b):
    newValue = a.value + b.value
    newDerivative = (
        (a, 1),  # a+b nach a abgeleitet gibt 1
        (b, 1)   # a+b nach b abgeleitet gibt 1
    )
    return FloatAad(newValue, newDerivative)

def mul(a, b):
    newValue = a.value * b.value
    newDerivative = (
        (a, b.value),  # a*b nach a abgeleitet gibt b
        (b, a.value)   # a*b nach b abgeleitet gibt a
    )
    return FloatAad(newValue, newDerivative)

def inv(a):
    newValue = 1. / a.value
    newDerivative = (
        (a, -1. / a.value**2), 
    )
    return FloatAad(newValue, newDerivative)

def pow(a, b):
    newValue = math.pow(a.value,b.value)
    newDerivative = (
        (a, b.value * math.pow(a.value, b.value-1)), # a^b nach a abgeleitet gibt b*a^(b-1)
        (b, math.pow(a.value, b.value) * math.log(a.value))  # a^b nach b abgeleitet gibt a^b * ln(a)
    )
    return FloatAad(newValue, newDerivative)

def getDerivatives(y):
    dy = defaultdict(lambda: 0)

    def computeDerivatives(y, pathValue):
        for node, localDerivative in y.derivatives:
            # Multipliziere entlang eines Weges im Graph
            valueOfPathToNode = pathValue * localDerivative
            # Addiere entlang unterschiedlicher Wege
            dy[node] = dy[node] + valueOfPathToNode
            # Rekursion zum Durchlaufen des ganzen Graphen
            computeDerivatives(node, valueOfPathToNode)

    # Initialisierung mit 1 (Ableitung von y nach y)
    computeDerivatives(y, pathValue = 1)
    return dy

def getGradient(x0, y):
    dy = getDerivatives(y)
    grad = []
    for i in range(len(x0)):
        grad.append(dy[x0[i]])
    return grad

