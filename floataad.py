## Implementierung AAD
## Quelle: https://sidsite.com/posts/autodiff/


from collections import defaultdict
import numpy as np

class FloatAad:

    def __init__(self, value, derivatives = ()):
        self.value = value
        self.derivatives = derivatives

    def __neg__(self):
        return neg(self)
        
    def __add__(self, other):
        if type(other) in [int, float]:
            return add(self, FloatAad(other))
        else:
            return add(self, other)
        
    def __radd__(self, other):
        if type(other) in [int, float]:
            return add(FloatAad(other), self)
        else:
            return add(other, self)
    
    def __sub__(self, other):
        if type(other) in [int, float]:
            return add(self, FloatAad(-other))
        else:
            return add(self, neg(other))
        
    def __rsub__(self, other):
        if type(other) in [int, float]:
            return add(FloatAad(other), neg(self))
        else:
            return add(neg(other), self)
    
    def __mul__(self, other):
        if type(other) in [int, float]:
            return mul(self, FloatAad(other))
        else:
            return mul(self, other)
        
    def __rmul__(self, other):
        if type(other) in [int, float]:
            return mul(FloatAad(other), self)
        else:
            return mul(other, self)
    
    def __truediv__(self, other):
        if type(other) in [int, float]:
            return mul(self, inv(FloatAad(other)))
        else:
            return mul(self, inv(other))
        
    def __rtruediv__(self, other):
        if type(other) in [int, float]:
            return mul(FloatAad(other), inv(self))
        else:
            return mul(other, inv(self))


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
        (a, b.value),  # ab nach a abgeleitet gibt b
        (b, a.value)   # ab nach b abgeleitet gibt a
    )
    return FloatAad(newValue, newDerivative)

def inv(a):
    newValue = 1. / a.value
    newDerivative = (
        (a, -1. / a.value**2), 
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

    # Initialisierung mit 1 weil Ableitung von y nach sich selber
    computeDerivatives(y, pathValue = 1)
    return dy

def getGradient(x0, y):
    dy = getDerivatives(y)
    grad = []
    for i in range(len(x0)):
        grad.append(dy[x0[i]])
    return grad

