import math
from floatsad import FloatSad


def sqrt(x:FloatSad) -> FloatSad:
    newValue = math.sqrt(x.value)
    newDerivative = 1/(2*math.sqrt(x.value)) * x.derivative
    return FloatSad(newValue, newDerivative)

def exp(x:FloatSad) -> FloatSad:
    newValue = math.exp(x.value)
    newDerivative = math.exp(x.value) * x.derivative
    return FloatSad(newValue, newDerivative)

#def log(x:FloatSad, b:float = math.e) -> FloatSad:
    newValue = math.log(x.value, b)
    newDerivative = 1 / (x.value * math.log(b)) * x.derivative
    return FloatSad(newValue, newDerivative)

def log(x:FloatSad, b = math.e) -> FloatSad:
    if type(b) in (float, int):
        newValue = math.log(x.value, b)
        newDerivative = 1 / (x.value * math.log(b)) * x.derivative
    else:
        newValue = math.log(x.value, b.value)
        newDerivative = (x.derivative/x.value * math.log(b.value) - math.log(x.value) * b.derivative / b.value) \
            / math.pow(math.log(b.value), 2)
    return FloatSad(newValue, newDerivative)

def sin(x:FloatSad) -> FloatSad:
    newValue = math.sin(x.value)
    newDerivative = math.cos(x.value) * x.derivative
    return FloatSad(newValue, newDerivative)

def cos(x:FloatSad) -> FloatSad:
    newValue = math.cos(x.value)
    newDerivative = -math.sin(x.value) * x.derivative
    return FloatSad(newValue, newDerivative)

def tan(x:FloatSad) -> FloatSad:
    return sin(x) / cos(x)

def asin(x:FloatSad) -> FloatSad:
    newValue = math.asin(x.value)
    newDerivative = 1/math.sqrt( 1 - math.pow(x.value, 2)) * x.derivative
    return FloatSad(newValue, newDerivative)

def acos(x:FloatSad) -> FloatSad:
    newValue = math.acos(x.value)
    newDerivative = -1/math.sqrt( 1 - math.pow(x.value, 2)) * x.derivative
    return FloatSad(newValue, newDerivative)

def atan(x:FloatSad) -> FloatSad:
    newValue = math.atan(x.value)
    newDerivative = 1/(math.pow(x.value, 2) + 1) * x.derivative
    return FloatSad(newValue, newDerivative)

def sinh(x:FloatSad) -> FloatSad:
    newValue = math.sinh(x.value)
    newDerivative = math.cosh(x.value) * x.derivative
    return FloatSad(newValue, newDerivative)

def cosh(x:FloatSad) -> FloatSad:
    newValue = math.cosh(x.value)
    newDerivative = math.sinh(x.value) * x.derivative
    return FloatSad(newValue, newDerivative)

def tanh(x:FloatSad) -> FloatSad:
    return sinh(x) / cosh(x)

def asinh(x:FloatSad) -> FloatSad:
    newValue = math.asinh(x.value)
    newDerivative = 1/math.sqrt(math.pow(x.value, 2) + 1) * x.derivative
    return FloatSad(newValue, newDerivative)

def acosh(x:FloatSad) -> FloatSad:
    newValue = math.acosh(x.value)
    newDerivative = 1/math.sqrt(math.pow(x.value, 2) - 1) * x.derivative
    return FloatSad(newValue, newDerivative)

def atanh(x:FloatSad) -> FloatSad:
    newValue = math.atanh(x.value)
    newDerivative = -1/(math.pow(x.value, 2) - 1) * x.derivative
    return FloatSad(newValue, newDerivative)