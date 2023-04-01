"""from floataad import float2FloatAad, getGradient
import numpy as np

def f(x):
    y0 = x[0] + x[1]**2 + 1/x[2]
    y1 = x[0] * x[1] * x[2]
    return [y0, y1]

x0 = [-2,-4,0.5]
x0 = float2FloatAad(x0)

y = f(x0)
y0 = y[0].value
y1 = y[1].value

dy0 = getGradient(x0, y[0])
dy1 = getGradient(x0, y[1])

dy = np.array([dy0, dy1])

print("Funktionswert: " + str([y0, y1]))
print("Gradient: ")
print(dy)
"""

from floataad import float2FloatAad, getGradient
import numpy as np

def f(x):
    y0 = x[0] + x[1]**2 + 1/x[2]
    y1 = x[0] * x[1] * x[2]
    return [y0, y1]

x0 = [-2,-4,0.5]
x0 = float2FloatAad(x0)

getValues = np.vectorize(lambda y : y.value)
getJacobian = lambda x,y : np.array([getGradient(x, y[i]) for i in range(len(y))])

y = f(x0)

val = getValues(y)
Jacobian = getJacobian(x0, y)

print(val)
print(Jacobian)
