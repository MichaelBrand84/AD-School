"""
## Version 1
## Variablen als FloatAad initialisiert, keine Funktion

from floataad import FloatAad, getDerivatives

x0 = FloatAad(2)
x1 = FloatAad(3)

y = (x0 + x1) * x0 - x1
dy = getDerivatives(y)

print(y.value)
print("")
print(dy[x0])
print(dy[x1])
"""






"""
## Version 2
## Funktion, Variablen ausserhalb als FloatAad initialisiert und übergeben

from floataad import FloatAad, getDerivatives

def f(x0, x1):
    y = (x0 + x1) * x0 - x1
    return y

x0 = FloatAad(2)
x1 = FloatAad(3)
y = f(x0, x1)
dy = getDerivatives(y)

print(y.value)
print("")
print(dy[x0])
print(dy[x1])
"""






"""
## Version 3
## Wie Version 2, aber Input als Liste und Konvertierung zu FloatAad mit vektorisierter Funktion

from floataad import FloatAad, getDerivatives
import numpy as np

def f(x):
    y = (x[0] + x[1]) * x[0] - x[1]
    return y

float2FloatAad = np.vectorize(lambda x: FloatAad(x))

x0 = [2, 3]
x0 = float2FloatAad(x0)
y = f(x0)
dy = getDerivatives(y)

print(y.value)
print("")
print(dy[x0[0]])
print(dy[x0[1]])
"""






"""
## Version 4
## Funktion, welche Gradient erzeugt

from floataad import FloatAad, getDerivatives
import numpy as np

def getGradient(x0, y):
    dy = getDerivatives(y)
    grad = []
    for i in range(len(x0)):
        grad.append(dy[x0[i]])
    return grad

def f(x):
    y = (x[0] + x[1]) * x[0] - x[1]
    return y

float2FloatAad = np.vectorize(lambda x: FloatAad(x))

x0 = [2, 3]
x0 = float2FloatAad(x0)
y = f(x0)
g = getGradient(x0, y)


print(y.value)
print("")
print(g)
"""






"""
## Version 5
## Umwandlung in FloatAad Objekte und Berechnung des Gradienten geschieht in der Funktion

from floataad import FloatAad, getDerivatives
import numpy as np

def getGradient(x0, y):
    dy = getDerivatives(y)
    grad = []
    for i in range(len(x0)):
        grad.append(dy[x0[i]])
    return grad

def f(x):
    x = float2FloatAad(x)

    y = (x[0] + x[1]) * x[0] - x[1]

    g = getGradient(x, y)
    return [y.value, g]

float2FloatAad = np.vectorize(lambda x: FloatAad(x))

x0 = [2, 3]
[y0, grad] = f(x0)

print(y0)
print("")
print(grad)
"""






"""
## Version 6
## Wie Version 5, aber Funktionen getGradient und float2FloatAad

from floataad import float2FloatAad, getGradient

def f(x):
    x = float2FloatAad(x)

    y = (x[0] + x[1]) * x[0] - x[1]

    g = getGradient(x, y)
    return [y.value, g]

x0 = [2, 3]
[y0, grad] = f(x0)

print(y0)
print("")
print(grad)
"""





"""
#Gradient Descent Beispiel

from floataad import float2FloatAad, getGradient

def f(x):
    x = float2FloatAad(x)

    v1 = x[0] * x[0] * x[0] * x[0]
    v2 = x[1] * x[1] * x[1] * x[1]
    v3 = x[0] * x[0] * x[1]
    v4 = x[0] * x[1] * x[1] * x[1]
    v5 = x[1] * x[1]
    y = v1 + v2 - v3 + v4 - v5

    g = getGradient(x, y)
    return [y.value, g]

x0 = [-0.5, 0]  # Startwert
lam = 0.01      # Proportionalitätsfaktor lambda
eps = 1e-8      # Toleranz

# 1. Schritt
[y0, grad] = f(x0)
x1 = [x0[0] - lam * grad[0], x0[1] - lam * grad[1]]

# Iteration
while (x1[0] - x0[0])**2 + (x1[1] - x0[1])**2 > eps:
    x0 = x1
    [y0, grad] = f(x0)
    x1 = [x0[0] - lam * grad[0], x0[1] - lam * grad[1]]

print("Das Minimum befindet sich bei " + str(x1))
print("Der minimale Funktionswert beträgt dort " + str(y0))
"""

from floataad import FloatAad, getDerivatives

x0 = FloatAad(2)
x1 = FloatAad(3)

y = 4 / (x0 * x1 * x1) - 5 * x0 + 3
dy = getDerivatives(y)

print(y.value)
print("")
print(dy[x0])
print(dy[x1])