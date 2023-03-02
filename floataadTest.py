"""
from floataad import FloatAad, getDerivatives

## Version 1
## Variablen als FloatAad initialisiert, keine Funktion
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
from floataad import FloatAad, getDerivatives

## Version 2
## Funktion, Variablen ausserhalb als FloatAad initialisiert und übergeben
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
from floataad import FloatAad, getDerivatives
import numpy as np

## Version 3
## Wie Version 2, aber Input als Liste und Konvertierung zu FloatAad mit vektorisierter Funktion
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
from floataad import FloatAad, getDerivatives
import numpy as np

## Version 4
## Funktion, welche Gradient erzeugt

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
from floataad import FloatAad, getDerivatives
import numpy as np

## Version 5
## Umwandlung in FloatAad Objekte und Berechnung des Gradienten geschieht in der Funktion

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







from floataad import float2FloatAad, getGradient

## Version 6
## Wie Version 5, aber Funktionen getGradient und float2FloatAad



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