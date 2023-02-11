"""
## Funktion f: R -> R^m (Parameterkurve)
## Ableitung mit Hilfe der FloatSad Klasse berechnet
## Problem: Liste von Floatsadobjekten
## nicht möglich: y.value[0] oder y.derivative[1], sondern y[0].value bzw. y[1].derivative

from floatsad import FloatSad
import mathsad

def f(t):
    t = FloatSad(t)
    f1 = t
    f2 = t**2
    f3 = mathsad.sin(t)
    return [f1, f2, f3]

t0 = 3
y = f(t0)
print(y)  # Liste von drei Flaotsad Objekten
# Alternative Darstellung; Liste der drei Werte und Liste der drei Ableitungen.
print("< " + str([element.value for element in y]) + " ; " + str([element.derivative for element in y]) + " >")
"""




"""
## Funktion f: R -> R^m (Parameterkurve)
## Manuelle Differentiation
## Vorteil: Ausgabe kann so gestaltet werden, wie man möchte, vgl Version 1
## Nachteil: Automatisieren in einer Klasse wird schwierig weil Listen erst am Schluss der Funktion definiert werden
import math

def f(t):
    v0dot = 1
    v0 = t
    v1dot = 2 * v0 * v0dot
    v1 = v0**2
    v2dot = math.cos(v0) * v0dot
    v2 = math.sin(v0)
    y = [v0, v1, v2]
    ydot = [v0dot, v1dot, v2dot]
    return [y, ydot]

t0 = 3
[y, ydot] = f(t0)
print(y)
print(ydot)
"""




## Funktion f: R -> R^m (Parameterkurve)
## Manuelle Differentiation mit Listen
import math

def f(t):
    pass