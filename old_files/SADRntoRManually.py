## Bsp aus Baydin et al. S. 10
## f(x1, x2) = ln(x1) + x1 * x2 - sin(x2)

"""
## Auswertung der Funktion
import math

def f(x):
    # Initialisierung
    v_1 = x[0]
    v0  = x[1]

    # Berechnung
    v1 = math.log(v_1)
    v2 = v_1 * v0
    v3 = math.sin(v0)
    v4 = v1 + v2
    v5 = v4 - v3
    
    # Rückgabe
    y = v5
    return y

x0 = [2, 5]
y0 = f(x0)
print(y0)
"""


## Auswertung der Funktion mit Ableitung
import math

def f(x):
    # Initialisierung
    v_1dot = -2
    v_1 = x[0]
    v0dot = 3
    v0  = x[1]

    # Berechnung
    v1dot = 1 / v_1 * v_1dot
    v1 = math.log(v_1)
    v2dot = v_1dot * v0 + v_1 * v0dot
    v2 = v_1 * v0
    v3dot = math.cos(v0) * v0dot
    v3 = math.sin(v0)
    v4dot = v1dot + v2dot
    v4 = v1 + v2
    v5dot = v4dot - v3dot
    v5 = v4 - v3
    
    # Rückgabe
    ydot = v5dot
    y = v5
    return [y, ydot]

x0 = [2, 5]
y0 = f(x0)
print(y0)


## Initialisierung mit vdot = r ergibt dann grad(f)(x0,y0) * r (Skalarprodukt).