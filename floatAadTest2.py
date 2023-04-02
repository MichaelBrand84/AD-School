"""
# Gradient berechnen (von Hand)

from floataad import float2FloatAad, getGradient
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

"""
## Jacobi Matrix berechnen (elegant)

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
"""


"""
## Gradient Descent

from floataad import float2FloatAad, getGradient
import numpy as np

def f(x):
    y = x[0]**4 + x[1]**4 + x[0] * x[1]**3
    y = y - x[0]**2 * x[1] - x[1]**2
    return y

getValues = np.vectorize(lambda y : y.value)

# Startwert und Lambda für Gradient Descent
x0 = [0.5, 0]
lam = 0.1
tol = 1e-6 # Toleranz für Abbruchbedingung

x0 = float2FloatAad(x0)
y0 = f(x0)
dy = np.array(getGradient(x0, y0))

# Erster Schritt
x1 = x0 - lam * dy

# Iteration bis Distanz zwischen zwei
# aufeinanderfolgenden Punkte kleiner als tol ist.
while np.linalg.norm(getValues(x0) - getValues(x1)) > tol:
    x0 = x1
    y0 = f(x0)
    dy = np.array(getGradient(x0, y0))
    x1 = x0 - lam * dy

print("Lokales Minimum gefunden in der Nähe von")
print(getValues(x1))
"""


# Lineare Regression ohne Vektorisierung

from floataad import FloatAad, getDerivatives
import numpy as np
import matplotlib.pyplot as plt

def loss(a, b):
    # X und Y werden im global space gefunden
    sum = 0
    for i in range(len(X)):
        d = float(Y[i]) - (a * float(X[i]) + b)
        sum += d**2
    return np.sum(sum)
    

# Parameter
anz = 50 # Anzahl Datenpunkte
xmin, xmax = 0, 10
s = 2    # Streuung

# Korrekte Funktion
f = lambda x: 2 * x + 3

# Daten erzeugen
X = np.linspace(xmin, xmax, anz)
Y = f(X)
# Rauschen hinzufügen
Y = Y + s * np.random.randn(anz)


# Gradient Descent
lam = 0.0005

a0, b0 = FloatAad(0), FloatAad(0) # Startwerte

Phi = loss(a0, b0)
dPhi = getDerivatives(Phi)

a1 = a0 - lam * dPhi[a0]
b1 = b0 - lam * dPhi[b0]

while (a1.value-a0.value)**2 + (b1.value-b0.value)**2 > 1e-9:
    a0, b0 = a1, b1
    Phi = loss(a0, b0)
    dPhi = getDerivatives(Phi)
    a1 = a0 - lam * dPhi[a0]
    b1 = b0 - lam * dPhi[b0]

# Regressionsgerade
a, b = a1.value, b1.value
g = np.vectorize(lambda x : a * x + b)
print("y = g(x) = " + str(a) + "x + " + str(b))

# Daten darstellen
plt.plot(X,Y, 'b.', X, f(X), 'r--', X, g(X), 'g-.')
plt.legend(["Datenpunkte", "Erzeugende Funktion", "Regressionsgerade"])
plt.show()