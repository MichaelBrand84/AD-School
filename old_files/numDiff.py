'''
def f(x):
    y = x ** 2
    return y

def fdot(f, x0, h):
    df = (f(x0 + h) - f(x0)) / h
    return df

x0 = 0.2
H = [10 ** -9, 10 ** -10, 10 ** -11, 10 ** -12]
for h in H:
    ydot = fdot(f, x0, h)
    print("h = " + str(h) + "\t=> f'(x0) = " + str(ydot))


x = -2
y = 1 if x>0 else -1

print(y)

'''

from floatsad import FloatSad
import math
import mathsad
import matplotlib.pyplot as plt

def d(t):
    t = FloatSad(t)
    Px = 2 * mathsad.cos(t) - 1    # x-Koordinate von P
    Py = 1.5 * mathsad.sin(t)      # y-Koordinate von P
    Pz = 0                         # z-Koordinate von P
    Qx = -3 * mathsad.sin(2*t)     # x-Koordinate von Q
    Qy = 2 * mathsad.cos(2*t) + 1  # y-Koordinate von Q
    Qz = 2 * mathsad.sin(2*t) + 1  # z-Koordinate von Q
    y = mathsad.sqrt((Px-Qx)**2 + (Py-Qy)**2 + (Pz-Qz)**2)
    return y

def gradient_descent(f, x0, lam):
    tol = 1e-9
    # Erster Schritt berechnen
    y0 = f(x0)
    x1 = x0 - lam * y0.derivative
    while math.fabs(x1-x0) > tol:
        x0 = x1
        y0 = f(x0)
        x1 = x0 - lam * y0.derivative
    return x1

if __name__ == "__main__":
    t0 = 3
    tmin = gradient_descent(d, t0, 0.01)
    dmin = d(tmin)
    print("Minimum bei (", tmin, dmin.value, ")")