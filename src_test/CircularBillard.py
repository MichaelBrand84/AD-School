from typing import Callable, Sequence
from floatsad import FloatSad
import matplotlib.pyplot as plt
import math
import mathsad

def f(angle:float, a:float, p:Sequence[float]) -> FloatSad:
    '''
    Berechnet das Skalarprodukt zwischen dem Vektor e_{XP} + e_{XQ}
    und dem Tangentenvektor r des Kreises.
    In: angle = Polarwinkel des Punktes X
        a = Abstand des Punktes Q vom Kreiszentrum
        p = Koordinaten des Punktes P mit ü[0] = xP, p[1] = yP
    
    Out: FloatSad des Skalarprodukts
    '''
    x = FloatSad(angle)
    c = mathsad.cos(x)
    s = mathsad.sin(x)
    # Einheitsvektor PX
    xp1 = p[0] - c
    xp2 = p[1] - s
    l = mathsad.sqrt(xp1 ** 2 + xp2 ** 2)
    ep1 = xp1 / l
    ep2 = xp2 / l
    # Einheitsvektor QX
    xq1 = a - c
    xq2 = -s
    l = mathsad.sqrt(xq1 ** 2 + xq2 ** 2)
    eq1 = xq1 / l
    eq2 = xq2 / l
    # Skalarprodukt
    sp = (ep1 + eq1) * s - (ep2 + eq2) * c
    return sp

def newton(f:Callable[[float], FloatSad], x0:float, tol:float) -> float:
    y = f(x0)
    x1 = x0 - y.value / y.derivative
    while math.fabs(x0 - x1) > tol:
        x0 = x1
        y = f(x0)
        x1 = x0 - y.value / y.derivative
    return x1


if __name__ == '__main__':
    # Parameter definieren
    a = -0.7
    p = [0.26, 0.69]

    # Lösungen für x bestimmen
    X = [2*math.pi * k / 100 for k in range(100)]
    lsg = set(())
    for x in X:
        winkel = newton(lambda x: f(x, a, p), x, 10 ** -8)
        lsg.add(winkel)
    print(lsg)

    # Grafik erstellen
    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlim((-1.2, 1.2))
    ax.set_ylim((-1.2, 1.2))
    ax.set_aspect('equal')
    circle = plt.Circle((0,0), 1, color='b', fill=False)
    qBall = plt.Circle((a,0), 0.02, color='k')
    pBall = plt.Circle(p, 0.02, color='k')
    ax.add_patch(circle)
    ax.add_patch(qBall)
    ax.add_patch(pBall)
    for x in lsg:
        xcoords = [a, math.cos(x), p[0]]
        ycoords = [0, math.sin(x), p[1]]
        plt.plot(xcoords, ycoords, linewidth=1, linestyle='--')
    plt.show()
