import math
import matplotlib.pyplot as plt

def f(x):
    # Parameter werden im global space gefunden
    # Berechnung des Skalarprodukts und dessen Ableitung
    v0dot = 1
    v0 = x
    v1dot = -math.sin(v0) * v0dot  # Ableitung von ...
    v1 = math.cos(v0)  # x-Koordinate von X
    v2dot = math.cos(v0) * v0dot   # Ableitung von ...
    v2 = math.sin(v0)  # y-Koordinate von X
    v3dot = - v1dot    # Ableitung von ...
    v3 = px - v1       # x-Komponente des Vektors XP
    v4dot = - v2dot    # Ableitung von ...
    v4 = py - v2       # y-Komponente des Vektors XP
    v5dot = 1 / (2*math.sqrt(v3**2 + v4**2)) \
        * (2*v3*v3dot + 2*v4*v4dot)  # Ableitung von ...
    v5 = math.sqrt(v3**2 + v4**2)  # Länge des Vektors XP
    v6dot = (v3dot * v5 - v3 * v5dot) / v5**2  # Ableitung von ...
    v6 = v3 / v5       # x-Komponente des Einheitsvektors eP
    v7dot = (v4dot * v5 - v4 * v5dot) / v5**2  # Ableitung von ...
    v7 = v4 / v5       # y-Komponente des Einheitsvektors eP
    v8dot = -v1dot     # Ableitung von ...
    v8 = a - v1        # x-Komponente des Vektors XQ
    v9dot = -v2dot     # Ableitung von ...
    v9 = -v2           # y-Komponente des Vektors XQ
    v10dot = 1 / (2*math.sqrt(v8**2 + v9**2)) \
        * (2*v8*v8dot + 2*v9*v9dot)  # Ableitung von ...
    v10 = math.sqrt(v8**2 + v9**2)  # Länge des Vektors XQ
    v11dot = (v8dot * v10 - v8 * v10dot) / v10**2  # Ableitung von ...
    v11 = v8 / v10     # x-Komponente des Vektors eQ    
    v12dot = (v9dot * v10 - v9 * v10dot) / v10**2  # Ableitung von ... 
    v12 = v9 / v10     # y-Komponente des Vektors eQ   
    ydot = (v6dot + v11dot) * v2 + (v6 + v11) * v2dot \
        - ( (v7dot + v12dot) * v1 + (v7 + v12) * v1dot )  # Ableitung von ...
    y = (v6 + v11) * v2 - (v7 + v12) * v1
    return [y, ydot]   

def newton(f, x0):
    tol = 1e-8
    # Erster Schritt berechnen
    [y0, y0dot] = f(x0)
    x1 = x0 - y0 / y0dot
    while math.fabs(x1 - x0) > tol:
        x0 = x1
        [y0, y0dot] = f(x0)
        x1 = x0 - y0 / y0dot
    return x1 


if __name__ == "__main__":

    # Parameter definieren
    a = -0.8           # Position von Q = (a|0)
    px, py = 0.5, 0.5  # Position von P = (px|py)

    # Lösung des Billardproblems berechnen
    sol = set({}) # leere Menge, in der die gefundenen Lösungen gespeichert werden
    X = [2*math.pi * k / 10 for k in range(10)]  # Liste der Startwerte für Newton
    for x0 in X:
        x = newton(f, x0)
        sol.add(x)

    # Lösungen grafisch darstellen
    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlim((-1.2, 1.2))
    ax.set_ylim((-1.2, 1.2))
    ax.set_aspect('equal')
    circle = plt.Circle((0,0), 1, color='b', fill=False)
    qBall = plt.Circle((a,0), 0.02, color='k')
    pBall = plt.Circle([px, py], 0.02, color='k')
    ax.add_patch(circle)
    ax.add_patch(qBall)
    ax.add_patch(pBall)
    for x in sol:
        xcoords = [a, math.cos(x), px]
        ycoords = [0, math.sin(x), py]
        plt.plot(xcoords, ycoords, linewidth=1, linestyle='--')
    plt.show()