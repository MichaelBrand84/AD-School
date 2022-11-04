"""
Algorithmic Differentiation in Finance Explained
Example p. 19 - 22

Function
    z = cos(a0 + exp(a1)) * (sin(a2) + cos(a3)) + a1^(3/2) + a3
"""

import math


def f(a):
    b1 = a[0] + math.exp(a[1])
    b2 = math.sin(a[2]) + math.cos(a[3])
    b3 = math.pow(a[1], 1.5) + a[3]
    b4 = math.cos(b1) * b2 + b3
    return b4


def f_sad(a):
    # forward sweep - function
    b1 = a[0] + math.exp(a[1])
    b2 = math.sin(a[2]) + math.cos(a[3])
    b3 = math.pow(a[1], 1.5) + a[3]
    b4 = math.cos(b1) * b2 + b3

    # forward sweep - derivatives
    nbA = len(a)
    
    # Ableitungen manuell berechnen
    b1dot = [0] * nbA
    b1dot[0] = 1
    b1dot[1] = math.exp(a[1])

    b2dot = [0] * nbA
    b2dot[2] = math.cos(a[2])
    b2dot[3] = -math.sin(a[3])

    b3dot = [0] * nbA
    b3dot[1] = 1.5 * math.sqrt(a[1])
    b3dot[3] = 1

    # restliche Ableitungen werden nun automatisch berechnet
    b4dot = [0] * nbA
    for i in range(nbA):
        b4dot[i] = -math.sin(b1) * b1dot[i] * b2 + math.cos(b1) * b2dot[i] + 1.0 * b3dot[i]

    return [b4, b4dot]


if __name__ == '__main__':
    [z, zdot] = f_sad([1,2,3,4])
    print("z = ", z)
    print("grad(z) = ", zdot)