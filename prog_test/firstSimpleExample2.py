"""
Eigenes Beispiel für f : R -> R

Function
    y = (x + exp(x)) * cos(x^2 + sqrt(x)) + 1/x
"""

import math


def f(x):
    b1 = x + math.exp(x)
    b2 = x**2 + math.sqrt(x)
    b3 = 1/x
    b4 = b1 * math.cos(b2) + b3
    return b4


def f_sad(x):
    # forward sweep - function
    b1 = x + math.exp(x)
    b2 = x**2 + math.sqrt(x)
    b3 = 1/x
    b4 = b1 * math.cos(b2) + b3

    # forward sweep - derivatives
    
    # Ableitungen manuell berechnen
    b1dot = 1 + math.exp(x)
    b2dot = 2 * x + 1 / (2 * math.sqrt(x))
    b3dot = -1 / x**2

    b4dot = b1dot * math.cos(b2) + b1 * (-math.sin(b2)) * b2dot + b3dot

    return [b4, b4dot]


if __name__ == '__main__':
    [z, zdot] = f_sad(2)
    print("z = ", z)
    print("grad(z) = ", zdot)