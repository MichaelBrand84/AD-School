import matplotlib.pyplot as plt
import math
import numpy as np


def fdot(f, x0, h):
    df = (f(x0 + h) - f(x0)) / h
    return df


x0 = 0.2
'''
H = [10**(k/100) for k in range(-1800, -300)]
E = [math.fabs(fdot(lambda x: x**2, x0, h) - 2*x0) for h in H]

print(np.finfo(float).eps)


fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.set(xlim=(10**-18, 10**-3), ylim=(10**-12, 10**0))
ax.set_xscale('log')
ax.set_yscale('log')
plt.plot(H,E)
plt.show()
'''


print(fdot(lambda x: x**2, x0, 10**-18) == 2*x0)