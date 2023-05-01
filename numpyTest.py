import numpy as np
from floataad import float2FloatAad, getGradient

getValues = np.vectorize(lambda x : x.value)

def getJacobian(A, y):
    J = np.array([])
    for row in A:
        dy = getGradient(row, y)
        J.append(dy)
    return J

A = np.array([[1, 2], [3, 4]])
A = float2FloatAad(A)
x = np.array([-2, 3])

y = A @ x
#Jy = getJacobian(A, y)
print(type(A[0][0]))

print(getValues(y))