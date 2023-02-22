"""
import numpy as np

class FloatMultiSad:

    def __init__(self, value, derivative):
        self.value = np.array(value)
        self.derivative = np.array(derivative)
        self.nbderivatives = len(derivative)

    
    def __add__(self, other):
        nbd = self.nbderivatives
        newValue = self.value + other.value
        newDerivative = [0] * nbd
        for i in range(nbd):
            newDerivative[i] = self.derivative[i] + other.derivative[i]
        return FloatMultiSad(newValue, newDerivative)

        





if __name__ == '__main__':

    def f(x):
        x = np.vectorize(lambda x : FloatMultiSad(x, [1, 0, 0]))
     #   y = x[0] + x[1]
     #   y1 = x[0] * x[1]**2 + x[2]
     #   y2 = x[2] / x[2] + x[1] * x[2]
     #   y = [y1, y2]
        return x

    getValue = np.vectorize(lambda y : y[0])

    x = [1, 2, 3]
    print(getValue(f(x)))
"""


class FloatMultiSad:

    def __init__(self, value, derivative):
       self.value = value             # reelle Zahl
       self.derivative = derivative   # dictionary 


    def __repr__(self):
        return "< " + str(self.value) + " ; " + str(self.derivative) + " >"

    
    def __add__(self, other):
        if other in [int, float]:
            return FloatMultiSad(self.value + other, self.derivative)
        else:
            newValue = self.value + other.value
            newDerivative = {}
            for var in self.derivative:
                newDerivative[var] = self.derivative[var]
            for var in other.derivative:
                if var in newDerivative:
                    newDerivative[var] = newDerivative[var] + other.derivative[var]
                else:
                    newDerivative[var] = other.derivative[var]
            return FloatMultiSad(newValue, newDerivative)



if __name__ == "__main__":

    def f(x, y, z):
        x = FloatMultiSad(value = x, derivative = {'x' : 1})
        y = FloatMultiSad(value = y, derivative = {'y' : 1})
        z = FloatMultiSad(value = z, derivative = {'z' : 1})

        return [x + x + y + y + y + z, x + y + z]

    res = f(1, 2, 3)
    print(res)