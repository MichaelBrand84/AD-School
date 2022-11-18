import math

class FloatSad:

    def __init__(self, value, derivative = 1.0) -> None:
        self.value = float(value)
        self.derivative = derivative

    def __repr__(self):
        return "< " + str(self.value) + " ; " + str(self.derivative) + " >"


    # unäre Operatoren

    def __pos__(self):
        return FloatSad(self.value, self.derivative)
    
    def __neg__(self):
        newValue = -self.value
        newDerivative = -self.derivative
        return FloatSad(newValue, newDerivative)
    

    # binäre Operatoren

    def __add__(self, other):
        if type(other) in (float, int):
            newValue = self.value + other
            newDerivative = self.derivative + 0
        else:
            newValue = self.value + other.value
            newDerivative = self.derivative + other.derivative
        return FloatSad(newValue, newDerivative)

    def __radd__(self, other):
        if type(other) in (float, int):
            newValue = other + self.value
            newDerivative = 0 + self.derivative
        else:
            newValue = other.value + self.value
            newDerivative = other.derivative + self.derivative
        return FloatSad(newValue, newDerivative)

    def __sub__(self, other):
        if type(other) in (float, int):
            newValue = self.value - other
            newDerivative = self.derivative - 0
        else:
            newValue = self.value - other.value
            newDerivative = self.derivative - other.derivative
        return FloatSad(newValue, newDerivative)

    def __rsub__(self, other):
        if type(other) in (float, int):
            newValue = other - self.value
            newDerivative = 0 - self.derivative
        else:
            newValue = other.value - self.value
            newDerivative = other.derivative - self.derivative
        return FloatSad(newValue, newDerivative)

    def __mul__(self, other):
        if type(other) in (float, int):
            newValue = self.value * other
            newDerivative = self.derivative * other
        else:
            newValue = self.value * other.value
            newDerivative = self.derivative * other.value + self.value * other.derivative
        return FloatSad(newValue, newDerivative)

    def __rmul__(self, other):
        if type(other) in (float, int):
            newValue = other * self.value
            newDerivative = other * self.derivative
        else:
            newValue = other.value * self.value
            newDerivative =  other.derivative * self.value + other.value * self.derivative
        return FloatSad(newValue, newDerivative)
    


if __name__ == '__main__':

    def f(x):
        v0 = FloatSad(x)
        v1 = -v0 
        v2 = 3 - v1
        v3 = v2 - v1
        y  = v3 - 5
        return y

    resultat = f(2)
    print(resultat)