################################################################################
#                                                                              #
#	Michael Brand                                                              #    
#	Augmented Float values for Standard Algorithmic Differentiation (SAD)      #
#	PYTHON 3.8.2                                                               #
#                                                                              #
################################################################################

import math

class FloatSad:
    """Augmented Float values for Standard Algorithmic Differentiation (SAD)"""

    def __init__(self, value, derivative = 1.0):
        self.value = float(value)
        self.derivative = derivative

    def __repr__(self):
        return "< " + str(self.value) + " ; " + str(self.derivative) + " >"


    # Vergleichsoperatoren 

    def __lt__(self, other):
        if type(other) in (float, int):
            return self.value < other
        else:
            return self.value < other.value

    def __le__(self, other):
        if type(other) in (float, int):
            return self.value <= other
        else:
            return self.value <= other.value

    def __eq__(self, other):
        if type(other) in (float, int):
            return self.value == other
        else:
            return self.value == other.value

    def __ne__(self, other):
        if type(other) in (float, int):
            return self.value != other
        else:
            return self.value != other.value

    def __gt__(self, other):
        if type(other) in (float, int):
            return self.value > other
        else:
            return self.value > other.value

    def __ge__(self, other):
        if type(other) in (float, int):
            return self.value >= other
        else:
            return self.value >= other.value


    # unäre Operatoren (Vorzeichen)

    def __pos__(self):
        return FloatSad(self.value, self.derivative)
    
    def __neg__(self):
        newValue = -self.value
        newDerivative = -self.derivative
        return FloatSad(newValue, newDerivative)
    

    # binäre Operatoren (+, -, *, /, **)

    def __add__(self, other):
        if type(other) in (float, int):
            newValue = self.value + other
            newDerivative = self.derivative + 0.0
        else:
            newValue = self.value + other.value
            newDerivative = self.derivative + other.derivative
        return FloatSad(newValue, newDerivative)

    def __radd__(self, other):
        if type(other) in (float, int):
            newValue = other + self.value
            newDerivative = 0.0 + self.derivative
        else:
            newValue = other.value + self.value
            newDerivative = other.derivative + self.derivative
        return FloatSad(newValue, newDerivative)

    def __sub__(self, other):
        if type(other) in (float, int):
            newValue = self.value - other
            newDerivative = self.derivative - 0.0
        else:
            newValue = self.value - other.value
            newDerivative = self.derivative - other.derivative
        return FloatSad(newValue, newDerivative)

    def __rsub__(self, other):
        if type(other) in (float, int):
            newValue = other - self.value
            newDerivative = 0.0 - self.derivative
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

    def __truediv__(self, other):
        if type(other) in (float, int):
            if other == 0:
                raise ZeroDivisionError("'other' must be different from zero.")
            else:
                newValue = self.value / other
                newDerivative = self.derivative / other
        else:
            if other.value == 0:
                raise ZeroDivisionError("'other.value' must be different from zero.")
            else:
                newValue = self.value / other.value
                newDerivative = (self.derivative * other.value - self.value * other.derivative) / math.pow(other.value, 2)
        return FloatSad(newValue, newDerivative)

    def __rtruediv__(self, other):
        if self.value == 0:
            raise ZeroDivisionError("'self.value' must be different from zero.")
        else:
            if type(other) in (float, int):
                newValue = other / self.value
                newDerivative = - other / math.pow(self.value, 2) * self.derivative
            else:
                newValue = other.value / self.value
                newDerivative = (other.derivative * self.value - other.value * self.derivative) / math.pow(self.value, 2) * self.derivative
        return FloatSad(newValue, newDerivative)

    def __pow__(self, other):
        if type(other) in (float, int):
            newValue = math.pow(self.value, other)
            newDerivative = other * math.pow(self.value, other - 1) * self.derivative
        else:
            if self.value < 0:
                raise ValueError("'self.value' must be positive.")
            else:
                newValue = math.pow(self.value, other.value)
                newDerivative = math.pow(self.value, other.value) * \
                    (other.derivative * math.log(self.value) + other.value * self.derivative / self.value)
        return FloatSad(newValue, newDerivative)

    def __rpow__(self, other):
        if type(other) in (float, int):
            if other <= 0:
                raise ValueError("'other' must be positive.")
            else:
                newValue = math.pow(other, self.value)
                newDerivative = math.pow(other, self.value) * math.log(other) * self.derivative
        else:
            if other.value <= 0:
                raise ValueError("'other.value' must be positive.")
            else:
                newValue = math.pow(other.value, self.value)
                newDerivative = math.pow(other.value, self.value) * \
                    (self.derivative * math.log(other.value) + self.value * other.derivative / other.value)
        return FloatSad(newValue, newDerivative)
    