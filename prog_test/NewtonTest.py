from floatsad import FloatSad
import math
import mathsad



def f(arg:float) -> FloatSad:
    x = FloatSad(arg)
    b1 = x + mathsad.exp(x)
    b2 = x ** 2 + mathsad.sqrt(x)
    b3 = 1/x
    b4 = b1 * mathsad.cos(b2) + b3
    return b4
    

def main():
    x0 = 0.5
    tol = 10 ** -12
    y = f(x0)
    x1 = x0 - y.value / y.derivative
    while math.fabs(x0 - x1) > tol:
        x0 = x1
        y = f(x0)
        x1 = x0 - y.value / y.derivative
    print(x0)





if __name__ == '__main__': 
    print("\n\n")
    main()
    print("\n\n")
